# SPDX-License-Identifier: BSD-3-Clause
"""OpenDSFile and its sync, async and stream I/O methods."""

import ctypes
import os

from . import cdll as _c
from .buffer import io_args, registry, stream_handle
from .driver import check, io_result, is_open, preserve_cuda_context, require_driver

class Future:
    """Completion of one read_async or write_async call.

    The backend writes the completion into storage this object owns, so it
    must stay alive until result() has returned. A Future dropped before
    that awaits in __del__ to honor the contract.
    """

    def __init__(self, c_future, buf):
        self._c_future = c_future
        self._buf = buf  # keep the buffer alive while the I/O is in flight
        self._result = None

    @property
    def done(self):
        return self._result is not None or bool(self._c_future.done)

    def result(self):
        """Block until the operation completes; return its byte count."""
        if self._result is None:
            if not self._c_future.done:
                # A closed driver completes nothing; fail instead of spinning.
                require_driver()
            with preserve_cuda_context():
                ret = _c.async_await(ctypes.byref(self._c_future))
            self._result = int(ret)
            self._buf = None
        return io_result(self._result)

    def __del__(self):
        try:
            if self._result is None and (self._c_future.done or is_open()):
                _c.async_await(ctypes.byref(self._c_future))
        except Exception:
            pass


# Initial byte count of a StreamOp: never a byte count, never a negated code.
_PENDING = -1


class StreamOp:
    """One read_stream or write_stream call.

    The backend reads the size and offsets, and writes the byte count, when
    the stream reaches the operation. Keep this object alive until the
    stream has been synchronized; result() raises until then.
    """

    def __init__(self, size, file_offset, dev_offset, buf):
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_long(file_offset)
        self._dev_offset = ctypes.c_long(dev_offset)
        self._bytes = ctypes.c_ssize_t(_PENDING)
        self._buf = buf

    @property
    def done(self):
        return self._bytes.value != _PENDING

    def result(self):
        """Byte count once the stream has executed the operation."""
        if not self.done:
            raise RuntimeError(
                "the stream has not reached this operation; synchronize it first"
            )
        return io_result(int(self._bytes.value))


_FLAGS = {
    "r": os.O_RDONLY,
    "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    "rw": os.O_RDWR | os.O_CREAT,
    "r+": os.O_RDWR,
    "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
}


class OpenDSFile:
    def __init__(self, path, flags="r", *, use_direct_io=None, mode=0o644):
        if flags not in _FLAGS:
            raise ValueError("unsupported flags %r" % flags)
        oflags = _FLAGS[flags]
        if use_direct_io is None:
            use_direct_io = True
        if use_direct_io:
            oflags |= os.O_DIRECT
        self._fh = None
        self._fd = -1
        require_driver()
        self._fd = os.open(path, oflags, mode)
        try:
            with preserve_cuda_context():
                fh = ctypes.c_void_p()
                check(_c.handle_register(ctypes.byref(fh), self._fd))
            self._fh = fh
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

    def _submit(self, c_fn, buf, size, file_offset, dev_offset):
        ptr, nbytes, size = io_args(buf, size, dev_offset)
        with preserve_cuda_context():
            registry.ensure(ptr, nbytes)
            return io_result(c_fn(self._fh, ptr, size, file_offset, dev_offset))

    def _submit_async(self, c_fn, buf, size, file_offset, dev_offset):
        ptr, nbytes, size = io_args(buf, size, dev_offset)
        c_future = _c.DsAsyncFuture()
        with preserve_cuda_context():
            registry.ensure(ptr, nbytes)
            check(
                c_fn(
                    self._fh,
                    ptr,
                    size,
                    file_offset,
                    dev_offset,
                    ctypes.byref(c_future),
                )
            )
        return Future(c_future, buf)

    def _submit_stream(self, c_fn, buf, size, file_offset, dev_offset, stream):
        ptr, nbytes, size = io_args(buf, size, dev_offset)
        op = StreamOp(size, file_offset, dev_offset, buf)
        with preserve_cuda_context():
            registry.ensure(ptr, nbytes)
            check(
                c_fn(
                    self._fh,
                    ptr,
                    ctypes.byref(op._size),
                    ctypes.byref(op._file_offset),
                    ctypes.byref(op._dev_offset),
                    ctypes.byref(op._bytes),
                    stream_handle(stream),
                )
            )
        return op

    def read_sync(self, buf, size=None, file_offset=0, dev_offset=0):
        return self._submit(_c.sync_read, buf, size, file_offset, dev_offset)

    def write_sync(self, buf, size=None, file_offset=0, dev_offset=0):
        return self._submit(_c.sync_write, buf, size, file_offset, dev_offset)

    def read_async(self, buf, size=None, file_offset=0, dev_offset=0):
        """Submit a read without waiting; the Future carries the byte count."""
        return self._submit_async(
            _c.async_read, buf, size, file_offset, dev_offset
        )

    def write_async(self, buf, size=None, file_offset=0, dev_offset=0):
        """Submit a write without waiting; the Future carries the byte count."""
        return self._submit_async(
            _c.async_write, buf, size, file_offset, dev_offset
        )

    def read_stream(self, buf, size=None, file_offset=0, dev_offset=0, stream=None):
        """Enqueue a read on a CUDA stream (cuFileReadAsync); see StreamOp."""
        return self._submit_stream(
            _c.stream_read, buf, size, file_offset, dev_offset, stream
        )

    def write_stream(self, buf, size=None, file_offset=0, dev_offset=0, stream=None):
        """Enqueue a write on a CUDA stream (cuFileWriteAsync); see StreamOp."""
        return self._submit_stream(
            _c.stream_write, buf, size, file_offset, dev_offset, stream
        )

    def fileno(self):
        return self._fd

    def close(self):
        if self._fh is not None:
            with preserve_cuda_context():
                _c.handle_deregister(self._fh)
            self._fh = None
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
