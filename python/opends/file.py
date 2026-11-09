# SPDX-License-Identifier: BSD-3-Clause
"""OpenDSFile and its sync, async and stream I/O methods."""

import ctypes
import os

from . import cdll as _c
from .buffer import io_args, registry
from .driver import check, io_result, is_open, require_driver
from .stream import StreamOp, stream_handle


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


_FLAGS = {
    "r": os.O_RDONLY,
    "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    "rw": os.O_RDWR | os.O_CREAT,
    "r+": os.O_RDWR,
    "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
}


class OpenDSFile:
    """A file registered with the driver.

    path is a file to open, or a descriptor that is already open, as with
    the builtin open(). A descriptor keeps the flags it was opened with, so
    flags, use_direct_io and mode do not apply to it. closefd=False leaves
    the descriptor open when the file closes. A path needs closefd=True.
    """

    def __init__(
        self, path, flags="r", *, use_direct_io=None, mode=0o644, closefd=True
    ):
        self._fh = None
        self._fd = -1
        self._closefd = closefd
        if isinstance(path, int):
            self._fd = path
        else:
            if not closefd:
                raise ValueError("closefd=False needs a descriptor, not a path")
            if flags not in _FLAGS:
                raise ValueError("unsupported flags %r" % flags)
            oflags = _FLAGS[flags]
            if use_direct_io is None:
                use_direct_io = True
            if use_direct_io:
                oflags |= os.O_DIRECT
            require_driver()
            self._fd = os.open(path, oflags, mode)
        try:
            require_driver()
            fh = ctypes.c_void_p()
            check(_c.handle_register(ctypes.byref(fh), self._fd))
            self._fh = fh
        except Exception:
            self._release_fd()
            raise

    def _submit(self, c_fn, buf, size, file_offset, dev_offset):
        ptr, nbytes, size = io_args(buf, size, dev_offset)
        registry.ensure(ptr, nbytes)
        return io_result(c_fn(self._fh, ptr, size, file_offset, dev_offset))

    def _submit_async(self, c_fn, buf, size, file_offset, dev_offset):
        ptr, nbytes, size = io_args(buf, size, dev_offset)
        c_future = _c.DsAsyncFuture()
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

    @property
    def handle(self):
        """The opends_handle_t, or None once closed."""
        return self._fh

    def close(self):
        if self._fh is not None:
            _c.handle_deregister(self._fh)
            self._fh = None
        self._release_fd()

    def _release_fd(self):
        if self._fd >= 0 and self._closefd:
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
