# SPDX-License-Identifier: BSD-3-Clause
"""Buffer registration and HostBuffer.

Buffers are registered on first use. Only a bare pointer, which has no
known extent, needs register_buffer up front.
"""

import ctypes
import math
import threading

from . import cdll as _c
from . import driver
from .driver import OpenDSError, check, preserve_cuda_context, require_driver


class Registry:
    """Base pointer to registered size, for the open driver."""

    def __init__(self):
        self._regs = {}
        # Reentrant: a GC finalizer (HostBuffer.free) can run inside ensure
        # while this thread holds the lock, and must not self-deadlock.
        self._lock = threading.RLock()

    def ensure(self, ptr, size):
        with self._lock:
            if ptr in self._regs:
                return
            if size is None:
                raise ValueError(
                    "bare pointer 0x%x is not registered; call "
                    "opends.register_buffer(ptr, size) on the base buffer "
                    "before using it for I/O" % ptr
                )
            err = _c.buf_register(ptr, size, 0)
            if err.err not in (
                _c.ErrorCode.SUCCESS,
                _c.ErrorCode.MEMORY_ALREADY_REGISTERED,
            ):
                raise OpenDSError(err.err, err.dev_err)
            self._regs[ptr] = size

    def drop(self, ptr):
        with self._lock:
            if self._regs.pop(ptr, None) is not None:
                _c.buf_deregister(ptr)

    def clear(self):
        with self._lock:
            for ptr in list(self._regs):
                _c.buf_deregister(ptr)
            self._regs.clear()


registry = Registry()
driver.on_close(registry.clear)


def register_buffer(buf, size=None):
    """Register a buffer for DMA before I/O (cuFileBufRegister).

    Structured buffers (HostBuffer, cupy/numpy/torch arrays) are
    registered on first use, so this is only needed for the cuFile
    pattern of registering one large base allocation and then indexing
    into it with a bare device pointer and dev_offset. For a bare
    pointer (ctypes.c_void_p or int) size is mandatory; for a structured
    buffer it defaults to the buffer's own extent. The registration
    lasts until deregister_buffer or the driver closes.
    """
    ptr, nbytes = buffer_view(buf)
    if size is None:
        if nbytes is None:
            raise ValueError("size is required to register a bare pointer")
        size = nbytes
    require_driver()
    with preserve_cuda_context():
        registry.ensure(ptr, int(size))


def deregister_buffer(buf):
    ptr, _ = buffer_view(buf)
    with preserve_cuda_context():
        registry.drop(ptr)


def stream_handle(stream):
    """Raw handle of a CUDA stream-like object. None is the default stream."""
    if stream is None:
        return 0
    if isinstance(stream, ctypes.c_void_p):
        return int(stream.value or 0)
    if isinstance(stream, int):
        return stream
    for attr in ("cuda_stream", "ptr"):  # torch.cuda.Stream, cupy.cuda.Stream
        value = getattr(stream, attr, None)
        if value is not None:
            return int(value)
    raise TypeError("unsupported stream object %r" % (stream,))


def register_stream(stream, flags=0):
    """cuFileStreamRegister."""
    require_driver()
    with preserve_cuda_context():
        check(_c.stream_register(stream_handle(stream), flags))


def deregister_stream(stream):
    with preserve_cuda_context():
        check(_c.stream_deregister(stream_handle(stream)))


class HostBuffer:
    """A backend-owned opends_alloc allocation: 4096-aligned host memory on
    ref, device memory on GPU backends. The backend allocates through the
    open driver and owns the memory for the driver's lifetime."""

    def __init__(self, size):
        self._ptr = 0
        self._size = int(size)
        require_driver()
        with preserve_cuda_context():
            ptr = _c.alloc(size)
        if not ptr:
            raise MemoryError("opends_alloc(%d) failed" % size)
        self._ptr = int(ptr)

    @property
    def ptr(self):
        return self._ptr

    @property
    def nbytes(self):
        return self._size

    def __len__(self):
        return self._size

    def as_ctypes(self):
        """The bytes as a ctypes array; host memory only, so the ref backend."""
        return (ctypes.c_char * self._size).from_address(self._ptr)

    def free(self):
        if self._ptr:
            with preserve_cuda_context():
                registry.drop(self._ptr)
                _c.free(self._ptr)
            self._ptr = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def _nbytes_from_iface(iface):
    itemsize = int(iface["typestr"][2:])
    return itemsize * math.prod(iface["shape"])


def buffer_view(buf):
    """Return (base_ptr:int, nbytes:int|None) for a buffer-like object.

    A bare device pointer (ctypes.c_void_p or int address, the cuFile
    calling convention) has no discoverable extent, so nbytes is None;
    such a buffer must be registered up front and given an explicit size
    at I/O time.
    """
    if isinstance(buf, HostBuffer):
        return buf.ptr, buf.nbytes
    if isinstance(buf, ctypes.c_void_p):
        return int(buf.value or 0), None
    if isinstance(buf, int):
        return buf, None
    cai = getattr(buf, "__cuda_array_interface__", None)
    if cai is not None:
        return int(cai["data"][0]), _nbytes_from_iface(cai)
    ai = getattr(buf, "__array_interface__", None)
    if ai is not None:
        return int(ai["data"][0]), _nbytes_from_iface(ai)
    if hasattr(buf, "data_ptr"):
        nbytes = getattr(buf, "nbytes", None)
        if nbytes is None:
            nbytes = buf.numel() * buf.element_size()
        return int(buf.data_ptr()), int(nbytes)
    mv = memoryview(buf)
    arr = (ctypes.c_char * mv.nbytes).from_buffer(buf)
    return ctypes.addressof(arr), mv.nbytes


def io_args(buf, size, dev_offset):
    """Resolve (ptr, nbytes, size) for an I/O call on buf.

    The transfer must lie inside a buffer with a known extent; a bare
    pointer has none, so only its size is checked for presence.
    """
    ptr, nbytes = buffer_view(buf)
    if dev_offset < 0:
        raise ValueError("dev_offset %d is negative" % dev_offset)
    if nbytes is None:
        if size is None:
            raise ValueError("size is required for a bare pointer")
        return ptr, nbytes, size
    if dev_offset > nbytes:
        raise ValueError(
            "dev_offset %d is past the end of the %d-byte buffer" % (dev_offset, nbytes)
        )
    if size is None:
        size = nbytes - dev_offset
    elif dev_offset + size > nbytes:
        raise ValueError(
            "%d bytes at dev_offset %d exceed the %d-byte buffer"
            % (size, dev_offset, nbytes)
        )
    return ptr, nbytes, size


def alloc(size):
    """Allocate a buffer owned by the backend: 4096-aligned host memory on
    ref, device memory on GPU backends. Needs an open Driver."""
    return HostBuffer(size)


def free(buf):
    buf.free()
