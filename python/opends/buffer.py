# SPDX-License-Identifier: BSD-3-Clause
"""Buffer registration and HostBuffer.

A buffer for I/O comes from alloc, or the caller registers it with
register_buffer before the first I/O on it.
"""

import ctypes
import math

from . import cdll as _c
from .driver import check, preserve_cuda_context, require_driver


def register_buffer(buf, size=None):
    """Register an externally allocated buffer for I/O (cuFileBufRegister).

    For a bare pointer (ctypes.c_void_p or int) size is mandatory; for a
    structured buffer it defaults to the buffer's own extent. The
    registration lasts until deregister_buffer or the driver closes.
    """
    ptr, nbytes = buffer_view(buf)
    if size is None:
        if nbytes is None:
            raise ValueError("size is required to register a bare pointer")
        size = nbytes
    require_driver()
    with preserve_cuda_context():
        check(_c.buf_register(ptr, int(size), 0))


def deregister_buffer(buf):
    ptr, _ = buffer_view(buf)
    with preserve_cuda_context():
        check(_c.buf_deregister(ptr))


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
    such a buffer is given an explicit size at I/O time.
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
    """Resolve (ptr, size) for an I/O call on buf.

    The transfer must lie inside a buffer with a known extent; a bare
    pointer has none, so only its size is checked for presence.
    """
    ptr, nbytes = buffer_view(buf)
    if dev_offset < 0:
        raise ValueError("dev_offset %d is negative" % dev_offset)
    if nbytes is None:
        if size is None:
            raise ValueError("size is required for a bare pointer")
        return ptr, size
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
    return ptr, size


def alloc(size):
    """Allocate a buffer owned by the backend: 4096-aligned host memory on
    ref, device memory on GPU backends. Needs an open Driver."""
    return HostBuffer(size)


def free(buf):
    buf.free()
