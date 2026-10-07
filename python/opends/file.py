# SPDX-License-Identifier: BSD-3-Clause
"""OpenDSFile and its sync I/O methods."""

import ctypes
import os

from . import cdll as _c
from . import driver
from .buffer import buffer_view, registry
from .driver import check, io_result, preserve_cuda_context

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
        self._fd = os.open(path, oflags, mode)
        try:
            with preserve_cuda_context():
                driver.ensure_driver()
                fh = ctypes.c_void_p()
                check(_c.handle_register(ctypes.byref(fh), self._fd))
            self._fh = fh
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

    def _submit(self, c_fn, buf, size, file_offset, dev_offset):
        ptr, nbytes = buffer_view(buf)
        if size is None:
            if nbytes is None:
                raise ValueError("size is required for a bare pointer")
            size = nbytes - dev_offset
        with preserve_cuda_context():
            registry.ensure(ptr, nbytes)
            return io_result(c_fn(self._fh, ptr, size, file_offset, dev_offset))

    def read_sync(self, buf, size=None, file_offset=0, dev_offset=0):
        return self._submit(_c.sync_read, buf, size, file_offset, dev_offset)

    def write_sync(self, buf, size=None, file_offset=0, dev_offset=0):
        return self._submit(_c.sync_write, buf, size, file_offset, dev_offset)

    def close(self):
        if self._fh is not None:
            with preserve_cuda_context():
                _c.handle_deregister(self._fh)
                self._fh = None
                driver.release_driver()
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
