# SPDX-License-Identifier: BSD-3-Clause
"""Driver lifetime and errors.

A Driver must be open before any file, buffer, stream or batch call, as
with cuFileDriverOpen.
"""

import atexit
import collections
import ctypes
import errno
import threading

from . import cdll as _c

class OpenDSError(RuntimeError):
    def __init__(self, code, dev_err=0):
        try:
            code = _c.ErrorCode(code)
        except ValueError:
            code = int(code)
        self.code = code
        self.dev_err = int(dev_err)
        msg = _c.op_status_error(self.code)
        msg = msg.decode() if msg else "unknown opends error"
        super().__init__("%s (code %d, dev_err %d)" % (msg, self.code, self.dev_err))


def check(err):
    """Raise OpenDSError for a failed opends_error_t."""
    if err.err != _c.ErrorCode.SUCCESS:
        raise OpenDSError(err.err, err.dev_err)


def io_error(ret):
    """The exception for a negative read or write result. -1 is an I/O
    error, as cuFile reports one. Any other value is a negated code."""
    if ret == -1:
        return OSError(errno.EIO, "I/O error")
    return OpenDSError(-ret)


def io_result(ret):
    """Byte count of a read or write result; raise for a negative one."""
    if ret < 0:
        raise io_error(ret)
    return ret


# ---------------------------------------------------------------------------
# Driver lifecycle, as cuFileDriverOpen/Close: Driver objects open and close
# the C driver explicitly and are counted. Files, buffers, streams and batches
# need an open Driver and do not keep it open.
# ---------------------------------------------------------------------------

# Reentrant: a host may call cleanup() from a signal handler while the same
# thread holds the lock.
_lock = threading.RLock()
_driver_refs = 0


def _acquire():
    global _driver_refs
    with _lock:
        if _driver_refs == 0:
            check(_c.driver_open())
        _driver_refs += 1


def _release():
    global _driver_refs
    with _lock:
        if _driver_refs == 0:
            return
        _driver_refs -= 1
        if _driver_refs == 0:
            check(_c.driver_close())


def require_driver():
    if _driver_refs == 0:
        raise OpenDSError(_c.ErrorCode.DRIVER_NOT_INITIALIZED)


def is_open():
    return _driver_refs > 0


# _cleaning guards against reentrancy: a host's signal handler that calls
# cleanup() can run while cleanup is inside the slow driver_close on the same
# thread. A reentrant call is a no-op so the first completes; reset in finally
# so a later driver lifecycle can clean up too.
_cleaning = False


@atexit.register
def cleanup():
    """Close the driver if it is still open.

    Idempotent. Runs at exit. A host may call it from a signal handler.
    """
    global _driver_refs, _cleaning
    if _cleaning:
        return
    _cleaning = True
    try:
        with _lock:
            if _driver_refs > 0:
                _c.driver_close()
                _driver_refs = 0
    finally:
        _cleaning = False


class Driver:
    """The explicit open and close, as cufile.CuFileDriver. Required before
    any file, buffer, stream or batch call."""

    def __init__(self):
        self._open = False
        self.open()

    def open(self):
        if not self._open:
            _acquire()
            self._open = True

    def close(self):
        if self._open:
            self._open = False
            _release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


DriverProperties = collections.namedtuple(
    "DriverProperties", [name for name, _ in _c.DsDrvProps._fields_]
)


def get_version():
    major, minor, patch = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    check(
        _c.get_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
    )
    return (major.value, minor.value, patch.value)


def get_properties():
    props = _c.DsDrvProps()
    require_driver()
    check(_c.driver_get_properties(ctypes.byref(props)))
    return DriverProperties(*(getattr(props, f) for f in DriverProperties._fields))


def use_count():
    """Number of registered file handles."""
    return int(_c.use_count())


def set_max_direct_io_size(size):
    require_driver()
    check(_c.driver_set_max_direct_io_size(int(size)))
