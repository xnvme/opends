# SPDX-License-Identifier: BSD-3-Clause
"""Driver lifetime, errors and the CUDA context guard.

A Driver must be open before any file or buffer call, as with
cuFileDriverOpen.
"""

import atexit
import ctypes
import os
import signal
import threading

from . import cdll as _c

# ---------------------------------------------------------------------------
# CUDA context preservation. The aisio (upcie-cuda) backend creates its own
# CUDA context and leaves it current across native calls. A host framework
# (e.g. PyTorch/vLLM) creates its stream and event handles in its own primary
# context; if our calls leave a different context current, those handles fault
# with CUDA "invalid resource handle". Save the caller's current context and
# restore it around every native call that may enter the CUDA backend. No-op
# when libcuda is unavailable (CPU/ref backend) or no context is current.
# ---------------------------------------------------------------------------
try:
    _libcuda = ctypes.CDLL("libcuda.so")
    # Set explicit signatures: default int marshalling happens to work for
    # pointer-sized values on x86-64 but must not be assumed.
    _libcuda.cuCtxGetCurrent.restype = ctypes.c_int
    _libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _libcuda.cuCtxSetCurrent.restype = ctypes.c_int
    _libcuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
except OSError:
    _libcuda = None


class preserve_cuda_context:
    """Save/restore the caller's current CUDA context around native calls.

    No-op when libcuda is unavailable (CPU/ref backend) or no context is
    current. Only preserves an existing context; never calls cuInit and
    never creates one. Per-thread-correct: cuCtxGet/SetCurrent act on the
    calling thread.
    """

    def __enter__(self):
        self._ctx = None
        if _libcuda is not None:
            c = ctypes.c_void_p()
            if _libcuda.cuCtxGetCurrent(ctypes.byref(c)) == 0 and c.value:
                self._ctx = c
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            _libcuda.cuCtxSetCurrent(self._ctx)
        return False


class OpenDSError(Exception):
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


def io_result(ret):
    """Byte count of a read or write result; raise for a negated error."""
    if ret < 0:
        raise OpenDSError(-ret)
    return ret


# ---------------------------------------------------------------------------
# Driver lifecycle, as cuFileDriverOpen/Close: Driver objects open and close
# the C driver explicitly and are counted. Files and buffers need an open
# Driver and do not keep it open.
# ---------------------------------------------------------------------------

# Reentrant so a signal-driven cleanup can re-acquire while the same thread
# already holds the lock, instead of deadlocking.
_lock = threading.RLock()
_driver_refs = 0
# Run before the C driver closes, so state tied to it (the registration
# cache in the buffer module) empties with it.
_close_hooks = []


def on_close(hook):
    _close_hooks.append(hook)


def _acquire():
    global _driver_refs
    with _lock:
        if _driver_refs == 0:
            check(_c.driver_open())
        _driver_refs += 1
    _install_signal_handlers()


def _release():
    global _driver_refs
    with _lock:
        if _driver_refs == 0:
            return
        _driver_refs -= 1
        if _driver_refs == 0:
            for hook in _close_hooks:
                hook()
            check(_c.driver_close())


def require_driver():
    if _driver_refs == 0:
        raise OpenDSError(_c.ErrorCode.DRIVER_NOT_INITIALIZED)


# _cleaning guards against signal reentrancy: a second SIGTERM/SIGINT arriving
# while cleanup is inside the slow buf_deregister re-enters it on the same
# thread. A reentrant call is a no-op so the first completes; reset in finally
# so a later driver lifecycle can clean up too.
_cleaning = False


@atexit.register
def cleanup():
    """Close the driver if it is still open, releasing the GPU dma-buf.

    Idempotent. Runs at exit and on SIGTERM/SIGINT. A framework that installs
    its own SIGTERM handler shadows ours, and its graceful shutdown may never
    reach the code that closes the Driver, so it calls opends.cleanup() from
    that handler instead.
    """
    global _driver_refs, _cleaning
    if _cleaning:
        return
    _cleaning = True
    try:
        with _lock:
            if _driver_refs > 0:
                for hook in _close_hooks:
                    hook()
                _c.driver_close()
                _driver_refs = 0
    finally:
        _cleaning = False


# ---------------------------------------------------------------------------
# Backstop for standalone consumers: atexit does not run on signal death, so a
# SIGTERM/SIGINT with no handler skips driver_close and leaks the GPU dma-buf.
# Install a handler that runs cleanup and chains to the prior one.
# ---------------------------------------------------------------------------

_CLEANUP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
_prev_handlers = {}


def _signal_cleanup(signum, frame):
    cleanup()
    prev = _prev_handlers.get(signum, signal.SIG_DFL)
    if prev is None:  # handler installed outside Python; treat as default
        prev = signal.SIG_DFL
    if callable(prev):
        prev(signum, frame)
    else:
        # SIG_DFL / SIG_IGN: restore it and re-raise so the original
        # disposition (terminate for SIGTERM) still applies.
        signal.signal(signum, prev)
        os.kill(os.getpid(), signum)


def _install_signal_handlers():
    """Route SIGTERM/SIGINT through _signal_cleanup, chaining to whatever was
    installed before. Idempotent (skips signals we already own, so it never
    chains to itself). A no-op off the main thread, where signal.signal raises;
    atexit still covers a clean exit there.
    """
    for sig in _CLEANUP_SIGNALS:
        try:
            cur = signal.getsignal(sig)
            if cur is _signal_cleanup:
                continue
            signal.signal(sig, _signal_cleanup)
            _prev_handlers[sig] = cur
        except (ValueError, OSError):
            pass


class Driver:
    """The explicit open and close, as cufile.CuFileDriver. Required before
    any file or buffer call."""

    def __init__(self):
        self._open = False
        self.open()

    def open(self):
        if not self._open:
            with preserve_cuda_context():
                _acquire()
            self._open = True

    def close(self):
        if self._open:
            self._open = False
            with preserve_cuda_context():
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


def get_version():
    major, minor, patch = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    check(
        _c.get_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
    )
    return (major.value, minor.value, patch.value)
