# SPDX-License-Identifier: BSD-3-Clause
"""The host owns its signals.

Opening a driver leaves the handlers of the process alone. A host that wants
the driver closed on a signal calls opends.cleanup() from its own handler, so
cleanup() keeps its name, is safe to re-enter and closes the driver when a
host's handler calls it.

Runs against the ref backend, so no GPU is needed.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time

import opends

# Child: a host with its own SIGTERM handler, which calls opends.cleanup().
# driver_close is wrapped so a close leaves proof on disk, and the handler
# leaves with os._exit so atexit cannot be what closed the driver.
_CHILD = r"""
import ctypes, os, signal
import opends
from opends import cdll

sentinel = os.environ["OPENDS_TEST_SENTINEL"]
ready = os.environ["OPENDS_TEST_READY"]

_orig_close = cdll.driver_close
def _close(*a, **k):
    with open(sentinel, "w") as fh:
        fh.write("closed"); fh.flush(); os.fsync(fh.fileno())
    return _orig_close(*a, **k)
cdll.driver_close = _close

def shutdown(signum, frame):
    opends.cleanup()
    os._exit(0)
signal.signal(signal.SIGTERM, shutdown)

driver = opends.Driver()
buf = opends.alloc(4096)
opends.register_buffer(ctypes.c_void_p(buf.ptr), 4096)

with open(ready, "w") as fh:
    fh.write("ready"); fh.flush(); os.fsync(fh.fileno())

while True:
    signal.pause()
"""


def test_driver_leaves_signals_alone():
    # The host decides what a signal does, and when the driver closes.
    signals = (signal.SIGTERM, signal.SIGINT)
    before = [signal.getsignal(sig) for sig in signals]
    with opends.Driver():
        during = [signal.getsignal(sig) for sig in signals]
    assert during == before, "opening a driver changed the signal handlers"


def test_host_handler_closes_driver():
    workdir = tempfile.mkdtemp(prefix="opends_sig_")
    sentinel = os.path.join(workdir, "closed")
    ready = os.path.join(workdir, "ready")
    child_py = os.path.join(workdir, "child.py")
    with open(child_py, "w") as fh:
        fh.write(_CHILD)

    env = dict(os.environ)
    env.setdefault("OPENDS_BACKEND", "ref")
    # Ensure the child imports the same opends package as this test.
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(opends.__file__)))
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENDS_TEST_SENTINEL"] = sentinel
    env["OPENDS_TEST_READY"] = ready

    proc = subprocess.Popen([sys.executable, child_py], env=env)
    try:
        for _ in range(200):  # up to ~10s for the child to open the driver
            if os.path.exists(ready):
                break
            if proc.poll() is not None:
                raise AssertionError("child exited before it was ready (rc=%s)" % proc.returncode)
            time.sleep(0.05)
        else:
            raise AssertionError("child never became ready")

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert proc.returncode == 0, "the host's SIGTERM handler did not run (rc=%s)" % proc.returncode
    assert os.path.exists(sentinel), (
        "driver_close did not run when the host's SIGTERM handler called "
        "opends.cleanup()"
    )


def test_cleanup_symbol_is_public():
    # External integrations call opends.cleanup() by name from their own
    # signal handlers (e.g. vLLM's EngineCore SIGTERM handler). Renaming it
    # makes that call fail and leaves the driver open. Guard the exact name.
    assert callable(getattr(opends, "cleanup", None)), (
        "opends.cleanup must exist and be callable. External signal handlers "
        "call it by this exact name to close the driver."
    )


def test_cleanup_reentrant_safe():
    # A host's signal handler can re-enter cleanup on the same thread while it
    # is inside the registry clear (buf_deregister can be slow). Without a
    # guard the reentrant call closes the driver under the outer clear and the
    # teardown can deadlock or double-close. Reproduce the re-entry by calling
    # cleanup from within buf_deregister and assert the teardown still
    # completes (does not hang).
    import ctypes
    import threading

    from opends import cdll, driver as d

    d._cleaning = False
    driver = opends.Driver()
    buf = opends.alloc(4096)
    opends.register_buffer(ctypes.c_void_p(buf.ptr), 4096)

    orig = cdll.buf_deregister

    def reenter(ptr):
        opends.cleanup()  # simulate a signal re-entering mid-clear
        return orig(ptr)

    done = threading.Event()

    def run_cleanup():
        opends.cleanup()
        done.set()

    cdll.buf_deregister = reenter
    try:
        threading.Thread(target=run_cleanup).start()
        assert done.wait(timeout=10), (
            "cleanup deadlocked on re-entry; the reentrancy guard is missing, "
            "so a second signal during teardown would leave the driver open"
        )
    finally:
        cdll.buf_deregister = orig
        d._cleaning = False
        driver.close()


if __name__ == "__main__":
    test_driver_leaves_signals_alone()
    test_cleanup_symbol_is_public()
    test_cleanup_reentrant_safe()
    test_host_handler_closes_driver()
    print("all ok")
