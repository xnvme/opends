# SPDX-License-Identifier: BSD-3-Clause
"""Regression test: a SIGTERM-killed host must still run driver cleanup.

atexit does not run on signal death, so without a signal handler a SIGTERM
(how orchestrators tear a process down) skips opends' driver_close and leaks
the pinned GPU dma-buf until reboot. This guards the handler in driver.py so
a future refactor cannot silently drop it again (it has regressed before).

Runs against the ref backend, so no GPU is needed: it asserts that
driver_close fires on SIGTERM, which is backend-independent.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time

import opends

# Child: open the driver (which installs the signal handler), register a
# buffer, and wrap driver_close so a SIGTERM-driven close leaves proof on
# disk. Then wait for the parent's SIGTERM. If the handler is missing,
# atexit does not run under SIGTERM and the sentinel is never written.
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

driver = opends.Driver()
buf = opends.alloc(4096)
opends.register_buffer(ctypes.c_void_p(buf.ptr), 4096)

with open(ready, "w") as fh:
    fh.write("ready"); fh.flush(); os.fsync(fh.fileno())

while True:
    signal.pause()
"""


def test_sigterm_runs_driver_cleanup():
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
        for _ in range(200):  # up to ~10s for the child to arm the handler
            if os.path.exists(ready):
                break
            if proc.poll() is not None:
                raise AssertionError("child exited before arming (rc=%s)" % proc.returncode)
            time.sleep(0.05)
        else:
            raise AssertionError("child never became ready")

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert os.path.exists(sentinel), (
        "driver_close did not run on SIGTERM: the signal cleanup handler "
        "regressed, so the pinned GPU dma-buf would leak until reboot"
    )


def test_cleanup_symbol_is_public():
    # External integrations call opends.cleanup() by name from their own
    # signal handlers (e.g. vLLM's EngineCore SIGTERM handler, which shadows any
    # handler opends installs). Renaming it makes that call a silent no-op and
    # leaks the GPU dma-buf. Guard the exact name.
    assert callable(getattr(opends, "cleanup", None)), (
        "opends.cleanup must exist and be callable. External signal handlers "
        "call it by this exact name to free the GPU dma-buf."
    )


def test_cleanup_reentrant_safe():
    # A second SIGTERM/SIGINT can re-enter cleanup on the same thread while it
    # is inside the registry clear (buf_deregister can be slow). The registry
    # lock is non-reentrant, so without a guard the reentrant call self-deadlocks
    # and driver_close never runs, leaking the GPU buffer. Reproduce the re-entry
    # by calling cleanup from within buf_deregister and assert the teardown
    # still completes (does not hang).
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
            "so a second signal during teardown would leak the GPU buffer"
        )
    finally:
        cdll.buf_deregister = orig
        d._cleaning = False
        driver.close()


if __name__ == "__main__":
    test_cleanup_symbol_is_public()
    test_cleanup_reentrant_safe()
    test_sigterm_runs_driver_cleanup()
    print("all ok")
