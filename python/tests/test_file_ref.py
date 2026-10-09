# SPDX-License-Identifier: BSD-3-Clause
"""Smoke test for the OpenDS Python bindings against the ref backend.

Dependency-free: uses opends_alloc-backed HostBuffers, so it runs on
any host without a GPU. Prints "all ok" on success.
"""

import ctypes
import os
import tempfile

import opends


def _payload(n):
    return bytes((i * 31 + 7) % 251 for i in range(n))


def test_version(driver):
    assert opends.get_version() == (0, 1, 0)


def test_roundtrip(driver):
    n = 4096
    payload = _payload(n)

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        src = opends.alloc(n)
        ctypes.memmove(src.as_ctypes(), payload, n)
        with opends.OpenDSFile(path, "w") as f:
            assert f.write_sync(src, size=n) == n

        dst = opends.alloc(n)
        with opends.OpenDSFile(path, "r") as f:
            assert f.read_sync(dst, size=n) == n
        assert bytes(dst.as_ctypes()) == payload
    finally:
        os.unlink(path)


def test_scatter_offsets(driver):
    n = 4096
    payload = _payload(n)
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        src = opends.alloc(n)
        ctypes.memmove(src.as_ctypes(), payload, n)
        with opends.OpenDSFile(path, "w") as f:
            f.write_sync(src, size=n)

        dst = opends.alloc(n)
        with opends.OpenDSFile(path, "r") as f:
            assert f.read_sync(dst, size=2048, file_offset=2048, dev_offset=0) == 2048
        assert bytes(dst.as_ctypes())[:2048] == payload[2048:]
    finally:
        os.unlink(path)


def test_raw_pointer_with_offset(driver):
    # cuFile/LMCache pattern: register one base buffer, then I/O through a
    # bare pointer plus dev_offset. Halves are written and read back at
    # distinct offsets into the same registered allocation.
    n = 4096
    half = n // 2
    payload = _payload(n)
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        src = opends.alloc(n)
        ctypes.memmove(src.as_ctypes(), payload, n)
        with opends.OpenDSFile(path, "w") as f:
            f.write_sync(src, size=n)

        dst = opends.alloc(n)
        base = ctypes.c_void_p(dst.ptr)
        opends.register_buffer(base, n)
        with opends.OpenDSFile(path, "r") as f:
            assert f.read_sync(base, size=half, file_offset=0, dev_offset=0) == half
            n = f.read_sync(base, size=half, file_offset=half, dev_offset=half)
            assert n == half
        assert bytes(dst.as_ctypes()) == payload
        opends.deregister_buffer(base)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    with opends.Driver() as drv:
        test_version(drv)
        test_roundtrip(drv)
        test_scatter_offsets(drv)
        test_raw_pointer_with_offset(drv)
    print("all ok")
