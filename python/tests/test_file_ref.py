# SPDX-License-Identifier: BSD-3-Clause
"""Smoke test for the OpenDS Python bindings against the ref backend.

Dependency-free: uses opends_alloc-backed HostBuffers, so it runs on
any host without a GPU. Prints "all ok" on success.
"""

import ctypes
import mmap
import os
import tempfile

import pytest

import opends


def _payload(n):
    return bytes((i * 31 + 7) % 251 for i in range(n))


def test_version(driver):
    assert opends.get_version() == (0, 3, 0)


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

        dst = mmap.mmap(-1, n)
        base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(dst)))
        opends.register_buffer(base, n)
        with opends.OpenDSFile(path, "r") as f:
            assert f.read_sync(base, size=half, file_offset=0, dev_offset=0) == half
            n = f.read_sync(base, size=half, file_offset=half, dev_offset=half)
            assert n == half
        assert dst[:] == payload
        opends.deregister_buffer(base)
    finally:
        os.unlink(path)


def test_registers_an_open_descriptor(driver, tmp_path):
    n = 4096
    payload = _payload(n)
    path = tmp_path / "fd.bin"
    path.write_bytes(payload)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    dst = opends.alloc(n)
    with opends.OpenDSFile(fd) as f:
        assert f.fileno() == fd
        assert f.read_sync(dst, size=n) == n
    assert bytes(dst.as_ctypes()) == payload
    with pytest.raises(OSError):
        os.fstat(fd)


def test_closefd_false_leaves_the_descriptor_open(driver, tmp_path):
    path = tmp_path / "fd.bin"
    path.write_bytes(_payload(4096))
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        with opends.OpenDSFile(fd, closefd=False) as f:
            assert f.fileno() == fd
        os.fstat(fd)
        with pytest.raises(ValueError):
            opends.OpenDSFile(str(path), closefd=False)
    finally:
        os.close(fd)


@pytest.mark.parametrize("closefd", [True, False])
def test_failed_registration_closes_only_an_owned_descriptor(
    driver, tmp_path, closefd
):
    path = tmp_path / "fd.bin"
    path.write_bytes(_payload(4096))
    fd = os.open(path, os.O_RDONLY)  # no O_DIRECT: the backend refuses it
    with pytest.raises(opends.OpenDSError) as info:
        opends.OpenDSFile(fd, closefd=closefd)
    assert info.value.code == opends.ErrorCode.DIO_NOT_SET
    if closefd:
        with pytest.raises(OSError):
            os.fstat(fd)
    else:
        os.close(fd)


if __name__ == "__main__":
    with opends.Driver() as drv:
        test_version(drv)
        test_roundtrip(drv)
        test_scatter_offsets(drv)
        test_raw_pointer_with_offset(drv)
    print("all ok")
