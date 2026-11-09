# SPDX-License-Identifier: BSD-3-Clause
"""Buffer argument checks against the ref backend."""

import ctypes
import mmap

import pytest

import opends
from opends import cdll


def _address(buf):
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def test_transfer_must_fit_the_buffer(driver, tmp_path):
    buf = opends.alloc(4096)
    with opends.OpenDSFile(str(tmp_path / "a"), "w") as f:
        assert f.write_sync(buf) == 4096
        assert f.write_sync(buf, dev_offset=1024) == 3072
        with pytest.raises(ValueError):
            f.write_sync(buf, dev_offset=4097)
        with pytest.raises(ValueError):
            f.write_sync(buf, size=4096, dev_offset=1)
        with pytest.raises(ValueError):
            f.write_sync(buf, dev_offset=-1)
        with pytest.raises(ValueError):
            f.write_async(buf, size=8192)
        with opends.Batch(1) as batch, pytest.raises(ValueError):
            batch.submit([opends.BatchOp(f, buf, size=1, dev_offset=4096)])


def test_io_registers_nothing(driver, tmp_path):
    allocated = opends.alloc(4096)
    external = mmap.mmap(-1, 4096)
    with opends.OpenDSFile(str(tmp_path / "a"), "w") as f:
        assert f.write_sync(allocated) == 4096
        # ref takes unregistered host memory, a GPU backend refuses it
        assert f.write_sync(external) == 4096
    for ptr in (allocated.ptr, _address(external)):
        err = cdll.buf_deregister(ptr)
        assert err.err == opends.ErrorCode.MEMORY_NOT_REGISTERED
    allocated.free()


def test_bare_pointer_needs_a_size(driver, tmp_path):
    buf = mmap.mmap(-1, 4096)
    base = ctypes.c_void_p(_address(buf))
    opends.register_buffer(base, 4096)
    with opends.OpenDSFile(str(tmp_path / "a"), "w") as f:
        with pytest.raises(ValueError):
            f.write_sync(base)
        assert f.write_sync(base, size=4096) == 4096
