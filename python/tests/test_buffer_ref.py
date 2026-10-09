# SPDX-License-Identifier: BSD-3-Clause
"""Buffer argument checks against the ref backend."""

import ctypes

import pytest

import opends


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


def test_bare_pointer_needs_a_size(driver, tmp_path):
    buf = opends.alloc(4096)
    base = ctypes.c_void_p(buf.ptr)
    opends.register_buffer(base, 4096)
    with opends.OpenDSFile(str(tmp_path / "a"), "w") as f:
        with pytest.raises(ValueError):
            f.write_sync(base)
        assert f.write_sync(base, size=4096) == 4096
