# SPDX-License-Identifier: BSD-3-Clause
"""Stream bindings against the ref backend.

The ref backend runs stream I/O synchronously, so the StreamOp result is
readable right after the call. Dependency-free (HostBuffer only).
"""

import ctypes
import errno

import pytest

import opends
from opends import stream

PAGE = 4096


def _pattern(n):
    return bytes((i * 31 + 7) % 251 for i in range(n))


def test_roundtrip(driver, tmp_path):
    path = str(tmp_path / "stream.bin")
    payload = _pattern(2 * PAGE)
    src = opends.alloc(2 * PAGE)
    ctypes.memmove(src.as_ctypes(), payload, 2 * PAGE)
    with opends.OpenDSFile(path, "w") as f:
        op = f.write_stream(src, size=2 * PAGE, stream=0)
        assert isinstance(op, opends.StreamOp)
        assert op.result() == 2 * PAGE
        assert op.nbytes == 2 * PAGE

    dst = opends.alloc(2 * PAGE)
    with opends.OpenDSFile(path, "r") as f:
        op = f.read_stream(dst, size=PAGE, file_offset=PAGE, dev_offset=PAGE)
        assert op.result() == PAGE
    assert bytes(dst.as_ctypes())[PAGE:] == payload[PAGE:]


def test_register_deregister(driver):
    opends.register_stream(0x1234)
    opends.deregister_stream(0x1234)


def test_before_the_stream_ran():
    op = opends.StreamOp(10, 0, 0, None)
    assert op.nbytes == 0
    assert op.result() == 0


def test_failed_operation():
    op = opends.StreamOp(10, 0, 0, None)
    # ref reports a failed stream I/O at submission, so store the result
    # the way a GPU backend does when the stream reaches the operation
    op._bytes.value = -opends.ErrorCode.FS_SETUP_ERROR
    assert op.nbytes == -opends.ErrorCode.FS_SETUP_ERROR
    with pytest.raises(opends.OpenDSError) as info:
        op.result()
    assert info.value.code is opends.ErrorCode.FS_SETUP_ERROR


def test_io_error():
    op = opends.StreamOp(10, 0, 0, None)
    op._bytes.value = -1  # cuFile's I/O error
    assert op.nbytes == -1
    with pytest.raises(OSError) as info:
        op.result()
    assert info.value.errno == errno.EIO


class _TorchLike:
    cuda_stream = 0xBEEF


class _CupyLike:
    ptr = 0xCAFE


def test_stream_handle_conversion(driver):
    assert stream.stream_handle(None) == 0
    assert stream.stream_handle(7) == 7
    assert stream.stream_handle(ctypes.c_void_p(9)) == 9
    assert stream.stream_handle(_TorchLike()) == 0xBEEF
    assert stream.stream_handle(_CupyLike()) == 0xCAFE
    with pytest.raises(TypeError):
        stream.stream_handle(object())
