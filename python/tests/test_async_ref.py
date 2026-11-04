# SPDX-License-Identifier: BSD-3-Clause
"""Async bindings against the ref backend.

Mirrors tests/test_async_read.h: a burst of reads at mixed offsets and
sizes, awaited in reverse submission order, each verified against the
pattern. Dependency-free (HostBuffer only).
"""

import ctypes

import pytest

import opends

PAGE = 4096
FILE_PAGES = 16
FILE_SIZE = FILE_PAGES * PAGE

CASES = [
    (0, PAGE),
    (0, FILE_SIZE),
    (PAGE, 2 * PAGE),
    (2 * PAGE, PAGE),
    (4 * PAGE, 4 * PAGE),
    (8 * PAGE, 8 * PAGE),
    ((FILE_PAGES - 1) * PAGE, PAGE),
    (0, 700),
    (2 * PAGE, PAGE + 300),
    (0, FILE_SIZE // 2),
    (4, 700),
    (PAGE + 4, 2 * PAGE),
    (1024 + 64, 5000),
    (2 * PAGE + 516, 300),
    (PAGE - 1, 2),
]


def _pattern(n):
    return bytes((i * 31 + 7) % 251 for i in range(n))


def _host(buf, n):
    return bytes(buf.as_ctypes())[:n]


def _write_pattern(path):
    payload = _pattern(FILE_SIZE)
    src = opends.alloc(FILE_SIZE)
    ctypes.memmove(src.as_ctypes(), payload, FILE_SIZE)
    with opends.OpenDSFile(path, "w") as f:
        fut = f.write_async(src, size=FILE_SIZE)
        assert isinstance(fut, opends.Future)
        assert fut.result() == FILE_SIZE
    return payload


def test_reads_awaited_in_reverse(driver, tmp_path):
    path = str(tmp_path / "pattern.bin")
    payload = _write_pattern(path)
    bufs = [opends.alloc(size) for _, size in CASES]
    with opends.OpenDSFile(path, "r") as f:
        futs = [
            f.read_async(buf, size=size, file_offset=off)
            for buf, (off, size) in zip(bufs, CASES)
        ]
        for fut, buf, (off, size) in reversed(list(zip(futs, bufs, CASES))):
            assert fut.result() == size
            assert fut.done
            assert _host(buf, size) == payload[off : off + size]
            # Awaiting a completed future again returns the same result.
            assert fut.result() == size


def test_scatter_writes_then_read(driver, tmp_path):
    path = str(tmp_path / "scatter.bin")
    payload = _pattern(2 * PAGE)
    src = opends.alloc(2 * PAGE)
    ctypes.memmove(src.as_ctypes(), payload, 2 * PAGE)
    with opends.OpenDSFile(path, "w") as f:
        hi = f.write_async(src, size=PAGE, file_offset=PAGE, dev_offset=PAGE)
        lo = f.write_async(src, size=PAGE, file_offset=0, dev_offset=0)
        assert lo.result() == PAGE
        assert hi.result() == PAGE

    dst = opends.alloc(2 * PAGE)
    with opends.OpenDSFile(path, "r") as f:
        assert f.read_sync(dst, size=2 * PAGE) == 2 * PAGE
    assert _host(dst, 2 * PAGE) == payload


def test_size_defaults_to_buffer_extent(driver, tmp_path):
    path = str(tmp_path / "pattern.bin")
    payload = _write_pattern(path)
    dst = opends.alloc(PAGE)
    with opends.OpenDSFile(path, "r") as f:
        fut = f.read_async(dst, file_offset=3 * PAGE, dev_offset=PAGE // 2)
        assert fut.result() == PAGE // 2
    assert _host(dst, PAGE)[PAGE // 2 :] == payload[3 * PAGE : 3 * PAGE + PAGE // 2]


def test_pending_future_needs_an_open_driver():
    # A future the backend never completed must not spin once the driver is
    # gone, and collecting it must not hang either. This one was never
    # submitted, so collect it here, while no driver is open; a later
    # collection under an open driver would await it forever.
    import gc

    fut = opends.Future(opends.cdll.DsAsyncFuture(), None)
    assert not fut.done
    with pytest.raises(opends.OpenDSError) as info:
        fut.result()
    assert info.value.code is opends.ErrorCode.DRIVER_NOT_INITIALIZED
    del info, fut
    gc.collect()
