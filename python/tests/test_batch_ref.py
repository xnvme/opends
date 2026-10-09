# SPDX-License-Identifier: BSD-3-Clause
"""Batch bindings against the ref backend.

Mirrors tests/test_batch_read.h: one batch of reads at mixed offsets and
sizes, reaped with a single get_status and verified by cookie; a repoll
delivers nothing; a second round reuses the freed slots; an overfill
raises BATCH_FULL. Dependency-free (HostBuffer only).
"""

import ctypes

import pytest

import opends
from opends import cdll as _c

PAGE = 4096
FILE_PAGES = 16
FILE_SIZE = FILE_PAGES * PAGE

CASES = [
    (0, PAGE),
    (PAGE, 2 * PAGE),
    (4 * PAGE, 4 * PAGE),
    (0, 700),
    (2 * PAGE, PAGE + 300),
    ((FILE_PAGES - 1) * PAGE, PAGE),
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
        assert f.write_sync(src, size=FILE_SIZE) == FILE_SIZE
    return payload


def test_abi_layout(driver):
    # Pin the struct layouts to the C ABI (x86-64 Linux offsets).
    assert ctypes.sizeof(_c.DsError) == 8
    assert ctypes.sizeof(_c.DsDrvProps) == 24
    assert ctypes.sizeof(_c.DsAsyncFuture) == 16
    assert _c.DsAsyncFuture.result.offset == 8
    assert ctypes.sizeof(_c.DsIoParams) == 64
    assert _c.DsIoParams.u.offset == 8
    assert _c.DsIoParams.fh.offset == 40
    assert _c.DsIoParams.opcode.offset == 48
    assert _c.DsIoParams.cookie.offset == 56
    assert ctypes.sizeof(_c.DsIoEvents) == 24
    assert _c.DsIoEvents.ret.offset == 16
    assert ctypes.sizeof(_c.Timespec) == 16


def _reap_round(batch, bufs, payload):
    events = batch.get_status(min_nr=len(CASES), max_nr=len(CASES) + 2)
    assert len(events) == len(CASES)
    seen = set()
    for ev in events:
        idx = ev.cookie["case"]
        assert idx not in seen
        seen.add(idx)
        off, size = CASES[idx]
        assert ev.status == opends.Status.COMPLETE
        assert ev.ret == size
        assert _host(bufs[idx], size) == payload[off : off + size]
    # Completions are delivered once: a repoll must be empty.
    assert batch.get_status(min_nr=0) == []


def test_two_rounds_and_overfill(driver, tmp_path):
    path = str(tmp_path / "pattern.bin")
    payload = _write_pattern(path)
    bufs = [opends.alloc(size) for _, size in CASES]
    with opends.OpenDSFile(path, "r") as f, opends.Batch(len(CASES) + 2) as batch:
        assert batch.capacity == len(CASES) + 2
        ops = [
            opends.BatchOp(f, buf, size=size, file_offset=off, cookie={"case": i})
            for i, (buf, (off, size)) in enumerate(zip(bufs, CASES))
        ]
        batch.submit(ops)
        assert batch.inflight == len(CASES)
        _reap_round(batch, bufs, payload)
        assert batch.inflight == 0

        # Delivered slots are free again, so a second full round fits.
        batch.submit(ops)
        # len(CASES) in flight leave two free slots, so three must not fit.
        with pytest.raises(opends.OpenDSError) as info:
            batch.submit(ops[:3])
        assert info.value.code == opends.ErrorCode.BATCH_FULL
        assert batch.inflight == len(CASES)
        _reap_round(batch, bufs, payload)


def test_write_then_read(driver, tmp_path):
    path = str(tmp_path / "scatter.bin")
    payload = _pattern(2 * PAGE)
    src = opends.alloc(2 * PAGE)
    ctypes.memmove(src.as_ctypes(), payload, 2 * PAGE)
    with opends.OpenDSFile(path, "w") as f, opends.Batch(2) as batch:
        batch.submit(
            [
                opends.BatchOp(
                    f, src, size=PAGE, opcode=opends.WRITE, cookie="lo"
                ),
                opends.BatchOp(
                    f,
                    src,
                    size=PAGE,
                    file_offset=PAGE,
                    dev_offset=PAGE,
                    opcode=opends.WRITE,
                    cookie="hi",
                ),
            ]
        )
        events = batch.get_status(min_nr=2)
        assert sorted(ev.cookie for ev in events) == ["hi", "lo"]
        assert all(ev.status is opends.Status.COMPLETE for ev in events)
        assert all(ev.ret == PAGE for ev in events)

    dst = opends.alloc(2 * PAGE)
    with opends.OpenDSFile(path, "r") as f, opends.Batch(1) as batch:
        batch.submit([opends.BatchOp(f, dst)])  # size defaults to the extent
        [ev] = batch.get_status()
        assert ev == opends.BatchEvent(None, opends.Status.COMPLETE, 2 * PAGE)
    assert _host(dst, 2 * PAGE) == payload


def test_cancel_reports_canceled(driver, tmp_path):
    path = str(tmp_path / "pattern.bin")
    _write_pattern(path)
    dst = opends.alloc(PAGE)
    with opends.OpenDSFile(path, "r") as f, opends.Batch(1) as batch:
        batch.submit([opends.BatchOp(f, dst, size=PAGE, cookie=42)])
        batch.cancel()
        [ev] = batch.get_status(min_nr=1)
        assert ev == opends.BatchEvent(42, opends.Status.CANCELED, 0)
        assert batch.inflight == 0


def test_timeout_poll_returns_empty(driver):
    with opends.Batch(1) as batch:
        assert batch.get_status(min_nr=0, timeout=0.01) == []
        assert batch.get_status(min_nr=0, timeout=0) == []


def test_close_is_idempotent(driver):
    batch = opends.Batch(1)
    batch.close()
    batch.close()
    with pytest.raises(opends.OpenDSError):
        batch.get_status(min_nr=0)
