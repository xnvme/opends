# SPDX-License-Identifier: BSD-3-Clause
"""Batch I/O: submit independent operations in one call, reap completions."""

import collections
import ctypes

from . import cdll as _c
from .buffer import io_args, registry
from .driver import check, preserve_cuda_context, require_driver

READ = _c.READ
WRITE = _c.WRITE
Status = _c.Status

BatchEvent = collections.namedtuple("BatchEvent", ["cookie", "status", "ret"])


class BatchOp:
    """One operation for Batch.submit."""

    __slots__ = ("file", "buf", "size", "file_offset", "dev_offset", "opcode", "cookie")

    def __init__(
        self,
        file,
        buf,
        size=None,
        file_offset=0,
        dev_offset=0,
        opcode=READ,
        cookie=None,
    ):
        self.file = file
        self.buf = buf
        self.size = size
        self.file_offset = file_offset
        self.dev_offset = dev_offset
        self.opcode = opcode
        self.cookie = cookie


def _status(value):
    try:
        return Status(value)
    except ValueError:
        return int(value)


def _timespec(timeout):
    if timeout is None:
        return None
    sec = int(timeout)
    nsec = int(round((timeout - sec) * 1e9))
    if nsec >= 1000000000:
        sec += 1
        nsec -= 1000000000
    return _c.Timespec(sec, nsec)


class Batch:
    def __init__(self, max_nr):
        self._handle = None
        self._capacity = int(max_nr)
        # slot id -> (cookie, buf); keeps buffers alive while in flight
        self._pending = {}
        self._next_slot = 1
        require_driver()
        with preserve_cuda_context():
            handle = ctypes.c_void_p()
            check(_c.batch_setup(ctypes.byref(handle), self._capacity))
        self._handle = handle

    @property
    def capacity(self):
        return self._capacity

    @property
    def inflight(self):
        return len(self._pending)

    def submit(self, ops, flags=0):
        ops = list(ops)
        params = (_c.DsIoParams * len(ops))()
        staged = {}
        with preserve_cuda_context():
            for p, op in zip(params, ops):
                if op.file.handle is None:
                    raise ValueError("file is closed")
                ptr, nbytes, size = io_args(op.buf, op.size, op.dev_offset)
                registry.ensure(ptr, nbytes)
                slot = self._next_slot
                self._next_slot += 1
                p.mode = _c.BATCH_MODE
                p.u.batch.dev_ptr_base = ptr
                p.u.batch.file_offset = op.file_offset
                p.u.batch.dev_ptr_offset = op.dev_offset
                p.u.batch.size = size
                p.fh = op.file.handle.value
                p.opcode = op.opcode
                p.cookie = slot
                staged[slot] = (op.cookie, op.buf)
            check(_c.batch_submit(self._handle, len(ops), params, flags))
        self._pending.update(staged)

    def get_status(self, min_nr=1, max_nr=None, timeout=None):
        """Wait for at least min_nr completions; return up to max_nr.

        timeout is in seconds, None waits without limit. Each completion
        is delivered once.
        """
        if max_nr is None:
            max_nr = self._capacity
        nr = ctypes.c_uint(max_nr)
        events = (_c.DsIoEvents * max_nr)()
        ts = _timespec(timeout)
        with preserve_cuda_context():
            check(
                _c.batch_get_status(
                    self._handle,
                    min_nr,
                    ctypes.byref(nr),
                    events,
                    None if ts is None else ctypes.byref(ts),
                )
            )
        out = []
        for ev in events[: nr.value]:
            cookie, _ = self._pending.pop(ev.cookie, (None, None))
            out.append(BatchEvent(cookie, _status(ev.status), int(ev.ret)))
        return out

    def cancel(self):
        with preserve_cuda_context():
            check(_c.batch_cancel(self._handle))

    def close(self):
        if self._handle is not None:
            with preserve_cuda_context():
                _c.batch_destroy(self._handle)
            self._handle = None
            self._pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
