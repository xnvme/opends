# SPDX-License-Identifier: BSD-3-Clause
"""Python bindings against the aisio backend on a live HOMI stack.

Reads the pattern file written by test_sync_read_prep through every mode
into device memory, copies it back with cudart and compares with a kernel
read; then writes a scratch file through sync, async and batch and reads it
back P2P and through the kernel. Skips unless OPENDS_BACKEND is aisio,
OPENDS_HOMI_MNT names the mount and the CUDA libraries load. Runs under
pytest or on its own, which needs no pytest on the target:

    OPENDS_BACKEND=aisio OPENDS_XAL_SHM=/xal_dev0 OPENDS_HOMI_MNT=/mnt/datasets \\
        python3 tests/test_aisio_cuda.py
"""

import ctypes
import mmap
import os
import sys
from ctypes import POINTER, byref, c_int, c_size_t, c_void_p

import opends

try:
    import pytest
except ImportError:  # standalone run on a target without pytest
    pytest = None

PAGE = 4096
MNT = os.environ.get("OPENDS_HOMI_MNT")
PATTERN = os.path.join(MNT, "opends_tests", "sync_read_pattern.bin") if MNT else None
SCRATCH = os.path.join(MNT, "opends_tests", "py_write_scratch.bin") if MNT else None
H2D, D2H = 1, 2


def unavailable():
    """Why this module cannot run here, or None."""
    if os.environ.get("OPENDS_BACKEND", "aisio") != "aisio":
        return "OPENDS_BACKEND is not aisio"
    if not MNT:
        return "OPENDS_HOMI_MNT is not set"
    if not os.path.exists(PATTERN):
        return "%s is missing; run test_sync_read_prep" % PATTERN
    try:
        Cuda.load()
    except OSError as exc:
        return "CUDA libraries do not load: %s" % exc
    return None


class Cuda:
    """libcuda and cudart through ctypes: a current context and device copies."""

    @staticmethod
    def load():
        cu = ctypes.CDLL("libcuda.so.1")
        names = (
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
        )
        for name in names:
            try:
                return cu, ctypes.CDLL(name)
            except OSError as exc:
                last = exc
        raise last

    def __init__(self):
        self.cu, self.rt = self.load()
        cu, rt = self.cu, self.rt
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        cu.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
        cu.cuCtxSetCurrent.argtypes = [c_void_p]
        cu.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        rt.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        rt.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
        rt.cudaFree.argtypes = [c_void_p]
        rt.cudaMemset.argtypes = [c_void_p, c_int, c_size_t]
        rt.cudaDeviceSynchronize.argtypes = []
        rt.cudaStreamCreate.argtypes = [POINTER(c_void_p)]
        rt.cudaStreamSynchronize.argtypes = [c_void_p]
        rt.cudaStreamDestroy.argtypes = [c_void_p]
        self.ok(cu.cuInit(0), "cuInit")
        dev = c_int()
        self.ok(cu.cuDeviceGet(byref(dev), 0), "cuDeviceGet")
        self.ctx = c_void_p()
        rc = cu.cuDevicePrimaryCtxRetain(byref(self.ctx), dev)
        self.ok(rc, "cuDevicePrimaryCtxRetain")
        self.ok(cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    @staticmethod
    def ok(rc, what):
        assert rc == 0, "%s: cuda rc %d" % (what, rc)

    def current(self):
        ctx = c_void_p()
        self.cu.cuCtxGetCurrent(byref(ctx))
        return ctx.value

    def d2h(self, ptr, n):
        host = (ctypes.c_char * n)()
        self.ok(self.rt.cudaMemcpy(host, ptr, n, D2H), "cudaMemcpy D2H")
        return bytes(host)

    def h2d(self, ptr, data):
        self.ok(self.rt.cudaMemcpy(ptr, data, len(data), H2D), "cudaMemcpy H2D")

    def zero(self, ptr, n):
        # cudaMemset is asynchronous and the NVMe DMA into the buffer is not
        # stream-ordered, so wait for it before submitting a read.
        self.ok(self.rt.cudaMemset(ptr, 0, n), "cudaMemset")
        self.ok(self.rt.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def stream(self):
        stream = c_void_p()
        self.ok(self.rt.cudaStreamCreate(byref(stream)), "cudaStreamCreate")
        return stream

    def sync(self, stream):
        self.ok(self.rt.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def malloc(self, n):
        ptr = c_void_p()
        self.ok(self.rt.cudaMalloc(byref(ptr), n), "cudaMalloc")
        return ptr


def cases(size):
    return [
        (0, PAGE),
        (PAGE, 2 * PAGE),
        (size - PAGE, PAGE),
        (0, 700),
        (2 * PAGE, PAGE + 300),
        (4, 700),
        (PAGE + 4, 2 * PAGE),
        (1024 + 64, 5000),
        (2 * PAGE + 516, 300),
        (PAGE - 4, 8),
    ]


def read_pattern():
    with open(PATTERN, "rb") as fh:
        return fh.read()


def check_properties():
    props = opends.get_properties()
    assert props.max_direct_io_size > 0
    assert opends.use_count() == 0


def check_sync(f, cuda, expected):
    size = len(expected)
    dst = opends.alloc(size)
    cuda.zero(dst.ptr, size)
    before = cuda.current()
    assert f.read_sync(dst, size=size) == size
    assert cuda.d2h(dst.ptr, size) == expected
    assert cuda.current() == before, "the CUDA context was not preserved"
    for off, n in cases(size):
        cuda.zero(dst.ptr, n)
        assert f.read_sync(dst, size=n, file_offset=off) == n, (off, n)
        assert cuda.d2h(dst.ptr, n) == expected[off : off + n], (off, n)
    dst.free()


def check_async(f, cuda, expected):
    size = len(expected)
    chunks = size // PAGE
    dst = opends.alloc(size)
    cuda.zero(dst.ptr, size)
    futs = [
        f.read_async(dst, size=PAGE, file_offset=i * PAGE, dev_offset=i * PAGE)
        for i in range(chunks)
    ]
    assert all(fut.result() == PAGE for fut in reversed(futs))
    assert cuda.d2h(dst.ptr, size) == expected
    assert futs[0].result() == PAGE and futs[0].done

    bufs = [opends.alloc(n) for _, n in cases(size)]
    for buf, (_, n) in zip(bufs, cases(size)):
        cuda.zero(buf.ptr, n)
    futs = [
        f.read_async(buf, size=n, file_offset=off)
        for buf, (off, n) in zip(bufs, cases(size))
    ]
    for fut, buf, (off, n) in zip(futs, bufs, cases(size)):
        assert fut.result() == n, (off, n)
        assert cuda.d2h(buf.ptr, n) == expected[off : off + n], (off, n)
    for buf in bufs:
        buf.free()
    dst.free()


def check_stream(f, cuda, expected):
    size = len(expected)
    chunks = size // PAGE
    stream = cuda.stream()
    opends.register_stream(stream)
    dst = opends.alloc(size)
    try:
        cuda.zero(dst.ptr, size)
        op = f.read_stream(dst, size=size, stream=stream)
        cuda.sync(stream)
        assert op.done
        assert op.result() == size
        assert cuda.d2h(dst.ptr, size) == expected

        cuda.zero(dst.ptr, size)
        ops = [
            f.read_stream(
                dst, size=PAGE, file_offset=i * PAGE, dev_offset=i * PAGE, stream=stream
            )
            for i in range(chunks)
        ]
        cuda.sync(stream)
        assert all(op.result() == PAGE for op in ops)
        assert cuda.d2h(dst.ptr, size) == expected

        cuda.zero(dst.ptr, size)
        op = f.read_stream(dst, size=PAGE + 300, file_offset=2 * PAGE, stream=stream)
        cuda.sync(stream)
        assert op.result() == PAGE + 300
        assert cuda.d2h(dst.ptr, PAGE + 300) == expected[2 * PAGE : 3 * PAGE + 300]
    finally:
        opends.deregister_stream(stream)
        cuda.rt.cudaStreamDestroy(stream)
        dst.free()


def check_batch(f, cuda, expected):
    size = len(expected)
    chunks = size // PAGE
    dst = opends.alloc(size)
    cuda.zero(dst.ptr, size)
    with opends.Batch(chunks) as batch:
        batch.submit(
            opends.BatchOp(
                f, dst, size=PAGE, file_offset=i * PAGE, dev_offset=i * PAGE, cookie=i
            )
            for i in range(chunks)
        )
        events = batch.get_status(min_nr=chunks, timeout=30)
        assert sorted(ev.cookie for ev in events) == list(range(chunks))
        assert all(ev.status is opends.Status.COMPLETE for ev in events)
        assert all(ev.ret == PAGE for ev in events)
        assert cuda.d2h(dst.ptr, size) == expected
        assert batch.get_status(min_nr=0, timeout=0) == []

        batch.submit(opends.BatchOp(f, dst, size=PAGE) for _ in range(chunks))
        try:
            batch.submit([opends.BatchOp(f, dst, size=PAGE)])
        except opends.OpenDSError as exc:
            assert exc.code is opends.ErrorCode.BATCH_FULL
        else:
            raise AssertionError("an overfilled batch was accepted")
        assert len(batch.get_status(min_nr=chunks, timeout=30)) == chunks
        assert batch.inflight == 0
    dst.free()


def check_external_buffer(f, cuda, expected):
    # A caller-owned cudaMalloc buffer registered up front, as cuFileBufRegister.
    size = len(expected)
    ext = cuda.malloc(size)
    try:
        opends.register_buffer(ext, size)
        cuda.zero(ext.value, size)
        assert f.read_sync(ext, size=size) == size
        assert cuda.d2h(ext.value, size) == expected
        opends.deregister_buffer(ext)
    finally:
        cuda.rt.cudaFree(ext)


def check_dio_not_set():
    try:
        opends.OpenDSFile(PATTERN, "r", use_direct_io=False)
    except opends.OpenDSError as exc:
        assert exc.code is opends.ErrorCode.DIO_NOT_SET
    else:
        raise AssertionError("a buffered fd was accepted")


def check_write_roundtrip(cuda, size):
    if os.path.exists(SCRATCH):
        os.unlink(SCRATCH)
    image = bytes((i * 7 + 0x5A) & 0xFF for i in range(size))
    quarter = size // 4
    src = opends.alloc(size)
    cuda.h2d(src.ptr, image)
    try:
        with opends.OpenDSFile(SCRATCH, "rw") as f:
            assert f.write_sync(src, size=quarter) == quarter
            assert os.fstat(f.fileno()).st_size == quarter
            fut = f.write_async(
                src, size=quarter, file_offset=quarter, dev_offset=quarter
            )
            assert fut.result() == quarter
            with opends.Batch(2) as batch:
                batch.submit(
                    opends.BatchOp(
                        f,
                        src,
                        size=quarter,
                        file_offset=k * quarter,
                        dev_offset=k * quarter,
                        opcode=opends.WRITE,
                        cookie=k,
                    )
                    for k in (2, 3)
                )
                events = batch.get_status(min_nr=2, timeout=60)
                assert all(ev.status is opends.Status.COMPLETE for ev in events)
                assert all(ev.ret == quarter for ev in events)
            assert os.fstat(f.fileno()).st_size == size
            dst = opends.alloc(size)
            cuda.zero(dst.ptr, size)
            # P2P read-back; waits out the xal reindex after the writes.
            assert f.read_sync(dst, size=size) == size
            assert cuda.d2h(dst.ptr, size) == image
            dst.free()
        fd = os.open(SCRATCH, os.O_RDONLY | os.O_DIRECT)
        try:
            host = mmap.mmap(-1, size)
            assert os.readv(fd, [host]) == size
            assert host[:size] == image
        finally:
            os.close(fd)
    finally:
        src.free()
        if os.path.exists(SCRATCH):
            os.unlink(SCRATCH)


if pytest is not None:

    @pytest.fixture(scope="module")
    def cuda():
        reason = unavailable()
        if reason:
            pytest.skip(reason)
        return Cuda()

    @pytest.fixture(scope="module")
    def expected(cuda):
        return read_pattern()

    @pytest.fixture
    def f(driver, cuda):
        with opends.OpenDSFile(PATTERN, "r") as f:
            yield f

    def test_properties(driver, cuda):
        check_properties()

    def test_sync(f, cuda, expected):
        check_sync(f, cuda, expected)

    def test_async(f, cuda, expected):
        check_async(f, cuda, expected)

    def test_stream(f, cuda, expected):
        check_stream(f, cuda, expected)

    def test_batch(f, cuda, expected):
        check_batch(f, cuda, expected)

    def test_external_buffer(f, cuda, expected):
        check_external_buffer(f, cuda, expected)

    def test_dio_not_set(driver, cuda):
        check_dio_not_set()

    def test_write_roundtrip(driver, cuda, expected):
        check_write_roundtrip(cuda, len(expected))


if __name__ == "__main__":
    reason = unavailable()
    if reason:
        sys.exit("cannot run: " + reason)
    cuda = Cuda()
    expected = read_pattern()
    with opends.Driver():
        check_properties()
        print("%-18s ok" % "properties")
        with opends.OpenDSFile(PATTERN, "r") as f:
            for check in (
                check_sync,
                check_async,
                check_stream,
                check_batch,
                check_external_buffer,
            ):
                check(f, cuda, expected)
                print("%-18s ok" % check.__name__[6:])
        check_dio_not_set()
        print("%-18s ok" % "dio_not_set")
        check_write_roundtrip(cuda, len(expected))
        print("%-18s ok" % "write_roundtrip")
    print("all ok")
