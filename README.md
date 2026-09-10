# OpenDS

Open source accelerator direct storage. Vendor-neutral API modeled on
NVIDIA's cuFile (GDS), powered by aisio for high-throughput PCIe P2P DMA from
NVMe straight into GPU memory.

## Backends

- **Reference** (`libopends_ref`): POSIX `pread`/`pwrite` on host buffers. No
  external dependencies. Serves as a correctness baseline and template for
  hardware-specific backends.
- **GDS** (`libopends_gds`): Wraps NVIDIA cuFile for GPUDirect Storage. Buffers
  are GPU memory allocated with `cudaMalloc` and registered via
  `cuFileBufRegister`. Requires CUDA toolkit and the cuFile (GDS) library. Built
  conditionally when both are found.
- **aisio** (`libopends_aisio`): PCIe P2P DMA between NVMe and GPU memory via
  [xNVMe](https://xnvme.io)'s `upcie-cuda` backend (no filesystem or kernel
  `nvme` driver in the read data path). Based on
  [aisio](https://github.com/xnvme/aisio). A HOMI daemon owns the userspace NVMe
  controller, serves an I/O qpair per file, and resolves each file's device
  extents on demand (`homic_get_extents`, FIEMAP over the qublk-exported block
  device). Reads and writes are supported. Requires xNVMe, the CUDA toolkit, and
  the HOMI/qublk stack.

## aisio configuration

The aisio backend reads its configuration from environment variables at
`opends_driver_open`. Values that are out of range, or not a number, fail
the open.

- `OPENDS_HOMI_DEV` (required): The NVMe device the HOMI daemon owns (PCI BDF).
- `OPENDS_HOMI_SOCKET`: HOMI daemon socket. Default `/run/homi/homi.sock`.
- `OPENDS_AISIO_IO_THREADS`: Number of internal IO worker threads. Default 2.
  Driver open attaches one NVMe qpair per worker.
- `OPENDS_AISIO_QUEUE_DEPTH`: xNVMe queue depth per worker. Default 8.
- `OPENDS_AISIO_CPU_MASK`: Hex mask of CPUs for the workers (e.g. `0xf0`).
  Each worker gets a one-CPU affinity via `pthread_attr_setaffinity_np(3)`:
  worker i takes set bit i mod popcount, so a mask with fewer bits than
  `OPENDS_AISIO_IO_THREADS` pins more than one worker to a CPU. Unset or `0`
  leaves placement to the scheduler.
- `OPENDS_AISIO_IDLE_SPIN`: How long an idle IO worker keeps yielding after
  its last activity before it naps, in microseconds. Default 200. `0` naps at
  once. `busy` never yields or naps: the worker polls flat out and holds a CPU
  until the driver closes.
- `OPENDS_AISIO_ASSUME_ALIGNED_ONLY`: `1` declares that every read is
  LBA-aligned. Reads that start or end off an LBA boundary fail with
  `OPENDS_INVALID_VALUE`, and the stream path stops enqueueing the bounce
  kernel.

## Performance

_Commit `93fd019` (kernel `6.8.12-dmabuf`, NVMe `Samsung S4LV008[Pascal]`, GPU
`NVIDIA RTX 2000 Ada Generation`). `OPENDS_AISIO_IO_THREADS=2` and
`OPENDS_AISIO_QUEUE_DEPTH=8`._

| Dataset       | mode   | gds (MiB/s) | opends (MiB/s) |
|---------------|--------|-------------|----------------|
| filesize8gib  | sync   |        6520 |           6794 |
| filesize8gib  | stream |        2599 |           7036 |
| filesize8gib  | async  |           - |           7065 |
| tiktokish     | sync   |        4614 |           5869 |
| tiktokish     | stream |        5101 |           4957 |
| tiktokish     | async  |           - |           5005 |
| imagenetish   | sync   |         343 |            583 |
| imagenetish   | stream |         875 |           2787 |
| imagenetish   | async  |           - |           3073 |
| lmcacheish    | sync   |        5533 |           6151 |
| lmcacheish    | stream |        4991 |           5368 |
| lmcacheish    | async  |           - |           5384 |

## opends API

| opends family     | cuFile equivalent                    |
|-------------------|--------------------------------------|
| `opends_async_*`  | none                                 |
| `opends_sync_*`   | `cuFileRead`/`cuFileWrite`           |
| `opends_stream_*` | `cuFileReadAsync`/`cuFileWriteAsync` |
| `opends_batch_*`  | `cuFileBatchIO*`                     |

### Threading and context

I/O submission and registration (handles, buffers, streams) are thread-safe once
`opends_driver_open` has returned. `opends_driver_open` and
`opends_driver_close` are not. Deregistering or freeing an object with I/O still
in flight on it is undefined, as with closing a file descriptor that has I/O
in flight.

The aisio backend captures the CUDA context current at `opends_driver_open` and
requires that same context to be current on every thread that submits I/O
through the API. A submit from a thread with a different context current fails
with `OPENDS_CONTEXT_MISMATCH`. An application that uses only the CUDA runtime
API meets this automatically, since every thread on the same device shares the
primary context. An application that creates contexts with the driver API
(`cuCtxCreate`) must bind the driver-open context on each submitting thread with
`cuCtxSetCurrent`. `cudaSetDevice` binds the primary context and is not
equivalent.

### Basic read

```c
#include <opends.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>

int main(void)
{
    opends_driver_open();

    int fd = open("/mnt/nvme/data.bin", O_RDONLY | O_DIRECT);

    opends_handle_t fh;
    opends_handle_register(&fh, fd);

    size_t size = 1024 * 1024;
    void *buf;
    cudaMalloc(&buf, size);
    opends_buf_register(buf, size, 0);

    ssize_t nread = opends_sync_read(fh, buf, size, 0, 0);
    printf("read %zd bytes\n", nread);

    opends_buf_deregister(buf);
    cudaFree(buf);
    opends_handle_deregister(fh);
    close(fd);
    opends_driver_close();

    return 0;
}
```

### Buffer offset

The last parameter to `opends_sync_read` is a byte offset into the destination
buffer. It mirrors cuFile's signature: rather than doing arithmetic on a device
pointer from host code, pass the registered base pointer plus an offset and let
the backend apply it within the mapping it owns.

```c
/* Read two 4 KiB blocks into different regions of a device buffer. */
opends_sync_read(fh, dev_buf, 4096, 0,    0);     /* -> dev_buf[0..4095]    */
opends_sync_read(fh, dev_buf, 4096, 4096, 4096);  /* -> dev_buf[4096..8191] */
```

### Async I/O

`opends_async_*` is per-operation async without streams (no cuFile
counterpart).
`opends_async_read` and `opends_async_write` are the synchronous calls with
completion reaping deferred: submit returns immediately after initializing
the caller-provided future, and `opends_async_await` blocks until that
operation completes, returning its byte count (or a negated error). The
caller owns the future storage (stack allocation is fine) and must keep it
valid at the same address until awaited. Futures may be awaited in any
order, and awaiting a completed future again returns the same result.

```c
opends_async_future_t fut0, fut1;

opends_async_read(fh, dev_buf, 4096, 0,    0,    &fut0);
opends_async_read(fh, dev_buf, 4096, 4096, 4096, &fut1);

/* ... overlap with computation ... */

ssize_t n0 = opends_async_await(&fut0);
ssize_t n1 = opends_async_await(&fut1);
```

### Error handling

Functions returning `opends_error_t` carry both an opends error code and an
optional backend-specific code. Functions returning `ssize_t` (read/write)
return the byte count on success or a negated error on failure.

```c
opends_error_t err = opends_handle_register(&fh, fd);

if (err.err != OPENDS_SUCCESS) {
    fprintf(stderr, "%s\n", opends_op_status_error(err.err));
}

ssize_t n = opends_sync_read(fh, buf, size, offset, 0);
if (n < 0) {
    fprintf(stderr, "read: %s\n",
            opends_op_status_error((opends_op_error_t)-n));
}
```

## Building

Requires [Meson](https://mesonbuild.com) and a C11 compiler. The GDS backend
additionally requires the CUDA toolkit and cuFile library.

```sh
meson setup build
meson compile -C build
```

Meson reports which backends are enabled at configure time:

```
Backends
  Reference backend        : true
  GDS backend              : true
  aisio backend            : true
  aisio accelerator vendor : cuda
```

## Installing

Install headers, libraries, and a pkg-config file so other projects can find
OpenDS via `pkg-config --cflags --libs opends` or meson's
`dependency('opends')`:

```sh
meson install -C build
```

## Testing

Run the reference backend smoke test locally:

```sh
./build/test_smoke_ref
```

Run the full synchronous-read suite against the ref backend locally.
`test_sync_read_prep` writes a deterministic 16-page pattern to a file; each
backend test reads it back through its backend and verifies against an in-memory
oracle:

```sh
f=$(mktemp) && ./build/test_sync_read_prep "$f" \
  && ./build/test_sync_read_ref "$f"; rm -f "$f"
```

### Remote testing with CIJOE

Integration tests run on a remote target via
[CIJOE](https://github.com/refenv/cijoe). Target requirements:

- A dedicated NVMe device (not the boot disk; the aisio phase unbinds it from
  the kernel `nvme` driver).
- An NVIDIA GPU with the CUDA toolkit; GDS (GPUDirect Storage) for the gds
  tests; xNVMe's `upcie-cuda` backend for the aisio tests.
- A kernel built with UDMABUF-import support, IOMMU disabled, and 2 MiB
  hugepages allocated (prerequisites for the GPU↔NVMe dma-buf P2P path that
  aisio uses).
- An XFS filesystem on the test namespace; the mount step does not format. Test
  artifacts live under `<mount_point>/opends_tests/`.

The [aisio](https://github.com/xnvme/aisio) project ships cijoe tasks that take
a fresh Ubuntu 24.04 install through every step above (custom kernel, NVIDIA
stack, hugepages, XFS format, reference datasets). Follow its README first to
bring up a target that meets these requirements. OpenDS pins its own dependency
refs (xNVMe, xal, fil, HOMI, qublk) in `configs/deps.toml` and installs the
stack via `scripts/setup_deps.py` for reproducible test runs.

`test_sync_read_prep` writes a deterministic pattern file (and a small extents
record external benchmarks can deserialize) while the FS is mounted. The ref and
gds tests read the pattern back through the kernel FS. The aisio phase runs last
against the HOMI/qublk stack: the kernel driver is unbound and the controller
handed to a HOMI daemon, qublk re-exports it as a block device, and the same XFS
is remounted over it. Each aisio test opens a file on that mount and registers
it, which resolves the file's extents through the daemon (`homic_get_extents`,
FIEMAP over the qublk device); reads and writes DMA straight to and from GPU
memory. The stack is then torn down and nvme rebound.

1. Copy the example configs and fill in target details:

   ```sh
   cp configs/transport.toml.example configs/transport.toml
   cp configs/test.toml.example configs/test.toml
   ```

   `configs/deps.toml` is tracked and needs no editing.

2. Bootstrap (first run only):

   ```sh
   python scripts/rsync.py
   python scripts/setup_deps.py   # xNVMe, xal, HOMI, qublk, OpenDS, fil
   python scripts/build.py
   ```

   Iterative loop: `python scripts/rsync.py && python scripts/build.py`.

3. Run all test suites:

   ```sh
   python scripts/run_tests.py
   ```

### Benchmarking with filperf

Throughput benchmarks use `filperf` from [fil](https://github.com/xnvme/fil)
against four reference datasets (`filesize8gib`, `tiktokish`, `imagenetish`,
`lmcacheish`). The first three come from the aisio project's
`tasks/setup_dataset.yaml` during target provisioning; `lmcacheish` is OpenDS's
own, populated by `scripts/bench/setup_dataset.py`. Both are one-time, not per
bench run.

With `scripts/setup_deps.py`, `scripts/build.py` and
`scripts/bench/setup_dataset.py` run on the target:

```sh
python scripts/bench/run.py          # --full-sweep measures the whole grid
python scripts/bench/report.py
```

`run.py` measures the configs in `scripts/bench/sweep.toml`. `report.py` turns
the records into `report.md`, `sweep.csv` and `report.png`. Each script's
`--help` covers its own flags.

The perf table above is edited by hand from these reports. Every `filperf` run
drops page caches first, so the numbers are cold-cache, N=1.
