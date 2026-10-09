# OpenDS

Open source accelerator direct storage. Vendor-neutral API modeled on
NVIDIA's cuFile (GDS), powered by AiSIO for high-throughput PCIe P2P DMA from
NVMe straight into GPU memory.

## Quick start

1. Provision a target machine by following the
   [AiSIO](https://github.com/xnvme/aisio) guide.

2. Create the configs and fill in the target details:

   ```sh
   cp configs/transport.toml.example configs/transport.toml  # hostname, ssh key
   cp configs/test.toml.example configs/test.toml            # NVMe BDF, mount point
   ```

3. Sync the tree, install the pinned dependencies (xNVMe, xal, fil, first
   run only), build, and run the test suites:

   ```sh
   python scripts/rsync.py
   python scripts/setup_deps.py
   python scripts/build.py
   python scripts/run_tests.py
   ```

## Basic example

```c
#define _GNU_SOURCE
#include <opends.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int main(void)
{
    opends_driver_open();

    int fd0 = open("/mnt/nvme/a.bin", O_RDONLY | O_DIRECT);
    int fd1 = open("/mnt/nvme/b.bin", O_RDONLY | O_DIRECT);

    opends_handle_t fh0, fh1;
    opends_handle_register(&fh0, fd0);
    opends_handle_register(&fh1, fd1);

    size_t size = 1024 * 1024;
    void *buf;
    cudaMalloc(&buf, 2 * size);
    opends_buf_register(buf, 2 * size, 0);

    opends_async_future_t fut0, fut1;
    opends_async_read(fh0, buf, size, 0, 0,    &fut0);
    opends_async_read(fh1, buf, size, 0, size, &fut1);

    /* ... overlap with computation ... */

    ssize_t n0 = opends_async_await(&fut0);
    ssize_t n1 = opends_async_await(&fut1);
    printf("read %zd and %zd bytes\n", n0, n1);

    opends_buf_deregister(buf);
    cudaFree(buf);
    opends_handle_deregister(fh0);
    opends_handle_deregister(fh1);
    close(fd0);
    close(fd1);
    opends_driver_close();

    return 0;
}
```

## Backends

- **Reference** (`libopends_ref`): POSIX `pread`/`pwrite` on host buffers. No
  external dependencies. Serves as a correctness baseline and template for
  hardware-specific backends.
- **cuFile** (`libopends_cufile`): Wraps NVIDIA cuFile for GPUDirect Storage. Buffers
  are GPU memory allocated with `cudaMalloc` and registered via
  `cuFileBufRegister`. Requires CUDA toolkit and the cuFile (GDS) library. Built
  conditionally when both are found.
- **AiSIO** (`libopends_aisio`): PCIe P2P DMA between NVMe and GPU memory via
  [xNVMe](https://xnvme.io)'s `upcie-cuda` backend (no filesystem or kernel
  `nvme` driver in the read data path). Based on
  [AiSIO](https://github.com/xnvme/aisio). A homi server (an xNVMe tool) is
  the primary of an xNVMe multi-process group and holds the userspace NVMe
  controller up. The driver joins that group as a secondary and allocates its
  own I/O queues. File extents come from the index xal-server publishes over
  POSIX shared memory, built over the qublk-exported block device. Reads and
  writes are supported. Requires xNVMe, xal, and the CUDA toolkit.

## OpenDS AiSIO configuration

The AiSIO backend reads its configuration from environment variables at
`opends_driver_open`. Values that are out of range, or not a number, fail
the open.

- `OPENDS_HOMI_DEV` (required): The NVMe device the homi server owns (PCI BDF).
- `OPENDS_XAL_SHM`: Shared-memory name of the xal-server extent index.
  Default `/xal_dev0`.
- `OPENDS_AISIO_IO_THREADS`: Number of internal IO worker threads. Default 2.
  Driver open creates one NVMe queue per worker.
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
- `OPENDS_AISIO_HOMI_ID`: xNVMe multi-process group to join. Default 1, which
  the test tasks also pass to homi, so both sides agree.
- `OPENDS_AISIO_HOST_HEAP_MB`: Host DMA heap for this process, holding its own
  queue rings and PRP lists. Default 256. The heap comes out of the hugepages
  every process in the group shares, so xNVMe's 1 GiB default is too large.
- `OPENDS_AISIO_DEVICE_HEAP_MB`: GPU device heap for this process. Default 0,
  which leaves it at the xNVMe default.

## Performance

_Commit `93fd019` (kernel `6.8.12-dmabuf`, NVMe `Samsung S4LV008[Pascal]`, GPU
`NVIDIA RTX 2000 Ada Generation`). `OPENDS_AISIO_IO_THREADS=2` and
`OPENDS_AISIO_QUEUE_DEPTH=8`._

| Dataset       | mode   | cuFile (MiB/s) | OpenDS (MiB/s) |
|---------------|--------|----------------|----------------|
| filesize8gib  | sync   |           6520 |           6794 |
| filesize8gib  | stream |           2599 |           7036 |
| filesize8gib  | async  |              - |           7065 |
| tiktokish     | sync   |           4614 |           5869 |
| tiktokish     | stream |           5101 |           4957 |
| tiktokish     | async  |              - |           5005 |
| imagenetish   | sync   |            343 |            583 |
| imagenetish   | stream |            875 |           2787 |
| imagenetish   | async  |              - |           3073 |
| lmcacheish    | sync   |           5533 |           6151 |
| lmcacheish    | stream |           4991 |           5368 |
| lmcacheish    | async  |              - |           5384 |

## OpenDS API

| OpenDS family     | cuFile equivalent                    |
|-------------------|--------------------------------------|
| `opends_async_*`  | none                                 |
| `opends_sync_*`   | `cuFileRead`/`cuFileWrite`           |
| `opends_stream_*` | `cuFileReadAsync`/`cuFileWriteAsync` |
| `opends_batch_*`  | `cuFileBatchIO*`                     |

`opends_async_*` is per-operation async without streams (no cuFile
counterpart), shown in the basic example. The future must stay at the
same address until awaited.

`opends_sync_*` blocks until completion.

`opends_stream_*` is stream-ordered I/O, cuFile's ReadAsync/WriteAsync:
operations enqueue on a registered stream (e.g. a CUDA stream) and complete
in stream order. Sizes, offsets, and the byte-count result are passed as
pointers, read and written at stream execution time rather than submission
time.

`opends_batch_*` submits many independent operations in one call and reaps
completions by polling `opends_batch_get_status`. Operations complete in
any order. `opends_batch_setup` fixes how many the handle holds in flight,
and a slot frees once its completion is delivered.

### Threading and context

I/O submission and registration (handles, buffers, streams) are thread-safe once
`opends_driver_open` has returned. `opends_driver_open` and
`opends_driver_close` are not. Deregistering or freeing an object with I/O still
in flight on it is undefined, as with closing a file descriptor that has I/O
in flight.

The AiSIO backend captures the CUDA context current at `opends_driver_open` and
requires that same context to be current on every thread that submits I/O
through the API. A submit from a thread with a different context current fails
with `OPENDS_CONTEXT_MISMATCH`. An application that uses only the CUDA runtime
API meets this automatically, since every thread on the same device shares the
primary context. An application that creates contexts with the driver API
(`cuCtxCreate`) must bind the driver-open context on each submitting thread with
`cuCtxSetCurrent`. `cudaSetDevice` binds the primary context and is not
equivalent.

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

## Building and installing locally

Requires [Meson](https://mesonbuild.com) and a C11 compiler. The cuFile backend
additionally requires the CUDA toolkit and cuFile library.

```sh
meson setup build
meson compile -C build
meson install -C build
```

`meson install` installs headers, libraries, and a pkg-config file so other
projects can find OpenDS via `pkg-config --cflags --libs opends` or meson's
`dependency('opends')`.

## Benchmarking with filperf

Throughput benchmarks use `filperf` from [fil](https://github.com/xnvme/fil)
against four reference datasets. The AiSIO guide's `setup_dataset.yaml`
writes `filesize8gib`, `tiktokish` and `imagenetish` during provisioning.
`scripts/bench/setup_dataset.py` writes `lmcacheish` once.

With the quick start done and the datasets in place:

```sh
python scripts/bench/run.py          # --full-sweep measures the whole grid
python scripts/bench/report.py
python scripts/bench/artefacts.py --push
```

`run.py` measures the configs in `scripts/bench/sweep.toml`. `report.py`
turns the records into `report.md`, `sweep.csv` and `report.png`.
`artefacts.py` publishes those to the orphan `artefacts` branch. Each
script's `--help` covers its own flags.

The perf table above is edited by hand from these reports. Every `filperf`
run drops page caches first, so the numbers are cold-cache, N=1.
