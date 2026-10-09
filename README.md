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
  controllers up; the driver joins that group as a secondary and allocates its
  own I/O queues. The driver takes its device set from the group's runtime:
  every device the server holds gets its own queues and I/O threads. File
  extents come from per-device indexes xal-server publishes over POSIX shared
  memory, built over the qublk-exported block devices, and a registered file
  is bound to the device whose index resolves its path. Reads and writes are
  supported. Requires xNVMe, xal, and the CUDA toolkit.

## OpenDS AiSIO configuration

The AiSIO backend reads its configuration from environment variables at
`opends_driver_open`. Values that are out of range, or not a number, fail
the open. The devices are not configured here: the driver takes them from
the multi-process runtime the homi server created, so the homi command line
is the only place they are named.

- `OPENDS_XAL_SHM`: Shared-memory names of the xal-server extent indexes, a
  comma-separated list paired with the runtime's device order. Default
  `/xal_dev<i>` for device i.
- `OPENDS_AISIO_IO_THREADS`: Total number of internal IO worker threads.
  Default 2, or the device count when that is larger. Driver open creates one
  NVMe queue per worker and gives the workers to the devices round-robin, so a
  worker serves one device. It fails when the workers are fewer than the
  devices, since a device with no worker cannot be read.
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
  kernel. A single stream makes the same declaration with the
  `OPENDS_STREAM_PAGE_ALIGNED_INPUTS` registration flag.
- `OPENDS_AISIO_HOMI_ID`: xNVMe multi-process group to join. Default 1, which
  the test tasks also pass to homi, so both sides agree.
- `OPENDS_AISIO_HOST_HEAP_MB`: Host DMA heap for this process, holding its own
  queue rings and PRP lists. Default 256. The heap comes out of the hugepages
  every process in the group shares, so xNVMe's 1 GiB default is too large.
- `OPENDS_AISIO_DEVICE_HEAP_MB`: GPU device heap for this process. Default 0,
  which leaves it at the xNVMe default.
- `OPENDS_AISIO_CQ_MIRROR`: `1` places the I/O workers' completion queues in
  GPU memory, mirrored to the host by a resident kernel
  (`XNVME_QUEUE_P2P_CQ_MIRROR`). Default off.

### GPU-initiated stream reads

By default an I/O worker thread issues the NVMe commands of a stream read
when the user's stream reaches the op, and reads `size_p`, `file_offset_p`
and `buf_offset_p` then. With `OPENDS_AISIO_GPU_INITIATED=1` the GPU issues
them itself on every stream registered with `OPENDS_STREAM_FIXED_SHAPE`:
`opends_stream_read` resolves the file's extents and builds the commands,
PRP lists included, when it is called, and enqueues a kernel on the stream
that submits them from GPU-resident NVMe queues and reaps the completions.
No host thread is on the path once the stream reaches the op.

The flags fix the shape of the read at call time, which is what lets the
commands be built there; a stream registered without them keeps the I/O
workers and deferred evaluation while the engine is on. Stream writes, and
reads too large for a context (see `OPENDS_AISIO_GPU_MAX_CMDS`), still go
through the I/O workers. GPU-resident queues need the privileges xNVMe
documents for GPU-issued I/O (the controller's BAR through sysfs, so root)
and, behind a translating IOMMU, the `iommu_map_pa` module.

- `OPENDS_AISIO_GPU_INITIATED`: `1` enables the GPU engine for the reads of
  fixed-shape streams. Default off. Driver open fails when the GPU queues
  cannot be created.
- `OPENDS_AISIO_GPU_QUEUE_DEPTH`: Depth of each GPU-resident queue, which is
  also the size of the CUDA block that drives it. A read with more commands
  than the depth goes in several rounds, each waiting for the slowest
  completion before the next, so a depth that holds a whole read is fastest.
  Default 256, at most 1023.
- `OPENDS_AISIO_GPU_CTXS`: GPU contexts per device, each one op in flight
  with its own queues, command array and PRP lists. A submit waits for a
  free context. Default 4. The controller's I/O queue count bounds contexts
  times queues per op; a Samsung 990 PRO has 16, shared with homi, qublk
  and the I/O workers.
- `OPENDS_AISIO_GPU_QUEUES_PER_OP`: Queues (CUDA blocks) an op is spread
  over. Default 1, at most 8.
- `OPENDS_AISIO_GPU_MAX_CMDS`: Commands a context holds; a read needing more
  falls back to the I/O workers. Each command moves up to the smaller of the
  device's MDTS and 2 MiB, and each costs a 4 KiB PRP list page in host and
  GPU memory per context. Default 2048.
- `OPENDS_AISIO_GPU_SQ_HOSTMEM`: `1` places the GPU queues' submission
  queues in host memory (`XNVME_QUEUE_SQ_HOSTMEM`) instead of GPU memory.
  Default off.

`aisio_stream_compute` (`examples/aisio_stream_compute.cu`) shows who works
in each engine. Per iteration it enqueues a fill kernel, a stream read and a
checksum kernel on one stream, then the host sleeps while the chain runs and
samples every thread's CPU time from `/proc` before and after; the sums are
checked against the host's, and GPU timestamps bound each read. Its streams
are registered with `OPENDS_STREAM_FIXED_SHAPE`, so the GPU engine takes the
reads when it is enabled.

```bash
export OPENDS_XAL_SHM=/xal_dev0 OPENDS_HOMI_MNT=/mnt/datasets
# <file> [iters] [host-sleep-ms] [read-bytes] [streams]
aisio_stream_compute /mnt/datasets/opends_tests/gpu_demo.bin 20 1500
OPENDS_AISIO_GPU_INITIATED=1 aisio_stream_compute /mnt/datasets/opends_tests/gpu_demo.bin 20 1500
OPENDS_AISIO_GPU_INITIATED=1 aisio_stream_compute /mnt/datasets/opends_tests/gpu_demo.bin 4000 200 4096 8
```

The last line of the output is a machine-readable summary, so a sweep over
sizes and stream counts can be tabulated.

A stream op's result is written by a host callback behind the I/O, which
costs 20 to 40 us of stream time per op. When `bytes_read_p` points into a
block from `opends_result_alloc`, pinned and device-mapped, the GPU engine
writes the result from the kernel instead and no callback is enqueued; the
demo uses such a block unless `AISIO_DEMO_PLAIN_RESULTS=1`.

With the host engine the two I/O workers use about one core for as long as
the chain runs; with the GPU engine the host spends a few percent of a core,
nearly all of it the idle workers waking, while the reads run at the same
rate. Submitting a 256 MiB read costs the caller about 0.6 ms of CPU, and a
submit blocks when every context is in flight.

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
