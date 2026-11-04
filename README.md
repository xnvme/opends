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
    size_t size = 1024 * 1024;
    void *buf;

    cudaMalloc(&buf, size);
    opends_driver_open();
    opends_buf_register(buf, size, 0);

    int fd = open("/mnt/nvme/a.bin", O_RDONLY | O_DIRECT);
    opends_handle_t fh;
    opends_handle_register(&fh, fd);

    ssize_t n = opends_sync_read(fh, buf, size, 0, 0);
    printf("read %zd bytes\n", n);

    opends_handle_deregister(fh);
    close(fd);
    opends_buf_deregister(buf);
    cudaFree(buf);
    opends_driver_close();

    return 0;
}
```

`examples/` holds complete programs with error handling. `sync_read.c` reads a
file and checks the bytes against a kernel read, `async_read.c` reads two
files at once with the async API, and `stream_read.cu` enqueues a read, a
kernel over the bytes and a copy back on one CUDA stream.

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
  the test tasks also pass to homi, so both sides agree. The driver asks the
  homi server of this group for its device, and the group must hold exactly
  one.
- `OPENDS_AISIO_HOST_HEAP_MB`: Host DMA heap for this process, holding its own
  queue rings and PRP lists. Default 256. The heap comes out of the hugepages
  every process in the group shares, so xNVMe's 1 GiB default is too large.
- `OPENDS_AISIO_DEVICE_HEAP_MB`: GPU device heap for this process. Default 0,
  which leaves it at the xNVMe default.

### Writes and the extent index

Every read through the AiSIO backend resolves the file's extents in the index
that xal-server publishes. The stack runs xal-server with `watchmode = 0`, so
the index describes the file system as it was when xal-server started, and
nothing updates it while the stack is up.

Writes through OpenDS (`opends_sync_write`, `opends_async_write`,
`opends_stream_write`) go through the kernel and are correct: the bytes land
in the file, and a kernel read sees them.

**Undefined behaviour:** reading through OpenDS what was written after
xal-server indexed. That read may return the bytes from before the write,
zeros, or another file's blocks, and no error. It covers a file created
after the index, bytes appended to a file, a hole that a write filled, and an
overwrite the file system placed in new blocks. An in-place overwrite of
blocks that were allocated at index time usually reads back correctly, but
XFS does not promise it, and OpenDS does not check it.

To read written data through OpenDS, restart xal-server after the writes so it
indexes again. A process that opened the driver before the restart stays on the
old index: its next read fails with `OPENDS_FS_SETUP_ERROR` after five seconds,
and it must call `opends_driver_close()` and `opends_driver_open()` to attach to
the new index. This restriction is temporary: a re-index on request is being
pursued in xal-server, and the restriction goes when it lands.

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
counterpart), shown in `examples/async_read.c`. The future must stay at
the same address until awaited.

`opends_sync_*` blocks until completion.

`opends_stream_*` is stream-ordered I/O, cuFile's ReadAsync/WriteAsync, shown
in `examples/stream_read.cu`: operations enqueue on a registered stream (e.g. a
CUDA stream) and complete in stream order. Sizes, offsets, and the byte-count result are passed as
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
```

`run.py` measures the configs in `scripts/bench/sweep.toml`. `report.py`
turns the records into `report.md`, `sweep.csv` and `report.png`. Each
script's `--help` covers its own flags.

The perf table above is edited by hand from these reports. Every `filperf`
run drops page caches first, so the numbers are cold-cache, N=1.

## Model loading with llama.cpp

Every loader llama.cpp ships moves the weights twice: off the drive into
host memory, then across the bus again into VRAM. OpenDS cuts the host
out and lets the drive write them into VRAM itself. Our fork of llama.cpp,
[fmma/llama.cpp](https://github.com/fmma/llama.cpp/tree/opends-loader),
adds `--load-mode opends`, which reads every tensor of a GGUF file that
way, over PCIe P2P DMA with no kernel in the data path. One drive is
enough to beat dio, the fastest loader llama.cpp ships.

![dio and opends loading the same model](docs/llama-load.gif)

Qwen3-14B, 11.5 GiB of weights, one Samsung 990 PRO:

| `--load-mode` | Whole process | Model load   |
|---------------|---------------|--------------|
| opends        | 3.1 s         | 2.0 s        |
| dio           | 3.3 to 3.6 s  | 2.2 to 2.5 s |
| none          | 3.9 to 4.0 s  | 2.8 s        |
| mmap          | 6.3 s         | 5.1 s        |

dio, llama.cpp's direct I/O loader, is the fastest of the existing ones,
and it runs the drive at its rate, about 6.9 GB/s, as opends does. What
opends saves is the host round trip: dio's pinned staging buffers, their
setup and their teardown.

### Try it

With OpenDS installed as in the quick start, build the `opends-loader`
branch of the fork, bring the stack up over the drive that holds the
model, and load. `$OPENDS` is this repository, `$BDF` the drive and `$MNT`
its mount point:

```sh
git clone -b opends-loader https://github.com/fmma/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENDS=ON
cmake --build build --target llama-completion

$OPENDS/tasks/steps/homi_stack_up.sh $BDF $MNT
export OPENDS_AISIO_DEVICE_HEAP_MB=64
echo "What is the capital of France? /no_think" |
    build/bin/llama-completion -m $MNT/models/Qwen3-14B-Q6_K.gguf -st \
        --load-mode opends
```

For the baselines, stop the stack, give the drive back to the kernel
(`clear_homi_stack.sh`, `rebind_nvme.sh`) and run the same command with
`--load-mode dio`, `none` or `mmap`.

_Measured 2026-11-02: OpenDS `5d37dff`, llama.cpp `opends-loader` at
`86d65e6`, kernel `6.8.12-dmabuf`, RTX 2000 Ada, Qwen3-14B-Q6_K, five runs
per mode with the page cache dropped, GPU persistence mode on and the token
embedding in VRAM (`-ot token_embd.weight=CUDA0`). The recording is one run
of each mode with the command above, recorded one after the other since
the drive cannot be on both drivers at once, laid side by side with the
two Enters on the same frame. The environment of both holds
`OPENDS_AISIO_DEVICE_HEAP_MB=64`, which keeps the backend's GPU heap out
of the way of llama.cpp's memory fit on a 16 GB card._
