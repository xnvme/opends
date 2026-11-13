# OpenDS

Open source accelerator direct storage. Vendor-neutral API modeled on
NVIDIA's cuFile (GDS), powered by AiSIO for high-throughput PCIe P2P DMA from
NVMe straight into GPU memory.

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

## Getting started

### Requirements

NOTE: The [AiSIO](https://github.com/xnvme/aisio) guide sets all of this up on a
fresh Ubuntu installation.

- An NVIDIA GPU with the NVIDIA driver and the CUDA toolkit, `nvcc` on the
  `PATH`.
- In the BIOS setup:
  - Resizable BAR enabled, so that the GPU's BAR1 window covers all of its
    memory. Check with `nvidia-smi -q -d memory`.
  - ACS disabled.
- An NVMe drive that is not the boot disk, with an XFS file system that holds
  the files to read.
- The two kernel modules of [uPCIe](https://github.com/safl/upcie), from its
  `upcie-experimental-dkms` package in the version of the uPCIe that the
  pinned xNVMe builds, 0.10.4. `dmabuf_import` resolves the DMA addresses of
  GPU memory, and `iommu_map_pa` maps GPU memory for a drive behind the
  IOMMU.
- One of the two ways to attach the drive:
  - The IOMMU on, and the drive bound to `vfio-pci`. Where xNVMe's default
    vfio mode is refused, as on a 6.8 kernel, set
    `XNVME_UPCIE_VFIO_MODE=type1`.
  - The IOMMU off or in passthrough mode (`iommu=pt`), and the drive bound to
    `uio_pci_generic`.
- The `ublk_drv` kernel module.
- 1024 hugepages of 2 MiB, which are enough for the default settings.
- Meson, Ninja, a C11 compiler, Git and liburing, to build the dependencies.

### Build

xNVMe (for-aisio-02112026), xal (v0.3.4) and OpenDS:

```sh
git clone https://github.com/xnvme/xnvme
cd xnvme
git checkout for-aisio-02112026
git submodule update --init --recursive
meson setup build
meson compile -C build
meson install -C build
cd ..

git clone https://github.com/xnvme/xal
cd xal
git checkout v0.3.4
git submodule update --init --recursive
meson setup build -Dxal-server=enabled
meson compile -C build
meson install -C build
cd ..

git clone https://github.com/xnvme/opends
cd opends
meson setup build
meson compile -C build
meson install -C build
ldconfig
```

xNVMe brings the `homi` and `qublk` tools, and xal brings `xal-server`.

### Run

As root, in one terminal. Name the drive, its mount point and the driver that
will hold it:

```sh
export BDF=<pci-address>   # of the drive, as lspci -D prints it: 0000:01:00.0
export MNT=<mount-point>   # where its file system is mounted
export DRIVER=vfio-pci     # uio_pci_generic with the IOMMU off or in passthrough mode
```

Check the address twice. The next block takes that drive away from the
kernel, and it must not be the one the system runs from.

Take the drive from the kernel's NVMe driver and bind it to that driver, so
that xNVMe can open it:

```sh
umount $MNT
echo $BDF > /sys/bus/pci/drivers/nvme/unbind
echo 1 > /sys/bus/pci/devices/$BDF/reset
modprobe $DRIVER
modprobe ublk_drv
echo $DRIVER > /sys/bus/pci/devices/$BDF/driver_override
echo $BDF > /sys/bus/pci/drivers/$DRIVER/bind
```

Start `homi`, which holds the controller, and wait until it prints
`HOMI started successfully`:

```sh
# 512 MiB of host memory: room for qublk and for OpenDS
homi serve $BDF --homi-id 1 --be upcie --host-heap-size $((512 * 1024 * 1024)) &
```

Start `qublk`, which exports the drive as a block device for the file system,
and wait until it prints `/dev/ublkb0 ... ready`:

```sh
qublk serve $BDF --homi-id 1 --be upcie --nqueues 1 &
```

`xal-server` publishes where the files are on the drive. Create its
configuration, `xal-server.conf`, with the mount point of the drive:

```toml
devices = [
  { uri = "/dev/ublkb0", shm_name = "/xal_dev0", mountpoint = "<mount-point>" },
]
```

Mount the file system and start it, and wait until it prints `published`,
which is when its first index is ready:

```sh
mount /dev/ublkb0 $MNT
xal-server --config xal-server.conf &
```

Run your OpenDS application, with the two variables that tell OpenDS where
the index and the mount are. For example, from the OpenDS checkout, the
example that reads a file into GPU memory:

```sh
export OPENDS_XAL_SHM=/xal_dev0 OPENDS_HOMI_MNT=$MNT
build/sync_read_example $MNT/<file>
```

A file has to be on the drive before `xal-server` starts (see
[Writes and the extent index](#writes-and-the-extent-index)).

Stop the three in reverse order, and give the drive back to the kernel:

```sh
kill %xal-server; wait %xal-server
umount $MNT
kill %qublk; wait %qublk
kill %homi; wait %homi
echo $BDF > /sys/bus/pci/drivers/$DRIVER/unbind
echo > /sys/bus/pci/devices/$BDF/driver_override
echo $BDF > /sys/bus/pci/drivers/nvme/bind
```

`tasks/steps/homi_stack_up.sh` and `homi_stack_down.sh` do the same in one
call, with waits and checks. They bind the driver that fits the IOMMU's mode,
or the one `HOMI_NVME_DRIVER` names.

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
  writes are supported. A handle can also be the drive itself (raw block
  mode, below), with reads and writes as P2P DMA at namespace offsets.
  Requires xNVMe, xal, and the CUDA toolkit.

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
- `OPENDS_AISIO_RAW_BLOCK`: `1` turns raw block mode on (below). Default off,
  which changes nothing for a process that uses files only.
- `OPENDS_AISIO_WRITE_BUF_MB`: Size of one pinned host buffer a write is staged
  through, in MiB. Default 16. A larger write goes through its buffer in
  rounds.
- `OPENDS_AISIO_WRITE_BUFS`: Most buffers the driver pins, allocated as writes
  need them. Default four per I/O thread. Every write in flight holds one: a
  stream write from its call, the others from their dispatch. A stream write
  call that finds none free waits for one, so an application that queues many
  stream writes behind work its own thread must still release needs a pool
  at least that large.

### Writes and the extent index

Every read through the AiSIO backend resolves the file's extents in the index
that xal-server publishes. The stack runs xal-server with `watchmode = 0`, so
the index describes the file system as it was when xal-server started, and
nothing updates it while the stack is up.

Writes through OpenDS (`opends_sync_write`, `opends_async_write`,
`opends_stream_write`) go through the kernel and are correct: the bytes land
in the file, and a kernel read sees them. The source is copied out of GPU
memory into a pinned buffer, on the user's stream for a stream write, and the
I/O thread only polls for the copy: it never makes a CUDA call that waits,
since a stream may be waiting at a gate for that thread.

A stream write on a stream registered with the three `FIXED` flags of
`opends_stream_register` (`OPENDS_STREAM_FIXED_BUF_OFFSET`,
`_FIXED_FILE_OFFSET`, `_FIXED_FILE_SIZE`, the flags of
`cuFileStreamRegister`) takes its size and offsets at the call and copies
with the copy engine; a value the stream changes later fails the write with
`OPENDS_INVALID_VALUE`. Without the flags a kernel on the stream reads the
values when the stream reaches the write and copies them, staying resident
until the last round of a write larger than the buffer has landed. A write
of more than 64 buffers goes through the kernel with the flags too, since a
stream takes only so many queued commands. Pass the flags when the values
are set before the call, which is the common case.

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

### Raw block mode

With `OPENDS_AISIO_RAW_BLOCK=1`, `opends_handle_register` accepts a character
device that answers for the controller, such as the one homi serves
(`/dev/xnvme/<bdf>`), opened without `O_DIRECT`. On such a handle
`file_offset` is the byte offset on the namespace: `file_offset / lba_size` is
the LBA, with no filesystem or extent lookup in the path. All I/O families
work: sync, async, stream and batch.

In this mode the driver opens without an index and attaches to it when the
first regular file is registered, with the wait and the error the open has
otherwise; a failed attach is final until the driver is closed and opened
again. A read that needs the index, through a file handle kept across
`opends_driver_close()` and `opends_driver_open()`, fails with
`OPENDS_FS_SETUP_ERROR` until a regular file is registered again.

A raw handle needs no index, so the stack for raw block mode is `homi serve`
alone: no qublk, no file system and no xal-server. This is the uGDS-style use,
a KV cache for instance, where the process owns the drive and lays it out
itself. `test_raw_block_aisio` runs that way in the test suite.

- Reads and writes go between GPU memory and the NVMe as P2P DMA, and are
  short when they cross the end of the namespace.
- A read may start and end anywhere, but its whole LBAs must land on a 4-byte
  boundary in GPU memory: a raw handle has no kernel path to fall back to. A
  stream read must start on an LBA boundary, as on a file.
- A write, unless empty, must start and end on an LBA boundary, come from a
  GPU address on a 4-byte boundary and start before the end of the namespace.
- I/O that breaks these rules fails with `OPENDS_INVALID_VALUE`, I/O against
  the access mode of the fd with `OPENDS_INVALID_FILE_OPEN_FLAG`. A refused
  write writes nothing.
- Only a character device that answers for the controller this driver opened
  is accepted, told by its serial number. homi serves one when xNVMe is built
  with CUSE, the `cuse` module is loaded and `--no-cuse` is not given.
- The process must own the drive: do not run qublk or mount a file system on
  a drive used in raw block mode. Raw writes bypass them, and they would not
  see the change.

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

C programs find OpenDS through `pkg-config --cflags --libs opends` or meson's
`dependency('opends')`. Python bindings are in `python/`; see
`python/README.md`.

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

## Testing on a target machine

The test suites and the benchmarks run from a workstation against a target
machine over ssh, through [cijoe](https://github.com/refenv/cijoe)
(`pipx install cijoe`). Create the configs and fill in the target details:

```sh
cp configs/transport.toml.example configs/transport.toml  # hostname, ssh key
cp configs/test.toml.example configs/test.toml            # NVMe BDF, mount point
```

Then sync the tree, install the pinned dependencies (xNVMe, xal, fil, first
run only), build, and run the suites. They attach the drive through
`uio_pci_generic`:

```sh
python scripts/rsync.py
python scripts/setup_deps.py
python scripts/build.py
python scripts/run_tests.py
```

## Benchmarking with filperf

Throughput benchmarks use `filperf` from [fil](https://github.com/xnvme/fil)
against four reference datasets. The AiSIO guide's `setup_dataset.yaml`
writes `filesize8gib`, `tiktokish` and `imagenetish` during provisioning.
`scripts/bench/setup_dataset.py` writes `lmcacheish` once.

With OpenDS installed and the datasets in place:

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

With OpenDS installed as in [Getting started](#getting-started), build the `opends-loader`
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
