# OpenDS

Open source accelerator direct storage. Vendor-neutral, drop-in replacement for
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
`opends_driver_open`. Out-of-range values fail the open.

- `OPENDS_HOMI_DEV` (required): The NVMe device the HOMI daemon owns (PCI BDF).
- `OPENDS_HOMI_SOCKET`: HOMI daemon socket. Default `/run/homi/homi.sock`.
- `OPENDS_AISIO_IO_THREADS`: Number of internal IO worker threads. Default 1.
  Driver open attaches one NVMe qpair per worker.
- `OPENDS_AISIO_QUEUE_DEPTH`: xNVMe queue depth per worker. Default 512.
- `OPENDS_AISIO_CPU_MASK`: CPU affinity mask for the workers (e.g. `0xf0`).
  Worker i is pinned to the i-th set bit, round-robin. Unset or `0` leaves
  workers unpinned.

## Performance

Headline read throughput across the four reference datasets, cold-cache, N=1.
The table below is maintained by hand. Regenerate the numbers with
`scripts/bench/run.py` and `scripts/bench/report.py`, then copy in the figures of
interest; the full per-leg data lives on the orphan `artefacts` branch (see
"Benchmarking with filperf").

_Commit `c4553df` (kernel `6.8.12-dmabuf`, NVMe `Samsung S4LV008[Pascal]`, GPU `NVIDIA RTX 2000 Ada Generation`)._

| Dataset       | mode  | gds (MiB/s) | opends (MiB/s) |
|---------------|-------|--------------|--------------|
| filesize8gib  | sync  |         6967 |         7100 |
| filesize8gib  | async |         2426 |         6974 |
| tiktokish     | sync  |         3899 |         4880 |
| tiktokish     | async |         2515 |         5285 |
| imagenetish   | sync  |          335 |          351 |
| imagenetish   | async |          868 |         2783 |
| lmcacheish    | sync  |         5317 |         6025 |
| lmcacheish    | async |         4961 |         4960 |

## opends API

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

Read offsets must be LBA-aligned and the file opened with `O_DIRECT`. The size
need not be: the aisio backend reads a sub-LBA tail through a bounce buffer and
copies it into place. A read starting at an unaligned offset returns
`OPENDS_INVALID_VALUE`.

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

    ssize_t nread = opends_read(fh, buf, size, 0, 0);
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

The last parameter to `opends_read` is a byte offset into the destination
buffer. It mirrors cuFile's signature: rather than doing arithmetic on a device
pointer from host code, pass the registered base pointer plus an offset and let
the backend apply it within the mapping it owns.

```c
/* Read two 4 KiB blocks into different regions of a device buffer. */
opends_read(fh, dev_buf, 4096, 0,    0);     /* -> dev_buf[0..4095]    */
opends_read(fh, dev_buf, 4096, 4096, 4096);  /* -> dev_buf[4096..8191] */
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

ssize_t n = opends_read(fh, buf, size, offset, 0);
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
   python scripts/setup_deps.py   # Installs xNVMe, xal, HOMI, qublk, OpenDS
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
`lmcacheish`). Two suites: `tasks/bench_gds.yaml` and `tasks/bench_opends.yaml`.
Datasets are populated once, not per bench run. The first three come from the
aisio project's `tasks/setup_dataset.yaml` during target provisioning.
`lmcacheish` is OpenDS's own dataset: populate it with
`scripts/bench/setup_dataset.py`.

Prerequisites: `scripts/setup_deps.py`, `scripts/build.py`, and
`scripts/bench/setup_dataset.py` have run on the target, and the aisio-provisioned
datasets are in place.

Full benchmark run and reporting:

```sh
python scripts/bench/run.py
python scripts/bench/report.py --spec-mbs 7450
python scripts/bench/artefacts.py --push
```

`run.py` sweeps every suite over its own knob grid into
`cijoe-output-bench/<suite>/`. The gds suite has no knobs, so its sweep is the
singleton: one run of the whole suite. The opends suite sweeps io_threads x
queue_depth x assume_aligned_only: the HOMI/qublk stack comes up once, and
each grid point runs into `cijoe-output-bench/opends/t<t>_q<q>[_aligned]/`;
the grid comes from `--io-threads` (default 1,2,4,8), `--queue-depth` (default
1..512) and `--assume-aligned-only` (default 0,1). Restrict with `--suite`,
`--mode`, and `--dataset`. An aligned leg rejects any read with a sub-LBA
tail, so datasets whose files are not LBA-multiples fail by construction;
those legs are recorded with an empty result, the sweep continues to the
datasets that do qualify, and `report.py` gives them their own section. The
aisio knobs travel as environment variables (`OPENDS_AISIO_IO_THREADS`,
`OPENDS_AISIO_QUEUE_DEPTH`, `OPENDS_AISIO_CPU_MASK`). `--cpu-mask` sets the
last one: a hex mask whose CPUs the aisio IO workers are pinned to round-robin
(`0x0` or unset leaves placement to the scheduler). Outside the sweep,
`OPENDS_AISIO_ASSUME_ALIGNED_ONLY=1` declares that every read is LBA-aligned:
reads whose span ends off an LBA boundary fail with `OPENDS_INVALID_VALUE`,
and the async path stops enqueueing the per-read bounce kernel.

Every leg appends a structured record to
`<out>/**/artifacts/history.jsonl`: config (backend, dataset, mode,
io_threads, queue_depth, cpu_mask), result (MiB/s, IOPS, timings), and
environment (commit, host, kernel, NVMe, GPU, from `meta.json`), next to the
verbatim `filperf` stdout (`<backend>_<dataset>[_async].log`). Each `filperf`
run drops page caches first, so numbers are cold-cache, N=1.

`report.py` reads every `history.jsonl` under its `--in` dirs (repeatable,
default `cijoe-output-bench`) and writes `report.md`, `sweep.csv` (all records,
flat), and `report.png` to the first `--in` dir (or `--out`). `report.md`
holds one MiB/s pivot matrix over (io_threads x queue_depth) per opends
(dataset, mode) and, when gds legs are present, a GDS-vs-OpenDS comparison
table per (queue_depth, io_threads) point with per-(dataset, mode) speedups.
`--spec-mbs` draws the device's spec sequential-read line in the plot (7450
for the reference target's Samsung 990 PRO).

`artefacts.py` publishes the reports and records to the orphan `artefacts`
branch (no shared history with the code) through a throwaway worktree, plain
fast-forward pushes only; `--push` sends the branch to origin. Layout:

```
README.md                  Generated snapshot index.
latest/                    Newest snapshot's report.md/report.png/sweep.csv,
                           overwritten each publish.
snapshots/<date>-<sha>/    One immutable snapshot per publish: the report
                           files plus history.jsonl (every leg, concatenated).
```

The perf table above is edited by hand from these reports.
