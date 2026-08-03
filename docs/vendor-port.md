# Adding a GPU vendor to the aisio backend

The aisio backend reads NVMe straight into GPU memory over xNVMe P2P DMA.
Its only GPU dependency is a narrow runtime surface, `struct ds_accel_ops`
in `src/ds_accel.h`: context binding, pinned-mapped host allocation,
host/device copy, a per-stream gate, and a deferred stream-ordered copy.
The NVMe and extent code calls only through the active-ops pointer
`ds_accel`, never a vendor symbol.

A port is one implementation file, `src/ds_accel_<backend>.c`, that fills in
`struct ds_accel_ops` and binds `ds_accel` to it, plus a `meson.build`
branch (keyed on the `accel_backend` option) that selects it and links the
vendor runtime. Read `ds_accel.h` for the contract and `ds_accel_cuda.c` for
a worked example; only CUDA is implemented today. Device buffers are not in
the table: allocation is delegated to xNVMe (`xnvme_buf_alloc`).

## What needs care

Most of the surface is mechanical. Two things are not, and both are worth
validating on real hardware before trusting a port:

- Async ordering. The I/O thread hands completed DMA to the user's stream
  through a gate word: the stream waits on a value the I/O thread publishes
  with a plain host store to mapped memory. This relies on the vendor's
  stream wait-value primitive actually waiting on host-written mapped memory
  updated by another thread. If it cannot, the ordering has to be reworked
  in the NVMe code, not just in the vendor file.
- Thread-to-device binding (`ctx_set`) for the I/O thread.

`copy_stream` is a deferred copy on the user's stream: its parameters are
published after it is enqueued. CUDA needs a kernel for that, shipped in
`ds_bounce_kernel_cuda.cu`; a vendor whose runtime can defer a stream copy
may not need a separate kernel translation unit at all. Setting
`OPENDS_AISIO_ASSUME_ALIGNED_ONLY=1` drops the tail fixup entirely
(unaligned reads then fail with `OPENDS_INVALID_VALUE`), so a port can be
brought up and measured on LBA-aligned workloads before `copy_stream` works.

Beyond the backend, the Python loader and the test harness still assume
CUDA and would need the same vendor-awareness.

External prerequisites this codebase does not own: an xNVMe `upcie-<vendor>`
P2P backend, and the hardware to validate on.
