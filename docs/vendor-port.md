# Adding a GPU vendor to the aisio backend

The aisio backend reads NVMe straight into GPU memory over xNVMe P2P DMA.
Its only GPU dependency is a narrow runtime surface, `struct ds_accel_ops`
in `src/ds_accel.h`: context capture and binding, pinned-mapped host
allocation, a host/device copy, a stream-ordered host callback, and a
deferred stream-ordered copy. The NVMe and extent code calls only through
the active-ops pointer `ds_accel`, never a vendor symbol.

A port is one implementation file, `src/ds_accel_<backend>.c`, that fills in
`struct ds_accel_ops` and binds `ds_accel` to it, plus a `meson.build`
branch (keyed on the `accel_backend` option) that selects it and links the
vendor runtime. Read `ds_accel.h` for the contract and `ds_accel_cuda.c` for
a worked example; only CUDA is implemented today. Device buffers are not in
the table: allocation is delegated to xNVMe (`xnvme_buf_alloc`).

## What needs care

Most of the surface is mechanical. Two things are not, and both are worth
validating on real hardware before trusting a port:

- Async ordering. `launch_host_func` must run a host callback in stream
  order and hold the stream until it returns, and the runtime must retire it
  host-side rather than through the device's command processor. A primitive
  the device has to schedule serialises behind foreign GPU work and costs
  milliseconds per operation under contention.
- Thread-to-device binding (`ctx_set`) for the I/O thread.

`copy_stream` closes out a sub-LBA tail on the user's stream: its parameters
are published after it is enqueued, by host stores the I/O thread makes
while the stream is parked in the callback. CUDA needs a kernel for that,
shipped in `ds_bounce_kernel_cuda.cu`; a vendor whose runtime can defer a
stream copy may not need a separate kernel translation unit at all. Setting
`OPENDS_AISIO_ASSUME_ALIGNED_ONLY=1` drops the tail fixup entirely
(unaligned reads then fail with `OPENDS_INVALID_VALUE`), so a port can be
brought up and measured on LBA-aligned workloads before `copy_stream` works.
Driver open validates the ops the chosen mode drives and fails cleanly when
one is missing, so a partial table may leave the other mode's ops NULL.

Beyond the backend, the Python loader and the test harness still assume
CUDA and would need the same vendor-awareness.

External prerequisites this codebase does not own: an xNVMe `upcie-<vendor>`
P2P backend, and the hardware to validate on.
