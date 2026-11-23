/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends_accel.h - vendor-neutral accelerator runtime interface for the aisio
 * backend.
 *
 * A vendor supports the backend by defining one struct opends_accel_ops
 * instance and binding the active-ops pointer opends_accel to it (see
 * opends_accel_cuda.c). The NVMe and extent code calls only through
 * opends_accel, never a vendor symbol.
 *
 * Interface types are vendor-uniform so the ops ABI is fixed:
 *   - opends_accel_devptr_t is a 64-bit device address.
 *   - opends_accel_ctx_t and opends_accel_stream_t are opaque handles (void *).
 * Each vendor converts to and from its native pointer at the boundary (identity
 * for CUDA's CUdeviceptr; a cast for HIP's void *).
 */
#ifndef OPENDS_ACCEL_H_
#define OPENDS_ACCEL_H_

#include <stddef.h>
#include <stdint.h>

typedef uint64_t opends_accel_devptr_t;
typedef void *opends_accel_ctx_t;
typedef void *opends_accel_stream_t;

/* The interface an accelerator vendor implements. Each int-returning method
 * returns 0 on success and a nonzero vendor error code on failure (CUDA: a
 * driver or runtime API error code). The code is opaque to the core, reported
 * through dev_err but never interpreted; -1 serves where no vendor code
 * applies. */
struct opends_accel_ops {
	/* xNVMe P2P backend string for this vendor (e.g. "upcie-cuda"). */
	const char *xnvme_be;

	/* Load the device code now, on the thread that opens the driver. A
	 * runtime that loads a module at a kernel's first launch may
	 * synchronize the context to do it, and that launch can come with a
	 * stream already parked for an I/O thread. */
	int (*load)(void);
	/* Capture the caller's current context (fails if there is none).
	 * Captured values compare equal iff they name the same context. */
	int (*ctx_get)(opends_accel_ctx_t *out);
	/* Make this thread's accelerator work target the captured context
	 * (CUDA: cuCtxSetCurrent; HIP: hipSetDevice). */
	void (*ctx_set)(opends_accel_ctx_t ctx);
	/* Granularity of the runtime's device allocations. A dma-buf export
	 * of the whole chunk around any device pointer is backed, so buffer
	 * registrations round to it (CUDA: cuMemGetAllocationGranularity). */
	int (*alloc_granularity)(size_t *out);

	/* Host pinned memory the accelerator can read by device pointer. */
	int (*host_alloc_mapped)(size_t bytes, void **host,
	                         opends_accel_devptr_t *dptr);
	/* Free a host_alloc_mapped allocation. */
	void (*host_free)(void *host);
	/* Pin host memory the caller owns, for the async copies below, and map
	 * it for the device (dptr may be NULL when no kernel will touch it). */
	int (*host_pin)(void *host, size_t bytes, opends_accel_devptr_t *dptr);
	void (*host_unpin)(void *host);

	/* Host/device bulk copy (direction inferred from the pointers). the
	 * calling thread must bind the accelerator context via ctx_set before
	 * using copy. */
	int (*copy)(void *dst, const void *src, size_t bytes);
	/* Queue a copy from pinned host memory to the device on the default
	 * stream and return at once. src must stay as it is until the copy
	 * has run; a stream_write_value32 queued behind it on the default
	 * stream (a NULL stream) tells when. Required only when
	 * assume_aligned_only is off. */
	int (*copy_to_dev_async)(void *dst, const void *src, size_t bytes);
	/* Queue a fill of device memory with zeros on the default stream and
	 * return at once; a stream_write_value32 queued behind it tells when
	 * it has run. Optional: without it a read of a hole fails. */
	int (*fill_dev_async)(void *dst, size_t bytes);
	/* Queue a copy from the device to pinned host memory on stream s (NULL
	 * is the default stream) and return at once. dst must stay unused
	 * until a stream_write_value32 or a launch_host_func queued behind it
	 * on the same stream tells that the copy has run. */
	int (*copy_from_dev_async)(void *dst, const void *src, size_t bytes,
	                           opends_accel_stream_t s);

	/* Per-stream gate, required only when assume_aligned_only is off:
	 * write a gate word in stream order. */
	int (*stream_write_value32)(opends_accel_stream_t s,
	                            opends_accel_devptr_t addr, uint32_t value);
	/* Hold the stream until the gate word reaches value (>=). */
	int (*stream_wait_value32_geq)(opends_accel_stream_t s,
	                               opends_accel_devptr_t addr,
	                               uint32_t value);

	/* Run fn(arg) on a host thread in stream order, holding the stream
	 * until it returns. fn must not call accelerator APIs. Required only
	 * when assume_aligned_only is on. */
	int (*launch_host_func)(opends_accel_stream_t s, void (*fn)(void *),
	                        void *arg);

	/* Execute the copy described at desc (a opends_bounce_copy; n_bytes ==
	 * 0 no-ops), ordered on the stream. The mechanism (a copy kernel or a
	 * stream-ordered memcpy) is the vendor's choice. */
	int (*copy_stream)(opends_accel_devptr_t desc,
	                   opends_accel_stream_t stream);
	/* Start the resident copy of a stream write whose parameters are read
	 * late (a opends_stage_copy at desc, the turn word, OPENDS_STAGE_BLOCKS
	 * landed words), ordered on the stream. It must hold the stream until
	 * it has copied the round marked last; see opends_bounce_kernel.h. */
	int (*stage_stream)(opends_accel_devptr_t desc,
	                    opends_accel_devptr_t turn,
	                    opends_accel_devptr_t landed,
	                    opends_accel_stream_t stream);
};

/* The active vendor, bound at link time by the one opends_accel_<vendor>.c
 * compiled into the backend. */
extern const struct opends_accel_ops *const opends_accel;

#endif /* OPENDS_ACCEL_H_ */
