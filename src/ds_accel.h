/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_accel.h - vendor-neutral accelerator runtime interface for the aisio
 * backend.
 *
 * A vendor supports the backend by defining one struct ds_accel_ops instance
 * and binding the active-ops pointer ds_accel to it (see ds_accel_cuda.c). The
 * NVMe and extent code calls only through ds_accel, never a vendor symbol.
 *
 * Interface types are vendor-uniform so the ops ABI is fixed:
 *   - ds_accel_devptr_t is a 64-bit device address.
 *   - ds_accel_ctx_t and ds_accel_stream_t are opaque handles (void *).
 * Each vendor converts to and from its native pointer at the boundary (identity
 * for CUDA's CUdeviceptr; a cast for HIP's void *).
 */
#ifndef DS_ACCEL_H
#define DS_ACCEL_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t ds_accel_devptr_t;
typedef void *ds_accel_ctx_t;
typedef void *ds_accel_stream_t;

/* The interface an accelerator vendor implements. Each method returns 0 on
 * success and -1 on failure unless noted. */
struct ds_accel_ops {
	/* xNVMe P2P backend string for this vendor (e.g. "upcie-cuda"). */
	const char *xnvme_be;

	/* Capture the caller's current context (fails if there is none). */
	int (*ctx_get)(ds_accel_ctx_t *out);
	/* Make this thread's accelerator work target the captured context
	 * (CUDA: cuCtxSetCurrent; HIP: hipSetDevice). */
	void (*ctx_set)(ds_accel_ctx_t ctx);

	/* Host pinned memory the accelerator can read by device pointer. */
	int (*host_alloc_mapped)(size_t bytes, void **host,
	                         ds_accel_devptr_t *dptr);
	/* Free a host_alloc_mapped allocation. */
	void (*host_free)(void *host);

	/* Host/device bulk copy (direction inferred from the pointers). the
	 * calling thread must bind the accelerator context via ctx_set before
	 * using copy. */
	int (*copy)(void *dst, const void *src, size_t bytes);

	/* Per-stream gate, required only when assume_aligned_only is off:
	 * write a gate word in stream order. */
	int (*stream_write_value32)(ds_accel_stream_t s, ds_accel_devptr_t addr,
	                            uint32_t value);
	/* Hold the stream until the gate word reaches value (>=). */
	int (*stream_wait_value32_geq)(ds_accel_stream_t s,
	                               ds_accel_devptr_t addr, uint32_t value);

	/* Run fn(arg) on a host thread in stream order, holding the stream
	 * until it returns. fn must not call accelerator APIs. Required only
	 * when assume_aligned_only is on. */
	int (*launch_host_func)(ds_accel_stream_t s, void (*fn)(void *),
	                        void *arg);

	/* Execute the copy described at desc (a ds_bounce_copy; n_bytes == 0
	 * no-ops), ordered on the stream. The mechanism (a copy kernel or a
	 * stream-ordered memcpy) is the vendor's choice. */
	int (*copy_stream)(ds_accel_devptr_t desc, ds_accel_stream_t stream);
};

/* The active vendor, bound at link time by the one ds_accel_<vendor>.c
 * compiled into the backend. */
extern const struct ds_accel_ops *const ds_accel;

#endif /* DS_ACCEL_H */
