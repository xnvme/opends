/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_accel.h - vendor-neutral accelerator runtime interface for the aisio
 * backend.
 *
 * The aisio NVMe path needs only a narrow accelerator surface: bind a
 * thread to the caller's context, allocate host pinned memory the
 * accelerator can read, copy between host and device, gate a stream on a
 * host-written word, and run a bounce copy ordered on a stream.
 * struct ds_accel_ops is that surface. A vendor supports the backend by
 * defining one ds_accel_ops instance and binding the active-ops pointer
 * ds_accel to it (see ds_accel_cuda.c). The NVMe and extent code calls
 * only through ds_accel, never a vendor symbol.
 *
 * Interface types are vendor-uniform so the ops ABI is fixed:
 *   - ds_accel_ctx_t and ds_accel_stream_t are opaque handles (void *).
 *   - ds_accel_devptr_t is an accelerator device address as a plain integer.
 * Each vendor converts to and from its native pointer at the boundary (identity
 * for CUDA's CUdeviceptr; a cast for HIP's void *), so callers can do ordinary
 * arithmetic on it.
 *
 * ctx_set is a semantic, "make this thread's accelerator work target the
 * captured device", not a specific primitive: CUDA uses cuCtxSetCurrent,
 * HIP would use hipSetDevice.
 *
 * The async path orders the I/O thread against the user's stream with a
 * per-stream gate word the NVMe code owns: stream_write_value32 publishes
 * the stream's arrival, stream_wait_value32_geq parks it, and the I/O
 * thread's plain host store releases it. Only the two stream enqueue ops
 * are vendor-specific; the word storage and the host-side poll/release
 * live in the NVMe code.
 */
#ifndef DS_ACCEL_H
#define DS_ACCEL_H

#include <stddef.h>
#include <stdint.h>

typedef void *ds_accel_ctx_t;
typedef void *ds_accel_stream_t;
typedef uint64_t ds_accel_devptr_t;

/* The interface an accelerator vendor implements. Each method returns 0 on
 * success and -1 on failure unless noted. */
struct ds_accel_ops {
	/* xNVMe P2P backend string for this vendor (e.g. "upcie-cuda"). */
	const char *xnvme_be;

	/* Capture the caller's current context (fails if there is none). */
	int (*ctx_get)(ds_accel_ctx_t *out);
	/* Make this thread's accelerator work target the captured context. */
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

	/* Per-stream gate: write a gate word in stream order. */
	int (*stream_write_value32)(ds_accel_stream_t s, ds_accel_devptr_t addr,
	                            uint32_t value);
	/* Hold the stream until the gate word reaches value (>=). */
	int (*stream_wait_value32_geq)(ds_accel_stream_t s,
	                               ds_accel_devptr_t addr, uint32_t value);

	/* Execute the copy described at desc (a ds_bounce_copy; n_bytes == 0
	 * no-ops), ordered on the stream. The mechanism (a copy kernel or a
	 * stream-ordered memcpy) is the vendor's choice. */
	int (*copy_stream)(ds_accel_devptr_t desc, ds_accel_stream_t stream);
};

/* The active vendor, bound at link time by the one ds_accel_<vendor>.c
 * compiled into the backend. */
extern const struct ds_accel_ops *const ds_accel;

#endif /* DS_ACCEL_H */
