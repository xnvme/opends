/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Shared CUDA helpers for GPU-backed backend tests (cufile, aisio).
 *
 * Provides the test_env callbacks that copy device buffers to host,
 * zero device buffers, and assert that opends_alloc returned CUDA
 * device memory.
 */

#ifndef OPENDS_TEST_CUDA_COMMON_H
#define OPENDS_TEST_CUDA_COMMON_H

#include "opends.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

static inline void *
cuda_buf_to_host(void *dst, const void *src, size_t n)
{
	cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
	return dst;
}

static inline void
cuda_buf_from_host(void *dst, const void *src, size_t n)
{
	cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

static inline void
cuda_buf_zero(void *buf, size_t n)
{
	cudaMemset(buf, 0, n);
	cudaDeviceSynchronize();
}

static inline void
cuda_check_buffer(const void *buf)
{
	struct cudaPointerAttributes attrs;
	cudaError_t rc = cudaPointerGetAttributes(&attrs, buf);
	if (rc != cudaSuccess) {
		fprintf(stderr, "cudaPointerGetAttributes: %s\n",
		        cudaGetErrorString(rc));
		abort();
	}
	if (attrs.type != cudaMemoryTypeDevice) {
		fprintf(stderr,
		        "opends_alloc returned non-device memory "
		        "(type=%d)\n",
		        (int)attrs.type);
		abort();
	}
}

static inline void *
cuda_alloc_acquire(size_t size)
{
	return opends_alloc(size);
}

static inline void
cuda_alloc_release(void *buf)
{
	opends_free(buf);
}

static inline void *
cuda_register_acquire(size_t size)
{
	void *buf = NULL;
	cudaError_t rc = cudaMalloc(&buf, size);
	if (rc != cudaSuccess) {
		fprintf(stderr, "  cudaMalloc: %s\n", cudaGetErrorString(rc));
		return NULL;
	}
	opends_error_t err = opends_buf_register(buf, size, 0);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  buf_register: %s\n",
		        opends_op_status_error(err.err));
		cudaFree(buf);
		return NULL;
	}
	return buf;
}

static inline void
cuda_register_release(void *buf)
{
	if (!buf)
		return;
	opends_buf_deregister(buf);
	cudaFree(buf);
}

/*
 * Register mode with a 4 KiB spacer allocated first: cudaMalloc then packs
 * the buffer behind it, off the GPU's 64 KiB page and sharing the spacer's
 * allocation chunk.
 */
#define CUDA_PACKED_SLOTS 16
static struct {
	void *buf;
	void *spacer;
} cuda_packed[CUDA_PACKED_SLOTS];

static inline void *
cuda_register_packed_acquire(size_t size)
{
	int slot;
	for (slot = 0; slot < CUDA_PACKED_SLOTS && cuda_packed[slot].buf;
	     slot++)
		;
	if (slot == CUDA_PACKED_SLOTS)
		return NULL;
	void *spacer = NULL;
	if (cudaMalloc(&spacer, 4096) != cudaSuccess)
		return NULL;
	void *buf = cuda_register_acquire(size);
	if (!buf) {
		cudaFree(spacer);
		return NULL;
	}
	cuda_packed[slot].buf = buf;
	cuda_packed[slot].spacer = spacer;
	return buf;
}

static inline void
cuda_register_packed_release(void *buf)
{
	if (!buf)
		return;
	cuda_register_release(buf);
	for (int slot = 0; slot < CUDA_PACKED_SLOTS; slot++) {
		if (cuda_packed[slot].buf == buf) {
			cudaFree(cuda_packed[slot].spacer);
			cuda_packed[slot].buf = NULL;
			cuda_packed[slot].spacer = NULL;
			break;
		}
	}
}

#endif /* OPENDS_TEST_CUDA_COMMON_H */
