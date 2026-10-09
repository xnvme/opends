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

#include <stdint.h>
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

/*
 * xNVMe's upcie-cuda backend exports a registered range as a dma-buf, which
 * needs its start and size aligned to cudamem_config.device_pagesize.
 * cudaMalloc promises 256-byte alignment only, and small allocations land
 * wherever its sub-allocator is, so register-mode allocations are padded and
 * their start rounded up; the raw pointer is kept for cudaFree. The tests
 * acquire and release on one thread.
 */
#define CUDA_REGISTER_PAGE 65536
#define CUDA_REGISTER_ALIGN(x)                                                 \
	(((x) + (CUDA_REGISTER_PAGE - 1)) & ~((size_t)CUDA_REGISTER_PAGE - 1))
#define CUDA_REGISTER_SLOTS 4096

static void *cuda_register_raw[CUDA_REGISTER_SLOTS][2];

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
	size_t aligned = CUDA_REGISTER_ALIGN(size);
	void *raw = NULL;
	void *buf;
	int slot;
	cudaError_t rc = cudaMalloc(&raw, aligned + CUDA_REGISTER_PAGE);
	if (rc != cudaSuccess) {
		fprintf(stderr, "  cudaMalloc: %s\n", cudaGetErrorString(rc));
		return NULL;
	}
	for (slot = 0; slot < CUDA_REGISTER_SLOTS; slot++)
		if (!cuda_register_raw[slot][0])
			break;
	if (slot == CUDA_REGISTER_SLOTS) {
		fprintf(stderr, "  cuda_register_acquire: no free slot\n");
		cudaFree(raw);
		return NULL;
	}
	buf = (void *)(((uintptr_t)raw + CUDA_REGISTER_PAGE - 1) &
	               ~((uintptr_t)CUDA_REGISTER_PAGE - 1));
	opends_error_t err = opends_buf_register(buf, aligned, 0);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  buf_register: %s\n",
		        opends_op_status_error(err.err));
		cudaFree(raw);
		return NULL;
	}
	cuda_register_raw[slot][0] = buf;
	cuda_register_raw[slot][1] = raw;
	return buf;
}

static inline void
cuda_register_release(void *buf)
{
	if (!buf)
		return;
	opends_buf_deregister(buf);
	for (int slot = 0; slot < CUDA_REGISTER_SLOTS; slot++) {
		if (cuda_register_raw[slot][0] == buf) {
			cudaFree(cuda_register_raw[slot][1]);
			cuda_register_raw[slot][0] = NULL;
			cuda_register_raw[slot][1] = NULL;
			return;
		}
	}
	cudaFree(buf);
}

#endif /* OPENDS_TEST_CUDA_COMMON_H */
