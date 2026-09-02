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

/*
 * xNVMe's upcie-cuda backend requires mem_map alignment to
 * cudamem_config.device_pagesize. Pad register-mode allocations up to that
 * granularity.
 */
#define CUDA_REGISTER_PAGE 65536
#define CUDA_REGISTER_ALIGN(x)                                                 \
	(((x) + (CUDA_REGISTER_PAGE - 1)) & ~((size_t)CUDA_REGISTER_PAGE - 1))

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
	void *buf = NULL;
	cudaError_t rc = cudaMalloc(&buf, aligned);
	if (rc != cudaSuccess) {
		fprintf(stderr, "  cudaMalloc: %s\n", cudaGetErrorString(rc));
		return NULL;
	}
	opends_error_t err = opends_buf_register(buf, aligned, 0);
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

#endif /* OPENDS_TEST_CUDA_COMMON_H */
