/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_host_common.h - test_env callbacks for a buffer in host memory
 * (opends_mem_alloc with OPENDS_MEM_HOST).
 *
 * The aisio backend serves host memory only when
 * OPENDS_AISIO_WORKERS_PER_DRIVE names host, so host_mem_available lets a test
 * skip the mode instead of failing.
 */
#ifndef OPENDS_TEST_HOST_COMMON_H
#define OPENDS_TEST_HOST_COMMON_H

#include "opends.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void *
host_buf_to_host(void *dst, const void *src, size_t n)
{
	return memcpy(dst, src, n);
}

static inline void
host_buf_from_host(void *dst, const void *src, size_t n)
{
	memcpy(dst, src, n);
}

static inline void
host_buf_zero(void *buf, size_t n)
{
	memset(buf, 0, n);
}

static inline void
host_check_buffer(const void *buf)
{
	struct cudaPointerAttributes attrs;
	cudaError_t rc = cudaPointerGetAttributes(&attrs, buf);
	if (rc != cudaSuccess) {
		fprintf(stderr, "cudaPointerGetAttributes: %s\n",
		        cudaGetErrorString(rc));
		abort();
	}
	if (attrs.type == cudaMemoryTypeDevice) {
		fprintf(stderr,
		        "opends_mem_alloc(host) returned device memory\n");
		abort();
	}
}

static inline void *
host_alloc_acquire(size_t size)
{
	void *buf = NULL;
	opends_error_t err = opends_mem_alloc(size, OPENDS_MEM_HOST, 0, &buf);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  opends_mem_alloc(host, %zu): %s\n", size,
		        opends_op_status_error(err.err));
		return NULL;
	}
	return buf;
}

static inline void
host_alloc_release(void *buf)
{
	opends_free(buf);
}

/* 1 when the driver was opened with host memory, 0 when it was not (a
 * message says so), -1 when it is there but fails. */
static inline int
host_mem_available(void)
{
	void *buf = NULL;
	opends_error_t err = opends_mem_alloc(4096, OPENDS_MEM_HOST, 0, &buf);
	if (err.err == OPENDS_MEMORY_TYPE_INVALID) {
		fprintf(stderr, "host memory not configured "
		                "(OPENDS_AISIO_WORKERS_PER_DRIVE), "
		                "skipping host mode\n");
		return 0;
	}
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "opends_mem_alloc(host): %s\n",
		        opends_op_status_error(err.err));
		return -1;
	}
	opends_free(buf);
	return 1;
}

#endif /* OPENDS_TEST_HOST_COMMON_H */
