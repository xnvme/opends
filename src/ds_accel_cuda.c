/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_accel_cuda.c - CUDA implementation of the ds_accel.h interface.
 *
 * Defines the ds_accel_ops instance for CUDA and binds the active-ops
 * pointer ds_accel to it. The bounce launch lives in ds_bounce_kernel_cuda.cu
 * (the device translation unit); every other op maps onto the CUDA driver
 * API (cu*).
 */

#include "ds_accel.h"

#include <cuda.h>

/* The copy_stream kernel launcher, defined in the nvcc TU
 * (ds_bounce_kernel_cuda.cu). */
int cuda_copy_stream(uint64_t desc_dev, void *stream);

static int
cuda_ctx_get(ds_accel_ctx_t *out)
{
	CUcontext ctx;
	if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx)
		return -1;
	*out = ctx;
	return 0;
}

static void
cuda_ctx_set(ds_accel_ctx_t ctx)
{
	cuCtxSetCurrent((CUcontext)ctx);
}

static int
cuda_host_alloc_mapped(size_t bytes, void **host, ds_accel_devptr_t *dptr)
{
	if (cuMemHostAlloc(host, bytes,
	                   CU_MEMHOSTALLOC_DEVICEMAP |
	                           CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS)
		return -1;
	CUdeviceptr d = 0;
	if (cuMemHostGetDevicePointer(&d, *host, 0) != CUDA_SUCCESS) {
		cuMemFreeHost(*host);
		*host = NULL;
		return -1;
	}
	*dptr = (ds_accel_devptr_t)d;
	return 0;
}

static void
cuda_host_free(void *host)
{
	cuMemFreeHost(host);
}

static int
cuda_copy(void *dst, const void *src, size_t bytes)
{
	return cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, bytes) ==
	                       CUDA_SUCCESS
	               ? 0
	               : -1;
}

static int
cuda_stream_write_value32(ds_accel_stream_t s, ds_accel_devptr_t addr,
                          uint32_t value)
{
	return cuStreamWriteValue32((CUstream)s, (CUdeviceptr)addr, value,
	                            CU_STREAM_WRITE_VALUE_DEFAULT) ==
	                       CUDA_SUCCESS
	               ? 0
	               : -1;
}

static int
cuda_stream_wait_value32_geq(ds_accel_stream_t s, ds_accel_devptr_t addr,
                             uint32_t value)
{
	return cuStreamWaitValue32((CUstream)s, (CUdeviceptr)addr, value,
	                           CU_STREAM_WAIT_VALUE_GEQ) == CUDA_SUCCESS
	               ? 0
	               : -1;
}

static int
cuda_launch_host_func(ds_accel_stream_t s, void (*fn)(void *), void *arg)
{
	return cuLaunchHostFunc((CUstream)s, (CUhostFn)fn, arg) == CUDA_SUCCESS
	               ? 0
	               : -1;
}

static const struct ds_accel_ops cuda_ops = {
        .xnvme_be = "upcie-cuda",
        .ctx_get = cuda_ctx_get,
        .ctx_set = cuda_ctx_set,
        .host_alloc_mapped = cuda_host_alloc_mapped,
        .host_free = cuda_host_free,
        .copy = cuda_copy,
        .stream_write_value32 = cuda_stream_write_value32,
        .stream_wait_value32_geq = cuda_stream_wait_value32_geq,
        .launch_host_func = cuda_launch_host_func,
        .copy_stream = cuda_copy_stream,
};

const struct ds_accel_ops *const ds_accel = &cuda_ops;
