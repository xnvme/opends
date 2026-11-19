/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_accel_rocm.c - HIP implementation of the ds_accel.h interface, for AMD
 * GPUs under ROCm.
 *
 * Defines the ds_accel_ops instance for HIP and binds the active-ops pointer
 * ds_accel to it. The kernel launches live in ds_bounce_kernel_rocm.hip (the
 * hipcc translation unit); every other op maps onto the HIP runtime API.
 */

#define __HIP_PLATFORM_AMD__

#include "ds_accel.h"

#include <hip/hip_runtime_api.h>

/* The launchers and the loader, defined in the hipcc TU
 * (ds_bounce_kernel_rocm.hip). */
int rocm_copy_stream(uint64_t desc_dev, void *stream);
int rocm_stage_stream(uint64_t desc_dev, uint64_t turn_dev, uint64_t landed_dev,
                      void *stream);
int rocm_kernels_load(void);

/* When the runtime cannot say, round registrations to the largest
 * granularity an AMD GPU uses for device memory. */
#define ROCM_FALLBACK_GRANULARITY (2u << 20)

static int
rocm_ctx_get(ds_accel_ctx_t *out)
{
	hipCtx_t ctx = NULL;
	hipError_t rc;

	rc = hipCtxGetCurrent(&ctx);
	if (rc != hipSuccess) {
		return (int)rc;
	}
	if (!ctx) {
		return -1;
	}
	*out = ctx;
	return 0;
}

static void
rocm_ctx_set(ds_accel_ctx_t ctx)
{
	hipCtxSetCurrent((hipCtx_t)ctx);
}

static int
rocm_alloc_granularity(size_t *out)
{
	hipMemAllocationProp prop = {0};
	int dev = 0;
	hipError_t rc;

	rc = hipGetDevice(&dev);
	if (rc != hipSuccess) {
		return (int)rc;
	}
	prop.type = hipMemAllocationTypePinned;
	prop.location.type = hipMemLocationTypeDevice;
	prop.location.id = dev;
	rc = hipMemGetAllocationGranularity(out, &prop,
	                                    hipMemAllocationGranularityMinimum);
	if (rc != hipSuccess || *out == 0) {
		*out = ROCM_FALLBACK_GRANULARITY;
	}
	return 0;
}

static int
rocm_host_alloc_mapped(size_t bytes, void **host, ds_accel_devptr_t *dptr)
{
	void *d = NULL;
	hipError_t rc;

	rc = hipHostMalloc(host, bytes,
	                   hipHostMallocMapped | hipHostMallocPortable |
	                           hipHostMallocCoherent);
	if (rc != hipSuccess) {
		return (int)rc;
	}
	rc = hipHostGetDevicePointer(&d, *host, 0);
	if (rc != hipSuccess) {
		hipHostFree(*host);
		*host = NULL;
		return (int)rc;
	}
	*dptr = (ds_accel_devptr_t)(uintptr_t)d;
	return 0;
}

static void
rocm_host_free(void *host)
{
	hipHostFree(host);
}

static int
rocm_host_pin(void *host, size_t bytes, ds_accel_devptr_t *dptr)
{
	void *d = NULL;
	hipError_t rc;

	rc = hipHostRegister(host, bytes, hipHostRegisterMapped);
	if (rc != hipSuccess) {
		return (int)rc;
	}
	if (dptr) {
		rc = hipHostGetDevicePointer(&d, host, 0);
		if (rc != hipSuccess) {
			hipHostUnregister(host);
			return (int)rc;
		}
		*dptr = (ds_accel_devptr_t)(uintptr_t)d;
	}
	return 0;
}

static void
rocm_host_unpin(void *host)
{
	hipHostUnregister(host);
}

static int
rocm_copy(void *dst, const void *src, size_t bytes)
{
	return (int)hipMemcpy(dst, src, bytes, hipMemcpyDefault);
}

static int
rocm_copy_to_dev_async(void *dst, const void *src, size_t bytes)
{
	return (int)hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
	                           NULL);
}

static int
rocm_copy_from_dev_async(void *dst, const void *src, size_t bytes,
                         ds_accel_stream_t s)
{
	return (int)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
	                           (hipStream_t)s);
}

static int
rocm_stream_write_value32(ds_accel_stream_t s, ds_accel_devptr_t addr,
                          uint32_t value)
{
	return (int)hipStreamWriteValue32((hipStream_t)s,
	                                  (void *)(uintptr_t)addr, value, 0);
}

static int
rocm_stream_wait_value32_geq(ds_accel_stream_t s, ds_accel_devptr_t addr,
                             uint32_t value)
{
	return (int)hipStreamWaitValue32((hipStream_t)s,
	                                 (void *)(uintptr_t)addr, value,
	                                 hipStreamWaitValueGte, 0xffffffffu);
}

static int
rocm_launch_host_func(ds_accel_stream_t s, void (*fn)(void *), void *arg)
{
	return (int)hipLaunchHostFunc((hipStream_t)s, (hipHostFn_t)fn, arg);
}

static const struct ds_accel_ops rocm_ops = {
        .xnvme_be = "upcie-hip",
        .load = rocm_kernels_load,
        .ctx_get = rocm_ctx_get,
        .ctx_set = rocm_ctx_set,
        .alloc_granularity = rocm_alloc_granularity,
        .host_alloc_mapped = rocm_host_alloc_mapped,
        .host_free = rocm_host_free,
        .host_pin = rocm_host_pin,
        .host_unpin = rocm_host_unpin,
        .copy = rocm_copy,
        .copy_to_dev_async = rocm_copy_to_dev_async,
        .copy_from_dev_async = rocm_copy_from_dev_async,
        .stream_write_value32 = rocm_stream_write_value32,
        .stream_wait_value32_geq = rocm_stream_wait_value32_geq,
        .launch_host_func = rocm_launch_host_func,
        .copy_stream = rocm_copy_stream,
        .stage_stream = rocm_stage_stream,
};

const struct ds_accel_ops *const ds_accel = &rocm_ops;
