/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * CUDA bounce copy kernel for the aisio async path. The host enqueues it
 * behind the stream wait-value gate (see ds_bounce_kernel.h), so by launch
 * time NVMe has filled the bounce slot and the descriptor is published.
 * Threads stride bytes within the single copy; n_bytes == 0 falls through,
 * so the launch is safe for a no-op op.
 *
 * Compiled by nvcc; does not include the host ds_accel.h shim.
 */

#include "ds_bounce_kernel.h"

#include <cuda_runtime.h>
#include <stdint.h>

static __global__ void
cuda_bounce_kernel(const struct ds_bounce_copy *desc)
{
	uint8_t *dst = (uint8_t *)(uintptr_t)desc->dst;
	const uint8_t *src = (const uint8_t *)(uintptr_t)desc->src;
	uint32_t n = desc->n_bytes;

	for (uint32_t off = threadIdx.x; off < n; off += blockDim.x)
		dst[off] = src[off];
}

extern "C" int
cuda_copy_stream(uint64_t desc_dev, void *stream)
{
	cuda_bounce_kernel<<<1, 128, 0, (cudaStream_t)stream>>>(
	        (const struct ds_bounce_copy *)(uintptr_t)desc_dev);
	return (int)cudaGetLastError();
}

static __device__ void
stage_sleep(void)
{
#if __CUDA_ARCH__ >= 700
	__nanosleep(2000);
#endif
}

/* Device to mapped host memory, the grid striding over the bytes. 16-byte
 * stores when both ends allow, so a warp fills whole link packets. */
static __device__ void
stage_copy(uint64_t dst, uint64_t src, uint64_t n)
{
	uint8_t *d = (uint8_t *)(uintptr_t)dst;
	const uint8_t *s = (const uint8_t *)(uintptr_t)src;
	uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t nthreads = (uint64_t)gridDim.x * blockDim.x;
	uint64_t nvec = 0;

	if (((dst | src) & 15) == 0) {
		const uint4 *sv = (const uint4 *)s;
		uint4 *dv = (uint4 *)d;

		nvec = n / 16;
		for (uint64_t i = tid; i < nvec; i += nthreads)
			dv[i] = sv[i];
	}
	for (uint64_t i = nvec * 16 + tid; i < n; i += nthreads)
		d[i] = s[i];
}

/* Resident for one stream write: round after round as the I/O thread grants
 * them through turn. Every word it polls or writes is in mapped host memory,
 * so the thread sees a landed round as a plain load. */
static __global__ void
cuda_stage_kernel(const volatile struct ds_stage_copy *desc,
                  const volatile uint32_t *turn, volatile uint32_t *landed)
{
	__shared__ uint64_t s_src;
	__shared__ uint64_t s_dst;
	__shared__ uint64_t s_n;
	__shared__ uint32_t s_last;
	uint32_t round = 1;
	uint32_t last;

	for (;;) {
		if (threadIdx.x == 0) {
			while ((int32_t)(*turn - round) < 0)
				stage_sleep();
			__threadfence_system();
			s_src = desc->src;
			s_dst = desc->dst;
			s_n = desc->n_bytes;
			s_last = desc->last;
		}
		__syncthreads();
		stage_copy(s_dst, s_src, s_n);
		__threadfence_system();
		__syncthreads();
		if (threadIdx.x == 0)
			landed[blockIdx.x] = round;
		last = s_last;
		__syncthreads();
		if (last)
			break;
		round++;
	}
}

/* Under lazy loading a kernel's module loads at its first launch, with a
 * context synchronization; asking for the attributes loads it now. */
extern "C" int
cuda_kernels_load(void)
{
	struct cudaFuncAttributes attr;
	cudaError_t rc;

	rc = cudaFuncGetAttributes(&attr, cuda_bounce_kernel);
	if (rc != cudaSuccess)
		return (int)rc;
	rc = cudaFuncGetAttributes(&attr, cuda_stage_kernel);
	return (int)rc;
}

extern "C" int
cuda_stage_stream(uint64_t desc_dev, uint64_t turn_dev, uint64_t landed_dev,
                  void *stream)
{
	cuda_stage_kernel<<<DS_STAGE_BLOCKS, DS_STAGE_THREADS, 0,
	                    (cudaStream_t)stream>>>(
	        (const volatile struct ds_stage_copy *)(uintptr_t)desc_dev,
	        (const volatile uint32_t *)(uintptr_t)turn_dev,
	        (volatile uint32_t *)(uintptr_t)landed_dev);
	return (int)cudaGetLastError();
}
