/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * CUDA kernel for GPU-initiated stream reads. One block drives one
 * GPU-resident NVMe queue and the block size is the queue depth. The blocks
 * take turns over the command array: each submits a round of up to depth
 * commands and reaps it before the next round. The block that issued the
 * last command copies the sub-LBA tail out of the bounce slot once that
 * command has completed.
 *
 * Compiled by nvcc; does not include the host ds_accel.h shim.
 */

#include "ds_gpu_io.h"

#include <libxnvme.h>

#include <cuda_runtime.h>
#include <stdint.h>

static __global__ void
aisio_gpu_io_kernel(struct ds_gpu_op *op)
{
	struct xnvme_cuda_queue *qp =
	        (struct xnvme_cuda_queue *)(uintptr_t)op->queues[blockIdx.x];
	const struct xnvme_spec_cmd *cmds =
	        (const struct xnvme_spec_cmd *)(uintptr_t)op->cmds;
	uint32_t n = op->n_cmds;
	uint32_t depth = blockDim.x;
	uint32_t tid = threadIdx.x;
	uint32_t stride = gridDim.x * depth;
	int err = 0;

	for (uint32_t base = blockIdx.x * depth; base < n; base += stride) {
		uint32_t batch = min(depth, n - base);
		__align__(16) struct xnvme_spec_cmd cmd = {};
		int rc;

		if (tid < batch)
			cmd = cmds[base + tid];
		/* Every thread takes part in the round's barriers; threads past
		 * the batch submit nothing. */
		rc = xnvme_cuda_cmd_io(qp, &cmd, tid, batch);
		if (rc && !err)
			err = rc;
	}
	if (err)
		atomicCAS((unsigned int *)&op->status, 0u, (unsigned int)err);

	if (n && op->tail_nbytes &&
	    blockIdx.x == ((n - 1) / depth) % gridDim.x) {
		uint8_t *dst = (uint8_t *)(uintptr_t)op->tail_dst;
		/* Landed by DMA, so read it past the per-SM cache. */
		const volatile uint8_t *src =
		        (const volatile uint8_t *)(uintptr_t)op->tail_src;
		uint32_t nbytes = op->tail_nbytes;

		for (uint32_t off = tid; off < nbytes; off += depth)
			dst[off] = src[off];
	}

	/* The last block to finish publishes the result and the done word. */
	__syncthreads();
	if (tid == 0) {
		__threadfence();
		if (atomicAdd((unsigned int *)&op->blocks_done, 1u) ==
		    gridDim.x - 1) {
			uint32_t st;

			__threadfence();
			st = *(volatile uint32_t *)&op->status;
			if (op->result)
				*(volatile long long *)(uintptr_t)op->result =
				        st ? op->result_err : op->result_ok;
			__threadfence_system();
			*(volatile uint32_t *)&op->done = 1;
		}
	}
}

extern "C" int
cuda_gpu_io_launch(uint64_t op_dev, uint32_t nblocks, uint32_t depth,
                   void *stream)
{
	aisio_gpu_io_kernel<<<nblocks, depth, 0, (cudaStream_t)stream>>>(
	        (struct ds_gpu_op *)(uintptr_t)op_dev);
	return (int)cudaGetLastError();
}
