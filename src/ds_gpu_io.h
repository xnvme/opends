/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_gpu_io.h - GPU-initiated read descriptor shared between opends_aisio.c
 * (host) and the vendor kernel TU (device).
 *
 * The host builds the NVMe commands and PRP lists when the op is submitted
 * and fills this descriptor; the kernel on the user's stream submits the
 * commands from GPU-resident queues and reaps them, so no host thread is on
 * the path once the stream reaches the op.
 */
#ifndef DS_GPU_IO_H
#define DS_GPU_IO_H

#include <stdint.h>

/* Blocks per op, one GPU-resident queue each. */
#define DS_GPU_MAX_BLOCKS 8

struct ds_gpu_op {
	/* struct xnvme_spec_cmd[n_cmds], device-visible. */
	uint64_t cmds;
	/* struct xnvme_cuda_queue *, one per block. */
	uint64_t queues[DS_GPU_MAX_BLOCKS];
	/* Sub-LBA tail copied out of the bounce slot; 0 bytes means none. */
	uint64_t tail_dst;
	uint64_t tail_src;
	uint32_t tail_nbytes;
	uint32_t n_cmds;
	/* Out: 0, or the first xnvme_cuda_cmd_io() error. */
	uint32_t status;
	/* Kernel: blocks that finished; the last one publishes the result. */
	uint32_t blocks_done;
	/* Result slot the kernel writes, device-visible, 0 = none, with the
	 * value on success and on failure. */
	uint64_t result;
	int64_t result_ok;
	int64_t result_err;
	/* Kernel: set once the op has completed and the result is written. */
	uint32_t done;
	uint32_t _pad;
};

#endif /* DS_GPU_IO_H */
