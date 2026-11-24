/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Regression test: opends_buf_register on an 8 GiB cudaMalloc'd
 * buffer. Mirrors the bench-time flow where fil allocates a single
 * device buffer sized to the dataset's max file (8 GiB for the
 * filesize8gib dataset) and hands it to opends_buf_register.
 *
 * Catches regressions in the cudaMalloc → xnvme_buf_register →
 * cuMemGetHandleForAddressRange path at scale (LUT capacity,
 * alloc_granularity handling) that small-buffer tests
 * don't exercise.
 */

#define _GNU_SOURCE

#include "opends.h"
#include "test_aisio_homi.h"

#include <cuda_runtime.h>

#include <stdio.h>

#define BUF_SIZE ((size_t)8 * 1024 * 1024 * 1024)

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <file-on-mount>\n", argv[0]);
		return 1;
	}

	struct aisio_homi a;
	if (aisio_homi_setup(argv[1], &a) < 0)
		return 1;

	void *buf = NULL;
	cudaError_t cerr = cudaMalloc(&buf, BUF_SIZE);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(%zu): %s\n", BUF_SIZE,
		        cudaGetErrorString(cerr));
		aisio_homi_teardown(&a);
		return 1;
	}

	opends_error_t err = opends_buf_register(buf, BUF_SIZE, 0);
	int failed = 0;
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "buf_register(%zu): %s\n", BUF_SIZE,
		        opends_op_status_error(err.err));
		failed = 1;
	} else {
		opends_buf_deregister(buf);
	}

	cudaFree(buf);
	aisio_homi_teardown(&a);

	if (failed)
		fprintf(stderr, "register_large_aisio FAIL\n");
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
