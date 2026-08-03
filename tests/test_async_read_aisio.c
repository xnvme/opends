/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_async_read.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <pattern-file-on-mount>\n", argv[0]);
		return 1;
	}

	struct aisio_homi a;
	if (aisio_homi_setup(argv[1], &a) < 0)
		return 1;

	CUstream main_stream;
	if (cuStreamCreate(&main_stream, CU_STREAM_NON_BLOCKING) !=
	    CUDA_SUCCESS) {
		fprintf(stderr, "cuStreamCreate(main) failed\n");
		aisio_homi_teardown(&a);
		return 1;
	}

	if (opends_stream_register(main_stream, 0).err != OPENDS_SUCCESS) {
		fprintf(stderr, "opends_stream_register(main) failed\n");
		cuStreamDestroy(main_stream);
		aisio_homi_teardown(&a);
		return 1;
	}

	CUstream extras[ASYNC_TEST_MAX_STREAMS];
	int extra_count = ASYNC_TEST_MAX_STREAMS;
	for (int i = 0; i < extra_count; i++) {
		if (cuStreamCreate(&extras[i], CU_STREAM_NON_BLOCKING) !=
		    CUDA_SUCCESS) {
			fprintf(stderr, "cuStreamCreate(extra[%d]) failed\n",
			        i);
			extra_count = i;
			break;
		}
		if (opends_stream_register(extras[i], 0).err !=
		    OPENDS_SUCCESS) {
			fprintf(stderr,
			        "opends_stream_register(extra[%d]) failed\n",
			        i);
			cuStreamDestroy(extras[i]);
			extra_count = i;
			break;
		}
	}

	fprintf(stderr, "opends_read_async tests (aisio backend, HOMI)\n");

	const char *aligned = getenv("OPENDS_AISIO_ASSUME_ALIGNED_ONLY");
	bool no_sub_lba = aligned && aligned[0] && aligned[0] != '0';

	struct async_test_env env_alloc = {
	        .fh = a.fh,
	        .stream = main_stream,
	        .extra_stream_count = extra_count,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .check_buffer = cuda_check_buffer,
	        .buf_acquire = cuda_alloc_acquire,
	        .buf_release = cuda_alloc_release,
	        .mode_label = "alloc",
	        .sub_lba_unsupported = no_sub_lba,
	};
	for (int i = 0; i < extra_count; i++)
		env_alloc.extra_streams[i] = extras[i];
	int failed = run_async_read_tests(&env_alloc);
	if (failed) {
		/* A failed alloc-mode test left a stream stuck on
		 * cuStreamWaitValue32; running register mode on the same
		 * streams would hang too. Bail out now. */
		fprintf(stderr, "%d test(s) failed\n", failed);
		fflush(NULL);
		_exit(1);
	}

	struct async_test_env env_register = {
	        .fh = a.fh,
	        .stream = main_stream,
	        .extra_stream_count = extra_count,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .check_buffer = cuda_check_buffer,
	        .buf_acquire = cuda_register_acquire,
	        .buf_release = cuda_register_release,
	        .mode_label = "register",
	        .sub_lba_unsupported = no_sub_lba,
	};
	for (int i = 0; i < extra_count; i++)
		env_register.extra_streams[i] = extras[i];
	failed += run_async_read_tests(&env_register);

	if (failed) {
		/* A timed-out test leaves streams stuck on
		 * cuStreamWaitValue32 and the I/O thread polling a never-
		 * ready event. Orderly teardown would block forever waiting
		 * for that work to drain, so report and bail out. */
		fprintf(stderr, "%d test(s) failed\n", failed);
		fflush(NULL);
		_exit(1);
	}

	for (int i = 0; i < extra_count; i++)
		cuStreamDestroy(extras[i]);
	cuStreamDestroy(main_stream);

	aisio_homi_teardown(&a);

	fprintf(stderr, "all ok\n");
	return 0;
}
