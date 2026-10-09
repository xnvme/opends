/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_stream_read.h" /* stream_test_stream_sync_timeout */
#include "test_cuda_common.h"
#include "test_write_homi.h"

#include <cuda.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static ssize_t
stream_submit_write(struct write_homi_env *e, void *gpu, size_t size,
                    off_t foff)
{
	size_t sz = size;
	off_t fo = foff;
	off_t bo = 0;
	ssize_t bytes = -1;
	opends_error_t err = opends_stream_write(e->fh, gpu, &sz, &fo, &bo,
	                                         &bytes, e->stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  stream_write submit: %s\n",
		        opends_op_status_error(err.err));
		return -1;
	}
	if (stream_test_stream_sync_timeout(e->stream, 10.0, "stream_write") !=
	    0)
		return -1;
	return bytes;
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <scratch-file-on-mount>\n", argv[0]);
		return 1;
	}
	const char *path = argv[1];

	struct aisio_homi a;
	if (aisio_homi_setup_flags(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT,
	                           &a) < 0)
		return 1;

	CUstream stream;
	if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
	    opends_stream_register(stream, 0).err != OPENDS_SUCCESS) {
		fprintf(stderr, "stream setup failed\n");
		aisio_homi_teardown(&a);
		unlink(path);
		return 1;
	}

	fprintf(stderr, "opends_stream_write tests (aisio backend, HOMI)\n");

	struct write_homi_env env = {
	        .fh = a.fh,
	        .fd = a.fd,
	        .stream = stream,
	        .submit_write = stream_submit_write,
	        .buf_acquire = cuda_alloc_acquire,
	        .buf_release = cuda_alloc_release,
	        .buf_from_host = cuda_buf_from_host,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .mode_label = "stream",
	};
	int failed = run_write_homi_tests(&env);
	failed += run_write_homi_tests_host(path, &env);

	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
	aisio_homi_teardown(&a);
	unlink(path);

	if (failed) {
		fprintf(stderr, "%d test(s) failed\n", failed);
		fflush(NULL);
		_exit(1);
	}
	fprintf(stderr, "all ok\n");
	return 0;
}
