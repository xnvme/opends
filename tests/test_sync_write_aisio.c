/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_cuda_common.h"
#include "test_write_homi.h"

#include <cuda.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static ssize_t
sync_submit_write(struct write_homi_env *e, void *gpu, size_t size, off_t foff)
{
	return opends_sync_write(e->fh, gpu, size, foff, 0);
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

	fprintf(stderr, "opends_sync_write tests (aisio backend, HOMI)\n");

	struct write_homi_env env = {
	        .fh = a.fh,
	        .fd = a.fd,
	        .stream = NULL,
	        .submit_write = sync_submit_write,
	        .buf_acquire = cuda_alloc_acquire,
	        .buf_release = cuda_alloc_release,
	        .buf_from_host = cuda_buf_from_host,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .mode_label = "sync",
	};
	int failed = run_write_homi_tests(&env);
	failed += run_write_homi_tests_host(path, &env);

	int bfd = open(path, O_RDWR);
	opends_handle_t bfh = NULL;
	opends_error_t rerr = opends_handle_register(&bfh, bfd);
	if (bfd < 0 || rerr.err != OPENDS_DIO_NOT_SET) {
		fprintf(stderr, "buffered fd not refused (fd %d, err %d)\n",
		        bfd, rerr.err);
		if (rerr.err == OPENDS_SUCCESS)
			opends_handle_deregister(bfh);
		failed++;
	} else {
		fprintf(stderr, "buffered fd refused ok\n");
	}
	if (bfd >= 0)
		close(bfd);

	aisio_homi_teardown(&a);
	unlink(path);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
