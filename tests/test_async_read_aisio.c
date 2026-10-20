/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_cuda_common.h"
#include "test_host_common.h"
#include "test_async_read.h"

#include <stdio.h>

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

	fprintf(stderr, "opends_async read test (aisio backend, HOMI)\n");

	struct async_read_env env = {
	        .fh = a.fh,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_acquire = cuda_alloc_acquire,
	        .buf_release = cuda_alloc_release,
	};
	int failed = run_async_read_test(&env);

	int host = host_mem_available();
	if (host < 0) {
		failed++;
	} else if (host) {
		fprintf(stderr, "opends_async read test (host buffers)\n");
		struct async_read_env env_host = {
		        .fh = a.fh,
		        .buf_to_host = host_buf_to_host,
		        .buf_acquire = host_alloc_acquire,
		        .buf_release = host_alloc_release,
		};
		failed += run_async_read_test(&env_host);
	}

	aisio_homi_teardown(&a);

	if (failed)
		fprintf(stderr, "%d failure(s)\n", failed);
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
