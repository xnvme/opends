/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_sync_read_host_aisio.c - the sync read tests on host buffers only,
 * for a driver that serves no GPU memory (OPENDS_AISIO_FABRICS).
 */
#define _GNU_SOURCE
#include "test_aisio_homi.h"
#include "test_host_common.h"
#include "test_sync_read.h"

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

	fprintf(stderr,
	        "opends_sync_read sync tests (aisio backend, host buffers)\n");

	int failed = 0;
	int host = host_mem_available();
	if (host <= 0) {
		fprintf(stderr, "host memory is required for this test\n");
		failed++;
	} else {
		struct test_env env_host = {
		        .fh = a.fh,
		        .buf_to_host = host_buf_to_host,
		        .buf_zero = host_buf_zero,
		        .check_buffer = host_check_buffer,
		        .buf_acquire = host_alloc_acquire,
		        .buf_release = host_alloc_release,
		        .mode_label = "host",
		        .rejects_nondword_head = 1,
		};
		failed += run_sync_read_tests(&env_host);
	}

	aisio_homi_teardown(&a);
	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
