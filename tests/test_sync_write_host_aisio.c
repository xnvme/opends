/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_sync_write_host_aisio.c - the write tests on host buffers only, for a
 * driver that serves no GPU memory (OPENDS_AISIO_FABRICS).
 */
#define _GNU_SOURCE
#include "test_aisio_homi.h"
#include "test_host_common.h"
#include "test_write_homi.h"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

static ssize_t
sync_submit_write(struct write_homi_env *e, void *buf, size_t size, off_t foff)
{
	return opends_sync_write(e->fh, buf, size, foff, 0);
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

	fprintf(stderr,
	        "opends_sync_write tests (aisio backend, host buffers)\n");

	int failed = 0;
	int host = host_mem_available();
	if (host <= 0) {
		fprintf(stderr, "host memory is required for this test\n");
		failed++;
	} else {
		struct write_homi_env env = {
		        .fh = a.fh,
		        .fd = a.fd,
		        .stream = NULL,
		        .submit_write = sync_submit_write,
		        .buf_acquire = host_alloc_acquire,
		        .buf_release = host_alloc_release,
		        .buf_from_host = host_buf_from_host,
		        .buf_to_host = host_buf_to_host,
		        .buf_zero = host_buf_zero,
		        .mode_label = "sync/host",
		};
		failed += run_write_homi_tests(&env);
	}

	aisio_homi_teardown(&a);
	unlink(path);
	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
