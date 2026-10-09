/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

/*
 * Allocating-write test for the aisio backend on a live HOMI/qublk stack.
 *
 * The LMCache-representative case: create an empty file, write one chunk into
 * it (which must allocate blocks through the kernel-mounted XFS), then P2P
 * read the chunk back and verify. Previously the aisio backend rejected
 * allocating writes with OPENDS_IO_NOT_SUPPORTED; with the pwrite path they
 * succeed.
 */

#include "opends.h"
#include "read_pattern.h"
#include "test_aisio_homi.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <scratch-file-on-mount>\n", argv[0]);
		return 1;
	}
	const char *path = argv[1];
	const size_t blk = PAGE;

	struct aisio_homi a;
	if (aisio_homi_setup_flags(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT,
	                           &a) < 0)
		return 1;

	fprintf(stderr,
	        "opends_sync_write block-alloc test (aisio backend, HOMI)\n");

	int failed = 0;
	void *gpu = cuda_alloc_acquire(blk);
	unsigned char *exp = malloc(blk);
	unsigned char *host = malloc(blk);
	if (!gpu || !exp || !host) {
		fprintf(stderr, "allocation failed\n");
		failed++;
		goto out;
	}

	for (size_t i = 0; i < blk; i++)
		exp[i] = (unsigned char)((i + 0x33) & 0xff);
	cuda_buf_from_host(gpu, exp, blk);

	ssize_t w = opends_sync_write(a.fh, gpu, blk, 0, 0);
	if (w != (ssize_t)blk) {
		fprintf(stderr, "  block-alloc write: %zd\n", w);
		failed++;
		goto out;
	}

	struct stat st;
	if (fstat(a.fd, &st) == 0 && st.st_size < (off_t)blk) {
		fprintf(stderr, "  file did not grow (size=%lld)\n",
		        (long long)st.st_size);
		failed++;
	}

	cuda_buf_zero(gpu, blk);
	ssize_t r = opends_sync_read(a.fh, gpu, blk, 0, 0);
	cuda_buf_to_host(host, gpu, blk);
	if (r != (ssize_t)blk || memcmp(host, exp, blk)) {
		fprintf(stderr, "  block-alloc verify failed (r=%zd)\n", r);
		failed++;
	} else {
		fprintf(stderr, "  block-alloc write + read-back ok\n");
	}

out:
	free(host);
	free(exp);
	if (gpu)
		opends_free(gpu);
	aisio_homi_teardown(&a);
	unlink(path);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
