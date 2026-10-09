/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Large-read coherency test for the aisio backend on a live HOMI/qublk stack.
 *
 * Reads a multi-MiB file through the aisio P2P path and verifies it
 * byte-for-byte against a kernel pread of the same file. This exercises
 * extent resolution and the MDTS chunking that the small (64 KiB) sync/async
 * read tests do not reach: a corruption that only appears above a few MiB
 * (wrong slbas, off-by-one chunking) shows up here as a byte mismatch.
 *
 * The file is passed on the command line; it must already exist on the mount
 * (e.g. one of the bench datasets or a prepared scratch file).
 */
#define _GNU_SOURCE

#include "opends.h"
#include "test_aisio_homi.h"
#include "test_cuda_common.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define READ_CAP ((size_t)64 * 1024 * 1024)

int
main(int argc, char **argv)
{
	if (argc < 2 || argc > 3) {
		fprintf(stderr, "usage: %s <file-on-mount> [dump-path]\n",
		        argv[0]);
		return 1;
	}
	const char *path = argv[1];

	struct stat st;
	if (stat(path, &st) < 0) {
		fprintf(stderr, "stat(%s) failed\n", path);
		return 1;
	}
	size_t size = (size_t)st.st_size;
	if (size > READ_CAP)
		size = READ_CAP;
	if (size == 0) {
		fprintf(stderr, "empty file: %s\n", path);
		return 1;
	}

	struct aisio_homi a;
	if (aisio_homi_setup(path, &a) < 0)
		return 1;

	/* Compare a block-aligned span: the kernel reference uses O_DIRECT to
	 * read the true on-disk bytes (a buffered read can return stale
	 * page-cache pages that differ from what the device, and thus the P2P
	 * path, sees). */
	size = size & ~((size_t)4095);
	if (size == 0) {
		fprintf(stderr, "file smaller than one block: %s\n", path);
		aisio_homi_teardown(&a);
		return 1;
	}

	int failed = 0;
	void *gpu = cuda_alloc_acquire(size);
	unsigned char *got = malloc(size);
	unsigned char *want = NULL;
	if (posix_memalign((void **)&want, 4096, size) != 0)
		want = NULL;
	if (!gpu || !got || !want) {
		fprintf(stderr, "allocation failed\n");
		failed = 1;
		goto out;
	}

	cuda_buf_zero(gpu, size);
	ssize_t n = opends_sync_read(a.fh, gpu, size, 0, 0);
	if (n != (ssize_t)size) {
		fprintf(stderr, "opends_sync_read: %zd (want %zu)\n", n, size);
		failed = 1;
		goto out;
	}
	cuda_buf_to_host(got, gpu, size);

	/* Optional: dump the exact aisio-read bytes for external comparison
	 * (sha256sum against dd O_DIRECT through qublk / kernel nvme). */
	if (argc >= 3) {
		FILE *df = fopen(argv[2], "wb");
		if (df) {
			fwrite(got, 1, size, df);
			fclose(df);
			fprintf(stderr, "aisio bytes dumped to %s (%zu)\n",
			        argv[2], size);
		}
	}

	/* Device-true kernel read of the same file for the reference bytes. */
	int kfd = open(path, O_RDONLY | O_DIRECT);
	if (kfd < 0) {
		fprintf(stderr, "open(%s, O_DIRECT) for verify failed\n", path);
		failed = 1;
		goto out;
	}
	/* Chunk the reference read into 1 MiB preads. */
	size_t done = 0;
	while (done < size) {
		size_t want_n = size - done;
		if (want_n > (size_t)1024 * 1024)
			want_n = (size_t)1024 * 1024;
		ssize_t r = pread(kfd, want + done, want_n, (off_t)done);
		if (r <= 0) {
			fprintf(stderr, "pread failed at %zu\n", done);
			failed = 1;
			break;
		}
		done += (size_t)r;
	}
	close(kfd);
	if (failed)
		goto out;

	size_t mism = 0, first = (size_t)-1;
	for (size_t i = 0; i < size; i++) {
		if (got[i] != want[i]) {
			if (first == (size_t)-1)
				first = i;
			mism++;
		}
	}
	if (mism) {
		fprintf(stderr,
		        "COHERENCY FAIL: %zu/%zu bytes differ, first at offset "
		        "%zu (got 0x%02x want 0x%02x)\n",
		        mism, size, first, got[first], want[first]);
		failed = 1;
	} else {
		fprintf(stderr, "coherency ok: %zu bytes match (%.2f MiB)\n",
		        size, size / (1024.0 * 1024.0));
	}

out:
	free(want);
	free(got);
	if (gpu)
		opends_free(gpu);
	aisio_homi_teardown(&a);

	if (!failed)
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
