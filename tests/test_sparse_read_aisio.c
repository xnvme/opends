/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MIB ((size_t)1024 * 1024)

/* The file tasks/test.yaml prepares: data, a hole, data, an unwritten extent,
 * and a hole up to the end, 1 MiB each. */
static const struct {
	const char *label;
	off_t off;
	size_t size;
} cases[] = {
        {"data", 0, MIB},
        {"hole", MIB, MIB},
        {"data into hole", MIB / 2, MIB},
        {"hole into data", MIB + MIB / 2, MIB},
        {"inside a hole, off an LBA", MIB + 100, 5000},
        {"unwritten extent", 3 * MIB, MIB},
        {"data into unwritten extent", 2 * MIB + MIB / 2, MIB},
        {"hole at the end", 4 * MIB, MIB},
        {"whole file", 0, 5 * MIB},
};

static int
stream_read(opends_handle_t fh, void *gpu, CUstream stream, off_t off,
            size_t size, ssize_t *bytes_read)
{
	off_t boff = 0;
	opends_error_t err;

	*bytes_read = 0;
	err = opends_stream_read(fh, gpu, &size, &off, &boff, bytes_read,
	                         stream);
	if (err.err != OPENDS_SUCCESS) {
		return -1;
	}
	return cuStreamSynchronize(stream) == CUDA_SUCCESS ? 0 : -1;
}

/* A stream read has no fill for a hole: it fails, also when extents come
 * before the hole, and the stream reads data afterwards. */
static int
stream_read_of_hole(struct aisio_homi *a, void *gpu, const unsigned char *want,
                    unsigned char *got)
{
	static const off_t holes[] = {MIB, MIB / 2};
	CUstream stream = NULL;
	ssize_t bytes_read;
	int failed = 1;
	int err;

	if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
	    opends_stream_register(stream, 0).err != OPENDS_SUCCESS) {
		fprintf(stderr, "[stream] no stream\n");
		return 1;
	}
	for (size_t i = 0; i < sizeof(holes) / sizeof(holes[0]); i++) {
		err = stream_read(a->fh, gpu, stream, holes[i], MIB,
		                  &bytes_read);
		if (err != 0 || bytes_read != -(ssize_t)OPENDS_INVALID_VALUE) {
			fprintf(stderr,
			        "[stream] read at %lld: %zd, expected %zd\n",
			        (long long)holes[i], bytes_read,
			        -(ssize_t)OPENDS_INVALID_VALUE);
			goto out;
		}
	}
	err = stream_read(a->fh, gpu, stream, 0, MIB, &bytes_read);
	cuda_buf_to_host(got, gpu, MIB);
	if (err != 0 || bytes_read != (ssize_t)MIB || memcmp(got, want, MIB)) {
		fprintf(stderr, "[stream] read of data after a hole: %zd\n",
		        bytes_read);
		goto out;
	}
	fprintf(stderr, "[stream] read of a hole refused ok\n");
	failed = 0;
out:
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
	return failed;
}

int
main(int argc, char **argv)
{
	size_t ncases = sizeof(cases) / sizeof(cases[0]);
	unsigned char *want = NULL;
	unsigned char *got = NULL;
	unsigned char *fill = NULL;
	struct aisio_homi a;
	struct stat st;
	void *gpu = NULL;
	int failed = 0;
	off_t hole;
	ssize_t r;
	int err;

	if (argc != 2) {
		fprintf(stderr, "usage: %s <sparse-file-on-mount>\n", argv[0]);
		return 1;
	}
	err = aisio_homi_setup(argv[1], &a);
	if (err < 0) {
		return 1;
	}
	err = fstat(a.fd, &st);
	hole = lseek(a.fd, 0, SEEK_HOLE);
	if (err != 0 || st.st_size != (off_t)(5 * MIB) || hole >= st.st_size) {
		fprintf(stderr, "%s is not the prepared sparse file\n",
		        argv[1]);
		aisio_homi_teardown(&a);
		return 1;
	}

	err = posix_memalign((void **)&want, 4096, 5 * MIB);
	got = malloc(5 * MIB);
	fill = malloc(5 * MIB);
	gpu = opends_alloc(5 * MIB);
	if (err != 0 || !got || !fill || !gpu) {
		fprintf(stderr, "allocation failed\n");
		aisio_homi_teardown(&a);
		return 1;
	}
	memset(fill, 0xA5, 5 * MIB);
	r = pread(a.fd, want, 5 * MIB, 0);
	if (r != (ssize_t)(5 * MIB)) {
		fprintf(stderr, "pread of the file: %zd\n", r);
		aisio_homi_teardown(&a);
		return 1;
	}

	for (size_t i = 0; i < ncases; i++) {
		cuda_buf_from_host(gpu, fill, cases[i].size);
		r = opends_sync_read(a.fh, gpu, cases[i].size, cases[i].off, 0);
		if (r != (ssize_t)cases[i].size) {
			fprintf(stderr, "[%s] read: %zd\n", cases[i].label, r);
			failed++;
			continue;
		}
		cuda_buf_to_host(got, gpu, cases[i].size);
		if (memcmp(got, want + cases[i].off, cases[i].size)) {
			fprintf(stderr, "[%s] differs from pread\n",
			        cases[i].label);
			failed++;
			continue;
		}
		fprintf(stderr, "[%s] ok\n", cases[i].label);
	}

	failed += stream_read_of_hole(&a, gpu, want, got);

	opends_free(gpu);
	free(want);
	free(got);
	free(fill);
	aisio_homi_teardown(&a);
	if (failed) {
		fprintf(stderr, "%d test(s) failed\n", failed);
	} else {
		fprintf(stderr, "all ok\n");
	}
	return failed ? 1 : 0;
}
