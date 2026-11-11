/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_write_homi.h - Shared write-test logic for the aisio backend on a live
 * HOMI/qublk stack.
 *
 * Writes go through the kernel-mounted filesystem (pwrite), so these tests run
 * against a real, writable file on the mount. The flow exercises an allocating
 * write into a freshly created (empty) file, aligned and unaligned overwrites
 * that must leave the surrounding bytes intact, and an unaligned write across
 * the end of the file. Each write is checked by the file size and a kernel
 * read of the enclosing blocks through the same O_DIRECT descriptor; a read
 * through OpenDS would go through the extent index, which does not follow
 * writes (see the README). The submit_write callback selects sync vs async.
 */
#ifndef TEST_WRITE_HOMI_H_
#define TEST_WRITE_HOMI_H_

#include "opends.h"
#include "read_pattern.h"
#include "test_cuda_common.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define WRITE_TAIL_KEEP 100
#define WRITE_PARTIAL_LEN (PAGE - WRITE_TAIL_KEEP)
#define WRITE_IMAGE_SIZE (FILE_SIZE + PAGE)

struct write_homi_env {
	opends_handle_t fh;
	int fd;
	opends_stream_t stream; /* unused by sync submit. */
	ssize_t (*submit_write)(struct write_homi_env *e, void *gpu,
	                        size_t size, off_t foff);
	const char *mode_label;
};

static inline unsigned char
write_pattern_byte(size_t i)
{
	return (unsigned char)((i * 7 + 0x5a) & 0xff);
}

/* Write image[foff, foff + size) at foff, check the file size, then read
 * the enclosing blocks back through the kernel and compare them with the
 * image. The descriptor is O_DIRECT, so the read is block-aligned and may
 * come back short at the end of the file. */
static int
write_verify(struct write_homi_env *e, void *gpu, unsigned char *host,
             const unsigned char *image, size_t *file_size, size_t size,
             off_t foff)
{
	cuda_buf_from_host(gpu, image + foff, size);
	ssize_t w = e->submit_write(e, gpu, size, foff);
	if (w != (ssize_t)size) {
		fprintf(stderr, "[%s] write %zu@%lld: %zd (%s)\n",
		        e->mode_label, size, (long long)foff, w,
		        w < 0 ? OPENDS_ERRSTR(w) : "short write");
		return -1;
	}

	if ((size_t)foff + size > *file_size)
		*file_size = (size_t)foff + size;
	struct stat st;
	int src = fstat(e->fd, &st);
	if (src < 0 || (size_t)st.st_size != *file_size) {
		fprintf(stderr,
		        "[%s] size after write %zu@%lld: %lld, want %zu\n",
		        e->mode_label, size, (long long)foff,
		        src < 0 ? -1LL : (long long)st.st_size, *file_size);
		return -1;
	}

	off_t roff = foff & ~(off_t)(PAGE - 1);
	size_t rend = ((size_t)foff + size + PAGE - 1) & ~(size_t)(PAGE - 1);
	size_t rlen = rend - (size_t)roff;
	size_t want = (*file_size < rend ? *file_size : rend) - (size_t)roff;
	memset(host, 0, rlen);
	ssize_t r = pread(e->fd, host, rlen, roff);
	if (r != (ssize_t)want || memcmp(host, image + roff, want)) {
		fprintf(stderr,
		        "[%s] verify %zu@%lld failed (r=%zd, want %zu)\n",
		        e->mode_label, rlen, (long long)roff, r, want);
		return -1;
	}
	fprintf(stderr, "[%s] write %zu@%lld + kernel read-back ok\n",
	        e->mode_label, size, (long long)foff);
	return 0;
}

/* Run against a handle whose file was created empty (O_CREAT|O_TRUNC). */
static int
run_write_homi_tests(struct write_homi_env *e)
{
	int failed = 0;
	size_t file_size = 0;
	void *gpu = opends_alloc(FILE_SIZE);
	unsigned char *host = NULL;
	unsigned char *image = malloc(WRITE_IMAGE_SIZE);
	if (posix_memalign((void **)&host, PAGE, WRITE_IMAGE_SIZE))
		host = NULL;
	if (!gpu || !host || !image) {
		fprintf(stderr, "[%s] allocation failed\n", e->mode_label);
		failed++;
		goto out;
	}

	/* 1. Allocating write of the whole file from empty. */
	for (size_t i = 0; i < FILE_SIZE; i++)
		image[i] = write_pattern_byte(i);
	if (write_verify(e, gpu, host, image, &file_size, FILE_SIZE, 0) < 0) {
		failed++;
		goto out;
	}

	/* 2. In-place sub-page overwrite; the page tail must survive. */
	memset(image, 0xBB, WRITE_PARTIAL_LEN);
	if (write_verify(e, gpu, host, image, &file_size, WRITE_PARTIAL_LEN,
	                 0) < 0)
		failed++;

	/* 3. One block in place at a block-aligned offset. */
	memset(image + PAGE, 0xAA, PAGE);
	if (write_verify(e, gpu, host, image, &file_size, PAGE, PAGE) < 0)
		failed++;

	/* 4. Unaligned span across a block boundary, in place. */
	memset(image + 2 * PAGE + 100, 0xCC, 5000);
	if (write_verify(e, gpu, host, image, &file_size, 5000,
	                 2 * PAGE + 100) < 0)
		failed++;

	/* 5. Unaligned span across the end of the file. */
	memset(image + FILE_SIZE - 500, 0xDD, 1000);
	if (write_verify(e, gpu, host, image, &file_size, 1000,
	                 (off_t)(FILE_SIZE - 500)) < 0)
		failed++;

	/* An unwritten extent past the end, for 7. It goes before the first
	 * read, which maps the file again. */
	if (fallocate(e->fd, 0, (off_t)(FILE_SIZE + PAGE), 2 * PAGE) < 0) {
		fprintf(stderr, "[%s] fallocate failed: %s\n", e->mode_label,
		        strerror(errno));
		failed++;
		goto out;
	}

	/* 6. The whole file back through OpenDS. The writes changed the file
	 * under its lease, and the first read after them maps it again. */
	cuda_buf_zero(gpu, FILE_SIZE);
	ssize_t r = opends_sync_read(e->fh, gpu, FILE_SIZE, 0, 0);
	cuda_buf_to_host(host, gpu, FILE_SIZE);
	if (r != (ssize_t)FILE_SIZE || memcmp(host, image, FILE_SIZE)) {
		fprintf(stderr,
		        "[%s] read-back through OpenDS failed (r=%zd)\n",
		        e->mode_label, r);
		failed++;
	} else {
		fprintf(stderr, "[%s] read-back through OpenDS ok\n",
		        e->mode_label);
	}

	/* 7. An unwritten extent reads as zeros through OpenDS, whatever its
	 * blocks hold. */
	cuda_buf_from_host(gpu, image, 2 * PAGE);
	r = opends_sync_read(e->fh, gpu, 2 * PAGE, (off_t)(FILE_SIZE + PAGE),
	                     0);
	cuda_buf_to_host(host, gpu, 2 * PAGE);
	size_t nonzero = 0;
	for (size_t i = 0; i < 2 * PAGE; i++)
		nonzero += host[i] != 0;
	if (r != (ssize_t)(2 * PAGE) || nonzero) {
		fprintf(stderr,
		        "[%s] unwritten extent read failed (r=%zd, %zu bytes "
		        "not zero)\n",
		        e->mode_label, r, nonzero);
		failed++;
	} else {
		fprintf(stderr, "[%s] unwritten extent reads as zeros ok\n",
		        e->mode_label);
	}

out:
	free(image);
	free(host);
	if (gpu)
		opends_free(gpu);
	return failed;
}

#endif /* TEST_WRITE_HOMI_H_ */
