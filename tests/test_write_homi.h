/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_write_homi.h - Shared write-test logic for the aisio backend on a live
 * HOMI/qublk stack.
 *
 * Writes go through the kernel-mounted filesystem (pwrite), so these tests run
 * against a real, writable file on the mount. The flow exercises an allocating
 * write into a freshly created (empty) file, aligned and unaligned overwrites
 * that must leave the surrounding bytes intact, and an unaligned write across
 * the end of the file. Each write is checked by the file size and a P2P
 * read-back of the enclosing blocks. The submit_write callback selects sync vs
 * async.
 */
#ifndef TEST_WRITE_HOMI_H_
#define TEST_WRITE_HOMI_H_

#include "opends.h"
#include "read_pattern.h"
#include "test_cuda_common.h"
#include "test_host_common.h"

#include <fcntl.h>
#include <limits.h>
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
	/* The buffer written from and read back into, and how to reach it. */
	void *(*buf_acquire)(size_t size);
	void (*buf_release)(void *buf);
	void (*buf_from_host)(void *dst, const void *src, size_t n);
	void *(*buf_to_host)(void *dst, const void *src, size_t n);
	void (*buf_zero)(void *buf, size_t n);
	const char *mode_label;
};

static inline unsigned char
write_pattern_byte(size_t i)
{
	return (unsigned char)((i * 7 + 0x5a) & 0xff);
}

/* Write image[foff, foff + size) at foff, check the file size, then read
 * the enclosing blocks back P2P and compare them with the image. */
static int
write_verify(struct write_homi_env *e, void *gpu, unsigned char *host,
             const unsigned char *image, size_t *file_size, size_t size,
             off_t foff)
{
	e->buf_from_host(gpu, image + foff, size);
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
	if (rend > *file_size)
		rend = *file_size;
	size_t rlen = rend - (size_t)roff;
	e->buf_zero(gpu, rlen);
	ssize_t r = opends_sync_read(e->fh, gpu, rlen, roff, 0);
	e->buf_to_host(host, gpu, rlen);
	if (r != (ssize_t)rlen || memcmp(host, image + roff, rlen)) {
		fprintf(stderr, "[%s] verify %zu@%lld failed (r=%zd)\n",
		        e->mode_label, rlen, (long long)roff, r);
		return -1;
	}
	fprintf(stderr, "[%s] write %zu@%lld + read-back ok\n", e->mode_label,
	        size, (long long)foff);
	return 0;
}

/* Run against a handle whose file was created empty (O_CREAT|O_TRUNC). */
static int
run_write_homi_tests(struct write_homi_env *e)
{
	int failed = 0;
	size_t file_size = 0;
	void *gpu = e->buf_acquire(FILE_SIZE);
	unsigned char *host = malloc(FILE_SIZE);
	unsigned char *image = malloc(WRITE_IMAGE_SIZE);
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

out:
	free(image);
	free(host);
	if (gpu)
		e->buf_release(gpu);
	return failed;
}

/* Run the write tests again with host buffers, on a scratch file next to
 * path, when the driver was opened with host memory. Returns the
 * failure count. */
static int
run_write_homi_tests_host(const char *path, const struct write_homi_env *base)
{
	struct write_homi_env e = *base;
	char hpath[PATH_MAX];
	char label[32];
	opends_error_t err;
	int host = host_mem_available();
	int failed;

	if (host <= 0)
		return host < 0 ? 1 : 0;

	snprintf(hpath, sizeof(hpath), "%s.host", path);
	snprintf(label, sizeof(label), "%s/host", base->mode_label);
	e.fh = NULL;
	e.fd = open(hpath, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);
	if (e.fd < 0) {
		fprintf(stderr, "open(%s) failed\n", hpath);
		return 1;
	}
	err = opends_handle_register(&e.fh, e.fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "handle_register(%s): %s\n", hpath,
		        opends_op_status_error(err.err));
		close(e.fd);
		unlink(hpath);
		return 1;
	}
	e.buf_acquire = host_alloc_acquire;
	e.buf_release = host_alloc_release;
	e.buf_from_host = host_buf_from_host;
	e.buf_to_host = host_buf_to_host;
	e.buf_zero = host_buf_zero;
	e.mode_label = label;

	failed = run_write_homi_tests(&e);

	opends_handle_deregister(e.fh);
	close(e.fd);
	unlink(hpath);
	return failed;
}

#endif /* TEST_WRITE_HOMI_H_ */
