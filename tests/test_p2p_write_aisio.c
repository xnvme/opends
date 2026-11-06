/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

/*
 * In-place write test for the aisio backend on a live HOMI/qublk stack.
 *
 * A write inside a file's allocated extents goes GPU to NVMe by DMA and leaves
 * the extent index alone; the kernel path carries the rest. The file was
 * written before xal-server indexed, so its extents are in the index, and
 * every write here stays inside them. The parts:
 *   1. DMA overwrite of a file the kernel wrote: the kernel reads the new
 *      bytes back, the size and mtime do not move, and a read issued right
 *      after does not wait for a re-index.
 *   2. A source buffer that is not registered takes the kernel path: the
 *      kernel reads the bytes back and mtime moves.
 */

#include "opends.h"
#include "test_aisio_homi.h"
#include "test_cuda_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define KIB ((size_t)1024)
#define MIB (1024 * KIB)
#define ALIGN 4096
#define FILE1_SIZE (1 * MIB)
#define READ_AFTER_WRITE_MAX_MS 80.0

static void
fill(unsigned char *p, size_t n, unsigned seed)
{
	for (size_t i = 0; i < n; i++)
		p[i] = (unsigned char)((i * 7 + seed) & 0xff);
}

static void *
host_aligned(size_t n)
{
	void *p = NULL;

	if (posix_memalign(&p, ALIGN, n) != 0)
		return NULL;
	memset(p, 0, n);
	return p;
}

static int
kernel_write(int fd, const void *host, size_t n, off_t off)
{
	size_t done = 0;

	while (done < n) {
		ssize_t w = pwrite(fd, (const unsigned char *)host + done,
		                   n - done, off + (off_t)done);
		if (w < 0 && errno == EINTR)
			continue;
		if (w <= 0)
			return -1;
		done += (size_t)w;
	}
	return 0;
}

static int
kernel_read(int fd, void *host, size_t n, off_t off)
{
	size_t done = 0;

	while (done < n) {
		ssize_t r = pread(fd, (unsigned char *)host + done, n - done,
		                  off + (off_t)done);
		if (r < 0 && errno == EINTR)
			continue;
		if (r <= 0)
			return -1;
		done += (size_t)r;
	}
	return 0;
}

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static bool
mtime_eq(const struct stat *a, const struct stat *b)
{
	return a->st_mtim.tv_sec == b->st_mtim.tv_sec &&
	       a->st_mtim.tv_nsec == b->st_mtim.tv_nsec;
}

static int
opends_read_check(const char *what, opends_handle_t fh, void *gpu,
                  unsigned char *host, const unsigned char *want, size_t n,
                  off_t off, double *ms)
{
	double t0 = now_ms();
	ssize_t r = opends_sync_read(fh, gpu, n, off, 0);
	double dt = now_ms() - t0;

	cuda_buf_to_host(host, gpu, n);
	if (ms)
		*ms = dt;
	if (r != (ssize_t)n || memcmp(host, want, n)) {
		fprintf(stderr,
		        "  %s: opends read-back %zu@%lld failed (r=%zd)\n",
		        what, n, (long long)off, r);
		return -1;
	}
	return 0;
}

static int
kernel_read_check(const char *what, int fd, unsigned char *host,
                  const unsigned char *want, size_t n, off_t off)
{
	int err = kernel_read(fd, host, n, off);
	if (err < 0 || memcmp(host, want, n)) {
		fprintf(stderr, "  %s: kernel read-back %zu@%lld failed\n",
		        what, n, (long long)off);
		return -1;
	}
	return 0;
}

/* Write src through OpenDS and expect the DMA path: the size and mtime stay.
 * A write the kernel carried moves mtime, which is how a file the index does
 * not hold shows up. */
static int
dma_write(const char *what, opends_handle_t fh, int fd, void *gpu, size_t n,
          off_t off, off_t want_size)
{
	struct stat before, after;

	if (fstat(fd, &before) < 0)
		return -1;
	ssize_t w = opends_sync_write(fh, gpu, n, off, 0);
	if (w != (ssize_t)n) {
		fprintf(stderr, "  %s: write %zu@%lld: %zd (%s)\n", what, n,
		        (long long)off, w,
		        w < 0 ? OPENDS_ERRSTR(w) : "short write");
		return -1;
	}
	if (fstat(fd, &after) < 0)
		return -1;
	if (after.st_size != want_size) {
		fprintf(stderr, "  %s: size %lld after write, want %lld\n",
		        what, (long long)after.st_size, (long long)want_size);
		return -1;
	}
	if (!mtime_eq(&before, &after)) {
		fprintf(stderr,
		        "  %s: mtime moved, the write took the kernel path (is "
		        "the file in the index?)\n",
		        what);
		return -1;
	}
	return 0;
}

static int
part_written_file(struct aisio_homi *a, void *gpu, unsigned char *host,
                  unsigned char *image)
{
	const size_t off = 256 * KIB, len = 512 * KIB;
	double ms = 0;

	fill(image, FILE1_SIZE, 1);
	if (kernel_write(a->fd, image, FILE1_SIZE, 0) < 0) {
		fprintf(stderr, "  kernel write of the file failed\n");
		return -1;
	}

	fill(image + off, len, 2);
	cuda_buf_from_host(gpu, image + off, len);
	if (dma_write("written", a->fh, a->fd, gpu, len, (off_t)off,
	              (off_t)FILE1_SIZE) < 0)
		return -1;

	cuda_buf_zero(gpu, len);
	if (opends_read_check("written", a->fh, gpu, host, image + off, len,
	                      (off_t)off, &ms) < 0)
		return -1;
	fprintf(stderr, "  read after DMA write: %.1f ms\n", ms);
	if (ms > READ_AFTER_WRITE_MAX_MS) {
		fprintf(stderr,
		        "  read after DMA write waited for a re-index\n");
		return -1;
	}
	if (kernel_read_check("written", a->fd, host, image, FILE1_SIZE, 0) < 0)
		return -1;
	fprintf(stderr, "[1] DMA overwrite of a written file ok\n");
	return 0;
}

static int
part_unregistered_source(struct aisio_homi *a, unsigned char *host,
                         unsigned char *image)
{
	const size_t off = 768 * KIB, len = 128 * KIB;
	struct stat before, after;
	void *raw = NULL;
	int failed = -1;

	if (cudaMalloc(&raw, len) != cudaSuccess) {
		fprintf(stderr, "  cudaMalloc failed\n");
		return -1;
	}
	fill(image, len, 4);
	cuda_buf_from_host(raw, image, len);

	/* mtime is coarse; put the write in a later tick than the last one. */
	usleep(20000);
	if (fstat(a->fd, &before) < 0)
		goto out;
	ssize_t w = opends_sync_write(a->fh, raw, len, (off_t)off, 0);
	if (w != (ssize_t)len) {
		fprintf(stderr, "  unregistered: write %zd\n", w);
		goto out;
	}
	if (fstat(a->fd, &after) < 0)
		goto out;
	if (after.st_size != (off_t)FILE1_SIZE || mtime_eq(&before, &after)) {
		fprintf(stderr,
		        "  unregistered: size %lld, mtime %s; want the kernel "
		        "path\n",
		        (long long)after.st_size,
		        mtime_eq(&before, &after) ? "unchanged" : "moved");
		goto out;
	}
	if (kernel_read_check("unregistered", a->fd, host, image, len,
	                      (off_t)off) < 0)
		goto out;
	fprintf(stderr, "[2] unregistered source took the kernel path ok\n");
	failed = 0;
out:
	cudaFree(raw);
	return failed;
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <indexed-file-on-mount>\n", argv[0]);
		return 1;
	}
	const char *path = argv[1];
	int failed = 0;

	struct aisio_homi a;
	if (aisio_homi_setup_flags(path, O_RDWR | O_DIRECT, &a) < 0)
		return 1;

	fprintf(stderr, "in-place write tests (aisio backend, HOMI)\n");

	void *gpu = opends_alloc(FILE1_SIZE);
	unsigned char *host = host_aligned(FILE1_SIZE);
	unsigned char *image = host_aligned(FILE1_SIZE);
	if (!gpu || !host || !image) {
		fprintf(stderr, "allocation failed\n");
		failed = 1;
		goto out;
	}

	if (part_written_file(&a, gpu, host, image) < 0)
		failed++;
	if (part_unregistered_source(&a, host, image) < 0)
		failed++;

out:
	free(image);
	free(host);
	if (gpu)
		opends_free(gpu);
	aisio_homi_teardown(&a);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
