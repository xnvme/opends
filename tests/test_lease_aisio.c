/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_lease_aisio.c - What the lease on a registered file holds.
 *
 * A read goes to the device blocks the file had when it was mapped. The
 * kernel knows more: the page cache, delayed allocation, the unwritten state
 * of an extent, and a writer about to free blocks. The driver holds a lease
 * on the registered descriptor, so a writer has to come past it, and the
 * file is mapped again, with a flush, before the next read.
 *
 *   second_descriptor  a file is registered once
 *   open_for_writing   a file a writer has open is refused, and taken later
 *   pagecache          a block overwritten through the page cache
 *   delalloc           a new file written through the page cache
 *   unwritten          an extent preallocated over blocks with old data
 *   writer_waits       a truncate during a read returns after the read
 *
 * pagecache, delalloc and unwritten pass when OpenDS returns what an
 * O_DIRECT read(2) does. A case whose state did not hold when the read ran
 * reports SKIP and does not count.
 *
 * Takes a directory on the mount to create its files in.
 */
#define _GNU_SOURCE

#include "opends.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <fcntl.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define BLK ((size_t)4096)
#define MIB ((size_t)1024 * 1024)
#define BIG_FILE (1024 * MIB)
#define BIG_TAIL (64 * MIB)
#define WRITER_AFTER_US 20000
#define READ_TIMEOUT_S 30.0
/* the test sees a read done a moment after the read gave its extents back */
#define WAIT_SLACK_S 0.010

enum outcome {
	PASS,
	FAIL,
	SKIPPED, /* the state did not hold when the read ran */
	BROKEN,  /* the case could not be set up */
};

struct lease_case {
	const char *name;
	enum outcome (*run)(const char *dir);
};

static void *gpu;       /* 1 MiB device buffer the reads land in */
static uint8_t *host;   /* 1 MiB aligned buffer for the kernel read */
static uint8_t *mirror; /* host copy of the device buffer */

static char case_file[4096]; /* the file of the running case */

static double
now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

static const char *
case_path(const char *dir, const char *name)
{
	snprintf(case_file, sizeof(case_file), "%s/lease_%s.bin", dir, name);
	return case_file;
}

/* Replace path with size bytes of byte, written O_DIRECT and synced. */
static int
write_direct(const char *path, size_t size, int byte)
{
	uint8_t *buf = aligned_alloc(BLK, MIB);
	ssize_t n;
	int err = 0;
	int fd;

	if (!buf) {
		return -1;
	}
	memset(buf, byte, MIB);
	unlink(path);
	fd = open(path, O_RDWR | O_CREAT | O_DIRECT, 0644);
	if (fd < 0) {
		free(buf);
		return -1;
	}
	for (size_t off = 0; off < size && !err; off += MIB) {
		n = pwrite(fd, buf, MIB, off);
		if (n != (ssize_t)MIB) {
			err = -1;
		}
	}
	if (!err) {
		err = fsync(fd);
	}
	close(fd);
	free(buf);
	return err;
}

/* The first extent of [off, off + len), or -1 when there is none. */
static int
first_extent(int fd, off_t off, size_t len, struct fiemap_extent *ext)
{
	union {
		struct fiemap fm;
		char room[sizeof(struct fiemap) + sizeof(struct fiemap_extent)];
	} q;
	int err;

	memset(&q, 0, sizeof(q));
	q.fm.fm_start = off;
	q.fm.fm_length = len;
	q.fm.fm_extent_count = 1;
	err = ioctl(fd, FS_IOC_FIEMAP, &q.fm);
	if (err < 0 || q.fm.fm_mapped_extents == 0) {
		return -1;
	}
	*ext = q.fm.fm_extents[0];
	return 0;
}

static ssize_t
kernel_read(const char *path, size_t len, off_t off)
{
	int fd = open(path, O_RDONLY | O_DIRECT);
	ssize_t n;

	if (fd < 0) {
		return -1;
	}
	n = pread(fd, host, len, off);
	close(fd);
	return n;
}

/* A read that gets no lease stays pending, and the driver cannot close under
 * it. Give up on the whole test then. */
static ssize_t
await_read(opends_async_future_t *future)
{
	double t0 = now_s();

	while (!__atomic_load_n(&future->done, __ATOMIC_ACQUIRE)) {
		if (now_s() - t0 > READ_TIMEOUT_S) {
			fprintf(stderr,
			        "FAIL: read still pending after %.0f s\n",
			        READ_TIMEOUT_S);
			unlink(case_file);
			_exit(1);
		}
		usleep(100);
	}
	return opends_async_await(future);
}

static ssize_t
aisio_read(opends_handle_t fh, void *buf, size_t len, off_t off)
{
	opends_async_future_t future;
	opends_error_t err;

	err = opends_async_read(fh, buf, len, off, 0, &future);
	if (err.err != OPENDS_SUCCESS) {
		return -(ssize_t)err.err;
	}
	return await_read(&future);
}

/* Open path read-only and register it. */
static opends_op_error_t
register_file(const char *path, int *fd, opends_handle_t *fh)
{
	opends_error_t err;

	*fh = NULL;
	*fd = open(path, O_RDONLY | O_DIRECT);
	if (*fd < 0) {
		return OPENDS_INVALID_VALUE;
	}
	err = opends_handle_register(fh, *fd);
	if (err.err != OPENDS_SUCCESS) {
		close(*fd);
		*fd = -1;
		*fh = NULL;
	}
	return err.err;
}

static void
deregister_file(int fd, opends_handle_t fh)
{
	opends_handle_deregister(fh);
	close(fd);
}

/* 1 when the device holds zeros under the file's first block, 0 when it
 * holds data, -1 when the device cannot be read. The stack-up step leaves
 * the name of the ublk device in /run/homi/ublk_dev. */
static int
device_block_zero(int fd)
{
	struct fiemap_extent ext;
	char dev[256] = {0};
	uint8_t *blk;
	int zero = 1;
	ssize_t n;
	FILE *f;
	int dfd;
	int err;

	err = first_extent(fd, 0, BLK, &ext);
	if (err < 0) {
		return -1;
	}
	f = fopen("/run/homi/ublk_dev", "r");
	if (!f) {
		return -1;
	}
	if (!fgets(dev, sizeof(dev), f)) {
		fclose(f);
		return -1;
	}
	fclose(f);
	dev[strcspn(dev, "\n")] = 0;

	blk = aligned_alloc(BLK, BLK);
	if (!blk) {
		return -1;
	}
	dfd = open(dev, O_RDONLY | O_DIRECT);
	if (dfd < 0) {
		free(blk);
		return -1;
	}
	n = pread(dfd, blk, BLK, ext.fe_physical);
	close(dfd);
	if (n != (ssize_t)BLK) {
		free(blk);
		return -1;
	}
	for (size_t i = 0; i < BLK; i++) {
		if (blk[i]) {
			zero = 0;
			break;
		}
	}
	free(blk);
	return zero;
}

/* Compare the OpenDS result in the device buffer with the kernel's in host. */
static enum outcome
compare(const char *what, ssize_t an, ssize_t kn, size_t len)
{
	size_t first = len;
	size_t diff = 0;

	if (an < 0) {
		fprintf(stderr,
		        "  %s: OpenDS read failed (%s), kernel read %zd\n",
		        what, opends_op_status_error((opends_op_error_t)-an),
		        kn);
		return FAIL;
	}
	cuda_buf_to_host(mirror, gpu, len);
	for (size_t i = 0; i < len; i++) {
		if (mirror[i] != host[i]) {
			if (first == len) {
				first = i;
			}
			diff++;
		}
	}
	fprintf(stderr, "  %s: OpenDS %zd bytes, kernel %zd bytes, %zu differ",
	        what, an, kn, diff);
	if (diff) {
		fprintf(stderr, ", first at %zu: OpenDS 0x%02x kernel 0x%02x",
		        first, mirror[first], host[first]);
	}
	fprintf(stderr, "\n");
	return an == kn && diff == 0 ? PASS : FAIL;
}

/* A second descriptor of a registered file is refused, and the first one
 * goes on reading. */
static enum outcome
case_second_descriptor(const char *dir)
{
	const char *path = case_path(dir, "second_descriptor");
	opends_handle_t fh, fh2;
	enum outcome out = BROKEN;
	opends_op_error_t st;
	int fd, fd2;
	ssize_t n;
	int err;

	err = write_direct(path, MIB, 0xa1);
	if (err < 0) {
		goto out;
	}
	st = register_file(path, &fd, &fh);
	if (st != OPENDS_SUCCESS) {
		goto out;
	}
	st = register_file(path, &fd2, &fh2);
	n = aisio_read(fh, gpu, BLK, 0);
	fprintf(stderr,
	        "  second registration: %s; read through the first: %zd\n",
	        opends_op_status_error(st), n);
	out = FAIL;
	if (st == OPENDS_HANDLE_ALREADY_REGISTERED && n == (ssize_t)BLK) {
		out = PASS;
	}
	if (fh2) {
		deregister_file(fd2, fh2);
	}
	deregister_file(fd, fh);
out:
	unlink(path);
	return out;
}

/* A file that a writer has open gets no lease, so it is not registered. It
 * is once the writer has closed it. */
static enum outcome
case_open_for_writing(const char *dir)
{
	const char *path = case_path(dir, "open_for_writing");
	opends_op_error_t held, closed;
	enum outcome out = BROKEN;
	opends_handle_t fh;
	ssize_t n = -1;
	int fd, wfd;
	int err;

	err = write_direct(path, MIB, 0xa1);
	if (err < 0) {
		goto out;
	}
	wfd = open(path, O_RDWR);
	if (wfd < 0) {
		goto out;
	}
	held = register_file(path, &fd, &fh);
	if (held == OPENDS_SUCCESS) {
		deregister_file(fd, fh);
	}
	close(wfd);
	closed = register_file(path, &fd, &fh);
	if (closed == OPENDS_SUCCESS) {
		n = aisio_read(fh, gpu, BLK, 0);
		deregister_file(fd, fh);
	}
	fprintf(stderr, "  with a writer: %s; after its close: %s, read %zd\n",
	        opends_op_status_error(held), opends_op_status_error(closed),
	        n);
	out = FAIL;
	if (held == OPENDS_FS_SETUP_ERROR && closed == OPENDS_SUCCESS &&
	    n == (ssize_t)BLK) {
		out = PASS;
	}
out:
	unlink(path);
	return out;
}

/* A block overwritten through the page cache and not written back. The
 * writer's open takes the lease away, and the read maps the file again. */
static enum outcome
case_pagecache(const char *dir)
{
	const char *path = case_path(dir, "pagecache");
	uint8_t *blk = aligned_alloc(BLK, BLK);
	enum outcome out = BROKEN;
	opends_op_error_t st;
	opends_handle_t fh;
	ssize_t an, kn, n;
	int fd, bfd;
	int err;

	if (!blk) {
		return BROKEN;
	}
	err = write_direct(path, MIB, 0xa1);
	if (err < 0) {
		goto out;
	}
	st = register_file(path, &fd, &fh);
	if (st != OPENDS_SUCCESS) {
		goto out;
	}
	an = aisio_read(fh, gpu, BLK, 0);
	if (an != (ssize_t)BLK) {
		fprintf(stderr, "  first read through OpenDS: %zd\n", an);
		goto out_dereg;
	}
	bfd = open(path, O_RDWR);
	if (bfd < 0) {
		goto out_dereg;
	}
	memset(blk, 0xb2, BLK);
	n = pwrite(bfd, blk, BLK, 0);
	close(bfd);
	if (n != (ssize_t)BLK) {
		goto out_dereg;
	}
	/* the kernel read writes the block back, so OpenDS reads first */
	an = aisio_read(fh, gpu, BLK, 0);
	kn = kernel_read(path, BLK, 0);
	out = compare("4 KiB after a buffered overwrite", an, kn, BLK);
out_dereg:
	deregister_file(fd, fh);
out:
	free(blk);
	unlink(path);
	return out;
}

/* A new file written through the page cache, with no blocks allocated when
 * it is registered. */
static enum outcome
case_delalloc(const char *dir)
{
	const char *path = case_path(dir, "delalloc");
	uint8_t *buf = aligned_alloc(BLK, 16 * BLK);
	enum outcome out = BROKEN;
	struct fiemap_extent ext;
	opends_op_error_t st;
	opends_handle_t fh;
	ssize_t an, kn, n;
	bool delalloc;
	int fd, bfd;
	int err;

	if (!buf) {
		return BROKEN;
	}
	unlink(path);
	bfd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (bfd < 0) {
		goto out;
	}
	memset(buf, 0xc3, 16 * BLK);
	n = pwrite(bfd, buf, 16 * BLK, 0);
	err = first_extent(bfd, 0, BLK, &ext);
	delalloc = err == 0 && (ext.fe_flags & FIEMAP_EXTENT_DELALLOC);
	close(bfd);
	if (n != (ssize_t)(16 * BLK)) {
		goto out;
	}
	st = register_file(path, &fd, &fh);
	if (st != OPENDS_SUCCESS) {
		goto out;
	}
	an = aisio_read(fh, gpu, BLK, 0);
	kn = kernel_read(path, BLK, 0);
	out = compare("4 KiB of a delayed-allocation file", an, kn, BLK);
	if (out == PASS && !delalloc) {
		fprintf(stderr, "  written back before the registration\n");
		out = SKIPPED;
	}
	deregister_file(fd, fh);
out:
	free(buf);
	unlink(path);
	return out;
}

/* An extent preallocated over the blocks of a deleted file: the kernel
 * reads zeros, the device still holds the deleted file's data. */
static enum outcome
case_unwritten(const char *dir)
{
	const char *path = case_path(dir, "unwritten");
	enum outcome out = BROKEN;
	struct fiemap_extent ext;
	opends_op_error_t st;
	opends_handle_t fh;
	ssize_t an, kn;
	char old[4096];
	int fd, wfd;
	int zero;
	int err;

	snprintf(old, sizeof(old), "%s/lease_unwritten_old.bin", dir);
	err = write_direct(old, MIB, 0xd4);
	unlink(old);
	if (err < 0) {
		return BROKEN;
	}
	sync();

	unlink(path);
	wfd = open(path, O_RDWR | O_CREAT | O_DIRECT, 0644);
	if (wfd < 0) {
		return BROKEN;
	}
	err = fallocate(wfd, 0, 0, MIB);
	if (!err) {
		err = fsync(wfd);
	}
	if (!err) {
		err = first_extent(wfd, 0, MIB, &ext);
	}
	close(wfd);
	if (err) {
		goto out;
	}
	if (!(ext.fe_flags & FIEMAP_EXTENT_UNWRITTEN)) {
		fprintf(stderr,
		        "  fallocate left no unwritten extent (flags 0x%x)\n",
		        ext.fe_flags);
		goto out;
	}
	st = register_file(path, &fd, &fh);
	if (st != OPENDS_SUCCESS) {
		goto out;
	}
	an = aisio_read(fh, gpu, MIB, 0);
	kn = kernel_read(path, MIB, 0);
	out = compare("1 MiB of an unwritten extent", an, kn, MIB);
	/* zeros from both say nothing when the device holds zeros there */
	zero = device_block_zero(fd);
	if (out == PASS && zero != 0) {
		fprintf(stderr, "  %s\n",
		        zero < 0 ? "device not readable, inconclusive"
		                 : "device holds zeros under the extent");
		out = SKIPPED;
	}
	deregister_file(fd, fh);
out:
	unlink(path);
	return out;
}

struct writer {
	const char *path;
	double t_start;
	double t_done;
	int err;
};

static void *
writer_thread(void *arg)
{
	struct writer *w = arg;

	usleep(WRITER_AFTER_US);
	w->t_start = now_s();
	w->err = truncate(w->path, (off_t)(BIG_FILE - BIG_TAIL));
	w->t_done = now_s();
	return NULL;
}

/* The tail of the file truncated away while one read of the whole file is
 * in flight. The truncate would free blocks the read has yet to reach, so
 * it has to wait for the read, and the read returns the whole file. */
static enum outcome
case_writer_waits(const char *dir)
{
	const char *path = case_path(dir, "writer_waits");
	struct writer wr = {.path = path};
	opends_async_future_t future;
	enum outcome out = BROKEN;
	double t0, t_read_done;
	opends_op_error_t st;
	uint8_t *tail = NULL;
	opends_handle_t fh;
	opends_error_t rerr;
	void *big = NULL;
	size_t data = 0;
	pthread_t thread;
	bool waited;
	ssize_t an;
	int err;
	int fd;

	err = write_direct(path, BIG_FILE, 0x58);
	if (err < 0) {
		goto out;
	}
	big = cuda_register_acquire(BIG_FILE);
	tail = aligned_alloc(BLK, BIG_TAIL);
	if (!big || !tail) {
		goto out;
	}
	st = register_file(path, &fd, &fh);
	if (st != OPENDS_SUCCESS) {
		goto out;
	}

	t0 = now_s();
	rerr = opends_async_read(fh, big, BIG_FILE, 0, 0, &future);
	if (rerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  async_read: %s\n",
		        opends_op_status_error(rerr.err));
		goto out_dereg;
	}
	err = pthread_create(&thread, NULL, writer_thread, &wr);
	if (err) {
		await_read(&future);
		goto out_dereg;
	}
	an = await_read(&future);
	t_read_done = now_s();
	pthread_join(thread, NULL);
	fprintf(stderr,
	        "  read %zd bytes, done at +%.3f s; truncate started at "
	        "+%.3f s, returned at +%.3f s\n",
	        an, t_read_done - t0, wr.t_start - t0, wr.t_done - t0);
	if (wr.err) {
		fprintf(stderr, "  truncate failed\n");
		goto out_dereg;
	}
	/* a read that maps the file after the truncate ends at the new size */
	if (t_read_done < wr.t_start || an == (ssize_t)(BIG_FILE - BIG_TAIL)) {
		fprintf(stderr,
		        "  the read was not in flight at the truncate\n");
		out = SKIPPED;
		goto out_dereg;
	}
	if (an == (ssize_t)BIG_FILE) {
		cuda_buf_to_host(tail, (uint8_t *)big + BIG_FILE - BIG_TAIL,
		                 BIG_TAIL);
		for (size_t i = 0; i < BIG_TAIL; i++) {
			data += tail[i] == 0x58;
		}
	}
	waited = wr.t_done > t_read_done - WAIT_SLACK_S;
	fprintf(stderr,
	        "  the truncate %s the read; %zu of %zu tail bytes are the "
	        "file's\n",
	        waited ? "waited for" : "returned before", data,
	        (size_t)BIG_TAIL);
	out = FAIL;
	if (waited && an == (ssize_t)BIG_FILE && data == BIG_TAIL) {
		out = PASS;
	}
out_dereg:
	deregister_file(fd, fh);
out:
	cuda_register_release(big);
	free(tail);
	unlink(path);
	return out;
}

int
main(int argc, char **argv)
{
	static const struct lease_case cases[] = {
	        {"second_descriptor", case_second_descriptor},
	        {"open_for_writing", case_open_for_writing},
	        {"pagecache", case_pagecache},
	        {"delalloc", case_delalloc},
	        {"unwritten", case_unwritten},
	        {"writer_waits", case_writer_waits},
	};
	size_t ncases = sizeof(cases) / sizeof(cases[0]);
	opends_error_t oerr;
	CUcontext cuctx;
	int failed = 0;
	int err;

	if (argc != 2) {
		fprintf(stderr, "usage: %s <dir-on-mount>\n", argv[0]);
		return 1;
	}
	err = cuda_ctx_create(&cuctx);
	if (err < 0) {
		fprintf(stderr, "cuCtxCreate failed\n");
		return 1;
	}
	oerr = opends_driver_open();
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr,
		        "driver_open: %s (is the homi stack running?)\n",
		        opends_op_status_error(oerr.err));
		cuCtxDestroy(cuctx);
		return 1;
	}
	gpu = cuda_alloc_acquire(MIB);
	host = aligned_alloc(BLK, MIB);
	mirror = malloc(MIB);
	if (!gpu || !host || !mirror) {
		fprintf(stderr, "buffer allocation failed\n");
		opends_driver_close();
		cuCtxDestroy(cuctx);
		return 1;
	}

	for (size_t i = 0; i < ncases; i++) {
		switch (cases[i].run(argv[1])) {
		case PASS: fprintf(stderr, "PASS: %s\n", cases[i].name); break;
		case SKIPPED:
			fprintf(stderr, "SKIP: %s\n", cases[i].name);
			break;
		case FAIL:
			fprintf(stderr, "FAIL: %s\n", cases[i].name);
			failed++;
			break;
		case BROKEN:
			fprintf(stderr, "FAIL: %s (setup)\n", cases[i].name);
			failed++;
			break;
		}
	}

	cuda_alloc_release(gpu);
	free(host);
	free(mirror);
	opends_driver_close();
	cuCtxDestroy(cuctx);

	if (failed) {
		fprintf(stderr, "%d test(s) FAILED\n", failed);
		return 1;
	}
	fprintf(stderr, "all tests passed\n");
	return 0;
}
