/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block_aisio - Raw block mode on the aisio backend, with homi
 * alone: no qublk, no file system, no xal-server.
 *
 * The harness (tasks/test.yaml) writes the sync-read pattern to a file while
 * the file system is mounted, with room behind it for a large write, records
 * the byte range of the file's first extent on the namespace and the size of
 * the namespace, and takes the stack down to homi. The file is removed only
 * after the test, so the file system still owns those blocks. The test
 * registers the character device homi serves and reads the range first: it must
 * hold the pattern the kernel wrote, which proves the offsets before anything
 * is written. Every write goes into that range, and the test ends with the
 * pattern back in its first pages.
 *
 * usage: test_raw_block_aisio <homi_chardev> <dev_off> <nbytes> <ns_nbytes>
 */
#define _GNU_SOURCE

#include "read_pattern.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define REGION_PAGES 8
#define REGION_BYTES ((size_t)PAGE * REGION_PAGES)
/* Behind the pattern file, and more than one NVMe command carries. */
#define LARGE_BYTES ((size_t)4 << 20)
#define REFUSED (-(ssize_t)OPENDS_INVALID_VALUE)
#define BAD_MODE (-(ssize_t)OPENDS_INVALID_FILE_OPEN_FLAG)

/* An index name with a malformed state region. Attaching to it fails at once;
 * a missing index would be waited for, for a minute. */
#define BAD_INDEX "/opends_raw_block_bad_index"
#define ENV_RAW_BLOCK "OPENDS_AISIO_RAW_BLOCK"

static int
cuda_ctx_create(CUcontext *ctx)
{
	CUdevice dev;
	cuInit(0);
	cuDeviceGet(&dev, 0);
	/* CUDA 13 maps cuCtxCreate to the 4-argument _v4 form. */
#if CUDA_VERSION >= 13000
	return cuCtxCreate(ctx, NULL, 0, dev) == CUDA_SUCCESS ? 0 : -1;
#else
	return cuCtxCreate(ctx, 0, dev) == CUDA_SUCCESS ? 0 : -1;
#endif
}

/* Compare n bytes of host memory with the pattern starting at offset off.
 * Returns the number of mismatching bytes. */
static size_t
check_pattern(const unsigned char *host, uint64_t off, size_t n)
{
	size_t bad = 0;
	for (size_t i = 0; i < n; i++)
		if (host[i] != pattern_byte((off_t)(off + i)))
			bad++;
	return bad;
}

/* Raw read of n bytes at device offset dev_off into gpu, copied to host.
 * Returns the backend's result (bytes or -error). */
static ssize_t
raw_read(opends_handle_t fh, void *gpu, unsigned char *host, size_t n,
         uint64_t dev_off)
{
	cuda_buf_zero(gpu, n);
	ssize_t r = opends_sync_read(fh, gpu, n, (off_t)dev_off, 0);
	if (r > 0)
		cuda_buf_to_host(host, gpu, (size_t)r);
	return r;
}

static ssize_t
raw_write(opends_handle_t fh, void *gpu, const unsigned char *host, size_t n,
          uint64_t dev_off)
{
	cuda_buf_from_host(gpu, host, n);
	return opends_sync_write(fh, gpu, n, (off_t)dev_off, 0);
}

/* Wait for a stream to drain, bounded by a wall-clock timeout so a stuck
 * backend fails the test instead of hanging the runner. Returns 0 when the
 * stream is idle, -1 after a message on timeout or driver error. */
static int
raw_stream_wait(CUstream stream, double timeout_s, const char *label)
{
	struct timespec t0, now;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (;;) {
		CUresult cr = cuStreamQuery(stream);
		if (cr == CUDA_SUCCESS)
			return 0;
		if (cr != CUDA_ERROR_NOT_READY) {
			fprintf(stderr, "  %s: cuStreamQuery: %d\n", label,
			        (int)cr);
			return -1;
		}
		clock_gettime(CLOCK_MONOTONIC, &now);
		double elapsed = (now.tv_sec - t0.tv_sec) +
		                 (now.tv_nsec - t0.tv_nsec) * 1e-9;
		if (elapsed >= timeout_s) {
			fprintf(stderr, "  %s: timed out after %.1fs\n", label,
			        elapsed);
			return -1;
		}
		struct timespec sl = {0, 1000000};
		nanosleep(&sl, NULL);
	}
}

/* Stream read of n bytes at device offset dev_off into gpu, waited for on
 * the stream, then copied to host. Returns the backend's result (bytes or
 * -error), or -OPENDS_INTERNAL_ERROR when the submit or the wait failed. */
static ssize_t
raw_stream_read(opends_handle_t fh, CUstream stream, void *gpu,
                unsigned char *host, size_t n, uint64_t dev_off)
{
	size_t size = n;
	off_t foff = (off_t)dev_off;
	off_t boff = 0;
	ssize_t bytes = -1;
	cuda_buf_zero(gpu, n);
	opends_error_t err = opends_stream_read(fh, gpu, &size, &foff, &boff,
	                                        &bytes, stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  stream_read submit: %s\n",
		        opends_op_status_error(err.err));
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	}
	if (raw_stream_wait(stream, 10.0, "stream_read") < 0)
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	if (bytes > 0)
		cuda_buf_to_host(host, gpu, (size_t)bytes);
	return bytes;
}

static ssize_t
raw_stream_write(opends_handle_t fh, CUstream stream, void *gpu,
                 const unsigned char *host, size_t n, uint64_t dev_off)
{
	size_t size = n;
	off_t foff = (off_t)dev_off;
	off_t boff = 0;
	ssize_t bytes = -1;
	cuda_buf_from_host(gpu, host, n);
	opends_error_t err = opends_stream_write(fh, gpu, &size, &foff, &boff,
	                                         &bytes, stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  stream_write submit: %s\n",
		        opends_op_status_error(err.err));
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	}
	if (raw_stream_wait(stream, 10.0, "stream_write") < 0)
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	return bytes;
}

static int failed;

static void
check(bool ok, const char *what)
{
	fprintf(stderr, "%s: %s\n", ok ? "ok" : "FAIL", what);
	failed += !ok;
}

/* The pattern of the page at offset off of the range, inverted. */
static void
inverse(unsigned char *dst, uint64_t off, size_t n)
{
	for (size_t i = 0; i < n; i++)
		dst[i] = (unsigned char)~pattern_byte((off_t)(off + i));
}

static bool
is_inverse(const unsigned char *host, uint64_t off, size_t n)
{
	for (size_t i = 0; i < n; i++)
		if (host[i] != (unsigned char)~pattern_byte((off_t)(off + i)))
			return false;
	return true;
}

/* Register path opened with flags; NULL when either step fails. */
static opends_handle_t
register_path(const char *path, int flags, int *fd, opends_op_error_t *err)
{
	opends_handle_t fh = NULL;

	*fd = open(path, flags);
	*err = *fd < 0 ? OPENDS_INVALID_VALUE
	               : opends_handle_register(&fh, *fd).err;
	return *err == OPENDS_SUCCESS ? fh : NULL;
}

static void
release(opends_handle_t fh, int fd)
{
	if (fh)
		opends_handle_deregister(fh);
	if (fd >= 0)
		close(fd);
}

/* One batch of two one-page ops on consecutive pages at dev_off, from or into
 * the first two pages of gpu. True when both completed in full. */
static bool
batch_two_pages(opends_handle_t fh, void *gpu, uint64_t dev_off,
                opends_opcode_t opcode)
{
	opends_batch_handle_t batch = NULL;
	opends_io_params_t io[2];
	opends_io_events_t ev[2];
	unsigned nr = 2;

	if (opends_batch_setup(&batch, 2).err != OPENDS_SUCCESS)
		return false;
	memset(io, 0, sizeof(io));
	for (int i = 0; i < 2; i++) {
		io[i].mode = OPENDS_BATCH;
		io[i].fh = fh;
		io[i].opcode = opcode;
		io[i].cookie = &io[i];
		io[i].u.batch.dev_ptr_base = gpu;
		io[i].u.batch.dev_ptr_offset = (off_t)i * PAGE;
		io[i].u.batch.file_offset = (off_t)(dev_off + i * PAGE);
		io[i].u.batch.size = PAGE;
	}
	bool ok = opends_batch_submit(batch, 2, io, 0).err == OPENDS_SUCCESS &&
	          opends_batch_get_status(batch, 2, &nr, ev, NULL).err ==
	                  OPENDS_SUCCESS &&
	          nr == 2 && ev[0].status == OPENDS_COMPLETE &&
	          ev[1].status == OPENDS_COMPLETE && ev[0].ret == PAGE &&
	          ev[1].ret == PAGE;
	opends_batch_destroy(batch);
	return ok;
}

/* Open the driver with no index published. Without OPENDS_AISIO_RAW_BLOCK
 * that fails, as it always has. With it the driver opens, and a regular file
 * (this binary) is refused for the lack of an index. That failure is final:
 * with the malformed index gone, a second registration that tried again would
 * wait a minute. Returns 0 with the driver open in raw block mode. */
static int
open_without_index(const char *file_path)
{
	opends_op_error_t err, err2;
	int fd, fd2;

	int sfd = shm_open(BAD_INDEX "_state", O_CREAT | O_RDWR, 0600);
	if (sfd < 0 || write(sfd, "x", 1) != 1) {
		fprintf(stderr, "shm_open(%s): %s\n", BAD_INDEX "_state",
		        strerror(errno));
		return -1;
	}
	close(sfd);
	setenv("OPENDS_XAL_SHM", BAD_INDEX, 1);

	unsetenv(ENV_RAW_BLOCK);
	err = opends_driver_open().err;
	check(err == OPENDS_FS_SETUP_ERROR,
	      "raw mode off: the driver needs an index to open");
	if (err == OPENDS_SUCCESS)
		opends_driver_close();

	setenv(ENV_RAW_BLOCK, "1", 1);
	if (opends_driver_open().err != OPENDS_SUCCESS) {
		fprintf(stderr, "driver_open without an index failed\n");
		shm_unlink(BAD_INDEX "_state");
		return -1;
	}
	opends_handle_t fh =
	        register_path(file_path, O_RDONLY | O_DIRECT, &fd, &err);
	release(fh, fd);
	shm_unlink(BAD_INDEX "_state");
	time_t t0 = time(NULL);
	fh = register_path(file_path, O_RDONLY | O_DIRECT, &fd2, &err2);
	release(fh, fd2);
	check(err == OPENDS_FS_SETUP_ERROR && err2 == OPENDS_FS_SETUP_ERROR &&
	              time(NULL) - t0 <= 5,
	      "no index: regular file refused, the second time at once");
	return 0;
}

/* A read may start and end off an LBA when its whole LBAs land on a 4-byte
 * boundary; one that puts them off it has no kernel path to take. */
static void
test_unaligned_reads(opends_handle_t fh, void *gpu, unsigned char *host,
                     uint64_t dev_off)
{
	size_t n = PAGE - 8;
	ssize_t r = raw_read(fh, gpu, host, n, dev_off + 4);

	check(r == (ssize_t)n && !check_pattern(host, 4, n),
	      "read with a bounced head and tail");
	/* long enough to hold a whole LBA of 4096 bytes too */
	check(raw_read(fh, gpu, host, 3 * PAGE - 2, dev_off + 1) == REFUSED,
	      "read with whole LBAs off a 4-byte boundary refused");
}

/* Page 0: a write is read back. A write that breaks a rule or the access
 * mode of its fd is refused and writes nothing: each carries the inverse, and
 * pages 0 and 1 hold the pattern afterwards. */
static void
test_sync_and_refusals(const char *raw_path, opends_handle_t fh, void *gpu,
                       unsigned char *host, uint64_t dev_off)
{
	opends_op_error_t err;
	int fd;

	inverse(host, 0, PAGE);
	ssize_t w = raw_write(fh, gpu, host, PAGE, dev_off);
	ssize_t r = raw_read(fh, gpu, host, PAGE, dev_off);
	check(w == PAGE && r == PAGE && is_inverse(host, 0, PAGE),
	      "sync write read back");

	for (size_t i = 0; i < PAGE; i++)
		host[i] = pattern_byte((off_t)i);
	check(raw_write(fh, gpu, host, PAGE, dev_off) == PAGE,
	      "pattern written back");
	inverse(host, 0, 2 * PAGE);
	cuda_buf_from_host(gpu, host, 2 * PAGE);
	check(opends_sync_write(fh, gpu, PAGE, (off_t)dev_off + 1, 0) ==
	              REFUSED,
	      "write starting off an LBA refused");
	check(opends_sync_write(fh, gpu, PAGE + 1, (off_t)dev_off, 0) ==
	              REFUSED,
	      "write ending off an LBA refused");
	check(opends_sync_write(fh, gpu, PAGE, (off_t)dev_off, 1) == REFUSED,
	      "write from a source off a 4-byte boundary refused");
	check(opends_sync_write(fh, gpu, 0, (off_t)dev_off + 1, 0) == 0,
	      "write of nothing returns 0 at any offset");

	opends_handle_t ro = register_path(raw_path, O_RDONLY, &fd, &err);
	check(ro && opends_sync_write(ro, gpu, PAGE, (off_t)dev_off, 0) ==
	                      BAD_MODE,
	      "write through a read-only fd refused");
	release(ro, fd);

	opends_handle_t wo = register_path(raw_path, O_WRONLY, &fd, &err);
	check(wo && opends_sync_read(wo, gpu, PAGE, (off_t)dev_off, 0) ==
	                      BAD_MODE,
	      "read through a write-only fd refused");
	release(wo, fd);

	r = raw_read(fh, gpu, host, 2 * PAGE, dev_off);
	check(r == 2 * PAGE && !check_pattern(host, 0, 2 * PAGE),
	      "refused writes wrote nothing");
}

/* Page 1 is read and page 2 written through the async calls, pages 3 and 4
 * through a batch. */
static void
test_async_and_batch(opends_handle_t fh, void *gpu, unsigned char *host,
                     uint64_t dev_off)
{
	opends_async_future_t fut;

	cuda_buf_zero(gpu, PAGE);
	opends_error_t err = opends_async_read(
	        fh, gpu, PAGE, (off_t)(dev_off + PAGE), 0, &fut);
	ssize_t r = err.err == OPENDS_SUCCESS ? opends_async_await(&fut) : -1;
	cuda_buf_to_host(host, gpu, PAGE);
	check(r == PAGE && !check_pattern(host, PAGE, PAGE), "async read");

	inverse(host, 2 * PAGE, PAGE);
	cuda_buf_from_host(gpu, host, PAGE);
	err = opends_async_write(fh, gpu, PAGE, (off_t)(dev_off + 2 * PAGE), 0,
	                         &fut);
	ssize_t w = err.err == OPENDS_SUCCESS ? opends_async_await(&fut) : -1;
	r = raw_read(fh, gpu, host, PAGE, dev_off + 2 * PAGE);
	check(w == PAGE && r == PAGE && is_inverse(host, 2 * PAGE, PAGE),
	      "async write read back");

	inverse(host, 3 * PAGE, 2 * PAGE);
	cuda_buf_from_host(gpu, host, 2 * PAGE);
	bool wok = batch_two_pages(fh, gpu, dev_off + 3 * PAGE, OPENDS_WRITE);
	cuda_buf_zero(gpu, 2 * PAGE);
	bool rok = batch_two_pages(fh, gpu, dev_off + 3 * PAGE, OPENDS_READ);
	cuda_buf_to_host(host, gpu, 2 * PAGE);
	check(wok && rok && is_inverse(host, 3 * PAGE, 2 * PAGE),
	      "batch write read back by a batch");
}

/* Pages 5 and 6 on a stream. A stream read may be partial only at its tail.
 * A stream write and a read of the page it writes are queued together and
 * waited for once: the read must see the write. */
static void
test_stream(opends_handle_t fh, void *gpu, unsigned char *host,
            uint64_t dev_off)
{
	CUstream stream;
	size_t n = PAGE - 8;

	if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
	    opends_stream_register(stream, 0).err != OPENDS_SUCCESS) {
		check(false, "stream setup");
		return;
	}

	ssize_t r =
	        raw_stream_read(fh, stream, gpu, host, n, dev_off + 5 * PAGE);
	check(r == (ssize_t)n && !check_pattern(host, 5 * PAGE, n),
	      "stream read with a partial tail");
	check(raw_stream_read(fh, stream, gpu, host, n,
	                      dev_off + 5 * PAGE + 4) == REFUSED,
	      "stream read starting off an LBA refused");

	inverse(host, 6 * PAGE, PAGE);
	cuda_buf_from_host(gpu, host, PAGE);
	cuda_buf_zero((uint8_t *)gpu + PAGE, PAGE);
	size_t wsz = PAGE, rsz = PAGE;
	off_t wfo = (off_t)(dev_off + 6 * PAGE), rfo = wfo;
	off_t wbo = 0, rbo = PAGE;
	ssize_t wb = -1, rb = -1;
	opends_error_t werr =
	        opends_stream_write(fh, gpu, &wsz, &wfo, &wbo, &wb, stream);
	opends_error_t rerr =
	        opends_stream_read(fh, gpu, &rsz, &rfo, &rbo, &rb, stream);
	int waited = raw_stream_wait(stream, 10.0, "stream write + read");
	cuda_buf_to_host(host, (uint8_t *)gpu + PAGE, PAGE);
	check(werr.err == OPENDS_SUCCESS && rerr.err == OPENDS_SUCCESS &&
	              waited == 0 && wb == PAGE && rb == PAGE &&
	              is_inverse(host, 6 * PAGE, PAGE),
	      "stream write seen by the read queued behind it");

	for (size_t i = 0; i < PAGE + 1; i++)
		host[i] = pattern_byte((off_t)(6 * PAGE + i));
	ssize_t w = raw_stream_write(fh, stream, gpu, host, PAGE + 1,
	                             dev_off + 6 * PAGE);
	r = raw_read(fh, gpu, host, PAGE, dev_off + 6 * PAGE);
	check(w == REFUSED && r == PAGE && is_inverse(host, 6 * PAGE, PAGE),
	      "stream write ending off an LBA refused, nothing written");

	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
}

/* One write and one read of LARGE_BYTES, which the driver splits into
 * commands of the device's transfer limit. */
static void
test_large(opends_handle_t fh, uint64_t dev_off)
{
	void *gpu = cuda_alloc_acquire(LARGE_BYTES);
	unsigned char *out = malloc(LARGE_BYTES);
	unsigned char *in = malloc(LARGE_BYTES);

	if (!gpu || !out || !in) {
		check(false, "large I/O: allocation");
		return;
	}
	/* The harness left zeros there: anything else is not our range. */
	memset(out, 0, LARGE_BYTES);
	ssize_t r = raw_read(fh, gpu, in, LARGE_BYTES, dev_off);
	bool ours = r == (ssize_t)LARGE_BYTES && !memcmp(in, out, LARGE_BYTES);
	check(ours, "range of the large write holds the harness' zeros");
	if (ours) {
		for (size_t i = 0; i < LARGE_BYTES; i++)
			out[i] = (unsigned char)((i * 2654435761u) >> 24);
		ssize_t w = raw_write(fh, gpu, out, LARGE_BYTES, dev_off);
		r = raw_read(fh, gpu, in, LARGE_BYTES, dev_off);
		check(w == (ssize_t)LARGE_BYTES && r == (ssize_t)LARGE_BYTES &&
		              !memcmp(in, out, LARGE_BYTES),
		      "write and read of several commands");
		memset(out, 0, LARGE_BYTES);
		check(raw_write(fh, gpu, out, LARGE_BYTES, dev_off) ==
		              (ssize_t)LARGE_BYTES,
		      "zeros written back");
	}
	cuda_alloc_release(gpu);
	free(out);
	free(in);
}

/* A namespace has no end of file to grow: a read across the end is short,
 * one past it returns 0 and a write past it is refused. Nothing is written
 * near the end: those blocks are not the test's. */
static void
test_namespace_end(opends_handle_t fh, void *gpu, unsigned char *host,
                   uint64_t ns_nbytes)
{
	off_t last = (off_t)(ns_nbytes - PAGE);

	check(raw_read(fh, gpu, host, 2 * PAGE, (uint64_t)last) == PAGE,
	      "read across the end is short");
	check(opends_sync_read(fh, gpu, PAGE, (off_t)ns_nbytes, 0) == 0,
	      "read past the end returns 0");
	check(opends_sync_write(fh, gpu, PAGE, (off_t)ns_nbytes, 0) == REFUSED,
	      "write past the end refused");
}

int
main(int argc, char **argv)
{
	if (argc != 5) {
		fprintf(stderr,
		        "usage: %s <homi_chardev> <dev_off> <nbytes> "
		        "<ns_nbytes>\n",
		        argv[0]);
		return 1;
	}
	const char *raw_path = argv[1];
	uint64_t dev_off = strtoull(argv[2], NULL, 0);
	uint64_t nbytes = strtoull(argv[3], NULL, 0);
	uint64_t ns_nbytes = strtoull(argv[4], NULL, 0);
	opends_op_error_t err;
	int raw_fd, fd;

	fprintf(stderr, "raw block mode (aisio backend, %s)\n", raw_path);
	/* Offset 0 is no file's extent (the superblock is there), and it is
	 * what a bad argument parses to. */
	if (!dev_off || (dev_off & (PAGE - 1)) ||
	    nbytes < FILE_SIZE + LARGE_BYTES || ns_nbytes < dev_off + nbytes) {
		fprintf(stderr,
		        "need a page-aligned range of %zu bytes past offset 0 "
		        "of the namespace\n",
		        FILE_SIZE + LARGE_BYTES);
		return 1;
	}

	CUcontext cuctx;
	if (cuda_ctx_create(&cuctx) < 0) {
		fprintf(stderr, "cuCtxCreate failed\n");
		return 1;
	}
	if (open_without_index(argv[0]) < 0)
		return 1;
	opends_handle_t fh = register_path("/dev/null", O_RDWR, &fd, &err);
	check(fd >= 0 && err == OPENDS_INVALID_VALUE,
	      "another character device refused");
	release(fh, fd);

	fh = register_path(raw_path, O_RDWR, &raw_fd, &err);
	void *gpu = cuda_alloc_acquire(REGION_BYTES);
	unsigned char *host = malloc(REGION_BYTES);
	if (!fh || !gpu || !host) {
		fprintf(stderr, "register %s: %s\n", raw_path,
		        opends_op_status_error(err));
		return 1;
	}

	ssize_t r = raw_read(fh, gpu, host, REGION_BYTES, dev_off);
	if (r != (ssize_t)REGION_BYTES ||
	    check_pattern(host, 0, REGION_BYTES)) {
		fprintf(stderr,
		        "FAIL: the range does not hold the pattern (read "
		        "%zd); nothing is written\n",
		        r);
		return 1;
	}
	fprintf(stderr, "ok: the range holds the pattern the kernel wrote\n");

	test_unaligned_reads(fh, gpu, host, dev_off);
	test_sync_and_refusals(raw_path, fh, gpu, host, dev_off);
	test_async_and_batch(fh, gpu, host, dev_off);
	test_stream(fh, gpu, host, dev_off);
	test_large(fh, dev_off + FILE_SIZE);
	test_namespace_end(fh, gpu, host, ns_nbytes);

	for (size_t i = 0; i < REGION_BYTES; i++)
		host[i] = pattern_byte((off_t)i);
	ssize_t w = raw_write(fh, gpu, host, REGION_BYTES, dev_off);
	r = raw_read(fh, gpu, host, REGION_BYTES, dev_off);
	check(w == (ssize_t)REGION_BYTES && r == (ssize_t)REGION_BYTES &&
	              !check_pattern(host, 0, REGION_BYTES),
	      "pattern back in the range");

	cuda_alloc_release(gpu);
	free(host);
	release(fh, raw_fd);
	opends_driver_close();
	cuCtxDestroy(cuctx);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
