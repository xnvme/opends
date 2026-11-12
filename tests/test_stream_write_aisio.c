/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_stream_read.h" /* stream_test_stream_sync_timeout, the plug */
#include "test_cuda_common.h"
#include "test_write_homi.h"

#include <cuda.h>

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* At least two stream writes and one probe for each I/O thread. */
#define PLUG_STREAM_WRITES 32
#define PLUG_PROBES 16

/* The write buffers the binary runs with: a few MiB make rounds, and the
 * plug case holds more writes than the default pool before it lets the
 * stream go. */
#define WRITE_BUF_MB "1"
#define WRITE_BUFS "64"
#define ROUNDS_WRITE_SIZE (2 * 1024 * 1024 + 2 * PAGE + 300)
#define ROUNDS_WRITE_OFF 100
/* More rounds than a stream holds queued commands for. */
#define MANY_ROUNDS_SIZE ((size_t)2 * 1024 * 1024 * 1024)

#define STREAM_FIXED                                                           \
	(OPENDS_STREAM_FIXED_BUF_OFFSET | OPENDS_STREAM_FIXED_FILE_OFFSET |    \
	 OPENDS_STREAM_FIXED_FILE_SIZE)

static ssize_t
stream_submit_write(struct write_homi_env *e, void *gpu, size_t size,
                    off_t foff)
{
	size_t sz = size;
	off_t fo = foff;
	off_t bo = 0;
	ssize_t bytes = -1;
	opends_error_t err = opends_stream_write(e->fh, gpu, &sz, &fo, &bo,
	                                         &bytes, e->stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  stream_write submit: %s\n",
		        opends_op_status_error(err.err));
		return -1;
	}
	if (stream_test_stream_sync_timeout(e->stream, 10.0, "stream_write") !=
	    0)
		return -1;
	return bytes;
}

static int
stream_create(CUstream *stream, unsigned cu_flags, unsigned flags,
              const char *label)
{
	if (cuStreamCreate(stream, cu_flags) != CUDA_SUCCESS) {
		fprintf(stderr, "  %s: cuStreamCreate\n", label);
		return -1;
	}
	if (opends_stream_register(*stream, flags).err != OPENDS_SUCCESS) {
		fprintf(stderr, "  %s: stream register\n", label);
		cuStreamDestroy(*stream);
		return -1;
	}
	return 0;
}

static int
blocking_stream_create(CUstream *stream, const char *label)
{
	return stream_create(stream, CU_STREAM_DEFAULT, 0, label);
}

/*
 * The write cases on one stream. A stream made with default flags and the
 * legacy default stream wait for each other, so a copy that waits on an
 * I/O thread waits for the parked stream, and the gate never opens. The
 * FIXED flags pick the copy engine with the values of the call; without
 * them the resident kernel reads the values late.
 */
static int
run_pass(struct aisio_homi *a, unsigned cu_flags, unsigned flags,
         const char *label)
{
	struct write_homi_env env = {
	        .fh = a->fh,
	        .fd = a->fd,
	        .submit_write = stream_submit_write,
	        .mode_label = label,
	};
	CUstream stream;
	int failed;
	int err;

	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("  ftruncate");
		return 1;
	}
	err = stream_create(&stream, cu_flags, flags, label);
	if (err < 0) {
		return 1;
	}
	env.stream = stream;
	failed = run_write_homi_tests(&env);
	if (failed) {
		/* the stream may be stuck at a gate: leave it */
		return failed;
	}
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
	return 0;
}

/* A write the file system refuses: the handle is on a read-only descriptor,
 * so the pwrite fails. The stream has to come free, and the byte count has
 * to carry the error. */
static int
stream_write_refused(const char *path, unsigned cu_flags, unsigned flags,
                     const char *label)
{
	const ssize_t want = -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR;
	opends_handle_t rfh = NULL;
	void *gpu = opends_alloc(PAGE);
	size_t sz = PAGE;
	off_t fo = 0;
	off_t bo = 0;
	ssize_t bytes = 0;
	CUstream stream;
	opends_error_t oerr;
	int failed = 1;
	int rfd;
	int err;

	rfd = open(path, O_RDONLY | O_DIRECT);
	if (rfd < 0 || !gpu) {
		fprintf(stderr, "[%s] refused: setup failed\n", label);
		goto out;
	}
	oerr = opends_handle_register(&rfh, rfd);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[%s] refused: register: %s\n", label,
		        opends_op_status_error(oerr.err));
		rfh = NULL;
		goto out;
	}
	err = stream_create(&stream, cu_flags, flags, label);
	if (err < 0) {
		goto out;
	}
	oerr = opends_stream_write(rfh, gpu, &sz, &fo, &bo, &bytes, stream);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[%s] refused: submit: %s\n", label,
		        opends_op_status_error(oerr.err));
		goto out_stream;
	}
	err = stream_test_stream_sync_timeout(stream, 15.0, label);
	if (err != 0) {
		/* the stream is stuck at its gate: leave it and the handle */
		return 1;
	}
	if (bytes != want) {
		fprintf(stderr, "[%s] refused: bytes %zd, want %zd\n", label,
		        bytes, want);
		goto out_stream;
	}
	fprintf(stderr, "[%s] refused write fails and frees the stream ok\n",
	        label);
	failed = 0;
out_stream:
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
out:
	if (rfh) {
		opends_handle_deregister(rfh);
	}
	if (rfd >= 0) {
		close(rfd);
	}
	if (gpu) {
		opends_free(gpu);
	}
	return failed;
}

/* A NULL stream is not a stream of this backend. */
static int
stream_write_null_refused(struct aisio_homi *a)
{
	void *gpu = opends_alloc(PAGE);
	size_t sz = PAGE;
	off_t fo = 0;
	off_t bo = 0;
	ssize_t bytes = 0;
	opends_error_t oerr;

	if (!gpu) {
		fprintf(stderr, "[null] allocation failed\n");
		return 1;
	}
	oerr = opends_stream_write(a->fh, gpu, &sz, &fo, &bo, &bytes, NULL);
	opends_free(gpu);
	if (oerr.err != OPENDS_INVALID_VALUE) {
		fprintf(stderr, "[null] a NULL stream was not refused: %s\n",
		        opends_op_status_error(oerr.err));
		return 1;
	}
	fprintf(stderr, "[null] NULL stream refused ok\n");
	return 0;
}

struct plain_thread_call {
	opends_handle_t fh;
	void *gpu;
	CUstream stream;
	CUcontext ctx; /* made current for the call, or NULL for none */
	size_t size;
	off_t foff;
	off_t boff;
	ssize_t bytes;
	opends_error_t oerr;
	int done;
};

static void *
plain_thread_stream_write(void *arg)
{
	struct plain_thread_call *c = arg;

	if (c->ctx) {
		cuCtxSetCurrent(c->ctx);
	}
	c->oerr = opends_stream_write(c->fh, c->gpu, &c->size, &c->foff,
	                              &c->boff, &c->bytes, c->stream);
	__atomic_store_n(&c->done, 1, __ATOMIC_RELEASE);
	return NULL;
}

/* Returns false when the call has not returned within the timeout. */
static bool
plain_thread_call_wait(pthread_t thread, struct plain_thread_call *c,
                       int timeout_ms)
{
	struct timespec nap = {0, 1000000};

	for (int ms = 0; ms < timeout_ms; ms++) {
		if (__atomic_load_n(&c->done, __ATOMIC_ACQUIRE)) {
			pthread_join(thread, NULL);
			return true;
		}
		nanosleep(&nap, NULL);
	}
	return false;
}

/* A write of nothing writes nothing, whatever its offset. */
static int
stream_write_empty(struct aisio_homi *a, unsigned flags, const char *label)
{
	size_t sz = 0;
	off_t fo = 100;
	off_t bo = 0;
	ssize_t bytes = -77;
	void *gpu = opends_alloc(PAGE);
	opends_error_t oerr;
	CUstream stream;
	struct stat st;
	int rc = 1;
	int err;

	if (!gpu) {
		fprintf(stderr, "[%s] empty: allocation failed\n", label);
		return 1;
	}
	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("  empty: ftruncate");
		goto out;
	}
	err = stream_create(&stream, CU_STREAM_NON_BLOCKING, flags, label);
	if (err < 0) {
		goto out;
	}
	oerr = opends_stream_write(a->fh, gpu, &sz, &fo, &bo, &bytes, stream);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[%s] empty: submit: %s\n", label,
		        opends_op_status_error(oerr.err));
		goto out_stream;
	}
	err = stream_test_stream_sync_timeout(stream, 15.0, label);
	if (err != 0) {
		return 1;
	}
	err = fstat(a->fd, &st);
	if (bytes != 0 || err != 0 || st.st_size != 0) {
		fprintf(stderr, "[%s] empty: bytes %zd, file size %lld\n",
		        label, bytes, err != 0 ? -1LL : (long long)st.st_size);
		goto out_stream;
	}
	fprintf(stderr, "[%s] a write of nothing leaves the file ok\n", label);
	rc = 0;
out_stream:
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
out:
	opends_free(gpu);
	return rc;
}

/* One write of thousands of rounds: its commands must not fill the stream
 * before the I/O thread can take the first round. */
static int
stream_write_many_rounds(struct aisio_homi *a, unsigned flags,
                         const char *label)
{
	static struct plain_thread_call call;
	const size_t size = MANY_ROUNDS_SIZE;
	const off_t probes[3] = {0, (off_t)(size / 2), (off_t)(size - PAGE)};
	unsigned char *host = NULL;
	unsigned char want[PAGE];
	CUdeviceptr dptr = 0;
	CUstream stream;
	pthread_t thread;
	opends_error_t oerr;
	struct stat st;
	int rc = 1;
	ssize_t r;
	int err;

	memset(want, 0x5a, sizeof(want));
	err = posix_memalign((void **)&host, PAGE, PAGE);
	if (err != 0) {
		return 1;
	}
	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("  many rounds: ftruncate");
		goto out;
	}
	if (cuMemAlloc(&dptr, size) != CUDA_SUCCESS ||
	    cuMemsetD8(dptr, 0x5a, size) != CUDA_SUCCESS) {
		fprintf(stderr, "[%s] many rounds: device memory\n", label);
		goto out;
	}
	oerr = opends_buf_register((void *)(uintptr_t)dptr, size, 0);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[%s] many rounds: register: %s\n", label,
		        opends_op_status_error(oerr.err));
		goto out;
	}
	err = stream_create(&stream, CU_STREAM_NON_BLOCKING, flags, label);
	if (err < 0) {
		goto out_buf;
	}

	memset(&call, 0, sizeof(call));
	call.fh = a->fh;
	call.gpu = (void *)(uintptr_t)dptr;
	call.stream = stream;
	call.ctx = a->cuctx;
	call.size = size;
	call.bytes = -77;
	err = pthread_create(&thread, NULL, plain_thread_stream_write, &call);
	if (err != 0) {
		fprintf(stderr, "[%s] many rounds: pthread_create\n", label);
		goto out_stream;
	}
	if (!plain_thread_call_wait(thread, &call, 30000)) {
		fprintf(stderr, "[%s] many rounds: the call did not return\n",
		        label);
		return 1;
	}
	if (call.oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[%s] many rounds: submit: %s\n", label,
		        opends_op_status_error(call.oerr.err));
		goto out_stream;
	}
	err = stream_test_stream_sync_timeout(stream, 120.0, label);
	if (err != 0) {
		return 1;
	}
	err = fstat(a->fd, &st);
	if (call.bytes != (ssize_t)size || err != 0 ||
	    (size_t)st.st_size != size) {
		fprintf(stderr, "[%s] many rounds: bytes %zd, file size %lld\n",
		        label, call.bytes,
		        err != 0 ? -1LL : (long long)st.st_size);
		goto out_stream;
	}
	for (int i = 0; i < 3; i++) {
		r = pread(a->fd, host, PAGE, probes[i]);
		if (r != PAGE || memcmp(host, want, PAGE)) {
			fprintf(stderr,
			        "[%s] many rounds: read-back at %lld (r=%zd)\n",
			        label, (long long)probes[i], r);
			goto out_stream;
		}
	}
	fprintf(stderr, "[%s] write of %zu rounds + read-back ok\n", label,
	        size / (1024 * 1024));
	rc = 0;
out_stream:
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
out_buf:
	opends_buf_deregister((void *)(uintptr_t)dptr);
out:
	if (dptr) {
		cuMemFree(dptr);
	}
	free(host);
	return rc;
}

/* A write of several rounds through the buffer, with a head and a tail off a
 * block, into an empty file: the head reads back as zeros, the file ends
 * with the data. */
static int
stream_write_rounds(struct aisio_homi *a, unsigned flags, const char *label)
{
	const size_t size = ROUNDS_WRITE_SIZE;
	const off_t foff = ROUNDS_WRITE_OFF;
	const size_t total = (size_t)foff + size;
	const size_t span = (total + PAGE - 1) & ~(size_t)(PAGE - 1);
	CUstream stream;
	void *gpu = NULL;
	unsigned char *image = NULL;
	unsigned char *host = NULL;
	struct stat st;
	ssize_t w;
	ssize_t r;
	int rc = 1;
	int err;

	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("  rounds: ftruncate");
		return 1;
	}
	err = stream_create(&stream, CU_STREAM_NON_BLOCKING, flags, label);
	if (err < 0) {
		return 1;
	}
	gpu = opends_alloc(size);
	image = calloc(1, span);
	err = posix_memalign((void **)&host, PAGE, span);
	if (err != 0) {
		host = NULL;
	}
	if (!gpu || !image || !host) {
		fprintf(stderr, "  %s: allocation failed\n", label);
		goto out;
	}
	for (size_t i = 0; i < size; i++) {
		image[(size_t)foff + i] = write_pattern_byte(i);
	}
	cuda_buf_from_host(gpu, image + foff, size);

	{
		struct write_homi_env env = {
		        .fh = a->fh,
		        .fd = a->fd,
		        .stream = stream,
		        .mode_label = label,
		};

		w = stream_submit_write(&env, gpu, size, foff);
	}
	if (w != (ssize_t)size) {
		fprintf(stderr, "  %s: write %zu@%lld = %zd\n", label, size,
		        (long long)foff, w);
		goto out;
	}
	err = fstat(a->fd, &st);
	if (err < 0 || (size_t)st.st_size != total) {
		fprintf(stderr, "  %s: file size %lld, want %zu\n", label,
		        err < 0 ? -1LL : (long long)st.st_size, total);
		goto out;
	}
	r = pread(a->fd, host, span, 0);
	if (r != (ssize_t)total || memcmp(host, image, total) != 0) {
		fprintf(stderr, "  %s: read-back mismatch (r=%zd)\n", label, r);
		goto out;
	}
	fprintf(stderr, "[%s] write %zu@%lld in rounds + read-back ok\n", label,
	        size, (long long)foff);
	rc = 0;
out:
	free(image);
	free(host);
	if (gpu) {
		opends_free(gpu);
	}
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
	return rc;
}

/* Values set by earlier work in the stream. A stream without the FIXED
 * flags writes with them; one with the flags refuses the write. */
static int
stream_write_late_values(struct aisio_homi *a, unsigned flags,
                         const char *label)
{
	const size_t alloc = 2 * PAGE;
	CUstream stream;
	void *gpu = NULL;
	unsigned char *image = NULL;
	unsigned char *host = NULL;
	size_t sz = 0;
	off_t foff = 0;
	off_t boff = 0;
	ssize_t bytes = 0;
	struct deferred_mutate_args mut = {
	        .size_p = &sz,
	        .file_offset_p = &foff,
	        .buf_offset_p = &boff,
	        .target_size = PAGE,
	        .target_file_offset = 3 * PAGE,
	        .target_buf_offset = PAGE,
	};
	bool fixed = (flags & STREAM_FIXED) == STREAM_FIXED;
	opends_error_t oerr;
	struct stat st;
	ssize_t r;
	int rc = 1;
	int err;

	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("  late: ftruncate");
		return 1;
	}
	err = stream_create(&stream, CU_STREAM_NON_BLOCKING, flags, label);
	if (err < 0) {
		return 1;
	}
	gpu = opends_alloc(alloc);
	image = malloc(alloc);
	err = posix_memalign((void **)&host, PAGE, 4 * PAGE);
	if (err != 0) {
		host = NULL;
	}
	if (!gpu || !image || !host) {
		fprintf(stderr, "  %s: allocation failed\n", label);
		goto out;
	}
	for (size_t i = 0; i < alloc; i++) {
		image[i] = write_pattern_byte(i);
	}
	cuda_buf_from_host(gpu, image, alloc);

	if (cuLaunchHostFunc(stream, stream_test_deferred_mutate, &mut) !=
	    CUDA_SUCCESS) {
		fprintf(stderr, "  %s: cuLaunchHostFunc\n", label);
		goto out;
	}
	oerr = opends_stream_write(a->fh, gpu, &sz, &foff, &boff, &bytes,
	                           stream);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  %s: submit: %s\n", label,
		        opends_op_status_error(oerr.err));
		goto out;
	}
	if (stream_test_stream_sync_timeout(stream, 10.0, label) != 0) {
		free(image);
		free(host);
		return 1;
	}
	if (fixed) {
		if (bytes != -(ssize_t)OPENDS_INVALID_VALUE) {
			fprintf(stderr, "  %s: bytes %zd, want refused\n",
			        label, bytes);
			goto out;
		}
		err = fstat(a->fd, &st);
		if (err < 0 || st.st_size != 0) {
			fprintf(stderr, "  %s: the file changed\n", label);
			goto out;
		}
		fprintf(stderr, "[%s] late values refused ok\n", label);
		rc = 0;
		goto out;
	}
	if (bytes != (ssize_t)PAGE) {
		fprintf(stderr, "  %s: bytes %zd (call-time values used)\n",
		        label, bytes);
		goto out;
	}
	err = fstat(a->fd, &st);
	if (err < 0 || st.st_size != 4 * PAGE) {
		fprintf(stderr, "  %s: file size %lld, want %d\n", label,
		        err < 0 ? -1LL : (long long)st.st_size, 4 * PAGE);
		goto out;
	}
	r = pread(a->fd, host, PAGE, 3 * PAGE);
	if (r != (ssize_t)PAGE || memcmp(host, image + PAGE, PAGE) != 0) {
		fprintf(stderr, "  %s: read-back mismatch (r=%zd)\n", label, r);
		goto out;
	}
	fprintf(stderr, "[%s] late values written ok\n", label);
	rc = 0;
out:
	free(image);
	free(host);
	if (gpu) {
		opends_free(gpu);
	}
	opends_stream_deregister(stream);
	cuStreamDestroy(stream);
	return rc;
}

/*
 * An I/O thread must not wait for a stream. A plug holds a stream made with
 * default flags, with stream writes behind it, so every I/O thread owes a
 * gate once the plug is out. Each write holds a write buffer from its call,
 * so the pool must have room for all of them: a call that finds none waits
 * for one, and the plug is pulled by this thread. An async write stages on an
 * I/O thread while the plug is in. Probes (writes of zero bytes, which stage
 * nothing) must finish on every thread while the plug is in, and everything
 * once it is out.
 */
static int
stream_write_plugged(struct aisio_homi *a)
{
	/* static: a stuck run returns while the I/O threads still use these */
	static struct stream_test_plug plug;
	static size_t sizes[PLUG_STREAM_WRITES];
	static off_t foffs[PLUG_STREAM_WRITES];
	static off_t boffs[PLUG_STREAM_WRITES];
	static ssize_t bws[PLUG_STREAM_WRITES];
	static opends_async_future_t futs[1 + PLUG_PROBES];
	const size_t total = (PLUG_STREAM_WRITES + 1) * PAGE;
	const off_t async_off = (off_t)(PLUG_STREAM_WRITES * PAGE);
	CUstream blocking;
	void *gpu = NULL;
	unsigned char *image = NULL;
	unsigned char *host = NULL;
	opends_error_t oerr;
	struct stat st;
	ssize_t r;
	bool ok = false;
	int nfuts = 0;
	int rc = 1;
	int err;

	memset(futs, 0, sizeof(futs));
	plug.pulled = 0;

	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("  plugged: ftruncate");
		return 1;
	}
	err = blocking_stream_create(&blocking, "plugged");
	if (err < 0) {
		return 1;
	}

	gpu = opends_alloc(total);
	image = malloc(total);
	err = posix_memalign((void **)&host, PAGE, total);
	if (err != 0) {
		host = NULL;
	}
	if (!gpu || !image || !host) {
		fprintf(stderr, "  plugged: allocation failed\n");
		goto out;
	}
	for (size_t i = 0; i < total; i++) {
		image[i] = write_pattern_byte(i);
	}
	cuda_buf_from_host(gpu, image, total);

	/* From here to the drain, a CUDA call that waits would wait here. */
	if (cuLaunchHostFunc(blocking, stream_test_plug_cb, &plug) !=
	    CUDA_SUCCESS) {
		fprintf(stderr, "  plugged: cuLaunchHostFunc\n");
		goto out;
	}

	for (int i = 0; i < PLUG_STREAM_WRITES; i++) {
		sizes[i] = PAGE;
		foffs[i] = (off_t)((size_t)i * PAGE);
		boffs[i] = foffs[i];
		bws[i] = 0;
		oerr = opends_stream_write(a->fh, gpu, &sizes[i], &foffs[i],
		                           &boffs[i], &bws[i], blocking);
		if (oerr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  plugged: submit %d: %s\n", i,
			        opends_op_status_error(oerr.err));
			goto drain;
		}
	}

	oerr = opends_async_write(a->fh, gpu, PAGE, async_off, async_off,
	                          &futs[0]);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  plugged: async write: %s\n",
		        opends_op_status_error(oerr.err));
		goto drain;
	}
	nfuts = 1;
	for (int i = 0; i < PLUG_PROBES; i++) {
		oerr = opends_async_write(a->fh, gpu, 0, 0, 0, &futs[1 + i]);
		if (oerr.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  plugged: probe %d: %s\n", i,
			        opends_op_status_error(oerr.err));
			goto drain;
		}
		nfuts++;
	}

	ok = stream_test_futures_done(futs + 1, PLUG_PROBES, 5.0);
	if (!ok) {
		fprintf(stderr,
		        "  plugged: an I/O thread waits for the stream\n");
	}
drain:
	__atomic_store_n(&plug.pulled, 1, __ATOMIC_RELEASE);
	if (!stream_test_futures_done(futs, nfuts, 20.0) ||
	    stream_test_stream_sync_timeout(blocking, 20.0, "plugged") != 0) {
		fprintf(stderr, "  plugged: the writes and the stream wait "
		                "for each other\n");
		/* The runner exits on a failure; the buffers are in use. */
		free(image);
		free(host);
		return 1;
	}
	if (!ok) {
		goto out;
	}

	if (opends_async_await(&futs[0]) != (ssize_t)PAGE) {
		fprintf(stderr, "  plugged: async write = %zd\n",
		        futs[0].result);
		goto out;
	}
	for (int i = 0; i < PLUG_PROBES; i++) {
		if (opends_async_await(&futs[1 + i]) != 0) {
			fprintf(stderr, "  plugged: probe %d = %zd\n", i,
			        futs[1 + i].result);
			goto out;
		}
	}
	for (int i = 0; i < PLUG_STREAM_WRITES; i++) {
		if (bws[i] != (ssize_t)PAGE) {
			fprintf(stderr, "  plugged: write %d = %zd\n", i,
			        bws[i]);
			goto out;
		}
	}
	err = fstat(a->fd, &st);
	if (err < 0 || (size_t)st.st_size != total) {
		fprintf(stderr, "  plugged: file size %lld, want %zu\n",
		        err < 0 ? -1LL : (long long)st.st_size, total);
		goto out;
	}
	r = pread(a->fd, host, total, 0);
	if (r != (ssize_t)total || memcmp(host, image, total) != 0) {
		fprintf(stderr, "  plugged: read-back mismatch (r=%zd)\n", r);
		goto out;
	}
	fprintf(stderr, "[stream] %d writes behind a plug + read-back ok\n",
	        PLUG_STREAM_WRITES + 1);
	rc = 0;
out:
	free(image);
	free(host);
	if (gpu) {
		opends_free(gpu);
	}
	opends_stream_deregister(blocking);
	cuStreamDestroy(blocking);
	return rc;
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <scratch-file-on-mount>\n", argv[0]);
		return 1;
	}
	const char *path = argv[1];

	setenv("OPENDS_AISIO_WRITE_BUF_MB", WRITE_BUF_MB, 1);
	setenv("OPENDS_AISIO_WRITE_BUFS", WRITE_BUFS, 1);

	struct aisio_homi a;
	if (aisio_homi_setup_flags(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT,
	                           &a) < 0)
		return 1;

	fprintf(stderr, "opends_stream_write tests (aisio backend, HOMI)\n");

	int failed =
	        run_pass(&a, CU_STREAM_NON_BLOCKING, STREAM_FIXED, "stream");
	if (!failed) {
		failed = run_pass(&a, CU_STREAM_NON_BLOCKING, 0,
		                  "stream/deferred");
	}
	if (!failed) {
		failed = run_pass(&a, CU_STREAM_DEFAULT, STREAM_FIXED,
		                  "stream/blocking");
	}
	if (!failed) {
		failed = run_pass(&a, CU_STREAM_DEFAULT, 0,
		                  "stream/blocking/deferred");
	}
	if (!failed) {
		failed = stream_write_null_refused(&a);
	}
	if (!failed) {
		failed = stream_write_refused(path, CU_STREAM_NON_BLOCKING,
		                              STREAM_FIXED, "stream");
	}
	if (!failed) {
		failed = stream_write_refused(path, CU_STREAM_NON_BLOCKING, 0,
		                              "stream/deferred");
	}
	if (!failed) {
		failed = stream_write_refused(path, CU_STREAM_DEFAULT,
		                              STREAM_FIXED, "stream/blocking");
	}
	if (!failed) {
		failed = stream_write_refused(path, CU_STREAM_DEFAULT, 0,
		                              "stream/blocking/deferred");
	}
	if (!failed) {
		failed = stream_write_empty(&a, STREAM_FIXED, "stream");
	}
	if (!failed) {
		failed = stream_write_empty(&a, 0, "stream/deferred");
	}
	if (!failed) {
		failed = stream_write_rounds(&a, STREAM_FIXED, "rounds");
	}
	if (!failed) {
		failed = stream_write_rounds(&a, 0, "rounds/deferred");
	}
	if (!failed) {
		failed = stream_write_late_values(&a, STREAM_FIXED, "late");
	}
	if (!failed) {
		failed = stream_write_late_values(&a, 0, "late/deferred");
	}
	if (!failed) {
		failed = stream_write_plugged(&a);
	}
	if (!failed) {
		failed =
		        stream_write_many_rounds(&a, 0, "rounds/many/deferred");
	}
	if (!failed) {
		failed = stream_write_many_rounds(&a, STREAM_FIXED,
		                                  "rounds/many");
	}
	if (failed) {
		fprintf(stderr, "%d test(s) failed\n", failed);
		/* an I/O thread may be stuck: no driver close */
		fflush(NULL);
		_exit(1);
	}

	aisio_homi_teardown(&a);
	unlink(path);

	fprintf(stderr, "all ok\n");
	return 0;
}
