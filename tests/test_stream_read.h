/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_stream_read.h - Backend-agnostic unit tests for stream-based
 * opends_stream_read.
 *
 * Mirrors test_sync_read.h but routes every read through
 * opends_stream_read with a real CUDA stream and waits for completion
 * via cuStreamSynchronize. Includes additional cases that exercise
 * stream ordering and concurrent submission across multiple streams.
 *
 * Include from a backend-specific source file that provides main() and
 * initializes a struct stream_test_env. Expects the same 16-page
 * (65536-byte) pattern file used by the sync tests.
 */
#ifndef TEST_STREAM_READ_H_
#define TEST_STREAM_READ_H_

#include "opends.h"
#include "opends.h"
#include "read_pattern.h"

#include <cuda.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define STREAM_TEST_MAX_STREAMS 32

/*
 * Bounded sync: poll cuStreamQuery against a wall-clock deadline.
 * Returns 0 on success, 1 on timeout, 2 on driver error. Used by
 * tests that must not block the whole runner if the backend hangs.
 */
static int
stream_test_stream_sync_timeout(CUstream stream, double timeout_s,
                                const char *label)
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
			return 2;
		}
		clock_gettime(CLOCK_MONOTONIC, &now);
		double elapsed = (now.tv_sec - t0.tv_sec) +
		                 (now.tv_nsec - t0.tv_nsec) * 1e-9;
		if (elapsed >= timeout_s) {
			fprintf(stderr,
			        "  %s: timed out after %.1fs (stream stuck)\n",
			        label, elapsed);
			return 1;
		}
		struct timespec sl = {0, 1000000};
		nanosleep(&sl, NULL);
	}
}

struct stream_test_env {
	opends_handle_t fh;
	CUstream stream;
	CUstream extra_streams[STREAM_TEST_MAX_STREAMS];
	int extra_stream_count;
	void *(*buf_to_host)(void *dst, const void *src, size_t n);
	void (*buf_zero)(void *buf, size_t n);
	void (*check_buffer)(const void *buf);
	void *(*buf_acquire)(size_t size);
	void (*buf_release)(void *buf);
	const char *mode_label;
	bool sub_lba_unsupported;
};

static ssize_t
stream_expected_bytes(char *dst, off_t offset, size_t size)
{
	if (offset < 0 || (size_t)offset >= FILE_SIZE)
		return 0;

	size_t avail = FILE_SIZE - (size_t)offset;
	if (size > avail)
		size = avail;

	for (size_t i = 0; i < size; i++)
		dst[i] = (char)pattern_byte(offset + (off_t)i);
	return (ssize_t)size;
}

static int
verify_stream_read(struct stream_test_env *env, void *buf, size_t alloc_size,
                   size_t size, off_t file_offset, off_t buf_offset,
                   const char *label)
{
	env->buf_zero(buf, alloc_size);

	size_t sz = size;
	off_t foff = file_offset;
	off_t boff = buf_offset;
	ssize_t bytes_read = 0;

	opends_error_t err = opends_stream_read(env->fh, buf, &sz, &foff, &boff,
	                                        &bytes_read, env->stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  %s: opends_stream_read: %s (dev_err=%d)\n",
		        label, opends_op_status_error(err.err),
		        (int)err.dev_err);
		return 1;
	}

	CUresult cr = cuStreamSynchronize(env->stream);
	if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "  %s: cuStreamSynchronize: %d\n", label,
		        (int)cr);
		return 1;
	}

	if (bytes_read < 0) {
		fprintf(stderr, "  %s: bytes_read = %s\n", label,
		        opends_op_status_error(
		                (opends_op_error_t)(-bytes_read)));
		return 1;
	}

	char *host = malloc(alloc_size);
	char *expected = malloc(size ? size : 1);
	if (!host || !expected) {
		free(host);
		free(expected);
		return 1;
	}

	env->buf_to_host(host, buf, alloc_size);

	ssize_t expected_n = stream_expected_bytes(expected, file_offset, size);
	if (bytes_read != expected_n) {
		fprintf(stderr, "  %s: got %zd bytes, expected %zd\n", label,
		        bytes_read, expected_n);
		free(host);
		free(expected);
		return 1;
	}

	if (bytes_read > 0 &&
	    memcmp(host + buf_offset, expected, (size_t)bytes_read) != 0) {
		fprintf(stderr, "  %s: data mismatch\n", label);
		free(host);
		free(expected);
		return 1;
	}

	free(host);
	free(expected);
	return 0;
}

/* --- Test cases ------------------------------------------------- */

static int
stream_test_single_block(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, PAGE, PAGE, 0, 0, "single_block");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_multi_block(struct stream_test_env *env)
{
	size_t size = 4 * PAGE;
	void *buf = env->buf_acquire(size);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, size, size, 0, 0, "multi_block");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_at_offset(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, PAGE, PAGE, 2 * PAGE, 0,
	                            "at_offset");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_full_file(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(FILE_SIZE);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, FILE_SIZE, FILE_SIZE, 0, 0,
	                            "full_file");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_last_block(struct stream_test_env *env)
{
	off_t off = (off_t)((FILE_PAGES - 1) * PAGE);
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, PAGE, PAGE, off, 0, "last_block");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_short_at_eof(struct stream_test_env *env)
{
	size_t req = 2 * PAGE;
	off_t off = (off_t)((FILE_PAGES - 1) * PAGE);
	void *buf = env->buf_acquire(req);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, req, req, off, 0, "short_at_eof");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_zero_size(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, PAGE, 0, 0, 0, "zero_size");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_buf_offset(struct stream_test_env *env)
{
	size_t alloc = 2 * PAGE;
	void *buf = env->buf_acquire(alloc);
	if (!buf)
		return 1;
	int rc = verify_stream_read(env, buf, alloc, PAGE, 0, PAGE,
	                            "buf_offset");
	env->buf_release(buf);
	return rc;
}

static int
stream_test_offset_size_sweep(struct stream_test_env *env)
{
	static const struct {
		off_t offset;
		size_t size;
	} cases[] = {
	        {0, 4096},  {0, 65536},    {4096, 4096}, {8192, 4096},
	        {0, 32768}, {4096, 65536}, {0, 16384},   {8192, 32768},
	};

	size_t max_size = 0;
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		if (cases[i].size > max_size)
			max_size = cases[i].size;
	}

	void *buf = env->buf_acquire(max_size);
	if (!buf)
		return 1;

	int rc = 0;
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		char label[64];
		snprintf(label, sizeof(label), "sweep[off=%ld,sz=%zu]",
		         (long)cases[i].offset, cases[i].size);
		rc = verify_stream_read(env, buf, max_size, cases[i].size,
		                        cases[i].offset, 0, label);
		if (rc)
			break;
	}

	env->buf_release(buf);
	return rc;
}

/*
 * Stream ordering: submit two reads back-to-back on the same stream
 * before syncing. Both must complete and contain the right data.
 */
static int
stream_test_stream_ordering(struct stream_test_env *env)
{
	void *buf_a = env->buf_acquire(PAGE);
	void *buf_b = env->buf_acquire(PAGE);
	if (!buf_a || !buf_b) {
		if (buf_a)
			env->buf_release(buf_a);
		if (buf_b)
			env->buf_release(buf_b);
		return 1;
	}

	env->buf_zero(buf_a, PAGE);
	env->buf_zero(buf_b, PAGE);

	size_t sz_a = PAGE, sz_b = PAGE;
	off_t off_a = 0, off_b = 4 * PAGE;
	off_t boff = 0;
	ssize_t br_a = 0, br_b = 0;

	opends_error_t err;
	err = opends_stream_read(env->fh, buf_a, &sz_a, &off_a, &boff, &br_a,
	                         env->stream);
	if (err.err != OPENDS_SUCCESS)
		goto fail;
	err = opends_stream_read(env->fh, buf_b, &sz_b, &off_b, &boff, &br_b,
	                         env->stream);
	if (err.err != OPENDS_SUCCESS)
		goto fail;

	if (cuStreamSynchronize(env->stream) != CUDA_SUCCESS)
		goto fail;
	if (br_a != (ssize_t)PAGE || br_b != (ssize_t)PAGE) {
		fprintf(stderr, "  ordering: br_a=%zd br_b=%zd\n", br_a, br_b);
		goto fail;
	}

	char *host_a = malloc(PAGE);
	char *host_b = malloc(PAGE);
	char *exp_a = malloc(PAGE);
	char *exp_b = malloc(PAGE);
	int rc = 1;
	if (!host_a || !host_b || !exp_a || !exp_b)
		goto out;

	env->buf_to_host(host_a, buf_a, PAGE);
	env->buf_to_host(host_b, buf_b, PAGE);
	stream_expected_bytes(exp_a, 0, PAGE);
	stream_expected_bytes(exp_b, 4 * PAGE, PAGE);

	if (memcmp(host_a, exp_a, PAGE) != 0) {
		fprintf(stderr, "  ordering: read A mismatch\n");
		goto out;
	}
	if (memcmp(host_b, exp_b, PAGE) != 0) {
		fprintf(stderr, "  ordering: read B mismatch\n");
		goto out;
	}
	rc = 0;
out:
	free(host_a);
	free(host_b);
	free(exp_a);
	free(exp_b);
	env->buf_release(buf_a);
	env->buf_release(buf_b);
	return rc;
fail:
	env->buf_release(buf_a);
	env->buf_release(buf_b);
	return 1;
}

/*
 * Submit one read per extra stream concurrently, then sync each.
 * Verifies that the orchestrator handles many streams in flight at the
 * same time.
 */
static int
stream_test_concurrent_streams(struct stream_test_env *env)
{
	if (env->extra_stream_count <= 0)
		return 0;

	int n = env->extra_stream_count;
	void **bufs = calloc(n, sizeof(*bufs));
	size_t *sizes = calloc(n, sizeof(*sizes));
	off_t *foffs = calloc(n, sizeof(*foffs));
	off_t *boffs = calloc(n, sizeof(*boffs));
	ssize_t *brs = calloc(n, sizeof(*brs));

	int rc = 1;
	if (!bufs || !sizes || !foffs || !boffs || !brs)
		goto out_free_arrays;

	for (int i = 0; i < n; i++) {
		bufs[i] = env->buf_acquire(PAGE);
		if (!bufs[i])
			goto out_free_bufs;
		env->buf_zero(bufs[i], PAGE);
		sizes[i] = PAGE;
		foffs[i] = (off_t)((i % FILE_PAGES) * PAGE);
		boffs[i] = 0;
	}

	for (int i = 0; i < n; i++) {
		opends_error_t err = opends_stream_read(
		        env->fh, bufs[i], &sizes[i], &foffs[i], &boffs[i],
		        &brs[i], env->extra_streams[i]);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  concurrent: stream[%d] submit: %s\n",
			        i, opends_op_status_error(err.err));
			goto out_free_bufs;
		}
	}

	for (int i = 0; i < n; i++) {
		if (cuStreamSynchronize(env->extra_streams[i]) !=
		    CUDA_SUCCESS) {
			fprintf(stderr,
			        "  concurrent: stream[%d] sync failed\n", i);
			goto out_free_bufs;
		}
	}

	for (int i = 0; i < n; i++) {
		if (brs[i] != (ssize_t)PAGE) {
			fprintf(stderr, "  concurrent[%d]: bytes=%zd\n", i,
			        brs[i]);
			goto out_free_bufs;
		}
		char *host = malloc(PAGE);
		char *exp = malloc(PAGE);
		if (!host || !exp) {
			free(host);
			free(exp);
			goto out_free_bufs;
		}
		env->buf_to_host(host, bufs[i], PAGE);
		stream_expected_bytes(exp, foffs[i], PAGE);
		int mm = memcmp(host, exp, PAGE);
		free(host);
		free(exp);
		if (mm != 0) {
			fprintf(stderr, "  concurrent[%d]: mismatch\n", i);
			goto out_free_bufs;
		}
	}

	rc = 0;

out_free_bufs:
	for (int i = 0; i < n; i++)
		if (bufs[i])
			env->buf_release(bufs[i]);
out_free_arrays:
	free(bufs);
	free(sizes);
	free(foffs);
	free(boffs);
	free(brs);
	return rc;
}

/*
 * Submit one sub-LBA read per extra stream concurrently. Each read is
 * small enough that the backend must serve it from the head/tail
 * bounce path. Reproduces a multi-stream + bounce hang that the
 * page-aligned concurrent_streams test cannot catch.
 */
static int
stream_test_concurrent_short_reads(struct stream_test_env *env)
{
	if (env->extra_stream_count <= 0)
		return 0;

	int n = env->extra_stream_count;
	const size_t short_size = 167;

	void **bufs = calloc(n, sizeof(*bufs));
	size_t *sizes = calloc(n, sizeof(*sizes));
	off_t *foffs = calloc(n, sizeof(*foffs));
	off_t *boffs = calloc(n, sizeof(*boffs));
	ssize_t *brs = calloc(n, sizeof(*brs));

	int rc = 1;
	if (!bufs || !sizes || !foffs || !boffs || !brs)
		goto out_free_arrays;

	for (int i = 0; i < n; i++) {
		bufs[i] = env->buf_acquire(PAGE);
		if (!bufs[i])
			goto out_free_bufs;
		env->buf_zero(bufs[i], PAGE);
		sizes[i] = short_size;
		/* LBA-aligned page offsets with sub-LBA size: each op reads
		 * the first short_size bytes of a distinct LBA-aligned page,
		 * which forces the backend into the tail-bounce path. */
		foffs[i] = (off_t)((i % FILE_PAGES) * PAGE);
		boffs[i] = 0;
	}

	for (int i = 0; i < n; i++) {
		opends_error_t err = opends_stream_read(
		        env->fh, bufs[i], &sizes[i], &foffs[i], &boffs[i],
		        &brs[i], env->extra_streams[i]);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  concurrent_short[%d]: submit: %s\n",
			        i, opends_op_status_error(err.err));
			goto out_free_bufs;
		}
	}

	for (int i = 0; i < n; i++) {
		char label[64];
		snprintf(label, sizeof(label), "concurrent_short[%d]", i);
		if (stream_test_stream_sync_timeout(env->extra_streams[i], 20.0,
		                                    label) != 0)
			goto out_free_bufs;
	}

	for (int i = 0; i < n; i++) {
		if (brs[i] != (ssize_t)short_size) {
			fprintf(stderr, "  concurrent_short[%d]: bytes=%zd\n",
			        i, brs[i]);
			goto out_free_bufs;
		}
		char *host = malloc(short_size);
		char *exp = malloc(short_size);
		if (!host || !exp) {
			free(host);
			free(exp);
			goto out_free_bufs;
		}
		env->buf_to_host(host, bufs[i], short_size);
		stream_expected_bytes(exp, foffs[i], short_size);
		int mm = memcmp(host, exp, short_size);
		free(host);
		free(exp);
		if (mm != 0) {
			fprintf(stderr, "  concurrent_short[%d]: mismatch\n",
			        i);
			goto out_free_bufs;
		}
	}

	rc = 0;

out_free_bufs:
	for (int i = 0; i < n; i++)
		if (bufs[i])
			env->buf_release(bufs[i]);
out_free_arrays:
	free(bufs);
	free(sizes);
	free(foffs);
	free(boffs);
	free(brs);
	return rc;
}

/*
 * Submit many reads on a single stream to exceed casual queue depth.
 * All must complete with the right data, demonstrating that the
 * orchestrator drains a backlog correctly.
 */
static int
stream_test_burst_single_stream(struct stream_test_env *env)
{
	enum { N = 32 };
	void *bufs[N] = {0};
	size_t sizes[N];
	off_t foffs[N];
	off_t boffs[N];
	ssize_t brs[N] = {0};

	int rc = 1;
	for (int i = 0; i < N; i++) {
		bufs[i] = env->buf_acquire(PAGE);
		if (!bufs[i])
			goto out;
		env->buf_zero(bufs[i], PAGE);
		sizes[i] = PAGE;
		foffs[i] = (off_t)((i % FILE_PAGES) * PAGE);
		boffs[i] = 0;
	}

	for (int i = 0; i < N; i++) {
		opends_error_t err = opends_stream_read(
		        env->fh, bufs[i], &sizes[i], &foffs[i], &boffs[i],
		        &brs[i], env->stream);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  burst: submit %d: %s\n", i,
			        opends_op_status_error(err.err));
			goto out;
		}
	}

	if (cuStreamSynchronize(env->stream) != CUDA_SUCCESS)
		goto out;

	for (int i = 0; i < N; i++) {
		if (brs[i] != (ssize_t)PAGE) {
			fprintf(stderr, "  burst[%d]: bytes=%zd\n", i, brs[i]);
			goto out;
		}
		char *host = malloc(PAGE);
		char *exp = malloc(PAGE);
		if (!host || !exp) {
			free(host);
			free(exp);
			goto out;
		}
		env->buf_to_host(host, bufs[i], PAGE);
		stream_expected_bytes(exp, foffs[i], PAGE);
		int mm = memcmp(host, exp, PAGE);
		free(host);
		free(exp);
		if (mm != 0) {
			fprintf(stderr, "  burst[%d]: mismatch\n", i);
			goto out;
		}
	}
	rc = 0;
out:
	for (int i = 0; i < N; i++)
		if (bufs[i])
			env->buf_release(bufs[i]);
	return rc;
}

/*
 * Deferred-evaluation semantics: opends_stream_read must dereference
 * size_p / file_offset_p / buf_offset_p when the stream actually runs
 * the op, not when the call is posted. Submit a host fn that mutates
 * those values, then submit the read on the same stream with the
 * caller-side values still pointing at sentinels. The read must see
 * the post-mutation values.
 */
struct deferred_mutate_args {
	size_t *size_p;
	off_t *file_offset_p;
	off_t *buf_offset_p;
	size_t target_size;
	off_t target_file_offset;
	off_t target_buf_offset;
};

static void CUDA_CB
stream_test_deferred_mutate(void *userdata)
{
	struct deferred_mutate_args *a = userdata;
	*a->size_p = a->target_size;
	*a->file_offset_p = a->target_file_offset;
	*a->buf_offset_p = a->target_buf_offset;
}

static int
stream_test_deferred_eval(struct stream_test_env *env)
{
	size_t alloc = 2 * PAGE;
	void *buf = env->buf_acquire(alloc);
	if (!buf)
		return 1;
	env->buf_zero(buf, alloc);

	/* Sentinels that would be wrong if read at call time. */
	size_t sz = 0;
	off_t foff = 0;
	off_t boff = 0;
	ssize_t br = 0;

	struct deferred_mutate_args mut = {
	        .size_p = &sz,
	        .file_offset_p = &foff,
	        .buf_offset_p = &boff,
	        .target_size = PAGE,
	        .target_file_offset = 3 * PAGE,
	        .target_buf_offset = PAGE,
	};

	int rc = 1;
	if (cuLaunchHostFunc(env->stream, stream_test_deferred_mutate, &mut) !=
	    CUDA_SUCCESS) {
		fprintf(stderr, "  deferred_eval: cuLaunchHostFunc failed\n");
		goto out;
	}

	opends_error_t err = opends_stream_read(env->fh, buf, &sz, &foff, &boff,
	                                        &br, env->stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  deferred_eval: submit: %s\n",
		        opends_op_status_error(err.err));
		goto out;
	}

	if (cuStreamSynchronize(env->stream) != CUDA_SUCCESS)
		goto out;

	if (br != (ssize_t)PAGE) {
		fprintf(stderr,
		        "  deferred_eval: bytes=%zd (call-time values "
		        "leaked through)\n",
		        br);
		goto out;
	}

	char *host = malloc(alloc);
	char *exp = malloc(PAGE);
	if (!host || !exp) {
		free(host);
		free(exp);
		goto out;
	}
	env->buf_to_host(host, buf, alloc);
	stream_expected_bytes(exp, mut.target_file_offset, PAGE);
	int mm = memcmp(host + mut.target_buf_offset, exp, PAGE);
	free(host);
	free(exp);
	if (mm != 0) {
		fprintf(stderr, "  deferred_eval: data mismatch at boff=%ld\n",
		        (long)mut.target_buf_offset);
		goto out;
	}
	rc = 0;
out:
	env->buf_release(buf);
	return rc;
}

/*
 * Multi-stream burst: each of N streams gets a backlog of M reads
 * before any are synced. Exercises orchestrator throughput, the xnvme
 * cmd_ctx pool under pressure, and CUDA's host-fn worker pool with
 * many simultaneously-blocked host fns from distinct streams.
 *
 * Each stream owns one buffer of size per*PAGE; op j on stream s
 * writes a single page at buf_offset j*PAGE. This keeps the backend's
 * registered-buffer table from filling up and mirrors the realistic
 * pattern of one DMA-eligible buffer per worker stream.
 */
static int
stream_test_multi_stream_burst(struct stream_test_env *env)
{
	if (env->extra_stream_count <= 0)
		return 0;

	const int n = env->extra_stream_count;
	const int per = 8;
	const int total = n * per;
	const size_t per_buf = (size_t)per * PAGE;

	void **bufs = calloc(n, sizeof(*bufs));
	size_t *sizes = calloc(total, sizeof(*sizes));
	off_t *foffs = calloc(total, sizeof(*foffs));
	off_t *boffs = calloc(total, sizeof(*boffs));
	ssize_t *brs = calloc(total, sizeof(*brs));

	int rc = 1;
	if (!bufs || !sizes || !foffs || !boffs || !brs)
		goto out_free_arrays;

	for (int s = 0; s < n; s++) {
		bufs[s] = env->buf_acquire(per_buf);
		if (!bufs[s])
			goto out_free_bufs;
		env->buf_zero(bufs[s], per_buf);
	}
	for (int i = 0; i < total; i++) {
		int j = i % per;
		sizes[i] = PAGE;
		foffs[i] = (off_t)((i % FILE_PAGES) * PAGE);
		boffs[i] = (off_t)((size_t)j * PAGE);
	}

	/* Submit per-stream backlog interleaved across streams to maximise
	 * orchestrator-side fan-in. */
	for (int j = 0; j < per; j++) {
		for (int s = 0; s < n; s++) {
			int i = s * per + j;
			opends_error_t err = opends_stream_read(
			        env->fh, bufs[s], &sizes[i], &foffs[i],
			        &boffs[i], &brs[i], env->extra_streams[s]);
			if (err.err != OPENDS_SUCCESS) {
				fprintf(stderr,
				        "  multi_burst: s=%d j=%d submit: %s\n",
				        s, j, opends_op_status_error(err.err));
				goto out_free_bufs;
			}
		}
	}

	for (int s = 0; s < n; s++) {
		if (cuStreamSynchronize(env->extra_streams[s]) !=
		    CUDA_SUCCESS) {
			fprintf(stderr,
			        "  multi_burst: stream[%d] sync failed\n", s);
			goto out_free_bufs;
		}
	}

	char *host = malloc(per_buf);
	char *exp = malloc(PAGE);
	if (!host || !exp) {
		free(host);
		free(exp);
		goto out_free_bufs;
	}

	for (int s = 0; s < n; s++) {
		env->buf_to_host(host, bufs[s], per_buf);
		for (int j = 0; j < per; j++) {
			int i = s * per + j;
			if (brs[i] != (ssize_t)PAGE) {
				fprintf(stderr,
				        "  multi_burst[s=%d j=%d]: "
				        "bytes=%zd\n",
				        s, j, brs[i]);
				goto out_free_host;
			}
			stream_expected_bytes(exp, foffs[i], PAGE);
			if (memcmp(host + boffs[i], exp, PAGE) != 0) {
				fprintf(stderr,
				        "  multi_burst[s=%d j=%d]: "
				        "mismatch\n",
				        s, j);
				goto out_free_host;
			}
		}
	}
	rc = 0;
out_free_host:
	free(host);
	free(exp);

out_free_bufs:
	for (int s = 0; s < n; s++)
		if (bufs[s])
			env->buf_release(bufs[s]);
out_free_arrays:
	free(bufs);
	free(sizes);
	free(foffs);
	free(boffs);
	free(brs);
	return rc;
}

/*
 * Stream-ordered consumer: an op enqueued on the user's stream right
 * after opends_stream_read must observe the bytes written by the
 * read, without any host-side sync between them. We zero the GPU buf
 * first (under a full device sync), then submit:
 *   1) opends_stream_read on stream
 *   2) cuMemcpyDtoHAsync(pinned, buf, ...) on the same stream
 * and only sync after both are queued. The memcpy is stream-ordered
 * behind the read's cuStreamWaitValue32(done_seq) gate. If that gate
 * releases early — or if done_seq advances before the NVMe DMA bytes
 * are visible to the GPU — the memcpy snapshots zeros, not the
 * expected pattern. Loops to make any race observable across runs.
 */
static int
stream_test_stream_consumer(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	void *host_pinned = NULL;
	if (cuMemHostAlloc(&host_pinned, PAGE, CU_MEMHOSTALLOC_PORTABLE) !=
	    CUDA_SUCCESS) {
		env->buf_release(buf);
		return 1;
	}

	char *expected = malloc(PAGE);
	if (!expected) {
		cuMemFreeHost(host_pinned);
		env->buf_release(buf);
		return 1;
	}

	enum { ITERS = 16 };
	int rc = 1;

	for (int i = 0; i < ITERS; i++) {
		env->buf_zero(buf, PAGE);
		memset(host_pinned, 0xff, PAGE);

		size_t sz = PAGE;
		off_t foff = (off_t)((i % FILE_PAGES) * PAGE);
		off_t boff = 0;
		ssize_t br = 0;
		opends_error_t err = opends_stream_read(
		        env->fh, buf, &sz, &foff, &boff, &br, env->stream);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  stream_consumer[%d]: submit: %s\n",
			        i, opends_op_status_error(err.err));
			goto out;
		}

		if (cuMemcpyDtoHAsync(host_pinned, (CUdeviceptr)buf, PAGE,
		                      env->stream) != CUDA_SUCCESS) {
			fprintf(stderr,
			        "  stream_consumer[%d]: cuMemcpyDtoHAsync "
			        "failed\n",
			        i);
			goto out;
		}

		if (cuStreamSynchronize(env->stream) != CUDA_SUCCESS) {
			fprintf(stderr, "  stream_consumer[%d]: sync failed\n",
			        i);
			goto out;
		}

		if (br != (ssize_t)PAGE) {
			fprintf(stderr, "  stream_consumer[%d]: bytes=%zd\n", i,
			        br);
			goto out;
		}
		stream_expected_bytes(expected, foff, PAGE);
		if (memcmp(host_pinned, expected, PAGE) != 0) {
			fprintf(stderr,
			        "  stream_consumer[%d]: consumer observed "
			        "wrong "
			        "bytes (read/done_seq gate broken?)\n",
			        i);
			goto out;
		}
	}

	rc = 0;
out:
	free(expected);
	cuMemFreeHost(host_pinned);
	env->buf_release(buf);
	return rc;
}

/*
 * Only meaningful when the backend rejects sub-LBA tails: the rejection
 * must report OPENDS_INVALID_VALUE through bytes_read, it must release
 * the stream (rejection happens behind the gate, so a leaked gate hangs
 * the stream forever), and an aligned read on the same stream must still
 * succeed afterwards.
 */
static int
stream_test_unaligned_rejected(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	int rc = 1;
	env->buf_zero(buf, PAGE);

	/* Sub-LBA for any NVMe LBA format (min 512 bytes). */
	size_t sz = 167;
	off_t foff = 0;
	off_t boff = 0;
	ssize_t bytes_read = 0;

	opends_error_t err = opends_stream_read(env->fh, buf, &sz, &foff, &boff,
	                                        &bytes_read, env->stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  unaligned_rejected: submit: %s\n",
		        opends_op_status_error(err.err));
		goto out;
	}

	if (stream_test_stream_sync_timeout(env->stream, 20.0,
	                                    "unaligned_rejected") != 0)
		goto out;

	if (bytes_read != -(ssize_t)OPENDS_INVALID_VALUE) {
		fprintf(stderr,
		        "  unaligned_rejected: bytes_read = %zd, expected "
		        "%zd\n",
		        bytes_read, -(ssize_t)OPENDS_INVALID_VALUE);
		goto out;
	}

	rc = verify_stream_read(env, buf, PAGE, PAGE, 0, 0,
	                        "unaligned_rejected/aligned_after");
out:
	env->buf_release(buf);
	return rc;
}

/*
 * The stream path carries one bounce copy per stream, so it stages a
 * sub-LBA tail but not an unaligned start. The rejection must report
 * OPENDS_INVALID_VALUE through bytes_read, release the stream, and leave
 * an aligned read on that stream working.
 */
static int
stream_test_unaligned_head_rejected(struct stream_test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	int rc = 1;
	env->buf_zero(buf, PAGE);

	size_t sz = PAGE;
	off_t foff = 4;
	off_t boff = 0;
	ssize_t bytes_read = 0;

	opends_error_t err = opends_stream_read(env->fh, buf, &sz, &foff, &boff,
	                                        &bytes_read, env->stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  unaligned_head_rejected: submit: %s\n",
		        opends_op_status_error(err.err));
		goto out;
	}

	if (stream_test_stream_sync_timeout(env->stream, 20.0,
	                                    "unaligned_head_rejected") != 0)
		goto out;

	if (bytes_read != -(ssize_t)OPENDS_INVALID_VALUE) {
		fprintf(stderr,
		        "  unaligned_head_rejected: bytes_read = %zd, expected "
		        "%zd\n",
		        bytes_read, -(ssize_t)OPENDS_INVALID_VALUE);
		goto out;
	}

	rc = verify_stream_read(env, buf, PAGE, PAGE, 0, 0,
	                        "unaligned_head_rejected/aligned_after");
out:
	env->buf_release(buf);
	return rc;
}

/* --- Test runner ------------------------------------------------ */

struct stream_test_entry {
	const char *name;
	int (*fn)(struct stream_test_env *);
	bool needs_sub_lba;
	bool needs_aligned_only;
};

/* clang-format off */
static const struct stream_test_entry stream_read_tests[] = {
	{"single_block",        stream_test_single_block},
	{"multi_block",         stream_test_multi_block},
	{"at_offset",           stream_test_at_offset},
	{"full_file",           stream_test_full_file},
	{"last_block",          stream_test_last_block},
	{"short_at_eof",        stream_test_short_at_eof},
	{"zero_size",           stream_test_zero_size},
	{"buf_offset",          stream_test_buf_offset},
	{"offset_size_sweep",   stream_test_offset_size_sweep},
	{"stream_ordering",     stream_test_stream_ordering},
	{"stream_consumer",     stream_test_stream_consumer},
	{"deferred_eval",       stream_test_deferred_eval},
	{"concurrent_streams",      stream_test_concurrent_streams},
	{"concurrent_short_reads",  stream_test_concurrent_short_reads, true},
	{"unaligned_rejected",      stream_test_unaligned_rejected, false, true},
	{"unaligned_head_rejected", stream_test_unaligned_head_rejected},
	{"burst_single_stream",     stream_test_burst_single_stream},
	{"multi_stream_burst",      stream_test_multi_stream_burst},
};
/* clang-format on */

#define NSTREAM_READ_TESTS                                                     \
	(sizeof(stream_read_tests) / sizeof(stream_read_tests[0]))

static int
run_stream_read_tests(struct stream_test_env *env)
{
	fprintf(stderr, "  [%s mode]\n",
	        env->mode_label ? env->mode_label : "default");

	if (env->check_buffer) {
		void *probe = env->buf_acquire(PAGE);
		if (!probe) {
			fprintf(stderr, "  buf_acquire probe failed\n");
			return 1;
		}
		env->check_buffer(probe);
		env->buf_release(probe);
	}

	int failed = 0;
	for (size_t i = 0; i < NSTREAM_READ_TESTS; i++) {
		if ((stream_read_tests[i].needs_sub_lba &&
		     env->sub_lba_unsupported) ||
		    (stream_read_tests[i].needs_aligned_only &&
		     !env->sub_lba_unsupported)) {
			fprintf(stderr, "  %-24s skip\n",
			        stream_read_tests[i].name);
			continue;
		}
		int rc = stream_read_tests[i].fn(env);
		fprintf(stderr, "  %-24s %s\n", stream_read_tests[i].name,
		        rc ? "FAIL" : "ok");
		if (rc) {
			/* A failed stream test (especially a timed-out
			 * cuStreamQuery) leaves CUDA streams and backend
			 * state poisoned; later tests would hang or report
			 * cascading errors. Stop immediately. */
			failed = rc;
			break;
		}
	}
	return failed;
}

#endif /* TEST_STREAM_READ_H_ */
