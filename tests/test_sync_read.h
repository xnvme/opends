/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_sync_read.h - Backend-agnostic unit tests for synchronous
 * opends_sync_read.
 *
 * Include from a backend-specific source file that provides main()
 * and initializes a struct test_env with buffer access callbacks.
 *
 * The test expects a 16-page (65536-byte) pattern file already
 * written by test_sync_read_prep. Expected read results are computed
 * in memory via expected_bytes(), so the test binary never needs to
 * read the file outside the backend.
 */
#ifndef TEST_SYNC_READ_H_
#define TEST_SYNC_READ_H_

#include "opends.h"
#include "read_pattern.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct test_env {
	opends_handle_t fh;
	void *(*buf_to_host)(void *dst, const void *src, size_t n);
	void (*buf_zero)(void *buf, size_t n);
	/* Optional: backend-specific assertion on acquired buffers
	 * (e.g. "this is a CUDA device pointer"). NULL means no check. */
	void (*check_buffer)(const void *buf);
	/* Buffer acquisition. In alloc-mode these wrap
	 * opends_mem_alloc/opends_free; in register-mode they wrap a
	 * backend-specific allocator plus opends_buf_register/deregister. */
	void *(*buf_acquire)(size_t size);
	void (*buf_release)(void *buf);
	const char *mode_label;
	/* Backend refuses a read that starts off a dword once whole LBAs
	 * follow (aisio). 0 means such reads succeed and verify data. */
	int rejects_nondword_head;
};

/*
 * Fill dst with the bytes that a read at (offset, size) should return.
 * Size is clamped to FILE_SIZE. Returns the number of bytes written.
 */
static ssize_t
expected_bytes(char *dst, off_t offset, size_t size)
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
verify_read(struct test_env *env, void *buf, size_t alloc_size, size_t size,
            off_t file_offset, off_t buf_offset, const char *label)
{
	env->buf_zero(buf, alloc_size);

	ssize_t n =
	        opends_sync_read(env->fh, buf, size, file_offset, buf_offset);
	if (n < 0) {
		fprintf(stderr, "  %s: opends_sync_read: %s\n", label,
		        opends_op_status_error((opends_op_error_t)(-n)));
		return 1;
	}

	char *host = malloc(alloc_size);
	char *expected = malloc(size);

	if (!host || !expected) {
		free(host);
		free(expected);
		fprintf(stderr, "  %s: malloc failed\n", label);
		return 1;
	}

	env->buf_to_host(host, buf, alloc_size);

	ssize_t expected_n = expected_bytes(expected, file_offset, size);
	if (n != expected_n) {
		fprintf(stderr, "  %s: got %zd bytes, expected %zd\n", label, n,
		        expected_n);
		free(host);
		free(expected);
		return 1;
	}

	if (n > 0 && memcmp(host + buf_offset, expected, (size_t)n) != 0) {
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
test_read_single_block(struct test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, PAGE, PAGE, 0, 0, "single_block");

	env->buf_release(buf);
	return rc;
}

static int
test_read_multi_block(struct test_env *env)
{
	size_t size = 4 * PAGE;
	void *buf = env->buf_acquire(size);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, size, size, 0, 0, "multi_block");

	env->buf_release(buf);
	return rc;
}

static int
test_read_at_offset(struct test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, PAGE, PAGE, 2 * PAGE, 0, "at_offset");

	env->buf_release(buf);
	return rc;
}

static int
test_read_multi_at_offset(struct test_env *env)
{
	size_t size = 2 * PAGE;
	void *buf = env->buf_acquire(size);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, size, size, PAGE, 0, "multi_at_offset");

	env->buf_release(buf);
	return rc;
}

static int
test_read_full_file(struct test_env *env)
{
	void *buf = env->buf_acquire(FILE_SIZE);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, FILE_SIZE, FILE_SIZE, 0, 0, "full_file");

	env->buf_release(buf);
	return rc;
}

static int
test_read_last_block(struct test_env *env)
{
	off_t off = (off_t)((FILE_PAGES - 1) * PAGE);
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, PAGE, PAGE, off, 0, "last_block");

	env->buf_release(buf);
	return rc;
}

/*
 * Request more bytes than remain at EOF. One page remains; request
 * two. Verify the return count and content of the short read.
 */
static int
test_read_short_at_eof(struct test_env *env)
{
	size_t req = 2 * PAGE;
	off_t off = (off_t)((FILE_PAGES - 1) * PAGE);
	void *buf = env->buf_acquire(req);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, req, req, off, 0, "short_at_eof");

	env->buf_release(buf);
	return rc;
}

/* Read at exact end of file; expect 0 bytes returned. */
static int
test_read_at_eof(struct test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	env->buf_zero(buf, PAGE);
	ssize_t n = opends_sync_read(env->fh, buf, PAGE, (off_t)FILE_SIZE, 0);
	if (n < 0) {
		fprintf(stderr, "  at_eof: %s\n",
		        opends_op_status_error((opends_op_error_t)(-n)));
		env->buf_release(buf);
		return 1;
	}

	env->buf_release(buf);
	if (n != 0) {
		fprintf(stderr, "  at_eof: got %zd bytes, expected 0\n", n);
		return 1;
	}
	return 0;
}

/* File offset past end of file; expect 0 bytes returned. */
static int
test_read_beyond_eof(struct test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	env->buf_zero(buf, PAGE);
	ssize_t n = opends_sync_read(env->fh, buf, PAGE,
	                             (off_t)(FILE_SIZE + PAGE), 0);
	if (n < 0) {
		fprintf(stderr, "  beyond_eof: %s\n",
		        opends_op_status_error((opends_op_error_t)(-n)));
		env->buf_release(buf);
		return 1;
	}

	env->buf_release(buf);
	if (n != 0) {
		fprintf(stderr, "  beyond_eof: got %zd bytes, expected 0\n", n);
		return 1;
	}
	return 0;
}

static int
test_read_zero_size(struct test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	ssize_t n = opends_sync_read(env->fh, buf, 0, 0, 0);

	env->buf_release(buf);
	if (n != 0) {
		fprintf(stderr, "  zero_size: got %zd, expected 0\n", n);
		return 1;
	}
	return 0;
}

static int
test_read_buf_offset(struct test_env *env)
{
	size_t alloc = 2 * PAGE;
	off_t boff = PAGE;
	void *buf = env->buf_acquire(alloc);
	if (!buf)
		return 1;

	int rc = verify_read(env, buf, alloc, PAGE, 0, boff, "buf_offset");

	env->buf_release(buf);
	return rc;
}

/*
 * Verify that a read with buf_offset modifies only the target
 * region. The prefix and suffix of the buffer must remain zero.
 */
static int
test_read_buf_offset_boundaries(struct test_env *env)
{
	size_t alloc = 3 * PAGE;
	off_t boff = PAGE;
	void *buf = env->buf_acquire(alloc);
	if (!buf)
		return 1;

	env->buf_zero(buf, alloc);

	ssize_t n = opends_sync_read(env->fh, buf, PAGE, 0, boff);
	if (n != (ssize_t)PAGE) {
		fprintf(stderr, "  buf_offset_bounds: got %zd bytes\n", n);
		env->buf_release(buf);
		return 1;
	}

	char *host = malloc(alloc);
	if (!host) {
		env->buf_release(buf);
		return 1;
	}
	env->buf_to_host(host, buf, alloc);

	char *zeroes = calloc(1, PAGE);
	int rc = 0;

	if (memcmp(host, zeroes, PAGE) != 0) {
		fprintf(stderr, "  buf_offset_bounds: prefix modified\n");
		rc = 1;
	}

	if (memcmp(host + 2 * PAGE, zeroes, PAGE) != 0) {
		fprintf(stderr, "  buf_offset_bounds: suffix modified\n");
		rc = 1;
	}

	free(zeroes);
	free(host);
	env->buf_release(buf);
	return rc;
}

static int
test_read_sequential_regions(struct test_env *env)
{
	size_t quarter = FILE_SIZE / 4;
	void *buf = env->buf_acquire(quarter);
	if (!buf)
		return 1;

	int rc = 0;
	for (int i = 0; i < 4; i++) {
		char label[64];
		snprintf(label, sizeof(label), "sequential[%d]", i);
		rc = verify_read(env, buf, quarter, quarter,
		                 (off_t)(i * (off_t)quarter), 0, label);
		if (rc)
			break;
	}

	env->buf_release(buf);
	return rc;
}

/*
 * Read two overlapping regions (pages 0-1 and pages 1-2) and verify
 * the shared page is identical in both reads.
 */
static int
test_read_overlapping_regions(struct test_env *env)
{
	size_t size = 2 * PAGE;
	void *buf = env->buf_acquire(size);
	if (!buf)
		return 1;

	char *host_a = malloc(size);
	char *host_b = malloc(size);
	if (!host_a || !host_b) {
		free(host_a);
		free(host_b);
		env->buf_release(buf);
		return 1;
	}

	env->buf_zero(buf, size);
	ssize_t n1 = opends_sync_read(env->fh, buf, size, 0, 0);
	if (n1 != (ssize_t)size) {
		fprintf(stderr, "  overlap: read A got %zd\n", n1);
		free(host_a);
		free(host_b);
		env->buf_release(buf);
		return 1;
	}
	env->buf_to_host(host_a, buf, size);

	env->buf_zero(buf, size);
	ssize_t n2 = opends_sync_read(env->fh, buf, size, PAGE, 0);
	if (n2 != (ssize_t)size) {
		fprintf(stderr, "  overlap: read B got %zd\n", n2);
		free(host_a);
		free(host_b);
		env->buf_release(buf);
		return 1;
	}
	env->buf_to_host(host_b, buf, size);

	/* Page 1 appears at offset PAGE in read A and offset 0 in read B. */
	int rc = 0;
	if (memcmp(host_a + PAGE, host_b, PAGE) != 0) {
		fprintf(stderr, "  overlap: shared region mismatch\n");
		rc = 1;
	}

	free(host_a);
	free(host_b);
	env->buf_release(buf);
	return rc;
}

/* Read the same block twice; both reads must return identical data. */
static int
test_read_repeated(struct test_env *env)
{
	void *buf = env->buf_acquire(PAGE);
	if (!buf)
		return 1;

	char *host_a = malloc(PAGE);
	char *host_b = malloc(PAGE);
	if (!host_a || !host_b) {
		free(host_a);
		free(host_b);
		env->buf_release(buf);
		return 1;
	}

	env->buf_zero(buf, PAGE);
	ssize_t n1 = opends_sync_read(env->fh, buf, PAGE, 0, 0);
	env->buf_to_host(host_a, buf, PAGE);

	env->buf_zero(buf, PAGE);
	ssize_t n2 = opends_sync_read(env->fh, buf, PAGE, 0, 0);
	env->buf_to_host(host_b, buf, PAGE);

	int rc = 0;
	if (n1 != n2) {
		fprintf(stderr, "  repeated: byte counts differ (%zd vs %zd)\n",
		        n1, n2);
		rc = 1;
	} else if (n1 > 0 && memcmp(host_a, host_b, (size_t)n1) != 0) {
		fprintf(stderr, "  repeated: data mismatch\n");
		rc = 1;
	}

	free(host_a);
	free(host_b);
	env->buf_release(buf);
	return rc;
}

/*
 * Sweep a table of (offset, size) pairs covering non-zero starts,
 * mid-file reads that span pages, and a short read that extends past
 * EOF. All offsets are page-aligned. Each pair is verified against
 * the in-memory pattern oracle.
 */
static int
test_read_offset_size_sweep(struct test_env *env)
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
		rc = verify_read(env, buf, max_size, cases[i].size,
		                 cases[i].offset, 0, label);
		if (rc)
			break;
	}

	env->buf_release(buf);
	return rc;
}

/*
 * Sweep reads that start mid-LBA. The backend has to stage the leading
 * partial LBA through a bounce buffer, so the cases cover a head alone,
 * a head plus whole LBAs, a head plus a trailing partial LBA, and a head
 * that runs past EOF. A dword-offset head is what a PRP offset permits
 * once whole LBAs follow it, so the offsets that are not multiples of 4
 * stay short enough to reach no whole LBA. The 512 and 4096 variants make
 * the cases meaningful whichever LBA size the device reports.
 */
static int
test_read_unaligned_head(struct test_env *env)
{
	static const struct {
		off_t offset;
		size_t size;
	} cases[] = {
	        {4, 8},
	        {4, 500},
	        {100, 412},
	        {512 + 4, 508},
	        {512 + 4, 600},
	        {4, 4092},
	        {4, 8192},
	        {4096 + 4, 12288},
	        {1024 + 64, 5000},
	        {8192 + 12, 20},
	        {(off_t)FILE_SIZE - 100, 200},
	        {1, 400},
	        {511, 2},
	        {4095, 4},
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
		snprintf(label, sizeof(label), "head[off=%ld,sz=%zu]",
		         (long)cases[i].offset, cases[i].size);
		rc = verify_read(env, buf, max_size, cases[i].size,
		                 cases[i].offset, 0, label);
		if (rc)
			break;
	}

	env->buf_release(buf);
	return rc;
}

/* An unaligned head landing at a non-zero buf_offset: both ends of the
 * copy are then offset, which is the case most likely to be got wrong.
 * The second case is sized so whole LBAs follow the head on a 4096-byte
 * device too, putting a direct DMA at a destination that is neither page
 * nor LBA aligned. */
static int
test_read_unaligned_head_buf_offset(struct test_env *env)
{
	static const struct {
		off_t offset;
		size_t size;
	} cases[] = {
	        {1024 + 4, 5000},
	        {PAGE + 4, 3 * PAGE},
	};
	const off_t boff = 64;

	size_t max_size = 0;
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		if (cases[i].size > max_size)
			max_size = cases[i].size;
	}
	size_t alloc = max_size + (size_t)boff;

	void *buf = env->buf_acquire(alloc);
	if (!buf)
		return 1;

	int rc = 0;
	for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		char label[64];
		snprintf(label, sizeof(label), "head_boff[off=%ld,sz=%zu]",
		         (long)cases[i].offset, cases[i].size);
		rc = verify_read(env, buf, alloc, cases[i].size,
		                 cases[i].offset, boff, label);
		if (rc)
			break;
	}

	env->buf_release(buf);
	return rc;
}

/*
 * Pin the rejection boundary. A read that starts off a dword and covers
 * whole LBAs past the first cannot express its start as a PRP offset,
 * so a backend that DMAs those LBAs directly refuses it with
 * OPENDS_INVALID_VALUE. A backend that bounces every block accepts the
 * same reads, so it verifies data instead. Both sizes reach whole LBAs
 * whether the device reports 512 or 4096.
 */
static int
test_read_nondword_head(struct test_env *env)
{
	static const struct {
		off_t offset;
		size_t size;
	} cases[] = {
	        {1, 12288},
	        {4095, 8192},
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
		snprintf(label, sizeof(label), "nondword[off=%ld,sz=%zu]",
		         (long)cases[i].offset, cases[i].size);
		if (!env->rejects_nondword_head) {
			rc = verify_read(env, buf, max_size, cases[i].size,
			                 cases[i].offset, 0, label);
		} else {
			ssize_t n =
			        opends_sync_read(env->fh, buf, cases[i].size,
			                         cases[i].offset, 0);
			if (n != -(ssize_t)OPENDS_INVALID_VALUE) {
				fprintf(stderr, "  %s: got %zd, expected %zd\n",
				        label, n,
				        -(ssize_t)OPENDS_INVALID_VALUE);
				rc = 1;
			}
		}
		if (rc)
			break;
	}

	env->buf_release(buf);
	return rc;
}

/* --- Test runner ------------------------------------------------ */

struct test_entry {
	const char *name;
	int (*fn)(struct test_env *);
};

/* clang-format off */
static const struct test_entry sync_read_tests[] = {
	{"single_block",        test_read_single_block},
	{"multi_block",         test_read_multi_block},
	{"at_offset",           test_read_at_offset},
	{"multi_at_offset",     test_read_multi_at_offset},
	{"full_file",           test_read_full_file},
	{"last_block",          test_read_last_block},
	{"short_at_eof",        test_read_short_at_eof},
	{"at_eof",              test_read_at_eof},
	{"beyond_eof",          test_read_beyond_eof},
	{"zero_size",           test_read_zero_size},
	{"buf_offset",          test_read_buf_offset},
	{"buf_offset_bounds",   test_read_buf_offset_boundaries},
	{"sequential_regions",  test_read_sequential_regions},
	{"overlapping_regions", test_read_overlapping_regions},
	{"repeated",            test_read_repeated},
	{"offset_size_sweep",   test_read_offset_size_sweep},
	{"unaligned_head",      test_read_unaligned_head},
	{"unaligned_head_boff", test_read_unaligned_head_buf_offset},
	{"nondword_head",       test_read_nondword_head},
};
/* clang-format on */

#define NSYNC_READ_TESTS (sizeof(sync_read_tests) / sizeof(sync_read_tests[0]))

static int
run_sync_read_tests(struct test_env *env)
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

	for (size_t i = 0; i < NSYNC_READ_TESTS; i++) {
		int rc = sync_read_tests[i].fn(env);
		fprintf(stderr, "  %-24s %s\n", sync_read_tests[i].name,
		        rc ? "FAIL" : "ok");
		failed += rc;
	}
	return failed;
}

#endif /* TEST_SYNC_READ_H_ */
