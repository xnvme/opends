/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_async_read.h - Backend-agnostic async read test.
 *
 * Include from a backend-specific source file that provides main().
 *
 * Phase 1 pipelines a window of in-flight futures: each await frees a
 * slot that is refilled with the next case, so many operations overlap
 * on the backend. Each completion is verified against the in-memory
 * pattern. The case table includes sub-block sizes and offsets that start
 * mid-LBA, so concurrent operations exercise per-op staging at both ends
 * of a span.
 *
 * Phase 2 submits a full window in one burst and awaits the futures in
 * reverse submission order: await order must not matter.
 *
 * Expects the 16-page pattern file written by test_sync_read_prep.
 */
#ifndef TEST_ASYNC_READ_H_
#define TEST_ASYNC_READ_H_

#include "opends.h"
#include "read_pattern.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASYNC_TEST_WINDOW 8
#define ASYNC_TEST_OPS 64

struct async_read_env {
	opends_handle_t fh;
	void *(*buf_to_host)(void *dst, const void *src, size_t n);
	void *(*buf_acquire)(size_t size);
	void (*buf_release)(void *buf);
};

static const struct {
	off_t offset;
	size_t size;
} async_read_cases[] = {
        {0, PAGE},
        {0, FILE_SIZE},
        {PAGE, 2 * PAGE},
        {2 * PAGE, PAGE},
        {4 * PAGE, 4 * PAGE},
        {8 * PAGE, 8 * PAGE},
        {(FILE_PAGES - 1) * PAGE, PAGE},
        {0, 700},
        {2 * PAGE, PAGE + 300},
        {0, FILE_SIZE / 2},
        {4, 700},
        {PAGE + 4, 2 * PAGE},
        {1024 + 64, 5000},
        {2 * PAGE + 516, 300},
        {PAGE - 1, 2},
};

#define ASYNC_READ_NCASES                                                      \
	(sizeof(async_read_cases) / sizeof(async_read_cases[0]))

struct async_read_slot {
	void *buf;
	opends_async_future_t future;
	off_t offset;
	size_t size;
};

static int
async_submit_case(struct async_read_env *env, struct async_read_slot *slot,
                  unsigned case_idx)
{
	size_t ci = case_idx % ASYNC_READ_NCASES;

	slot->offset = async_read_cases[ci].offset;
	slot->size = async_read_cases[ci].size;

	opends_error_t err = opends_async_read(env->fh, slot->buf, slot->size,
	                                       slot->offset, 0, &slot->future);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  submit(op=%u, off=%ld, size=%zu): %s\n",
		        case_idx, (long)slot->offset, slot->size,
		        opends_op_status_error(err.err));
		return -1;
	}
	return 0;
}

static int
async_await_verify(struct async_read_env *env, struct async_read_slot *slot,
                   char *host)
{
	ssize_t n = opends_async_await(&slot->future);

	if (n != (ssize_t)slot->size) {
		fprintf(stderr, "  await(off=%ld, size=%zu): rc=%zd\n",
		        (long)slot->offset, slot->size, n);
		return -1;
	}

	env->buf_to_host(host, slot->buf, slot->size);
	for (size_t b = 0; b < slot->size; b++) {
		if ((unsigned char)host[b] !=
		    pattern_byte(slot->offset + (off_t)b)) {
			fprintf(stderr,
			        "  mismatch at byte %zu (off=%ld, size=%zu)\n",
			        b, (long)slot->offset, slot->size);
			return -1;
		}
	}
	return 0;
}

static int
run_async_reverse_test(struct async_read_env *env,
                       struct async_read_slot *slots, char *host)
{
	int failures = 0;
	unsigned submitted = 0;

	for (unsigned i = 0; i < ASYNC_TEST_WINDOW; i++) {
		if (async_submit_case(env, &slots[i], i) < 0) {
			failures++;
			break;
		}
		submitted++;
	}

	for (unsigned i = submitted; i-- > 0;)
		if (async_await_verify(env, &slots[i], host) < 0)
			failures++;

	return failures;
}

static int
run_async_read_test(struct async_read_env *env)
{
	struct async_read_slot slots[ASYNC_TEST_WINDOW];
	int failures = 0;

	char *host = malloc(FILE_SIZE);
	if (!host) {
		fprintf(stderr, "  host buffer alloc failed\n");
		return 1;
	}

	memset(slots, 0, sizeof(slots));
	for (size_t i = 0; i < ASYNC_TEST_WINDOW; i++) {
		slots[i].buf = env->buf_acquire(FILE_SIZE);
		if (!slots[i].buf) {
			fprintf(stderr, "  slot %zu buffer alloc failed\n", i);
			failures = 1;
			goto out_bufs;
		}
	}

	unsigned submitted = 0;
	unsigned done = 0;

	while (submitted < ASYNC_TEST_WINDOW) {
		if (async_submit_case(env, &slots[submitted], submitted) < 0) {
			failures++;
			goto out_drain;
		}
		submitted++;
	}

	while (done < ASYNC_TEST_OPS) {
		struct async_read_slot *slot = &slots[done % ASYNC_TEST_WINDOW];

		if (async_await_verify(env, slot, host) < 0)
			failures++;
		done++;

		if (submitted < ASYNC_TEST_OPS) {
			if (async_submit_case(env, slot, submitted) < 0) {
				failures++;
				goto out_drain;
			}
			submitted++;
		}
	}

	failures += run_async_reverse_test(env, slots, host);
	goto out_bufs;

out_drain:
	/* Every submitted op must be awaited before buffers go away. */
	while (done < submitted) {
		opends_async_await(&slots[done % ASYNC_TEST_WINDOW].future);
		done++;
	}
out_bufs:
	for (size_t i = 0; i < ASYNC_TEST_WINDOW; i++)
		if (slots[i].buf)
			env->buf_release(slots[i].buf);
	free(host);

	fprintf(stderr, "  %-24s %s\n", "async_read", failures ? "FAIL" : "ok");
	return failures;
}

#endif /* TEST_ASYNC_READ_H_ */
