/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_batch_read.h - Backend-agnostic batch read test.
 *
 * Include from a backend-specific source file that provides main().
 *
 * Submits one batch of concurrent reads at mixed offsets and sizes,
 * including sub-block sizes to cover per-op tail staging, reaps them
 * with a single get_status, and verifies each completion against the
 * in-memory pattern by cookie. A second get_status must deliver
 * nothing: completions are reported exactly once. A second round on the
 * same handle then checks that delivered slots are free again, and that
 * a submit past the free slots returns OPENDS_BATCH_FULL. Before all of
 * that, on handles of their own, a submit with one undefined opcode or
 * mode must fail with OPENDS_INVALID_VALUE and take no slot.
 *
 * Expects the 16-page pattern file written by test_sync_read_prep.
 */
#ifndef TEST_BATCH_READ_H_
#define TEST_BATCH_READ_H_

#include "opends.h"
#include "read_pattern.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct batch_read_env {
	opends_handle_t fh;
	void *(*buf_to_host)(void *dst, const void *src, size_t n);
	void *(*buf_acquire)(size_t size);
	void (*buf_release)(void *buf);
};

static const struct {
	off_t offset;
	size_t size;
} batch_read_cases[] = {
        {0, PAGE}, {PAGE, 2 * PAGE},       {4 * PAGE, 4 * PAGE},
        {0, 700},  {2 * PAGE, PAGE + 300}, {(FILE_PAGES - 1) * PAGE, PAGE},
};

#define BATCH_READ_NCASES                                                      \
	(sizeof(batch_read_cases) / sizeof(batch_read_cases[0]))

static int
batch_verify_event(struct batch_read_env *env, const opends_io_events_t *ev,
                   void **bufs, char *host)
{
	size_t ci = (size_t)(uintptr_t)ev->cookie;

	if (ci >= BATCH_READ_NCASES) {
		fprintf(stderr, "  bad cookie %zu\n", ci);
		return -1;
	}

	off_t offset = batch_read_cases[ci].offset;
	size_t size = batch_read_cases[ci].size;

	if (ev->status != OPENDS_COMPLETE || ev->ret != size) {
		fprintf(stderr,
		        "  case %zu (off=%ld, size=%zu): status=%d "
		        "ret=%zu\n",
		        ci, (long)offset, size, ev->status, ev->ret);
		return -1;
	}

	env->buf_to_host(host, bufs[ci], size);
	for (size_t b = 0; b < size; b++) {
		if ((unsigned char)host[b] != pattern_byte(offset + (off_t)b)) {
			fprintf(stderr,
			        "  mismatch at byte %zu (off=%ld, size=%zu)\n",
			        b, (long)offset, size);
			return -1;
		}
	}
	return 0;
}

/* Reap one round of BATCH_READ_NCASES completions and verify each once. */
static int
batch_reap_round(struct batch_read_env *env, opends_batch_handle_t batch,
                 void **bufs, char *host, const char *tag)
{
	opends_io_events_t events[BATCH_READ_NCASES + 2];
	bool seen[BATCH_READ_NCASES];
	unsigned nr = BATCH_READ_NCASES + 2;
	int failures = 0;

	memset(seen, 0, sizeof(seen));
	opends_error_t err = opends_batch_get_status(batch, BATCH_READ_NCASES,
	                                             &nr, events, NULL);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  batch_get_status(%s): %s\n", tag,
		        opends_op_status_error(err.err));
		return 1;
	}

	if (nr != BATCH_READ_NCASES) {
		fprintf(stderr, "  events(%s): %u, expected %zu\n", tag, nr,
		        BATCH_READ_NCASES);
		failures++;
	}

	for (unsigned i = 0; i < nr; i++) {
		size_t ci = (size_t)(uintptr_t)events[i].cookie;

		if (ci < BATCH_READ_NCASES && seen[ci]) {
			fprintf(stderr, "  duplicate event for case %zu\n", ci);
			failures++;
			continue;
		}
		if (batch_verify_event(env, &events[i], bufs, host) < 0)
			failures++;
		else
			seen[ci] = true;
	}

	/* Completions are delivered once. A second poll must be empty. */
	nr = BATCH_READ_NCASES + 2;
	err = opends_batch_get_status(batch, 0, &nr, events, NULL);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  batch_get_status(%s repoll): %s\n", tag,
		        opends_op_status_error(err.err));
		failures++;
	} else if (nr != 0) {
		fprintf(stderr, "  repoll events(%s): %u, expected 0\n", tag,
		        nr);
		failures++;
	}

	return failures;
}

/* Reap the two valid reads that follow a rejected submit in
 * batch_invalid_round: cases 0 and 1, each delivered once. */
static int
batch_reap_pair(struct batch_read_env *env, opends_batch_handle_t batch,
                void **bufs, char *host, const char *what)
{
	opends_io_events_t events[2];
	bool seen[2] = {false, false};
	unsigned nr = 2;
	int failures = 0;

	opends_error_t err =
	        opends_batch_get_status(batch, 2, &nr, events, NULL);
	if (err.err != OPENDS_SUCCESS || nr != 2) {
		fprintf(stderr,
		        "  batch_get_status(after %s): %s, %u events, "
		        "expected 2\n",
		        what, opends_op_status_error(err.err), nr);
		return 1;
	}

	for (unsigned k = 0; k < nr; k++) {
		size_t ci = (size_t)(uintptr_t)events[k].cookie;

		if (batch_verify_event(env, &events[k], bufs, host) < 0)
			failures++;
		else if (ci < 2)
			seen[ci] = true;
	}
	if (!seen[0] || !seen[1]) {
		fprintf(stderr,
		        "  batch_get_status(after %s): cases 0 and 1 not "
		        "both delivered\n",
		        what);
		failures++;
	}
	return failures;
}

/* Submit a valid read next to an operation whose opcode or mode is not
 * one opends.h defines, on a handle with two slots. The submit must fail
 * with OPENDS_INVALID_VALUE and start neither: both slots stay free, so
 * the two valid reads fit right after and complete once each. Without
 * the check the bad operation may run as a write. It points past EOF,
 * where every backend writes through the file descriptor, and the
 * read-only handle refuses it, so a regression cannot change the pattern
 * file. */
static int
batch_invalid_round(struct batch_read_env *env,
                    const opends_io_params_t *params, void **bufs, char *host)
{
	static const struct {
		int mode;
		int opcode;
		const char *what;
	} bad[] = {
	        {OPENDS_BATCH, 7, "opcode 7"},
	        {OPENDS_BATCH, -1, "opcode -1"},
	        {0, OPENDS_READ, "mode 0"},
	};
	int failures = 0;

	for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
		opends_io_params_t p[2] = {params[0], params[1]};
		opends_batch_handle_t batch;

		opends_error_t err = opends_batch_setup(&batch, 2);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  batch_setup(%s): %s\n", bad[i].what,
			        opends_op_status_error(err.err));
			failures++;
			continue;
		}

		p[1].mode = (opends_batch_mode_t)bad[i].mode;
		p[1].opcode = (opends_opcode_t)bad[i].opcode;
		p[1].u.batch.file_offset = FILE_SIZE;
		err = opends_batch_submit(batch, 2, p, 0);
		if (err.err != OPENDS_INVALID_VALUE) {
			fprintf(stderr, "  batch_submit(%s): %s, expected %s\n",
			        bad[i].what, opends_op_status_error(err.err),
			        opends_op_status_error(OPENDS_INVALID_VALUE));
			failures++;
		}

		p[1] = params[1];
		err = opends_batch_submit(batch, 2, p, 0);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  batch_submit(after %s): %s\n",
			        bad[i].what, opends_op_status_error(err.err));
			failures++;
		} else {
			failures += batch_reap_pair(env, batch, bufs, host,
			                            bad[i].what);
		}
		opends_batch_destroy(batch);
	}
	return failures;
}

static int
run_batch_read_test(struct batch_read_env *env)
{
	void *bufs[BATCH_READ_NCASES];
	opends_io_params_t params[BATCH_READ_NCASES];
	opends_batch_handle_t batch = NULL;
	int failures = 0;

	char *host = malloc(FILE_SIZE);
	if (!host) {
		fprintf(stderr, "  host buffer alloc failed\n");
		return 1;
	}

	memset(bufs, 0, sizeof(bufs));
	for (size_t i = 0; i < BATCH_READ_NCASES; i++) {
		bufs[i] = env->buf_acquire(FILE_SIZE);
		if (!bufs[i]) {
			fprintf(stderr, "  buffer %zu alloc failed\n", i);
			failures = 1;
			goto out;
		}
		params[i] = (opends_io_params_t){
		        .mode = OPENDS_BATCH,
		        .u.batch = {bufs[i], batch_read_cases[i].offset, 0,
		                    batch_read_cases[i].size},
		        .fh = env->fh,
		        .opcode = OPENDS_READ,
		        .cookie = (void *)(uintptr_t)i,
		};
	}

	opends_error_t err = opends_batch_setup(&batch, BATCH_READ_NCASES + 2);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  batch_setup: %s\n",
		        opends_op_status_error(err.err));
		failures = 1;
		goto out;
	}

	failures += batch_invalid_round(env, params, bufs, host);
	if (failures)
		goto out;

	err = opends_batch_submit(batch, BATCH_READ_NCASES, params, 0);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  batch_submit: %s\n",
		        opends_op_status_error(err.err));
		failures = 1;
		goto out;
	}

	failures += batch_reap_round(env, batch, bufs, host, "round 1");
	if (failures)
		goto out;

	/* The handle holds NCASES + 2 slots and NCASES were used. A full
	 * second round only fits if delivered slots are free again. */
	err = opends_batch_submit(batch, BATCH_READ_NCASES, params, 0);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  batch_submit(round 2): %s\n",
		        opends_op_status_error(err.err));
		failures = 1;
		goto out;
	}

	/* NCASES in flight leave two free slots, so three must not fit. */
	err = opends_batch_submit(batch, 3, params, 0);
	if (err.err != OPENDS_BATCH_FULL) {
		fprintf(stderr, "  batch_submit(overfill): %s, expected %s\n",
		        opends_op_status_error(err.err),
		        opends_op_status_error(OPENDS_BATCH_FULL));
		failures++;
	}

	failures += batch_reap_round(env, batch, bufs, host, "round 2");

out:
	if (batch)
		opends_batch_destroy(batch);
	for (size_t i = 0; i < BATCH_READ_NCASES; i++)
		if (bufs[i])
			env->buf_release(bufs[i]);
	free(host);

	fprintf(stderr, "  %-24s %s\n", "batch_read", failures ? "FAIL" : "ok");
	return failures;
}

#endif /* TEST_BATCH_READ_H_ */
