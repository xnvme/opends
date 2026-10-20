/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Batch I/O implemented on the async API. A batch is a set of
 * independent async operations whose futures live in the batch object,
 * so any backend that provides opends_async_read/write/await gets the
 * batch family from this file. Backends with a native batch engine
 * (cufile) provide their own and do not compile it.
 *
 * Operations in a batch are unordered. opends_batch_cancel cannot stop
 * I/O the backend already accepted: it awaits outstanding operations
 * and reports their undelivered completions as canceled. A batch is
 * not thread-safe; guard shared batches externally.
 */
#define _GNU_SOURCE

#include "opends_internal.h"

#include <sched.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

struct batch_entry {
	opends_async_future_t future;
	void *cookie;
	bool failed;
	bool canceled;
	bool delivered;
};

struct async_batch {
	struct batch_entry *entries;
	unsigned used;
	unsigned inflight;
	unsigned capacity;
};

static bool
entry_done(struct batch_entry *e)
{
	return e->failed || __atomic_load_n(&e->future.done, __ATOMIC_ACQUIRE);
}

/* A slot frees once its completion is delivered. */
static struct batch_entry *
entry_take(struct async_batch *b)
{
	for (unsigned i = 0; i < b->used; i++)
		if (b->entries[i].delivered) {
			memset(&b->entries[i], 0, sizeof(b->entries[i]));
			return &b->entries[i];
		}
	return &b->entries[b->used++];
}

opends_error_t
opends_batch_setup(opends_batch_handle_t *batch_idp, unsigned nr)
{
	if (!batch_idp || nr == 0)
		return opends_err(OPENDS_INVALID_VALUE);

	struct async_batch *b = calloc(1, sizeof(*b));
	if (!b)
		return opends_err(OPENDS_INTERNAL_ERROR);

	b->entries = calloc(nr, sizeof(*b->entries));
	if (!b->entries) {
		free(b);
		return opends_err(OPENDS_INTERNAL_ERROR);
	}

	b->capacity = nr;
	*batch_idp = b;
	return opends_ok();
}

opends_error_t
opends_batch_submit(opends_batch_handle_t batch_idp, unsigned nr,
                    opends_io_params_t *iocbp, unsigned int flags)
{
	(void)flags;
	struct async_batch *b = batch_idp;

	if (!b || !iocbp)
		return opends_err(OPENDS_INVALID_VALUE);
	if (nr > b->capacity - b->inflight)
		return opends_err(OPENDS_BATCH_FULL);

	for (unsigned i = 0; i < nr; i++) {
		opends_io_params_t *p = &iocbp[i];
		struct batch_entry *e = entry_take(b);
		opends_error_t err;

		e->cookie = p->cookie;

		if (p->opcode == OPENDS_READ)
			err = opends_async_read(
			        p->fh, p->u.batch.dev_ptr_base, p->u.batch.size,
			        p->u.batch.file_offset,
			        p->u.batch.dev_ptr_offset, &e->future);
		else
			err = opends_async_write(
			        p->fh, p->u.batch.dev_ptr_base, p->u.batch.size,
			        p->u.batch.file_offset,
			        p->u.batch.dev_ptr_offset, &e->future);
		if (err.err != OPENDS_SUCCESS)
			e->failed = true;
		b->inflight++;
	}

	return opends_ok();
}

static unsigned
count_ready(struct async_batch *b)
{
	unsigned ready = 0;

	for (unsigned i = 0; i < b->used; i++)
		if (!b->entries[i].delivered && entry_done(&b->entries[i]))
			ready++;
	return ready;
}

opends_error_t
opends_batch_get_status(opends_batch_handle_t batch_idp, unsigned min_nr,
                        unsigned *nr, opends_io_events_t *iocbp,
                        struct timespec *timeout)
{
	struct async_batch *b = batch_idp;

	if (!b || !nr || !iocbp)
		return opends_err(OPENDS_INVALID_VALUE);
	if (min_nr > *nr || min_nr > b->inflight)
		return opends_err(OPENDS_INVALID_VALUE);

	struct timespec deadline;
	if (timeout) {
		clock_gettime(CLOCK_MONOTONIC, &deadline);
		deadline.tv_sec += timeout->tv_sec;
		deadline.tv_nsec += timeout->tv_nsec;
		if (deadline.tv_nsec >= 1000000000L) {
			deadline.tv_sec++;
			deadline.tv_nsec -= 1000000000L;
		}
	}

	while (count_ready(b) < min_nr) {
		if (timeout) {
			struct timespec now;

			clock_gettime(CLOCK_MONOTONIC, &now);
			if (now.tv_sec > deadline.tv_sec ||
			    (now.tv_sec == deadline.tv_sec &&
			     now.tv_nsec >= deadline.tv_nsec))
				break;
		}
		sched_yield();
	}

	unsigned out = 0;
	for (unsigned i = 0; i < b->used && out < *nr; i++) {
		struct batch_entry *e = &b->entries[i];

		if (e->delivered || !entry_done(e))
			continue;

		opends_io_events_t *ev = &iocbp[out++];
		ev->cookie = e->cookie;
		if (e->canceled) {
			ev->status = OPENDS_CANCELED;
			ev->ret = 0;
		} else if (e->failed) {
			ev->status = OPENDS_FAILED;
			ev->ret = 0;
		} else {
			ssize_t r = opends_async_await(&e->future);

			ev->status = r < 0 ? OPENDS_FAILED : OPENDS_COMPLETE;
			ev->ret = r < 0 ? 0 : (size_t)r;
		}
		e->delivered = true;
		b->inflight--;
	}

	*nr = out;
	return opends_ok();
}

opends_error_t
opends_batch_cancel(opends_batch_handle_t batch_idp)
{
	struct async_batch *b = batch_idp;

	if (!b)
		return opends_err(OPENDS_INVALID_VALUE);

	for (unsigned i = 0; i < b->used; i++) {
		struct batch_entry *e = &b->entries[i];

		if (e->delivered)
			continue;
		if (!e->failed)
			opends_async_await(&e->future);
		e->canceled = true;
	}

	return opends_ok();
}

void
opends_batch_destroy(opends_batch_handle_t batch_idp)
{
	struct async_batch *b = batch_idp;

	if (!b)
		return;

	for (unsigned i = 0; i < b->used; i++)
		if (!b->entries[i].failed)
			opends_async_await(&b->entries[i].future);

	free(b->entries);
	free(b);
}
