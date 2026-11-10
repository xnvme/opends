/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_aisio_posix.c - the aisio backend's path through the mounted file
 * system: writes, staged out of GPU memory into pinned host buffers and
 * pwritten through the fd. The copies are queued, on the user's stream
 * ahead of its gate for a stream write and on the default stream for the
 * rest, and an I/O thread only polls for them. An I/O thread never makes
 * an accelerator call that waits: a stream can be waiting for it.
 */

#define _GNU_SOURCE

#include "ds_aisio.h"

struct write_buf *
ds_write_buf_take(struct driver *d)
{
	struct write_buf *s;

	pthread_mutex_lock(&d->write_buf_lock);
	s = d->write_bufs_free;
	if (s) {
		d->write_bufs_free = s->next;
	}
	pthread_mutex_unlock(&d->write_buf_lock);
	return s;
}

void
ds_write_buf_put(struct driver *d, struct write_buf *s)
{
	pthread_mutex_lock(&d->write_buf_lock);
	s->next = d->write_bufs_free;
	d->write_bufs_free = s;
	pthread_mutex_unlock(&d->write_buf_lock);
}

/* A free buffer, or a new one while the pool is under its cap. Pinning is an
 * accelerator call, so only a submitting thread comes here. */
struct write_buf *
ds_write_buf_take_or_grow(struct driver *d)
{
	size_t nbytes = d->cfg.write_buf_nbytes;
	struct write_buf *s = ds_write_buf_take(d);
	int err;

	if (s) {
		return s;
	}
	pthread_mutex_lock(&d->write_buf_lock);
	if (d->write_buf_count >= d->cfg.write_bufs) {
		pthread_mutex_unlock(&d->write_buf_lock);
		return NULL;
	}
	d->write_buf_count++;
	pthread_mutex_unlock(&d->write_buf_lock);

	s = calloc(1, sizeof(*s));
	if (s) {
		err = posix_memalign(&s->host, OPENDS_DIRECT_ALIGN, nbytes);
		if (err != 0) {
			free(s);
			s = NULL;
		}
	}
	if (s) {
		err = ds_accel->host_pin(s->host, nbytes, &s->dptr);
		if (err != 0) {
			free(s->host);
			free(s);
			s = NULL;
		}
	}
	pthread_mutex_lock(&d->write_buf_lock);
	if (s) {
		s->all_next = d->write_bufs_all;
		d->write_bufs_all = s;
	} else {
		d->write_buf_count--;
	}
	pthread_mutex_unlock(&d->write_buf_lock);
	return s;
}

void
ds_write_buf_teardown(struct driver *d)
{
	struct write_buf *s = d->write_bufs_all;
	struct write_buf *next;

	while (s) {
		next = s->all_next;
		ds_accel->host_unpin(s->host);
		free(s->host);
		free(s);
		s = next;
	}
	d->write_bufs_all = NULL;
	d->write_bufs_free = NULL;
	d->write_buf_count = 0;
}

void
ds_write_init(struct file_op_write *wr, const uint8_t *src, size_t size,
              off_t file_offset, off_t buf_offset)
{
	wr->buf = NULL;
	wr->deferred = false;
	wr->src = src;
	wr->size = size;
	wr->file_offset = file_offset;
	wr->buf_offset = buf_offset;
	wr->start = 0;
	wr->span = 0;
	wr->head = 0;
	wr->tail = 0;
	wr->trunc_to = -1;
	wr->rounds = 0;
	wr->round = 0;
	wr->granted = 0;
	wr->copies = 0;
	wr->enqueued = 0;
}

/* Rounds that have landed: for the resident kernel, the least of its
 * blocks. */
static uint32_t
write_landed(const struct file_op_write *wr)
{
	uint32_t n = __atomic_load_n(&wr->landed[0], __ATOMIC_ACQUIRE);

	if (wr->deferred) {
		for (int b = 1; b < DS_STAGE_BLOCKS; b++) {
			uint32_t v = __atomic_load_n(&wr->landed[b],
			                             __ATOMIC_ACQUIRE);

			if ((int32_t)(v - n) < 0) {
				n = v;
			}
		}
	}
	return n;
}

void
ds_write_geometry(struct driver *d, struct file_op_write *wr)
{
	size_t nbytes = d->cfg.write_buf_nbytes;
	struct opends_direct_geom g;

	opends_direct_span(wr->file_offset, wr->size, &g);
	wr->start = g.start;
	wr->span = g.span;
	wr->head = g.head;
	wr->tail = g.tail;
	wr->rounds = (uint32_t)((g.span + nbytes - 1) / nbytes);
}

/* Round k of a write: [rs, rs + rl) of the span goes through the buffer, and
 * its data is the n bytes of the span from ds. */
static void
write_round(const struct file_op_write *wr, size_t nbytes, uint32_t k,
            size_t *rs, size_t *rl, size_t *ds, size_t *n)
{
	size_t re = (size_t)k * nbytes + nbytes;
	size_t de = wr->head + wr->size;

	*rs = (size_t)k * nbytes;
	if (re > wr->span) {
		re = wr->span;
	}
	*rl = re - *rs;
	*ds = *rs > wr->head ? *rs : wr->head;
	*n = (re < de ? re : de) - *ds;
}

/* Aligned-only mode: a round of a stream write has landed. Hold the stream
 * for the thread's turn before the next round, or at the gate after the
 * last one. */
static void
stage_round_cb(void *arg)
{
	struct file_op *op = arg;
	struct file_op_write *wr = &op->wr;
	struct opends_stream *s = op->u.stream.opends_stream;
	uint32_t seq = op->u.stream.seq;
	uint32_t k = __atomic_add_fetch(wr->landed, 1, __ATOMIC_ACQ_REL);

	if (k < wr->rounds) {
		while ((int32_t)(__atomic_load_n(wr->turn, __ATOMIC_ACQUIRE) -
		                 k) < 0)
			;
		return;
	}
	while ((int32_t)(__atomic_load_n(s->gate, __ATOMIC_ACQUIRE) -
	                 (2 * seq + 1)) < 0)
		;
}

/* Aligned-only mode: a round of an async write has landed. */
static void
stage_mark_cb(void *arg)
{
	struct file_op *op = arg;

	__atomic_add_fetch(op->wr.landed, 1, __ATOMIC_ACQ_REL);
}

/* Aligned-only mode, deferred stream write: publish the arrival and hold
 * the stream until the thread has filled the first round. */
static void
claim_turn_cb(void *arg)
{
	struct file_op *op = arg;
	struct opends_stream *s = op->u.stream.opends_stream;

	__atomic_store_n(s->gate, 2 * op->u.stream.seq, __ATOMIC_RELEASE);
	while ((int32_t)(__atomic_load_n(op->wr.turn, __ATOMIC_ACQUIRE) - 1) <
	       0)
		;
}

/* Aligned-only mode, deferred stream write: hold the stream at the gate
 * after the kernel. */
static void
park_release_cb(void *arg)
{
	struct file_op *op = arg;
	struct opends_stream *s = op->u.stream.opends_stream;
	uint32_t seq = op->u.stream.seq;

	while ((int32_t)(__atomic_load_n(s->gate, __ATOMIC_ACQUIRE) -
	                 (2 * seq + 1)) < 0)
		;
}

/* Let a stream op copy round k. The resident kernel reads the round from
 * the descriptor once turn reaches k + 1; the copies of a fixed stream
 * wait for turn to reach k, the first one for nothing. */
static void
write_grant(struct driver *d, struct file_op *op, uint32_t k)
{
	struct file_op_write *wr = &op->wr;
	struct ds_stage_copy *desc = wr->desc;
	size_t rs, rl, ds, n;

	if (!wr->deferred) {
		__atomic_store_n(wr->turn, k, __ATOMIC_RELEASE);
		return;
	}
	write_round(wr, d->cfg.write_buf_nbytes, k, &rs, &rl, &ds, &n);
	desc->src = (uint64_t)(uintptr_t)(wr->src + (ds - wr->head));
	desc->dst = wr->buf->dptr + (ds - rs);
	desc->n_bytes = n;
	desc->round = k + 1;
	desc->last = k + 1 == wr->rounds;
	wr->granted = k + 1;
	__atomic_store_n(wr->turn, k + 1, __ATOMIC_RELEASE);
}

/* The commands of a deferred stream write: the arrival on the gate, a hold
 * until the thread has read the parameters and filled the first round, the
 * resident kernel, and the gate. */
int
ds_stream_write_enqueue_deferred(struct driver *d, struct file_op *op,
                                 ds_accel_stream_t cus)
{
	struct file_op_write *wr = &op->wr;
	struct opends_stream *s = op->u.stream.opends_stream;
	uint32_t seq = op->u.stream.seq;
	int rc;

	if (d->cfg.assume_aligned_only) {
		rc = ds_accel->launch_host_func(cus, claim_turn_cb, op);
	} else {
		rc = ds_accel->stream_write_value32(cus, s->gate_dptr, 2 * seq);
		if (rc == 0) {
			rc = ds_accel->stream_wait_value32_geq(
			        cus, wr->turn_dptr, 1);
		}
	}
	if (rc != 0) {
		return rc;
	}
	rc = ds_accel->stage_stream(wr->desc_dptr, wr->turn_dptr,
	                            wr->landed_dptr, cus);
	if (rc != 0) {
		return rc;
	}
	/* the kernel is live: it lands every round it is granted */
	wr->copies = 1;
	wr->enqueued = 1;
	if (d->cfg.assume_aligned_only) {
		return ds_accel->launch_host_func(cus, park_release_cb, op);
	}
	return ds_accel->stream_wait_value32_geq(cus, s->gate_dptr,
	                                         2 * seq + 1);
}

/* Queue round k of a stream write on the user's stream: the thread's turn
 * (not before the first round), the copy, and the landed signal. */
static int
stream_write_enqueue_round(struct driver *d, struct file_op *op,
                           ds_accel_stream_t cus, uint32_t k)
{
	struct file_op_write *wr = &op->wr;
	size_t rs, rl, ds, n;
	int rc;

	write_round(wr, d->cfg.write_buf_nbytes, k, &rs, &rl, &ds, &n);
	if (k > 0 && !d->cfg.assume_aligned_only) {
		rc = ds_accel->stream_wait_value32_geq(cus, wr->turn_dptr, k);
		if (rc != 0) {
			return rc;
		}
	}
	rc = ds_accel->copy_from_dev_async((uint8_t *)wr->buf->host + (ds - rs),
	                                   wr->src + (ds - wr->head), n, cus);
	if (rc != 0) {
		return rc;
	}
	wr->copies = k + 1;
	if (d->cfg.assume_aligned_only) {
		rc = ds_accel->launch_host_func(cus, stage_round_cb, op);
	} else {
		rc = ds_accel->stream_write_value32(cus, wr->landed_dptr,
		                                    k + 1);
	}
	if (rc != 0) {
		return rc;
	}
	wr->enqueued = k + 1;
	return 0;
}

/* Every round of a stream write, then the gate. The copies run in stream
 * order ahead of the gate, so the stream has reached the write when the
 * first round lands, and the parameters are the values of the call. */
int
ds_stream_write_enqueue(struct driver *d, struct file_op *op,
                        ds_accel_stream_t cus)
{
	struct file_op_write *wr = &op->wr;
	struct opends_stream *s = op->u.stream.opends_stream;
	int rc;

	for (uint32_t k = 0; k < wr->rounds; k++) {
		rc = stream_write_enqueue_round(d, op, cus, k);
		if (rc != 0) {
			return rc;
		}
	}
	if (d->cfg.assume_aligned_only) {
		/* the last round's callback holds the stream at the gate */
		return 0;
	}
	return ds_accel->stream_wait_value32_geq(cus, s->gate_dptr,
	                                         2 * op->u.stream.seq + 1);
}

/* Queue round k of an async write on the default stream. The copy can wait
 * there for streams parked at other gates; this thread does not. */
static void
async_write_enqueue_round(struct io_worker *w, struct file_op *op, uint32_t k)
{
	struct driver *d = w->drv;
	struct file_op_write *wr = &op->wr;
	size_t rs, rl, ds, n;
	int rc;

	write_round(wr, d->cfg.write_buf_nbytes, k, &rs, &rl, &ds, &n);
	rc = ds_accel->copy_from_dev_async((uint8_t *)wr->buf->host + (ds - rs),
	                                   wr->src + (ds - wr->head), n, NULL);
	if (rc != 0) {
		op->err = OPENDS_DEVICE_DRIVER_ERROR;
		return;
	}
	wr->copies = k + 1;
	if (d->cfg.assume_aligned_only) {
		rc = ds_accel->launch_host_func(NULL, stage_mark_cb, op);
	} else {
		rc = ds_accel->stream_write_value32(NULL, wr->landed_dptr,
		                                    k + 1);
	}
	if (rc != 0) {
		op->err = OPENDS_DEVICE_DRIVER_ERROR;
		return;
	}
	wr->enqueued = k + 1;
}

static void
write_pwrite_round(struct io_worker *w, struct file_op *op)
{
	struct file_op_write *wr = &op->wr;
	size_t head = wr->round == 0 ? wr->head : 0;
	size_t tail = wr->round + 1 == wr->rounds ? wr->tail : 0;
	int fd = op->h->fd;
	size_t rs, rl, ds, n;
	int err;

	write_round(wr, w->drv->cfg.write_buf_nbytes, wr->round, &rs, &rl, &ds,
	            &n);
	if (rl == 0) {
		return;
	}
	err = opends_direct_fill_edges(fd, op->h->oflags, wr->buf->host,
	                               wr->start + (off_t)rs, rl, head, tail,
	                               w->edge_buf);
	if (err == 0) {
		err = opends_direct_pwrite_span(fd, wr->buf->host, rl,
		                                wr->start + (off_t)rs);
	}
	if (err == 0 && tail && wr->trunc_to >= 0) {
		err = ftruncate(fd, wr->trunc_to);
		if (err < 0) {
			err = -errno;
		}
	}
	if (err != 0) {
		op->err = err == -EINVAL ? OPENDS_INVALID_VALUE
		                         : OPENDS_DEVICE_DRIVER_ERROR;
	}
}

/* No more rounds are written. The queued rounds of a fixed stream write
 * still run into the buffer, so the stream reaches its gate, and it is
 * returned once every queued copy has landed. A live resident kernel gets
 * an empty last round, so it exits. */
static void
write_abort(struct file_op *op)
{
	struct file_op_write *wr = &op->wr;

	if (wr->deferred) {
		if (wr->copies) {
			/* an empty last round: the kernel copies nothing and
			 * exits */
			struct ds_stage_copy *desc = wr->desc;

			wr->rounds = wr->granted + 1;
			wr->enqueued = wr->rounds;
			desc->src = 0;
			desc->dst = wr->buf->dptr;
			desc->n_bytes = 0;
			desc->round = wr->rounds;
			desc->last = 1;
			wr->granted = wr->rounds;
			__atomic_store_n(wr->turn, wr->rounds,
			                 __ATOMIC_RELEASE);
		}
		wr->round = wr->rounds;
		return;
	}
	wr->round = wr->rounds;
	if (op->mode == FILE_OP_STREAM) {
		__atomic_store_n(wr->turn, wr->rounds, __ATOMIC_RELEASE);
	}
}

static void
write_complete(struct driver *d, struct file_op *op)
{
	struct file_op_write *wr = &op->wr;
	ssize_t n = op->err ? -(ssize_t)op->err : (ssize_t)wr->size;

	(void)d;
	if (wr->buf) {
		/* a copy with no signal behind it may still land: keep the
		 * buffer out of the pool */
		if (wr->deferred || wr->copies == wr->enqueued) {
			ds_write_buf_put(d, wr->buf);
		}
		wr->buf = NULL;
	}
	for (int b = 0; b < DS_STAGE_BLOCKS; b++) {
		wr->landed[b] = 0;
	}
	*wr->turn = 0;
	if (op->mode == FILE_OP_STREAM) {
		*op->u.stream.bytes_read_p = n;
		release_gate(op->u.stream.opends_stream, op->u.stream.seq);
	} else {
		ds_aisio_future_complete(op->u.async.future, n);
	}
	__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
}

/* Service a write in flight: pwrite each round that has landed, let the
 * next one in, and complete after the last. Never waits. */
void
ds_write_poll(struct io_worker *w, struct file_op *op)
{
	struct file_op_write *wr = &op->wr;
	uint32_t landed = write_landed(wr);

	while (!op->err && wr->round < wr->rounds && landed > wr->round) {
		write_pwrite_round(w, op);
		wr->round++;
		if (op->err) {
			write_abort(op);
		} else if (wr->round < wr->rounds) {
			if (op->mode == FILE_OP_STREAM) {
				write_grant(w->drv, op, wr->round);
			} else {
				async_write_enqueue_round(w, op, wr->round);
				if (op->err) {
					write_abort(op);
				}
			}
		}
		landed = write_landed(wr);
	}
	if (wr->round < wr->rounds || landed < wr->enqueued) {
		return;
	}
	write_complete(w->drv, op);
}

/* Start a PENDING write. Returns true when it has to stay pending: an async
 * write with no free buffer. */
bool
ds_dispatch_write(struct io_worker *w, struct file_op *op)
{
	struct driver *d = w->drv;
	struct file_op_write *wr = &op->wr;
	int err;

	if (op->mode == FILE_OP_STREAM && wr->deferred) {
		/* the stream has arrived: the parameters are read now, and the
		 * resident kernel gets at least an empty round, so it exits */
		wr->size = *op->u.stream.size_p;
		wr->file_offset = *op->u.stream.file_offset_p;
		wr->buf_offset = *op->u.stream.buf_offset_p;
		wr->src = (const uint8_t *)op->buf_base + wr->buf_offset;
		if (!op->err && wr->size &&
		    !ds_aisio_range_registered(d, wr->src, wr->size)) {
			op->err = OPENDS_MEMORY_NOT_REGISTERED;
		}
		if (!op->err) {
			ds_write_geometry(d, wr);
			if (wr->rounds == 0) {
				wr->rounds = 1;
			}
			if (wr->copies) {
				wr->enqueued = wr->rounds;
			}
		}
	} else if (op->mode == FILE_OP_STREAM) {
		/* the queued copies used the values of the call */
		if (!op->err &&
		    (*op->u.stream.size_p != wr->size ||
		     *op->u.stream.file_offset_p != wr->file_offset ||
		     *op->u.stream.buf_offset_p != wr->buf_offset)) {
			op->err = OPENDS_INVALID_VALUE;
		}
	} else if (wr->size) {
		if (!ds_aisio_range_registered(d, wr->src, wr->size)) {
			op->err = OPENDS_MEMORY_NOT_REGISTERED;
		} else if (!wr->buf) {
			wr->buf = ds_write_buf_take(d);
			if (!wr->buf) {
				return true;
			}
		}
		if (!op->err) {
			ds_write_geometry(d, wr);
		}
	}
	if (!op->err && wr->tail) {
		err = opends_direct_trunc_to(op->h->fd, wr->start, wr->span,
		                             wr->file_offset + (off_t)wr->size,
		                             &wr->trunc_to);
		if (err != 0) {
			op->err = OPENDS_DEVICE_DRIVER_ERROR;
		}
	}
	__atomic_store_n(&op->state, FILE_OP_IN_FLIGHT, __ATOMIC_RELEASE);
	if (op->err) {
		write_abort(op);
	} else if (op->mode == FILE_OP_STREAM && wr->deferred) {
		if (wr->copies) {
			write_grant(d, op, 0);
		}
	} else if (op->mode == FILE_OP_ASYNC && wr->size) {
		async_write_enqueue_round(w, op, 0);
		if (op->err) {
			write_abort(op);
		}
	}
	ds_write_poll(w, op);
	return false;
}

/* Returns 0 on success; on failure, the dev_err to report (vendor code or
 * -1). */
/* The turn and landed words of worker idx's op slots, behind the stream
 * gates and the kernel read words, and the stage descriptors. */
int
ds_write_words_setup(struct driver *d, struct io_worker *w, int idx)
{
	uint32_t *words = (uint32_t *)d->stream_words_host;
	size_t base = MAX_STREAMS + (size_t)d->cfg.n_io_threads +
	              (size_t)idx * FILE_OP_QUEUE_SIZE * (1 + DS_STAGE_BLOCKS);
	size_t descs_bytes = FILE_OP_QUEUE_SIZE * sizeof(struct ds_stage_copy);
	void *host = NULL;
	int rc;

	rc = ds_accel->host_alloc_mapped(descs_bytes, &host, &w->descs_dptr);
	if (rc != 0) {
		return -1;
	}
	memset(host, 0, descs_bytes);
	w->descs = host;

	for (uint32_t s = 0; s < FILE_OP_QUEUE_SIZE; s++) {
		struct file_op_write *wr = &w->file_op_queue[s].wr;
		size_t i = base + (size_t)s * (1 + DS_STAGE_BLOCKS);

		wr->turn = &words[i];
		wr->landed = &words[i + 1];
		wr->turn_dptr = d->stream_words_dptr + i * sizeof(uint32_t);
		wr->landed_dptr =
		        d->stream_words_dptr + (i + 1) * sizeof(uint32_t);
		wr->desc = &w->descs[s];
		wr->desc_dptr = w->descs_dptr +
		                (size_t)s * sizeof(struct ds_stage_copy);
	}
	return 0;
}
