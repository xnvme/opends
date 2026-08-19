/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends_aisio.c - aisio backend for raw-NVMe direct storage.
 *
 * Reads go straight from an NVMe device into GPU memory via xNVMe's upcie-cuda
 * backend (PCIe P2P DMA). HOMI owns the device: it hands out the I/O qpair the
 * reads are driven over. A registered file's extents come from
 * homic_get_extents.
 *
 * Requires: libxnvme and the CUDA toolkit. The NVMe kernel driver must be
 * unbound from the target device before opends_driver_open runs.
 *
 * Writes go through the kernel-mounted filesystem: the source is staged to a
 * host buffer and pwritten via the fd, so XFS over qublk allocates blocks and
 * writes the data; the file's extents are then re-resolved for later P2P reads.
 * Batch paths report OPENDS_ASYNC_NOT_SUPPORTED for now.
 */

#define _GNU_SOURCE

#include "ds_accel.h"
#include "ds_stream_map.h"
#include "ds_bounce_kernel.h"
#include "opends_internal.h"
#include "ds_extent.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <homic.h>
#include <libxnvme.h>

#define ENV_HOMI_SOCKET "OPENDS_HOMI_SOCKET"
#define ENV_HOMI_DEV "OPENDS_HOMI_DEV"
#define ENV_IO_THREADS "OPENDS_AISIO_IO_THREADS"
#define ENV_QUEUE_DEPTH "OPENDS_AISIO_QUEUE_DEPTH"
#define ENV_CPU_MASK "OPENDS_AISIO_CPU_MASK"
#define ENV_ASSUME_ALIGNED_ONLY "OPENDS_AISIO_ASSUME_ALIGNED_ONLY"
#define DEFAULT_HOMI_SOCKET "/run/homi/homi.sock"
#define DEFAULT_IO_THREADS 1
#define MAX_IO_THREADS 15
#define MAX_BUF_ENTRIES 8192
#define DEFAULT_BOUNCE_SIZE (128 * 1024)
#define NVME_MAX_NLB 65536
#define NVME_PRP_PAGE 4096
#define DEFAULT_QUEUE_DEPTH 512
#define MAX_QUEUE_DEPTH 4096
#define MAX_STREAMS 8192
#define STREAM_WORDS_BYTES (MAX_STREAMS * sizeof(uint32_t))
#define FILE_OP_QUEUE_SIZE 1024
#define FILE_OP_QUEUE_MASK (FILE_OP_QUEUE_SIZE - 1)
#define STREAM_MAP_SIZE (2 * MAX_STREAMS)
#define STREAM_MAP_MASK (STREAM_MAP_SIZE - 1)

enum file_op_state {
	FILE_OP_FREE = 0,
	FILE_OP_PENDING = 1,
	FILE_OP_IN_FLIGHT = 2,
};

struct buf_entry {
	const void *base;
	size_t length;
	bool owned; /* true: xnvme_buf_alloc; false: xnvme_mem_map. */
};

struct registered_file {
	int fd;
};

struct opends_stream {
	uint32_t *gate;
	ds_accel_devptr_t gate_dptr;
	uint32_t next_seq;
	void *bounce_buf;
	struct ds_bounce_copy *bounce_desc_host;
	ds_accel_devptr_t bounce_desc_dev;
};

enum file_op_mode {
	FILE_OP_SYNC,
	FILE_OP_ASYNC,
	FILE_OP_BATCH,
};

struct file_op_sync {
	size_t size;
	off_t file_offset;
	off_t buf_offset;
	ssize_t result;
};

struct file_op_async {
	size_t *size_p;
	off_t *file_offset_p;
	off_t *buf_offset_p;
	ssize_t *bytes_read_p;
	struct opends_stream *opends_stream;
	uint32_t seq;
};

struct file_op_batch {
	opends_io_params_t *iocbp;
	unsigned nr;
};

struct file_op {
	enum file_op_mode mode;
	bool is_write;
	enum file_op_state state;
	struct registered_file *h;
	void *buf_base;
	int chunks_remaining;
	int bounces_outstanding;
	size_t bytes_acc;
	int err;
	void *tail_dst;
	size_t tail_nbytes;
	union {
		struct file_op_sync sync;
		struct file_op_async async;
		struct file_op_batch batch;
	} u;
};

struct read_cursor {
	struct xnvme_queue *queue;
	uint64_t cur_slba;
	uint8_t *abs_dst;
	size_t remaining;
};

struct driver;

struct io_worker {
	struct driver *drv;
	struct xnvme_queue *queue;
	void *sync_bounce_buf;
	struct file_op file_op_queue[FILE_OP_QUEUE_SIZE];
	uint32_t queue_head;
	uint32_t queue_tail;
	pthread_t thread;
};

struct driver {
	char dev_uri[64];      ///< NVMe device (BDF) the HOMI daemon owns
	char *attach_descpath; ///< HOMI-served qpair attach descriptor file
	struct xnvme_dev *xdev;
	uint32_t nsid;
	uint32_t lba_size;
	uint32_t lba_shift;
	uint32_t mdts_nbytes;
	struct buf_entry bufs[MAX_BUF_ENTRIES];
	int buf_count;

	bool async_ready;
	ds_accel_ctx_t accel_ctx;

	struct opends_stream streams[MAX_STREAMS];
	int n_streams;
	void *stream_words_host;
	ds_accel_devptr_t stream_words_dptr;

	struct ds_stream_map_entry stream_map[STREAM_MAP_SIZE];

	int n_io_threads;
	uint32_t queue_depth;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	struct io_worker *workers;
	uint32_t rr_next;
	pthread_mutex_t submit_lock;
	pthread_mutex_t reg_lock;
	bool stop;
};

static struct driver *drv;
static long use_count;

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
	return a > b ? a : b;
}

static int
ensure_bounce_buf(struct driver *d, void **bounce_buf_p)
{
	if (*bounce_buf_p)
		return 0;
	*bounce_buf_p = xnvme_buf_alloc(d->xdev, NVME_PRP_PAGE);
	return *bounce_buf_p ? 0 : -ENOMEM;
}

/* Allocate the tail-bounce slot and copy descriptor for a stream. GPU alloc
 * APIs do a device-wide sync, so this must run on a host thread: on the aisio
 * I/O thread it could deadlock waiting on the thread's own gated stream.
 * Returns 0 on success; on failure, the dev_err to report (vendor code or
 * -1), with the fields left zeroed. */
static int
stream_bounce_alloc(struct opends_stream *s, struct xnvme_dev *xdev)
{
	s->bounce_buf = NULL;
	s->bounce_desc_host = NULL;
	s->bounce_desc_dev = 0;

	/* DEVICEMAP'd so the I/O thread fills it via host stores and the
	 * kernel reads it on the GPU. Zero-init keeps a bounce-free op
	 * (n_bytes == 0) kernel-safe. */
	void *desc_host = NULL;
	ds_accel_devptr_t desc_dev = 0;
	int rc = ds_accel->host_alloc_mapped(sizeof(struct ds_bounce_copy),
	                                     &desc_host, &desc_dev);
	if (rc != 0)
		return rc;
	memset(desc_host, 0, sizeof(struct ds_bounce_copy));

	/* One slot suffices: at most one bounce per op, and the per-stream gate
	 * serialises ops. */
	void *buf = xnvme_buf_alloc(xdev, NVME_PRP_PAGE);
	if (!buf) {
		ds_accel->host_free(desc_host);
		return -1;
	}

	s->bounce_desc_host = (struct ds_bounce_copy *)desc_host;
	s->bounce_desc_dev = desc_dev;
	s->bounce_buf = buf;
	return 0;
}

static void
stream_bounce_free(struct opends_stream *s, struct xnvme_dev *xdev)
{
	if (s->bounce_buf) {
		xnvme_buf_free(xdev, s->bounce_buf);
		s->bounce_buf = NULL;
	}
	if (s->bounce_desc_host) {
		ds_accel->host_free(s->bounce_desc_host);
		s->bounce_desc_host = NULL;
	}
	s->bounce_desc_dev = 0;
}

#define ESTALE_RETRIES 6000
#define ESTALE_BACKOFF_US 50000

static int
resolve_extents(int fd, struct ds_extent **out, uint32_t *out_n)
{
	struct homic_extent *hx = NULL;
	uint32_t n = 0;
	int rc = -ESTALE;

	for (int attempt = 0; attempt < ESTALE_RETRIES; attempt++) {
		rc = homic_get_extents(fd, &hx, &n);
		if (rc != -ESTALE)
			break;
		usleep(ESTALE_BACKOFF_US);
	}
	if (rc < 0)
		return rc;

	struct ds_extent *ex = calloc(n ? n : 1, sizeof(*ex));
	if (!ex) {
		free(hx);
		return -ENOMEM;
	}
	for (uint32_t i = 0; i < n; i++) {
		ex[i].file_offset = hx[i].file_offset;
		ex[i].slba = hx[i].slba;
		ex[i].length = hx[i].length;
	}
	free(hx);

	*out = ex;
	*out_n = n;
	return 0;
}

static ssize_t
pwrite_op(struct driver *d, struct registered_file *h, const void *src,
          size_t size, off_t file_offset)
{
	if (size == 0)
		return 0;

	void *bounce = malloc(size);
	if (!bounce)
		return -ENOMEM;

	ssize_t ret = (ssize_t)size;
	if (ds_accel->copy(bounce, src, size) != 0) {
		ret = -EIO;
		goto out;
	}

	size_t done = 0;
	while (done < size) {
		ssize_t w = pwrite(h->fd, (uint8_t *)bounce + done, size - done,
		                   file_offset + (off_t)done);
		if (w < 0) {
			if (errno == EINTR)
				continue;
			ret = -errno;
			goto out;
		}
		if (w == 0) {
			ret = -EIO;
			goto out;
		}
		done += (size_t)w;
	}

	if (fsync(h->fd) < 0) {
		ret = -errno;
		goto out;
	}

	int rrc = homic_mark_dirty(d->dev_uri);
	if (rrc < 0)
		ret = rrc;
out:
	free(bounce);
	return ret;
}

static int
open_device(struct driver *d)
{
	int nqpairs = 1 + d->n_io_threads;
	int rc = homic_attach_qpair(d->dev_uri, nqpairs, &d->attach_descpath);
	if (rc < 0) {
		fprintf(stderr,
		        "aisio open_device: homic_attach_qpair(%s) rc=%d\n",
		        d->dev_uri, rc);
		if (rc == -ENOMEM || rc == -EINVAL)
			fprintf(stderr,
			        "aisio: HOMI refused %d qpairs "
			        "(1 producer + %d IO threads); lower %s\n",
			        nqpairs, d->n_io_threads, ENV_IO_THREADS);
		return rc;
	}

	setenv("XNVME_UPCIE_ATTACH", d->attach_descpath, 1);
	struct xnvme_opts opts = xnvme_opts_default();
	opts.be = ds_accel->xnvme_be;

	d->xdev = xnvme_dev_open(d->dev_uri, &opts);
	unsetenv("XNVME_UPCIE_ATTACH");
	if (!d->xdev)
		return -EIO;

	const struct xnvme_geo *geo = xnvme_dev_get_geo(d->xdev);
	d->nsid = xnvme_dev_get_nsid(d->xdev);
	d->lba_size = geo->lba_nbytes ? geo->lba_nbytes : geo->nbytes;
	d->mdts_nbytes =
	        geo->mdts_nbytes ? geo->mdts_nbytes : DEFAULT_BOUNCE_SIZE;

	if (d->lba_size == 0) {
		fprintf(stderr,
		        "aisio open_device: zero geometry from xnvme_dev_open "
		        "(lba_nbytes=%u nbytes=%u mdts_nbytes=%u); "
		        "controller likely needs a PCI reset before this "
		        "open\n",
		        geo->lba_nbytes, geo->nbytes, geo->mdts_nbytes);
		xnvme_dev_close(d->xdev);
		d->xdev = NULL;
		return -EIO;
	}
	if (d->lba_size & (d->lba_size - 1)) {
		fprintf(stderr,
		        "aisio open_device: lba_size=%u is not a power of 2\n",
		        d->lba_size);
		xnvme_dev_close(d->xdev);
		d->xdev = NULL;
		return -EIO;
	}
	d->lba_shift = (uint32_t)__builtin_ctz(d->lba_size);

	return 0;
}

/* ------------------------------------------------------------------ */
/*  I/O engine                                                        */
/* ------------------------------------------------------------------ */

static void
chunk_cb(struct xnvme_cmd_ctx *ctx, void *opaque)
{
	struct file_op *op = opaque;
	if (xnvme_cmd_ctx_cpl_status(ctx)) {
		op->err = OPENDS_DEVICE_DRIVER_ERROR;
	} else {
		uint32_t lba_size = drv->lba_size;
		op->bytes_acc += ((uint64_t)ctx->cmd.nvm.nlb + 1) * lba_size;
	}
	op->chunks_remaining--;
	xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
}

static int
submit_read_middle(struct driver *d, struct read_cursor *c, struct file_op *op,
                   uint64_t middle_lbas)
{
	uint32_t lba_nbytes = d->lba_size;
	uint64_t max_chunk_lbas = d->mdts_nbytes >> d->lba_shift;
	uint64_t lbas_done = 0;
	while (lbas_done < middle_lbas) {
		uint64_t chunk_lbas =
		        XNVME_MIN_U64(middle_lbas - lbas_done, max_chunk_lbas);
		chunk_lbas = XNVME_MIN_U64(chunk_lbas, NVME_MAX_NLB);

		struct xnvme_cmd_ctx *ctx;
		for (;;) {
			ctx = xnvme_queue_get_cmd_ctx(c->queue);
			if (ctx)
				break;
			xnvme_queue_poke(c->queue, 0);
		}
		xnvme_cmd_ctx_set_cb(ctx, chunk_cb, op);

		uint16_t nlb = (uint16_t)(chunk_lbas - 1);
		uint64_t chunk_slba = c->cur_slba + lbas_done;
		uint8_t *dst_chunk = c->abs_dst + lbas_done * lba_nbytes;

		op->chunks_remaining++;
		int srv = xnvme_nvm_read(ctx, d->nsid, chunk_slba, nlb,
		                         dst_chunk, NULL);
		if (srv == -EBUSY) {
			op->chunks_remaining--;
			xnvme_queue_put_cmd_ctx(c->queue, ctx);
			xnvme_queue_poke(c->queue, 0);
			continue;
		}
		if (srv < 0) {
			op->chunks_remaining--;
			xnvme_queue_put_cmd_ctx(c->queue, ctx);
			op->err = OPENDS_DEVICE_DRIVER_ERROR;
			return -1;
		}

		lbas_done += chunk_lbas;
	}

	c->cur_slba += middle_lbas;
	c->abs_dst += middle_lbas * lba_nbytes;
	c->remaining -= middle_lbas * lba_nbytes;
	return 0;
}

static void
bounce_cb(struct xnvme_cmd_ctx *ctx, void *opaque)
{
	struct file_op *op = opaque;
	if (xnvme_cmd_ctx_cpl_status(ctx) && !op->err)
		op->err = OPENDS_DEVICE_DRIVER_ERROR;
	op->bounces_outstanding--;
	xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
}

static int
submit_read_bounce(struct io_worker *w, struct file_op *op, uint8_t *abs_dst,
                   uint64_t cur_slba, size_t nbytes)
{
	struct driver *d = w->drv;
	struct opends_stream *s = op->u.async.opends_stream;
	void *bounce_src = s->bounce_buf;

	uint64_t nlbas = (nbytes + (d->lba_size - 1)) >> d->lba_shift;
	uint16_t nlb = (uint16_t)(nlbas - 1);

	for (;;) {
		struct xnvme_cmd_ctx *ctx;
		for (;;) {
			ctx = xnvme_queue_get_cmd_ctx(w->queue);
			if (ctx)
				break;
			xnvme_queue_poke(w->queue, 0);
		}
		xnvme_cmd_ctx_set_cb(ctx, bounce_cb, op);

		int srv = xnvme_nvm_read(ctx, d->nsid, cur_slba, nlb,
		                         bounce_src, NULL);
		if (srv == -EBUSY) {
			xnvme_queue_put_cmd_ctx(w->queue, ctx);
			xnvme_queue_poke(w->queue, 0);
			continue;
		}
		if (srv < 0) {
			xnvme_queue_put_cmd_ctx(w->queue, ctx);
			op->err = OPENDS_DEVICE_DRIVER_ERROR;
			return -1;
		}
		break;
	}

	op->tail_dst = abs_dst;
	op->tail_nbytes = nbytes;
	s->bounce_desc_host->dst = (uint64_t)(uintptr_t)op->tail_dst;
	s->bounce_desc_host->src = (uint64_t)(uintptr_t)bounce_src;

	op->bounces_outstanding++;
	op->bytes_acc += nbytes;
	return 0;
}

static int
submit_sync_tail(struct io_worker *w, struct file_op *op, uint8_t *abs_dst,
                 uint64_t cur_slba, size_t nbytes)
{
	struct driver *d = w->drv;

	if (ensure_bounce_buf(d, &w->sync_bounce_buf) < 0) {
		op->err = OPENDS_INTERNAL_ERROR;
		return -1;
	}

	uint64_t nlbas = (nbytes + (d->lba_size - 1)) >> d->lba_shift;
	uint16_t nlb = (uint16_t)(nlbas - 1);

	for (;;) {
		struct xnvme_cmd_ctx *ctx;
		for (;;) {
			ctx = xnvme_queue_get_cmd_ctx(w->queue);
			if (ctx)
				break;
			xnvme_queue_poke(w->queue, 0);
		}
		xnvme_cmd_ctx_set_cb(ctx, bounce_cb, op);

		int srv = xnvme_nvm_read(ctx, d->nsid, cur_slba, nlb,
		                         w->sync_bounce_buf, NULL);
		if (srv == -EBUSY) {
			xnvme_queue_put_cmd_ctx(w->queue, ctx);
			xnvme_queue_poke(w->queue, 0);
			continue;
		}
		if (srv < 0) {
			xnvme_queue_put_cmd_ctx(w->queue, ctx);
			op->err = OPENDS_DEVICE_DRIVER_ERROR;
			return -1;
		}
		break;
	}

	op->tail_dst = abs_dst;
	op->tail_nbytes = nbytes;
	op->bounces_outstanding++;
	op->bytes_acc += nbytes;
	return 0;
}

static void
start_read_op(struct io_worker *w, struct file_op *op)
{
	struct driver *d = w->drv;

	if (d->lba_size == 0) {
		op->err = OPENDS_INTERNAL_ERROR;
		return;
	}
	if ((d->mdts_nbytes >> d->lba_shift) == 0) {
		op->err = OPENDS_INVALID_VALUE;
		return;
	}

	size_t size;
	uint64_t req_start;
	uint8_t *dst_base;
	if (op->mode == FILE_OP_ASYNC) {
		size = *op->u.async.size_p;
		req_start = (uint64_t)*op->u.async.file_offset_p;
		dst_base = (uint8_t *)op->buf_base + *op->u.async.buf_offset_p;
	} else {
		size = op->u.sync.size;
		req_start = (uint64_t)op->u.sync.file_offset;
		dst_base = (uint8_t *)op->buf_base + op->u.sync.buf_offset;
	}
	if (size > UINT64_MAX - req_start) {
		op->err = OPENDS_INVALID_VALUE;
		return;
	}
	uint64_t req_end = req_start + size;

	struct ds_extent *extents;
	uint32_t extent_count;
	int frc = resolve_extents(op->h->fd, &extents, &extent_count);
	if (frc < 0) {
		op->err = OPENDS_FS_SETUP_ERROR;
		return;
	}

	uint32_t lba_shift = d->lba_shift;
	uint32_t lba_mask = d->lba_size - 1;

	for (uint32_t i = 0; i < extent_count; i++) {
		const struct ds_extent *e = &extents[i];
		uint64_t ext_start = e->file_offset;
		uint64_t ext_end = ext_start + e->length;
		if (ext_start >= req_end)
			break;

		uint64_t span_start = max_u64(req_start, ext_start);
		uint64_t span_end = XNVME_MIN_U64(req_end, ext_end);
		if (span_start >= span_end)
			continue;

		uint64_t off_in_ext = span_start - ext_start;
		if (off_in_ext & lba_mask) {
			op->err = OPENDS_INVALID_VALUE;
			goto out;
		}
		size_t buf_off = span_start - req_start;
		uint8_t *abs_dst = dst_base + buf_off;
		uint64_t cur_slba = e->slba + (off_in_ext >> lba_shift);
		size_t remaining = span_end - span_start;

		size_t tail_bytes = remaining & lba_mask;
		uint64_t middle_lbas = (remaining - tail_bytes) >> lba_shift;
		if (middle_lbas) {
			struct read_cursor c = {
			        .queue = w->queue,
			        .cur_slba = cur_slba,
			        .abs_dst = abs_dst,
			        .remaining = remaining,
			};
			if (submit_read_middle(d, &c, op, middle_lbas) < 0)
				goto out;
			cur_slba = c.cur_slba;
			abs_dst = c.abs_dst;
			remaining = c.remaining;
		}

		if (tail_bytes) {
			if (d->assume_aligned_only) {
				op->err = OPENDS_INVALID_VALUE;
				goto out;
			}
			int trc;
			if (op->mode == FILE_OP_ASYNC)
				trc = submit_read_bounce(w, op, abs_dst,
				                         cur_slba, tail_bytes);
			else
				trc = submit_sync_tail(w, op, abs_dst, cur_slba,
				                       tail_bytes);
			if (trc < 0)
				goto out;
		}
	}
out:
	free(extents);
}

static void
release_gate(struct opends_stream *s, uint32_t seq)
{
	__atomic_store_n(s->gate, 2 * seq + 1, __ATOMIC_RELEASE);
}

static void
park_gate_cb(void *arg)
{
	struct file_op *op = arg;
	struct opends_stream *s = op->u.async.opends_stream;
	uint32_t seq = op->u.async.seq;

	__atomic_store_n(s->gate, 2 * seq, __ATOMIC_RELEASE);

	while ((int32_t)(__atomic_load_n(s->gate, __ATOMIC_ACQUIRE) -
	                 (2 * seq + 1)) < 0)
		;
}

static void
complete_read_op(struct io_worker *w, struct file_op *op)
{
	ssize_t n = op->err ? -(ssize_t)op->err : (ssize_t)op->bytes_acc;
	uint32_t tail_bytes = op->err ? 0 : (uint32_t)op->tail_nbytes;

	if (op->mode == FILE_OP_ASYNC) {
		struct opends_stream *s = op->u.async.opends_stream;
		*op->u.async.bytes_read_p = n;
		s->bounce_desc_host->n_bytes = tail_bytes;
		release_gate(s, op->u.async.seq);
		__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
		return;
	}

	if (tail_bytes) {
		int rc = ds_accel->copy(op->tail_dst, w->sync_bounce_buf,
		                        op->tail_nbytes);
		if (rc != 0)
			n = -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR;
	}

	op->u.sync.result = n;
	__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
}

static void
dispatch_write(struct io_worker *w, struct file_op *op)
{
	struct driver *d = w->drv;
	const void *src;
	size_t size;
	off_t file_offset;
	if (op->mode == FILE_OP_ASYNC) {
		src = (const uint8_t *)op->buf_base + *op->u.async.buf_offset_p;
		size = *op->u.async.size_p;
		file_offset = *op->u.async.file_offset_p;
	} else {
		src = (const uint8_t *)op->buf_base + op->u.sync.buf_offset;
		size = op->u.sync.size;
		file_offset = op->u.sync.file_offset;
	}

	ssize_t n = pwrite_op(d, op->h, src, size, file_offset);
	if (n < 0)
		n = (n == -(ssize_t)EINVAL)
		            ? -(ssize_t)OPENDS_INVALID_VALUE
		            : -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR;

	if (op->mode == FILE_OP_ASYNC) {
		*op->u.async.bytes_read_p = n;
		release_gate(op->u.async.opends_stream, op->u.async.seq);
	} else {
		op->u.sync.result = n;
	}
	__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
}

static void
dispatch_pending(struct io_worker *w, struct file_op *op)
{
	if (op->is_write) {
		dispatch_write(w, op);
		return;
	}
	op->chunks_remaining = 0;
	op->bounces_outstanding = 0;
	op->bytes_acc = 0;
	op->err = 0;
	op->tail_nbytes = 0;
	__atomic_store_n(&op->state, FILE_OP_IN_FLIGHT, __ATOMIC_RELEASE);
	start_read_op(w, op);
}

static void
reap_in_flight(struct io_worker *w, struct file_op *op)
{
	if (op->chunks_remaining == 0 && op->bounces_outstanding == 0)
		complete_read_op(w, op);
}

/* Service a PENDING async op: dispatch it once its stream gate has opened,
 * otherwise leave it for a later pass. Returns true while still gated. The gate
 * is a free-running +1 counter (the submitter's arrival publish then this
 * thread's release each tick it by one), so compare with serial/cyclic
 * arithmetic to match the device-side GEQ wait, keeping the sequence
 * wrap-safe. */
static bool
poll_async_pending(struct io_worker *w, struct file_op *op)
{
	uint32_t gate = __atomic_load_n(op->u.async.opends_stream->gate,
	                                __ATOMIC_ACQUIRE);

	if ((int32_t)(gate - 2 * op->u.async.seq) < 0)
		return true;
	dispatch_pending(w, op);
	return false;
}

static void *
io_thread_main(void *arg)
{
	struct io_worker *w = arg;
	struct driver *d = w->drv;
	ds_accel->ctx_set(d->accel_ctx);

	for (;;) {
		bool busy = false;

		uint32_t head =
		        __atomic_load_n(&w->queue_head, __ATOMIC_ACQUIRE);
		for (uint32_t i = w->queue_tail; i != head; i++) {
			struct file_op *op =
			        &w->file_op_queue[i & FILE_OP_QUEUE_MASK];
			switch (__atomic_load_n(&op->state, __ATOMIC_ACQUIRE)) {
			case FILE_OP_PENDING:
				switch (op->mode) {
				case FILE_OP_SYNC:
					dispatch_pending(w, op);
					break;
				case FILE_OP_ASYNC:
					if (poll_async_pending(w, op))
						busy = true;
					break;
				case FILE_OP_BATCH: break;
				}
				break;
			case FILE_OP_IN_FLIGHT: reap_in_flight(w, op); break;
			default: break;
			}
		}

		while (w->queue_tail != head) {
			struct file_op *op =
			        &w->file_op_queue[w->queue_tail &
			                          FILE_OP_QUEUE_MASK];
			if (__atomic_load_n(&op->state, __ATOMIC_ACQUIRE) !=
			    FILE_OP_FREE)
				break;
			__atomic_store_n(&w->queue_tail, w->queue_tail + 1,
			                 __ATOMIC_RELEASE);
		}

		if (w->queue_tail != head)
			busy = true;

		if (__atomic_load_n(&d->stop, __ATOMIC_ACQUIRE) && !busy)
			break;

		if (busy) {
			xnvme_queue_poke(w->queue, 0);
			sched_yield();
		} else {
			struct timespec ts = {0, 100000};
			nanosleep(&ts, NULL);
		}
	}

	return NULL;
}

static int
mask_nth_cpu(uint64_t mask, int n)
{
	for (int cpu = 0; cpu < 64; cpu++)
		if (((mask >> cpu) & 1) && n-- == 0)
			return cpu;
	return -1;
}

/* Returns 0 on success; on failure, the dev_err to report (vendor code or
 * -1). */
static int
async_setup(struct driver *d)
{
	int rc = ds_accel->ctx_get(&d->accel_ctx);
	if (rc != 0)
		return rc;

	/* Mapped so the device-side gate can address the word; the host
	 * callback path uses the host view only. */
	void *host = NULL;
	ds_accel_devptr_t dptr = 0;
	rc = ds_accel->host_alloc_mapped(STREAM_WORDS_BYTES, &host, &dptr);
	if (rc != 0)
		return rc;
	memset(host, 0, STREAM_WORDS_BYTES);
	d->stream_words_host = host;
	d->stream_words_dptr = dptr;

	d->workers = calloc((size_t)d->n_io_threads, sizeof(*d->workers));
	if (!d->workers)
		goto fail_words;

	d->stop = false;
	int started = 0;
	int mask_cpus = __builtin_popcountll(d->cpu_mask);
	for (int i = 0; i < d->n_io_threads; i++) {
		struct io_worker *w = &d->workers[i];
		w->drv = d;
		if (xnvme_queue_init(d->xdev, d->queue_depth, 0, &w->queue) <
		    0) {
			w->queue = NULL;
			goto fail;
		}
		pthread_attr_t attr;
		pthread_attr_t *attrp = NULL;
		if (mask_cpus) {
			cpu_set_t set;
			CPU_ZERO(&set);
			CPU_SET(mask_nth_cpu(d->cpu_mask, i % mask_cpus), &set);
			pthread_attr_init(&attr);
			pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
			attrp = &attr;
		}
		int rc = pthread_create(&w->thread, attrp, io_thread_main, w);
		if (attrp)
			pthread_attr_destroy(&attr);
		if (rc != 0) {
			xnvme_queue_term(w->queue);
			w->queue = NULL;
			goto fail;
		}
		started++;
	}

	d->async_ready = true;
	return 0;

fail:
	__atomic_store_n(&d->stop, true, __ATOMIC_RELEASE);
	for (int i = 0; i < started; i++)
		pthread_join(d->workers[i].thread, NULL);
	for (int i = 0; i < d->n_io_threads; i++) {
		struct io_worker *w = &d->workers[i];
		if (w->sync_bounce_buf)
			xnvme_buf_free(d->xdev, w->sync_bounce_buf);
		if (w->queue)
			xnvme_queue_term(w->queue);
	}
	free(d->workers);
	d->workers = NULL;
fail_words:
	ds_accel->host_free(d->stream_words_host);
	d->stream_words_host = NULL;
	return -1;
}

static void
async_teardown(struct driver *d)
{
	if (!d->async_ready)
		return;

	__atomic_store_n(&d->stop, true, __ATOMIC_RELEASE);
	for (int i = 0; i < d->n_io_threads; i++)
		pthread_join(d->workers[i].thread, NULL);

	for (int i = 0; i < d->n_streams; i++) {
		stream_bounce_free(&d->streams[i], d->xdev);
	}

	for (int i = 0; i < d->n_io_threads; i++) {
		struct io_worker *w = &d->workers[i];
		if (w->sync_bounce_buf) {
			xnvme_buf_free(d->xdev, w->sync_bounce_buf);
			w->sync_bounce_buf = NULL;
		}
		if (w->queue) {
			xnvme_queue_term(w->queue);
			w->queue = NULL;
		}
	}
	free(d->workers);
	d->workers = NULL;

	ds_accel->host_free(d->stream_words_host);
	d->stream_words_host = NULL;

	d->async_ready = false;
}

static struct opends_stream *
opends_stream_get(struct driver *d, ds_accel_stream_t stream)
{
	int idx = ds_stream_map_get(d->stream_map, STREAM_MAP_MASK, stream);
	if (idx < 0)
		return NULL;
	return &d->streams[idx];
}

static int
env_int(const char *name, int def, int lo, int hi, int *out)
{
	const char *v = getenv(name);
	if (!v || !v[0]) {
		*out = def;
		return 0;
	}
	long n = strtol(v, NULL, 10);
	if (n < lo || n > hi) {
		fprintf(stderr, "aisio: %s=%s out of range [%d, %d]\n", name, v,
		        lo, hi);
		return -EINVAL;
	}
	*out = (int)n;
	return 0;
}

static int
read_env_config(struct driver *d)
{
	int n;

	if (env_int(ENV_IO_THREADS, DEFAULT_IO_THREADS, 1, MAX_IO_THREADS, &n) <
	    0)
		return -EINVAL;
	d->n_io_threads = n;

	if (env_int(ENV_QUEUE_DEPTH, DEFAULT_QUEUE_DEPTH, 1, MAX_QUEUE_DEPTH,
	            &n) < 0)
		return -EINVAL;
	d->queue_depth = (uint32_t)n;

	const char *mask = getenv(ENV_CPU_MASK);
	d->cpu_mask = mask && mask[0] ? strtoull(mask, NULL, 0) : 0;

	const char *aligned = getenv(ENV_ASSUME_ALIGNED_ONLY);
	d->assume_aligned_only = aligned && aligned[0] && aligned[0] != '0';

	/* The tail mode picks the async gate mechanism, and the vendor ops it
	 * drives are required only for that mode (see ds_accel.h). A partial
	 * port may leave the other mode's ops NULL; fail open instead of
	 * crashing on the first submission. */
	if (d->assume_aligned_only && !ds_accel->launch_host_func) {
		fprintf(stderr,
		        "aisio: %s=1 needs launch_host_func, which the vendor "
		        "ops table does not provide\n",
		        ENV_ASSUME_ALIGNED_ONLY);
		return -EINVAL;
	}
	if (!d->assume_aligned_only && (!ds_accel->stream_write_value32 ||
	                                !ds_accel->stream_wait_value32_geq)) {
		fprintf(stderr,
		        "aisio: the vendor ops table does not provide the "
		        "stream gate ops; set %s=1 to gate via "
		        "launch_host_func\n",
		        ENV_ASSUME_ALIGNED_ONLY);
		return -EINVAL;
	}

	return 0;
}

static struct io_worker *
route_op(struct driver *d)
{
	return &d->workers[d->rr_next++ % (uint32_t)d->n_io_threads];
}

static struct file_op *
claim_slot_locked(struct driver *d, struct io_worker **wp, uint32_t *headp)
{
	for (;;) {
		struct io_worker *w = route_op(d);
		uint32_t head = w->queue_head;

		if (head - __atomic_load_n(&w->queue_tail, __ATOMIC_ACQUIRE) <
		    FILE_OP_QUEUE_SIZE) {
			*wp = w;
			*headp = head;
			return &w->file_op_queue[head & FILE_OP_QUEUE_MASK];
		}

		pthread_mutex_unlock(&d->submit_lock);
		sched_yield();
		pthread_mutex_lock(&d->submit_lock);
	}
}

/* ------------------------------------------------------------------ */
/*  Driver lifecycle                                                  */
/* ------------------------------------------------------------------ */

opends_error_t
opends_driver_open(void)
{
	if (drv)
		return opends_err(OPENDS_DRIVER_ALREADY_OPEN);

	const char *dev = getenv(ENV_HOMI_DEV);
	if (!dev || !dev[0]) {
		fprintf(stderr,
		        "aisio: %s must name the NVMe device the HOMI daemon "
		        "owns\n",
		        ENV_HOMI_DEV);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}

	struct driver *d = calloc(1, sizeof(*d));
	if (!d)
		return opends_err(OPENDS_INTERNAL_ERROR);

	snprintf(d->dev_uri, sizeof(d->dev_uri), "%s", dev);
	if (read_env_config(d) < 0) {
		free(d);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}
	pthread_mutex_init(&d->submit_lock, NULL);
	pthread_mutex_init(&d->reg_lock, NULL);

	const char *sock = getenv(ENV_HOMI_SOCKET);
	int rc = homic_connect(
	        (char *)(sock && sock[0] ? sock : DEFAULT_HOMI_SOCKET));
	if (rc < 0) {
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}

	drv = d;

	int orc = open_device(d);
	if (orc < 0) {
		homic_disconnect();
		free(d->attach_descpath);
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		drv = NULL;
		return opends_err(OPENDS_DEVICE_NOT_FOUND);
	}
	int arc = async_setup(d);
	if (arc != 0) {
		fprintf(stderr, "aisio: async_setup failed\n");
		xnvme_dev_close(d->xdev);
		homic_detach_qpair();
		homic_disconnect();
		free(d->attach_descpath);
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		drv = NULL;
		return opends_err_dev(OPENDS_DEVICE_DRIVER_ERROR, arc);
	}

	return opends_ok();
}

opends_error_t
opends_driver_close(void)
{
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);

	async_teardown(drv);

	for (int i = 0; i < drv->buf_count; i++) {
		struct buf_entry *e = &drv->bufs[i];
		if (e->owned)
			xnvme_buf_free(drv->xdev, (void *)e->base);
		else
			xnvme_mem_unmap(drv->xdev, (void *)e->base);
	}
	drv->buf_count = 0;

	if (drv->xdev)
		xnvme_dev_close(drv->xdev);

	homic_detach_qpair();
	homic_disconnect();
	free(drv->attach_descpath);

	pthread_mutex_destroy(&drv->submit_lock);
	pthread_mutex_destroy(&drv->reg_lock);
	free(drv);
	drv = NULL;
	return opends_ok();
}

long
opends_use_count(void)
{
	return __atomic_load_n(&use_count, __ATOMIC_RELAXED);
}

opends_error_t
opends_driver_get_properties(opends_drv_props_t *props)
{
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!props)
		return opends_err(OPENDS_INVALID_VALUE);

	memset(props, 0, sizeof(*props));
	props->major_version = 0;
	props->minor_version = 1;
	props->max_direct_io_size = drv->mdts_nbytes;
	return opends_ok();
}

opends_error_t
opends_driver_set_max_direct_io_size(size_t max_direct_io_size)
{
	(void)max_direct_io_size;
	return drv ? opends_ok() : opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
}

opends_error_t
opends_get_version(unsigned *major, unsigned *minor, unsigned *patch)
{
	if (major)
		*major = 0;
	if (minor)
		*minor = 1;
	if (patch)
		*patch = 0;
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Handle registration                                               */
/* ------------------------------------------------------------------ */

opends_error_t
opends_handle_register(opends_handle_t *fh, int fd)
{
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh)
		return opends_err(OPENDS_INVALID_VALUE);

	struct registered_file *h = calloc(1, sizeof(*h));
	if (!h)
		return opends_err(OPENDS_INTERNAL_ERROR);
	h->fd = fd;

	*fh = h;
	__atomic_fetch_add(&use_count, 1, __ATOMIC_RELAXED);
	return opends_ok();
}

void
opends_handle_deregister(opends_handle_t fh)
{
	if (!fh)
		return;
	free(fh);
	__atomic_fetch_sub(&use_count, 1, __ATOMIC_RELAXED);
}

/* ------------------------------------------------------------------ */
/*  Buffer allocation                                                 */
/* ------------------------------------------------------------------ */

void *
opends_alloc(size_t size)
{
	if (!drv || !drv->xdev)
		return NULL;

	pthread_mutex_lock(&drv->reg_lock);
	if (drv->buf_count >= MAX_BUF_ENTRIES) {
		pthread_mutex_unlock(&drv->reg_lock);
		return NULL;
	}

	void *buf = xnvme_buf_alloc(drv->xdev, size);
	if (!buf) {
		pthread_mutex_unlock(&drv->reg_lock);
		return NULL;
	}

	struct buf_entry *e = &drv->bufs[drv->buf_count++];
	e->base = buf;
	e->length = size;
	e->owned = true;
	pthread_mutex_unlock(&drv->reg_lock);
	return buf;
}

void
opends_free(void *buf)
{
	if (!drv || !buf)
		return;

	pthread_mutex_lock(&drv->reg_lock);
	for (int i = 0; i < drv->buf_count; i++) {
		if (drv->bufs[i].base == buf) {
			if (!drv->bufs[i].owned)
				break;
			xnvme_buf_free(drv->xdev, buf);
			drv->bufs[i] = drv->bufs[drv->buf_count - 1];
			drv->buf_count--;
			break;
		}
	}
	pthread_mutex_unlock(&drv->reg_lock);
}

opends_error_t
opends_buf_register(const void *buf_base, size_t size, int flags)
{
	(void)flags;

	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!drv->xdev)
		return opends_err(OPENDS_DEVICE_NOT_FOUND);
	if (!buf_base || !size)
		return opends_err(OPENDS_INVALID_VALUE);

	pthread_mutex_lock(&drv->reg_lock);
	for (int i = 0; i < drv->buf_count; i++) {
		if (drv->bufs[i].base == buf_base) {
			pthread_mutex_unlock(&drv->reg_lock);
			return opends_err(OPENDS_MEMORY_ALREADY_REGISTERED);
		}
	}
	if (drv->buf_count >= MAX_BUF_ENTRIES) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_INTERNAL_ERROR);
	}

	int rc = xnvme_mem_map(drv->xdev, (void *)buf_base, size);
	if (rc < 0) {
		pthread_mutex_unlock(&drv->reg_lock);
		fprintf(stderr,
		        "opends_buf_register: xnvme_mem_map(%p, %zu) rc=%d\n",
		        buf_base, size, rc);
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);
	}

	struct buf_entry *e = &drv->bufs[drv->buf_count++];
	e->base = buf_base;
	e->length = size;
	e->owned = false;
	pthread_mutex_unlock(&drv->reg_lock);
	return opends_ok();
}

opends_error_t
opends_buf_deregister(const void *buf_base)
{
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!drv->xdev)
		return opends_err(OPENDS_DEVICE_NOT_FOUND);
	if (!buf_base)
		return opends_err(OPENDS_INVALID_VALUE);

	pthread_mutex_lock(&drv->reg_lock);
	for (int i = 0; i < drv->buf_count; i++) {
		if (drv->bufs[i].base == buf_base) {
			if (drv->bufs[i].owned) {
				pthread_mutex_unlock(&drv->reg_lock);
				return opends_err(OPENDS_INVALID_VALUE);
			}
			drv->bufs[i] = drv->bufs[drv->buf_count - 1];
			drv->buf_count--;
			xnvme_mem_unmap(drv->xdev, (void *)buf_base);
			pthread_mutex_unlock(&drv->reg_lock);
			return opends_ok();
		}
	}
	pthread_mutex_unlock(&drv->reg_lock);
	return opends_err(OPENDS_MEMORY_NOT_REGISTERED);
}

/* ------------------------------------------------------------------ */
/*  Synchronous I/O                                                   */
/* ------------------------------------------------------------------ */

static ssize_t
submit_sync_op(struct driver *d, bool is_write, opends_handle_t fh,
               void *buf_base, size_t size, off_t file_offset, off_t buf_offset)
{
	if (!d)
		return -(ssize_t)OPENDS_DRIVER_NOT_INITIALIZED;
	if (!fh || !buf_base)
		return -(ssize_t)OPENDS_INVALID_VALUE;
	if (!d->async_ready)
		return -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR;

	struct io_worker *w;
	uint32_t head;

	pthread_mutex_lock(&d->submit_lock);
	struct file_op *op = claim_slot_locked(d, &w, &head);
	op->mode = FILE_OP_SYNC;
	op->is_write = is_write;
	op->h = (struct registered_file *)fh;
	op->buf_base = buf_base;
	op->u.sync.size = size;
	op->u.sync.file_offset = file_offset;
	op->u.sync.buf_offset = buf_offset;
	__atomic_store_n(&op->state, FILE_OP_PENDING, __ATOMIC_RELEASE);
	__atomic_store_n(&w->queue_head, head + 1, __ATOMIC_RELEASE);
	pthread_mutex_unlock(&d->submit_lock);

	while (__atomic_load_n(&op->state, __ATOMIC_ACQUIRE) != FILE_OP_FREE)
		sched_yield();

	ssize_t n = op->u.sync.result;
	if (n < 0)
		fprintf(stderr, "opends_%s(size=%zu, off=%ld) failed: rc=%zd\n",
		        is_write ? "write" : "read", size, (long)file_offset,
		        n);
	return n;
}

ssize_t
opends_read(opends_handle_t fh, void *buf_base, size_t size, off_t file_offset,
            off_t buf_offset)
{
	return submit_sync_op(drv, false, fh, buf_base, size, file_offset,
	                      buf_offset);
}

ssize_t
opends_write(opends_handle_t fh, const void *buf_base, size_t size,
             off_t file_offset, off_t buf_offset)
{
	return submit_sync_op(drv, true, fh, (void *)buf_base, size,
	                      file_offset, buf_offset);
}

/* ------------------------------------------------------------------ */
/*  Stream-based async I/O                                            */
/* ------------------------------------------------------------------ */

static opends_error_t
classify_accel_failure(struct driver *d, int accel_rc)
{
	ds_accel_ctx_t cur;

	if (ds_accel->ctx_get(&cur) != 0 || cur != d->accel_ctx)
		return opends_err_dev(OPENDS_CONTEXT_MISMATCH, accel_rc);
	return opends_err_dev(OPENDS_INTERNAL_ERROR, accel_rc);
}

static opends_error_t
submit_async_op(struct driver *d, bool is_write, opends_handle_t fh,
                void *buf_base, size_t *size_p, off_t *file_offset_p,
                off_t *buf_offset_p, ssize_t *bytes_p, opends_stream_t stream)
{
	if (!d)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh || !buf_base || !size_p || !file_offset_p || !buf_offset_p ||
	    !bytes_p)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!stream)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!d->async_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	ds_accel_stream_t cus = (ds_accel_stream_t)stream;
	struct opends_stream *opends_stream = opends_stream_get(d, cus);
	if (!opends_stream)
		return opends_err(OPENDS_INTERNAL_ERROR);

	struct io_worker *w;
	uint32_t head;

	pthread_mutex_lock(&d->submit_lock);
	struct file_op *op = claim_slot_locked(d, &w, &head);
	uint32_t seq = ++opends_stream->next_seq;
	op->mode = FILE_OP_ASYNC;
	op->is_write = is_write;
	op->h = (struct registered_file *)fh;
	op->buf_base = buf_base;
	op->u.async.size_p = size_p;
	op->u.async.file_offset_p = file_offset_p;
	op->u.async.buf_offset_p = buf_offset_p;
	op->u.async.bytes_read_p = bytes_p;
	op->u.async.opends_stream = opends_stream;
	op->u.async.seq = seq;

	/* Order the I/O thread against the user's stream through a per-stream
	 * gate word (strictly monotonic, two phases per op): the submitter
	 * publishes 2*seq on arrival and parks, and the I/O thread's store of
	 * 2*seq+1 releases it once the I/O is done (a read's DMA landed, or a
	 * write's source was staged). Do not roll back seq on failure: a reused
	 * seq would let the I/O thread's gate check pass before the stream is
	 * ready.
	 *
	 * launch_host_func is faster when no copy kernel is enqueued per op.
	 * Otherwise stream_write/stream_wait is. The gate ops are commands the
	 * GPU runs, so they wait for this context to be scheduled, and another
	 * process using the GPU delays them by orders of magnitude. The
	 * callback runs on the CPU and never waits. A copy kernel waits for
	 * the GPU regardless, so once one is enqueued the callback wins
	 * nothing. Hence the coupling to assume_aligned_only. */
	int accel_rc;
	if (d->assume_aligned_only) {
		accel_rc = ds_accel->launch_host_func(cus, park_gate_cb, op);
		if (accel_rc != 0) {
			pthread_mutex_unlock(&d->submit_lock);
			return classify_accel_failure(d, accel_rc);
		}
	} else {
		ds_accel_devptr_t gate = opends_stream->gate_dptr;
		accel_rc = ds_accel->stream_write_value32(cus, gate, 2 * seq);
		if (accel_rc != 0) {
			pthread_mutex_unlock(&d->submit_lock);
			return classify_accel_failure(d, accel_rc);
		}
		accel_rc = ds_accel->stream_wait_value32_geq(cus, gate,
		                                             2 * seq + 1);
		if (accel_rc != 0) {
			pthread_mutex_unlock(&d->submit_lock);
			return classify_accel_failure(d, accel_rc);
		}
	}

	__atomic_store_n(&op->state, FILE_OP_PENDING, __ATOMIC_RELEASE);
	__atomic_store_n(&w->queue_head, head + 1, __ATOMIC_RELEASE);
	pthread_mutex_unlock(&d->submit_lock);

	/* Reads enqueue the deferred tail copy (writes run none): offsets
	 * resolve behind the gate, so the copy size is unknown here, and
	 * copy_stream no-ops when it is zero. Enqueue after publishing so a
	 * failed enqueue is still drained by the I/O thread (which releases the
	 * gate); only this read is lost. */
	if (!d->assume_aligned_only && !is_write) {
		accel_rc = ds_accel->copy_stream(opends_stream->bounce_desc_dev,
		                                 cus);
		if (accel_rc != 0)
			return classify_accel_failure(d, accel_rc);
	}

	return opends_ok();
}

opends_error_t
opends_read_async(opends_handle_t fh, void *buf_base, size_t *size_p,
                  off_t *file_offset_p, off_t *buf_offset_p,
                  ssize_t *bytes_read_p, opends_stream_t stream)
{
	return submit_async_op(drv, false, fh, buf_base, size_p, file_offset_p,
	                       buf_offset_p, bytes_read_p, stream);
}

opends_error_t
opends_write_async(opends_handle_t fh, void *buf_base, size_t *size_p,
                   off_t *file_offset_p, off_t *buf_offset_p,
                   ssize_t *bytes_written_p, opends_stream_t stream)
{
	return submit_async_op(drv, true, fh, buf_base, size_p, file_offset_p,
	                       buf_offset_p, bytes_written_p, stream);
}

opends_error_t
opends_stream_register(opends_stream_t stream, unsigned flags)
{
	(void)flags;

	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!stream)
		return opends_err(OPENDS_INVALID_VALUE);

	if (!drv->async_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	ds_accel_stream_t cus = (ds_accel_stream_t)stream;

	pthread_mutex_lock(&drv->reg_lock);
	if (ds_stream_map_get(drv->stream_map, STREAM_MAP_MASK, cus) >= 0) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_ok();
	}
	if (drv->n_streams >= MAX_STREAMS) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_INTERNAL_ERROR);
	}

	int n = drv->n_streams;
	struct opends_stream *opends_stream = &drv->streams[n];
	uint32_t *words = (uint32_t *)drv->stream_words_host;
	opends_stream->gate = &words[n];
	opends_stream->gate_dptr =
	        drv->stream_words_dptr + (size_t)n * sizeof(uint32_t);
	*opends_stream->gate = 0;
	opends_stream->next_seq = 0;

	int rc = stream_bounce_alloc(opends_stream, drv->xdev);
	if (rc != 0) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err_dev(OPENDS_INTERNAL_ERROR, rc);
	}

	if (ds_stream_map_put(drv->stream_map, STREAM_MAP_MASK, cus, n) < 0) {
		stream_bounce_free(opends_stream, drv->xdev);
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_INTERNAL_ERROR);
	}

	drv->n_streams = n + 1;
	pthread_mutex_unlock(&drv->reg_lock);
	return opends_ok();
}

opends_error_t
opends_stream_deregister(opends_stream_t stream)
{
	(void)stream;
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Batch I/O (not implemented)                                       */
/* ------------------------------------------------------------------ */

opends_error_t
opends_batch_io_setup(opends_batch_handle_t *batch_idp, unsigned nr)
{
	(void)batch_idp;
	(void)nr;
	return opends_err(OPENDS_ASYNC_NOT_SUPPORTED);
}

opends_error_t
opends_batch_io_submit(opends_batch_handle_t batch_idp, unsigned nr,
                       opends_io_params_t *iocbp, unsigned int flags)
{
	(void)batch_idp;
	(void)nr;
	(void)iocbp;
	(void)flags;
	return opends_err(OPENDS_ASYNC_NOT_SUPPORTED);
}

opends_error_t
opends_batch_io_get_status(opends_batch_handle_t batch_idp, unsigned min_nr,
                           unsigned *nr, opends_io_events_t *iocbp,
                           struct timespec *timeout)
{
	(void)batch_idp;
	(void)min_nr;
	(void)nr;
	(void)iocbp;
	(void)timeout;
	return opends_err(OPENDS_ASYNC_NOT_SUPPORTED);
}

opends_error_t
opends_batch_io_cancel(opends_batch_handle_t batch_idp)
{
	(void)batch_idp;
	return opends_err(OPENDS_ASYNC_NOT_SUPPORTED);
}

void
opends_batch_io_destroy(opends_batch_handle_t batch_idp)
{
	(void)batch_idp;
}
