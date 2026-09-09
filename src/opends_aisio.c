/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends_aisio.c - aisio backend for raw-NVMe direct storage.
 *
 * Reads go straight from an NVMe device into GPU memory via xNVMe's upcie-cuda
 * backend (PCIe P2P DMA). The HOMI server (xnvme's "homi serve") is the
 * primary of an xNVMe multi-process group and holds the controllers up. This
 * driver joins the same group as a secondary, takes its device set from the
 * group's runtime, and allocates its own I/O queues. A registered file's
 * extents come from xal, which maps the file under a lease held on the
 * registered descriptor, through the handle of the device whose mount holds
 * the file. They are in byte units, and this driver converts to LBAs with
 * its own device geometry.
 *
 * Requires: libxnvme and the CUDA toolkit. The NVMe kernel driver must be
 * unbound from the target device before opends_driver_open runs.
 *
 * Writes go through the kernel-mounted filesystem: the source is staged
 * through a pinned host buffer and pwritten via the fd, so XFS over qublk
 * allocates blocks and writes the data; the file's extents are then
 * re-resolved for later P2P reads. The copies into the buffer are queued, on
 * the user's stream ahead of its gate for a stream write and on the default
 * stream for the rest, and an I/O thread only polls for them. An I/O thread
 * never makes an accelerator call that waits: a stream can be waiting for it.
 */

#define _GNU_SOURCE

#include "ds_aisio.h"

#define NO_WAITER UINT64_MAX

static struct driver *drv;
static long use_count;

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
	return a > b ? a : b;
}

static inline uint64_t
monotonic_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void
cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__)
	__asm__ __volatile__("yield");
#endif
}

/* The upcie DMA registries are process-global and ignore the dev argument, so
 * one device stands in for all of them on every buffer call. */
static inline struct xnvme_dev *
mem_dev(struct driver *d)
{
	return d->devices[0].xdev;
}

/* ------------------------------------------------------------------ */
/*  Buffer table and chunk mappings                                   */
/* ------------------------------------------------------------------ */

static int
buf_lower_bound(const struct driver *d, const void *base)
{
	int lo = 0, hi = d->buf_count;

	while (lo < hi) {
		int mid = lo + (hi - lo) / 2;
		if ((uintptr_t)d->bufs[mid].base < (uintptr_t)base)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

static struct buf_entry *
buf_find(struct driver *d, const void *base)
{
	int i = buf_lower_bound(d, base);

	if (i < d->buf_count && d->bufs[i].base == base)
		return &d->bufs[i];
	return NULL;
}

/* The entry whose range holds [ptr, ptr + size), or NULL. */
static const struct buf_entry *
buf_covering(const struct driver *d, const void *ptr, size_t size)
{
	int i = buf_lower_bound(d, ptr);

	if (i < d->buf_count && d->bufs[i].base == ptr)
		i++;
	while (i-- > 0) {
		const struct buf_entry *e = &d->bufs[i];
		uintptr_t off = (uintptr_t)ptr - (uintptr_t)e->base;

		if (off <= e->length && size <= e->length - off)
			return e;
	}
	return NULL;
}

static struct buf_entry *
buf_insert(struct driver *d, const void *base, size_t length, bool owned)
{
	int i = buf_lower_bound(d, base);
	struct buf_entry *e;

	if (d->buf_count >= MAX_BUF_ENTRIES)
		return NULL;
	memmove(&d->bufs[i + 1], &d->bufs[i],
	        (size_t)(d->buf_count - i) * sizeof(*e));
	d->buf_count++;
	e = &d->bufs[i];
	e->base = base;
	e->length = length;
	e->owned = owned;
	return e;
}

static void
buf_remove(struct driver *d, struct buf_entry *e)
{
	int i = (int)(e - d->bufs);

	memmove(&d->bufs[i], &d->bufs[i + 1],
	        (size_t)(d->buf_count - i - 1) * sizeof(*e));
	d->buf_count--;
}

/* A chunk mapping also covers neighbouring allocations, which xnvme then
 * cannot refuse, so every op is checked against the table it came from. */
bool
ds_aisio_range_registered(struct driver *d, const void *ptr, size_t size)
{
	bool ok;

	pthread_mutex_lock(&d->reg_lock);
	ok = buf_covering(d, ptr, size) != NULL;
	pthread_mutex_unlock(&d->reg_lock);
	return ok;
}

/* First run ending after addr; runs are disjoint and sorted. */
static int
run_lower_bound(const struct driver *d, uintptr_t addr)
{
	int lo = 0, hi = d->run_count;

	while (lo < hi) {
		int mid = lo + (hi - lo) / 2;
		if (d->runs[mid].base + d->runs[mid].nbytes <= addr)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

static void
run_insert(struct driver *d, uintptr_t base, size_t nbytes)
{
	int i = run_lower_bound(d, base);

	memmove(&d->runs[i + 1], &d->runs[i],
	        (size_t)(d->run_count - i) * sizeof(d->runs[0]));
	d->run_count++;
	d->runs[i].base = base;
	d->runs[i].nbytes = nbytes;
	d->runs[i].refs = 1;
}

static void
run_remove(struct driver *d, int i)
{
	memmove(&d->runs[i], &d->runs[i + 1],
	        (size_t)(d->run_count - i - 1) * sizeof(d->runs[0]));
	d->run_count--;
}

struct run_gap {
	uintptr_t base;
	size_t nbytes;
};

/* Map the chunks of [lo, hi) no run covers yet, one xnvme_mem_map per gap, and
 * reference every run the range touches. A gap never overlaps a live mapping,
 * which is what xnvme's registry refuses. Returns 0 or a negative errno. */
static int
runs_acquire(struct driver *d, uintptr_t lo, uintptr_t hi)
{
	int err = 0;
	int first = run_lower_bound(d, lo);
	int ngaps = 0, mapped;
	uintptr_t cursor = lo;
	struct run_gap *gaps;

	for (int i = first; i < d->run_count && d->runs[i].base < hi; i++) {
		if (cursor < d->runs[i].base)
			ngaps++;
		cursor = d->runs[i].base + d->runs[i].nbytes;
	}
	if (cursor < hi)
		ngaps++;
	if (d->run_count + ngaps > MAX_MAP_RUNS)
		return -ENOSPC;

	gaps = calloc(ngaps ? (size_t)ngaps : 1, sizeof(*gaps));
	if (!gaps)
		return -ENOMEM;
	ngaps = 0;
	cursor = lo;
	for (int i = first; i < d->run_count && d->runs[i].base < hi; i++) {
		if (cursor < d->runs[i].base) {
			gaps[ngaps].base = cursor;
			gaps[ngaps].nbytes = d->runs[i].base - cursor;
			ngaps++;
		}
		cursor = d->runs[i].base + d->runs[i].nbytes;
	}
	if (cursor < hi) {
		gaps[ngaps].base = cursor;
		gaps[ngaps].nbytes = hi - cursor;
		ngaps++;
	}

	for (mapped = 0; mapped < ngaps; mapped++) {
		err = xnvme_mem_map(mem_dev(d), (void *)gaps[mapped].base,
		                    gaps[mapped].nbytes);
		if (err < 0) {
			fprintf(stderr,
			        "opends_buf_register: xnvme_mem_map(%p, %zu) "
			        "rc=%d\n",
			        (void *)gaps[mapped].base, gaps[mapped].nbytes,
			        err);
			break;
		}
	}
	if (err < 0) {
		for (int i = 0; i < mapped; i++)
			xnvme_mem_unmap(mem_dev(d), (void *)gaps[i].base);
		free(gaps);
		return err;
	}

	for (int i = first; i < d->run_count && d->runs[i].base < hi; i++)
		d->runs[i].refs++;
	for (int i = 0; i < ngaps; i++)
		run_insert(d, gaps[i].base, gaps[i].nbytes);
	free(gaps);
	return 0;
}

/* Drop the references [lo, hi) holds; unmap the runs nobody references. */
static void
runs_release(struct driver *d, uintptr_t lo, uintptr_t hi)
{
	int i = run_lower_bound(d, lo);

	while (i < d->run_count && d->runs[i].base < hi) {
		if (--d->runs[i].refs > 0) {
			i++;
			continue;
		}
		xnvme_mem_unmap(mem_dev(d), (void *)d->runs[i].base);
		run_remove(d, i);
	}
}

static void
runs_clear(struct driver *d)
{
	for (int i = 0; i < d->run_count; i++)
		xnvme_mem_unmap(mem_dev(d), (void *)d->runs[i].base);
	d->run_count = 0;
}

/* The chunk span a buffer falls in. */
static void
buf_chunks(const struct driver *d, const void *base, size_t size, uintptr_t *lo,
           uintptr_t *hi)
{
	uintptr_t mask = d->chunk_nbytes - 1;

	*lo = (uintptr_t)base & ~mask;
	*hi = ((uintptr_t)base + size + mask) & ~mask;
}

static void *
buf_alloc_locked(struct driver *d, size_t nbytes)
{
	pthread_mutex_lock(&d->alloc_lock);
	void *p = xnvme_buf_alloc(mem_dev(d), nbytes);
	pthread_mutex_unlock(&d->alloc_lock);
	return p;
}

static void
buf_free_locked(struct driver *d, void *p)
{
	pthread_mutex_lock(&d->alloc_lock);
	xnvme_buf_free(mem_dev(d), p);
	pthread_mutex_unlock(&d->alloc_lock);
}

/* ------------------------------------------------------------------ */
/*  Staging buffers                                                     */
/* ------------------------------------------------------------------ */

void
ds_aisio_future_complete(opends_async_future_t *fut, ssize_t result)
{
	fut->result = result;
	__atomic_store_n(&fut->done, 1, __ATOMIC_RELEASE);
}

/* Allocate the tail-bounce slot and copy descriptor for a stream. GPU alloc
 * APIs do a device-wide sync, so this must run on a host thread: on the aisio
 * I/O thread it could deadlock waiting on the thread's own gated stream.
 * Returns 0 on success; on failure, the dev_err to report (vendor code or
 * -1), with the fields left zeroed. */
static int
stream_bounce_alloc(struct opends_stream *s, struct driver *d)
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
	 * serialises ops. A stream is not bound to a device, so the slot takes
	 * the largest lba_size in the set. */
	void *buf = buf_alloc_locked(d, d->max_lba_size);
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
stream_bounce_free(struct opends_stream *s, struct driver *d)
{
	if (s->bounce_buf) {
		buf_free_locked(d, s->bounce_buf);
		s->bounce_buf = NULL;
	}
	if (s->bounce_desc_host) {
		ds_accel->host_free(s->bounce_desc_host);
		s->bounce_desc_host = NULL;
	}
	s->bounce_desc_dev = 0;
}

/* The extents of a registered file, from the lease taken at registration.
 * xal counts the caller as a reader until the matching release, which is
 * what keeps a writer's open() or truncate() waiting until the DMA is done.
 * An extent without an address cannot be read raw. The handle carries no
 * LBA size, so convert with our own geometry. -EAGAIN: the old map is being
 * recalled, try again later. */
static int
lease_extents(struct registered_file *h, struct ds_extent **out,
              uint32_t *out_n)
{
	struct nvme_device *dev = h->dev;
	struct xal_extents *xe;
	struct ds_extent *ex;
	int rc;

	/* A write through the descriptor leaves the map behind. A release
	 * finds that, and the acquire after it maps the file again; the lock
	 * keeps another acquire from taking the old map meanwhile. */
	pthread_mutex_lock(&h->lock);
	if (__atomic_exchange_n(&h->remap, false, __ATOMIC_ACQ_REL)) {
		rc = xal_extents_acquire(dev->xal, h->lease, &xe);
		if (rc == 0) {
			xal_extents_release(dev->xal, h->lease);
		} else {
			__atomic_store_n(&h->remap, true, __ATOMIC_RELEASE);
		}
	}
	rc = xal_extents_acquire(dev->xal, h->lease, &xe);
	pthread_mutex_unlock(&h->lock);
	if (rc < 0)
		return rc;

	ex = calloc(xe->count ? xe->count : 1, sizeof(*ex));
	if (!ex) {
		rc = -ENOMEM;
		goto err;
	}
	for (uint32_t i = 0; i < xe->count; i++) {
		struct xal_extent *e =
		        xal_extent_at(dev->xal, xe->extent_idx + i);
		struct xal_extent_converted b = {0};

		if (!e || (e->flag & XAL_EXTENT_NOADDR)) {
			rc = -ENOENT;
			goto err;
		}
		rc = xal_extent_in_bytes(dev->xal, e, &b);
		if (rc < 0)
			goto err;
		if (b.start_block & (uint64_t)(dev->lba_size - 1)) {
			rc = -EIO;
			goto err;
		}
		ex[i].file_offset = b.start_offset;
		ex[i].slba = b.start_block >> dev->lba_shift;
		ex[i].length = b.size;
		ex[i].flag = e->flag;
	}
	*out = ex;
	*out_n = xe->count;
	return 0;

err:
	free(ex);
	xal_extents_release(dev->xal, h->lease);
	return rc;
}

/* Whole LBAs cannot be DMAed to a destination off a dword: upcie wants the
 * data buffer dword aligned, and the PRP offset is dword granular. Every LBA
 * boundary of a read lands at the same offset modulo 4, so one test covers
 * them all. A read with no whole LBA is bounced whatever its destination. */
static bool
needs_kernel_read(const struct nvme_device *dev, const uint8_t *dst,
                  size_t size, uint64_t req_start)
{
	uint64_t lba_mask = dev->lba_size - 1;
	uint64_t first_lba = (req_start + lba_mask) & ~lba_mask;
	uintptr_t shift = (uintptr_t)dst - req_start;

	if (size < dev->lba_size ||
	    first_lba + dev->lba_size > req_start + size)
		return false;
	return (shift & (NVME_PRP_OFFSET_ALIGN - 1)) != 0;
}

/* Read the next chunk into the pinned buffer and queue its copy on the
 * default stream, with the write of the word behind it. An I/O thread must
 * not wait for the accelerator: a stream can wait for this thread. */
static void
kernel_read_chunk(struct io_worker *w)
{
	struct kernel_read *k = &w->kread;
	struct file_op *op = k->op;
	off_t start = (off_t)k->off & ~(off_t)(OPENDS_DIRECT_ALIGN - 1);
	size_t head = (size_t)((off_t)k->off - start);
	size_t n = k->buf_nbytes - head;
	size_t span;
	ssize_t got;
	int err;

	if (n > k->remaining) {
		n = k->remaining;
	}
	span = (head + n + OPENDS_DIRECT_ALIGN - 1) &
	       ~(size_t)(OPENDS_DIRECT_ALIGN - 1);
	got = opends_direct_pread_span(op->h->fd, k->buf, span, start);
	if (got < 0) {
		op->err = (got == -(ssize_t)EINVAL)
		                  ? OPENDS_INVALID_VALUE
		                  : OPENDS_DEVICE_DRIVER_ERROR;
		goto done;
	}
	if ((size_t)got <= head) {
		goto done;
	}
	if ((size_t)got - head < n) {
		n = (size_t)got - head;
		k->last = true;
	}

	err = ds_accel->copy_to_dev_async(k->dst, (uint8_t *)k->buf + head, n);
	if (err != 0) {
		op->err = OPENDS_DEVICE_DRIVER_ERROR;
		goto done;
	}
	k->seq++;
	err = ds_accel->stream_write_value32(NULL, k->word_dptr, k->seq);
	if (err != 0) {
		/* nothing tells when the queued copy has run, so the buffer
		 * cannot be used again */
		k->broken = true;
		op->err = OPENDS_DEVICE_DRIVER_ERROR;
		goto done;
	}
	op->bytes_acc += n;
	k->dst += n;
	k->off += n;
	k->remaining -= n;
	return;
done:
	op->chunks_remaining = 0;
	k->op = NULL;
}

static void
kernel_read_poll(struct io_worker *w)
{
	struct kernel_read *k = &w->kread;

	if (__atomic_load_n(k->word, __ATOMIC_ACQUIRE) != k->seq) {
		return;
	}
	if (k->remaining == 0 || k->last) {
		k->op->chunks_remaining = 0;
		k->op = NULL;
		return;
	}
	kernel_read_chunk(w);
}

static void
kernel_read_start(struct io_worker *w, struct file_op *op, uint8_t *dst,
                  size_t size, uint64_t req_start)
{
	struct kernel_read *k = &w->kread;

	k->op = op;
	k->dst = dst;
	k->off = req_start;
	k->remaining = size;
	k->last = false;
	op->chunks_remaining = 1;
	kernel_read_chunk(w);
}

static int
kernel_read_setup(struct driver *d, struct io_worker *w, int idx)
{
	struct kernel_read *k = &w->kread;
	size_t word = MAX_STREAMS + (size_t)idx;
	size_t mdts = w->dev->mdts_nbytes;
	int err;

	/* the same chunk size as on the direct path */
	k->buf_nbytes = (mdts + OPENDS_DIRECT_ALIGN - 1) &
	                ~(size_t)(OPENDS_DIRECT_ALIGN - 1);
	err = posix_memalign(&k->buf, OPENDS_DIRECT_ALIGN, k->buf_nbytes);
	if (err != 0) {
		k->buf = NULL;
		return -1;
	}
	err = ds_accel->host_pin(k->buf, k->buf_nbytes, NULL);
	if (err != 0) {
		free(k->buf);
		k->buf = NULL;
		return -1;
	}
	k->word = (uint32_t *)d->stream_words_host + word;
	k->word_dptr = d->stream_words_dptr + word * sizeof(uint32_t);
	return 0;
}

static void
kernel_read_teardown(struct io_worker *w)
{
	if (!w->kread.buf) {
		return;
	}
	ds_accel->host_unpin(w->kread.buf);
	free(w->kread.buf);
	w->kread.buf = NULL;
}

static int
open_device(struct driver *d, struct nvme_device *dev)
{
	/* Joins the group the HOMI server is primary of. Whoever opens first
	 * wins the role election, so a homi_id the server does not serve would
	 * silently make this process the controller owner; both sides use
	 * homi_id 1 by convention unless overridden. */
	struct xnvme_opts opts = xnvme_opts_default();
	opts.be = ds_accel->xnvme_be;
	opts.homi_id = d->cfg.homi_id;
	opts.host_heap_size = d->cfg.host_heap_nbytes;
	opts.device_heap_size = d->cfg.device_heap_nbytes;

	dev->xdev = xnvme_dev_open(dev->dev_uri, &opts);
	if (!dev->xdev) {
		fprintf(stderr,
		        "aisio open_device: xnvme_dev_open(%s, be=%s, "
		        "homi_id=%u) failed\n",
		        dev->dev_uri, opts.be, d->cfg.homi_id);
		return -EIO;
	}

	const struct xnvme_geo *geo = xnvme_dev_get_geo(dev->xdev);
	dev->nsid = xnvme_dev_get_nsid(dev->xdev);
	dev->lba_size = geo->lba_nbytes ? geo->lba_nbytes : geo->nbytes;
	dev->mdts_nbytes =
	        geo->mdts_nbytes ? geo->mdts_nbytes : DEFAULT_BOUNCE_SIZE;

	if (dev->lba_size == 0) {
		fprintf(stderr,
		        "aisio open_device(%s): zero geometry from "
		        "xnvme_dev_open (lba_nbytes=%u nbytes=%u "
		        "mdts_nbytes=%u); controller likely needs a PCI reset "
		        "before this open\n",
		        dev->dev_uri, geo->lba_nbytes, geo->nbytes,
		        geo->mdts_nbytes);
		xnvme_dev_close(dev->xdev);
		dev->xdev = NULL;
		return -EIO;
	}
	if (dev->lba_size & (dev->lba_size - 1)) {
		fprintf(stderr,
		        "aisio open_device(%s): lba_size=%u is not a power of "
		        "2\n",
		        dev->dev_uri, dev->lba_size);
		xnvme_dev_close(dev->xdev);
		dev->xdev = NULL;
		return -EIO;
	}
	dev->lba_shift = (uint32_t)__builtin_ctz(dev->lba_size);
	pthread_mutex_init(&dev->admit_lock, NULL);
	dev->waiting_ticket = NO_WAITER;

	return 0;
}

static void
close_devices(struct driver *d)
{
	for (int i = 0; i < d->n_devices; i++) {
		if (d->devices[i].xdev) {
			xnvme_dev_close(d->devices[i].xdev);
			d->devices[i].xdev = NULL;
			pthread_mutex_destroy(&d->devices[i].admit_lock);
		}
	}
}

static int
open_devices(struct driver *d)
{
	int err;

	for (int i = 0; i < d->n_devices; i++) {
		err = open_device(d, &d->devices[i]);
		if (err < 0) {
			close_devices(d);
			return err;
		}
	}

	d->max_lba_size = 0;
	d->min_mdts_nbytes = UINT32_MAX;
	for (int i = 0; i < d->n_devices; i++) {
		struct nvme_device *dev = &d->devices[i];

		if (dev->lba_size > d->max_lba_size) {
			d->max_lba_size = dev->lba_size;
		}
		if (dev->mdts_nbytes < d->min_mdts_nbytes) {
			d->min_mdts_nbytes = dev->mdts_nbytes;
		}
	}

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
		uint32_t lba_size = op->h->dev->lba_size;
		op->bytes_acc += ((uint64_t)ctx->cmd.nvm.nlb + 1) * lba_size;
	}
	op->chunks_remaining--;
	xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
}

static int
submit_read_middle(struct nvme_device *dev, struct read_cursor *c,
                   struct file_op *op, uint64_t middle_lbas)
{
	uint32_t lba_nbytes = dev->lba_size;
	uint64_t max_chunk_lbas = dev->mdts_nbytes >> dev->lba_shift;
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
		int srv = xnvme_nvm_read(ctx, dev->nsid, chunk_slba, nlb,
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
submit_bounce_read(struct io_worker *w, struct file_op *op, void *bounce_buf,
                   uint8_t *abs_dst, uint64_t slba, size_t src_off,
                   size_t nbytes)
{
	struct nvme_device *dev = w->dev;

	if (op->n_bounces >= BOUNCE_SLOTS || src_off + nbytes > dev->lba_size) {
		op->err = OPENDS_INTERNAL_ERROR;
		return -1;
	}

	int slot = op->n_bounces;
	uint8_t *bounce = (uint8_t *)bounce_buf + (size_t)slot * dev->lba_size;
	uint16_t nlb = 0;

	for (;;) {
		struct xnvme_cmd_ctx *ctx;
		for (;;) {
			ctx = xnvme_queue_get_cmd_ctx(w->queue);
			if (ctx)
				break;
			xnvme_queue_poke(w->queue, 0);
		}
		xnvme_cmd_ctx_set_cb(ctx, bounce_cb, op);

		int srv =
		        xnvme_nvm_read(ctx, dev->nsid, slba, nlb, bounce, NULL);
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

	op->bounces[slot].dst = abs_dst;
	op->bounces[slot].src = bounce + src_off;
	op->bounces[slot].nbytes = nbytes;
	op->n_bounces = slot + 1;

	op->bounces_outstanding++;
	op->bytes_acc += nbytes;
	return 0;
}

static int
submit_stream_bounce(struct io_worker *w, struct file_op *op, uint8_t *abs_dst,
                     uint64_t cur_slba, size_t nbytes)
{
	struct opends_stream *s = op->u.stream.opends_stream;

	if (op->n_bounces) {
		op->err = OPENDS_INTERNAL_ERROR;
		return -1;
	}
	if (submit_bounce_read(w, op, s->bounce_buf, abs_dst, cur_slba, 0,
	                       nbytes) < 0)
		return -1;

	s->bounce_desc_host->dst = (uint64_t)(uintptr_t)abs_dst;
	s->bounce_desc_host->src = (uint64_t)(uintptr_t)s->bounce_buf;
	return 0;
}

static int
submit_host_bounce(struct io_worker *w, struct file_op *op, uint8_t *abs_dst,
                   uint64_t slba, size_t src_off, size_t nbytes)
{
	if (!op->bounce_buf)
		op->bounce_buf = buf_alloc_locked(
		        w->drv, BOUNCE_SLOTS * (size_t)w->dev->lba_size);
	if (!op->bounce_buf) {
		op->err = OPENDS_INTERNAL_ERROR;
		return -1;
	}
	return submit_bounce_read(w, op, op->bounce_buf, abs_dst, slba, src_off,
	                          nbytes);
}

static int
submit_partial(struct io_worker *w, struct file_op *op, uint8_t *abs_dst,
               uint64_t slba, size_t src_off, size_t nbytes)
{
	if (w->drv->cfg.assume_aligned_only) {
		op->err = OPENDS_INVALID_VALUE;
		return -1;
	}
	if (op->mode == FILE_OP_STREAM) {
		if (src_off) {
			op->err = OPENDS_INVALID_VALUE;
			return -1;
		}
		return submit_stream_bounce(w, op, abs_dst, slba, nbytes);
	}
	return submit_host_bounce(w, op, abs_dst, slba, src_off, nbytes);
}

/* The filesystem reads an unwritten extent as zeros, whatever its blocks
 * hold, and so does OpenDS. The stream path copies by descriptor on the
 * device, which this does not extend. */
static int
zero_fill(struct file_op *op, uint8_t *dst, size_t nbytes)
{
	static const uint8_t zeros[1 << 20];

	if (op->mode == FILE_OP_STREAM) {
		op->err = OPENDS_INVALID_VALUE;
		return -1;
	}
	while (nbytes) {
		size_t n = nbytes < sizeof(zeros) ? nbytes : sizeof(zeros);

		if (ds_accel->copy(dst, zeros, n) != 0) {
			op->err = OPENDS_DEVICE_DRIVER_ERROR;
			return -1;
		}
		dst += n;
		nbytes -= n;
		op->bytes_acc += n;
	}
	return 0;
}

/* Resolve where a read lands on the drive. Sets op->err on failure. */
static void
plan_read_op(struct io_worker *w, struct file_op *op)
{
	struct driver *d = w->drv;
	struct nvme_device *dev = w->dev;
	struct read_plan *p = &op->plan;
	uint32_t lba_mask = dev->lba_size - 1;
	size_t size;
	int err;

	memset(p, 0, sizeof(*p));
	if (dev->lba_size == 0) {
		op->err = OPENDS_INTERNAL_ERROR;
		return;
	}
	if ((dev->mdts_nbytes >> dev->lba_shift) == 0) {
		op->err = OPENDS_INVALID_VALUE;
		return;
	}

	if (op->mode == FILE_OP_STREAM) {
		size = *op->u.stream.size_p;
		p->req_start = (uint64_t)*op->u.stream.file_offset_p;
		p->dst_base =
		        (uint8_t *)op->buf_base + *op->u.stream.buf_offset_p;
	} else {
		size = op->u.async.size;
		p->req_start = (uint64_t)op->u.async.file_offset;
		p->dst_base = (uint8_t *)op->buf_base + op->u.async.buf_offset;
	}
	if (!ds_aisio_range_registered(d, p->dst_base, size)) {
		op->err = OPENDS_MEMORY_NOT_REGISTERED;
		return;
	}
	if (size > UINT64_MAX - p->req_start) {
		op->err = OPENDS_INVALID_VALUE;
		return;
	}
	p->req_end = p->req_start + size;
	if (needs_kernel_read(dev, p->dst_base, size, p->req_start)) {
		if (d->cfg.assume_aligned_only || op->mode == FILE_OP_STREAM) {
			op->err = OPENDS_INVALID_VALUE;
			return;
		}
		/* read by the kernel: no commands, so not admitted */
		p->kernel = true;
		return;
	}

	err = lease_extents(op->h, &p->extents, &p->extent_count);
	if (err == -EAGAIN) {
		/* the old map is being recalled: plan again on a later pass */
		op->planned = false;
		return;
	}
	if (err < 0) {
		op->err = OPENDS_FS_SETUP_ERROR;
		return;
	}
	op->held = true;

	for (uint32_t i = 0; i < p->extent_count; i++) {
		const struct ds_extent *e = &p->extents[i];
		uint64_t ext_start = e->file_offset;
		uint64_t span_start = max_u64(p->req_start, ext_start);
		uint64_t span_end =
		        XNVME_MIN_U64(p->req_end, ext_start + e->length);

		if (ext_start >= p->req_end) {
			break;
		}
		if (span_start >= span_end) {
			continue;
		}
		if (!p->spans) {
			p->first_lba = e->slba + ((span_start - ext_start) >>
			                          dev->lba_shift);
		}
		p->end_lba = e->slba + ((span_end - ext_start + lba_mask) >>
		                        dev->lba_shift);
		p->spans++;
	}
}

/* Whether the drive should take this read now.
 *
 * A read of several commands that starts away from where the last one ended
 * is slow while other commands are in flight, so it waits for the drive to
 * drain. A read that continues the last one, or fits in one command, does not
 * wait. The oldest refused read goes first and holds back younger ones, so a
 * flow of small reads cannot starve a waiting one. Each device admits on its
 * own, so a read on one drive never waits for another to drain.
 *
 * At queue depth 1 no read waits: a read alone then has one command in flight,
 * which is slower than the reads of several I/O threads side by side. */
static bool
admit_read(struct io_worker *w, struct file_op *op)
{
	struct nvme_device *dev = w->dev;
	const struct read_plan *p = &op->plan;
	bool one_cmd =
	        p->spans == 1 && p->req_end - p->req_start <= dev->mdts_nbytes;
	bool admit;

	if (w->drv->cfg.queue_depth == 1) {
		return true;
	}

	pthread_mutex_lock(&dev->admit_lock);
	if (dev->active_reads == 0) {
		admit = op->ticket <= dev->waiting_ticket;
	} else {
		admit = (one_cmd || p->first_lba == dev->next_lba) &&
		        op->ticket < dev->waiting_ticket;
	}
	if (admit) {
		op->admitted = true;
		dev->active_reads++;
		dev->next_lba = p->end_lba;
		if (dev->waiting_ticket == op->ticket) {
			dev->waiting_ticket = NO_WAITER;
		}
	} else if (op->ticket < dev->waiting_ticket) {
		dev->waiting_ticket = op->ticket;
	}
	pthread_mutex_unlock(&dev->admit_lock);

	return admit;
}

static void
retire_read(struct nvme_device *dev, struct file_op *op)
{
	if (!op->admitted) {
		return;
	}
	op->admitted = false;
	pthread_mutex_lock(&dev->admit_lock);
	dev->active_reads--;
	pthread_mutex_unlock(&dev->admit_lock);
}

static void
submit_read_op(struct io_worker *w, struct file_op *op)
{
	struct nvme_device *dev = w->dev;
	struct read_plan *p = &op->plan;
	uint64_t req_start = p->req_start;
	uint64_t req_end = p->req_end;
	uint8_t *dst_base = p->dst_base;
	struct ds_extent *extents = p->extents;
	uint32_t extent_count = p->extent_count;
	uint32_t lba_shift = dev->lba_shift;
	uint32_t lba_mask = dev->lba_size - 1;

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
		size_t buf_off = span_start - req_start;
		uint8_t *abs_dst = dst_base + buf_off;
		uint64_t cur_slba = e->slba + (off_in_ext >> lba_shift);
		size_t remaining = span_end - span_start;

		if (e->flag & XAL_EXTENT_UNWRITTEN) {
			if (zero_fill(op, abs_dst, remaining) < 0)
				goto out;
			continue;
		}

		size_t head_off = off_in_ext & lba_mask;
		if (head_off) {
			size_t head_bytes = dev->lba_size - head_off;

			if (head_bytes > remaining)
				head_bytes = remaining;
			if (submit_partial(w, op, abs_dst, cur_slba, head_off,
			                   head_bytes) < 0)
				goto out;
			cur_slba++;
			abs_dst += head_bytes;
			remaining -= head_bytes;
		}

		size_t tail_bytes = remaining & lba_mask;
		uint64_t middle_lbas = (remaining - tail_bytes) >> lba_shift;
		if (middle_lbas) {
			struct read_cursor c = {
			        .queue = w->queue,
			        .cur_slba = cur_slba,
			        .abs_dst = abs_dst,
			        .remaining = remaining,
			};
			if (submit_read_middle(dev, &c, op, middle_lbas) < 0)
				goto out;
			cur_slba = c.cur_slba;
			abs_dst = c.abs_dst;
			remaining = c.remaining;
		}

		if (tail_bytes) {
			if (submit_partial(w, op, abs_dst, cur_slba, 0,
			                   tail_bytes) < 0)
				goto out;
		}
	}
out:
	free(extents);
	p->extents = NULL;
}

static void
park_gate_cb(void *arg)
{
	struct file_op *op = arg;
	struct opends_stream *s = op->u.stream.opends_stream;
	uint32_t seq = op->u.stream.seq;

	__atomic_store_n(s->gate, 2 * seq, __ATOMIC_RELEASE);

	while ((int32_t)(__atomic_load_n(s->gate, __ATOMIC_ACQUIRE) -
	                 (2 * seq + 1)) < 0)
		;
}

static int
run_bounce_copies(struct file_op *op)
{
	for (int i = 0; i < op->n_bounces; i++) {
		if (ds_accel->copy(op->bounces[i].dst, op->bounces[i].src,
		                   op->bounces[i].nbytes) != 0)
			return -1;
	}
	return 0;
}

/* Count the op out as a reader once its DMA is done. xal says -ESTALE when a
 * writer got past the lease meanwhile; the data may then not be the file's. */
static void
release_extents(struct file_op *op)
{
	int rc;

	if (!op->held)
		return;
	rc = xal_extents_release(op->h->dev->xal, op->h->lease);
	op->held = false;
	if (rc < 0 && !op->err)
		op->err = OPENDS_FS_SETUP_ERROR;
}

static void
complete_read_op(struct file_op *op)
{
	release_extents(op);

	ssize_t n = op->err ? -(ssize_t)op->err : (ssize_t)op->bytes_acc;

	retire_read(op->h->dev, op);

	if (op->mode == FILE_OP_STREAM) {
		struct opends_stream *s = op->u.stream.opends_stream;
		uint32_t tail_bytes = (op->err || !op->n_bounces)
		                              ? 0
		                              : (uint32_t)op->bounces[0].nbytes;

		*op->u.stream.bytes_read_p = n;
		s->bounce_desc_host->n_bytes = tail_bytes;
		release_gate(s, op->u.stream.seq);
		__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
		return;
	}

	if (!op->err && run_bounce_copies(op) < 0)
		n = -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR;

	ds_aisio_future_complete(op->u.async.future, n);
	__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
}

/* ------------------------------------------------------------------ */
/*  Writes                                                            */
/* ------------------------------------------------------------------ */

/* Start a PENDING op. Returns true when it has to stay pending: a write
 * that waits for a buffer, or a read that waits to be admitted. */
static bool
dispatch_pending(struct io_worker *w, struct file_op *op)
{
	if (op->is_write) {
		return ds_dispatch_write(w, op);
	}
	if (!op->planned) {
		op->chunks_remaining = 0;
		op->bounces_outstanding = 0;
		op->bytes_acc = 0;
		op->err = 0;
		op->n_bounces = 0;
		op->held = false;
		op->planned = true;
		plan_read_op(w, op);
		if (!op->planned) {
			return true;
		}
	}
	if (!op->err && op->plan.kernel) {
		if (w->kread.broken) {
			op->err = OPENDS_DEVICE_DRIVER_ERROR;
		} else if (w->kread.op) {
			/* the buffer is in use: stay pending */
			return true;
		}
	}
	if (!op->err && op->plan.spans && !admit_read(w, op)) {
		return true;
	}
	__atomic_store_n(&op->state, FILE_OP_IN_FLIGHT, __ATOMIC_RELEASE);
	if (op->err) {
		return false;
	}
	if (op->plan.kernel) {
		kernel_read_start(w, op, op->plan.dst_base,
		                  op->plan.req_end - op->plan.req_start,
		                  op->plan.req_start);
	} else {
		submit_read_op(w, op);
	}
	return false;
}

static void
reap_in_flight(struct io_worker *w, struct file_op *op)
{
	if (op->is_write) {
		ds_write_poll(w, op);
		return;
	}
	if (w->kread.op == op) {
		kernel_read_poll(w);
	}
	if (op->chunks_remaining == 0 && op->bounces_outstanding == 0)
		complete_read_op(op);
}

/* Service a PENDING stream op: dispatch it once its stream has reached it,
 * otherwise leave it for a later pass. Returns true while it is still gated
 * or has to stay pending. A read, and a write that stages nothing, publish
 * 2*seq on the gate on arrival; a staged write has reached the stream when
 * its first round has landed. The gate is a free-running counter (an arrival
 * publish and this thread's release each tick it), so compare with
 * serial/cyclic arithmetic to match the device-side GEQ wait, keeping the
 * sequence wrap-safe. */
static bool
poll_stream_pending(struct io_worker *w, struct file_op *op)
{
	uint32_t gate;

	if (op->is_write && op->wr.rounds) {
		if (!op->err &&
		    __atomic_load_n(op->wr.landed, __ATOMIC_ACQUIRE) < 1)
			return true;
	} else {
		gate = __atomic_load_n(op->u.stream.opends_stream->gate,
		                       __ATOMIC_ACQUIRE);
		if ((int32_t)(gate - 2 * op->u.stream.seq) < 0)
			return true;
	}
	return dispatch_pending(w, op);
}

static void *
io_thread_main(void *arg)
{
	struct io_worker *w = arg;
	struct driver *d = w->drv;
	uint64_t idle_spin_ns = (uint64_t)d->cfg.idle_spin_us * 1000;
	uint64_t spin_until_ns = 0;
	bool busy_spin = d->cfg.busy_spin;
	bool stay_hot = true;

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
				case FILE_OP_ASYNC:
					if (dispatch_pending(w, op))
						busy = true;
					break;
				case FILE_OP_STREAM:
					if (poll_stream_pending(w, op))
						busy = true;
					break;
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
			stay_hot = true;
			xnvme_queue_poke(w->queue, 0);
		}

		if (busy_spin) {
			cpu_relax();
		} else if (busy) {
			sched_yield();
		} else {
			if (stay_hot) {
				spin_until_ns = monotonic_ns() + idle_spin_ns;
				stay_hot = false;
			}
			if (monotonic_ns() < spin_until_ns) {
				sched_yield();
			} else {
				struct timespec ts = {0, 100000};
				nanosleep(&ts, NULL);
			}
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

/* Workers go round-robin over the devices: worker i serves device
 * i % n_devices. */
static int
workers_of_device(const struct driver *d, int di)
{
	int base = d->cfg.n_io_threads / d->n_devices;

	return base + (di < d->cfg.n_io_threads % d->n_devices ? 1 : 0);
}

static void
workers_stop(struct driver *d)
{
	__atomic_store_n(&d->stop, true, __ATOMIC_RELEASE);
	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];

		for (int i = 0; i < dev->n_workers; i++) {
			pthread_join(dev->workers[i].thread, NULL);
		}
	}
}

static void
workers_free(struct driver *d)
{
	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];

		if (!dev->workers) {
			continue;
		}
		/* every slot, started or not: one that failed in setup
		 * holds what it got */
		for (int i = 0; i < workers_of_device(d, di); i++) {
			struct io_worker *w = &dev->workers[i];

			for (uint32_t s = 0; s < FILE_OP_QUEUE_SIZE; s++) {
				struct file_op *op = &w->file_op_queue[s];

				if (op->bounce_buf) {
					buf_free_locked(d, op->bounce_buf);
					op->bounce_buf = NULL;
				}
			}
			if (w->queue) {
				xnvme_queue_term(w->queue);
				w->queue = NULL;
			}
			kernel_read_teardown(w);
			free(w->edge_buf);
			w->edge_buf = NULL;
			if (w->descs) {
				ds_accel->host_free(w->descs);
				w->descs = NULL;
			}
		}
		free(dev->workers);
		dev->workers = NULL;
		dev->n_workers = 0;
	}
}

/* Returns 0 on success; on failure, the dev_err to report (vendor code or
 * -1). */
static int
workers_setup(struct driver *d)
{
	int qopts = d->cfg.cq_mirror ? XNVME_QUEUE_P2P_CQ_MIRROR : 0;
	int rc = ds_accel->ctx_get(&d->accel_ctx);
	if (rc != 0)
		return rc;
	rc = ds_accel->load();
	if (rc != 0)
		return rc;
	rc = ds_accel->alloc_granularity(&d->chunk_nbytes);
	if (rc != 0)
		return rc;
	if (!d->chunk_nbytes || (d->chunk_nbytes & (d->chunk_nbytes - 1)))
		return -1;

	/* Mapped so the device-side gate can address the word; the host
	 * callback path uses the host view only. */
	void *host = NULL;
	ds_accel_devptr_t dptr = 0;
	/* one more word for the kernel reads of each worker, then for each
	 * op slot a write's turn and the landed word of every stage block */
	size_t words_bytes = STREAM_WORDS_BYTES +
	                     (size_t)d->cfg.n_io_threads * sizeof(uint32_t) +
	                     (size_t)d->cfg.n_io_threads * FILE_OP_QUEUE_SIZE *
	                             (1 + DS_STAGE_BLOCKS) * sizeof(uint32_t);
	rc = ds_accel->host_alloc_mapped(words_bytes, &host, &dptr);
	if (rc != 0)
		return rc;
	memset(host, 0, words_bytes);
	d->stream_words_host = host;
	d->stream_words_dptr = dptr;

	d->stop = false;
	int mask_cpus = __builtin_popcountll(d->cfg.cpu_mask);
	int started = 0;
	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];
		int nw = workers_of_device(d, di);

		dev->workers = calloc((size_t)nw, sizeof(*dev->workers));
		if (!dev->workers)
			goto fail;

		for (int i = 0; i < nw; i++) {
			struct io_worker *w = &dev->workers[i];
			w->drv = d;
			w->dev = dev;
			if (ds_write_words_setup(d, w, started) < 0)
				goto fail;
			if (posix_memalign(&w->edge_buf, OPENDS_DIRECT_ALIGN,
			                   OPENDS_DIRECT_ALIGN) != 0) {
				w->edge_buf = NULL;
				goto fail;
			}
			if (!d->cfg.assume_aligned_only &&
			    kernel_read_setup(d, w, started) < 0)
				goto fail;
			if (xnvme_queue_init(dev->xdev, d->cfg.queue_depth,
			                     qopts, &w->queue) < 0) {
				w->queue = NULL;
				goto fail;
			}
			pthread_attr_t attr;
			pthread_attr_t *attrp = NULL;
			if (mask_cpus) {
				cpu_set_t set;
				CPU_ZERO(&set);
				CPU_SET(mask_nth_cpu(d->cfg.cpu_mask,
				                     started % mask_cpus),
				        &set);
				pthread_attr_init(&attr);
				pthread_attr_setaffinity_np(&attr, sizeof(set),
				                            &set);
				attrp = &attr;
			}
			int rc = pthread_create(&w->thread, attrp,
			                        io_thread_main, w);
			if (attrp)
				pthread_attr_destroy(&attr);
			if (rc != 0)
				goto fail;
			started++;
			dev->n_workers = i + 1;
		}
	}

	d->workers_ready = true;
	return 0;

fail:
	workers_stop(d);
	workers_free(d);
	ds_accel->host_free(d->stream_words_host);
	d->stream_words_host = NULL;
	return -1;
}

static void
workers_teardown(struct driver *d)
{
	if (!d->workers_ready)
		return;

	workers_stop(d);

	for (int i = 0; i < d->n_streams; i++) {
		stream_bounce_free(&d->streams[i], d);
	}

	workers_free(d);
	ds_write_buf_teardown(d);

	ds_accel->host_free(d->stream_words_host);
	d->stream_words_host = NULL;

	d->workers_ready = false;
}

static struct opends_stream *
opends_stream_get(struct driver *d, ds_accel_stream_t stream)
{
	int idx = ds_stream_map_get(d->stream_map, STREAM_MAP_MASK, stream);
	if (idx < 0)
		return NULL;
	return &d->streams[idx];
}

static struct io_worker *
route_op(struct nvme_device *dev)
{
	return &dev->workers[dev->rr_next++ % (uint32_t)dev->n_workers];
}

static struct file_op *
claim_slot_locked(struct driver *d, struct nvme_device *dev,
                  struct io_worker **wp, uint32_t *headp)
{
	for (;;) {
		struct io_worker *w = route_op(dev);
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

#define ATTACH_RETRIES 1200
#define ATTACH_BACKOFF_US 50000

/*
 * The device set comes from the multi-process runtime the HOMI server
 * created: the driver serves every controller the group holds, in the
 * runtime's order. -ENOENT (no runtime yet), -EAGAIN (mid-creation) and a
 * group that holds no controller yet all mean the server is still starting,
 * so retry on them.
 */
static int
discover_devices(struct driver *d)
{
	struct xnvme_cplane_info info;
	int err;

	if (aisio_config_homi_id(&d->cfg.homi_id) < 0) {
		return -EINVAL;
	}

	err = -ENOENT;
	for (int i = 0; i < ATTACH_RETRIES; i++) {
		err = xnvme_cplane_get_info(d->cfg.homi_id, &info);
		if (err == 0 && info.nctrlrs > 0) {
			break;
		}
		if (err < 0 && err != -ENOENT && err != -EAGAIN) {
			break;
		}
		usleep(ATTACH_BACKOFF_US);
	}
	if (err < 0) {
		fprintf(stderr,
		        "aisio: xnvme_cplane_get_info(homi_id=%u) failed; "
		        "err(%d)\n",
		        d->cfg.homi_id, err);
		return err;
	}
	if (info.nctrlrs == 0) {
		fprintf(stderr,
		        "aisio: the homi group (homi_id=%u) serves no device\n",
		        d->cfg.homi_id);
		return -ENODEV;
	}
	if (info.nctrlrs > MAX_DEVICES) {
		fprintf(stderr,
		        "aisio: the homi group (homi_id=%u) serves %u devices; "
		        "the driver takes at most %d\n",
		        d->cfg.homi_id, info.nctrlrs, MAX_DEVICES);
		return -EINVAL;
	}

	for (uint32_t i = 0; i < info.nctrlrs; i++) {
		snprintf(d->devices[i].dev_uri, sizeof(d->devices[i].dev_uri),
		         "%s", info.ctrlrs[i]);
	}
	d->n_devices = (int)info.nctrlrs;

	return 0;
}

static void
xal_close_leases(struct driver *d)
{
	for (int i = 0; i < d->n_devices; i++) {
		if (d->devices[i].xal) {
			xal_close(d->devices[i].xal);
			d->devices[i].xal = NULL;
		}
	}
}

/* Open xal on each device's mount in the read-lease mode: the driver maps a
 * registered file itself, under a lease on its descriptor, and reads no
 * index. */
static int
xal_open_leases(struct driver *d)
{
	struct xal_opts opts;
	int err;

	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];

		err = aisio_config_mount(di, d->n_devices, dev->mount,
		                         sizeof(dev->mount));
		if (err < 0) {
			xal_close_leases(d);
			return err;
		}
		memset(&opts, 0, sizeof(opts));
		opts.watch_mode = XAL_WATCHMODE_READ_LEASE;
		opts.mountpoint = dev->mount;
		err = xal_open_from_uri(dev->mount, &dev->xal, &opts);
		if (err < 0) {
			fprintf(stderr,
			        "aisio: xal_open_from_uri(%s) failed; "
			        "err(%d)\n",
			        dev->mount, err);
			xal_close_leases(d);
			return err;
		}
	}

	return 0;
}

opends_error_t
opends_driver_open(void)
{
	if (drv)
		return opends_err(OPENDS_DRIVER_ALREADY_OPEN);

	struct driver *d = calloc(1, sizeof(*d));
	if (!d)
		return opends_err(OPENDS_INTERNAL_ERROR);

	/* The thread-count default follows the device count, so discovery
	 * comes before the rest of the environment. */
	if (discover_devices(d) < 0 ||
	    aisio_config_read(&d->cfg, d->n_devices) < 0) {
		free(d);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}
	pthread_mutex_init(&d->submit_lock, NULL);
	pthread_mutex_init(&d->reg_lock, NULL);
	pthread_mutex_init(&d->alloc_lock, NULL);
	pthread_mutex_init(&d->write_buf_lock, NULL);

	int rc = xal_open_leases(d);
	if (rc < 0) {
		pthread_mutex_destroy(&d->write_buf_lock);
		pthread_mutex_destroy(&d->alloc_lock);
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}

	drv = d;

	int orc = open_devices(d);
	if (orc < 0) {
		xal_close_leases(d);
		pthread_mutex_destroy(&d->write_buf_lock);
		pthread_mutex_destroy(&d->alloc_lock);
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		drv = NULL;
		return opends_err(OPENDS_DEVICE_NOT_FOUND);
	}
	int arc = workers_setup(d);
	if (arc != 0) {
		fprintf(stderr, "aisio: workers_setup failed\n");
		close_devices(d);
		xal_close_leases(d);
		pthread_mutex_destroy(&d->write_buf_lock);
		pthread_mutex_destroy(&d->alloc_lock);
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

	workers_teardown(drv);

	for (int i = 0; i < drv->buf_count; i++) {
		struct buf_entry *e = &drv->bufs[i];
		if (e->owned)
			buf_free_locked(drv, (void *)e->base);
	}
	drv->buf_count = 0;
	runs_clear(drv);

	close_devices(drv);

	xal_close_leases(drv);

	pthread_mutex_destroy(&drv->write_buf_lock);
	pthread_mutex_destroy(&drv->alloc_lock);
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
	props->major_version = OPENDS_VERSION_MAJOR;
	props->minor_version = OPENDS_VERSION_MINOR;
	props->max_direct_io_size = drv->min_mdts_nbytes;
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
		*major = OPENDS_VERSION_MAJOR;
	if (minor)
		*minor = OPENDS_VERSION_MINOR;
	if (patch)
		*patch = OPENDS_VERSION_PATCH;
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Handle registration                                               */
/* ------------------------------------------------------------------ */

opends_error_t
opends_handle_register(opends_handle_t *fh, int fd)
{
	struct nvme_device *dev = NULL;
	int rc;

	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh)
		return opends_err(OPENDS_INVALID_VALUE);

	int oflags = fcntl(fd, F_GETFL);
	if (oflags < 0)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!(oflags & O_DIRECT))
		return opends_err(OPENDS_DIO_NOT_SET);

	struct registered_file *h = calloc(1, sizeof(*h));
	if (!h)
		return opends_err(OPENDS_INTERNAL_ERROR);
	h->fd = fd;
	h->oflags = oflags;
	pthread_mutex_init(&h->lock, NULL);

	/* A read lease for a read-only descriptor, a write lease otherwise, on
	 * the device whose mount holds the file: xal says -EXDEV for the
	 * others. -EAGAIN: the file is open elsewhere. -EBUSY: its open file is
	 * registered already. */
	rc = -EXDEV;
	for (int i = 0; i < drv->n_devices && rc == -EXDEV; i++) {
		dev = &drv->devices[i];
		rc = xal_extents_attach_fd(dev->xal, NULL, fd, &h->lease);
	}
	if (rc < 0) {
		fprintf(stderr,
		        "aisio: xal_extents_attach_fd(fd=%d) failed; "
		        "err(%d)%s\n",
		        fd, rc,
		        rc == -EAGAIN ? " (the file is open elsewhere)" : "");
		pthread_mutex_destroy(&h->lock);
		free(h);
		switch (rc) {
		case -EXDEV: return opends_err(OPENDS_DEVICE_NOT_FOUND);
		case -EBUSY:
			return opends_err(OPENDS_HANDLE_ALREADY_REGISTERED);
		case -EACCES: return opends_err(OPENDS_PERMISSION_DENIED);
		default: return opends_err(OPENDS_FS_SETUP_ERROR);
		}
	}
	h->dev = dev;

	*fh = h;
	__atomic_fetch_add(&use_count, 1, __ATOMIC_RELAXED);
	return opends_ok();
}

void
opends_handle_deregister(opends_handle_t fh)
{
	struct registered_file *h = fh;

	if (!h)
		return;
	if (drv && h->lease)
		xal_extents_detach_fd(h->dev->xal, h->lease);
	pthread_mutex_destroy(&h->lock);
	free(h);
	__atomic_fetch_sub(&use_count, 1, __ATOMIC_RELAXED);
}

/* ------------------------------------------------------------------ */
/*  Buffer allocation                                                 */
/* ------------------------------------------------------------------ */

void *
opends_alloc(size_t size)
{
	if (!drv || !mem_dev(drv))
		return NULL;

	pthread_mutex_lock(&drv->reg_lock);
	if (drv->buf_count >= MAX_BUF_ENTRIES) {
		pthread_mutex_unlock(&drv->reg_lock);
		return NULL;
	}

	void *buf = buf_alloc_locked(drv, size);
	if (!buf) {
		pthread_mutex_unlock(&drv->reg_lock);
		return NULL;
	}
	buf_insert(drv, buf, size, true);
	pthread_mutex_unlock(&drv->reg_lock);
	return buf;
}

void
opends_free(void *buf)
{
	if (!drv || !buf)
		return;

	pthread_mutex_lock(&drv->reg_lock);
	struct buf_entry *e = buf_find(drv, buf);
	if (e && e->owned) {
		buf_free_locked(drv, buf);
		buf_remove(drv, e);
	}
	pthread_mutex_unlock(&drv->reg_lock);
}

opends_error_t
opends_buf_register(const void *buf_base, size_t size, int flags)
{
	(void)flags;

	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!mem_dev(drv))
		return opends_err(OPENDS_DEVICE_NOT_FOUND);
	if (!buf_base || !size)
		return opends_err(OPENDS_INVALID_VALUE);
	if (size > UINTPTR_MAX - (uintptr_t)buf_base - drv->chunk_nbytes)
		return opends_err(OPENDS_INVALID_VALUE);

	int err;
	uintptr_t lo, hi;

	pthread_mutex_lock(&drv->reg_lock);
	if (buf_find(drv, buf_base)) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_MEMORY_ALREADY_REGISTERED);
	}
	if (drv->buf_count >= MAX_BUF_ENTRIES) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_INTERNAL_ERROR);
	}

	buf_chunks(drv, buf_base, size, &lo, &hi);
	err = runs_acquire(drv, lo, hi);
	if (err < 0) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);
	}
	buf_insert(drv, buf_base, size, false);
	pthread_mutex_unlock(&drv->reg_lock);
	return opends_ok();
}

opends_error_t
opends_buf_deregister(const void *buf_base)
{
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!mem_dev(drv))
		return opends_err(OPENDS_DEVICE_NOT_FOUND);
	if (!buf_base)
		return opends_err(OPENDS_INVALID_VALUE);

	uintptr_t lo, hi;

	pthread_mutex_lock(&drv->reg_lock);
	struct buf_entry *e = buf_find(drv, buf_base);
	if (!e) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_MEMORY_NOT_REGISTERED);
	}
	if (e->owned) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err(OPENDS_INVALID_VALUE);
	}
	buf_chunks(drv, e->base, e->length, &lo, &hi);
	buf_remove(drv, e);
	runs_release(drv, lo, hi);
	pthread_mutex_unlock(&drv->reg_lock);
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Async I/O                                                         */
/* ------------------------------------------------------------------ */

static opends_error_t
submit_async_op(struct driver *d, bool is_write, opends_handle_t fh,
                void *buf_base, size_t size, off_t file_offset,
                off_t buf_offset, opends_async_future_t *future)
{
	if (!d)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh || !buf_base || !future)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!d->workers_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	future->done = 0;
	future->result = 0;

	/* A write's buffer, while the pool can give one; the I/O thread takes
	 * one later otherwise. Pinning is an accelerator call, so it is this
	 * thread's. */
	struct write_buf *buf = NULL;
	if (is_write && size)
		buf = ds_write_buf_take_or_grow(d);

	struct registered_file *h = (struct registered_file *)fh;
	struct io_worker *w;
	uint32_t head;

	pthread_mutex_lock(&d->submit_lock);
	struct file_op *op = claim_slot_locked(d, h->dev, &w, &head);
	op->mode = FILE_OP_ASYNC;
	op->is_write = is_write;
	op->h = h;
	op->buf_base = buf_base;
	op->err = 0;
	if (is_write) {
		ds_write_init(&op->wr, (const uint8_t *)buf_base + buf_offset,
		              size, file_offset, buf_offset);
		op->wr.buf = buf;
	}
	op->u.async.size = size;
	op->u.async.file_offset = file_offset;
	op->u.async.buf_offset = buf_offset;
	op->u.async.future = future;
	op->ticket = d->next_ticket++;
	op->planned = false;
	op->admitted = false;
	__atomic_store_n(&op->state, FILE_OP_PENDING, __ATOMIC_RELEASE);
	__atomic_store_n(&w->queue_head, head + 1, __ATOMIC_RELEASE);
	pthread_mutex_unlock(&d->submit_lock);

	return opends_ok();
}

opends_error_t
opends_async_read(opends_handle_t fh, void *buf_base, size_t size,
                  off_t file_offset, off_t buf_offset,
                  opends_async_future_t *future)
{
	return submit_async_op(drv, false, fh, buf_base, size, file_offset,
	                       buf_offset, future);
}

opends_error_t
opends_async_write(opends_handle_t fh, const void *buf_base, size_t size,
                   off_t file_offset, off_t buf_offset,
                   opends_async_future_t *future)
{
	return submit_async_op(drv, true, fh, (void *)buf_base, size,
	                       file_offset, buf_offset, future);
}

ssize_t
opends_async_await(opends_async_future_t *future)
{
	if (!future)
		return -(ssize_t)OPENDS_INVALID_VALUE;

	while (!__atomic_load_n(&future->done, __ATOMIC_ACQUIRE))
		sched_yield();

	return future->result;
}

/* ------------------------------------------------------------------ */
/*  Stream-ordered I/O                                                */
/* ------------------------------------------------------------------ */

static opends_error_t
classify_accel_failure(struct driver *d, int accel_rc)
{
	ds_accel_ctx_t cur;

	if (ds_accel->ctx_get(&cur) != 0 || cur != d->accel_ctx)
		return opends_err_dev(OPENDS_CONTEXT_MISMATCH, accel_rc);
	return opends_err_dev(OPENDS_INTERNAL_ERROR, accel_rc);
}

/* The gate of a read, and of a write that stages nothing: the submitter
 * publishes 2*seq on arrival and parks the stream until the I/O thread's
 * release. */
static int
gate_enqueue(struct driver *d, struct file_op *op, ds_accel_stream_t cus)
{
	struct opends_stream *s = op->u.stream.opends_stream;
	uint32_t seq = op->u.stream.seq;
	int rc;

	if (d->cfg.assume_aligned_only) {
		return ds_accel->launch_host_func(cus, park_gate_cb, op);
	}
	rc = ds_accel->stream_write_value32(cus, s->gate_dptr, 2 * seq);
	if (rc != 0) {
		return rc;
	}
	return ds_accel->stream_wait_value32_geq(cus, s->gate_dptr,
	                                         2 * seq + 1);
}

static opends_error_t
submit_stream_op(struct driver *d, bool is_write, opends_handle_t fh,
                 void *buf_base, size_t *size_p, off_t *file_offset_p,
                 off_t *buf_offset_p, ssize_t *bytes_p, opends_stream_t stream)
{
	if (!d)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh || !buf_base || !size_p || !file_offset_p || !buf_offset_p ||
	    !bytes_p)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!d->workers_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	ds_accel_stream_t cus = (ds_accel_stream_t)stream;
	struct opends_stream *opends_stream = opends_stream_get(d, cus);
	if (!opends_stream)
		return opends_err(OPENDS_INTERNAL_ERROR);

	/* A write is staged by copies queued below, ahead of the gate, so it
	 * takes the parameters of the call and its buffer now; a full pool
	 * waits for the I/O threads. A source the thread would refuse stages
	 * nothing and fails at the gate, as before. */
	size_t wsize = 0;
	off_t wfoff = 0;
	off_t wboff = 0;
	struct write_buf *buf = NULL;
	opends_op_error_t werr = OPENDS_SUCCESS;
	bool deferred = is_write && !opends_stream->fixed;
	if (deferred) {
		buf = ds_write_buf_take_or_grow(d);
		while (!buf) {
			sched_yield();
			buf = ds_write_buf_take(d);
		}
	} else if (is_write) {
		wsize = *size_p;
		wfoff = *file_offset_p;
		wboff = *buf_offset_p;
		if (wsize && !ds_aisio_range_registered(
		                     d, (uint8_t *)buf_base + wboff, wsize)) {
			werr = OPENDS_MEMORY_NOT_REGISTERED;
		} else if (wsize) {
			buf = ds_write_buf_take_or_grow(d);
			while (!buf) {
				sched_yield();
				buf = ds_write_buf_take(d);
			}
		}
	}

	struct registered_file *h = (struct registered_file *)fh;
	struct io_worker *w;
	uint32_t head;

	pthread_mutex_lock(&d->submit_lock);
	struct file_op *op = claim_slot_locked(d, h->dev, &w, &head);
	uint32_t seq = ++opends_stream->next_seq;
	op->mode = FILE_OP_STREAM;
	op->is_write = is_write;
	op->h = h;
	op->buf_base = buf_base;
	op->err = werr;
	op->u.stream.size_p = size_p;
	op->u.stream.file_offset_p = file_offset_p;
	op->u.stream.buf_offset_p = buf_offset_p;
	op->u.stream.bytes_read_p = bytes_p;
	op->u.stream.opends_stream = opends_stream;
	op->u.stream.seq = seq;
	if (is_write) {
		ds_write_init(&op->wr, (const uint8_t *)buf_base + wboff, wsize,
		              wfoff, wboff);
		op->wr.deferred = deferred;
		op->wr.buf = buf;
		if (buf && !deferred) {
			ds_write_geometry(d, &op->wr);
		}
	}
	op->ticket = d->next_ticket++;
	op->planned = false;
	op->admitted = false;

	/* Order the I/O thread against the user's stream through a per-stream
	 * gate word (strictly monotonic, two phases per op): the submitter
	 * publishes 2*seq on arrival and parks, and the I/O thread's store of
	 * 2*seq+1 releases it once the I/O is done (a read's DMA landed, or a
	 * write's source was staged). A staged write publishes nothing: its
	 * first landed round says the stream has arrived, and it parks at the
	 * release. Do not roll back seq on failure: a reused seq would let the
	 * I/O thread's gate check pass before the stream is ready.
	 *
	 * launch_host_func is faster when no copy kernel is enqueued per op.
	 * Otherwise stream_write/stream_wait is. The gate ops are commands the
	 * GPU runs, so they wait for this context to be scheduled, and another
	 * process using the GPU delays them by orders of magnitude. The
	 * callback runs on the CPU and never waits. A copy kernel waits for
	 * the GPU regardless, so once one is enqueued the callback wins
	 * nothing. Hence the coupling to assume_aligned_only.
	 *
	 * The gate enqueues run under submit_lock so gate values reach the
	 * stream in seq order; an enqueue that blocks (a full stream queue)
	 * stalls all submission. */
	int accel_rc;
	if (deferred) {
		accel_rc = ds_stream_write_enqueue_deferred(d, op, cus);
		if (accel_rc != 0 && !op->wr.copies) {
			/* no kernel runs: nothing to drain */
			ds_write_buf_put(d, buf);
			pthread_mutex_unlock(&d->submit_lock);
			return classify_accel_failure(d, accel_rc);
		}
		if (accel_rc != 0) {
			op->err = OPENDS_INTERNAL_ERROR;
		}
	} else if (is_write && op->wr.rounds) {
		accel_rc = ds_stream_write_enqueue(d, op, cus);
		if (accel_rc != 0) {
			/* the rounds already queued run: the I/O thread drains
			 * them and returns the buffer */
			op->err = OPENDS_INTERNAL_ERROR;
		}
	} else {
		accel_rc = gate_enqueue(d, op, cus);
		if (accel_rc != 0) {
			pthread_mutex_unlock(&d->submit_lock);
			return classify_accel_failure(d, accel_rc);
		}
	}

	__atomic_store_n(&op->state, FILE_OP_PENDING, __ATOMIC_RELEASE);
	__atomic_store_n(&w->queue_head, head + 1, __ATOMIC_RELEASE);
	pthread_mutex_unlock(&d->submit_lock);
	if (accel_rc != 0)
		return classify_accel_failure(d, accel_rc);

	/* Reads enqueue the deferred tail copy (writes run none): offsets
	 * resolve behind the gate, so the copy size is unknown here, and
	 * copy_stream no-ops when it is zero. Enqueue after publishing so a
	 * failed enqueue is still drained by the I/O thread (which releases the
	 * gate); only this read is lost. */
	if (!d->cfg.assume_aligned_only && !is_write) {
		accel_rc = ds_accel->copy_stream(opends_stream->bounce_desc_dev,
		                                 cus);
		if (accel_rc != 0)
			return classify_accel_failure(d, accel_rc);
	}

	return opends_ok();
}

opends_error_t
opends_stream_read(opends_handle_t fh, void *buf_base, size_t *size_p,
                   off_t *file_offset_p, off_t *buf_offset_p,
                   ssize_t *bytes_read_p, opends_stream_t stream)
{
	return submit_stream_op(drv, false, fh, buf_base, size_p, file_offset_p,
	                        buf_offset_p, bytes_read_p, stream);
}

opends_error_t
opends_stream_write(opends_handle_t fh, void *buf_base, size_t *size_p,
                    off_t *file_offset_p, off_t *buf_offset_p,
                    ssize_t *bytes_written_p, opends_stream_t stream)
{
	return submit_stream_op(drv, true, fh, buf_base, size_p, file_offset_p,
	                        buf_offset_p, bytes_written_p, stream);
}

opends_error_t
opends_stream_register(opends_stream_t stream, unsigned flags)
{
	const unsigned fixed = OPENDS_STREAM_FIXED_BUF_OFFSET |
	                       OPENDS_STREAM_FIXED_FILE_OFFSET |
	                       OPENDS_STREAM_FIXED_FILE_SIZE;

	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!stream)
		return opends_err(OPENDS_INVALID_VALUE);

	if (!drv->workers_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	ds_accel_stream_t cus = (ds_accel_stream_t)stream;

	pthread_mutex_lock(&drv->reg_lock);
	int idx = ds_stream_map_get(drv->stream_map, STREAM_MAP_MASK, cus);
	if (idx >= 0) {
		/* The entry outlives a deregistration, and the runtime hands
		 * out a destroyed stream's handle again, so the flags of the
		 * new registration apply. */
		drv->streams[idx].fixed = (flags & fixed) == fixed;
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
	opends_stream->fixed = (flags & fixed) == fixed;

	int rc = stream_bounce_alloc(opends_stream, drv);
	if (rc != 0) {
		pthread_mutex_unlock(&drv->reg_lock);
		return opends_err_dev(OPENDS_INTERNAL_ERROR, rc);
	}

	if (ds_stream_map_put(drv->stream_map, STREAM_MAP_MASK, cus, n) < 0) {
		stream_bounce_free(opends_stream, drv);
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
