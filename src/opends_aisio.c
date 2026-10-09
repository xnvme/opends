/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends_aisio.c - aisio backend for raw-NVMe direct storage.
 *
 * Reads go straight from an NVMe device into GPU memory via xNVMe's upcie-cuda
 * backend (PCIe P2P DMA). The HOMI server (xnvme's "homi serve") holds the
 * controllers up and serves them over its control plane; this driver attaches
 * as a client, takes its device set from the server's runtime, and allocates
 * its own I/O queues. A registered file's
 * extents come from a per-device xal index that xal-server publishes over
 * POSIX shared memory; the index is in byte units, and this driver converts
 * to LBAs with its own device geometry. The index whose mountpoint covers a
 * file's path also binds the file to its device.
 *
 * Requires: libxnvme and the CUDA toolkit. The NVMe kernel driver must be
 * unbound from the target device before opends_driver_open runs.
 *
 * Writes go through the kernel-mounted filesystem: the source is staged to a
 * host buffer and pwritten via the fd, so XFS over qublk allocates blocks and
 * writes the data; the file's extents are then re-resolved for later P2P reads.
 *
 * Stream reads have two engines. By default an I/O thread issues the NVMe
 * commands when the user's stream reaches the op, reading the size and
 * offsets then. With OPENDS_AISIO_GPU_INITIATED=1 the GPU issues them itself
 * from GPU-resident queues on a stream registered with
 * OPENDS_STREAM_FIXED_SHAPE: those flags let the commands be built when the
 * op is submitted, and a kernel on the user's stream submits and reaps them.
 */

#define _GNU_SOURCE

#include "ds_accel.h"
#include "ds_stream_map.h"
#include "ds_bounce_kernel.h"
#include "opends_internal.h"
#include "ds_extent.h"
#include "ds_gpu_io.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <libxal.h>
#include <libxnvme.h>

#define ENV_XAL_SHM "OPENDS_XAL_SHM"
#define ENV_IO_THREADS "OPENDS_AISIO_IO_THREADS"
#define ENV_QUEUE_DEPTH "OPENDS_AISIO_QUEUE_DEPTH"
#define ENV_CPU_MASK "OPENDS_AISIO_CPU_MASK"
#define ENV_ASSUME_ALIGNED_ONLY "OPENDS_AISIO_ASSUME_ALIGNED_ONLY"
#define ENV_IDLE_SPIN "OPENDS_AISIO_IDLE_SPIN"
#define ENV_HOMI_ID "OPENDS_AISIO_HOMI_ID"
#define ENV_HOST_HEAP_MB "OPENDS_AISIO_HOST_HEAP_MB"
#define ENV_DEVICE_HEAP_MB "OPENDS_AISIO_DEVICE_HEAP_MB"
#define ENV_CQ_MIRROR "OPENDS_AISIO_CQ_MIRROR"
#define ENV_GPU_INITIATED "OPENDS_AISIO_GPU_INITIATED"
#define ENV_GPU_QUEUE_DEPTH "OPENDS_AISIO_GPU_QUEUE_DEPTH"
#define ENV_GPU_CTXS "OPENDS_AISIO_GPU_CTXS"
#define ENV_GPU_QUEUES_PER_OP "OPENDS_AISIO_GPU_QUEUES_PER_OP"
#define ENV_GPU_MAX_CMDS "OPENDS_AISIO_GPU_MAX_CMDS"
#define ENV_GPU_SQ_HOSTMEM "OPENDS_AISIO_GPU_SQ_HOSTMEM"
#define DEFAULT_XAL_SHM_FMT "/xal_dev%d"
#define DEFAULT_HOMI_ID 1
#define DEFAULT_IO_THREADS 2
#define MAX_IO_THREADS 15
/* A device with no worker cannot be read, so the workers bound the set. */
#define MAX_DEVICES MAX_IO_THREADS
#define MAX_BUF_ENTRIES 8192
#define MAX_RESULT_RANGES 64
#define DEFAULT_BOUNCE_SIZE (128 * 1024)
#define NVME_MAX_NLB 65536
#define NVME_PRP_OFFSET_ALIGN 4
#define BOUNCE_SLOTS 2
#define DEFAULT_QUEUE_DEPTH 8
#define MAX_QUEUE_DEPTH 4096

/* GPU-initiated reads. A GPU queue is one CUDA block, so its depth is bounded
 * by the block size, and a submission queue in device memory must fit a 64 KiB
 * device page at depth + 1 entries. The controller's memory page is 4 KiB, and
 * every command gets one page for its PRP list, so an op may span at most
 * gpu_max_cmds chunks of up to 512 pages each. */
#define DEFAULT_GPU_QUEUE_DEPTH 256
#define MAX_GPU_QUEUE_DEPTH 1023
#define DEFAULT_GPU_CTXS 4
#define MAX_GPU_CTXS 32
#define DEFAULT_GPU_MAX_CMDS 2048
#define MAX_GPU_MAX_CMDS 65536
#define NVME_PAGE_SIZE 4096
#define NVME_PRP_LIST_ENTRIES (NVME_PAGE_SIZE / sizeof(uint64_t))
/* Device memory is exported to the controller in 64 KiB pages, each physically
 * contiguous, and xNVMe's translation table resolves at that granule. */
#define GPU_PHYS_GRANULE 65536
#define GPU_DRAIN_TIMEOUT_NS (10ull * 1000000000ull)

/* The host DMA heap holds this process's own SQ/CQ rings and PRP lists, one set
 * per I/O thread. The heap is process-wide and the thread count does not grow
 * with the device count, so 256 MiB covers the largest configuration the knobs
 * allow (MAX_IO_THREADS queues at MAX_QUEUE_DEPTH). xNVMe would otherwise
 * default to 1 GiB, which does not multiply across the processes sharing the
 * hugepages. */
#define DEFAULT_HOST_HEAP_MB 256
#define MAX_HEAP_MB (64 * 1024)
#define DEFAULT_IDLE_SPIN_US 200
#define MAX_IDLE_SPIN_US 1000000
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

/* A block from opends_result_alloc: pinned host memory the GPU can write. */
struct result_range {
	void *host;
	ds_accel_devptr_t dptr;
	size_t size;
};

struct nvme_device;

struct registered_file {
	int fd;
	int oflags;
	struct nvme_device *dev;
};

struct opends_stream {
	uint32_t *gate;
	ds_accel_devptr_t gate_dptr;
	uint32_t next_seq;
	void *bounce_buf;
	struct ds_bounce_copy *bounce_desc_host;
	ds_accel_devptr_t bounce_desc_dev;
	unsigned flags; ///< opends_stream_register flags
	bool aligned;   ///< Sub-LBA tails rejected (flag or the global knob)
};

enum file_op_mode {
	FILE_OP_STREAM,
	FILE_OP_ASYNC,
};

struct file_op_stream {
	size_t *size_p;
	off_t *file_offset_p;
	off_t *buf_offset_p;
	ssize_t *bytes_read_p;
	struct opends_stream *opends_stream;
	uint32_t seq;
};

struct file_op_async {
	size_t size;
	off_t file_offset;
	off_t buf_offset;
	opends_async_future_t *future;
};

struct bounce_slot {
	void *dst;
	void *src;
	size_t nbytes;
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
	struct bounce_slot bounces[BOUNCE_SLOTS];
	int n_bounces;
	void *bounce_buf;
	union {
		struct file_op_stream stream;
		struct file_op_async async;
	} u;
};

struct read_cursor {
	struct xnvme_queue *queue;
	uint64_t cur_slba;
	uint8_t *abs_dst;
	size_t remaining;
};

struct driver;

/* One GPU-initiated op in flight: its queues, the command array and PRP lists
 * the host builds, and the bounce slot for a sub-LBA tail. Claimed from the
 * submitting thread, returned by the callback that runs behind the kernel. */
struct gpu_ctx {
	struct nvme_device *dev;
	int busy;
	bool broken; ///< A timed-out round left the queue state unknown
	bool fast;   ///< The kernel publishes the result; no callback behind it
	int nq;
	struct xnvme_cuda_queue *queues[DS_GPU_MAX_BLOCKS];
	struct ds_gpu_op *op; ///< Host-mapped
	ds_accel_devptr_t op_dptr;
	struct xnvme_spec_cmd *cmds; ///< Host-mapped, read by the kernel
	uint64_t *prp_stage;         ///< Host-mapped, one page per command
	void *prp_dev;               ///< Device heap, same layout as prp_stage
	uint64_t *prp_phys;          ///< Physical address of each prp_dev page
	void *bounce;                ///< Device heap, one LBA
	uint32_t prp_pages;          ///< Pages of prp_stage the op uses
	size_t bytes_total;
	ssize_t *bytes_p;
};

struct io_worker {
	struct driver *drv;
	struct nvme_device *dev;
	struct xnvme_queue *queue;
	struct file_op file_op_queue[FILE_OP_QUEUE_SIZE];
	uint32_t queue_head;
	uint32_t queue_tail;
	pthread_t thread;
};

struct nvme_device {
	char dev_uri[XNVME_IDENT_URI_LEN];
	struct xal *xal; ///< Attach to the index xal-server publishes
	struct xnvme_dev *xdev;
	uint32_t nsid;
	uint32_t lba_size;
	uint32_t lba_shift;
	uint32_t mdts_nbytes;
	struct io_worker *workers;
	int n_workers;
	uint32_t rr_next;
	struct gpu_ctx *gpu_ctxs;
	int n_gpu_ctxs;
};

struct driver {
	struct nvme_device devices[MAX_DEVICES];
	int n_devices;
	uint32_t homi_id; ///< Multi-process group the HOMI server is primary of
	size_t host_heap_nbytes;
	size_t device_heap_nbytes;
	uint32_t max_lba_size;    ///< Largest over the devices
	uint32_t min_mdts_nbytes; ///< Smallest over the devices
	struct buf_entry bufs[MAX_BUF_ENTRIES];
	int buf_count;
	struct result_range results[MAX_RESULT_RANGES];
	int n_results;

	bool workers_ready;
	ds_accel_ctx_t accel_ctx;

	struct opends_stream streams[MAX_STREAMS];
	int n_streams;
	void *stream_words_host;
	ds_accel_devptr_t stream_words_dptr;

	struct ds_stream_map_entry stream_map[STREAM_MAP_SIZE];

	int n_io_threads;
	uint32_t queue_depth;
	uint32_t idle_spin_us;
	bool busy_spin;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	bool cq_mirror; ///< CQ in GPU memory, warp-mirrored to host (upcie-cuda
	                ///< only)
	bool gpu_initiated; ///< Stream reads issued by the GPU
	bool gpu_ready;
	bool gpu_sq_hostmem;
	uint32_t gpu_queue_depth;
	uint32_t gpu_max_cmds;
	int gpu_ctxs;
	int gpu_queues_per_op;
	pthread_mutex_t submit_lock;
	pthread_mutex_t reg_lock;
	pthread_mutex_t alloc_lock;
	bool stop;
};

static struct driver *drv;
static long use_count;

static int gpu_pool_setup(struct driver *d);
static void gpu_pool_teardown(struct driver *d);

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

static void
async_future_complete(opends_async_future_t *fut, ssize_t result)
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

#define ESTALE_RETRIES 6000
#define ESTALE_BACKOFF_US 50000

/* True while the seqlock snapshot taken at `seq` is still valid: the server
 * is not rewriting the shared pools, they did not move since `seq` was read,
 * and the filesystem is not dirty (changed but not yet re-indexed). */
static bool
snapshot_held(struct xal *xal, int seq)
{
	int cur;
	bool dirty;

	atomic_thread_fence(memory_order_acquire);
	cur = xal_get_seq_lock(xal);
	dirty = xal_is_dirty(xal);
	return !(seq & 1) && cur == seq && !dirty;
}

/* Copy the extents for `path` out of the shared index. The server rewrites
 * the shared pools in place, concurrent with this read, so validate the
 * snapshot before trusting anything read from them. -ESTALE means retry. */
static int
extents_snapshot(struct nvme_device *dev, char *path, struct ds_extent **out,
                 uint32_t *out_n)
{
	struct xal *xal = dev->xal;
	struct xal_extents *xe = NULL;
	struct ds_extent *ex = NULL;
	uint32_t n = 0, base;
	bool held;
	int seq, rc;

	seq = xal_get_seq_lock(xal);
	held = snapshot_held(xal, seq);
	if (!held)
		return -ESTALE;

	rc = xal_get_extents(xal, path, &xe);
	if (rc < 0)
		goto out;

	/* xe points into the shared inode pool, so count/extent_idx may be
	 * torn. Capture them, then validate before using them as allocation
	 * size and pool indices. */
	n = xe->count;
	base = xe->extent_idx;
	held = snapshot_held(xal, seq);
	if (!held)
		return -ESTALE;

	ex = calloc(n ? n : 1, sizeof(*ex));
	if (!ex)
		return -ENOMEM;

	for (uint32_t i = 0; i < n; i++) {
		struct xal_extent *e = xal_extent_at(xal, base + i);
		struct xal_extent_converted b = {0};

		rc = xal_extent_in_bytes(xal, e, &b);
		if (rc < 0)
			goto out;
		/* xal-server never opens the device, so its index carries no
		 * LBA size; convert with our own geometry. */
		if (b.start_block & (uint64_t)(dev->lba_size - 1)) {
			rc = -EIO;
			goto out;
		}
		ex[i].file_offset = b.start_offset;
		ex[i].slba = b.start_block >> dev->lba_shift;
		ex[i].length = b.size;
	}

out:
	/* A torn read during a rewrite can surface as a spurious error, and a
	 * clean result may hold torn extents; only trust either if the
	 * snapshot held. */
	held = snapshot_held(xal, seq);
	if (!held) {
		free(ex);
		return -ESTALE;
	}
	if (rc < 0) {
		free(ex);
		return rc;
	}
	*out = ex;
	*out_n = n;
	return 0;
}

static int
resolve_extents(struct registered_file *h, struct ds_extent **out,
                uint32_t *out_n)
{
	char fdpath[64], path[PATH_MAX];
	ssize_t plen;
	int rc;

	snprintf(fdpath, sizeof(fdpath), "/proc/self/fd/%d", h->fd);
	plen = readlink(fdpath, path, sizeof(path) - 1);
	if (plen < 0)
		return -errno;
	path[plen] = '\0';

	/* A write marks the index dirty (dispatch_write), and the server's
	 * watcher re-indexes on filesystem events. Retry until a clean
	 * snapshot resolves. */
	rc = -ESTALE;
	for (int attempt = 0; attempt < ESTALE_RETRIES; attempt++) {
		rc = extents_snapshot(h->dev, path, out, out_n);
		if (rc != -ESTALE)
			break;
		usleep(ESTALE_BACKOFF_US);
	}
	return rc;
}

static ssize_t
pwrite_op(struct registered_file *h, const void *src, size_t size,
          off_t file_offset)
{
	return opends_direct_pwrite(h->fd, h->oflags, src, size, file_offset,
	                            ds_accel->copy);
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
	opts.homi_id = d->homi_id;
	opts.host_heap_size = d->host_heap_nbytes;
	opts.device_heap_size = d->device_heap_nbytes;

	dev->xdev = xnvme_dev_open(dev->dev_uri, &opts);
	if (!dev->xdev) {
		fprintf(stderr,
		        "aisio open_device: xnvme_dev_open(%s, be=%s, "
		        "homi_id=%u) failed\n",
		        dev->dev_uri, opts.be, d->homi_id);
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

	return 0;
}

static void
close_devices(struct driver *d)
{
	for (int i = 0; i < d->n_devices; i++) {
		if (d->devices[i].xdev) {
			xnvme_dev_close(d->devices[i].xdev);
			d->devices[i].xdev = NULL;
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
	bool aligned = op->mode == FILE_OP_STREAM
	                       ? op->u.stream.opends_stream->aligned
	                       : w->drv->assume_aligned_only;

	if (aligned) {
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

static void
start_read_op(struct io_worker *w, struct file_op *op)
{
	struct nvme_device *dev = w->dev;

	if (dev->lba_size == 0) {
		op->err = OPENDS_INTERNAL_ERROR;
		return;
	}
	if ((dev->mdts_nbytes >> dev->lba_shift) == 0) {
		op->err = OPENDS_INVALID_VALUE;
		return;
	}

	size_t size;
	uint64_t req_start;
	uint8_t *dst_base;
	if (op->mode == FILE_OP_STREAM) {
		size = *op->u.stream.size_p;
		req_start = (uint64_t)*op->u.stream.file_offset_p;
		dst_base = (uint8_t *)op->buf_base + *op->u.stream.buf_offset_p;
	} else {
		size = op->u.async.size;
		req_start = (uint64_t)op->u.async.file_offset;
		dst_base = (uint8_t *)op->buf_base + op->u.async.buf_offset;
	}
	if (size > UINT64_MAX - req_start) {
		op->err = OPENDS_INVALID_VALUE;
		return;
	}
	uint64_t req_end = req_start + size;

	struct ds_extent *extents;
	uint32_t extent_count;
	int frc = resolve_extents(op->h, &extents, &extent_count);
	if (frc < 0) {
		op->err = OPENDS_FS_SETUP_ERROR;
		return;
	}

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

		size_t head_off = off_in_ext & lba_mask;
		if (head_off) {
			size_t head_bytes = dev->lba_size - head_off;

			if (head_bytes > remaining)
				head_bytes = remaining;
			if (remaining - head_bytes >= dev->lba_size &&
			    (head_bytes & (NVME_PRP_OFFSET_ALIGN - 1))) {
				op->err = OPENDS_INVALID_VALUE;
				goto out;
			}
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

static void
complete_read_op(struct file_op *op)
{
	ssize_t n = op->err ? -(ssize_t)op->err : (ssize_t)op->bytes_acc;

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

	async_future_complete(op->u.async.future, n);
	__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
}

static void
dispatch_write(struct file_op *op)
{
	const void *src;
	size_t size;
	off_t file_offset;
	if (op->mode == FILE_OP_STREAM) {
		src = (const uint8_t *)op->buf_base +
		      *op->u.stream.buf_offset_p;
		size = *op->u.stream.size_p;
		file_offset = *op->u.stream.file_offset_p;
	} else {
		src = (const uint8_t *)op->buf_base + op->u.async.buf_offset;
		size = op->u.async.size;
		file_offset = op->u.async.file_offset;
	}

	ssize_t n = pwrite_op(op->h, src, size, file_offset);
	if (n >= 0)
		xal_mark_dirty(op->h->dev->xal);
	if (n < 0)
		n = (n == -(ssize_t)EINVAL)
		            ? -(ssize_t)OPENDS_INVALID_VALUE
		            : -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR;

	if (op->mode == FILE_OP_STREAM) {
		*op->u.stream.bytes_read_p = n;
		release_gate(op->u.stream.opends_stream, op->u.stream.seq);
	} else {
		async_future_complete(op->u.async.future, n);
	}
	__atomic_store_n(&op->state, FILE_OP_FREE, __ATOMIC_RELEASE);
}

static void
dispatch_pending(struct io_worker *w, struct file_op *op)
{
	if (op->is_write) {
		dispatch_write(op);
		return;
	}
	op->chunks_remaining = 0;
	op->bounces_outstanding = 0;
	op->bytes_acc = 0;
	op->err = 0;
	op->n_bounces = 0;
	__atomic_store_n(&op->state, FILE_OP_IN_FLIGHT, __ATOMIC_RELEASE);
	start_read_op(w, op);
}

static void
reap_in_flight(struct file_op *op)
{
	if (op->chunks_remaining == 0 && op->bounces_outstanding == 0)
		complete_read_op(op);
}

/* Service a PENDING async op: dispatch it once its stream gate has opened,
 * otherwise leave it for a later pass. Returns true while still gated. The gate
 * is a free-running +1 counter (the submitter's arrival publish then this
 * thread's release each tick it by one), so compare with serial/cyclic
 * arithmetic to match the device-side GEQ wait, keeping the sequence
 * wrap-safe. */
static bool
poll_stream_pending(struct io_worker *w, struct file_op *op)
{
	uint32_t gate = __atomic_load_n(op->u.stream.opends_stream->gate,
	                                __ATOMIC_ACQUIRE);

	if ((int32_t)(gate - 2 * op->u.stream.seq) < 0)
		return true;
	dispatch_pending(w, op);
	return false;
}

static void *
io_thread_main(void *arg)
{
	struct io_worker *w = arg;
	struct driver *d = w->drv;
	uint64_t idle_spin_ns = (uint64_t)d->idle_spin_us * 1000;
	uint64_t spin_until_ns = 0;
	bool busy_spin = d->busy_spin;
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
					dispatch_pending(w, op);
					break;
				case FILE_OP_STREAM:
					if (poll_stream_pending(w, op))
						busy = true;
					break;
				}
				break;
			case FILE_OP_IN_FLIGHT: reap_in_flight(op); break;
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
	int base = d->n_io_threads / d->n_devices;

	return base + (di < d->n_io_threads % d->n_devices ? 1 : 0);
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
		for (int i = 0; i < dev->n_workers; i++) {
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
	int qopts = d->cq_mirror ? XNVME_QUEUE_P2P_CQ_MIRROR : 0;
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

	d->stop = false;
	int mask_cpus = __builtin_popcountll(d->cpu_mask);
	int pinned = 0;
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
			if (xnvme_queue_init(dev->xdev, d->queue_depth, qopts,
			                     &w->queue) < 0) {
				w->queue = NULL;
				goto fail;
			}
			pthread_attr_t attr;
			pthread_attr_t *attrp = NULL;
			if (mask_cpus) {
				cpu_set_t set;
				CPU_ZERO(&set);
				CPU_SET(mask_nth_cpu(d->cpu_mask,
				                     pinned % mask_cpus),
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
			if (rc != 0) {
				xnvme_queue_term(w->queue);
				w->queue = NULL;
				goto fail;
			}
			pthread_setname_np(w->thread, "aisio-io");
			pinned++;
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

static int
env_int(const char *name, int def, int lo, int hi, int *out)
{
	const char *v = getenv(name);
	if (!v || !v[0]) {
		*out = def;
		return 0;
	}
	char *end;
	long n = strtol(v, &end, 10);
	if (end == v || *end) {
		fprintf(stderr, "aisio: %s=%s is not a number\n", name, v);
		return -EINVAL;
	}
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
	int thr_def;

	/* One worker per device at least, since a device with no worker cannot
	 * be read. The threads are the total, not a per-device count. */
	thr_def = DEFAULT_IO_THREADS < d->n_devices ? d->n_devices
	                                            : DEFAULT_IO_THREADS;
	if (env_int(ENV_IO_THREADS, thr_def, 1, MAX_IO_THREADS, &n) < 0)
		return -EINVAL;
	if (n < d->n_devices) {
		fprintf(stderr,
		        "aisio: %s=%d leaves some of the %d devices without a "
		        "worker\n",
		        ENV_IO_THREADS, n, d->n_devices);
		return -EINVAL;
	}
	d->n_io_threads = n;

	if (env_int(ENV_QUEUE_DEPTH, DEFAULT_QUEUE_DEPTH, 1, MAX_QUEUE_DEPTH,
	            &n) < 0)
		return -EINVAL;
	d->queue_depth = (uint32_t)n;

	const char *spin = getenv(ENV_IDLE_SPIN);
	d->busy_spin = spin && !strcmp(spin, "busy");
	if (d->busy_spin) {
		d->idle_spin_us = DEFAULT_IDLE_SPIN_US;
	} else {
		if (env_int(ENV_IDLE_SPIN, DEFAULT_IDLE_SPIN_US, 0,
		            MAX_IDLE_SPIN_US, &n) < 0) {
			fprintf(stderr,
			        "aisio: %s takes microseconds, or \"busy\"\n",
			        ENV_IDLE_SPIN);
			return -EINVAL;
		}
		d->idle_spin_us = (uint32_t)n;
	}

	const char *mask = getenv(ENV_CPU_MASK);
	d->cpu_mask = mask && mask[0] ? strtoull(mask, NULL, 0) : 0;

	const char *aligned = getenv(ENV_ASSUME_ALIGNED_ONLY);
	d->assume_aligned_only = aligned && aligned[0] && aligned[0] != '0';

	const char *cqm = getenv(ENV_CQ_MIRROR);
	d->cq_mirror = cqm && cqm[0] && cqm[0] != '0';

	const char *gpu = getenv(ENV_GPU_INITIATED);
	d->gpu_initiated = gpu && gpu[0] && gpu[0] != '0';
	if (env_int(ENV_GPU_QUEUE_DEPTH, DEFAULT_GPU_QUEUE_DEPTH, 1,
	            MAX_GPU_QUEUE_DEPTH, &n) < 0)
		return -EINVAL;
	d->gpu_queue_depth = (uint32_t)n;
	if (env_int(ENV_GPU_CTXS, DEFAULT_GPU_CTXS, 1, MAX_GPU_CTXS, &n) < 0)
		return -EINVAL;
	d->gpu_ctxs = n;
	if (env_int(ENV_GPU_QUEUES_PER_OP, 1, 1, DS_GPU_MAX_BLOCKS, &n) < 0)
		return -EINVAL;
	d->gpu_queues_per_op = n;
	if (env_int(ENV_GPU_MAX_CMDS, DEFAULT_GPU_MAX_CMDS, 1, MAX_GPU_MAX_CMDS,
	            &n) < 0)
		return -EINVAL;
	d->gpu_max_cmds = (uint32_t)n;
	const char *hsq = getenv(ENV_GPU_SQ_HOSTMEM);
	d->gpu_sq_hostmem = hsq && hsq[0] && hsq[0] != '0';
	if (d->gpu_initiated &&
	    (!ds_accel->launch_host_func || !ds_accel->copy_async ||
	     !ds_accel->gpu_io_launch)) {
		fprintf(stderr,
		        "aisio: %s=1 needs launch_host_func, copy_async and "
		        "gpu_io_launch, which the vendor ops table does not "
		        "provide\n",
		        ENV_GPU_INITIATED);
		return -EINVAL;
	}

	if (env_int(ENV_HOST_HEAP_MB, DEFAULT_HOST_HEAP_MB, 1, MAX_HEAP_MB,
	            &n) < 0)
		return -EINVAL;
	d->host_heap_nbytes = (size_t)n << 20;

	/* 0 leaves the device heap at the xNVMe default; GPU memory is not the
	 * scarce resource the host hugepages are. */
	if (env_int(ENV_DEVICE_HEAP_MB, 0, 0, MAX_HEAP_MB, &n) < 0)
		return -EINVAL;
	d->device_heap_nbytes = (size_t)n << 20;

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
	int n;
	int err;

	if (env_int(ENV_HOMI_ID, DEFAULT_HOMI_ID, 0, INT_MAX, &n) < 0) {
		return -EINVAL;
	}
	d->homi_id = (uint32_t)n;

	err = -ENOENT;
	for (int i = 0; i < ATTACH_RETRIES; i++) {
		err = xnvme_cplane_get_info(d->homi_id, &info);
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
		        d->homi_id, err);
		return err;
	}
	if (info.nctrlrs == 0) {
		fprintf(stderr,
		        "aisio: the homi group (homi_id=%u) serves no device\n",
		        d->homi_id);
		return -ENODEV;
	}
	if (info.nctrlrs > MAX_DEVICES) {
		fprintf(stderr,
		        "aisio: the homi group (homi_id=%u) serves %u devices; "
		        "the driver takes at most %d\n",
		        d->homi_id, info.nctrlrs, MAX_DEVICES);
		return -EINVAL;
	}

	for (uint32_t i = 0; i < info.nctrlrs; i++) {
		snprintf(d->devices[i].dev_uri, sizeof(d->devices[i].dev_uri),
		         "%s", info.ctrlrs[i]);
	}
	d->n_devices = (int)info.nctrlrs;

	return 0;
}

/* OPENDS_XAL_SHM optionally overrides the index names as a comma-separated
 * list, paired with the runtime's device order. Default: /xal_dev<i>. */
static int
xal_shm_name(int di, int n_devices, char *out, size_t out_len)
{
	const char *spec = getenv(ENV_XAL_SHM);
	const char *p;
	const char *end;
	size_t len;
	int n;

	if (!spec || !spec[0]) {
		snprintf(out, out_len, DEFAULT_XAL_SHM_FMT, di);
		return 0;
	}

	n = 1;
	for (p = spec; *p; p++) {
		if (*p == ',') {
			n++;
		}
	}
	if (n != n_devices) {
		fprintf(stderr, "aisio: %s names %d indexes for %d devices\n",
		        ENV_XAL_SHM, n, n_devices);
		return -EINVAL;
	}

	p = spec;
	for (int i = 0; i < di; i++) {
		p = strchr(p, ',') + 1;
	}
	end = strchr(p, ',');
	len = end ? (size_t)(end - p) : strlen(p);
	if (len == 0 || len >= out_len) {
		fprintf(stderr, "aisio: %s entry %d is empty or too long\n",
		        ENV_XAL_SHM, di);
		return -EINVAL;
	}
	memcpy(out, p, len);
	out[len] = '\0';

	return 0;
}

static void
xal_detach(struct driver *d)
{
	for (int i = 0; i < d->n_devices; i++) {
		if (d->devices[i].xal) {
			xal_close(d->devices[i].xal);
			d->devices[i].xal = NULL;
		}
	}
}

/*
 * Attach to the per-device indexes xal-server publishes. -ENOENT (not yet
 * published), -EAGAIN (still being set up) and -ESTALE (first index not
 * finished) all mean "not yet" for a server starting alongside us, so retry
 * on them.
 */
static int
xal_attach(struct driver *d)
{
	char shm[256];
	int err;

	for (int di = 0; di < d->n_devices; di++) {
		err = xal_shm_name(di, d->n_devices, shm, sizeof(shm));
		if (err < 0) {
			xal_detach(d);
			return err;
		}

		err = -ENOENT;
		for (int i = 0; i < ATTACH_RETRIES; i++) {
			err = xal_from_shm(shm, &d->devices[di].xal);
			if (err != -ENOENT && err != -EAGAIN &&
			    err != -ESTALE) {
				break;
			}
			usleep(ATTACH_BACKOFF_US);
		}
		if (err < 0) {
			fprintf(stderr,
			        "aisio: xal_from_shm(%s) failed; err(%d)\n",
			        shm, err);
			xal_detach(d);
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
	if (discover_devices(d) < 0 || read_env_config(d) < 0) {
		free(d);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}
	pthread_mutex_init(&d->submit_lock, NULL);
	pthread_mutex_init(&d->reg_lock, NULL);
	pthread_mutex_init(&d->alloc_lock, NULL);

	int rc = xal_attach(d);
	if (rc < 0) {
		pthread_mutex_destroy(&d->alloc_lock);
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		return opends_err(OPENDS_FS_SETUP_ERROR);
	}

	drv = d;

	int orc = open_devices(d);
	if (orc < 0) {
		xal_detach(d);
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
		xal_detach(d);
		pthread_mutex_destroy(&d->alloc_lock);
		pthread_mutex_destroy(&d->submit_lock);
		pthread_mutex_destroy(&d->reg_lock);
		free(d);
		drv = NULL;
		return opends_err_dev(OPENDS_DEVICE_DRIVER_ERROR, arc);
	}

	if (d->gpu_initiated) {
		int grc = gpu_pool_setup(d);

		if (grc != 0) {
			fprintf(stderr,
			        "aisio: GPU-initiated reads could not be set "
			        "up; "
			        "err(%d)\n",
			        grc);
			workers_teardown(d);
			close_devices(d);
			xal_detach(d);
			pthread_mutex_destroy(&d->alloc_lock);
			pthread_mutex_destroy(&d->submit_lock);
			pthread_mutex_destroy(&d->reg_lock);
			free(d);
			drv = NULL;
			return opends_err_dev(OPENDS_DEVICE_DRIVER_ERROR, grc);
		}
	}

	return opends_ok();
}

opends_error_t
opends_driver_close(void)
{
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);

	gpu_pool_teardown(drv);
	workers_teardown(drv);

	for (int i = 0; i < drv->buf_count; i++) {
		struct buf_entry *e = &drv->bufs[i];
		if (e->owned)
			buf_free_locked(drv, (void *)e->base);
		else
			xnvme_mem_unmap(mem_dev(drv), (void *)e->base);
	}
	drv->buf_count = 0;
	for (int i = 0; i < drv->n_results; i++) {
		ds_accel->host_free(drv->results[i].host);
	}
	drv->n_results = 0;

	close_devices(drv);

	xal_detach(drv);

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
	props->major_version = 0;
	props->minor_version = 1;
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

/*
 * Find the device that serves `fd` by asking each device's index for the
 * file's path. The indexes cover disjoint filesystems, so at most one can
 * answer: -EINVAL means the path is outside that index's mountpoint, any
 * other failure means the index is not settled yet and is worth another
 * pass (a freshly created file appears once the server re-indexes).
 */
static int
bind_device(struct driver *d, int fd, struct nvme_device **out)
{
	char fdpath[64], path[PATH_MAX];
	struct xal_inode *inode;
	ssize_t plen;
	int rc;

	snprintf(fdpath, sizeof(fdpath), "/proc/self/fd/%d", fd);
	plen = readlink(fdpath, path, sizeof(path) - 1);
	if (plen < 0) {
		return -errno;
	}
	path[plen] = '\0';

	for (int attempt = 0; attempt < ESTALE_RETRIES; attempt++) {
		bool retryable = false;

		for (int i = 0; i < d->n_devices; i++) {
			struct nvme_device *dev = &d->devices[i];

			rc = xal_get_inode(dev->xal, path, &inode);
			if (rc == 0) {
				*out = dev;
				return 0;
			}
			if (rc != -EINVAL) {
				retryable = true;
			}
		}
		if (!retryable) {
			break;
		}
		usleep(ESTALE_BACKOFF_US);
	}

	fprintf(stderr,
	        "aisio: %s is not on a filesystem the homi stack serves\n",
	        path);
	return -ENODEV;
}

opends_error_t
opends_handle_register(opends_handle_t *fh, int fd)
{
	struct nvme_device *dev;
	int err;

	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh)
		return opends_err(OPENDS_INVALID_VALUE);

	int oflags = fcntl(fd, F_GETFL);
	if (oflags < 0)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!(oflags & O_DIRECT))
		return opends_err(OPENDS_DIO_NOT_SET);

	/* With one device the extent lookup at I/O time already answers this,
	 * and skipping the probe lets a not-yet-indexed fresh file register. */
	dev = &drv->devices[0];
	if (drv->n_devices > 1) {
		err = bind_device(drv, fd, &dev);
		if (err < 0)
			return opends_err(OPENDS_FS_SETUP_ERROR);
	}

	struct registered_file *h = calloc(1, sizeof(*h));
	if (!h)
		return opends_err(OPENDS_INTERNAL_ERROR);
	h->fd = fd;
	h->oflags = oflags;
	h->dev = dev;

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
			buf_free_locked(drv, buf);
			drv->bufs[i] = drv->bufs[drv->buf_count - 1];
			drv->buf_count--;
			break;
		}
	}
	pthread_mutex_unlock(&drv->reg_lock);
}

void *
opends_result_alloc(size_t size)
{
	void *host = NULL;
	ds_accel_devptr_t dptr = 0;

	if (!drv || !size) {
		return NULL;
	}
	pthread_mutex_lock(&drv->reg_lock);
	if (drv->n_results >= MAX_RESULT_RANGES ||
	    ds_accel->host_alloc_mapped(size, &host, &dptr) != 0) {
		pthread_mutex_unlock(&drv->reg_lock);
		return NULL;
	}
	drv->results[drv->n_results].host = host;
	drv->results[drv->n_results].dptr = dptr;
	drv->results[drv->n_results].size = size;
	drv->n_results++;
	pthread_mutex_unlock(&drv->reg_lock);
	return host;
}

void
opends_result_free(void *p)
{
	if (!drv || !p) {
		return;
	}
	pthread_mutex_lock(&drv->reg_lock);
	for (int i = 0; i < drv->n_results; i++) {
		if (drv->results[i].host == p) {
			ds_accel->host_free(p);
			drv->results[i] = drv->results[drv->n_results - 1];
			drv->n_results--;
			break;
		}
	}
	pthread_mutex_unlock(&drv->reg_lock);
}

/* Device address of a result slot; 0 when p is not in a block from
 * opends_result_alloc or is not aligned for the GPU's store. */
static ds_accel_devptr_t
result_dptr(struct driver *d, const ssize_t *p)
{
	ds_accel_devptr_t r = 0;

	if ((uintptr_t)p & (sizeof(*p) - 1)) {
		return 0;
	}
	pthread_mutex_lock(&d->reg_lock);
	for (int i = 0; i < d->n_results; i++) {
		const struct result_range *rr = &d->results[i];
		uintptr_t off = (uintptr_t)p - (uintptr_t)rr->host;

		if ((uintptr_t)p >= (uintptr_t)rr->host &&
		    off + sizeof(*p) <= rr->size) {
			r = rr->dptr + off;
			break;
		}
	}
	pthread_mutex_unlock(&d->reg_lock);
	return r;
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

	int rc = xnvme_mem_map(mem_dev(drv), (void *)buf_base, size);
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
	if (!mem_dev(drv))
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
			xnvme_mem_unmap(mem_dev(drv), (void *)buf_base);
			pthread_mutex_unlock(&drv->reg_lock);
			return opends_ok();
		}
	}
	pthread_mutex_unlock(&drv->reg_lock);
	return opends_err(OPENDS_MEMORY_NOT_REGISTERED);
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

	struct registered_file *h = (struct registered_file *)fh;
	struct io_worker *w;
	uint32_t head;

	pthread_mutex_lock(&d->submit_lock);
	struct file_op *op = claim_slot_locked(d, h->dev, &w, &head);
	op->mode = FILE_OP_ASYNC;
	op->is_write = is_write;
	op->h = h;
	op->buf_base = buf_base;
	op->u.async.size = size;
	op->u.async.file_offset = file_offset;
	op->u.async.buf_offset = buf_offset;
	op->u.async.future = future;
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

/* ------------------------------------------------------------------ */
/*  GPU-initiated reads                                               */
/* ------------------------------------------------------------------ */

/*
 * A stream read the GPU issues itself. The submitting thread resolves the
 * extents and builds one NVMe read per chunk, PRP lists included, into a GPU
 * context claimed for the op. A kernel on the user's stream then submits the
 * commands from the context's GPU-resident queues and reaps them, and a host
 * callback behind it publishes bytes_read and returns the context. The shape
 * (size and offsets) is read when the op is submitted, not when the stream
 * reaches it.
 */

static void
gpu_ctx_free(struct driver *d, struct gpu_ctx *c)
{
	for (int i = 0; i < c->nq; i++) {
		xnvme_cuda_queue_destroy(c->dev->xdev, c->queues[i]);
		c->queues[i] = NULL;
	}
	c->nq = 0;
	if (c->bounce) {
		buf_free_locked(d, c->bounce);
		c->bounce = NULL;
	}
	if (c->prp_dev) {
		buf_free_locked(d, c->prp_dev);
		c->prp_dev = NULL;
	}
	free(c->prp_phys);
	c->prp_phys = NULL;
	if (c->prp_stage) {
		ds_accel->host_free(c->prp_stage);
		c->prp_stage = NULL;
	}
	if (c->cmds) {
		ds_accel->host_free(c->cmds);
		c->cmds = NULL;
	}
	if (c->op) {
		ds_accel->host_free(c->op);
		c->op = NULL;
	}
}

/* Returns 0 on success; on failure, the dev_err to report (a negative errno
 * or a vendor code). */
static int
gpu_ctx_setup(struct driver *d, struct nvme_device *dev, struct gpu_ctx *c)
{
	size_t cap = d->gpu_max_cmds;
	size_t prp_bytes = cap * NVME_PAGE_SIZE;
	int qopts = d->gpu_sq_hostmem ? XNVME_QUEUE_SQ_HOSTMEM : 0;
	void *host = NULL;
	ds_accel_devptr_t dptr = 0;
	int rc;

	c->dev = dev;

	rc = ds_accel->host_alloc_mapped(sizeof(*c->op), &host, &dptr);
	if (rc != 0) {
		return rc;
	}
	memset(host, 0, sizeof(*c->op));
	c->op = host;
	c->op_dptr = dptr;

	rc = ds_accel->host_alloc_mapped(cap * sizeof(*c->cmds), &host, &dptr);
	if (rc != 0) {
		return rc;
	}
	c->cmds = host;
	c->op->cmds = dptr;

	/* The lists are staged here and copied to the device heap on the
	 * stream ahead of the kernel; the controller reads them from there. */
	rc = ds_accel->host_alloc_mapped(prp_bytes, &host, &dptr);
	if (rc != 0) {
		return rc;
	}
	c->prp_stage = host;

	c->prp_phys = calloc(cap, sizeof(*c->prp_phys));
	if (!c->prp_phys) {
		return -ENOMEM;
	}
	c->prp_dev = buf_alloc_locked(d, prp_bytes);
	if (!c->prp_dev) {
		return -ENOMEM;
	}
	if ((uintptr_t)c->prp_dev & (NVME_PAGE_SIZE - 1)) {
		fprintf(stderr,
		        "aisio: the PRP list region is not page aligned\n");
		return -EINVAL;
	}
	for (size_t i = 0; i < cap; i++) {
		rc = xnvme_buf_vtophys(
		        dev->xdev, (uint8_t *)c->prp_dev + i * NVME_PAGE_SIZE,
		        &c->prp_phys[i]);
		if (rc < 0) {
			return rc;
		}
	}

	c->bounce = buf_alloc_locked(d, dev->lba_size);
	if (!c->bounce) {
		return -ENOMEM;
	}

	for (int i = 0; i < d->gpu_queues_per_op; i++) {
		rc = xnvme_cuda_queue_create(dev->xdev,
		                             (uint16_t)d->gpu_queue_depth,
		                             qopts, &c->queues[i]);
		if (rc < 0) {
			fprintf(stderr,
			        "aisio: xnvme_cuda_queue_create(%s, depth=%u) "
			        "failed; err(%d)\n",
			        dev->dev_uri, d->gpu_queue_depth, rc);
			c->queues[i] = NULL;
			return rc;
		}
		c->nq = i + 1;
		c->op->queues[i] = (uint64_t)(uintptr_t)c->queues[i];
	}

	return 0;
}

static void
gpu_pool_free(struct driver *d)
{
	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];

		for (int i = 0; i < dev->n_gpu_ctxs; i++) {
			gpu_ctx_free(d, &dev->gpu_ctxs[i]);
		}
		free(dev->gpu_ctxs);
		dev->gpu_ctxs = NULL;
		dev->n_gpu_ctxs = 0;
	}
	d->gpu_ready = false;
}

static int
gpu_pool_setup(struct driver *d)
{
	int rc;

	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];

		dev->gpu_ctxs =
		        calloc((size_t)d->gpu_ctxs, sizeof(*dev->gpu_ctxs));
		if (!dev->gpu_ctxs) {
			rc = -ENOMEM;
			goto fail;
		}
		for (int i = 0; i < d->gpu_ctxs; i++) {
			dev->n_gpu_ctxs = i + 1;
			rc = gpu_ctx_setup(d, dev, &dev->gpu_ctxs[i]);
			if (rc != 0) {
				goto fail;
			}
		}
	}
	d->gpu_ready = true;
	return 0;

fail:
	gpu_pool_free(d);
	return rc;
}

static void
gpu_pool_teardown(struct driver *d)
{
	uint64_t deadline = monotonic_ns() + GPU_DRAIN_TIMEOUT_NS;

	if (!d->gpu_ready) {
		return;
	}
	/* An op in flight owns its context. The kernel runs without the host,
	 * so it finishes (or times out) on its own, and either its done word
	 * or the callback behind it marks that; wait before deleting the
	 * queues. */
	for (int di = 0; di < d->n_devices; di++) {
		struct nvme_device *dev = &d->devices[di];

		for (int i = 0; i < dev->n_gpu_ctxs; i++) {
			struct gpu_ctx *c = &dev->gpu_ctxs[i];

			while (!__atomic_load_n(&c->broken, __ATOMIC_ACQUIRE) &&
			       __atomic_load_n(&c->busy, __ATOMIC_ACQUIRE) &&
			       !(__atomic_load_n(&c->fast, __ATOMIC_ACQUIRE) &&
			         __atomic_load_n(&c->op->done,
			                         __ATOMIC_ACQUIRE)) &&
			       monotonic_ns() < deadline) {
				struct timespec ts = {0, 1000000};

				nanosleep(&ts, NULL);
			}
		}
	}
	gpu_pool_free(d);
}

/* Physical address of a device virtual address, resolving once per granule. */
static int
gpu_vtophys(const struct xnvme_dev *xdev, uintptr_t va, uintptr_t *granule_va,
            uint64_t *granule_phys, uint64_t *phys)
{
	uintptr_t g = va & ~(uintptr_t)(GPU_PHYS_GRANULE - 1);
	int rc;

	if (g != *granule_va) {
		rc = xnvme_buf_vtophys(xdev, (void *)g, granule_phys);
		if (rc < 0) {
			return -EFAULT;
		}
		*granule_va = g;
	}
	*phys = *granule_phys + (va - g);
	return 0;
}

/* PRP1 takes the first page at its offset; PRP2 the second page, or the list
 * staged in page k when the transfer spans more. Returns 1 when a list was
 * used, 0 when not, or a negative errno. */
static int
gpu_set_prps(struct gpu_ctx *c, uint32_t k, uint8_t *dst, size_t len,
             struct xnvme_spec_cmd *cmd)
{
	const struct xnvme_dev *xdev = c->dev->xdev;
	uintptr_t first = (uintptr_t)dst & ~(uintptr_t)(NVME_PAGE_SIZE - 1);
	size_t off = (uintptr_t)dst - first;
	size_t npages = (off + len + NVME_PAGE_SIZE - 1) / NVME_PAGE_SIZE;
	uint64_t *list = c->prp_stage + (size_t)k * NVME_PRP_LIST_ENTRIES;
	uintptr_t granule_va =
	        1; /* Never a granule start, so the first page resolves. */
	uint64_t granule_phys = 0;
	uint64_t phys;
	int rc;

	if (npages - 1 > NVME_PRP_LIST_ENTRIES) {
		return -E2BIG;
	}
	rc = gpu_vtophys(xdev, (uintptr_t)dst, &granule_va, &granule_phys,
	                 &phys);
	if (rc < 0) {
		return rc;
	}
	cmd->common.dptr.prp.prp1 = phys;
	cmd->common.dptr.prp.prp2 = 0;
	if (npages == 1) {
		return 0;
	}
	for (size_t p = 1; p < npages; p++) {
		rc = gpu_vtophys(xdev, first + p * NVME_PAGE_SIZE, &granule_va,
		                 &granule_phys, &phys);
		if (rc < 0) {
			return rc;
		}
		list[p - 1] = phys;
	}
	if (npages == 2) {
		cmd->common.dptr.prp.prp2 = list[0];
		return 0;
	}
	cmd->common.dptr.prp.prp2 = c->prp_phys[k];
	return 1;
}

static int
gpu_add_read(struct driver *d, struct gpu_ctx *c, uint64_t slba, uint32_t nlbas,
             uint8_t *dst, size_t len)
{
	struct ds_gpu_op *op = c->op;
	struct xnvme_spec_cmd *cmd;
	int rc;

	if (op->n_cmds >= d->gpu_max_cmds) {
		return -E2BIG;
	}
	cmd = &c->cmds[op->n_cmds];
	memset(cmd, 0, sizeof(*cmd));
	cmd->common.opcode = XNVME_SPEC_NVM_OPC_READ;
	cmd->common.nsid = c->dev->nsid;
	cmd->nvm.slba = slba;
	cmd->nvm.nlb = nlbas - 1;
	rc = gpu_set_prps(c, op->n_cmds, dst, len, cmd);
	if (rc < 0) {
		return rc;
	}
	if (rc > 0) {
		c->prp_pages = op->n_cmds + 1;
	}
	op->n_cmds++;
	return 0;
}

/* Build the commands for a read of size bytes at req_start into dst_base.
 * Returns 0, -E2BIG when the op does not fit the context (the host path takes
 * it), -EINVAL for a shape the stream path rejects, -EIO when the extents do
 * not resolve, or -EFAULT when a buffer does not translate. */
static int
gpu_build_read(struct driver *d, struct gpu_ctx *c, struct registered_file *h,
               uint8_t *dst_base, uint64_t req_start, size_t size, bool aligned)
{
	struct nvme_device *dev = h->dev;
	struct ds_gpu_op *op = c->op;
	struct ds_extent *extents = NULL;
	uint32_t extent_count = 0;
	uint32_t lba_shift = dev->lba_shift;
	uint32_t lba_mask = dev->lba_size - 1;
	size_t chunk_max = dev->mdts_nbytes;
	size_t bytes = 0;
	uint64_t req_end;
	int rc;

	if (size > UINT64_MAX - req_start) {
		return -EINVAL;
	}
	req_end = req_start + size;

	/* One PRP list page per command bounds a chunk at 512 pages; so does
	 * the 16-bit block count. */
	chunk_max = XNVME_MIN_U64(chunk_max, (size_t)NVME_PRP_LIST_ENTRIES *
	                                             NVME_PAGE_SIZE);
	chunk_max = XNVME_MIN_U64(chunk_max, (size_t)NVME_MAX_NLB << lba_shift);
	chunk_max &= ~(size_t)lba_mask;
	if (chunk_max == 0) {
		return -EINVAL;
	}

	op->n_cmds = 0;
	op->status = 0;
	op->tail_nbytes = 0;
	op->tail_dst = 0;
	op->tail_src = 0;
	op->blocks_done = 0;
	op->result = 0;
	c->prp_pages = 0;

	rc = resolve_extents(h, &extents, &extent_count);
	if (rc < 0) {
		return -EIO;
	}

	for (uint32_t i = 0; i < extent_count; i++) {
		const struct ds_extent *e = &extents[i];
		uint64_t ext_start = e->file_offset;
		uint64_t ext_end = ext_start + e->length;
		uint64_t span_start, span_end, off_in_ext, cur_slba;
		uint8_t *abs_dst;
		size_t remaining, tail_bytes, middle;

		if (ext_start >= req_end) {
			break;
		}
		span_start = max_u64(req_start, ext_start);
		span_end = XNVME_MIN_U64(req_end, ext_end);
		if (span_start >= span_end) {
			continue;
		}
		off_in_ext = span_start - ext_start;
		abs_dst = dst_base + (span_start - req_start);
		cur_slba = e->slba + (off_in_ext >> lba_shift);
		remaining = span_end - span_start;

		/* As on the host path: a stream read starts on an LBA boundary
		 * and only its tail may be partial, through the bounce slot. */
		if (off_in_ext & lba_mask) {
			rc = -EINVAL;
			goto out;
		}
		tail_bytes = remaining & lba_mask;
		middle = remaining - tail_bytes;
		while (middle) {
			size_t chunk = XNVME_MIN_U64(middle, chunk_max);

			rc = gpu_add_read(d, c, cur_slba,
			                  (uint32_t)(chunk >> lba_shift),
			                  abs_dst, chunk);
			if (rc < 0) {
				goto out;
			}
			cur_slba += chunk >> lba_shift;
			abs_dst += chunk;
			middle -= chunk;
			bytes += chunk;
		}
		if (tail_bytes) {
			if (aligned || op->tail_nbytes) {
				rc = -EINVAL;
				goto out;
			}
			rc = gpu_add_read(d, c, cur_slba, 1, c->bounce,
			                  dev->lba_size);
			if (rc < 0) {
				goto out;
			}
			op->tail_nbytes = (uint32_t)tail_bytes;
			op->tail_dst = (uint64_t)(uintptr_t)abs_dst;
			op->tail_src = (uint64_t)(uintptr_t)c->bounce;
			bytes += tail_bytes;
		}
	}
	rc = 0;
out:
	free(extents);
	c->bytes_total = bytes;
	return rc;
}

struct gpu_report {
	ssize_t *bytes_p;
	ssize_t value;
};

/* Publishes a result decided at submit time, in stream order. */
static void
gpu_report_cb(void *arg)
{
	struct gpu_report *r = arg;

	*r->bytes_p = r->value;
	free(r);
}

static opends_error_t
gpu_report(struct driver *d, ds_accel_stream_t cus, ssize_t *bytes_p,
           ssize_t value)
{
	struct gpu_report *r = malloc(sizeof(*r));
	int accel_rc;

	if (!r) {
		return opends_err(OPENDS_INTERNAL_ERROR);
	}
	r->bytes_p = bytes_p;
	r->value = value;
	accel_rc = ds_accel->launch_host_func(cus, gpu_report_cb, r);
	if (accel_rc != 0) {
		free(r);
		return classify_accel_failure(d, accel_rc);
	}
	return opends_ok();
}

static void
gpu_ctx_release(struct gpu_ctx *c)
{
	__atomic_store_n(&c->busy, 0, __ATOMIC_RELEASE);
}

/* A timed-out round leaves the completion head out of step with the
 * controller, so the context is retired, not reused. */
static void
gpu_ctx_retire(struct gpu_ctx *c)
{
	fprintf(stderr,
	        "aisio: a GPU-issued read on %s timed out; retiring its "
	        "queues\n",
	        c->dev->dev_uri);
	__atomic_store_n(&c->broken, true, __ATOMIC_RELEASE);
}

/* Runs behind the kernel on the user's stream, for a result the kernel
 * could not write itself. */
static void
gpu_op_done_cb(void *arg)
{
	struct gpu_ctx *c = arg;
	int status = (int)c->op->status;

	*c->bytes_p = status ? -(ssize_t)OPENDS_DEVICE_DRIVER_ERROR
	                     : (ssize_t)c->bytes_total;
	if (status == -EAGAIN) {
		gpu_ctx_retire(c);
		return;
	}
	gpu_ctx_release(c);
}

/* The previous op's done word must be gone before this op can be seen as
 * fast, or a claimer would take the context away under its owner. */
static void
gpu_ctx_take(struct gpu_ctx *c)
{
	__atomic_store_n(&c->fast, false, __ATOMIC_RELEASE);
	__atomic_store_n(&c->op->done, 0, __ATOMIC_RELEASE);
}

/* Claim a context of the device, waiting for one in flight to come back.
 * NULL when the device has none left to claim. */
static struct gpu_ctx *
gpu_ctx_claim(struct nvme_device *dev)
{
	for (;;) {
		bool any = false;

		for (int i = 0; i < dev->n_gpu_ctxs; i++) {
			struct gpu_ctx *c = &dev->gpu_ctxs[i];
			int expected = 0;

			if (__atomic_load_n(&c->broken, __ATOMIC_ACQUIRE)) {
				continue;
			}
			any = true;
			if (__atomic_compare_exchange_n(&c->busy, &expected, 1,
			                                false, __ATOMIC_ACQUIRE,
			                                __ATOMIC_RELAXED)) {
				gpu_ctx_take(c);
				return c;
			}
			/* An op the kernel completed on its own: the first
			 * claimer to see it takes the context over through a
			 * third state, so there is no free window for another
			 * to race into. */
			expected = 1;
			if (__atomic_load_n(&c->fast, __ATOMIC_ACQUIRE) &&
			    __atomic_load_n(&c->op->done, __ATOMIC_ACQUIRE) &&
			    __atomic_compare_exchange_n(&c->busy, &expected, 2,
			                                false, __ATOMIC_ACQUIRE,
			                                __ATOMIC_RELAXED)) {
				if ((int)c->op->status == -EAGAIN) {
					gpu_ctx_retire(c);
					continue;
				}
				gpu_ctx_take(c);
				__atomic_store_n(&c->busy, 1, __ATOMIC_RELEASE);
				return c;
			}
		}
		if (!any) {
			return NULL;
		}
		sched_yield();
	}
}

/* Submit a stream read the GPU issues. Returns true when the op was taken,
 * with *err the result of submitting it, or false to leave it to the host
 * path. */
static bool
submit_stream_read_gpu(struct driver *d, struct registered_file *h,
                       void *buf_base, size_t size, off_t file_offset,
                       off_t buf_offset, ssize_t *bytes_p, bool aligned,
                       ds_accel_stream_t cus, opends_error_t *err)
{
	struct gpu_ctx *c = gpu_ctx_claim(h->dev);
	opends_op_error_t e;
	int rc, accel_rc;

	if (!c) {
		return false;
	}
	rc = gpu_build_read(d, c, h, (uint8_t *)buf_base + buf_offset,
	                    (uint64_t)file_offset, size, aligned);
	if (rc == -E2BIG) {
		gpu_ctx_release(c);
		return false;
	}
	if (rc < 0) {
		gpu_ctx_release(c);
		e = rc == -EINVAL ? OPENDS_INVALID_VALUE
		    : rc == -EIO  ? OPENDS_FS_SETUP_ERROR
		                  : OPENDS_DEVICE_DRIVER_ERROR;
		*err = gpu_report(d, cus, bytes_p, -(ssize_t)e);
		return true;
	}
	if (c->op->n_cmds == 0) {
		ssize_t n = (ssize_t)c->bytes_total;

		gpu_ctx_release(c);
		*err = gpu_report(d, cus, bytes_p, n);
		return true;
	}
	c->bytes_p = bytes_p;
	c->op->result = result_dptr(d, bytes_p);
	if (c->op->result) {
		c->op->result_ok = (int64_t)c->bytes_total;
		c->op->result_err = -(int64_t)OPENDS_DEVICE_DRIVER_ERROR;
		__atomic_store_n(&c->fast, true, __ATOMIC_RELEASE);
	}

	if (c->prp_pages) {
		accel_rc = ds_accel->copy_async(
		        c->prp_dev, c->prp_stage,
		        (size_t)c->prp_pages * NVME_PAGE_SIZE, cus);
		if (accel_rc != 0) {
			goto fail;
		}
	}
	accel_rc = ds_accel->gpu_io_launch(c->op_dptr, (uint32_t)c->nq,
	                                   d->gpu_queue_depth, cus);
	if (accel_rc != 0) {
		goto fail;
	}
	if (!c->fast) {
		accel_rc = ds_accel->launch_host_func(cus, gpu_op_done_cb, c);
		if (accel_rc != 0) {
			goto fail;
		}
	}
	*err = opends_ok();
	return true;

fail:
	/* Part of the op may be on the stream, so the context cannot be
	 * taken back; retire it. */
	__atomic_store_n(&c->broken, true, __ATOMIC_RELEASE);
	*err = classify_accel_failure(d, accel_rc);
	return true;
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
	if (!stream)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!d->workers_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	ds_accel_stream_t cus = (ds_accel_stream_t)stream;
	struct opends_stream *opends_stream = opends_stream_get(d, cus);
	if (!opends_stream)
		return opends_err(OPENDS_INTERNAL_ERROR);

	struct registered_file *h = (struct registered_file *)fh;
	struct io_worker *w;
	uint32_t head;

	/* The GPU engine takes a read it has room for when the stream's flags
	 * fix the shape at submit; the rest, and every write, goes to an I/O
	 * thread, which reads the size and offsets at stream time. */
	if (!is_write && d->gpu_ready &&
	    (opends_stream->flags & OPENDS_STREAM_FIXED_SHAPE) ==
	            OPENDS_STREAM_FIXED_SHAPE) {
		opends_error_t gerr;

		if (submit_stream_read_gpu(d, h, buf_base, *size_p,
		                           *file_offset_p, *buf_offset_p,
		                           bytes_p, opends_stream->aligned, cus,
		                           &gerr))
			return gerr;
	}

	pthread_mutex_lock(&d->submit_lock);
	struct file_op *op = claim_slot_locked(d, h->dev, &w, &head);
	uint32_t seq = ++opends_stream->next_seq;
	op->mode = FILE_OP_STREAM;
	op->is_write = is_write;
	op->h = h;
	op->buf_base = buf_base;
	op->u.stream.size_p = size_p;
	op->u.stream.file_offset_p = file_offset_p;
	op->u.stream.buf_offset_p = buf_offset_p;
	op->u.stream.bytes_read_p = bytes_p;
	op->u.stream.opends_stream = opends_stream;
	op->u.stream.seq = seq;

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
	 * nothing. Hence the coupling to the stream's alignment.
	 *
	 * The gate enqueues run under submit_lock so gate values reach the
	 * stream in seq order; an enqueue that blocks (a full stream queue)
	 * stalls all submission. */
	int accel_rc;
	if (opends_stream->aligned && ds_accel->launch_host_func) {
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
	if (!opends_stream->aligned && !is_write) {
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
	if (!drv)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!stream || (flags & ~OPENDS_STREAM_FLAGS_ALL))
		return opends_err(OPENDS_INVALID_VALUE);

	if (!drv->workers_ready)
		return opends_err(OPENDS_DEVICE_DRIVER_ERROR);

	ds_accel_stream_t cus = (ds_accel_stream_t)stream;
	int idx;

	pthread_mutex_lock(&drv->reg_lock);
	idx = ds_stream_map_get(drv->stream_map, STREAM_MAP_MASK, cus);
	if (idx >= 0) {
		bool same = drv->streams[idx].flags == flags;

		pthread_mutex_unlock(&drv->reg_lock);
		return same ? opends_ok() : opends_err(OPENDS_INVALID_VALUE);
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
	opends_stream->flags = flags;
	opends_stream->aligned = drv->assume_aligned_only ||
	                         (flags & OPENDS_STREAM_PAGE_ALIGNED_INPUTS);

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
