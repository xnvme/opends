/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_aisio.h - the aisio backend's own types and the calls its source
 * files share. opends_aisio.c holds the P2P paths and the driver;
 * ds_aisio_posix.c the path through the mounted file system.
 */
#ifndef DS_AISIO_H_
#define DS_AISIO_H_

#include "ds_accel.h"
#include "ds_aisio_config.h"
#include "ds_stream_map.h"
#include "ds_bounce_kernel.h"
#include "opends_internal.h"
#include "ds_extent.h"

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

#define MAX_BUF_ENTRIES 8192
#define DEFAULT_BOUNCE_SIZE (128 * 1024)
#define NVME_MAX_NLB 65536
#define NVME_PRP_OFFSET_ALIGN 4
#define BOUNCE_SLOTS 2
#define MAX_STREAMS 8192
#define STREAM_WORDS_BYTES (MAX_STREAMS * sizeof(uint32_t))
#define FILE_OP_QUEUE_SIZE 1024
/* Rounds of one stream write that are queued at the call. Each is a few
 * stream commands, a stream takes a bounded number, and the I/O thread
 * cannot take a round before the call has returned. */
#define DS_FIXED_ROUNDS_MAX 64
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

/* One xnvme_mem_map of whole allocation chunks. A registered buffer references
 * every run its chunks fall in; the run is unmapped when the last one leaves.
 */
struct map_run {
	uintptr_t base;
	size_t nbytes;
	int refs;
};
#define MAX_MAP_RUNS MAX_BUF_ENTRIES

struct registered_file {
	int fd;
	int oflags;
	struct xal_inode *lease; ///< The file as attached to xal
	bool remap; ///< A write through fd since the map: map the file again
	pthread_mutex_t lock; ///< Orders the map again against acquires
};

struct opends_stream {
	uint32_t *gate;
	ds_accel_devptr_t gate_dptr;
	uint32_t next_seq;
	void *bounce_buf;
	struct ds_bounce_copy *bounce_desc_host;
	ds_accel_devptr_t bounce_desc_dev;
	bool fixed; /* registered with the three FIXED flags */
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

/* A pinned host buffer a write is staged through. */
struct write_buf {
	void *host;
	ds_accel_devptr_t dptr;     /* the buffer as the stage kernel sees it */
	struct write_buf *next;     /* the free list */
	struct write_buf *all_next; /* every buffer, for teardown */
};

/* A write's staging: the aligned span of the data goes through a buffer in
 * rounds of the buffer's size. Each round's copy is queued with a write of
 * landed behind it. The I/O thread pwrites a landed round and, for a stream
 * op, lets the stream copy the next one with turn. A deferred stream op
 * (a stream without the FIXED flags) is copied by the resident kernel of
 * ds_bounce_kernel.h instead, from a descriptor the thread fills for each
 * round, and every block of the kernel has a landed word. */
struct file_op_write {
	struct write_buf *buf;
	bool deferred;
	struct ds_stage_copy *desc;
	ds_accel_devptr_t desc_dptr;
	const uint8_t *src;
	size_t size;
	off_t file_offset;
	off_t buf_offset;
	off_t start;
	size_t span;
	size_t head;
	size_t tail;
	uint32_t rounds;
	uint32_t round;    /* the next round to pwrite */
	uint32_t granted;  /* rounds the kernel was given (deferred) */
	uint32_t copies;   /* rounds whose copy is queued */
	uint32_t enqueued; /* rounds that will land */
	uint32_t *landed;  /* DS_STAGE_BLOCKS words */
	uint32_t *turn;
	ds_accel_devptr_t landed_dptr;
	ds_accel_devptr_t turn_dptr;
};

/* Where a read lands on the drive, resolved once before it is admitted. */
struct read_plan {
	struct ds_extent *extents;
	uint32_t extent_count;
	uint64_t req_start;
	uint64_t req_end;
	uint8_t *dst_base;
	bool kernel;        ///< Whole LBAs land off a dword: a kernel read
	uint32_t spans;     ///< Extents the request overlaps
	uint64_t first_lba; ///< First LBA it reads
	uint64_t end_lba;   ///< One past the last LBA it reads
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
	bool held; ///< Counted as a reader of the file's extents until released
	struct file_op_write wr;
	uint64_t ticket; ///< Submission order
	bool planned;
	bool admitted;
	struct read_plan plan;
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

/* One read through the kernel at a time for each worker, in chunks through
 * buf. The default stream writes seq to word when a chunk has landed. */
struct kernel_read {
	void *buf;
	size_t buf_nbytes;
	uint32_t *word;
	ds_accel_devptr_t word_dptr;
	uint32_t seq;
	struct file_op *op;
	uint8_t *dst;
	uint64_t off;
	size_t remaining;
	bool last;
	bool broken;
};

struct io_worker {
	struct driver *drv;
	struct xnvme_queue *queue;
	struct kernel_read kread;
	void *edge_buf; /* one block, for the edges of a staged write */
	struct ds_stage_copy *descs; /* one for each op slot */
	ds_accel_devptr_t descs_dptr;
	struct file_op file_op_queue[FILE_OP_QUEUE_SIZE];
	uint32_t queue_head;
	uint32_t queue_tail;
	pthread_t thread;
};

struct driver {
	char dev_uri[XNVME_IDENT_URI_LEN]; ///< NVMe device the HOMI server owns
	struct aisio_config cfg;           ///< From the environment at open
	struct xal *xal; ///< Attach to the index xal-server publishes
	struct xnvme_dev *xdev;
	uint32_t nsid;
	uint32_t lba_size;
	uint32_t lba_shift;
	uint32_t mdts_nbytes;
	struct buf_entry bufs[MAX_BUF_ENTRIES]; ///< Sorted by base
	int buf_count;
	size_t chunk_nbytes; ///< Allocation granularity registrations round to
	struct map_run runs[MAX_MAP_RUNS]; ///< Disjoint, sorted by base
	int run_count;

	bool workers_ready;
	ds_accel_ctx_t accel_ctx;

	struct opends_stream streams[MAX_STREAMS];
	int n_streams;
	void *stream_words_host;
	ds_accel_devptr_t stream_words_dptr;

	struct ds_stream_map_entry stream_map[STREAM_MAP_SIZE];

	struct write_buf *write_bufs_free;
	struct write_buf *write_bufs_all;
	int write_buf_count;
	int write_buf_live; ///< Made and not kept out of the pool
	pthread_mutex_t write_buf_lock;

	struct io_worker *workers;
	uint32_t rr_next;
	uint64_t next_ticket; ///< Under submit_lock
	pthread_mutex_t submit_lock;
	pthread_mutex_t reg_lock;
	pthread_mutex_t alloc_lock;
	bool stop;

	/* Read admission, see admit_read(). Under admit_lock. */
	pthread_mutex_t admit_lock;
	uint32_t active_reads;   ///< Admitted and not yet complete
	uint64_t next_lba;       ///< Where the last admitted read ends
	uint64_t waiting_ticket; ///< Oldest read refused, NO_WAITER if none
};

/* Shared between the backend's source files, hidden from the library's
 * users. */
#define DS_HIDDEN __attribute__((visibility("hidden")))

static inline void
release_gate(struct opends_stream *s, uint32_t seq)
{
	__atomic_store_n(s->gate, 2 * seq + 1, __ATOMIC_RELEASE);
}

/* opends_aisio.c */
DS_HIDDEN bool ds_aisio_range_registered(struct driver *d, const void *ptr,
                                         size_t size);
DS_HIDDEN void ds_aisio_future_complete(opends_async_future_t *fut,
                                        ssize_t result);

/* ds_aisio_posix.c: writes through the file system, staged in pinned
 * buffers an I/O thread only polls for. */
DS_HIDDEN struct write_buf *ds_write_buf_take(struct driver *d);
DS_HIDDEN int ds_write_buf_take_or_grow(struct driver *d,
                                        struct write_buf **out);
DS_HIDDEN void ds_write_buf_put(struct driver *d, struct write_buf *s);
DS_HIDDEN void ds_write_buf_teardown(struct driver *d);
DS_HIDDEN int ds_write_words_setup(struct driver *d, struct io_worker *w,
                                   int idx);
DS_HIDDEN void ds_write_init(struct file_op_write *wr, const uint8_t *src,
                             size_t size, off_t file_offset, off_t buf_offset);
DS_HIDDEN void ds_write_geometry(struct driver *d, struct file_op_write *wr);
DS_HIDDEN int ds_stream_write_enqueue(struct driver *d, struct file_op *op,
                                      ds_accel_stream_t cus);
DS_HIDDEN int ds_stream_write_enqueue_deferred(struct driver *d,
                                               struct file_op *op,
                                               ds_accel_stream_t cus);
DS_HIDDEN bool ds_dispatch_write(struct io_worker *w, struct file_op *op);
DS_HIDDEN void ds_write_poll(struct io_worker *w, struct file_op *op);

#endif /* DS_AISIO_H_ */
