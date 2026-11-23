/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends_aisio.h - the aisio backend's own types and the calls its source
 * files share. opends_aisio.c holds the P2P paths and the driver;
 * opends_aisio_posix.c the path through the mounted file system.
 */
#ifndef OPENDS_AISIO_H_
#define OPENDS_AISIO_H_

#include "accel/opends_accel.h"
#include "opends_aisio_config.h"
#include "opends_stream_map.h"
#include "accel/opends_bounce_kernel.h"
#include "opends_internal.h"

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
#define OPENDS_FIXED_ROUNDS_MAX 64
#define FILE_OP_QUEUE_MASK (FILE_OP_QUEUE_SIZE - 1)
#define STREAM_MAP_SIZE (2 * MAX_STREAMS)
#define STREAM_MAP_MASK (STREAM_MAP_SIZE - 1)

enum file_op_state {
	FILE_OP_FREE = 0,
	FILE_OP_PENDING = 1,
	FILE_OP_IN_FLIGHT = 2,
};

/* One extent of a file: a byte range and the LBA it starts at. */
struct opends_extent {
	uint64_t file_offset;
	uint64_t slba;
	uint64_t length;
};

struct buf_entry {
	const void *base;
	size_t length;
	bool owned; /* true: xnvme_buf_alloc; false: xnvme_buf_register. */
};

/* One xnvme_buf_register of whole allocation chunks. A registered buffer
 * references every run its chunks fall in; the run is unmapped when the last
 * one leaves.
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
	bool readable;
	bool writable;
	bool raw;
};

struct opends_stream {
	uint32_t *gate;
	opends_accel_devptr_t gate_dptr;
	uint32_t next_seq;
	void *bounce_buf;
	struct opends_bounce_copy *bounce_desc_host;
	opends_accel_devptr_t bounce_desc_dev;
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
	opends_accel_devptr_t dptr; /* the buffer as the stage kernel sees it */
	struct write_buf *next;     /* the free list */
	struct write_buf *all_next; /* every buffer, for teardown */
};

/* A write's staging: the aligned span of the data goes through a buffer in
 * rounds of the buffer's size. Each round's copy is queued with a write of
 * landed behind it. The I/O thread pwrites a landed round and, for a stream
 * op, lets the stream copy the next one with turn. A deferred stream op
 * (a stream without the FIXED flags) is copied by the resident kernel of
 * opends_bounce_kernel.h instead, from a descriptor the thread fills for each
 * round, and every block of the kernel has a landed word. */
struct file_op_write {
	struct write_buf *buf;
	bool deferred;
	struct opends_stage_copy *desc;
	opends_accel_devptr_t desc_dptr;
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
	uint32_t *landed;  /* OPENDS_STAGE_BLOCKS words */
	uint32_t *turn;
	opends_accel_devptr_t landed_dptr;
	opends_accel_devptr_t turn_dptr;
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
	bool fill; /* waits for the worker's marker to reach fill_seq */
	uint32_t fill_seq;
	struct bounce_slot bounces[BOUNCE_SLOTS];
	int n_bounces;
	void *bounce_buf;
	struct file_op_write wr;
	union {
		struct file_op_stream stream;
		struct file_op_async async;
	} u;
};

struct io_cursor {
	struct xnvme_queue *queue;
	uint64_t cur_slba;
	uint8_t *abs_buf;
	size_t remaining;
};

struct driver;

/* One read through the kernel at a time for each worker, in chunks through
 * buf. The default stream writes seq to word when a chunk has landed. */
struct kernel_read {
	void *buf;
	size_t buf_nbytes;
	uint32_t *word;
	opends_accel_devptr_t word_dptr;
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
	struct opends_stage_copy *descs; /* one for each op slot */
	opends_accel_devptr_t descs_dptr;
	struct file_op file_op_queue[FILE_OP_QUEUE_SIZE];
	uint32_t queue_head;
	uint32_t queue_tail;
	pthread_t thread;
};

struct driver {
	/* NVMe device the HOMI server owns */
	char dev_uri[XNVME_IDENT_URI_LEN];
	struct aisio_config cfg; /* from the environment at open */
	struct xal *xal; /* the index xal-server publishes (index_attach) */
	int attach_rc;   /* why the attach failed; it is tried only once */
	struct xnvme_dev *xdev;
	uint32_t nsid;
	uint32_t lba_size;
	uint32_t lba_shift;
	uint32_t mdts_nbytes;
	uint64_t ns_nbytes;
	struct buf_entry bufs[MAX_BUF_ENTRIES]; /* sorted by base */
	int buf_count;
	size_t chunk_nbytes; /* allocation granularity registrations round to */
	struct map_run runs[MAX_MAP_RUNS]; /* disjoint, sorted by base */
	int run_count;

	bool workers_ready;
	opends_accel_ctx_t accel_ctx;

	struct opends_stream streams[MAX_STREAMS];
	int n_streams;
	void *stream_words_host;
	opends_accel_devptr_t stream_words_dptr;

	struct opends_stream_map_entry stream_map[STREAM_MAP_SIZE];

	struct write_buf *write_bufs_free;
	struct write_buf *write_bufs_all;
	int write_buf_count;
	int write_buf_live; /* made or being made, not kept out of the pool */
	pthread_mutex_t write_buf_lock;

	struct io_worker *workers;
	uint32_t rr_next;
	pthread_mutex_t submit_lock;
	pthread_mutex_t reg_lock;
	pthread_mutex_t alloc_lock;
	pthread_mutex_t attach_lock;
	bool stop;
};

/* Shared between the backend's source files, hidden from the library's
 * users. */
static inline void
release_gate(struct opends_stream *s, uint32_t seq)
{
	__atomic_store_n(s->gate, 2 * seq + 1, __ATOMIC_RELEASE);
}

/* opends_aisio.c */
bool opends_aisio_range_registered(struct driver *d, const void *ptr,
                                   size_t size);
void opends_aisio_future_complete(opends_async_future_t *fut, ssize_t result);

/* opends_aisio_posix.c: writes through the file system, staged in pinned
 * buffers an I/O thread only polls for, and reads through the kernel. */
struct write_buf *opends_write_buf_take(struct driver *d);
int opends_write_buf_take_or_grow(struct driver *d, struct write_buf **out);
void opends_write_buf_put(struct driver *d, struct write_buf *s);
void opends_write_buf_teardown(struct driver *d);
int opends_write_words_setup(struct driver *d, struct io_worker *w, int idx);
void opends_write_init(struct file_op_write *wr, const uint8_t *src,
                       size_t size, off_t file_offset, off_t buf_offset);
void opends_write_geometry(struct driver *d, struct file_op_write *wr);
int opends_stream_write_enqueue(struct driver *d, struct file_op *op,
                                opends_accel_stream_t cus);
int opends_stream_write_enqueue_deferred(struct driver *d, struct file_op *op,
                                         opends_accel_stream_t cus);
bool opends_dispatch_write(struct io_worker *w, struct file_op *op);
void opends_write_poll(struct io_worker *w, struct file_op *op);
void opends_kernel_read_start(struct io_worker *w, struct file_op *op,
                              uint8_t *dst, size_t size, uint64_t req_start);
void opends_kernel_read_poll(struct io_worker *w);
int opends_kernel_read_setup(struct driver *d, struct io_worker *w);
void opends_kernel_read_teardown(struct io_worker *w);

#endif /* OPENDS_AISIO_H_ */
