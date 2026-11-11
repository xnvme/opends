/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef OPENDS_INTERNAL_H_
#define OPENDS_INTERNAL_H_

#include "opends.h"

#include <stdint.h>
#include <sys/types.h>

/* Practical upper bound of the logical block size O_DIRECT aligns to. */
#define OPENDS_DIRECT_ALIGN 4096

/* O_DIRECT pwrite of size bytes at off, from src through copy(). The edge
 * blocks are completed by read-modify-write when off or size is unaligned.
 * Returns size or a negative errno. */
ssize_t opends_direct_pwrite(int fd, int oflags, const void *src, size_t size,
                             off_t off,
                             int (*copy)(void *dst, const void *src,
                                         size_t bytes));

/* The O_DIRECT span that holds [off, off + size): start is the first block,
 * span the whole length, head and tail the bytes before and after the data. */
struct opends_direct_geom {
	off_t start;
	size_t span;
	size_t head;
	size_t tail;
};

void opends_direct_span(off_t off, size_t size, struct opends_direct_geom *g);

/* Complete the edge blocks of a span whose data is in bounce already: the
 * file's bytes [0, head) and [span - tail, span) go around it, read through
 * scratch (one aligned block). Returns 0 or a negative errno. */
int opends_direct_fill_edges(int fd, int oflags, uint8_t *bounce, off_t start,
                             size_t span, size_t head, size_t tail,
                             void *scratch);

/* O_DIRECT pwrite of the whole span at start. Returns 0 or a negative errno. */
int opends_direct_pwrite_span(int fd, const void *bounce, size_t span,
                              off_t start);

/* The size to truncate the file to once the span is written, or -1 when the
 * span ends inside the file, for a write whose data ends at end. Whole
 * blocks go to the file, so a span past the data grows the file past it.
 * Returns 0 or a negative errno. */
int opends_direct_trunc_to(int fd, off_t start, size_t span, off_t end,
                           off_t *trunc_to);

/* O_DIRECT pread of the aligned span at start into bounce. Returns the bytes
 * read (short at EOF) or a negative errno. */
ssize_t opends_direct_pread_span(int fd, void *bounce, size_t span,
                                 off_t start);

/* O_DIRECT pread of size bytes at off into dst through a bounce buffer and
 * copy(). Returns the bytes read (short at EOF) or a negative errno. */
ssize_t opends_direct_pread(int fd, void *dst, size_t size, off_t off,
                            int (*copy)(void *dst, const void *src,
                                        size_t bytes));

/* OPENDS_INVALID_VALUE unless every operation in iocbp[0..nr) has mode
 * OPENDS_BATCH and opcode OPENDS_READ or OPENDS_WRITE. */
opends_error_t opends_batch_check(unsigned nr, const opends_io_params_t *iocbp);

static inline opends_error_t
opends_ok(void)
{
	return (opends_error_t){OPENDS_SUCCESS, 0};
}

static inline opends_error_t
opends_err(opends_op_error_t e)
{
	return (opends_error_t){e, 0};
}

static inline opends_error_t
opends_err_dev(opends_op_error_t e, opends_result_t dev_err)
{
	return (opends_error_t){e, dev_err};
}

#endif /* OPENDS_INTERNAL_H_ */
