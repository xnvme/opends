/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef OPENDS_INTERNAL_H_
#define OPENDS_INTERNAL_H_

#include "opends.h"

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

/* O_DIRECT pread of size bytes at off into dst through a bounce buffer.
 * Returns the bytes read (short at EOF) or a negative errno. */
ssize_t opends_direct_pread(int fd, void *dst, size_t size, off_t off);

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

/* The memory type named by the opends_mem_alloc arguments: exactly one
 * known type bit, and device 0 for host memory. 0 when malformed. */
static inline int
opends_mem_type(int flags, int device)
{
	int type = flags & (OPENDS_MEM_DEVICE | OPENDS_MEM_HOST);

	if (flags != type || (type & (type - 1))) {
		return 0;
	}
	if (type == OPENDS_MEM_HOST && device != 0) {
		return 0;
	}
	return type;
}

#endif /* OPENDS_INTERNAL_H_ */
