/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Shared helpers for the ref backend tests: its buffers are host memory, and
 * every test opens its file, the driver and a handle the same way.
 */
#ifndef TEST_REF_COMMON_H_
#define TEST_REF_COMMON_H_

#include "opends.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static inline void *
ref_buf_to_host(void *dst, const void *src, size_t n)
{
	return memcpy(dst, src, n);
}

static inline void *
ref_alloc_acquire(size_t size)
{
	return opends_alloc(size);
}

static inline void
ref_alloc_release(void *buf)
{
	opends_free(buf);
}

/* Open path with O_DIRECT, open the driver and register the file. Returns 0,
 * or -1 after a message. */
static inline int
ref_setup(const char *path, int *fd, opends_handle_t *fh)
{
	opends_error_t err;

	*fd = open(path, O_RDONLY | O_DIRECT);
	if (*fd < 0) {
		perror("open");
		return -1;
	}

	err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "driver_open: %s\n",
		        opends_op_status_error(err.err));
		close(*fd);
		return -1;
	}

	err = opends_handle_register(fh, *fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "handle_register: %s\n",
		        opends_op_status_error(err.err));
		close(*fd);
		return -1;
	}
	return 0;
}

static inline void
ref_teardown(int fd, opends_handle_t fh)
{
	opends_handle_deregister(fh);
	close(fd);
	opends_driver_close();
}

#endif /* TEST_REF_COMMON_H_ */
