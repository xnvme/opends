/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "opends_internal.h"

#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

struct ref_handle {
	int fd;
	int oflags;
};

static bool driver_open;
static long use_count;

#define REF_MAX_BUFS 64
static const void *ref_registered[REF_MAX_BUFS];
static int ref_registered_count;

opends_error_t
opends_driver_open(void)
{
	if (driver_open)
		return opends_err(OPENDS_DRIVER_ALREADY_OPEN);
	driver_open = true;
	return opends_ok();
}

opends_error_t
opends_driver_close(void)
{
	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	ref_registered_count = 0;
	driver_open = false;
	return opends_ok();
}

long
opends_use_count(void)
{
	return use_count;
}

opends_error_t
opends_driver_get_properties(opends_drv_props_t *props)
{
	if (!props)
		return opends_err(OPENDS_INVALID_VALUE);
	memset(props, 0, sizeof(*props));
	props->major_version = 1;
	props->minor_version = 0;
	return opends_ok();
}

opends_error_t
opends_driver_set_max_direct_io_size(size_t max_direct_io_size)
{
	(void)max_direct_io_size;
	return opends_ok();
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

opends_error_t
opends_handle_register(opends_handle_t *fh, int fd)
{
	if (!fh || fd < 0)
		return opends_err(OPENDS_INVALID_VALUE);

	int oflags = fcntl(fd, F_GETFL);
	if (oflags < 0)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!(oflags & O_DIRECT))
		return opends_err(OPENDS_DIO_NOT_SET);

	struct ref_handle *h = malloc(sizeof(*h));
	if (!h)
		return opends_err(OPENDS_INTERNAL_ERROR);

	h->fd = fd;
	h->oflags = oflags;
	*fh = h;
	use_count++;
	return opends_ok();
}

void
opends_handle_deregister(opends_handle_t fh)
{
	if (!fh)
		return;
	free(fh);
	use_count--;
}

#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

void *
opends_result_alloc(size_t size)
{
	return malloc(size);
}

void
opends_result_free(void *p)
{
	free(p);
}

void *
opends_alloc(size_t size)
{
	return aligned_alloc(4096, ALIGN_UP(size, 4096));
}

void
opends_free(void *buf)
{
	if (!buf)
		return;

	for (int i = 0; i < ref_registered_count; i++) {
		if (ref_registered[i] == buf)
			return;
	}
	free(buf);
}

opends_error_t
opends_buf_register(const void *buf_base, size_t size, int flags)
{
	(void)flags;

	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!buf_base || !size)
		return opends_err(OPENDS_INVALID_VALUE);

	for (int i = 0; i < ref_registered_count; i++) {
		if (ref_registered[i] == buf_base)
			return opends_err(OPENDS_MEMORY_ALREADY_REGISTERED);
	}
	if (ref_registered_count >= REF_MAX_BUFS)
		return opends_err(OPENDS_INTERNAL_ERROR);

	ref_registered[ref_registered_count++] = buf_base;
	return opends_ok();
}

opends_error_t
opends_buf_deregister(const void *buf_base)
{
	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!buf_base)
		return opends_err(OPENDS_INVALID_VALUE);

	for (int i = 0; i < ref_registered_count; i++) {
		if (ref_registered[i] == buf_base) {
			ref_registered[i] =
			        ref_registered[--ref_registered_count];
			return opends_ok();
		}
	}
	return opends_err(OPENDS_MEMORY_NOT_REGISTERED);
}

static int
host_copy(void *dst, const void *src, size_t bytes)
{
	memcpy(dst, src, bytes);
	return 0;
}

static bool
direct_aligned(const void *p, size_t size, off_t off)
{
	return (((uintptr_t)p | size | (uintptr_t)off) &
	        (OPENDS_DIRECT_ALIGN - 1)) == 0;
}

static opends_error_t
ref_async_submit(bool is_write, opends_handle_t fh, void *buf_base, size_t size,
                 off_t file_offset, off_t buf_offset,
                 opends_async_future_t *future)
{
	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!fh || !buf_base || !future)
		return opends_err(OPENDS_INVALID_VALUE);

	struct ref_handle *h = fh;
	char *p = (char *)buf_base + buf_offset;
	bool aligned = direct_aligned(p, size, file_offset);
	ssize_t result;
	if (is_write && aligned)
		result = pwrite(h->fd, p, size, file_offset);
	else if (is_write)
		result = opends_direct_pwrite(h->fd, h->oflags, p, size,
		                              file_offset, host_copy);
	else if (aligned)
		result = pread(h->fd, p, size, file_offset);
	else
		result = opends_direct_pread(h->fd, p, size, file_offset);
	if (result < 0)
		result = -(ssize_t)OPENDS_INTERNAL_ERROR;

	future->result = result;
	__atomic_store_n(&future->done, 1, __ATOMIC_RELEASE);
	return opends_ok();
}

opends_error_t
opends_async_read(opends_handle_t fh, void *buf_base, size_t size,
                  off_t file_offset, off_t buf_offset,
                  opends_async_future_t *future)
{
	return ref_async_submit(false, fh, buf_base, size, file_offset,
	                        buf_offset, future);
}

opends_error_t
opends_async_write(opends_handle_t fh, const void *buf_base, size_t size,
                   off_t file_offset, off_t buf_offset,
                   opends_async_future_t *future)
{
	return ref_async_submit(true, fh, (void *)buf_base, size, file_offset,
	                        buf_offset, future);
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

opends_error_t
opends_stream_read(opends_handle_t fh, void *buf_base, size_t *size_p,
                   off_t *file_offset_p, off_t *buf_offset_p,
                   ssize_t *bytes_read_p, opends_stream_t stream)
{
	(void)stream;
	ssize_t ret = opends_sync_read(fh, buf_base, *size_p, *file_offset_p,
	                               *buf_offset_p);
	*bytes_read_p = ret;
	return ret < 0 ? opends_err(OPENDS_INTERNAL_ERROR) : opends_ok();
}

opends_error_t
opends_stream_write(opends_handle_t fh, void *buf_base, size_t *size_p,
                    off_t *file_offset_p, off_t *buf_offset_p,
                    ssize_t *bytes_written_p, opends_stream_t stream)
{
	(void)stream;
	ssize_t ret = opends_sync_write(fh, buf_base, *size_p, *file_offset_p,
	                                *buf_offset_p);
	*bytes_written_p = ret;
	return ret < 0 ? opends_err(OPENDS_INTERNAL_ERROR) : opends_ok();
}

opends_error_t
opends_stream_register(opends_stream_t stream, unsigned flags)
{
	(void)stream;
	if (flags & ~OPENDS_STREAM_FLAGS_ALL) {
		return opends_err(OPENDS_INVALID_VALUE);
	}
	return opends_ok();
}

opends_error_t
opends_stream_deregister(opends_stream_t stream)
{
	(void)stream;
	return opends_ok();
}
