/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Synchronous I/O implemented on the async API: submit, then await in
 * place. Any backend that provides opends_async_read/write/await gets
 * the sync family from this file. Backends with a native sync path
 * (cufile) provide their own and do not compile it.
 */
#define _GNU_SOURCE

#include "opends_internal.h"

ssize_t
opends_sync_read(opends_handle_t fh, void *buf_base, size_t size,
                 off_t file_offset, off_t buf_offset)
{
	opends_async_future_t future;

	opends_error_t err = opends_async_read(fh, buf_base, size, file_offset,
	                                       buf_offset, &future);
	if (err.err != OPENDS_SUCCESS)
		return -(ssize_t)err.err;

	return opends_async_await(&future);
}

ssize_t
opends_sync_write(opends_handle_t fh, const void *buf_base, size_t size,
                  off_t file_offset, off_t buf_offset)
{
	opends_async_future_t future;

	opends_error_t err = opends_async_write(fh, buf_base, size, file_offset,
	                                        buf_offset, &future);
	if (err.err != OPENDS_SUCCESS)
		return -(ssize_t)err.err;

	return opends_async_await(&future);
}
