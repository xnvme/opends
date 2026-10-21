/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "opends_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* clang-format off */
const char *
opends_op_status_error(opends_op_error_t status)
{
	switch (status) {
	case OPENDS_SUCCESS:                   return "opends success";
	case OPENDS_DRIVER_NOT_INITIALIZED:    return "driver is not loaded";
	case OPENDS_DRIVER_INVALID_PROPS:      return "invalid property";
	case OPENDS_DRIVER_UNSUPPORTED_LIMIT:  return "property range error";
	case OPENDS_DRIVER_VERSION_MISMATCH:   return "driver version mismatch";
	case OPENDS_DRIVER_VERSION_READ_ERROR: return "driver version read error";
	case OPENDS_DRIVER_CLOSING:            return "driver shutdown in progress";
	case OPENDS_PLATFORM_NOT_SUPPORTED:    return "direct storage not supported on current platform";
	case OPENDS_IO_NOT_SUPPORTED:          return "direct storage not supported on current file";
	case OPENDS_DEVICE_NOT_SUPPORTED:      return "direct storage not supported on current device";
	case OPENDS_FS_DRIVER_ERROR:           return "filesystem driver ioctl error";
	case OPENDS_DEVICE_DRIVER_ERROR:       return "device driver API error";
	case OPENDS_POINTER_INVALID:           return "invalid device pointer";
	case OPENDS_MEMORY_TYPE_INVALID:       return "invalid pointer memory type";
	case OPENDS_POINTER_RANGE_ERROR:       return "pointer range exceeds allocated address range";
	case OPENDS_CONTEXT_MISMATCH:          return "device context mismatch";
	case OPENDS_INVALID_MAPPING_SIZE:      return "access beyond maximum pinned size";
	case OPENDS_INVALID_MAPPING_RANGE:     return "access beyond mapped size";
	case OPENDS_INVALID_FILE_TYPE:         return "unsupported file type";
	case OPENDS_INVALID_FILE_OPEN_FLAG:    return "unsupported file open flags";
	case OPENDS_DIO_NOT_SET:               return "fd direct IO not set";
	case OPENDS_INVALID_VALUE:             return "invalid arguments";
	case OPENDS_MEMORY_ALREADY_REGISTERED: return "device pointer already registered";
	case OPENDS_MEMORY_NOT_REGISTERED:     return "device pointer lookup failure";
	case OPENDS_PERMISSION_DENIED:         return "driver or file access error";
	case OPENDS_DRIVER_ALREADY_OPEN:       return "driver is already open";
	case OPENDS_HANDLE_NOT_REGISTERED:     return "file descriptor is not registered";
	case OPENDS_HANDLE_ALREADY_REGISTERED: return "file descriptor is already registered";
	case OPENDS_DEVICE_NOT_FOUND:          return "device not found";
	case OPENDS_INTERNAL_ERROR:            return "internal error";
	case OPENDS_GETNEWFD_FAILED:           return "failed to obtain new file descriptor";
	case OPENDS_FS_SETUP_ERROR:            return "filesystem driver initialization error";
	case OPENDS_IO_DISABLED:               return "direct storage disabled by config on current file";
	case OPENDS_BATCH_SUBMIT_FAILED:       return "failed to submit batch operation";
	case OPENDS_MEMORY_PINNING_FAILED:     return "failed to allocate pinned device memory";
	case OPENDS_BATCH_FULL:                return "queue full for batch operation";
	case OPENDS_ASYNC_NOT_SUPPORTED:       return "async I/O not supported by this backend";
	case OPENDS_IO_MAX_ERROR:              return "max error";
	default:                                return "unknown opends error";
	}
}
/* clang-format on */

static int
read_block(int fd, void *buf, off_t off)
{
	ssize_t r;

	do {
		r = pread(fd, buf, OPENDS_DIRECT_ALIGN, off);
	} while (r < 0 && errno == EINTR);
	if (r < 0)
		return -errno;
	memset((uint8_t *)buf + r, 0, OPENDS_DIRECT_ALIGN - (size_t)r);
	return 0;
}

/* Fill the partial edge blocks of [start, start + span) from the file. */
static int
rmw_fill(int fd, int oflags, uint8_t *bounce, off_t start, size_t span,
         size_t head, size_t tail)
{
	int rfd = fd;
	int err = 0;
	size_t last = span - OPENDS_DIRECT_ALIGN;

	if ((oflags & O_ACCMODE) == O_WRONLY) {
		char path[64];
		snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
		rfd = open(path, O_RDONLY | O_DIRECT);
		if (rfd < 0)
			return -errno;
	}
	if (head)
		err = read_block(rfd, bounce, start);
	if (err == 0 && tail && (last || !head))
		err = read_block(rfd, bounce + last, start + (off_t)last);
	if (rfd != fd)
		close(rfd);
	return err;
}

ssize_t
opends_direct_pwrite(int fd, int oflags, const void *src, size_t size,
                     off_t off,
                     int (*copy)(void *dst, const void *src, size_t bytes))
{
	if (size == 0)
		return 0;

	off_t start = off & ~(off_t)(OPENDS_DIRECT_ALIGN - 1);
	off_t end = off + (off_t)size;
	size_t head = (size_t)(off - start);
	size_t span = (head + size + OPENDS_DIRECT_ALIGN - 1) &
	              ~(size_t)(OPENDS_DIRECT_ALIGN - 1);
	size_t tail = span - head - size;

	void *bounce = NULL;
	int err = posix_memalign(&bounce, OPENDS_DIRECT_ALIGN, span);
	if (err != 0)
		return -ENOMEM;

	ssize_t ret = (ssize_t)size;
	off_t trunc_to = -1;
	if (tail) {
		struct stat st;
		err = fstat(fd, &st);
		if (err < 0) {
			ret = -errno;
			goto out;
		}
		if (st.st_size < end)
			st.st_size = end;
		if (start + (off_t)span > st.st_size)
			trunc_to = st.st_size;
	}
	if (head || tail) {
		err = rmw_fill(fd, oflags, bounce, start, span, head, tail);
		if (err < 0) {
			ret = err;
			goto out;
		}
	}
	err = copy((uint8_t *)bounce + head, src, size);
	if (err != 0) {
		ret = -EIO;
		goto out;
	}

	size_t done = 0;
	while (done < span) {
		ssize_t w = pwrite(fd, (uint8_t *)bounce + done, span - done,
		                   start + (off_t)done);
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

	if (trunc_to >= 0) {
		err = ftruncate(fd, trunc_to);
		if (err < 0) {
			ret = -errno;
			goto out;
		}
	}
out:
	free(bounce);
	return ret;
}

ssize_t
opends_direct_pread(int fd, void *dst, size_t size, off_t off)
{
	if (size == 0)
		return 0;

	off_t start = off & ~(off_t)(OPENDS_DIRECT_ALIGN - 1);
	size_t head = (size_t)(off - start);
	size_t span = (head + size + OPENDS_DIRECT_ALIGN - 1) &
	              ~(size_t)(OPENDS_DIRECT_ALIGN - 1);

	void *bounce = NULL;
	int err = posix_memalign(&bounce, OPENDS_DIRECT_ALIGN, span);
	if (err != 0)
		return -ENOMEM;

	ssize_t ret = 0;
	size_t got = 0;
	while (got < span) {
		ssize_t r = pread(fd, (uint8_t *)bounce + got, span - got,
		                  start + (off_t)got);
		if (r < 0) {
			if (errno == EINTR)
				continue;
			ret = -errno;
			goto out;
		}
		if (r == 0)
			break;
		got += (size_t)r;
	}

	if (got > head) {
		size_t n = got - head;
		if (n > size)
			n = size;
		memcpy(dst, (uint8_t *)bounce + head, n);
		ret = (ssize_t)n;
	}
out:
	free(bounce);
	return ret;
}

opends_error_t
opends_batch_check(unsigned nr, const opends_io_params_t *iocbp)
{
	for (unsigned i = 0; i < nr; i++)
		if (iocbp[i].mode != OPENDS_BATCH ||
		    (iocbp[i].opcode != OPENDS_READ &&
		     iocbp[i].opcode != OPENDS_WRITE))
			return opends_err(OPENDS_INVALID_VALUE);
	return opends_ok();
}
