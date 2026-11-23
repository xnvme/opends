/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * read_pattern.h - Shared pattern rule for sync-read tests.
 *
 * Both the prep binary (which writes the pattern to a file) and the
 * backend test binaries (which verify reads against the pattern in
 * memory) derive bytes from pattern_byte(). Keeping one definition
 * guarantees prep and verify never drift apart.
 *
 * POSIX only; no OpenDS dependency.
 */
#ifndef READ_PATTERN_H_
#define READ_PATTERN_H_

#include <stddef.h>
#include <sys/types.h>

#define PAGE 4096
#define FILE_PAGES 16
#define FILE_SIZE ((size_t)(PAGE * FILE_PAGES))

static inline unsigned char
pattern_byte(off_t abs_offset)
{
	unsigned long p = (unsigned long)abs_offset / PAGE;
	unsigned long o = (unsigned long)abs_offset % PAGE;
	return (unsigned char)((o + p) & 0xff);
}

/* The bytes a read of size at offset returns, cut at the end of the file.
 * Returns their number. */
static inline ssize_t
pattern_expected(char *dst, off_t offset, size_t size)
{
	if (offset < 0 || (size_t)offset >= FILE_SIZE)
		return 0;

	size_t avail = FILE_SIZE - (size_t)offset;
	if (size > avail)
		size = avail;

	for (size_t i = 0; i < size; i++)
		dst[i] = (char)pattern_byte(offset + (off_t)i);
	return (ssize_t)size;
}

#endif /* READ_PATTERN_H_ */
