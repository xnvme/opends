/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_sync_read_prep - Write the shared pattern file consumed by
 * the per-backend sync-read tests.
 *
 * Runs while the target filesystem is mounted and writable. Truncates
 * and rewrites the pattern on every invocation so a stale file from a
 * crashed prior run cannot silently satisfy a later test.
 */
#define _GNU_SOURCE

#include "read_pattern.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <pattern_path>\n", argv[0]);
		return 1;
	}

	const char *path = argv[1];

	int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		fprintf(stderr, "open %s: %s\n", path, strerror(errno));
		return 1;
	}

	char page[PAGE];
	for (int p = 0; p < FILE_PAGES; p++) {
		for (size_t i = 0; i < PAGE; i++)
			page[i] = (char)pattern_byte((off_t)(p * PAGE + i));

		if (pwrite(fd, page, PAGE, (off_t)(p * PAGE)) != PAGE) {
			fprintf(stderr, "pwrite page %d: %s\n", p,
			        strerror(errno));
			close(fd);
			return 1;
		}
	}

	if (fsync(fd) < 0) {
		fprintf(stderr, "fsync: %s\n", strerror(errno));
		close(fd);
		return 1;
	}

	close(fd);
	fprintf(stderr, "pattern written to %s (%zu bytes)\n", path, FILE_SIZE);
	return 0;
}
