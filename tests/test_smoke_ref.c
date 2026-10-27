/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "opends.h"
#include "test_mem.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BUF_SIZE 4096

static int
check(opends_error_t err, const char *label)
{
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "%s: %s\n", label,
		        opends_op_status_error(err.err));
		return 1;
	}
	return 0;
}

static void
fill_pattern(char *buf, size_t size, unsigned char seed)
{
	for (size_t i = 0; i < size; i++)
		buf[i] = (i + seed) & 0xff;
}

static int
test_sync_read_write(opends_handle_t fh, char *wbuf, char *rbuf)
{
	fill_pattern(wbuf, BUF_SIZE, 0);

	ssize_t w = opends_sync_write(fh, wbuf, BUF_SIZE, 0, 0);
	if (w != BUF_SIZE) {
		fprintf(stderr, "sync write: %zd, expected %d\n", w, BUF_SIZE);
		return 1;
	}

	memset(rbuf, 0, BUF_SIZE);
	ssize_t n = opends_sync_read(fh, rbuf, BUF_SIZE, 0, 0);
	if (n != BUF_SIZE) {
		fprintf(stderr, "sync read: %zd, expected %d\n", n, BUF_SIZE);
		return 1;
	}

	if (memcmp(wbuf, rbuf, BUF_SIZE) != 0) {
		fprintf(stderr, "sync read/write mismatch\n");
		return 1;
	}

	return 0;
}

static int
test_buf_offset(opends_handle_t fh, char *wbuf)
{
	fill_pattern(wbuf, BUF_SIZE, 0);

	ssize_t w = opends_sync_write(fh, wbuf, BUF_SIZE, 4096, 0);
	if (w != BUF_SIZE) {
		fprintf(stderr, "buf_offset write: %zd, expected %d\n", w,
		        BUF_SIZE);
		return 1;
	}

	size_t big_size = BUF_SIZE + 512;
	char *bigbuf = test_dev_alloc(big_size);
	if (!bigbuf) {
		fprintf(stderr, "opends_mem_alloc failed\n");
		return 1;
	}
	memset(bigbuf, 0, big_size);

	ssize_t n = opends_sync_read(fh, bigbuf, BUF_SIZE, 4096, 512);
	if (n != BUF_SIZE) {
		fprintf(stderr, "buf_offset read: %zd, expected %d\n", n,
		        BUF_SIZE);
		opends_free(bigbuf);
		return 1;
	}

	int rc = memcmp(bigbuf + 512, wbuf, BUF_SIZE) != 0;
	if (rc)
		fprintf(stderr, "buf_offset read/write mismatch\n");

	opends_free(bigbuf);
	return rc;
}

static int
test_driver_properties(void)
{
	unsigned major, minor, patch;
	if (check(opends_get_version(&major, &minor, &patch), "get_version"))
		return 1;
	if (major != 0 || minor != 1 || patch != 0) {
		fprintf(stderr, "version: %u.%u.%u, expected 0.1.0\n", major,
		        minor, patch);
		return 1;
	}

	if (opends_use_count() != 1) {
		fprintf(stderr, "use_count: %ld, expected 1\n",
		        opends_use_count());
		return 1;
	}

	return 0;
}

static int
test_batch_io(opends_handle_t fh, char *wbuf, char *rbuf)
{
	opends_batch_handle_t batch;
	if (check(opends_batch_setup(&batch, 4), "batch_setup"))
		return 1;

	fill_pattern(wbuf, BUF_SIZE, 0x42);

	/* Operations in a batch are unordered, so the write and the
	 * read-back go in separate submits with a get_status between
	 * them. Each get_status must deliver its completion exactly
	 * once. */
	opends_io_params_t params[2] = {
	        {
	                .mode = OPENDS_BATCH,
	                .u.batch = {wbuf, 8192, 0, BUF_SIZE},
	                .fh = fh,
	                .opcode = OPENDS_WRITE,
	                .cookie = (void *)1,
	        },
	        {
	                .mode = OPENDS_BATCH,
	                .u.batch = {rbuf, 8192, 0, BUF_SIZE},
	                .fh = fh,
	                .opcode = OPENDS_READ,
	                .cookie = (void *)2,
	        },
	};

	for (unsigned i = 0; i < 2; i++) {
		if (check(opends_batch_submit(batch, 1, &params[i], 0),
		          "batch_submit")) {
			opends_batch_destroy(batch);
			return 1;
		}

		opends_io_events_t events[2];
		unsigned nr_events = 2;
		if (check(opends_batch_get_status(batch, 1, &nr_events, events,
		                                  NULL),
		          "batch_get_status")) {
			opends_batch_destroy(batch);
			return 1;
		}

		if (nr_events != 1 || events[0].cookie != params[i].cookie ||
		    events[0].status != OPENDS_COMPLETE ||
		    events[0].ret != BUF_SIZE) {
			fprintf(stderr,
			        "batch event %u: nr=%u status=%d ret=%zu\n", i,
			        nr_events, events[0].status, events[0].ret);
			opends_batch_destroy(batch);
			return 1;
		}
	}

	opends_batch_destroy(batch);

	if (memcmp(wbuf, rbuf, BUF_SIZE) != 0) {
		fprintf(stderr, "batch read/write mismatch\n");
		return 1;
	}

	return 0;
}

static int
test_stream_io(opends_handle_t fh, char *wbuf, char *rbuf)
{
	opends_stream_t stream = NULL;
	if (check(opends_stream_register(stream, 0), "stream_register"))
		return 1;

	fill_pattern(wbuf, BUF_SIZE, 0x99);

	size_t size = BUF_SIZE;
	off_t file_off = 12288;
	off_t buf_off = 0;
	ssize_t bytes = 0;

	if (check(opends_stream_write(fh, wbuf, &size, &file_off, &buf_off,
	                              &bytes, stream),
	          "stream_write"))
		return 1;
	if (bytes != BUF_SIZE) {
		fprintf(stderr, "stream_write bytes: %zd, expected %d\n", bytes,
		        BUF_SIZE);
		return 1;
	}

	memset(rbuf, 0, BUF_SIZE);
	bytes = 0;

	if (check(opends_stream_read(fh, rbuf, &size, &file_off, &buf_off,
	                             &bytes, stream),
	          "stream_read"))
		return 1;
	if (bytes != BUF_SIZE) {
		fprintf(stderr, "stream_read bytes: %zd, expected %d\n", bytes,
		        BUF_SIZE);
		return 1;
	}

	if (memcmp(wbuf, rbuf, BUF_SIZE) != 0) {
		fprintf(stderr, "stream read/write mismatch\n");
		return 1;
	}

	if (check(opends_stream_deregister(stream), "stream_deregister"))
		return 1;
	return 0;
}

int
main(int argc, char **argv)
{
	/* /tmp is a tmpfs on some distributions, and tmpfs rejects O_DIRECT;
	 * argv[1] points the test file at a directory that supports it. */
	const char *dir = "/tmp";
	char path[4096];
	int fd;
	int n;

	if (argc > 1) {
		dir = argv[1];
	}
	n = snprintf(path, sizeof(path), "%s/smoke_ref_XXXXXX", dir);
	if (n < 0 || (size_t)n >= sizeof(path)) {
		fprintf(stderr, "directory path too long\n");
		return 1;
	}
	fd = mkstemp(path);
	if (fd < 0) {
		perror("mkstemp");
		return 1;
	}
	unlink(path);

	if (check(opends_driver_open(), "driver_open"))
		return 1;

	opends_handle_t fh;
	opends_error_t berr = opends_handle_register(&fh, fd);
	if (berr.err != OPENDS_DIO_NOT_SET) {
		fprintf(stderr, "buffered fd not refused: %d\n", berr.err);
		return 1;
	}
	int err = fcntl(fd, F_SETFL, O_DIRECT);
	if (err < 0) {
		perror("fcntl");
		return 1;
	}
	if (check(opends_handle_register(&fh, fd), "handle_register"))
		return 1;

	char *wbuf = test_dev_alloc(BUF_SIZE);
	char *rbuf = test_dev_alloc(BUF_SIZE);
	if (!wbuf || !rbuf) {
		fprintf(stderr, "opends_mem_alloc failed\n");
		return 1;
	}

	if (test_sync_read_write(fh, wbuf, rbuf))
		return 1;
	if (test_buf_offset(fh, wbuf))
		return 1;
	if (test_driver_properties())
		return 1;
	if (test_batch_io(fh, wbuf, rbuf))
		return 1;
	if (test_stream_io(fh, wbuf, rbuf))
		return 1;

	opends_free(rbuf);
	opends_free(wbuf);
	opends_handle_deregister(fh);
	close(fd);

	if (opends_use_count() != 0) {
		fprintf(stderr, "use_count after deregister: %ld, expected 0\n",
		        opends_use_count());
		return 1;
	}

	opends_driver_close();

	fprintf(stderr, "all ok\n");
	return 0;
}
