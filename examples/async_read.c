// SPDX-License-Identifier: BSD-3-Clause
// Read the first MiB of two files into one GPU buffer with the async API.
#define _GNU_SOURCE
#include <opends.h>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SIZE (1024 * 1024)

static void
check(opends_error_t e, const char *what)
{
	if (e.err) {
		fprintf(stderr, "%s: %s\n", what,
		        opends_op_status_error(e.err));
		exit(1);
	}
}

static int
open_file(const char *path, opends_handle_t *fh)
{
	int fd;

	fd = open(path, O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror(path);
		exit(1);
	}
	check(opends_handle_register(fh, fd), path);
	return fd;
}

int
main(int argc, char **argv)
{
	void *buf;
	int fd0, fd1;
	opends_handle_t fh0, fh1;
	opends_async_future_t fut0, fut1;
	ssize_t n0, n1;
	cudaError_t rc;

	if (argc != 3) {
		fprintf(stderr, "usage: %s <file-a> <file-b>\n", argv[0]);
		return 2;
	}

	rc = cudaMalloc(&buf, 2 * SIZE);
	if (rc != cudaSuccess) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(rc));
		return 1;
	}
	check(opends_driver_open(), "opends_driver_open");
	check(opends_buf_register(buf, 2 * SIZE, 0), "opends_buf_register");
	fd0 = open_file(argv[1], &fh0);
	fd1 = open_file(argv[2], &fh1);

	/* Both reads are in flight while this thread goes on. */
	check(opends_async_read(fh0, buf, SIZE, 0, 0, &fut0), argv[1]);
	check(opends_async_read(fh1, buf, SIZE, 0, SIZE, &fut1), argv[2]);

	n0 = opends_async_await(&fut0);
	n1 = opends_async_await(&fut1);
	if (n0 < 0 || n1 < 0) {
		fprintf(stderr, "read: %s\n",
		        OPENDS_ERRSTR((int)(n0 < 0 ? n0 : n1)));
		return 1;
	}
	printf("read %zd and %zd bytes\n", n0, n1);

	opends_handle_deregister(fh0);
	opends_handle_deregister(fh1);
	close(fd0);
	close(fd1);
	opends_buf_deregister(buf);
	cudaFree(buf);
	opends_driver_close();
	return 0;
}
