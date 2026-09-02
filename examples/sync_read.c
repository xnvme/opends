// SPDX-License-Identifier: BSD-3-Clause
// Read a whole file into GPU memory with the sync API and check the bytes
// against a kernel read.
#define _GNU_SOURCE
#include <opends.h>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ALIGN 4096

static void
check(opends_error_t e, const char *what)
{
	if (e.err) {
		fprintf(stderr, "%s: %s\n", what,
		        opends_op_status_error(e.err));
		exit(1);
	}
}

static void
check_cuda(cudaError_t rc, const char *what)
{
	if (rc != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(rc));
		exit(1);
	}
}

int
main(int argc, char **argv)
{
	void *buf;
	void *host;
	void *ref;
	int fd;
	opends_handle_t fh;
	struct stat st;
	size_t size;
	size_t alloc;
	ssize_t n;
	ssize_t r;
	int err;

	if (argc != 2) {
		fprintf(stderr, "usage: %s <file>\n", argv[0]);
		return 2;
	}

	fd = open(argv[1], O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror(argv[1]);
		return 1;
	}
	err = fstat(fd, &st);
	if (err) {
		perror("fstat");
		return 1;
	}
	size = (size_t)st.st_size;
	/* O_DIRECT wants an aligned length. The kernel read stops at EOF. */
	alloc = (size + ALIGN - 1) / ALIGN * ALIGN;

	check_cuda(cudaMalloc(&buf, alloc), "cudaMalloc");
	check(opends_driver_open(), "opends_driver_open");
	check(opends_buf_register(buf, alloc, 0), "opends_buf_register");
	check(opends_handle_register(&fh, fd), "opends_handle_register");

	n = opends_sync_read(fh, buf, size, 0, 0);
	if (n < 0) {
		fprintf(stderr, "opends_sync_read: %s\n",
		        OPENDS_ERRSTR((int)n));
		return 1;
	}

	host = malloc(alloc);
	err = posix_memalign(&ref, ALIGN, alloc);
	if (!host || err) {
		fprintf(stderr, "out of memory\n");
		return 1;
	}
	check_cuda(cudaMemcpy(host, buf, (size_t)n, cudaMemcpyDeviceToHost),
	           "cudaMemcpy");
	r = pread(fd, ref, alloc, 0);
	if (r != n || memcmp(host, ref, (size_t)n)) {
		fprintf(stderr,
		        "GPU bytes differ from the file: read %zd, kernel read "
		        "%zd\n",
		        n, r);
		return 1;
	}
	printf("read %zd bytes into GPU memory, equal to the file\n", n);

	opends_handle_deregister(fh);
	close(fd);
	opends_buf_deregister(buf);
	cudaFree(buf);
	free(host);
	free(ref);
	opends_driver_close();
	return 0;
}
