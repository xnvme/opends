// SPDX-License-Identifier: BSD-3-Clause
// Read a file on a CUDA stream and sum its bytes with a kernel behind the read.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <opends.h>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SIZE (1024 * 1024)

static __global__ void
sum_bytes(const unsigned char *p, size_t n, unsigned long long *sum)
{
	unsigned long long s = 0;

	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += (size_t)gridDim.x * blockDim.x) {
		s += p[i];
	}
	atomicAdd(sum, s);
}

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
	void *ref;
	unsigned long long *dsum;
	unsigned long long sum = 0;
	unsigned long long expect = 0;
	cudaStream_t stream;
	int fd;
	opends_handle_t fh;
	size_t size = SIZE;
	off_t file_offset = 0;
	off_t buf_offset = 0;
	ssize_t bytes_read = 0;
	ssize_t n;
	int err;

	if (argc != 2) {
		fprintf(stderr, "usage: %s <file>\n", argv[0]);
		return 2;
	}

	check_cuda(cudaMalloc(&buf, SIZE), "cudaMalloc");
	check_cuda(cudaMalloc((void **)&dsum, sizeof(*dsum)), "cudaMalloc");
	check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
	check(opends_driver_open(), "opends_driver_open");
	check(opends_buf_register(buf, SIZE, 0), "opends_buf_register");
	check(opends_stream_register(stream, 0), "opends_stream_register");

	fd = open(argv[1], O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror(argv[1]);
		return 1;
	}
	check(opends_handle_register(&fh, fd), "opends_handle_register");

	/* One stream, no host sync: the kernel runs behind the read. The read's
	 * size, offsets and byte count must outlive the call. */
	/* The kernel sums the whole buffer, the file may be shorter. */
	check_cuda(cudaMemsetAsync(buf, 0, SIZE, stream), "cudaMemsetAsync");
	check_cuda(cudaMemsetAsync(dsum, 0, sizeof(*dsum), stream),
	           "cudaMemsetAsync");
	check(opends_stream_read(fh, buf, &size, &file_offset, &buf_offset,
	                         &bytes_read, stream),
	      "opends_stream_read");
	sum_bytes<<<64, 256, 0, stream>>>((const unsigned char *)buf, SIZE,
	                                  dsum);
	check_cuda(cudaGetLastError(), "sum_bytes");
	check_cuda(cudaMemcpyAsync(&sum, dsum, sizeof(sum),
	                           cudaMemcpyDeviceToHost, stream),
	           "cudaMemcpyAsync");
	check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
	if (bytes_read < 0) {
		fprintf(stderr, "opends_stream_read: %s\n",
		        OPENDS_ERRSTR((int)bytes_read));
		return 1;
	}

	err = posix_memalign(&ref, 4096, SIZE);
	if (err) {
		fprintf(stderr, "posix_memalign failed\n");
		return 1;
	}
	n = pread(fd, ref, SIZE, 0);
	if (n != bytes_read) {
		fprintf(stderr, "kernel read %zd bytes, stream read %zd\n", n,
		        bytes_read);
		return 1;
	}
	for (ssize_t i = 0; i < n; i++) {
		expect += ((const unsigned char *)ref)[i];
	}
	if (sum != expect) {
		fprintf(stderr,
		        "the kernel summed %llu, the file sums to %llu\n", sum,
		        expect);
		return 1;
	}
	printf("read %zd bytes on the stream, the kernel behind it summed them "
	       "to %llu\n",
	       bytes_read, sum);

	opends_handle_deregister(fh);
	close(fd);
	opends_stream_deregister(stream);
	opends_buf_deregister(buf);
	cudaStreamDestroy(stream);
	cudaFree(dsum);
	cudaFree(buf);
	free(ref);
	opends_driver_close();
	return 0;
}
