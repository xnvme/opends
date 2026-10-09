// SPDX-License-Identifier: BSD-3-Clause
// End-to-end demo: read a file on a HOMI/qublk-backed mounted XFS into GPU
// memory via the OpenDS aisio backend, and verify against the kernel-read bytes.
#define _GNU_SOURCE
#include <opends.h>

#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <file-on-mount>\n", argv[0]);
		return 2;
	}
	const char *path = argv[1];
	opends_error_t e;

	/* The upcie-cuda backend needs a current CUDA context for device init
	 * (identify) and buffer allocation; create one up front (driver API). */
	cuInit(0);
	CUdevice cudev;
	CUcontext cuctx;
	cuDeviceGet(&cudev, 0);
	cuCtxCreate(&cuctx, 0, cudev);

	e = opends_driver_open();
	if (e.err) {
		fprintf(stderr, "FAILED: opends_driver_open err=%d\n", e.err);
		return 1;
	}

	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		perror("open");
		return 1;
	}
	struct stat st;
	if (fstat(fd, &st) < 0) {
		perror("fstat");
		return 1;
	}
	size_t size = (size_t)st.st_size;
	printf("file %s size=%zu\n", path, size);

	opends_handle_t h;
	e = opends_handle_register(&h, fd);
	if (e.err) {
		fprintf(stderr, "FAILED: opends_handle_register err=%d\n", e.err);
		return 1;
	}

	void *gbuf = opends_alloc(size);
	if (!gbuf) {
		fprintf(stderr, "FAILED: opends_alloc(%zu)\n", size);
		return 1;
	}

	ssize_t n = opends_sync_read(h, gbuf, size, 0, 0);
	if (n < 0) {
		fprintf(stderr, "FAILED: opends_sync_read rc=%zd\n", n);
		return 1;
	}
	printf("aisio: read %zd bytes into GPU memory over a HOMI-shared controller\n",
	       n);

	void *got = malloc((size_t)n);
	void *ref = malloc((size_t)n);
	if (!got || !ref) {
		fprintf(stderr, "FAILED: malloc\n");
		return 1;
	}
	CUresult cr = cuMemcpyDtoH(got, (CUdeviceptr)(uintptr_t)gbuf, (size_t)n);
	if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "FAILED: cuMemcpyDtoH cr=%d\n", cr);
		return 1;
	}
	if (pread(fd, ref, (size_t)n, 0) != n) {
		fprintf(stderr, "FAILED: pread reference\n");
		return 1;
	}

	if (memcmp(got, ref, (size_t)n) == 0) {
		printf("SUCCESS: GPU bytes == file bytes (%zd) — OpenDS aisio read a file "
		       "on the HOMI/qublk-mounted XFS into GPU memory\n",
		       n);
	} else {
		size_t i = 0;
		while (i < (size_t)n && ((char *)got)[i] == ((char *)ref)[i])
			i++;
		fprintf(stderr, "FAILED: mismatch at byte %zu\n", i);
		fprintf(stderr, "got(GPU):");
		for (int k = 0; k < 16; k++)
			fprintf(stderr, " %02x", ((unsigned char *)got)[k]);
		fprintf(stderr, "\nref(file):");
		for (int k = 0; k < 16; k++)
			fprintf(stderr, " %02x", ((unsigned char *)ref)[k]);
		fprintf(stderr, "\n");
		return 1;
	}

	opends_free(gbuf);
	opends_handle_deregister(h);
	opends_driver_close();
	cuCtxDestroy(cuctx);
	free(got);
	free(ref);
	close(fd);
	return 0;
}
