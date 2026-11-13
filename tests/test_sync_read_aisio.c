/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_cuda_common.h"
#include "test_sync_read.h"

#include <cuda.h>

#include <stdbool.h>
#include <stdio.h>

int
main(int argc, char **argv)
{
	if (argc != 2 && argc != 3) {
		fprintf(stderr,
		        "usage: %s <pattern-file-on-mount> [<homi-chardev>]\n",
		        argv[0]);
		return 1;
	}

	struct aisio_homi a;
	if (aisio_homi_setup(argv[1], &a) < 0)
		return 1;

	fprintf(stderr, "opends_sync_read sync tests (aisio backend, HOMI)\n");

	struct test_env env_alloc = {
	        .fh = a.fh,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .check_buffer = cuda_check_buffer,
	        .buf_acquire = cuda_alloc_acquire,
	        .buf_release = cuda_alloc_release,
	        .mode_label = "alloc",
	};
	int failed = run_sync_read_tests(&env_alloc);

	struct test_env env_register = {
	        .fh = a.fh,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .check_buffer = cuda_check_buffer,
	        .buf_acquire = cuda_register_acquire,
	        .buf_release = cuda_register_release,
	        .mode_label = "register",
	};
	failed += run_sync_read_tests(&env_register);

	struct test_env env_packed = {
	        .fh = a.fh,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_zero = cuda_buf_zero,
	        .check_buffer = cuda_check_buffer,
	        .buf_acquire = cuda_register_packed_acquire,
	        .buf_release = cuda_register_packed_release,
	        .mode_label = "register-packed",
	};
	failed += run_sync_read_tests(&env_packed);

	/* The chunk mapping behind a registration covers more than the
	 * buffer, so the backend has to refuse I/O past it itself. */
	void *small = cuda_register_acquire(PAGE);
	if (!small) {
		failed++;
	} else {
		ssize_t n = opends_sync_read(a.fh, small, 2 * PAGE, 0, 0);
		if (n == -(ssize_t)OPENDS_MEMORY_NOT_REGISTERED) {
			fprintf(stderr, "PASS: read past the registered range "
			                "refused\n");
		} else {
			fprintf(stderr,
			        "FAIL: read past the registered range returned "
			        "%zd\n",
			        n);
			failed++;
		}
		cuda_register_release(small);
	}

	/* A read goes to the device without passing the kernel, so the driver
	 * has to honour the access mode of the fd itself. */
	int wo_fd = open(argv[1], O_WRONLY | O_DIRECT);
	opends_handle_t wo_fh = NULL;
	void *wo_buf = cuda_alloc_acquire(PAGE);
	if (wo_fd < 0 || !wo_buf ||
	    opends_handle_register(&wo_fh, wo_fd).err != OPENDS_SUCCESS ||
	    opends_sync_read(wo_fh, wo_buf, PAGE, 0, 0) !=
	            -(ssize_t)OPENDS_INVALID_FILE_OPEN_FLAG) {
		fprintf(stderr, "FAIL: read through a write-only fd\n");
		failed++;
	} else {
		fprintf(stderr, "PASS: read through a write-only fd refused\n");
	}
	if (wo_fh)
		opends_handle_deregister(wo_fh);
	if (wo_buf)
		cuda_alloc_release(wo_buf);
	if (wo_fd >= 0)
		close(wo_fd);

	/* homi's character device is a raw handle in raw block mode, and no
	 * handle at all without it: it is not opened with O_DIRECT. */
	const char *raw = getenv("OPENDS_AISIO_RAW_BLOCK");
	bool raw_mode = raw && raw[0] == '1';
	if (argc == 3) {
		opends_op_error_t want =
		        raw_mode ? OPENDS_SUCCESS : OPENDS_DIO_NOT_SET;
		int cfd = open(argv[2], O_RDWR);
		opends_handle_t cfh = NULL;
		opends_op_error_t got =
		        cfd < 0 ? OPENDS_INVALID_VALUE
		                : opends_handle_register(&cfh, cfd).err;
		fprintf(stderr, "%s: register %s: %s\n",
		        got == want ? "PASS" : "FAIL", argv[2],
		        opends_op_status_error(got));
		failed += got != want;
		if (cfh)
			opends_handle_deregister(cfh);
		if (cfd >= 0)
			close(cfd);
	}

	/* A handle outlives the driver. In raw block mode the reopened driver
	 * has no extent index until a regular file is registered: a read
	 * through the kept handle fails before that and works after it. */
	opends_driver_close();
	opends_error_t err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "FAIL: driver reopen: %s\n",
		        opends_op_status_error(err.err));
		failed++;
	} else {
		void *buf = cuda_alloc_acquire(PAGE);
		int bad = !buf;
		if (buf && raw_mode) {
			int fd2 = open(argv[1], O_RDONLY | O_DIRECT);
			opends_handle_t fh2 = NULL;

			bad = opends_sync_read(a.fh, buf, PAGE, 0, 0) !=
			              -(ssize_t)OPENDS_FS_SETUP_ERROR ||
			      fd2 < 0 ||
			      opends_handle_register(&fh2, fd2).err !=
			              OPENDS_SUCCESS;
			fprintf(stderr,
			        "%s: no index after a driver reopen until a "
			        "file is registered\n",
			        bad ? "FAIL" : "PASS");
			if (fh2)
				opends_handle_deregister(fh2);
			if (fd2 >= 0)
				close(fd2);
		}
		bad = bad || verify_read(&env_alloc, buf, PAGE, PAGE, 0, 0,
		                         "read after driver reopen");
		fprintf(stderr,
		        "%s: read through a handle kept across a "
		        "driver reopen\n",
		        bad ? "FAIL" : "PASS");
		failed += bad;
		if (buf)
			cuda_alloc_release(buf);
	}

	aisio_homi_teardown(&a);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
