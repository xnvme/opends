/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_sync_read.h"
#include "test_mem.h"

#include <fcntl.h>
#include <stdlib.h>

static void *
ref_buf_to_host(void *dst, const void *src, size_t n)
{
	return memcpy(dst, src, n);
}

static void
ref_buf_zero(void *buf, size_t n)
{
	memset(buf, 0, n);
}

static void *
ref_alloc_acquire(size_t size)
{
	return test_dev_alloc(size);
}

static void
ref_alloc_release(void *buf)
{
	opends_free(buf);
}

#define PAGE_ALIGN(x) (((x) + 4095) & ~((size_t)4095))

static void *
ref_register_acquire(size_t size)
{
	size_t aligned = PAGE_ALIGN(size);
	void *buf = aligned_alloc(4096, aligned);
	if (!buf)
		return NULL;
	opends_error_t err = opends_buf_register(buf, aligned, 0);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  buf_register: %s\n",
		        opends_op_status_error(err.err));
		free(buf);
		return NULL;
	}
	return buf;
}

static void
ref_register_release(void *buf)
{
	if (!buf)
		return;
	opends_buf_deregister(buf);
	free(buf);
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <path>\n", argv[0]);
		return 1;
	}

	int fd = open(argv[1], O_RDONLY | O_DIRECT);
	if (fd < 0) {
		perror("open");
		return 1;
	}

	opends_error_t err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "driver_open: %s\n",
		        opends_op_status_error(err.err));
		close(fd);
		return 1;
	}

	opends_handle_t fh;
	err = opends_handle_register(&fh, fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "handle_register: %s\n",
		        opends_op_status_error(err.err));
		close(fd);
		return 1;
	}

	fprintf(stderr, "opends_sync_read sync tests (ref backend)\n");

	struct test_env env_alloc = {
	        .fh = fh,
	        .buf_to_host = ref_buf_to_host,
	        .buf_zero = ref_buf_zero,
	        .buf_acquire = ref_alloc_acquire,
	        .buf_release = ref_alloc_release,
	        .mode_label = "alloc",
	};
	int failed = run_sync_read_tests(&env_alloc);

	struct test_env env_register = {
	        .fh = fh,
	        .buf_to_host = ref_buf_to_host,
	        .buf_zero = ref_buf_zero,
	        .buf_acquire = ref_register_acquire,
	        .buf_release = ref_register_release,
	        .mode_label = "register",
	};
	failed += run_sync_read_tests(&env_register);

	opends_handle_deregister(fh);
	close(fd);
	opends_driver_close();

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
