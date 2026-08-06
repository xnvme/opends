#define _GNU_SOURCE

#include "test_mt_read.h"

#include <fcntl.h>
#include <unistd.h>

static void *
ref_buf_to_host(void *dst, const void *src, size_t n)
{
	return memcpy(dst, src, n);
}

static void *
ref_buf_acquire(size_t size)
{
	return opends_alloc(size);
}

static void
ref_buf_release(void *buf)
{
	opends_free(buf);
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <path>\n", argv[0]);
		return 1;
	}

	int fd = open(argv[1], O_RDONLY);
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

	fprintf(stderr, "opends_read multithreaded test (ref backend)\n");

	struct mt_read_env env = {
	        .fh = fh,
	        .buf_to_host = ref_buf_to_host,
	        .buf_acquire = ref_buf_acquire,
	        .buf_release = ref_buf_release,
	};
	int failed = run_mt_read_test(&env);

	opends_handle_deregister(fh);
	close(fd);
	opends_driver_close();

	if (failed)
		fprintf(stderr, "%d failure(s)\n", failed);
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
