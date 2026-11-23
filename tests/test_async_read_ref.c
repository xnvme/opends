/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_ref_common.h"
#include "test_async_read.h"

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <path>\n", argv[0]);
		return 1;
	}

	int fd;
	opends_handle_t fh;
	if (ref_setup(argv[1], &fd, &fh) < 0)
		return 1;

	fprintf(stderr, "opends_async read test (ref backend)\n");

	struct async_read_env env = {
	        .fh = fh,
	        .buf_to_host = ref_buf_to_host,
	        .buf_acquire = ref_alloc_acquire,
	        .buf_release = ref_alloc_release,
	};
	int failed = run_async_read_test(&env);

	ref_teardown(fd, fh);

	if (failed)
		fprintf(stderr, "%d failure(s)\n", failed);
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
