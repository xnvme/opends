#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_cuda_common.h"
#include "test_mt_read.h"

#include <cuda.h>

#include <stdio.h>

/* Worker threads need the main thread's CUDA context current: the read
 * path stages sub-LBA tails through cudaMemcpy. */
static void
mt_thread_init(void *arg)
{
	cuCtxSetCurrent((CUcontext)arg);
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <pattern-file-on-mount>\n", argv[0]);
		return 1;
	}

	struct aisio_homi a;
	if (aisio_homi_setup(argv[1], &a) < 0)
		return 1;

	fprintf(stderr,
	        "opends_read multithreaded test (aisio backend, HOMI)\n");

	struct mt_read_env env = {
	        .fh = a.fh,
	        .buf_to_host = cuda_buf_to_host,
	        .buf_acquire = cuda_alloc_acquire,
	        .buf_release = cuda_alloc_release,
	        .thread_init = mt_thread_init,
	        .thread_init_arg = a.cuctx,
	};
	int failed = run_mt_read_test(&env);

	aisio_homi_teardown(&a);

	if (failed)
		fprintf(stderr, "%d failure(s)\n", failed);
	else
		fprintf(stderr, "all ok\n");

	return failed ? 1 : 0;
}
