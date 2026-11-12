/* SPDX-License-Identifier: BSD-3-Clause */
#define _GNU_SOURCE

#include "test_aisio_homi.h"
#include "test_cuda_common.h"
#include "test_write_homi.h"

#include <cuda.h>

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Write buffers of 1 MiB, so a write of a few MiB takes several rounds. */
#define WRITE_BUF_MB "1"
#define BEHIND_A_SIZE ((size_t)4 * 1024 * 1024 + 100)
#define BEHIND_B_OFF ((off_t)8 * 1024 * 1024)
#define FIRST_WRITERS 16
#define FIRST_WRITE_SIZE ((size_t)1024 * 1024)

struct first_writer {
	struct aisio_homi *a;
	pthread_barrier_t *start;
	void *gpu;
	int idx;
	ssize_t ret;
};

static void *
first_writer(void *arg)
{
	struct first_writer *w = arg;

	cuCtxSetCurrent(w->a->cuctx);
	pthread_barrier_wait(w->start);
	w->ret = opends_sync_write(w->a->fh, w->gpu, FIRST_WRITE_SIZE,
	                           (off_t)w->idx * FIRST_WRITE_SIZE, 0);
	return NULL;
}

/* More threads than write buffers start the first writes of a driver at the
 * same moment, so some arrive while every buffer is still being pinned. Each
 * write has to wait for a buffer and complete. */
static int
concurrent_first_writes(const char *path)
{
	struct first_writer w[FIRST_WRITERS];
	pthread_t threads[FIRST_WRITERS];
	pthread_barrier_t start;
	struct aisio_homi a;
	int started = 0;
	int failed = 0;
	void *gpu;
	int err;

	err = aisio_homi_setup_flags(path,
	                             O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, &a);
	if (err < 0) {
		return 1;
	}
	gpu = opends_alloc(FIRST_WRITE_SIZE);
	if (!gpu) {
		fprintf(stderr, "[first writes] allocation failed\n");
		aisio_homi_teardown(&a);
		return 1;
	}
	pthread_barrier_init(&start, NULL, FIRST_WRITERS);
	for (int i = 0; i < FIRST_WRITERS; i++) {
		w[i] = (struct first_writer){
		        .a = &a, .start = &start, .gpu = gpu, .idx = i};
		err = pthread_create(&threads[i], NULL, first_writer, &w[i]);
		if (err != 0) {
			/* the barrier would not open: no way back */
			fprintf(stderr, "[first writes] pthread_create: %d\n",
			        err);
			_exit(1);
		}
		started++;
	}
	for (int i = 0; i < started; i++) {
		pthread_join(threads[i], NULL);
		if (w[i].ret != (ssize_t)FIRST_WRITE_SIZE) {
			fprintf(stderr, "[first writes] write %d: %zd (%s)\n",
			        i, w[i].ret,
			        w[i].ret < 0 ? OPENDS_ERRSTR(w[i].ret)
			                     : "short write");
			failed = 1;
		}
	}
	pthread_barrier_destroy(&start);
	opends_free(gpu);
	aisio_homi_teardown(&a);
	if (!failed) {
		fprintf(stderr,
		        "[first writes] %d concurrent first writes ok\n",
		        FIRST_WRITERS);
	}
	return failed;
}

struct plain_thread_write {
	opends_handle_t fh;
	void *gpu;
	ssize_t ret;
	CUcontext ctx_after;
	int done;
};

static void *
plain_thread_writer(void *arg)
{
	struct plain_thread_write *t = arg;

	t->ret = opends_sync_write(t->fh, t->gpu, PAGE, 0, 0);
	cuCtxGetCurrent(&t->ctx_after);
	__atomic_store_n(&t->done, 1, __ATOMIC_RELEASE);
	return NULL;
}

/* A write from a thread with no CUDA context current, as the first write of
 * the process, so no write buffer is pinned yet. It has to complete and to
 * leave the thread without a context. Leaves the file empty. */
static int
write_from_plain_thread(struct aisio_homi *a)
{
	struct plain_thread_write t = {.fh = a->fh, .ctx_after = NULL};
	struct timespec nap = {0, 1000000};
	unsigned char image[PAGE];
	unsigned char *host = NULL;
	pthread_t thread;
	int failed = 1;
	ssize_t r;
	int err;

	err = posix_memalign((void **)&host, PAGE, PAGE);
	t.gpu = opends_alloc(PAGE);
	if (err != 0 || !t.gpu) {
		fprintf(stderr, "[plain thread] allocation failed\n");
		goto out;
	}
	for (size_t i = 0; i < PAGE; i++) {
		image[i] = write_pattern_byte(i);
	}
	cuda_buf_from_host(t.gpu, image, PAGE);

	err = pthread_create(&thread, NULL, plain_thread_writer, &t);
	if (err != 0) {
		fprintf(stderr, "[plain thread] pthread_create: %d\n", err);
		goto out;
	}
	for (int ms = 0; ms < 10000; ms++) {
		if (__atomic_load_n(&t.done, __ATOMIC_ACQUIRE)) {
			break;
		}
		nanosleep(&nap, NULL);
	}
	if (!__atomic_load_n(&t.done, __ATOMIC_ACQUIRE)) {
		fprintf(stderr, "[plain thread] write did not complete\n");
		/* the thread is stuck in the write: no join, no close */
		return -1;
	}
	pthread_join(thread, NULL);
	if (t.ret != PAGE) {
		fprintf(stderr, "[plain thread] write: %zd (%s)\n", t.ret,
		        t.ret < 0 ? OPENDS_ERRSTR(t.ret) : "short write");
		goto out;
	}
	if (t.ctx_after) {
		fprintf(stderr, "[plain thread] write left a context\n");
		goto out;
	}
	r = pread(a->fd, host, PAGE, 0);
	if (r != PAGE || memcmp(host, image, PAGE)) {
		fprintf(stderr, "[plain thread] read-back failed (r=%zd)\n", r);
		goto out;
	}
	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("[plain thread] ftruncate");
		goto out;
	}
	fprintf(stderr, "[plain thread] write without a context ok\n");
	failed = 0;
out:
	free(host);
	if (t.gpu) {
		opends_free(t.gpu);
	}
	return failed;
}

/* Write A takes several rounds and ends inside a block, so it pads its last
 * block and cuts the file back to its end. Write B lands behind A while A is
 * in its rounds. B has to stay. */
static int
write_behind_unaligned_write(struct aisio_homi *a)
{
	const off_t a_tail = (off_t)(BEHIND_A_SIZE & ~(size_t)(PAGE - 1));
	const size_t a_rest = BEHIND_A_SIZE & (PAGE - 1);
	const off_t want = BEHIND_B_OFF + PAGE;
	opends_async_future_t fa, fb;
	unsigned char *image = malloc(BEHIND_A_SIZE);
	unsigned char *host = NULL;
	void *gpu = opends_alloc(BEHIND_A_SIZE);
	opends_error_t oerr;
	struct stat st;
	int failed = 1;
	ssize_t wa, wb, r;
	int err;

	err = posix_memalign((void **)&host, PAGE, PAGE);
	if (err != 0 || !image || !gpu) {
		fprintf(stderr, "[behind] allocation failed\n");
		goto out;
	}
	for (size_t i = 0; i < BEHIND_A_SIZE; i++) {
		image[i] = write_pattern_byte(i);
	}
	cuda_buf_from_host(gpu, image, BEHIND_A_SIZE);
	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("[behind] ftruncate");
		goto out;
	}

	oerr = opends_async_write(a->fh, gpu, BEHIND_A_SIZE, 0, 0, &fa);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[behind] submit A: %s\n",
		        opends_op_status_error(oerr.err));
		goto out;
	}
	oerr = opends_async_write(a->fh, gpu, PAGE, BEHIND_B_OFF, 0, &fb);
	wa = opends_async_await(&fa);
	if (oerr.err != OPENDS_SUCCESS) {
		fprintf(stderr, "[behind] submit B: %s\n",
		        opends_op_status_error(oerr.err));
		goto out;
	}
	wb = opends_async_await(&fb);
	if (wa != (ssize_t)BEHIND_A_SIZE || wb != PAGE) {
		fprintf(stderr, "[behind] writes: A %zd, B %zd\n", wa, wb);
		goto out;
	}

	err = fstat(a->fd, &st);
	if (err != 0 || st.st_size != want) {
		fprintf(stderr, "[behind] size %lld, want %lld\n",
		        err != 0 ? -1LL : (long long)st.st_size,
		        (long long)want);
		goto out;
	}
	r = pread(a->fd, host, PAGE, BEHIND_B_OFF);
	if (r != PAGE || memcmp(host, image, PAGE)) {
		fprintf(stderr, "[behind] B is not in the file (r=%zd)\n", r);
		goto out;
	}
	r = pread(a->fd, host, PAGE, a_tail);
	if (r != PAGE || memcmp(host, image + a_tail, a_rest)) {
		fprintf(stderr, "[behind] A's end is not in the file (r=%zd)\n",
		        r);
		goto out;
	}
	err = ftruncate(a->fd, 0);
	if (err != 0) {
		perror("[behind] ftruncate");
		goto out;
	}
	fprintf(stderr,
	        "[behind] a write behind an unaligned write stays ok\n");
	failed = 0;
out:
	free(image);
	free(host);
	if (gpu) {
		opends_free(gpu);
	}
	return failed;
}

static ssize_t
sync_submit_write(struct write_homi_env *e, void *gpu, size_t size, off_t foff)
{
	return opends_sync_write(e->fh, gpu, size, foff, 0);
}

int
main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "usage: %s <scratch-file-on-mount>\n", argv[0]);
		return 1;
	}
	const char *path = argv[1];

	setenv("OPENDS_AISIO_WRITE_BUF_MB", WRITE_BUF_MB, 1);

	fprintf(stderr, "opends_sync_write tests (aisio backend, HOMI)\n");

	int first = concurrent_first_writes(path);

	struct aisio_homi a;
	if (aisio_homi_setup_flags(path, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT,
	                           &a) < 0)
		return 1;

	int plain = write_from_plain_thread(&a);
	if (plain < 0) {
		fprintf(stderr, "1 test(s) failed\n");
		fflush(NULL);
		_exit(1);
	}

	struct write_homi_env env = {
	        .fh = a.fh,
	        .fd = a.fd,
	        .stream = NULL,
	        .submit_write = sync_submit_write,
	        .mode_label = "sync",
	};
	int failed = first + plain + run_write_homi_tests(&env);

	failed += write_behind_unaligned_write(&a);

	int bfd = open(path, O_RDWR);
	opends_handle_t bfh = NULL;
	opends_error_t rerr = opends_handle_register(&bfh, bfd);
	if (bfd < 0 || rerr.err != OPENDS_DIO_NOT_SET) {
		fprintf(stderr, "buffered fd not refused (fd %d, err %d)\n",
		        bfd, rerr.err);
		if (rerr.err == OPENDS_SUCCESS)
			opends_handle_deregister(bfh);
		failed++;
	} else {
		fprintf(stderr, "buffered fd refused ok\n");
	}
	if (bfd >= 0)
		close(bfd);

	aisio_homi_teardown(&a);
	unlink(path);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
