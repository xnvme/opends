/*
 * test_mt_read.h - Backend-agnostic multithreaded opends_read test.
 *
 * Include from a backend-specific source file that provides main().
 * N threads start behind a barrier and hammer opends_read on a shared
 * handle, each with its own buffer, verifying every read against the
 * in-memory pattern from read_pattern.h. Each thread walks the case
 * table from a different starting index so concurrent reads cover
 * different (offset, size) pairs.
 *
 * Expects the 16-page pattern file written by test_sync_read_prep.
 */
#ifndef TEST_MT_READ_H_
#define TEST_MT_READ_H_

#include "opends.h"
#include "read_pattern.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MT_READ_THREADS 4
#define MT_READ_ITERS 64

struct mt_read_env {
	opends_handle_t fh;
	void *(*buf_to_host)(void *dst, const void *src, size_t n);
	void *(*buf_acquire)(size_t size);
	void (*buf_release)(void *buf);
	/* Optional per-thread setup (e.g. make a CUDA context current). */
	void (*thread_init)(void *arg);
	void *thread_init_arg;
};

static const struct {
	off_t offset;
	size_t size;
} mt_read_cases[] = {
        {0, PAGE},
        {0, FILE_SIZE},
        {PAGE, 2 * PAGE},
        {2 * PAGE, PAGE},
        {4 * PAGE, 4 * PAGE},
        {8 * PAGE, 8 * PAGE},
        {(FILE_PAGES - 1) * PAGE, PAGE},
        {0, FILE_SIZE / 2},
};

#define MT_READ_NCASES (sizeof(mt_read_cases) / sizeof(mt_read_cases[0]))

struct mt_read_thread {
	struct mt_read_env *env;
	pthread_barrier_t *barrier;
	int id;
	int failures;
};

static void *
mt_read_thread_main(void *arg)
{
	struct mt_read_thread *t = arg;
	struct mt_read_env *env = t->env;

	if (env->thread_init)
		env->thread_init(env->thread_init_arg);

	void *buf = env->buf_acquire(FILE_SIZE);
	char *host = malloc(FILE_SIZE);
	if (!buf || !host) {
		fprintf(stderr, "  thread %d: buffer setup failed\n", t->id);
		t->failures = 1;
		free(host);
		if (buf)
			env->buf_release(buf);
		pthread_barrier_wait(t->barrier);
		return NULL;
	}

	pthread_barrier_wait(t->barrier);

	for (int i = 0; i < MT_READ_ITERS; i++) {
		size_t ci = ((size_t)t->id + (size_t)i) % MT_READ_NCASES;
		off_t off = mt_read_cases[ci].offset;
		size_t size = mt_read_cases[ci].size;

		ssize_t n = opends_read(env->fh, buf, size, off, 0);
		if (n != (ssize_t)size) {
			fprintf(stderr,
			        "  thread %d iter %d: read(off=%ld, size=%zu) "
			        "rc=%zd\n",
			        t->id, i, (long)off, size, n);
			t->failures++;
			continue;
		}

		env->buf_to_host(host, buf, size);
		for (size_t b = 0; b < size; b++) {
			if ((unsigned char)host[b] !=
			    pattern_byte(off + (off_t)b)) {
				fprintf(stderr,
				        "  thread %d iter %d: mismatch at "
				        "byte %zu (off=%ld, size=%zu)\n",
				        t->id, i, b, (long)off, size);
				t->failures++;
				break;
			}
		}
	}

	free(host);
	env->buf_release(buf);
	return NULL;
}

static int
run_mt_read_test(struct mt_read_env *env)
{
	pthread_t threads[MT_READ_THREADS];
	struct mt_read_thread state[MT_READ_THREADS];
	pthread_barrier_t barrier;

	pthread_barrier_init(&barrier, NULL, MT_READ_THREADS);

	for (int i = 0; i < MT_READ_THREADS; i++) {
		state[i] = (struct mt_read_thread){
		        .env = env,
		        .barrier = &barrier,
		        .id = i,
		};
		if (pthread_create(&threads[i], NULL, mt_read_thread_main,
		                   &state[i]) != 0) {
			fprintf(stderr, "pthread_create(%d) failed\n", i);
			exit(1);
		}
	}

	int failures = 0;
	for (int i = 0; i < MT_READ_THREADS; i++) {
		pthread_join(threads[i], NULL);
		failures += state[i].failures;
	}
	pthread_barrier_destroy(&barrier);

	fprintf(stderr, "  %-24s %s\n", "mt_read", failures ? "FAIL" : "ok");
	return failures;
}

#endif /* TEST_MT_READ_H_ */
