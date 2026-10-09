// SPDX-License-Identifier: BSD-3-Clause
/*
 * Compute kernels and stream reads on one or more CUDA streams, with the
 * host's part measured.
 *
 * Each iteration enqueues: a kernel that overwrites the buffer, a stamp of the
 * GPU clock, opends_stream_read of the file into the buffer, another stamp,
 * and a kernel that sums the buffer. Everything is enqueued up front. The host
 * then sleeps, and samples every thread's CPU time from /proc before and
 * after. The sums must equal the host's sum over the file, which shows each
 * read landed between its two kernels; the per-thread CPU times show which
 * host threads took part while the chain ran; the stamps bound each read on
 * the GPU timeline.
 *
 * With several streams the ops go round-robin over them, each stream with a
 * buffer of its own, so the engines are also compared under concurrency.
 * AISIO_DEMO_COMPUTE_US=N puts N microseconds of GPU compute ahead of every
 * read, so the reads are spaced out the way they would be between real
 * kernels. The last line is a machine-readable summary.
 *
 * Usage: aisio_stream_compute <file-on-mount> [iters] [host-sleep-ms]
 *                             [read-bytes] [streams]
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <opends.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_READ_BYTES (256u << 20)
#define MAX_STREAMS 32
#define MAX_THREADS 64
#define THREADS 256
#define FILL_PATTERN 0xA5A5A5A5A5A5A5A5ull

struct thread_cpu {
	int tid;
	char comm[32];
	uint64_t run_ns;
};

struct cpu_snapshot {
	struct thread_cpu t[MAX_THREADS];
	int n;
};

static __global__ void
fill_kernel(uint64_t *p, size_t n, uint64_t v)
{
	size_t stride = (size_t)gridDim.x * blockDim.x;

	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
	     i += stride)
		p[i] = v;
}

/* Stands in for compute between reads: one thread busy-waits ns. */
static __global__ void
spin_kernel(uint64_t ns)
{
	uint64_t t0, t;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
	do {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	} while (t - t0 < ns);
}

static __global__ void
stamp_kernel(uint64_t *ts)
{
	uint64_t t;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	*ts = t;
}

static __global__ void
sum_kernel(const uint64_t *p, size_t n, unsigned long long *out)
{
	__shared__ unsigned long long sh[THREADS];
	size_t stride = (size_t)gridDim.x * blockDim.x;
	unsigned long long acc = 0;

	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
	     i += stride)
		acc += p[i];
	sh[threadIdx.x] = acc;
	__syncthreads();
	for (unsigned s = THREADS / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s)
			sh[threadIdx.x] += sh[threadIdx.x + s];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(out, sh[0]);
}

static double
now_ms(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void
sleep_ms(long ms)
{
	struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};

	nanosleep(&ts, NULL);
}

/* Time on CPU in ns (schedstat) and the name of one thread. */
static int
read_thread_cpu(int tid, struct thread_cpu *out)
{
	char path[64];
	FILE *f;
	int n;

	snprintf(path, sizeof(path), "/proc/self/task/%d/schedstat", tid);
	f = fopen(path, "r");
	if (!f)
		return -1;
	n = fscanf(f, "%" SCNu64, &out->run_ns);
	fclose(f);
	if (n != 1)
		return -1;

	snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
	f = fopen(path, "r");
	if (!f)
		return -1;
	if (!fgets(out->comm, sizeof(out->comm), f))
		out->comm[0] = '\0';
	fclose(f);
	out->comm[strcspn(out->comm, "\n")] = '\0';
	out->tid = tid;
	return 0;
}

static void
snapshot_cpu(struct cpu_snapshot *s)
{
	DIR *d = opendir("/proc/self/task");
	struct dirent *e;

	s->n = 0;
	if (!d)
		return;
	while ((e = readdir(d)) && s->n < MAX_THREADS) {
		if (e->d_name[0] == '.')
			continue;
		if (read_thread_cpu(atoi(e->d_name), &s->t[s->n]) == 0)
			s->n++;
	}
	closedir(d);
}

/* 0 for a thread that did not exist yet. */
static uint64_t
run_ns_of(const struct cpu_snapshot *s, int tid)
{
	for (int i = 0; i < s->n; i++)
		if (s->t[i].tid == tid)
			return s->t[i].run_ns;
	return 0;
}

/* Per-thread CPU time between two snapshots; returns the total in ms. */
static double
report_cpu(const struct cpu_snapshot *a, const struct cpu_snapshot *b,
           double wall_ms)
{
	double total = 0;

	for (int i = 0; i < b->n; i++) {
		double ms = (b->t[i].run_ns - run_ns_of(a, b->t[i].tid)) / 1e6;

		total += ms;
		printf("    tid %-7d %-16s %9.3f ms\n", b->t[i].tid,
		       b->t[i].comm, ms);
	}
	printf("    %-24s %9.3f ms  (%.2f%% of one core over %.0f ms)\n",
	       "total", total, 100.0 * total / wall_ms, wall_ms);
	return total;
}

static int
host_sum(int fd, size_t nbytes, unsigned long long *out)
{
	void *host = NULL;
	const uint64_t *w;
	unsigned long long acc = 0;
	size_t done = 0;

	if (posix_memalign(&host, 4096, nbytes))
		return -1;
	while (done < nbytes) {
		ssize_t n = pread(fd, (char *)host + done, nbytes - done, done);

		if (n <= 0) {
			free(host);
			return -1;
		}
		done += (size_t)n;
	}
	w = (const uint64_t *)host;
	for (size_t i = 0; i < nbytes / sizeof(uint64_t); i++)
		acc += w[i];
	free(host);
	*out = acc;
	return 0;
}

int
main(int argc, char **argv)
{
	const char *path;
	int iters = 20;
	long host_sleep_ms = 1500;
	size_t read_bytes = 0;
	int nstreams = 1;
	const char *gpu_env = getenv("OPENDS_AISIO_GPU_INITIATED");
	bool gpu_engine = gpu_env && gpu_env[0] && gpu_env[0] != '0';
	const char *plain_env = getenv("AISIO_DEMO_PLAIN_RESULTS");
	bool want_slots = !(plain_env && plain_env[0] && plain_env[0] != '0');
	bool slots = false;
	const char *compute_env = getenv("AISIO_DEMO_COMPUTE_US");
	long compute_us = compute_env ? atol(compute_env) : 0;
	CUdevice cudev;
	CUcontext cuctx;
	CUstream streams[MAX_STREAMS];
	CUresult cres;
	opends_error_t err;
	opends_handle_t fh = NULL;
	struct stat st;
	size_t nbytes, n_words;
	unsigned grid;
	unsigned long long ref = 0;
	void *bufs[MAX_STREAMS];
	uint64_t *ts_dev, *ts;
	unsigned long long *sums_dev, *sums;
	size_t *sz;
	off_t *foff, *boff;
	ssize_t *bytes;
	struct cpu_snapshot s0, s1, s2;
	double t0, t1, t_wake, t2, cpu_enq, cpu_run;
	bool done_asleep;
	int polls = 0, bad = 0, fd, rc = 1, created = 0;

	if (argc < 2 || argc > 6) {
		fprintf(stderr,
		        "usage: %s <file-on-mount> [iters] [host-sleep-ms] "
		        "[read-bytes] [streams]\n",
		        argv[0]);
		return 2;
	}
	path = argv[1];
	if (argc > 2)
		iters = atoi(argv[2]);
	if (argc > 3)
		host_sleep_ms = atol(argv[3]);
	if (argc > 4)
		read_bytes = strtoull(argv[4], NULL, 0);
	if (argc > 5)
		nstreams = atoi(argv[5]);
	if (iters < 1 || host_sleep_ms < 0 || nstreams < 1 ||
	    nstreams > MAX_STREAMS) {
		fprintf(stderr, "bad iters, sleep or streams\n");
		return 2;
	}

	cuInit(0);
	cuDeviceGet(&cudev, 0);
#if CUDA_VERSION >= 13000
	cres = cuCtxCreate(&cuctx, NULL, 0, cudev);
#else
	cres = cuCtxCreate(&cuctx, 0, cudev);
#endif
	if (cres != CUDA_SUCCESS) {
		fprintf(stderr, "cuCtxCreate failed: %d\n", (int)cres);
		return 1;
	}

	err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr,
		        "driver_open: %s (is the homi stack running?)\n",
		        opends_op_status_error(err.err));
		return 1;
	}

	fd = open(path, O_RDONLY | O_DIRECT);
	if (fd < 0 || fstat(fd, &st) < 0) {
		perror(path);
		goto out_driver;
	}
	nbytes = (size_t)st.st_size;
	if (nbytes > MAX_READ_BYTES)
		nbytes = MAX_READ_BYTES;
	if (read_bytes && read_bytes < nbytes)
		nbytes = read_bytes;
	nbytes &= ~(size_t)4095;
	if (!nbytes) {
		fprintf(stderr, "%s: too small\n", path);
		goto out_fd;
	}
	n_words = nbytes / sizeof(uint64_t);
	grid = (unsigned)((n_words + THREADS - 1) / THREADS);
	if (grid > 1024)
		grid = 1024;

	err = opends_handle_register(&fh, fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "handle_register: %s\n",
		        opends_op_status_error(err.err));
		goto out_fd;
	}
	if (host_sum(fd, nbytes, &ref) < 0) {
		fprintf(stderr, "host read of %s failed\n", path);
		goto out_handle;
	}

	memset(bufs, 0, sizeof(bufs));
	for (int s = 0; s < nstreams; s++) {
		bufs[s] = opends_alloc(nbytes);
		if (!bufs[s]) {
			fprintf(stderr, "opends_alloc(%zu) failed\n", nbytes);
			goto out_buf;
		}
	}
	for (int s = 0; s < nstreams; s++) {
		if (cuStreamCreate(&streams[s], CU_STREAM_NON_BLOCKING) !=
		    CUDA_SUCCESS) {
			fprintf(stderr, "cuStreamCreate failed\n");
			goto out_stream;
		}
		created = s + 1;
		err = opends_stream_register(streams[s],
		                             OPENDS_STREAM_FIXED_SHAPE);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "stream_register: %s\n",
			        opends_op_status_error(err.err));
			goto out_stream;
		}
	}
	if (cudaMalloc((void **)&ts_dev, 2 * iters * sizeof(*ts_dev)) ||
	    cudaMalloc((void **)&sums_dev, iters * sizeof(*sums_dev)) ||
	    cudaMemset(sums_dev, 0, iters * sizeof(*sums_dev))) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto out_stream;
	}
	ts = (uint64_t *)calloc(2 * iters, sizeof(*ts));
	sums = (unsigned long long *)calloc(iters, sizeof(*sums));
	sz = (size_t *)calloc(iters, sizeof(*sz));
	foff = (off_t *)calloc(iters, sizeof(*foff));
	boff = (off_t *)calloc(iters, sizeof(*boff));
	/* bytes_read lands in a result block so the GPU engine can write it
	 * from the device; AISIO_DEMO_PLAIN_RESULTS=1 keeps it in plain memory
	 * to compare against the callback path. */
	bytes = NULL;
	if (want_slots)
		bytes = (ssize_t *)opends_result_alloc(iters * sizeof(*bytes));
	if (bytes) {
		memset(bytes, 0, iters * sizeof(*bytes));
		slots = true;
	} else {
		bytes = (ssize_t *)calloc(iters, sizeof(*bytes));
	}
	if (!ts || !sums || !sz || !foff || !boff || !bytes) {
		fprintf(stderr, "calloc failed\n");
		goto out_stream;
	}

	printf("aisio_stream_compute: %d x (fill, read %zu KiB, sum) on %d "
	       "stream%s, GPU engine %s, results in %s, %ld us of compute "
	       "before each read\n",
	       iters, nbytes >> 10, nstreams, nstreams > 1 ? "s" : "",
	       gpu_engine ? "on" : "off",
	       slots ? "a result block" : "plain memory", compute_us);

	snapshot_cpu(&s0);
	t0 = now_ms();
	for (int i = 0; i < iters; i++) {
		cudaStream_t cs = (cudaStream_t)streams[i % nstreams];
		void *buf = bufs[i % nstreams];

		fill_kernel<<<grid, THREADS, 0, cs>>>((uint64_t *)buf, n_words,
		                                      FILL_PATTERN);
		if (compute_us > 0)
			spin_kernel<<<1, 1, 0, cs>>>((uint64_t)compute_us *
			                             1000);
		stamp_kernel<<<1, 1, 0, cs>>>(&ts_dev[2 * i]);
		sz[i] = nbytes;
		err = opends_stream_read(
		        fh, buf, &sz[i], &foff[i], &boff[i], &bytes[i],
		        (opends_stream_t)streams[i % nstreams]);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "stream_read %d: %s\n", i,
			        opends_op_status_error(err.err));
			goto out_stream;
		}
		stamp_kernel<<<1, 1, 0, cs>>>(&ts_dev[2 * i + 1]);
		sum_kernel<<<grid, THREADS, 0, cs>>>((const uint64_t *)buf,
		                                     n_words, &sums_dev[i]);
	}
	t1 = now_ms();
	snapshot_cpu(&s1);

	/* Nothing on this thread drives the chain from here on. */
	sleep_ms(host_sleep_ms);
	cres = CUDA_SUCCESS;
	for (int s = 0; s < nstreams && cres == CUDA_SUCCESS; s++)
		cres = cuStreamQuery(streams[s]);
	done_asleep = cres == CUDA_SUCCESS;
	t_wake = now_ms();
	while (cres == CUDA_ERROR_NOT_READY) {
		sleep_ms(1);
		polls++;
		cres = CUDA_SUCCESS;
		for (int s = 0; s < nstreams && cres == CUDA_SUCCESS; s++)
			cres = cuStreamQuery(streams[s]);
	}
	t2 = now_ms();
	snapshot_cpu(&s2);
	if (cres != CUDA_SUCCESS) {
		fprintf(stderr, "stream failed: %d\n", (int)cres);
		goto out_stream;
	}

	cudaMemcpy(ts, ts_dev, 2 * iters * sizeof(*ts), cudaMemcpyDeviceToHost);
	cudaMemcpy(sums, sums_dev, iters * sizeof(*sums),
	           cudaMemcpyDeviceToHost);
	for (int i = 0; i < iters; i++) {
		if (sums[i] != ref || bytes[i] != (ssize_t)nbytes) {
			printf("  iteration %d: sum %#llx vs %#llx, bytes_read "
			       "%zd\n",
			       i, sums[i], ref, bytes[i]);
			bad++;
		}
	}
	printf("  data: %s\n", bad ? "MISMATCH"
	                           : "every sum matches the host's, every "
	                             "bytes_read is the full size");

	printf("  enqueue: %.1f ms wall\n", t1 - t0);
	cpu_enq = report_cpu(&s0, &s1, t1 - t0);

	printf("  run: host slept %ld ms, chain %s while it slept%s\n",
	       host_sleep_ms, done_asleep ? "finished" : "was still running",
	       done_asleep ? "" : "; polled until done");
	if (!done_asleep)
		printf("  run: finished %.1f ms after waking (%d polls)\n",
		       t2 - t_wake, polls);
	printf("  run: %.1f ms wall from the last enqueue to completion\n",
	       t2 - t1);
	cpu_run = report_cpu(&s1, &s2, t2 - t1);

	{
		double lo = 1e300, hi = 0, acc = 0, chain, caller_ms = 0,
		       io_ms = 0, cpu_all = cpu_enq + cpu_run;
		uint64_t first = UINT64_MAX, last = 0;

		for (int i = 0; i < iters; i++) {
			double w = (ts[2 * i + 1] - ts[2 * i]) / 1e6;

			if (w < lo)
				lo = w;
			if (w > hi)
				hi = w;
			acc += w;
			if (ts[2 * i] < first)
				first = ts[2 * i];
			if (ts[2 * i + 1] > last)
				last = ts[2 * i + 1];
		}
		chain = (last - first) / 1e6;
		/* The caller's thread and the aisio I/O workers over the whole
		 * interval. */
		for (int i = 0; i < s2.n; i++) {
			double ms =
			        (s2.t[i].run_ns - run_ns_of(&s0, s2.t[i].tid)) /
			        1e6;

			if (s2.t[i].tid == getpid())
				caller_ms = ms;
			else if (!strcmp(s2.t[i].comm, "aisio-io"))
				io_ms += ms;
		}
		printf("  GPU timeline: read window min/avg/max %.3f/%.3f/%.3f "
		       "ms "
		       "(%.2f GB/s per read); all reads span %.1f ms: %.0f "
		       "reads/s, %.2f GB/s aggregate\n",
		       lo, acc / iters, hi, nbytes / (acc / iters) / 1e6, chain,
		       iters / (chain / 1e3),
		       (double)nbytes * iters / chain / 1e6);
		printf("SUMMARY engine=%s results=%s compute_us=%ld "
		       "read_kib=%zu streams=%d "
		       "iters=%d "
		       "avg_ms=%.3f max_ms=%.3f reads_per_s=%.0f gbps=%.2f "
		       "wall_ms=%.1f cpu_ms=%.1f cpu_caller_ms=%.1f "
		       "cpu_io_ms=%.1f "
		       "cpu_pct=%.1f\n",
		       gpu_engine ? "gpu" : "host", slots ? "slots" : "plain",
		       compute_us, nbytes >> 10, nstreams, iters, acc / iters,
		       hi, iters / (chain / 1e3),
		       (double)nbytes * iters / chain / 1e6, t2 - t0, cpu_all,
		       caller_ms, io_ms, 100.0 * cpu_all / (t2 - t0));
	}
	rc = bad ? 1 : 0;

out_stream:
	for (int s = 0; s < created; s++)
		cuStreamDestroy(streams[s]);
out_buf:
	for (int s = 0; s < nstreams; s++)
		if (bufs[s])
			opends_free(bufs[s]);
out_handle:
	opends_handle_deregister(fh);
out_fd:
	close(fd);
out_driver:
	opends_driver_close();
	cuCtxDestroy(cuctx);
	return rc;
}
