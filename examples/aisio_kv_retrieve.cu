// SPDX-License-Identifier: BSD-3-Clause
/*
 * KV cache retrieve shaped like LMCache's: chunks of a file land in staging
 * buffers and a scatter kernel writes them into a paged KV cache through a
 * block table. Chunk mode reads a whole chunk, then scatters it. Layer mode
 * reads one layer's slice, scatters it, and runs a compute kernel standing in
 * for that layer's prefill, so the chain is many small reads interleaved with
 * compute. Every request runs on its own stream, everything is enqueued up
 * front, and the host then only waits.
 *
 * Reported per run: time to KV ready per request, the read window per step
 * from GPU stamps, enqueue time, host CPU by thread, and the rate of an
 * optional co-running compute kernel. The cache content is checked against
 * the file's pattern. The last line is a machine-readable summary.
 *
 * Usage: aisio_kv_retrieve <file-on-mount> [key=value ...]
 *   mode=chunk|layer chunk_mib=32 layers=32 tokens=256 chunks=8 requests=1
 *   compute_us=0 coload=0 coload_ms=3000 tag=<label> make_file=<bytes>
 * make_file writes the pattern file through the kernel mount and exits.
 * requests=0 with coload=1 measures the co-runner alone for coload_ms.
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

#define MAX_REQUESTS 32
#define MAX_THREADS 64
#define THREADS 256
#define BLOCK_TOKENS 16
#define PATTERN_MUL 0x9E3779B97F4A7C15ull

struct thread_cpu {
	int tid;
	char comm[32];
	uint64_t run_ns;
};

struct cpu_snapshot {
	struct thread_cpu t[MAX_THREADS];
	int n;
};

struct request {
	CUstream stream;
	void *staging[2];
	uint8_t *cache;
	uint32_t *block_table;
	uint64_t *ts;
	unsigned long long *sum;
};

/* The co-running kernel's stream, stop flag (host-mapped) and round count.
 * Driver API throughout, like the OpenDS stream handles. */
struct coload {
	CUstream stream;
	volatile int *stop;
	CUdeviceptr stop_dptr;
	CUdeviceptr rounds;
	unsigned long long n;
	double t0, t1;
};

struct params {
	bool layer_mode;
	size_t chunk_bytes;
	uint32_t layers;
	uint32_t tokens;
	uint32_t chunks;
	int requests;
	long compute_us;
	bool coload;
	long coload_ms;
	const char *tag;
	uint64_t make_file;
};

static __global__ void
stamp_kernel(uint64_t *ts)
{
	uint64_t t;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	*ts = t;
}

/* Stands in for a layer's compute: one thread busy-waits ns. */
static __global__ void
spin_kernel(uint64_t ns)
{
	uint64_t t0, t;

	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
	do {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	} while (t - t0 < ns);
}

/* src holds nlayers layers of one chunk, laid out [layer][token][tok_vecs]
 * in 16-byte vectors. The cache is [block][token in block][layer][tok_vecs];
 * block_table maps the chunk's token blocks to cache blocks. */
static __global__ void
scatter_kernel(const uint4 *src, uint4 *cache, const uint32_t *block_table,
               uint32_t tokens, uint32_t layers, uint32_t layer0,
               uint32_t nlayers, uint32_t tok_vecs)
{
	size_t per_layer = (size_t)tokens * tok_vecs;
	size_t n = per_layer * nlayers;
	size_t stride = (size_t)gridDim.x * blockDim.x;

	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
	     i += stride) {
		uint32_t l = layer0 + (uint32_t)(i / per_layer);
		size_t j = i % per_layer;
		uint32_t t = (uint32_t)(j / tok_vecs);
		uint32_t v = (uint32_t)(j % tok_vecs);
		uint32_t blk = block_table[t / BLOCK_TOKENS];
		size_t slot = ((size_t)blk * BLOCK_TOKENS + t % BLOCK_TOKENS) *
		                      layers +
		              l;

		cache[slot * tok_vecs + v] = src[i];
	}
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

/* Stands in for a decode batch sharing the GPU: one block per SM runs FMA
 * rounds until the host raises stop, counting rounds. Thread 0 reads the flag
 * and the block agrees on it, so the barriers stay convergent. */
static __global__ void
coload_kernel(const volatile int *stop, unsigned long long *rounds)
{
	__shared__ int go;
	float a = threadIdx.x * 1e-3f;
	const float b = 1.0001f;

	for (;;) {
		if (threadIdx.x == 0)
			go = !*stop;
		__syncthreads();
		if (!go)
			break;
		for (int i = 0; i < 1 << 16; i++)
			a = fmaf(a, b, 0.5f);
		if (threadIdx.x == 0)
			atomicAdd(rounds, 1ull);
		__syncthreads();
	}
	if (a == 12345.f)
		*rounds = 0;
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

static uint64_t
run_ns_of(const struct cpu_snapshot *s, int tid)
{
	for (int i = 0; i < s->n; i++)
		if (s->t[i].tid == tid)
			return s->t[i].run_ns;
	return 0;
}

/* Sum of the pattern words in [off, off + nbytes): word i is i * PATTERN_MUL,
 * so the sum is PATTERN_MUL times a difference of triangular numbers, all
 * modulo 2^64 like the GPU's additions. */
static uint64_t
tri(uint64_t n)
{
	unsigned __int128 p = (unsigned __int128)n * (n - 1);

	return (uint64_t)(p / 2);
}

static uint64_t
pattern_sum(uint64_t off, uint64_t nbytes)
{
	uint64_t a = off / 8, b = (off + nbytes) / 8;

	return (tri(b) - tri(a)) * PATTERN_MUL;
}

static int
make_file(const char *path, uint64_t nbytes)
{
	size_t step = 1u << 20;
	uint64_t *buf;
	int fd;

	fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		perror(path);
		return -1;
	}
	if (posix_memalign((void **)&buf, 4096, step)) {
		close(fd);
		return -1;
	}
	for (uint64_t off = 0; off < nbytes; off += step) {
		size_t n = nbytes - off < step ? (size_t)(nbytes - off) : step;
		size_t done = 0;

		for (size_t i = 0; i < n / 8; i++)
			buf[i] = (off / 8 + i) * PATTERN_MUL;
		while (done < n) {
			ssize_t w = pwrite(fd, (char *)buf + done, n - done,
			                   (off_t)(off + done));

			if (w <= 0) {
				perror("pwrite");
				free(buf);
				close(fd);
				return -1;
			}
			done += (size_t)w;
		}
	}
	free(buf);
	if (fsync(fd) < 0 || close(fd) < 0) {
		perror("fsync");
		return -1;
	}
	printf("wrote %" PRIu64 " MiB of pattern to %s\n", nbytes >> 20, path);
	return 0;
}

static uint64_t
xorshift(uint64_t *s)
{
	uint64_t x = *s;

	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	*s = x;
	return x;
}

static int
coload_setup(struct coload *c)
{
	if (cuStreamCreate(&c->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
		return -1;
	if (cuMemHostAlloc((void **)&c->stop, sizeof(int),
	                   CU_MEMHOSTALLOC_DEVICEMAP) != CUDA_SUCCESS)
		return -1;
	*c->stop = 0;
	if (cuMemHostGetDevicePointer(&c->stop_dptr, (void *)c->stop, 0) !=
	    CUDA_SUCCESS)
		return -1;
	if (cuMemAlloc(&c->rounds, sizeof(c->n)) != CUDA_SUCCESS)
		return -1;
	cuMemsetD8(c->rounds, 0, sizeof(c->n));
	return 0;
}

static void
coload_start(struct coload *c, int sm_count)
{
	c->t0 = now_ms();
	coload_kernel<<<sm_count, THREADS, 0, (cudaStream_t)c->stream>>>(
	        (const volatile int *)(uintptr_t)c->stop_dptr,
	        (unsigned long long *)(uintptr_t)c->rounds);
}

static void
coload_finish(struct coload *c)
{
	if (!c->stream)
		return;
	if (c->stop)
		*c->stop = 1;
	cuStreamSynchronize(c->stream);
	c->t1 = now_ms();
	if (c->rounds)
		cuMemcpyDtoH(&c->n, c->rounds, sizeof(c->n));
}

static double
coload_rate(const struct coload *c)
{
	return c->t1 > c->t0 ? c->n / ((c->t1 - c->t0) / 1e3) : 0.0;
}

static void
coload_free(struct coload *c)
{
	if (c->rounds)
		cuMemFree(c->rounds);
	if (c->stop)
		cuMemFreeHost((void *)c->stop);
	if (c->stream)
		cuStreamDestroy(c->stream);
	memset(c, 0, sizeof(*c));
}

static int
parse_args(int argc, char **argv, struct params *p)
{
	for (int i = 2; i < argc; i++) {
		char *eq = strchr(argv[i], '=');
		const char *v;

		if (!eq) {
			fprintf(stderr, "bad argument %s\n", argv[i]);
			return -1;
		}
		*eq = '\0';
		v = eq + 1;
		if (!strcmp(argv[i], "mode")) {
			if (!strcmp(v, "layer"))
				p->layer_mode = true;
			else if (!strcmp(v, "chunk"))
				p->layer_mode = false;
			else
				return -1;
		} else if (!strcmp(argv[i], "chunk_mib")) {
			p->chunk_bytes = strtoull(v, NULL, 0) << 20;
		} else if (!strcmp(argv[i], "layers")) {
			p->layers = (uint32_t)strtoul(v, NULL, 0);
		} else if (!strcmp(argv[i], "tokens")) {
			p->tokens = (uint32_t)strtoul(v, NULL, 0);
		} else if (!strcmp(argv[i], "chunks")) {
			p->chunks = (uint32_t)strtoul(v, NULL, 0);
		} else if (!strcmp(argv[i], "requests")) {
			p->requests = atoi(v);
		} else if (!strcmp(argv[i], "compute_us")) {
			p->compute_us = atol(v);
		} else if (!strcmp(argv[i], "coload")) {
			p->coload = atoi(v) != 0;
		} else if (!strcmp(argv[i], "coload_ms")) {
			p->coload_ms = atol(v);
		} else if (!strcmp(argv[i], "tag")) {
			p->tag = v;
		} else if (!strcmp(argv[i], "make_file")) {
			p->make_file = strtoull(v, NULL, 0);
		} else {
			fprintf(stderr, "unknown key %s\n", argv[i]);
			return -1;
		}
	}
	return 0;
}

int
main(int argc, char **argv)
{
	struct params p = {
	        .layer_mode = false,
	        .chunk_bytes = 32u << 20,
	        .layers = 32,
	        .tokens = 256,
	        .chunks = 8,
	        .requests = 1,
	        .compute_us = 0,
	        .coload = false,
	        .coload_ms = 3000,
	        .tag = "none",
	        .make_file = 0,
	};
	const char *gpu_env = getenv("OPENDS_AISIO_GPU_INITIATED");
	bool gpu_engine = gpu_env && gpu_env[0] && gpu_env[0] != '0';
	const char *path;
	struct request req[MAX_REQUESTS];
	struct coload cl;
	CUdevice cudev;
	CUcontext cuctx;
	CUresult cres;
	int sm_count = 1;
	opends_error_t err;
	opends_handle_t fh = NULL;
	struct stat st;
	size_t slice_bytes, stage_bytes, tok_bytes, cache_bytes, total_bytes;
	uint32_t steps, n_file_chunks, blocks_per_req;
	size_t *sz = NULL;
	off_t *foff = NULL, *boff = NULL;
	ssize_t *bytes = NULL;
	uint64_t *ts = NULL;
	unsigned long long *sums = NULL;
	struct cpu_snapshot s0, s1, s2;
	double t0, t1, t2;
	int fd = -1, rc = 1, created = 0, bad = 0;

	if (argc < 2) {
		fprintf(stderr, "usage: %s <file-on-mount> [key=value ...]\n",
		        argv[0]);
		return 2;
	}
	path = argv[1];
	if (parse_args(argc, argv, &p) < 0)
		return 2;
	if (p.make_file)
		return make_file(path, p.make_file) < 0 ? 1 : 0;

	if (p.layers < 1 || p.tokens < BLOCK_TOKENS ||
	    p.tokens % BLOCK_TOKENS || p.chunks < 1 || p.requests < 0 ||
	    p.requests > MAX_REQUESTS || p.chunk_bytes % 4096 ||
	    p.chunk_bytes % ((size_t)p.layers * p.tokens * 16)) {
		fprintf(stderr, "bad shape\n");
		return 2;
	}
	if (p.requests == 0 && !p.coload) {
		fprintf(stderr, "requests=0 needs coload=1\n");
		return 2;
	}
	slice_bytes = p.chunk_bytes / p.layers;
	if (p.layer_mode && slice_bytes % 4096) {
		fprintf(stderr, "layer slice is not 4 KiB aligned\n");
		return 2;
	}
	tok_bytes = p.chunk_bytes / ((size_t)p.layers * p.tokens);
	stage_bytes = p.layer_mode ? slice_bytes : p.chunk_bytes;
	steps = p.layer_mode ? p.chunks * p.layers : p.chunks;
	cache_bytes = (size_t)p.chunks * p.chunk_bytes;
	blocks_per_req = p.chunks * p.tokens / BLOCK_TOKENS;
	total_bytes = (size_t)p.requests * cache_bytes;

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
	cuDeviceGetAttribute(&sm_count,
	                     CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudev);

	memset(req, 0, sizeof(req));
	memset(&cl, 0, sizeof(cl));

	if (p.requests == 0) {
		if (coload_setup(&cl) < 0) {
			fprintf(stderr, "coload setup failed\n");
			goto out_ctx;
		}
		coload_start(&cl, sm_count);
		sleep_ms(p.coload_ms);
		coload_finish(&cl);
		printf("SUMMARY tag=%s engine=none mode=none requests=0 "
		       "coload=1 coload_rounds_s=%.0f data=ok\n",
		       p.tag, coload_rate(&cl));
		rc = 0;
		goto out_ctx;
	}

	err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr,
		        "driver_open: %s (is the homi stack running?)\n",
		        opends_op_status_error(err.err));
		goto out_ctx;
	}

	fd = open(path, O_RDONLY | O_DIRECT);
	if (fd < 0 || fstat(fd, &st) < 0) {
		perror(path);
		goto out_driver;
	}
	n_file_chunks = (uint32_t)((uint64_t)st.st_size / p.chunk_bytes);
	if (n_file_chunks == 0) {
		fprintf(stderr, "%s: smaller than one chunk\n", path);
		goto out_fd;
	}
	err = opends_handle_register(&fh, fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "handle_register: %s\n",
		        opends_op_status_error(err.err));
		goto out_fd;
	}

	for (int r = 0; r < p.requests; r++) {
		struct request *q = &req[r];
		uint32_t *table;
		uint64_t seed = 0x1234567ull + (uint64_t)r * 0x9E37ull;

		if (cuStreamCreate(&q->stream, CU_STREAM_NON_BLOCKING) !=
		    CUDA_SUCCESS) {
			fprintf(stderr, "cuStreamCreate failed\n");
			goto out_req;
		}
		created = r + 1;
		err = opends_stream_register(q->stream,
		                             OPENDS_STREAM_FIXED_SHAPE);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "stream_register: %s (dev_err=%d)\n",
			        opends_op_status_error(err.err),
			        (int)err.dev_err);
			goto out_req;
		}
		for (int b = 0; b < 2; b++) {
			q->staging[b] = opends_alloc(stage_bytes);
			if (!q->staging[b]) {
				fprintf(stderr, "opends_alloc(%zu) failed\n",
				        stage_bytes);
				goto out_req;
			}
		}
		if (cudaMalloc((void **)&q->cache, cache_bytes) ||
		    cudaMemset(q->cache, 0, cache_bytes) ||
		    cudaMalloc((void **)&q->block_table,
		               blocks_per_req * sizeof(uint32_t)) ||
		    cudaMalloc((void **)&q->ts,
		               (2 * (size_t)steps + 1) * sizeof(uint64_t)) ||
		    cudaMalloc((void **)&q->sum, sizeof(*q->sum)) ||
		    cudaMemset(q->sum, 0, sizeof(*q->sum))) {
			fprintf(stderr, "cudaMalloc failed for request %d\n",
			        r);
			goto out_req;
		}
		/* A random permutation of the cache's blocks, as a block table
		 * built over a busy cache would be. */
		table = (uint32_t *)malloc(blocks_per_req * sizeof(*table));
		if (!table)
			goto out_req;
		for (uint32_t i = 0; i < blocks_per_req; i++)
			table[i] = i;
		for (uint32_t i = blocks_per_req - 1; i > 0; i--) {
			uint32_t j = (uint32_t)(xorshift(&seed) % (i + 1));
			uint32_t tmp = table[i];

			table[i] = table[j];
			table[j] = tmp;
		}
		cudaMemcpy(q->block_table, table,
		           blocks_per_req * sizeof(*table),
		           cudaMemcpyHostToDevice);
		free(table);
	}
	cudaDeviceSynchronize();

	if (p.coload && coload_setup(&cl) < 0) {
		fprintf(stderr, "coload setup failed\n");
		goto out_req;
	}

	sz = (size_t *)calloc((size_t)p.requests * steps, sizeof(*sz));
	foff = (off_t *)calloc((size_t)p.requests * steps, sizeof(*foff));
	boff = (off_t *)calloc((size_t)p.requests * steps, sizeof(*boff));
	ts = (uint64_t *)calloc(2 * (size_t)steps + 1, sizeof(*ts));
	sums = (unsigned long long *)calloc((size_t)p.requests, sizeof(*sums));
	bytes = (ssize_t *)opends_result_alloc((size_t)p.requests * steps *
	                                       sizeof(*bytes));
	if (!sz || !foff || !boff || !ts || !sums || !bytes) {
		fprintf(stderr, "allocation failed\n");
		goto out_req;
	}
	memset(bytes, 0, (size_t)p.requests * steps * sizeof(*bytes));

	printf("aisio_kv_retrieve: %d request%s x %u chunks of %zu MiB, %s "
	       "mode "
	       "(%u steps of %zu KiB per request), %u layers x %u tokens x %zu "
	       "B, "
	       "%ld us compute per step, co-load %s, GPU engine %s\n",
	       p.requests, p.requests > 1 ? "s" : "", p.chunks,
	       p.chunk_bytes >> 20, p.layer_mode ? "layer" : "chunk", steps,
	       stage_bytes >> 10, p.layers, p.tokens, tok_bytes, p.compute_us,
	       p.coload ? "on" : "off", gpu_engine ? "on" : "off");

	if (p.coload)
		coload_start(&cl, sm_count);

	snapshot_cpu(&s0);
	t0 = now_ms();
	for (uint32_t s = 0; s < steps; s++) {
		for (int r = 0; r < p.requests; r++) {
			struct request *q = &req[r];
			cudaStream_t cs = (cudaStream_t)q->stream;
			size_t idx = (size_t)r * steps + s;
			uint32_t k = p.layer_mode ? s / p.layers : s;
			uint32_t l = p.layer_mode ? s % p.layers : 0;
			uint32_t nl = p.layer_mode ? 1 : p.layers;
			uint64_t file_chunk =
			        ((uint64_t)r * p.chunks + k) % n_file_chunks;
			void *buf = q->staging[s % 2];
			size_t nvec = (size_t)nl * p.tokens * (tok_bytes / 16);
			unsigned grid =
			        (unsigned)((nvec + THREADS - 1) / THREADS);

			if (grid > 1024)
				grid = 1024;
			sz[idx] = stage_bytes;
			foff[idx] = (off_t)(file_chunk * p.chunk_bytes +
			                    (uint64_t)l * slice_bytes);
			stamp_kernel<<<1, 1, 0, cs>>>(&q->ts[2 * s]);
			err = opends_stream_read(fh, buf, &sz[idx], &foff[idx],
			                         &boff[idx], &bytes[idx],
			                         (opends_stream_t)q->stream);
			if (err.err != OPENDS_SUCCESS) {
				const char *cs_str = NULL;

				cuGetErrorString((CUresult)err.dev_err,
				                 &cs_str);
				fprintf(stderr,
				        "stream_read r%d s%u: %s (dev_err=%d "
				        "%s)\n",
				        r, s, opends_op_status_error(err.err),
				        (int)err.dev_err,
				        cs_str ? cs_str : "?");
				goto out_stop;
			}
			stamp_kernel<<<1, 1, 0, cs>>>(&q->ts[2 * s + 1]);
			scatter_kernel<<<grid, THREADS, 0, cs>>>(
			        (const uint4 *)buf, (uint4 *)q->cache,
			        q->block_table +
			                (size_t)k * p.tokens / BLOCK_TOKENS,
			        p.tokens, p.layers, l, nl,
			        (uint32_t)(tok_bytes / 16));
			if (p.compute_us > 0)
				spin_kernel<<<1, 1, 0, cs>>>(
				        (uint64_t)p.compute_us * 1000);
			if (s == steps - 1) {
				size_t nwords = cache_bytes / sizeof(uint64_t);
				unsigned g = (unsigned)((nwords + THREADS - 1) /
				                        THREADS);

				if (g > 1024)
					g = 1024;
				stamp_kernel<<<1, 1, 0, cs>>>(
				        &q->ts[2 * steps]);
				sum_kernel<<<g, THREADS, 0, cs>>>(
				        (const uint64_t *)q->cache, nwords,
				        q->sum);
			}
		}
	}
	t1 = now_ms();
	snapshot_cpu(&s1);

	/* Nothing on this thread drives the chains from here on. */
	for (;;) {
		cres = CUDA_SUCCESS;
		for (int r = 0; r < p.requests && cres == CUDA_SUCCESS; r++)
			cres = cuStreamQuery(req[r].stream);
		if (cres != CUDA_ERROR_NOT_READY)
			break;
		sleep_ms(1);
	}
	t2 = now_ms();
	snapshot_cpu(&s2);
	coload_finish(&cl);
	if (cres != CUDA_SUCCESS) {
		fprintf(stderr, "stream failed: %d\n", (int)cres);
		goto out_req;
	}

	{
		double ready_acc = 0, ready_max = 0, step_acc = 0, step_max = 0;
		double enq_ms = t1 - t0, wall_ms = t2 - t0, span_ms;
		double cpu_all = 0, caller_ms = 0, io_ms = 0;
		uint64_t first = UINT64_MAX, last = 0;

		for (int r = 0; r < p.requests; r++) {
			struct request *q = &req[r];
			uint64_t expect = 0;
			double ready;

			cudaMemcpy(ts, q->ts,
			           (2 * (size_t)steps + 1) * sizeof(*ts),
			           cudaMemcpyDeviceToHost);
			cudaMemcpy(&sums[r], q->sum, sizeof(sums[r]),
			           cudaMemcpyDeviceToHost);
			for (uint32_t k = 0; k < p.chunks; k++) {
				uint64_t file_chunk =
				        ((uint64_t)r * p.chunks + k) %
				        n_file_chunks;

				expect +=
				        pattern_sum(file_chunk * p.chunk_bytes,
				                    p.chunk_bytes);
			}
			if (sums[r] != expect) {
				printf("  request %d: cache sum %#llx, "
				       "expected "
				       "%#" PRIx64 "\n",
				       r, sums[r], expect);
				bad++;
			}
			for (uint32_t s = 0; s < steps; s++) {
				double w = (ts[2 * s + 1] - ts[2 * s]) / 1e6;

				if (bytes[(size_t)r * steps + s] !=
				    (ssize_t)stage_bytes) {
					printf("  request %d step %u: "
					       "bytes_read "
					       "%zd\n",
					       r, s,
					       bytes[(size_t)r * steps + s]);
					bad++;
				}
				step_acc += w;
				if (w > step_max)
					step_max = w;
			}
			ready = (ts[2 * steps] - ts[0]) / 1e6;
			ready_acc += ready;
			if (ready > ready_max)
				ready_max = ready;
			if (ts[0] < first)
				first = ts[0];
			if (ts[2 * steps] > last)
				last = ts[2 * steps];
		}
		span_ms = (last - first) / 1e6;

		printf("  data: %s\n",
		       bad ? "MISMATCH" : "every cache matches the pattern");
		printf("  enqueue: %.1f ms; chains done %.1f ms after the last "
		       "enqueue; %.1f ms wall\n",
		       enq_ms, t2 - t1, wall_ms);
		for (int i = 0; i < s2.n; i++) {
			double ms =
			        (s2.t[i].run_ns - run_ns_of(&s0, s2.t[i].tid)) /
			        1e6;

			cpu_all += ms;
			if (s2.t[i].tid == getpid())
				caller_ms = ms;
			else if (!strcmp(s2.t[i].comm, "aisio-io"))
				io_ms += ms;
			printf("    tid %-7d %-16s %9.3f ms\n", s2.t[i].tid,
			       s2.t[i].comm, ms);
		}
		printf("  time to KV ready per request avg/max %.1f/%.1f ms; "
		       "read window per step avg/max %.3f/%.3f ms; %.2f GB/s "
		       "over all requests\n",
		       ready_acc / p.requests, ready_max,
		       step_acc / ((double)steps * p.requests), step_max,
		       total_bytes / span_ms / 1e6);
		printf("SUMMARY tag=%s engine=%s mode=%s chunk_mib=%zu "
		       "slice_kib=%zu layers=%u tokens=%u chunks=%u "
		       "requests=%d "
		       "compute_us=%ld coload=%d steps=%u ready_ms_avg=%.1f "
		       "ready_ms_max=%.1f step_ms_avg=%.3f step_ms_max=%.3f "
		       "enqueue_ms=%.1f wall_ms=%.1f gbps=%.2f cpu_ms=%.1f "
		       "cpu_caller_ms=%.1f cpu_io_ms=%.1f cpu_pct=%.1f "
		       "coload_rounds_s=%.0f data=%s\n",
		       p.tag, gpu_engine ? "gpu" : "host",
		       p.layer_mode ? "layer" : "chunk", p.chunk_bytes >> 20,
		       stage_bytes >> 10, p.layers, p.tokens, p.chunks,
		       p.requests, p.compute_us, p.coload ? 1 : 0, steps,
		       ready_acc / p.requests, ready_max,
		       step_acc / ((double)steps * p.requests), step_max,
		       enq_ms, wall_ms, total_bytes / span_ms / 1e6, cpu_all,
		       caller_ms, io_ms, 100.0 * cpu_all / wall_ms,
		       coload_rate(&cl), bad ? "MISMATCH" : "ok");
	}
	rc = bad ? 1 : 0;
	goto out_req;

out_stop:
	coload_finish(&cl);
	for (int r = 0; r < created; r++)
		cuStreamSynchronize(req[r].stream);
out_req:
	for (int r = 0; r < created; r++)
		cuStreamDestroy(req[r].stream);
	for (int r = 0; r < p.requests; r++) {
		for (int b = 0; b < 2; b++)
			if (req[r].staging[b])
				opends_free(req[r].staging[b]);
		cudaFree(req[r].cache);
		cudaFree(req[r].block_table);
		cudaFree(req[r].ts);
		cudaFree(req[r].sum);
	}
	if (bytes)
		opends_result_free(bytes);
	free(sz);
	free(foff);
	free(boff);
	free(ts);
	free(sums);
	opends_handle_deregister(fh);
out_fd:
	if (fd >= 0)
		close(fd);
out_driver:
	opends_driver_close();
out_ctx:
	coload_finish(&cl);
	coload_free(&cl);
	cuCtxDestroy(cuctx);
	return rc;
}
