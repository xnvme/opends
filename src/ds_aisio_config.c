/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Environment parsing for the aisio backend. Every variable is read here,
 * checked against its bounds, and handed over as a struct aisio_config.
 */
#include "ds_aisio_config.h"
#include "ds_accel.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENV_XAL_SHM "OPENDS_XAL_SHM"
#define ENV_WORKERS_PER_DRIVE "OPENDS_AISIO_WORKERS_PER_DRIVE"
#define ENV_QUEUE_DEPTH "OPENDS_AISIO_QUEUE_DEPTH"
#define ENV_CPU_MASK "OPENDS_AISIO_CPU_MASK"
#define ENV_ASSUME_ALIGNED_ONLY "OPENDS_AISIO_ASSUME_ALIGNED_ONLY"
#define ENV_IDLE_SPIN "OPENDS_AISIO_IDLE_SPIN"
#define ENV_HOMI_ID "OPENDS_AISIO_HOMI_ID"
#define ENV_HOST_HEAP_MB "OPENDS_AISIO_HOST_HEAP_MB"
#define ENV_DEVICE_HEAP_MB "OPENDS_AISIO_DEVICE_HEAP_MB"
#define ENV_CQ_MIRROR "OPENDS_AISIO_CQ_MIRROR"
#define DEFAULT_XAL_SHM_FMT "/xal_dev%d"
#define DEFAULT_HOMI_ID 1
#define DEFAULT_GPU_WORKERS 2
#define MAX_WORKERS_PER_MEM 16
#define DEFAULT_QUEUE_DEPTH 8
#define MAX_QUEUE_DEPTH 1024

/* The host DMA heap holds this process's own SQ/CQ rings and PRP lists, about
 * 4 MiB per queue, so 256 MiB covers some sixty queues. xNVMe would otherwise
 * default to 1 GiB, which does not multiply across the processes sharing the
 * hugepages. */
#define DEFAULT_HOST_HEAP_MB 256
#define MAX_HEAP_MB (64 * 1024)
#define DEFAULT_IDLE_SPIN_US 200
#define MAX_IDLE_SPIN_US 1000000

static int
env_int(const char *name, int def, int lo, int hi, int *out)
{
	const char *v = getenv(name);
	if (!v || !v[0]) {
		*out = def;
		return 0;
	}
	char *end;
	long n = strtol(v, &end, 10);
	if (end == v || *end) {
		fprintf(stderr, "aisio: %s=%s is not a number\n", name, v);
		return -EINVAL;
	}
	if (n < lo || n > hi) {
		fprintf(stderr, "aisio: %s=%s out of range [%d, %d]\n", name, v,
		        lo, hi);
		return -EINVAL;
	}
	*out = (int)n;
	return 0;
}

/* The memory a key of OPENDS_AISIO_WORKERS_PER_DRIVE names. */
static int
mem_from_key(int gpu_ordinal, const char *key)
{
	if (strcmp(key, "host") == 0) {
		return MEM_HOST;
	}
	if (strncmp(key, "gpu", 3) == 0) {
		int ord = gpu_ordinal;

		if (key[3]) {
			char *tail;

			ord = (int)strtol(key + 3, &tail, 10);
			if (*tail || tail == key + 3) {
				fprintf(stderr,
				        "aisio: %s: unknown key \"%s\"\n",
				        ENV_WORKERS_PER_DRIVE, key);
				return -EINVAL;
			}
		}
		if (ord != gpu_ordinal) {
			fprintf(stderr,
			        "aisio: %s: gpu%d is not the accelerator of "
			        "the current context (gpu%d)\n",
			        ENV_WORKERS_PER_DRIVE, ord, gpu_ordinal);
			return -EINVAL;
		}
		return MEM_GPU;
	}
	fprintf(stderr, "aisio: %s: unknown key \"%s\"\n",
	        ENV_WORKERS_PER_DRIVE, key);
	return -EINVAL;
}

/* OPENDS_AISIO_WORKERS_PER_DRIVE, "key=value,...": the workers every drive runs
 * for each memory. Unset, the current GPU gets DEFAULT_GPU_WORKERS. */
static int
read_workers_per_drive(struct aisio_config *cfg, int gpu_ordinal)
{
	const char *spec = getenv(ENV_WORKERS_PER_DRIVE);
	const char *p = spec;

	memset(cfg->n_workers, 0, sizeof(cfg->n_workers));
	if (!spec || !spec[0]) {
		cfg->n_workers[MEM_GPU] = DEFAULT_GPU_WORKERS;
		return 0;
	}
	while (*p) {
		const char *end = strchr(p, ',');
		const char *eq;
		char key[16];
		char *tail;
		size_t klen;
		long val;
		int mem;

		if (!end) {
			end = p + strlen(p);
		}
		eq = memchr(p, '=', (size_t)(end - p));
		klen = eq ? (size_t)(eq - p) : 0;
		if (!klen || klen >= sizeof(key)) {
			fprintf(stderr,
			        "aisio: %s: expected key=value at \"%.*s\"\n",
			        ENV_WORKERS_PER_DRIVE, (int)(end - p), p);
			return -EINVAL;
		}
		memcpy(key, p, klen);
		key[klen] = '\0';
		val = strtol(eq + 1, &tail, 10);
		if (tail == eq + 1 || tail != end || val < 1) {
			fprintf(stderr,
			        "aisio: %s: %s needs a positive count\n",
			        ENV_WORKERS_PER_DRIVE, key);
			return -EINVAL;
		}
		mem = mem_from_key(gpu_ordinal, key);
		if (mem < 0) {
			return mem;
		}
		if (val > MAX_WORKERS_PER_MEM) {
			fprintf(stderr, "aisio: %s: %s is at most %d workers\n",
			        ENV_WORKERS_PER_DRIVE, key,
			        MAX_WORKERS_PER_MEM);
			return -EINVAL;
		}
		cfg->n_workers[mem] = (int)val;
		p = *end ? end + 1 : end;
	}
	if (!cfg->n_workers[MEM_GPU]) {
		fprintf(stderr, "aisio: %s must give gpu%d a worker count\n",
		        ENV_WORKERS_PER_DRIVE, gpu_ordinal);
		return -EINVAL;
	}
	return 0;
}

int
aisio_config_homi_id(uint32_t *out)
{
	int n;

	if (env_int(ENV_HOMI_ID, DEFAULT_HOMI_ID, 0, INT_MAX, &n) < 0) {
		return -EINVAL;
	}
	*out = (uint32_t)n;
	return 0;
}

int
aisio_config_read(struct aisio_config *cfg, int gpu_ordinal)
{
	int n;

	if (read_workers_per_drive(cfg, gpu_ordinal) < 0) {
		return -EINVAL;
	}

	if (env_int(ENV_QUEUE_DEPTH, DEFAULT_QUEUE_DEPTH, 1, MAX_QUEUE_DEPTH,
	            &n) < 0) {
		return -EINVAL;
	}
	cfg->queue_depth = (uint32_t)n;

	const char *spin = getenv(ENV_IDLE_SPIN);
	cfg->busy_spin = spin && !strcmp(spin, "busy");
	if (cfg->busy_spin) {
		cfg->idle_spin_us = DEFAULT_IDLE_SPIN_US;
	} else {
		if (env_int(ENV_IDLE_SPIN, DEFAULT_IDLE_SPIN_US, 0,
		            MAX_IDLE_SPIN_US, &n) < 0) {
			fprintf(stderr,
			        "aisio: %s takes microseconds, or \"busy\"\n",
			        ENV_IDLE_SPIN);
			return -EINVAL;
		}
		cfg->idle_spin_us = (uint32_t)n;
	}

	const char *mask = getenv(ENV_CPU_MASK);
	cfg->cpu_mask = mask && mask[0] ? strtoull(mask, NULL, 0) : 0;

	const char *aligned = getenv(ENV_ASSUME_ALIGNED_ONLY);
	cfg->assume_aligned_only = aligned && aligned[0] && aligned[0] != '0';

	const char *cqm = getenv(ENV_CQ_MIRROR);
	cfg->cq_mirror = cqm && cqm[0] && cqm[0] != '0';

	if (env_int(ENV_HOST_HEAP_MB, DEFAULT_HOST_HEAP_MB, 1, MAX_HEAP_MB,
	            &n) < 0) {
		return -EINVAL;
	}
	cfg->host_heap_nbytes = (size_t)n << 20;

	/* 0 leaves the device heap at the xNVMe default; GPU memory is not the
	 * scarce resource the host hugepages are. */
	if (env_int(ENV_DEVICE_HEAP_MB, 0, 0, MAX_HEAP_MB, &n) < 0) {
		return -EINVAL;
	}
	cfg->device_heap_nbytes = (size_t)n << 20;

	/* The tail mode picks the async gate mechanism, and the vendor ops it
	 * drives are required only for that mode (see ds_accel.h). A partial
	 * port may leave the other mode's ops NULL; fail open instead of
	 * crashing on the first submission. */
	if (cfg->assume_aligned_only && !ds_accel->launch_host_func) {
		fprintf(stderr,
		        "aisio: %s=1 needs launch_host_func, which the vendor "
		        "ops table does not provide\n",
		        ENV_ASSUME_ALIGNED_ONLY);
		return -EINVAL;
	}
	if (!cfg->assume_aligned_only && (!ds_accel->stream_write_value32 ||
	                                  !ds_accel->stream_wait_value32_geq)) {
		fprintf(stderr,
		        "aisio: the vendor ops table does not provide the "
		        "stream gate ops; set %s=1 to gate via "
		        "launch_host_func\n",
		        ENV_ASSUME_ALIGNED_ONLY);
		return -EINVAL;
	}

	return 0;
}

/* OPENDS_XAL_SHM optionally overrides the index names as a comma-separated
 * list, paired with the runtime's device order. Default: /xal_dev<i>. */
int
aisio_config_xal_shm(int di, int n_devices, char *out, size_t out_len)
{
	const char *spec = getenv(ENV_XAL_SHM);
	const char *p;
	const char *end;
	size_t len;
	int n;

	if (!spec || !spec[0]) {
		snprintf(out, out_len, DEFAULT_XAL_SHM_FMT, di);
		return 0;
	}

	n = 1;
	for (p = spec; *p; p++) {
		if (*p == ',') {
			n++;
		}
	}
	if (n != n_devices) {
		fprintf(stderr, "aisio: %s names %d indexes for %d devices\n",
		        ENV_XAL_SHM, n, n_devices);
		return -EINVAL;
	}

	p = spec;
	for (int i = 0; i < di; i++) {
		p = strchr(p, ',') + 1;
	}
	end = strchr(p, ',');
	len = end ? (size_t)(end - p) : strlen(p);
	if (len == 0 || len >= out_len) {
		fprintf(stderr, "aisio: %s entry %d is empty or too long\n",
		        ENV_XAL_SHM, di);
		return -EINVAL;
	}
	memcpy(out, p, len);
	out[len] = '\0';

	return 0;
}
