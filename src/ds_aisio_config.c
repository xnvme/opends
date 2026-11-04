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
#define ENV_IO_THREADS "OPENDS_AISIO_IO_THREADS"
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
#define DEFAULT_IO_THREADS 2
#define MAX_IO_THREADS 15
#define DEFAULT_QUEUE_DEPTH 8
#define MAX_QUEUE_DEPTH 1024

/* The host DMA heap holds this process's own SQ/CQ rings and PRP lists, one set
 * per I/O thread. The heap is process-wide and the thread count does not grow
 * with the device count, so 256 MiB covers the largest configuration the knobs
 * allow (MAX_IO_THREADS queues at MAX_QUEUE_DEPTH). xNVMe would otherwise
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
aisio_config_read(struct aisio_config *cfg, int n_devices)
{
	int n;
	int thr_def;

	/* One worker per device at least, since a device with no worker cannot
	 * be read. The threads are the total, not a per-device count. */
	thr_def =
	        DEFAULT_IO_THREADS < n_devices ? n_devices : DEFAULT_IO_THREADS;
	if (env_int(ENV_IO_THREADS, thr_def, 1, MAX_IO_THREADS, &n) < 0) {
		return -EINVAL;
	}
	if (n < n_devices) {
		fprintf(stderr,
		        "aisio: %s=%d leaves some of the %d devices without a "
		        "worker\n",
		        ENV_IO_THREADS, n, n_devices);
		return -EINVAL;
	}
	cfg->n_io_threads = n;

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
