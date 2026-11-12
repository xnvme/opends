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

#define ENV_MOUNT "OPENDS_HOMI_MNT"
#define ENV_IO_THREADS "OPENDS_AISIO_IO_THREADS"
#define ENV_QUEUE_DEPTH "OPENDS_AISIO_QUEUE_DEPTH"
#define ENV_CPU_MASK "OPENDS_AISIO_CPU_MASK"
#define ENV_ASSUME_ALIGNED_ONLY "OPENDS_AISIO_ASSUME_ALIGNED_ONLY"
#define ENV_IDLE_SPIN "OPENDS_AISIO_IDLE_SPIN"
#define ENV_HOMI_ID "OPENDS_AISIO_HOMI_ID"
#define ENV_HOST_HEAP_MB "OPENDS_AISIO_HOST_HEAP_MB"
#define ENV_DEVICE_HEAP_MB "OPENDS_AISIO_DEVICE_HEAP_MB"
#define ENV_WRITE_BUF_MB "OPENDS_AISIO_WRITE_BUF_MB"
#define ENV_WRITE_BUFS "OPENDS_AISIO_WRITE_BUFS"
#define DEFAULT_HOMI_ID 1
#define DEFAULT_IO_THREADS 2
#define MAX_IO_THREADS 15
#define DEFAULT_QUEUE_DEPTH 8
#define MAX_QUEUE_DEPTH 1024

/* The host DMA heap holds this process's own SQ/CQ rings and PRP lists, one set
 * per I/O thread. 256 MiB covers the largest configuration the knobs allow
 * (MAX_IO_THREADS queues at MAX_QUEUE_DEPTH). xNVMe would otherwise default to
 * 1 GiB, which does not multiply across the processes sharing the hugepages. */
#define DEFAULT_HOST_HEAP_MB 256
#define MAX_HEAP_MB (64 * 1024)
#define DEFAULT_IDLE_SPIN_US 200
#define MAX_IDLE_SPIN_US 1000000
#define DEFAULT_WRITE_BUF_MB 16
#define MAX_WRITE_BUF_MB 1024
#define DEFAULT_WRITE_BUFS_PER_THREAD 4
#define MAX_WRITE_BUFS 1024

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
aisio_config_read(struct aisio_config *cfg)
{
	int n;

	if (env_int(ENV_HOMI_ID, DEFAULT_HOMI_ID, 0, INT_MAX, &n) < 0) {
		return -EINVAL;
	}
	cfg->homi_id = (uint32_t)n;

	const char *mount = getenv(ENV_MOUNT);
	if (!mount || !mount[0]) {
		fprintf(stderr, "aisio: %s is not set\n", ENV_MOUNT);
		return -EINVAL;
	}
	if (strlen(mount) >= sizeof(cfg->mount)) {
		fprintf(stderr, "aisio: %s is longer than %zu\n", ENV_MOUNT,
		        sizeof(cfg->mount) - 1);
		return -EINVAL;
	}
	strcpy(cfg->mount, mount);

	if (env_int(ENV_IO_THREADS, DEFAULT_IO_THREADS, 1, MAX_IO_THREADS, &n) <
	    0) {
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

	if (env_int(ENV_WRITE_BUF_MB, DEFAULT_WRITE_BUF_MB, 1, MAX_WRITE_BUF_MB,
	            &n) < 0) {
		return -EINVAL;
	}
	cfg->write_buf_nbytes = (size_t)n << 20;

	if (env_int(ENV_WRITE_BUFS,
	            DEFAULT_WRITE_BUFS_PER_THREAD * cfg->n_io_threads, 1,
	            MAX_WRITE_BUFS, &n) < 0) {
		return -EINVAL;
	}
	cfg->write_bufs = n;

	/* Every write stages through pinned buffers with async copies, and a
	 * stream write with late parameters through the resident kernel. */
	if (!ds_accel->load || !ds_accel->host_pin || !ds_accel->host_unpin ||
	    !ds_accel->copy_from_dev_async || !ds_accel->stage_stream) {
		fprintf(stderr, "aisio: the vendor ops table does not provide "
		                "the copy ops writes stage through\n");
		return -EINVAL;
	}

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
	                                  !ds_accel->stream_wait_value32_geq ||
	                                  !ds_accel->copy_to_dev_async)) {
		fprintf(stderr,
		        "aisio: the vendor ops table does not provide the "
		        "stream gate and pinned copy ops; set %s=1 to gate "
		        "via launch_host_func\n",
		        ENV_ASSUME_ALIGNED_ONLY);
		return -EINVAL;
	}

	return 0;
}
