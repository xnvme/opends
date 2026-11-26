/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * The aisio backend's configuration, read from the environment at driver
 * open. The driver consumes the values and never reads the environment
 * itself.
 */
#ifndef OPENDS_AISIO_CONFIG_H_
#define OPENDS_AISIO_CONFIG_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AISIO_CONFIG_MOUNT_MAXLEN 256

struct aisio_config {
	/* multi-process group the HOMI server is primary of */
	uint32_t homi_id;
	/* mountpoint of the filesystem on the group's device, or empty */
	char mount[AISIO_CONFIG_MOUNT_MAXLEN];
	int n_io_threads;
	uint32_t queue_depth;
	uint32_t idle_spin_us;
	bool busy_spin;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	bool raw_block_enabled;
	size_t host_heap_nbytes;
	size_t device_heap_nbytes;
	size_t write_buf_nbytes; /* pinned host buf a write is staged through */
	int write_bufs;          /* most buffers the driver pins */
};

/* Returns 0, or -EINVAL after a message on stderr. */
int opends_aisio_config_read(struct aisio_config *cfg);

#endif /* OPENDS_AISIO_CONFIG_H_ */
