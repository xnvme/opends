/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * The aisio backend's configuration, read from the environment at driver
 * open. The driver consumes the values and never reads the environment
 * itself.
 */
#ifndef DS_AISIO_CONFIG_H_
#define DS_AISIO_CONFIG_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AISIO_CONFIG_MOUNT_MAXLEN 256

struct aisio_config {
	uint32_t homi_id; ///< Multi-process group the HOMI server is primary of
	char mount[AISIO_CONFIG_MOUNT_MAXLEN]; ///< Mountpoint of the filesystem
	                                       ///< on the group's device
	int n_io_threads;
	uint32_t queue_depth;
	uint32_t idle_spin_us;
	bool busy_spin;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	size_t host_heap_nbytes;
	size_t device_heap_nbytes;
	size_t write_buf_nbytes; ///< Pinned host buf a write is staged
	                         ///< through
	int write_bufs;          ///< Most buffers the driver pins
};

/* Returns 0, or -EINVAL after a message on stderr. */
int aisio_config_read(struct aisio_config *cfg);

#endif /* DS_AISIO_CONFIG_H_ */
