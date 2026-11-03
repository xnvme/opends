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

#define AISIO_CONFIG_XAL_SHM_MAXLEN 128

struct aisio_config {
	uint32_t homi_id; ///< Multi-process group the HOMI server is primary of
	char xal_shm[AISIO_CONFIG_XAL_SHM_MAXLEN]; ///< Index xal-server
	                                           ///< publishes
	int n_io_threads;
	uint32_t queue_depth;
	uint32_t idle_spin_us;
	bool busy_spin;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	size_t host_heap_nbytes;
	size_t device_heap_nbytes;
};

/* Returns 0, or -EINVAL after a message on stderr. */
int aisio_config_read(struct aisio_config *cfg);

#endif /* DS_AISIO_CONFIG_H_ */
