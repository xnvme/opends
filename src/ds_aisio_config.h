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

#include "opends.h"

/* The per-memory tables are indexed by enum opends_mem_kind. */
#define MAX_MEMS 2

/* One NVMe-oF target of OPENDS_AISIO_FABRICS: the addr:port the SPDK backend
 * takes as the device URI, and the subsystem NQN to connect. */
#define AISIO_TRADDR_LEN 64
#define AISIO_NQN_LEN 256
struct aisio_fabrics_target {
	char traddr[AISIO_TRADDR_LEN];
	char subnqn[AISIO_NQN_LEN];
};

struct aisio_config {
	uint32_t homi_id; ///< Multi-process group the HOMI server is primary of
	int n_workers[MAX_MEMS]; ///< Workers every drive runs, per memory
	uint32_t queue_depth;
	uint32_t idle_spin_us;
	bool busy_spin;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	bool cq_mirror; ///< CQ in GPU memory, warp-mirrored to host (upcie-cuda
	                ///< only)
	size_t host_heap_nbytes;
	size_t device_heap_nbytes;
};

/* The homi group to join, on its own because device discovery needs it
 * before the rest is read. */
int aisio_config_homi_id(uint32_t *out);

/* Read the NVMe-oF targets from OPENDS_AISIO_FABRICS into out, up to max
 * entries. Unset leaves *n_out at 0, and the drives then come from the homi
 * group. Returns 0, or -EINVAL after a message on stderr. */
int aisio_config_fabrics(struct aisio_fabrics_target *out, int max, int *n_out);

/* Everything else. gpu_ordinal is the accelerator of the current context,
 * the only one a gpu<k> key may name. fabrics drives serve host memory
 * only, so the workers must be given, with a count for cpu0 and none for
 * the GPU. Returns 0, or -EINVAL after a message on stderr. */
int aisio_config_read(struct aisio_config *cfg, int gpu_ordinal, bool fabrics);

/* The xal-server index name of device di: OPENDS_XAL_SHM's entry, or
 * /xal_dev<di>. */
int aisio_config_xal_shm(int di, int n_devices, char *out, size_t out_len);

#endif /* DS_AISIO_CONFIG_H_ */
