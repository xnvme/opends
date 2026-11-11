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
	int n_io_threads; ///< Process total, split over the devices
	uint32_t queue_depth;
	uint32_t idle_spin_us;
	bool busy_spin;
	uint64_t cpu_mask;
	bool assume_aligned_only;
	bool cq_mirror; ///< CQ in GPU memory, warp-mirrored to host (upcie-cuda
	                ///< only)
	size_t host_heap_nbytes;
	size_t device_heap_nbytes;
	size_t write_buf_nbytes; ///< Pinned host buf a write is staged
	                         ///< through
	int write_bufs;          ///< Most buffers the driver pins
};

/* The homi group to join, on its own because device discovery needs it
 * before the rest is read. */
int aisio_config_homi_id(uint32_t *out);

/* Everything else. Returns 0, or -EINVAL after a message on stderr. */
int aisio_config_read(struct aisio_config *cfg, int n_devices);

/* The mountpoint of device di's filesystem: OPENDS_HOMI_MNT's entry. No
 * default. */
int aisio_config_mount(int di, int n_devices, char *out, size_t out_len);

#endif /* DS_AISIO_CONFIG_H_ */
