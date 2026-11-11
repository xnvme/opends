/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_extent.h - File-to-LBA extent triple.
 *
 * The aisio backend resolves a file's byte ranges to device LBAs with FIEMAP
 * and drives DMA per extent. This is the internal currency between FIEMAP,
 * the read engine, and the extent record the tests serialize.
 */
#ifndef DS_EXTENT_H_
#define DS_EXTENT_H_

#include <stdint.h>

struct ds_extent {
	uint64_t file_offset;
	uint64_t slba;
	uint64_t length;
	uint8_t flag; /* bits of enum xal_extent_flag; 0 holds the file's data
	               */
};

#endif /* DS_EXTENT_H_ */
