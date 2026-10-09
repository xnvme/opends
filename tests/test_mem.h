/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Backend-owned buffer at the caller's current accelerator, for the acquire
 * callbacks that report failure as NULL.
 */

#ifndef OPENDS_TEST_MEM_H
#define OPENDS_TEST_MEM_H

#include "opends.h"

#include <stddef.h>

static inline void *
test_dev_alloc(size_t size)
{
	void *buf = NULL;
	opends_error_t err;

	err = opends_mem_alloc(size, OPENDS_MEM_DEVICE, OPENDS_DEVICE_CURRENT,
	                       &buf);
	return err.err == OPENDS_SUCCESS ? buf : NULL;
}

#endif /* OPENDS_TEST_MEM_H */
