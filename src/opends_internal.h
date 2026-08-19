/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef OPENDS_INTERNAL_H_
#define OPENDS_INTERNAL_H_

#include "opends.h"

static inline opends_error_t
opends_ok(void)
{
	return (opends_error_t){OPENDS_SUCCESS, 0};
}

static inline opends_error_t
opends_err(opends_op_error_t e)
{
	return (opends_error_t){e, 0};
}

static inline opends_error_t
opends_err_dev(opends_op_error_t e, opends_result_t dev_err)
{
	return (opends_error_t){e, dev_err};
}

#endif /* OPENDS_INTERNAL_H_ */
