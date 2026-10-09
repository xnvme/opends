/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends_cufile.c - cuFile backend for NVIDIA GPUDirect Storage.
 *
 * Thin wrapper around the cuFile API. Buffers are allocated on the GPU
 * with cudaMalloc and registered with cuFileBufRegister for DMA. Batch
 * and async operations delegate directly to cuFile.
 *
 * Requires: NVIDIA CUDA toolkit and cuFile (GDS) library.
 * Tested against cuFile API 1.x (GDS 1.4+).
 */

#define _GNU_SOURCE

#include "opends_internal.h"

#include <cufile.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

_Static_assert((int)OPENDS_SUCCESS == (int)CU_FILE_SUCCESS,
               "opends/cuFile error code mismatch: SUCCESS");
_Static_assert((int)OPENDS_DRIVER_NOT_INITIALIZED ==
                       (int)CU_FILE_DRIVER_NOT_INITIALIZED,
               "opends/cuFile error code mismatch: DRIVER_NOT_INITIALIZED");
_Static_assert((int)OPENDS_INTERNAL_ERROR == (int)CU_FILE_INTERNAL_ERROR,
               "opends/cuFile error code mismatch: INTERNAL_ERROR");

struct cufile_handle {
	CUfileHandle_t cufh;
};

static bool driver_open;
static long use_count;

static opends_error_t
from_cufile_error(CUfileError_t e)
{
	return (opends_error_t){(opends_op_error_t)e.err,
	                        (opends_result_t)e.cu_err};
}

/* ------------------------------------------------------------------ */
/*  Driver lifecycle                                                   */
/* ------------------------------------------------------------------ */

opends_error_t
opends_driver_open(void)
{
	if (driver_open)
		return opends_err(OPENDS_DRIVER_ALREADY_OPEN);
	CUfileError_t err = cuFileDriverOpen();
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	driver_open = true;
	return opends_ok();
}

opends_error_t
opends_driver_close(void)
{
	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	CUfileError_t err = cuFileDriverClose();
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	driver_open = false;
	return opends_ok();
}

long
opends_use_count(void)
{
	return use_count;
}

opends_error_t
opends_driver_get_properties(opends_drv_props_t *props)
{
	if (!props)
		return opends_err(OPENDS_INVALID_VALUE);

	CUfileDrvProps_t cu_props;
	CUfileError_t err = cuFileDriverGetProperties(&cu_props);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);

	memset(props, 0, sizeof(*props));
	props->major_version = cu_props.nvfs.major_version;
	props->minor_version = cu_props.nvfs.minor_version;
	props->max_direct_io_size = cu_props.nvfs.max_direct_io_size;
	props->max_batch_io_size = cu_props.max_batch_io_size;
	props->max_batch_io_timeout_msecs = cu_props.max_batch_io_timeout_msecs;
	return opends_ok();
}

opends_error_t
opends_driver_set_max_direct_io_size(size_t max_direct_io_size)
{
	CUfileError_t err = cuFileDriverSetMaxDirectIOSize(max_direct_io_size);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_get_version(unsigned *major, unsigned *minor, unsigned *patch)
{
	CUfileDrvProps_t cu_props;
	CUfileError_t err = cuFileDriverGetProperties(&cu_props);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);

	if (major)
		*major = cu_props.nvfs.major_version;
	if (minor)
		*minor = cu_props.nvfs.minor_version;
	if (patch)
		*patch = 0;
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Handle registration                                                */
/* ------------------------------------------------------------------ */

opends_error_t
opends_handle_register(opends_handle_t *fh, int fd)
{
	if (!fh || fd < 0)
		return opends_err(OPENDS_INVALID_VALUE);

	int oflags = fcntl(fd, F_GETFL);
	if (oflags < 0)
		return opends_err(OPENDS_INVALID_VALUE);
	if (!(oflags & O_DIRECT))
		return opends_err(OPENDS_DIO_NOT_SET);

	struct cufile_handle *h = malloc(sizeof(*h));
	if (!h)
		return opends_err(OPENDS_INTERNAL_ERROR);

	CUfileDescr_t desc;
	memset(&desc, 0, sizeof(desc));
	desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	desc.handle.fd = fd;

	CUfileError_t err = cuFileHandleRegister(&h->cufh, &desc);
	if (err.err != CU_FILE_SUCCESS) {
		free(h);
		return from_cufile_error(err);
	}

	*fh = h;
	use_count++;
	return opends_ok();
}

void
opends_handle_deregister(opends_handle_t fh)
{
	if (!fh)
		return;
	struct cufile_handle *h = fh;
	cuFileHandleDeregister(h->cufh);
	free(h);
	use_count--;
}

/* ------------------------------------------------------------------ */
/*  Buffer allocation                                                  */
/* ------------------------------------------------------------------ */

void *
opends_alloc(size_t size)
{
	void *ptr;
	if (cudaMalloc(&ptr, size) != cudaSuccess)
		return NULL;
	CUfileError_t err = cuFileBufRegister(ptr, size, 0);
	if (err.err != CU_FILE_SUCCESS) {
		cudaFree(ptr);
		return NULL;
	}
	return ptr;
}

void
opends_free(void *buf)
{
	if (!buf)
		return;
	cuFileBufDeregister(buf);
	cudaFree(buf);
}

opends_error_t
opends_buf_register(const void *buf_base, size_t size, int flags)
{
	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!buf_base || !size)
		return opends_err(OPENDS_INVALID_VALUE);
	CUfileError_t err = cuFileBufRegister(buf_base, size, flags);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_buf_deregister(const void *buf_base)
{
	if (!driver_open)
		return opends_err(OPENDS_DRIVER_NOT_INITIALIZED);
	if (!buf_base)
		return opends_err(OPENDS_INVALID_VALUE);
	CUfileError_t err = cuFileBufDeregister(buf_base);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Async I/O (not implemented)                                        */
/* ------------------------------------------------------------------ */

/* cuFile has no stream-free async primitive to map these onto. */

opends_error_t
opends_async_read(opends_handle_t fh, void *buf_base, size_t size,
                  off_t file_offset, off_t buf_offset,
                  opends_async_future_t *future)
{
	(void)fh;
	(void)buf_base;
	(void)size;
	(void)file_offset;
	(void)buf_offset;
	(void)future;
	return opends_err(OPENDS_ASYNC_NOT_SUPPORTED);
}

opends_error_t
opends_async_write(opends_handle_t fh, const void *buf_base, size_t size,
                   off_t file_offset, off_t buf_offset,
                   opends_async_future_t *future)
{
	(void)fh;
	(void)buf_base;
	(void)size;
	(void)file_offset;
	(void)buf_offset;
	(void)future;
	return opends_err(OPENDS_ASYNC_NOT_SUPPORTED);
}

ssize_t
opends_async_await(opends_async_future_t *future)
{
	(void)future;
	return -(ssize_t)OPENDS_ASYNC_NOT_SUPPORTED;
}

/* ------------------------------------------------------------------ */
/*  Synchronous I/O                                                    */
/* ------------------------------------------------------------------ */

ssize_t
opends_sync_read(opends_handle_t fh, void *buf_base, size_t size,
                 off_t file_offset, off_t buf_offset)
{
	struct cufile_handle *h = fh;
	return cuFileRead(h->cufh, buf_base, size, file_offset, buf_offset);
}

ssize_t
opends_sync_write(opends_handle_t fh, const void *buf_base, size_t size,
                  off_t file_offset, off_t buf_offset)
{
	struct cufile_handle *h = fh;
	return cuFileWrite(h->cufh, buf_base, size, file_offset, buf_offset);
}

/* ------------------------------------------------------------------ */
/*  Stream-ordered I/O                                                 */
/* ------------------------------------------------------------------ */

opends_error_t
opends_stream_read(opends_handle_t fh, void *buf_base, size_t *size_p,
                   off_t *file_offset_p, off_t *buf_offset_p,
                   ssize_t *bytes_read_p, opends_stream_t stream)
{
	struct cufile_handle *h = fh;
	CUfileError_t err =
	        cuFileReadAsync(h->cufh, buf_base, size_p, file_offset_p,
	                        buf_offset_p, bytes_read_p, (CUstream)stream);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_stream_write(opends_handle_t fh, void *buf_base, size_t *size_p,
                    off_t *file_offset_p, off_t *buf_offset_p,
                    ssize_t *bytes_written_p, opends_stream_t stream)
{
	struct cufile_handle *h = fh;
	CUfileError_t err = cuFileWriteAsync(h->cufh, buf_base, size_p,
	                                     file_offset_p, buf_offset_p,
	                                     bytes_written_p, (CUstream)stream);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_stream_register(opends_stream_t stream, unsigned flags)
{
	CUfileError_t err = cuFileStreamRegister((CUstream)stream, flags);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_stream_deregister(opends_stream_t stream)
{
	CUfileError_t err = cuFileStreamDeregister((CUstream)stream);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

/* ------------------------------------------------------------------ */
/*  Batch I/O                                                          */
/* ------------------------------------------------------------------ */

opends_error_t
opends_batch_setup(opends_batch_handle_t *batch_idp, unsigned nr)
{
	if (!batch_idp || nr == 0)
		return opends_err(OPENDS_INVALID_VALUE);

	CUfileError_t err =
	        cuFileBatchIOSetUp((CUfileBatchHandle_t *)batch_idp, nr);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_batch_submit(opends_batch_handle_t batch_idp, unsigned nr,
                    opends_io_params_t *iocbp, unsigned int flags)
{
	if (!batch_idp || !iocbp)
		return opends_err(OPENDS_INVALID_VALUE);

	CUfileIOParams_t *cu_params = calloc(nr, sizeof(*cu_params));
	if (!cu_params)
		return opends_err(OPENDS_INTERNAL_ERROR);

	for (unsigned i = 0; i < nr; i++) {
		struct cufile_handle *h = iocbp[i].fh;
		cu_params[i].mode = (CUfileBatchMode_t)iocbp[i].mode;
		cu_params[i].fh = h->cufh;
		cu_params[i].opcode = (CUfileOpcode_t)iocbp[i].opcode;
		cu_params[i].cookie = iocbp[i].cookie;
		cu_params[i].u.batch.devPtr_base =
		        iocbp[i].u.batch.dev_ptr_base;
		cu_params[i].u.batch.file_offset = iocbp[i].u.batch.file_offset;
		cu_params[i].u.batch.devPtr_offset =
		        iocbp[i].u.batch.dev_ptr_offset;
		cu_params[i].u.batch.size = iocbp[i].u.batch.size;
	}

	CUfileError_t err = cuFileBatchIOSubmit((CUfileBatchHandle_t)batch_idp,
	                                        nr, cu_params, flags);
	free(cu_params);

	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

opends_error_t
opends_batch_get_status(opends_batch_handle_t batch_idp, unsigned min_nr,
                        unsigned *nr, opends_io_events_t *iocbp,
                        struct timespec *timeout)
{
	if (!batch_idp || !nr || !iocbp)
		return opends_err(OPENDS_INVALID_VALUE);

	unsigned max_nr = *nr;
	CUfileIOEvents_t *cu_events = calloc(max_nr, sizeof(*cu_events));
	if (!cu_events)
		return opends_err(OPENDS_INTERNAL_ERROR);

	CUfileError_t err = cuFileBatchIOGetStatus(
	        (CUfileBatchHandle_t)batch_idp, min_nr, nr, cu_events, timeout);

	if (err.err != CU_FILE_SUCCESS) {
		free(cu_events);
		return from_cufile_error(err);
	}

	for (unsigned i = 0; i < *nr; i++) {
		iocbp[i].cookie = cu_events[i].cookie;
		iocbp[i].status = (opends_status_t)cu_events[i].status;
		iocbp[i].ret = cu_events[i].ret;
	}

	free(cu_events);
	return opends_ok();
}

opends_error_t
opends_batch_cancel(opends_batch_handle_t batch_idp)
{
	if (!batch_idp)
		return opends_err(OPENDS_INVALID_VALUE);
	CUfileError_t err = cuFileBatchIOCancel((CUfileBatchHandle_t)batch_idp);
	if (err.err != CU_FILE_SUCCESS)
		return from_cufile_error(err);
	return opends_ok();
}

void
opends_batch_destroy(opends_batch_handle_t batch_idp)
{
	if (!batch_idp)
		return;
	cuFileBatchIODestroy((CUfileBatchHandle_t)batch_idp);
}
