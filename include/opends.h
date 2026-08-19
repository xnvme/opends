/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * opends.h - Accelerator Direct Storage File I/O Interface
 *
 * Vendor-neutral API for direct storage access, structurally compatible
 * with NVIDIA's cuFile API. Applications written against cuFile can be
 * ported by renaming symbols.
 *
 * The reference backend (libopends_ref) uses POSIX pread/pwrite on
 * host buffers and has no external dependencies.
 *
 * Deviations from cuFile:
 *
 *   - Linux only. File handles are plain file descriptors; the
 *     cuFileDescr_t type/union and USERSPACE_FS handle type are
 *     not supported.
 *
 *   - Buffer registration (cuFileBufRegister/Deregister) is offered
 *     alongside backend-owned allocation via opends_alloc/opends_free.
 *     Callers may allocate their own memory (e.g. cudaMalloc) and call
 *     opends_buf_register, or let the backend handle both with
 *     opends_alloc.
 *
 * Error reporting:
 *
 *   Functions returning opends_error_t use a structured error with
 *   both an opends error code and an optional backend-specific code.
 *
 *   Functions returning ssize_t (read/write) return the byte count on
 *   success or a negated opends_op_error_t on failure.
 *
 * Thread safety:
 *
 *   Once opends_driver_open has returned, I/O submission and
 *   registration (handles, buffers, streams) may run concurrently
 *   from multiple threads. opends_driver_open and opends_driver_close
 *   must not run concurrently with any other opends call.
 *   Deregistering or freeing an object with I/O still in flight on it
 *   is undefined, as with closing a file descriptor that has I/O in
 *   flight.
 */
#ifndef OPENDS_H_
#define OPENDS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

typedef int opends_result_t;

/* clang-format off */
typedef enum opends_op_error {
	OPENDS_SUCCESS                  = 0,
	OPENDS_DRIVER_NOT_INITIALIZED   = 5001,
	OPENDS_DRIVER_INVALID_PROPS     = 5002,
	OPENDS_DRIVER_UNSUPPORTED_LIMIT = 5003,
	OPENDS_DRIVER_VERSION_MISMATCH  = 5004,
	OPENDS_DRIVER_VERSION_READ_ERROR = 5005,
	OPENDS_DRIVER_CLOSING           = 5006,
	OPENDS_PLATFORM_NOT_SUPPORTED   = 5007,
	OPENDS_IO_NOT_SUPPORTED         = 5008,
	OPENDS_DEVICE_NOT_SUPPORTED     = 5009,
	OPENDS_FS_DRIVER_ERROR          = 5010,
	OPENDS_DEVICE_DRIVER_ERROR      = 5011,
	OPENDS_POINTER_INVALID          = 5012,
	OPENDS_MEMORY_TYPE_INVALID      = 5013,
	OPENDS_POINTER_RANGE_ERROR      = 5014,
	OPENDS_CONTEXT_MISMATCH         = 5015,
	OPENDS_INVALID_MAPPING_SIZE     = 5016,
	OPENDS_INVALID_MAPPING_RANGE    = 5017,
	OPENDS_INVALID_FILE_TYPE        = 5018,
	OPENDS_INVALID_FILE_OPEN_FLAG   = 5019,
	OPENDS_DIO_NOT_SET              = 5020,
	/* 5021 intentionally unused */
	OPENDS_INVALID_VALUE            = 5022,
	OPENDS_MEMORY_ALREADY_REGISTERED = 5023,
	OPENDS_MEMORY_NOT_REGISTERED    = 5024,
	OPENDS_PERMISSION_DENIED        = 5025,
	OPENDS_DRIVER_ALREADY_OPEN      = 5026,
	OPENDS_HANDLE_NOT_REGISTERED    = 5027,
	OPENDS_HANDLE_ALREADY_REGISTERED = 5028,
	OPENDS_DEVICE_NOT_FOUND         = 5029,
	OPENDS_INTERNAL_ERROR           = 5030,
	OPENDS_GETNEWFD_FAILED          = 5031,
	/* 5032 intentionally unused */
	OPENDS_FS_SETUP_ERROR           = 5033,
	OPENDS_IO_DISABLED              = 5034,
	OPENDS_BATCH_SUBMIT_FAILED      = 5035,
	OPENDS_MEMORY_PINNING_FAILED    = 5036,
	OPENDS_BATCH_FULL               = 5037,
	OPENDS_ASYNC_NOT_SUPPORTED      = 5038,
	OPENDS_IO_MAX_ERROR             = 5039,
} opends_op_error_t;
/* clang-format on */

typedef struct opends_error {
	opends_op_error_t err;
	/* Backend-specific subcode for err (e.g. a CUDA error code from the
	 * aisio backend's accelerator runtime); 0 or -1 when none applies. */
	opends_result_t dev_err;
} opends_error_t;

#define OPENDS_BASE_ERR 5000
#define IS_OPENDS_ERR(err) (abs((err)) > OPENDS_BASE_ERR)
#define OPENDS_ERRSTR(err) opends_op_status_error((opends_op_error_t)abs((err)))

const char *opends_op_status_error(opends_op_error_t status);

typedef struct opends_drv_props {
	unsigned int major_version;
	unsigned int minor_version;
	size_t max_direct_io_size;
	unsigned int max_batch_io_size;
	unsigned int max_batch_io_timeout_msecs;
} opends_drv_props_t;

typedef void *opends_handle_t;

/*
 * Driver lifecycle. Call opends_driver_open() once before any other
 * opends operation. Call opends_driver_close() to release all
 * resources when done.
 *
 * Backends bound to an accelerator context (e.g. aisio) capture the
 * context current at opends_driver_open. Every thread that submits
 * I/O must have that same context current; the aisio backend fails
 * such submits with OPENDS_CONTEXT_MISMATCH.
 */
opends_error_t opends_driver_open(void);
opends_error_t opends_driver_close(void);
long opends_use_count(void);
opends_error_t opends_get_version(unsigned *major, unsigned *minor,
                                  unsigned *patch);
opends_error_t opends_driver_get_properties(opends_drv_props_t *props);
opends_error_t opends_driver_set_max_direct_io_size(size_t max_direct_io_size);

/*
 * File handle registration. Each file descriptor must be registered
 * before it can be used with opends_read/write. The file must be
 * opened with O_DIRECT. Linux only.
 */
opends_error_t opends_handle_register(opends_handle_t *fh, int fd);
void opends_handle_deregister(opends_handle_t fh);

/*
 * Buffer allocation. Buffers used with opends_read/write must be
 * either allocated through opends_alloc or registered with
 * opends_buf_register so the backend can set up DMA mappings.
 */
void *opends_alloc(size_t size);
void opends_free(void *buf);

/*
 * Register an externally allocated buffer for use with opends_read
 * and opends_write. The caller retains ownership of the allocation;
 * deregister before freeing. flags is forwarded to the backend (e.g.
 * cuFileBufRegister flags for gds); the ref and aisio backends ignore
 * it.
 */
opends_error_t opends_buf_register(const void *buf_base, size_t size,
                                   int flags);
opends_error_t opends_buf_deregister(const void *buf_base);

/*
 * Synchronous I/O. Returns byte count on success or a negated
 * opends_op_error_t on failure. The buf_offset parameter writes
 * into the buffer at an offset, useful for scatter reads into a
 * single allocation.
 */
ssize_t opends_read(opends_handle_t fh, void *buf_base, size_t size,
                    off_t file_offset, off_t buf_offset);
ssize_t opends_write(opends_handle_t fh, const void *buf_base, size_t size,
                     off_t file_offset, off_t buf_offset);

/*
 * Batch I/O. Submit multiple operations in a single call and poll for
 * completion, useful for overlapping I/O with computation. The reference
 * backend executes operations synchronously on submit and marks them
 * complete immediately.
 */
typedef enum opends_opcode {
	OPENDS_READ = 0,
	OPENDS_WRITE = 1,
} opends_opcode_t;

typedef enum opends_status {
	OPENDS_WAITING = 0x000001,
	OPENDS_PENDING = 0x000002,
	OPENDS_INVALID = 0x000004,
	OPENDS_CANCELED = 0x000008,
	OPENDS_COMPLETE = 0x000010,
	OPENDS_TIMEOUT = 0x000020,
	OPENDS_FAILED = 0x000040,
} opends_status_t;

typedef enum opends_batch_mode {
	OPENDS_BATCH = 1,
} opends_batch_mode_t;

typedef struct opends_io_params {
	opends_batch_mode_t mode;
	union {
		struct {
			void *dev_ptr_base;
			off_t file_offset;
			off_t dev_ptr_offset;
			size_t size;
		} batch;
	} u;
	opends_handle_t fh;
	opends_opcode_t opcode;
	void *cookie;
} opends_io_params_t;

typedef struct opends_io_events {
	void *cookie;
	opends_status_t status;
	size_t ret;
} opends_io_events_t;

typedef void *opends_batch_handle_t;

opends_error_t opends_batch_io_setup(opends_batch_handle_t *batch_idp,
                                     unsigned nr);
opends_error_t opends_batch_io_submit(opends_batch_handle_t batch_idp,
                                      unsigned nr, opends_io_params_t *iocbp,
                                      unsigned int flags);
/* Poll for at least min_nr completions, returning up to *nr events. */
opends_error_t opends_batch_io_get_status(opends_batch_handle_t batch_idp,
                                          unsigned min_nr, unsigned *nr,
                                          opends_io_events_t *iocbp,
                                          struct timespec *timeout);
opends_error_t opends_batch_io_cancel(opends_batch_handle_t batch_idp);
void opends_batch_io_destroy(opends_batch_handle_t batch_idp);

/*
 * Stream-based async I/O. Operations are associated with a stream (e.g. a
 * CUDA stream) and complete asynchronously. The size, offset, and byte
 * count parameters are pointers so the values can be read at stream
 * execution time rather than submission time. The reference backend reads
 * them immediately.
 *
 * Backend limits (aisio): at most 8192 streams may be registered at once,
 * and at most 1024 async ops may be in flight across all streams. Exceeding
 * the stream limit returns OPENDS_INTERNAL_ERROR from opends_stream_register;
 * the in-flight limit applies back-pressure rather than failing
 * (opends_read_async spins until a slot frees).
 */
typedef void *opends_stream_t;

opends_error_t opends_read_async(opends_handle_t fh, void *buf_base,
                                 size_t *size_p, off_t *file_offset_p,
                                 off_t *buf_offset_p, ssize_t *bytes_read_p,
                                 opends_stream_t stream);
opends_error_t opends_write_async(opends_handle_t fh, void *buf_base,
                                  size_t *size_p, off_t *file_offset_p,
                                  off_t *buf_offset_p, ssize_t *bytes_written_p,
                                  opends_stream_t stream);
opends_error_t opends_stream_register(opends_stream_t stream, unsigned flags);
opends_error_t opends_stream_deregister(opends_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* OPENDS_H_ */
