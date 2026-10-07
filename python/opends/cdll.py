# SPDX-License-Identifier: BSD-3-Clause
"""Library loader and ctypes prototypes for the OpenDS C ABI.

ABI churn lives here. Every signature mirrors the headers under
include/; keep them in sync when the C API changes.
"""

import ctypes
import enum
import os
from ctypes import (
    POINTER,
    c_char_p,
    c_int,
    c_long,
    c_size_t,
    c_ssize_t,
    c_uint,
    c_void_p,
)
from pathlib import Path


class ErrorCode(enum.IntEnum):
    """opends_op_error_t."""

    SUCCESS = 0
    DRIVER_NOT_INITIALIZED = 5001
    DRIVER_INVALID_PROPS = 5002
    DRIVER_UNSUPPORTED_LIMIT = 5003
    DRIVER_VERSION_MISMATCH = 5004
    DRIVER_VERSION_READ_ERROR = 5005
    DRIVER_CLOSING = 5006
    PLATFORM_NOT_SUPPORTED = 5007
    IO_NOT_SUPPORTED = 5008
    DEVICE_NOT_SUPPORTED = 5009
    FS_DRIVER_ERROR = 5010
    DEVICE_DRIVER_ERROR = 5011
    POINTER_INVALID = 5012
    MEMORY_TYPE_INVALID = 5013
    POINTER_RANGE_ERROR = 5014
    CONTEXT_MISMATCH = 5015
    INVALID_MAPPING_SIZE = 5016
    INVALID_MAPPING_RANGE = 5017
    INVALID_FILE_TYPE = 5018
    INVALID_FILE_OPEN_FLAG = 5019
    DIO_NOT_SET = 5020
    INVALID_VALUE = 5022
    MEMORY_ALREADY_REGISTERED = 5023
    MEMORY_NOT_REGISTERED = 5024
    PERMISSION_DENIED = 5025
    DRIVER_ALREADY_OPEN = 5026
    HANDLE_NOT_REGISTERED = 5027
    HANDLE_ALREADY_REGISTERED = 5028
    DEVICE_NOT_FOUND = 5029
    INTERNAL_ERROR = 5030
    GETNEWFD_FAILED = 5031
    FS_SETUP_ERROR = 5033
    IO_DISABLED = 5034
    BATCH_SUBMIT_FAILED = 5035
    MEMORY_PINNING_FAILED = 5036
    BATCH_FULL = 5037
    ASYNC_NOT_SUPPORTED = 5038
    IO_MAX_ERROR = 5039


class DsError(ctypes.Structure):
    """opends_error_t: {opends_op_error_t err; opends_result_t dev_err;}."""

    _fields_ = [("err", c_int), ("dev_err", c_int)]


class DsDrvProps(ctypes.Structure):
    """opends_drv_props_t."""

    _fields_ = [
        ("major_version", c_uint),
        ("minor_version", c_uint),
        ("max_direct_io_size", c_size_t),
        ("max_batch_io_size", c_uint),
        ("max_batch_io_timeout_msecs", c_uint),
    ]


class DsAsyncFuture(ctypes.Structure):
    """opends_async_future_t."""

    _fields_ = [("done", c_uint), ("result", c_ssize_t)]


BACKEND = os.environ.get("OPENDS_BACKEND", "aisio")


def _candidate_paths():
    soname = "libopends_%s.so" % BACKEND
    explicit = os.environ.get("OPENDS_LIBRARY")
    if explicit:
        yield explicit
    repo = Path(__file__).resolve().parents[2]
    yield str(repo / "build" / soname)
    yield soname


def _load():
    last = None
    for path in _candidate_paths():
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            last = exc
    raise OSError(
        "could not load libopends_%s.so; set OPENDS_LIBRARY or build the "
        "C library first (meson compile -C build). Last error: %s"
        % (BACKEND, last)
    )


_lib = _load()


def _decl(name, restype, argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


driver_open = _decl("opends_driver_open", DsError, [])
driver_close = _decl("opends_driver_close", DsError, [])
use_count = _decl("opends_use_count", c_long, [])
get_version = _decl(
    "opends_get_version",
    DsError,
    [POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)],
)
driver_get_properties = _decl(
    "opends_driver_get_properties", DsError, [POINTER(DsDrvProps)]
)
driver_set_max_direct_io_size = _decl(
    "opends_driver_set_max_direct_io_size", DsError, [c_size_t]
)

handle_register = _decl(
    "opends_handle_register", DsError, [POINTER(c_void_p), c_int]
)
handle_deregister = _decl("opends_handle_deregister", None, [c_void_p])

alloc = _decl("opends_alloc", c_void_p, [c_size_t])
free = _decl("opends_free", None, [c_void_p])

buf_register = _decl(
    "opends_buf_register", DsError, [c_void_p, c_size_t, c_int]
)
buf_deregister = _decl("opends_buf_deregister", DsError, [c_void_p])

sync_read = _decl(
    "opends_sync_read", c_ssize_t, [c_void_p, c_void_p, c_size_t, c_long, c_long]
)
sync_write = _decl(
    "opends_sync_write", c_ssize_t, [c_void_p, c_void_p, c_size_t, c_long, c_long]
)

async_read = _decl(
    "opends_async_read",
    DsError,
    [c_void_p, c_void_p, c_size_t, c_long, c_long, POINTER(DsAsyncFuture)],
)
async_write = _decl(
    "opends_async_write",
    DsError,
    [c_void_p, c_void_p, c_size_t, c_long, c_long, POINTER(DsAsyncFuture)],
)
async_await = _decl("opends_async_await", c_ssize_t, [POINTER(DsAsyncFuture)])

stream_read = _decl(
    "opends_stream_read",
    DsError,
    [
        c_void_p,
        c_void_p,
        POINTER(c_size_t),
        POINTER(c_long),
        POINTER(c_long),
        POINTER(c_ssize_t),
        c_void_p,
    ],
)
stream_write = _decl(
    "opends_stream_write",
    DsError,
    [
        c_void_p,
        c_void_p,
        POINTER(c_size_t),
        POINTER(c_long),
        POINTER(c_long),
        POINTER(c_ssize_t),
        c_void_p,
    ],
)
stream_register = _decl("opends_stream_register", DsError, [c_void_p, c_uint])
stream_deregister = _decl("opends_stream_deregister", DsError, [c_void_p])

op_status_error = _decl("opends_op_status_error", c_char_p, [c_int])
