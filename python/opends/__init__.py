# SPDX-License-Identifier: BSD-3-Clause
"""OpenDS Python bindings: a ctypes layer over the OpenDS C ABI.

OPENDS_BACKEND selects libopends_<backend>.so (default aisio);
OPENDS_LIBRARY loads a specific file instead.
"""

from .batch import READ, WRITE, Batch, BatchEvent, BatchOp, Status
from .buffer import (
    HostBuffer,
    alloc,
    deregister_buffer,
    deregister_stream,
    free,
    register_buffer,
    register_stream,
)
from .cdll import ErrorCode
from .driver import (
    Driver,
    DriverProperties,
    OpenDSError,
    cleanup,
    get_properties,
    get_version,
    set_max_direct_io_size,
    use_count,
)
from .file import Future, OpenDSFile, StreamOp

__all__ = [
    "Batch",
    "BatchEvent",
    "BatchOp",
    "Driver",
    "DriverProperties",
    "ErrorCode",
    "Future",
    "HostBuffer",
    "OpenDSError",
    "OpenDSFile",
    "READ",
    "Status",
    "StreamOp",
    "WRITE",
    "alloc",
    "cleanup",
    "deregister_buffer",
    "deregister_stream",
    "free",
    "get_properties",
    "get_version",
    "register_buffer",
    "register_stream",
    "set_max_direct_io_size",
    "use_count",
]
