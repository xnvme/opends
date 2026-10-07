# SPDX-License-Identifier: BSD-3-Clause
"""OpenDS Python bindings: a ctypes layer over the OpenDS C ABI.

OPENDS_BACKEND selects libopends_<backend>.so (default aisio);
OPENDS_LIBRARY loads a specific file instead.
"""

from .buffer import HostBuffer, alloc, deregister_buffer, free, register_buffer
from .driver import OpenDSError, cleanup, get_version
from .file import OpenDSFile

__all__ = [
    "HostBuffer",
    "OpenDSError",
    "OpenDSFile",
    "alloc",
    "cleanup",
    "deregister_buffer",
    "free",
    "get_version",
    "register_buffer",
]
