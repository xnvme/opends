# SPDX-License-Identifier: BSD-3-Clause
"""Stream registration and StreamOp, the result of one stream I/O call."""

import ctypes

from . import cdll as _c
from .driver import check, io_result, require_driver


def stream_handle(stream):
    """Raw handle of a CUDA stream-like object. None is the default stream."""
    if stream is None:
        return 0
    if isinstance(stream, ctypes.c_void_p):
        return int(stream.value or 0)
    if isinstance(stream, int):
        return stream
    for attr in ("cuda_stream", "ptr"):  # torch.cuda.Stream, cupy.cuda.Stream
        value = getattr(stream, attr, None)
        if value is not None:
            return int(value)
    raise TypeError("unsupported stream object %r" % (stream,))


def register_stream(stream, flags=0):
    """cuFileStreamRegister."""
    require_driver()
    check(_c.stream_register(stream_handle(stream), flags))


def deregister_stream(stream):
    check(_c.stream_deregister(stream_handle(stream)))


# Initial byte count of a StreamOp: never a byte count, never a negated code.
_PENDING = -1


class StreamOp:
    """One read_stream or write_stream call.

    The backend reads the size and offsets, and writes the byte count, when
    the stream reaches the operation. Keep this object alive until the
    stream has been synchronized; result() raises until then.
    """

    def __init__(self, size, file_offset, dev_offset, buf):
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_long(file_offset)
        self._dev_offset = ctypes.c_long(dev_offset)
        self._bytes = ctypes.c_ssize_t(_PENDING)
        self._buf = buf

    @property
    def done(self):
        return self._bytes.value != _PENDING

    def result(self):
        """Byte count once the stream has executed the operation."""
        if not self.done:
            raise RuntimeError(
                "the stream has not reached this operation; synchronize it first"
            )
        return io_result(int(self._bytes.value))
