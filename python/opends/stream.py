# SPDX-License-Identifier: BSD-3-Clause
"""Stream registration and StreamOp, the result of one stream I/O call."""

import ctypes

from . import cdll as _c
from .driver import check, io_result, preserve_cuda_context, require_driver

# register_stream flags, cuFileStreamRegister's. With the three FIXED ones a
# stream write takes its size and offsets at the call and the copy engine
# stages it; without them a kernel on the stream reads them late.
STREAM_FIXED_BUF_OFFSET = 1
STREAM_FIXED_FILE_OFFSET = 2
STREAM_FIXED_FILE_SIZE = 4
STREAM_PAGE_ALIGNED_INPUTS = 8
STREAM_FIXED = (
    STREAM_FIXED_BUF_OFFSET | STREAM_FIXED_FILE_OFFSET | STREAM_FIXED_FILE_SIZE
)


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
    with preserve_cuda_context():
        check(_c.stream_register(stream_handle(stream), flags))


def deregister_stream(stream):
    with preserve_cuda_context():
        check(_c.stream_deregister(stream_handle(stream)))


class StreamOp:
    """One read_stream or write_stream call.

    The backend reads the size and offsets, and writes the byte count, when
    the stream reaches the operation. Keep this object alive until the
    stream has been synchronized. The byte count is 0 until then, as with
    cuFile, which requires the storage to start at 0.
    """

    def __init__(self, size, file_offset, dev_offset, buf):
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_long(file_offset)
        self._dev_offset = ctypes.c_long(dev_offset)
        self._bytes = ctypes.c_ssize_t(0)
        self._buf = buf

    @property
    def nbytes(self):
        """cuFile's byte count as written: 0 before the stream has run, the
        count, -1 for an I/O error, or a negated code."""
        return int(self._bytes.value)

    def result(self):
        """Byte count of the operation; raises for a failed one."""
        return io_result(self.nbytes)
