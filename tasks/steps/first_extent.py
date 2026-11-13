#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Print the device byte offset and length of a file's first extent.

FIEMAP through the mounted file system; the offset is on the block device the
file system is mounted on, which for the test suite is the ublk device qublk
exports, so it is also the namespace byte offset. Used to hand the raw block
test a range of the namespace that a file owns.

usage: first_extent.py FILE  ->  "<offset> <length>"
"""
import fcntl
import struct
import sys

FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 1
# The extent is the file's bytes on disk only when none of these is set:
# unknown, delalloc, encoded, encrypted, not aligned, inline, tail, unwritten.
FIEMAP_NOT_PLAIN = 0x2 | 0x4 | 0x8 | 0x80 | 0x100 | 0x200 | 0x400 | 0x800

with open(sys.argv[1], "rb") as f:
    n = 8
    hdr = struct.pack("QQIIII", 0, 2**64 - 1, FIEMAP_FLAG_SYNC, 0, n, 0)
    buf = bytearray(hdr + b"\0" * (56 * n))
    fcntl.ioctl(f, FS_IOC_FIEMAP, buf)
    mapped = struct.unpack_from("I", buf, 20)[0]
    if mapped == 0:
        sys.exit("no extents mapped for %s" % sys.argv[1])
    logical, physical, length = struct.unpack_from("QQQ", buf, 32)
    flags = struct.unpack_from("I", buf, 32 + 40)[0]
    if logical != 0 or flags & FIEMAP_NOT_PLAIN:
        sys.exit("first extent of %s is not a plain written extent at 0" % sys.argv[1])
    print(physical, length)
