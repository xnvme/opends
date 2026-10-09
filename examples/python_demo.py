#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Read one file through every OpenDS API mode from Python, into GPU memory.

Writes a pattern file on the OpenDS mount, then reads it back with the sync,
async, stream and batch families and checks every byte.

    OPENDS_XAL_SHM=/xal_dev0 OPENDS_HOMI_MNT=/mnt/datasets \\
        python3 examples/python_demo.py /mnt/datasets/demo.bin
"""

import os
import sys

import torch

try:
    import opends
except ImportError:  # run from a source checkout without installing
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    import opends

PAGE = 4096
CHUNKS = 8
SIZE = CHUNKS * PAGE
REGIONS = 5


def pattern(n):
    return bytes((i * 31 + 7) % 251 for i in range(n))


def expect(cond, what):
    if not cond:
        sys.exit("FAILED: %s" % what)


def check(mode, tensor, region, payload):
    got = tensor[region * SIZE : (region + 1) * SIZE].cpu().numpy().tobytes()
    expect(got == payload, "%s read returned wrong data" % mode)
    print("%-8s ok" % mode)


def main(path):
    payload = pattern(SIZE)
    tensor = torch.zeros(REGIONS * SIZE, dtype=torch.uint8, device="cuda")
    host = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    tensor[:SIZE] = host
    stream = torch.cuda.Stream()

    with opends.Driver():
        opends.register_buffer(tensor)
        opends.register_stream(stream)
        try:
            with opends.OpenDSFile(path, "w") as f:
                expect(f.write_sync(tensor, size=SIZE) == SIZE, "write")

            with opends.OpenDSFile(path, "r") as f:
                props = opends.get_properties()
                print(
                    "opends %d.%d.%d, driver %d.%d, max direct io %d"
                    % (
                        *opends.get_version(),
                        props.major_version,
                        props.minor_version,
                        props.max_direct_io_size,
                    )
                )

                # Sync:
                n = f.read_sync(tensor, size=SIZE, dev_offset=1 * SIZE)
                expect(n == SIZE, "sync byte count")
                check("sync", tensor, 1, payload)

                # Async:
                futs = [
                    f.read_async(
                        tensor,
                        size=PAGE,
                        file_offset=i * PAGE,
                        dev_offset=2 * SIZE + i * PAGE,
                    )
                    for i in range(CHUNKS)
                ]
                for fut in futs:
                    expect(fut.result() == PAGE, "async byte count")
                check("async", tensor, 2, payload)

                # Stream:
                op = f.read_stream(
                    tensor, size=SIZE, dev_offset=3 * SIZE, stream=stream
                )
                stream.synchronize()
                expect(op.result() == SIZE, "stream byte count")
                check("stream", tensor, 3, payload)

                # Batch:
                with opends.Batch(CHUNKS) as batch:
                    batch.submit(
                        opends.BatchOp(
                            f,
                            tensor,
                            size=PAGE,
                            file_offset=i * PAGE,
                            dev_offset=4 * SIZE + i * PAGE,
                            cookie=i,
                        )
                        for i in range(CHUNKS)
                    )
                    events = batch.get_status(min_nr=CHUNKS)
                cookies = sorted(ev.cookie for ev in events)
                expect(cookies == list(range(CHUNKS)), "batch cookies")
                for ev in events:
                    expect(ev.status is opends.Status.COMPLETE, "batch status")
                    expect(ev.ret == PAGE, "batch byte count")
                check("batch", tensor, 4, payload)
        finally:
            opends.deregister_stream(stream)
            opends.deregister_buffer(tensor)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python_demo.py <file-on-opends-mount>")
    main(sys.argv[1])
