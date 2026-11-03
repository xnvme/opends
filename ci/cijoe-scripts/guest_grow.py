# SPDX-License-Identifier: BSD-3-Clause
"""Grow the derive guest's disk before its first boot, since CUDA does not fit the base's 16 GB."""
import logging as log
from argparse import ArgumentParser

from cijoe.qemu.wrapper import Guest


def add_args(parser: ArgumentParser):
    parser.add_argument("--guest_name", type=str, default="derive")


def main(args, cijoe):
    """Resize the guest's boot image"""

    guest = Guest(cijoe, cijoe.config, args.guest_name)
    size = cijoe.getconf("guest.disk_size", "64G")
    err, _ = cijoe.run_local(f"qemu-img resize {guest.boot_img} {size}")
    if err:
        log.error(f"could not grow {guest.boot_img} to {size}: err({err})")
    return err
