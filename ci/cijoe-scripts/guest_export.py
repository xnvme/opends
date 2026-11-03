# SPDX-License-Identifier: BSD-3-Clause
"""Power the derive guest off and flatten its disk into the image the ci guest boots."""
import errno
import logging as log
import time
from argparse import ArgumentParser
from pathlib import Path

from cijoe.qemu.wrapper import Guest


def add_args(parser: ArgumentParser):
    parser.add_argument("--guest_name", type=str, default="derive")
    parser.add_argument("--image_name", type=str, default="opends-guest-x86_64")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds to wait for the guest to power off")


def main(args, cijoe):
    """Shut the guest down, then flatten its disk into the image"""

    guest = Guest(cijoe, cijoe.config, args.guest_name)
    path = cijoe.getconf(f"system-imaging.images.{args.image_name}.disk.path")
    if not path:
        log.error(f"system-imaging.images.{args.image_name}.disk.path is not set")
        return errno.EINVAL
    image = Path(path)

    err, _ = cijoe.run("truncate -s 0 /etc/machine-id && sync")
    if err:
        return err
    # Detached, so the shell returns before sshd goes away
    cijoe.run("systemd-run --on-active=2 systemctl poweroff")
    began = time.time()
    while guest.is_running() and time.time() - began < args.timeout:
        time.sleep(5)
    if guest.is_running():
        log.error("the guest did not power off")
        return errno.ETIMEDOUT

    image.parent.mkdir(parents=True, exist_ok=True)
    err, _ = cijoe.run_local(
        f"qemu-img convert -O qcow2 {guest.boot_img} {image} && qemu-img info {image}"
    )
    return err
