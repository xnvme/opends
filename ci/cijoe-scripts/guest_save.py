# SPDX-License-Identifier: BSD-3-Clause
"""Power the ci guest off and keep its disk as an image on the one it booted.

The overlay names its backing file without a directory, so the pair works wherever the cache is mounted.
"""
import errno
import logging as log
import time
from argparse import ArgumentParser
from pathlib import Path

from cijoe.qemu.wrapper import Guest


def add_args(parser: ArgumentParser):
    parser.add_argument("--guest_name", type=str, default="ci")
    parser.add_argument("--image_name", type=str, default="opends-deps-x86_64")
    parser.add_argument("--base_name", type=str, default="opends-guest-x86_64")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds to wait for the guest to power off")


def main(args, cijoe):
    """Shut the guest down, then move its overlay beside the image under it"""

    guest = Guest(cijoe, cijoe.config, args.guest_name)
    paths = []
    for name in [args.image_name, args.base_name]:
        path = cijoe.getconf(f"system-imaging.images.{name}.disk.path")
        if not path:
            log.error(f"system-imaging.images.{name}.disk.path is not set")
            return errno.EINVAL
        paths.append(Path(path))
    image, base = paths
    if image.parent != base.parent:
        log.error(f"{image} and {base} are not in one directory")
        return errno.EINVAL

    err, _ = cijoe.run(
        "rm -f /root/.ssh/authorized_keys && truncate -s 0 /etc/machine-id && sync"
    )
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

    # Under another name until it is whole, as a job boots what it finds
    err, _ = cijoe.run_local(
        f"mv {guest.boot_img} {image}.tmp"
        f" && qemu-img rebase -u -F qcow2 -b {base.name} {image}.tmp"
        f" && mv {image}.tmp {image} && qemu-img info --backing-chain {image}"
    )
    return err
