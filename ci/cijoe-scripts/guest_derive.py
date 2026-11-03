# SPDX-License-Identifier: BSD-3-Clause
"""Turn the derive guest into the OpenDS guest.

Runs tasks/steps/guest_derive.sh in it, in two phases around a reboot into the kernel the image boots.
"""
import errno
import logging as log
import shlex
import time
from argparse import ArgumentParser
from pathlib import Path

REMOTE = "/root/opends-guest"
SCRIPT = Path(__file__).resolve().parent.parent / "tasks" / "steps" / "guest_derive.sh"


def add_args(parser: ArgumentParser):
    parser.add_argument("--guest_name", type=str, default="derive")
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds to wait for the guest after its reboot")


def reboot(cijoe, timeout):
    """Reboot the guest and return once it answers again, or False"""

    cijoe.run("systemd-run --on-active=2 systemctl reboot")
    time.sleep(15)
    deadline = time.time() + timeout
    while time.time() < deadline:
        err, _ = cijoe.run("true")
        if not err:
            return True
        time.sleep(5)
    return False


def main(args, cijoe):
    """Put the inputs in the guest and run the two phases around a reboot"""

    debs = sorted(Path(cijoe.getconf("guest.debs")).glob("*.deb"))
    if not debs:
        log.error("no debs fetched; fetch_debs runs before this step")
        return errno.ENOENT
    nvidia = cijoe.getconf("guest.nvidia")
    if not nvidia:
        log.error("guest.nvidia is not set; is configs/guest-lock.toml given?")
        return errno.EINVAL

    err, _ = cijoe.run(f"mkdir -p {REMOTE}")
    if err:
        return err
    for path in [SCRIPT, *debs]:
        if not cijoe.put(str(path), f"{REMOTE}/{path.name}"):
            log.error(f"could not put {path} in the guest")
            return errno.EIO

    err, _ = cijoe.run(f"bash {REMOTE}/guest_derive.sh prepare")
    if err:
        return err
    if not reboot(cijoe, args.timeout):
        log.error("the guest did not come back from its reboot")
        return errno.ETIMEDOUT

    packages = " ".join(shlex.quote(p) for p in nvidia["packages"])
    err, _ = cijoe.run(
        f"bash {REMOTE}/guest_derive.sh install {REMOTE} "
        f"{shlex.quote(nvidia['repo'])} {shlex.quote(nvidia['cuda'])} "
        f"{shlex.quote(nvidia['pinning'])} {packages}"
    )
    if err:
        return err
    err, _ = cijoe.run(f"rm -rf {REMOTE}")
    return err
