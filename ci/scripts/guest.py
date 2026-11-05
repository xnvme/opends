#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Boot and stop the guest a hardware job runs in.

up boots the ci guest with the machine's NVMe device and GPU and authorizes a key for this run in it; down kills it.

Run it as root in the xnvme-devenv host container. The devices' addresses on the host come from /etc/opends-ci/rig.toml, or the file OPENDS_CI_RIG names.
"""
import argparse
import os
import pwd
import shutil
import subprocess
import sys
from pathlib import Path

from _helpers import ROOT, fail, ok

RIG = Path(os.environ.get("OPENDS_CI_RIG", "/etc/opends-ci/rig.toml"))
# OpenSSH reads the config of the account's home, whatever HOME says
SSH_DIR = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".ssh"
KEY = SSH_DIR / "opends-guest"
SSH_CONFIG = """\
# The guest of OpenDS's scripts/guest.py, on the port QEMU forwards
Host localhost
    Port 4200
    User root
    IdentityFile {key}
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
"""


def cijoe(task, env=None):
    """Run a guest task with the guest's configs; returns the rc"""

    configs = ["-c", "configs/guest.toml", "-c", "configs/guest-lock.toml"]
    if RIG.exists():
        configs += ["-c", str(RIG)]
    return subprocess.run(
        ["cijoe", "-m", "-o", f"cijoe-output-{Path(task).stem}", *configs, task],
        cwd=ROOT, env={**os.environ, **(env or {})},
    ).returncode


def make_key():
    """A key pair for this run and the SSH config entry that uses it"""

    SSH_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not KEY.exists():
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "opends-guest",
             "-f", str(KEY)],
            check=True,
        )
    config = SSH_DIR / "config"
    entry = SSH_CONFIG.format(key=KEY)
    if not config.exists() or entry not in config.read_text():
        with config.open("a") as f:
            f.write(entry)
        config.chmod(0o600)
    return KEY.with_suffix(".pub").read_text().strip()


def up(_args):
    if not RIG.exists():
        fail(f"{RIG} is missing; it gives the devices' addresses on the host")
        sys.exit(1)
    rc = cijoe("tasks/guest_up.yaml", env={"OPENDS_GUEST_PUBKEY": make_key()})
    if rc:
        fail(f"tasks/guest_up.yaml (rc={rc})")
        sys.exit(rc)
    ok("guest up")


def down(_args):
    rc = cijoe("tasks/guest_down.yaml")
    serial = Path(os.environ["HOME"]) / "guests" / "opends-ci" / "serial.output"
    out = ROOT / "cijoe-output-guest_down"
    if serial.exists() and out.is_dir():
        shutil.copy(serial, out / "serial.output")
    if rc:
        fail(f"tasks/guest_down.yaml (rc={rc})")
        sys.exit(rc)
    ok("guest down")


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("up", help="boot the guest and make it the target").set_defaults(func=up)
sub.add_parser("down", help="kill the guest").set_defaults(func=down)
args = parser.parse_args()
args.func(args)
