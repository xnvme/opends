#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Boot and stop the guest a hardware job runs in.

up boots the ci guest with the machine's NVMe device and GPU and authorizes a key for this run in it; down kills it.
The guest boots from an image in the cache that has xnvme and xal installed. Images are named after a key of their inputs and built when missing; keys prints the names.

Run it as root in the xnvme-devenv host container. The devices' addresses on the host come from /etc/opends-ci/rig.toml, or the file OPENDS_CI_RIG names.
"""
import argparse
import hashlib
import os
import pwd
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from _helpers import ROOT, fail, guest_image_stem, ok

RIG = Path(os.environ.get("OPENDS_CI_RIG", "/etc/opends-ci/rig.toml"))
DISK = Path(os.environ["HOME"]) / "system_imaging" / "disk"
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
# What decides how xnvme and xal are built, beside their commits
DEPS_RECIPE = ["tasks/setup_xnvme.yaml", "tasks/setup_xal.yaml",
               "tasks/steps/setup_meson_dep.sh", "tasks/steps/install_prefix.sh"]
KEEP = {"guest": 2, "deps": 4}


def die(msg, rc=1):
    fail(msg)
    sys.exit(rc)


def cijoe(task, out=None, env=None):
    """Run a guest task with the guest's configs; exits when it fails"""

    configs = ["-c", "configs/guest.toml", "-c", "configs/guest-lock.toml"]
    if RIG.exists():
        configs += ["-c", str(RIG)]
    out = out or f"cijoe-output-{Path(task).stem}"
    rc = subprocess.run(
        ["cijoe", "-m", "-o", out, *configs, task],
        cwd=ROOT, env={**os.environ, **(env or {})},
    ).returncode
    if rc:
        die(f"{task} (rc={rc})", rc)


def script(*argv):
    """Run one of the suite's scripts against the guest; exits when it fails"""

    rc = subprocess.run([str(ROOT / "scripts" / argv[0]), *argv[1:]]).returncode
    if rc:
        sys.exit(rc)


def commit(remote, ref):
    """The commit 'ref' names in 'remote' today; a branch before a tag, as setup_meson_dep.sh checks out"""

    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    res = subprocess.run(
        ["git", "ls-remote", remote, ref, f"{ref}^{{}}"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if res.returncode:
        die(f"git ls-remote {remote} {ref}: {res.stderr.strip()}")
    refs = {name: sha for sha, name in
            (line.split("\t") for line in res.stdout.splitlines() if "\t" in line)}
    for name in [f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"]:
        if name in refs:
            return refs[name]
    return ref


def deps_stem(guest_stem):
    """The name of the image with the dependencies, for the given guest image"""

    with open(ROOT / "configs" / "deps.toml", "rb") as f:
        pins = tomllib.load(f)
    test = ROOT / "configs" / "test.toml"
    with open(test if test.exists() else test.with_name("guest-test.toml"), "rb") as f:
        paths = tomllib.load(f)["deps"]
    h = hashlib.sha256()
    for dep in ["xnvme", "xal"]:
        repo = pins[dep]["repository"]
        sha = commit(repo["remote"], repo["ref"])
        print(f"{dep}: {repo['ref']} is {sha}", flush=True)
        h.update(f"{dep} {repo['remote']} {sha}\n".encode())
    for path in DEPS_RECIPE:
        h.update((ROOT / path).read_bytes())
    h.update(repr(sorted(paths.items())).encode())
    return f"opends-deps-{guest_stem.removeprefix('opends-guest-')}-{h.hexdigest()[:12]}"


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


def prune(in_use):
    """Remove the images used longest ago, and the ones on a removed image"""

    for kind, pattern in [("guest", r"opends-guest-[0-9a-f]{12}"),
                          ("deps", r"opends-deps-([0-9a-f]{12})-[0-9a-f]{12}")]:
        found = []
        for path in DISK.glob(f"opends-{kind}-*.qcow2"):
            match = re.fullmatch(pattern, path.stem)
            if not match:
                continue
            orphan = (kind == "deps"
                      and not (DISK / f"opends-guest-{match.group(1)}.qcow2").exists())
            found.append((orphan, -path.stat().st_mtime, path))
        for i, (orphan, _, path) in enumerate(sorted(found)):
            if path.stem not in in_use and (orphan or i >= KEEP[kind]):
                print(f"removing {path.name}", flush=True)
                path.unlink()


def up(_args):
    if not RIG.exists():
        die(f"{RIG} is missing; it gives the devices' addresses on the host")
    env = {"OPENDS_GUEST_PUBKEY": make_key(), "OPENDS_GUEST_BASE": guest_image_stem()}
    env["OPENDS_GUEST_DEPS"] = deps_stem(env["OPENDS_GUEST_BASE"])
    guest = DISK / f"{env['OPENDS_GUEST_BASE']}.qcow2"
    deps = DISK / f"{env['OPENDS_GUEST_DEPS']}.qcow2"

    if not guest.exists():
        print(f"\n=== no {guest.name}; deriving the guest image ===", flush=True)
        cijoe("tasks/guest_derive.yaml", env=env)
    if not deps.exists():
        print(f"\n=== no {deps.name}; building xnvme and xal ===", flush=True)
        cijoe("tasks/guest_up.yaml", out="cijoe-output-guest_up-deps", env=env)
        script("rsync.py")
        script("setup_deps.py", "xnvme", "xal")
        cijoe("tasks/guest_save.yaml", env=env)
    cijoe("tasks/guest_up.yaml", env={**env, "OPENDS_GUEST_IMAGE": "opends-deps-x86_64"})

    for path in [guest, deps]:
        os.utime(path)
    prune({guest.stem, deps.stem})
    ok(f"guest up on {deps.name}")


def down(_args):
    try:
        cijoe("tasks/guest_down.yaml")
    finally:
        serial = Path(os.environ["HOME"]) / "guests" / "opends-ci" / "serial.output"
        out = ROOT / "cijoe-output-guest_down"
        if serial.exists() and out.is_dir():
            shutil.copy(serial, out / "serial.output")
    ok("guest down")


def keys(_args):
    stem = guest_image_stem()
    print(stem)
    print(deps_stem(stem))


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("up", help="boot the guest and make it the target").set_defaults(func=up)
sub.add_parser("down", help="kill the guest").set_defaults(func=down)
sub.add_parser("keys", help="print the names of the two images").set_defaults(func=keys)
args = parser.parse_args()
args.func(args)
