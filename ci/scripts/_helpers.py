# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

# ROOT is ci/, where cijoe runs and its configs, tasks and output live
ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
CONFIGS = ["-c", "configs/transport.toml", "-c", "configs/deps.toml",
           "-c", "configs/test.toml"]
# What the guest image is made from, beside the derive guest in guest.toml
GUEST_RECIPE = ["configs/guest-lock.toml", "tasks/guest_derive.yaml",
                "tasks/steps/guest_derive.sh", "cijoe-scripts/guest_derive.py",
                "cijoe-scripts/guest_grow.py", "cijoe-scripts/guest_export.py"]


def guest_image_stem():
    """The name of the guest image this checkout derives: opends-guest-<key of the lock and the recipe>"""

    with open(ROOT / "configs" / "guest.toml", "rb") as f:
        guest = tomllib.load(f)
    h = hashlib.sha256()
    for path in GUEST_RECIPE:
        h.update((ROOT / path).read_bytes())
    for part in [guest["qemu"]["guests"]["derive"], guest["guest"],
                 guest["system-imaging"]["images"]["debian-13-x86_64"]]:
        h.update(repr(part).encode())
    return f"opends-guest-{h.hexdigest()[:12]}"


def iter_history(in_dirs):
    """Yield (base_dir, history.jsonl path) under each dir, live legs only.

    Skips cijoe-archive/ backups so stale snapshots don't double-count.
    """
    for d in in_dirs:
        d = Path(d)
        for p in sorted(d.rglob("history.jsonl")):
            if "cijoe-archive" not in p.parts:
                yield d, p


def fail(msg):
    """Print a bold-red FAILED banner to stderr.

    ANSI escapes are gated on stderr.isatty() so redirected logs stay clean.
    """
    banner = f"FAILED: {msg}"
    if sys.stderr.isatty():
        sys.stderr.write(f"\n\033[1;31m{banner}\033[0m\n")
    else:
        sys.stderr.write(f"\n{banner}\n")


def ok(msg):
    """Print a bold-green OK banner to stdout.

    ANSI escapes are gated on stdout.isatty() so redirected logs stay clean.
    """
    banner = f"OK: {msg}"
    if sys.stdout.isatty():
        sys.stdout.write(f"\n\033[1;32m{banner}\033[0m\n")
    else:
        sys.stdout.write(f"\n{banner}\n")


def run_cijoe(task, *steps, out=None, check=True):
    """Run a cijoe task with the standard configs.

    Output dir defaults to cijoe-output-<task-stem>. Returns the rc. When
    check is true, a nonzero rc prints a FAILED banner and exits; pass
    check=False to keep going (e.g. so a sweep still runs teardown).
    """
    out = out or f"cijoe-output-{Path(task).stem}"
    rc = subprocess.run(
        ["cijoe", "-m", "-s", "-o", out, *CONFIGS, task, *steps],
        cwd=ROOT,
    ).returncode
    if rc and check:
        fail(f"{task} (rc={rc})")
        sys.exit(rc)
    return rc
