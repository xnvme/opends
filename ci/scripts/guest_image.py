#!/usr/bin/env python3
"""Build the guest image by hand: run tasks/guest_derive.yaml in the xnvme-devenv host container.

The container runs as root, so what it writes to the image cache is root's.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from _helpers import REPO, guest_image_stem

IMAGE = "ghcr.io/xnvme/xnvme-devenv/host:v0.15.0"

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--image", default=os.environ.get("OPENDS_DEVENV_IMAGE", IMAGE),
                    help="the host container image (default: %(default)s)")
parser.add_argument("--engine", default="docker", help="docker or podman")
parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "xnvme-devenv",
                    help="the image cache, mounted as the container's system_imaging")
parser.add_argument("--home", type=Path, default=Path.home() / ".cache" / "opends-guest",
                    help="the guests and the cijoe output, mounted as the container's home")
args = parser.parse_args()

args.cache.mkdir(parents=True, exist_ok=True)
(args.home / "guests").mkdir(parents=True, exist_ok=True)

cijoe = ("export PATH=$HOME/.local/bin:$PATH && "
         "cijoe -m -s -o /root/guests/cijoe-output-guest_derive "
         "-c configs/guest.toml -c configs/guest-lock.toml tasks/guest_derive.yaml")
cmd = [
    args.engine, "run", "--rm", "--device", "/dev/kvm",
    "-e", f"OPENDS_GUEST_BASE={guest_image_stem()}",
    "-v", f"{REPO}:/tmp/opends:ro",
    "-v", f"{args.cache}:/root/system_imaging",
    "-v", f"{args.home / 'guests'}:/root/guests",
    "-w", "/tmp/opends/ci",
    args.image, "bash", "-euo", "pipefail", "-c", cijoe,
]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd).returncode)
