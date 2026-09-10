#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Rsync the local working tree to the target.

After the rsync succeeds, the host's `git describe --always --dirty`
is stamped to `<repo_path>/.commit_stamp` on the target. bench_meta
reads that stamp to tag bench artifacts with the commit that was
actually synced (post-rsync host edits do not flip the stamp).
"""

import subprocess
import sys
import tomllib
from pathlib import Path

from _helpers import fail, ok

root = Path(__file__).resolve().parent.parent

with open(root / "configs" / "transport.toml", "rb") as f:
    transport = tomllib.load(f)

with open(root / "configs" / "test.toml", "rb") as f:
    test = tomllib.load(f)

ssh = transport["cijoe"]["transport"]["ssh"]
dest = test["test"]["repo_path"]
target = f"{ssh['username']}@{ssh['hostname']}"

rc = subprocess.run(
    ["rsync", "-az", "--delete", "--mkpath",
     "--filter=:- .gitignore",
     "--exclude=.git/",
     "--exclude=cijoe-output*/",
     "--exclude=cijoe-archive/",
     f"{root}/", f"{target}:{dest}/"],
).returncode

if rc != 0:
    fail(f"rsync (rc={rc})")
    sys.exit(rc)

desc = subprocess.check_output(
    ["git", "describe", "--always", "--dirty", "--abbrev=7"],
    cwd=root, text=True,
).strip()
rc = subprocess.run(
    ["ssh", target, f"printf '%s\\n' '{desc}' > '{dest}/.commit_stamp'"],
).returncode
if rc != 0:
    fail(f"commit stamp (rc={rc})")
    sys.exit(rc)

ok(f"synced to {target}:{dest} @ {desc}")
