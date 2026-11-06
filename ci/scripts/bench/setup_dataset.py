#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Populate the lmcacheish reference dataset on the target (one-time).

The other three reference datasets (filesize8gib, tiktokish, imagenetish)
come from aisio's tasks/setup_dataset.yaml during target provisioning;
lmcacheish is OpenDS's own and is generated here. Binds the kernel nvme
driver, mounts the dataset XFS, and runs the idempotent generation step,
so re-running is a cheap no-op. Assumes the tree has been synced to the
target (rsync.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import ok, run_cijoe

run_cijoe("tasks/setup_dataset.yaml")
ok("setup_dataset")
