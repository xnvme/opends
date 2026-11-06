#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build OpenDS on the target.

Assumes the tree has been synced to the target (rsync.py).
"""

from _helpers import ok, run_cijoe

run_cijoe("tasks/setup_opends.yaml", out="cijoe-output-build")
ok("build")
