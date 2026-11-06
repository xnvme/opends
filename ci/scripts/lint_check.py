#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
patterns = ("src/*.c", "src/*.h", "src/*.cu", "include/*.h", "tests/*.c", "tests/*.h")
sources = sorted(p for pat in patterns for p in root.glob(pat))

rc = 0
for src in sources:
    result = subprocess.run(
        ["clang-format", "--dry-run", "--Werror", str(src)],
        capture_output=True, text=True,
    )
    if result.returncode:
        print(result.stderr, end="")
        rc = 1

sys.exit(rc)
