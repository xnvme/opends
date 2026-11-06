#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
patterns = ("src/*.c", "src/*.h", "src/*.cu", "include/*.h", "tests/*.c", "tests/*.h")
sources = sorted(p for pat in patterns for p in root.glob(pat))

for src in sources:
    subprocess.run(["clang-format", "-i", str(src)])
