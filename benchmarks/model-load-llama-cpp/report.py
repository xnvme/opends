#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Render the model loading records into report.md.

Reads the llama.cpp records under the bench output dir and writes one row per
load mode with the range of the process's wall time and of llama.cpp's load
time over the runs.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from _helpers import ROOT, fail, iter_history, ok

MODES = ("opends", "dio", "none", "mmap")


def _range(values):
    lo, hi = min(values), max(values)
    return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f} to {hi:.2f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", nargs="?", default="cijoe-output-bench",
                    help="Bench output dir (default: cijoe-output-bench); "
                         "report.md is written into it")
    a = ap.parse_args()
    in_dir = Path(a.in_dir) if Path(a.in_dir).is_absolute() else ROOT / a.in_dir

    records = {}
    for _, path in iter_history([in_dir]):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("config", {}).get("backend") == "llama.cpp":
                records[rec["config"]["mode"]] = rec
    if not records:
        fail(f"no llama.cpp records under {in_dir}")
        sys.exit(1)

    cfg = next(iter(records.values()))["config"]
    md = ["## Model loading with llama.cpp", "",
          f"{cfg['dataset']}, llama.cpp `{cfg['llama_cpp_sha'][:7]}`, "
          f"{cfg['n']} runs per mode, each a new process from a cold page "
          "cache.", "",
          "| `--load-mode` | Whole process (s) | Model load (s) |",
          "|---|---|---|"]
    for mode in MODES:
        if mode in records:
            res = records[mode]["result"]
            md.append(f"| {mode} | {_range(res['real_s'])} | "
                      f"{_range(res['load_s'])} |")
    (in_dir / "report.md").write_text("\n".join(md) + "\n")
    ok(f"report: {len(records)} modes -> {in_dir / 'report.md'}")


if __name__ == "__main__":
    main()
