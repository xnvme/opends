#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Render the LMCache KV cache records into report.md.

Reads the lmcache records under the bench output dir and writes one row per
GDS backend with the store and retrieve latencies LMCache reports.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from _helpers import ROOT, fail, iter_history, ok

LEGS = ("opends", "cufile")


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
            if rec.get("config", {}).get("backend") == "lmcache":
                records[rec["config"]["mode"]] = rec
    if not records:
        fail(f"no lmcache records under {in_dir}")
        sys.exit(1)

    cfg = next(iter(records.values()))["config"]
    md = ["## KV cache with LMCache", "",
          f"`lmcache bench server`, LMCache `{cfg['lmcache_sha'][:7]}`, "
          f"{cfg['n']} requests of {cfg['num_tokens']} tokens per backend. "
          "A store writes the KV cache of a request from GPU memory to the "
          "slab, a retrieve reads it back.", "",
          "| `--gds-l1-backend` | Store p50 (ms) | Store p99 (ms) | "
          "Retrieve p50 (ms) | Retrieve p99 (ms) |",
          "|---|---|---|---|---|"]
    for leg in LEGS:
        if leg in records:
            res = records[leg]["result"]
            md.append(f"| {leg} | {res['store_ms']['p50']:.1f} | "
                      f"{res['store_ms']['p99']:.1f} | "
                      f"{res['retrieve_ms']['p50']:.1f} | "
                      f"{res['retrieve_ms']['p99']:.1f} |")
    (in_dir / "report.md").write_text("\n".join(md) + "\n")
    ok(f"report: {len(records)} backends -> {in_dir / 'report.md'}")


if __name__ == "__main__":
    main()
