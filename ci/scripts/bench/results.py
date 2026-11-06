#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Bundle a bench run into one results document for a collector.

Gathers every history.jsonl under the bench output dir, the raw filperf
logs the records name and the rendered report.md into results/results.json,
with the run's provenance once at the top and each record reduced to what
differs between records, and copies report.md beside it. A collector then
fetches one artifact and keeps two files.

The environment is the first leg's sample: the machine is leased to one user
for the run, so later legs see the same one. The code and the run identity
must match across legs.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import ROOT, fail, iter_history, ok

SHARED = ("opends", "deps", "env", "ci")
IDENTITY = ("opends", "deps", "ci")
PER_RECORD = ("run_timestamp", "config", "result", "raw_log", "error")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir", nargs="?", default="cijoe-output-bench",
                    help="Bench output dir (default: cijoe-output-bench)")
    ap.add_argument("--job", default=os.environ.get("OPENDS_CI_JOB", ""),
                    help="Job name recorded in the document "
                         "(default: $OPENDS_CI_JOB)")
    ap.add_argument("--out", default="results",
                    help="Output dir for results.json and report.md "
                         "(default: results)")
    a = ap.parse_args()
    in_dir = Path(a.in_dir) if Path(a.in_dir).is_absolute() else ROOT / a.in_dir
    out_dir = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out

    shared, records, logs, schema = None, [], {}, None
    for base, path in iter_history([in_dir]):
        leg = str(path.parent.parent.relative_to(base))
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if shared is None:
                shared = {k: rec.get(k) for k in SHARED}
                schema = rec.get("schema_version")
            elif any(rec.get(k) != shared[k] for k in IDENTITY):
                fail(f"{path}: code or run identity differs from the first "
                     "record")
                sys.exit(1)
            entry = {"leg": leg}
            entry.update({k: rec[k] for k in PER_RECORD if k in rec})
            records.append(entry)
            log = path.parent / rec.get("raw_log", "")
            if rec.get("raw_log") and log.is_file():
                logs[f"{leg}/{rec['raw_log']}"] = log.read_text()
    if not records:
        fail(f"no records under {in_dir}")
        sys.exit(1)

    report = in_dir / "report.md"
    doc = {
        "schema_version": 1,
        "job": a.job,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records_schema_version": schema,
        **shared,
        "records": records,
        "logs": logs,
        "report": report.read_text() if report.is_file() else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(doc, indent=1) + "\n")
    if report.is_file():
        shutil.copyfile(report, out_dir / "report.md")
    ok(f"results: {len(records)} records, {len(logs)} logs -> {out_dir}")


if __name__ == "__main__":
    main()
