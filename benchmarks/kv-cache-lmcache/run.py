#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run the LMCache KV cache benchmark on the target.

Installs LMCache at the commit in deps.toml beside this file, runs
`lmcache bench server` against a GDS slab on the test drive with the cufile
and the opends backend and writes the records to
cijoe-output-bench/kv-cache-lmcache/. The slab is removed and the stack
brought down afterwards, also when a step fails. report.py beside this file
renders the table.

A record holds the latency summaries LMCache prints, in the shape
cijoe-scripts/bench_run.py writes, so scripts/bench/results.py bundles it.
"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from _helpers import CONFIGS, ROOT, fail, ok

TASK = str(HERE.relative_to(ROOT) / "bench.yaml")
DEPS = str(HERE.relative_to(ROOT) / "deps.toml")
OUT = "cijoe-output-bench/kv-cache-lmcache"

STEPS = ["check_stack_down", "cpu_governor", "load_nvidia_fs", "meta",
         "install", "bind_nvme", "mount", "bench_cufile", "slab",
         "homi_stack_up", "bench_opends"]
TEARDOWN = ["remove_slab", "homi_stack_down", "cpu_governor_restore"]
LEGS = ["cufile", "opends"]
ENV_KEYS = ("hostname", "kernel", "kernel_cmdline", "cpu_model",
            "cpu_governor", "cuda", "nvme_bdf", "nvme_model", "nvme_link",
            "fs", "gpu_model", "hugepages_2m")


def cijoe(out, steps):
    return subprocess.run(
        ["cijoe", "-m", "-s", "-o", out, *CONFIGS, "-c", DEPS, TASK, *steps],
        cwd=ROOT,
    ).returncode


def stats(summary, section):
    """One latency section of LMCache's summary, keys without the prefix."""
    return {k.removeprefix(f"{section}_"): v
            for k, v in summary[section].items()}


def write_records(out):
    """Turn the output of the bench steps into history.jsonl records.

    Returns the legs that have no record or read back wrong data.
    """
    artifacts = out / "artifacts"
    meta = json.loads((artifacts / "meta.json").read_text())
    with open(HERE / "deps.toml", "rb") as f:
        pins = tomllib.load(f)["lmcache"]
    bad = []
    for leg in LEGS:
        text = "".join(p.read_text(errors="replace") for step in
                       sorted(out.glob(f"[0-9]*_bench_{leg}"))
                       for p in sorted(step.glob("cmd_*.output")))
        found = re.search(r"^--- result ---$(.*?)^--- server log ---$", text,
                          re.MULTILINE | re.DOTALL)
        try:
            summary = json.loads(found.group(1))["metrics"]
            store = stats(summary, "cold_store")
            retrieve = stats(summary, "warm_retrieve")
        except (AttributeError, ValueError, KeyError):
            bad.append(leg)
            continue
        raw = artifacts / f"lmcache_{leg}.log"
        raw.write_text(text)
        if summary["results"]["checksum_fail"]:
            bad.append(leg)
        record = {
            "schema_version": 3,
            "run_id": f"{meta['timestamp']}-{meta['hostname']}",
            "run_timestamp": meta["timestamp"],
            "opends": {"sha": meta.get("commit_sha", ""),
                       "dirty": meta.get("dirty", False)},
            "deps": meta.get("deps", {}),
            "env": {k: meta.get(k) for k in ENV_KEYS},
            "ci": meta.get("ci"),
            "config": {
                "backend": "lmcache",
                "dataset": "bench-server",
                "mode": leg,
                "n": summary["results"]["total_requests"],
                "num_tokens": summary["config"]["num_tokens"],
                "lmcache_sha": pins["repository"]["ref"],
                "torch": pins["torch"]["version"],
            },
            "result": {
                "checksum_fail": summary["results"]["checksum_fail"],
                "store_ms": store,
                "retrieve_ms": retrieve,
            },
            "raw_log": raw.name,
        }
        with (artifacts / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    return bad


def main():
    rc = cijoe(OUT, STEPS)
    bad = []
    # No meta.json: check_stack_down failed, and a stack that was up before
    # the run is not this run's to take down.
    if (ROOT / OUT / "artifacts" / "meta.json").is_file():
        bad = write_records(ROOT / OUT)
        cijoe(f"{OUT}/_teardown", TEARDOWN)
    if rc:
        fail(f"{TASK} (rc={rc})")
        sys.exit(rc)
    if bad:
        fail(f"{TASK}: no result or a wrong checksum for {', '.join(bad)}")
        sys.exit(1)
    ok("kv-cache-lmcache")


if __name__ == "__main__":
    main()
