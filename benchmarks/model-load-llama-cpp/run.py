#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run the llama.cpp model loading benchmark on the target.

Builds llama.cpp at the commit in deps.toml beside this file, puts the model
on the test drive, loads it in every load mode and writes the records to
cijoe-output-bench/model-load-llama-cpp/. The model's copy on the drive is
removed and the stack brought down afterwards, also when a step fails.
report.py beside this file renders the table.

A record holds each run's wall time and the load time llama.cpp prints, in
the shape cijoe-scripts/bench_run.py writes, so scripts/bench/results.py
bundles it.
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
OUT = "cijoe-output-bench/model-load-llama-cpp"

STEPS = ["check_stack_down", "meta", "build", "fetch_model", "bind_nvme",
         "mount", "stage_model", "gpu_persistence", "load_dio", "load_none",
         "load_mmap", "homi_stack_up", "load_opends"]
TEARDOWN = ["unstage_model", "homi_stack_down", "gpu_persistence_restore"]
MODES = ["dio", "none", "mmap", "opends"]
ENV_KEYS = ("hostname", "kernel", "kernel_cmdline", "cpu_model",
            "cpu_governor", "cuda", "nvme_bdf", "nvme_model", "nvme_link",
            "fs", "gpu_model", "hugepages_2m")


def cijoe(out, steps):
    return subprocess.run(
        ["cijoe", "-m", "-s", "-o", out, *CONFIGS, "-c", DEPS, TASK, *steps],
        cwd=ROOT,
    ).returncode


def write_records(out):
    """Turn the output of the load steps into history.jsonl records."""
    artifacts = out / "artifacts"
    meta = json.loads((artifacts / "meta.json").read_text())
    with open(HERE / "deps.toml", "rb") as f:
        pins = tomllib.load(f)["llama_cpp"]
    for mode in MODES:
        text = "".join(p.read_text(errors="replace") for step in
                       sorted(out.glob(f"[0-9]*_load_{mode}"))
                       for p in sorted(step.glob("cmd_*.output")))
        real = re.findall(r"^real_ms=(\d+)$", text, re.MULTILINE)
        load = re.findall(r"load time =\s*([0-9.]+) ms", text)
        if not real or len(real) != len(load):
            continue
        persistence = re.search(r"^gpu_persistence_mode=(.*)$", text,
                                re.MULTILINE)
        raw = artifacts / f"llama_cpp_{mode}.log"
        raw.write_text(text)
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
                "backend": "llama.cpp",
                "dataset": pins["model"]["name"],
                "mode": mode,
                "cold_cache": True,
                "n": len(real),
                "llama_cpp_sha": pins["repository"]["ref"],
                "model_sha256": pins["model"]["sha256"],
                "gpu_persistence_mode": persistence.group(1)
                if persistence else None,
            },
            "result": {"real_s": [int(x) / 1000 for x in real],
                       "load_s": [round(float(x) / 1000, 3) for x in load]},
            "raw_log": raw.name,
        }
        with (artifacts / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")


rc = cijoe(OUT, STEPS)
# No meta.json: check_stack_down failed, and a stack that was up before the
# run is not this run's to take down.
if (ROOT / OUT / "artifacts" / "meta.json").is_file():
    write_records(ROOT / OUT)
    cijoe(f"{OUT}/_teardown", TEARDOWN)
if rc:
    fail(f"{TASK} (rc={rc})")
    sys.exit(rc)
ok("model-load-llama-cpp")
