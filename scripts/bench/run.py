#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Sweep the filperf-driven bench suites on the target.

Every suite sweeps its own knob grid into cijoe-output-bench/<suite>/.
gds has no knobs: its sweep is the singleton, one run of the whole suite.
opends runs the config list in scripts/bench/sweep.toml. --full-sweep runs the
whole io_threads x queue_depth x assume_aligned_only x idle_spin cross product
instead, --sweep-file swaps in another list, and the knob flags sweep the
values they name, each knob they leave out at its aisio default.

Either way the HOMI/qublk stack comes up once, each config runs the bench steps
into opends/t<t>_q<q>[_aligned][_spin<v>][_busy]/ with the aisio knobs passed
through the environment, and the stack is torn down. Both suites set the CPU
governor for the run and restore it in teardown. Restrict with --suite,
--mode, --dataset.

An assume_aligned_only leg rejects any read with a sub-LBA tail, so datasets
whose files are not LBA-multiples fail by construction. Those legs are recorded
with an empty result and the sweep carries on to the datasets that do qualify.

aisio's tasks/setup_dataset.yaml must have populated the reference datasets
(filesize8gib, tiktokish, imagenetish) under config.test.mount_point, and
scripts/bench/setup_dataset.py must have populated lmcacheish (both one-time).
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import fail, ok, run_cijoe

SUITES = {
    "gds": "tasks/bench_gds.yaml",
    "opends": "tasks/bench_opends.yaml",
}

DATASETS = ["filesize8gib", "tiktokish", "imagenetish", "lmcacheish"]

MODES = ["sync", "stream", "async"]
# fil implements --async for the opends backend only
BACKEND_MODES = {"gds": ["sync", "stream"]}

DEFAULT_CONFIGS = Path(__file__).resolve().parent / "sweep.toml"

GRID_AXES = {"io_threads": [1, 2, 4, 8],
             "queue_depth": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
             "assume_aligned_only": [0, 1],
             "idle_spin": [0, 200, "busy"]}

# The aisio defaults. A knob the flags leave out sweeps this one value, so
# --io-threads 2 is one leg, not a plane.
KNOB_DEFAULTS = {"io_threads": 2, "queue_depth": 8,
                 "assume_aligned_only": 0, "idle_spin": 200}

GDS_SETUP = ["cpu_governor", "load_nvidia_fs", "meta", "bind_nvme", "mount"]

def _bench_steps(backend, modes, datasets):
    avail = BACKEND_MODES.get(backend, MODES)
    return [f"{backend}_{ds}" + ("" if m == "sync" else f"_{m}")
            for ds in datasets for m in modes if m in avail]


def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _spin_list(s):
    """Comma list of idle-spin values: microseconds, or the token "busy",
    where a worker never naps at all."""
    try:
        return [x.strip() if x.strip() == "busy" else int(x)
                for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expects a comma list of microseconds, or \"busy\"")


def _mode_list(s):
    modes = [x.strip() for x in s.split(",") if x.strip()]
    unknown = set(modes) - set(MODES)
    if not modes or unknown:
        raise argparse.ArgumentTypeError(
            "expects a comma list of " + ", ".join(MODES))
    return modes


def _run_gds(args, modes, datasets):
    """Singleton sweep: the gds suite has no knobs to grid over."""
    bench = _bench_steps("gds", modes, datasets)
    if not bench:
        print(f"\n=== gds suite: nothing to run for --mode "
              f"{','.join(modes)} ===", flush=True)
        return
    if set(modes) == set(MODES) and set(datasets) == set(DATASETS):
        steps = []
    else:
        steps = GDS_SETUP + bench
    print(f"\n=== gds suite: "
          f"{', '.join(steps) if steps else 'all steps'} ===", flush=True)
    try:
        run_cijoe(SUITES["gds"], *steps, out=f"{args.out}/gds")
    finally:
        run_cijoe(SUITES["gds"], "cpu_governor_restore",
                  out=f"{args.out}/gds/_teardown")


def _axis_configs(axes, modes, datasets):
    """The cross product of the axes, in sweep order."""
    for s in axes["idle_spin"]:
        for a in axes["assume_aligned_only"]:
            for t in axes["io_threads"]:
                for q in axes["queue_depth"]:
                    yield {"io_threads": t, "queue_depth": q,
                           "assume_aligned_only": a, "idle_spin": s,
                           "datasets": datasets, "mode": modes}


def _leg_name(p):
    """The dir a config runs into."""
    if p.get("name"):
        return p["name"]
    s = p["idle_spin"]
    return (f"t{p['io_threads']}_q{p['queue_depth']}"
            + ("_aligned" if p["assume_aligned_only"] else "")
            + ("_busy" if s == "busy" else f"_spin{s}" if s != 200 else ""))


CONFIG_KEYS = {"io_threads", "queue_depth", "datasets", "mode",
               "assume_aligned_only", "idle_spin", "name"}


def _file_configs(path, fail):
    """Read a config list, e.g. scripts/bench/sweep.toml."""
    try:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"{path}: {exc}")
    configs = doc.get("config")
    if not configs:
        fail(f"{path}: no [[config]] entries")
    seen = {}
    for i, p in enumerate(configs, 1):
        where = f"{path}: config {i}"
        unknown = set(p) - CONFIG_KEYS
        if unknown:
            fail(f"{where} has unknown keys {', '.join(sorted(unknown))}")
        for key in ("io_threads", "queue_depth", "assume_aligned_only"):
            v = p.get(key)
            # a bool passes isinstance(int), and reaches aisio as "True"
            if isinstance(v, bool) or not isinstance(v, int):
                fail(f"{where} needs an integer {key}")
        spin = p.get("idle_spin")
        if isinstance(spin, bool) or (not isinstance(spin, int)
                                      and spin != "busy"):
            fail(f"{where} needs an integer idle_spin, or \"busy\"")
        if not isinstance(p.get("datasets"), list) or not p["datasets"]:
            fail(f"{where} needs a datasets list")
        unknown = set(p["datasets"]) - set(DATASETS)
        if unknown:
            fail(f"{where} names unknown datasets "
                 f"{', '.join(sorted(unknown))}")
        modes = p.get("mode")
        if not isinstance(modes, list) or not modes:
            fail(f"{where} needs a mode list")
        unknown = set(modes) - set(MODES)
        if unknown:
            fail(f"{where} names unknown modes "
                 f"{', '.join(sorted(unknown))}")
        if "name" in p and not isinstance(p["name"], str):
            fail(f"{where} needs a string name")
        leg = _leg_name(p)
        if leg in seen:
            fail(f"{where} runs into {leg}, which config {seen[leg]} took")
        seen[leg] = i
    return configs


def _run_opends(args, modes, datasets, configs):
    """Run the configs; returns the legs cijoe reported a failure for."""
    failed = []
    suite = SUITES["opends"]
    out = f"{args.out}/opends"
    allowed = set(_bench_steps("opends", modes, datasets))
    plan = [(p, [x for x in _bench_steps("opends", p["mode"], p["datasets"])
                 if x in allowed]) for p in configs]
    if any(not steps for _, steps in plan):
        plan = [(p, steps) for p, steps in plan if steps]
        print(f"--dataset/--mode keeps {len(plan)} of {len(configs)} configs",
              flush=True)
    if not args.skip_setup:
        run_cijoe(suite, "cpu_governor", "homi_stack_up", out=f"{out}/_setup")
    try:
        for i, (p, steps) in enumerate(plan, 1):
            t, q = p["io_threads"], p["queue_depth"]
            a = p["assume_aligned_only"]
            s = p["idle_spin"]
            os.environ["OPENDS_AISIO_IDLE_SPIN"] = str(s)
            os.environ["OPENDS_AISIO_IO_THREADS"] = str(t)
            os.environ["OPENDS_AISIO_QUEUE_DEPTH"] = str(q)
            os.environ["OPENDS_AISIO_ASSUME_ALIGNED_ONLY"] = str(a)
            leg = _leg_name(p)
            print(f"\n=== opends [{i}/{len(plan)}] io_threads={t} queue_depth={q} "
                  f"assume_aligned_only={a} idle_spin={s}"
                  f": {', '.join(steps)} ===", flush=True)
            rc = run_cijoe(suite, "meta", *steps, out=f"{out}/{leg}",
                           check=False)
            if rc:
                print(f"leg {leg} failed (rc={rc}); continuing", flush=True)
                failed.append(leg)
    finally:
        if not args.keep_stack:
            run_cijoe(suite, "homi_stack_down", "cpu_governor_restore",
                      out=f"{out}/_teardown")
    return failed


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--suite", action="append", choices=list(SUITES),
                    help="Run only this suite. Repeat to combine. Default: "
                         "all.")
parser.add_argument("--mode", type=_mode_list,
                    metavar="LIST",
                    help="Comma-separated modes to keep: sync, stream, "
                         "async. Default all three; the gds suite has no "
                         "async leg and skips it.")
parser.add_argument("--dataset", action="append", choices=DATASETS,
                    help="Narrow every config to this dataset, dropping the "
                         "configs that do not name it. Repeat to combine.")
parser.add_argument("--batches", action="append", metavar="DS=N",
                    help="Override batches for a dataset (e.g. tiktokish=50) "
                         "to lengthen its measurement window. Repeatable. "
                         "Forwarded via OPENDS_BENCH_BATCHES; batch-size, the "
                         "concurrency under study, is unchanged.")
parser.add_argument("--cpu-mask", metavar="MASK",
                    help="Pin aisio IO workers round-robin to the CPUs in "
                         "this hex mask (e.g. 0x3f = cores 0-5); 0x0 or "
                         "unset leaves placement to the scheduler. Forwarded "
                         "via OPENDS_AISIO_CPU_MASK; opends suite only.")
parser.add_argument("--out", default="cijoe-output-bench",
                    help="Output dir root; each suite writes under "
                         "<out>/<suite>/. Default cijoe-output-bench. A full "
                         "sweep and a config list share their leg names, so "
                         "give one of the two a distinct dir to keep both "
                         "(cijoe wipes a reused leg dir).")

grid = parser.add_argument_group("opends sweep configs")
which = grid.add_mutually_exclusive_group()
which.add_argument("--full-sweep", action="store_true",
                   help="Sweep the whole io_threads x queue_depth x "
                        "assume_aligned_only x idle_spin cross product "
                        "instead of the config list.")
which.add_argument("--sweep-file", metavar="FILE",
                   help="Run the configs listed in FILE (TOML) instead of the "
                        "default list, scripts/bench/sweep.toml. A config "
                        "states every aisio knob, and may narrow "
                        "datasets and mode, so a costly dataset runs only "
                        "where its own optimum is.")
grid.add_argument("--io-threads", type=_int_list, metavar="LIST",
                  help="Comma-separated OPENDS_AISIO_IO_THREADS values to "
                       "sweep instead of the config list. Any knob flag "
                       "switches to that mode, where the knobs left out sit "
                       "at their aisio default, 2 here. Datasets and mode "
                       "come from --dataset/--mode, and each leg writes to "
                       "the dir its knobs name, as a list entry would.")
grid.add_argument("--queue-depth", type=_int_list, metavar="LIST",
                  help="Comma-separated OPENDS_AISIO_QUEUE_DEPTH values. "
                       "Default 8; over 512 may exceed the device MQES and "
                       "fail.")
grid.add_argument("--assume-aligned-only", type=_int_list, metavar="LIST",
                  help="Comma-separated OPENDS_AISIO_ASSUME_ALIGNED_ONLY "
                       "values. Default 0. An aligned leg writes to "
                       "t<t>_q<q>_aligned/.")
grid.add_argument("--idle-spin", type=_spin_list, metavar="LIST",
                  help="Comma-separated OPENDS_AISIO_IDLE_SPIN values in "
                       "microseconds, or 'busy'. Default 200. 0 naps "
                       "immediately (the pre-idle-spin behavior), busy never "
                       "naps and burns a core, so pair it with --cpu-mask. "
                       "Anything but 200 writes to t<t>_q<q>_spin<v>/ or "
                       "_busy/.")
grid.add_argument("--skip-setup", action="store_true",
                  help="Assume the HOMI stack is already up.")
grid.add_argument("--keep-stack", action="store_true",
                  help="Leave the HOMI stack up after the sweep.")
args = parser.parse_args()

datasets = args.dataset or DATASETS
modes = args.mode or MODES
if args.batches:
    for spec in args.batches:
        ds, _, n = spec.partition("=")
        if not ds or not n.isdigit() or int(n) == 0:
            parser.error("--batches expects <dataset>=<positive integer>, "
                         f"got {spec!r}")
    os.environ["OPENDS_BENCH_BATCHES"] = ",".join(args.batches)
if args.cpu_mask:
    os.environ["OPENDS_AISIO_CPU_MASK"] = args.cpu_mask

knobs = {k: getattr(args, k) for k in KNOB_DEFAULTS
         if getattr(args, k) is not None}
if knobs and (args.full_sweep or args.sweep_file):
    other = "--full-sweep" if args.full_sweep else "--sweep-file"
    parser.error(f"the knob flags sweep the values they name, which {other} "
                 "replaces")
if knobs:
    axes = {k: knobs.get(k, [v]) for k, v in KNOB_DEFAULTS.items()}
    configs = list(_axis_configs(axes, modes, datasets))
elif args.full_sweep:
    configs = list(_axis_configs(GRID_AXES, modes, datasets))
else:
    configs = _file_configs(args.sweep_file or DEFAULT_CONFIGS, parser.error)

failed = []
for name in args.suite or list(SUITES):
    if name == "gds":
        _run_gds(args, modes, datasets)
    else:
        failed += _run_opends(args, modes, datasets, configs)

if failed:
    fail(f"run_bench: {len(failed)} of {len(configs)} legs failed: "
         f"{', '.join(failed)}")
    sys.exit(1)
ok("run_bench")
