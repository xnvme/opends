#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Sweep the filperf-driven bench suites on the target.

Every suite sweeps its own knob grid into cijoe-output-bench/<suite>/.
gds has no knobs: its sweep is the singleton, one run of the whole suite.
opends sweeps io_threads x queue_depth x assume_aligned_only x idle_spin
(--io-threads, --queue-depth, --assume-aligned-only, --idle-spin): the
HOMI/qublk stack comes up once, each grid point runs the bench steps into
opends/t<t>_q<q>[_aligned][_spin<v>][_busy]/ with the aisio knobs passed
through the environment, and the stack is torn down. --busy-spin applies
OPENDS_AISIO_BUSY_SPIN=1 to the whole invocation. Both suites set the CPU
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import ok, run_cijoe

SUITES = {
    "gds": "tasks/bench_gds.yaml",
    "opends": "tasks/bench_opends.yaml",
}

DATASETS = ["filesize8gib", "tiktokish", "imagenetish", "lmcacheish"]

GDS_SETUP = ["cpu_governor", "load_nvidia_fs", "meta", "bind_nvme", "mount"]


def _bench_steps(backend, mode, datasets):
    modes = ["sync", "async"] if mode == "both" else [mode]
    return [f"{backend}_{ds}" + ("_async" if m == "async" else "")
            for ds in datasets for m in modes]


def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _spin_list(s):
    """Comma list of idle-spin values: "default" keeps the library default
    (the env var stays unset), an integer is microseconds (0 naps
    immediately, the pre-idle-spin behavior)."""
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(None if x == "default" else int(x))
    return out


def _run_gds(args, datasets):
    """Singleton sweep: the gds suite has no knobs to grid over."""
    if args.mode == "both" and set(datasets) == set(DATASETS):
        steps = []
    else:
        steps = GDS_SETUP + _bench_steps("gds", args.mode, datasets)
    print(f"\n=== gds suite: "
          f"{', '.join(steps) if steps else 'all steps'} ===", flush=True)
    try:
        run_cijoe(SUITES["gds"], *steps, out=f"{args.out}/gds")
    finally:
        run_cijoe(SUITES["gds"], "cpu_governor_restore",
                  out=f"{args.out}/gds/_teardown")


def _run_opends(args, datasets):
    suite = SUITES["opends"]
    steps = _bench_steps("opends", args.mode, datasets)
    out = f"{args.out}/opends"
    if args.busy_spin:
        os.environ["OPENDS_AISIO_BUSY_SPIN"] = "1"
    else:
        os.environ.pop("OPENDS_AISIO_BUSY_SPIN", None)
    if not args.skip_setup:
        run_cijoe(suite, "cpu_governor", "homi_stack_up", out=f"{out}/_setup")
    try:
        for s in args.idle_spin:
            if s is None:
                os.environ.pop("OPENDS_AISIO_IDLE_SPIN_US", None)
            else:
                os.environ["OPENDS_AISIO_IDLE_SPIN_US"] = str(s)
            for a in args.assume_aligned_only:
                for t in args.io_threads:
                    for q in args.queue_depth:
                        os.environ["OPENDS_AISIO_IO_THREADS"] = str(t)
                        os.environ["OPENDS_AISIO_QUEUE_DEPTH"] = str(q)
                        os.environ["OPENDS_AISIO_ASSUME_ALIGNED_ONLY"] = str(a)
                        leg = (f"t{t}_q{q}" + ("_aligned" if a else "")
                               + (f"_spin{s}" if s is not None else "")
                               + ("_busy" if args.busy_spin else ""))
                        print(f"\n=== opends io_threads={t} queue_depth={q} "
                              f"assume_aligned_only={a}"
                              + (f" idle_spin_us={s}" if s is not None else "")
                              + (" busy_spin=1" if args.busy_spin else "")
                              + f": {', '.join(steps)} ===", flush=True)
                        rc = run_cijoe(suite, "meta", *steps,
                                       out=f"{out}/{leg}", check=False)
                        if rc:
                            print(f"leg {leg} failed (rc={rc}); continuing",
                                  flush=True)
    finally:
        if not args.keep_stack:
            run_cijoe(suite, "homi_stack_down", "cpu_governor_restore",
                      out=f"{out}/_teardown")


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--suite", action="append", choices=list(SUITES),
                    help="Run only this suite. Repeat to combine. Default: "
                         "all.")
parser.add_argument("--mode", choices=["sync", "async", "both"], default="both")
parser.add_argument("--dataset", action="append", choices=DATASETS,
                    help="Run only this dataset. Repeat to combine.")
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
                         "<out>/<suite>/. Default cijoe-output-bench. Use a "
                         "distinct dir to add legs without overwriting prior "
                         "ones (cijoe wipes a reused leg dir).")

grid = parser.add_argument_group("opends sweep grid")
grid.add_argument("--io-threads", type=_int_list, default=[1, 2, 4, 8],
                  help="Comma-separated IO-thread counts. Default 1,2,4,8.")
grid.add_argument("--queue-depth", type=_int_list,
                  default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                  help="Comma-separated NVMe queue depths. Default 1..512 "
                       "(>512 may exceed the device MQES and fail).")
grid.add_argument("--assume-aligned-only", type=_int_list, default=[0, 1],
                  metavar="LIST",
                  help="Comma-separated OPENDS_AISIO_ASSUME_ALIGNED_ONLY "
                       "values. Default 0,1 (both), which doubles the grid; "
                       "pass 0 for the tail-fixup sweep alone. An aligned leg "
                       "writes to t<t>_q<q>_aligned/, so 0-legs keep their "
                       "paths.")
grid.add_argument("--idle-spin", type=_spin_list, default=[None],
                  metavar="LIST",
                  help="Comma-separated OPENDS_AISIO_IDLE_SPIN_US values in "
                       "microseconds; the token 'default' keeps the library "
                       "default (env unset), 0 naps immediately (the "
                       "pre-idle-spin behavior). E.g. 'default,0' doubles the "
                       "grid into an on/off comparison. An explicit value "
                       "writes to t<t>_q<q>_spin<v>/, so default legs keep "
                       "their paths. Default: 'default'.")
grid.add_argument("--busy-spin", action="store_true",
                  help="Set OPENDS_AISIO_BUSY_SPIN=1 for every leg of this "
                       "invocation (workers poll flat out; pair with "
                       "--cpu-mask). Legs write to t<t>_q<q>[...]_busy/.")
grid.add_argument("--skip-setup", action="store_true",
                  help="Assume the HOMI stack is already up.")
grid.add_argument("--keep-stack", action="store_true",
                  help="Leave the HOMI stack up after the sweep.")
args = parser.parse_args()

datasets = args.dataset or DATASETS
if args.batches:
    for spec in args.batches:
        ds, _, n = spec.partition("=")
        if not ds or not n.isdigit() or int(n) == 0:
            parser.error("--batches expects <dataset>=<positive integer>, "
                         f"got {spec!r}")
    os.environ["OPENDS_BENCH_BATCHES"] = ",".join(args.batches)
if args.cpu_mask:
    os.environ["OPENDS_AISIO_CPU_MASK"] = args.cpu_mask

for name in args.suite or list(SUITES):
    if name == "gds":
        _run_gds(args, datasets)
    else:
        _run_opends(args, datasets)
ok("run_bench")
