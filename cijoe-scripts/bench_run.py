# SPDX-License-Identifier: BSD-3-Clause
"""Run a single filperf bench and stash its stdout as an artifact.

The artifact is the verbatim filperf output (CSV plus `--summary`
block) written to
artifacts/<backend>_<data_dir>[_async].log (the `_async` suffix is
only appended when `mode: async`). Parsing is deferred to
scripts/bench/report.py so the cijoe step stays oblivious to
filperf's exact output shape.

Each bench run runs filperf under `prlimit --nofile`. nofile is
required for opends and harmless elsewhere. gds runs get a cold-cache `drop_caches`; opends
reads files on the HOMI/qublk mount via DMA into GPU memory, so the
kernel page cache is irrelevant and the device's BDF/socket are
passed through OPENDS_HOMI_DEV/OPENDS_HOMI_SOCKET.
"""

import json
import logging as log
import os
import re
from pathlib import Path

NOFILE = 1048576

SUMMARY_FIELDS = {
    "Total time": "total_time_s",
    "Prep time": "prep_time_s",
    "IO time": "io_time_s",
    "File/s": "files_per_s",
    "MiB/s": "mib_s",
    "IOPS": "iops",
    "Number of IOs": "num_ios",
}


INT_FIELDS = {"num_ios"}


def _parse_summary(text):
    out = {}
    for label, field in SUMMARY_FIELDS.items():
        m = re.search(rf"^\s*{re.escape(label)}:\s*([0-9.]+)\s*$", text, re.MULTILINE)
        if m:
            out[field] = int(float(m.group(1))) if field in INT_FIELDS \
                else float(m.group(1))
    return out


def _append_record(cijoe, args, log_name, text, io_threads, queue_depth,
                   cpu_mask, assume_aligned_only, idle_spin, busy_spin,
                   error=None):
    artifacts = Path(cijoe.output_path) / "artifacts"
    meta_path = artifacts / "meta.json"
    if not meta_path.is_file():
        log.warning("no meta.json; skipping history record")
        return
    meta = json.loads(meta_path.read_text())
    opends = args.backend == "opends"
    env_keys = ("hostname", "kernel", "cuda", "nvme_bdf", "nvme_model",
                "gpu_model", "hugepages_2m")
    record = {
        "schema_version": 1,
        "run_id": f"{meta['timestamp']}-{meta['hostname']}",
        "run_timestamp": meta["timestamp"],
        "opends": {"sha": meta.get("commit_sha", ""),
                   "dirty": meta.get("dirty", False)},
        "deps": meta.get("deps", {}),
        "env": {k: meta.get(k) for k in env_keys},
        "config": {
            "backend": args.backend,
            "dataset": args.data_dir,
            "mode": args.mode,
            "cold_cache": True,
            "batches": args.batches,
            "batch_size": args.batch_size,
            "io_threads": int(io_threads) if opends and io_threads else None,
            "queue_depth": int(queue_depth) if opends and queue_depth else None,
            "cpu_mask": cpu_mask if opends and cpu_mask else None,
            "assume_aligned_only": bool(assume_aligned_only) if opends else None,
            "idle_spin_us": int(idle_spin) if opends and idle_spin is not None
            else None,
            "busy_spin": bool(busy_spin) if opends else None,
            "n": 1,
        },
        "result": _parse_summary(text),
        "raw_log": log_name,
    }
    if error is not None:
        record["error"] = error
    with (artifacts / "history.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def _batches_override(data_dir, default):
    """Per-dataset batches override from OPENDS_BENCH_BATCHES ("ds=N,ds=N").

    Returns None (fails the step) on a matching entry whose count is not
    a positive integer.
    """
    spec = os.environ.get("OPENDS_BENCH_BATCHES", "")
    for item in spec.split(","):
        ds, _, n = item.partition("=")
        if ds.strip() == data_dir and n.strip():
            if not n.strip().isdigit() or int(n) == 0:
                log.error(f"OPENDS_BENCH_BATCHES: {item.strip()!r} is not "
                          "<dataset>=<positive integer>")
                return None
            return int(n)
    return default


def add_args(parser):
    parser.add_argument("--backend", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batches", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--mnt", default="")
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")


def main(args, cijoe):
    args.batches = _batches_override(args.data_dir, args.batches)
    if args.batches is None:
        return 1
    io_threads = os.environ.get("OPENDS_AISIO_IO_THREADS")
    queue_depth = os.environ.get("OPENDS_AISIO_QUEUE_DEPTH")
    cpu_mask = os.environ.get("OPENDS_AISIO_CPU_MASK")
    aligned = os.environ.get("OPENDS_AISIO_ASSUME_ALIGNED_ONLY")
    assume_aligned_only = bool(aligned and aligned != "0")
    idle_spin = os.environ.get("OPENDS_AISIO_IDLE_SPIN_US")
    busy = os.environ.get("OPENDS_AISIO_BUSY_SPIN")
    busy_spin = bool(busy and busy != "0")
    knobs = (f" io_threads={io_threads} queue_depth={queue_depth}"
             f" cpu_mask={cpu_mask}"
             f" assume_aligned_only={int(assume_aligned_only)}"
             f" idle_spin_us={idle_spin} busy_spin={int(busy_spin)}"
             if args.backend == "opends" else "")
    print(f"--- {args.backend} {args.data_dir} {args.mode} "
          f"(batches={args.batches} batch_size={args.batch_size}{knobs}) ---",
          flush=True)
    bdf = cijoe.getconf("test.nvme_bdf")
    env = ""
    mnt = args.mnt
    if args.backend == "gds":
        repo = cijoe.getconf("test.repo_path")
        err, state = cijoe.run(f"'{repo}/tasks/steps/resolve_nvme_ns.sh' '{bdf}'")
        if err:
            return err
        target = state.output().strip().splitlines()[-1]
    elif args.backend == "opends":
        # HOMI owns the controller; qublk re-exports it as a ublk block device
        # that fil walks with xal for enumeration, exactly like gds walks the
        # kernel-bound NVMe. The aisio backend attaches a qpair from HOMI per
        # file (via OPENDS_HOMI_DEV/SOCKET), so the device read path bypasses
        # the ublk target.
        err, state = cijoe.run("cat /run/homi/ublk_dev")
        if err:
            return err
        target = state.output().strip().splitlines()[-1]
        sock = "/run/homi/homi.sock"
        if not mnt:
            mnt = cijoe.getconf("test.mount_point")
        env = (
            f"OPENDS_HOMI_DEV='{bdf}' OPENDS_HOMI_SOCKET='{sock}' "
            f"OPENDS_HOMI_MNT='{mnt}' "
        )
        if io_threads:
            env += f"OPENDS_AISIO_IO_THREADS='{io_threads}' "
        if queue_depth:
            env += f"OPENDS_AISIO_QUEUE_DEPTH='{queue_depth}' "
        if cpu_mask:
            env += f"OPENDS_AISIO_CPU_MASK='{cpu_mask}' "
        if assume_aligned_only:
            env += "OPENDS_AISIO_ASSUME_ALIGNED_ONLY='1' "
        if idle_spin is not None:
            env += f"OPENDS_AISIO_IDLE_SPIN_US='{idle_spin}' "
        if busy_spin:
            env += "OPENDS_AISIO_BUSY_SPIN='1' "
    else:
        target = bdf

    filperf = [
        f"{env}prlimit --nofile={NOFILE}:{NOFILE} --",
        f"filperf '{target}'",
        f"--backend {args.backend}",
        f"--data-dir {args.data_dir}",
        f"--batches {args.batches} --batch-size {args.batch_size}",
        "--summary",
    ]
    if mnt:
        filperf.insert(2, f"--mnt '{mnt}'")
    if args.mode == "async":
        filperf.append("--async")

    cmd = "echo 3 > /proc/sys/vm/drop_caches\n" + " \\\n  ".join(filperf)
    err, state = cijoe.run(cmd)
    # assume_aligned_only rejects any read with a sub-LBA tail, so a dataset
    # whose files are not LBA-multiples fails by construction. Record it and
    # let the sweep reach the datasets that do qualify.
    if err and not assume_aligned_only:
        return err

    suffix = "_async" if args.mode == "async" else ""
    out = (Path(cijoe.output_path) / "artifacts"
           / f"{args.backend}_{args.data_dir}{suffix}.log")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(state.output())
    log.info(f"wrote {out}")
    _append_record(cijoe, args, out.name, state.output(), io_threads,
                   queue_depth, cpu_mask, assume_aligned_only, idle_spin,
                   busy_spin, error=f"filperf rc={err}" if err else None)
    if err:
        log.warning(f"{args.data_dir}/{args.mode}: filperf rc={err} under "
                    f"assume_aligned_only; recorded and continuing")
    return 0
