# SPDX-License-Identifier: BSD-3-Clause
"""Capture build commit and target platform info to artifacts/meta.json.

Runs early in a bench suite so its results carry the commit hash and
host/kernel/NVMe/GPU they were taken on. The OpenDS commit comes from
`<repo_path>/.commit_stamp` on the target (written by
scripts/rsync.py); reading from the target instead of the host avoids
the decoupling where host edits after rsync would mislabel binaries
built from the prior tree.

Dependency SHAs (xnvme, xal, homi, qublk, fil) are resolved from their
checkouts under config.deps.src_root, so a throughput change is
attributable to OpenDS versus a dep rebase under the same branch pin.

Under GitHub Actions the record also carries the run's identity: the
run id and attempt, the job, the pull request and its head and base
commits, from the GITHUB_* variables and the OPENDS_CI_* ones the
workflow exports. A stored record then names the code it measured and
the run it came from, which the merge ref's commit alone does not.
"""

import json
import logging as log
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

DEPS = ["xnvme", "xal", "homi", "qublk", "fil"]


def _target_stdout(cijoe, cmd, quiet=False):
    err, state = cijoe.run(cmd)
    if err:
        if not quiet:
            log.error(f"failed({cmd}); err({err})")
        return None
    return state.output().strip()


def _short_nvme(machine_readable):
    """Render a tighter NVMe identifier from `lspci -mm -s <bdf>` output.

    The verbose `lspci -s` form prefixes the device with the full legal
    vendor name and a generic `NVMe SSD Controller` qualifier, making the
    stamp unwieldy. Take the short vendor (first token) plus the device
    string with that qualifier stripped.
    """
    parts = shlex.split(machine_readable)
    if len(parts) < 4:
        return machine_readable
    vendor = parts[2].split()[0]
    device = re.sub(r"^(?:NVMe SSD Controller|NVMe Controller)\s+", "", parts[3])
    return f"{vendor} {device}".strip()


def _split_commit(stamp):
    if not stamp:
        return "", False
    dirty = stamp.endswith("-dirty")
    return (stamp[: -len("-dirty")] if dirty else stamp), dirty


def _cuda_version(cijoe):
    out = _target_stdout(
        cijoe, "/usr/local/cuda/bin/nvcc --version 2>/dev/null || nvcc --version",
    ) or ""
    m = re.search(r"release ([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else ""


def _hugepages_2m(cijoe):
    out = _target_stdout(
        cijoe, "cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages",
    )
    return int(out) if out and out.isdigit() else None


def _ci_identity():
    """The GitHub Actions run this bench belongs to, or None outside CI."""
    env = os.environ
    run_id = env.get("GITHUB_RUN_ID")
    if not run_id:
        return None
    repo = env.get("GITHUB_REPOSITORY", "")
    attempt = env.get("GITHUB_RUN_ATTEMPT", "1")
    pr = env.get("OPENDS_CI_PR") or None
    return {
        "provider": "github",
        "repository": repo,
        "workflow": env.get("GITHUB_WORKFLOW", ""),
        "event": env.get("GITHUB_EVENT_NAME", ""),
        "run_id": int(run_id),
        "run_attempt": int(attempt),
        "job": env.get("OPENDS_CI_JOB") or env.get("GITHUB_JOB", ""),
        "pr": int(pr) if pr else None,
        "head_sha": env.get("OPENDS_CI_HEAD_SHA") or env.get("GITHUB_SHA", ""),
        "base_sha": env.get("OPENDS_CI_BASE_SHA") or None,
        "url": f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}"
               f"/actions/runs/{run_id}/attempts/{attempt}",
    }


def _xfs_info(cijoe, repo_path, nvme_bdf, mount_point):
    """xfs_info of the dataset filesystem as {section: {key: value}}.

    Read through the mount when there is one, which is the case inside a
    bench leg where the stack owns the drive, and from the namespace
    otherwise.
    """
    target = None
    if mount_point and _target_stdout(
        cijoe, f"findmnt -rno SOURCE '{mount_point}'", quiet=True,
    ):
        target = mount_point
    else:
        ns = _target_stdout(
            cijoe, f"'{repo_path}/tasks/steps/resolve_nvme_ns.sh' '{nvme_bdf}'",
            quiet=True,
        )
        target = ns.splitlines()[-1] if ns else None
    if not target:
        return {}
    out = _target_stdout(cijoe, f"xfs_info '{target}' 2>/dev/null") or ""
    info, section = {}, None
    for line in out.splitlines():
        m = re.match(r"^([a-z-]+)\s*=", line)
        if m:
            section = m.group(1)
        if section is None:
            continue
        for key, val in re.findall(r"([A-Za-z0-9_-]+)=([A-Za-z0-9_./-]+)", line):
            info.setdefault(section, {})[key] = val
    return info


def _nvme_link(cijoe, nvme_bdf):
    out = _target_stdout(cijoe, f"lspci -vv -s '{nvme_bdf}' 2>/dev/null") or ""
    m = re.search(r"LnkSta:\s*Speed ([^,]+), Width (x\d+)", out)
    return f"{m.group(1)} {m.group(2)}" if m else ""


def _dep_versions(cijoe, src_root):
    deps = {}
    for dep in DEPS:
        d = f"{src_root}/{dep}"
        sha = _target_stdout(cijoe, f"git -C '{d}' rev-parse HEAD 2>/dev/null") or ""
        status = _target_stdout(
            cijoe, f"git -C '{d}' status --porcelain 2>/dev/null",
        )
        deps[dep] = {
            "sha": sha,
            "ref": cijoe.getconf(f"{dep}.repository.ref", ""),
            "dirty": bool(status),
        }
    return deps


def main(args, cijoe):
    repo_path = cijoe.getconf("test.repo_path")
    src_root = cijoe.getconf("deps.src_root", "")
    nvme_bdf = cijoe.getconf("test.nvme_bdf", "")

    nvme_model = ""
    if nvme_bdf:
        mm = _target_stdout(cijoe, f"lspci -mm -s '{nvme_bdf}'") or ""
        nvme_model = _short_nvme(mm) if mm else ""

    stamp = _target_stdout(cijoe, f"cat '{repo_path}/.commit_stamp'")
    commit_sha, dirty = _split_commit(stamp)

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": stamp,
        "commit_sha": commit_sha,
        "dirty": dirty,
        "hostname": _target_stdout(cijoe, "uname -n"),
        "kernel": _target_stdout(cijoe, "uname -r"),
        "kernel_cmdline": _target_stdout(cijoe, "cat /proc/cmdline") or "",
        "cpu_model": _target_stdout(
            cijoe, "lscpu | sed -n 's/^Model name: *//p'",
        ) or "",
        "cpu_governor": _target_stdout(
            cijoe, "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        ) or "",
        "cuda": _cuda_version(cijoe),
        "hugepages_2m": _hugepages_2m(cijoe),
        "nvme_bdf": nvme_bdf,
        "nvme_model": nvme_model,
        "nvme_link": _nvme_link(cijoe, nvme_bdf) if nvme_bdf else "",
        "fs": _xfs_info(cijoe, repo_path, nvme_bdf,
                        cijoe.getconf("test.mount_point", "")) if nvme_bdf else {},
        "gpu_model": _target_stdout(
            cijoe, "nvidia-smi --query-gpu=name --format=csv,noheader | head -1",
        ) or "",
        "deps": _dep_versions(cijoe, src_root) if src_root else {},
        "ci": _ci_identity(),
    }

    out = Path(cijoe.output_path) / "artifacts" / "meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n")
    log.info(f"wrote {out}")

    if os.environ.get("OPENDS_BENCH_STRICT") and (
        meta["dirty"] or any(d.get("dirty") for d in meta["deps"].values())
    ):
        log.error("OPENDS_BENCH_STRICT: refusing to bench a dirty tree "
                  "(OpenDS or a dependency)")
        return 1
    return 0
