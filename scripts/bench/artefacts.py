#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Publish bench results to the orphan `artefacts` branch.

Layout on the branch (no shared history with the code):

  README.md                    generated index of snapshots
  latest/                      newest snapshot's report.* (overwritten)
  snapshots/<date>-<sha>/      one immutable snapshot per publish:
      report.md report.png sweep.csv
      history.jsonl            every per-leg record, concatenated

Collects report.md / sweep.csv / report.png and every history.jsonl under the
input dirs and commits a new snapshot through a throwaway git worktree; the
working branch and its uncommitted changes are never touched. A missing local
branch is created from origin/artefacts when the remote has one, so snapshot
history continues across clones. --push then publishes the branch to origin
(default: commit only).
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import ROOT, iter_history, ok

BRANCH = "artefacts"
WORKTREE = ROOT / ".artefacts"
REPORT_FILES = ("report.md", "sweep.csv", "report.png")
SNAP_DIR = "snapshots"
LATEST_DIR = "latest"


def _git(*a, where=ROOT, **kw):
    return subprocess.run(["git", "-C", str(where), *a],
                          text=True, capture_output=True, **kw)


def _ensure_branch():
    if _git("rev-parse", "--verify", "--quiet", BRANCH).returncode == 0:
        return
    if _git("rev-parse", "--verify", "--quiet",
            f"origin/{BRANCH}").returncode == 0:
        r = _git("branch", BRANCH, f"origin/{BRANCH}")
    else:
        tree = _git("mktree", input="").stdout.strip()
        commit = _git("commit-tree", tree, "-m",
                      f"init {BRANCH}").stdout.strip()
        if not commit:
            sys.exit(f"failed to seed orphan {BRANCH} branch")
        r = _git("branch", BRANCH, commit)
    if r.returncode:
        sys.exit(r.stderr.strip())


def _label():
    sha = _git("rev-parse", "--short", "HEAD").stdout.strip() or "nocommit"
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{date}-{sha}" + ("-dirty" if dirty else "")


def _collect(in_dirs, dest):
    n = 0
    for d in in_dirs:
        for name in REPORT_FILES:
            if (d / name).is_file():
                shutil.copy2(d / name, dest / name)
                n += 1
    with (dest / "history.jsonl").open("w") as out:
        for _, hist in iter_history(in_dirs):
            text = hist.read_text()
            out.write(text if text.endswith("\n") else text + "\n")
            n += 1
    return n


def _write_index(worktree):
    snaps = sorted((p.name for p in (worktree / SNAP_DIR).iterdir()
                    if p.is_dir()), reverse=True)
    lines = [
        "# OpenDS benchmark artefacts",
        "",
        "Generated bench data published by `scripts/bench/artefacts.py`. Orphan "
        "branch: no shared history with the code.",
        "",
        f"- `{LATEST_DIR}/` mirrors the newest snapshot's report, overwritten "
        "each publish.",
        f"- `{SNAP_DIR}/<date>-<sha>/` is one immutable snapshot per publish: "
        "`report.md`, `report.png`, `sweep.csv`, and `history.jsonl` (all legs "
        "concatenated).",
        "",
        "## Snapshots",
        "",
    ]
    lines += [f"- [`{s}`]({SNAP_DIR}/{s}/report.md)" for s in snaps]
    (worktree / "README.md").write_text("\n".join(lines) + "\n")


def _resolve(dirs):
    return [d if d.is_absolute() else ROOT / d for d in (Path(x) for x in dirs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dirs", action="append", metavar="DIR",
                    help="Dir with report.* and history.jsonl. Repeatable. "
                         "Default cijoe-output-bench.")
    ap.add_argument("-m", "--message", help="Commit message.")
    ap.add_argument("--push", action="store_true",
                    help="Push the branch to origin after committing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be published; touch no refs.")
    a = ap.parse_args()

    in_dirs = _resolve(a.in_dirs or ["cijoe-output-bench"])
    missing = [str(d) for d in in_dirs if not d.is_dir()]
    if missing:
        sys.exit("no such dir: " + ", ".join(missing))

    label = _label()
    msg = a.message or f"{label} bench snapshot"

    if a.dry_run:
        print(f"would publish {BRANCH}:{SNAP_DIR}/{label} (+ {LATEST_DIR}/) "
              "from " + ", ".join(str(d) for d in in_dirs))
        for d in in_dirs:
            for name in REPORT_FILES:
                if (d / name).is_file():
                    print(f"  {SNAP_DIR}/{label}/{name}")
        legs = sum(1 for _ in iter_history(in_dirs))
        print(f"  {SNAP_DIR}/{label}/history.jsonl  ({legs} legs)")
        return

    if WORKTREE.exists():
        _git("worktree", "remove", "--force", str(WORKTREE))
    _ensure_branch()
    r = _git("worktree", "add", str(WORKTREE), BRANCH)
    if r.returncode:
        sys.exit(r.stderr.strip())
    try:
        dest = WORKTREE / SNAP_DIR / label
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        n = _collect(in_dirs, dest)

        latest = WORKTREE / LATEST_DIR
        latest.mkdir(exist_ok=True)
        for name in REPORT_FILES:
            if (dest / name).is_file():
                shutil.copy2(dest / name, latest / name)

        _write_index(WORKTREE)

        _git("add", "-A", where=WORKTREE)
        c = _git("commit", "-m", msg, where=WORKTREE)
        if c.returncode and "nothing to commit" not in (c.stdout + c.stderr):
            sys.exit((c.stderr or c.stdout).strip())
        head = _git("rev-parse", "--short", BRANCH).stdout.strip()
        print(f"committed {n} files to {BRANCH}:{SNAP_DIR}/{label} ({head})")
        if a.push:
            p = _git("push", "origin", BRANCH)
            sys.stdout.write(p.stdout)
            sys.stderr.write(p.stderr)
            if p.returncode:
                sys.exit(f"push to origin {BRANCH} failed (rc {p.returncode})")
            print(f"pushed {BRANCH} to origin")
        else:
            print(f"push with: git push origin {BRANCH}  (or rerun --push)")
    finally:
        _git("worktree", "remove", "--force", str(WORKTREE))
    ok("artefacts")


if __name__ == "__main__":
    main()
