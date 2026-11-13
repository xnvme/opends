#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run OpenDS test suites on the target.

Each test step is identified by (backend, test). When no flag is
passed every test runs. Use --backend to restrict to one or more
backends (ref, cufile, aisio) and --test to restrict to one or more
test names. Required setup steps (bind/mount/prep, plus unbind for
aisio), the cleanup of the suite's files and the rebind teardown are
added automatically based on the selection.
"""

import argparse

from _helpers import ok, run_cijoe

BACKENDS = ["ref", "cufile", "aisio"]
TESTS_BY_BACKEND = {
    "ref":   ["smoke", "sync_read", "mt_read", "async_read", "batch_read"],
    "cufile": ["sync_read"],
    "aisio": ["sync_read", "sync_read_raw_mode", "mt_read", "register_large",
              "coherency", "async_read", "async_read_busy_spin", "batch_read",
              "stream_read", "stream_read_busy_spin", "sync_write",
              "stream_write", "python", "raw_block"],
}
ALL_TESTS = sorted({t for ts in TESTS_BY_BACKEND.values() for t in ts})


def _steps_for(backends, tests):
    """Build the cijoe step list for the requested (backend, test) matrix.

    Returns an empty list (run every step in test.yaml) when the full
    matrix is requested, matching bench/run.py's convention.
    """
    if set(backends) == set(BACKENDS) and set(tests) == set(ALL_TESTS):
        return []

    selected = [
        (b, t) for b in backends for t in tests
        if t in TESTS_BY_BACKEND[b]
    ]
    if not selected:
        return []

    needs_prep = any(t != "smoke" for _, t in selected)
    needs_stack = any(b == "aisio" for b, _ in selected)

    steps = ["check_stack_down", "bind_nvme", "mount", "prepare_test_dir"]
    if needs_prep:
        steps.append("test_sync_read_prep")
    if any(t == "coherency" for _, t in selected):
        steps.append("test_coherency_prep")
    for b, t in selected:
        if b == "aisio":
            continue
        steps.append(f"test_{t}_{b}")
    raw = ("aisio", "raw_block") in selected
    if needs_stack:
        # Hand the controller to HOMI, expose it via qublk, remount the
        # same XFS, run the aisio tests against it, then restore nvme.
        steps.append("homi_stack_up")
        for b, t in selected:
            if b == "aisio" and t != "raw_block":
                steps.append(f"test_{t}_{b}")
    if raw:
        # Records where its pattern file is while the file system is
        # mounted, then runs with the stack reduced to homi.
        steps.append("raw_block_prep")
    steps.append("clean_test_dir")
    if raw:
        steps += ["homi_stack_bare", "test_raw_block_aisio"]
    if needs_stack:
        steps.append("homi_stack_down")
    if raw:
        steps.append("raw_block_clean")
    return steps


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--backend", action="append", choices=BACKENDS,
    help="Run only this backend's tests. Repeat to combine. Default: all.",
)
parser.add_argument(
    "--test", action="append", choices=ALL_TESTS,
    help="Run only this test. Repeat to combine. Default: all.",
)
args = parser.parse_args()

steps = _steps_for(args.backend or BACKENDS, args.test or ALL_TESTS)
print(f"\n=== test suite: {', '.join(steps) if steps else 'all steps'} ===",
      flush=True)
run_cijoe("tasks/test.yaml", *steps, out="cijoe-output-test")
ok("run_tests")
