#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run OpenDS test suites on the target.

Each test step is identified by (backend, test). When no flag is
passed every test runs. Use --backend to restrict to one or more
backends (ref, gds, aisio) and --test to restrict to one or more
test names. Required setup steps (bind/mount/prep, plus unbind for
aisio) and the rebind teardown are added automatically based on the
selection.
"""

import argparse

from _helpers import ok, run_cijoe

BACKENDS = ["ref", "gds", "aisio"]
TESTS_BY_BACKEND = {
    "ref":   ["smoke", "sync_read"],
    "gds":   ["sync_read"],
    "aisio": ["sync_read", "register_large", "async_read",
              "sync_write", "async_write", "block_alloc"],
}
ALL_TESTS = sorted({t for ts in TESTS_BY_BACKEND.values() for t in ts})


def _steps_for(backends, tests):
    """Build the cijoe step list for the requested (backend, test) matrix.

    Returns an empty list (run every step in test.yaml) when the full
    matrix is requested, matching run_bench.py's convention.
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

    steps = ["bind_nvme", "mount", "prepare_test_dir"]
    if needs_prep:
        steps.append("test_sync_read_prep")
    for b, t in selected:
        if b == "aisio":
            continue
        steps.append(f"test_{t}_{b}")
    if needs_stack:
        # Hand the controller to HOMI, expose it via qublk, remount the
        # same XFS, run the aisio tests against it, then restore nvme.
        steps.append("homi_stack_up")
        for b, t in selected:
            if b == "aisio":
                steps.append(f"test_{t}_{b}")
        steps.append("homi_stack_down")
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
