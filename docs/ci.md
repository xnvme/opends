# Continuous integration

One workflow, `.github/workflows/ci.yml`, running on GitHub-hosted
runners for every push to main and every pull request. Both jobs are
intended as required checks.

`lint` runs `scripts/lint_check.py` (clang-format 18, pinned by the
ubuntu-24.04 image).

`test-ref` builds without CUDA, so only the ref backend and its tests
are compiled, then runs the ref test binaries against a pattern file in
the build directory and the Python binding tests under `python/tests`.
No block device is involved; the ref backend does not require O_DIRECT.

The gds and aisio backends are not covered by CI. Compiling them
requires CUDA and the pinned dependency stack, and running them requires
a machine with an NVMe device and a GPU with dma-buf P2P support. That
suite remains the manual flow described in the README: `rsync.py`,
`build.py`, `run_tests.py` against a configured target.
