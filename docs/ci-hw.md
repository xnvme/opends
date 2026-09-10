# Hardware CI

The `test-full` check runs the full test suite, aisio included, on a
hardware machine. It is opt-in per PR.

A maintainer puts the `test-full` label on the PR. The label starts a
`ci / test-full` check. The check waits for the hosted `lint` and
`test-ref` jobs. When they pass, the job runs on the hardware
machine's persistent runner. If the runner service is stopped, a
labeled run queues until it starts again. A job that stays queued for
more than 24 hours fails. The check is required, so a labeled PR
cannot merge before the suite passes. A new push or a label change
cancels the superseded run, queued or mid-suite, and starts a fresh
one.

The job checks out the PR on the machine. It drives the hardware
through cijoe over a local SSH loop (`root@localhost`). This is the
same rsync/build/run_tests flow as manual development (see "Remote
testing with CIJOE" in the README). The full log lands on the
workflow run.

The bench labels work the same way. `bench` runs one leg at the
default knobs, `bench-sweep` runs the standard sweep
(`scripts/bench/sweep.toml`), and `bench-full-sweep` runs the whole
knob cross product. Each uploads its cijoe output, `report.md`
included, as a workflow artifact on the run. The reference datasets
must already be present on the machine.

## Repository setup

One-time GitHub configuration on the repository the workflow runs in:

1. Create the `test-full`, `bench`, `bench-sweep`, and
   `bench-full-sweep` labels.
2. Add the four checks to the required status checks. A skipped
   check satisfies the requirement, so unlabeled PRs are unaffected.

## Runner machine

One-time setup on the hardware machine, as an unprivileged user:

1. Bring the machine to the target requirements ("Remote testing with
   CIJOE" in the README). Populate the reference datasets for bench.
2. Install the GitHub Actions runner under `~/actions-runner`.
   Register it with an extra `nvme-cuda` label. Install it as a
   service (`sudo ./svc.sh install <user> && sudo ./svc.sh start`).
3. Create the local SSH loop: Make a passphrase-less key pair for the
   runner user and authorize it for `root@localhost`.
4. Copy `configs/{test,transport}.toml.example` to
   `~/.config/opends-ci/{test,transport}.toml` and fill them in. Set
   `localhost` as the hostname. Use a `repo_path` and a `build_dir`
   dedicated to CI. The workflow copies the files into the checkout.
5. Install cijoe and the python dependencies of `scripts/` for the
   runner user.
