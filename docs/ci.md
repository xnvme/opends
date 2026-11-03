# Continuous integration

One workflow, `.github/workflows/ci.yml`, whose hardware jobs share
their steps through `run-on-nvme-cuda.yml`. What the jobs run is in
`ci/`: the cijoe tasks and configs, and the scripts that drive them.
The `lint` and `test-ref` jobs run on GitHub-hosted runners for every
push to main and every pull request, and both are intended as required
checks.

`lint` runs `ci/scripts/lint_check.py` (clang-format 18, pinned by the
ubuntu-24.04 image).

`test-ref` builds without CUDA, so only the ref backend and its tests
are compiled, then runs the ref test binaries against a pattern file in
the build directory and the Python binding tests under `python/tests`.

The cufile and aisio backends need CUDA, the pinned dependency stack
and a machine with an NVMe device and a GPU with dma-buf P2P support,
so the hosted jobs do not cover them. Their suite runs on a self-hosted
runner behind the `test-full` label, and the benchmarks behind the
`bench` labels, as described below. Without a label those checks are
skipped, and the manual flow in the README (`rsync.py`, `build.py`,
`run_tests.py` against a configured target) remains available.

## Hardware checks

The `test-full` check runs the full test suite, aisio included, on a
hardware machine. It is opt-in per PR.

A maintainer puts the `test-full` label on the PR. The label starts a
`test-full / nvme-cuda` check. The check waits for the hosted `lint` and
`test-ref` jobs. When they pass, the job runs on the hardware
machine's persistent runner. The job first queues for the machine's
labgrid place as `gh-run-<run id>-<job>`, printing who holds it, and
waits up to a day; each job's timeout is its run budget plus that day.
It then checks that the stack is down: no HOMI process running,
nothing mounted over ublk, and the NVMe controller on the kernel
driver.
Otherwise the job fails before it installs anything, names what is in
the way, and leaves the machine alone. The place is given back when
the job ends, cancelled or not. Someone who needs the machine while a
job waits or runs cancels the run and re-runs it afterwards. The check
is required, so a labeled PR cannot merge before the suite passes. A
new push or a label change cancels the superseded run, queued or
mid-suite, and starts a fresh one.

The job runs the PR's own workflow and scripts as root on the machine.
The label is the only gate, and it stays on the PR, so later pushes by
the author run on the hardware without another look. The machine is
therefore sacrificial: it holds no credentials and nothing trusts it.
GitHub's setting for workflows from fork pull requests, which can
require approval for outside collaborators, covers these jobs like the
hosted ones.

The job checks out the PR on the machine. It drives the hardware
through cijoe over a local SSH loop (`root@localhost`). This is the
same rsync/build/run_tests flow as manual development (the quick
start in the README). The full log lands on the workflow run.

The bench labels work the same way, except that a bench job waits for
`test-full` and is skipped when that one fails; without the `test-full`
label it runs on its own. `bench` runs one leg at the default knobs,
`bench-sweep` runs the standard sweep (`ci/scripts/bench/sweep.toml`), and
`bench-full-sweep` runs the whole knob cross product. Every job uploads its cijoe output as a workflow
artifact on the run. A bench job that ran its legs also uploads
`results-<job>`: `results.json` with the run's provenance, every
record, the raw filperf logs and the report, and `report.md` beside it,
so a collector fetches one artifact per job. The reference datasets
must already be present on the machine, and nothing else should be:
filperf indexes the whole filesystem, so the suite removes its files at
the end of a run, and a file left by other work changes the bench or
crashes it.

### Repository setup

One-time GitHub configuration on the repository the workflow runs in:

1. Create the `test-full`, `bench`, `bench-sweep`, and
   `bench-full-sweep` labels.
2. Add the four checks, `test-full / nvme-cuda`, `bench / nvme-cuda`,
   `bench-sweep / nvme-cuda` and `bench-full-sweep / nvme-cuda`, to the
   required status checks. A skipped check satisfies the requirement,
   so unlabeled PRs are unaffected.

### Runner machine

One-time setup on the hardware machine, as an unprivileged user:

1. Provision the machine as the quick start in the README says.
   Populate the reference datasets for bench.
2. Install the GitHub Actions runner under `~/actions-runner`.
   Register it with an extra `nvme-cuda` label. Install it as a
   service (`sudo ./svc.sh install <user> && sudo ./svc.sh start`) and
   leave it running: GitHub deletes the registration of a runner that
   has not connected for 14 days. The stack check, not a stopped
   service, is what keeps a labeled run off a machine in use.
3. Create the local SSH loop: Make a passphrase-less key pair for the
   runner user and authorize it for `root@localhost`.
4. Copy `ci/configs/{test,transport}.toml.example` to
   `~/.config/opends-ci/{test,transport}.toml` and fill them in. Set
   `localhost` as the hostname. Use a `repo_path`, a `build_dir`, a
   `deps.src_root`, a `deps.build_root` and a `deps.install_prefix`
   dedicated to CI: a job deletes and clones the dependency checkouts
   afresh, and installs nothing into /usr/local; homi's install still
   writes its unit to /usr/lib/systemd and its default config to
   /etc/homi. The workflow copies the files into the checkout.
5. Install cijoe and the python dependencies of `ci/scripts/` for the
   runner user. `config.sh` records the PATH of the shell that runs it
   in the runner's `.path`, so configure from a shell that finds them.
6. Check that rsync on the machine is 3.2.3 or newer: `ci/scripts/rsync.py`
   passes `--mkpath`.
7. Install [labgrid](https://labgrid.readthedocs.io/), run
   `labgrid-coordinator` as a service bound to localhost, and create a
   place for the machine. Put `LG_PLACE=<place>` in the runner's
   `.env`. Everyone who uses the machine acquires that place first;
   the job queues for it with a reservation and gives it back with
   `release-from`.

## The guest image

The hardware jobs can also run inside a guest on the machine, with the
NVMe device and the GPU passed through to it with vfio-pci, so that every
job starts on a pristine machine and leaves nothing on the host. The
guest is nosi's Debian 13, the image xnvme's CI boots, with NVIDIA's
driver and CUDA, nvidia-fs, upcie's dma-buf modules, the packages the
dependency builds need and the settings the suite expects added on top.
It is built with xnvme-devenv's tooling, in its host container, the
way xnvme-devenv derives its own guests.

Everything that goes into the image is pinned in `ci/configs/guest-lock.toml`:
the base by digest, the NVIDIA packages by version and the upcie
release by checksum. The kernel is Debian's current one at build time;
the image records it, with the other versions, in `/etc/opends-guest`.
Moving a pin and building again is how anything in the image changes;
nothing in it is edited in place. The image holds the NVIDIA driver and
CUDA, so it stays on the machine and is not published.

Build it on the machine, as the runner user, with docker installed and
`/dev/kvm` present:

```sh
ci/scripts/guest_image.py
```

That runs `ci/tasks/guest_derive.yaml` in the container: stage the base,
make a scratch guest from it with its disk grown to 64 GB, boot it,
install the build packages and the kernel, reboot into that kernel, drop
the base's older kernels, install the pinned NVIDIA packages and the
upcie modules so their DKMS builds target it, check that every DKMS
module built and load the ones that do not need the hardware, and
export the disk as `opends-guest-x86_64.qcow2` in the image cache,
`~/.cache/xnvme-devenv` by default. The build takes about ten
minutes once the base is cached; the first one also fetches the base.
The `ci` guest in `ci/configs/guest.toml` is sized for the suite and boots
that image through the usual initialise step, a fresh overlay per boot.

One module is known not to load in the image: nvidia-fs builds without
the open driver's symbol versions and the kernel refuses it, so it is
left out of the modules the image loads at boot and cuFile runs in its
compatibility mode in the guest until that is solved. The jobs
themselves still run on the host; booting the guest from them, with the
passthrough rig, is the next step.
