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
machine's persistent runner, in xnvme-devenv's host container, and the
suite runs in a guest the job boots there: the guest image described
below, on a disk made afresh for the job, with the machine's NVMe
device and GPU passed through with vfio-pci. Every job so starts on a
pristine machine, and what it installs or leaves running goes with the
guest, which is killed when the job ends, cancelled or not. Only the
filesystem on the NVMe device outlives a job.

The machine serves one job at a time, and may serve people too. The
workflow does not arbitrate that; the runner does, with a script it
runs before each job and one after it, as the setup below says. A job
that has to wait for the machine waits there, up to a day; each job's
timeout is its run budget plus that day. The check is required, so a
labeled PR cannot merge before the suite passes. A new push or a label
change cancels the superseded run, queued or mid-suite, and starts a
fresh one.

The job runs the PR's own workflow and scripts as root in a privileged
container, which is root on the machine in all but name. The label is
the only gate, and it stays on the PR, so later pushes by the author
run on the hardware without another look. The machine is therefore
sacrificial: it holds no credentials and nothing trusts it. GitHub's
setting for workflows from fork pull requests, which can require
approval for outside collaborators, covers these jobs like the hosted
ones.

The job checks out the PR in the container. `ci/scripts/guest.py up` boots
the guest and authorizes a key made for the run in it;
`ci/configs/guest-transport.toml` and `ci/configs/guest-test.toml` become the
job's `transport.toml` and `test.toml`, so cijoe drives the guest over
SSH on the port QEMU forwards. From there it is the same
rsync/build/run_tests flow as manual development (the quick start in
the README), except that the guest boots with xnvme and xal installed
and the job builds only OpenDS and fil. The full log lands on the
workflow run.

The bench labels work the same way, except that a bench job waits for
`test-full` and is skipped when that one fails; without the `test-full`
label it runs on its own. `bench` runs one leg at the default knobs,
`bench-sweep` runs the standard sweep (`ci/scripts/bench/sweep.toml`), and
`bench-full-sweep` runs the whole knob cross product. Every job uploads its cijoe output as a workflow
artifact on the run. A bench job that ran its legs also uploads
`results-<job>`: `results.json` with the run's provenance, every
record, the raw filperf logs and the report, and `report.md` beside it,
so a collector fetches one artifact per job. The reference datasets
must already be present on the machine's NVMe device, and nothing else
should be: filperf indexes the whole filesystem, so the suite removes
its files at the end of a run, and a file left by other work changes
the bench or crashes it. A guest cannot set the CPU's clocks, so the
machine keeps them at the bench's setting.

### Repository setup

One-time GitHub configuration on the repository the workflow runs in:

1. Create the `test-full`, `bench`, `bench-sweep`, and
   `bench-full-sweep` labels.
2. Add the four checks, `test-full / nvme-cuda`, `bench / nvme-cuda`,
   `bench-sweep / nvme-cuda` and `bench-full-sweep / nvme-cuda`, to the
   required status checks. A skipped check satisfies the requirement,
   so unlabeled PRs are unaffected.

### Runner machine

One-time setup on the hardware machine:

1. Boot with the IOMMU on in a mode that gives devices their groups
   (`iommu=pt` does), check that `/dev/kvm` is there, and install
   docker with the runner's user in its group.
2. Bind the NVMe device and the GPU to vfio-pci at boot, with the other
   functions in their IOMMU groups, so that the host never drives
   them. The job binds them when they are not, but that fails while a
   service of the host holds the GPU.
3. Give their addresses on the host in `/etc/opends-ci/rig.toml`, the
   layer over the rig in `ci/configs/guest.toml`:

   ```toml
   [rig.devices.nvme0]
   host = "0000:01:00.0"

   [rig.devices.gpu0]
   host = "0000:2b:00.0"
   ```

4. Make `/var/cache/opends-ci`, the image cache. The first job builds
   the images into it, as the section below says.
5. Set the clocks the bench wants at boot: the schedutil governor and
   boost off. A guest cannot set them.
6. Provision the NVMe device as the quick start in the README says
   for a target, with an XFS filesystem and the reference datasets for
   bench, before it goes to vfio-pci or from a guest.
7. Install the GitHub Actions runner under `~/actions-runner`.
   Register it with an extra `nvme-cuda` label. Install it as a
   service (`sudo ./svc.sh install <user> && sudo ./svc.sh start`) and
   leave it running: GitHub deletes the registration of a runner that
   has not connected for 14 days.
8. Where people use the machine too, let the runner take the lock they
   use: name a script that waits for it and takes it as
   `ACTIONS_RUNNER_HOOK_JOB_STARTED` in the runner's `.env`, and one
   that gives it back as `ACTIONS_RUNNER_HOOK_JOB_COMPLETED`. The
   runner runs the first before a job's steps and the second after
   them, and a job waits for as long as the first takes. The project's
   machine uses a [labgrid](https://labgrid.readthedocs.io/) place for
   this, held as `gh-run-<run id>-<job>`.

## The guest image

The hardware jobs run inside a guest on the machine, with the NVMe
device and the GPU passed through to it with vfio-pci, so that every
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

A job builds it when the cache holds none for its checkout. To build it
by hand, on the machine, as the runner user, with docker installed and
`/dev/kvm` present:

```sh
ci/scripts/guest_image.py --cache /var/cache/opends-ci
```

That runs `ci/tasks/guest_derive.yaml` in the container: stage the base,
make a scratch guest from it with its disk grown to 64 GB, boot it,
install the build packages and the kernel, reboot into that kernel, drop
the base's older kernels, install the pinned NVIDIA packages and the
upcie modules so their DKMS builds target it, check that every DKMS
module built and load the ones that do not need the hardware, and
export the disk as `opends-guest-<key>.qcow2` in the image cache,
`~/.cache/xnvme-devenv` by default. The build takes about ten
minutes once the base is cached; the first one also fetches the base.
The `ci` guest in `ci/configs/guest.toml` is sized for the suite and boots
that image through the usual initialise step, a fresh overlay per boot.
The hardware jobs boot it with the devices, through `ci/scripts/guest.py`.

The cache holds two kinds of image, each named after a key of what it
is made from, so that a job finds the one for its checkout, builds it
when there is none, and leaves the ones other checkouts use alone. The
key of the guest image covers the lock, the derive recipe and the
derive guest. On it sits the image a job boots,
`opends-deps-<key>-<key>.qcow2`: the guest image with xnvme and xal
built and installed, an overlay that `ci/scripts/guest.py` makes by
booting the guest image, running the two setup tasks in it and keeping
its disk. Its key covers the commits that the refs in
`ci/configs/deps.toml` point at when the job starts, so a branch that
moved gets a new image, and how the two are built.
`ci/scripts/guest.py keys` prints both names. The two guest images and the
four images with dependencies used last are kept, the rest removed.

One module is known not to load in the image: nvidia-fs builds without
the open driver's symbol versions and the kernel refuses it, so it is
left out of the modules the image loads at boot and cuFile runs in its
compatibility mode in the guest until that is solved.
