#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Runs as root in the derive guest, in two phases around a reboot into the kernel the image boots.
# prepare installs the build packages and that kernel; install adds NVIDIA's stack, the upcie modules and the suite's settings.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

usage() {
	echo "usage: guest_derive.sh prepare" >&2
	echo "       guest_derive.sh install DEBS_DIR REPO CUDA_VERSION PINNING PACKAGE..." >&2
	exit 1
}

# What xnvme's toolbox installs on Debian 13 for its build, plus what xal
# (xfslibs-dev), the suite (xfsprogs, rsync) and the DKMS builds need.
BUILD_PKGS="autoconf automake bash build-essential findutils git libaio-dev
libarchive-dev libcunit1-dev libibverbs-dev libisal-dev libncurses5-dev
libnuma-dev librdmacm-dev libssl-dev libtool liburing-dev make meson nasm
openssl patch pipx pkg-config python3 python3-jinja2 python3-pip
python3-pyelftools python3-tabulate python3-venv rdma-core systemd-dev
xfslibs-dev xfsprogs rsync dkms curl ca-certificates"

prepare() {
	# growpart says NOCHANGE when the partition already fills the disk
	growpart /dev/vda 1 || true
	resize2fs "$(findmnt -no SOURCE /)"
	apt-get -qy update
	# shellcheck disable=SC2086
	apt-get -qy install --no-install-recommends $BUILD_PKGS \
		linux-image-amd64 linux-headers-amd64
	apt-get -qy clean
	df -h /
	echo "prepared; reboot into $(ls /boot/vmlinuz-* | sort -V | tail -1)"
}

install() {
	local debs=$1 repo=$2 cuda=$3 pinning=$4
	shift 4
	local kernel old status
	kernel=$(uname -r)

	# The base ships older kernels; DKMS would build for each of them
	old=$(dpkg-query -W -f '${binary:Package}\n' \
		'linux-image-[0-9]*-amd64' 'linux-headers-[0-9]*-amd64' \
		| grep -v -- "-$kernel\$" || true)
	if [ -n "$old" ]; then
		# shellcheck disable=SC2086
		apt-get -qy purge $old
		apt-get -qy autoremove --purge
	fi

	dpkg -i "$debs"/cuda-keyring_*.deb
	# The keyring package adds its own sources entry for the repository the
	# lock names; add one when the two differ, or the pins would not resolve.
	if ! grep -rqs "$repo" /etc/apt/sources.list.d/; then
		echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $repo /" \
			> /etc/apt/sources.list.d/cuda-opends.list
	fi
	apt-get -qy update
	# apt reads its pins when it starts, so the driver pin goes in first
	apt-get -qy install "$pinning"
	apt-get -qy install --no-install-recommends "$@"
	apt-get -qy install "$debs"/upcie-experimental-dkms_*.deb
	apt-get -qy clean
	ln -sfn "/usr/local/cuda-$cuda" /usr/local/cuda

	# The settings the suite expects: upcie's hugepages, the modules the
	# stack and the baselines load, and CUDA on the PATH of ssh sessions.
	cat > /etc/sysctl.d/10-hugepages.conf <<-EOF
	# 1024 x 2 MiB hugepages for upcie (xnvme upcie-cuda)
	vm.nr_hugepages = 1024
	EOF
	printf '%s\n' ublk_drv uio_pci_generic dmabuf_import iommu_map_pa nvidia \
		> /etc/modules-load.d/opends.conf
	sed -i '/^PATH=/d' /etc/environment
	echo 'PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' \
		>> /etc/environment

	# What the image holds, for whoever boots it
	cat > /etc/opends-guest <<-EOF
	derived=$(date -u +%Y-%m-%dT%H:%M:%SZ)
	kernel=$kernel
	nvidia=$(dpkg-query -W -f '${Version}' nvidia-kernel-open-dkms)
	cuda=$cuda
	nvidia_fs=$(dpkg-query -W -f '${Version}' nvidia-fs-dkms)
	upcie=$(dpkg-query -W -f '${Version}' upcie-experimental-dkms)
	EOF

	echo "modules built for $kernel:"
	status=$(dkms status)
	for m in nvidia nvidia-fs upcie-experimental; do
		if ! grep -q "^$m/.*, $kernel, .*: installed" <<< "$status"; then
			echo "error: $m did not build for $kernel" >&2
			echo "$status" >&2
			exit 1
		fi
	done
	grep ", $kernel, " <<< "$status"
	# nvidia and nvidia_fs need the GPU, which this guest does not have
	modprobe -a ublk_drv uio_pci_generic dmabuf_import iommu_map_pa
	cat /etc/opends-guest
	df -h /
}

case "${1:-}" in
prepare)
	[ $# -eq 1 ] || usage
	prepare
	;;
install)
	[ $# -ge 6 ] || usage
	shift
	install "$@"
	;;
*)
	usage
	;;
esac
