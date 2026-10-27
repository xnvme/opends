#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Unmount and hand an NVMe device to userspace (xnvme upcie-cuda), leaving
# the controller in a clean, verified power-on state.
set -e

if [ $# -ne 2 ]; then
	echo "usage: unbind_nvme.sh BDF MOUNT" >&2
	exit 1
fi

BDF=$1
MOUNT=$2
HERE=$(dirname "$0")

if mountpoint -q "$MOUNT"; then
	echo "unmounting $MOUNT"
	umount "$MOUNT"
fi

# Re-enumerate the function before unbinding: a PCI remove + rescan forces the
# kernel nvme probe (reset, wait for CSTS.RDY, identify), leaving the
# controller in a verified-live state before handing it to userspace.
# OPENDS_NVME_VERIFY=0 skips the re-enumeration and probe verification, for
# hosts where the kernel driver must not probe the device (e.g. a duplicate
# subsystem NQN, or interrupts the driver cannot use). The FLR below still
# returns the controller to a clean power-on state.
if [ "${OPENDS_NVME_VERIFY:-1}" != "0" ]; then
	echo "re-enumerating $BDF"
	echo 1 > "/sys/bus/pci/devices/$BDF/remove"
	echo 1 > /sys/bus/pci/rescan

	NS=""
	for _ in $(seq 1 50); do
		if [ -d "/sys/bus/pci/devices/$BDF/nvme" ]; then
			if NS=$("$HERE/resolve_nvme_ns.sh" "$BDF" 2>/dev/null) &&
			   [ -b "$NS" ]; then
				break
			fi
		fi
		NS=""
		sleep 0.2
	done
	if [ -z "$NS" ]; then
		echo "error: $BDF did not come back as a live nvme namespace after rescan" >&2
		exit 1
	fi
	echo "$BDF live as $NS"
fi

if [ -e "/sys/bus/pci/devices/$BDF/driver" ]; then
	echo "unbinding $BDF"
	echo "$BDF" > "/sys/bus/pci/devices/$BDF/driver/unbind"
fi

if [ -e "/sys/bus/pci/devices/$BDF/driver" ]; then
	DRIVER=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")")
	echo "error: $BDF still bound to $DRIVER after unbind"
	exit 1
fi
echo "$BDF unbound"

# Unbinding shuts the controller down (CC.SHN). Follow with an FLR to return
# it to a clean power-on state xnvme's open can enable, then settle so the
# open does not race the controller-level reset.
echo "resetting $BDF"
echo 1 > "/sys/bus/pci/devices/$BDF/reset"
sleep 1

# The upcie owner path (xnvme SYSFS backend) claims the function through
# uio_pci_generic; a fully unbound device is no longer accepted. driver_override
# pins the binding to this device only.
modprobe uio_pci_generic
echo uio_pci_generic > "/sys/bus/pci/devices/$BDF/driver_override"
echo "$BDF" > /sys/bus/pci/drivers/uio_pci_generic/bind
DRIVER=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")" 2>/dev/null)
if [ "$DRIVER" != "uio_pci_generic" ]; then
	echo "error: $BDF not bound to uio_pci_generic (got '$DRIVER')" >&2
	exit 1
fi
echo "$BDF bound to uio_pci_generic"
