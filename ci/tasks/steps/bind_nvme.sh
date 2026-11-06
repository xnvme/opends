#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Bind NVMe device to kernel driver, print the resolved namespace path.
# Use resolve_nvme_ns.sh to discover the actual namespace and print it
# on stdout so the caller can capture it.
set -e

if [ $# -ne 1 ]; then
	echo "usage: bind_nvme.sh BDF" >&2
	exit 1
fi

BDF=$1
HERE=$(dirname "$0")

if [ ! -d "/sys/bus/pci/devices/$BDF" ]; then
	echo "error: PCI device $BDF not found in sysfs" >&2
	ls /sys/bus/pci/devices/ >&2
	exit 1
fi

# A HOMI stack leaked by an aborted prior run still owns this controller via
# upcie; tear it down first, or it contends with the kernel nvme driver bound
# below and wedges the device.
"$HERE/stop_homi_stack.sh"

DRIVER=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")" 2>/dev/null)
if [ "$DRIVER" != "nvme" ]; then
	if [ -n "$DRIVER" ]; then
		echo "unbinding $BDF from $DRIVER" >&2
		echo "$BDF" > "/sys/bus/pci/drivers/$DRIVER/unbind"
	fi

	# Clear any driver_override (e.g. uio_pci_generic from the owner path),
	# otherwise it pins the function away from nvme.
	echo > "/sys/bus/pci/devices/$BDF/driver_override" 2>/dev/null || true

	echo "resetting $BDF" >&2
	echo 1 > "/sys/bus/pci/devices/$BDF/reset"

	echo "binding $BDF to nvme" >&2
	echo "$BDF" > /sys/bus/pci/drivers/nvme/bind
	udevadm settle
fi

NS=$("$HERE/resolve_nvme_ns.sh" "$BDF") || {
	echo "available block devices:" >&2
	ls /dev/nvme* >&2 || true
	exit 1
}

if [ ! -b "$NS" ]; then
	echo "error: $NS resolved from $BDF is not a block device" >&2
	exit 1
fi

echo "$NS"
