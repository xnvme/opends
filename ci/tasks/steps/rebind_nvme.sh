#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
set -e

if [ $# -ne 1 ]; then
	echo "usage: rebind_nvme.sh BDF" >&2
	exit 1
fi

BDF=$1

CUR=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")" 2>/dev/null)
if [ "$CUR" = nvme ]; then
	echo "$BDF already bound to nvme"
	exit 0
fi

# Release the function from the owner path's uio_pci_generic binding and clear
# the driver_override, otherwise the override pins it away from nvme.
if [ -n "$CUR" ]; then
	echo "$BDF" > "/sys/bus/pci/drivers/$CUR/unbind"
fi
echo > "/sys/bus/pci/devices/$BDF/driver_override" 2>/dev/null || true

echo "rebinding $BDF to nvme"
echo "$BDF" > /sys/bus/pci/drivers/nvme/bind

if [ ! -e "/sys/bus/pci/devices/$BDF/driver" ]; then
	echo "error: $BDF has no driver after bind"
	exit 1
fi
DRIVER=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver")")
if [ "$DRIVER" != "nvme" ]; then
	echo "error: $BDF bound to $DRIVER, expected nvme"
	exit 1
fi
echo "$BDF bound to nvme"
