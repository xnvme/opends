#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Tear down the HOMI/qublk/xal-server stack and return the NVMe device to the
# kernel driver. clear_homi_stack.sh stops the stack; this rebinds nvme after it.
set -e

if [ $# -ne 2 ]; then
	echo "usage: homi_stack_down.sh BDF MOUNT" >&2
	exit 1
fi

BDF=$1
MOUNT=$2
HERE=$(dirname "$0")

"$HERE/clear_homi_stack.sh"

# Rebind nvme; recover with a PCI remove+rescan if a wedged userspace owner
# left the controller in a bad state.
for _ in 1 2 3; do
	if "$HERE/rebind_nvme.sh" "$BDF"; then
		break
	fi
	echo 1 > "/sys/bus/pci/devices/$BDF/remove" 2>/dev/null || true
	echo 1 > /sys/bus/pci/rescan 2>/dev/null || true
	sleep 1
done

rm -f /run/homi/ublk_dev
echo "HOMI stack down: $BDF rebound to nvme"
