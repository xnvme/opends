#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Tear down the HOMI/qublk stack and return the NVMe device to the kernel
# driver. Unmounts the ublk-backed filesystem, stops qublk and homid (a clean
# SIGTERM lets qublk run STOP_DEV/DEL_DEV), then rebinds nvme.
set -e

if [ $# -ne 2 ]; then
	echo "usage: homi_stack_down.sh BDF MOUNT" >&2
	exit 1
fi

BDF=$1
MOUNT=$2
HERE=$(dirname "$0")

umount "$MOUNT" 2>/dev/null || umount -l "$MOUNT" 2>/dev/null || true

# Match the bare binary name: the servers run as "qublk"/"homid" (PATH
# resolved), so pkill -f on an absolute path would never match and would leak
# them. A clean SIGTERM lets qublk run
# STOP_DEV/DEL_DEV before exiting.
pkill -TERM -x qublk 2>/dev/null || true
sleep 2
pkill -TERM -x homid 2>/dev/null || true
sleep 2
pkill -KILL -x qublk 2>/dev/null || true
pkill -KILL -x homid 2>/dev/null || true

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

rm -f /run/homi/ublk_dev /run/homi/*.desc /dev/shm/homid_dev*
echo "HOMI stack down: $BDF rebound to nvme"
