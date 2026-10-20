#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Tear down what fabrics_loop_up.sh set up, tolerating a partial or absent
# loop: xal-server, the mount at MOUNT, the nvme-rdma connection, the nvmet
# port and subsystem, and the Soft-RoCE link. Modules stay loaded.
set -u

if [ $# -ne 1 ]; then
	echo "usage: fabrics_loop_down.sh MOUNT" >&2
	exit 1
fi

MOUNT=$1
NQN=nqn.2014-08.org.nvmexpress:uuid:3f9d2e6c-0c5b-4b1e-9b7a-6d2f1c8a4e10
CFG=/sys/kernel/config/nvmet

pkill -TERM -x xal-server 2>/dev/null || true
sleep 1
pkill -KILL -x xal-server 2>/dev/null || true
rm -f /dev/shm/xal_dev*

umount "$MOUNT" 2>/dev/null || umount -l "$MOUNT" 2>/dev/null || true

nvme disconnect -n "$NQN" > /dev/null 2>&1 || true

if [ -d "$CFG/ports/1" ]; then
	rm -f "$CFG/ports/1/subsystems/$NQN"
	rmdir "$CFG/ports/1" 2>/dev/null || true
fi
if [ -d "$CFG/subsystems/$NQN" ]; then
	echo 0 > "$CFG/subsystems/$NQN/namespaces/1/enable" 2>/dev/null || true
	rmdir "$CFG/subsystems/$NQN/namespaces/1" 2>/dev/null || true
	rmdir "$CFG/subsystems/$NQN" 2>/dev/null || true
fi

for link in $(rdma link show 2>/dev/null | awk '{print $2}' | sed 's#/.*##' | sort -u); do
	case "$link" in
	rxe_*) rdma link del "$link" 2>/dev/null || true ;;
	esac
done
