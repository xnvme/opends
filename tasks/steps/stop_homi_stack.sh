#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Stop any running HOMI/qublk stack and clear its ublk state. A stack leaked by
# a crashed or aborted prior run (homid/qublk left owning the controller via
# upcie) contends with whatever claims the device next -- e.g. bind_nvme handing
# it to the kernel nvme driver -- and wedges it. Run this before claiming the
# controller.
#
# Best-effort and idempotent: never fails, and a no-op when nothing is running.
set -u

findmnt -rno TARGET,SOURCE 2>/dev/null | awk '$2 ~ /\/dev\/ublkb/ { print $1 }' |
	while read -r mnt; do
		umount "$mnt" 2>/dev/null || umount -l "$mnt" 2>/dev/null || true
	done

# A clean SIGTERM lets qublk run STOP_DEV/DEL_DEV before exiting; escalate to
# KILL if it does not. Match the bare binary name (the servers run PATH-resolved
# as "qublk"/"homid", so -f on an absolute path would never match).
pkill -TERM -x qublk 2>/dev/null || true
pkill -TERM -x homid 2>/dev/null || true
sleep 2
pkill -KILL -x qublk 2>/dev/null || true
pkill -KILL -x homid 2>/dev/null || true

rm -f /run/homi/*.desc /dev/shm/homid_dev*

# Reload ublk_drv to drop stale device ids a killed qublk may have left behind.
rmmod ublk_drv 2>/dev/null || true
modprobe ublk_drv 2>/dev/null || true
