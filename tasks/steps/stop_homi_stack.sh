#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Stop any running HOMI/qublk/xal-server stack and clear its state. A stack
# leaked by a crashed or aborted prior run (homi/qublk left owning the
# controller via upcie) contends with whatever claims the device next -- e.g.
# bind_nvme handing it to the kernel nvme driver -- and wedges it. Run this
# before claiming the controller.
#
# Best-effort and idempotent: never fails, and a no-op when nothing is running.
set -u

findmnt -rno TARGET,SOURCE 2>/dev/null | awk '$2 ~ /\/dev\/ublkb/ { print $1 }' |
	while read -r mnt; do
		umount "$mnt" 2>/dev/null || umount -l "$mnt" 2>/dev/null || true
	done

# Kill order: xal-server first (it unlinks its shm on a clean exit and readers
# must be done before then), qublk before homi so the ublk device is gone
# before the controller it forwards to. A clean SIGTERM lets qublk run
# STOP_DEV/DEL_DEV before exiting; escalate to KILL if it does not. Match the
# bare binary names (the servers run PATH-resolved, so -f on an absolute path
# would never match). "homid" is the daemon of the old stack; clear it too so
# a leftover from an older build cannot hold the controller.
pkill -TERM -x xal-server 2>/dev/null || true
pkill -TERM -x qublk 2>/dev/null || true
sleep 2
pkill -KILL -x qublk 2>/dev/null || true
pkill -TERM -x homi 2>/dev/null || true
pkill -TERM -x homid 2>/dev/null || true
sleep 1
pkill -KILL -x xal-server 2>/dev/null || true
pkill -KILL -x homi 2>/dev/null || true
pkill -KILL -x homid 2>/dev/null || true

rm -f /dev/shm/xal_dev* /dev/shm/homid_dev*

# A killed primary leaves its multi-process segments behind, and the next
# server would join the dead group rather than electing itself primary.
rm -f /dev/shm/xnvme-upcie* /tmp/xnvme-upcie-flock-* /tmp/xnvme-homi-*.sock

# Reload ublk_drv to drop stale device ids a killed qublk may have left behind.
rmmod ublk_drv 2>/dev/null || true
modprobe ublk_drv 2>/dev/null || true
