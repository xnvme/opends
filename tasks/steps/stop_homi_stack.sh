#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Stop a running HOMI/qublk/xal-server stack and clear its state. Run it before
# claiming the controller: a leaked homi/qublk wedges whatever claims it next.
#
# The order matters. xal-server holds the mount, the mount must be gone before
# qublk stops, and qublk must exit on its own: a lazy unmount or a killed qublk
# leaves ublk I/O in flight until a reboot. So the script fails, with the stack
# left as it is, when the filesystem will not unmount or qublk will not exit.
# A no-op when nothing is running.
set -u

wait_gone() { # NAME SECONDS -> 0 when no process named NAME remains
	for _ in $(seq 1 "$2"); do
		pgrep -x "$1" > /dev/null || return 0
		sleep 1
	done
	! pgrep -x "$1" > /dev/null
}

# 1. xal-server; a stop during an index waits for the index, so give it time
pkill -TERM -x xal-server 2>/dev/null
if ! wait_gone xal-server 120; then
	echo "warning: xal-server ignored SIGTERM for 120 s, killing it" >&2
	pkill -KILL -x xal-server 2>/dev/null
	wait_gone xal-server 10
fi

# 2. the ublk mounts, plain unmounts only
for _ in $(seq 1 30); do
	mnts=$(findmnt -rno TARGET,SOURCE 2>/dev/null | awk '$2 ~ /\/dev\/ublkb/ { print $1 }')
	[ -z "$mnts" ] && break
	for mnt in $mnts; do
		umount "$mnt" 2>/dev/null || true
	done
	sleep 2
done
mnts=$(findmnt -rno TARGET,SOURCE 2>/dev/null | awk '$2 ~ /\/dev\/ublkb/ { print $1 }')
if [ -n "$mnts" ]; then
	echo "error: still mounted over ublk after 60 s: $mnts" >&2
	echo "       holders: $(fuser -vm $mnts 2>&1 | tr '\n' ' ')" >&2
	exit 1
fi

# 3. qublk; SIGTERM lets it run STOP_DEV/DEL_DEV, never SIGKILL it
pkill -TERM -x qublk 2>/dev/null
if ! wait_gone qublk 60; then
	echo "error: qublk did not exit within 60 s of SIGTERM; not killing it" >&2
	exit 1
fi

# 4. homi, and homid of the old stack
pkill -TERM -x homi 2>/dev/null
pkill -TERM -x homid 2>/dev/null
if ! wait_gone homi 30 || ! wait_gone homid 5; then
	pkill -KILL -x homi 2>/dev/null
	pkill -KILL -x homid 2>/dev/null
fi

rm -f /dev/shm/xal_dev* /dev/shm/homid_dev*

# A killed primary leaves its multi-process segments behind, and the next
# server would join the dead group rather than electing itself primary.
rm -f /dev/shm/xnvme-upcie* /tmp/xnvme-upcie-flock-* /tmp/xnvme-homi-*.sock

# reload ublk_drv to drop stale device ids a dead qublk left behind
if ! ls /dev/ublkb* > /dev/null 2>&1; then
	rmmod ublk_drv 2>/dev/null || true
	modprobe ublk_drv 2>/dev/null || true
fi
