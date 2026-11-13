#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Reduce a running HOMI stack to homi alone: stop xal-server, unmount the file
# system and stop qublk. Raw block mode needs no more than homi, and nothing
# else may use the drive meanwhile. homi_stack_down.sh tears the rest down
# afterwards.
set -e

if [ $# -ne 1 ]; then
	echo "usage: homi_stack_bare.sh MOUNT" >&2
	exit 1
fi

MOUNT=$1

wait_gone() { # NAME SECONDS
	for _ in $(seq 1 "$2"); do
		pgrep -x "$1" > /dev/null || return 0
		sleep 1
	done
	echo "error: $1 did not exit" >&2
	exit 1
}

pkill -TERM -x xal-server 2>/dev/null || true
wait_gone xal-server 120
rm -f /dev/shm/xal_dev*

if mountpoint -q "$MOUNT"; then
	umount "$MOUNT"
fi

# SIGTERM lets qublk delete its ublk device; never kill it
pkill -TERM -x qublk 2>/dev/null || true
wait_gone qublk 60
rm -f /run/homi/ublk_dev

pgrep -x homi > /dev/null || { echo "error: homi is not running" >&2; exit 1; }
echo "bare stack: homi alone, nothing mounted, no index"
