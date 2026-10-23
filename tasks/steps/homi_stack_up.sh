#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Bring up the HOMI/qublk stack so the aisio backend can attach qpairs and read
# files on a mounted filesystem. Hands the NVMe controller to userspace (HOMI),
# exposes it as a ublk block device (qublk), and mounts the existing XFS over it
# at MOUNT. The same XFS is what the ref/gds phases mounted via the kernel
# driver, so any file written there (e.g. the sync-read pattern) is still
# present after the remount.
set -e

if [ $# -ne 2 ]; then
	echo "usage: homi_stack_up.sh BDF MOUNT" >&2
	exit 1
fi

BDF=$1
MOUNT=$2
HERE=$(dirname "$0")
SOCK=/run/homi/homi.sock
CONF=/run/homi/homi-test.conf

mkdir -p /run/homi

# Clear any stale stack from a crashed or aborted prior run before bringing up
# a fresh one.
"$HERE/stop_homi_stack.sh"

# Hand the controller to userspace (unmounts the kernel mount, unbinds nvme,
# leaves the controller in a clean power-on state).
"$HERE/unbind_nvme.sh" "$BDF" "$MOUNT"

# xal config; mirror the libxal.h enums (homid parses these as TOML integers).
# shellcheck disable=SC2034  # full enum sets defined for clarity; only some used
XAL_BACKEND_XFS=1
XAL_BACKEND_FIEMAP=2
XAL_WATCHMODE_NONE=0
XAL_WATCHMODE_DIRTY_DETECTION=1
XAL_WATCHMODE_EXTENT_UPDATE=2
XAL_FILE_LOOKUPMODE_TRAVERSE=0
XAL_FILE_LOOKUPMODE_HASHMAP=1
cat > "$CONF" <<EOF
log_level = 2
devices = [ "$BDF" ]
ipc_socket = "$SOCK"

[xal]
backend = $XAL_BACKEND_FIEMAP
watchmode = $XAL_WATCHMODE_DIRTY_DETECTION
file_lookupmode = $XAL_FILE_LOOKUPMODE_TRAVERSE
mountpoint = "$MOUNT"
EOF
rm -f "$SOCK" /run/homi/*.desc /dev/shm/homid_dev*

echo "starting homid"
# Diagnostic: let the daemons dump a core on crash (paired with a file
# core_pattern). Harmless when cores are disabled by policy.
ulimit -c unlimited 2>/dev/null || true
# Detach the daemon fully: redirect on the outer command so neither the daemon
# nor any wrapper keeps this step's stdout/stderr open.
setsid homid --config "$CONF" < /dev/null > /run/homi/homid.log 2>&1 &
# homid is launched via a bash wrapper, so the named process appears a moment
# after the '&'. Wait for it to show up before watching for it to die.
for _ in $(seq 1 20); do
	pgrep -x homid > /dev/null && break
	sleep 0.2
done
# Wait for the listening socket, bailing if homid dies. homid binds the socket
# only after owning the controller and indexing xal (a few seconds), so the
# socket's appearance means it is ready to serve qpair-attach and xal requests.
for _ in $(seq 1 120); do
	[ -S "$SOCK" ] && break
	if ! pgrep -x homid > /dev/null; then
		echo "error: homid exited during startup" >&2
		journalctl -t homi -t homid -n 20 --no-pager -o short-iso >&2
		exit 1
	fi
	sleep 0.5
done
if [ ! -S "$SOCK" ]; then
	echo "error: homid socket did not come up" >&2
	journalctl -t homi -t homid -n 20 --no-pager -o short-iso >&2
	exit 1
fi

echo "starting qublk"
setsid qublk "$BDF" --homi "$SOCK" --nr-queues 1 < /dev/null > /run/homi/qublk.log 2>&1 &
UBLK=""
for _ in $(seq 1 60); do
	UBLK=$(grep -oE '/dev/ublkb[0-9]+' /run/homi/qublk.log 2>/dev/null | head -1)
	[ -n "$UBLK" ] && [ -b "$UBLK" ] && break
	sleep 0.3
done
if [ -z "$UBLK" ] || [ ! -b "$UBLK" ]; then
	echo "error: qublk did not expose a ublk device" >&2
	cat /run/homi/qublk.log >&2 || true
	exit 1
fi
echo "$UBLK" > /run/homi/ublk_dev

mkdir -p "$MOUNT"
mount "$UBLK" "$MOUNT"

for _ in $(seq 1 120); do
	[ -e /dev/shm/homid_dev0.ready ] && break
	sleep 0.5
done
if [ ! -e /dev/shm/homid_dev0.ready ]; then
	echo "error: homid did not index xal after mount" >&2
	exit 1
fi

echo "HOMI stack up: homid + qublk ($UBLK) mounted at $MOUNT"
