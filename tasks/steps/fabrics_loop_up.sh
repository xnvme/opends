#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Bring up a single-host NVMe-oF loop so the aisio backend can read a mounted
# filesystem over fabrics without a homi stack: a Soft-RoCE link on NETDEV, a
# kernel nvmet target exporting BLOCKDEV over RDMA, a kernel nvme-rdma
# connection to that target, the existing XFS on it mounted at MOUNT from that
# connection, and an xal-server publishing the mount's extent index. The last
# line printed is the OPENDS_AISIO_FABRICS value that reaches the same target
# through xNVMe's SPDK backend, as a second controller on the subsystem.
# TRTYPE tcp skips the Soft-RoCE link and exports over NVMe/TCP instead, to
# check everything but the RDMA transport on a host that cannot load rxe.
set -e

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
	echo "usage: fabrics_loop_up.sh NETDEV BLOCKDEV MOUNT [TRTYPE]" >&2
	exit 1
fi

NETDEV=$1
BLOCKDEV=$2
MOUNT=$3
TRTYPE=${4:-rdma}
HERE=$(dirname "$0")
NQN=nqn.2014-08.org.nvmexpress:uuid:3f9d2e6c-0c5b-4b1e-9b7a-6d2f1c8a4e10
PORT=4420
RUN=/run/opends-fabrics
CFG=/sys/kernel/config/nvmet
XAL_SHM=/xal_dev0

die() {
	echo "error: $1" >&2
	[ -n "${2:-}" ] && cat "$2" >&2 || true
	exit 1
}

mkdir -p "$RUN"

"$HERE/fabrics_loop_down.sh" "$MOUNT"

if findmnt -rn -S "$BLOCKDEV" > /dev/null; then
	die "$BLOCKDEV is mounted; unmount it before exporting it"
fi
if [ "$(blkid -o value -s TYPE "$BLOCKDEV" 2>/dev/null)" != "xfs" ]; then
	die "no xfs filesystem on $BLOCKDEV"
fi

ADDR=$(ip -4 -o addr show dev "$NETDEV" | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$ADDR" ] || die "$NETDEV has no IPv4 address"

if [ "$TRTYPE" = rdma ]; then
	# A host whose RDMA core is not the in-tree one (Mellanox OFED) has to
	# load an rxe built against it beforehand; modprobe then has nothing to
	# add.
	if ! lsmod | grep -q "^rdma_rxe "; then
		modprobe rdma_rxe
	fi
	if ! rdma link show 2>/dev/null | grep -q "netdev $NETDEV"; then
		rdma link add "rxe_$NETDEV" type rxe netdev "$NETDEV"
	fi
fi
modprobe nvmet
modprobe "nvmet_$TRTYPE"
modprobe "nvme_$TRTYPE"

mkdir -p "$CFG/subsystems/$NQN"
echo 1 > "$CFG/subsystems/$NQN/attr_allow_any_host"
mkdir -p "$CFG/subsystems/$NQN/namespaces/1"
echo "$BLOCKDEV" > "$CFG/subsystems/$NQN/namespaces/1/device_path"
echo 1 > "$CFG/subsystems/$NQN/namespaces/1/enable"
mkdir -p "$CFG/ports/1"
echo "$TRTYPE" > "$CFG/ports/1/addr_trtype"
echo ipv4 > "$CFG/ports/1/addr_adrfam"
echo "$ADDR" > "$CFG/ports/1/addr_traddr"
echo "$PORT" > "$CFG/ports/1/addr_trsvcid"
ln -sf "$CFG/subsystems/$NQN" "$CFG/ports/1/subsystems/$NQN"

nvme connect -t "$TRTYPE" -a "$ADDR" -s "$PORT" -n "$NQN"

# The connection's namespace shows up under the subsystem NQN once udev has
# settled.
NS=""
for _ in $(seq 1 50); do
	for blk in /sys/block/nvme*; do
		[ -e "$blk/device/subsysnqn" ] || continue
		if [ "$(cat "$blk/device/subsysnqn")" = "$NQN" ]; then
			NS="/dev/$(basename "$blk")"
		fi
	done
	[ -n "$NS" ] && [ -b "$NS" ] && break
	sleep 0.2
done
[ -n "$NS" ] && [ -b "$NS" ] || die "no namespace appeared for $NQN"
echo "$NS" > "$RUN/ns"

mkdir -p "$MOUNT"
mount "$NS" "$MOUNT"

# watchmode 2 = extent update: the server re-indexes on filesystem changes,
# which is what makes the extents of freshly written files resolvable.
cat > "$RUN/xal-server.conf" <<CONF
log_level = 2
devices = [
  { uri = "$NS", shm_name = "$XAL_SHM", mountpoint = "$MOUNT" },
]

[xal]
watchmode = 2
CONF

echo "starting xal-server"
setsid xal-server --config "$RUN/xal-server.conf" < /dev/null > "$RUN/xal-server.log" 2>&1 &
for _ in $(seq 1 120); do
	[ -e "/dev/shm${XAL_SHM}_state" ] && break
	if ! pgrep -x xal-server > /dev/null; then
		die "xal-server exited during startup" "$RUN/xal-server.log"
	fi
	sleep 0.5
done
if [ ! -e "/dev/shm${XAL_SHM}_state" ]; then
	die "xal-server did not publish $XAL_SHM" "$RUN/xal-server.log"
fi

echo "fabrics loop up: $BLOCKDEV as $NQN on $TRTYPE $ADDR:$PORT, $NS mounted at $MOUNT"
echo "OPENDS_AISIO_FABRICS=$ADDR:$PORT/$NQN"
