#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
set -e

if [ $# -ne 2 ]; then
	echo "usage: mount_nvme.sh BDF MOUNT" >&2
	exit 1
fi

BDF=$1
MOUNT=$2
HERE=$(dirname "$0")

NS=$("$HERE/resolve_nvme_ns.sh" "$BDF") || exit 1

if [ ! -b "$NS" ]; then
	echo "error: $NS is not a block device" >&2
	exit 1
fi

if [ "$(blkid -o value -s TYPE "$NS" 2>/dev/null)" != "xfs" ]; then
	echo "error: no xfs filesystem on $NS; format and populate it first (aisio's setup_dataset.yaml)" >&2
	exit 1
fi

umount -l "$MOUNT" 2>/dev/null || true
mkdir -p "$MOUNT"

mount "$NS" "$MOUNT"
echo "$NS mounted on $MOUNT"
