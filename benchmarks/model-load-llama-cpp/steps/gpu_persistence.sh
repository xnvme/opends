#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# set (default): turn GPU persistence mode on, saving the previous state.
# restore: put the saved state back. Without the mode every process on the
# kernel NVMe driver pays the GPU's start-up, which a process does not pay
# while the HOMI stack holds the GPU open.
set -e

STATE=/tmp/opends-bench-gpu-persistence

mode=${1:-set}

if [ "$mode" = restore ]; then
	if [ ! -f "$STATE" ]; then
		echo "no saved persistence mode; nothing to restore"
		exit 0
	fi
	nvidia-smi --persistence-mode="$(cat "$STATE")" > /dev/null
	echo "restored persistence mode $(cat "$STATE")"
	rm -f "$STATE"
	exit 0
elif [ "$mode" != set ]; then
	echo "usage: $0 [set|restore]" >&2
	exit 1
fi

# Keep the oldest saved state if a previous run never restored.
if [ ! -f "$STATE" ]; then
	case "$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)" in
		Enabled) echo 1 > "$STATE" ;;
		*) echo 0 > "$STATE" ;;
	esac
fi
nvidia-smi --persistence-mode=1 > /dev/null
echo "persistence mode on (was $(cat "$STATE"))"
