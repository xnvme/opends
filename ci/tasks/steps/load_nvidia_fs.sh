#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Ensure nvidia-fs.ko is loaded so cuFile uses the real P2P path
# (NVMe -> GPU via nvidia-fs) instead of CPU-staged compat mode.
set -e

if lsmod | grep -q '^nvidia_fs '; then
	echo "nvidia_fs already loaded"
	exit 0
fi

modprobe nvidia_fs
echo "nvidia_fs loaded"
