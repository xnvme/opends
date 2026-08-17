#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# set (default): put the CPU in a stable bench mode, schedutil governor
# with boost disabled, saving the previous state. restore: put the saved
# state back. A reboot also reverts to the kernel default.
set -e

GOV=schedutil
STATE=/tmp/opends-bench-cpu-state

mode=${1:-set}

if [ "$mode" = restore ]; then
	if [ ! -f "$STATE" ]; then
		echo "no saved CPU state; nothing to restore"
		exit 0
	fi
	read -r gov boost < "$STATE"
	for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
		echo "$gov" > "$g"
	done
	if [ -n "$boost" ] && [ -w /sys/devices/system/cpu/cpufreq/boost ]; then
		echo "$boost" > /sys/devices/system/cpu/cpufreq/boost
	fi
	rm -f "$STATE"
	echo "restored governor=$gov, boost=${boost:-n/a}"
	exit 0
elif [ "$mode" != set ]; then
	echo "usage: $0 [set|restore]" >&2
	exit 1
fi

# Keep the oldest saved state if a previous run never restored.
if [ ! -f "$STATE" ]; then
	gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
	boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true)
	echo "$gov $boost" > "$STATE"
fi

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
	echo "$GOV" > "$g"
done

if [ -w /sys/devices/system/cpu/cpufreq/boost ]; then
	echo 0 > /sys/devices/system/cpu/cpufreq/boost
fi

echo "governor=$GOV, boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"
