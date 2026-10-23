#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Fail unless the stack is down: no HOMI process running, no ublk device
# mounted, and the NVMe controller on the kernel driver. Run it before
# installing or claiming anything, so a run on a shared machine stops instead
# of replacing the libraries and daemons someone else is using.
set -u

if [ $# -ne 1 ]; then
	echo "usage: check_stack_down.sh BDF" >&2
	exit 1
fi

BDF=$1
busy=0

for name in homid homi qublk xal-server; do
	if pids=$(pgrep -d, -x "$name"); then
		echo "error: $name is running (pid $pids)" >&2
		busy=1
	fi
done

mnts=$(findmnt -rno TARGET,SOURCE 2>/dev/null | awk '$2 ~ /\/dev\/ublkb/ { print $1 }')
if [ -n "$mnts" ]; then
	echo "error: mounted over ublk: $mnts" >&2
	busy=1
fi

driver=$(basename "$(readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null)")
if [ "$driver" != "nvme" ]; then
	echo "error: $BDF is bound to '${driver:-nothing}', not nvme" >&2
	busy=1
fi

if [ "$busy" -ne 0 ]; then
	echo "error: the stack is up on $(hostname); bring it down and run again" >&2
	exit 1
fi
echo "stack down on $(hostname): $BDF on nvme"
