#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Load MODEL RUNS times, each in a new process from a cold page cache. After
# llama.cpp's own output of a run comes its wall time as real_ms=<n>. In
# opends mode a run fails unless OpenDS read all the tensor data.
set -e

if [ $# -ne 4 ]; then
	echo "usage: load.sh LLAMA_COMPLETION MODEL MODE RUNS" >&2
	exit 1
fi

# the loader falls back to its own direct I/O without an error, only its debug lines tell
verbose=()
if [ "$3" = opends ]; then
	verbose=(--verbose)
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT

echo "gpu_persistence_mode=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)"
for i in $(seq "$4"); do
	echo "--- $3 [$i/$4] ---"
	sync
	echo 3 > /proc/sys/vm/drop_caches

	rc=0
	start=$EPOCHREALTIME
	"$1" -m "$2" --load-mode "$3" -ngl 99 -ot token_embd.weight=CUDA0 \
		-c 512 -n 1 -st --fit off -p hello "${verbose[@]}" > "$log" 2>&1 || rc=$?
	end=$EPOCHREALTIME
	# of the debug lines, print only the loader's byte counts
	awk '!/^[0-9.]+ D / || / D load_all_data: /' "$log"
	if [ "$rc" -ne 0 ]; then
		exit "$rc"
	fi

	if [ "$3" = opends ]; then
		via=$(grep -oE 'opends [0-9]+ ops \([0-9]+ tensors\) [0-9.]+ MiB' "$log" | awk '{ s += $(NF - 1) } END { printf "%.1f", s }')
		all=$(grep -oE ': [0-9.]+ MiB in [0-9.]+ ms' "$log" | awk '{ s += $2 } END { printf "%.1f", s }')
		if [ "$all" = 0.0 ] || [ "$via" != "$all" ]; then
			echo "error: OpenDS read $via MiB of the $all MiB that llama.cpp loaded" >&2
			exit 1
		fi
	fi
	echo "real_ms=$(( (${end/[.,]/} - ${start/[.,]/}) / 1000 ))"
done
