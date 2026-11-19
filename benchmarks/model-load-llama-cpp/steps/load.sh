#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Load MODEL RUNS times, each in a new process from a cold page cache. After
# llama.cpp's own output of a run comes its wall time as real_ms=<n>.
set -e

if [ $# -ne 4 ]; then
	echo "usage: load.sh LLAMA_COMPLETION MODEL MODE RUNS" >&2
	exit 1
fi

echo "gpu_persistence_mode=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)"
for i in $(seq "$4"); do
	echo "--- $3 [$i/$4] ---"
	sync
	echo 3 > /proc/sys/vm/drop_caches

	start=$EPOCHREALTIME
	"$1" -m "$2" --load-mode "$3" -ngl 99 -ot token_embd.weight=CUDA0 \
		-c 512 -n 1 -st --fit off -p hello 2>&1
	end=$EPOCHREALTIME
	echo "real_ms=$(( (${end/[.,]/} - ${start/[.,]/}) / 1000 ))"
done
