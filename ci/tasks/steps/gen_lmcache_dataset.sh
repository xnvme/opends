#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Generate a synthetic LMCache-shaped dataset: N files, each laid out like
# LMCache's GdsBackend writes a KV chunk -- a metadata header
# (_METADATA_MAX_SIZE = 4096) followed by the KV blob at offset 4096. KV size
# defaults to 14 MiB (Qwen-7B, 256-token chunk: 2*28*4*128*2*256 = 14680064
# bytes, 4096-aligned), so the whole file is 14684160 bytes = 3585*4096, also
# 4096-aligned -- reads are aligned in offset and size, matching real LMCache
# (unlike tiktokish's arbitrary sizes, which force cuFile's unaligned bounce
# path).
#
# Files are spread across subdirectories (data-dir/classNNN/chunk_*.kvcache) so
# filperf can enumerate them via xal (it expects a 2-level layout, like the
# tiktokish/imagenetish datasets). Idempotent: skips generation when the layout
# already holds exactly N files.
#
# args: <dir> [n_files] [kv_bytes] [n_subdirs]
set -euo pipefail

DIR=${1:?dir}
N=${2:-256}
KV=${3:-14680064}
SUBDIRS=${4:-8}
HDR=4096
TOTAL=$((HDR + KV))

[ $((KV % 4096)) -eq 0 ] || { echo "kv_bytes ($KV) must be 4096-aligned"; exit 1; }

mkdir -p "$DIR"
have=$(find "$DIR" -mindepth 2 -name 'chunk_*.kvcache' 2>/dev/null | wc -l)
if [ "$have" -eq "$N" ]; then
	exit 0
else
	rm -rf "${DIR:?}"/*  # clear any stale/flat layout
	echo "generating $N x $TOTAL bytes (~$((TOTAL / 1024 / 1024)) MiB KV + 4K hdr) across $SUBDIRS dirs in $DIR"
	base=$(mktemp)
	dd if=/dev/urandom of="$base" bs=4096 count=$((TOTAL / 4096)) status=none
	per=$(((N + SUBDIRS - 1) / SUBDIRS))
	k=0
	for s in $(seq 0 $((SUBDIRS - 1))); do
		sd=$(printf '%s/class%03d' "$DIR" "$s")
		mkdir -p "$sd"
		for _ in $(seq 1 "$per"); do
			[ "$k" -ge "$N" ] && break
			cp "$base" "$(printf '%s/chunk_%05d.kvcache' "$sd" "$k")"
			k=$((k + 1))
		done
	done
	rm -f "$base"
fi
