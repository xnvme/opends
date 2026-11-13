#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Clone llama.cpp at REF and build llama-completion with the opends load mode
# against the OpenDS in INSTALL_PREFIX.
set -e

if [ $# -ne 5 ]; then
	echo "usage: build.sh REMOTE REF SRC_DIR BUILD_DIR INSTALL_PREFIX" >&2
	echo "  SRC_DIR and BUILD_DIR are deleted and made afresh" >&2
	exit 1
fi

REMOTE=$1
REF=$2
SRC_DIR=$3
BUILD_DIR=$4
INSTALL_PREFIX=$5

export PATH="/usr/local/cuda/bin:$PATH"

rm -rf "$SRC_DIR" "$BUILD_DIR"
git init -q "$SRC_DIR"
cd "$SRC_DIR"
git fetch -q --depth 1 "$REMOTE" "$REF"
git checkout -q --detach FETCH_HEAD

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
	-DLLAMA_CURL=OFF -DLLAMA_OPENDS=ON -DLLAMA_OPENDS_DIR="$INSTALL_PREFIX"
cmake --build "$BUILD_DIR" --target llama-completion -j "$(nproc)"
