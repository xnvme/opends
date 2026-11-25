#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Put LMCache at REF in a venv under DIR, with torch TORCH from TORCH_INDEX
# and LMCache's CUDA kernels. A venv made from the same pins is kept.
set -e

if [ $# -ne 5 ]; then
	echo "usage: install.sh REMOTE REF TORCH TORCH_INDEX DIR" >&2
	echo "  DIR is deleted and made afresh unless it holds these pins" >&2
	exit 1
fi

REMOTE=$1
REF=$2
TORCH=$3
TORCH_INDEX=$4
DIR=$5
PINS="$REMOTE $REF torch==$TORCH $TORCH_INDEX"

if [ "$(cat "$DIR/pins" 2>/dev/null)" = "$PINS" ]; then
	echo "$DIR: cached"
	exit 0
fi

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

rm -rf "$DIR"
git init -q "$DIR/src"
cd "$DIR/src"
git fetch -q --depth 1 "$REMOTE" "$REF"
git checkout -q --detach FETCH_HEAD

python3 -m venv "$DIR/venv"
pip_install() {
	"$DIR/venv/bin/pip" install --quiet --no-cache-dir "$@"
}
pip_install --index-url "$TORCH_INDEX" "torch==$TORCH"
pip_install -r requirements/build.txt

# LMCache builds for CUDA 13 unless told the major of the installed torch
LMCACHE_CUDA_MAJOR=$("$DIR/venv/bin/python" -c 'import torch; print(torch.version.cuda.split(".")[0])')
export LMCACHE_CUDA_MAJOR
# a shallow clone has no tag to take the version from
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0
pip_install --no-build-isolation .

echo "$PINS" > "$DIR/pins"
echo "$DIR: installed"
