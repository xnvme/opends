#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
set -e

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
	echo "usage: setup_meson_dep.sh REMOTE REF SRC_DIR BUILD_DIR [INSTALL_PREFIX]" >&2
	echo "  REF is a branch, tag, or commit hash; INSTALL_PREFIX defaults to /usr/local" >&2
	exit 1
fi

REMOTE=$1
REF=$2
SRC_DIR=$3
BUILD_DIR=$4
HERE=$(dirname "$0")
. "$HERE/install_prefix.sh" "${5:-}"

if [ ! -d "$SRC_DIR/.git" ]; then
	rm -rf "$SRC_DIR"
	git clone --recursive "$REMOTE" "$SRC_DIR"
fi

cd "$SRC_DIR"

if [ -n "$(git status --porcelain)" ]; then
	echo "error: dependency checkout is not clean: $SRC_DIR" >&2
	git status --short >&2
	echo "       commit, stash, or 'git -C $SRC_DIR reset --hard' it, then retry" >&2
	exit 1
fi

git remote set-url origin "$REMOTE"
git fetch --all --tags
# Prefer origin/REF so branch refs pick up the latest remote tip; fall
# back to REF directly for tags and commit hashes.
if git rev-parse --verify --quiet "origin/$REF^{commit}" > /dev/null; then
	git checkout --detach "origin/$REF"
else
	git checkout --detach "$REF"
fi
git submodule update --init --recursive

rm -rf "$BUILD_DIR"
meson setup "${MESON_ARGS[@]}" "$BUILD_DIR" "$SRC_DIR"
meson compile -C "$BUILD_DIR"
meson install -C "$BUILD_DIR"
if [ "$INSTALL_PREFIX" = /usr/local ]; then
	ldconfig
fi
