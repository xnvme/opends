#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
set -e

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
	echo "usage: setup_meson_dep.sh REMOTE REF SRC_DIR BUILD_DIR [INSTALL_PREFIX]" >&2
	echo "  REF is a branch, tag, or commit hash; INSTALL_PREFIX defaults to /usr/local" >&2
	echo "  SRC_DIR and BUILD_DIR are deleted and made afresh" >&2
	exit 1
fi

REMOTE=$1
REF=$2
SRC_DIR=$3
BUILD_DIR=$4
HERE=$(dirname "$0")
. "$HERE/install_prefix.sh" "${5:-}"

rm -rf "$SRC_DIR" "$BUILD_DIR"
git clone "$REMOTE" "$SRC_DIR"
cd "$SRC_DIR"
# origin/REF for a branch; a tag or a commit hash is REF itself.
if git rev-parse --verify --quiet "origin/$REF^{commit}" > /dev/null; then
	git checkout --detach "origin/$REF"
else
	git checkout --detach "$REF"
fi
git submodule update --init --recursive

meson setup "${MESON_ARGS[@]}" "$BUILD_DIR" "$SRC_DIR"
meson compile -C "$BUILD_DIR"
meson install -C "$BUILD_DIR"
if [ "$INSTALL_PREFIX" = /usr/local ]; then
	ldconfig
fi
