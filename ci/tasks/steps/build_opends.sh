#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
set -e

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
	echo "usage: build_opends.sh SRC_DIR BUILD_DIR [INSTALL_PREFIX]" >&2
	exit 1
fi

SRC_DIR=$1
BUILD_DIR=$2
HERE=$(dirname "$0")
. "$HERE/install_prefix.sh" "${3:-}"

if [ ! -f "$SRC_DIR/meson.build" ]; then
	echo "error: $SRC_DIR/meson.build not found"
	exit 1
fi

rm -rf "$BUILD_DIR"
cd "$SRC_DIR"
meson setup "${MESON_ARGS[@]}" "$BUILD_DIR"
meson compile -C "$BUILD_DIR"
meson install -C "$BUILD_DIR"
if [ "$INSTALL_PREFIX" = /usr/local ]; then
	ldconfig
fi
