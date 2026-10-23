#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# The build and test steps source this with the install prefix as the argument.
# It sets MESON_ARGS and the environment to install into that prefix, link
# against it and run what it holds.
INSTALL_PREFIX=${1:-/usr/local}
MESON_ARGS=()
if [ "$INSTALL_PREFIX" != /usr/local ]; then
	MESON_ARGS=(--prefix "$INSTALL_PREFIX" --libdir lib)
	export PATH="$INSTALL_PREFIX/bin:$PATH"
	export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
	export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
