#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Put the model in CACHE_DIR, downloading it only when the file there is
# missing or does not match SHA256.
set -e

if [ $# -ne 4 ]; then
	echo "usage: fetch_model.sh URL SHA256 CACHE_DIR NAME" >&2
	exit 1
fi

URL=$1
SHA256=$2
CACHE_DIR=$3
FILE=$CACHE_DIR/$4

matches() {
	[ -f "$FILE" ] && echo "$SHA256  $FILE" | sha256sum --check --status
}

mkdir -p "$CACHE_DIR"
if matches; then
	echo "$FILE: cached"
	exit 0
fi

curl --fail --location --retry 3 --silent --show-error --output "$FILE.part" "$URL"
mv "$FILE.part" "$FILE"
if ! matches; then
	echo "error: $FILE does not match sha256 $SHA256" >&2
	exit 1
fi
echo "$FILE: downloaded"
