#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Start an LMCache server whose L1 tier is a GDS slab in SLAB_DIR, run
# `lmcache bench server` against it and stop it. The JSON summary of the
# bench comes after the line "--- result ---", the server's log after it.
set -e

if [ $# -ne 5 ]; then
	echo "usage: bench_server.sh VENV BACKEND SLAB_DIR SLAB_GB REQUESTS" >&2
	exit 1
fi

LMCACHE=$1/bin/lmcache
BACKEND=$2
SLAB_DIR=$3
SLAB_GB=$4
REQUESTS=$5
RPC_PORT=15556
HTTP_PORT=18080

export LMCACHE_DISABLE_BANNER=1

log=$(mktemp)
out=$(mktemp)

"$LMCACHE" server --transport grpc --port "$RPC_PORT" --http-port "$HTTP_PORT" \
	--l1-size-gb "$SLAB_GB" --eviction-policy LRU \
	--gds-l1-path "$SLAB_DIR" --gds-l1-backend "$BACKEND" > "$log" 2>&1 &
server=$!

stop_server() {
	kill "$server" 2>/dev/null || true
	for _ in $(seq 30); do
		kill -0 "$server" 2>/dev/null || break
		sleep 1
	done
	# a server that hangs would keep the slab open and block the unmount
	kill -9 "$server" 2>/dev/null || true
	wait "$server" 2>/dev/null || true
	echo "--- server log ---"
	cat "$log"
	rm -f "$log" "$out"
}
trap stop_server EXIT

ready=0
for _ in $(seq 120); do
	kill -0 "$server" 2>/dev/null || break
	if curl --fail --silent --output /dev/null "http://localhost:$HTTP_PORT/healthcheck"; then
		ready=1
		break
	fi
	sleep 1
done
if [ "$ready" -ne 1 ]; then
	echo "error: the LMCache server did not come up" >&2
	exit 1
fi

# --url "": no checksum requests over HTTP, the bench still checks what it reads back
"$LMCACHE" bench server --rpc-url "grpc://localhost:$RPC_PORT" --url "" \
	--end "$REQUESTS" --format json --output "$out" --quiet
echo "--- result ---"
cat "$out"
