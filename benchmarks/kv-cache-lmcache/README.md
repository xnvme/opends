# KV cache with LMCache

Measures how long LMCache takes to store the KV cache of a request from GPU
memory on one NVMe drive and to read it back, with its GDS tier on NVIDIA
cuFile and on OpenDS. The benchmark is LMCache's own, `lmcache bench server`.
It needs no inference engine and no model, because it makes the KV cache of
each request itself. The `opends` backend comes from a fork of LMCache.
OpenDS itself does not depend on LMCache; only this benchmark fetches it.

`deps.toml` pins what is measured: the LMCache commit, and the torch build it
runs on. The other Python packages are the newest that LMCache's requirements
allow on the day of the install.

## What a run does

`run.py` installs LMCache at the pinned commit into a venv in the cache
directory, unless a venv made from the same pins is there already. For each
backend it starts `lmcache server` with a 16 GiB slab on the test drive as
the L1 tier, and runs `lmcache bench server` for 128 requests against it.
`cufile` runs while the kernel `nvme` driver owns the drive, with nvidia-fs
loaded. Then the HOMI stack comes up and `opends` runs. Then the slab is
removed and the stack brought down.

A request is 512 tokens, 64 MiB of KV cache in LMCache's default shape. The
bench stores each request once and retrieves it once, and compares the
checksums of what it reads back with what it stored; a mismatch fails the
run. LMCache times a store and a retrieve from the call to the server until
the data has arrived, so a time holds the request to the server and the I/O.
128 requests fill half the slab, so nothing is evicted.

Both slabs come from `fallocate`. LMCache makes the slab of the `cufile` leg.
The run makes the slab of the `opends` leg before the stack comes up, because
xal-server takes the extents of the files when it starts.

## Run it

With OpenDS installed on the target as for the other benchmarks:

```sh
benchmarks/kv-cache-lmcache/run.py
benchmarks/kv-cache-lmcache/report.py
```

The records land in `cijoe-output-bench/kv-cache-lmcache/`, the table in
`cijoe-output-bench/report.md`. In CI the `bench-kv-cache-lmcache` label
runs both.

The cache directory is `/var/cache/opends-bench` on the target, or
`model_cache` under `[test]` in `configs/test.toml`.
