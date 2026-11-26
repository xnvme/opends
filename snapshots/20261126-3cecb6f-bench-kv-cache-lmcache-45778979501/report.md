## KV cache with LMCache

`lmcache bench server`, LMCache `4c0111d`, 128 requests of 512 tokens per backend. A store writes the KV cache of a request from GPU memory to the slab, a retrieve reads it back.

| `--gds-l1-backend` | Store p50 (ms) | Store p99 (ms) | Retrieve p50 (ms) | Retrieve p99 (ms) |
|---|---|---|---|---|
| opends | 23.9 | 28.7 | 12.0 | 12.7 |
| cufile | 28.0 | 33.2 | 31.4 | 33.3 |
