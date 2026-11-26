## KV cache with LMCache

`lmcache bench server`, LMCache `4c0111d`, 128 requests of 512 tokens per backend. A store writes the KV cache of a request from GPU memory to the slab, a retrieve reads it back.

| `--gds-l1-backend` | Store p50 (ms) | Store p99 (ms) | Retrieve p50 (ms) | Retrieve p99 (ms) |
|---|---|---|---|---|
| opends | 24.4 | 30.2 | 12.1 | 13.0 |
| cufile | 28.3 | 34.1 | 31.7 | 33.9 |
