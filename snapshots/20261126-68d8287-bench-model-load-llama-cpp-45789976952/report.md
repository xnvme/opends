## Model loading with llama.cpp

Qwen3-14B-Q6_K.gguf, llama.cpp `d9e7aec`, 5 runs per mode, each a new process from a cold page cache.

| `--load-mode` | Whole process (s) | Model load (s) |
|---|---|---|
| opends | 3.12 to 3.14 | 2.00 to 2.02 |
| dio | 3.41 to 3.66 | 2.24 to 2.48 |
| none | 3.97 to 4.00 | 2.80 to 2.82 |
| mmap | 6.25 to 6.28 | 5.07 to 5.09 |
