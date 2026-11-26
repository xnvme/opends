## Model loading with llama.cpp

Qwen3-14B-Q6_K.gguf, llama.cpp `d9e7aec`, 5 runs per mode, each a new process from a cold page cache.

| `--load-mode` | Whole process (s) | Model load (s) |
|---|---|---|
| opends | 3.12 to 3.18 | 2.00 to 2.04 |
| dio | 3.51 to 3.74 | 2.32 to 2.58 |
| none | 3.97 to 4.03 | 2.80 to 2.86 |
| mmap | 6.22 to 6.27 | 5.06 to 5.09 |
