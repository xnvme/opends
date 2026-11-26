## Model loading with llama.cpp

Qwen3-14B-Q6_K.gguf, llama.cpp `d9e7aec`, 5 runs per mode, each a new process from a cold page cache.

| `--load-mode` | Whole process (s) | Model load (s) |
|---|---|---|
| opends | 3.11 to 3.15 | 1.99 to 2.01 |
| dio | 3.39 to 3.58 | 2.21 to 2.40 |
| none | 3.96 to 4.07 | 2.79 to 2.89 |
| mmap | 6.26 to 6.28 | 5.08 to 5.09 |
