## Model loading with llama.cpp

Qwen3-14B-Q6_K.gguf, llama.cpp `d9e7aec`, 5 runs per mode, each a new process from a cold page cache.

| `--load-mode` | Whole process (s) | Model load (s) |
|---|---|---|
| opends | 3.13 to 3.17 | 2.00 to 2.03 |
| dio | 3.38 to 3.64 | 2.20 to 2.46 |
| none | 3.96 to 4.07 | 2.80 to 2.89 |
| mmap | 6.30 to 6.33 | 5.13 to 5.15 |
