## Model loading with llama.cpp

Qwen3-14B-Q6_K.gguf, llama.cpp `d9e7aec`, 5 runs per mode, each a new process from a cold page cache.

| `--load-mode` | Whole process (s) | Model load (s) |
|---|---|---|
| opends | 3.14 to 3.15 | 1.99 to 2.02 |
| dio | 3.33 to 3.47 | 2.16 to 2.29 |
| none | 3.97 to 3.98 | 2.79 to 2.81 |
| mmap | 6.25 to 6.26 | 5.07 to 5.08 |
