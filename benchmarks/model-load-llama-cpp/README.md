# Model loading with llama.cpp

Measures how long llama.cpp takes to load one GGUF model from one NVMe drive
into GPU memory, in each of its load modes: `opends`, `dio`, `none` and
`mmap`. The `opends` mode comes from a fork of llama.cpp that reads the
tensors through OpenDS. OpenDS itself does not depend on llama.cpp; only this
benchmark fetches it.

`deps.toml` pins what is measured: the llama.cpp commit, and the model by URL
and sha256.

## What a run does

`run.py` builds `llama-completion` at the pinned commit against the installed
OpenDS, puts the model in the cache directory unless a file with the pinned
checksum is there already, and copies it onto the test drive. It loads the
model five times in `dio`, `none` and `mmap` mode while the kernel `nvme`
driver owns the drive, brings the HOMI stack up, and loads it five times in
`opends` mode. Then it removes the copy from the drive and brings the stack
down.

Every load is a new process, started after the page cache is dropped. Two
times are recorded: the wall time of the whole process, and the load time
llama.cpp prints.

GPU persistence mode is on during the run and put back afterwards. Without
it, a process on the kernel driver pays for the start-up of the GPU, 1.7 s
on an RTX 2000 Ada, which a process does not pay while the stack holds the
GPU open.

## Run it

With OpenDS installed on the target as for the other benchmarks:

```sh
benchmarks/model-load-llama-cpp/run.py
benchmarks/model-load-llama-cpp/report.py
```

The records land in `cijoe-output-bench/model-load-llama-cpp/`, the table in
`cijoe-output-bench/report.md`. In CI the `bench-model-load-llama-cpp` label
runs both.

The cache directory is `/var/cache/opends-bench` on the target, or
`model_cache` under `[test]` in `configs/test.toml`. The model is 12 GB.
