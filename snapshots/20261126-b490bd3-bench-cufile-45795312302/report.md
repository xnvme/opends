### filesize8gib / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7038 |

### filesize8gib / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7102 |

### filesize8gib / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7056 |

### imagenetish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 3193 |

### imagenetish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 2987 |

### imagenetish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 604 |

### lmcacheish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4940 |

### lmcacheish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5071 |

### lmcacheish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 6203 |

### tiktokish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4980 |

### tiktokish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4930 |

### tiktokish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5780 |

## cuFile vs OpenDS

### queue_depth=8, io_threads=2

| Dataset | Mode | cuFile (MiB/s) | OpenDS (MiB/s) | Speedup |
|---|---|---|---|---|
| filesize8gib | async | - | 7038 | - |
| filesize8gib | stream | 1759 | 7102 | 4.04 |
| filesize8gib | sync | 5713 | 7056 | 1.24 |
| imagenetish | async | - | 3193 | - |
| imagenetish | stream | 969 | 2987 | 3.08 |
| imagenetish | sync | 656 | 604 | 0.92 |
| lmcacheish | async | - | 4940 | - |
| lmcacheish | stream | 4979 | 5071 | 1.02 |
| lmcacheish | sync | 4300 | 6203 | 1.44 |
| tiktokish | async | - | 4980 | - |
| tiktokish | stream | 5130 | 4930 | 0.96 |
| tiktokish | sync | 2957 | 5780 | 1.96 |

