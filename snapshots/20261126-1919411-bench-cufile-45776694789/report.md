### filesize8gib / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7069 |

### filesize8gib / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7065 |

### filesize8gib / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7060 |

### imagenetish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 3086 |

### imagenetish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 2802 |

### imagenetish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 556 |

### lmcacheish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5346 |

### lmcacheish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5444 |

### lmcacheish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 6194 |

### tiktokish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5013 |

### tiktokish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5015 |

### tiktokish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5804 |

## cuFile vs OpenDS

### queue_depth=8, io_threads=2

| Dataset | Mode | cuFile (MiB/s) | OpenDS (MiB/s) | Speedup |
|---|---|---|---|---|
| filesize8gib | async | - | 7069 | - |
| filesize8gib | stream | 1672 | 7065 | 4.23 |
| filesize8gib | sync | 5708 | 7060 | 1.24 |
| imagenetish | async | - | 3086 | - |
| imagenetish | stream | 826 | 2802 | 3.39 |
| imagenetish | sync | 632 | 556 | 0.88 |
| lmcacheish | async | - | 5346 | - |
| lmcacheish | stream | 4996 | 5444 | 1.09 |
| lmcacheish | sync | 4307 | 6194 | 1.44 |
| tiktokish | async | - | 5013 | - |
| tiktokish | stream | 5067 | 5015 | 0.99 |
| tiktokish | sync | 2960 | 5804 | 1.96 |

