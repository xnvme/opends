### filesize8gib / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 6970 |

### filesize8gib / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7052 |

### filesize8gib / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7079 |

### imagenetish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 3086 |

### imagenetish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 2794 |

### imagenetish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 553 |

### lmcacheish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5415 |

### lmcacheish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5462 |

### lmcacheish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 6182 |

### tiktokish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4994 |

### tiktokish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4981 |

### tiktokish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5818 |

## cuFile vs OpenDS

### queue_depth=8, io_threads=2

| Dataset | Mode | cuFile (MiB/s) | OpenDS (MiB/s) | Speedup |
|---|---|---|---|---|
| filesize8gib | async | - | 6970 | - |
| filesize8gib | stream | 1734 | 7052 | 4.07 |
| filesize8gib | sync | 6456 | 7079 | 1.10 |
| imagenetish | async | - | 3086 | - |
| imagenetish | stream | 867 | 2794 | 3.22 |
| imagenetish | sync | 657 | 553 | 0.84 |
| lmcacheish | async | - | 5415 | - |
| lmcacheish | stream | 4977 | 5462 | 1.10 |
| lmcacheish | sync | 4350 | 6182 | 1.42 |
| tiktokish | async | - | 4994 | - |
| tiktokish | stream | 5096 | 4981 | 0.98 |
| tiktokish | sync | 2957 | 5818 | 1.97 |

