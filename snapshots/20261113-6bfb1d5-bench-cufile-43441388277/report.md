### filesize8gib / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7075 |

### filesize8gib / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7069 |

### filesize8gib / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7056 |

### imagenetish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 3088 |

### imagenetish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 2790 |

### imagenetish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 549 |

### lmcacheish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5328 |

### lmcacheish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5366 |

### lmcacheish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 6181 |

### tiktokish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5009 |

### tiktokish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5012 |

### tiktokish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5801 |

## cuFile vs OpenDS

### queue_depth=8, io_threads=2

| Dataset | Mode | cuFile (MiB/s) | OpenDS (MiB/s) | Speedup |
|---|---|---|---|---|
| filesize8gib | async | - | 7075 | - |
| filesize8gib | stream | 1739 | 7069 | 4.06 |
| filesize8gib | sync | 6445 | 7056 | 1.09 |
| imagenetish | async | - | 3088 | - |
| imagenetish | stream | 852 | 2790 | 3.27 |
| imagenetish | sync | 657 | 549 | 0.84 |
| lmcacheish | async | - | 5328 | - |
| lmcacheish | stream | 5005 | 5366 | 1.07 |
| lmcacheish | sync | 4336 | 6181 | 1.43 |
| tiktokish | async | - | 5009 | - |
| tiktokish | stream | 5068 | 5012 | 0.99 |
| tiktokish | sync | 3092 | 5801 | 1.88 |

