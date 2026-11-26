### filesize8gib / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7087 |

### filesize8gib / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7057 |

### filesize8gib / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 7031 |

### imagenetish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 3087 |

### imagenetish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 2799 |

### imagenetish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 558 |

### lmcacheish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5461 |

### lmcacheish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5579 |

### lmcacheish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 6177 |

### tiktokish / async (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4943 |

### tiktokish / stream (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 4969 |

### tiktokish / sync (MiB/s)

| io_threads \ queue_depth | 8 |
|---|---|
| 2 | 5808 |

## cuFile vs OpenDS

### queue_depth=8, io_threads=2

| Dataset | Mode | cuFile (MiB/s) | OpenDS (MiB/s) | Speedup |
|---|---|---|---|---|
| filesize8gib | async | - | 7087 | - |
| filesize8gib | stream | 1698 | 7057 | 4.15 |
| filesize8gib | sync | 6478 | 7031 | 1.09 |
| imagenetish | async | - | 3087 | - |
| imagenetish | stream | 886 | 2799 | 3.16 |
| imagenetish | sync | 635 | 558 | 0.88 |
| lmcacheish | async | - | 5461 | - |
| lmcacheish | stream | 4991 | 5579 | 1.12 |
| lmcacheish | sync | 4308 | 6177 | 1.43 |
| tiktokish | async | - | 4943 | - |
| tiktokish | stream | 5183 | 4969 | 0.96 |
| tiktokish | sync | 2955 | 5808 | 1.97 |

