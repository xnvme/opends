/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * ds_stream_map.h - tiny open-addressing map keyed on a GPU stream.
 *
 * Backs opends_aisio's stream -> stream_state lookup. The driver
 * registers up to MAX_STREAMS streams and then performs an
 * O(1) lookup per opends_read_async call. Grow-only: there is no
 * delete primitive, so the table needs no tombstones.
 *
 * Caller owns the entry array (typically inline in the driver struct).
 * Capacity must be a power of two; pass mask = capacity - 1. Load
 * factor stays below 50% if capacity >= 2 * MAX_STREAMS.
 *
 * The value is a small int (typically an index into the driver's
 * streams[] array). NULL key means empty slot; the GPU runtime never
 * returns a NULL stream and the pseudo-streams (legacy = 0x1,
 * per-thread = 0x2) are nonzero, so NULL is safe as the empty sentinel.
 */

#ifndef OPENDS_DS_STREAM_MAP_H
#define OPENDS_DS_STREAM_MAP_H

#include "ds_accel.h"

#include <stdint.h>

struct ds_stream_map_entry {
	ds_accel_stream_t key;
	int idx;
};

static inline uint32_t
ds_stream_map_hash(ds_accel_stream_t s, uint32_t mask)
{
	uintptr_t v = (uintptr_t)s;
	v *= 0x9E3779B97F4A7C15ULL;
	return (uint32_t)(v >> 32) & mask;
}

/* Returns idx on hit, -1 on miss. Lock-free against concurrent
 * ds_stream_map_put on a different key: a reader for K1 only walks
 * slots [hash(K1), K1's_slot], and those were all non-empty at K1's
 * registration (else K1 would have stopped sooner). The table is
 * grow-only, so a slot in that range cannot later become empty or
 * be the destination of another concurrent insert. The key is stored
 * with release and loaded with acquire, so an observed key publishes
 * its idx and everything the caller wrote before the put. */
static inline int
ds_stream_map_get(const struct ds_stream_map_entry *e, uint32_t mask,
                  ds_accel_stream_t stream)
{
	uint32_t h = ds_stream_map_hash(stream, mask);
	uint32_t cap = mask + 1;
	for (uint32_t p = 0; p < cap; p++) {
		const struct ds_stream_map_entry *s = &e[(h + p) & mask];
		ds_accel_stream_t k =
		        __atomic_load_n(&s->key, __ATOMIC_ACQUIRE);
		if (k == stream)
			return s->idx;
		if (k == NULL)
			return -1;
	}
	return -1;
}

/* Insert, returning 0 on success and -1 if full. Caller must
 * serialize inserts (in opends_aisio.c, reg_lock) and keep the load
 * factor below 50%. See ds_stream_map_get for the lock-free reader
 * invariant. */
static inline int
ds_stream_map_put(struct ds_stream_map_entry *e, uint32_t mask,
                  ds_accel_stream_t stream, int idx)
{
	uint32_t h = ds_stream_map_hash(stream, mask);
	uint32_t cap = mask + 1;
	for (uint32_t p = 0; p < cap; p++) {
		struct ds_stream_map_entry *s = &e[(h + p) & mask];
		if (s->key == NULL) {
			s->idx = idx;
			__atomic_store_n(&s->key, stream, __ATOMIC_RELEASE);
			return 0;
		}
	}
	return -1;
}

#endif /* OPENDS_DS_STREAM_MAP_H */
