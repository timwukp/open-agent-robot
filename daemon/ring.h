/* ring.h — lock-free single-producer/single-consumer byte ring.
 * One producer thread, one consumer thread, no locks (C11 atomics).
 * Capacity must be a power of two. */
#ifndef OAR_RING_H
#define OAR_RING_H
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint8_t *buf;
    size_t cap;                 /* power of two */
    _Atomic size_t head;        /* write index (producer) */
    _Atomic size_t tail;        /* read index (consumer)  */
} ring_t;

static inline int ring_init(ring_t *r, uint8_t *storage, size_t cap_pow2) {
    if (cap_pow2 == 0 || (cap_pow2 & (cap_pow2 - 1))) return -1;
    r->buf = storage; r->cap = cap_pow2;
    atomic_store(&r->head, 0); atomic_store(&r->tail, 0);
    return 0;
}

static inline size_t ring_used(const ring_t *r) {
    return atomic_load_explicit(&((ring_t *)r)->head, memory_order_acquire)
         - atomic_load_explicit(&((ring_t *)r)->tail, memory_order_acquire);
}
static inline size_t ring_free(const ring_t *r) { return r->cap - ring_used(r); }

/* returns bytes written (0 if insufficient space — frame dropped whole) */
static inline size_t ring_write(ring_t *r, const uint8_t *src, size_t n) {
    if (ring_free(r) < n) return 0;
    size_t head = atomic_load_explicit(&r->head, memory_order_relaxed);
    size_t idx = head & (r->cap - 1);
    size_t first = r->cap - idx; if (first > n) first = n;
    memcpy(r->buf + idx, src, first);
    memcpy(r->buf, src + first, n - first);
    atomic_store_explicit(&r->head, head + n, memory_order_release);
    return n;
}

/* returns bytes read (<= n) */
static inline size_t ring_read(ring_t *r, uint8_t *dst, size_t n) {
    size_t used = ring_used(r); if (n > used) n = used;
    if (n == 0) return 0;
    size_t tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
    size_t idx = tail & (r->cap - 1);
    size_t first = r->cap - idx; if (first > n) first = n;
    memcpy(dst, r->buf + idx, first);
    memcpy(dst + first, r->buf, n - first);
    atomic_store_explicit(&r->tail, tail + n, memory_order_release);
    return n;
}
#endif
