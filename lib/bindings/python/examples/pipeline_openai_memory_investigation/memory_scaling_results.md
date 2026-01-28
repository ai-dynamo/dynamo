# Memory Scaling Results

## Test 1: Concurrency Scaling (Fixed Requests: 10k per test)

```
┌─────────────┬────────────┬─────────────┬──────────────┬──────────────┬───────────────┐
│ Concurrency │ 1K Payload │ 1K Growth/  │ 200K Payload │ 200K Growth/ │ 200K/1K Ratio │
│             │            │ Base        │              │ Base         │               │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 2           │ 66 MB      │ 0%          │ 338 MB       │ 0%           │ 5.1x          │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 4           │ 67 MB      │ 2%          │ 382 MB       │ 13%          │ 5.7x          │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 8           │ 70 MB      │ 6%          │ 420 MB       │ 24%          │ 6.0x          │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 16          │ 76 MB      │ 15%         │ 477 MB       │ 41%          │ 6.3x          │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 32          │ 87 MB      │ 32%         │ 501 MB       │ 48%          │ 5.8x          │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 64          │ 108 MB     │ 64%         │ 1085 MB      │ 221%         │ 10.0x         │
├─────────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 96          │ 125 MB     │ 89%         │ 1165 MB      │ 245%         │ 9.3x          │
└─────────────┴────────────┴─────────────┴──────────────┴──────────────┴───────────────┘
```

### Observations:
- 1K payload: Growth is gradual (0% → 89%)
- 200K payload: Growth accelerates sharply at 64 concurrency (48% → 221%)
- The 200K growth rate is ~2.7x the 1K growth rate (245%/89%)
- Memory growth is NOT proportional to payload size (200x payload → ~5x base memory)

### Key Ratios:
- Payload size ratio: 200K/1K = 200x
- Base memory ratio: 338/66 = 5.1x
- Growth amount ratio: 827/59 = 14x

---

## Test 2: Request Count Scaling (Fixed Concurrency: 48)

```
┌──────────┬────────────┬─────────────┬──────────────┬──────────────┬───────────────┐
│ Requests │ 1K Payload │ 1K Growth/  │ 200K Payload │ 200K Growth/ │ 200K/1K Ratio │
│          │            │ Base        │              │ Base         │               │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 1000     │ 86 MB      │ 0%          │ 659 MB       │ 0%           │ 7.7x          │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 2000     │ 92 MB      │ 7%          │ 728 MB       │ 10%          │ 7.9x          │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 4000     │ 98 MB      │ 14%         │ 840 MB       │ 27%          │ 8.6x          │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 8000     │ 104 MB     │ 21%         │ 813 MB       │ 23%          │ 7.8x          │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 16000    │ 109 MB     │ 27%         │ 907 MB       │ 38%          │ 8.3x          │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 32000    │ 113 MB     │ 31%         │ 967 MB       │ 47%          │ 8.6x          │
├──────────┼────────────┼─────────────┼──────────────┼──────────────┼───────────────┤
│ 64000    │ 118 MB     │ 37%         │ 992 MB       │ 51%          │ 8.4x          │
└──────────┴────────────┴─────────────┴──────────────┴──────────────┴───────────────┘
```

### Observations:
- 1K payload: 64x more requests → only 37% memory growth
- 200K payload: 64x more requests → only 51% memory growth
- Memory grows **sub-linearly** with request count (logarithmic pattern)

### Key Ratios:
- Request count ratio: 64k/1k = 64x
- 1K growth: 86 → 118 MB = +32 MB (+37%)
- 200K growth: 659 → 992 MB = +333 MB (+51%)

---

## Comparison: Concurrency vs Request Count Impact

| Factor | 1K Growth | 200K Growth |
|--------|-----------|-------------|
| 48x Concurrency (2→96) | +89% | +245% |
| 64x Requests (1k→64k) | +37% | +51% |

**Conclusion:** Concurrency has ~2-5x more impact on memory than request count.

---

## Root Cause Analysis

See `memory_debug_report.md` for detailed debug instrumentation results.

**Key Finding**: Memory issue is NOT caused by Python object retention.

- PyO3 objects (Context, AsyncResponseStream) are properly freed
- No reference cycles blocking GC
- Memory correlates with thread count (148 threads), not object count

**Root Cause**: glibc malloc arena fragmentation in tokio spawn_blocking threads.

Each blocking thread maintains its own malloc arena. With high concurrency and large payloads, arena fragmentation causes memory to accumulate but not return to OS.

---

## Allocator Comparison: jemalloc vs glibc

Test: 10k requests, 200K payload, 96 concurrency

| Allocator | VmRSS | VmPeak | Throughput |
|-----------|-------|--------|------------|
| jemalloc (LD_PRELOAD) | 1121 MB | 7180 MB | 149 req/s |
| glibc (default) | 1608 MB | 13754 MB | 169 req/s |
| **Savings** | **487 MB (30%)** | **6574 MB (48%)** | -12% |

**Conclusion**: jemalloc reduces memory by 30% with a 12% throughput cost.

### How to use jemalloc

```bash
# Install jemalloc
apt-get install -y libjemalloc2

# Run with LD_PRELOAD
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python frontend.py
```

---

## Arena Fragmentation Proof

Using `mallinfo2()` to diagnose glibc arena state after 5000 requests:

### Default glibc allocator
```
[FRONTEND-FINAL] === Arena Info ===
[FRONTEND-FINAL]   Arena (non-mmap): 1368.9 MB
[FRONTEND-FINAL]   Allocated:        10.6 MB
[FRONTEND-FINAL]   Free in arenas:   1358.3 MB
[FRONTEND-FINAL]   Releasable:       11.6 MB
[FRONTEND-FINAL]   Mmap regions:     4 (1.0 MB)
[FRONTEND-FINAL]   Free chunks:      4827
[FRONTEND-FINAL]   Arena utilization: 0.8%
[FRONTEND-FINAL]   WARNING: Low utilization suggests arena fragmentation!
[FRONTEND-FINAL]   WARNING: 1358.3MB free but only 11.6MB releasable!
```

**Key Insight**: 1358 MB is free in arenas but only 11.6 MB can be returned to OS.
This proves memory is trapped due to fragmentation, not actual usage.

### With MALLOC_ARENA_MAX=2
```
[FRONTEND-FINAL] === Arena Info ===
[FRONTEND-FINAL]   Arena (non-mmap): 842.4 MB
[FRONTEND-FINAL]   Allocated:        10.0 MB
[FRONTEND-FINAL]   Free in arenas:   832.4 MB
[FRONTEND-FINAL]   Releasable:       3.9 MB
[FRONTEND-FINAL]   Arena utilization: 1.2%
```

| Setting | Arena Size | RSS | Improvement |
|---------|-----------|-----|-------------|
| Default glibc | 1368.9 MB | 1293 MB | - |
| MALLOC_ARENA_MAX=2 | 842.4 MB | 897 MB | **31% reduction** |

---

## Mitigation Options Summary

| Option | Memory Reduction | Throughput Impact | Implementation |
|--------|-----------------|-------------------|----------------|
| jemalloc (LD_PRELOAD) | 30% | -12% | Easy (env var) |
| MALLOC_ARENA_MAX=2 | 31% | Minimal | Easy (env var) |
| jemalloc + ARENA_MAX | ~35-40% (est.) | -12% | Easy (env vars) |
| thread_keep_alive | TBD | None | Code change |

### Recommended Production Settings

```bash
# Option 1: Use MALLOC_ARENA_MAX (simplest, no throughput loss)
MALLOC_ARENA_MAX=2 python frontend.py

# Option 2: Use jemalloc (best memory savings)
apt-get install -y libjemalloc2
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python frontend.py

# Option 3: Both (maximum memory savings)
MALLOC_ARENA_MAX=2 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 python frontend.py
```

---

## Future Improvements

### 1. Tokio thread_keep_alive Configuration

The Dynamo runtime currently uses tokio's default `thread_keep_alive` of 10 seconds.
Reducing this could help release arena memory faster during idle periods.

**Location**: `lib/runtime/src/config.rs:368-377`

```rust
// Current (no thread_keep_alive set)
pub(crate) fn create_runtime(&self) -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(...)
        .max_blocking_threads(self.max_blocking_threads)
        .enable_all()
        .build()
}

// Potential improvement
pub(crate) fn create_runtime(&self) -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(...)
        .max_blocking_threads(self.max_blocking_threads)
        .thread_keep_alive(Duration::from_secs(self.blocking_thread_keep_alive_secs))
        .enable_all()
        .build()
}
```

### 2. Periodic malloc_trim() Calls

Adding periodic `malloc_trim(0)` calls during idle periods could help release
releasable memory back to OS.

### 3. Build Dynamo with jemalloc Linked

Instead of relying on LD_PRELOAD, build the Rust components with jemalloc as the
global allocator for more consistent behavior.
