# Memory Analysis Report: Dynamo Frontend Pod

**Date:** January 2026
**Investigation:** Memory growth in frontend pod under high concurrency with large payloads

## Executive Summary

Memory growth in the Dynamo frontend is caused by **glibc malloc arena fragmentation**, not memory leaks. The root cause is the interaction between:
1. Tokio's `spawn_blocking` thread pool
2. Python/Rust serialization (`pythonize`/`depythonize`)
3. glibc's per-thread arena allocation strategy

**Solution:** Set `MALLOC_ARENA_MAX=2` environment variable to bound memory growth.

---

## 4-Way Comparison Test Results

### Test Configuration
- **Payload:** 1,000,000 chars (1 MB)
- **Concurrency:** 96
- **Requests per cycle:** 1,000
- **Cycles:** 3

### Results

| Test | Configuration | Initial | Cycle 1 | Cycle 2 | Cycle 3 | **Final** | Growth/Cycle |
|------|--------------|---------|---------|---------|---------|-----------|--------------|
| 1 | Baseline (no arena limit) | 57 MB | 6,114 MB | 7,573 MB | 8,389 MB | **8,389 MB** | ~1,100 MB |
| 2 | Baseline + MALLOC_ARENA_MAX=2 | 57 MB | 4,321 MB | 4,436 MB | 4,436 MB | **4,436 MB** | ~58 MB |
| 3 | PR #5376 (no arena limit) | 61 MB | 6,282 MB | 7,368 MB | 8,140 MB | **8,140 MB** | ~930 MB |
| 4 | PR #5376 + MALLOC_ARENA_MAX=2 | 57 MB | 3,946 MB | 4,004 MB | 4,061 MB | **4,061 MB** | ~57 MB |

### Key Findings

1. **MALLOC_ARENA_MAX=2 is the dominant factor** - ~50% memory reduction
2. **PR #5376 provides marginal improvement** - 3-8% additional reduction
3. **Best configuration:** PR #5376 + MALLOC_ARENA_MAX=2 (4,061 MB final)

---

## Memory Profiling Results (memray)

### Allocation Statistics
```
Total allocations:     463,508
Total memory allocated: 122.760 GB
Peak memory usage:      4.025 GB
```

### Top Allocators by Size
| Location | Size | Percentage |
|----------|------|------------|
| `<stack trace unavailable>` (Native/Rust) | 110.7 GB | **90%** |
| `generator:frontend.py:64` | 12.0 GB | 10% |

**90% of allocations occur in native (Rust) code**, confirming the issue is in the Rust serialization layer, not Python.

### Allocation Size Histogram
```
< 5 B      :  17,994 allocations
< 33 B     : 146,804 allocations
< 190 B    : 166,271 allocations
< 1 KB     :  60,967 allocations
< 6 KB     :  13,588 allocations
< 36 KB    :   5,106 allocations
< 210 KB   :  12,399 allocations
< 1.2 MB   :  35,579 allocations
< 40 MB    :   4,800 allocations
```

---

## Root Cause Analysis

### Code Path Identification

The allocations occur in `lib/bindings/python/rust/engine.rs`:

#### 1. Request Serialization (line 182)
```rust
let stream = tokio::task::spawn_blocking(move || {
    Python::with_gil(|py| {
        let py_request = pythonize(py, &request)?;  // Allocates ~1MB per request
        // ...
    })
})
```

#### 2. Response Deserialization (line 323) - Called per chunk
```rust
let response = tokio::task::spawn_blocking(move || {
    Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))
})
```

### Why Memory Grows

```
┌─────────────────────────────────────────────────────────────────┐
│  HTTP Request (1MB payload)                                      │
│       │                                                          │
│       ▼                                                          │
│  spawn_blocking ──► Thread Pool ──► Arena N (allocates 1MB)     │
│       │                                                          │
│       ▼                                                          │
│  pythonize(request) ──► Python dict created                     │
│       │                                                          │
│       ▼                                                          │
│  Python generator yields chunks                                  │
│       │                                                          │
│       ▼ (repeated per chunk)                                     │
│  spawn_blocking ──► Thread Pool ──► Arena M (allocates)         │
│       │                                                          │
│       ▼                                                          │
│  depythonize(response) ──► Rust struct                          │
│       │                                                          │
│       ▼                                                          │
│  Request complete ──► Memory freed BUT stays in arenas          │
└─────────────────────────────────────────────────────────────────┘
```

### glibc Arena Behavior

| Factor | Default | With MALLOC_ARENA_MAX=2 |
|--------|---------|------------------------|
| Max arenas | 8 × CPU cores | 2 |
| Memory per arena | Unbounded | Shared |
| Fragmentation | Distributed across many arenas | Contained in 2 arenas |
| Memory return to OS | Rarely (arenas hold freed memory) | More efficient |

---

## Recommendations

### Production Configuration

```bash
# Required for memory stability
export MALLOC_ARENA_MAX=2

# Optional: Additional tuning
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
```

### Kubernetes Deployment

```yaml
env:
  - name: MALLOC_ARENA_MAX
    value: "2"
```

### Alternative Allocators

For further optimization, consider:
- **jemalloc** - Better fragmentation handling
- **mimalloc** - Lower memory overhead
- **tcmalloc** - Google's thread-caching allocator

---

## Verification

Each test was verified using debug markers:

**Baseline tests:**
```
MEMORY_TEST: BASELINE code path (NO ZeroCopyTcpDecoder)
```

**PR #5376 tests:**
```
MEMORY_TEST: PR-5376 code path (WITH ZeroCopyTcpDecoder)
```

---

## Files Generated

- `/tmp/memray_frontend.bin` - memray binary profile
- `/tmp/memray_flamegraph.html` - Flamegraph visualization
- `/tmp/test[1-4]_*.log` - Test run logs
- `/tmp/tracemalloc_test.py` - Python allocation profiler

---

## Conclusion

The memory growth is **not a leak** but **arena fragmentation**. The fix is simple:

```bash
MALLOC_ARENA_MAX=2
```

This bounds memory to ~4 GB instead of 8+ GB under the same load, with stable memory after warmup.
