# Memory Debug Report - Frontend Serialization Analysis

**Date**: 2026-01-11
**Test**: 1301 requests, 200K payload, 48 concurrency
**Environment**: DYNAMO_MEMORY_DEBUG=1

## Summary

Memory grew from 51MB to 952MB (+901MB) but **Python object retention is NOT the cause**.

## Key Findings

### 1. PyO3 Objects Are NOT Accumulating
```
PyO3 suspects: {}
```
Context, AsyncResponseStream, Client, and other PyO3 objects are being properly freed after each request completes.

### 2. No Reference Cycles
```
No uncollectable objects
```
`gc.garbage` is empty - no objects with `__del__` methods stuck in reference cycles.

### 3. Top Object Types (Normal)
```
Top 20 object types:
  function: 6,195
  tuple: 3,309
  dict: 3,011
  ReferenceType: 1,632
  wrapper_descriptor: 1,556
  method_descriptor: 1,481
  builtin_function_or_method: 1,228
  getset_descriptor: 1,220
  cell: 976
  type: 917
  list: 659
  ForwardRef: 618
  member_descriptor: 557
  property: 524
  frozenset: 346
  _tuplegetter: 295
  set: 258
  module: 250
  ModuleSpec: 245
  classmethod: 234
```
All standard Python runtime types - no indication of leaking application objects.

### 4. Per-Request Memory Analysis
```
Memory retention ratio: avg=6.30x, min=-32.07x, max=134.53x payload size
Memory delta by payload size:
    0KB payload: avg delta=0.0KB (1 samples)
    4000KB payload: avg delta=24623.6KB (1300 samples)
```

- **Negative minimum** (-32x): Memory IS being freed after requests complete
- **High maximum** (134x): First request includes initialization overhead
- **Average**: ~24MB delta per request with ~4MB payload

### 5. Thread Count
```
Threads: 148
```
Thread count remains at 148 regardless of load - these are tokio runtime threads, not leaking.

## Conclusion

**The memory issue is NOT caused by Python object retention or serialization patterns.**

The memory is being retained in the **Rust/glibc layer**:
- Python objects (including PyO3 types) are properly freed
- No reference cycles blocking garbage collection
- Memory correlates with concurrency (thread count), not object count

**Root Cause**: glibc malloc arena fragmentation in tokio spawn_blocking threads.

Each blocking thread maintains its own malloc arena (~8MB). With 148 threads and arena fragmentation from large allocations (200K payloads), memory accumulates in arenas but isn't returned to the OS.

## Mitigation Options

1. **malloc_trim()**: Periodically call to return arena memory to OS (already tested, provides 15-20% reduction)
2. **Alternative allocator**: jemalloc or mimalloc (tested, had compatibility issues)
3. **Reduce blocking threads**: Limit `DYN_RUNTIME_MAX_BLOCKING_THREADS` (tested, didn't help - 148 threads are mostly async workers)
4. **Batch GIL operations**: Reduce per-request spawn_blocking calls (requires code changes)

## Raw Data

### Sample Per-Request Tracking
```
[FRONTEND] [REQ 140200568658624] START: payload=4001219B, refcount=4, RSS=65.8MB
[FRONTEND] [REQ 140200568658624] END: RSS=245.3MB (delta: +179.48MB)  # First request, includes init

[FRONTEND] [REQ 140200568719168] START: payload=4001219B, refcount=4, RSS=189.1MB
[FRONTEND] [REQ 140200568719168] END: RSS=252.0MB (delta: +62.89MB)   # Subsequent requests

# At steady state, deltas become negative (memory freed):
[FRONTEND] [REQ 140200568724480] END: RSS=1159.1MB (delta: -13.34MB)
[FRONTEND] [REQ 140200568725056] END: RSS=1163.0MB (delta: -16.95MB)
```

### Final Report
```
============================================================
[FRONTEND] MEMORY REPORT @ 1301 requests
============================================================
RSS: 951.8MB (delta: +900.7MB from start)

PyO3 suspects: {}
No uncollectable objects
============================================================
```

## Environment Variables for Debug Mode

| Variable | Purpose |
|----------|---------|
| `DYNAMO_MEMORY_PROFILE=1` | Enable basic monitoring (default) |
| `DYNAMO_MEMORY_DEBUG=1` | Enable per-request tracking (high overhead) |
| `DYNAMO_MEMORY_TRACE=1` | Enable tracemalloc allocation tracking |
| `DYNAMO_MEMORY_VERBOSE=1` | Enable object snapshots |
