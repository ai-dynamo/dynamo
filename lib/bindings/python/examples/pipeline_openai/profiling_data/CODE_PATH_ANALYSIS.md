# Code Path Analysis: Memory Allocations

## Allocation Hotspot: `lib/bindings/python/rust/engine.rs`

### 1. Request Serialization (lines 182-206)

```rust
// Acquiring the GIL is similar to acquiring a standard lock/mutex
// Performing this in an tokio async task could block the thread for an undefined amount of time
// To avoid this, we spawn a blocking task to acquire the GIL and perform the operations needed
// while holding the GIL.
//
// Under low GIL contention, we wouldn't need to do this.
// However, under high GIL contention, this can lead to significant performance degradation.
//
// Since we cannot predict the GIL contention, we will always use the blocking task and pay the
// cost. The Python GIL is the gift that keeps on giving -- performance hits...
let stream = tokio::task::spawn_blocking(move || {
    Python::with_gil(|py| {
        let py_request = pythonize(py, &request)?;  // <-- ALLOCATES ~1MB PER REQUEST

        let gen_result = if has_context {
            // Create context with trace information - only when the Python callable accepts it
            let py_ctx =
                Py::new(py, Context::new(ctx_python.clone(), current_trace_context))?;
            // Pass context as a kwarg
            let kwarg = PyDict::new(py);
            kwarg.set_item("context", &py_ctx)?;
            generator.call(py, (py_request,), Some(&kwarg))
        } else {
            // Legacy: No `context` arg - don't create Context object to avoid unnecessary allocations
            generator.call1(py, (py_request,))
        }?;

        let locals = TaskLocals::new(event_loop.bind(py).clone());
        pyo3_async_runtimes::tokio::into_stream_with_locals_v1(
            locals,
            gen_result.into_bound(py),
        )
    })
})
.await??;
```

**Impact:**
- Each request with 1MB payload allocates ~1MB in the blocking thread's arena
- `pythonize` converts Rust struct to Python dict (deep copy)

---

### 2. Response Deserialization (lines 304-333)

```rust
async fn process_item<Resp>(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(|e| {
        // ... error handling ...
    })?;

    let response = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))  // <-- CALLED PER CHUNK
    })
    .await
    .map_err(|e| ResponseProcessingError::OffloadError(e.to_string()))?
    .map_err(|e| ResponseProcessingError::DeserializeError(e.to_string()))?;

    let response = Annotated::from_data(response);

    Ok(response)
}
```

**Impact:**
- Called once per response chunk (streaming responses)
- Each `spawn_blocking` may use a different thread → different arena
- `depythonize` converts Python dict to Rust struct

---

## Memory Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REQUEST PROCESSING                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  HTTP Request (1MB JSON payload)                                         │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ tokio::task::spawn_blocking                                       │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  Thread Pool Thread N                                             │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  malloc() in Arena N (glibc)  ← 1MB allocated here                │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  pythonize(py, &request)                                          │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  Python dict created (1MB)                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│       ▼                                                                  │
│  Python async generator processes request                                │
│       │                                                                  │
│       ▼ (repeated for each response chunk)                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ tokio::task::spawn_blocking                                       │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  Thread Pool Thread M (may be different from N!)                  │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  malloc() in Arena M (glibc)  ← allocates here                    │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  depythonize(&item)                                               │  │
│  │     │                                                             │  │
│  │     ▼                                                             │  │
│  │  Rust struct created                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│       ▼                                                                  │
│  Request complete                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  Memory freed BUT remains in arenas (not returned to OS)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## glibc Arena Behavior

### Default Configuration
- Max arenas: `8 × num_cpus` (e.g., 128 arenas on 16-core machine)
- Each blocking thread can get its own arena
- Memory freed in an arena stays in that arena
- OS memory is not released until arena is destroyed

### With MALLOC_ARENA_MAX=2
- Max arenas: 2
- All threads share these 2 arenas
- Fragmentation contained
- Better memory reuse

---

## memray Profiling Results

```
Total allocations:     463,508
Total memory allocated: 122.760 GB  ← High churn!
Peak memory usage:      4.025 GB    ← Actual peak

Top allocators by size:
- <stack trace unavailable> (Native/Rust): 110.708 GB (90%)
- generator:frontend.py:64:               12.005 GB (10%)
```

**Key insight:** 90% of allocations happen in native code that memray cannot trace.
This confirms the allocations are in the Rust serialization layer.

---

## Potential Optimizations (Future Work)

1. **Reuse buffers** in pythonize/depythonize
2. **Thread-local caching** for serialization
3. **Zero-copy serialization** where possible
4. **Alternative allocators** (jemalloc, mimalloc)
5. **Reduce spawn_blocking calls** by batching
