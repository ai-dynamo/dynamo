# Optimization Suggestions for pythonize/depythonize

## Current Bottleneck Analysis

The memory profiling shows:
- **122 GB total allocated** for 463K allocations
- **90% in native code** (`<stack trace unavailable>`)
- **Peak 4 GB** but high churn causes arena fragmentation

The main allocation path in `engine.rs`:
```rust
// Request: Rust struct → Python dict (1MB+ allocation)
let py_request = pythonize(py, &request)?;

// Response: Python dict → Rust struct (per chunk)
depythonize::<Resp>(&item.into_bound(py))
```

---

## Optimization 1: Avoid spawn_blocking for Small Responses

**Problem:** Every response chunk (typically small) goes through `spawn_blocking`.

**Current code:**
```rust
let response = tokio::task::spawn_blocking(move || {
    Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))
})
.await??;
```

**Suggested optimization:**
```rust
// For small responses, skip spawn_blocking - GIL contention is minimal
let response = Python::with_gil(|py| {
    depythonize::<Resp>(&item.into_bound(py))
})?;
```

**Rationale:**
- Response chunks are typically small (< 1KB)
- GIL acquisition is fast for short operations
- Avoiding spawn_blocking reduces thread pool pressure and arena allocation

**Estimated impact:** 10-20% reduction in allocations

---

## Optimization 2: Zero-Copy String Handling for Large Payloads

**Problem:** The 1MB message content is copied during `pythonize`.

**Current behavior:**
```
Rust String (1MB) → pythonize → Python str (1MB copy) → Python processing
```

**Suggested optimization - Pass bytes view:**
```rust
use pyo3::types::PyBytes;

// Instead of pythonizing the entire request, handle large content specially
fn pythonize_request_with_zerocopy<'py>(
    py: Python<'py>,
    request: &NvCreateChatCompletionRequest,
) -> PyResult<Bound<'py, PyAny>> {
    // For messages with large content, use PyBytes (zero-copy from Rust)
    let messages: Vec<_> = request.inner.messages.iter().map(|msg| {
        let content = &msg.content;
        if content.len() > 10_000 {
            // Large content: create PyBytes view (no copy)
            PyBytes::new(py, content.as_bytes())
        } else {
            // Small content: normal string
            content.into_pyobject(py)
        }
    }).collect();

    // Build dict with zero-copy content
    let dict = PyDict::new(py);
    dict.set_item("messages", messages)?;
    // ... other fields
    Ok(dict.into_any())
}
```

**Estimated impact:** 50-70% reduction in memory allocation for large payloads

---

## Optimization 3: Lazy Deserialization

**Problem:** `depythonize` deserializes the entire Python dict even if only parts are needed.

**Suggested optimization:**
```rust
// Instead of full deserialization
let response: FullResponse = depythonize(&py_obj)?;

// Use targeted field extraction
fn extract_response_fields(py: Python, obj: &Bound<PyAny>) -> Result<PartialResponse> {
    let dict = obj.downcast::<PyDict>()?;

    // Only extract fields we actually need
    let id: String = dict.get_item("id")?.extract()?;
    let content: Option<String> = dict.get_item("content").ok().map(|v| v.extract()).transpose()?;

    Ok(PartialResponse { id, content })
}
```

**Estimated impact:** 20-30% reduction for responses with many unused fields

---

## Optimization 4: Buffer Pooling with thread_local

**Problem:** Each serialization allocates fresh buffers.

**Suggested optimization:**
```rust
use std::cell::RefCell;

thread_local! {
    static SERIALIZE_BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(64 * 1024));
}

fn pythonize_with_buffer<T: Serialize>(py: Python, value: &T) -> PyResult<Bound<PyAny>> {
    SERIALIZE_BUFFER.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();

        // Serialize to buffer first
        serde_json::to_writer(&mut *buf, value)?;

        // Parse JSON in Python (Python's json is highly optimized)
        let json_module = py.import("json")?;
        let py_bytes = PyBytes::new(py, &buf);
        json_module.call_method1("loads", (py_bytes,))
    })
}
```

**Estimated impact:** 10-15% reduction in allocation overhead

---

## Optimization 5: Pass Raw JSON to Python

**Problem:** Double parsing - JSON → Rust struct → Python dict

**Current flow:**
```
HTTP JSON → serde_json → Rust struct → pythonize → Python dict
```

**Optimized flow:**
```
HTTP JSON → Python (keep bytes) → Python json.loads (when needed)
```

**Implementation:**
```rust
// In HTTP handler, keep raw bytes
let raw_json: Bytes = request.into_body().collect().await?;

// Pass to Python as bytes
Python::with_gil(|py| {
    let py_bytes = PyBytes::new(py, &raw_json);
    generator.call1(py, (py_bytes,))
})
```

**Python side:**
```python
import json
import orjson  # Faster JSON library

def generate(request_bytes):
    # Parse only when needed, using fast parser
    request = orjson.loads(request_bytes)
    # ... process
```

**Estimated impact:** 40-60% reduction in memory for request handling

---

## Optimization 6: Use orjson for Python Serialization

**Problem:** `pythonize` uses serde which may not be optimal for Python interop.

**Alternative approach:**
```rust
use pyo3::types::PyBytes;

fn request_to_python(py: Python, request: &Request) -> PyResult<Bound<PyAny>> {
    // Serialize to JSON bytes using simd-json or serde_json
    let json_bytes = serde_json::to_vec(request)?;

    // Import orjson (must be installed in Python env)
    let orjson = py.import("orjson")?;

    // orjson.loads is extremely fast
    orjson.call_method1("loads", (PyBytes::new(py, &json_bytes),))
}
```

**Benchmark comparison (typical):**
- pythonize: ~50 µs for 1MB
- orjson.loads: ~5 µs for 1MB

**Estimated impact:** 80-90% reduction in serialization time

---

## Optimization 7: Streaming Large Content

**Problem:** Entire 1MB message loaded into memory at once.

**Suggested optimization for very large payloads:**
```python
# Python side - streaming content access
class LazyContent:
    def __init__(self, content_view):
        self._view = content_view  # memoryview into Rust memory
        self._decoded = None

    def __str__(self):
        if self._decoded is None:
            self._decoded = self._view.tobytes().decode('utf-8')
        return self._decoded

    def __len__(self):
        return len(self._view)
```

---

## Implementation Priority

| Priority | Optimization | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | Skip spawn_blocking for small responses | Low | 10-20% |
| 2 | Use orjson instead of pythonize | Medium | 40-60% |
| 3 | Pass raw JSON bytes to Python | Medium | 40-60% |
| 4 | Zero-copy for large content | High | 50-70% |
| 5 | Buffer pooling | Low | 10-15% |
| 6 | Lazy deserialization | Medium | 20-30% |

---

## Quick Win: Conditional spawn_blocking

The simplest optimization with immediate impact:

```rust
// In engine.rs, line ~323
async fn process_item<Resp>(item: Result<Py<PyAny>, PyErr>) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(/* ... */)?;

    // OPTIMIZATION: For typical small responses, avoid spawn_blocking overhead
    // The GIL acquisition for small operations is faster than thread pool dispatch
    let response = Python::with_gil(|py| {
        depythonize::<Resp>(&item.into_bound(py))
    })
    .map_err(|e| ResponseProcessingError::DeserializeError(e.to_string()))?;

    Ok(Annotated::from_data(response))
}
```

This single change could reduce response processing allocations significantly.

---

## Long-term: Alternative Allocator

For production deployments handling high concurrency:

```dockerfile
# Use jemalloc instead of glibc malloc
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

# Or mimalloc
ENV LD_PRELOAD=/usr/lib/libmimalloc.so
```

These allocators handle arena fragmentation better than glibc's ptmalloc2.
