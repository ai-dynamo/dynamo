# Dynamo Codegen

Python code generator for Dynamo Python bindings.

## gen-python-prometheus-names

Generates `prometheus_names.py` from Rust source `lib/runtime/src/metrics/prometheus_names.rs`.

### Usage

```bash
cargo run -p dynamo-codegen --bin gen-python-prometheus-names
```

### What it does

- Parses Rust AST from `lib/runtime/src/metrics/prometheus_names.rs`
- Generates Python classes with constants at `lib/bindings/python/src/dynamo/prometheus_names.py`

### Example

**Rust input:**
```rust
pub mod kvrouter {
    pub const KV_CACHE_EVENTS_APPLIED: &str = "kv_cache_events_applied";
}
```

**Python output:**
```python
class kvrouter:
    KV_CACHE_EVENTS_APPLIED = "kv_cache_events_applied"
```

### When to run

Run after modifying `lib/runtime/src/metrics/prometheus_names.rs` to regenerate the Python file.
