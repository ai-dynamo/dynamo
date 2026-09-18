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
- Mirrors nested `pub mod` as nested Python classes, so a Rust path and a Python path read the same

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

### Nested modules

A nested `pub mod` becomes a nested class. `transport::tcp::ERRORS_TOTAL` in Rust is `transport.tcp.ERRORS_TOTAL` in Python.

**Rust input:**
```rust
pub mod transport {
    pub mod tcp {
        pub const ERRORS_TOTAL: &str = "tcp_errors_total";
    }
}
```

**Python output:**
```python
class transport:
    class tcp:
        ERRORS_TOTAL = "tcp_errors_total"
```

Only `pub mod` is exported; a private nested module stays out of the generated file. Nested modules are sorted by name, so reordering the Rust source does not churn the Python diff.

### When to run

Run after modifying `lib/runtime/src/metrics/prometheus_names.rs`:

```bash
cargo run -p dynamo-codegen --bin gen-python-prometheus-names
```

That single command is enough — the generator emits `black`-canonical output (88-column limit, no blank line under a docstring-free class header), so a second formatting pass changes nothing and re-running produces a byte-identical file.
