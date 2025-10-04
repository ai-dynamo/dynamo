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
- Handles macro-generated constants (e.g., `kvstats_name!("active_blocks")` → `"kvstats_active_blocks"`)

### Example

**Rust input:**
```rust
pub mod kvstats {
    pub const ACTIVE_BLOCKS: &str = kvstats_name!("active_blocks");
}
```

**Python output:**
```python
class kvstats:
    ACTIVE_BLOCKS = "kvstats_active_blocks"
```

### When to run

Run after modifying `lib/runtime/src/metrics/prometheus_names.rs` to regenerate the Python file.
