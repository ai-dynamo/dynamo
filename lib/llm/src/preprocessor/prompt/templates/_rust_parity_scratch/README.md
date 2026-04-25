# Rust minijinja parity scratch

Standalone Cargo project used during the DIS-1850 spike to validate
templates against the actual production-runtime minijinja crate (the
in-tree `_deepseek_jinja_parity.rs.rejected` integration test was
blocked by a missing `libnixl` in the local-dev sglang container).

## Run

```bash
\cp -f Cargo.toml main.rs /tmp/claude/dis1850-rust/   # or any dir with cargo
cd /tmp/claude/dis1850-rust
cargo build --release && ./target/release/dis1850-rust
```

## Status

- Fixture 2 (no preprocessing) renders byte-identical via Rust minijinja.
- Fixtures 1, 3 (with tools / tool_calls) blocked on three jinja2-only
  idioms used in the templates: `.get()`, `.append()`, `.update()`.
  See spike-results doc for the port-pass plan.

This is throwaway scaffolding — delete on impl PR landing.
