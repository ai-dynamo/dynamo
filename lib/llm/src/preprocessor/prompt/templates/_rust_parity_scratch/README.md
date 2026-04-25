# Rust minijinja parity + bench scratch

Standalone Cargo project used during the DIS-1850 spike to validate
the templates against the actual production-runtime minijinja crate
(the in-tree `_deepseek_jinja_parity.rs.rejected` integration test was
blocked by missing `libnixl` in local-dev sglang containers).

## Files

- `Cargo.toml` — needs cargo 1.85+ (edition 2024)
- `main.rs` — parity binary: renders fixtures via Rust minijinja, byte-diffs against `test_output_*.txt`
- `bench.rs` — head-to-head bench: minijinja+pre-pass vs `encode_messages` (the existing Rust port)
- `deepseek_v4_port.rs` — copy of `lib/llm/src/preprocessor/prompt/deepseek_v4.rs` (sans `DeepSeekV4Formatter` impl, which depends on dynamo-llm internals); used by `bench.rs` as `mod deepseek_v4` for the head-to-head

## Run

```bash
mkdir -p /tmp/dis1850-rust/src
cp Cargo.toml /tmp/dis1850-rust/
cp main.rs /tmp/dis1850-rust/src/
cp bench.rs deepseek_v4_port.rs /tmp/dis1850-rust/src/
mv /tmp/dis1850-rust/src/deepseek_v4_port.rs /tmp/dis1850-rust/src/deepseek_v4.rs
cd /tmp/dis1850-rust
cargo build --release && ./target/release/parity && ./target/release/bench
```

(Or run from inside the dynamo devcontainer at `/workspace/_dis1850_rust/` —
that's where the spike actually ran.)

## Status / measured numbers

See `../../../../notes/DIS-1850-spike-results.md` → **Rust head-to-head bench**.

This is throwaway scaffolding — delete on impl PR landing.
