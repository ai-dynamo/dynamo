# dynamo-bench

Benchmarks and trace-export entrypoints for Dynamo. Hosts:

- `multiturn_bench` — concurrent multi-turn chat benchmark against an
  OpenAI-compatible endpoint, with optional speculative prefill.
- `offline_replay_bench` — Rust-native replay loop using the mocker's perf
  model, for profiling replay overhead.
- `kv_router/{mooncake,active_sequences}_bench` — kv-router microbenchmarks.
- `claude_trace_export` — converts local Claude sessions into canonical Dynamo
  request traces for direct replay.
- `request_trace_to_mooncake` — opt-in export of Dynamo request traces to
  Mooncake replay JSONL.
- `request_trace_to_satf` — opt-in export of Dynamo request traces to SATF 2.0.

## Guardrails

- Dynamo request traces replay directly through `--trace-format dynamo`; do not
  insert an intermediate Mooncake file into that replay path.
  `request_trace_to_mooncake` is an opt-in export for Mooncake-compatible
  consumers.
- Benchmarks here are CI-checked via clippy (`--all-targets -- -D warnings`).
  Keep that validation green.
- This is a benchmark crate, not a library — keep public surface area
  minimal and treat the binaries as the primary product.
