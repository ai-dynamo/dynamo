# dynamo-bench

Benchmarks and trace-export entrypoints for Dynamo. Hosts:

- `multiturn_bench` — concurrent multi-turn chat benchmark against an
  OpenAI-compatible endpoint, with optional speculative prefill.
- `offline_replay_bench` — Rust-native replay loop using the mocker's perf
  model, for profiling replay overhead.
- `kv_router/{mooncake,active_sequences}_bench` — kv-router microbenchmarks.
- `claude_trace_export` and `request_trace_to_mooncake` binaries — convert
  Claude / request traces into Mooncake JSONL for replay.

## Guardrails

- The Mooncake JSONL row schema, rolling block-hash-to-id mapper, and
  JSONL writer live in `dynamo-data-gen`, not here. Producers in this crate
  must serialize `dynamo_data_gen::MooncakeRow` directly — do not redefine
  the row type locally.
- Benchmarks here are CI-checked via clippy (`--all-targets -- -D warnings`)
  and the dedicated `mooncake_trace` test under
  `--features mocker-kvbm-offload`. Keep both green.
- This is a benchmark crate, not a library — keep public surface area
  minimal and treat the binaries as the primary product.
