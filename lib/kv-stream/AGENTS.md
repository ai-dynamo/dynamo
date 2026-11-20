# lib/kv-stream

`dynamo-kv-stream` is the engine-side half of KV state: one rank's ordered KV event batches folded into a residency ledger, with the replay window and snapshots an engine's Rust frontend (vLLM, SGLang) or a Dynamo sidecar serves from it. The contract is in the crate docs (`src/lib.rs`).

- **No Dynamo dependencies, no runtime.** Engines embed this crate, so it depends on `rustc-hash` and `thiserror` only, and has no threads, locks, channels or async. What needs a Dynamo type belongs in `dynamo_kv_indexer::stream`, which attaches the prefix indexes to a stream.
- **It must build on the engines' Rust.** SGLang pins 1.92 and vLLM 1.95, both older than Dynamo's toolchain. `clippy.toml` sets the MSRV so clippy flags newer standard library APIs; it does not catch newer syntax.
- **A ledger entry is per block, millions per rank.** Work out the new `size_of` before adding a field to `Block` in `ledger.rs`, and keep its `const` size asserts.
- **Per-event paths stay per event.** `apply_batch`, `store` and `remove` do one table probe per block. The O(blocks) steps are listed on `RankStream`; do not add one to the ingest path.
- **A batch applies whole or poisons the stream.** A new mutation path that can fail halfway sets the poison flag first, as `apply_batch` and `reseed` do.
