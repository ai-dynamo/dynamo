# lib/kv-ledger

`dynamo-kv-ledger` is one KV cache event stream as a replay window and a block ledger. `LocalKvIndexer` in `lib/router/kv-indexer/src/local.rs` runs on it, and engines embed it. The contract is in the crate docs (`src/lib.rs`).

- **Adding a dependency or a Dynamo type.** Don't. Engines build this crate in their own workspace, so it takes `rustc-hash` and the standard library only. What needs `RouterEvent` or any other Dynamo type goes in `local.rs`.
- **Using a newer standard library API or syntax.** Check it against Rust 1.92, which SGLang pins: `cargo +1.92 check -p dynamo-kv-ledger`. `clippy.toml` sets the MSRV so clippy flags newer APIs; it does not catch newer syntax.
- **Adding a method, option or type.** Add it only with a caller in `local.rs`. This crate holds what `LocalKvIndexer` uses, so that there is one implementation.
