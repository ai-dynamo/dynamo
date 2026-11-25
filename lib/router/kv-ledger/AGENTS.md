# lib/router/kv-ledger

`dynamo-kv-ledger` is one KV cache event stream as a replay window and a block ledger. `LocalKvIndexer` in `lib/router/kv-indexer/src/local.rs` runs on it, and engines embed it. The contract is in the crate docs (`src/lib.rs`).

- **Adding a dependency or a Dynamo type.** Don't. Engines build this crate in their own workspace, so it takes `rustc-hash` and the standard library only. What needs `RouterEvent` or any other Dynamo type goes in `local.rs`.
- **Using a newer standard library API or syntax.** Check it by hand against Rust 1.92, which SGLang pins: `cargo +1.92 check -p dynamo-kv-ledger`. CI does not run this. `clippy.toml` sets the MSRV so clippy flags newer APIs; it does not catch newer syntax.
- **Changing `Block`, the slot type or their derives.** Keep Dynamo's slot, `(u64, Block<u64>)`, `Copy` and 32 bytes; two compile-time asserts in `src/ledger.rs` fail otherwise. `LocalKvIndexer` copies its ledgers under the lock its publisher takes, and that copy is a memory copy only while the slot is `Copy`.
- **Changing where `store` or `remove` put a block.** Keep `chains()` independent of slot order: the seeded test in `src/tests.rs` compares it with a plain map after every step, so run it. Then time one cold dump (`LocalKvIndexer::get_events_in_id_range(None, None)`) of 4M blocks stored as sequences, before and after, with a throwaway driver: `chains()` is fast only while a block's parent is usually in a neighbouring slot. ai-dynamo/dynamo#19429 has the numbers to compare with.
- **Adding a method, option or type.** Add it only with a caller in `local.rs`. This crate holds what `LocalKvIndexer` uses, so that there is one implementation.
