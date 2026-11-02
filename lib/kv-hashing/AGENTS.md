# lib/kv-hashing

- This crate is the block-hash contract engines and the router share: the block hash identities `LocalBlockHash` and `ExternalSequenceBlockHash` (`block_hash`, re-exported at the root), `routing::compute_block_hash` and `compute_block_hash_for_seq`, the sequence hashes, and `XXH3_SEED`; it folds a block's multimodal metadata (`dynamo_multimodal::BlockExtraInfo`) into the block hash, but the pad and identifier rules themselves are `dynamo-multimodal`'s. Pure computation: no async, no transports, no event types, no service configuration.
- Depend only on `dynamo-multimodal` and `dynamo-tokens` among Dynamo crates (plus xxhash-rust, serde, serde_json, thiserror, derive_builder). Never add `dynamo-runtime`, `dynamo-kv-router`, `dynamo-router-protocol`, `dynamo-kv-events`, tokio, axum, reqwest, zmq, or prometheus; the gate rejects them.
- Changing a hash here changes what engines publish and what the router matches. Keep the seed-equality and known-answer tests next to the function.
- Run `scripts/router_layering_check.sh` before committing.
