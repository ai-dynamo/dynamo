# lib/kv-hashing

- This crate is the block-hash contract engines and the router share: `routing::compute_block_hash` and `compute_block_hash_for_seq`, the sequence hashes, `XXH3_SEED`, and multimodal padding. Pure computation: no async, no transports, no event types, no service configuration.
- Depend only on `dynamo-router-protocol` and `dynamo-tokens` among Dynamo crates (plus xxhash-rust, serde, serde_json, thiserror, derive_builder). Never add `dynamo-runtime`, `dynamo-kv-router`, `dynamo-kv-events`, tokio, axum, reqwest, zmq, or prometheus; the gate rejects them.
- Changing a hash here changes what engines publish and what the router matches. Keep the seed-equality and known-answer tests next to the function.
- Run `scripts/router_layering_check.sh` before committing.
