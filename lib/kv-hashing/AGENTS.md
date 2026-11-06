# lib/kv-hashing

`dynamo-kv-hashing` is the block-hash contract engines and the router share: `LocalBlockHash` and `ExternalSequenceBlockHash`, `routing::compute_block_hash` and `compute_block_hash_for_seq`, the sequence hashes, and `XXH3_SEED`. Pure computation; changing a hash changes what engines publish and what the router matches, so keep the seed-equality and known-answer tests next to the function.

- The `multimodal` module (`src/multimodal/`) is the multimodal identity both ends of the wire compute: `BlockExtraInfo`, `RequestExtraInfo`, the pad and identifier rules, the vLLM extra-key marker. Consumers name it in full (`dynamo_kv_hashing::multimodal::`); the root re-exports none of it, and it uses nothing else in the crate. Keep the known-answer tests next to each rule.
- Depends on (Dynamo crates): `dynamo-tokens`.
- Must not depend on: the runtime, tokio, any transport, any router crate (`dynamo-router-protocol` and `dynamo-kv-events` sit above it).
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
