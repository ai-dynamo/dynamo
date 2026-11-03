# lib/kv-hashing

`dynamo-kv-hashing` is the block-hash contract engines and the router share: `LocalBlockHash` and `ExternalSequenceBlockHash`, `routing::compute_block_hash` and `compute_block_hash_for_seq`, the sequence hashes, and `XXH3_SEED`. Pure computation; changing a hash changes what engines publish and what the router matches, so keep the seed-equality and known-answer tests next to the function.

- Depends on (Dynamo crates): `dynamo-multimodal`, `dynamo-tokens`.
- Must not depend on: the runtime, tokio, any transport, any router crate beyond those two (`dynamo-router-protocol` and `dynamo-kv-events` sit above it).
- Hosts import this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
