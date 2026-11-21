# lib/kv-hashing

`dynamo-kv-hashing` is the block-hash contract engines and the router share (`Request`, `LocalBlockHash`, `ExternalSequenceBlockHash`, `routing::compute_block_hash_for_seq`). It is pure computation. The KV-memory area in `.github/codeowners/areas.yaml` owns it.

## Rules

- **Changing a hash output.** Don't, unless every producer changes in the same release. The router recomputes from request tokens the hashes that workers publish, and a mismatch gives no error, only zero cache overlap. The seed (`XXH3_SEED` in `src/routing.rs`), the salt derivation (`src/salt.rs`), the multimodal byte frame, and the Eagle window are all part of the contract. `tests/request_hashing.rs` pins it (`producer_block_hash_matches_router_local_block_hash_default`, `producer_block_hash_matches_router_local_block_hash_lora`, `request_sequence_hashes_match_canonical_chain`).
- **Adding a request-wide identity that must isolate caches** (LoRA, `cache_salt`). Fold it into the salt in `compute_salt_hash`, and treat an empty string as absent (`empty_lora_normalizes_to_none`). Multimodal identity is not in the salt; it folds into each block so requests still share the blocks before an image (#8197, #10652).
- **Changing the chain recurrence.** Change `dynamo_tokens::compute_next_sequence_hash`, which `compute_next_seq_hash` calls. Don't add a second copy.
- **Touching the hashing loop.** Text blocks hash in place with no copy; only a block that overlaps a multimodal run allocates. Run the bench before and after (#15701).
- **Adding async, a transport, an event type, or a config trait.** Not here. The crate also sets `#![deny(missing_docs)]`, so document every public item.

## Test and bench

```bash
cargo test -p dynamo-kv-hashing
cargo bench -p dynamo-kv-hashing --bench request_hash
```
