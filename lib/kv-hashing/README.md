# `dynamo-kv-hashing`

Universal `Request → Vec<PositionalLineageHash>` contract for KV cache identity across the Dynamo stack (router, consolidator, kvbm, framework workers).

This crate is **pure computation**: a library, not a service. No async, no transports, no traits over runtime/scheduler config, no event types. Those concerns belong in higher layers.

---

## 1. The three-representation problem

Today three different "block hash" representations coexist in Dynamo and don't agree on what a block is hashed *over* nor on the encoding of position/lineage:

1. **`lib/kv-router`** computes `LocalBlockHash(u64)` (`lib/kv-router/src/protocols.rs`) via XXH3 with **LoRA encoded as a seed mutation** (`XXH3_SEED.wrapping_add(xxh3_64(lora_name))`) and **multimodal hashes appended to per-block bytes** before XXH3. The router *also* computes a parent-chained `SequenceHash(u64)` (`compute_seq_hash_for_block`) for active-block tracking — so it already uses **a form of positional encoding via chain depth**, just not the bit-packed `(position | parent_fragment | current_fragment)` layout that PLH uses. Radix tree edges are keyed on `LocalBlockHash`; nodes also carry `ExternalSequenceBlockHash(u64)` for eviction lookup.

2. **`lib/tokens` + kvbm** share one representation. `TokenBlock` (`lib/tokens/src/lib.rs`) computes `BlockHash(u64)` seeded by `SaltHash`, where `SaltHash = compute_hash_v2(json(SaltPayload{salt, lora_name}), 0)` — a canonical pre-mixed salt. `PositionalLineageHash` is a 128-bit identifier with mode-selected layout `(mode | position | parent_fragment | current_fragment)` and is already aliased as `kvbm_common::SequenceHash` — every kvbm crate keys on it.

3. **Framework workers** (vLLM / TRT-LLM) emit `KvCacheEvent`s carrying *both* an `ExternalSequenceBlockHash` (their own scheme) and a re-derived `tokens_hash: LocalBlockHash` so the router can index. Removal events ship only the external hash, forcing the router's secondary lookup index.

The cost of keeping these reconciled is the consolidator's translation layer plus a dual-hash field on every `KvCacheStoredBlockData`. `ryan/kvbm-consolidator-v2` tried to unify this and went too deep — 238 files / 36k insertions — by bundling hashing with observability, ZMQ transport, async event services, and abstract worker/event traits.

## 2. The multimodal gap on the PLH side

Until this crate landed, `TokenBlock::from_chunk` and `SaltPayload` only knew about tokens + salt + LoRA. **kvbm and the kvbm-connector `Request` did not disambiguate two requests with identical tokens but different mid-prompt images** — a correctness gap relative to what the router already does. Closing this gap is a hard requirement for PLH-as-universal.

This crate's `Request` carries `mm_info: Vec<RequestMmObjectInfo>`. Block formation in `lib/tokens` was extended (`TokenBlockSequence::new_with_mm`, `TokenBlock::from_chunk_with_mm`-equivalent path) so MM placeholders are first-class block slots, matching vLLM's prefix-cache model: a block of `block_size=16` with 7 placeholder slots holds 9 real tokens, and the block hash incorporates the placeholder identifiers at their correct in-block positions.

**Per-slot byte encoding:**
- Real token slot: 4 bytes (`u32 LE` token id).
- Placeholder slot: 12 bytes (`u64 LE mm_hash` + `u32 LE run_offset`), where `run_offset = global_position - mm_run.offset`.

`run_offset` (relative to the start of the multimodal run, not the block) ensures that the same image at the same global token position produces identical placeholder bytes regardless of where block boundaries fall — preserving cross-request prefix sharing across alignment shifts. A multi-block MM run produces distinct `block_hash`es for each of its blocks because `run_offset` increases monotonically across blocks (verified by `tokens_mm_multi_block_run`).

Bytewise compatibility with vLLM's hashing is impossible regardless: vLLM uses SHA256 over a Python tuple of strings, while this path uses XXH3 over a packed LE buffer. We choose the encoding with better semantics for our cross-request prefix-sharing use cases.

## 3. Why PLH (and what PLH cannot do alone)

`PositionalLineageHash` is already implemented in `lib/tokens` and already adopted by every kvbm crate (`kvbm-common::SequenceHash` is literally a re-alias). It is:

- **Position-encoded**, so radix-prefix sharing still works.
- **Fragment-keyed**, enabling O(1) parent lineage lookup at position N − 1.
- **128-bit**, large enough to disambiguate cross-cluster identity.

Adopting PLH everywhere is **not a wholesale change in *kind* of hashing for the router** (which is already chain-positional via parent chaining). It's a change in *encoding*, plus salt canonicalization, plus MM coverage.

**PLH is self-extending.** PLH carries the full 64-bit current sequence hash inline, alongside the position and a parent-fragment. A child PLH is built from a parent PLH plus the child's `BlockHash` via `PositionalLineageHash::extend(child_block_hash)` — no out-of-band `SequenceHash` needs to be tracked.

**Salt enters the chain only at the root.** `BlockHash` (a `LocalBlockHash`) is content-only: `xxh3(block_bytes, LOCAL_BLOCK_HASH_SEED)` with a fixed seed shared with the kv-router (`XXH3_SEED = 1337`). Salt is mixed in once, at `PositionalLineageHash::root_with_salt(local, salt, block_size)`, where the root's `current = xxh3([salt.0, local_block_hash.0], 0)`. Every subsequent `extend(child_block_hash)` is `xxh3([parent_seq_u64, child_block_hash], 0)` with seed `0` — salt baked into `current` at the root propagates through every parent thereafter. Two sequences with identical tokens but different salts diverge starting at the root; identical tokens + identical salt produce bit-identical chains. The content-only `BlockHash` stays request-independent so it remains compatible with the kv-router's `tokens_hash` indexing.

```rust
pub struct UniversalBlock {
    pub block_hash: BlockHash,                // u64
    pub plh: PositionalLineageHash,           // u128 — carries full u64 seq hash inline
}

impl UniversalBlock {
    pub fn position(&self) -> u64 { self.plh.position() }
    pub fn sequence_hash(&self) -> SequenceHash { self.plh.current_sequence_hash() }
}
```

Salt is a per-request constant (identical for every block in a sequence), so it is not stored on `UniversalBlock`. Read it once from `Request::salt_hash(block_size)` — `block_size` is mixed into the salt so two requests with identical tokens but different `block_size` cannot collide on per-block hashes.

Wire formats that carry only PLH (e.g., a slimmed-down `KvCacheEvent`) are now sufficient on their own — receivers do not need a side table mapping PLH → u64. The consolidator's translation layer collapses to a thin pass-through.

> **Encoding break.** The PLH u128 layout and the chain-step xxh3 seed both changed in this revision. Old serialized PLHs are not bytewise-compatible with new ones, and the mode bits overlap so the two cannot be distinguished. Persisted radix trees, on-disk caches, and any cross-version replay must be invalidated.

## 4. Why a new crate (and what changed in `lib/tokens`)

`dynamo-tokens` owns the low-level primitive: block formation from token IDs, the salt-seeded chain, and the PLH encoding. `dynamo-kv-hashing` is the *application-level* `Request → Hash` contract that adds:

- Salt canonicalization (`SaltPayload { salt, lora_name }` → `SaltHash`).
- The user-facing `Request` shape with LoRA + multimodal.
- Per-block `UniversalBlock` projection.

`lib/tokens` got a targeted, additive change to support multimodal block formation:

- `TokenBlockMmInfo` (new public type).
- `TokenBlockSequence::new_with_mm` and `split_tokens_with_mm` (new constructors).
- `TokenBlockSequence::push_token` / `push_mm_run` / `extend_with_mm` (new streaming API).
- Internal `commit_current` that routes per-block hash computation through the MM-aware byte path when `mm_runs` is non-empty.
- `validate_and_sort_mm_info` (new public helper). The MM byte-encoding routine itself stays `pub(crate)` — consumers chain through `PositionalLineageHash::root_with_salt` / `extend(local_block_hash)` and never hand-roll the byte buffer.

**Existing zero-MM constructors are unchanged.** All pre-existing tests pass unchanged, and `tokens_mm_zero_mm_equivalence` proves field-for-field equality between the MM-empty `new_with_mm` path and the existing `new` path. `cross_check_tokens_zero_mm` proves the same gate at the kv-hashing level.

## 5. Non-goals

This crate intentionally does NOT contain — and should NOT grow:

- Async / tokio / runtime — block hashing is pure CPU; let callers schedule.
- Traits over worker, scheduler, or transport configs (`WorkerConfigLike`, `RouterEventSink`) — the consolidator-v2 mistake. Hardening event/transport abstractions before the Request → hash mapping was stable led to repeated refactoring there.
- `KvCacheEvent` / `KvCacheStoredBlockData` / wire formats — they belong in `lib/kv-router` or a protocols crate.
- ZMQ / NATS / any networking — out of scope for a hashing library.
- Observability / metrics — instrumentation is a caller concern.

## 6. Migration sketch (informational, not executed in this PR)

This PR delivers the contract. Adoption is in follow-up phases:

- **Phase A — kvbm-connector**: replace `kvbm-connector::Request` with `kv_hashing::Request`; salt-hash computation moves to this crate; the multimodal field is added on the kvbm side for the first time. Tests in `kvbm-connector` should switch to using `request.into_blocks(block_size)`.
- **Phase B — framework workers**: vLLM/TRT-LLM bridge code computes PLH locally (using this crate or `dynamo_tokens` directly) and emits it as the *single* hash on `KvCacheEvent`. The dual `block_hash` / `tokens_hash` fields on `KvCacheStoredBlockData` collapse. Removal events also carry PLH (or u64 SequenceHash, decided in Phase C).
- **Phase C — kv-router**: `RadixTree` edge key changes to either `PositionalLineageHash` (u128) or extracted `BlockHash` (u64) — choice deferred until we have benchmark data on radix size. The router's existing chained `SequenceHash` becomes redundant once PLH is the edge key. `ExternalSequenceBlockHash` and the secondary lookup index are removed.
- **Phase D — consolidator**: drops the per-event translation logic. If router and consolidator agree on the same edge key (from Phase C), the consolidator is a thin dedup + rebroadcast pass.

## 7. Quick reference

```rust
use dynamo_kv_hashing::{Request, RequestMmObjectInfo};

let request = Request::new(
    tokens,                                    // Vec<u32>
    Some("lora-name".into()),                  // Option<String>
    Some("model-arch-tag".into()),             // Option<String> — free-form salt
    vec![RequestMmObjectInfo {                 // multimodal placeholder runs
        mm_hash: image_hash,
        offset: 12,
        length: 256,
    }],
)?;

let blocks = request.into_blocks(16)?;         // Vec<UniversalBlock>
let plhs   = request.positional_lineage_hashes(16)?;  // transport-friendly
```

Public surface re-exports the underlying `dynamo_tokens` primitives (`PositionalLineageHash`, `compute_hash_v2`, `compute_salt_hash_from_bytes`, etc.) so consumers can depend on this crate alone.
