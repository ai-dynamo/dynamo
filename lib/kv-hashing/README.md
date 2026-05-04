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
    pub fn position(&self) -> u32 { self.plh.position_u32() }
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

## 7. Universal-hashing API rollout (lib/tokens cleanup)

Section 6 covers the *wire-format* migration (kvbm-connector → framework
workers → kv-router → consolidator). This section covers the *Rust API*
migration inside `lib/tokens` and `lib/kv-hashing` themselves: the back-compat
shims that exist today purely to keep the diff against `main` small, and the
phased plan to retire them.

### End-state contract

`PositionalLineageHash` is the single chain primitive. Its layout already
carries everything the rest of the stack needs to stop tracking out-of-band
fields:

- `current: SequenceHash` — full 64-bit sequence hash of this block
- `parent: SequenceHash` — full 64-bit parent sequence hash (0 at root)
- `flags: PlhFlags` (32-bit) — `position` (20 bits, ≤ `2^20-1`),
  `log2_block_size` (4 bits, `block_size ∈ {1, 2, …, 32768}`), `partition`
  (5 bits), `synthetic` (1 bit), `version` (2 bits, V1 = `0b01`)

`SequenceHash`, `LocalBlockHash` (= `BlockHash`), and `SaltHash` become
newtype wrappers around `u64`. Each is "already a hash," so each implements
`Hash` as a single `state.write_u64(self.0)` — the same pattern PLH itself
uses today (`lib/tokens/src/lib.rs:840–844`). All four types pair with one
shared passthrough hasher — call it `IdentityU64Hasher` — modelled on the
existing `PlhHasher` (`lib/tokens/src/lib.rs:855–870`). Map keys "already
hashed" go through it as `HashMap<K, V, BuildHasherDefault<IdentityU64Hasher>>`
so the inner `u64` is captured verbatim, not re-hashed.

### What is still on a shim today

Each row points at the file:line where the shim lives and names the
replacement.

#### A. Type aliases that should be newtypes

| File:line | What | Replacement |
|---|---|---|
| `lib/tokens/src/lib.rs:40` | `pub type SaltHash = u64;` | `pub struct SaltHash(pub u64)` with `Hash` passthrough |
| `lib/tokens/src/lib.rs:53` | `pub type BlockHash = u64;` | `pub struct BlockHash(pub u64)` with `Hash` passthrough |
| `lib/tokens/src/lib.rs:58` | `pub type LocalBlockHash = BlockHash;` | Stays as alias once `BlockHash` is a newtype |
| `lib/tokens/src/lib.rs:69` | `pub type SequenceHash = u64;` | `pub struct SequenceHash(pub u64)` with `Hash` passthrough |

#### B. Salt-free wire chain on `TokenBlock` (added to fix the mooncake parity test)

The mocker emits KV-cache events whose `block_hash` field comes from
`TokenBlock::sequence_hash()`. Universal hashing made `PLH::current` salt-mixed
at the root, but the kv-router event consumers
(`compute_seq_hash_for_block`, `ensure_seq_hash_computed`) still encode the
historic salt-free convention `seq[0] = local_hash[0]`. To unblock CI without
touching kv-router, `TokenBlock` carries a parallel salt-free chain that
`sequence_hash()` projects.

| File:line | What | Why it's there |
|---|---|---|
| `lib/tokens/src/lib.rs:1340–1346` | `TokenBlock::{wire_sequence_hash, parent_wire_sequence_hash}` fields | Wire chain storage |
| `lib/tokens/src/lib.rs:1118` | `PartialTokenBlock::parent_wire_sequence_hash` field | Wire chain plumbing |
| `lib/tokens/src/lib.rs:1373, 1386–1394` | `from_chunk` accepts and computes the wire chain | Wire chain init |
| `lib/tokens/src/lib.rs:1424–1431` | `TokenBlock::sequence_hash()` returns the wire chain | Mocker → kv-router event compatibility |
| `lib/tokens/src/lib.rs:1436–1440` | `TokenBlock::parent_sequence_hash()` returns the wire chain | Same |
| `lib/tokens/src/lib.rs:1262–1267` | `PartialTokenBlock::parent_sequence_hash()` returns the wire chain | Same |
| `lib/tokens/src/lib.rs` (`chain_chunks`, `commit`, `truncate`, MM commit) | Wire-chain threading through the sequence APIs | Same |

Removal collapses both projections back onto
`self.plh.current_sequence_hash()` / `self.plh.parent_sequence_hash()`.
**Pre-condition:** kv-router event consumers must accept salt-mixed chains
first (see §D).

#### C. PLH back-compat accessors and constructors

| File:line | What | Replacement |
|---|---|---|
| `lib/tokens/src/lib.rs:697–699` | `PLH::position(&self) -> u64` | `position_u32()` (already exists; rename to `position()` after callers) |
| `lib/tokens/src/lib.rs:722–729` | 3-arg `PLH::new(current, parent: Option<u64>, position)` | `from_raw_parts` or `root_with_salt` + `extend` |
| `lib/tokens/src/lib.rs:808–810` | `PLH::parent_hash_fragment(&self) -> u64` | `parent_raw()` (currently `pub(crate)`; promote on removal) |
| `lib/tokens/src/lib.rs:820–822` | `PLH::current_hash_fragment(&self) -> u64` | `current_sequence_hash()` |
| `lib/tokens/src/lib.rs:831–833` | `PLH::as_u128(&self) -> u128` | PLH-keyed maps via `IdentityU64Hasher`; PLH's own `Hash` already reduces to `current` |
| `lib/tokens/src/lib.rs:451–460` | `PlhFlags::for_block_size` non-power-of-two tolerance | Power-of-two only; kvbm-* tests pre-validate |
| `lib/tokens/src/lib.rs:639–648` | `PLH::synthetic_unique` (transitional, `pub(crate)`) | Drop the mocker's prefix-caching-disabled registration; see TODOs at lib/tokens/src/lib.rs:633–637, lib/mocker/src/common/sequence.rs, lib/mocker/src/kv_manager/kvbm_backend.rs |

#### D. kv-router event boundary (consumes events; needs to learn salt-mixed chain)

| File:line | What | Change |
|---|---|---|
| `lib/kv-router/src/protocols.rs:131–167` | `compute_seq_hash_for_block` — `seq[0] = local_hash[0]`, then xxh3 step | Accept request-level salt; root becomes `xxh3([salt, local_hash[0]], 0)` |
| `lib/kv-router/src/indexer/positional.rs:521–540` | `ensure_seq_hash_computed` — `seq_hashes[0] = sequence[0]` | Accept request-level salt and seed the root with it |
| `lib/kv-router/src/protocols.rs:382–392` | local `LocalBlockHash(pub u64)` and `ExternalSequenceBlockHash(pub u64)` | Re-export `LocalBlockHash` from `dynamo-tokens`; `ExternalSequenceBlockHash` collapses into PLH (or its `current` projection) |
| `KvCacheStoredBlockData` (kv-router event types) | Dual `block_hash` + `tokens_hash` fields | Single PLH (or `current`) field; framework workers (vLLM/sglang) emit PLH directly |

Once §D ships, §B has no remaining reader.

#### E. Kvbm-side cleanups (consumers of the §C shims)

These files use `current_hash_fragment`, `parent_hash_fragment`, `as_u128`,
or 3-arg `SequenceHash::new` and need rewrites in lockstep with §C:

- `lib/kvbm-logical/src/pools/inactive/backends/lineage.rs`
- `lib/kvbm-logical/src/pools/inactive/backends/multi_lru_backend.rs`
- `lib/kvbm-logical/src/registry/mod.rs`
- `lib/kvbm-engine/src/leader/session/{handle,server_session}.rs`
- `lib/kvbm-engine/src/offload/{pending,pipeline,policy}.rs`
- `lib/kvbm-engine/src/object/s3/lock.rs`
- `lib/kvbm-engine/src/testing/offloading/object_flow.rs`
- `lib/mocker/src/common/sequence.rs`
- `lib/mocker/src/kv_manager/kvbm_backend.rs`
- `lib/mocker/src/kvbm_offload/engine.rs`
- `lib/kvbm-physical/src/manager/handle.rs`
- `lib/llm/src/block_manager/v2/physical/manager/handle.rs`

`as_u128`-keyed `HashMap`s (frequency tracker, lineage tracker) become
`HashMap<PositionalLineageHash, _, BuildHasherDefault<IdentityU64Hasher>>`.
PLH's `Hash` impl already feeds only `current` to the hasher
(`lib/tokens/src/lib.rs:840–844`), so these maps lose nothing on lookup
fidelity that the `as_u128()` keys provided.

### Phased rollout

Each phase is its own PR. Dependencies between phases are explicit; running
them out of order breaks compilation or runtime parity.

- **Phase 1 — newtype promotion in lib/tokens.** Promote the four type
  aliases in §A; add `IdentityU64Hasher` next to `PlhHasher`; implement
  `Hash` on each newtype as a single `write_u64`. Touch only `lib/tokens` and
  `lib/kv-hashing`. Downstream crates that read `.0` or pass bare `u64`
  break — that's the bucket-(1) churn from
  `.claude/plans/the-blast-radius-on-radiant-valley.md` undone in reverse,
  deliberately.
- **Phase 2 — kv-router accepts salt-mixed chains.** Update §D:
  `compute_seq_hash_for_block` and `ensure_seq_hash_computed` learn salt at
  request entry; framework-worker bridges (vLLM/sglang) emit PLH directly.
  Once merged, §B is dead code. **Depends on:** Phase 1.
- **Phase 3 — drop the wire-chain projection.** Remove §B fields, threading,
  and the `TokenBlock::sequence_hash()` / `parent_sequence_hash()`
  projections; they collapse back to `plh.current_sequence_hash()` /
  `plh.parent_sequence_hash()`. Remove the matching TODO blocks
  (`lib/tokens/src/lib.rs:71–83` and the per-field TODOs cited in §B).
  **Depends on:** Phase 2.
- **Phase 4 — drop PLH back-compat constructors and projections (§C).**
  After the kvbm-* and mocker callers in §E migrate to `from_raw_parts`,
  `current_sequence_hash`, `position_u32`, etc., delete `PLH::new` (3-arg),
  `position() -> u64`, `current_hash_fragment`, `parent_hash_fragment`,
  `as_u128`, `PlhFlags::for_block_size`'s non-pow2 tolerance, and
  `synthetic_unique`. **Depends on:** Phase 1, plus §E callers rewritten.
- **Phase 5 — collapse `kvbm_common::SequenceHash` re-aliases.** With aliases
  no longer needed for source compatibility, decide whether to keep
  `pub use dynamo_tokens::PositionalLineageHash as SequenceHash` as a
  documentation re-export or import PLH directly. **Depends on:** Phase 4.

### Verification gate per phase

| Phase | Cargo invocations |
|---|---|
| 1 | `cargo test -p dynamo-tokens`, `cargo test -p dynamo-kv-hashing`, `cargo build --workspace`, `cargo clippy -p dynamo-tokens -p dynamo-kv-hashing --all-features` |
| 2 | `cargo test -p dynamo-kv-router`, `cargo test -p dynamo-bench --test mooncake_trace`, `cargo test -p dynamo-mocker --lib` |
| 3 | `cargo test -p dynamo-tokens`, `cargo test -p dynamo-bench --test mooncake_trace`, `cargo test -p kvbm-logical`, `cargo test -p kvbm-engine --features testing --lib` |
| 4 | `cargo test --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings` |
| 5 | `cargo test --workspace --all-targets`, `cargo doc --workspace --no-deps` |

## 8. Quick reference

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
