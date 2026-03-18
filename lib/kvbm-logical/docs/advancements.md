# Design Doc: `block_manager` (v1) vs `kvbm-logical` — Evaluation and Migration Path

## Context

`lib/llm/src/block_manager/` (referred to as **v1** throughout) manages the KV cache block lifecycle for LLM inference. Its logical layer — block state machines, registry, pools, events — is interleaved with physical concerns (storage backends, CUDA, NIXL, offloading).

`lib/kvbm-logical/` is a standalone crate that provides an independent logical block lifecycle layer. This document compares the two implementations factually across API surface, registry design, pool architecture, testability, and usability — and evaluates how `kvbm-logical` aligns with the future direction of operating on blocks by sequence hash.

---

## 1. Registry

The registry is where the two implementations diverge most significantly.

### v1 Registry (`block_manager/block/registry.rs`)

- **Data structure**: `HashMap<SequenceHash, Weak<BlockHandle>>` per pool, plus a `GlobalRegistry = Arc<Mutex<HashMap<SequenceHash, Weak<RegistrationHandle>>>>` shared across pools.
- **Registration handle**: `RegistrationHandle` is a fixed struct storing a full `TokenBlock` clone, `block_hash`, `sequence_hash`, `parent_sequence_hash`, and an `Arc<dyn EventReleaseManager>`. Every registration clones the token data.
- **Lookup**: O(1) average via HashMap. No prefix-aware or lineage-aware operations.
- **Cleanup**: A background tokio task listens on an `mpsc::UnboundedChannel<SequenceHash>` for `BlockHandle` drops, then checks `Weak::upgrade()` and removes dead entries from both the per-pool and global maps.
- **Extensibility**: To attach new data to a registered block, the `RegistrationHandle` struct must be modified. Fields are predetermined at compile time.
- **Frequency tracking**: None at the registry level. Access frequency is not recorded.

### kvbm-logical Registry (`kvbm-logical/src/registry/`)

- **Data structure**: `PositionalRadixTree<Weak<BlockRegistrationHandleInner>>`. Lookups are O(log n) and prefix-aware — the tree structure mirrors sequence hash lineage.
- **Registration handle**: `BlockRegistryHandle` is lightweight. Instead of cloning tokens, it uses an `AttachmentStore` that supports typed data association:
  - `attach_unique<T>(value)` — one value per type
  - `attach<T>(value)` — multiple values per type
  - `get<T>() -> TypedAttachments<T>` — accessor with `with_unique()`, `with_multiple()`, `with_all()` and mutable variants
  - Extensible without modifying the handle struct itself.
- **Lookup**: `register_sequence_hash()`, `match_sequence_hash()`, `is_registered()`, `check_presence<T>()`, `check_presence_any()`.
- **Cleanup**: Drop-based via `BlockRegistrationHandleInner::Drop`. No background task needed.
- **Frequency tracking**: Optional `TinyLFU` tracker (Count-Min Sketch, 4 hash functions, 4-bit counters with configurable decay). Methods: `touch(seq_hash)`, `count(seq_hash)`, `frequency_tracker()`. Integrated into `register_sequence_hash()` — each registration increments frequency.
- **Touch callbacks**: `on_touch(callback)` registers callbacks triggered on hash access. `touch()` fires all registered callbacks.
- **Presence tracking**: Explicit `mark_present<T>()` / `mark_absent<T>()` per block type. `has_block<T>()` and `has_any_block(type_ids)` query which typed blocks are currently alive on a handle.

### Comparison Notes

Both registries track blocks by `SequenceHash` and use weak references for lifetime management. v1's HashMap has O(1) average lookup but searches across all registered blocks in the pool — every lookup hashes and probes the full keyspace. kvbm-logical's `PositionalRadixTree` first narrows to a positional offset (the sequence position within the token stream), reducing the candidate set by orders of magnitude before searching within that bucket. For workloads with many registered blocks across many sequence positions, this partitioning matters.

The attachment system in kvbm-logical is the other key architectural difference — it allows associating arbitrary typed data with a registered hash without modifying the handle struct.

---

## 2. Block Guard Types: MutableBlock, ImmutableBlock, WeakBlock

These are the most-used types in both codebases — every block allocation, registration, match, and return goes through them.

### v1 MutableBlock (`block.rs:572-688`)

```rust
pub struct MutableBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    block: Option<Block<S, L, M>>,
    return_tx: tokio::sync::mpsc::UnboundedSender<Block<S, L, M>>,
    parent: Option<Arc<MutableBlock<S, L, M>>>,
}
```

- **3 generic type parameters** on every usage: `Storage`, `LocalityProvider`, `BlockMetadata`.
- **Option wrapping**: Inner block is `Option<Block>`. Accessed via `Deref`/`DerefMut` that calls `.expect("block was dropped")` (`block.rs:680,686`). If the block is taken (via `try_take_block`) and then accessed via Deref, this panics.
- **RAII return**: On drop, sends the block through an `mpsc::UnboundedSender<Block>` channel to the pool's background task (`block.rs:646-653`).
- **Parent chain**: `parent: Option<Arc<MutableBlock>>` tracks block lineage. Drop impl has explicit iterative unwinding to avoid stack overflow from deeply nested parent chains (`block.rs:657-673`).
- **State transitions**: `MutableBlock` derefs to `Block`, which exposes `init_sequence()`, `add_token()`, `commit()`, `apply_token_block()`, `reset()` — all return `Result<()>`. Caller must ensure block is in the right state.

### kvbm-logical MutableBlock (`blocks/mutable.rs:14-99`)

```rust
pub struct MutableBlock<T: BlockMetadata> {
    block: Option<Block<T, Reset>>,
    return_fn: ResetReturnFn<T>,
}
```

- **1 generic type parameter**: `T: BlockMetadata` — but `BlockMetadata` is a blanket trait with no methods:
  ```rust
  pub trait BlockMetadata: Clone + Send + Sync + 'static {}
  impl<T: Clone + Send + Sync + 'static> BlockMetadata for T {}
  ```
  In practice, `T` is a **zero-sized marker type** (storage tier marker) like `struct G1;` or `struct G2;`. It carries no data and no behavior — it exists purely to distinguish block types at the type level. A `MutableBlock<G1>` cannot be mixed with a `MutableBlock<G2>` at compile time. This is a significant difference from v1's `BlockMetadata` which is a trait with 4 required methods (`on_acquired`, `on_returned`, `reset_metadata`, `offload_priority`).
- **Option wrapping**: Same `Option<Block>` pattern. `.expect("MutableBlock missing block")` on access (`mutable.rs:75,80`). Same potential panic, but the take-then-access window is narrower — `take_block()` is only called in `complete()` and `stage()`, which consume `self`.
- **RAII return**: On drop, calls `(self.return_fn)(block)` — a closure, not a channel send (`mutable.rs:84-90`). No async overhead, no channel allocation, no background task recv.
- **No parent chain**. No iterative drop unwinding needed.
- **State transitions**: `MutableBlock` has two transitions: `complete(token_block) -> CompleteBlock<T>` and `stage(seq_hash) -> CompleteBlock<T>`. Both **consume self** and return a new type. You cannot call `complete()` twice, and you cannot call any method on a `MutableBlock` after completing it — the compiler enforces this.
- **Error recovery**: `complete()` returns `Result<CompleteBlock<T>, BlockError<MutableBlock<T>>>`. On failure (size mismatch), the `MutableBlock` is returned inside the error. v1's `apply_token_block()` returns `Result<()>` — the block remains in its previous state, but the caller gets no typed guarantee of that.

### v1 ImmutableBlock (`block.rs:772-957`)

```rust
pub struct ImmutableBlock<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    block: Arc<MutableBlock<S, L, M>>,
    sequence_hash: SequenceHash,
    duplicate: Option<Arc<MutableBlock<S, L, M>>>,
}
```

- **Wraps Arc\<MutableBlock\>**: An immutable reference is literally `Arc<MutableBlock>`. The immutability is convention, not type-enforced — the inner block is still a `MutableBlock` with all its mutable methods accessible via `Deref`.
- **Deref chain**: `ImmutableBlock -> (deref) -> Block -> (methods)`. Two levels of indirection to reach block data.
- **Duplicate field**: `duplicate: Option<Arc<MutableBlock>>` — added as an optional field. When duplication is allowed and a second block registers with the same hash, the duplicate MutableBlock is attached here. The `block_id()` method checks: if duplicate exists, return duplicate's ID; otherwise return primary's ID (`block.rs:843-847`). This means `ImmutableBlock.block_id()` can return different values depending on whether it's a duplicate.
- **Clone**: Clones the Arc + sequence_hash + optional duplicate Arc. Multiple ImmutableBlock instances can point to the same underlying data.
- **No downgrade**: No way to create a weak reference. If you hold an ImmutableBlock, you hold a strong reference that prevents the block from being evicted.

### kvbm-logical ImmutableBlock (`blocks/immutable.rs:13-67`)

```rust
pub struct ImmutableBlock<T: BlockMetadata> {
    block: Arc<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
}
```

- **Wraps Arc\<dyn RegisteredBlock\<T\>\>**: The inner type is a trait object, not a concrete `MutableBlock`. The `RegisteredBlock<T>` trait exposes only `block_id()`, `sequence_hash()`, `registration_handle()` — no mutable methods, no state transitions. Immutability is structurally enforced.
- **1 type parameter**: Just `T: BlockMetadata` — a zero-sized storage tier marker (see MutableBlock section above).
- **`downgrade() -> WeakBlock<T>`** (`immutable.rs:33-39`): Creates a weak reference. Weak references don't prevent eviction. This is critical for caches that want to track blocks without pinning them.
- **`use_count() -> usize`** (`immutable.rs:55-57`): Exposes `Arc::strong_count()` for diagnostics.
- **`registration_handle()`** (`immutable.rs:51-53`): Direct access to the registry handle, and through it, the attachment system.

### kvbm-logical WeakBlock (`blocks/immutable.rs:19-97`)

```rust
pub struct WeakBlock<T: BlockMetadata> {
    sequence_hash: SequenceHash,
    block: Weak<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
}
```

- **v1 has no equivalent**. If you hold a reference to a registered block in v1, it's a strong Arc.
- **Two-stage upgrade** (`immutable.rs:71-83`): First tries `Weak::upgrade()` (fast path — block still alive). If that fails, calls `upgrade_fn(sequence_hash)` which searches the active + inactive pools for the block (slow path — block may have moved pools).
- **Clone**: Cloneable. Multiple WeakBlocks can track the same sequence hash without preventing eviction.
- **Use case**: Schedulers, caches, and state trackers that need to reference blocks without preventing eviction.

### Comparison Notes

v1's `ImmutableBlock` wrapping `Arc<MutableBlock>` means the "immutable" block still carries all the mutable API surface through Deref. The naming is misleading — it's more accurately "shared access to a mutable block." kvbm-logical's `Arc<dyn RegisteredBlock<T>>` structurally limits the API to read-only operations.

v1 has no WeakBlock. Every reference is strong. This means a scheduler holding a reference to a block prevents that block from being evicted, even if the scheduler only needs to know the block exists. kvbm-logical's WeakBlock addresses this directly.

The return path differs: v1 sends through an async channel (`mpsc::UnboundedSender`), kvbm-logical calls a synchronous closure. For MutableBlock drops — which happen frequently — this means v1 incurs channel send overhead + background task recv + processing per block return. kvbm-logical's closure executes inline.

---

## 3. Block Duplication Strategies

Both implementations handle the case where a new block is registered with the same sequence hash as an existing registered block. The two implementations take meaningfully different approaches.

### v1 Duplication (`pool/managed/state.rs:200-257`, `block.rs:820-831`)

**Configuration**: `BlockRegistrationDuplicationSetting` — `Allowed` or `Disabled` — set per pool at construction time.

**Registration flow** (simplified):
1. Block enters registration.
2. Check inactive pool: `inactive.match_sequence_hash(seq_hash)`. If found, the existing block becomes the "primary" and the new block becomes the "duplicate."
3. If not in inactive, attempt `block.register(&mut self.registry)`. If `BlockAlreadyRegistered`, call `wait_for_returned_block()` — an **async wait** that blocks the registration path until the existing block transitions back.
4. Wrap in `ImmutableBlock::new(mutable_block)`.
5. Based on policy:
   - **Allowed**: `immutable.with_duplicate(duplicate_arc)` — attaches the duplicate as an `Option<Arc<MutableBlock>>` field on the ImmutableBlock.
   - **Disabled**: `duplicate.try_take_block()` — extracts raw blocks, returns them to inactive pool.

**Duplicate storage**: The duplicate is stored inside the `ImmutableBlock` as `duplicate: Option<Arc<MutableBlock>>`. This means:
- `block_id()` returns the duplicate's ID when duplicate exists, otherwise the primary's ID.
- `try_take_block()` must unwrap both Arcs.
- The primary block's lifetime is tied to the duplicate's `ImmutableBlock` — both stay alive until the `ImmutableBlock` is dropped.
- `is_duplicate()` check exists but is `pub(crate)`.

**Race handling**: `wait_for_returned_block()` is an async function that recv's from the block return channel until the target hash appears. This can block the entire registration path.

### kvbm-logical Duplication (`registry/registration.rs:119-166`, `blocks/registered.rs:26-154`)

**Configuration**: `BlockDuplicationPolicy` — `Allow` or `Reject` — set per BlockManager.

**Registration flow** (simplified):
1. Block enters registration via `register_block_inner()`.
2. Acquire attachment lock on the registry handle.
3. Call `try_find_existing_block()` — checks:
   - **Presence markers**: `attachments.presence_markers.contains_key(&TypeId::of::<T>())`. If no marker, return None immediately (no existing block of this type).
   - **Weak block references**: `weak_block.primary_block.upgrade()` — try to find active primary.
   - **Inactive pool**: `inactive_pool.find_block_as_primary(hash, false)` — uses `new_unattached()` to avoid deadlock (caller holds attachment lock).
   - **Spin-loop retry**: If presence marker exists but block not found in either pool, spin-loop up to 100 times (block is transitioning between pools).
4. If existing found, handle by policy:
   - **Allow**: `DuplicateBlock::new(registered_block, existing_primary, reset_return_fn)` — creates a dedicated `DuplicateBlock<T>` type. The primary is held as `_primary: Arc<PrimaryBlock<T>>`.
   - **Reject**: `extracted.discard(&reset_return_fn)` — the new block is returned to the reset pool immediately. Returns the existing primary.
5. If no existing, register as new `PrimaryBlock<T>` via `PrimaryBlock::new_attached()`.

**Type separation**: `PrimaryBlock<T>` and `DuplicateBlock<T>` are separate `pub(crate)` types. Both implement `RegisteredBlock<T>` but have distinct drop behaviors:
- `PrimaryBlock::drop()` returns `Arc<Block<T, Registered>>` to the inactive pool via `registered_return_fn`.
- `DuplicateBlock::drop()` calls `block.reset()` and returns `Block<T, Reset>` to the reset pool via `reset_return_fn`. The duplicate block is reclaimed, not preserved.

**Weak ref storage**: `PrimaryBlock::store_weak_refs()` stores both `Weak<Block<T, Registered>>` and `Weak<PrimaryBlock<T>>` in the handle's attachment system (`registered.rs:76-92`). This enables resurrection: if a block is being returned (PrimaryBlock dropped, Arc being moved to pool), the raw_block weak ref can still catch it.

**Lock discipline**: `new_attached()` vs `new_unattached()` (`registered.rs:42-68`) prevents double-lock when the caller already holds the attachment lock. 5 of 6 creation sites use `new_attached`; only `find_block_as_primary` uses `new_unattached`.

### Which version has tighter logic

kvbm-logical's duplication logic is tighter in several concrete ways:

1. **Type-enforced roles**: `PrimaryBlock` and `DuplicateBlock` are distinct types with distinct drop behaviors. v1 uses a single `ImmutableBlock` with an `Option<Arc<MutableBlock>>` field — the "is this a duplicate?" question is answered by checking if the Option is Some. The type system doesn't distinguish the two roles.

2. **Drop path separation**: In kvbm-logical, duplicate blocks reset and return to the reset pool on drop. Primary blocks return to the inactive pool as registered. These are separate code paths in separate types. In v1, `ImmutableBlock::try_take_block()` must handle both cases by checking whether `self.duplicate` exists, unwrapping both Arcs, and collecting results into a Vec (`block.rs:936-957`).

3. **No async wait**: v1's `wait_for_returned_block()` blocks the registration path waiting on an async channel for a specific hash to appear. kvbm-logical's `try_find_existing_block()` uses presence markers + weak reference upgrade + spin-loop (max 100 iterations). The spin-loop is bounded and doesn't involve channel communication.

4. **Presence markers**: kvbm-logical tracks whether a block type is present via explicit `mark_present<T>()` / `mark_absent<T>()` calls on the registry handle. This allows `try_find_existing_block` to short-circuit immediately when no block of that type exists. v1 must check the registry's HashMap, then the inactive pool's HashMap, then fall back to async waiting.

5. **Weak ref resurrection**: kvbm-logical stores `WeakBlockEntry { raw_block, primary_block }` in the attachment store. This handles the race window between PrimaryBlock::drop taking the Arc out of the Option and the return_fn completing the insert into the pool. v1 doesn't have this — it relies on the async return channel and `wait_for_returned_block()`.

6. **Lock awareness**: kvbm-logical has `new_attached()` / `new_unattached()` to explicitly manage lock ordering. v1 doesn't have this concern because its pool operations go through async channels rather than direct lock acquisition.

The tradeoff: v1's `wait_for_returned_block()` is guaranteed to find the block (it waits indefinitely on the channel). kvbm-logical's spin-loop is bounded at 100 iterations and can give up — the `MAX_RETRIES` limit means a block transitioning very slowly could be missed, though this hasn't been observed in the 194+ test suite.

---

## 4. Inactive Pool Backends

### v1 Inactive Pool (`pool/managed/inactive.rs`)

- **Data structures**: `HashMap<SequenceHash, Block<S, L, M>>` for lookup + `BTreeSet<PriorityKey<M>>` for eviction ordering + `VecDeque<Block>` for uninitialized blocks.
- **Eviction ordering**: `PriorityKey<M>` delegates to `M: Ord`. For `BasicMetadata`, the derived `Ord` compares: `priority` (u32) first, then `returned_tick` (u64), then `acquired_tick` (u64). Lower priority values are evicted first. Among equal priority, older returned timestamps are evicted first (LRU-like).
- **Allocation**: Pops from `uninitialized_set` (FIFO) first, then from `priority_set.pop_first()` (lowest priority key).
- **Hash matching**: `HashMap::remove()` — O(1) average.
- **Single strategy**: The eviction policy is determined entirely by the `BlockMetadata` impl's `Ord`. No way to swap strategies without changing the metadata type.

### kvbm-logical Inactive Pool (`pools/inactive/`)

- **Backend trait**: `InactivePoolBackend<T: BlockMetadata>` with methods: `find_matches()`, `scan_matches()`, `allocate()`, `insert()`, `len()`, `has_block()`, `allocate_all()`.
- **Built-in backends**:
  1. **HashMap** — similar to v1. Pluggable `ReusePolicy` (FIFO, LRU, etc.) determines which block to evict.
  2. **LRU** — simple LRU eviction.
  3. **MultiLRU** — 4 frequency tiers (Cold, Warm, Hot, Very Hot) with configurable thresholds (default: [3, 8, 15]). Blocks promote between tiers based on their frequency count from the TinyLFU tracker. Eviction prefers the coldest tier first.
  4. **Lineage** — custom lineage-based eviction that considers sequence hash parent relationships.
- **Backend selection**: Configured at build time via `BlockManager::builder()`: `.with_lru_backend()`, `.with_multi_lru_backend()`, `.with_multi_lru_backend_custom_thresholds(t1, t2, t3)`, `.with_hashmap_backend(reuse_policy)`, `.with_lineage_backend()`.

### Comparison Notes

v1 has one eviction strategy tied to `BasicMetadata`'s `Ord` implementation. This works but is rigid — changing eviction behavior requires changing the metadata type or its ordering. kvbm-logical decouples eviction strategy from metadata via the `InactivePoolBackend` trait. The MultiLRU backend specifically leverages the TinyLFU frequency tracker to make informed tier-based eviction decisions, which v1 has no equivalent for.

v1's `BTreeSet<PriorityKey>` approach does handle priority-aware ordering well for the offload use case (G2->G3), where blocks with higher `priority` values are offloaded first via the `OffloadRequestKey` (which reverses the comparison for the offload queue). This priority mechanism exists outside the inactive pool itself — it's in the offload path.

---

## 5. Block Pool Architecture

### v1 Pools (`pool.rs`, `pool/managed.rs`)

- **Async-first**: All pool operations (`allocate_blocks`, `register_blocks`, `match_sequence_hashes`, `touch_blocks`, `try_return_block`) are `async` methods on the `BlockPool` trait. Each also has a `_blocking` variant.
- **Communication**: Uses `priority_tx: PriorityChannelSender` and `ctrl_tx: mpsc::Sender` channels with `oneshot` response channels. A background tokio task processes requests from these channels.
- **Trait surface**: `BlockPool<S, L, M>` requires implementing `BlockPoolController` + `AsyncBlockPoolController` — 3 traits total, ~14 async methods + ~14 blocking variants = ~28 method signatures per pool type.
- **Block return**: Blocks return via `UnboundedSender<Block>`. `MutableBlock::drop()` sends the inner block through this channel.

### kvbm-logical Pools (`pools/`, `manager/`)

- **Synchronous core**: Pool operations use `parking_lot` locks. No async runtime required. No channels, no background tasks, no oneshot receivers.
- **Pool separation**: `ResetPool<T>` (available blocks), `ActivePool<T>` (registered block lookup via registry), `InactivePool<T>` (evictable registered blocks). Each pool is a simple struct with direct method calls.
- **Orchestrator**: `BlockManager<T>` composes the three pools + `BlockRegistry`. Public methods: `allocate_blocks()`, `register_blocks()`, `match_blocks()`, `scan_matches()`, `reset_inactive_pool()`, `total_blocks()`, `available_blocks()`, `block_size()`.
- **Block return**: `MutableBlock::drop()` calls a `return_fn: Arc<dyn Fn(Block<T, Reset>)>` closure. `InactivePool` uses `Arc::try_unwrap()` to reclaim blocks on return.
- **Reset pool allocator**: Pluggable `BlockAllocator<T>` trait. Default is `DequeBlockAllocator` (FIFO).

### Comparison Notes

v1 wraps every pool operation in an async request/response cycle even for in-memory operations. This was likely designed to serialize access and enable cross-task communication, but it means every `allocate_blocks()` call involves: sender.send() -> background task recv() -> process -> oneshot.send() -> caller.await. For operations that are fundamentally in-memory lookups and HashMap mutations, this introduces overhead per operation.

kvbm-logical's synchronous approach with `parking_lot` mutexes is simpler. The tradeoff is that callers block on lock acquisition rather than yielding to the async runtime. For the typical workload (short critical sections doing HashMap ops and BTreeSet mutations), this is appropriate — the lock hold times are measured in microseconds.

---

## 6. Block State Machine

### v1 (`block/state.rs`, `block.rs`)

- **Runtime enum**: `BlockState { Reset, Partial(PartialState), Complete(CompleteState), Registered(Arc<RegistrationHandle>, Arc<BlockHandle>) }`
- **Transitions**: Mutating methods on `Block` that return `Result<()>`. Invalid transitions produce `BlockStateInvalid` errors at runtime.
- **States**: Reset -> Partial (via `init_sequence`) -> Complete (via `commit`) -> Registered (via pool registration). Also Reset -> Complete (via `apply_token_block`).
- **Block type**: `Block<S: Storage, L: LocalityProvider, M: BlockMetadata>` — 3 generic parameters.
- **Partial state**: Supports incremental token building: `add_token()`, `add_tokens()`, `pop_token()`, `pop_tokens()`, `commit()`.

### kvbm-logical (`blocks/`)

- **Compile-time type-state**: `Block<T: BlockMetadata, State>` where `T` is a zero-sized storage tier marker (`struct G1;`, `struct G2;`, etc.) and `State` is one of `Reset`, `Staged`, `Registered` (also zero-sized markers). `BlockMetadata` is a blanket trait with no methods — any `Clone + Send + Sync + 'static` type satisfies it. The `T` parameter prevents mixing blocks from different storage tiers at compile time.
- **Transitions**: Consuming methods that return a new type. `MutableBlock.complete(token_block) -> CompleteBlock<T>`. `CompleteBlock.register() -> ImmutableBlock<T>`. Invalid transitions are compile errors.
- **States**: Reset -> Staged (via `complete()`) -> Registered (via registration). No Partial state — token building happens outside the block.
- **Block type**: `Block<T, State>` — 2 generic parameters. State is usually inferred.
- **Error handling**: `BlockError<B>` carries the block back on failure, preventing resource leaks. v1's errors don't return the block.

### Comparison Notes

v1 has a `Partial` state for incremental token building directly on the block. kvbm-logical does not — it expects a completed `TokenBlock` to be passed in. This means token-by-token building must happen externally in kvbm-logical.

kvbm-logical's type-state pattern prevents an entire class of runtime errors at compile time. v1's runtime enum is more flexible (can query state dynamically) but requires every caller to handle invalid state transitions.

---

## 7. Testability

| Aspect | v1 | kvbm-logical |
|--------|-----|-------------|
| Runtime requirement | Most tests need `#[tokio::test]` due to async pool ops | Synchronous. No async runtime needed |
| Block creation | Requires layout allocation, `BlockData::new(Arc<layout>, idx, set_idx, worker_id)`, `Block::new(data, metadata)` | `TestBlockBuilder::new(id).with_block_size(8).fill_iota(100).build_staged()` |
| Test utilities shipped | `create_reference_block_manager_config()` — requires physical setup | `testing` feature: `create_test_manager()`, `TestBlockBuilder`, `BlockSequenceBuilder`, `create_staged_block()`, `create_reset_blocks()` |
| Test metadata | `BasicMetadata` (production type) | `TestMeta(u64)`, `MetadataA`/`B`/`C` (purpose-built) |
| Test count | ~15 active tests, several commented-out test blocks | 194+ tests |
| Downstream test support | No test feature | `#[cfg(any(test, feature = "testing"))]` exposes utilities to consumers |

---

## 8. Future Strategy: Operating on Blocks by Sequence Hash

A key future direction is the ability to **operate on blocks by their sequence hash** across all pools — attaching metadata, pinning blocks, holding them for defined durations, and updating eviction ordering on touch or other events.

### How each implementation supports this

**v1**: The `GlobalRegistry` maps `SequenceHash -> Weak<RegistrationHandle>`, and `RegistrationHandle` is a fixed struct. To pin a block or attach metadata by hash, you would need to:
1. Add fields to `RegistrationHandle` (e.g., `pinned: AtomicBool`, `custom_metadata: HashMap<TypeId, Box<dyn Any>>`)
2. Modify pool eviction to check pinned state
3. Add touch/frequency tracking from scratch
4. Thread these changes through the async channel boundaries

**kvbm-logical**: The `BlockRegistry` maps `SequenceHash -> BlockRegistryHandle`, and `BlockRegistryHandle` has the attachment system. To pin a block or attach metadata by hash:
1. `handle.attach_unique::<PinState>(PinState::Pinned)` — no struct modification needed
2. `handle.on_touch(callback)` — register eviction-order update logic
3. Frequency tracking already integrated via TinyLFU
4. `handle.get::<PinState>().with_unique(|pin| ...)` to query pin state during eviction
5. Pools can check `handle.has_block::<T>()` and read attachments during eviction decisions

The attachment system + touch callbacks + presence tracking in kvbm-logical are directly designed for this workflow. In v1, each new "operate by hash" capability requires structural changes to `RegistrationHandle` and threading through async boundaries.

### Specific future patterns enabled by kvbm-logical

- **Pin by hash**: `registry.match_sequence_hash(hash)?.attach_unique::<Pinned>(Pinned(duration))`
- **Metadata by hash**: `handle.attach::<CustomMetadata>(data)` — multiple attachments per type supported
- **Cross-pool eviction update on touch**: `handle.on_touch(|h| update_eviction_order(h))` — fires when any pool touches the hash
- **Frequency-aware eviction**: MultiLRU backend reads frequency from TinyLFU tracker, promotes/demotes blocks across tiers automatically
- **Presence queries**: `handle.has_block::<G1Block>()` / `handle.has_any_block(&[g1_id, g2_id])` — check which storage tiers hold the block

---

## 9. Summary

| Dimension | v1 (`block_manager`) | `kvbm-logical` |
|-----------|---------------------|----------------|
| Registry data structure | HashMap (searches full keyspace) | PositionalRadixTree (narrows by position first, then searches within bucket) |
| Registry extensibility | Fixed struct fields | Typed attachment system |
| Frequency tracking | None (timestamp-based priority in offload path) | TinyLFU Count-Min Sketch, integrated with registry and MultiLRU |
| Inactive pool backends | Single (BTreeSet on metadata Ord) | 4 pluggable backends (HashMap, LRU, MultiLRU, Lineage) |
| Pool communication | Async channels + background task | Synchronous with parking_lot locks |
| Block state enforcement | Runtime enum | Compile-time type-state |
| Type parameters | 3 (`Storage`, `LocalityProvider`, `BlockMetadata`) | 2 (`T` storage tier marker, `State`) |
| BlockMetadata semantics | Trait with 4 methods (`on_acquired`, `on_returned`, `reset_metadata`, `offload_priority`) | Blanket trait (no methods) — `T` is a zero-sized marker like `struct G1;` |
| Error recovery | Block lost on error | `BlockError<B>` returns block |
| Test count | ~15 | 194+ |
| Test runtime | Async (tokio) | Synchronous |
| Token building on block | Yes (Partial state) | No (external) |
| Touch callbacks | No | Yes |
| WeakBlock support | No | Yes |
| MutableBlock type params | 3 (`S`, `L`, `M`) | 1 (`T`) |
| MutableBlock RAII return | async channel send (`mpsc::UnboundedSender`) | synchronous closure call |
| MutableBlock parent chain | Yes (iterative drop to avoid stack overflow) | No |
| ImmutableBlock inner type | `Arc<MutableBlock>` (mutable API still accessible via Deref) | `Arc<dyn RegisteredBlock<T>>` (read-only trait object) |
| WeakBlock | Not supported | `WeakBlock<T>` with two-stage upgrade (weak ref then pool search) |
| Block duplication types | Optional field on ImmutableBlock | Dedicated PrimaryBlock / DuplicateBlock types |
| Duplicate drop behavior | Unified with primary (try_take_block handles both) | Separate: DuplicateBlock resets and returns to reset pool; PrimaryBlock returns to inactive pool |
| Duplicate detection | async `wait_for_returned_block()` (unbounded wait on channel) | Presence markers + weak ref upgrade + bounded spin-loop (max 100 iterations) |
| Error recovery on registration | Block retained but no typed guarantee | `BlockError<B>` returns block inside error type |

### What kvbm-logical replaces

- `block/state.rs` -> `kvbm-logical::blocks::state`
- `block/registry.rs` -> `kvbm-logical::registry/`
- `block.rs` (Block, MutableBlock, ImmutableBlock) -> `kvbm-logical::blocks/`
- `pool.rs` + `pool/managed.rs` (logical pool ops) -> `kvbm-logical::pools/` + `kvbm-logical::manager/`
- `events.rs` -> `kvbm-logical::events/`

---

## 10. Verification

1. `cd lib/kvbm-logical && cargo test` — 194+ tests pass
2. Compare registries: `kvbm-logical/src/registry/` vs `block_manager/block/registry.rs`
3. Compare inactive pool backends: `kvbm-logical/src/pools/inactive/backends/` vs `block_manager/pool/managed/inactive.rs`
4. Compare pool communication: `kvbm-logical/src/manager/mod.rs` vs `block_manager/pool/managed.rs`
5. Compare block guard types: `kvbm-logical/src/blocks/` vs `block_manager/block.rs`
