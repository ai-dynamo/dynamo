# KVBM Extension Guide

KVBM exposes three extension points through the `kvbm_logical::ext` module for
customizing block lifecycle behavior:

| Extension Point        | Purpose                                    | Wired Via                                             |
|------------------------|--------------------------------------------|-------------------------------------------------------|
| `BlockAllocator`       | Control which reset block is allocated     | `BlockManager::builder().block_allocator(alloc)`      |
| `PresenceDelegate`    | React to block registration/deregistration | `BlockRegistry::builder().presence_delegate(d)`       |
| `InactivePoolBackend` | Control inactive pool storage and reset    | `BlockManager::builder().with_inactive_backend(b)`    |

All are optional — KVBM ships sensible defaults (`FifoBlockAllocator`, no
delegate, and `LineageBackend`).

## Custom BlockAllocator

### Trait signature

```rust
pub trait BlockAllocator<T: BlockMetadata> {
    fn insert(&mut self, block: Block<T, Reset>);
    fn pop(&mut self) -> Option<Block<T, Reset>>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool { self.len() == 0 }
}
```

`insert` is called whenever a block returns to the reset pool (e.g. a
`MutableBlock` is dropped without being staged). `pop` is called during
allocation to acquire a free block. The allocator decides the **order** in
which blocks are handed out.

### Example: DefragAllocator

The `defrag_allocator` example implements a lowest-ID-first strategy using a
`BTreeMap`:

```rust
struct DefragAllocator<T: BlockMetadata> {
    available: BTreeMap<BlockId, Block<T, Reset>>,
    shared: Arc<SharedDefragState>,
}

impl<T: BlockMetadata> BlockAllocator<T> for DefragAllocator<T> {
    fn insert(&mut self, block: Block<T, Reset>) {
        self.available.insert(block.block_id(), block);
    }

    fn pop(&mut self) -> Option<Block<T, Reset>> {
        let (_id, block) = self.available.pop_first()?;
        Some(block)
    }

    fn len(&self) -> usize {
        self.available.len()
    }
}
```

`BTreeMap::pop_first()` always returns the entry with the smallest key, so
freed block IDs are recycled in ascending order. This clusters active blocks
toward the low end of the ID space — a defragmentation effect.

### Wiring

```rust
let allocator = DefragAllocator::<GpuTier>::new(shared);

let manager = BlockManager::<GpuTier>::builder()
    .block_count(8)
    .block_size(4)
    .registry(registry)
    .block_allocator(allocator)
    .build()
    .expect("failed to build BlockManager");
```

The allocator is moved into the builder and wrapped in `Arc<Mutex<..>>` internally. It must implement `Send + Sync + 'static`.

### When blocks flow through the allocator

Blocks enter the allocator (`insert`) when:
- The `BlockManager` is first built (initial pool of `block_count` blocks)
- A `MutableBlock` is dropped (returned to reset pool)

Blocks leave the allocator (`pop`) when:
- `BlockManager::allocate_blocks()` draws from the reset pool

Note that blocks evicted from the **inactive pool** bypass the allocator —
they go directly to the caller as `MutableBlock`s. They only re-enter the
allocator if the caller drops the `MutableBlock` without staging it.

## PresenceDelegate

### Trait signature

```rust
pub trait PresenceDelegate: Send + Sync {
    fn on_present(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        handle: &BlockRegistrationHandle,
    );

    fn on_absent(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        handle: &BlockRegistrationHandle,
    );
}
```

`on_present` fires when a block transitions **Staged → Registered** (via
`BlockManager::register_block`). `on_absent` fires when a registered block is
evicted (transitions **Registered → Reset**).

Both callbacks execute **synchronously** on the calling thread, outside any
pool lock. Implementations must be lightweight and non-blocking.

### TypeId filtering

The `type_id` parameter carries `TypeId::of::<T>()` where `T` is the
`BlockMetadata` type of the block. This allows a single delegate to handle
multiple tiers and filter by type:

```rust
impl PresenceDelegate for DefragPresenceDelegate {
    fn on_present(&self, seq_hash: SequenceHash, block_id: BlockId,
                  type_id: TypeId, _handle: &BlockRegistrationHandle) {
        if self.filter_type_id.is_some() && self.filter_type_id != Some(type_id) {
            return; // ignore blocks from other tiers
        }
        // handle the event
    }
    // ...
}
```

### `typed_presence_delegate` helper

For simple cases, use the `typed_presence_delegate` function to create a
delegate from two closures that only fires for a specific metadata type:

```rust
use kvbm_logical::ext::typed_presence_delegate;

let delegate = typed_presence_delegate::<GpuTier>(
    |seq_hash, block_id, handle| {
        println!("GPU block {block_id} registered");
    },
    |seq_hash, block_id, handle| {
        println!("GPU block {block_id} deregistered");
    },
);
```

### Wiring

```rust
let registry = BlockRegistry::builder()
    .presence_delegate(delegate)  // Arc<dyn PresenceDelegate>
    .build();
```

Multiple delegates can be added by calling `.presence_delegate()` repeatedly.

## Custom InactivePoolBackend

### Trait signature

```rust
pub trait InactivePoolBackend<T: BlockMetadata>: Send + Sync {
    fn find_matches(&mut self, hashes: &[SequenceHash], touch: bool) -> Vec<Block<T, Registered>>;
    fn scan_matches(&mut self, hashes: &[SequenceHash], touch: bool)
        -> Vec<(SequenceHash, Block<T, Registered>)>;
    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>>;
    fn insert(&mut self, block: Block<T, Registered>);
    fn len(&self) -> usize;
    fn has_block(&self, seq_hash: SequenceHash) -> bool;

    // Provided defaults:
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn allocate_all(&mut self) -> Vec<Block<T, Registered>> { /* calls len() + allocate() */ }
    fn should_reset(&self, block: &Block<T, Registered>) -> bool { false }
}
```

The inactive pool stores blocks in the **Registered** state after their last
`ImmutableBlock` guard is dropped. It serves two roles:

1. **Cache** — blocks can be found again by `SequenceHash` via `find_matches`
   or `scan_matches`, avoiding recomputation.
2. **Eviction source** — `allocate` removes blocks to satisfy new allocation
   requests when the reset pool is empty.

### The `should_reset` hook

When an `ImmutableBlock` is dropped and its `Arc` refcount reaches zero, the
inactive pool calls `should_reset(&block)` on the backend **before** inserting
the block. If `should_reset` returns `true`, the block is immediately reset
and returned to the reset pool, bypassing the inactive pool entirely.

This is useful when you want certain blocks to skip caching — for example,
single-use blocks that are unlikely to be reused.

### Built-in: `ResetInactiveBlocksBackend`

`ResetInactiveBlocksBackend` is a no-op backend where `should_reset` always
returns `true`. Every dropped block goes straight back to the reset pool:

```rust
use kvbm_logical::ext::ResetInactiveBlocksBackend;

let manager = BlockManager::<MyMeta>::builder()
    .block_count(64)
    .block_size(16)
    .registry(registry)
    .with_inactive_backend(ResetInactiveBlocksBackend)
    .build()
    .expect("build failed");
```

This effectively disables the inactive cache. No block is ever stored or
looked up — all blocks are recycled immediately.

### Wrapping pattern: LoggingInactiveBackend

The `defrag_allocator` example demonstrates wrapping an existing backend with
a logging decorator:

```rust
struct LoggingInactiveBackend<T: BlockMetadata> {
    inner: Box<dyn InactivePoolBackend<T>>,
}

impl<T: BlockMetadata> InactivePoolBackend<T> for LoggingInactiveBackend<T> {
    fn insert(&mut self, block: Block<T, Registered>) {
        let block_id = block.block_id();
        self.inner.insert(block);
        eprintln!("[LoggingBackend] insert(block={block_id})");
    }

    fn should_reset(&self, block: &Block<T, Registered>) -> bool {
        let result = self.inner.should_reset(block);
        eprintln!("[LoggingBackend] should_reset(block={}) → {result}", block.block_id());
        result
    }

    // ... delegate remaining methods to self.inner
}
```

This pattern lets you add observability, filtering, or policy logic around
any inner backend without modifying it.

### Wiring

```rust
use kvbm_logical::pools::LineageBackend;

let logging_backend = LoggingInactiveBackend::<GpuTier>::new(
    Box::new(LineageBackend::default()),
);

let manager = BlockManager::<GpuTier>::builder()
    .block_count(8)
    .block_size(4)
    .registry(registry)
    .with_inactive_backend(logging_backend)
    .build()
    .expect("failed to build BlockManager");
```

When `with_inactive_backend()` is set, it takes priority over the built-in
backend config methods (`with_lru_backend()`, `with_lineage_backend()`, etc.).
The backend must implement `InactivePoolBackend<T> + 'static`.

## Connecting Allocator and Delegate

The allocator and delegate serve different purposes, but sometimes they need to
coordinate — for example, an allocator that adapts its strategy based on how
many blocks are currently registered.

Use an `Arc<SharedState>` to share data between them:

```rust
struct SharedDefragState {
    registered_count: AtomicUsize,
}

let shared = Arc::new(SharedDefragState {
    registered_count: AtomicUsize::new(0),
});

// Both components hold an Arc to the same state
let allocator = DefragAllocator::new(Arc::clone(&shared));
let delegate = DefragPresenceDelegate::for_tier::<GpuTier>(
    Arc::clone(&shared), "GPU",
);
```

The delegate updates `registered_count` in its callbacks, and the allocator can
read it when deciding which block to return. Because the allocator runs under a
`Mutex` and the delegate callbacks run outside any lock, use atomic types or
other lock-free primitives for shared fields.

## Running the Example

```bash
cargo run -p kvbm-logical --example defrag_allocator
```

**stdout** shows the allocation flow:
- Step 1–2: Initial allocation and registration of 8 blocks (IDs 0–7)
- Step 3: Drop blocks at odd indices (1, 3, 5, 7) — they move to inactive pool
- Step 4: Allocate 4 blocks — evicts from inactive pool
- Step 5–6: Return evicted blocks to reset pool, then reallocate — DefragAllocator
  returns IDs 1, 3, 5, 7 (lowest first)

**stderr** shows all three extension points in action:
- `[DefragAllocator]` — allocator pop events with pool size and registered count
- `[GPUDelegate]` — `on_present` / `on_absent` events with sequence hashes
- `[LoggingBackend]` — `insert`, `allocate`, and `should_reset` calls on the inactive pool

## When to Use Each

### Custom BlockAllocator

- **Defragmentation**: Return lowest IDs first to pack allocations tightly
- **Priority allocation**: Prefer blocks with specific properties (e.g., NUMA locality)
- **Remote/networked pools**: Allocate blocks from a remote block server
- **Metrics/debugging**: Log every allocation decision

### PresenceDelegate

- **Cross-tier coordination**: Notify a CPU-tier manager when a GPU-tier block is registered
- **External cache invalidation**: Trigger invalidation in an external system when blocks are evicted
- **Monitoring**: Track registration rates, detect anomalies
- **Replication**: Mirror block presence state to a remote replica

### Custom InactivePoolBackend

- **Disable caching**: Use `ResetInactiveBlocksBackend` to skip the inactive pool entirely
- **Observability**: Wrap a backend to log or meter insert/allocate/reset decisions
- **Custom eviction**: Implement domain-specific eviction policies beyond the built-in LRU/Lineage options
- **Conditional reset**: Use `should_reset` to selectively bypass caching for certain blocks (e.g., single-use prefills)
