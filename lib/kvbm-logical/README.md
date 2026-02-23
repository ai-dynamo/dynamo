# kvbm-logical

Logical block lifecycle management for KVBM (KV Block Manager). Manages KV cache blocks for LLM inference through a type-safe state machine, registry, and pool system.

## Block Lifecycle

Blocks follow a compile-time enforced state machine via the type-state pattern:

```text
MutableBlock<T> → CompleteBlock<T> → ImmutableBlock<T> ⇄ WeakBlock<T>
   (Reset)           (Staged)          (Registered)       (Non-owning)
```

- **MutableBlock** — Allocated from the reset pool, writable. Drop returns to the reset pool.
- **CompleteBlock** — Staged with a `SequenceHash` but not yet registered. Drop returns to the reset pool.
- **ImmutableBlock** — Registered in the block registry. Strong-ref prevents eviction. Drop moves to the inactive pool for caching.
- **WeakBlock** — Non-owning reference that does not prevent eviction. Upgradeable back to `ImmutableBlock` via two-phase lookup.

The type parameter `T: BlockMetadata` is a marker for the storage tier (e.g. GPU, CPU, disk).

## Usage

```rust,no_run
use kvbm_logical::{
    BlockManager, BlockRegistry, MutableBlock, CompleteBlock, ImmutableBlock, WeakBlock,
    SequenceHash,
    manager::FrequencyTrackingCapacity,
};

# fn main() {
// Any Clone + Send + Sync + 'static type satisfies BlockMetadata.
#[derive(Clone)]
struct G2; // CPU tier marker

// Build a registry with TinyLFU frequency tracking.
let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
let registry = BlockRegistry::builder()
    .frequency_tracker(tracker)
    .build();

// Build the block manager with an LRU eviction backend.
let manager = BlockManager::<G2>::builder()
    .block_count(1024)
    .block_size(16)
    .registry(registry)
    .with_lru_backend()
    .build()
    .expect("failed to build block manager");

// Allocate mutable blocks from the reset pool.
let mut blocks: Vec<MutableBlock<G2>> = manager
    .allocate_blocks(2)
    .expect("not enough blocks available");

// Stage a block with a pre-computed sequence hash, producing a CompleteBlock.
// SequenceHash wraps a positional lineage hash computed from token data.
let seq_hash_0 = SequenceHash::new(42, None, 0);
let complete: CompleteBlock<G2> = blocks
    .remove(0)
    .stage(seq_hash_0, manager.block_size())
    .expect("block size should match");

// Register the staged block, producing an ImmutableBlock.
let immutable: ImmutableBlock<G2> = manager.register_block(complete);

// Prefix-match registered blocks by sequence hash.
let matched: Vec<ImmutableBlock<G2>> = manager.match_blocks(&[seq_hash_0]);
assert_eq!(matched.len(), 1);

// Downgrade to a WeakBlock (does not prevent eviction).
let weak: WeakBlock<G2> = immutable.downgrade();

// Upgrade back to ImmutableBlock if the block hasn't been evicted.
if let Some(restored) = weak.upgrade() {
    assert_eq!(restored.sequence_hash(), seq_hash_0);
}

// RAII: dropping an ImmutableBlock moves it to the inactive pool for caching.
{
    let temporary = manager.match_blocks(&[seq_hash_0]);
    // `temporary` dropped here → block returns to inactive pool
}

// Introspect pool state.
let available = manager.available_blocks();
let total = manager.total_blocks();
# }
```

## Prometheus Metrics

All metrics carry a `pool` label identifying the storage tier.

### Counters

| Name | Description |
|------|-------------|
| `kvbm_allocations_total` | Total blocks allocated from pools |
| `kvbm_allocations_from_reset_total` | Total blocks allocated from the reset pool |
| `kvbm_evictions_total` | Total blocks evicted from inactive pool |
| `kvbm_registrations_total` | Total blocks registered (CompleteBlock → ImmutableBlock) |
| `kvbm_duplicate_blocks_total` | Total duplicate blocks created (Allow policy) |
| `kvbm_registration_dedup_total` | Total block registrations deduplicated (Reject policy) |
| `kvbm_stagings_total` | Total MutableBlock → CompleteBlock transitions |
| `kvbm_match_hashes_requested_total` | Total hashes requested in match_blocks calls |
| `kvbm_match_blocks_returned_total` | Total blocks returned from match_blocks calls |
| `kvbm_scan_hashes_requested_total` | Total hashes requested in scan_matches calls |
| `kvbm_scan_blocks_returned_total` | Total blocks returned from scan_matches calls |

### Gauges

| Name | Description |
|------|-------------|
| `kvbm_inflight_mutable` | Current MutableBlocks held outside pool |
| `kvbm_inflight_immutable` | Current ImmutableBlocks held outside pool |
| `kvbm_reset_pool_size` | Current reset pool size |
| `kvbm_inactive_pool_size` | Current inactive pool size |
