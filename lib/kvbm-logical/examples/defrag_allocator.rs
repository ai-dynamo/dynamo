// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example: DefragAllocator with PresenceDelegate and LoggingInactiveBackend
//!
//! Demonstrates all three KVBM extension points:
//!
//! - **[`BlockAllocator`]** — `DefragAllocator` uses a `BTreeMap` so that
//!   `pop()` always returns the lowest available `BlockId`.
//! - **[`PresenceDelegate`]** — `DefragPresenceDelegate` logs registration
//!   events and maintains a live count of registered blocks.
//! - **[`InactivePoolBackend`]** — `LoggingInactiveBackend` wraps any inner
//!   backend, logging `insert`, `allocate`, and `should_reset` calls to stderr.
//!
//! All three share state through an `Arc<SharedDefragState>`.
//!
//! Run with:
//! ```sh
//! cargo run -p kvbm-logical --example defrag_allocator
//! ```

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kvbm_logical::ext::{
    Block, BlockAllocator, BlockId, BlockMetadata, InactivePoolBackend, PresenceDelegate,
    Registered, Reset, SequenceHash,
};
use kvbm_logical::pools::LineageBackend;
use kvbm_logical::registry::BlockRegistrationHandle;
use kvbm_logical::{BlockManager, BlockRegistry};

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// State shared between the allocator and the presence delegate.
struct SharedDefragState {
    registered_count: AtomicUsize,
}

// ---------------------------------------------------------------------------
// Tier markers
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GpuTier;

#[derive(Clone)]
#[allow(dead_code)]
struct CpuTier;

// ---------------------------------------------------------------------------
// DefragAllocator
// ---------------------------------------------------------------------------

/// A block allocator that always returns the block with the lowest `BlockId`.
///
/// Internally it keeps a `BTreeMap<BlockId, Block>` so that `pop()` removes the
/// first (smallest) entry. This produces a compact, defragmented allocation
/// pattern — reused IDs cluster at the low end of the ID space.
struct DefragAllocator<T: BlockMetadata> {
    /// Blocks sorted by ID — `BTreeMap::pop_first()` gives the lowest ID.
    available: BTreeMap<BlockId, Block<T, Reset>>,
    shared: Arc<SharedDefragState>,
}

impl<T: BlockMetadata> DefragAllocator<T> {
    fn new(shared: Arc<SharedDefragState>) -> Self {
        Self {
            available: BTreeMap::new(),
            shared,
        }
    }
}

impl<T: BlockMetadata> BlockAllocator<T> for DefragAllocator<T> {
    fn insert(&mut self, block: Block<T, Reset>) {
        self.available.insert(block.block_id(), block);
    }

    fn pop(&mut self) -> Option<Block<T, Reset>> {
        let (_id, block) = self.available.pop_first()?;
        eprintln!(
            "  [DefragAllocator] popped block {} (pool size now {}, registered {})",
            block.block_id(),
            self.available.len(),
            self.shared.registered_count.load(Ordering::Relaxed),
        );
        Some(block)
    }

    fn len(&self) -> usize {
        self.available.len()
    }
}

// ---------------------------------------------------------------------------
// DefragPresenceDelegate
// ---------------------------------------------------------------------------

/// A presence delegate that logs registration events and updates shared state.
///
/// Optionally filters by a specific `TypeId` so that it only fires for blocks
/// of a particular metadata tier.
struct DefragPresenceDelegate {
    shared: Arc<SharedDefragState>,
    filter_type_id: Option<TypeId>,
    tier_name: String,
}

impl DefragPresenceDelegate {
    /// Create a delegate that fires only for blocks with metadata type `T`.
    fn for_tier<T: BlockMetadata>(shared: Arc<SharedDefragState>, tier_name: &str) -> Self {
        Self {
            shared,
            filter_type_id: Some(TypeId::of::<T>()),
            tier_name: tier_name.to_string(),
        }
    }
}

impl PresenceDelegate for DefragPresenceDelegate {
    fn on_present(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        _handle: &BlockRegistrationHandle,
    ) {
        if self.filter_type_id.is_some() && self.filter_type_id != Some(type_id) {
            return;
        }
        let prev = self.shared.registered_count.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "  [{}Delegate] on_present  block={block_id}  seq_hash={seq_hash}  registered={}",
            self.tier_name,
            prev + 1,
        );
    }

    fn on_absent(
        &self,
        seq_hash: SequenceHash,
        block_id: BlockId,
        type_id: TypeId,
        _handle: &BlockRegistrationHandle,
    ) {
        if self.filter_type_id.is_some() && self.filter_type_id != Some(type_id) {
            return;
        }
        let prev = self.shared.registered_count.fetch_sub(1, Ordering::Relaxed);
        eprintln!(
            "  [{}Delegate] on_absent   block={block_id}  seq_hash={seq_hash}  registered={}",
            self.tier_name,
            prev - 1,
        );
    }
}

// ---------------------------------------------------------------------------
// LoggingInactiveBackend
// ---------------------------------------------------------------------------

/// A custom [`InactivePoolBackend`] that wraps an inner backend and logs
/// `insert`, `allocate`, and `should_reset` calls to stderr.
///
/// This demonstrates the wrapping/decorator pattern for inactive backends:
/// all real work is delegated to the inner backend while the wrapper adds
/// observability.
struct LoggingInactiveBackend<T: BlockMetadata> {
    inner: Box<dyn InactivePoolBackend<T>>,
}

impl<T: BlockMetadata> LoggingInactiveBackend<T> {
    fn new(inner: Box<dyn InactivePoolBackend<T>>) -> Self {
        Self { inner }
    }
}

impl<T: BlockMetadata> InactivePoolBackend<T> for LoggingInactiveBackend<T> {
    fn find_matches(&mut self, hashes: &[SequenceHash], touch: bool) -> Vec<Block<T, Registered>> {
        let result = self.inner.find_matches(hashes, touch);
        eprintln!(
            "  [LoggingBackend] find_matches({} hashes, touch={touch}) → {} hits",
            hashes.len(),
            result.len(),
        );
        result
    }

    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Block<T, Registered>)> {
        let result = self.inner.scan_matches(hashes, touch);
        eprintln!(
            "  [LoggingBackend] scan_matches({} hashes, touch={touch}) → {} hits",
            hashes.len(),
            result.len(),
        );
        result
    }

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>> {
        let result = self.inner.allocate(count);
        eprintln!(
            "  [LoggingBackend] allocate({count}) → {} blocks (pool now {})",
            result.len(),
            self.inner.len(),
        );
        result
    }

    fn insert(&mut self, block: Block<T, Registered>) {
        let block_id = block.block_id();
        self.inner.insert(block);
        eprintln!(
            "  [LoggingBackend] insert(block={block_id}) (pool now {})",
            self.inner.len(),
        );
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.inner.has_block(seq_hash)
    }

    fn should_reset(&self, block: &Block<T, Registered>) -> bool {
        let result = self.inner.should_reset(block);
        eprintln!(
            "  [LoggingBackend] should_reset(block={}) → {result}",
            block.block_id(),
        );
        result
    }
}

// ---------------------------------------------------------------------------
// Helper: create unique sequence hashes
// ---------------------------------------------------------------------------

fn make_seq_hash(index: u64) -> SequenceHash {
    // Each block gets a unique hash value and position.
    // Use index as position and a derived value as the hash.
    SequenceHash::new(0x1000 + index, None, index)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    const BLOCK_COUNT: usize = 8;
    const BLOCK_SIZE: usize = 4; // tokens per block (must be power of 2)

    println!("=== KVBM DefragAllocator Example ===\n");

    // 1. Shared state ----------------------------------------------------------
    let shared = Arc::new(SharedDefragState {
        registered_count: AtomicUsize::new(0),
    });

    // 2. Build registry with presence delegate ---------------------------------
    let delegate = Arc::new(DefragPresenceDelegate::for_tier::<GpuTier>(
        Arc::clone(&shared),
        "GPU",
    ));

    let registry = BlockRegistry::builder()
        .presence_delegate(delegate)
        .build();

    // 3. Build BlockManager with DefragAllocator + LoggingInactiveBackend --------
    let allocator = DefragAllocator::<GpuTier>::new(Arc::clone(&shared));

    // Wrap the default LineageBackend in a logging decorator
    let logging_backend = LoggingInactiveBackend::<GpuTier>::new(
        Box::new(LineageBackend::default()),
    );

    let manager = BlockManager::<GpuTier>::builder()
        .block_count(BLOCK_COUNT)
        .block_size(BLOCK_SIZE)
        .registry(registry)
        .block_allocator(allocator)
        .with_inactive_backend(logging_backend)
        .build()
        .expect("failed to build BlockManager");

    // 4. Allocate all 8 blocks --------------------------------------------------
    println!("Step 1: Allocate {BLOCK_COUNT} blocks");
    let mutable_blocks = manager
        .allocate_blocks(BLOCK_COUNT)
        .expect("failed to allocate blocks");

    let mut block_ids: Vec<BlockId> = Vec::new();
    for mb in &mutable_blocks {
        block_ids.push(mb.block_id());
    }
    println!("  Allocated block IDs: {block_ids:?}");

    // 5. Stage and register all blocks ------------------------------------------
    println!("\nStep 2: Stage and register all {BLOCK_COUNT} blocks");
    let mut immutable_blocks = Vec::new();
    let mut seq_hashes = HashMap::new();

    for (i, mb) in mutable_blocks.into_iter().enumerate() {
        let bid = mb.block_id();
        let sh = make_seq_hash(i as u64);
        seq_hashes.insert(bid, sh);

        let complete = mb.stage(sh, BLOCK_SIZE).expect("stage failed");
        let immutable = manager.register_block(complete);
        println!(
            "  Registered block {} with seq_hash {sh}",
            immutable.block_id()
        );
        immutable_blocks.push(immutable);
    }

    println!(
        "\n  Registered count (shared state): {}",
        shared.registered_count.load(Ordering::Relaxed)
    );

    // 6. Drop blocks at odd indices (1, 3, 5, 7) -------------------------------
    println!("\nStep 3: Drop blocks at odd indices (1, 3, 5, 7)");
    // Remove odd-indexed blocks (reverse order to keep indices stable)
    for idx in [7, 5, 3, 1] {
        let block = immutable_blocks.remove(idx);
        println!("  Dropping block {} (index {idx})", block.block_id());
        drop(block);
    }

    println!(
        "\n  Registered count after drops: {}",
        shared.registered_count.load(Ordering::Relaxed)
    );

    // 7. Allocate 4 more blocks -------------------------------------------------
    // The reset pool is empty, so blocks are evicted from the inactive pool.
    // Eviction order depends on the inactive backend (default LRU).
    // On return (drop of MutableBlock), blocks re-enter the DefragAllocator
    // where BTreeMap ordering ensures lowest-ID-first allocation on the NEXT cycle.
    println!("\nStep 4: Allocate 4 new blocks (evicted from inactive pool)");
    let new_blocks = manager
        .allocate_blocks(4)
        .expect("failed to allocate new blocks");

    let new_ids: Vec<BlockId> = new_blocks.iter().map(|b| b.block_id()).collect();
    println!("  New block IDs (defrag order): {new_ids:?}");

    // 8. Drop the evicted blocks back — they return to DefragAllocator ----------
    println!("\nStep 5: Drop 4 evicted blocks (return to DefragAllocator reset pool)");
    for mb in new_blocks {
        println!("  Dropping mutable block {}", mb.block_id());
        drop(mb);
    }

    // 9. Reallocate — DefragAllocator now returns lowest IDs first ---------------
    println!("\nStep 6: Reallocate 4 blocks — DefragAllocator returns lowest IDs first");
    let defrag_blocks = manager
        .allocate_blocks(4)
        .expect("failed to allocate blocks");

    let defrag_ids: Vec<BlockId> = defrag_blocks.iter().map(|b| b.block_id()).collect();
    println!("  Defragmented block IDs: {defrag_ids:?}");

    // Stage and register them
    println!("\nStep 7: Stage and register the defragmented blocks");
    for (i, mb) in defrag_blocks.into_iter().enumerate() {
        let sh = make_seq_hash(200 + i as u64);
        let complete = mb.stage(sh, BLOCK_SIZE).expect("stage failed");
        let immutable = manager.register_block(complete);
        println!("  Registered block {} with seq_hash {sh}", immutable.block_id());
        immutable_blocks.push(immutable);
    }

    // 10. Summary ----------------------------------------------------------------
    println!("\n=== Summary ===");
    println!(
        "  Total registered blocks: {}",
        shared.registered_count.load(Ordering::Relaxed)
    );
    let final_ids: Vec<BlockId> = immutable_blocks.iter().map(|b| b.block_id()).collect();
    println!("  Active block IDs: {final_ids:?}");
    println!(
        "  Defrag effect: evicted IDs were re-sorted by DefragAllocator (lowest first)"
    );
    println!("\n=== Done ===");
}
