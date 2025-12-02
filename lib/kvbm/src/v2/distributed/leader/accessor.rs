// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block accessor for policy-based scanning.
//!
//! Provides a stateless interface for acquiring blocks from G2/G3 tiers.
//! Designed for use with custom scanning policies that control iteration
//! and can yield results incrementally.

use crate::{
    logical::blocks::ImmutableBlock,
    v2::{BlockId, G2, G3, SequenceHash, logical::LogicalLayoutHandle},
};

use super::InstanceLeader;

/// A block from either G2 or G3 tier.
///
/// Provides RAII ownership - blocks are released when dropped.
#[derive(Debug)]
pub enum TieredBlock {
    /// Block from G2 (host memory) tier.
    G2(ImmutableBlock<G2>),
    /// Block from G3 (disk) tier.
    G3(ImmutableBlock<G3>),
}

impl TieredBlock {
    /// Get the storage tier of this block.
    pub fn tier(&self) -> LogicalLayoutHandle {
        match self {
            TieredBlock::G2(_) => LogicalLayoutHandle::G2,
            TieredBlock::G3(_) => LogicalLayoutHandle::G3,
        }
    }

    /// Get the sequence hash.
    pub fn sequence_hash(&self) -> SequenceHash {
        match self {
            TieredBlock::G2(b) => b.sequence_hash(),
            TieredBlock::G3(b) => b.sequence_hash(),
        }
    }

    /// Get the block ID.
    pub fn block_id(&self) -> BlockId {
        match self {
            TieredBlock::G2(b) => b.block_id(),
            TieredBlock::G3(b) => b.block_id(),
        }
    }

    /// Get the position in the sequence (for ordering).
    pub fn position(&self) -> u64 {
        self.sequence_hash().position()
    }

    /// Check if this is a G2 block.
    pub fn is_g2(&self) -> bool {
        matches!(self, TieredBlock::G2(_))
    }

    /// Check if this is a G3 block.
    pub fn is_g3(&self) -> bool {
        matches!(self, TieredBlock::G3(_))
    }

    /// Convert to G2 block, consuming self.
    pub fn into_g2(self) -> Option<ImmutableBlock<G2>> {
        match self {
            TieredBlock::G2(b) => Some(b),
            TieredBlock::G3(_) => None,
        }
    }

    /// Convert to G3 block, consuming self.
    pub fn into_g3(self) -> Option<ImmutableBlock<G3>> {
        match self {
            TieredBlock::G3(b) => Some(b),
            TieredBlock::G2(_) => None,
        }
    }
}

/// Stateless accessor for block acquisition.
///
/// Each method call is independent - no locks are held between calls.
/// This enables parallel policy execution (e.g., with rayon).
///
/// # Thread Safety
///
/// `BlockAccessor` is `Send + Sync` because:
/// - It only holds a shared reference to `InstanceLeader`
/// - `InstanceLeader` contains `Arc<BlockManager<T>>` which is `Send + Sync`
/// - All operations use internal locking per call
/// - No mutable state is held between method calls
pub struct BlockAccessor<'a> {
    instance: &'a InstanceLeader,
    touch: bool,
}

impl<'a> BlockAccessor<'a> {
    /// Create a new accessor.
    pub(crate) fn new(instance: &'a InstanceLeader, touch: bool) -> Self {
        Self { instance, touch }
    }

    /// Find and take a block from G2 or G3.
    ///
    /// Searches G2 first, then G3 if not found. The block is acquired/removed
    /// from the pool - caller owns via RAII until dropped.
    ///
    /// Returns `None` if the block is not found in either tier.
    pub fn find(&self, hash: SequenceHash) -> Option<TieredBlock> {
        // Try G2 first (match_blocks acquires the block)
        let g2_matches = self.instance.g2_manager.match_blocks(&[hash]);
        if let Some(block) = g2_matches.into_iter().next() {
            return Some(TieredBlock::G2(block));
        }

        // Try G3 if available
        if let Some(ref g3) = self.instance.g3_manager {
            let g3_matches = g3.match_blocks(&[hash]);
            if let Some(block) = g3_matches.into_iter().next() {
                return Some(TieredBlock::G3(block));
            }
        }

        None
    }

    /// Get the touch setting for this accessor.
    ///
    /// When `true`, frequency tracking is updated on block access
    /// (affects MultiLRU eviction priority).
    pub fn touch(&self) -> bool {
        self.touch
    }
}

// Safety: BlockAccessor is Send + Sync because:
// - It only holds a shared reference to InstanceLeader
// - InstanceLeader contains Arc<BlockManager<T>> which is Send + Sync
// - All operations use internal locking per call (RwLock in InactivePool)
// - No mutable state is held between method calls
unsafe impl Send for BlockAccessor<'_> {}
unsafe impl Sync for BlockAccessor<'_> {}

/// Context for policy execution with result collection.
///
/// Provides access to the `BlockAccessor` for block lookups and a
/// `yield_item` method for streaming results back to the caller.
pub struct PolicyContext<'a, T> {
    pub(crate) accessor: BlockAccessor<'a>,
    pub(crate) results: Vec<T>,
}

impl<'a, T> PolicyContext<'a, T> {
    /// Get access to the block accessor.
    pub fn accessor(&self) -> &BlockAccessor<'a> {
        &self.accessor
    }

    /// Yield a result item.
    ///
    /// Items are collected and returned as a `Vec<T>` when the policy completes.
    pub fn yield_item(&mut self, item: T) {
        self.results.push(item);
    }

    /// Yield multiple result items at once.
    pub fn yield_items(&mut self, items: impl IntoIterator<Item = T>) {
        self.results.extend(items);
    }
}

// =============================================================================
// TODO: Parallel policy support via rayon::scope
//
// Requirements to enable:
// 1. Add `rayon` to Cargo.toml dependencies
// 2. Ensure BlockAccessor is truly Send+Sync (verify internal locking is correct)
// 3. Add feature flag `parallel` to gate this code
// 4. Test thread-safety of concurrent BlockManager::match_blocks calls
// 5. Benchmark to ensure parallel overhead is worth it (likely only for large hash sets)
//
// The design uses rayon::scope instead of par_chunks because:
// - par_chunks could split across logical boundaries (e.g., middle of a contiguous run)
// - rayon::scope lets the policy control parallelism granularity
// - Policy can identify natural split points (e.g., gaps in position sequence)
//
// use std::sync::Mutex;
// use rayon;
//
// /// Context for parallel policy execution.
// /// Provides thread-safe result collection via Mutex.
// pub struct ParallelPolicyContext<'a, 's, T> {
//     pub(crate) accessor: &'a BlockAccessor<'a>,
//     pub(crate) scope: &'s rayon::Scope<'s>,
//     pub(crate) results: &'a Mutex<Vec<T>>,
// }
//
// impl<'a, 's, T: Send> ParallelPolicyContext<'a, 's, T> {
//     /// Get access to the block accessor.
//     pub fn accessor(&self) -> &BlockAccessor<'a> {
//         self.accessor
//     }
//
//     /// Yield a result item (thread-safe).
//     pub fn yield_item(&self, item: T) {
//         self.results.lock().unwrap().push(item);
//     }
//
//     /// Yield multiple result items (thread-safe, single lock acquisition).
//     pub fn yield_items(&self, items: impl IntoIterator<Item = T>) {
//         self.results.lock().unwrap().extend(items);
//     }
//
//     /// Spawn parallel work within the rayon scope.
//     ///
//     /// The closure receives the accessor and results mutex, allowing it to
//     /// perform lookups and yield items from a separate thread.
//     ///
//     /// # Example
//     /// ```ignore
//     /// ctx.spawn(|accessor, results| {
//     ///     for hash in my_segment {
//     ///         if let Some(block) = accessor.find(hash) {
//     ///             results.lock().unwrap().push(block);
//     ///         }
//     ///     }
//     /// });
//     /// ```
//     pub fn spawn<F>(&self, f: F)
//     where
//         F: FnOnce(&BlockAccessor, &Mutex<Vec<T>>) + Send + 'a,
//     {
//         let accessor = self.accessor;
//         let results = self.results;
//         self.scope.spawn(move |_| {
//             f(accessor, results);
//         });
//     }
// }
// =============================================================================
