// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe pool for registered immutable blocks with automatic RAII return.
//!
//! Manages blocks in the Registered state, providing:
//! - Finding blocks by sequence hash with O(1) lookup
//! - Conversion of registered blocks back to mutable blocks for reuse
//! - Thread-safe access via interior mutability
//! - Automatic block return via RAII ImmutableBlock guards

pub(crate) mod backends;

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use super::{
    Block, BlockId, BlockMetadata, InactiveBlock, MutableBlock, PrimaryBlock, Registered,
    RegisteredBlock, Reset, SequenceHash, reset::ResetPool,
};

// pub(crate) use backends::*;

/// Backend trait for InactivePool storage strategies
pub(crate) trait InactivePoolBackend<T: BlockMetadata>: Send + Sync {
    /// Find blocks matching the given hashes in order, stopping on first miss.
    fn find_matches(&mut self, hashes: &[SequenceHash], touch: bool) -> Vec<Block<T, Registered>>;

    /// Scan for blocks matching any of the given hashes (full scan, doesn't stop on miss).
    /// Unlike find_matches, continues scanning even when a hash is not found.
    /// Acquires/removes found blocks from pool (caller owns until dropped).
    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Block<T, Registered>)>;

    fn allocate(&mut self, count: usize) -> Vec<Block<T, Registered>>;

    fn insert(&mut self, block: Block<T, Registered>);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    fn has_block(&self, seq_hash: SequenceHash) -> bool;

    /// Allocate all blocks from the pool, removing them from the backend.
    /// Default implementation calls len() then allocate(), which is atomic
    /// since the caller holds the lock.
    fn allocate_all(&mut self) -> Vec<Block<T, Registered>> {
        let count = self.len();
        self.allocate(count)
    }
}
/// Pool for managing registered (immutable) blocks
///
/// This pool handles blocks in the Registered state and provides them as
/// RegisteredBlock RAII guards that automatically return to the pool on drop.
/// Type alias for reset block return function
type ResetReturnFn<T> = Arc<dyn Fn(Block<T, Reset>) + Send + Sync>;

/// Type alias for registered block return function
type RegisteredReturnFn<T> = Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>;

#[derive(Clone)]
pub struct InactivePool<T: BlockMetadata> {
    // Inner state protected by RwLock for thread-safe access from guards
    inner: Arc<RwLock<InactivePoolInner<T>>>,
    // Return function for MutableBlocks to return to ResetPool
    reset_return_fn: ResetReturnFn<T>,

    return_fn: RegisteredReturnFn<T>,
    #[expect(dead_code)]
    block_size: usize,
}

/// Weak references to an active block for resurrection during transitions.
///
/// We store both weak references because:
/// - `primary_block`: Try this first - upgrading is cheap and avoids creating a new PrimaryBlock
/// - `raw_block`: Fallback when PrimaryBlock is dropping but Arc<Block> not yet returned to backend
///
/// This enables resurrection during the race window when one thread is dropping a PrimaryBlock
/// while another is searching for the same block.
struct WeakBlockEntry<T: BlockMetadata + Sync> {
    /// Weak reference to the underlying Block Arc
    raw_block: Weak<Block<T, Registered>>,
    /// Weak reference to the PrimaryBlock RAII guard
    primary_block: Weak<PrimaryBlock<T>>,
}

struct InactivePoolInner<T: BlockMetadata + Sync> {
    backend: Box<dyn InactivePoolBackend<T>>,
    /// Active blocks tracked via weak references for resurrection.
    /// Key is sequence hash, value contains weak refs to Block and PrimaryBlock.
    weak_blocks: HashMap<SequenceHash, WeakBlockEntry<T>>,
}

impl<T: BlockMetadata + Sync> InactivePool<T> {
    /// Create a new InactivePool with the given backend and reset pool
    pub fn new(backend: Box<dyn InactivePoolBackend<T>>, reset_pool: &ResetPool<T>) -> Self {
        let inner = Arc::new(RwLock::new(InactivePoolInner {
            backend,
            weak_blocks: HashMap::new(),
        }));

        let inner_clone = inner.clone();
        let return_fn = Arc::new(move |block: Arc<Block<T, Registered>>| {
            let seq_hash = block.sequence_hash();
            let strong_count = Arc::strong_count(&block);

            let mut inner = inner_clone.write();
            match Arc::try_unwrap(block) {
                Ok(block) => {
                    // Block is truly inactive now (refcount was 1)
                    // Remove from weak_blocks and add to backend
                    let block_id = block.block_id();
                    inner.weak_blocks.remove(&seq_hash);
                    inner.backend.insert(block);
                    tracing::debug!(?seq_hash, block_id, "Block stored in inactive pool");
                }
                Err(_block) => {
                    // Refcount > 1 - another thread grabbed it via find_or_promote
                    // Block stays active, weak_blocks entry remains valid
                    tracing::trace!(
                        ?seq_hash,
                        strong_count,
                        "Arc::try_unwrap failed - block resurrected by another thread"
                    );
                }
            }
        }) as Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>;

        Self {
            inner,
            reset_return_fn: reset_pool.return_fn(),
            return_fn,
            block_size: reset_pool.block_size(),
        }
    }

    /// Find blocks by sequence hashes and return them as RegisteredBlock guards.
    /// Stops on first miss. Checks both weak_blocks (active) and backend (inactive).
    pub fn find_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        let mut inner = self.inner.write();
        let mut results = Vec::with_capacity(hashes.len());

        for &hash in hashes {
            // First check weak_blocks (active blocks)
            if let Some(primary) = Self::try_weak_lookup(&mut inner, hash, &self.return_fn) {
                results.push(primary as Arc<dyn RegisteredBlock<T>>);
                continue;
            }

            // Not in weak_blocks, try backend (inactive blocks)
            let matched = inner.backend.find_matches(&[hash], touch);
            if let Some(block) = matched.into_iter().next() {
                let arc_block = Arc::new(block);
                let primary =
                    Arc::new(PrimaryBlock::new(arc_block.clone(), self.return_fn.clone()));

                // Add to weak_blocks for future lookups
                inner.weak_blocks.insert(
                    hash,
                    WeakBlockEntry {
                        raw_block: Arc::downgrade(&arc_block),
                        primary_block: Arc::downgrade(&primary),
                    },
                );

                results.push(primary as Arc<dyn RegisteredBlock<T>>);
            } else {
                // Miss - stop searching
                break;
            }
        }

        results
    }

    /// Scan for all blocks matching the given hashes (doesn't stop on miss).
    /// Checks both weak_blocks (active) and backend (inactive).
    /// Returns RAII guards (PrimaryBlocks) for found blocks.
    pub fn scan_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        let mut inner = self.inner.write();
        let mut results = Vec::with_capacity(hashes.len());

        for &hash in hashes {
            // First check weak_blocks (active blocks)
            if let Some(primary) = Self::try_weak_lookup(&mut inner, hash, &self.return_fn) {
                results.push((hash, primary as Arc<dyn RegisteredBlock<T>>));
                continue;
            }

            // Not in weak_blocks, try backend (inactive blocks)
            let found = inner.backend.scan_matches(&[hash], touch);
            if let Some((_, block)) = found.into_iter().next() {
                let arc_block = Arc::new(block);
                let primary =
                    Arc::new(PrimaryBlock::new(arc_block.clone(), self.return_fn.clone()));

                // Add to weak_blocks for future lookups
                inner.weak_blocks.insert(
                    hash,
                    WeakBlockEntry {
                        raw_block: Arc::downgrade(&arc_block),
                        primary_block: Arc::downgrade(&primary),
                    },
                );

                results.push((hash, primary as Arc<dyn RegisteredBlock<T>>));
            }
            // Miss - continue scanning (unlike find_blocks)
        }

        results
    }

    /// Helper to try upgrading weak references from weak_blocks.
    /// Returns Some(Arc<PrimaryBlock>) if successful, None otherwise.
    fn try_weak_lookup(
        inner: &mut InactivePoolInner<T>,
        hash: SequenceHash,
        return_fn: &RegisteredReturnFn<T>,
    ) -> Option<Arc<PrimaryBlock<T>>> {
        let weak_result = inner.weak_blocks.get(&hash).map(|weak_entry| {
            (
                weak_entry.primary_block.upgrade(),
                weak_entry.raw_block.clone(),
            )
        });

        if let Some((maybe_primary, raw_block_weak)) = weak_result {
            // Try PrimaryBlock first
            if let Some(primary) = maybe_primary {
                return Some(primary);
            }

            // Fallback: upgrade raw block and create new PrimaryBlock
            if let Some(raw_arc) = raw_block_weak.upgrade() {
                let primary = Arc::new(PrimaryBlock::new(raw_arc, return_fn.clone()));

                inner.weak_blocks.insert(
                    hash,
                    WeakBlockEntry {
                        raw_block: raw_block_weak,
                        primary_block: Arc::downgrade(&primary),
                    },
                );

                return Some(primary);
            }

            // Both weaks dead - remove stale entry
            inner.weak_blocks.remove(&hash);
        }

        None
    }

    /// Allocate blocks from registered pool, converting them to MutableBlocks for ResetPool
    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        if count == 0 {
            return Some(Vec::new());
        }

        let mut inner = self.inner.write();

        if inner.backend.len() < count {
            return None;
        }

        let allocated_blocks = inner.backend.allocate(count);

        if allocated_blocks.len() == count {
            let mut mutable_blocks = Vec::with_capacity(count);
            mutable_blocks.extend(allocated_blocks.into_iter().map(|registered_block| {
                let reset_block = registered_block.reset();
                MutableBlock::new(reset_block, self.reset_return_fn.clone())
            }));
            Some(mutable_blocks)
        } else {
            for block in allocated_blocks {
                inner.backend.insert(block);
            }
            None
        }
    }

    /// Check if a block exists in the pool
    #[allow(dead_code)]
    pub fn has_block(&self, hash: SequenceHash) -> bool {
        let inner = self.inner.read();
        inner.backend.has_block(hash)
    }

    /// Find and promote a single block from inactive to active by sequence hash.
    /// Returns the concrete `Arc<PrimaryBlock<T>>` for duplicate referencing.
    ///
    /// This differs from `find_blocks()` which returns trait objects. This method
    /// returns the concrete type needed when creating `DuplicateBlock` references.
    ///
    /// **Note**: The caller is responsible for calling `attach_block_ref()` on the
    /// returned PrimaryBlock's registration handle to update the weak reference.
    /// This is not done here to avoid deadlocks when called while holding the
    /// registry attachments lock.
    pub fn find_block_as_primary(
        &self,
        hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<PrimaryBlock<T>>> {
        let mut inner = self.inner.write();
        let matched = inner.backend.find_matches(&[hash], touch);
        matched.into_iter().next().map(|block| {
            let primary = PrimaryBlock::new(Arc::new(block), self.return_fn.clone());
            Arc::new(primary)
        })
    }

    /// Unified lookup that checks both active (weak_blocks) and inactive (backend) blocks.
    ///
    /// This is the primary lookup method that replaces the separate active/inactive pool searches.
    /// Under a single lock, it provides a consistent view with no "in transition" window.
    ///
    /// Lookup order:
    /// 1. Try weak_blocks - upgrade Weak<PrimaryBlock> (cheap if still alive)
    /// 2. Fallback: upgrade Weak<Block> and create new PrimaryBlock (handles race during drop)
    /// 3. Try backend - promote from inactive storage
    ///
    /// Returns `Some(Arc<PrimaryBlock<T>>)` if found, `None` otherwise.
    pub fn find_or_promote(&self, hash: SequenceHash) -> Option<Arc<PrimaryBlock<T>>> {
        let mut inner = self.inner.write();

        // 1. Try weak_blocks first (active blocks)
        // We need to handle the borrow carefully - clone weak refs before mutating
        let weak_result = inner.weak_blocks.get(&hash).map(|weak_entry| {
            (
                weak_entry.primary_block.upgrade(),
                weak_entry.raw_block.clone(),
            )
        });

        if let Some((maybe_primary, raw_block_weak)) = weak_result {
            // Try PrimaryBlock first (cheap - just Arc upgrade)
            if let Some(primary) = maybe_primary {
                tracing::trace!(?hash, "find_or_promote: found via weak PrimaryBlock");
                return Some(primary);
            }

            // Fallback: PrimaryBlock is dropping but Arc<Block> not yet returned
            // This handles the race where another thread is in PrimaryBlock::drop()
            if let Some(raw_arc) = raw_block_weak.upgrade() {
                tracing::trace!(?hash, "find_or_promote: resurrecting via weak Block");
                let primary = Arc::new(PrimaryBlock::new(raw_arc, self.return_fn.clone()));

                // Update weak entry with new PrimaryBlock reference
                inner.weak_blocks.insert(
                    hash,
                    WeakBlockEntry {
                        raw_block: raw_block_weak,
                        primary_block: Arc::downgrade(&primary),
                    },
                );

                return Some(primary);
            }

            // Both weaks are dead - remove stale entry
            tracing::trace!(?hash, "find_or_promote: removing stale weak entry");
            inner.weak_blocks.remove(&hash);
        }

        // 2. Try backend (inactive blocks)
        let matched = inner.backend.find_matches(&[hash], false);
        if let Some(block) = matched.into_iter().next() {
            let arc_block = Arc::new(block);
            let primary = Arc::new(PrimaryBlock::new(arc_block.clone(), self.return_fn.clone()));

            // Add to weak_blocks for future lookups
            inner.weak_blocks.insert(
                hash,
                WeakBlockEntry {
                    raw_block: Arc::downgrade(&arc_block),
                    primary_block: Arc::downgrade(&primary),
                },
            );

            tracing::trace!(?hash, "find_or_promote: promoted from backend");
            return Some(primary);
        }

        None
    }

    /// Unified lookup returning trait object instead of concrete type.
    ///
    /// Convenience wrapper around `find_or_promote` for callers that need `Arc<dyn RegisteredBlock<T>>`.
    pub fn find_or_promote_dyn(&self, hash: SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> {
        self.find_or_promote(hash)
            .map(|primary| primary as Arc<dyn RegisteredBlock<T>>)
    }

    /// Register a newly created block in the active tracking (weak_blocks).
    ///
    /// This is called when a new block is registered (not promoted from inactive pool).
    /// It adds weak references to enable future lookups via `find_or_promote`.
    ///
    /// The block will automatically be moved to the backend when the PrimaryBlock is dropped
    /// (via return_fn), unless another thread resurrects it first.
    pub fn register_active(&self, primary: &Arc<PrimaryBlock<T>>) {
        let hash = primary.sequence_hash();
        let raw_block = Arc::downgrade(
            primary
                .block
                .as_ref()
                .expect("PrimaryBlock should have block"),
        );
        let primary_weak = Arc::downgrade(primary);

        let mut inner = self.inner.write();
        inner.weak_blocks.insert(
            hash,
            WeakBlockEntry {
                raw_block,
                primary_block: primary_weak,
            },
        );

        tracing::trace!(?hash, "register_active: added weak entry for new block");
    }

    /// Get the number of blocks in the pool
    pub fn len(&self) -> usize {
        let inner = self.inner.read();
        inner.backend.len()
    }

    /// Check if the pool is empty
    #[expect(dead_code)]
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.read();
        inner.backend.is_empty()
    }

    pub(crate) fn return_fn(&self) -> RegisteredReturnFn<T> {
        self.return_fn.clone()
    }

    /// Allocate all blocks from the pool, converting them to MutableBlocks.
    /// The MutableBlocks will return to the ResetPool when dropped via RAII.
    pub fn allocate_all_blocks(&self) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.write();
        let blocks = inner.backend.allocate_all();
        blocks
            .into_iter()
            .map(|registered_block| {
                let reset_block = registered_block.reset();
                MutableBlock::new(reset_block, self.reset_return_fn.clone())
            })
            .collect()
    }
}

// // Create pools
// let inactive_pool = InactivePool::new(backend, &reset_pool);

// #[cfg(test)]
// mod tests {
//     use super::backends::FifoReusePolicy;
//     use super::*;
//     use crate::v2::pools::test_utils::{TestData, fixtures::*};

//     impl<T: BlockMetadata> InactivePool<T> {
//         fn insert(&self, block: Block<T, Registered>) {
//             let mut inner = self.inner.write();
//             inner.backend.insert(block);
//         }
//     }

//     fn create_test_pool() -> (InactivePool<TestData>, ResetPool<TestData>) {
//         use super::backends::hashmap_backend::HashMapBackend;

//         let reuse_policy = Box::new(FifoReusePolicy::new());
//         let backend = Box::new(HashMapBackend::new(reuse_policy));

//         let reset_blocks = (0..10).map(|i| Block::new(i, 4)).collect();
//         let reset_pool = ResetPool::new(reset_blocks, 4);

//         let inactive_pool = InactivePool::new(backend, &reset_pool);
//         (inactive_pool, reset_pool)
//     }

//     #[test]
//     fn test_new_pool_starts_empty() {
//         let (pool, _reset_pool) = create_test_pool();
//         assert_eq!(pool.len(), 0);
//         assert!(pool.is_empty());
//         assert!(!pool.has_block(100));
//     }

//     #[test]
//     fn test_return_and_find_single_block() {
//         let (pool, _reset_pool) = create_test_pool();
//         let (block, seq_hash) = create_registered_block(1, &tokens_for_id(1));

//         // Return block directly (simulating manual return)
//         pool.insert(block);

//         assert_eq!(pool.len(), 1);
//         assert!(pool.has_block(seq_hash));

//         // Find the block
//         let found_blocks = pool.find_blocks(&[seq_hash], true);
//         assert_eq!(found_blocks.len(), 1);
//         assert_eq!(found_blocks[0].block_id(), 1);
//         assert_eq!(found_blocks[0].sequence_hash(), seq_hash);

//         // Block should be removed from pool after finding
//         assert_eq!(pool.len(), 0);
//         assert!(!pool.has_block(seq_hash));

//         // Blocks will auto-return when dropped at end of scope
//     }

//     #[test]
//     fn test_find_blocks_stops_on_first_miss() {
//         let (pool, _reset_pool) = create_test_pool();

//         // Add blocks with different sequence hashes
//         let (block1, seq_hash1) = create_registered_block(1, &tokens_for_id(1));
//         let (block3, seq_hash3) = create_registered_block(3, &tokens_for_id(3));
//         pool.insert(block1);
//         pool.insert(block3);

//         assert_eq!(pool.len(), 2);

//         // Try to find blocks - use a sequence hash that doesn't exist to test first miss behavior
//         let nonexistent_hash = 99999;
//         let found_blocks = pool.find_blocks(&[seq_hash1, nonexistent_hash, seq_hash3], true);
//         assert_eq!(found_blocks.len(), 1); // Only found first block
//         assert_eq!(found_blocks[0].sequence_hash(), seq_hash1);

//         // Block 3 should still be in pool since search stopped at first miss
//         assert_eq!(pool.len(), 1);
//         assert!(pool.has_block(seq_hash3));
//     }

//     #[test]
//     fn test_raii_auto_return() {
//         let (pool, _reset_pool) = create_test_pool();
//         let (block, seq_hash) = create_registered_block(1, &tokens_for_id(1));
//         pool.insert(block);

//         assert_eq!(pool.len(), 1);

//         {
//             let _found_blocks = pool.find_blocks(&[seq_hash], true);
//             assert_eq!(pool.len(), 0);
//         }

//         assert_eq!(pool.len(), 1);
//         assert!(pool.has_block(seq_hash));
//     }

//     #[test]
//     fn test_allocate_blocks() {
//         let (pool, reset_pool) = create_test_pool();

//         // Add some registered blocks to the pool
//         let (block1, _seq_hash1) = create_registered_block(1, &tokens_for_id(1));
//         let (block2, _seq_hash2) = create_registered_block(2, &tokens_for_id(2));
//         let (block3, _seq_hash3) = create_registered_block(3, &tokens_for_id(3));
//         pool.insert(block1);
//         pool.insert(block2);
//         pool.insert(block3);

//         assert_eq!(pool.len(), 3);

//         // Allocate 1 block - should convert to MutableBlocks
//         // Note: Due to test setup limitations with reuse policy, we can only allocate 1 block
//         let mutable_blocks = pool.allocate_blocks(1).expect("Should allocate 1 block");
//         assert_eq!(mutable_blocks.len(), 1);

//         // Pool should have one less block
//         assert_eq!(pool.len(), 2);

//         // The MutableBlocks should have the correct IDs
//         let block_ids: Vec<u64> = mutable_blocks.iter().map(|b| b.block_id()).collect();
//         assert!(block_ids.contains(&1) || block_ids.contains(&2) || block_ids.contains(&3));

//         drop(mutable_blocks);

//         assert_eq!(pool.len(), 2);
//         assert_eq!(reset_pool.available_blocks(), 11);
//     }

//     #[test]
//     fn test_allocate_more_than_available_fails() {
//         let (pool, _reset_pool) = create_test_pool();

//         // Add only 2 blocks
//         let (block1, _seq_hash1) = create_registered_block(1, &tokens_for_id(1));
//         let (block2, _seq_hash2) = create_registered_block(2, &tokens_for_id(2));
//         pool.insert(block1);
//         pool.insert(block2);

//         assert_eq!(pool.len(), 2);

//         // Try to allocate 3 blocks - should fail
//         let result = pool.allocate_blocks(3);
//         assert!(result.is_none());

//         // Pool should be unchanged
//         assert_eq!(pool.len(), 2);
//     }
// }
