// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe pool for registered immutable blocks with automatic RAII return.
//!
//! Manages blocks in the Registered state, providing:
//! - Finding blocks by sequence hash with O(1) lookup
//! - Conversion of registered blocks back to mutable blocks for reuse
//! - Thread-safe access via interior mutability
//! - Automatic block return via RAII ImmutableBlock guards

pub mod backends;

use parking_lot::RwLock;
use std::sync::Arc;

use crate::metrics::BlockPoolMetrics;

use super::{
    Block, BlockId, BlockMetadata, InactiveBlock, MutableBlock, PrimaryBlock, Registered,
    RegisteredBlock, SequenceHash, reset::ResetPool,
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

    #[allow(dead_code)]
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
use crate::blocks::{RegisteredReturnFn, ResetReturnFn};

/// Pool for managing registered (immutable) blocks
///
/// This pool handles blocks in the Registered state and provides them as
/// RegisteredBlock RAII guards that automatically return to the pool on drop.

#[derive(Clone)]
pub(crate) struct InactivePool<T: BlockMetadata> {
    // Inner state protected by RwLock for thread-safe access from guards
    inner: Arc<RwLock<InactivePoolInner<T>>>,
    // Return function for MutableBlocks to return to ResetPool
    reset_return_fn: ResetReturnFn<T>,

    return_fn: RegisteredReturnFn<T>,
    #[expect(dead_code)]
    block_size: usize,
    metrics: Option<Arc<BlockPoolMetrics>>,
}

struct InactivePoolInner<T: BlockMetadata> {
    backend: Box<dyn InactivePoolBackend<T>>,
}

impl<T: BlockMetadata + Sync> InactivePool<T> {
    /// Create a new InactivePool with the given backend and reset pool
    pub(crate) fn new(
        backend: Box<dyn InactivePoolBackend<T>>,
        reset_pool: &ResetPool<T>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Self {
        let inner = Arc::new(RwLock::new(InactivePoolInner { backend }));

        let inner_clone = inner.clone();
        let metrics_clone = metrics.clone();
        let return_fn = Arc::new(move |block: Arc<Block<T, Registered>>| {
            let seq_hash = block.sequence_hash();

            let mut inner = inner_clone.write();
            match Arc::try_unwrap(block) {
                Ok(block) => {
                    let block_id = block.block_id();
                    inner.backend.insert(block);
                    if let Some(ref m) = metrics_clone {
                        m.inc_inactive_pool_size();
                    }
                    tracing::trace!(?seq_hash, block_id, "Block stored in inactive pool");
                }
                Err(block) => {
                    let block_id = block.block_id();
                    let weak = Arc::downgrade(&block);
                    drop(block);
                    if weak.strong_count() == 0 {
                        tracing::warn!(?seq_hash, block_id, "Possible KV Block leak detected");
                    }
                }
            }
        }) as Arc<dyn Fn(Arc<Block<T, Registered>>) + Send + Sync>;

        Self {
            inner,
            reset_return_fn: reset_pool.return_fn(),
            return_fn,
            block_size: reset_pool.block_size(),
            metrics,
        }
    }

    /// Find blocks by sequence hashes and return them as RegisteredBlock guards.
    /// Stops on first miss.
    pub(crate) fn find_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        let mut inner = self.inner.write();
        let matched_blocks = inner.backend.find_matches(hashes, touch);

        let count = matched_blocks.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }

        matched_blocks
            .into_iter()
            .map(|block| {
                PrimaryBlock::new_attached(Arc::new(block), self.return_fn.clone())
                    as Arc<dyn RegisteredBlock<T>>
            })
            .collect()
    }

    /// Scan for all blocks matching the given hashes (doesn't stop on miss).
    /// Acquires/removes found blocks from pool - caller owns until dropped.
    /// Returns RAII guards (PrimaryBlocks) for found blocks.
    pub(crate) fn scan_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        let mut inner = self.inner.write();
        let found = inner.backend.scan_matches(hashes, touch);

        let count = found.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }

        found
            .into_iter()
            .map(|(hash, block)| {
                let registered = PrimaryBlock::new_attached(Arc::new(block), self.return_fn.clone())
                    as Arc<dyn RegisteredBlock<T>>;
                (hash, registered)
            })
            .collect()
    }

    /// Allocate blocks from registered pool, converting them to MutableBlocks for ResetPool
    pub(crate) fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        if count == 0 {
            return Some(Vec::new());
        }

        let mut inner = self.inner.write();

        if inner.backend.len() < count {
            return None;
        }

        let allocated_blocks = inner.backend.allocate(count);

        if allocated_blocks.len() == count {
            if let Some(ref m) = self.metrics {
                for _ in 0..count {
                    m.dec_inactive_pool_size();
                }
            }
            let mut mutable_blocks = Vec::with_capacity(count);
            mutable_blocks.extend(allocated_blocks.into_iter().map(|registered_block| {
                let reset_block = registered_block.reset();
                MutableBlock::new(
                    reset_block,
                    self.reset_return_fn.clone(),
                    self.metrics.clone(),
                )
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
    pub(crate) fn has_block(&self, hash: SequenceHash) -> bool {
        let inner = self.inner.read();
        inner.backend.has_block(hash)
    }

    /// Find and promote a single block from inactive to active by sequence hash.
    /// Returns the concrete `Arc<PrimaryBlock<T>>` for duplicate referencing.
    ///
    /// This differs from `find_blocks()` which returns trait objects. This method
    /// returns the concrete type needed when creating `DuplicateBlock` references.
    ///
    /// Uses `new_unattached` because this is called from `try_find_existing_block`
    /// while the attachments lock is held. The caller MUST call
    /// `PrimaryBlock::store_weak_refs()` after dropping the attachments lock.
    pub(crate) fn find_block_as_primary(
        &self,
        hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<PrimaryBlock<T>>> {
        let mut inner = self.inner.write();
        let matched = inner.backend.find_matches(&[hash], touch);
        matched.into_iter().next().map(|block| {
            if let Some(ref m) = self.metrics {
                m.dec_inactive_pool_size();
            }
            PrimaryBlock::new_unattached(Arc::new(block), self.return_fn.clone())
        })
    }

    /// Get the number of blocks in the pool
    pub(crate) fn len(&self) -> usize {
        let inner = self.inner.read();
        inner.backend.len()
    }

    /// Check if the pool is empty
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        let inner = self.inner.read();
        inner.backend.is_empty()
    }

    pub(crate) fn return_fn(&self) -> RegisteredReturnFn<T> {
        self.return_fn.clone()
    }

    /// Allocate all blocks from the pool, converting them to MutableBlocks.
    /// The MutableBlocks will return to the ResetPool when dropped via RAII.
    pub(crate) fn allocate_all_blocks(&self) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.write();
        let blocks = inner.backend.allocate_all();
        let count = blocks.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        blocks
            .into_iter()
            .map(|registered_block| {
                let reset_block = registered_block.reset();
                MutableBlock::new(
                    reset_block,
                    self.reset_return_fn.clone(),
                    self.metrics.clone(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::backends::FifoReusePolicy;
    use super::*;
    use crate::testing::{TestMeta, create_registered_block, tokens_for_id};

    impl<T: BlockMetadata> InactivePool<T> {
        fn insert(&self, block: Block<T, Registered>) {
            let mut inner = self.inner.write();
            inner.backend.insert(block);
        }
    }

    fn create_test_pool() -> (InactivePool<TestMeta>, ResetPool<TestMeta>) {
        use super::backends::HashMapBackend;

        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));

        let reset_blocks: Vec<_> = (0..10_usize).map(|i| Block::new(i, 4)).collect();
        let reset_pool = ResetPool::new(reset_blocks, 4, None);

        let inactive_pool = InactivePool::new(backend, &reset_pool, None);
        (inactive_pool, reset_pool)
    }

    /// Create a sequence hash for a block that doesn't exist in any pool.
    fn nonexistent_hash() -> SequenceHash {
        // Create a registered block just to get its sequence hash, then drop it
        let (_, seq_hash) = create_registered_block::<TestMeta>(999, &[9999, 9998, 9997, 9996]);
        seq_hash
    }

    #[test]
    fn test_new_pool_starts_empty() {
        let (pool, _reset_pool) = create_test_pool();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert!(!pool.has_block(nonexistent_hash()));
    }

    #[test]
    fn test_return_and_find_single_block() {
        let (pool, _reset_pool) = create_test_pool();
        let (block, seq_hash) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));

        pool.insert(block);

        assert_eq!(pool.len(), 1);
        assert!(pool.has_block(seq_hash));

        let found_blocks = pool.find_blocks(&[seq_hash], true);
        assert_eq!(found_blocks.len(), 1);
        assert_eq!(found_blocks[0].block_id(), 1);
        assert_eq!(found_blocks[0].sequence_hash(), seq_hash);

        // Block should be removed from pool after finding
        assert_eq!(pool.len(), 0);
        assert!(!pool.has_block(seq_hash));
    }

    #[test]
    fn test_find_blocks_stops_on_first_miss() {
        let (pool, _reset_pool) = create_test_pool();

        let (block1, seq_hash1) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block3, seq_hash3) = create_registered_block::<TestMeta>(3, &tokens_for_id(3));
        pool.insert(block1);
        pool.insert(block3);

        assert_eq!(pool.len(), 2);

        let missing = nonexistent_hash();
        let found_blocks = pool.find_blocks(&[seq_hash1, missing, seq_hash3], true);
        assert_eq!(found_blocks.len(), 1);
        assert_eq!(found_blocks[0].sequence_hash(), seq_hash1);

        // Block 3 should still be in pool since search stopped at first miss
        assert_eq!(pool.len(), 1);
        assert!(pool.has_block(seq_hash3));
    }

    #[test]
    fn test_raii_auto_return() {
        let (pool, _reset_pool) = create_test_pool();
        let (block, seq_hash) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        pool.insert(block);

        assert_eq!(pool.len(), 1);

        {
            let _found_blocks = pool.find_blocks(&[seq_hash], true);
            assert_eq!(pool.len(), 0);
        }

        assert_eq!(pool.len(), 1);
        assert!(pool.has_block(seq_hash));
    }

    #[test]
    fn test_allocate_blocks() {
        let (pool, reset_pool) = create_test_pool();

        let (block1, _) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block2, _) = create_registered_block::<TestMeta>(2, &tokens_for_id(2));
        let (block3, _) = create_registered_block::<TestMeta>(3, &tokens_for_id(3));
        pool.insert(block1);
        pool.insert(block2);
        pool.insert(block3);

        assert_eq!(pool.len(), 3);

        let mutable_blocks = pool.allocate_blocks(1).expect("Should allocate 1 block");
        assert_eq!(mutable_blocks.len(), 1);
        assert_eq!(pool.len(), 2);

        drop(mutable_blocks);

        assert_eq!(pool.len(), 2);
        assert_eq!(reset_pool.available_blocks(), 11);
    }

    #[test]
    fn test_allocate_more_than_available_fails() {
        let (pool, _reset_pool) = create_test_pool();

        let (block1, _) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block2, _) = create_registered_block::<TestMeta>(2, &tokens_for_id(2));
        pool.insert(block1);
        pool.insert(block2);

        assert_eq!(pool.len(), 2);

        let result = pool.allocate_blocks(3);
        assert!(result.is_none());

        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_scan_blocks() {
        let (pool, _reset_pool) = create_test_pool();

        let (block1, seq_hash1) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block3, seq_hash3) = create_registered_block::<TestMeta>(3, &tokens_for_id(3));
        pool.insert(block1);
        // Sleep for FIFO timestamp uniqueness (HashMap backend)
        std::thread::sleep(std::time::Duration::from_millis(2));
        pool.insert(block3);

        assert_eq!(pool.len(), 2);

        let missing = nonexistent_hash();

        // scan_blocks should NOT stop on miss — should find both hash1 and hash3
        let found = pool.scan_blocks(&[seq_hash1, missing, seq_hash3], true);
        assert_eq!(
            found.len(),
            2,
            "scan_blocks should find both blocks, skipping the miss"
        );

        let found_hashes: Vec<_> = found.iter().map(|(h, _)| *h).collect();
        assert!(found_hashes.contains(&seq_hash1));
        assert!(found_hashes.contains(&seq_hash3));

        // Both blocks were removed from the pool
        assert_eq!(pool.len(), 0);

        // RAII return: dropping the found blocks should return them
        drop(found);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn test_allocate_all_blocks() {
        let (pool, reset_pool) = create_test_pool();

        let (block1, _) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block2, _) = create_registered_block::<TestMeta>(2, &tokens_for_id(2));
        let (block3, _) = create_registered_block::<TestMeta>(3, &tokens_for_id(3));
        pool.insert(block1);
        pool.insert(block2);
        pool.insert(block3);

        assert_eq!(pool.len(), 3);

        let mutable_blocks = pool.allocate_all_blocks();
        assert_eq!(mutable_blocks.len(), 3);
        assert_eq!(pool.len(), 0);

        // Verify they are MutableBlocks by checking block_id
        for block in &mutable_blocks {
            let _id = block.block_id();
        }

        // Drop them — they should return to the reset pool
        drop(mutable_blocks);
        // 10 original reset blocks + 3 returned = 13
        assert_eq!(reset_pool.available_blocks(), 13);
    }
}
