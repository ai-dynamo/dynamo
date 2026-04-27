// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe pool for registered immutable blocks with automatic RAII return.
//!
//! Manages blocks in the Registered state, providing:
//! - Finding blocks by sequence hash with O(1) lookup
//! - Conversion of registered blocks back to mutable blocks for reuse
//! - Thread-safe access via interior mutability
//! - Automatic block return via RAII ImmutableBlock guards
//!
//! Internally, eviction order lives in a [`InactiveIndex`](crate::pools::store::InactiveIndex)
//! backend (T-free, indexed by `BlockId`+`SequenceHash`); the actual
//! `Block<T, Registered>` payloads live alongside in a `HashMap<BlockId, _>`
//! side-table. This split is the transitional layer for the unified
//! `BlockStore<T>` refactor — once guards talk to the store directly the
//! side-table goes away.

pub mod backends;

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::metrics::BlockPoolMetrics;
use crate::pools::store::InactiveIndex;

use super::{
    Block, BlockId, BlockMetadata, InactiveBlock, MutableBlock, PrimaryBlock, Registered,
    RegisteredBlock, SequenceHash, reset::ResetPool,
};

use crate::blocks::{RegisteredReturnFn, ResetReturnFn};

/// Pool for managing registered (immutable) blocks
///
/// This pool handles blocks in the Registered state and provides them as
/// RegisteredBlock RAII guards that automatically return to the pool on drop.
#[derive(Clone)]
pub(crate) struct InactivePool<T: BlockMetadata> {
    inner: Arc<RwLock<InactivePoolInner<T>>>,
    reset_return_fn: ResetReturnFn<T>,
    return_fn: RegisteredReturnFn<T>,
    #[expect(dead_code)]
    block_size: usize,
    metrics: Option<Arc<BlockPoolMetrics>>,
}

struct InactivePoolInner<T: BlockMetadata> {
    /// Eviction-order tracker (T-free).
    index: Box<dyn InactiveIndex>,
    /// Side-table of the actual `Block<T, Registered>` payloads, keyed by
    /// `BlockId`. Always kept in sync with `index`.
    blocks: HashMap<BlockId, Block<T, Registered>>,
}

impl<T: BlockMetadata + Sync> InactivePool<T> {
    pub(crate) fn new(
        index: Box<dyn InactiveIndex>,
        reset_pool: &ResetPool<T>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Self {
        let inner = Arc::new(RwLock::new(InactivePoolInner {
            index,
            blocks: HashMap::new(),
        }));

        let inner_clone = inner.clone();
        let metrics_clone = metrics.clone();
        let return_fn = Arc::new(move |block: Arc<Block<T, Registered>>| {
            let seq_hash = block.sequence_hash();
            let mut inner = inner_clone.write();
            match Arc::try_unwrap(block) {
                Ok(block) => {
                    let block_id = block.block_id();
                    inner.index.insert(seq_hash, block_id);
                    inner.blocks.insert(block_id, block);
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
        let matched = inner.index.find_matches(hashes, touch);

        let count = matched.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }

        matched
            .into_iter()
            .map(|(_seq_hash, block_id)| {
                let block = inner
                    .blocks
                    .remove(&block_id)
                    .expect("inactive index/side-table out of sync");
                PrimaryBlock::new_attached(Arc::new(block), self.return_fn.clone())
                    as Arc<dyn RegisteredBlock<T>>
            })
            .collect()
    }

    /// Scan for all blocks matching the given hashes (doesn't stop on miss).
    /// Acquires/removes found blocks from pool — caller owns until dropped.
    pub(crate) fn scan_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        let mut inner = self.inner.write();
        let found = inner.index.scan_matches(hashes, touch);

        let count = found.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }

        found
            .into_iter()
            .map(|(seq_hash, block_id)| {
                let block = inner
                    .blocks
                    .remove(&block_id)
                    .expect("inactive index/side-table out of sync");
                let registered = PrimaryBlock::new_attached(Arc::new(block), self.return_fn.clone())
                    as Arc<dyn RegisteredBlock<T>>;
                (seq_hash, registered)
            })
            .collect()
    }

    /// Allocate blocks from the registered pool, converting them to
    /// [`MutableBlock`]s for the [`ResetPool`]. Also reports the
    /// [`SequenceHash`] of each evicted block.
    pub(crate) fn allocate_blocks(
        &self,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        if count == 0 {
            return Some((Vec::new(), Vec::new()));
        }

        let mut inner = self.inner.write();

        if inner.index.len() < count {
            return None;
        }

        let evicted_pairs = inner.index.allocate(count);

        if evicted_pairs.len() != count {
            for (h, id) in evicted_pairs {
                inner.index.insert(h, id);
            }
            return None;
        }

        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        let mut mutable_blocks = Vec::with_capacity(count);
        let mut evicted = Vec::with_capacity(count);
        for (seq_hash, block_id) in evicted_pairs {
            let block = inner
                .blocks
                .remove(&block_id)
                .expect("inactive index/side-table out of sync");
            evicted.push(seq_hash);
            let reset_block = block.reset();
            mutable_blocks.push(MutableBlock::new(
                reset_block,
                self.reset_return_fn.clone(),
                self.metrics.clone(),
            ));
        }
        Some((mutable_blocks, evicted))
    }

    /// Check if a block exists in the pool.
    #[allow(dead_code)]
    pub(crate) fn has_block(&self, hash: SequenceHash) -> bool {
        let inner = self.inner.read();
        inner.index.has(hash)
    }

    /// Find and promote a single block from inactive to active by sequence hash.
    /// Returns the concrete `Arc<PrimaryBlock<T>>` for duplicate referencing.
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
        let mut matched = inner.index.find_matches(&[hash], touch);
        let (_, block_id) = matched.pop()?;
        let block = inner
            .blocks
            .remove(&block_id)
            .expect("inactive index/side-table out of sync");
        if let Some(ref m) = self.metrics {
            m.dec_inactive_pool_size();
        }
        Some(PrimaryBlock::new_unattached(
            Arc::new(block),
            self.return_fn.clone(),
        ))
    }

    pub(crate) fn len(&self) -> usize {
        let inner = self.inner.read();
        inner.index.len()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        let inner = self.inner.read();
        inner.index.is_empty()
    }

    pub(crate) fn return_fn(&self) -> RegisteredReturnFn<T> {
        self.return_fn.clone()
    }

    /// Allocate all blocks from the pool, converting them to MutableBlocks.
    /// The MutableBlocks will return to the ResetPool when dropped via RAII.
    pub(crate) fn allocate_all_blocks(&self) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.write();
        let drained = inner.index.allocate_all();
        let count = drained.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        drained
            .into_iter()
            .map(|(_seq_hash, block_id)| {
                let block = inner
                    .blocks
                    .remove(&block_id)
                    .expect("inactive index/side-table out of sync");
                let reset_block = block.reset();
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
            let seq_hash = block.sequence_hash();
            let block_id = block.block_id();
            inner.index.insert(seq_hash, block_id);
            inner.blocks.insert(block_id, block);
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

    fn nonexistent_hash() -> SequenceHash {
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

        let (block1, seq_hash1) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block2, seq_hash2) = create_registered_block::<TestMeta>(2, &tokens_for_id(2));
        let (block3, seq_hash3) = create_registered_block::<TestMeta>(3, &tokens_for_id(3));
        pool.insert(block1);
        pool.insert(block2);
        pool.insert(block3);

        assert_eq!(pool.len(), 3);

        let (mutable_blocks, evicted) = pool.allocate_blocks(1).expect("Should allocate 1 block");
        assert_eq!(mutable_blocks.len(), 1);
        assert_eq!(evicted.len(), 1);
        assert!(
            [seq_hash1, seq_hash2, seq_hash3].contains(&evicted[0]),
            "evicted hash must match one of the inserted blocks; got {:?}",
            evicted[0]
        );
        assert_eq!(pool.len(), 2);

        drop(mutable_blocks);

        assert_eq!(pool.len(), 2);
        assert_eq!(reset_pool.available_blocks(), 11);
    }

    #[test]
    fn test_allocate_blocks_reports_all_evicted_hashes() {
        let (pool, _reset_pool) = create_test_pool();

        let (block1, seq_hash1) = create_registered_block::<TestMeta>(1, &tokens_for_id(1));
        let (block2, seq_hash2) = create_registered_block::<TestMeta>(2, &tokens_for_id(2));
        let (block3, seq_hash3) = create_registered_block::<TestMeta>(3, &tokens_for_id(3));
        pool.insert(block1);
        pool.insert(block2);
        pool.insert(block3);
        let inserted = [seq_hash1, seq_hash2, seq_hash3];

        let (mutable_blocks, evicted) = pool
            .allocate_blocks(3)
            .expect("Should allocate all three blocks");
        assert_eq!(mutable_blocks.len(), 3);
        assert_eq!(evicted.len(), 3);
        for h in &evicted {
            assert!(inserted.contains(h), "evicted hash {h:?} not in inserted set");
        }
        let unique: std::collections::HashSet<_> = evicted.iter().copied().collect();
        assert_eq!(unique.len(), 3, "evicted hashes must all be distinct");
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
        std::thread::sleep(std::time::Duration::from_millis(2));
        pool.insert(block3);

        assert_eq!(pool.len(), 2);

        let missing = nonexistent_hash();

        let found = pool.scan_blocks(&[seq_hash1, missing, seq_hash3], true);
        assert_eq!(
            found.len(),
            2,
            "scan_blocks should find both blocks, skipping the miss"
        );

        let found_hashes: Vec<_> = found.iter().map(|(h, _)| *h).collect();
        assert!(found_hashes.contains(&seq_hash1));
        assert!(found_hashes.contains(&seq_hash3));

        assert_eq!(pool.len(), 0);

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

        for block in &mutable_blocks {
            let _id = block.block_id();
        }

        drop(mutable_blocks);
        assert_eq!(reset_pool.available_blocks(), 13);
    }
}
