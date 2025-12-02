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
use std::sync::Arc;

use super::{
    Block, BlockId, BlockMetadata, InactiveBlock, MutableBlock, PrimaryBlock, Registered,
    RegisteredBlock, Reset, SequenceHash, reset::ResetPool,
};

// pub(crate) use backends::*;

/// Backend trait for InactivePool storage strategies.
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

    fn has_block(&self, seq_hash: SequenceHash) -> bool;
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

struct InactivePoolInner<T: BlockMetadata> {
    backend: Box<dyn InactivePoolBackend<T>>,
}

impl<T: BlockMetadata + Sync> InactivePool<T> {
    /// Create a new InactivePool with the given backend and reset pool
    pub fn new(backend: Box<dyn InactivePoolBackend<T>>, reset_pool: &ResetPool<T>) -> Self {
        let inner = Arc::new(RwLock::new(InactivePoolInner { backend }));

        let inner_clone = inner.clone();
        let return_fn = Arc::new(move |block: Arc<Block<T, Registered>>| {
            let mut inner = inner_clone.write();

            if let Ok(block) = Arc::try_unwrap(block) {
                inner.backend.insert(block);
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
    /// Stops on first miss.
    pub fn find_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        let mut inner = self.inner.write();
        let matched_blocks = inner.backend.find_matches(hashes, touch);

        matched_blocks
            .into_iter()
            .map(|block| PrimaryBlock::new(Arc::new(block), self.return_fn.clone()).register())
            .collect()
    }

    /// Scan for all blocks matching the given hashes (doesn't stop on miss).
    /// Acquires/removes found blocks from pool - caller owns until dropped.
    /// Returns RAII guards (PrimaryBlocks) for found blocks.
    pub fn scan_blocks(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        let mut inner = self.inner.write();
        let found = inner.backend.scan_matches(hashes, touch);

        found
            .into_iter()
            .map(|(hash, block)| {
                // Same pattern as find_blocks: PrimaryBlock::new(...).register()
                let registered =
                    PrimaryBlock::new(Arc::new(block), self.return_fn.clone()).register();
                (hash, registered)
            })
            .collect()
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
    #[expect(dead_code)]
    pub fn has_block(&self, hash: SequenceHash) -> bool {
        let inner = self.inner.read();
        inner.backend.has_block(hash)
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
