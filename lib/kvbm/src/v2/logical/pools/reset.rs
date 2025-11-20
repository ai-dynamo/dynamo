// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thread-safe pool for mutable blocks in reset state with pluggable allocation strategies.
//!
//! The ResetPool manages blocks available for allocation, using:
//! - Pluggable BlockAllocator for flexible allocation strategies
//! - RAII MutableBlock guards for automatic return
//! - Thread-safe access via parking_lot::Mutex

use crate::BlockId;

use super::{Block, BlockAllocator, BlockMetadata, MutableBlock, Reset};
use parking_lot::Mutex;
use std::{collections::VecDeque, sync::Arc};

pub struct ResetPool<T> {
    block_allocator: Arc<Mutex<dyn BlockAllocator<T> + Send + Sync>>,
    return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    block_size: usize,
}

impl<T: BlockMetadata> ResetPool<T> {
    pub fn new(blocks: Vec<Block<T, Reset>>, block_size: usize) -> Self {
        let allocator = DequeBlockAllocator::new();
        Self::from_block_allocator(allocator, blocks, block_size)
    }

    pub fn from_block_allocator(
        mut allocator: impl BlockAllocator<T> + Send + Sync + 'static,
        blocks: Vec<Block<T, Reset>>,
        block_size: usize,
    ) -> Self {
        for (i, block) in blocks.iter().enumerate() {
            if block.block_id() != i as BlockId {
                panic!("Block ids must be monotonically increasing starting at 0");
            }
        }

        for block in blocks {
            allocator.insert(block);
        }

        let block_allocator = Arc::new(Mutex::new(allocator));

        let allocator_clone = block_allocator.clone();
        let return_fn = Arc::new(move |block: Block<T, Reset>| {
            allocator_clone.lock().insert(block);
        });

        Self {
            block_allocator,
            return_fn,
            block_size,
        }
    }

    /// Tries to allocate upto `count` blocks from the pool.
    /// Will return less than `count` blocks if the pool has less than `count` blocks available.
    pub fn allocate_blocks(&self, count: usize) -> Vec<MutableBlock<T>> {
        let mut blocks = Vec::with_capacity(count);
        let mut allocator = self.block_allocator.lock();
        let available_count = std::cmp::min(count, allocator.len());

        for _ in 0..available_count {
            blocks.push(MutableBlock::new(
                allocator.pop().unwrap(),
                self.return_fn.clone(),
            ));
        }

        blocks
    }

    /// Get the number of available blocks
    #[expect(dead_code)]
    pub fn available_blocks(&self) -> usize {
        self.block_allocator.lock().len()
    }

    pub fn len(&self) -> usize {
        self.block_allocator.lock().len()
    }

    /// Check if the pool is empty
    #[expect(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.block_allocator.lock().is_empty()
    }

    /// Create a return function for blocks to return to this pool
    /// This allows other pools to create MutableBlocks that return here
    pub(crate) fn return_fn(&self) -> Arc<dyn Fn(Block<T, Reset>) + Send + Sync> {
        self.return_fn.clone()
    }

    /// Get the expected block size for this pool
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }
}

#[derive(Debug)]
pub struct DequeBlockAllocator<T: BlockMetadata> {
    blocks: VecDeque<Block<T, Reset>>,
}

impl<T: BlockMetadata> Default for DequeBlockAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BlockMetadata> DequeBlockAllocator<T> {
    pub fn new() -> Self {
        Self {
            blocks: VecDeque::new(),
        }
    }
}

impl<T: BlockMetadata> BlockAllocator<T> for DequeBlockAllocator<T> {
    fn insert(&mut self, block: Block<T, Reset>) {
        self.blocks.push_back(block);
    }

    fn pop(&mut self) -> Option<Block<T, Reset>> {
        self.blocks.pop_front()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::super::test_utils::TestData;
//     use super::*;

//     fn create_test_blocks(count: usize) -> Vec<Block<TestData, Reset>> {
//         (0..count as u64).map(|id| Block::new(id, 4)).collect()
//     }

//     #[test]
//     fn test_mutable_block_raii_return() {
//         let blocks = create_test_blocks(3);
//         let pool = ResetPool::new(blocks, 4);

//         assert_eq!(pool.len(), 3);

//         {
//             let allocated = pool.allocate_blocks(2).unwrap();
//             assert_eq!(allocated.len(), 2);
//             assert_eq!(pool.len(), 1);
//         }

//         assert_eq!(pool.len(), 3);
//     }

//     #[test]
//     fn test_pool_allocation_and_return_cycle() {
//         let blocks = create_test_blocks(5);
//         let pool = ResetPool::new(blocks, 4);

//         for _ in 0..3 {
//             assert_eq!(pool.len(), 5);

//             {
//                 let allocated = pool.allocate_blocks(2).unwrap();
//                 assert_eq!(allocated.len(), 2);
//                 assert_eq!(pool.len(), 3);
//             }

//             assert_eq!(pool.len(), 5);
//         }
//     }

//     #[test]
//     fn test_try_allocate_blocks_partial() {
//         let blocks = create_test_blocks(3);
//         let pool = ResetPool::new(blocks, 4);

//         let allocated = pool.try_allocate_blocks(5);
//         assert_eq!(allocated.len(), 3);
//         assert_eq!(pool.len(), 0);
//     }

//     #[test]
//     fn test_allocate_blocks_insufficient() {
//         let blocks = create_test_blocks(2);
//         let pool = ResetPool::new(blocks, 4);

//         let result = pool.allocate_blocks(3);
//         assert!(result.is_none());
//         assert_eq!(pool.len(), 2);
//     }
// }
