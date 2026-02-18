// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! FIFO reuse policy for inactive registered blocks.
//!
//! Allocates blocks in first-in-first-out order using a monotonic counter.
//! Uses BTreeMap for O(log n) insertion/removal with priority key ordering.

#![allow(dead_code)]

use super::reuse_policy::*;

use std::collections::{BTreeMap, HashMap};

use super::{BlockId, InactiveBlock};

/// Monotonic sequence number used as priority key for FIFO ordering.
pub type PriorityKey = u64;

/// FIFO reuse policy
#[derive(Debug)]
pub struct FifoReusePolicy {
    keys: HashMap<BlockId, PriorityKey>,
    blocks: BTreeMap<PriorityKey, InactiveBlock>,
    next_seq: u64,
}

impl Default for FifoReusePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl FifoReusePolicy {
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            blocks: BTreeMap::new(),
            next_seq: 0,
        }
    }
}

impl ReusePolicy for FifoReusePolicy {
    fn insert(&mut self, inactive_block: InactiveBlock) -> Result<(), ReusePolicyError> {
        assert!(
            !self.keys.contains_key(&inactive_block.block_id),
            "block already exists"
        );
        let priority_key = self.next_seq;
        self.next_seq += 1;
        self.keys.insert(inactive_block.block_id, priority_key);
        self.blocks.insert(priority_key, inactive_block);
        Ok(())
    }

    fn remove(&mut self, block_id: BlockId) -> Result<(), ReusePolicyError> {
        let priority_key = self
            .keys
            .remove(&block_id)
            .ok_or(ReusePolicyError::BlockNotFound(block_id))?;

        assert!(
            self.blocks.remove(&priority_key).is_some(),
            "block not found"
        );
        Ok(())
    }

    fn next_free(&mut self) -> Option<InactiveBlock> {
        let next_block = self.blocks.pop_first();
        if let Some((_, block)) = next_block {
            assert!(
                self.keys.remove(&block.block_id).is_some(),
                "block not found"
            );
            Some(block)
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestMeta, create_staged_block};

    // Use TestMeta instead of local TestData
    type TestData = TestMeta;

    // Wrapper for backward compatibility
    fn create_completed_block<T: crate::blocks::BlockMetadata + std::fmt::Debug>(
        tokens: &[u32],
        block_id: BlockId,
    ) -> crate::blocks::Block<T, crate::blocks::state::Staged> {
        create_staged_block::<T>(block_id, tokens)
    }

    /// Helper function to create InactiveBlock instances for testing
    fn create_inactive_block(block_id: BlockId, seq_hash: u64) -> InactiveBlock {
        let complete_block = create_completed_block::<TestData>(&[seq_hash as u32], block_id);
        InactiveBlock {
            block_id,
            seq_hash: complete_block.sequence_hash(),
        }
    }

    #[test]
    fn test_fifo_ordering_basic() {
        let mut policy = FifoReusePolicy::new();

        // Insert blocks
        let block1 = create_inactive_block(1, 100);
        let block2 = create_inactive_block(2, 200);
        let block3 = create_inactive_block(3, 300);

        let seq_hash1 = block1.seq_hash;
        let seq_hash2 = block2.seq_hash;
        let seq_hash3 = block3.seq_hash;

        policy.insert(block1).unwrap();
        policy.insert(block2).unwrap();
        policy.insert(block3).unwrap();

        // Verify FIFO order - first inserted should come out first
        assert_eq!(policy.len(), 3);
        assert!(!policy.is_empty());

        let retrieved1 = policy.next_free().unwrap();
        assert_eq!(retrieved1.block_id, 1);
        assert_eq!(retrieved1.seq_hash, seq_hash1);

        let retrieved2 = policy.next_free().unwrap();
        assert_eq!(retrieved2.block_id, 2);
        assert_eq!(retrieved2.seq_hash, seq_hash2);

        let retrieved3 = policy.next_free().unwrap();
        assert_eq!(retrieved3.block_id, 3);
        assert_eq!(retrieved3.seq_hash, seq_hash3);

        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
    }

    #[test]
    fn test_fifo_ordering_with_delays() {
        let mut policy = FifoReusePolicy::new();

        // Insert blocks
        let blocks = vec![
            create_inactive_block(10, 1000),
            create_inactive_block(20, 2000),
            create_inactive_block(30, 3000),
            create_inactive_block(40, 4000),
        ];

        for block in blocks {
            policy.insert(block).unwrap();
        }

        // Retrieve all blocks and verify FIFO order
        let expected_order = vec![10, 20, 30, 40];
        let mut retrieved_order = Vec::new();

        while let Some(block) = policy.next_free() {
            retrieved_order.push(block.block_id);
        }

        assert_eq!(retrieved_order, expected_order);
    }

    #[test]
    fn test_insert_and_remove() {
        let mut policy = FifoReusePolicy::new();

        // Insert several blocks
        let blocks = vec![
            create_inactive_block(1, 100),
            create_inactive_block(2, 200),
            create_inactive_block(3, 300),
            create_inactive_block(4, 400),
        ];

        for block in blocks {
            policy.insert(block).unwrap();
        }

        assert_eq!(policy.len(), 4);

        // Remove block 2 (second inserted)
        policy.remove(2).unwrap();
        assert_eq!(policy.len(), 3);

        // Retrieve remaining blocks - should be 1, 3, 4 in that order
        let retrieved1 = policy.next_free().unwrap();
        assert_eq!(retrieved1.block_id, 1);

        let retrieved2 = policy.next_free().unwrap();
        assert_eq!(retrieved2.block_id, 3);

        let retrieved3 = policy.next_free().unwrap();
        assert_eq!(retrieved3.block_id, 4);

        assert!(policy.is_empty());
    }

    #[test]
    fn test_empty_operations() {
        let mut policy = FifoReusePolicy::new();

        // Test empty state
        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
        assert!(policy.next_free().is_none());

        // Insert and remove a block
        let block = create_inactive_block(1, 100);
        policy.insert(block).unwrap();
        assert!(!policy.is_empty());
        assert_eq!(policy.len(), 1);

        let retrieved = policy.next_free().unwrap();
        assert_eq!(retrieved.block_id, 1);

        // Should be empty again
        assert!(policy.is_empty());
        assert_eq!(policy.len(), 0);
        assert!(policy.next_free().is_none());
    }

    #[test]
    #[should_panic(expected = "block already exists")]
    fn test_duplicate_block_panic() {
        let mut policy = FifoReusePolicy::new();

        let block = create_inactive_block(1, 100);
        policy.insert(block).unwrap();

        // Inserting the same block ID again should panic
        let duplicate_block = create_inactive_block(1, 200); // Same ID, different hash
        policy.insert(duplicate_block).unwrap();
    }

    #[test]
    fn test_remove_nonexistent_block() {
        let mut policy = FifoReusePolicy::new();

        // Try to remove from empty policy
        let result = policy.remove(999);
        assert!(matches!(result, Err(ReusePolicyError::BlockNotFound(_))));

        // Insert a block and try to remove a different one
        let block = create_inactive_block(1, 100);
        policy.insert(block).unwrap();

        let result = policy.remove(999);
        assert!(matches!(result, Err(ReusePolicyError::BlockNotFound(_))));

        // Verify the original block is still there
        assert_eq!(policy.len(), 1);
        let retrieved = policy.next_free().unwrap();
        assert_eq!(retrieved.block_id, 1);
    }

    #[test]
    fn test_interleaved_operations() {
        let mut policy = FifoReusePolicy::new();

        // Insert some blocks
        policy.insert(create_inactive_block(1, 100)).unwrap();
        policy.insert(create_inactive_block(2, 200)).unwrap();
        policy.insert(create_inactive_block(3, 300)).unwrap();

        // Remove the first one
        let first = policy.next_free().unwrap();
        assert_eq!(first.block_id, 1);

        // Insert another block
        policy.insert(create_inactive_block(4, 400)).unwrap();

        // Remove a specific block by ID
        policy.remove(3).unwrap();

        // Insert another block
        policy.insert(create_inactive_block(5, 500)).unwrap();

        // The remaining blocks should come out in order: 2, 4, 5
        let second = policy.next_free().unwrap();
        assert_eq!(second.block_id, 2);

        let third = policy.next_free().unwrap();
        assert_eq!(third.block_id, 4);

        let fourth = policy.next_free().unwrap();
        assert_eq!(fourth.block_id, 5);

        assert!(policy.is_empty());
    }

    #[test]
    fn test_priority_key_ordering() {
        let mut policy = FifoReusePolicy::new();

        // Insert blocks — monotonic counter guarantees unique, ordered keys
        for i in 1..=5 {
            policy
                .insert(create_inactive_block(i, i as u64 * 100))
                .unwrap();
        }

        // Verify each key is unique and strictly increasing
        let keys: Vec<PriorityKey> = (1..=5).map(|id| *policy.keys.get(&id).unwrap()).collect();
        for window in keys.windows(2) {
            assert!(window[0] < window[1], "keys must be strictly increasing");
        }

        // Retrieve all blocks and verify they come out in insertion order
        let mut retrieval_order: Vec<BlockId> = Vec::new();
        while let Some(block) = policy.next_free() {
            retrieval_order.push(block.block_id);
        }

        let expected_order: Vec<BlockId> = (1..=5).collect();
        assert_eq!(retrieval_order, expected_order);
    }

    #[test]
    fn test_btreemap_ordering_assumption() {
        use std::collections::BTreeMap;

        // Verify our assumption about BTreeMap ordering with u64 keys
        let mut map = BTreeMap::new();

        // Insert keys in non-sorted order
        map.insert(100u64, "hundred");
        map.insert(10u64, "ten");
        map.insert(50u64, "fifty");
        map.insert(1u64, "one");
        map.insert(200u64, "two_hundred");

        // pop_first should return the smallest key first
        assert_eq!(map.pop_first(), Some((1, "one")));
        assert_eq!(map.pop_first(), Some((10, "ten")));
        assert_eq!(map.pop_first(), Some((50, "fifty")));
        assert_eq!(map.pop_first(), Some((100, "hundred")));
        assert_eq!(map.pop_first(), Some((200, "two_hundred")));
        assert_eq!(map.pop_first(), None);
    }
}
