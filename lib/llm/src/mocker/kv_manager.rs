// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # KV Manager
//! A synchronous implementation of a block manager that handles MoveBlock signals for caching KV blocks.
//!
//! ## Block Operations
//! The KV manager processes four types of MoveBlock signals:
//!
//! ### Use
//! - Checks if block exists in active pool → increment reference count
//! - If in inactive pool → move to active pool
//! - If neither → try evicting from inactive pool to make room
//! - If inactive pool is empty → pre-empt the oldest running request
//!
//! ### Destroy
//! - Removes the block from the active pool
//!
//! ### Deref
//! - Decrements reference count of a block in active pool
//! - If count reaches zero → move block to inactive pool
//!
//! ### Promote
//! - Converts a partial block (uuid) into a full block (global block hash)
//!
//! ## Preemption
//! If a Use operation fails (typically due to insufficient space), a false boolean signal
//! is returned to the scheduler for preemption. Initial KV block allocations for new requests
//! should not fail due to the watermark checking.
//!
//! ## NOTE
//! For simplicity (or non-simplicity), reference counting is tracked manually instead of using
//! the more idiomatic built-in Arc reference counter. This can be considered a shadow / mirror
//! implementation of the main block manager.

use crate::mocker::evictor::LRUEvictor;
use crate::mocker::protocols::{MoveBlock, PrefillCost, UniqueBlock};
use crate::mocker::sequence::ActiveSequence;
use derive_getters::Getters;
use std::collections::{HashMap, HashSet};

#[derive(Getters)]
pub struct KvManager {
    #[getter(copy)]
    max_capacity: usize,

    #[getter(copy)]
    block_size: usize,

    active_blocks: HashMap<UniqueBlock, usize>,

    inactive_blocks: LRUEvictor<UniqueBlock>,

    all_blocks: HashSet<UniqueBlock>,
}

impl KvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        let active_blocks = HashMap::new();
        let inactive_blocks = LRUEvictor::default();
        let all_blocks = HashSet::new();

        KvManager {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            all_blocks,
        }
    }

    /// Process a MoveBlock instruction synchronously
    pub fn process(&mut self, event: &MoveBlock) -> bool {
        match event {
            MoveBlock::Use(hashes, _) => {
                for hash in hashes {
                    // First check if it already exists in active blocks
                    if let Some(ref_count) = self.active_blocks.get_mut(hash) {
                        // Block already active, just increment reference count
                        *ref_count += 1;
                        continue;
                    }

                    // Then check if it exists in inactive and move it to active if found
                    if self.inactive_blocks.remove(hash) {
                        // Insert into active with reference count 1
                        self.active_blocks.insert(hash.clone(), 1);
                        continue;
                    }

                    // Get counts for capacity check
                    let active_count = self.active_blocks.len();
                    let inactive_count = self.inactive_blocks.len();

                    // If at max capacity, evict the oldest entry from inactive blocks
                    if active_count + inactive_count >= self.max_capacity {
                        if let Some(evicted) = self.inactive_blocks.evict() {
                            // Remove evicted block from all_blocks
                            self.all_blocks.remove(&evicted);
                        } else {
                            // Cannot evict block, meaning no free blocks left in inactive pool
                            // Send a signal, scheduler would expect to handle preemption upon receiving this
                            return false;
                        }
                    }

                    // Now insert the new block in active blocks with reference count 1
                    self.active_blocks.insert(hash.clone(), 1);
                    // Add to all_blocks as it's a new block
                    self.all_blocks.insert(hash.clone());
                }
            }
            MoveBlock::Destroy(hashes) => {
                // Loop in inverse direction
                for hash in hashes.iter().rev() {
                    self.active_blocks.remove(hash).unwrap();
                    // Remove from all_blocks when destroyed
                    assert!(self.all_blocks.remove(hash));
                }
            }
            MoveBlock::Deref(hashes) => {
                // Loop in inverse direction
                for hash in hashes.iter().rev() {
                    // Decrement reference count and check if we need to move to inactive
                    if let Some(ref_count) = self.active_blocks.get_mut(hash) {
                        if *ref_count == 0 {
                            panic!("Negative reference count would be encountered after Deref.");
                        }
                        *ref_count -= 1;

                        // If reference count reaches zero, remove from active and move to inactive
                        if *ref_count == 0 {
                            self.active_blocks.remove(hash);
                            // Use the LRUEvictor's timing functionality
                            self.inactive_blocks.insert(hash.clone());
                        }
                    }
                }
            }
            MoveBlock::Promote(uuid, hash) => {
                let uuid_block = UniqueBlock::PartialBlock(*uuid);
                let hash_block = UniqueBlock::FullBlock(*hash);

                let Some(ref_count) = self.active_blocks.remove(&uuid_block) else {
                    let in_all_blocks = self.all_blocks.contains(&uuid_block);
                    panic!(
                        "Missing active block for promotion: {:?}. Block still exists: {}",
                        uuid_block, in_all_blocks
                    );
                };

                // Replace with hash block, keeping the same reference count
                self.active_blocks.insert(hash_block.clone(), ref_count);

                // Update all_blocks
                assert!(self.all_blocks.remove(&uuid_block));
                self.all_blocks.insert(hash_block);
            }
        }

        // Return true if we made it this far
        true
    }

    /// Get the count of blocks in the input list that aren't in all_blocks
    pub fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize {
        blocks
            .iter()
            .filter(|&block| !self.all_blocks.contains(block))
            .count()
    }

    /// Get the current capacity (active blocks + inactive blocks)
    pub fn current_capacity(&self) -> usize {
        let active = self.active_blocks.len();
        let inactive = self.inactive_blocks.len();
        active + inactive
    }

    /// Get the current capacity as a percentage of the maximum capacity
    pub fn current_capacity_perc(&self) -> f64 {
        let current = self.current_capacity() as f64;
        current / self.max_capacity as f64
    }

    /// Get the number of active blocks
    pub fn num_active_blocks(&self) -> usize {
        self.active_blocks.len()
    }

    /// Get the number of inactive blocks
    pub fn num_inactive_blocks(&self) -> usize {
        self.inactive_blocks.len()
    }

    /// Get the keys of inactive blocks
    pub fn get_inactive_blocks(&self) -> Vec<&UniqueBlock> {
        self.inactive_blocks.keys().collect()
    }

    /// Get the keys of active blocks
    pub fn get_active_blocks(&self) -> Vec<&UniqueBlock> {
        self.active_blocks.keys().collect()
    }

    /// Check if a sequence can be scheduled and calculate cost if possible
    pub fn try_schedule(
        &self,
        sequence: &ActiveSequence,
        watermark: f64,
        tokens_budget: usize,
    ) -> Option<PrefillCost> {
        // Return None immediately if tokens_budget is 0
        if tokens_budget == 0 {
            return None;
        }

        // Get unique blocks from the sequence
        let unique_blocks = sequence.unique_blocks();

        // Get the count of new blocks
        let new_blocks = self.probe_new_blocks(unique_blocks);

        // Calculate current usage and available capacity
        let active_count = self.active_blocks.len();

        // Check if we can schedule based on the watermark
        if (active_count + new_blocks) as f64 > (1.0 - watermark) * self.max_capacity as f64 {
            return None;
        }

        // Calculate overlap blocks
        let overlap_blocks = unique_blocks.len() - new_blocks;

        // Calculate new tokens
        let new_tokens = sequence.num_input_tokens() - overlap_blocks * self.block_size;

        // // Print the full equation with actual values substituted
        // println!("{} = {} - ({} * {}) (new_tokens = num_input_tokens - overlap_blocks * block_size)",
        //     new_tokens,
        //     sequence.num_input_tokens(),
        //     overlap_blocks,
        //     self.block_size);

        // Return None if new_tokens exceeds tokens_budget
        if new_tokens > tokens_budget {
            return None;
        }

        // Calculate prefill compute
        let prefill_compute =
            new_tokens as f64 * (new_tokens + overlap_blocks * self.block_size) as f64;

        Some(PrefillCost {
            new_tokens,
            prefill_compute,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failure_on_max_capacity() {
        // Create a KvManager with 10 blocks capacity
        let mut manager = KvManager::new(10, 16);

        // Helper function to use multiple blocks that returns the response
        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) -> bool {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Use(blocks, None))
        }

        // First use 10 blocks (0 to 9) in a batch
        let response = use_blocks(&mut manager, (0..10).collect());
        assert!(response, "Expected success response");

        // Verify we are at capacity
        assert_eq!(manager.current_capacity(), 10);

        // The 11th block should return false, not panic
        let response = use_blocks(&mut manager, vec![10]);
        assert!(
            !response,
            "Expected failure response when exceeding max capacity"
        );
    }

    #[test]
    // This is taken directly from the example in the vllm v1 prefix caching docs
    fn test_block_lifecycle_stringent() {
        // Create a KvManager with 10 blocks capacity
        let mut manager = KvManager::new(10, 16);

        // Helper function to use multiple blocks
        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Use(blocks, None));
        }

        // Helper function to destroy multiple blocks
        fn destroy_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Destroy(blocks));
        }

        // Helper function to deref multiple blocks
        fn deref_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Deref(blocks));
        }

        // Helper function to check if active blocks contain expected blocks with expected ref counts
        fn assert_active_blocks(manager: &KvManager, expected_blocks: &[(u64, usize)]) {
            assert_eq!(
                manager.active_blocks().len(),
                expected_blocks.len(),
                "Active blocks count doesn't match expected"
            );

            for &(id, ref_count) in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    manager.active_blocks().contains_key(&block),
                    "Block {} not found in active blocks",
                    id
                );
                assert_eq!(
                    manager.active_blocks().get(&block),
                    Some(&ref_count),
                    "Block {} has wrong reference count",
                    id
                );
            }
        }

        // Helper function to check if inactive blocks contain expected blocks
        fn assert_inactive_blocks(
            manager: &KvManager,
            expected_size: usize,
            expected_blocks: &[u64],
        ) {
            let inactive_blocks = manager.get_inactive_blocks();
            let inactive_blocks_count = manager.inactive_blocks().len();

            assert_eq!(
                inactive_blocks_count, expected_size,
                "Inactive blocks count doesn't match expected"
            );

            for &id in expected_blocks {
                let block = UniqueBlock::FullBlock(id);
                assert!(
                    inactive_blocks.iter().any(|&b| *b == block),
                    "Block {} not found in inactive blocks",
                    id
                );
            }
        }

        // First use blocks 0, 1, 2, 3, 4 in a batch
        use_blocks(&mut manager, (0..5).collect());

        // Then use blocks 0, 1, 5, 6 in a batch
        use_blocks(&mut manager, vec![0, 1, 5, 6]);

        // Check that the blocks 0 and 1 are in active blocks, both with reference counts of 2
        assert_active_blocks(
            &manager,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        // Now destroy block 4
        destroy_blocks(&mut manager, vec![4]);

        // And deref blocks 3, 2, 1, 0 in this order as a batch
        deref_blocks(&mut manager, vec![0, 1, 2, 3]);

        // Check that the inactive_blocks is size 2 (via num_objects) and contains 3 and 2
        assert_inactive_blocks(&manager, 2, &[3, 2]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (5, 1), (6, 1)]);

        // Now destroy block 6
        destroy_blocks(&mut manager, vec![6]);

        // And deref blocks 5, 1, 0 as a batch
        deref_blocks(&mut manager, vec![0, 1, 5]);

        // Check that the inactive_blocks is size 5, and contains 0, 1, 2, 3, 5
        assert_inactive_blocks(&manager, 5, &[0, 1, 2, 3, 5]);
        assert_active_blocks(&manager, &[]);

        // Now use 0, 1, 2, 7, 8, 9 as a batch
        use_blocks(&mut manager, vec![0, 1, 2, 7, 8, 9]);

        // Check that the inactive_blocks is size 2, and contains 3 and 5
        assert_inactive_blocks(&manager, 2, &[3, 5]);
        assert_active_blocks(&manager, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);

        // Test the new_blocks method - only block 4 should be new out of [0,1,2,3,4]
        let blocks_to_check: Vec<UniqueBlock> = vec![0, 1, 2, 3, 4]
            .into_iter()
            .map(UniqueBlock::FullBlock)
            .collect();
        assert_eq!(manager.probe_new_blocks(&blocks_to_check), 1);

        // Now use blocks 10, 11, 12 as a batch
        use_blocks(&mut manager, vec![10, 11, 12]);

        // Check that the inactive_blocks is size 1 and contains only 5
        assert_inactive_blocks(&manager, 1, &[5]);
    }
}
