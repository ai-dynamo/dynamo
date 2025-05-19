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

use crate::mocker::evictor::LRUEvictor;
use crate::mocker::protocols::{MoveBlock, MoveBlockResponse, PrefillCost, UniqueBlock};
use crate::mocker::sequence::ActiveSequence;
use std::collections::{HashMap, HashSet};

/// Mock implementation of worker for testing and simulation
pub struct KvManager {
    max_capacity: usize,
    block_size: usize,
    active_blocks: HashMap<UniqueBlock, usize>,
    inactive_blocks: LRUEvictor<UniqueBlock>,
    all_blocks: HashSet<UniqueBlock>,
}

impl KvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        let active_blocks = HashMap::new();
        let inactive_blocks = LRUEvictor::new();
        let all_blocks = HashSet::new();

        KvManager {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            all_blocks,
        }
    }

    /// Get the maximum capacity
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get a reference to the active blocks
    pub fn active_blocks(&self) -> &HashMap<UniqueBlock, usize> {
        &self.active_blocks
    }

    /// Get a reference to the inactive blocks
    pub fn inactive_blocks(&self) -> &LRUEvictor<UniqueBlock> {
        &self.inactive_blocks
    }

    /// Get a reference to all blocks
    pub fn all_blocks(&self) -> &HashSet<UniqueBlock> {
        &self.all_blocks
    }

    /// Process a MoveBlock instruction synchronously
    pub fn process(&mut self, event: &MoveBlock) -> MoveBlockResponse {
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
                    let inactive_count = self.inactive_blocks.num_objects();

                    // If at max capacity, evict the oldest entry from inactive blocks
                    if active_count + inactive_count >= self.max_capacity {
                        if let Some(evicted) = self.inactive_blocks.evict() {
                            // Remove evicted block from all_blocks
                            self.all_blocks.remove(&evicted);
                        } else {
                            // Return failure instead of panicking
                            return MoveBlockResponse::Failure;
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
                    self.active_blocks.remove(hash);
                    // Remove from all_blocks when destroyed
                    self.all_blocks.remove(hash);
                }
            }
            MoveBlock::Deref(hashes) => {
                // Loop in inverse direction
                for hash in hashes.iter().rev() {
                    // Decrement reference count and check if we need to move to inactive
                    if let Some(ref_count) = self.active_blocks.get_mut(hash) {
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

                // Check if the UUID block exists in active blocks
                if let Some(ref_count) = self.active_blocks.remove(&uuid_block) {
                    // Replace with hash block, keeping the same reference count
                    self.active_blocks.insert(hash_block.clone(), ref_count);

                    // Update all_blocks
                    self.all_blocks.remove(&uuid_block);
                    self.all_blocks.insert(hash_block);
                }
            }
        }

        // Return success if we made it this far
        MoveBlockResponse::Success
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
        let inactive = self.inactive_blocks.num_objects();
        active + inactive
    }

    /// Get the current capacity as a percentage of the maximum capacity
    pub fn current_capacity_perc(&self) -> f64 {
        let current = self.current_capacity() as f64;
        current / self.max_capacity as f64
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
        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) -> MoveBlockResponse {
            let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
            manager.process(&MoveBlock::Use(blocks, None))
        }

        // First use 10 blocks (0 to 9) in a batch
        let response = use_blocks(&mut manager, (0..10).collect());
        assert_eq!(response, MoveBlockResponse::Success);

        // Verify we are at capacity
        assert_eq!(manager.current_capacity(), 10);

        // The 11th block should return Failure, not panic
        let response = use_blocks(&mut manager, vec![10]);
        assert_eq!(
            response,
            MoveBlockResponse::Failure,
            "Expected Failure response when exceeding max capacity"
        );
    }

    #[test]
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
            let inactive_blocks_count = manager.inactive_blocks().num_objects();

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
