// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

use crate::kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use crate::kv_router::publisher::KvEventPublisher;
use crate::mocker::evictor::LRUEvictor;
use crate::mocker::protocols::{MoveBlock, PrefillCost};
use crate::mocker::sequence::ActiveSequence;
use crate::tokens::blocks::UniqueBlock;
use crate::tokens::{BlockHash, SequenceHash};
use derive_getters::Getters;
use dynamo_runtime::component::Component;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Getters)]
pub struct KvManager {
    #[getter(copy)]
    max_capacity: usize,

    #[getter(copy)]
    block_size: usize,

    active_blocks: HashMap<UniqueBlock, usize>,

    inactive_blocks: LRUEvictor<UniqueBlock>,

    all_blocks: HashSet<UniqueBlock>,

    kv_event_publisher: Option<Arc<KvEventPublisher>>,

    #[getter(copy)]
    dp_rank: u32,

    next_event_id: u64,
}

impl KvManager {
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        Self::new_with_publisher(max_capacity, block_size, None, 0, false)
    }

    pub fn new_with_publisher(
        max_capacity: usize,
        block_size: usize,
        component: Option<Component>,
        dp_rank: u32,
        enable_local_indexer: bool,
    ) -> Self {
        let active_blocks = HashMap::new();
        let inactive_blocks = LRUEvictor::default();
        let all_blocks = HashSet::new();

        let kv_event_publisher = component.map(|comp| {
            tracing::info!(
                "Initializing KV event publisher for DP rank {dp_rank} with block_size {block_size}, enable_local_indexer={enable_local_indexer}"
            );
            Arc::new(
                KvEventPublisher::new_with_local_indexer(comp, block_size as u32, None, enable_local_indexer)
                    .expect("Failed to create KV event publisher"),
            )
        });

        KvManager {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            all_blocks,
            kv_event_publisher,
            dp_rank,
            next_event_id: 0,
        }
    }

    /// Converts stored/removed blocks into KvCacheEventData and publishes if publisher is available
    fn publish_kv_event(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<u64>,
        is_store: bool,
    ) {
        if full_blocks.is_empty() {
            return;
        }

        let Some(ref publisher) = self.kv_event_publisher else {
            return;
        };

        let event_data = if is_store {
            let num_blocks = full_blocks.len();
            let local_hashes_slice = &local_hashes[local_hashes
                .len()
                .checked_sub(num_blocks)
                .expect("local hashes fewer than stored blocks")..];

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                blocks: full_blocks
                    .into_iter()
                    .zip(local_hashes_slice.iter())
                    .map(|(global_hash, local_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(*local_hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            })
        } else {
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: full_blocks
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            })
        };

        // Use incremental event ID starting from 0
        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let event = KvCacheEvent {
            event_id,
            data: event_data,
            dp_rank: self.dp_rank,
        };

        if let Err(e) = publisher.publish(event) {
            tracing::warn!("Failed to publish KV event: {e}");
        }
    }

    /// Process a MoveBlock instruction synchronously
    pub fn process(&mut self, event: &MoveBlock) -> bool {
        match event {
            MoveBlock::Use(hashes, local_hashes) => {
                let mut blocks_stored = Vec::<u64>::new();

                let mut parent_block: Option<&UniqueBlock> = None;
                for hash in hashes {
                    // First check if it already exists in active blocks
                    if let Some(ref_count) = self.active_blocks.get_mut(hash) {
                        // Block already active, just increment reference count
                        *ref_count += 1;
                        parent_block = Some(hash);
                        continue;
                    }

                    // Then check if it exists in inactive and move it to active if found
                    if self.inactive_blocks.remove(hash) {
                        // Insert into active with reference count 1
                        self.active_blocks.insert(hash.clone(), 1);
                        parent_block = Some(hash);
                        continue;
                    }

                    // Get counts for capacity check
                    let active_count = self.active_blocks.len();
                    let inactive_count = self.inactive_blocks.len();

                    // If at max capacity, evict the oldest entry from inactive blocks
                    if active_count + inactive_count >= self.max_capacity {
                        let Some(evicted) = self.inactive_blocks.evict() else {
                            return false;
                        };
                        tracing::trace!(
                            "Evicting block from inactive pool: {evicted:?}, dp_rank={}",
                            self.dp_rank
                        );
                        self.all_blocks.remove(&evicted);
                        if let UniqueBlock::FullBlock(evicted_full_block) = evicted {
                            self.publish_kv_event(vec![evicted_full_block], &[], None, false);
                        }
                    }

                    // Now insert the new block in active blocks with reference count 1
                    self.active_blocks.insert(hash.clone(), 1);
                    self.all_blocks.insert(hash.clone());
                    if self.kv_event_publisher.is_some()
                        && let UniqueBlock::FullBlock(stored_full_block) = hash
                    {
                        blocks_stored.push(*stored_full_block);
                    }
                }

                let parent_hash = match parent_block {
                    None => None,
                    Some(UniqueBlock::FullBlock(block)) => Some(*block),
                    Some(UniqueBlock::PartialBlock(_)) => panic!("parent block cannot be partial"),
                };
                self.publish_kv_event(blocks_stored, local_hashes, parent_hash, true);
            }

            MoveBlock::Destroy(hashes) => {
                let mut blocks_destroyed = Vec::<u64>::new();

                // Process blocks in order (already reversed by caller if needed)
                for hash in hashes.iter() {
                    self.active_blocks.remove(hash).unwrap();
                    // Remove from all_blocks when destroyed
                    assert!(self.all_blocks.remove(hash));

                    // Track blocks for batch sending
                    if self.kv_event_publisher.is_some()
                        && let UniqueBlock::FullBlock(destroyed_full_block) = hash
                    {
                        blocks_destroyed.push(*destroyed_full_block);
                    }
                }

                self.publish_kv_event(blocks_destroyed, &[], None, false);
            }

            MoveBlock::Deref(hashes) => {
                // Process blocks in order (already reversed by caller if needed)
                for hash in hashes.iter() {
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

            MoveBlock::Promote(uuid, hash, parent_hash, local_hash) => {
                let uuid_block = UniqueBlock::PartialBlock(*uuid);
                let hash_block = UniqueBlock::FullBlock(*hash);

                assert_eq!(
                    self.active_blocks.remove(&uuid_block),
                    Some(1),
                    "uuid_block {uuid_block:?} should exist and be unique with ref_count=1"
                );

                let hash_ref_count = if let Some(ref_count) = self.active_blocks.get(&hash_block) {
                    *ref_count
                } else if self.inactive_blocks.remove(&hash_block) {
                    0
                } else {
                    self.publish_kv_event(vec![*hash], &[*local_hash], *parent_hash, true);
                    0
                };

                self.active_blocks
                    .insert(hash_block.clone(), hash_ref_count + 1);

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
            // .filter(|&block| !self.active_blocks.contains_key(block))
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

    /// Get the percentage of active blocks relative to maximum capacity
    pub fn get_active_perc(&self) -> f64 {
        self.active_blocks.len() as f64 / self.max_capacity as f64
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
    pub fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        let seq_blocks = sequence.unique_blocks();
        let new_blocks = self.probe_new_blocks(seq_blocks);
        let overlap_blocks = seq_blocks.len() - new_blocks;
        let new_tokens = sequence.num_input_tokens() - overlap_blocks * self.block_size;

        PrefillCost {
            new_blocks,
            new_tokens,
        }
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
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes))
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
    fn test_block_lifecycle_stringent() {
        // Create a KvManager with 10 blocks capacity (no KV event publisher for tests)
        let mut manager = KvManager::new(10, 16);

        // Helper function to use multiple blocks
        fn use_blocks(manager: &mut KvManager, ids: Vec<u64>) {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let hashes: Vec<_> = ids.into_iter().collect();
            manager.process(&MoveBlock::Use(blocks, hashes));
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
                    "Block {id} not found in active blocks",
                );
                assert_eq!(
                    manager.active_blocks().get(&block),
                    Some(&ref_count),
                    "Block {id} has wrong reference count",
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
                    "Block {id} not found in inactive blocks",
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

        use_blocks(&mut manager, vec![13]);
    }
}
