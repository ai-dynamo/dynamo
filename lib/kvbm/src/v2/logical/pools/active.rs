// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active pool for managing blocks that are currently in use (have strong references).
//!
//! This pool provides a layer of abstraction for finding active blocks.
//! It delegates to InactivePool which now handles both active (weak_blocks)
//! and inactive (backend) lookups under a unified lock.

use std::sync::Arc;

use super::{BlockMetadata, InactivePool, RegisteredBlock, SequenceHash};

/// Pool for managing active (in-use) blocks.
///
/// This delegates to InactivePool which now manages both active and inactive blocks
/// via its weak_blocks map and backend storage respectively.
pub struct ActivePool<T: BlockMetadata + Sync> {
    inactive_pool: InactivePool<T>,
}

impl<T: BlockMetadata + Sync> ActivePool<T> {
    /// Create a new ActivePool that delegates to the given InactivePool.
    pub fn new(inactive_pool: InactivePool<T>) -> Self {
        Self { inactive_pool }
    }

    /// Find multiple blocks by sequence hashes, stopping on first miss.
    ///
    /// This searches for blocks (both active and inactive) and returns them as
    /// RegisteredBlock guards. If any hash is not found, the search stops.
    #[inline]
    pub fn find_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        // Delegate to InactivePool which now handles both active and inactive lookups
        self.inactive_pool.find_blocks(hashes, touch)
    }

    /// Scan for blocks (doesn't stop on miss).
    ///
    /// Unlike `find_matches`, this continues scanning even when a hash is not found.
    /// Returns all found blocks with their corresponding sequence hashes.
    #[inline]
    pub fn scan_matches(
        &self,
        hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        self.inactive_pool.scan_blocks(hashes, false)
    }

    /// Find a single block by sequence hash.
    ///
    /// Returns the block if found, None otherwise.
    #[inline]
    pub fn find_match(&self, seq_hash: SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> {
        self.inactive_pool.find_or_promote_dyn(seq_hash)
    }

    /// Check if a block with the given sequence hash exists.
    #[expect(dead_code)]
    pub fn has_block(&self, seq_hash: SequenceHash) -> bool {
        self.find_match(seq_hash).is_some()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::super::{BlockDuplicationPolicy, CompleteBlock, FifoReusePolicy, Reset};
//     use super::*;
//     use crate::v2::pools::test_utils::TestData;
//     use crate::v2::pools::{
//         block::Block,
//         frequency_sketch::TinyLFUTracker,
//         inactive::{InactivePool, backends::hashmap_backend::HashMapBackend},
//         registry::BlockRegistry,
//         reset::ResetPool,
//     };
//     use dynamo_tokens::TokenBlockSequence;
//     use std::sync::Arc;

//     fn create_test_setup() -> (
//         ActivePool<TestData>,
//         InactivePool<TestData>,
//         BlockRegistry,
//         ResetPool<TestData>,
//     ) {
//         let frequency_tracker = Arc::new(TinyLFUTracker::new(100));
//         let registry = BlockRegistry::with_frequency_tracker(frequency_tracker);

//         let reset_blocks: Vec<_> = (0..10).map(|i| Block::new(i, 4)).collect();
//         let reset_pool = ResetPool::new(reset_blocks, 4);

//         let reuse_policy = Box::new(FifoReusePolicy::new());
//         let backend = Box::new(HashMapBackend::new(reuse_policy));
//         let inactive_pool = InactivePool::new(backend, &reset_pool);

//         let active_pool = ActivePool::new(registry.clone(), inactive_pool.return_fn());

//         (active_pool, inactive_pool, registry, reset_pool)
//     }

//     fn create_complete_block(id: u64, tokens: &[u32]) -> (CompleteBlock<TestData>, u64) {
//         let sequence = TokenBlockSequence::from_slice(tokens, 4, Some(42));
//         let token_block = if let Some(block) = sequence.blocks().first() {
//             block.clone()
//         } else {
//             let mut partial = sequence.into_parts().1;
//             partial.commit().expect("Should be able to commit")
//         };

//         let seq_hash = token_block.sequence_hash();
//         let complete_block = Block::new(id, 4)
//             .complete(token_block)
//             .expect("Block size should match");

//         // Create a dummy return function for testing
//         let return_fn = Arc::new(|_block: Block<TestData, Reset>| {
//             // In real usage this would return blocks to the reset pool
//         });

//         let complete_guard = CompleteBlock {
//             block: Some(complete_block),
//             return_fn,
//         };
//         (complete_guard, seq_hash)
//     }

//     #[test]
//     fn test_active_pool_find_match() {
//         let (active_pool, _inactive_pool, registry, _reset_pool) = create_test_setup();

//         let (complete_block, seq_hash) = create_complete_block(1, &[100, 101, 102, 103]);

//         let handle = registry.register_sequence_hash(seq_hash);
//         let _immutable_block = handle.register_block(
//             complete_block,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         let found_block = active_pool.find_match(seq_hash);
//         assert!(found_block.is_some());

//         let found_block = found_block.unwrap();
//         assert_eq!(found_block.block_id(), 1);
//         assert_eq!(found_block.sequence_hash(), seq_hash);
//     }

//     #[test]
//     fn test_active_pool_find_matches() {
//         let (active_pool, _inactive_pool, registry, _reset_pool) = create_test_setup();

//         let (complete_block1, seq_hash1) = create_complete_block(1, &[100, 101, 102, 103]);
//         let (complete_block2, seq_hash2) = create_complete_block(2, &[200, 201, 202, 203]);
//         let (complete_block3, seq_hash3) = create_complete_block(3, &[300, 301, 302, 303]);

//         // Register blocks
//         let handle1 = registry.register_sequence_hash(seq_hash1);
//         let _immutable1 = handle1.register_block(
//             complete_block1,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         let handle2 = registry.register_sequence_hash(seq_hash2);
//         let _immutable2 = handle2.register_block(
//             complete_block2,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         let handle3 = registry.register_sequence_hash(seq_hash3);
//         let _immutable3 = handle3.register_block(
//             complete_block3,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         // Find all three blocks
//         let found_blocks = active_pool.find_matches(&[seq_hash1, seq_hash2, seq_hash3], true);
//         assert_eq!(found_blocks.len(), 3);
//         assert_eq!(found_blocks[0].block_id(), 1);
//         assert_eq!(found_blocks[1].block_id(), 2);
//         assert_eq!(found_blocks[2].block_id(), 3);
//     }

//     #[test]
//     fn test_active_pool_find_matches_stops_on_miss() {
//         let (active_pool, _inactive_pool, registry, _reset_pool) = create_test_setup();

//         let (complete_block1, seq_hash1) = create_complete_block(1, &[100, 101, 102, 103]);
//         let (complete_block3, seq_hash3) = create_complete_block(3, &[300, 301, 302, 303]);

//         // Register only blocks 1 and 3
//         let handle1 = registry.register_sequence_hash(seq_hash1);
//         let _immutable1 = handle1.register_block(
//             complete_block1,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         let handle3 = registry.register_sequence_hash(seq_hash3);
//         let _immutable3 = handle3.register_block(
//             complete_block3,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         // Try to find blocks 1, 2, 3 - should stop at 2 since it's missing
//         let missing_hash = 999;
//         let found_blocks = active_pool.find_matches(&[seq_hash1, missing_hash, seq_hash3], true);

//         // Should only find block 1, then stop on missing hash
//         assert_eq!(found_blocks.len(), 1);
//         assert_eq!(found_blocks[0].block_id(), 1);
//     }

//     #[test]
//     fn test_active_pool_has_block() {
//         let (active_pool, _inactive_pool, registry, _reset_pool) = create_test_setup();

//         let (complete_block, seq_hash) = create_complete_block(1, &[100, 101, 102, 103]);

//         // Block should not be found initially
//         assert!(!active_pool.has_block(seq_hash));

//         // Register the block
//         let handle = registry.register_sequence_hash(seq_hash);
//         let _immutable_block = handle.register_block(
//             complete_block,
//             BlockDuplicationPolicy::Allow,
//             active_pool.return_fn.clone(),
//         );

//         // Now block should be found
//         assert!(active_pool.has_block(seq_hash));

//         // Drop the immutable block
//         drop(_immutable_block);

//         // Block should no longer be active (but might still be registered)
//         // This depends on the exact behavior of the registry
//     }

//     #[test]
//     fn test_active_pool_empty_search() {
//         let (active_pool, _inactive_pool, _registry, _reset_pool) = create_test_setup();

//         let found_blocks = active_pool.find_matches(&[], true);
//         assert_eq!(found_blocks.len(), 0);

//         let found_block = active_pool.find_match(999);
//         assert!(found_block.is_none());

//         assert!(!active_pool.has_block(999));
//     }
// }
