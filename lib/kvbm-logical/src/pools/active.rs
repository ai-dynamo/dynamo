// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Active pool for managing blocks that are currently in use (have strong references).
//!
//! This pool provides a layer of abstraction over the BlockRegistry for finding
//! active blocks. Active blocks are those that have been registered and are
//! currently being used, as opposed to inactive blocks which are available
//! for reuse.

use std::sync::Arc;

use super::{BlockMetadata, RegisteredBlock, SequenceHash};
use crate::blocks::RegisteredReturnFn;
use crate::registry::BlockRegistry;

/// Pool for managing active (in-use) blocks.
///
/// This is a simple wrapper around BlockRegistry that encapsulates the logic
/// for finding blocks that are currently active (have strong references).
pub(crate) struct ActivePool<T: BlockMetadata> {
    block_registry: BlockRegistry,
    return_fn: RegisteredReturnFn<T>,
}

impl<T: BlockMetadata> ActivePool<T> {
    /// Create a new ActivePool with the given registry and return function.
    pub(crate) fn new(block_registry: BlockRegistry, return_fn: RegisteredReturnFn<T>) -> Self {
        Self {
            block_registry,
            return_fn,
        }
    }

    /// Find multiple blocks by sequence hashes, stopping on first miss.
    ///
    /// This searches for active blocks in the registry and returns them as
    /// RegisteredBlock guards. If any hash is not found or the block cannot
    /// be retrieved, the search stops and returns only the blocks found so far.
    #[inline]
    pub(crate) fn find_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        let mut matches = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(handle) = self.block_registry.match_sequence_hash(*hash, touch) {
                if let Some(block) = handle.try_get_block::<T>(self.return_fn.clone()) {
                    matches.push(block);
                } else {
                    break; // Stop on first miss
                }
            } else {
                break; // Stop on first miss
            }
        }

        matches
    }

    /// Scan for blocks in the active pool (doesn't stop on miss).
    ///
    /// Unlike `find_matches`, this continues scanning even when a hash is not found.
    /// Returns all found blocks with their corresponding sequence hashes.
    #[inline]
    pub(crate) fn scan_matches(
        &self,
        hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        hashes
            .iter()
            .filter_map(|hash| {
                self.block_registry
                    .match_sequence_hash(*hash, false)
                    .and_then(|handle| {
                        handle
                            .try_get_block::<T>(self.return_fn.clone())
                            .map(|block| (*hash, block))
                    })
            })
            .collect()
    }

    // /// Find a single block by sequence hash.
    // ///
    // /// Returns the block if found and active, None otherwise.
    // #[inline]
    // pub(crate) fn find_match(&self, seq_hash: SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> {
    //     self.block_registry
    //         .match_sequence_hash(seq_hash, true)
    //         .and_then(|handle| handle.try_get_block::<T>(self.return_fn.clone()))
    // }

    // /// Check if a block with the given sequence hash is currently active.
    // pub(crate) fn has_block(&self, seq_hash: SequenceHash) -> bool {
    //     self.find_match(seq_hash).is_some()
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::{Block, PrimaryBlock, state::Reset};
    use crate::pools::backends::{FifoReusePolicy, HashMapBackend};
    use crate::pools::inactive::InactivePool;
    use crate::pools::reset::ResetPool;
    use crate::testing::{TestMeta, create_staged_block, tokens_for_id};

    fn create_test_setup() -> (
        ActivePool<TestMeta>,
        InactivePool<TestMeta>,
        BlockRegistry,
        ResetPool<TestMeta>,
    ) {
        let registry = BlockRegistry::new();

        let reset_blocks: Vec<Block<TestMeta, Reset>> =
            (0..10_usize).map(|i| Block::new(i, 4)).collect();
        let reset_pool = ResetPool::new(reset_blocks, 4, None);

        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        let inactive_pool = InactivePool::new(backend, &reset_pool, None);

        let active_pool = ActivePool::new(registry.clone(), inactive_pool.return_fn());

        (active_pool, inactive_pool, registry, reset_pool)
    }

    /// Register a staged block and hold a strong reference to make it "active".
    fn make_active_block(
        registry: &BlockRegistry,
        return_fn: &RegisteredReturnFn<TestMeta>,
        id: usize,
        tokens: &[u32],
    ) -> (Arc<PrimaryBlock<TestMeta>>, SequenceHash) {
        let staged = create_staged_block::<TestMeta>(id, tokens);
        let seq_hash = staged.sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);
        let registered = staged.register_with_handle(handle);
        let primary = PrimaryBlock::new_attached(Arc::new(registered), return_fn.clone());
        (primary, seq_hash)
    }

    #[test]
    fn test_find_matches() {
        let (active_pool, inactive_pool, registry, _reset_pool) = create_test_setup();
        let return_fn = inactive_pool.return_fn();

        let (_hold1, hash1) = make_active_block(&registry, &return_fn, 1, &tokens_for_id(1));
        let (_hold2, hash2) = make_active_block(&registry, &return_fn, 2, &tokens_for_id(2));
        let (_hold3, hash3) = make_active_block(&registry, &return_fn, 3, &tokens_for_id(3));

        let found = active_pool.find_matches(&[hash1, hash2, hash3], true);
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].block_id(), 1);
        assert_eq!(found[1].block_id(), 2);
        assert_eq!(found[2].block_id(), 3);
    }

    #[test]
    fn test_find_matches_stops_on_miss() {
        let (active_pool, inactive_pool, registry, _reset_pool) = create_test_setup();
        let return_fn = inactive_pool.return_fn();

        let (_hold1, hash1) = make_active_block(&registry, &return_fn, 1, &tokens_for_id(1));
        let (_hold3, hash3) = make_active_block(&registry, &return_fn, 3, &tokens_for_id(3));

        // Create a hash that's not in the registry
        let missing_hash = {
            let staged = create_staged_block::<TestMeta>(999, &[9999, 9998, 9997, 9996]);
            staged.sequence_hash()
        };

        let found = active_pool.find_matches(&[hash1, missing_hash, hash3], true);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].block_id(), 1);
    }

    #[test]
    fn test_scan_matches() {
        let (active_pool, inactive_pool, registry, _reset_pool) = create_test_setup();
        let return_fn = inactive_pool.return_fn();

        let (_hold1, hash1) = make_active_block(&registry, &return_fn, 1, &tokens_for_id(1));
        let (_hold3, hash3) = make_active_block(&registry, &return_fn, 3, &tokens_for_id(3));

        let missing_hash = {
            let staged = create_staged_block::<TestMeta>(999, &[9999, 9998, 9997, 9996]);
            staged.sequence_hash()
        };

        // scan_matches doesn't stop on miss — should find both 1 and 3
        let found = active_pool.scan_matches(&[hash1, missing_hash, hash3]);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].0, hash1);
        assert_eq!(found[0].1.block_id(), 1);
        assert_eq!(found[1].0, hash3);
        assert_eq!(found[1].1.block_id(), 3);
    }

    #[test]
    fn test_find_matches_empty() {
        let (active_pool, _inactive_pool, _registry, _reset_pool) = create_test_setup();

        let found = active_pool.find_matches(&[], true);
        assert!(found.is_empty());
    }

    #[test]
    fn test_find_matches_no_active_blocks() {
        let (active_pool, _inactive_pool, _registry, _reset_pool) = create_test_setup();

        let missing_hash = {
            let staged = create_staged_block::<TestMeta>(999, &[9999, 9998, 9997, 9996]);
            staged.sequence_hash()
        };

        let found = active_pool.find_matches(&[missing_hash], true);
        assert!(found.is_empty());
    }
}
