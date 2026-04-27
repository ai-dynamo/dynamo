// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block lifecycle orchestration over the unified [`BlockStore`].
//!
//! [`BlockManager`] owns a single [`BlockStore`] (which holds reset and
//! inactive pool bookkeeping under one mutex) plus the [`BlockRegistry`].
//! It exposes allocation, registration, matching, and scanning operations
//! while keeping all pool transitions behind a single API surface.
//!
//! Construction uses a builder pattern — see [`BlockManagerConfigBuilder`].

mod builder;

#[cfg(test)]
mod tests;

pub use builder::{
    BlockManagerBuilderError, BlockManagerConfigBuilder, BlockManagerResetError,
    FrequencyTrackingCapacity, InactiveBackendConfig,
};

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::blocks::{BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock, RegisteredBlock, UpgradeFn};
use crate::metrics::BlockPoolMetrics;
use crate::pools::{BlockDuplicationPolicy, BlockStore, SequenceHash};
use crate::registry::BlockRegistry;

/// Manages the full block lifecycle over the unified [`BlockStore`].
///
/// Thread-safe: allocation is serialised via an internal [`Mutex`] for the
/// reset+evict combination; the underlying [`BlockStore`] uses a single
/// internal mutex for all pool bookkeeping.
///
/// Construct via [`BlockManager::builder()`].
pub struct BlockManager<T: BlockMetadata> {
    pub(crate) store: Arc<BlockStore<T>>,
    pub(crate) block_registry: BlockRegistry,
    pub(crate) duplication_policy: BlockDuplicationPolicy,
    pub(crate) upgrade_fn: UpgradeFn<T>,
    pub(crate) allocate_mutex: Mutex<()>,
    pub(crate) total_blocks: usize,
    pub(crate) block_size: usize,
    pub(crate) metrics: Arc<BlockPoolMetrics>,
}

impl<T: BlockMetadata + Sync> BlockManager<T> {
    /// Create a new builder for BlockManager.
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    /// Allocate `count` mutable blocks, drawing first from the reset pool
    /// then evicting from the inactive pool if needed.
    ///
    /// Returns `None` if fewer than `count` blocks are available across both pools.
    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        self.allocate_blocks_with_evictions(count)
            .map(|(blocks, _evicted)| blocks)
    }

    /// Like [`allocate_blocks`](Self::allocate_blocks) but also reports the
    /// [`SequenceHash`] of each block evicted from the inactive pool.
    pub fn allocate_blocks_with_evictions(
        &self,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        let _guard = self.allocate_mutex.lock();
        let from_reset = self.store.allocate_reset_blocks(count);
        let from_reset_count = from_reset.len();
        let mut blocks = from_reset;

        let remaining_needed = count - blocks.len();
        match self.store.evict_to_reset(remaining_needed) {
            Some((remaining, evicted)) => {
                let eviction_count = remaining.len() as u64;
                blocks.extend(remaining);

                self.metrics.inc_allocations(blocks.len() as u64);
                self.metrics
                    .inc_allocations_from_reset(from_reset_count as u64);
                self.metrics.inc_evictions(eviction_count);

                Some((blocks, evicted))
            }
            None => None,
        }
    }

    /// Drain the inactive pool, returning all blocks to the reset pool.
    pub fn reset_inactive_pool(&self) -> Result<(), BlockManagerResetError> {
        let blocks = self.store.drain_inactive_to_reset();
        drop(blocks);

        let reset_count = self.store.reset_len();
        if reset_count != self.total_blocks {
            return Err(BlockManagerResetError::BlockCountMismatch {
                expected: self.total_blocks,
                actual: reset_count,
            });
        }

        Ok(())
    }

    /// Register a batch of completed blocks, returning immutable handles.
    pub fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        blocks
            .into_iter()
            .map(|block| self.register_block(block))
            .collect()
    }

    /// Register a single completed block and return an immutable handle.
    pub fn register_block(&self, block: CompleteBlock<T>) -> ImmutableBlock<T> {
        self.metrics.inc_registrations();
        let handle = self
            .block_registry
            .register_sequence_hash(block.sequence_hash());
        let registered_block = handle.register_block(
            block,
            self.duplication_policy,
            &self.store,
            Some(self.metrics.as_ref()),
        );
        ImmutableBlock::new(
            registered_block,
            self.upgrade_fn.clone(),
            Some(self.metrics.clone()),
        )
    }

    /// Linear prefix match: walks `seq_hash` left-to-right, stopping on first miss.
    ///
    /// Checks the active pool first (via the registry's weak refs), then the
    /// inactive pool for remaining hashes.
    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        self.metrics
            .inc_match_hashes_requested(seq_hash.len() as u64);

        tracing::debug!(
            num_hashes = seq_hash.len(),
            inactive_pool_len = self.store.inactive_len(),
            "match_blocks called"
        );

        // First try to match against active blocks via the registry
        let mut matched: Vec<ImmutableBlock<T>> = Vec::with_capacity(seq_hash.len());
        matched.extend(
            self.find_active_matches(seq_hash, true)
                .into_iter()
                .map(|block| {
                    ImmutableBlock::new(block, self.upgrade_fn.clone(), Some(self.metrics.clone()))
                }),
        );

        let active_matched = matched.len();
        tracing::debug!(active_matched, "Matched from active pool");

        // If we didn't match all hashes, try inactive blocks for the remaining ones
        let remaining_hashes = &seq_hash[matched.len()..];
        if !remaining_hashes.is_empty() {
            let inactive_found: Vec<_> = self.store.find_inactive_blocks(remaining_hashes, true);
            let inactive_matched = inactive_found.len();
            tracing::debug!(
                remaining_to_check = remaining_hashes.len(),
                inactive_matched,
                "Matched from inactive pool"
            );
            matched.extend(inactive_found.into_iter().map(|block| {
                ImmutableBlock::new(block, self.upgrade_fn.clone(), Some(self.metrics.clone()))
            }));
        }

        self.metrics.inc_match_blocks_returned(matched.len() as u64);

        tracing::debug!(total_matched = matched.len(), "match_blocks result");
        tracing::trace!(matched = ?matched, "matched blocks");
        matched
    }

    /// Scatter-gather scan: finds all blocks matching any hash, without stopping on misses.
    pub fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>> {
        self.metrics
            .inc_scan_hashes_requested(seq_hashes.len() as u64);

        let mut result = HashMap::new();

        // 1. Check active pool for all hashes (via registry, no touch)
        let active_found = self.scan_active_matches(seq_hashes);
        for (hash, block) in active_found {
            result.insert(
                hash,
                ImmutableBlock::new(block, self.upgrade_fn.clone(), Some(self.metrics.clone())),
            );
        }

        // 2. Build remaining hashes set
        let remaining: Vec<SequenceHash> = seq_hashes
            .iter()
            .filter(|h| !result.contains_key(h))
            .copied()
            .collect();

        // 3. Scan inactive pool for remaining
        if !remaining.is_empty() {
            let inactive_found = self.store.scan_inactive_blocks(&remaining, touch);
            for (hash, block) in inactive_found {
                result.insert(
                    hash,
                    ImmutableBlock::new(block, self.upgrade_fn.clone(), Some(self.metrics.clone())),
                );
            }
        }

        self.metrics.inc_scan_blocks_returned(result.len() as u64);

        result
    }

    /// Find currently-active registered blocks for the given hashes via the
    /// registry's weak references. Stops on first miss.
    ///
    /// Replaces the deleted `ActivePool::find_matches`.
    fn find_active_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<dyn RegisteredBlock<T>>> {
        let return_fn = self.store.inactive_return_fn();
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(handle) = self.block_registry.match_sequence_hash(*hash, touch) {
                if let Some(block) = handle.try_get_block::<T>(return_fn.clone()) {
                    matches.push(block);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        matches
    }

    /// Scan-style version of [`find_active_matches`] — does not stop on miss.
    fn scan_active_matches(
        &self,
        hashes: &[SequenceHash],
    ) -> Vec<(SequenceHash, Arc<dyn RegisteredBlock<T>>)> {
        let return_fn = self.store.inactive_return_fn();
        hashes
            .iter()
            .filter_map(|hash| {
                self.block_registry
                    .match_sequence_hash(*hash, false)
                    .and_then(|handle| {
                        handle
                            .try_get_block::<T>(return_fn.clone())
                            .map(|block| (*hash, block))
                    })
            })
            .collect()
    }

    /// Total number of blocks managed (constant after construction).
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Blocks available for allocation (reset + inactive pools).
    pub fn available_blocks(&self) -> usize {
        self.store.reset_len() + self.store.inactive_len()
    }

    /// Tokens per block (constant after construction).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Current duplication policy.
    pub fn duplication_policy(&self) -> &BlockDuplicationPolicy {
        &self.duplication_policy
    }

    /// Reference to the shared block registry.
    pub fn block_registry(&self) -> &BlockRegistry {
        &self.block_registry
    }

    /// Reference to the block pool metrics.
    pub fn metrics(&self) -> &Arc<BlockPoolMetrics> {
        &self.metrics
    }
}
