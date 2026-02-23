// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block lifecycle orchestration across reset, active, and inactive pools.
//!
//! [`BlockManager`] is the top-level owner of the three pool tiers and the
//! block registry. It exposes allocation, registration, matching, and scanning
//! operations while keeping all pool transitions behind a single API surface.
//!
//! Construction uses a builder pattern — see [`BlockManagerConfigBuilder`].
//!
//! # Re-exported configuration types
//!
//! - [`FrequencyTrackingCapacity`] — TinyLFU tracker sizing
//! - [`InactiveBackendConfig`] — inactive pool backend selection
//! - [`BlockManagerBuilderError`] / [`BlockManagerResetError`] — error types

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

use crate::blocks::{BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock, UpgradeFn};
use crate::metrics::BlockPoolMetrics;
use crate::pools::{ActivePool, BlockDuplicationPolicy, InactivePool, ResetPool, SequenceHash};
use crate::registry::BlockRegistry;

/// Manages the full block lifecycle across three pool tiers:
/// reset (free), active (in-use), and inactive (cached, evictable).
///
/// Thread-safe: allocation is serialised via an internal [`Mutex`]; individual
/// pools use their own internal locking.
///
/// Construct via [`BlockManager::builder()`].
pub struct BlockManager<T: BlockMetadata> {
    reset_pool: ResetPool<T>,
    active_pool: ActivePool<T>,
    inactive_pool: InactivePool<T>,
    block_registry: BlockRegistry,
    duplication_policy: BlockDuplicationPolicy,
    upgrade_fn: UpgradeFn<T>,
    allocate_mutex: Mutex<()>,
    total_blocks: usize,
    block_size: usize,
    metrics: Arc<BlockPoolMetrics>,
}

impl<T: BlockMetadata> BlockManager<T> {
    /// Create a new builder for BlockManager.
    ///
    /// # Example
    /// ```ignore
    /// let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
    /// let registry = BlockRegistry::builder().frequency_tracker(tracker).build();
    ///
    /// let manager = BlockManager::builder()
    ///     .block_count(1000)
    ///     .registry(registry)
    ///     .with_multi_lru_backend()
    ///     .build()?;
    /// ```
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    /// Allocate `count` mutable blocks, drawing first from the reset pool
    /// then evicting from the inactive pool if needed.
    ///
    /// Returns `None` if fewer than `count` blocks are available across both pools.
    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        let _guard = self.allocate_mutex.lock();
        let from_reset = self.reset_pool.allocate_blocks(count);
        let from_reset_count = from_reset.len();
        let mut blocks = from_reset;

        let remaining_needed = count - blocks.len();
        match self.inactive_pool.allocate_blocks(remaining_needed) {
            Some(remaining) => {
                let eviction_count = remaining.len() as u64;
                blocks.extend(remaining);

                self.metrics.inc_allocations(blocks.len() as u64);
                self.metrics
                    .inc_allocations_from_reset(from_reset_count as u64);
                self.metrics.inc_evictions(eviction_count);

                Some(blocks)
            }
            None => None,
        }
    }

    /// Drain the inactive pool, returning all blocks to the reset pool.
    ///
    /// 1. Acquires the inactive pool lock and allocates all blocks.
    /// 2. Releases the lock.
    /// 3. Drops the allocated blocks (RAII returns them to reset).
    /// 4. Verifies the reset pool contains the expected total.
    ///
    /// Returns an error under contention when blocks are in active use.
    pub fn reset_inactive_pool(&self) -> Result<(), BlockManagerResetError> {
        // 1. Allocate all blocks from inactive pool (acquires lock internally)
        let blocks = self.inactive_pool.allocate_all_blocks();

        // 2. Drop blocks - RAII returns them to reset pool
        drop(blocks);

        // 3. Verify block count (may fail under contention - that's OK)
        let reset_count = self.reset_pool.len();
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
    ///
    /// Deduplication is governed by the configured [`BlockDuplicationPolicy`].
    pub fn register_block(&self, block: CompleteBlock<T>) -> ImmutableBlock<T> {
        self.metrics.inc_registrations();
        let handle = self
            .block_registry
            .register_sequence_hash(block.sequence_hash());
        let registered_block = handle.register_block(
            block,
            self.duplication_policy,
            &self.inactive_pool,
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
    /// Checks the active pool first, then the inactive pool for remaining hashes.
    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        self.metrics
            .inc_match_hashes_requested(seq_hash.len() as u64);

        tracing::debug!(
            num_hashes = seq_hash.len(),
            inactive_pool_len = self.inactive_pool.len(),
            "match_blocks called"
        );

        // First try to match against active blocks
        let mut matched: Vec<ImmutableBlock<T>> = Vec::with_capacity(seq_hash.len());
        matched.extend(
            self.active_pool
                .find_matches(seq_hash, true)
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
            let inactive_found: Vec<_> = self.inactive_pool.find_blocks(remaining_hashes, true);
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
    ///
    /// Returns a map of found hashes to immutable handles.
    pub fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>> {
        self.metrics
            .inc_scan_hashes_requested(seq_hashes.len() as u64);

        let mut result = HashMap::new();

        // 1. Check active pool for all hashes (read-only, no touch needed)
        let active_found = self.active_pool.scan_matches(seq_hashes);
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

        // 3. Scan inactive pool for remaining (acquires blocks, may touch)
        if !remaining.is_empty() {
            let inactive_found = self.inactive_pool.scan_blocks(&remaining, touch);
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

    /// Total number of blocks managed (constant after construction).
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Blocks available for allocation (reset + inactive pools).
    pub fn available_blocks(&self) -> usize {
        self.reset_pool.len() + self.inactive_pool.len()
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
