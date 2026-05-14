// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block lifecycle orchestration over the unified [`BlockStore`].
//!
//! [`BlockManager`] owns a single [`BlockStore`] and the [`BlockRegistry`].
//! All pool transitions go through the store's single mutex; the manager
//! adds the registry coordination, allocation eviction policy, and metrics.

mod builder;

#[cfg(test)]
mod tests;

pub use builder::{
    BlockManagerBuilderError, BlockManagerConfigBuilder, BlockManagerResetError,
    FrequencyTrackingCapacity, InactiveBackendConfig,
};

use std::collections::HashMap;
use std::sync::Arc;

use crate::blocks::{BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock};
use crate::metrics::BlockPoolMetrics;
use crate::pools::{BlockDuplicationPolicy, BlockStore, SequenceHash};
use crate::registry::BlockRegistry;

/// Manages the full block lifecycle over the unified [`BlockStore`].
///
/// Construct via [`BlockManager::builder()`].
pub struct BlockManager<T: BlockMetadata> {
    pub(crate) store: Arc<BlockStore<T>>,
    pub(crate) block_registry: BlockRegistry,
    pub(crate) duplication_policy: BlockDuplicationPolicy,
    pub(crate) total_blocks: usize,
    pub(crate) block_size: usize,
    pub(crate) metrics: Arc<BlockPoolMetrics>,
}

impl<T: BlockMetadata + Sync> BlockManager<T> {
    /// Create a new builder for `BlockManager`.
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    /// Stable, process-unique identifier for this manager's underlying
    /// [`BlockStore`](crate::pools::BlockStore). See [`crate::ManagerId`].
    /// Cheap (one field load via the store).
    ///
    /// Together with a [`BlockId`](crate::BlockId) this names a specific
    /// physical pool slot — the disambiguating runtime address that
    /// downstream consumers need after the policy parameter `T` has been
    /// type-erased through [`crate::LifecyclePinRef`].
    pub fn id(&self) -> crate::ManagerId {
        self.store.id()
    }

    /// Allocate `count` mutable blocks, drawing first from the reset pool
    /// and then evicting from the inactive pool if needed.
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
        self.store.allocate_atomic(count)
    }

    /// Drain the inactive pool, returning all blocks to the reset pool.
    pub fn reset_inactive_pool(&self) -> Result<(), BlockManagerResetError> {
        let blocks = self.store.drain_inactive_to_mutable();
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

    /// Register a batch of completed blocks.
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
        let inner = handle.register_block(block, self.duplication_policy, &self.store);
        ImmutableBlock::from_inner(inner)
    }

    /// Linear prefix match: walks `seq_hash` left-to-right, stopping on
    /// first miss. Checks the active pool first (via the registry's weak
    /// refs), then the inactive pool for the remaining hashes.
    ///
    /// Single-hash fast path: probes the first hash against active and
    /// then inactive, only allocating the result `Vec` once at least
    /// one hit is confirmed. The all-miss path is allocation-free.
    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        self.metrics
            .inc_match_hashes_requested(seq_hash.len() as u64);

        tracing::debug!(
            num_hashes = seq_hash.len(),
            inactive_pool_len = self.store.inactive_len(),
            "match_blocks called"
        );

        let Some((&first_hash, rest)) = seq_hash.split_first() else {
            self.metrics.inc_match_blocks_returned(0);
            tracing::debug!(total_matched = 0, "match_blocks result");
            return Vec::new();
        };

        // Active path: single-hash probe before allocating the Vec.
        if let Some(first_active) = self.find_active_match(first_hash, true) {
            let mut matched: Vec<ImmutableBlock<T>> = Vec::with_capacity(seq_hash.len());
            matched.push(ImmutableBlock::from_inner(first_active));
            if !rest.is_empty() {
                matched.extend(
                    self.find_active_matches(rest, true)
                        .into_iter()
                        .map(ImmutableBlock::from_inner),
                );
            }
            let active_matched = matched.len();
            tracing::debug!(active_matched, "Matched from active pool");

            let remaining_hashes = &seq_hash[matched.len()..];
            if !remaining_hashes.is_empty() {
                let inactive_found = self.store.find_inactive_primaries(remaining_hashes, true);
                let inactive_matched = inactive_found.len();
                tracing::debug!(
                    remaining_to_check = remaining_hashes.len(),
                    inactive_matched,
                    "Matched from inactive pool"
                );
                matched.extend(inactive_found.into_iter().map(ImmutableBlock::from_inner));
            }

            self.metrics.inc_match_blocks_returned(matched.len() as u64);
            tracing::debug!(total_matched = matched.len(), "match_blocks result");
            tracing::trace!(matched = ?matched, "matched blocks");
            return matched;
        }

        // Active missed on the first hash. Try inactive for the full
        // slice; `find_inactive_primaries` has its own first-hash
        // fast-path that returns empty without allocating when the
        // head misses.
        let inactive_found = self.store.find_inactive_primaries(seq_hash, true);
        if inactive_found.is_empty() {
            self.metrics.inc_match_blocks_returned(0);
            tracing::debug!(active_matched = 0, "Matched from active pool");
            tracing::debug!(total_matched = 0, "match_blocks result");
            return Vec::new();
        }

        let mut matched: Vec<ImmutableBlock<T>> = Vec::with_capacity(seq_hash.len());
        let inactive_matched = inactive_found.len();
        matched.extend(inactive_found.into_iter().map(ImmutableBlock::from_inner));
        tracing::debug!(active_matched = 0, "Matched from active pool");
        tracing::debug!(
            remaining_to_check = seq_hash.len(),
            inactive_matched,
            "Matched from inactive pool"
        );

        self.metrics.inc_match_blocks_returned(matched.len() as u64);
        tracing::debug!(total_matched = matched.len(), "match_blocks result");
        tracing::trace!(matched = ?matched, "matched blocks");
        matched
    }

    /// Scatter-gather scan: finds all blocks matching any hash, without
    /// stopping on misses.
    pub fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>> {
        self.metrics
            .inc_scan_hashes_requested(seq_hashes.len() as u64);

        let mut result = HashMap::new();

        let active_found = self.scan_active_matches(seq_hashes, touch);
        for (hash, inner) in active_found {
            result.insert(hash, ImmutableBlock::from_inner(inner));
        }

        let remaining: Vec<SequenceHash> = seq_hashes
            .iter()
            .filter(|h| !result.contains_key(h))
            .copied()
            .collect();

        if !remaining.is_empty() {
            let inactive_found = self.store.scan_inactive_primaries(&remaining, touch);
            for (hash, inner) in inactive_found {
                result.insert(hash, ImmutableBlock::from_inner(inner));
            }
        }

        self.metrics.inc_scan_blocks_returned(result.len() as u64);

        result
    }

    /// Look up currently-active registered blocks by sequence hash via the
    /// registry's stored Weak references. Stops on first miss.
    fn find_active_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<crate::blocks::ImmutableBlockInner<T>>> {
        let mut matches = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some(inner) = self.find_active_match(*hash, touch) {
                matches.push(inner);
            } else {
                break;
            }
        }
        matches
    }

    /// Single-hash variant of [`find_active_matches`] used by the
    /// `match_blocks` first-hash fast path to avoid allocating a
    /// one-element slice and result Vec.
    fn find_active_match(
        &self,
        hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<crate::blocks::ImmutableBlockInner<T>>> {
        let handle = self.block_registry.match_sequence_hash(hash, touch)?;
        handle.try_get_inner::<T>(&self.store, touch)
    }

    /// Scan-style version of `find_active_matches` — does not stop on miss.
    fn scan_active_matches(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<crate::blocks::ImmutableBlockInner<T>>)> {
        hashes
            .iter()
            .filter_map(|hash| {
                self.block_registry
                    .match_sequence_hash(*hash, touch)
                    .and_then(|handle| {
                        handle
                            .try_get_inner::<T>(&self.store, touch)
                            .map(|inner| (*hash, inner))
                    })
            })
            .collect()
    }

    /// Total number of blocks managed (constant after construction).
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Blocks available for allocation (reset + inactive pools).
    ///
    /// Reads both pool sizes under a single store-lock acquisition so the
    /// returned value is a coherent snapshot, never an over- or under-count
    /// produced by a concurrent reset↔inactive transition.
    pub fn available_blocks(&self) -> usize {
        self.store.available_len()
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

    /// Test-only accessor for the underlying [`BlockStore`]. Used to
    /// reach test hooks like `BlockStore::pause_release_primary` from
    /// race-window tests.
    #[cfg(test)]
    pub(crate) fn store_for_test(&self) -> &Arc<BlockStore<T>> {
        &self.store
    }
}
