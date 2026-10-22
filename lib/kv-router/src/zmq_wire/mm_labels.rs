// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exclusion of multimodal values the request side cannot reproduce.
//!
//! Since sgl-project/sglang#43043, SGLang fills each multimodal item with pad
//! values derived from its own identity, so event tokens no longer equal
//! Dynamo's routing pads. A store is cut at the first block whose hashed tokens
//! hold such a value. That block and every block chained after it are
//! excluded, including later stores under an excluded parent, and their
//! removals never reach the index. Text-prefix stores, removals and clears
//! still apply.
//!
//! Every source excludes values that did not fit in `u32`: they all narrow to
//! one sentinel, so indexing them would give different images one local hash.
//! A source flagged as engine-owned multimodal identity also excludes every
//! value at or above `MM_PAD_SHIFT_VALUE`. Legacy sources keep those values,
//! because their pads are Dynamo's own routing values.

use rustc_hash::FxHashSet;

use crate::protocols::{
    KvCacheEventData, MM_PAD_SHIFT_VALUE, PlacementEvent, StorageTier, WorkerWithDpRank,
};

use super::types::{BlockHashValue, RawKvEvent, WIDE_TOKEN_SENTINEL};

/// Bound on the exclusion set. Losing entries never indexes an excluded block:
/// its descendants then fail the indexer's parent lookup.
const DEFAULT_EXCLUDED_CAPACITY: usize = 1 << 20;

type Key = (WorkerWithDpRank, StorageTier, u64);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineMmStats {
    /// Blocks dropped because their hashed tokens hold multimodal values.
    pub excluded_unmapped: u64,
    /// Blocks dropped because their parent was dropped.
    pub excluded_parent: u64,
    /// Exclusion entries not recorded because the set was full.
    pub excluded_overflow: u64,
}

#[derive(Debug, Clone)]
pub(super) struct EngineMmState {
    engine_owned: bool,
    excluded: FxHashSet<Key>,
    excluded_capacity: usize,
    pub(super) stats: EngineMmStats,
}

impl EngineMmState {
    pub(super) fn new(engine_owned: bool) -> Self {
        Self {
            engine_owned,
            excluded: FxHashSet::default(),
            excluded_capacity: DEFAULT_EXCLUDED_CAPACITY,
            stats: EngineMmStats::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn with_excluded_capacity(engine_owned: bool, excluded_capacity: usize) -> Self {
        Self {
            excluded_capacity,
            ..Self::new(engine_owned)
        }
    }

    pub(super) fn engine_owned(&self) -> bool {
        self.engine_owned
    }

    fn is_multimodal(&self, token: u32) -> bool {
        if self.engine_owned {
            u64::from(token) >= MM_PAD_SHIFT_VALUE
        } else {
            token == WIDE_TOKEN_SENTINEL
        }
    }

    /// Cut the store at the first block whose hashed tokens hold a multimodal
    /// value. Returns `false` when nothing in the store may be indexed.
    pub(super) fn prepare_store(
        &mut self,
        raw: &mut RawKvEvent,
        worker: WorkerWithDpRank,
        tier: StorageTier,
    ) -> bool {
        let RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            is_eagle,
            ..
        } = raw
        else {
            return true;
        };

        if !self.excluded.is_empty()
            && let Some(parent) = parent_block_hash.map(BlockHashValue::into_u64)
            && self.excluded.contains(&(worker, tier, parent))
        {
            self.stats.excluded_parent += block_hashes.len() as u64;
            for hash in block_hashes.iter() {
                self.exclude((worker, tier, hash.into_u64()));
            }
            return false;
        }

        // Conversion hashes each block's tokens and, for Eagle, also the next
        // block's first token. Classify exactly those windows.
        let block_len = *block_size;
        let eagle = is_eagle.unwrap_or(false);
        let window = block_len + usize::from(eagle);
        let cut = (0..block_hashes.len()).find(|&block_idx| {
            let start = block_idx * block_len;
            token_ids
                .get(start..start + window)
                .is_some_and(|tokens| tokens.iter().any(|&token| self.is_multimodal(token)))
        });

        if let Some(block_idx) = cut {
            self.stats.excluded_unmapped += (block_hashes.len() - block_idx) as u64;
            for hash in block_hashes.drain(block_idx..) {
                self.exclude((worker, tier, hash.into_u64()));
            }
            // Keep the overlap token the last surviving Eagle block hashes.
            token_ids.truncate(block_idx * block_len + usize::from(eagle && block_idx > 0));
        }

        !block_hashes.is_empty()
    }

    /// Drop removed engine ids that were excluded: they never reached the index.
    pub(super) fn filter_remove(
        &mut self,
        event: &mut PlacementEvent,
        worker: WorkerWithDpRank,
        tier: StorageTier,
    ) {
        if self.excluded.is_empty() {
            return;
        }
        let KvCacheEventData::Removed(remove) = &mut event.event.data else {
            return;
        };
        remove
            .block_hashes
            .retain(|hash| !self.excluded.remove(&(worker, tier, hash.0)));
    }

    /// Forget all state for a worker after the engine cleared its cache.
    pub(super) fn clear(&mut self, worker: WorkerWithDpRank) {
        self.excluded.retain(|(known, _, _)| *known != worker);
    }

    fn exclude(&mut self, key: Key) {
        if self.excluded.len() < self.excluded_capacity {
            self.excluded.insert(key);
        } else if !self.excluded.contains(&key) {
            self.stats.excluded_overflow += 1;
        }
    }
}
