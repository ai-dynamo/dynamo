// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-owned multimodal identity for SGLang sources with wide pad values.
//!
//! Since sgl-project/sglang#43043, SGLang fills each multimodal item with pad
//! values derived from its own identity, so event tokens no longer equal
//! Dynamo's routing pads and the request side cannot reproduce them. This
//! module cuts a store at the first block holding such a value. That block and
//! every block chained after it are excluded, including later stores under an
//! excluded parent, and their removals never reach the index. Text-prefix
//! stores, removals and clears still apply.

use rustc_hash::FxHashSet;

use crate::protocols::{
    KvCacheEventData, MM_PAD_SHIFT_VALUE, PlacementEvent, StorageTier, WorkerWithDpRank,
};

use super::types::{BlockHashValue, RawKvEvent, WIDE_TOKEN_SENTINEL};

/// Bound on the advisory exclusion set. Losing entries never indexes an
/// unmapped block: its descendants then fail the indexer's parent lookup.
const DEFAULT_EXCLUDED_CAPACITY: usize = 1 << 20;

type Key = (WorkerWithDpRank, StorageTier, u64);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineMmStats {
    /// Blocks dropped because they hold multimodal values.
    pub excluded_unmapped: u64,
    /// Blocks dropped because their parent was dropped.
    pub excluded_parent: u64,
    /// Exclusion entries not recorded because the set was full.
    pub excluded_overflow: u64,
}

#[derive(Debug, Clone)]
pub(super) struct EngineMmState {
    excluded: FxHashSet<Key>,
    excluded_capacity: usize,
    pub(super) stats: EngineMmStats,
}

impl Default for EngineMmState {
    fn default() -> Self {
        Self {
            excluded: FxHashSet::default(),
            excluded_capacity: DEFAULT_EXCLUDED_CAPACITY,
            stats: EngineMmStats::default(),
        }
    }
}

impl EngineMmState {
    #[cfg(test)]
    pub(super) fn with_excluded_capacity(excluded_capacity: usize) -> Self {
        Self {
            excluded_capacity,
            ..Self::default()
        }
    }

    /// Cut the store at the first block holding a multimodal value. Returns
    /// `false` when nothing in the store may be indexed.
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

        if let Some(parent) = parent_block_hash.map(BlockHashValue::into_u64)
            && self.excluded.contains(&(worker, tier, parent))
        {
            self.stats.excluded_parent += block_hashes.len() as u64;
            for hash in block_hashes.iter() {
                self.exclude((worker, tier, hash.into_u64()));
            }
            return false;
        }

        let block_len = *block_size;
        let eagle = is_eagle.unwrap_or(false);
        let mut cut = None;
        for block_idx in 0..block_hashes.len() {
            let start = block_idx * block_len;
            let end = start + block_len;
            if end > token_ids.len() {
                // Conversion drops short blocks; nothing to classify here.
                break;
            }
            if token_ids[start..end]
                .iter()
                .any(|&token| u64::from(token) >= MM_PAD_SHIFT_VALUE)
            {
                cut = Some(block_idx);
                break;
            }
        }

        if let Some(block_idx) = cut {
            self.stats.excluded_unmapped += (block_hashes.len() - block_idx) as u64;
            for hash in block_hashes.drain(block_idx..) {
                self.exclude((worker, tier, hash.into_u64()));
            }
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

/// For sources not flagged as engine-owned multimodal identity: cut a store at
/// the first block holding a value that did not fit in `u32`. All such values
/// narrow to one sentinel, so indexing them would make different images share
/// a local hash.
pub(super) fn truncate_at_wide_tokens(raw: &mut RawKvEvent) -> bool {
    let RawKvEvent::BlockStored {
        block_hashes,
        token_ids,
        block_size,
        ..
    } = raw
    else {
        return true;
    };
    let Some(position) = token_ids
        .iter()
        .position(|&token| token == WIDE_TOKEN_SENTINEL)
    else {
        return true;
    };
    let block_len = (*block_size).max(1);
    let block_idx = position / block_len;
    block_hashes.truncate(block_idx);
    token_ids.truncate(block_idx * block_len);
    !block_hashes.is_empty()
}
