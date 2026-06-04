// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    PlacementEvent, RouterEvent, StorageTier,
};

/// Reference-counting filter that deduplicates KV cache events.
///
/// vLLM can emit multiple store/remove events for the same block hash.
/// Refcounts are tracked **per DP rank** because identical block hashes
/// on different ranks represent independent blocks.
///
/// - **Store**: always passes through; increments refcount for the rank.
/// - **Remove**: only passes through when refcount decrements to 0.
/// - **Cleared**: resets refcounts for all ranks.
pub(super) struct EventDedupFilter {
    /// Per-(dp_rank, storage_tier) refcounts.
    per_rank_tier: HashMap<(u32, StorageTier), HashMap<ExternalSequenceBlockHash, usize>>,
}

impl EventDedupFilter {
    pub(super) fn new() -> Self {
        Self {
            per_rank_tier: HashMap::new(),
        }
    }

    /// Track a store event. Increments refcount for each block hash on the
    /// given (DP rank, storage tier). Stores always pass through — this only
    /// updates bookkeeping.
    pub(super) fn track_store(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        data: &KvCacheStoreData,
    ) {
        let refcounts = self
            .per_rank_tier
            .entry((dp_rank, storage_tier))
            .or_default();
        for block in &data.blocks {
            *refcounts.entry(block.block_hash).or_insert(0) += 1;
        }
    }

    /// Filter a remove event. Retains only block hashes whose refcount on the
    /// given (DP rank, storage tier) decrements to 0 (removing them from the
    /// map). Returns `None` if no hashes survive filtering.
    pub(super) fn filter_remove(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        mut data: KvCacheRemoveData,
    ) -> Option<KvCacheRemoveData> {
        let refcounts = self
            .per_rank_tier
            .entry((dp_rank, storage_tier))
            .or_default();
        data.block_hashes.retain(|hash| {
            match refcounts.entry(*hash) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() -= 1;
                    if *entry.get() == 0 {
                        entry.remove();
                        true // refcount hit 0 -> pass through
                    } else {
                        false // still has references -> filter out
                    }
                }
                Entry::Vacant(_) => {
                    true // not tracked -> pass through defensively
                }
            }
        });
        if data.block_hashes.is_empty() {
            None
        } else {
            Some(data)
        }
    }

    /// Clear refcounts for all DP ranks and tiers. A `Cleared` event from any
    /// rank causes the indexer to wipe all blocks for the entire worker, so we
    /// must reset all refcounts to stay consistent.
    pub(super) fn clear(&mut self) {
        self.per_rank_tier.clear();
    }
}

/// Per-worker dedup for the direct (non-NATS) per-pod ZMQ KV-event path.
///
/// vLLM can emit duplicate store/remove events; applying them straight to the
/// indexer corrupts block refcounts (a remove could fire before the worker
/// actually evicts the block). This wraps [`EventDedupFilter`] so the per-pod
/// path (`KvRouter::register_worker_kv_events`) gets the same refcount-correct
/// dedup the batched NATS path already applies.
pub(crate) struct PerWorkerDedup {
    filter: EventDedupFilter,
}

impl PerWorkerDedup {
    pub(crate) fn new() -> Self {
        Self {
            filter: EventDedupFilter::new(),
        }
    }

    /// Apply dedup to a decoded placement event and return the `RouterEvent`
    /// to index, or `None` when the event is fully filtered (duplicate remove)
    /// or carries no local-worker placement.
    pub(crate) fn process(&mut self, mut event: PlacementEvent) -> Option<RouterEvent> {
        let dp_rank = event.event.dp_rank;
        let tier = event.placement.tier;
        // Take the data out to dedup (`filter_remove` consumes it), then restore.
        let data = std::mem::replace(&mut event.event.data, KvCacheEventData::Cleared);
        let new_data = match data {
            KvCacheEventData::Stored(store) => {
                self.filter.track_store(dp_rank, tier, &store);
                KvCacheEventData::Stored(store)
            }
            KvCacheEventData::Removed(remove) => {
                KvCacheEventData::Removed(self.filter.filter_remove(dp_rank, tier, remove)?)
            }
            KvCacheEventData::Cleared => {
                self.filter.clear();
                KvCacheEventData::Cleared
            }
        };
        event.event.data = new_data;
        event.into_router_event()
    }
}
