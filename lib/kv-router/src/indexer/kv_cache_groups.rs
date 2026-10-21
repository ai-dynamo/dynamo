// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-group residency for hybrid (attention + recurrent state) models.
//!
//! vLLM publishes one `BlockStored` per KV cache group at the same positional
//! block hash. For the main-attention group that hash names a reusable block;
//! for a recurrent (Mamba / GDN / linear-attention) group it names the state
//! checkpoint at that boundary. A hybrid engine can resume a prefix only at a
//! boundary where every group has a residency, so an attention-only match
//! overestimates the reusable prefix and over-sizes KV transfer hints
//! (ai-dynamo/dynamo#16794).
//!
//! Recurrent-group events are tagged at ingress (`RouterEvent::kv_cache_group`)
//! and recorded here instead of in the attention radix tree. After a lookup,
//! [`KvCacheGroupResidency::cut`] shortens each hybrid worker's match to the
//! rightmost matched position at which all of its recurrent groups hold a
//! state, and the KV transfer chain is truncated to the same point. Workers
//! that never published a recurrent-group event are left untouched.

use std::sync::RwLock;

use rustc_hash::FxHashMap;

use super::lower_tier::{LowerTierContinuation, LowerTierMatchDetails};
use super::types::MatchDetails;
use crate::kv_hints::KvTransferCandidateSource;
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, RouterEvent, StorageTier, WorkerId,
    WorkerWithDpRank,
};

type GroupPresence = FxHashMap<u32, FxHashMap<u64, u8>>;

fn tier_bit(tier: StorageTier) -> u8 {
    match tier {
        StorageTier::Device => 1,
        StorageTier::HostPinned => 2,
        StorageTier::Disk => 4,
        StorageTier::External => 8,
    }
}

/// Which positional block hashes hold a recurrent state, per worker and group,
/// across storage tiers (a state is loadable from any tier).
#[derive(Debug, Default)]
pub struct KvCacheGroupResidency {
    workers: RwLock<FxHashMap<WorkerWithDpRank, GroupPresence>>,
}

impl KvCacheGroupResidency {
    pub fn new() -> Self {
        Self::default()
    }

    /// True once any worker has published a recurrent-group event. Lookups
    /// then retain the matched hash chain so matches can be cut.
    pub fn is_active(&self) -> bool {
        !self.workers.read().unwrap().is_empty()
    }

    /// Apply a tagged event. Returns false if the event carried no group tag.
    pub fn apply(&self, event: &RouterEvent) -> bool {
        let Some(tag) = event.kv_cache_group else {
            return false;
        };
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let bit = tier_bit(event.storage_tier);
        let mut workers = self.workers.write().unwrap();
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                let group = workers
                    .entry(worker)
                    .or_default()
                    .entry(tag.group_idx)
                    .or_default();
                for block in &data.blocks {
                    *group.entry(block.block_hash.0).or_insert(0) |= bit;
                }
            }
            KvCacheEventData::Removed(data) => {
                if let Some(group) = workers
                    .get_mut(&worker)
                    .and_then(|groups| groups.get_mut(&tag.group_idx))
                {
                    for hash in &data.block_hashes {
                        if let Some(mask) = group.get_mut(&hash.0) {
                            *mask &= !bit;
                            if *mask == 0 {
                                group.remove(&hash.0);
                            }
                        }
                    }
                }
            }
            KvCacheEventData::Cleared => {
                // A group-tagged clear is not produced by vLLM; treat it as a
                // worker-wide clear of recurrent presence (the attention tree
                // handles its own untagged clear).
                workers.remove(&worker);
            }
        }
        true
    }

    /// Forget recurrent presence for a worker rank (untagged clear, or the
    /// worker left).
    pub fn clear_worker(&self, worker: WorkerWithDpRank) {
        self.workers.write().unwrap().remove(&worker);
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.workers
            .write()
            .unwrap()
            .retain(|worker, _| worker.worker_id != worker_id);
    }

    /// Rightmost `p <= blocks` such that every recurrent group of `worker`
    /// holds a state at `chain[p - 1]`. `None` means the worker has no
    /// recurrent groups and the attention match stands.
    pub fn reusable_blocks(
        &self,
        worker: WorkerWithDpRank,
        chain: &[ExternalSequenceBlockHash],
        blocks: usize,
    ) -> Option<usize> {
        let workers = self.workers.read().unwrap();
        let groups = workers.get(&worker)?;
        if groups.is_empty() {
            return None;
        }
        let blocks = blocks.min(chain.len());
        Some(
            (1..=blocks)
                .rev()
                .find(|&p| {
                    let hash = chain[p - 1].0;
                    groups.values().all(|group| group.contains_key(&hash))
                })
                .unwrap_or(0),
        )
    }

    /// Cut device-tier matches to each hybrid worker's last complete boundary.
    /// Needs the retained root-aligned chain (`kv_transfer_candidates`); without
    /// it the scores are left as they are.
    pub fn cut_device(&self, details: &mut MatchDetails) {
        if !self.is_active() {
            return;
        }
        let Some(chain) = details
            .kv_transfer_candidates
            .as_ref()
            .map(|candidates| candidates.block_hashes.clone())
        else {
            return;
        };
        let mut cuts: Vec<(WorkerWithDpRank, usize)> = Vec::new();
        for (worker, score) in &details.overlap_scores.scores {
            let blocks = *score as usize;
            if let Some(p) = self.reusable_blocks(*worker, &chain, blocks)
                && p < blocks
            {
                cuts.push((*worker, p));
            }
        }
        if cuts.is_empty() {
            return;
        }
        for (worker, p) in &cuts {
            if *p == 0 {
                details.overlap_scores.scores.remove(worker);
                details.last_matched_hashes.remove(worker);
            } else {
                details.overlap_scores.scores.insert(*worker, *p as u32);
                details.last_matched_hashes.insert(*worker, chain[*p - 1]);
            }
        }
        let cut_sources: FxHashMap<KvTransferCandidateSource, usize> = cuts
            .iter()
            .map(|(worker, p)| (KvTransferCandidateSource::from(*worker), *p))
            .collect();
        if let Some(candidates) = details.kv_transfer_candidates.as_mut() {
            candidates
                .owner_prefix_blocks
                .retain_mut(|(source, blocks)| match cut_sources.get(source) {
                    Some(0) => false,
                    Some(p) => {
                        *blocks = *p;
                        true
                    }
                    None => true,
                });
            let max_blocks = candidates
                .owner_prefix_blocks
                .iter()
                .map(|(_, blocks)| *blocks)
                .max()
                .unwrap_or(0);
            candidates.block_hashes.truncate(max_blocks);
            if candidates.owner_prefix_blocks.is_empty() {
                details.kv_transfer_candidates = None;
            }
        }
    }

    /// Cut one lower tier's hits the same way. `device_chain` is the retained
    /// device chain, used when the tier did not retain its own.
    pub fn cut_lower_tier(
        &self,
        details: &mut LowerTierMatchDetails,
        device_chain: &[ExternalSequenceBlockHash],
    ) {
        if !self.is_active() {
            return;
        }
        let chain: Vec<ExternalSequenceBlockHash> = match details.kv_transfer_candidates.as_ref() {
            Some(candidates) if candidates.block_hashes.len() >= device_chain.len() => {
                candidates.block_hashes.clone()
            }
            _ => device_chain.to_vec(),
        };
        if chain.is_empty() {
            return;
        }
        // `hits` counts blocks matched in this tier beyond the continuation
        // start; the continuation's `start_pos` is the absolute end position.
        let mut cuts: Vec<(WorkerWithDpRank, usize)> = Vec::new();
        for (worker, hits) in &details.hits {
            let Some(end) = details
                .next_continuations
                .get(worker)
                .map(|continuation| continuation.start_pos)
            else {
                continue;
            };
            let start = end - *hits;
            if end > chain.len() {
                // The chain does not cover this tier's extension (device
                // candidates were truncated below it): nothing to verify.
                continue;
            }
            if let Some(p) = self.reusable_blocks(*worker, &chain, end)
                && p < end
            {
                cuts.push((*worker, p.max(start)));
            }
        }
        if cuts.is_empty() {
            return;
        }
        for (worker, p) in &cuts {
            let end = details.next_continuations[worker].start_pos;
            let hits = details.hits[worker];
            details.hits.insert(*worker, hits - (end - *p));
            let continuation = if *p == 0 {
                LowerTierContinuation::from_root(0)
            } else {
                LowerTierContinuation::new(*p, chain[*p - 1])
            };
            details.next_continuations.insert(*worker, continuation);
        }
        let cut_sources: FxHashMap<KvTransferCandidateSource, usize> = cuts
            .iter()
            .map(|(worker, p)| (KvTransferCandidateSource::from(*worker), *p))
            .collect();
        if let Some(candidates) = details.kv_transfer_candidates.as_mut() {
            candidates
                .owner_prefix_blocks
                .retain_mut(|(source, blocks)| match cut_sources.get(source) {
                    Some(0) => false,
                    Some(p) => {
                        *blocks = *p;
                        true
                    }
                    None => true,
                });
            let max_blocks = candidates
                .owner_prefix_blocks
                .iter()
                .map(|(_, blocks)| *blocks)
                .max()
                .unwrap_or(0);
            candidates.block_hashes.truncate(max_blocks);
            if candidates.owner_prefix_blocks.is_empty() {
                details.kv_transfer_candidates = None;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn groups_for(&self, worker: WorkerWithDpRank) -> rustc_hash::FxHashSet<u32> {
        self.workers
            .read()
            .unwrap()
            .get(&worker)
            .map(|groups| groups.keys().copied().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_hints::KvTransferCandidates;
    use crate::protocols::{
        KvCacheEvent, KvCacheGroupKind, KvCacheGroupTag, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash,
    };

    fn tag(group_idx: u32) -> KvCacheGroupTag {
        KvCacheGroupTag {
            group_idx,
            kind: KvCacheGroupKind::Mamba,
        }
    }

    fn stored(
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group_idx: u32,
        hashes: &[u64],
    ) -> RouterEvent {
        let blocks = hashes
            .iter()
            .map(|&h| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(h),
                tokens_hash: LocalBlockHash(h),
                mm_extra_info: None,
            })
            .collect();
        RouterEvent::with_storage_tier(
            worker.worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: worker.dp_rank,
            },
            tier,
        )
        .with_kv_cache_group(tag(group_idx))
    }

    fn removed(
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group_idx: u32,
        hashes: &[u64],
    ) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker.worker_id,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes
                        .iter()
                        .map(|&h| ExternalSequenceBlockHash(h))
                        .collect(),
                }),
                dp_rank: worker.dp_rank,
            },
            tier,
        )
        .with_kv_cache_group(tag(group_idx))
    }

    fn chain(n: u64) -> Vec<ExternalSequenceBlockHash> {
        (1..=n).map(ExternalSequenceBlockHash).collect()
    }

    fn device_details(worker: WorkerWithDpRank, blocks: u32, chain_len: u64) -> MatchDetails {
        let mut details = MatchDetails::new();
        details.overlap_scores.scores.insert(worker, blocks);
        details
            .last_matched_hashes
            .insert(worker, ExternalSequenceBlockHash(blocks as u64));
        details.kv_transfer_candidates = Some(KvTransferCandidates {
            block_hashes: chain(chain_len),
            owner_prefix_blocks: vec![(worker.into(), blocks as usize)],
            routing_snapshot: None,
        });
        details
    }

    #[test]
    fn untagged_events_are_not_recorded() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        let mut event = stored(worker, StorageTier::Device, 1, &[1]);
        event.kv_cache_group = None;
        assert!(!residency.apply(&event));
        assert!(!residency.is_active());
    }

    #[test]
    fn dense_worker_is_left_alone() {
        let residency = KvCacheGroupResidency::new();
        let hybrid = WorkerWithDpRank::new(1, 0);
        let dense = WorkerWithDpRank::new(2, 0);
        residency.apply(&stored(hybrid, StorageTier::Device, 1, &[1]));
        assert_eq!(residency.reusable_blocks(dense, &chain(3), 3), None);
        let mut details = device_details(dense, 3, 3);
        residency.cut_device(&mut details);
        assert_eq!(details.overlap_scores.scores[&dense], 3);
    }

    #[test]
    fn match_is_cut_to_last_boundary_with_every_group() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        residency.apply(&stored(worker, StorageTier::Device, 1, &[1, 3]));
        residency.apply(&stored(worker, StorageTier::HostPinned, 2, &[1]));
        assert_eq!(residency.reusable_blocks(worker, &chain(3), 3), Some(1));

        let mut details = device_details(worker, 3, 3);
        residency.cut_device(&mut details);
        assert_eq!(details.overlap_scores.scores[&worker], 1);
        assert_eq!(
            details.last_matched_hashes[&worker],
            ExternalSequenceBlockHash(1)
        );
        let candidates = details.kv_transfer_candidates.as_ref().unwrap();
        assert_eq!(candidates.block_hashes, chain(1));
        assert_eq!(candidates.owner_prefix_blocks, vec![(worker.into(), 1)]);
    }

    #[test]
    fn worker_without_any_state_on_the_prefix_drops_out() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        residency.apply(&stored(worker, StorageTier::Device, 1, &[9]));
        let mut details = device_details(worker, 3, 3);
        residency.cut_device(&mut details);
        assert!(details.overlap_scores.scores.is_empty());
        assert!(details.last_matched_hashes.is_empty());
        assert!(details.kv_transfer_candidates.is_none());
    }

    #[test]
    fn removal_and_tiers_are_tracked_per_hash() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        residency.apply(&stored(worker, StorageTier::Device, 1, &[2]));
        residency.apply(&stored(worker, StorageTier::HostPinned, 1, &[2]));
        residency.apply(&removed(worker, StorageTier::Device, 1, &[2]));
        assert_eq!(residency.reusable_blocks(worker, &chain(2), 2), Some(2));
        residency.apply(&removed(worker, StorageTier::HostPinned, 1, &[2]));
        assert_eq!(residency.reusable_blocks(worker, &chain(2), 2), Some(0));
        assert_eq!(residency.groups_for(worker).len(), 1);
    }

    #[test]
    fn clear_and_remove_worker_forget_presence() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        residency.apply(&stored(worker, StorageTier::Device, 1, &[1]));
        residency.clear_worker(worker);
        assert!(!residency.is_active());
        residency.apply(&stored(worker, StorageTier::Device, 1, &[1]));
        residency.remove_worker(1);
        assert!(!residency.is_active());
    }

    #[test]
    fn lower_tier_hits_are_cut_with_the_device_chain() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        residency.apply(&stored(worker, StorageTier::HostPinned, 1, &[2]));
        let mut details = LowerTierMatchDetails::default();
        details.hits.insert(worker, 4);
        details.next_continuations.insert(
            worker,
            LowerTierContinuation::new(4, ExternalSequenceBlockHash(4)),
        );
        residency.cut_lower_tier(&mut details, &chain(4));
        assert_eq!(details.hits[&worker], 2);
        assert_eq!(details.next_continuations[&worker].start_pos, 2);
        assert_eq!(
            details.next_continuations[&worker].last_matched_hash,
            Some(ExternalSequenceBlockHash(2))
        );

        // Device already matched 2 blocks (continuation starts at 2); the tier
        // added 2 more but the only state is at block 3: one host hit survives.
        let mut details = LowerTierMatchDetails::default();
        details.hits.insert(worker, 2);
        details.next_continuations.insert(
            worker,
            LowerTierContinuation::new(4, ExternalSequenceBlockHash(4)),
        );
        residency.apply(&stored(worker, StorageTier::HostPinned, 1, &[3]));
        residency.cut_lower_tier(&mut details, &chain(4));
        assert_eq!(details.hits[&worker], 1);
        assert_eq!(details.next_continuations[&worker].start_pos, 3);
    }
}
