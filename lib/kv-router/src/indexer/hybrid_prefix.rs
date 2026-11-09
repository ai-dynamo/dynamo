// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashMap;
use std::collections::HashMap;

use super::{
    CheckpointLayout, CheckpointReadView, KvCacheGroupResidency, LowerTierMatchDetails,
    MatchDetails, MatchObserver,
};
use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwner, ResidencyOwnerKey, ResidencyRoutingSnapshot,
    StorageTier, WorkerWithDpRank,
};

#[derive(Clone, Copy, Debug)]
struct Checkpoint {
    position: usize,
    hash: ExternalSequenceBlockHash,
}

impl Checkpoint {
    const ZERO: Self = Self {
        position: 0,
        hash: ExternalSequenceBlockHash(0),
    };
}

struct DeviceWorkerState {
    captured: Option<CheckpointLayout>,
    checkpoint: Checkpoint,
    resume_cap: Option<usize>,
}

impl DeviceWorkerState {
    fn new(captured: Option<CheckpointLayout>) -> Self {
        Self {
            captured,
            checkpoint: Checkpoint::ZERO,
            resume_cap: None,
        }
    }

    fn checkpoint(&self, end: usize, floor: usize) -> Option<Checkpoint> {
        self.captured?;
        Some(
            if self.checkpoint.position <= end && self.checkpoint.position > floor {
                self.checkpoint
            } else {
                Checkpoint::ZERO
            },
        )
    }
}

/// Complete checkpoints collected at the block hashes visited by each tier walk.
pub(crate) struct HybridPrefixEvaluator {
    view: CheckpointReadView,
    candidates: FxHashMap<(ResidencyOwner, StorageTier), Vec<Checkpoint>>,
    tier: StorageTier,
    max_blocks: usize,
    workers: FxHashMap<WorkerWithDpRank, DeviceWorkerState>,
    device_prepared: bool,
}

impl HybridPrefixEvaluator {
    pub(crate) fn new(residency: &KvCacheGroupResidency, max_blocks: Option<usize>) -> Self {
        Self {
            view: residency.read_view(),
            candidates: FxHashMap::default(),
            tier: StorageTier::Device,
            max_blocks: max_blocks.unwrap_or(usize::MAX),
            workers: FxHashMap::default(),
            device_prepared: false,
        }
    }

    fn worker_state(&mut self, worker: WorkerWithDpRank) -> &mut DeviceWorkerState {
        self.workers.entry(worker).or_insert_with(|| {
            DeviceWorkerState::new(self.view.track_owner(ResidencyOwner::worker(worker)))
        })
    }

    pub(crate) fn on_tier(&mut self, tier: StorageTier) {
        self.tier = tier;
    }

    fn layout(&self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        if let Some(worker) = owner.as_worker()
            && let Some(state) = self.workers.get(&worker)
        {
            return state.captured;
        }
        self.view.layout(owner)
    }

    fn checkpoint(
        &self,
        owner: ResidencyOwner,
        tier: StorageTier,
        end: usize,
        floor: usize,
    ) -> Option<Checkpoint> {
        let end = end.min(self.max_blocks);
        if tier == StorageTier::Device
            && let Some(worker) = owner.as_worker()
            && let Some(state) = self.workers.get(&worker)
        {
            return state.checkpoint(end, floor);
        }
        self.layout(owner)?;
        if tier == StorageTier::Device {
            return Some(Checkpoint::ZERO);
        }
        Some(self.latest_checkpoint(owner, tier, end, floor))
    }

    fn latest_checkpoint(
        &self,
        owner: ResidencyOwner,
        tier: StorageTier,
        end: usize,
        floor: usize,
    ) -> Checkpoint {
        self.candidates
            .get(&(owner, tier))
            .and_then(|points| {
                let index = points.partition_point(|point| point.position <= end);
                points
                    .get(index.wrapping_sub(1))
                    .filter(|point| point.position > floor)
            })
            .copied()
            .unwrap_or(Checkpoint::ZERO)
    }

    pub(crate) fn prepare_device(&mut self, raw: &MatchDetails) {
        if self.device_prepared {
            return;
        }
        self.device_prepared = true;
        for (&worker, &blocks) in &raw.overlap_scores.scores {
            let point = self.latest_checkpoint(
                ResidencyOwner::worker(worker),
                StorageTier::Device,
                (blocks as usize).min(self.max_blocks),
                0,
            );
            let state = self.worker_state(worker);
            if state.captured.is_some() {
                state.checkpoint = point;
                state.resume_cap = Some(point.position);
            }
        }
    }

    /// A feasible GPU anchor seeds offloaded suffixes without mutating raw facts.
    pub(crate) fn query_anchor(&mut self, raw: &MatchDetails) -> MatchDetails {
        self.prepare_device(raw);
        let mut anchor = MatchDetails {
            kv_transfer_candidates: raw.kv_transfer_candidates.clone(),
            ..Default::default()
        };
        for (&worker, &blocks) in &raw.overlap_scores.scores {
            let state = self.workers.get(&worker);
            let blocks = state
                .filter(|state| state.captured.is_some())
                .map(|state| state.checkpoint.position)
                .unwrap_or(blocks as usize);
            if blocks == 0 {
                continue;
            }
            anchor.overlap_scores.scores.insert(worker, blocks as u32);
            if let Some(state) = state.filter(|state| state.checkpoint.position > 0) {
                anchor
                    .last_matched_hashes
                    .insert(worker, state.checkpoint.hash);
            } else if let Some(hash) = raw.last_matched_hashes.get(&worker) {
                anchor.last_matched_hashes.insert(worker, *hash);
            }
        }
        anchor
    }

    /// The sole credit decision, after tier walks and prediction merges. One
    /// endpoint P partitions the continuous GPU -> host -> disk -> external path.
    pub(crate) fn finish(
        &mut self,
        device: &mut MatchDetails,
        tiers: &mut HashMap<StorageTier, LowerTierMatchDetails>,
        snapshot: &ResidencyRoutingSnapshot,
    ) {
        for (tier, details) in tiers.iter() {
            for (worker, hits) in &details.hits {
                let Some(next) = details.next_continuations.get(worker) else {
                    continue;
                };
                let start = next.start_pos.saturating_sub(*hits);
                if let Some(point) = self.checkpoint(
                    ResidencyOwner::worker(*worker),
                    *tier,
                    next.start_pos,
                    start,
                ) {
                    let state = self.worker_state(*worker);
                    let cap = state.resume_cap.unwrap_or(0);
                    state.resume_cap = Some(if point.position > start {
                        cap.max(point.position)
                    } else {
                        cap
                    });
                }
            }
        }
        device.overlap_scores.scores.retain(|worker, blocks| {
            let state = self.worker_state(*worker);
            if state.captured.is_none() {
                return true;
            }
            state.resume_cap.get_or_insert(0);
            let cap = state.checkpoint.position;
            *blocks = (*blocks).min(cap as u32);
            if cap == 0 {
                device.last_matched_hashes.remove(worker);
            } else {
                device
                    .last_matched_hashes
                    .insert(*worker, state.checkpoint.hash);
            }
            *blocks > 0
        });
        trim_candidates(
            &mut device.kv_transfer_candidates,
            self,
            StorageTier::Device,
        );
        for (tier, details) in tiers.iter_mut() {
            for (worker, hits) in &mut details.hits {
                let Some(cap) = self.workers.get(worker).and_then(|state| state.resume_cap) else {
                    continue;
                };
                let Some(next) = details.next_continuations.get_mut(worker) else {
                    continue;
                };
                let start = next.start_pos.saturating_sub(*hits);
                let end = next.start_pos.min(cap);
                *hits = end.saturating_sub(start);
                next.start_pos = end;
                if end == 0 {
                    next.last_matched_hash = None;
                } else if let Some(point) =
                    self.checkpoint(ResidencyOwner::worker(*worker), *tier, end, start)
                {
                    // Continuations are no longer consumed after finalization.
                    if point.position == end {
                        next.last_matched_hash = Some(point.hash);
                    }
                }
            }
            trim_candidates(&mut details.kv_transfer_candidates, self, *tier);
            if let Some(extensions) = &mut details.kv_transfer_extensions {
                extensions.owner_prefix_blocks.retain(|source, blocks| {
                    if let Some(cap) = self.source_cap(*source, Some(snapshot), *blocks, *tier, 0) {
                        *blocks = (*blocks).min(cap);
                    }
                    *blocks > 0
                });
                extensions.source_intervals.retain_mut(|interval| {
                    if let Some(cap) = self.source_cap(
                        interval.source,
                        Some(snapshot),
                        interval.end,
                        *tier,
                        interval.start,
                    ) {
                        interval.end = interval.end.min(cap);
                    }
                    interval.start < interval.end
                });
            }
        }
        // Construct the wire map once, retaining pre-existing unrelated entries.
        // Some(0) is evidence for a known hybrid candidate, unlike absent state.
        let cap_count = self
            .workers
            .values()
            .filter(|state| state.resume_cap.is_some())
            .count();
        device.hybrid_resume_caps.reserve(cap_count);
        for (&worker, state) in &self.workers {
            if state.resume_cap.is_some() {
                device
                    .hybrid_resume_caps
                    .insert(worker, state.checkpoint.position as u32);
            }
        }
    }

    fn source_cap(
        &self,
        source: KvTransferCandidateSource,
        snapshot: Option<&ResidencyRoutingSnapshot>,
        end: usize,
        tier: StorageTier,
        floor: usize,
    ) -> Option<usize> {
        let (checkpoint, route_cap, device_prefix) = match source {
            KvTransferCandidateSource::Worker(worker) => {
                let state = self.workers.get(&worker);
                let checkpoint = if tier == StorageTier::Device {
                    match state {
                        Some(state) => state.checkpoint(end.min(self.max_blocks), floor),
                        None => self.checkpoint(ResidencyOwner::worker(worker), tier, end, floor),
                    }
                } else {
                    self.checkpoint(ResidencyOwner::worker(worker), tier, end, floor)
                };
                (
                    checkpoint,
                    state.and_then(|state| state.resume_cap),
                    state.map(|state| state.checkpoint.position).unwrap_or(0),
                )
            }
            KvTransferCandidateSource::CacheOwner(key) => (
                self.view
                    .owner_for_key(key)
                    .and_then(|owner| self.checkpoint(owner, tier, end, floor)),
                snapshot
                    .and_then(|snapshot| snapshot.projection().project_key(key))
                    .and_then(|worker| self.workers.get(&worker))
                    .and_then(|state| state.resume_cap),
                0,
            ),
        };
        match (checkpoint, route_cap) {
            (Some(point), route_cap) => Some(
                point
                    .position
                    .max(device_prefix)
                    .min(route_cap.unwrap_or(usize::MAX)),
            ),
            (None, Some(route_cap)) => Some(device_prefix.min(route_cap)),
            (None, None) => None,
        }
    }
}
impl MatchObserver for HybridPrefixEvaluator {
    fn reserve_workers(&mut self, count: usize) {
        if self.tier == StorageTier::Device && self.workers.is_empty() {
            self.workers.reserve(count);
        }
    }

    fn observe_worker(&mut self, worker: WorkerWithDpRank) {
        self.worker_state(worker);
    }

    fn observe_cache_owner(&mut self, key: ResidencyOwnerKey) {
        self.view.track_cache_owner(key);
    }

    fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
        if position > self.max_blocks {
            return;
        }
        let candidates = &mut self.candidates;
        let tier = self.tier;
        self.view.with_owners(hash, tier, |owners| {
            for (&owner, &alignment) in owners {
                if !position.is_multiple_of(alignment.max(1)) {
                    continue;
                }
                let points = candidates.entry((owner, tier)).or_default();
                if let Err(index) = points.binary_search_by_key(&position, |point| point.position) {
                    points.insert(index, Checkpoint { position, hash });
                }
            }
        });
    }
}

fn trim_candidates(
    candidates: &mut Option<KvTransferCandidates>,
    evaluator: &HybridPrefixEvaluator,
    tier: StorageTier,
) {
    let Some(data) = candidates.as_mut() else {
        return;
    };
    let snapshot = data.routing_snapshot.as_deref();
    data.owner_prefix_blocks.retain_mut(|(source, blocks)| {
        if let Some(cap) = evaluator.source_cap(*source, snapshot, *blocks, tier, 0) {
            *blocks = (*blocks).min(cap);
        }
        *blocks > 0
    });
    data.source_intervals.retain_mut(|interval| {
        if let Some(cap) = evaluator.source_cap(
            interval.source,
            snapshot,
            interval.end,
            tier,
            interval.start,
        ) {
            interval.end = interval.end.min(cap);
        }
        interval.start < interval.end
    });
    let end = data
        .owner_prefix_blocks
        .iter()
        .map(|(_, blocks)| *blocks)
        .chain(data.source_intervals.iter().map(|interval| interval.end))
        .max()
        .unwrap_or(0);
    data.block_hashes.truncate(end);
    if end == 0 {
        *candidates = None;
    }
}

/// Remote caps survive a caller's prediction merge, including explicit zero.
pub(crate) fn cap_side_overlay(
    residency: Option<&KvCacheGroupResidency>,
    details: &mut MatchDetails,
    confirmed: &FxHashMap<WorkerWithDpRank, u32>,
) {
    details.overlap_scores.scores.retain(|worker, blocks| {
        if let Some(cap) = details.hybrid_resume_caps.get(worker) {
            *blocks = (*blocks).min(*cap);
        } else if residency.is_some_and(|table| table.is_hybrid(*worker)) {
            *blocks = (*blocks).min(confirmed.get(worker).copied().unwrap_or(0));
        }
        *blocks > 0
    });
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::KvTransferExtensions;
    use crate::kv_hints::KvTransferSourceInterval;

    fn store_device_checkpoint(
        residency: &KvCacheGroupResidency,
        worker: WorkerWithDpRank,
        position: u64,
        alignment: u32,
    ) {
        let mut event =
            crate::test_utils::recurrent_store(worker, StorageTier::Device, position, 0, 1);
        event.kv_cache_group.as_mut().unwrap().alignment_blocks = Some(alignment);
        residency.apply(&event);
    }

    #[test]
    fn device_candidates_preserve_raw_bounds_alignment_and_dp() {
        for case in [
            "bound",
            "bound_missing",
            "alignment",
            "alignment_missing",
            "dp",
            "clear_readd",
        ] {
            let target = WorkerWithDpRank::new(64, u32::from(case == "dp"));
            let others: Vec<_> = (1..=63).map(|id| WorkerWithDpRank::new(id, 0)).collect();
            let residency = KvCacheGroupResidency::new();
            for &worker in &others {
                residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
                for position in 1..=8 {
                    store_device_checkpoint(&residency, worker, position, 1);
                }
            }
            let alignment = if case.starts_with("alignment") { 2 } else { 1 };
            residency.declare_layout(ResidencyOwner::worker(target), &[0], alignment);
            let positions: &[u64] = match case {
                "bound" => &[8, 3],
                "bound_missing" => &[8, 7],
                "alignment" => &[7, 2],
                "alignment_missing" => &[7],
                _ => &[999],
            };
            for &position in positions {
                store_device_checkpoint(&residency, target, position, alignment);
            }
            let wrong_dp = WorkerWithDpRank::new(64, 0);
            if case == "dp" {
                residency.declare_groups(ResidencyOwner::worker(wrong_dp), &[0]);
                for position in 1..=8 {
                    store_device_checkpoint(&residency, wrong_dp, position, 1);
                }
            }
            let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
            let mut raw = MatchDetails::default();
            for &worker in &others {
                evaluator.observe_worker(worker);
                raw.overlap_scores.scores.insert(worker, 8);
            }
            evaluator.observe_worker(target);
            raw.overlap_scores
                .scores
                .insert(target, if case.starts_with("bound") { 3 } else { 8 });
            raw.last_matched_hashes
                .insert(target, ExternalSequenceBlockHash(8));
            if case == "dp" {
                evaluator.observe_worker(wrong_dp);
            }
            if case == "clear_readd" {
                residency.clear_worker(target);
                for position in 1..=8 {
                    store_device_checkpoint(&residency, target, position, 1);
                }
            }
            for position in 1..=8 {
                evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
            }
            raw.kv_transfer_candidates = Some(KvTransferCandidates {
                block_hashes: (1..=8).map(ExternalSequenceBlockHash).collect(),
                owner_prefix_blocks: vec![(target.into(), 8)],
                source_intervals: Vec::new(),
                routing_snapshot: None,
            });
            evaluator.prepare_device(&raw);
            evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
            let expected = match case {
                "bound" => 3,
                "alignment" => 2,
                "clear_readd" => 8,
                _ => 0,
            };
            assert_eq!(
                raw.overlap_scores.scores.get(&target).copied().unwrap_or(0),
                expected,
                "{case}"
            );
            assert_eq!(
                raw.hybrid_resume_caps.get(&target),
                Some(&expected),
                "{case}"
            );
            if expected == 0 {
                assert!(!raw.last_matched_hashes.contains_key(&target), "{case}");
                assert!(raw.kv_transfer_candidates.is_none(), "{case}");
            } else {
                assert_eq!(
                    raw.last_matched_hashes[&target],
                    ExternalSequenceBlockHash(expected as u64),
                    "{case}"
                );
                assert_eq!(
                    raw.kv_transfer_candidates.unwrap().block_hashes.len(),
                    expected as usize,
                    "{case}"
                );
            }
            if case == "dp" {
                assert!(!raw.hybrid_resume_caps.contains_key(&wrong_dp));
            }
        }
    }

    #[test]
    fn device_worker_reservation_uses_the_hint_without_growing_during_announcements() {
        let residency = KvCacheGroupResidency::new();
        let workers: Vec<_> = (0..64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.reserve_workers(workers.len());
        let capacity = evaluator.workers.capacity();
        assert!(capacity >= workers.len());
        for worker in &workers {
            evaluator.observe_worker(*worker);
            assert_eq!(evaluator.workers.capacity(), capacity);
        }
        assert_eq!(evaluator.workers.len(), workers.len());
        evaluator.reserve_workers(capacity + 1);
        assert_eq!(evaluator.workers.capacity(), capacity);

        let empty: &[WorkerWithDpRank] = &[];
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.reserve_workers(empty.len());
        assert_eq!(evaluator.workers.capacity(), 0);
        evaluator.on_tier(StorageTier::HostPinned);
        evaluator.reserve_workers(workers.len());
        assert_eq!(evaluator.workers.capacity(), 0);
        assert_eq!(evaluator.candidates.capacity(), 0);
    }

    #[test]
    fn device_only_side_predictions_keep_zero_caps_and_trim_hints() {
        let hybrid = WorkerWithDpRank::new(7, 0);
        let dense = WorkerWithDpRank::new(8, 0);
        let residency = KvCacheGroupResidency::new();
        residency.declare_groups(ResidencyOwner::worker(hybrid), &[0]);
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.prepare_device(&MatchDetails::default());

        let mut merged = MatchDetails::default();
        merged.overlap_scores.scores.insert(hybrid, 3);
        merged.overlap_scores.scores.insert(dense, 3);
        merged.kv_transfer_candidates = Some(KvTransferCandidates {
            block_hashes: (1..=3).map(ExternalSequenceBlockHash).collect(),
            owner_prefix_blocks: vec![(hybrid.into(), 3), (dense.into(), 3)],
            source_intervals: Vec::new(),
            routing_snapshot: None,
        });
        evaluator.finish(&mut merged, &mut HashMap::new(), &Default::default());

        assert!(!merged.overlap_scores.scores.contains_key(&hybrid));
        assert_eq!(merged.hybrid_resume_caps.get(&hybrid), Some(&0));
        assert_eq!(merged.overlap_scores.scores.get(&dense), Some(&3));
        assert!(!merged.hybrid_resume_caps.contains_key(&dense));
        let hints = merged.kv_transfer_candidates.unwrap();
        assert_eq!(hints.owner_prefix_blocks, vec![(dense.into(), 3)]);
        assert_eq!(hints.block_hashes.len(), 3);
    }

    #[test]
    fn output_caps_preserve_foreign_entries_without_emitting_unmatched_workers() {
        let hybrid = WorkerWithDpRank::new(7, 0);
        let unmatched = WorkerWithDpRank::new(8, 0);
        let dense = WorkerWithDpRank::new(9, 0);
        let foreign = WorkerWithDpRank::new(10, 0);
        let residency = KvCacheGroupResidency::new();
        for worker in [hybrid, unmatched] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        for worker in [hybrid, unmatched, dense] {
            evaluator.observe_worker(worker);
        }
        let mut raw = MatchDetails::default();
        raw.overlap_scores.scores.insert(hybrid, 3);
        raw.overlap_scores.scores.insert(dense, 3);
        raw.hybrid_resume_caps.insert(hybrid, 99);
        raw.hybrid_resume_caps.insert(foreign, 12);
        evaluator.prepare_device(&raw);
        evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());

        assert_eq!(
            raw.hybrid_resume_caps,
            FxHashMap::from_iter([(hybrid, 0), (foreign, 12)])
        );
        assert_eq!(
            raw.overlap_scores.scores,
            FxHashMap::from_iter([(dense, 3)])
        );
    }

    #[test]
    fn current_checkpoints_after_clear_and_readd_are_used_across_tiers() {
        use crate::indexer::LowerTierContinuation;
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        for legacy_first in [false, true] {
            let residency = KvCacheGroupResidency::new();
            residency.declare_groups(owner, &[0]);
            let gpu = crate::test_utils::recurrent_store(worker, StorageTier::Device, 2, 0, 1);
            let host = crate::test_utils::recurrent_store(worker, StorageTier::HostPinned, 2, 0, 1);
            residency.apply(&gpu);
            residency.apply(&host);
            let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
            if legacy_first {
                evaluator.view.track_owner(owner);
            } else {
                evaluator.observe_worker(worker);
            }
            residency.clear_worker(worker);
            residency.apply(&gpu);
            residency.apply(&host);
            evaluator.observe_worker(worker);
            evaluator.observe_block(2, ExternalSequenceBlockHash(2));
            let mut raw = MatchDetails::default();
            raw.overlap_scores.scores.insert(worker, 3);
            evaluator.prepare_device(&raw);
            evaluator.on_tier(StorageTier::HostPinned);
            evaluator.observe_worker(worker);
            evaluator.observe_block(2, ExternalSequenceBlockHash(2));
            let mut lower = HashMap::from([(
                StorageTier::HostPinned,
                LowerTierMatchDetails {
                    hits: FxHashMap::from_iter([(worker, 3)]),
                    next_continuations: FxHashMap::from_iter([(
                        worker,
                        LowerTierContinuation::new(3, ExternalSequenceBlockHash(3)),
                    )]),
                    ..Default::default()
                },
            )]);
            evaluator.finish(&mut raw, &mut lower, &Default::default());
            assert_eq!(
                raw.overlap_scores.scores.get(&worker),
                Some(&2),
                "legacy_first={legacy_first}"
            );
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&2));
            assert_eq!(lower[&StorageTier::HostPinned].hits[&worker], 2);
        }
    }

    #[test]
    fn captured_absent_layout_is_not_relearned_when_entering_lower_tiers() {
        use crate::indexer::LowerTierContinuation;
        let worker = WorkerWithDpRank::new(7, 0);
        let residency = KvCacheGroupResidency::new();
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.observe_worker(worker);
        residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        residency.apply(&crate::test_utils::recurrent_store(
            worker,
            StorageTier::HostPinned,
            2,
            0,
            1,
        ));
        let mut raw = MatchDetails::default();
        raw.overlap_scores.scores.insert(worker, 3);
        evaluator.prepare_device(&raw);
        evaluator.on_tier(StorageTier::HostPinned);
        evaluator.observe_worker(worker);
        evaluator.observe_block(2, ExternalSequenceBlockHash(2));
        let mut lower = HashMap::from([(
            StorageTier::HostPinned,
            LowerTierMatchDetails {
                hits: FxHashMap::from_iter([(worker, 3)]),
                next_continuations: FxHashMap::from_iter([(
                    worker,
                    LowerTierContinuation::new(3, ExternalSequenceBlockHash(3)),
                )]),
                ..Default::default()
            },
        )]);
        evaluator.finish(&mut raw, &mut lower, &Default::default());
        assert_eq!(raw.overlap_scores.scores.get(&worker), Some(&3));
        assert!(!raw.hybrid_resume_caps.contains_key(&worker));
        assert_eq!(lower[&StorageTier::HostPinned].hits[&worker], 3);
        assert!(evaluator.workers[&worker].captured.is_none());
    }

    #[test]
    fn device_checkpoint_presence_requires_the_exact_owner_and_uses_current_state() {
        let worker = WorkerWithDpRank::new(7, 0);
        let other = WorkerWithDpRank::new(8, 0);
        for stale in [false, true] {
            let residency = KvCacheGroupResidency::new();
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            let source = if stale { worker } else { other };
            let event = crate::test_utils::recurrent_store(source, StorageTier::Device, 2, 0, 1);
            residency.apply(&event);
            let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
            evaluator.observe_worker(worker);
            if stale {
                residency.clear_worker(worker);
                residency.apply(&event);
            }
            evaluator.observe_block(2, ExternalSequenceBlockHash(2));

            let mut raw = MatchDetails::default();
            raw.overlap_scores.scores.insert(worker, 3);
            raw.last_matched_hashes
                .insert(worker, ExternalSequenceBlockHash(3));
            evaluator.prepare_device(&raw);
            evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
            let expected = if stale { 2 } else { 0 };
            assert_eq!(
                raw.overlap_scores.scores.get(&worker).copied().unwrap_or(0),
                expected
            );
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&expected));
        }
    }

    #[tokio::test]
    async fn device_membership_changes_before_observation_use_the_current_complete_owners() {
        use crate::ConcurrentRadixTreeCompressed;
        use crate::indexer::{RadixTree, ThreadPoolIndexer};
        use crate::protocols::{
            KvCacheEventData, KvCacheRemoveData, LocalBlockHash, RouterEvent,
            compute_seq_hash_for_block,
        };

        struct InterleavingObserver<'a> {
            evaluator: HybridPrefixEvaluator,
            residency: &'a KvCacheGroupResidency,
            updates: Vec<RouterEvent>,
        }

        impl MatchObserver for InterleavingObserver<'_> {
            fn observe_worker(&mut self, worker: WorkerWithDpRank) {
                self.evaluator.observe_worker(worker);
            }

            fn observe_cache_owner(&mut self, owner: ResidencyOwnerKey) {
                self.evaluator.observe_cache_owner(owner);
            }

            fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
                if position == 2 {
                    for event in self.updates.drain(..) {
                        self.residency.apply(&event);
                    }
                }
                self.evaluator.observe_block(position, hash);
            }
        }

        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let sequence: Vec<_> = (11..=13).map(LocalBlockHash).collect();
        let hashes: Vec<_> = compute_seq_hash_for_block(&sequence)
            .into_iter()
            .map(ExternalSequenceBlockHash)
            .collect();
        for concurrent in [false, true] {
            for mutation in ["unchanged", "add_owner", "remove", "remove_readd"] {
                let residency = KvCacheGroupResidency::new();
                for worker in [first, second] {
                    residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
                }
                let checkpoint = |worker, position: usize| {
                    crate::test_utils::recurrent_store(
                        worker,
                        StorageTier::Device,
                        hashes[position - 1].0,
                        0,
                        1,
                    )
                };
                residency.apply(&checkpoint(first, 1));
                residency.apply(&checkpoint(first, 2));
                let updates = match mutation {
                    "add_owner" => vec![checkpoint(second, 2)],
                    "remove" | "remove_readd" => {
                        let stored = checkpoint(first, 2);
                        let mut removed = stored.clone();
                        removed.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: vec![hashes[1]],
                        });
                        if mutation == "remove" {
                            vec![removed]
                        } else {
                            vec![removed, stored]
                        }
                    }
                    _ => Vec::new(),
                };
                let mut observer = InterleavingObserver {
                    evaluator: HybridPrefixEvaluator::new(&residency, None),
                    residency: &residency,
                    updates,
                };
                let events = [first, second].map(|worker| {
                    crate::test_utils::make_store_event(worker.worker_id, &[11, 12, 13])
                });
                let mut raw = if concurrent {
                    let indexer =
                        ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 4);
                    for event in events {
                        indexer.apply_event_and_wait(event).await.unwrap();
                    }
                    indexer.backend().find_match_details_impl_with_observer(
                        &sequence,
                        false,
                        true,
                        &mut observer,
                    )
                } else {
                    let mut tree = RadixTree::new();
                    for event in events {
                        tree.apply_event(event).unwrap();
                    }
                    tree.find_match_details_with_observer(
                        sequence.clone(),
                        false,
                        true,
                        &mut observer,
                    )
                };
                let context = format!("concurrent={concurrent} mutation={mutation}");
                assert_eq!(raw.overlap_scores.scores[&first], 3, "{context}");
                assert_eq!(raw.overlap_scores.scores[&second], 3, "{context}");
                observer.evaluator.prepare_device(&raw);
                observer
                    .evaluator
                    .finish(&mut raw, &mut HashMap::new(), &Default::default());

                let expected = if mutation == "remove" { 1 } else { 2 };
                let second_expected = if mutation == "add_owner" { 2 } else { 0 };
                assert_eq!(
                    raw.overlap_scores.scores.get(&first),
                    Some(&expected),
                    "{context}"
                );
                assert_eq!(
                    raw.overlap_scores.scores.get(&second).copied().unwrap_or(0),
                    second_expected,
                    "{context}"
                );
                assert_eq!(
                    raw.hybrid_resume_caps.get(&first),
                    Some(&expected),
                    "{context}"
                );
                assert_eq!(
                    raw.hybrid_resume_caps.get(&second),
                    Some(&second_expected),
                    "{context}"
                );
                assert_eq!(
                    raw.last_matched_hashes[&first],
                    hashes[expected as usize - 1],
                    "{context}"
                );
                let hints = raw.kv_transfer_candidates.unwrap();
                assert_eq!(hints.block_hashes, hashes[..expected as usize], "{context}");
                assert!(
                    hints
                        .owner_prefix_blocks
                        .contains(&(first.into(), expected as usize)),
                    "{context}"
                );
                assert_eq!(
                    hints.owner_prefix_blocks.len(),
                    if second_expected > 0 { 2 } else { 1 },
                    "{context}"
                );
                if second_expected > 0 {
                    assert!(
                        hints
                            .owner_prefix_blocks
                            .contains(&(second.into(), second_expected as usize)),
                        "{context}"
                    );
                }
            }
        }
    }

    #[test]
    fn cache_owner_hint_cannot_borrow_its_attachment_workers_gpu_checkpoint() {
        use crate::identity::{
            CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };
        use crate::protocols::ResidencyProjection;
        let owner = CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                ClusterId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        );
        let worker = WorkerWithDpRank::new(7, 0);
        let snapshot = ResidencyRoutingSnapshot::from_projection(
            ResidencyProjection::new([(owner, worker)]).unwrap(),
        );
        let table = KvCacheGroupResidency::new();
        let mut evaluator = HybridPrefixEvaluator::new(&table, None);
        let state = evaluator.worker_state(worker);
        state.checkpoint = Checkpoint {
            position: 2,
            hash: ExternalSequenceBlockHash(2),
        };
        state.resume_cap = Some(4);
        assert_eq!(
            evaluator.source_cap(
                KvTransferCandidateSource::CacheOwner(
                    ResidencyOwner::cache_owner(owner).compact_key()
                ),
                Some(&snapshot),
                4,
                StorageTier::HostPinned,
                0
            ),
            Some(0)
        );
    }

    #[test]
    fn suffix_only_hybrid_hint_source_requires_its_own_checkpoint() {
        let residency = KvCacheGroupResidency::new();
        let hybrid = WorkerWithDpRank::new(7, 0);
        let dense = WorkerWithDpRank::new(8, 0);
        residency.declare_groups(ResidencyOwner::worker(hybrid), &[0]);
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.observe_worker(hybrid);
        let details = LowerTierMatchDetails {
            kv_transfer_extensions: Some(KvTransferExtensions {
                source_intervals: vec![
                    KvTransferSourceInterval {
                        source: hybrid.into(),
                        start: 1,
                        end: 3,
                    },
                    KvTransferSourceInterval {
                        source: dense.into(),
                        start: 1,
                        end: 3,
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut tiers = HashMap::from([(StorageTier::HostPinned, details)]);
        evaluator.finish(
            &mut MatchDetails::default(),
            &mut tiers,
            &ResidencyRoutingSnapshot::default(),
        );
        let intervals = &tiers[&StorageTier::HostPinned]
            .kv_transfer_extensions
            .as_ref()
            .unwrap()
            .source_intervals;
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].source, dense.into());
    }

    #[cfg(feature = "standalone-indexer")]
    #[tokio::test]
    async fn single_and_concurrent_use_only_complete_visited_checkpoint_boundaries() {
        use crate::indexer::{KvIndexer, KvIndexerMetrics, LowerTierIndexers};
        use crate::protocols::*;
        use crate::services::indexer::backend::Indexer;
        use crate::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};
        use std::sync::Arc;
        use tokio_util::sync::CancellationToken;
        for concurrent in [false, true] {
            let lower_tier = LowerTierIndexers::new(1, 4);
            let indexer = if concurrent {
                Indexer::Concurrent {
                    primary: Arc::new(ThreadPoolIndexer::new(
                        ConcurrentRadixTreeCompressed::new(),
                        1,
                        4,
                    )),
                    lower_tier,
                    approx: None,
                    primary_records_routing_decisions: false,
                    session_updates: None,
                }
            } else {
                Indexer::Single {
                    primary: KvIndexer::new(
                        CancellationToken::new(),
                        4,
                        Arc::new(KvIndexerMetrics::new_unregistered()),
                    ),
                    lower_tier,
                    approx: None,
                    primary_records_routing_decisions: false,
                    session_updates: None,
                }
            };
            let worker = WorkerWithDpRank::new(27, 0);
            let locals: Vec<_> = (11..=18).map(LocalBlockHash).collect();
            let hashes: Vec<_> = compute_seq_hash_for_block(&locals)
                .into_iter()
                .map(ExternalSequenceBlockHash)
                .collect();
            indexer.declare_recurrent_groups(worker, &[0, 1, 2]);
            let stored = RouterEvent::new(
                27,
                KvCacheEvent {
                    event_id: 1,
                    dp_rank: 0,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
                        blocks: locals
                            .iter()
                            .zip(&hashes)
                            .map(|(&tokens_hash, &block_hash)| KvCacheStoredBlockData {
                                block_hash,
                                tokens_hash,
                                mm_extra_info: None,
                            })
                            .collect(),
                    }),
                },
            );
            indexer.try_apply_event(stored).await.unwrap();
            indexer.dump_events().await.unwrap();
            assert!(
                indexer
                    .find_primary_match_details(locals.clone())
                    .await
                    .unwrap()
                    .overlap_scores
                    .scores
                    .is_empty()
            );
            let checkpoint = |hash, group, tier| {
                RouterEvent::with_storage_tier(
                    27,
                    KvCacheEvent {
                        event_id: 2,
                        dp_rank: 0,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None,
                            start_position: None,
                            blocks: vec![KvCacheStoredBlockData {
                                block_hash: hash,
                                tokens_hash: LocalBlockHash(0),
                                mm_extra_info: None,
                            }],
                        }),
                    },
                    tier,
                )
                .with_kv_cache_group(KvCacheGroupTag {
                    group_idx: group,
                    kind: KvCacheGroupKind::Mamba,
                    expected_group_count: Some(3),
                    alignment_blocks: Some(1),
                })
            };
            // A state outside the matched attention path never proves reuse.
            for group in 0..3 {
                indexer
                    .try_apply_event(checkpoint(
                        ExternalSequenceBlockHash(999),
                        group,
                        StorageTier::Device,
                    ))
                    .await
                    .unwrap();
            }
            for group in 0..3 {
                indexer
                    .try_apply_event(checkpoint(
                        hashes[1],
                        group,
                        if group == 2 {
                            StorageTier::HostPinned
                        } else {
                            StorageTier::Device
                        },
                    ))
                    .await
                    .unwrap();
            }
            assert!(
                indexer
                    .find_primary_match_details(locals.clone())
                    .await
                    .unwrap()
                    .overlap_scores
                    .scores
                    .is_empty(),
                "vLLM cannot assemble a GPU state from groups split across physical tiers"
            );
            indexer
                .try_apply_event(checkpoint(hashes[1], 2, StorageTier::Device))
                .await
                .unwrap();
            for group in 0..3 {
                indexer
                    .try_apply_event(checkpoint(hashes[5], group, StorageTier::Device))
                    .await
                    .unwrap();
            }
            for (limit, expected) in [(None, 6), (Some(5), 2), (Some(0), 0)] {
                let result = indexer
                    .find_primary_match_details_with_limit(locals.clone(), limit)
                    .await
                    .unwrap();
                assert_eq!(
                    result
                        .overlap_scores
                        .scores
                        .get(&worker)
                        .copied()
                        .unwrap_or(0),
                    expected,
                    "concurrent={concurrent} limit={limit:?}"
                );
                assert_eq!(result.hybrid_resume_caps.get(&worker), Some(&expected));
            }
            let partial = indexer
                .find_primary_match_details(locals[..4].to_vec())
                .await
                .unwrap();
            assert_eq!(partial.overlap_scores.scores[&worker], 2);
        }
    }
    #[test]
    fn engine_alignment_never_rounds_a_checkpoint_into_an_unstored_boundary() {
        let worker = WorkerWithDpRank::new(29, 0);
        let table = KvCacheGroupResidency::new();
        table.declare_layout(ResidencyOwner::worker(worker), &[7], 2);
        let store = |position| {
            let mut event =
                crate::test_utils::recurrent_store(worker, StorageTier::Device, position, 7, 1);
            event.kv_cache_group.as_mut().unwrap().alignment_blocks = Some(2);
            table.apply(&event);
        };
        let raw = || {
            let mut details = MatchDetails::default();
            details.overlap_scores.scores.insert(worker, 4);
            details
                .last_matched_hashes
                .insert(worker, ExternalSequenceBlockHash(4));
            details
        };
        store(3);
        for (with_aligned_state, limit, expected) in
            [(false, None, 0), (true, None, 2), (true, Some(1), 0)]
        {
            if with_aligned_state {
                store(2);
            }
            let mut evaluator = HybridPrefixEvaluator::new(&table, limit);
            evaluator.observe_worker(worker);
            for position in 1..=4 {
                evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
            }
            let mut details = raw();
            evaluator.query_anchor(&details);
            evaluator.finish(&mut details, &mut Default::default(), &Default::default());
            assert_eq!(
                details
                    .overlap_scores
                    .scores
                    .get(&worker)
                    .copied()
                    .unwrap_or(0),
                expected
            );
        }
    }
}
