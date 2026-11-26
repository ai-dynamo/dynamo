// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::FxHashMap;
use std::cell::OnceCell;
use std::collections::HashMap;

use super::forward_prefix::{ForwardDeviceEvaluator, Pending};
use super::kv_cache_groups::tier_index;
use super::{
    CheckpointLayout, CheckpointReadView, KvCacheGroupResidency, LowerTierMatchDetails,
    MatchDetails, MatchObserver,
};
use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use crate::prelude::{
    ExternalSequenceBlockHash, LocalBlockHash, ResidencyOwner, ResidencyOwnerKey,
    ResidencyRoutingSnapshot, StorageTier, WorkerWithDpRank, compute_seq_hash_for_block,
};
use dynamo_tokens::SequenceHash;

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

/// The blocks one tier's walks visited, by one-based position.
#[derive(Default)]
struct TierBlocks {
    /// The hash seen first at each position.
    first: Vec<Option<ExternalSequenceBlockHash>>,
    /// Other hashes seen at a position, in visit order. Empty unless two
    /// walks of the tier disagree about a position.
    other: Vec<(usize, ExternalSequenceBlockHash)>,
}

impl TierBlocks {
    fn record(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
        if position >= self.first.len() {
            self.first.resize(position + 1, None);
        }
        match self.first[position] {
            None => self.first[position] = Some(hash),
            Some(first) if first == hash => {}
            Some(_) => self.other.push((position, hash)),
        }
    }

    /// The hashes seen at `position`, in visit order.
    fn at(&self, position: usize) -> impl Iterator<Item = ExternalSequenceBlockHash> + '_ {
        let first = self.first.get(position).copied().flatten();
        let other = self
            .other
            .iter()
            .filter(move |&&(other, _)| other == position)
            .map(|&(_, hash)| hash);
        first.into_iter().chain(other)
    }
}

/// One hybrid worker's hits on a lower tier, waiting for its deepest
/// checkpoint in `(floor, end]`.
struct LowerQuery {
    worker: WorkerWithDpRank,
    end: usize,
    floor: usize,
    /// The checkpoint, once settled; `ZERO` if the worker holds none.
    point: Checkpoint,
    /// Settled, or past its floor.
    done: bool,
}

type TierCandidates = FxHashMap<ResidencyOwner, Vec<Checkpoint>>;

/// Caps hybrid workers at their checkpoints across the Device and lower tiers.
///
/// The tier walks only note which blocks they visited. Checkpoint records are
/// read afterwards, outside the tree guards. Workers are settled together
/// from the deepest block downward: one record read serves every worker
/// waiting at that block, and reading stops when no worker is waiting. Workers
/// that hold a checkpoint where their match ends therefore cost one read,
/// however many of the visited blocks have one.
pub(crate) struct HybridPrefixEvaluator {
    view: CheckpointReadView,
    tier: StorageTier,
    max_blocks: usize,
    /// The query's length in blocks, to size a tier's block index before its
    /// walk takes any tree guard.
    blocks: usize,
    workers: FxHashMap<WorkerWithDpRank, DeviceWorkerState>,
    device_prepared: bool,
    /// Blocks each tier's walks visited, indexed like `StorageTier`.
    visited: [TierBlocks; 4],
    /// Every owner's checkpoints among a lower tier's visited blocks, by
    /// ascending position. Only a transfer hint asks for an arbitrary owner
    /// and range, so a tier's map is built the first time one does.
    candidates: [OnceCell<TierCandidates>; 4],
}

impl HybridPrefixEvaluator {
    pub(crate) fn new(residency: &KvCacheGroupResidency, max_blocks: Option<usize>) -> Self {
        Self {
            view: residency.read_view(),
            tier: StorageTier::Device,
            max_blocks: max_blocks.unwrap_or(usize::MAX),
            blocks: 0,
            workers: FxHashMap::default(),
            device_prepared: false,
            visited: Default::default(),
            candidates: Default::default(),
        }
    }

    /// Size the block index for a query of `blocks` blocks before the walk
    /// takes any tree guard.
    pub(crate) fn reserve_blocks(&mut self, blocks: usize) {
        self.blocks = blocks.min(self.max_blocks);
        self.size_tier(StorageTier::Device);
    }

    fn size_tier(&mut self, tier: StorageTier) {
        let first = &mut self.visited[tier_index(tier)].first;
        if self.blocks > 0 && first.len() <= self.blocks {
            first.resize(self.blocks + 1, None);
        }
    }

    fn worker_state(&mut self, worker: WorkerWithDpRank) -> &mut DeviceWorkerState {
        self.workers.entry(worker).or_insert_with(|| {
            DeviceWorkerState::new(self.view.track_owner(ResidencyOwner::worker(worker)))
        })
    }

    pub(crate) fn on_tier(&mut self, tier: StorageTier) {
        self.tier = tier;
        self.size_tier(tier);
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

    /// The deepest visited block of a lower `tier` in `(floor, end]` where
    /// `owner` holds a complete checkpoint, for any owner and range.
    fn latest_checkpoint(
        &self,
        owner: ResidencyOwner,
        tier: StorageTier,
        end: usize,
        floor: usize,
    ) -> Checkpoint {
        self.candidates[tier_index(tier)]
            .get_or_init(|| self.collect_candidates(tier))
            .get(&owner)
            .and_then(|points| {
                let index = points.partition_point(|point| point.position <= end);
                points
                    .get(index.wrapping_sub(1))
                    .filter(|point| point.position > floor)
            })
            .copied()
            .unwrap_or(Checkpoint::ZERO)
    }

    /// Read the record of every block `tier`'s walks visited. At one position
    /// an owner keeps the hash seen first.
    fn collect_candidates(&self, tier: StorageTier) -> TierCandidates {
        let mut candidates = TierCandidates::default();
        if !self.view.tier_has_checkpoints(tier) {
            return candidates;
        }
        let visited = &self.visited[tier_index(tier)];
        // Owners are copied out, so no shard guard is held while the map grows.
        let mut owners = Vec::with_capacity(self.workers.len().max(16));
        for position in 1..visited.first.len() {
            for hash in visited.at(position) {
                owners.clear();
                self.view.visit_owners(hash, tier, |current| {
                    owners.extend(current.iter().copied());
                });
                for &owner in &owners {
                    let points = candidates.entry(owner).or_default();
                    if points
                        .last()
                        .is_none_or(|point: &Checkpoint| point.position != position)
                    {
                        points.push(Checkpoint { position, hash });
                    }
                }
            }
        }
        candidates
    }

    /// Settle each query's deepest checkpoint on a lower `tier`, from the
    /// deepest block downward. A record is read once for every query open at
    /// its block, and reading stops when no query is open. At one position
    /// the hash seen first is tried first.
    fn settle_lower(&self, tier: StorageTier, queries: &mut [LowerQuery]) {
        if !self.view.tier_has_checkpoints(tier) {
            return;
        }
        let visited = &self.visited[tier_index(tier)];
        let last = visited.first.len().saturating_sub(1);
        // A query opens at its end and closes at its floor.
        let mut by_end: Vec<u32> = Vec::with_capacity(queries.len());
        for (slot, query) in queries.iter_mut().enumerate() {
            query.end = query.end.min(last);
            if query.end > query.floor {
                by_end.push(slot as u32);
            }
        }
        if by_end.is_empty() {
            return;
        }
        by_end.sort_unstable_by_key(|&slot| std::cmp::Reverse(queries[slot as usize].end));
        let mut by_floor = by_end.clone();
        by_floor.sort_unstable_by_key(|&slot| std::cmp::Reverse(queries[slot as usize].floor));
        let (mut opened, mut closed) = (0, 0);
        let mut open: Vec<u32> = Vec::with_capacity(by_end.len());
        let mut waiting = 0usize;
        // Built after the first record with few owners meets many open
        // queries, outside its shard guard, and used from the next record on.
        let mut slots: Option<FxHashMap<WorkerWithDpRank, u32>> = None;
        let mut sparse = false;
        let mut position = queries[by_end[0] as usize].end;
        while position > 0 {
            while let Some(&slot) = by_end.get(opened)
                && queries[slot as usize].end >= position
            {
                open.push(slot);
                waiting += 1;
                opened += 1;
            }
            while let Some(&slot) = by_floor.get(closed)
                && queries[slot as usize].floor >= position
            {
                let query = &mut queries[slot as usize];
                if !query.done {
                    query.done = true;
                    waiting -= 1;
                }
                closed += 1;
            }
            if waiting == 0 {
                // Nobody can use the blocks above the next query's end.
                let Some(&slot) = by_end.get(opened) else {
                    break;
                };
                position = queries[slot as usize].end;
                open.clear();
                continue;
            }
            for hash in visited.at(position) {
                self.view.visit_owners(hash, tier, |owners| {
                    let few_owners = owners.len() * 2 < waiting;
                    if few_owners && let Some(slots) = &slots {
                        for owner in owners {
                            if let Some(worker) = owner.as_worker()
                                && let Some(&slot) = slots.get(&worker)
                            {
                                let query = &mut queries[slot as usize];
                                if !query.done && query.end >= position {
                                    query.point = Checkpoint { position, hash };
                                    query.done = true;
                                    waiting -= 1;
                                }
                            }
                        }
                    } else {
                        sparse |= few_owners;
                        open.retain(|&slot| {
                            let query = &mut queries[slot as usize];
                            if query.done {
                                return false;
                            }
                            if owners.contains(&ResidencyOwner::worker(query.worker)) {
                                query.point = Checkpoint { position, hash };
                                query.done = true;
                                waiting -= 1;
                                return false;
                            }
                            true
                        });
                    }
                });
                if sparse && slots.is_none() {
                    slots = Some(
                        by_end
                            .iter()
                            .map(|&slot| (queries[slot as usize].worker, slot))
                            .collect(),
                    );
                }
                if waiting == 0 {
                    break;
                }
            }
            position -= 1;
        }
    }

    /// Settle every scored hybrid worker's Device resume point: its deepest
    /// matched block within its score where it holds a checkpoint. The workers
    /// are settled together from the deepest score downward, so one record
    /// read serves every worker waiting at that block.
    pub(crate) fn prepare_device(&mut self, raw: &MatchDetails) {
        if self.device_prepared {
            return;
        }
        self.device_prepared = true;
        let mut pending = Vec::with_capacity(raw.overlap_scores.scores.len());
        for (&worker, &blocks) in &raw.overlap_scores.scores {
            let bound = (blocks as usize).min(self.max_blocks);
            // A worker the registry does not summarize is searched in full.
            let (count, bit) = self
                .view
                .device_summary(worker)
                .unwrap_or((usize::MAX, u64::MAX));
            let state = self.worker_state(worker);
            if state.captured.is_none() {
                continue;
            }
            state.checkpoint = Checkpoint::ZERO;
            state.resume_cap = Some(0);
            // A zero count is exact: no record credits the worker on Device.
            if count > 0 && bound > 0 {
                pending.push(Pending {
                    worker,
                    bit,
                    bound: u32::try_from(bound).unwrap_or(u32::MAX),
                    resolved: None,
                });
            }
        }
        if pending.is_empty() {
            return;
        }
        let chain: Vec<_> = self.visited[tier_index(StorageTier::Device)]
            .first
            .iter()
            .enumerate()
            .filter_map(|(position, hash)| hash.map(|hash| (position, hash)))
            .collect();
        ForwardDeviceEvaluator::resolve(self.view.device_records(), &chain, &mut pending);
        for entry in pending {
            let Some(index) = entry.resolved else {
                continue;
            };
            let (position, hash) = chain[index as usize];
            if let Some(state) = self.workers.get_mut(&entry.worker) {
                state.checkpoint = Checkpoint { position, hash };
                state.resume_cap = Some(position);
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

    /// [`Self::finish_with_request`] without predicted checkpoints.
    #[cfg(test)]
    pub(crate) fn finish(
        &mut self,
        device: &mut MatchDetails,
        tiers: &mut HashMap<StorageTier, LowerTierMatchDetails>,
        snapshot: &ResidencyRoutingSnapshot,
    ) {
        self.finish_with_request(device, tiers, snapshot, None);
    }

    /// The sole credit decision, after tier walks and prediction merges. One
    /// endpoint P partitions the continuous GPU -> host -> disk -> external path.
    /// `request` is the query's block hashes when the device match includes
    /// predict-on-route scores. A hybrid worker whose score reaches past its
    /// confirmed checkpoint is then also credited up to its deepest predicted
    /// checkpoint on the request's own chain.
    pub(crate) fn finish_with_request(
        &mut self,
        device: &mut MatchDetails,
        tiers: &mut HashMap<StorageTier, LowerTierMatchDetails>,
        snapshot: &ResidencyRoutingSnapshot,
        request: Option<&[LocalBlockHash]>,
    ) {
        // The request's sequence hashes, computed only if a prediction is tried.
        let request_chain: OnceCell<Vec<SequenceHash>> = OnceCell::new();
        // Each hybrid worker's deepest checkpoint within its hits on a tier.
        let mut lower_points = FxHashMap::default();
        for (tier, details) in tiers.iter() {
            let mut queries = Vec::new();
            for (worker, hits) in &details.hits {
                let Some(next) = details.next_continuations.get(worker) else {
                    continue;
                };
                // Only a known hybrid worker is capped.
                if self.layout(ResidencyOwner::worker(*worker)).is_none() {
                    continue;
                }
                queries.push(LowerQuery {
                    worker: *worker,
                    end: next.start_pos.min(self.max_blocks),
                    floor: next.start_pos.saturating_sub(*hits),
                    point: Checkpoint::ZERO,
                    done: false,
                });
            }
            self.settle_lower(*tier, &mut queries);
            for query in queries {
                lower_points.insert((*tier, query.worker), query.point);
                let state = self.worker_state(query.worker);
                let cap = state.resume_cap.unwrap_or(0);
                state.resume_cap = Some(if query.point.position > query.floor {
                    cap.max(query.point.position)
                } else {
                    cap
                });
            }
        }
        device.overlap_scores.scores.retain(|worker, blocks| {
            let state = self.worker_state(*worker);
            if state.captured.is_none() {
                return true;
            }
            state.resume_cap.get_or_insert(0);
            let mut point = state.checkpoint;
            let reach = (*blocks as usize).min(self.max_blocks);
            if reach > point.position
                && let Some(request) = request
            {
                let chain = request_chain.get_or_init(|| compute_seq_hash_for_block(request));
                let predicted = self.view.predicted_checkpoint(*worker, reach, |position| {
                    chain
                        .get(position.checked_sub(1)?)
                        .map(|hash| ExternalSequenceBlockHash(*hash))
                });
                if let Some((position, hash)) = predicted
                    && position > point.position
                {
                    point = Checkpoint { position, hash };
                }
            }
            let cap = point.position;
            *blocks = (*blocks).min(cap as u32);
            if cap == 0 {
                device.last_matched_hashes.remove(worker);
            } else {
                device.last_matched_hashes.insert(*worker, point.hash);
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
                } else if let Some(point) = lower_points.get(&(*tier, *worker)) {
                    // The cap is at or past this checkpoint, so it is also the
                    // deepest one within the capped hits; no second read.
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
        self.visited[tier_index(self.tier)].record(position, hash);
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
    use crate::KvTransferExtensions;
    use crate::kv_hints::KvTransferSourceInterval;

    fn store_device_checkpoint(
        residency: &KvCacheGroupResidency,
        worker: WorkerWithDpRank,
        position: u64,
    ) {
        residency.apply(&crate::testing::recurrent_store(
            worker,
            StorageTier::Device,
            position,
        ));
    }

    #[test]
    fn device_candidates_preserve_raw_bounds_and_dp() {
        for case in [
            "bound",
            "bound_missing",
            "unsupported_alignment",
            "dp",
            "clear_readd",
        ] {
            let target = WorkerWithDpRank::new(64, u32::from(case == "dp"));
            let others: Vec<_> = (1..=63).map(|id| WorkerWithDpRank::new(id, 0)).collect();
            let residency = KvCacheGroupResidency::new();
            for &worker in &others {
                residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
                for position in 1..=8 {
                    store_device_checkpoint(&residency, worker, position);
                }
            }
            let alignment = if case == "unsupported_alignment" {
                2
            } else {
                1
            };
            residency.declare_layout(ResidencyOwner::worker(target), true, alignment);
            let positions: &[u64] = match case {
                "bound" => &[8, 3],
                "bound_missing" => &[8, 7],
                "unsupported_alignment" => &[7, 2],
                _ => &[999],
            };
            for &position in positions {
                store_device_checkpoint(&residency, target, position);
            }
            let wrong_dp = WorkerWithDpRank::new(64, 0);
            if case == "dp" {
                residency.declare_layout(ResidencyOwner::worker(wrong_dp), true, 1);
                for position in 1..=8 {
                    store_device_checkpoint(&residency, wrong_dp, position);
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
                    store_device_checkpoint(&residency, target, position);
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
        assert!(
            evaluator
                .visited
                .iter()
                .all(|tier| tier.first.capacity() == 0)
        );
    }

    #[test]
    fn device_only_side_predictions_keep_zero_caps_and_trim_hints() {
        let hybrid = WorkerWithDpRank::new(7, 0);
        let dense = WorkerWithDpRank::new(8, 0);
        let residency = KvCacheGroupResidency::new();
        residency.declare_layout(ResidencyOwner::worker(hybrid), true, 1);
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
            residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
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
        use crate::LowerTierContinuation;
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        for legacy_first in [false, true] {
            let residency = KvCacheGroupResidency::new();
            residency.declare_layout(owner, true, 1);
            let gpu = crate::testing::recurrent_store(worker, StorageTier::Device, 2);
            let host = crate::testing::recurrent_store(worker, StorageTier::HostPinned, 2);
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
    fn lower_tier_credit_ends_at_the_deepest_checkpoint_within_the_hits() {
        use crate::LowerTierContinuation;
        let worker = WorkerWithDpRank::new(7, 0);
        let hash = ExternalSequenceBlockHash;
        // The Device checkpoint is at block 2, and the host walk continues over
        // blocks 3 and 4. Host credit ends at the deepest host checkpoint among
        // those two; one at block 2 or above is outside the walk.
        for (host_checkpoints, expected_hits) in
            [(&[3, 4][..], 2), (&[3][..], 1), (&[2][..], 0), (&[][..], 0)]
        {
            let residency = KvCacheGroupResidency::new();
            residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
            store_device_checkpoint(&residency, worker, 2);
            for &position in host_checkpoints {
                residency.apply(&crate::testing::recurrent_store(
                    worker,
                    StorageTier::HostPinned,
                    position,
                ));
            }
            let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
            evaluator.observe_worker(worker);
            for position in 1..=2 {
                evaluator.observe_block(position, hash(position as u64));
            }
            let mut raw = MatchDetails::default();
            raw.overlap_scores.scores.insert(worker, 2);
            raw.last_matched_hashes.insert(worker, hash(2));
            evaluator.prepare_device(&raw);
            evaluator.on_tier(StorageTier::HostPinned);
            evaluator.observe_worker(worker);
            for position in 3..=4 {
                evaluator.observe_block(position, hash(position as u64));
            }
            let mut lower = HashMap::from([(
                StorageTier::HostPinned,
                LowerTierMatchDetails {
                    hits: FxHashMap::from_iter([(worker, 2)]),
                    next_continuations: FxHashMap::from_iter([(
                        worker,
                        LowerTierContinuation::new(4, hash(4)),
                    )]),
                    ..Default::default()
                },
            )]);
            evaluator.finish(&mut raw, &mut lower, &Default::default());

            let context = format!("host checkpoints {host_checkpoints:?}");
            let host = &lower[&StorageTier::HostPinned];
            assert_eq!(host.hits[&worker], expected_hits, "{context}");
            assert_eq!(
                host.next_continuations[&worker].start_pos,
                2 + expected_hits,
                "{context}"
            );
            assert_eq!(raw.overlap_scores.scores[&worker], 2, "{context}");
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&2), "{context}");
        }
    }

    #[test]
    fn lower_tier_walks_that_disagree_about_a_position_keep_both_hashes() {
        use crate::LowerTierContinuation;
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let hash = ExternalSequenceBlockHash;
        // Each worker reaches block 2 through its own hash and holds its host
        // checkpoint there.
        let walks = [(first, 21), (second, 22)];
        let residency = KvCacheGroupResidency::new();
        for (worker, at_two) in walks {
            residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
            residency.apply(&crate::testing::recurrent_store(
                worker,
                StorageTier::HostPinned,
                at_two,
            ));
        }
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.prepare_device(&MatchDetails::default());
        evaluator.on_tier(StorageTier::HostPinned);
        for (worker, at_two) in walks {
            evaluator.observe_worker(worker);
            evaluator.observe_block(1, hash(at_two - 10));
            evaluator.observe_block(2, hash(at_two));
            evaluator.observe_block(3, hash(at_two + 10));
        }
        let mut raw = MatchDetails::default();
        let mut lower = HashMap::from([(
            StorageTier::HostPinned,
            LowerTierMatchDetails {
                hits: FxHashMap::from_iter(walks.map(|(worker, _)| (worker, 3))),
                next_continuations: FxHashMap::from_iter(walks.map(|(worker, at_two)| {
                    (worker, LowerTierContinuation::new(3, hash(at_two + 10)))
                })),
                ..Default::default()
            },
        )]);
        evaluator.finish(&mut raw, &mut lower, &Default::default());

        let host = &lower[&StorageTier::HostPinned];
        for (worker, at_two) in walks {
            assert_eq!(host.hits[&worker], 2, "{worker:?}");
            assert_eq!(
                host.next_continuations[&worker].last_matched_hash,
                Some(hash(at_two)),
                "{worker:?}"
            );
        }
    }

    #[test]
    fn captured_absent_layout_is_not_relearned_when_entering_lower_tiers() {
        use crate::LowerTierContinuation;
        let worker = WorkerWithDpRank::new(7, 0);
        let residency = KvCacheGroupResidency::new();
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.observe_worker(worker);
        residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
        residency.apply(&crate::testing::recurrent_store(
            worker,
            StorageTier::HostPinned,
            2,
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
            residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
            let source = if stale { worker } else { other };
            let event = crate::testing::recurrent_store(source, StorageTier::Device, 2);
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
        use crate::prelude::{
            KvCacheEventData, KvCacheRemoveData, LocalBlockHash, RouterEvent,
            compute_seq_hash_for_block,
        };
        use crate::{RadixTree, ThreadPoolIndexer};

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
                    residency.declare_layout(ResidencyOwner::worker(worker), true, 1);
                }
                let checkpoint = |worker, position: usize| {
                    crate::testing::recurrent_store(
                        worker,
                        StorageTier::Device,
                        hashes[position - 1].0,
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
                    crate::testing::make_store_event(worker.worker_id, &[11, 12, 13])
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
        use crate::prelude::{ResidencyOwnerKeyExt, ResidencyProjection};
        use dynamo_router_protocol::identity::{
            CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };
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
        residency.declare_layout(ResidencyOwner::worker(hybrid), true, 1);
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

    #[tokio::test]
    async fn single_and_concurrent_use_only_complete_visited_checkpoint_boundaries() {
        use crate::backend::Indexer;
        use crate::prelude::*;
        use crate::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};
        use crate::{KvIndexer, KvIndexerMetrics, LowerTierIndexers};
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
            indexer.declare_recurrent_layout(worker, true, 1);
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
            let checkpoint = |hash, tier| {
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
                    kind: KvCacheGroupKind::RecurrentCheckpoint,
                    alignment_blocks: Some(1),
                })
            };
            // A state outside the matched attention path never proves reuse.
            indexer
                .try_apply_event(checkpoint(
                    ExternalSequenceBlockHash(999),
                    StorageTier::Device,
                ))
                .await
                .unwrap();
            // A checkpoint in host memory is not one on the device. Groups that
            // are split across tiers never get this far: the worker publishes a
            // checkpoint only once every group is resident in one tier.
            indexer
                .try_apply_event(checkpoint(hashes[1], StorageTier::HostPinned))
                .await
                .unwrap();
            assert!(
                indexer
                    .find_primary_match_details(locals.clone())
                    .await
                    .unwrap()
                    .overlap_scores
                    .scores
                    .is_empty(),
                "vLLM cannot resume on the GPU from a state held in host memory"
            );
            indexer
                .try_apply_event(checkpoint(hashes[1], StorageTier::Device))
                .await
                .unwrap();
            indexer
                .try_apply_event(checkpoint(hashes[5], StorageTier::Device))
                .await
                .unwrap();
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
}
