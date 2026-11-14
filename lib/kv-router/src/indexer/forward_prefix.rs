// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::{FxHashMap, FxHashSet};

use super::hybrid_prefix::HybridPrefixEvaluator;
use super::kv_cache_groups::{CheckpointWorkerSet, CheckpointWorkers};
use super::{KvCacheGroupResidency, MatchDetails, MatchEdge, MatchFirstEdge, MatchObserver};
use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

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

struct CurrentCheckpoint {
    workers: CheckpointWorkerSet,
    latest: Checkpoint,
}

#[derive(Clone, Copy)]
enum ForwardOverride {
    Live(Checkpoint),
    Frozen(Checkpoint),
}

impl ForwardOverride {
    fn is_frozen(self) -> bool {
        matches!(self, Self::Frozen(_))
    }

    fn point(self) -> Checkpoint {
        match self {
            Self::Live(point) | Self::Frozen(point) => point,
        }
    }
}

/// Consume current checkpoint worker sets during the forward attention walk.
/// Presence reads use a directory shard guard; subsequent matching uses owned
/// query-local sets. Like attention coverage, accumulated results are best-effort
/// during concurrent updates.
pub(crate) struct ForwardDeviceEvaluator<'a> {
    residency: &'a KvCacheGroupResidency,
    legacy: Option<Box<HybridPrefixEvaluator>>,
    view: Option<super::kv_cache_groups::KnownWorkerRead>,
    max_blocks: usize,
    current: Option<CurrentCheckpoint>,
    overrides: FxHashMap<WorkerWithDpRank, ForwardOverride>,
    cutoffs: Vec<(usize, WorkerWithDpRank)>,
    needs_checkpoints: bool,
    candidate_count: usize,
    default_cap_hint: u32,
    emitted: bool,
}

impl<'a> ForwardDeviceEvaluator<'a> {
    pub(crate) fn new(residency: &'a KvCacheGroupResidency, max_blocks: Option<usize>) -> Self {
        let unsupported = residency.requires_general_device_eval();
        Self {
            residency,
            legacy: unsupported
                .then(|| Box::new(HybridPrefixEvaluator::new(residency, max_blocks))),
            view: None,
            max_blocks: max_blocks.unwrap_or(usize::MAX),
            current: None,
            overrides: FxHashMap::default(),
            cutoffs: Vec::new(),
            needs_checkpoints: true,
            candidate_count: 0,
            default_cap_hint: 0,
            emitted: false,
        }
    }

    fn start(&mut self, count: usize) {
        if self.candidate_count == 0 {
            self.candidate_count = count;
        }
        if self.view.is_none()
            && self.legacy.is_none()
            && self.residency.requires_general_device_eval()
        {
            self.legacy = Some(Box::new(HybridPrefixEvaluator::new(
                self.residency,
                Some(self.max_blocks),
            )));
        }
        if let Some(legacy) = &mut self.legacy {
            legacy.reserve_workers(count);
        } else if self.view.is_none() {
            self.view = Some(self.residency.known_worker_read());
        }
    }

    fn known(&self, worker: WorkerWithDpRank) -> bool {
        self.view
            .as_ref()
            .is_some_and(|view| view.is_known_worker(worker))
    }

    fn accumulated_point(&self, worker: WorkerWithDpRank) -> Option<Checkpoint> {
        self.overrides
            .get(&worker)
            .map(|entry| entry.point())
            .or_else(|| {
                self.current.as_ref().and_then(|worker_set| {
                    worker_set
                        .workers
                        .contains(&worker)
                        .then_some(worker_set.latest)
                })
            })
    }

    fn freeze_worker(&mut self, worker: WorkerWithDpRank) {
        if self
            .overrides
            .get(&worker)
            .is_some_and(|entry| entry.is_frozen())
        {
            return;
        }
        if let Some(point) = self.accumulated_point(worker) {
            self.overrides
                .insert(worker, ForwardOverride::Frozen(point));
        }
    }

    fn transition_worker(
        overrides: &mut FxHashMap<WorkerWithDpRank, ForwardOverride>,
        previous: &CurrentCheckpoint,
        next: &FxHashSet<WorkerWithDpRank>,
        worker: WorkerWithDpRank,
    ) {
        if overrides
            .get(&worker)
            .is_some_and(|entry| entry.is_frozen())
        {
            return;
        }
        if next.contains(&worker) {
            overrides.remove(&worker);
        } else if previous.workers.contains(&worker) {
            overrides.insert(worker, ForwardOverride::Live(previous.latest));
        }
    }

    fn for_each_covered(
        edge: &MatchEdge<'_>,
        position: usize,
        mut visit: impl FnMut(WorkerWithDpRank),
    ) {
        let relative = position - edge.prefix_depth;
        if edge.first_node {
            for &worker in edge.full_workers {
                visit(worker);
            }
            for (&worker, &cutoff) in edge.worker_cutoffs {
                if relative <= cutoff {
                    visit(worker);
                }
            }
        } else {
            for &worker in edge.incoming_workers {
                if edge.full_workers.contains(&worker)
                    || edge
                        .worker_cutoffs
                        .get(&worker)
                        .is_some_and(|&cutoff| relative <= cutoff)
                {
                    visit(worker);
                }
            }
        }
    }

    fn is_covered(edge: &MatchEdge<'_>, position: usize, worker: WorkerWithDpRank) -> bool {
        (edge.first_node || edge.incoming_workers.contains(&worker))
            && (edge.full_workers.contains(&worker)
                || edge
                    .worker_cutoffs
                    .get(&worker)
                    .is_some_and(|&cutoff| position - edge.prefix_depth <= cutoff))
    }

    fn observe_checkpoint(
        &mut self,
        position: usize,
        hash: ExternalSequenceBlockHash,
        coverage: Option<&MatchEdge<'_>>,
    ) {
        while self
            .cutoffs
            .last()
            .is_some_and(|&(bound, _)| bound < position)
        {
            if let Some((_, worker)) = self.cutoffs.pop() {
                self.freeze_worker(worker);
            }
        }
        if position > self.max_blocks || !self.needs_checkpoints {
            return;
        }
        let current = &mut self.current;
        let overrides = &mut self.overrides;
        let default_cap_hint = &mut self.default_cap_hint;
        let majority = self.candidate_count.div_ceil(2);
        if let Some(update) = self
            .residency
            .checkpoint_workers_since(hash, current.as_ref().map(|current| &current.workers))
        {
            let point = Checkpoint { position, hash };
            let workers = match update {
                CheckpointWorkers::Unchanged => {
                    let previous = current
                        .as_mut()
                        .expect("unchanged membership requires a previous checkpoint");
                    if previous.workers.len() >= majority {
                        *default_cap_hint = position as u32;
                    }
                    previous.latest = point;
                    return;
                }
                CheckpointWorkers::Changed(workers) => workers,
            };
            if workers.len() >= majority {
                *default_cap_hint = position as u32;
            }
            if let Some(previous) = current.as_ref() {
                let mut visit =
                    |worker| Self::transition_worker(overrides, previous, &workers, worker);
                if let Some(edge) = coverage {
                    let candidates = if edge.first_node {
                        edge.full_workers.len() + edge.worker_cutoffs.len()
                    } else {
                        edge.incoming_workers.len()
                    };
                    if previous.workers.len().saturating_add(workers.len()) < candidates {
                        for &worker in previous.workers.iter().chain(workers.iter()) {
                            if Self::is_covered(edge, position, worker) {
                                visit(worker);
                            }
                        }
                    } else {
                        Self::for_each_covered(edge, position, visit);
                    }
                } else {
                    for &worker in previous.workers.iter().chain(workers.iter()) {
                        visit(worker);
                    }
                }
            }
            *current = Some(CurrentCheckpoint {
                workers,
                latest: point,
            });
        }
    }

    fn bounded_point(&self, worker: WorkerWithDpRank, raw_end: u32) -> Option<Checkpoint> {
        if !self.known(worker) {
            return None;
        }
        let Some(point) = self.accumulated_point(worker) else {
            return Some(Checkpoint::ZERO);
        };
        let bound = (raw_end as usize).min(self.max_blocks);
        Some(if point.position <= bound {
            point
        } else {
            Checkpoint::ZERO
        })
    }

    fn emit_supported(
        &mut self,
        mut active: FxHashSet<WorkerWithDpRank>,
        raw_end: u32,
        raw_hash: Option<ExternalSequenceBlockHash>,
        details: &mut MatchDetails,
    ) {
        self.emitted = true;
        let default_cap = self.default_cap_hint;
        let mut prior_known = Vec::new();
        let mut overrides = Vec::new();
        let last_hashes = &mut details.last_matched_hashes;
        details.overlap_scores.scores.retain(|worker, blocks| {
            let Some(point) = self.bounded_point(*worker, *blocks) else {
                return true;
            };
            prior_known.push(*worker);
            let cap = point.position as u32;
            if cap != default_cap {
                overrides.push((*worker, cap));
            }
            *blocks = cap;
            if cap == 0 {
                last_hashes.remove(worker);
            } else {
                last_hashes.insert(*worker, point.hash);
            }
            cap > 0
        });
        active.retain(|worker| {
            let Some(point) = self.bounded_point(*worker, raw_end) else {
                details.overlap_scores.scores.insert(*worker, raw_end);
                if let Some(hash) = raw_hash {
                    last_hashes.insert(*worker, hash);
                }
                return false;
            };
            let cap = point.position as u32;
            if cap != default_cap {
                overrides.push((*worker, cap));
            }
            if cap > 0 {
                details.overlap_scores.scores.insert(*worker, cap);
                last_hashes.insert(*worker, point.hash);
            } else {
                last_hashes.remove(worker);
            }
            true
        });
        active.extend(prior_known);
        self.view = None;
        details.hybrid_resume_caps =
            super::HybridResumeCaps::sparse_confirmed(active, default_cap, overrides);
    }

    pub(crate) fn finish(&mut self, details: &mut MatchDetails) {
        if let Some(legacy) = &mut self.legacy {
            legacy.prepare_device(details);
            legacy.finish(details, &mut Default::default(), &Default::default());
            return;
        }
        if !self.emitted {
            self.emit_supported(FxHashSet::default(), 0, None, details);
        }
        trim_forward_candidates(
            &mut details.kv_transfer_candidates,
            &details.hybrid_resume_caps,
        );
    }
}

impl MatchObserver for ForwardDeviceEvaluator<'_> {
    fn begin_edge(&mut self, edge: MatchEdge<'_>) -> bool {
        if edge.first_node {
            self.start(edge.full_workers.len() + edge.worker_cutoffs.len());
        }
        if self.legacy.is_some() {
            return false;
        }
        if edge.first_node {
            self.needs_checkpoints = self.max_blocks > 0
                && edge
                    .full_workers
                    .iter()
                    .chain(edge.worker_cutoffs.keys())
                    .any(|&worker| self.known(worker));
        }
        self.cutoffs.clear();
        if !self.needs_checkpoints {
            return true;
        }
        if !edge.first_node && !edge.all_workers_continue {
            for &worker in edge.incoming_workers {
                if !edge.full_workers.contains(&worker)
                    && !edge.worker_cutoffs.contains_key(&worker)
                {
                    self.freeze_worker(worker);
                }
            }
        }
        for (&worker, &cutoff) in edge.worker_cutoffs {
            if edge.first_node || edge.incoming_workers.contains(&worker) {
                self.cutoffs.push((edge.prefix_depth + cutoff, worker));
            }
        }
        self.cutoffs
            .sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
        true
    }

    fn reserve_workers(&mut self, count: usize) {
        self.start(count);
    }

    fn begin_path(&mut self, count: usize, _endpoint_hint: Option<ExternalSequenceBlockHash>) {
        self.start(count);
    }

    fn begin_first_edge(&mut self, edge: MatchFirstEdge<'_>) -> Option<usize> {
        if self.legacy.is_some() {
            return None;
        }
        self.begin_edge(MatchEdge {
            full_workers: edge.full_workers,
            worker_cutoffs: edge.worker_cutoffs,
            prefix_depth: 0,
            first_node: true,
            incoming_workers: &FxHashSet::default(),
            all_workers_continue: false,
        });
        None
    }

    fn observe_worker(&mut self, worker: WorkerWithDpRank) {
        if let Some(legacy) = &mut self.legacy {
            legacy.observe_worker(worker);
        }
    }

    fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
        if let Some(legacy) = &mut self.legacy {
            legacy.observe_block(position, hash);
        } else {
            self.observe_checkpoint(position, hash, None);
        }
    }

    fn observe_block_with_coverage(
        &mut self,
        position: usize,
        hash: ExternalSequenceBlockHash,
        edge: &MatchEdge<'_>,
    ) {
        if let Some(legacy) = &mut self.legacy {
            legacy.observe_block(position, hash);
        } else {
            self.observe_checkpoint(position, hash, Some(edge));
        }
    }

    fn emit_match_details(
        &mut self,
        active: FxHashSet<WorkerWithDpRank>,
        raw_end: u32,
        raw_hash: Option<ExternalSequenceBlockHash>,
        details: &mut MatchDetails,
    ) -> Result<(), FxHashSet<WorkerWithDpRank>> {
        if self.legacy.is_some() {
            return Err(active);
        }
        self.emit_supported(active, raw_end, raw_hash, details);
        Ok(())
    }
}

fn trim_forward_candidates(
    candidates: &mut Option<KvTransferCandidates>,
    caps: &super::HybridResumeCaps,
) {
    let Some(data) = candidates.as_mut() else {
        return;
    };
    let snapshot = data.routing_snapshot.as_deref();
    let cap_for = |source| {
        let worker = match source {
            KvTransferCandidateSource::Worker(worker) => Some(worker),
            KvTransferCandidateSource::CacheOwner(key) => {
                snapshot.and_then(|snapshot| snapshot.projection().project_key(key))
            }
        };
        worker.and_then(|worker| caps.get(&worker).copied())
    };
    data.owner_prefix_blocks.retain_mut(|(source, blocks)| {
        if let Some(cap) = cap_for(*source) {
            *blocks = (*blocks).min(cap as usize);
        }
        *blocks > 0
    });
    data.source_intervals.retain_mut(|interval| {
        if let Some(cap) = cap_for(interval.source) {
            interval.end = interval.end.min(cap as usize);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_hints::KvTransferSourceInterval;
    use crate::protocols::{ResidencyOwner, StorageTier};

    #[test]
    fn already_observed_checkpoint_uses_attention_best_effort_semantics() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        table.declare_groups(ResidencyOwner::worker(worker), &[0]);
        store_device_checkpoint(&table, worker, 1, 1);
        let raw = raw_device([(worker, 1)].into_iter());
        let mut observer = ForwardDeviceEvaluator::new(&table, None);
        begin_forward(&mut observer, &raw, 1);
        observer.observe_block(1, ExternalSequenceBlockHash(1));
        table.clear_worker(worker);
        let mut already_read = raw.clone();
        observer.finish(&mut already_read);
        assert_eq!(already_read.hybrid_resume_caps.get(&worker), Some(&1));

        // A later query reads the cleared record; an earlier observation is
        // not a lease on the physical checkpoint, just like attention coverage.
        let (after_clear, _) = forward_and_legacy(&table, &raw, None, 1);
        assert_eq!(after_clear.hybrid_resume_caps.get(&worker), Some(&0));
        assert!(!after_clear.overlap_scores.scores.contains_key(&worker));
    }

    #[test]
    fn late_worker_registration_keeps_initial_unknown_classification() {
        let table = KvCacheGroupResidency::new();
        let known = WorkerWithDpRank::new(1, 0);
        let unknown = WorkerWithDpRank::new(2, 0);
        table.declare_groups(ResidencyOwner::worker(known), &[0]);
        store_device_checkpoint(&table, known, 1, 1);
        let mut raw = raw_device([(known, 1), (unknown, 1)].into_iter());
        let mut observer = ForwardDeviceEvaluator::new(&table, None);
        begin_forward(&mut observer, &raw, 1);
        observer.observe_block(1, ExternalSequenceBlockHash(1));
        table.declare_groups(ResidencyOwner::worker(unknown), &[0]);
        store_device_checkpoint(&table, unknown, 1, 1);
        observer.finish(&mut raw);
        assert_eq!(raw.hybrid_resume_caps.get(&known), Some(&1));
        assert_eq!(raw.hybrid_resume_caps.get(&unknown), None);
        assert_eq!(raw.overlap_scores.scores.get(&unknown), Some(&1));
    }

    #[test]
    fn owned_checkpoint_sets_preserve_equal_sized_changes_and_missing_boundaries() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(1, 1);
        for worker in [first, second] {
            table.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        store_device_checkpoint(&table, first, 1, 1);
        store_device_checkpoint(&table, first, 2, 1);
        // B3 is absent; B4 has the same count but a different DP rank.
        store_device_checkpoint(&table, second, 4, 1);
        store_device_checkpoint(&table, second, 5, 1);
        let raw = raw_device([(first, 6), (second, 6)].into_iter());
        let (actual, expected) = forward_and_legacy(&table, &raw, None, 6);
        assert_forward_parity(&actual, &expected);
        for (worker, position) in [(first, 2), (second, 5)] {
            assert_eq!(actual.overlap_scores.scores.get(&worker), Some(&position));
            assert_eq!(actual.hybrid_resume_caps.get(&worker), Some(&position));
            assert_eq!(
                actual.last_matched_hashes.get(&worker),
                Some(&ExternalSequenceBlockHash(u64::from(position)))
            );
        }
    }

    #[test]
    fn sparse_forward_results_match_independent_checkpoint_oracle() {
        for seed in 0u64..12 {
            let table = KvCacheGroupResidency::new();
            let mut checkpoints = FxHashMap::default();
            let mut raw = MatchDetails::default();
            for id in 1u64..=24 {
                let worker = WorkerWithDpRank::new(id, (id % 3) as u32);
                table.declare_groups(ResidencyOwner::worker(worker), &[0]);
                let bound = 1 + ((id * 7 + seed * 3) % 32) as u32;
                raw.overlap_scores.scores.insert(worker, bound);
                raw.last_matched_hashes
                    .insert(worker, ExternalSequenceBlockHash(bound as u64));
                let points: Vec<_> = (1u64..=32)
                    .filter(|&p| (p * 13 + id * 11 + seed * 5).is_multiple_of(7))
                    .collect();
                for &point in &points {
                    store_device_checkpoint(&table, worker, point, 1);
                }
                checkpoints.insert(worker, points);
            }
            let unknown = WorkerWithDpRank::new(99, 0);
            raw.overlap_scores.scores.insert(unknown, 32);
            for limit in [0, 3, 17, 32] {
                let (actual, _) = forward_and_legacy(&table, &raw, Some(limit), 32);
                for (&worker, points) in &checkpoints {
                    let bound = (raw.overlap_scores.scores[&worker] as usize).min(limit);
                    let expected = points
                        .iter()
                        .copied()
                        .filter(|&p| p as usize <= bound)
                        .max()
                        .unwrap_or(0) as u32;
                    assert_eq!(
                        actual.hybrid_resume_caps.get(&worker),
                        Some(&expected),
                        "seed={seed} limit={limit} worker={worker:?}"
                    );
                    assert_eq!(
                        actual
                            .overlap_scores
                            .scores
                            .get(&worker)
                            .copied()
                            .unwrap_or(0),
                        expected
                    );
                    assert_eq!(
                        actual.last_matched_hashes.get(&worker).copied(),
                        (expected > 0).then_some(ExternalSequenceBlockHash(expected as u64))
                    );
                }
                assert_eq!(actual.hybrid_resume_caps.get(&unknown), None);
                assert_eq!(actual.overlap_scores.scores.get(&unknown), Some(&32));
            }
        }
    }
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

    fn forward_and_legacy(
        residency: &KvCacheGroupResidency,
        raw: &MatchDetails,
        max_blocks: Option<usize>,
        visited: usize,
    ) -> (MatchDetails, MatchDetails) {
        let mut forward = ForwardDeviceEvaluator::new(residency, max_blocks);
        let mut legacy = HybridPrefixEvaluator::new(residency, max_blocks);
        let (full_workers, worker_cutoffs) = begin_forward(&mut forward, raw, visited);
        legacy.reserve_workers(raw.overlap_scores.scores.len());
        for &worker in raw.overlap_scores.scores.keys() {
            forward.observe_worker(worker);
            legacy.observe_worker(worker);
        }
        for position in 1..=visited {
            let hash = ExternalSequenceBlockHash(position as u64);
            forward.observe_block_with_coverage(
                position,
                hash,
                &MatchEdge {
                    full_workers: &full_workers,
                    worker_cutoffs: &worker_cutoffs,
                    incoming_workers: &FxHashSet::default(),
                    prefix_depth: 0,
                    first_node: true,
                    all_workers_continue: false,
                },
            );
            legacy.observe_block(position, hash);
        }
        let mut actual = raw.clone();
        let mut expected = raw.clone();
        forward.finish(&mut actual);
        legacy.prepare_device(&expected);
        legacy.finish(&mut expected, &mut Default::default(), &Default::default());
        (actual, expected)
    }

    fn begin_forward(
        evaluator: &mut ForwardDeviceEvaluator<'_>,
        raw: &MatchDetails,
        visited: usize,
    ) -> (
        FxHashSet<WorkerWithDpRank>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        let mut full_workers = FxHashSet::default();
        let mut worker_cutoffs = FxHashMap::default();
        for (&worker, &blocks) in &raw.overlap_scores.scores {
            if blocks as usize >= visited {
                full_workers.insert(worker);
            } else {
                worker_cutoffs.insert(worker, blocks as usize);
            }
        }
        evaluator.begin_edge(MatchEdge {
            full_workers: &full_workers,
            worker_cutoffs: &worker_cutoffs,
            prefix_depth: 0,
            first_node: true,
            incoming_workers: &FxHashSet::default(),
            all_workers_continue: false,
        });
        (full_workers, worker_cutoffs)
    }

    fn raw_device(workers: impl Iterator<Item = (WorkerWithDpRank, u32)>) -> MatchDetails {
        let mut raw = MatchDetails::default();
        for (worker, blocks) in workers {
            raw.overlap_scores.scores.insert(worker, blocks);
            if blocks > 0 {
                raw.last_matched_hashes
                    .insert(worker, ExternalSequenceBlockHash(blocks as u64));
            }
        }
        raw
    }

    fn assert_forward_parity(actual: &MatchDetails, expected: &MatchDetails) {
        assert_eq!(actual.overlap_scores.scores, expected.overlap_scores.scores);
        assert_eq!(actual.last_matched_hashes, expected.last_matched_hashes);
        assert_eq!(actual.hybrid_resume_caps, expected.hybrid_resume_caps);
    }

    #[test]
    fn forward_device_dense_worker_set_matches_legacy_with_off_chain_missing_workers_and_limits() {
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        let unknown = WorkerWithDpRank::new(65, 0);
        for missing in [1, 32, 63, 64] {
            let residency = KvCacheGroupResidency::new();
            for (index, &worker) in workers.iter().enumerate() {
                residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
                if index >= workers.len() - missing {
                    store_device_checkpoint(&residency, worker, 999, 1);
                } else {
                    for position in 1..=8 {
                        store_device_checkpoint(&residency, worker, position, 1);
                    }
                }
            }
            let raw = raw_device(
                workers
                    .iter()
                    .copied()
                    .map(|worker| (worker, 8))
                    .chain([(unknown, 8)]),
            );
            for limit in [None, Some(0), Some(7), Some(8)] {
                let (actual, expected) = forward_and_legacy(&residency, &raw, limit, 8);
                assert_forward_parity(&actual, &expected);
                assert_eq!(actual.hybrid_resume_caps.len(), 64);
                assert_eq!(actual.hybrid_resume_caps.get(&unknown), None);
                assert_eq!(actual.overlap_scores.scores.get(&unknown), Some(&8));
            }
        }
    }

    #[test]
    fn forward_device_selects_earlier_checkpoint_inside_worker_raw_bounds() {
        let residency = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(1, 1);
        for worker in [first, second] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        for position in [2, 6] {
            store_device_checkpoint(&residency, first, position, 1);
        }
        for position in [4, 8] {
            store_device_checkpoint(&residency, second, position, 1);
        }
        for raw_first in [1, 3, 5, 8] {
            let raw = raw_device([(first, raw_first), (second, 8)].into_iter());
            for limit in [None, Some(4), Some(7)] {
                let (actual, expected) = forward_and_legacy(&residency, &raw, limit, 8);
                assert_forward_parity(&actual, &expected);
            }
        }

        let worker_set = KvCacheGroupResidency::new();
        for worker in [first, second] {
            worker_set.declare_groups(ResidencyOwner::worker(worker), &[0]);
            for position in [4, 8] {
                store_device_checkpoint(&worker_set, worker, position, 1);
            }
        }
        let raw = raw_device([(first, 3), (second, 8)].into_iter());
        let (actual, expected) = forward_and_legacy(&worker_set, &raw, None, 8);
        assert_forward_parity(&actual, &expected);
        assert_eq!(actual.hybrid_resume_caps.get(&first), Some(&0));
        assert_eq!(actual.hybrid_resume_caps.get(&second), Some(&8));
    }

    #[test]
    fn forward_device_freezes_a_shared_worker_set_at_each_attention_cutoff() {
        let residency = KvCacheGroupResidency::new();
        let short = WorkerWithDpRank::new(1, 0);
        let long = WorkerWithDpRank::new(2, 0);
        for worker in [short, long] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            for position in [4, 8] {
                store_device_checkpoint(&residency, worker, position, 1);
            }
        }
        let raw = raw_device([(short, 4), (long, 8)].into_iter());
        let (actual, expected) = forward_and_legacy(&residency, &raw, None, 8);
        assert_forward_parity(&actual, &expected);
        assert_eq!(actual.hybrid_resume_caps.get(&short), Some(&4));
        assert_eq!(actual.hybrid_resume_caps.get(&long), Some(&8));
        assert_eq!(
            actual.last_matched_hashes[&short],
            ExternalSequenceBlockHash(4)
        );
    }

    #[test]
    fn forward_device_dropped_before_first_checkpoint_keeps_zero_credit() {
        let residency = KvCacheGroupResidency::new();
        let dropped = WorkerWithDpRank::new(1, 0);
        let surviving = WorkerWithDpRank::new(2, 0);
        let first_workers = FxHashSet::from_iter([dropped, surviving]);
        let last_workers = FxHashSet::from_iter([surviving]);
        let cutoffs = FxHashMap::default();
        for worker in [dropped, surviving] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            store_device_checkpoint(&residency, worker, 8, 1);
        }
        let mut evaluator = ForwardDeviceEvaluator::new(&residency, None);
        let first = MatchEdge {
            full_workers: &first_workers,
            worker_cutoffs: &cutoffs,
            incoming_workers: &FxHashSet::default(),
            prefix_depth: 0,
            first_node: true,
            all_workers_continue: false,
        };
        assert!(evaluator.begin_edge(first));
        for position in 1..=4 {
            evaluator.observe_block_with_coverage(
                position,
                ExternalSequenceBlockHash(position as u64),
                &first,
            );
        }
        let last = MatchEdge {
            full_workers: &last_workers,
            worker_cutoffs: &cutoffs,
            incoming_workers: &first_workers,
            prefix_depth: 4,
            first_node: false,
            all_workers_continue: false,
        };
        assert!(evaluator.begin_edge(last));
        for position in 5..=8 {
            evaluator.observe_block_with_coverage(
                position,
                ExternalSequenceBlockHash(position as u64),
                &last,
            );
        }
        assert_eq!(evaluator.overrides.capacity(), 0);
        let mut raw = raw_device([(dropped, 4), (surviving, 8)].into_iter());
        evaluator.finish(&mut raw);
        assert_eq!(raw.hybrid_resume_caps.get(&dropped), Some(&0));
        assert_eq!(raw.hybrid_resume_caps.get(&surviving), Some(&8));
        assert!(!raw.overlap_scores.scores.contains_key(&dropped));
    }

    #[test]
    fn forward_device_shared_worker_set_preserves_many_distinct_worker_cutoffs() {
        let residency = KvCacheGroupResidency::new();
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        for &worker in &workers {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            for position in 1..=64 {
                store_device_checkpoint(&residency, worker, position, 1);
            }
        }
        let raw = raw_device(
            workers
                .iter()
                .enumerate()
                .map(|(index, &worker)| (worker, index as u32 + 1)),
        );
        let (actual, expected) = forward_and_legacy(&residency, &raw, None, 64);
        assert_forward_parity(&actual, &expected);
        for (index, worker) in workers.iter().enumerate() {
            assert_eq!(
                actual.hybrid_resume_caps.get(worker),
                Some(&(index as u32 + 1))
            );
        }
    }

    #[test]
    fn forward_device_freezes_dropped_workers_before_observing_the_next_edge() {
        let residency = KvCacheGroupResidency::new();
        let dropped = WorkerWithDpRank::new(1, 0);
        let surviving = WorkerWithDpRank::new(2, 0);
        for worker in [dropped, surviving] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            for position in [4, 8] {
                store_device_checkpoint(&residency, worker, position, 1);
            }
        }
        let mut evaluator = ForwardDeviceEvaluator::new(&residency, None);
        let full = FxHashSet::from_iter([dropped, surviving]);
        let partial = FxHashMap::default();
        assert!(evaluator.begin_edge(MatchEdge {
            full_workers: &full,
            worker_cutoffs: &partial,
            prefix_depth: 0,
            first_node: true,
            incoming_workers: &FxHashSet::default(),
            all_workers_continue: false,
        }));
        for position in 1..=4 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        assert!(evaluator.begin_edge(MatchEdge {
            full_workers: &FxHashSet::from_iter([surviving]),
            worker_cutoffs: &partial,
            prefix_depth: 4,
            incoming_workers: &full,
            first_node: false,
            all_workers_continue: false,
        }));
        for position in 5..=6 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        assert!(evaluator.begin_edge(MatchEdge {
            full_workers: &FxHashSet::from_iter([surviving]),
            worker_cutoffs: &partial,
            prefix_depth: 6,
            incoming_workers: &FxHashSet::from_iter([surviving]),
            first_node: false,
            all_workers_continue: true,
        }));
        for position in 7..=8 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        let mut actual = raw_device([(dropped, 4), (surviving, 8)].into_iter());
        evaluator.finish(&mut actual);
        assert_eq!(actual.hybrid_resume_caps.get(&dropped), Some(&4));
        assert_eq!(actual.hybrid_resume_caps.get(&surviving), Some(&8));
        assert_eq!(
            actual.last_matched_hashes[&dropped],
            ExternalSequenceBlockHash(4)
        );
    }

    #[test]
    fn forward_device_updates_overlapping_worker_sets_without_revising_frozen_points() {
        let residency = KvCacheGroupResidency::new();
        let short = WorkerWithDpRank::new(1, 0);
        let shared = WorkerWithDpRank::new(2, 0);
        let other = WorkerWithDpRank::new(3, 0);
        for worker in [short, shared, other] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        for position in [4, 8] {
            for worker in [short, shared] {
                store_device_checkpoint(&residency, worker, position, 1);
            }
        }
        for worker in [shared, other] {
            store_device_checkpoint(&residency, worker, 6, 1);
        }
        let raw = raw_device([(short, 4), (shared, 8), (other, 8)].into_iter());
        let (actual, expected) = forward_and_legacy(&residency, &raw, None, 8);
        assert_forward_parity(&actual, &expected);
        assert_eq!(actual.hybrid_resume_caps.get(&short), Some(&4));
        assert_eq!(actual.hybrid_resume_caps.get(&shared), Some(&8));
        assert_eq!(actual.hybrid_resume_caps.get(&other), Some(&6));
    }

    #[test]
    fn forward_device_rotating_sparse_worker_sets_respect_candidate_cutoffs() {
        let residency = KvCacheGroupResidency::new();
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        for &worker in &workers {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            store_device_checkpoint(&residency, worker, 999, 1);
        }
        for position in 1..=8 {
            store_device_checkpoint(
                &residency,
                workers[(position as usize - 1) % 4],
                position,
                1,
            );
        }
        let raw = raw_device(
            workers
                .iter()
                .enumerate()
                .map(|(index, &worker)| (worker, if index == 0 { 3 } else { 8 })),
        );
        let (actual, expected) = forward_and_legacy(&residency, &raw, None, 8);
        assert_forward_parity(&actual, &expected);
        for (index, cap) in [1, 6, 7, 8].into_iter().enumerate() {
            assert_eq!(actual.hybrid_resume_caps.get(&workers[index]), Some(&cap));
        }
        for worker in &workers[4..] {
            assert_eq!(actual.hybrid_resume_caps.get(worker), Some(&0));
        }
    }

    #[test]
    fn forward_device_unsupported_alignment_delegates_the_original_same_walk() {
        let residency = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        store_device_checkpoint(&residency, worker, 2, 2);
        let raw = raw_device([(worker, 3)].into_iter());
        let (actual, expected) = forward_and_legacy(&residency, &raw, None, 3);
        assert_forward_parity(&actual, &expected);
        assert_eq!(actual.hybrid_resume_caps.get(&worker), Some(&2));
    }

    #[test]
    fn forward_device_first_edge_requires_exact_workers_and_reachable_bound() {
        let residency = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let wrong = WorkerWithDpRank::new(3, 0);
        for worker in [first, second, wrong] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        for worker in [first, wrong] {
            store_device_checkpoint(&residency, worker, 3, 1);
        }
        store_device_checkpoint(&residency, second, 999, 1);
        let workers = FxHashSet::from_iter([first, second]);
        let blocks: Vec<_> = (1..=8)
            .map(|hash| {
                (
                    crate::protocols::LocalBlockHash(hash),
                    ExternalSequenceBlockHash(hash),
                )
            })
            .collect();
        let mut evaluator = ForwardDeviceEvaluator::new(&residency, Some(3));
        assert_eq!(
            evaluator.begin_first_edge(MatchFirstEdge {
                full_workers: &workers,
                worker_cutoffs: &FxHashMap::default(),
                matched_blocks: &blocks,
                query_blocks: 8,
                requested_endpoint: Some(ExternalSequenceBlockHash(8)),
                terminal: true,
            }),
            None,
            "equal cardinality cannot substitute a different eligible worker"
        );
        for position in 1..=8 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        let mut details = MatchDetails::default();
        assert!(
            evaluator
                .emit_match_details(workers, 8, Some(ExternalSequenceBlockHash(8)), &mut details)
                .is_ok()
        );
        evaluator.finish(&mut details);
        assert_eq!(details.hybrid_resume_caps.get(&first), Some(&3));
        assert_eq!(details.hybrid_resume_caps.get(&second), Some(&0));

        store_device_checkpoint(&residency, second, 3, 1);
        for cutoff in [2, 3] {
            let mut evaluator = ForwardDeviceEvaluator::new(&residency, Some(3));
            let full = FxHashSet::from_iter([first]);
            let partial = FxHashMap::from_iter([(second, cutoff)]);
            assert!(evaluator.begin_edge(MatchEdge {
                full_workers: &full,
                worker_cutoffs: &partial,
                prefix_depth: 0,
                first_node: true,
                incoming_workers: &FxHashSet::default(),
                all_workers_continue: false,
            }));
            for position in 1..=8 {
                evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
            }
            let mut raw = raw_device([(first, 8), (second, cutoff as u32)].into_iter());
            evaluator.finish(&mut raw);
            assert_eq!(raw.hybrid_resume_caps.get(&first), Some(&3));
            assert_eq!(
                raw.hybrid_resume_caps.get(&second),
                Some(&if cutoff >= 3 { 3 } else { 0 })
            );
        }
    }

    #[test]
    fn forward_device_fused_emission_preserves_partial_unknowns_caps_and_hints() {
        let residency = KvCacheGroupResidency::new();
        let partial = WorkerWithDpRank::new(1, 0);
        let full = WorkerWithDpRank::new(2, 0);
        let unknown_full = WorkerWithDpRank::new(3, 0);
        let unknown_partial = WorkerWithDpRank::new(4, 0);
        for worker in [partial, full] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            for position in [4, 8] {
                store_device_checkpoint(&residency, worker, position, 1);
            }
        }
        for limit in [None, Some(0), Some(7), Some(8)] {
            let mut raw = raw_device(
                [
                    (partial, 3),
                    (full, 8),
                    (unknown_full, 8),
                    (unknown_partial, 5),
                ]
                .into_iter(),
            );
            raw.kv_transfer_candidates = Some(KvTransferCandidates {
                block_hashes: (1..=8).map(ExternalSequenceBlockHash).collect(),
                owner_prefix_blocks: vec![
                    (partial.into(), 3),
                    (full.into(), 8),
                    (unknown_full.into(), 8),
                    (unknown_partial.into(), 5),
                ],
                source_intervals: vec![KvTransferSourceInterval {
                    source: full.into(),
                    start: 0,
                    end: 8,
                }],
                routing_snapshot: None,
            });
            let (_, expected) = forward_and_legacy(&residency, &raw, limit, 8);
            let mut evaluator = ForwardDeviceEvaluator::new(&residency, limit);
            begin_forward(&mut evaluator, &raw, 8);
            for position in 1..=8 {
                evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
            }
            let mut details = raw.clone();
            for worker in [full, unknown_full] {
                details.overlap_scores.scores.remove(&worker);
                details.last_matched_hashes.remove(&worker);
            }
            assert!(
                evaluator
                    .emit_match_details(
                        FxHashSet::from_iter([full, unknown_full]),
                        8,
                        Some(ExternalSequenceBlockHash(8)),
                        &mut details,
                    )
                    .is_ok()
            );
            evaluator.finish(&mut details);
            assert_forward_parity(&details, &expected);
            assert_eq!(details.hybrid_resume_caps.len(), 2);
            assert_eq!(details.hybrid_resume_caps.get(&partial), Some(&0));
            assert_eq!(details.hybrid_resume_caps.get(&unknown_partial), None);
            let actual_hints = details.kv_transfer_candidates.unwrap();
            let expected_hints = expected.kv_transfer_candidates.unwrap();
            assert_eq!(actual_hints.block_hashes, expected_hints.block_hashes);
            assert_eq!(
                actual_hints.owner_prefix_blocks,
                expected_hints.owner_prefix_blocks
            );
            assert_eq!(
                actual_hints.source_intervals,
                expected_hints.source_intervals
            );
        }
    }

    #[tokio::test]
    async fn forward_device_empty_and_root_miss_keep_uninitialized_scope() {
        use crate::ConcurrentRadixTreeCompressed;
        use crate::indexer::ThreadPoolIndexer;
        use crate::protocols::LocalBlockHash;

        let worker = WorkerWithDpRank::new(7, 0);
        let residency = KvCacheGroupResidency::new();
        residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        store_device_checkpoint(&residency, worker, 3, 1);
        let primary = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 4);
        primary
            .apply_event_and_wait(crate::test_utils::make_store_event(7, &[11, 12, 13]))
            .await
            .unwrap();
        for query in [Vec::new(), vec![LocalBlockHash(999)]] {
            for limit in [None, Some(0)] {
                for retain in [false, true] {
                    let mut forward = ForwardDeviceEvaluator::new(&residency, limit);
                    let mut legacy = HybridPrefixEvaluator::new(&residency, limit);
                    let mut actual = primary.backend().find_match_details_impl_with_observer(
                        &query,
                        false,
                        retain,
                        &mut forward,
                    );
                    let mut expected = primary.backend().find_match_details_impl_with_observer(
                        &query,
                        false,
                        retain,
                        &mut legacy,
                    );
                    forward.finish(&mut actual);
                    legacy.prepare_device(&expected);
                    legacy.finish(&mut expected, &mut Default::default(), &Default::default());
                    assert_forward_parity(&actual, &expected);
                    assert!(actual.overlap_scores.scores.is_empty());
                    assert!(actual.last_matched_hashes.is_empty());
                    assert!(actual.hybrid_resume_caps.is_empty());
                    assert!(actual.kv_transfer_candidates.is_none());
                    assert!(forward.view.is_none());
                    assert!(forward.current.is_none());
                }
            }
        }
    }
}
