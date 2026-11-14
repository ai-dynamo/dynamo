// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::{FxHashMap, FxHashSet};

use super::hybrid_prefix::HybridPrefixEvaluator;
use super::{KvCacheGroupResidency, MatchDetails, MatchEdge, MatchFirstEdge, MatchObserver};
use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use crate::prelude::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// One known hybrid candidate waiting for its resume point.
struct Pending {
    worker: WorkerWithDpRank,
    /// The worker's Device filter bit.
    bit: u64,
    /// Attention coverage end, capped by the query limit.
    bound: u32,
    /// Index into the recorded chain of the settled checkpoint.
    resolved: Option<u32>,
}

/// One bit per candidate slot. Up to 256 slots stay on the stack, and a larger
/// set is sized up front, so marking a slot never allocates.
struct SlotBitmap {
    inline: [u64; 4],
    spilled: Vec<u64>,
    marked: usize,
}

impl SlotBitmap {
    fn new(slots: usize) -> Self {
        let words = slots.div_ceil(64);
        Self {
            inline: [0; 4],
            spilled: if words > 4 {
                vec![0; words]
            } else {
                Vec::new()
            },
            marked: 0,
        }
    }

    /// Mark a slot that is not marked yet.
    fn mark(&mut self, slot: usize) {
        let words: &mut [u64] = if self.spilled.is_empty() {
            &mut self.inline
        } else {
            &mut self.spilled
        };
        words[slot / 64] |= 1 << (slot % 64);
        self.marked += 1;
    }

    fn is_marked(&self, slot: usize) -> bool {
        let words: &[u64] = if self.spilled.is_empty() {
            &self.inline
        } else {
            &self.spilled
        };
        (words[slot / 64] >> (slot % 64)) & 1 == 1
    }

    fn marked(&self) -> usize {
        self.marked
    }
}

/// Resolve hybrid resume points once the attention walk has finished.
///
/// The walk only records the matched external hashes. When every candidate's
/// attention bound is known, checkpoint records are read from the deepest
/// bound downward: one read settles every unsettled worker present at that
/// hash, and reading stops as soon as no candidate is left. A query whose
/// candidates all hold a checkpoint at their attention end therefore reads
/// one record per distinct end rather than one per matched block.
///
/// A read first asks the Device filter whether the record can hold any worker
/// that is still unsettled, so positions without one cost a single atomic
/// load and never touch the directory.
///
/// Reads take a directory shard guard and see current state. Like attention
/// coverage, results are best-effort during concurrent updates. Nothing
/// allocates while a guard is held: readers of one record otherwise collapse
/// under concurrency.
pub(crate) struct ForwardDeviceEvaluator<'a> {
    residency: &'a KvCacheGroupResidency,
    legacy: Option<Box<HybridPrefixEvaluator>>,
    view: Option<super::kv_cache_groups::KnownWorkerRead>,
    max_blocks: usize,
    /// Matched `(position, hash)` pairs in ascending one-based positions.
    chain: Vec<(usize, ExternalSequenceBlockHash)>,
    needs_checkpoints: bool,
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
            chain: Vec::new(),
            needs_checkpoints: true,
            emitted: false,
        }
    }

    fn start(&mut self, count: usize) {
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

    /// Device checkpoint count and filter bit; `None` for a worker that was
    /// unknown when the walk started.
    fn device_summary(&self, worker: WorkerWithDpRank) -> Option<(usize, u64)> {
        self.view
            .as_ref()
            .and_then(|view| view.device_summary(worker))
    }

    fn record_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
        if !self.needs_checkpoints || position > self.max_blocks {
            return;
        }
        // Positions arrive in ascending order; keep that true if a tree ever
        // revisits one, because resolution searches the chain by position.
        while self.chain.last().is_some_and(|&(last, _)| last >= position) {
            self.chain.pop();
        }
        self.chain.push((position, hash));
    }

    /// Settle candidates against current checkpoint records, deepest bound
    /// first. A worker's resume point is its deepest recorded position within
    /// its bound whose record holds the worker; unsettled workers get none.
    fn resolve(&self, pending: &mut [Pending]) {
        let Some(first) = pending.first() else {
            return;
        };
        // Candidates that survive the whole walk share one bound, so the
        // input is usually uniform already.
        let first_bound = first.bound;
        if pending.iter().any(|entry| entry.bound != first_bound) {
            pending.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.bound));
        }
        let deepest = pending[0].bound as usize;
        // Slots that have reached their bound; settled ones are dropped lazily.
        let mut open: Vec<u32> = Vec::with_capacity(pending.len());
        let mut unsettled = 0usize;
        // Filter bits of the open slots: a superset once some have settled,
        // rebuilt after each read that settles one.
        let mut open_bits = 0u64;
        let mut next = 0usize;
        // Built after the first record with few holders meets many open
        // slots, outside its shard guard, and used from the next record on.
        let mut slots: Option<FxHashMap<WorkerWithDpRank, u32>> = None;
        let mut sparse = false;
        // Members of the last populated record that every open slot was
        // checked against. An equal set cannot hold any of them.
        let mut checked = None;
        let mut index = self
            .chain
            .partition_point(|&(position, _)| position <= deepest);
        while index > 0 {
            let (position, hash) = self.chain[index - 1];
            while let Some(entry) = pending.get(next)
                && entry.bound as usize >= position
            {
                open.push(next as u32);
                open_bits |= entry.bit;
                next += 1;
                unsettled += 1;
                checked = None;
            }
            if unsettled == 0 {
                let Some(entry) = pending.get(next) else {
                    break;
                };
                // Nobody can use the positions above the next candidate's bound.
                let bound = entry.bound as usize;
                index = self.chain[..index - 1].partition_point(|&(position, _)| position <= bound);
                open.clear();
                open_bits = 0;
                continue;
            }
            let at = (index - 1) as u32;
            let before = unsettled;
            self.residency
                .visit_checkpoint_workers(hash, open_bits, |workers| {
                    let signature = workers.signature();
                    if checked == Some(signature) {
                        return;
                    }
                    checked = Some(signature);
                    let few_holders = workers.len() * 2 < unsettled;
                    if few_holders && let Some(slots) = &slots {
                        for worker in workers.iter() {
                            if let Some(&slot) = slots.get(worker)
                                && (slot as usize) < next
                                && pending[slot as usize].resolved.is_none()
                            {
                                pending[slot as usize].resolved = Some(at);
                                unsettled -= 1;
                            }
                        }
                    } else {
                        sparse |= few_holders;
                        open.retain(|&slot| {
                            let entry = &mut pending[slot as usize];
                            if entry.resolved.is_some() {
                                return false;
                            }
                            if workers.contains(&entry.worker) {
                                entry.resolved = Some(at);
                                unsettled -= 1;
                                return false;
                            }
                            true
                        });
                    }
                });
            if sparse && slots.is_none() {
                slots = Some(
                    pending
                        .iter()
                        .enumerate()
                        .map(|(slot, entry)| (entry.worker, slot as u32))
                        .collect(),
                );
            }
            if unsettled != before {
                open_bits = open
                    .iter()
                    .map(|&slot| &pending[slot as usize])
                    .filter(|entry| entry.resolved.is_none())
                    .fold(0, |bits, entry| bits | entry.bit);
            }
            index -= 1;
        }
    }

    fn emit_supported(
        &mut self,
        mut active: FxHashSet<WorkerWithDpRank>,
        raw_end: u32,
        raw_hash: Option<ExternalSequenceBlockHash>,
        details: &mut MatchDetails,
    ) {
        self.emitted = true;
        let limit = u32::try_from(self.max_blocks).unwrap_or(u32::MAX);
        // Candidates that need the general downward search.
        let mut pending: Vec<Pending> = Vec::new();
        // Known workers that cannot have a resume point: no Device checkpoint
        // anywhere, or no attention coverage within the limit.
        let mut uncredited = Vec::new();
        // Workers whose coverage ended before the final depth are already
        // scored; the known ones join the confirmed scope below.
        let mut earlier = Vec::new();
        for (&worker, &blocks) in &details.overlap_scores.scores {
            if let Some((count, bit)) = self.device_summary(worker) {
                earlier.push(worker);
                let bound = blocks.min(limit);
                if count == 0 || bound == 0 {
                    uncredited.push(worker);
                } else {
                    pending.push(Pending {
                        worker,
                        bit,
                        bound,
                        resolved: None,
                    });
                }
            }
        }
        let last_hashes = &mut details.last_matched_hashes;
        let scores = &mut details.overlap_scores.scores;
        // Known workers that survived the whole walk share one bound.
        let bound = raw_end.min(limit);
        let mut survivors: Vec<(WorkerWithDpRank, u64)> = Vec::with_capacity(active.len());
        let mut survivor_bits = 0u64;
        active.retain(|worker| {
            let Some((count, bit)) = self.device_summary(*worker) else {
                scores.insert(*worker, raw_end);
                if let Some(hash) = raw_hash {
                    last_hashes.insert(*worker, hash);
                }
                return false;
            };
            if count == 0 || bound == 0 {
                uncredited.push(*worker);
            } else {
                survivors.push((*worker, bit));
                survivor_bits |= bit;
            }
            true
        });

        // Read the record at the survivors' shared bound once. In the common
        // case it settles all of them and no further bookkeeping is needed.
        // The read is skipped when the filter rules every survivor out.
        let top = self
            .chain
            .partition_point(|&(position, _)| position <= bound as usize)
            .checked_sub(1)
            .filter(|_| !survivors.is_empty());
        // Survivor slots the top record does not hold. Sized before the read
        // so the visitor never allocates under the shard guard.
        let mut missed = SlotBitmap::new(survivors.len());
        let mut top_read = false;
        if let Some(index) = top {
            self.residency.visit_checkpoint_workers(
                self.chain[index].1,
                survivor_bits,
                |workers| {
                    top_read = true;
                    for (slot, (worker, _)) in survivors.iter().enumerate() {
                        if !workers.contains(worker) {
                            missed.mark(slot);
                        }
                    }
                },
            );
        }
        let settled_at_top = if top_read {
            survivors.len() - missed.marked()
        } else {
            0
        };
        match top {
            // Everything at or above the top position is ruled out for the
            // survivors it did not settle.
            Some(index) => {
                let below = (self.chain[index].0 - 1) as u32;
                let unsettled = |slot: usize| {
                    let (worker, bit) = survivors[slot];
                    Pending {
                        worker,
                        bit,
                        bound: below,
                        resolved: None,
                    }
                };
                if !top_read {
                    pending.extend((0..survivors.len()).map(unsettled));
                } else if missed.marked() > 0 {
                    pending.reserve(missed.marked());
                    pending.extend(
                        (0..survivors.len())
                            .filter(|&slot| missed.is_marked(slot))
                            .map(unsettled),
                    );
                }
            }
            // No recorded position lies within the bound.
            None => uncredited.extend(survivors.iter().map(|&(worker, _)| worker)),
        }
        pending.retain(|entry| {
            if entry.bound == 0 {
                uncredited.push(entry.worker);
            }
            entry.bound > 0
        });

        self.resolve(&mut pending);

        // The most common cap becomes the default so few overrides are stored.
        let top_cap = top.map_or(0, |index| self.chain[index].0 as u32);
        let cap_of = |entry: &Pending| {
            entry
                .resolved
                .map_or(0, |index| self.chain[index as usize].0 as u32)
        };
        let mut default_cap = top_cap;
        let mut votes = settled_at_top;
        let mut vote = |cap: u32, count: usize| {
            if cap == default_cap {
                votes += count;
            } else if votes >= count {
                votes -= count;
            } else {
                default_cap = cap;
                votes = count - votes;
            }
        };
        vote(0, uncredited.len());
        for entry in &pending {
            vote(cap_of(entry), 1);
        }

        let mut overrides = Vec::new();
        if settled_at_top > 0
            && let Some(index) = top
        {
            let hash = self.chain[index].1;
            let any_missed = missed.marked() > 0;
            for (slot, &(worker, _)) in survivors.iter().enumerate() {
                if any_missed && missed.is_marked(slot) {
                    continue;
                }
                scores.insert(worker, top_cap);
                last_hashes.insert(worker, hash);
                if top_cap != default_cap {
                    overrides.push((worker, top_cap));
                }
            }
        }
        for entry in &pending {
            match entry.resolved {
                Some(index) => {
                    let (position, hash) = self.chain[index as usize];
                    let cap = position as u32;
                    scores.insert(entry.worker, cap);
                    last_hashes.insert(entry.worker, hash);
                    if cap != default_cap {
                        overrides.push((entry.worker, cap));
                    }
                }
                None => uncredited.push(entry.worker),
            }
        }
        for worker in uncredited {
            scores.remove(&worker);
            last_hashes.remove(&worker);
            if default_cap != 0 {
                overrides.push((worker, 0));
            }
        }
        active.extend(earlier);
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
            self.chain.clear();
            // Nothing to record unless some candidate can hold a checkpoint.
            self.needs_checkpoints = self.max_blocks > 0
                && edge
                    .full_workers
                    .iter()
                    .chain(edge.worker_cutoffs.keys())
                    .any(|&worker| {
                        self.device_summary(worker)
                            .is_some_and(|(count, _)| count > 0)
                    });
            if self.needs_checkpoints {
                self.chain.reserve(64);
            }
        }
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
            self.record_block(position, hash);
        }
    }

    fn observe_block_with_coverage(
        &mut self,
        position: usize,
        hash: ExternalSequenceBlockHash,
        _edge: &MatchEdge<'_>,
    ) {
        self.observe_block(position, hash);
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
    use crate::prelude::{ResidencyOwner, StorageTier};

    #[test]
    fn checkpoints_are_read_at_emission_with_current_state_semantics() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(1, 0);
        table.declare_groups(ResidencyOwner::worker(worker), &[0]);
        store_device_checkpoint(&table, worker, 1, 1);
        let raw = raw_device([(worker, 1)].into_iter());
        let mut observer = ForwardDeviceEvaluator::new(&table, None);
        begin_forward(&mut observer, &raw, 1);
        observer.observe_block(1, ExternalSequenceBlockHash(1));

        // The walk only records the path. A record cleared before emission
        // gives no credit: a matched hash is not a lease on the checkpoint,
        // just like attention coverage.
        table.clear_worker(worker);
        let mut cleared = raw.clone();
        observer.finish(&mut cleared);
        assert_eq!(cleared.hybrid_resume_caps.get(&worker), Some(&0));
        assert!(!cleared.overlap_scores.scores.contains_key(&worker));

        store_device_checkpoint(&table, worker, 1, 1);
        let (restored, _) = forward_and_legacy(&table, &raw, None, 1);
        assert_eq!(restored.hybrid_resume_caps.get(&worker), Some(&1));
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
            crate::testing::recurrent_store(worker, StorageTier::Device, position, 0, 1);
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
                    crate::prelude::LocalBlockHash(hash),
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
        use crate::ThreadPoolIndexer;
        use crate::prelude::LocalBlockHash;

        let worker = WorkerWithDpRank::new(7, 0);
        let residency = KvCacheGroupResidency::new();
        residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        store_device_checkpoint(&residency, worker, 3, 1);
        let primary = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 4);
        primary
            .apply_event_and_wait(crate::testing::make_store_event(7, &[11, 12, 13]))
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
                    assert!(forward.chain.is_empty());
                }
            }
        }
    }
}
