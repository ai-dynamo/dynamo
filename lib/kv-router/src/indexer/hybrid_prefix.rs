// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

use super::{
    CapturedLayout, CheckpointLayout, CheckpointReadView, KvCacheGroupResidency,
    LowerTierMatchDetails, MatchDetails, MatchObserver,
};
use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwner, ResidencyOwnerKey, ResidencyRoutingSnapshot,
    StorageTier, WorkerWithDpRank,
};

/// Speculates on an exact, event-published certificate for a common endpoint.
/// Obvious misses collect ordinary evidence during the same walk. Inconclusive
/// attempts are discarded before retrying the ordinary evaluator on the full
/// query, preserving its first-observation and revision semantics.
pub(crate) struct EndpointPrefixProof<'a> {
    residency: &'a KvCacheGroupResidency,
    cutoff: u64,
    certificate: Option<super::kv_cache_groups::EndpointCertificateRead>,
    certificate_hash: Option<ExternalSequenceBlockHash>,
    fallback: Option<HybridPrefixEvaluator>,
    allow_early_fallback: bool,
}

impl<'a> EndpointPrefixProof<'a> {
    pub(crate) fn new(residency: &'a KvCacheGroupResidency) -> Self {
        Self {
            residency,
            cutoff: 0,
            certificate: None,
            certificate_hash: None,
            fallback: None,
            allow_early_fallback: true,
        }
    }

    pub(crate) fn bounded(residency: &'a KvCacheGroupResidency) -> Self {
        Self {
            allow_early_fallback: false,
            ..Self::new(residency)
        }
    }

    pub(crate) fn confirms(&self, details: &MatchDetails) -> bool {
        details.overlap_scores.scores.is_empty()
            || (!details.hybrid_resume_caps.is_empty()
                && self.certificate.as_ref().is_some_and(|certificate| {
                    self.residency.is_certificate_read_valid(certificate)
                }))
    }

    pub(crate) fn finish(&mut self, details: &mut MatchDetails) -> bool {
        if let Some(evaluator) = &mut self.fallback {
            evaluator.prepare_device(details);
            evaluator.finish(details, &mut Default::default(), &Default::default());
            true
        } else {
            self.confirms(details)
        }
    }
}

impl MatchObserver for EndpointPrefixProof<'_> {
    fn observes_path(&self) -> bool {
        self.fallback.is_some()
    }

    fn reserve_workers(&mut self, _count: usize) {
        self.cutoff = self.residency.capture_cutoff();
    }

    fn begin_path(&mut self, count: usize, endpoint_hint: Option<ExternalSequenceBlockHash>) {
        self.reserve_workers(count);
        if let Some(hash) = endpoint_hint {
            self.certificate = self.residency.endpoint_certificate_read(hash, self.cutoff);
            self.certificate_hash = Some(hash);
            if self.allow_early_fallback
                && self
                    .certificate
                    .as_ref()
                    .is_none_or(|certificate| certificate.workers().len() < count)
            {
                self.certificate = None;
                self.certificate_hash = None;
                let mut evaluator = HybridPrefixEvaluator::new(self.residency, None);
                evaluator.reserve_workers(count);
                self.fallback = Some(evaluator);
            }
        }
    }

    fn observe_worker(&mut self, worker: WorkerWithDpRank) {
        if let Some(evaluator) = &mut self.fallback {
            evaluator.observe_worker(worker);
        }
    }

    fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
        if let Some(evaluator) = &mut self.fallback {
            evaluator.observe_block(position, hash);
        }
    }

    fn uniform_resume_workers(
        &mut self,
        _position: u32,
        hash: ExternalSequenceBlockHash,
    ) -> Option<&FxHashSet<WorkerWithDpRank>> {
        if self.fallback.is_some() {
            return None;
        }
        if self.certificate_hash != Some(hash) {
            self.certificate = self.residency.endpoint_certificate_read(hash, self.cutoff);
            self.certificate_hash = Some(hash);
        }
        self.certificate
            .as_ref()
            .map(|certificate| certificate.workers())
    }
}

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

#[derive(Clone, Copy, Debug)]
struct DeviceCandidate {
    point: Checkpoint,
    membership_revision: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceResolutionProbe {
    PendingSet(usize),
    MembershipScan(usize),
    MembershipOwner,
    PendingLookups(usize),
    PendingOwnerLookup,
    PendingOwnerFound,
    PendingWords(usize),
    GenerationWordCheck,
    GenerationWordProbe,
    GenerationWordSkip,
    RawBoundSkip,
    PendingCoarseWords(usize),
    CoarseWordCheck,
    CoarseWordProbe,
    CoarseWordSkip,
}

struct PendingGeneration {
    generation: u64,
    count: usize,
    word_slot: usize,
}

struct PendingGenerationWord {
    index: u64,
    mask: u64,
}

/// An exact negative filter for the still-pending query generations. Word slots
/// stay sorted/stable; active indices remain sorted so queries never rescan
/// resolved zero words. Counts keep duplicate generations from clearing early.
struct PendingGenerationWords {
    generations: Vec<PendingGeneration>,
    words: Vec<PendingGenerationWord>,
    active: Vec<usize>,
    coarse: Option<Box<PendingGenerationWords>>,
}

impl PendingGenerationWords {
    const COARSE_WORD_THRESHOLD: usize = 8;

    fn new(generations: impl Iterator<Item = u64>) -> Self {
        Self::from_counts(
            generations
                .map(|generation| PendingGeneration {
                    generation,
                    count: 1,
                    word_slot: 0,
                })
                .collect(),
            true,
        )
    }

    fn from_counts(mut generations: Vec<PendingGeneration>, build_coarse: bool) -> Self {
        generations.sort_unstable_by_key(|entry| entry.generation);
        generations.dedup_by(|current, previous| {
            if current.generation == previous.generation {
                previous.count += current.count;
                true
            } else {
                false
            }
        });
        let word_count = usize::from(!generations.is_empty())
            + generations
                .windows(2)
                .filter(|pair| pair[0].generation >> 6 != pair[1].generation >> 6)
                .count();
        let mut words: Vec<PendingGenerationWord> = Vec::with_capacity(word_count);
        for entry in &mut generations {
            let index = entry.generation >> 6;
            let bit = 1 << (entry.generation & 63);
            if let Some(word) = words.last_mut()
                && word.index == index
            {
                word.mask |= bit;
            } else {
                words.push(PendingGenerationWord { index, mask: bit });
            }
            entry.word_slot = words.len() - 1;
        }
        let active = (0..words.len()).collect();
        // Preserve worker multiplicities, including distinct generations in the
        // same fine word. The inner constructor cannot build another level.
        let coarse = (build_coarse && word_count >= Self::COARSE_WORD_THRESHOLD).then(|| {
            Box::new(Self::from_counts(
                generations
                    .iter()
                    .map(|entry| PendingGeneration {
                        generation: entry.generation >> 6,
                        count: entry.count,
                        word_slot: 0,
                    })
                    .collect(),
                false,
            ))
        });
        Self {
            generations,
            words,
            active,
            coarse,
        }
    }

    fn remove(&mut self, generation: u64) {
        let index = self
            .generations
            .binary_search_by_key(&generation, |entry| entry.generation)
            .expect("a resolved generation was in the pending set");
        let entry = &mut self.generations[index];
        debug_assert!(entry.count > 0);
        if let Some(coarse) = self.coarse.as_mut() {
            coarse.remove(generation >> 6);
        }
        entry.count -= 1;
        if entry.count > 0 {
            return;
        }
        let word = &mut self.words[entry.word_slot];
        word.mask &= !(1 << (generation & 63));
        if word.mask == 0 {
            let position = self
                .active
                .binary_search(&entry.word_slot)
                .expect("a nonzero generation word was active");
            self.active.remove(position);
        }
    }

    fn mask(&self, index: u64) -> u64 {
        self.words
            .binary_search_by_key(&index, |word| word.index)
            .ok()
            .map(|slot| self.words[slot].mask)
            .unwrap_or(0)
    }

    /// Lower-bound jump after a word already known to precede the target.
    /// Exponential bracketing keeps short/alternating runs out of N-log-N work.
    fn lower_bound_after(
        known_less: usize,
        len: usize,
        target: u64,
        mut word_at: impl FnMut(usize) -> (u64, u64),
    ) -> usize {
        let mut low = known_less + 1;
        let mut high = len;
        let mut step = 1usize;
        while let Some(index) = known_less.checked_add(step).filter(|&index| index < len) {
            if word_at(index).0 >= target {
                high = index;
                break;
            }
            low = index + 1;
            let Some(next_step) = step.checked_mul(2) else {
                break;
            };
            step = next_step;
        }
        while low < high {
            let mid = low + (high - low) / 2;
            if word_at(mid).0 < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }

    fn may_intersect(
        &self,
        mut membership_words: impl ExactSizeIterator<Item = (u64, u64)> + DoubleEndedIterator + Clone,
        mut membership_mask: impl FnMut(u64) -> u64,
        #[cfg(test)] observe: impl FnMut(),
    ) -> bool {
        // Both word domains are sorted and unique. The membership API preserves
        // that contract while retaining constant-time indexed slice access.
        let pending_len = self.active.len();
        let membership_len = membership_words.len();
        if pending_len == 0 || membership_len == 0 {
            return false;
        }
        #[cfg(test)]
        let observe = std::cell::RefCell::new(observe);
        if pending_len == 1 {
            #[cfg(test)]
            observe.borrow_mut()();
            let word = &self.words[self.active[0]];
            return word.mask & membership_mask(word.index) != 0;
        }
        if membership_len == 1 {
            #[cfg(test)]
            observe.borrow_mut()();
            let (index, mask) = membership_words.next().unwrap();
            return mask & self.mask(index) != 0;
        }
        let mut pending_at = |index: usize| {
            #[cfg(test)]
            observe.borrow_mut()();
            let word = &self.words[self.active[index]];
            (word.index, word.mask)
        };
        let mut membership_at = |index: usize| {
            #[cfg(test)]
            observe.borrow_mut()();
            // The scoped membership API is a mapped slice iterator: cloning
            // and nth are O(1), with no owner or word container copy.
            membership_words.clone().nth(index).unwrap()
        };
        let mut pending_index = 0;
        let mut membership_index = 0;
        let mut pending = pending_at(0);
        let mut member = membership_at(0);
        let pending_last = pending_at(pending_len - 1).0;
        let membership_last = membership_at(membership_len - 1).0;
        loop {
            if pending.0 > membership_last || member.0 > pending_last {
                return false;
            }
            match pending.0.cmp(&member.0) {
                std::cmp::Ordering::Equal => {
                    if pending.1 & member.1 != 0 {
                        return true;
                    }
                    pending_index += 1;
                    membership_index += 1;
                    if pending_index == pending_len || membership_index == membership_len {
                        return false;
                    }
                    pending = pending_at(pending_index);
                    member = membership_at(membership_index);
                }
                std::cmp::Ordering::Less => {
                    let next_index = pending_index + 1;
                    if next_index == pending_len {
                        return false;
                    }
                    let next = pending_at(next_index);
                    if next.0 >= member.0 {
                        pending_index = next_index;
                        pending = next;
                        continue;
                    }
                    let second_index = next_index + 1;
                    if second_index == pending_len {
                        return false;
                    }
                    let second = pending_at(second_index);
                    if second.0 >= member.0 {
                        pending_index = second_index;
                        pending = second;
                    } else {
                        pending_index = Self::lower_bound_after(
                            second_index,
                            pending_len,
                            member.0,
                            &mut pending_at,
                        );
                        if pending_index == pending_len {
                            return false;
                        }
                        pending = pending_at(pending_index);
                    }
                }
                std::cmp::Ordering::Greater => {
                    let next_index = membership_index + 1;
                    if next_index == membership_len {
                        return false;
                    }
                    let next = membership_at(next_index);
                    if next.0 >= pending.0 {
                        membership_index = next_index;
                        member = next;
                        continue;
                    }
                    let second_index = next_index + 1;
                    if second_index == membership_len {
                        return false;
                    }
                    let second = membership_at(second_index);
                    if second.0 >= pending.0 {
                        membership_index = second_index;
                        member = second;
                    } else {
                        membership_index = Self::lower_bound_after(
                            second_index,
                            membership_len,
                            pending.0,
                            &mut membership_at,
                        );
                        if membership_index == membership_len {
                            return false;
                        }
                        member = membership_at(membership_index);
                    }
                }
            }
        }
    }
}

struct DeviceWorkerState {
    captured: Option<CapturedLayout>,
    checkpoint: Checkpoint,
    resume_cap: Option<usize>,
    raw_end: u32,
}

impl DeviceWorkerState {
    fn new(captured: Option<CapturedLayout>) -> Self {
        Self {
            captured,
            checkpoint: Checkpoint::ZERO,
            resume_cap: None,
            raw_end: 0,
        }
    }

    fn checkpoint(&self, end: usize, floor: usize) -> Option<Checkpoint> {
        let layout = self.captured?.layout();
        Some(
            if layout.complete_layout
                && self.checkpoint.position <= end
                && self.checkpoint.position > floor
            {
                self.checkpoint
            } else {
                Checkpoint::ZERO
            },
        )
    }

    fn resolve_checkpoint(&mut self, point: Checkpoint, generation: u64) -> bool {
        let Some(captured) = self.captured else {
            return false;
        };
        let alignment = captured.layout().alignment_blocks;
        if !captured.is_eligible_generation(generation)
            || self.checkpoint.position != 0
            || point.position > self.raw_end as usize
            || (alignment > 1 && !point.position.is_multiple_of(alignment))
        {
            return false;
        }
        self.checkpoint = point;
        self.resume_cap = Some(point.position);
        true
    }

    fn is_pending(&self) -> bool {
        self.raw_end > 0
            && self.resume_cap.is_some()
            && self.checkpoint.position == 0
            && self
                .captured
                .is_some_and(|captured| captured.layout().complete_layout)
    }
}

/// Query-owned evidence collected while the attention indexes visit their blocks.
/// Group completeness is maintained by event writers. The read path never scans
/// groups or probes every earlier boundary separately for each worker.
pub(crate) struct HybridPrefixEvaluator {
    view: CheckpointReadView,
    candidates: FxHashMap<(ResidencyOwner, StorageTier), Vec<Checkpoint>>,
    tier: StorageTier,
    max_blocks: usize,
    workers: FxHashMap<WorkerWithDpRank, DeviceWorkerState>,
    device_positions: Vec<DeviceCandidate>,
    device_prepared: bool,
    legacy_layouts_seeded: bool,
}

impl HybridPrefixEvaluator {
    // Strategy budgets, never limits on candidate workers or credited blocks.
    const DIRECT_SCAN_OWNER_BUDGET: usize = 256;
    const DIRECT_SCAN_SMALL_MEMBERSHIP: usize = 2;

    pub(crate) fn new(residency: &KvCacheGroupResidency, max_blocks: Option<usize>) -> Self {
        Self {
            view: residency.read_view(),
            candidates: FxHashMap::default(),
            tier: StorageTier::Device,
            max_blocks: max_blocks.unwrap_or(usize::MAX),
            workers: FxHashMap::default(),
            device_positions: Vec::new(),
            device_prepared: false,
            legacy_layouts_seeded: false,
        }
    }

    fn worker_state(&mut self, worker: WorkerWithDpRank) -> &mut DeviceWorkerState {
        self.workers.entry(worker).or_insert_with(|| {
            let owner = ResidencyOwner::worker(worker);
            let captured = self.view.capture_layout(owner);
            if self.legacy_layouts_seeded {
                self.view.remember_layout(owner, captured);
            }
            DeviceWorkerState::new(captured)
        })
    }

    pub(crate) fn on_tier(&mut self, tier: StorageTier) {
        if tier != StorageTier::Device && !self.legacy_layouts_seeded {
            // Lower-tier snapshots must use the exact first Device capture,
            // including None; a later layout must not be learned mid-query.
            for (&worker, state) in &self.workers {
                self.view
                    .remember_layout(ResidencyOwner::worker(worker), state.captured);
            }
            self.legacy_layouts_seeded = true;
        }
        self.tier = tier;
    }

    fn layout(&self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        if let Some(worker) = owner.as_worker()
            && let Some(state) = self.workers.get(&worker)
        {
            return state.captured.map(|captured| captured.layout());
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
        let layout = self.layout(owner)?;
        if !layout.complete_layout || tier == StorageTier::Device {
            return Some(Checkpoint::ZERO);
        }
        let found = self
            .candidates
            .get(&(owner, tier))
            .and_then(|points| {
                let idx = points.partition_point(|point| point.position <= end);
                points
                    .get(idx.wrapping_sub(1))
                    .filter(|point| point.position > floor)
            })
            .copied();
        Some(found.unwrap_or(Checkpoint::ZERO))
    }

    /// Resolve GPU candidates together on the hashes already visited by the tree.
    /// A dense checkpoint population needs only the last suitable owner list,
    /// rather than materializing one candidate per owner at every earlier block.
    pub(crate) fn prepare_device(&mut self, raw: &MatchDetails) {
        #[cfg(test)]
        self.prepare_device_observed(raw, |_| {});
        #[cfg(not(test))]
        self.prepare_device_observed(raw);
    }

    fn prepare_device_observed(
        &mut self,
        raw: &MatchDetails,
        #[cfg(test)] mut observe: impl FnMut(DeviceResolutionProbe),
    ) {
        if self.device_prepared {
            return;
        }
        self.device_prepared = true;
        let mut pending = 0;
        for (&worker, &blocks) in &raw.overlap_scores.scores {
            let state = self.worker_state(worker);
            state.raw_end = blocks;
            if let Some(captured) = state.captured {
                state.resume_cap = Some(0);
                pending += usize::from(blocks > 0 && captured.layout().complete_layout);
            }
        }
        let mut positions = std::mem::take(&mut self.device_positions).into_iter().rev();
        if pending == 0 {
            return;
        }
        let Some(candidate) = positions.next() else {
            return;
        };
        // Keep the common latest-boundary case to one membership walk and a
        // scalar count. No pending set is needed when that walk resolves all.
        let point = candidate.point;
        self.view.with_membership_at_revision(
            point.hash,
            StorageTier::Device,
            candidate.membership_revision,
            |membership| {
                #[cfg(test)]
                observe(DeviceResolutionProbe::MembershipScan(membership.len()));
                for (owner, generation) in membership.iter() {
                    #[cfg(test)]
                    observe(DeviceResolutionProbe::MembershipOwner);
                    if let Some(worker) = owner.as_worker()
                        && let Some(state) = self.workers.get_mut(&worker)
                        && state.resolve_checkpoint(point, generation)
                    {
                        pending -= 1;
                    }
                }
            },
        );
        if pending == 0 || positions.len() == 0 {
            return;
        }
        // Building pending sets costs more than a few cheap membership walks.
        // Inspect each actual list so a narrow tail can stay allocation-free,
        // while a wider earlier list can still switch to indexed resolution.
        let indexed_candidate = loop {
            let Some(candidate) = positions.next() else {
                return;
            };
            let mut use_index = false;
            let point = candidate.point;
            self.view.with_membership_at_revision(
                point.hash,
                StorageTier::Device,
                candidate.membership_revision,
                |membership| {
                    if membership.len() > Self::DIRECT_SCAN_SMALL_MEMBERSHIP
                        && positions
                            .len()
                            .saturating_add(1)
                            .saturating_mul(membership.len())
                            > Self::DIRECT_SCAN_OWNER_BUDGET
                    {
                        use_index = true;
                        return;
                    }
                    #[cfg(test)]
                    observe(DeviceResolutionProbe::MembershipScan(membership.len()));
                    for (owner, generation) in membership.iter() {
                        #[cfg(test)]
                        observe(DeviceResolutionProbe::MembershipOwner);
                        if let Some(worker) = owner.as_worker()
                            && let Some(state) = self.workers.get_mut(&worker)
                            && state.resolve_checkpoint(point, generation)
                        {
                            pending -= 1;
                        }
                    }
                },
            );
            if pending == 0 {
                return;
            }
            if use_index {
                break candidate;
            }
        };
        // A few missing query checkpoints must not make every earlier hash
        // revisit all owners that already resolved. Global checkpoint counts
        // cannot prove absence on this query's chain, so keep those workers.
        let mut unresolved: FxHashSet<_> = self
            .workers
            .iter()
            .filter_map(|(&worker, state)| state.is_pending().then_some(worker))
            .collect();
        #[cfg(test)]
        observe(DeviceResolutionProbe::PendingSet(unresolved.len()));
        let mut max_raw_end = 0;
        let mut pending_words = PendingGenerationWords::new(unresolved.iter().map(|worker| {
            let state = &self.workers[worker];
            max_raw_end = max_raw_end.max(state.raw_end);
            state
                .captured
                .expect("a pending worker has a captured layout")
                .generation()
        }));
        #[cfg(test)]
        observe(DeviceResolutionProbe::PendingWords(
            pending_words.active.len(),
        ));
        #[cfg(test)]
        if let Some(coarse) = pending_words.coarse.as_ref() {
            observe(DeviceResolutionProbe::PendingCoarseWords(
                coarse.active.len(),
            ));
        }
        for candidate in std::iter::once(indexed_candidate).chain(positions) {
            if unresolved.is_empty() {
                break;
            }
            let point = candidate.point;
            if point.position > max_raw_end as usize {
                #[cfg(test)]
                observe(DeviceResolutionProbe::RawBoundSkip);
                continue;
            }
            // Any changed H revision rejects this whole boundary, including
            // existing owners, and lets earlier unchanged candidates recover.
            self.view.with_membership_at_revision(
                point.hash,
                StorageTier::Device,
                candidate.membership_revision,
                |membership| {
                    if pending_words.active.len() >= PendingGenerationWords::COARSE_WORD_THRESHOLD
                        && let Some(coarse) = pending_words.coarse.as_ref()
                    {
                        #[cfg(test)]
                        observe(DeviceResolutionProbe::CoarseWordCheck);
                        #[cfg(test)]
                        let may_intersect = coarse.may_intersect(
                            membership.generation_word_groups(),
                            |group| membership.generation_word_group_mask(group),
                            || observe(DeviceResolutionProbe::CoarseWordProbe),
                        );
                        #[cfg(not(test))]
                        let may_intersect = coarse
                            .may_intersect(membership.generation_word_groups(), |group| {
                                membership.generation_word_group_mask(group)
                            });
                        if !may_intersect {
                            #[cfg(test)]
                            observe(DeviceResolutionProbe::CoarseWordSkip);
                            return;
                        }
                    }
                    #[cfg(test)]
                    observe(DeviceResolutionProbe::GenerationWordCheck);
                    #[cfg(test)]
                    let may_intersect = pending_words.may_intersect(
                        membership.generation_words(),
                        |word| membership.generation_mask(word),
                        || observe(DeviceResolutionProbe::GenerationWordProbe),
                    );
                    #[cfg(not(test))]
                    let may_intersect = pending_words
                        .may_intersect(membership.generation_words(), |word| {
                            membership.generation_mask(word)
                        });
                    if !may_intersect {
                        #[cfg(test)]
                        observe(DeviceResolutionProbe::GenerationWordSkip);
                        return;
                    }
                    if unresolved.len() < membership.len() {
                        #[cfg(test)]
                        observe(DeviceResolutionProbe::PendingLookups(unresolved.len()));
                        unresolved.retain(|worker| {
                            #[cfg(test)]
                            observe(DeviceResolutionProbe::PendingOwnerLookup);
                            let Some(generation) =
                                membership.generation_for(ResidencyOwner::worker(*worker))
                            else {
                                return true;
                            };
                            #[cfg(test)]
                            observe(DeviceResolutionProbe::PendingOwnerFound);
                            let resolved = self
                                .workers
                                .get_mut(worker)
                                .is_some_and(|state| state.resolve_checkpoint(point, generation));
                            if resolved {
                                pending_words.remove(generation);
                            }
                            !resolved
                        });
                    } else {
                        #[cfg(test)]
                        observe(DeviceResolutionProbe::MembershipScan(membership.len()));
                        for (owner, generation) in membership.iter() {
                            #[cfg(test)]
                            observe(DeviceResolutionProbe::MembershipOwner);
                            if let Some(worker) = owner.as_worker()
                                && let Some(state) = self.workers.get_mut(&worker)
                                && state.resolve_checkpoint(point, generation)
                            {
                                unresolved.remove(&worker);
                                pending_words.remove(generation);
                            }
                        }
                    }
                },
            );
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
        if self.tier == StorageTier::Device {
            if let Some(membership_revision) = self.view.checkpoint_hint_revision(hash, self.tier) {
                self.device_positions.push(DeviceCandidate {
                    point: Checkpoint { position, hash },
                    membership_revision,
                });
            }
            return;
        }
        for owner in self.view.owners_at(hash, self.tier) {
            if !position.is_multiple_of(
                self.layout(owner)
                    .map(|layout| layout.alignment_blocks)
                    .unwrap_or(1)
                    .max(1),
            ) {
                continue;
            }
            let points = self.candidates.entry((owner, self.tier)).or_default();
            match points.binary_search_by_key(&position, |point| point.position) {
                Ok(_) => {}
                Err(idx) => points.insert(idx, Checkpoint { position, hash }),
            }
        }
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

    #[derive(Default)]
    struct ResolutionCounts {
        pending_sets: Vec<usize>,
        membership_scans: Vec<usize>,
        membership_owners: usize,
        pending_selections: Vec<usize>,
        pending_lookups: usize,
        pending_found: usize,
        pending_words: Vec<usize>,
        word_checks: usize,
        word_probes: usize,
        word_skips: usize,
        bound_skips: usize,
        pending_coarse_words: Vec<usize>,
        coarse_checks: usize,
        coarse_probes: usize,
        coarse_skips: usize,
    }

    impl ResolutionCounts {
        fn observe(&mut self, probe: DeviceResolutionProbe) {
            match probe {
                DeviceResolutionProbe::PendingSet(count) => self.pending_sets.push(count),
                DeviceResolutionProbe::MembershipScan(count) => self.membership_scans.push(count),
                DeviceResolutionProbe::MembershipOwner => self.membership_owners += 1,
                DeviceResolutionProbe::PendingLookups(count) => self.pending_selections.push(count),
                DeviceResolutionProbe::PendingOwnerLookup => self.pending_lookups += 1,
                DeviceResolutionProbe::PendingOwnerFound => self.pending_found += 1,
                DeviceResolutionProbe::PendingWords(count) => self.pending_words.push(count),
                DeviceResolutionProbe::GenerationWordCheck => self.word_checks += 1,
                DeviceResolutionProbe::GenerationWordProbe => self.word_probes += 1,
                DeviceResolutionProbe::GenerationWordSkip => self.word_skips += 1,
                DeviceResolutionProbe::RawBoundSkip => self.bound_skips += 1,
                DeviceResolutionProbe::PendingCoarseWords(count) => {
                    self.pending_coarse_words.push(count)
                }
                DeviceResolutionProbe::CoarseWordCheck => self.coarse_checks += 1,
                DeviceResolutionProbe::CoarseWordProbe => self.coarse_probes += 1,
                DeviceResolutionProbe::CoarseWordSkip => self.coarse_skips += 1,
            }
        }
    }

    fn pending_words_intersect(
        pending: &PendingGenerationWords,
        membership: &[(u64, u64)],
    ) -> (bool, usize) {
        let mut probes = 0;
        let found = pending.may_intersect(
            membership.iter().copied(),
            |word| {
                membership
                    .binary_search_by_key(&word, |&(index, _)| index)
                    .ok()
                    .map(|slot| membership[slot].1)
                    .unwrap_or(0)
            },
            || probes += 1,
        );
        (found, probes)
    }

    #[test]
    fn pending_generation_words_keep_duplicate_bits_until_the_last_worker_resolves() {
        let mut pending = PendingGenerationWords::new([1, 1, 65].into_iter());
        assert!(pending.coarse.is_none());
        assert_eq!(pending_words_intersect(&pending, &[(0, 2)]), (true, 1));
        pending.remove(1);
        assert_eq!(pending_words_intersect(&pending, &[(0, 2)]), (true, 1));
        pending.remove(1);
        assert_eq!(pending_words_intersect(&pending, &[(0, 2)]), (false, 1));
        assert_eq!(pending.active.len(), 1);
        assert_eq!(pending_words_intersect(&pending, &[(1, 2)]), (true, 1));
        pending.remove(65);
        assert!(pending.active.is_empty());
        assert_eq!(pending_words_intersect(&pending, &[(1, 2)]), (false, 0));
    }

    #[test]
    fn coarse_generation_words_preserve_worker_counts_across_shared_fine_words() {
        let mut pending = PendingGenerationWords::new(
            [1, 2, 2]
                .into_iter()
                .chain((1..8).map(|word| word * 64 + 1)),
        );
        assert_eq!(pending.active.len(), 8);
        let coarse = pending.coarse.as_ref().unwrap();
        assert!(coarse.coarse.is_none());
        assert_eq!(pending_words_intersect(coarse, &[(0, 1)]), (true, 1));
        for generation in [2, 1] {
            pending.remove(generation);
            assert!(pending_words_intersect(pending.coarse.as_ref().unwrap(), &[(0, 1)]).0);
        }
        pending.remove(2);
        assert!(!pending_words_intersect(pending.coarse.as_ref().unwrap(), &[(0, 1)]).0);
        assert_eq!(pending.active.len(), 7);
        for word in 1..8 {
            pending.remove(word * 64 + 1);
        }
        assert!(pending.coarse.as_ref().unwrap().active.is_empty());
    }

    #[test]
    fn coarse_generation_words_are_one_level_and_only_prove_negative_membership() {
        let mut pending =
            PendingGenerationWords::new((0..8).map(|group| group << 12).chain([u64::MAX]));
        let coarse = pending.coarse.as_ref().unwrap();
        assert_eq!(coarse.active.len(), 9);
        assert!(coarse.coarse.is_none());
        assert!(pending_words_intersect(coarse, &[(u64::MAX >> 12, 1 << 63)]).0);
        pending.remove(u64::MAX);
        assert!(
            !pending_words_intersect(
                pending.coarse.as_ref().unwrap(),
                &[(u64::MAX >> 12, 1 << 63)]
            )
            .0
        );

        let pending = PendingGenerationWords::new((0..8).map(|word| word * 64 + 1));
        let different_bits: Vec<_> = (0..8).map(|word| (word, 1 << 2)).collect();
        assert!(pending_words_intersect(pending.coarse.as_ref().unwrap(), &[(0, 255)]).0);
        assert!(!pending_words_intersect(&pending, &different_bits).0);
    }

    #[test]
    fn pending_generation_words_probe_only_active_slots_after_a_thousand_word_removals() {
        let mut pending = PendingGenerationWords::new((0..1000).map(|word| word * 64));
        for word in 0..999 {
            pending.remove(word * 64);
        }
        assert_eq!(pending.words.len(), 1000);
        assert_eq!(pending.active.len(), 1);
        let membership: Vec<_> = (0..1000)
            .map(|word| (word, if word == 999 { 2 } else { 1 }))
            .collect();
        assert_eq!(pending_words_intersect(&pending, &membership), (false, 1));
        let mut far = PendingGenerationWords::new([0, 1 << 40, u64::MAX].into_iter());
        assert_eq!(
            pending_words_intersect(&far, &[(u64::MAX >> 6, 1 << 63)]),
            (true, 1)
        );
        far.remove(u64::MAX);
        assert_eq!(
            pending_words_intersect(&far, &[(u64::MAX >> 6, 1 << 63)]),
            (false, 1)
        );
        assert_eq!(pending_words_intersect(&far, &[(1 << 34, 1)]), (true, 1));
    }

    #[test]
    fn pending_generation_words_skip_sparse_ranges_without_penalizing_alternation() {
        let middle = PendingGenerationWords::new((16..48).map(|word| word * 64));
        let mut membership: Vec<_> = (0..16).chain(48..64).map(|word| (word, 1)).collect();
        let (found, probes) = pending_words_intersect(&middle, &membership);
        assert!(!found);
        assert!(probes < 24, "middle range used {probes} indexed probes");
        membership.push((33, 1));
        membership.sort_unstable_by_key(|entry| entry.0);
        assert!(pending_words_intersect(&middle, &membership).0);

        let last = PendingGenerationWords::new((32..64).map(|word| word * 64));
        let membership: Vec<_> = (0..32).map(|word| (word, 1)).collect();
        let (found, probes) = pending_words_intersect(&last, &membership);
        assert!(!found);
        assert!(probes <= 4, "disjoint ranges used {probes} indexed probes");

        let alternating = PendingGenerationWords::new((0..256).map(|index| (2 * index + 1) * 64));
        let membership: Vec<_> = (0..256).map(|index| (2 * index, 1)).collect();
        let (found, probes) = pending_words_intersect(&alternating, &membership);
        assert!(!found);
        assert!(probes <= 4 * (alternating.active.len() + membership.len()));
    }

    #[test]
    fn pending_generation_word_intersection_matches_an_exact_randomized_oracle() {
        use std::collections::BTreeMap;
        let mut seed = 0x739a_17c8_9e12_43d1u64;
        for case in 0..128 {
            let mut generations = vec![0, u64::MAX];
            let mut membership = BTreeMap::from([
                (0, if case % 3 == 0 { 1 } else { 2 }),
                (u64::MAX >> 6, if case % 5 == 0 { 1 << 63 } else { 1 }),
            ]);
            for word in 1..129u64 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                if seed & 7 == 0 {
                    generations.push(word * 64 + ((seed >> 17) & 63));
                }
                if (seed >> 3) & 7 == 0 {
                    membership.insert(word, 1 << ((seed >> 31) & 63));
                }
            }
            if case % 7 == 0 {
                generations.push(0);
            }
            let mut pending = PendingGenerationWords::new(generations.iter().copied());
            let mut remaining = Vec::new();
            for (index, &generation) in generations.iter().enumerate() {
                if (index + case) % 3 == 0 {
                    pending.remove(generation);
                } else {
                    remaining.push(generation);
                }
            }
            let expected = remaining.iter().any(|&generation| {
                membership.get(&(generation >> 6)).copied().unwrap_or(0) & (1 << (generation & 63))
                    != 0
            });
            if let Some(coarse) = pending.coarse.as_ref() {
                let mut groups = BTreeMap::<u64, u64>::new();
                for &word in membership.keys() {
                    *groups.entry(word >> 6).or_default() |= 1 << (word & 63);
                }
                let expected_coarse = remaining
                    .iter()
                    .any(|&generation| membership.contains_key(&(generation >> 6)));
                let actual_coarse =
                    pending_words_intersect(coarse, &groups.into_iter().collect::<Vec<_>>()).0;
                assert_eq!(actual_coarse, expected_coarse, "coarse case {case}");
                assert!(
                    !expected || actual_coarse,
                    "coarse false negative in case {case}"
                );
            }
            let membership: Vec<_> = membership.into_iter().collect();
            assert_eq!(
                pending_words_intersect(&pending, &membership).0,
                expected,
                "case {case}"
            );
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

    #[test]
    fn early_proof_rejection_finishes_the_same_walk_and_freezes_unknown_layout() {
        let residency = KvCacheGroupResidency::new();
        let known = WorkerWithDpRank::new(1, 0);
        let initially_unknown = WorkerWithDpRank::new(2, 0);
        residency.declare_groups(ResidencyOwner::worker(known), &[0]);
        store_device_checkpoint(&residency, known, 11, 1);
        let mut proof = EndpointPrefixProof::new(&residency);
        proof.begin_path(2, Some(ExternalSequenceBlockHash(11)));
        assert!(
            proof.observes_path(),
            "insufficient membership enters the ordinary evaluator early"
        );
        proof.observe_worker(known);
        proof.observe_worker(initially_unknown);
        assert!(
            proof.fallback.as_ref().unwrap().workers[&initially_unknown]
                .captured
                .is_none()
        );
        // A later registration cannot replace this walk's first None capture.
        residency.declare_groups(ResidencyOwner::worker(initially_unknown), &[0]);
        store_device_checkpoint(&residency, initially_unknown, 11, 1);
        proof.observe_block(1, ExternalSequenceBlockHash(11));
        let mut raw = MatchDetails::default();
        raw.overlap_scores
            .scores
            .extend([(known, 1), (initially_unknown, 1)]);
        assert!(
            proof.finish(&mut raw),
            "early fallback must finish without a second tree walk"
        );
        assert_eq!(raw.hybrid_resume_caps.get(&known), Some(&1));
        assert_eq!(raw.hybrid_resume_caps.get(&initially_unknown), None);
        assert_eq!(raw.overlap_scores.scores.get(&initially_unknown), Some(&1));
    }

    #[test]
    fn reverse_resolution_owner_probe_counts_cover_complete_and_mixed_populations() {
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        for missing_count in [0, 1, 32, 63] {
            let member_count = workers.len() - missing_count;
            let residency = KvCacheGroupResidency::new();
            for (index, &worker) in workers.iter().enumerate() {
                residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
                if index >= member_count {
                    // A globally resident state says nothing about this chain.
                    store_device_checkpoint(&residency, worker, 999, 1);
                } else {
                    for position in 1..=8 {
                        store_device_checkpoint(&residency, worker, position, 1);
                    }
                }
            }
            let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
            evaluator.reserve_workers(workers.len());
            let mut raw = MatchDetails::default();
            for &worker in &workers {
                evaluator.observe_worker(worker);
                raw.overlap_scores.scores.insert(worker, 8);
            }
            for position in 1..=8 {
                evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
            }
            raw.kv_transfer_candidates = Some(KvTransferCandidates {
                block_hashes: (1..=8).map(ExternalSequenceBlockHash).collect(),
                owner_prefix_blocks: workers.iter().map(|&worker| (worker.into(), 8)).collect(),
                source_intervals: Vec::new(),
                routing_snapshot: None,
            });
            let word_count = |workers: &[WorkerWithDpRank]| {
                workers
                    .iter()
                    .map(|worker| evaluator.workers[worker].captured.unwrap().generation() >> 6)
                    .collect::<FxHashSet<_>>()
                    .len()
            };
            let smaller_word_count =
                word_count(&workers[..member_count]).min(word_count(&workers[member_count..]));
            let mut counts = ResolutionCounts::default();
            evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
            assert!(counts.pending_coarse_words.is_empty());
            assert_eq!(counts.coarse_checks, 0);
            if missing_count == 0 {
                assert!(counts.pending_sets.is_empty());
                assert!(counts.pending_words.is_empty());
                assert_eq!(counts.membership_scans, [64]);
                assert_eq!(counts.membership_owners, 64);
                assert_eq!(counts.pending_lookups, 0);
                assert_eq!(counts.word_checks, 0);
            } else if missing_count == 1 {
                assert_eq!(counts.pending_sets, [missing_count]);
                assert_eq!(counts.membership_scans, [member_count]);
                assert_eq!(counts.membership_owners, member_count);
                assert_eq!(counts.pending_lookups, 0);
                assert_eq!(counts.word_checks, 7);
                assert!(counts.word_probes <= 7 * 8 * smaller_word_count.max(1));
                assert_eq!(counts.word_skips, 7);
            } else {
                assert!(counts.pending_sets.is_empty());
                assert!(counts.pending_words.is_empty());
                assert_eq!(counts.membership_scans, vec![member_count; 8]);
                assert_eq!(counts.membership_owners, member_count * 8);
                assert_eq!(counts.pending_lookups, 0);
                assert_eq!(counts.word_checks, 0);
            }
            evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
            for (index, &worker) in workers.iter().enumerate() {
                let expected = if index >= member_count { 0 } else { 8 };
                assert_eq!(
                    raw.overlap_scores.scores.get(&worker).copied().unwrap_or(0),
                    expected
                );
                assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&expected));
            }
            let hints = raw.kv_transfer_candidates.unwrap();
            assert_eq!(hints.block_hashes.len(), 8);
            assert_eq!(hints.owner_prefix_blocks.len(), member_count);
        }
    }

    #[test]
    fn coarse_generation_filter_skips_spaced_missing_workers_and_preserves_output() {
        let blocks = 64;
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        let residency = KvCacheGroupResidency::new();
        for (index, &worker) in workers.iter().enumerate() {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            // Give each captured layout a different fine generation word.
            for _ in 0..64 {
                residency.clear_worker(worker);
            }
            if index % 2 == 0 {
                for position in 1..=blocks {
                    store_device_checkpoint(&residency, worker, position, 1);
                }
            } else {
                // These states exist, but none lies on the queried chain.
                store_device_checkpoint(&residency, worker, 999, 1);
            }
        }
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        evaluator.reserve_workers(workers.len());
        let mut raw = MatchDetails::default();
        for &worker in &workers {
            evaluator.observe_worker(worker);
            raw.overlap_scores.scores.insert(worker, blocks as u32);
            raw.last_matched_hashes
                .insert(worker, ExternalSequenceBlockHash(blocks));
        }
        for position in 1..=blocks {
            evaluator.observe_block(position as usize, ExternalSequenceBlockHash(position));
        }
        raw.kv_transfer_candidates = Some(KvTransferCandidates {
            block_hashes: (1..=blocks).map(ExternalSequenceBlockHash).collect(),
            owner_prefix_blocks: workers
                .iter()
                .map(|&worker| (worker.into(), blocks as usize))
                .collect(),
            source_intervals: Vec::new(),
            routing_snapshot: None,
        });
        let mut counts = ResolutionCounts::default();
        evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
        assert_eq!(counts.pending_sets, [32]);
        assert_eq!(counts.pending_words, [32]);
        assert_eq!(counts.pending_coarse_words.len(), 1);
        assert_eq!(counts.membership_scans, [32]);
        assert_eq!(counts.membership_owners, 32);
        assert_eq!(counts.pending_lookups, 0);
        assert_eq!(counts.coarse_checks, 63);
        assert_eq!(counts.coarse_skips, 63);
        assert!(counts.coarse_probes <= 63 * 8);
        assert_eq!(counts.word_checks, 0);
        evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
        for (index, &worker) in workers.iter().enumerate() {
            let expected = if index % 2 == 0 { blocks as u32 } else { 0 };
            assert_eq!(
                raw.overlap_scores.scores.get(&worker).copied().unwrap_or(0),
                expected
            );
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&expected));
            assert_eq!(
                raw.last_matched_hashes.get(&worker).copied(),
                (expected > 0).then_some(ExternalSequenceBlockHash(blocks))
            );
        }
        let hints = raw.kv_transfer_candidates.unwrap();
        assert_eq!(hints.block_hashes.len(), blocks as usize);
        assert_eq!(
            hints.owner_prefix_blocks,
            workers
                .iter()
                .step_by(2)
                .map(|&worker| (worker.into(), blocks as usize))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn reverse_resolution_scans_a_smaller_membership_and_removes_only_resolved_workers() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let unmatched = WorkerWithDpRank::new(3, 0);
        let residency = KvCacheGroupResidency::new();
        for worker in [first, second, unmatched] {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        for (worker, position) in [
            (first, 8),
            (unmatched, 8),
            (first, 7),
            (second, 6),
            (first, 5),
        ] {
            store_device_checkpoint(&residency, worker, position, 1);
        }
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        for worker in [first, second, unmatched] {
            evaluator.observe_worker(worker);
        }
        for position in 1..=8 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        let mut raw = MatchDetails::default();
        raw.overlap_scores.scores.insert(first, 5);
        raw.overlap_scores.scores.insert(second, 8);
        let mut counts = ResolutionCounts::default();
        evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
        assert!(counts.pending_sets.is_empty());
        assert!(counts.pending_words.is_empty());
        assert_eq!(counts.membership_scans, [2, 1, 1, 1]);
        assert_eq!(counts.membership_owners, 5);
        assert_eq!(counts.pending_lookups, 0);
        assert_eq!(counts.word_checks, 0);
        assert_eq!(counts.word_skips, 0);
        evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
        assert_eq!(raw.overlap_scores.scores.get(&first), Some(&5));
        assert_eq!(raw.overlap_scores.scores.get(&second), Some(&6));
        assert!(!raw.hybrid_resume_caps.contains_key(&unmatched));
        assert!(evaluator.workers[&unmatched].resume_cap.is_none());
    }

    #[test]
    fn singleton_memberships_keep_long_missing_tails_free_of_pending_metadata() {
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        let residency = KvCacheGroupResidency::new();
        for (index, &worker) in workers.iter().enumerate() {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            if index == 0 {
                for position in 1..=1024 {
                    store_device_checkpoint(&residency, worker, position, 1);
                }
            } else {
                store_device_checkpoint(&residency, worker, 9999, 1);
            }
        }
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        let mut raw = MatchDetails::default();
        for &worker in &workers {
            evaluator.observe_worker(worker);
            raw.overlap_scores.scores.insert(worker, 1024);
        }
        for position in 1..=1024 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        let mut counts = ResolutionCounts::default();
        evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
        assert_eq!(counts.membership_scans, vec![1; 1024]);
        assert_eq!(counts.membership_owners, 1024);
        assert!(counts.pending_sets.is_empty());
        assert!(counts.pending_words.is_empty());
        assert!(counts.pending_coarse_words.is_empty());
        assert_eq!(counts.word_checks, 0);
        assert_eq!(counts.coarse_checks, 0);
        evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
        for (index, &worker) in workers.iter().enumerate() {
            let expected = if index == 0 { 1024 } else { 0 };
            assert_eq!(
                raw.overlap_scores.scores.get(&worker).copied().unwrap_or(0),
                expected
            );
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&expected));
        }
    }

    #[test]
    fn wider_earlier_memberships_switch_to_indexing_without_losing_the_current_candidate() {
        let workers: Vec<_> = (1..=64).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        let residency = KvCacheGroupResidency::new();
        for (index, &worker) in workers.iter().enumerate() {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            if index < 32 {
                for position in 1..=62 {
                    store_device_checkpoint(&residency, worker, position, 1);
                }
            } else {
                store_device_checkpoint(&residency, worker, 999, 1);
            }
        }
        store_device_checkpoint(&residency, workers[0], 64, 1);
        store_device_checkpoint(&residency, workers[1], 63, 1);
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        let mut raw = MatchDetails::default();
        for &worker in &workers {
            evaluator.observe_worker(worker);
            raw.overlap_scores.scores.insert(worker, 64);
        }
        for position in 1..=64 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        let mut counts = ResolutionCounts::default();
        evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
        assert_eq!(counts.pending_sets, [62]);
        assert_eq!(counts.membership_scans, [1, 1, 32]);
        assert_eq!(counts.membership_owners, 34);
        assert_eq!(counts.word_checks, 62);
        assert_eq!(counts.word_skips, 61);
        evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
        for (index, &worker) in workers.iter().enumerate() {
            let expected = match index {
                0 => 64,
                1 => 63,
                2..32 => 62,
                _ => 0,
            };
            assert_eq!(
                raw.overlap_scores.scores.get(&worker).copied().unwrap_or(0),
                expected
            );
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&expected));
            assert_eq!(
                raw.last_matched_hashes.get(&worker).copied(),
                (expected > 0).then_some(ExternalSequenceBlockHash(expected as u64))
            );
        }
    }

    #[test]
    fn pending_worker_lookup_preserves_raw_bounds_alignment_dp_and_generation() {
        for case in [
            "bound",
            "bound_missing",
            "alignment",
            "alignment_missing",
            "dp",
            "generation",
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
            if case == "generation" {
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
            let mut counts = ResolutionCounts::default();
            evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
            assert_eq!(counts.pending_sets, [1], "{case}");
            assert_eq!(counts.membership_scans.len(), 1, "{case}");
            if matches!(case, "bound" | "alignment" | "alignment_missing") {
                assert!(counts.pending_lookups > 0, "{case}");
            } else {
                assert_eq!(counts.pending_lookups, 0, "{case}");
            }
            if matches!(case, "dp" | "generation") {
                assert_eq!(counts.pending_found, 0);
                assert_eq!(counts.word_skips, 7);
            }
            evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
            let expected = match case {
                "bound" => 3,
                "alignment" => 2,
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
    fn no_pending_set_is_allocated_without_an_earlier_feasible_search() {
        let worker = WorkerWithDpRank::new(1, 0);
        let other = WorkerWithDpRank::new(2, 0);
        for (positions, zero_bound) in [(0, false), (1, false), (8, true)] {
            let residency = KvCacheGroupResidency::new();
            for worker in [worker, other] {
                residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
            }
            for position in 1..=8 {
                store_device_checkpoint(&residency, other, position, 1);
            }
            let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
            evaluator.observe_worker(worker);
            evaluator.observe_worker(other);
            for position in 1..=positions {
                evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
            }
            let mut raw = MatchDetails::default();
            raw.overlap_scores
                .scores
                .insert(worker, if zero_bound { 0 } else { 8 });
            let mut counts = ResolutionCounts::default();
            evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
            assert!(counts.pending_sets.is_empty());
            assert!(counts.pending_words.is_empty());
            assert!(counts.pending_coarse_words.is_empty());
            assert_eq!(counts.coarse_checks, 0);
            assert_eq!(counts.pending_lookups, 0);
            assert_eq!(counts.word_checks, 0);
            evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&0));
            assert!(!raw.hybrid_resume_caps.contains_key(&other));
            assert!(evaluator.workers[&other].resume_cap.is_none());
        }
    }

    #[test]
    fn pending_raw_bound_skips_deeper_memberships_before_accessing_the_view() {
        let full = WorkerWithDpRank::new(1, 0);
        let limited = WorkerWithDpRank::new(2, 0);
        let full_others = [WorkerWithDpRank::new(3, 0), WorkerWithDpRank::new(4, 0)];
        let residency = KvCacheGroupResidency::new();
        for worker in [full, limited].into_iter().chain(full_others) {
            residency.declare_groups(ResidencyOwner::worker(worker), &[0]);
        }
        for position in 1..=1024 {
            for worker in [full].into_iter().chain(full_others) {
                store_device_checkpoint(&residency, worker, position, 1);
            }
        }
        store_device_checkpoint(&residency, limited, 8, 1);
        let mut evaluator = HybridPrefixEvaluator::new(&residency, None);
        for worker in [full, limited].into_iter().chain(full_others) {
            evaluator.observe_worker(worker);
        }
        for position in 1..=1024 {
            evaluator.observe_block(position, ExternalSequenceBlockHash(position as u64));
        }
        let mut raw = MatchDetails::default();
        raw.overlap_scores.scores.insert(full, 1024);
        raw.overlap_scores.scores.insert(limited, 8);
        for worker in full_others {
            raw.overlap_scores.scores.insert(worker, 1024);
        }
        let mut counts = ResolutionCounts::default();
        evaluator.prepare_device_observed(&raw, |probe| counts.observe(probe));
        assert_eq!(counts.bound_skips, 1015);
        assert_eq!(counts.membership_scans, [3]);
        assert_eq!(counts.membership_owners, 3);
        assert_eq!(counts.word_checks, 1);
        assert_eq!(counts.pending_lookups, 1);
        evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
        assert_eq!(raw.overlap_scores.scores.get(&full), Some(&1024));
        assert_eq!(raw.overlap_scores.scores.get(&limited), Some(&8));
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
        assert_eq!(evaluator.device_positions.capacity(), 0);
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
    fn worker_and_legacy_capture_keep_the_first_generation_across_tiers() {
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
            assert!(
                raw.overlap_scores.scores.is_empty(),
                "legacy_first={legacy_first}"
            );
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&0));
            assert_eq!(lower[&StorageTier::HostPinned].hits[&worker], 0);
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
    fn checkpoint_hints_for_untracked_or_stale_owners_grant_no_device_credit() {
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
            assert_eq!(evaluator.device_positions.len(), 1);

            let mut raw = MatchDetails::default();
            raw.overlap_scores.scores.insert(worker, 3);
            raw.last_matched_hashes
                .insert(worker, ExternalSequenceBlockHash(3));
            evaluator.prepare_device(&raw);
            evaluator.finish(&mut raw, &mut HashMap::new(), &Default::default());
            assert!(raw.overlap_scores.scores.is_empty(), "stale={stale}");
            assert_eq!(raw.hybrid_resume_caps.get(&worker), Some(&0));
        }
    }

    #[tokio::test]
    async fn device_membership_changes_during_tree_walk_fall_back_to_an_unchanged_boundary() {
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
            revisions: Option<(u64, u64)>,
        }

        impl MatchObserver for InterleavingObserver<'_> {
            fn observe_worker(&mut self, worker: WorkerWithDpRank) {
                self.evaluator.observe_worker(worker);
            }

            fn observe_cache_owner(&mut self, owner: ResidencyOwnerKey) {
                self.evaluator.observe_cache_owner(owner);
            }

            fn observe_block(&mut self, position: usize, hash: ExternalSequenceBlockHash) {
                self.evaluator.observe_block(position, hash);
                if position == 2 {
                    let before = self
                        .evaluator
                        .device_positions
                        .last()
                        .unwrap()
                        .membership_revision;
                    // Deterministically interleave a separate residency writer.
                    // These events do not mutate or reenter the attention tree.
                    for event in self.updates.drain(..) {
                        self.residency.apply(&event);
                    }
                    let after = self
                        .residency
                        .read_view()
                        .checkpoint_hint_revision(hash, StorageTier::Device)
                        .unwrap();
                    self.revisions = Some((before, after));
                }
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
            for mutation in ["unchanged", "add_owner", "remove_readd"] {
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
                    "remove_readd" => {
                        let stored = checkpoint(first, 2);
                        let mut removed = stored.clone();
                        removed.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: vec![hashes[1]],
                        });
                        vec![removed, stored]
                    }
                    _ => Vec::new(),
                };
                let mut observer = InterleavingObserver {
                    evaluator: HybridPrefixEvaluator::new(&residency, None),
                    residency: &residency,
                    updates,
                    revisions: None,
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
                let (before, after) = observer.revisions.unwrap();
                assert_eq!(before == after, mutation == "unchanged", "{context}");
                observer.evaluator.prepare_device(&raw);
                observer
                    .evaluator
                    .finish(&mut raw, &mut HashMap::new(), &Default::default());

                // Any changed H2 membership invalidates the whole boundary,
                // including first's still-valid state when second was added.
                // It cannot grant second new credit or accept a remove/readd ABA.
                let expected = if mutation == "unchanged" { 2 } else { 1 };
                assert_eq!(raw.overlap_scores.scores[&first], expected, "{context}");
                assert!(
                    !raw.overlap_scores.scores.contains_key(&second),
                    "{context}"
                );
                assert_eq!(
                    raw.hybrid_resume_caps.get(&first),
                    Some(&expected),
                    "{context}"
                );
                assert_eq!(raw.hybrid_resume_caps.get(&second), Some(&0), "{context}");
                assert_eq!(
                    raw.last_matched_hashes[&first],
                    hashes[expected as usize - 1],
                    "{context}"
                );
                let hints = raw.kv_transfer_candidates.unwrap();
                assert_eq!(hints.block_hashes, hashes[..expected as usize], "{context}");
                assert_eq!(
                    hints.owner_prefix_blocks,
                    vec![(first.into(), expected as usize)],
                    "{context}"
                );
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
