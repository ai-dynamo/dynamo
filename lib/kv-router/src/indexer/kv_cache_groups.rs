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
//! Recurrent events update per-owner checkpoint membership, never attention edges.
//! Complete layouts include groups with no resident checkpoints and survive dumps.

use std::{
    cell::OnceCell,
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use arc_swap::{ArcSwap, Guard};
use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHashSet};

use super::concurrent_radix_tree_compressed::mix_worker;
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheGroupKind, KvCacheGroupTag,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, ResetScope, ResidencyDomain,
    ResidencyOwner, ResidencyOwnerKey, RouterEvent, StorageTier, WorkerId, WorkerWithDpRank,
};

type GroupPresence = FxHashMap<u32, FxHashMap<u64, u8>>;

const CHECKPOINT_SHARDS: usize = 256;
const COHORT_CACHE_CELLS: usize = 64;
const GENERATION_WORD_MASK: u64 = u64::MAX >> 6;
const GENERATION_TIER_SHIFT: u32 = 62;
const GENERATION_LEVEL_SHIFT: u32 = 60;

type CheckpointShardMap = FxHashMap<u64, Arc<CompleteMembership>>;
type CapturedCheckpointShards = [OnceCell<Arc<CheckpointShardMap>>; CHECKPOINT_SHARDS];

/// The existing per-owner layout allocation also carries a certificate-only
/// epoch. Zero closes certification before affected shard proofs are invalidated;
/// a nonzero value qualifies exactly that complete, alignment-one generation.
#[derive(Debug, Default)]
struct OwnerLayoutHandle {
    snapshot: ArcSwap<CapturedLayout>,
    eligible_generation: AtomicU64,
}

/// Event-published exact Device endpoint eligibility. This is a speculative
/// shortcut only: the query must check its first-observation publication cutoff,
/// cover every scored worker, and revalidate after constructing the result.
#[derive(Debug)]
pub(crate) struct EndpointCertificate {
    cohort: Arc<CheckpointCohort>,
    // Table identity is checked by the optional standalone-indexer lookup.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    table: Arc<AtomicU64>,
    publication: AtomicU64,
}

impl EndpointCertificate {
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn workers(&self) -> &FxHashSet<WorkerWithDpRank> {
        &self.cohort.workers
    }

    pub(crate) fn publication_id(&self) -> u64 {
        self.publication.load(Ordering::Acquire)
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.publication_id() != 0
    }

    fn invalidate(&self) {
        self.publication.store(0, Ordering::Release);
    }
}

/// Immutable exact eligibility can be shared across hashes and generations;
/// each endpoint still owns its independent publication and invalidation flag.
#[derive(Debug)]
struct CheckpointCohort {
    workers: FxHashSet<WorkerWithDpRank>,
    // A cache selector only. Equality and transition reuse remain owner-exact.
    fingerprint: u64,
}

#[derive(Debug)]
struct CohortTransition {
    // The Weak keeps the previous allocation's identity from being reused.
    previous: Weak<CheckpointCohort>,
    worker: WorkerWithDpRank,
    include: bool,
}

#[derive(Debug, Default)]
struct CohortCacheEntry {
    fingerprint: u64,
    cohort: Weak<CheckpointCohort>,
    transition: Option<CohortTransition>,
}

/// One query-scoped shard guard protects the immutable proof without cloning
/// its hot Arc. Hash lookup derives each borrow, so this has no self-reference.
#[derive(Debug)]
#[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
pub(crate) struct EndpointCertificateRead {
    snapshot: Guard<Arc<CheckpointShardMap>>,
    hash: u64,
}

// This proof reader is used by the optional standalone-indexer lookup.
#[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
impl EndpointCertificateRead {
    fn certificate(&self) -> &EndpointCertificate {
        self.snapshot
            .get(&self.hash)
            .and_then(|membership| membership.endpoint_certificate.as_deref())
            .expect("a protected endpoint proof remains in its immutable snapshot")
    }

    pub(crate) fn workers(&self) -> &FxHashSet<WorkerWithDpRank> {
        self.certificate().workers()
    }

    #[cfg(test)]
    fn publication_id(&self) -> u64 {
        self.certificate().publication_id()
    }

    #[cfg(test)]
    fn is_valid(&self) -> bool {
        self.certificate().is_valid()
    }
}

fn compare_owners(left: &ResidencyOwner, right: &ResidencyOwner) -> std::cmp::Ordering {
    match (left, right) {
        (ResidencyOwner::Worker(left), ResidencyOwner::Worker(right)) => left.cmp(right),
        (ResidencyOwner::CacheOwner(left), ResidencyOwner::CacheOwner(right)) => left.cmp(right),
        (ResidencyOwner::Worker(_), ResidencyOwner::CacheOwner(_)) => std::cmp::Ordering::Less,
        (ResidencyOwner::CacheOwner(_), ResidencyOwner::Worker(_)) => std::cmp::Ordering::Greater,
    }
}

/// Read-only complete membership for one hash and tier inside a scoped callback.
/// Facts still require the caller's captured owner layout/generation before credit.
/// The private membership reference borrows the immutable shard held by that callback.
#[derive(Clone, Copy)]
pub struct CheckpointMembershipView<'a> {
    membership: &'a CompleteMembership,
    tier_index: usize,
}

impl<'a> CheckpointMembershipView<'a> {
    fn new(membership: &'a CompleteMembership, tier_index: usize) -> Self {
        Self {
            membership,
            tier_index,
        }
    }

    fn owners(&self) -> &[(ResidencyOwner, u64)] {
        &self.membership.owners[self.tier_index]
    }

    pub fn len(&self) -> usize {
        self.owners().len()
    }

    pub fn is_empty(&self) -> bool {
        self.owners().is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ResidencyOwner, u64)> + '_ {
        self.owners().iter().copied()
    }

    pub fn generation_for(&self, owner: ResidencyOwner) -> Option<u64> {
        let owners = self.owners();
        owners
            .binary_search_by(|entry| compare_owners(&entry.0, &owner))
            .ok()
            .map(|index| owners[index].1)
    }

    /// Sorted unique nonempty generation words for this tier. Disjoint generation sets
    /// rule out an eligible owner; an intersection still requires exact owner
    /// and captured-generation validation. The range is found only on demand.
    /// The returned slice-backed iterator supports O(1) cloning, front/back
    /// access and `nth`, so callers can skip sparse ranges without scanning.
    pub fn generation_words(
        &self,
    ) -> impl ExactSizeIterator<Item = (u64, u64)> + DoubleEndedIterator + Clone + '_ {
        self.words_at_level(0)
    }

    /// Sorted unique occupancy words for this tier's generation-word indices.
    /// A bit names a nonempty generation word, so disjoint occupancy rules out
    /// an eligible owner without comparing every sparse generation word.
    /// The iterator has the same O(1) clone/indexing guarantees as `generation_words`.
    pub fn generation_word_groups(
        &self,
    ) -> impl ExactSizeIterator<Item = (u64, u64)> + DoubleEndedIterator + Clone + '_ {
        self.words_at_level(1)
    }

    fn words_at_level(
        &self,
        level: u32,
    ) -> impl ExactSizeIterator<Item = (u64, u64)> + DoubleEndedIterator + Clone + '_ {
        let tier_start = ((self.tier_index as u64) << GENERATION_TIER_SHIFT)
            | (u64::from(level) << GENERATION_LEVEL_SHIFT);
        let word_mask = GENERATION_WORD_MASK >> (6 * level);
        let words = &self.membership.generation_words;
        let start = words.partition_point(|&(key, _)| key < tier_start);
        let end = words.partition_point(|&(key, _)| key <= (tier_start | word_mask));
        words[start..end]
            .iter()
            .map(move |&(key, mask)| (key & word_mask, mask))
    }

    /// Returns this tier's generation bits for one word, or zero if absent.
    pub fn generation_mask(&self, word: u64) -> u64 {
        self.mask_at_level(word, 0)
    }

    /// Returns occupancy bits for one group of generation-word indices.
    pub fn generation_word_group_mask(&self, group: u64) -> u64 {
        self.mask_at_level(group, 1)
    }

    fn mask_at_level(&self, word: u64, level: u32) -> u64 {
        if word > GENERATION_WORD_MASK >> (6 * level) {
            return 0;
        }
        let key = ((self.tier_index as u64) << GENERATION_TIER_SHIFT)
            | (u64::from(level) << GENERATION_LEVEL_SHIFT)
            | word;
        self.membership
            .generation_words
            .binary_search_by_key(&key, |&(key, _)| key)
            .ok()
            .map_or(0, |index| self.membership.generation_words[index].1)
    }
}

// Keep the revision and Device owner header together, ahead of cold summaries.
// Rust's default field reordering puts the revision behind those summaries.
#[derive(Debug, Default, Clone)]
#[repr(C)]
struct CompleteMembership {
    revision: u64,
    endpoint_certificate: Option<Arc<EndpointCertificate>>,
    // Sorted by the exact logical owner, with one generation per owner/tier.
    owners: [Vec<(ResidencyOwner, u64)>; 4],
    // One sparse sidecar shared by all tiers. High two key bits encode tier;
    // bit 60 encodes fine words or their occupancy groups. Low 58/52 bits
    // encode generation / 64 or / 4096. Positive matches remain owner-exact.
    generation_words: Vec<(u64, u64)>,
}

impl CompleteMembership {
    fn update(&mut self, owner: ResidencyOwner, generation: u64, mask: u8) {
        for (index, owners) in self.owners.iter_mut().enumerate() {
            let found = owners.binary_search_by(|entry| compare_owners(&entry.0, &owner));
            let present = mask & (1 << index) != 0;
            let removed_generation = match (found, present) {
                (Ok(position), true) if owners[position].1 != generation => {
                    Some(std::mem::replace(&mut owners[position].1, generation))
                }
                (Err(position), true) => {
                    owners.insert(position, (owner, generation));
                    None
                }
                (Ok(position), false) if owners[position].1 == generation => {
                    owners.remove(position);
                    Some(generation)
                }
                _ => continue,
            };
            // Tests and generation wrap can associate the same token with more
            // than one owner. Never clear a bit while another exact fact uses it.
            if let Some(previous) = removed_generation
                && !owners.iter().any(|&(_, token)| token == previous)
            {
                Self::update_generation_word(&mut self.generation_words, index, previous, false);
            }
            if present {
                Self::update_generation_word(&mut self.generation_words, index, generation, true);
            }
        }
    }

    fn update_generation_word(
        words: &mut Vec<(u64, u64)>,
        tier_index: usize,
        generation: u64,
        present: bool,
    ) {
        let key = ((tier_index as u64) << GENERATION_TIER_SHIFT) | (generation >> 6);
        let bit = 1_u64 << (generation & 63);
        if let Some(occupied) = Self::update_sparse_word(words, key, bit, present) {
            let group_key = ((tier_index as u64) << GENERATION_TIER_SHIFT)
                | (1_u64 << GENERATION_LEVEL_SHIFT)
                | (generation >> 12);
            let _ =
                Self::update_sparse_word(words, group_key, 1 << ((generation >> 6) & 63), occupied);
        }
    }

    /// Returns a transition only when the entire sparse word appears/disappears.
    fn update_sparse_word(
        words: &mut Vec<(u64, u64)>,
        key: u64,
        bit: u64,
        present: bool,
    ) -> Option<bool> {
        match (words.binary_search_by_key(&key, |&(key, _)| key), present) {
            (Ok(index), true) => words[index].1 |= bit,
            (Err(index), true) => {
                // One fine word and its occupancy group need two slots, not
                // Vec's minimum four-element growth allocation.
                if words.capacity() == 0 {
                    words.reserve_exact(2);
                }
                words.insert(index, (key, bit));
                return Some(true);
            }
            (Ok(index), false) => {
                words[index].1 &= !bit;
                if words[index].1 == 0 {
                    words.remove(index);
                    return Some(false);
                }
            }
            (Err(_), false) => {}
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.owners.iter().all(Vec::is_empty)
    }
}

#[derive(Debug, Default)]
struct CheckpointShard {
    entries: ArcSwap<CheckpointShardMap>,
    // Revisions only compare memberships for the same hash, which always maps
    // to this shard. Retaining the counter across deletion prevents ABA reuse.
    writer: Mutex<u64>,
}

#[derive(Debug)]
struct CheckpointMembershipIndex {
    shards: [CheckpointShard; CHECKPOINT_SHARDS],
    publication: Arc<AtomicU64>,
    // Fixed-size, weak-only writer cache. Dense-only tables allocate no cells,
    // and dead cohorts never retain their worker tables through this cache.
    cohorts: OnceLock<Box<[Mutex<CohortCacheEntry>; COHORT_CACHE_CELLS]>>,
}

impl Default for CheckpointMembershipIndex {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| CheckpointShard::default()),
            publication: Arc::default(),
            cohorts: OnceLock::new(),
        }
    }
}

impl CheckpointMembershipIndex {
    fn shard(hash: u64) -> usize {
        // Sequence hashes are already uniformly distributed. Folding higher bits
        // also keeps synthetic sequential hashes spread across the shards.
        (hash ^ (hash >> 32) ^ (hash >> 16) ^ (hash >> 8)) as usize & (CHECKPOINT_SHARDS - 1)
    }

    fn cohort_cell(&self, fingerprint: u64) -> &Mutex<CohortCacheEntry> {
        &self.cohorts.get_or_init(|| {
            Box::new(std::array::from_fn(|_| {
                Mutex::new(CohortCacheEntry::default())
            }))
        })[Self::shard(fingerprint) & (COHORT_CACHE_CELLS - 1)]
    }

    fn intern_cohort(
        &self,
        workers: FxHashSet<WorkerWithDpRank>,
        fingerprint: u64,
    ) -> Arc<CheckpointCohort> {
        let mut cached = self.cohort_cell(fingerprint).lock();
        if cached.fingerprint == fingerprint
            && let Some(cohort) = cached.cohort.upgrade()
            && cohort.workers == workers
        {
            return cohort;
        }
        let cohort = Arc::new(CheckpointCohort {
            workers,
            fingerprint,
        });
        *cached = CohortCacheEntry {
            fingerprint,
            cohort: Arc::downgrade(&cohort),
            transition: None,
        };
        cohort
    }

    fn transition_cohort(
        &self,
        previous: &Arc<CheckpointCohort>,
        worker: WorkerWithDpRank,
        include: bool,
    ) -> Option<Arc<CheckpointCohort>> {
        if previous.workers.contains(&worker) == include {
            return Some(previous.clone());
        }
        if !include && previous.workers.len() == 1 {
            return None;
        }
        let fingerprint = previous.fingerprint ^ mix_worker(worker);
        let mut cached = self.cohort_cell(fingerprint).lock();
        if cached.fingerprint == fingerprint
            && let Some(cohort) = cached.cohort.upgrade()
        {
            let same_transition = cached.transition.as_ref().is_some_and(|transition| {
                transition.previous.as_ptr() == Arc::as_ptr(previous)
                    && transition.worker == worker
                    && transition.include == include
            });
            // A selector collision, or a different path to the same cohort,
            // must establish the complete expected set relation exactly.
            let exact_transition = same_transition
                || if include {
                    cohort.workers.len() == previous.workers.len() + 1
                        && cohort.workers.contains(&worker)
                        && previous.workers.is_subset(&cohort.workers)
                } else {
                    cohort.workers.len() + 1 == previous.workers.len()
                        && !cohort.workers.contains(&worker)
                        && cohort.workers.is_subset(&previous.workers)
                };
            if exact_transition {
                if !same_transition {
                    cached.transition = Some(CohortTransition {
                        previous: Arc::downgrade(previous),
                        worker,
                        include,
                    });
                }
                return Some(cohort);
            }
        }
        let mut workers = previous.workers.clone();
        if include {
            workers.insert(worker);
        } else {
            workers.remove(&worker);
        }
        let cohort = Arc::new(CheckpointCohort {
            workers,
            fingerprint,
        });
        *cached = CohortCacheEntry {
            fingerprint,
            cohort: Arc::downgrade(&cohort),
            transition: Some(CohortTransition {
                previous: Arc::downgrade(previous),
                worker,
                include,
            }),
        };
        Some(cohort)
    }

    /// The owner is already closed and write-locked. A rebuilding writer either
    /// precedes this shard lock (and its new proof is invalidated here), or sees
    /// the closed/new generation. Never acquire an owner lock from this path.
    fn invalidate_certificates(&self, hashes: impl Iterator<Item = u64>) {
        let mut batches: FxHashMap<usize, Vec<u64>> = FxHashMap::default();
        for hash in hashes {
            batches.entry(Self::shard(hash)).or_default().push(hash);
        }
        for (index, hashes) in batches {
            let shard = &self.shards[index];
            let _writer = shard.writer.lock();
            let current = shard.entries.load();
            for hash in hashes {
                if let Some(certificate) = current
                    .get(&hash)
                    .and_then(|membership| membership.endpoint_certificate.as_ref())
                {
                    certificate.invalidate();
                }
            }
        }
    }

    /// The caller holds the owner's write guard through publication. Shard
    /// writers serialize different owners and never hold two shard locks.
    fn publish(
        &self,
        owner: ResidencyOwner,
        updates: Vec<(u64, u64, u8)>,
        registry: Option<&Arc<OwnerRegistry>>,
    ) {
        let mut batches: FxHashMap<usize, Vec<(u64, u64, u8)>> = FxHashMap::default();
        for update in updates {
            batches
                .entry(Self::shard(update.0))
                .or_default()
                .push(update);
        }
        for (shard_index, updates) in batches {
            let shard = &self.shards[shard_index];
            let mut counter = shard.writer.lock();
            // Zero permanently disables live hints after counter exhaustion;
            // mutations still publish, and no revision can wrap or be reused.
            let revision = counter.checked_add(1).unwrap_or(0);
            if revision != 0 {
                *counter = revision;
            }
            let mut next = (*shard.entries.load_full()).clone();
            let mut certificates = Vec::new();
            for (hash, generation, mask) in updates {
                let previous = next
                    .get(&hash)
                    .and_then(|entry| entry.endpoint_certificate.as_ref())
                    .cloned();
                // Preserve this decision before invalidating our own old proof.
                // Another owner's close is serialized by this same shard lock.
                let reuse = previous.as_ref().filter(|proof| proof.is_valid());
                if let Some(previous) = &previous {
                    previous.invalidate();
                }
                let mut membership = next
                    .get(&hash)
                    .map(|entry| (**entry).clone())
                    .unwrap_or_default();
                membership.update(owner, generation, mask);
                membership.revision = revision;
                membership.endpoint_certificate =
                    registry.filter(|_| revision != 0).and_then(|registry| {
                        self.build_certificate(&membership, registry, owner, reuse)
                    });
                if let Some(certificate) = &membership.endpoint_certificate {
                    certificates.push((hash, certificate.clone()));
                }
                if membership.is_empty() {
                    next.remove(&hash);
                } else {
                    next.insert(hash, Arc::new(membership));
                }
            }
            let next = Arc::new(next);
            shard.entries.store(next.clone());
            // Allocation occurs only after the map is visible. A query's cutoff
            // can never admit a proof of facts first published after that cutoff.
            for (hash, certificate) in certificates {
                if !next
                    .get(&hash)
                    .and_then(|membership| membership.endpoint_certificate.as_ref())
                    .is_some_and(|current| Arc::ptr_eq(current, &certificate))
                {
                    continue;
                }
                if let Ok(previous) =
                    self.publication
                        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                            value.checked_add(1)
                        })
                {
                    certificate
                        .publication
                        .store(previous + 1, Ordering::Release);
                }
            }
        }
    }

    fn build_certificate(
        &self,
        membership: &CompleteMembership,
        registry: &Arc<OwnerRegistry>,
        changed_owner: ResidencyOwner,
        previous: Option<&Arc<EndpointCertificate>>,
    ) -> Option<Arc<EndpointCertificate>> {
        let eligible = |owner, generation| {
            generation != 0
                && registry.layouts.get(&owner).is_some_and(|handle| {
                    handle.eligible_generation.load(Ordering::Acquire) == generation
                })
        };
        let cohort = if let Some(previous) = previous {
            if let Some(worker) = changed_owner.as_worker() {
                let generation = membership.owners[0]
                    .binary_search_by(|entry| compare_owners(&entry.0, &changed_owner))
                    .ok()
                    .map(|index| membership.owners[0][index].1);
                let include =
                    generation.is_some_and(|generation| eligible(changed_owner, generation));
                self.transition_cohort(&previous.cohort, worker, include)?
            } else {
                previous.cohort.clone()
            }
        } else {
            let mut workers = FxHashSet::default();
            let mut fingerprint = 0;
            workers.reserve(membership.owners[0].len());
            for &(owner, generation) in &membership.owners[0] {
                if let Some(worker) = owner.as_worker()
                    && eligible(owner, generation)
                    && workers.insert(worker)
                {
                    fingerprint ^= mix_worker(worker);
                }
            }
            if workers.is_empty() {
                return None;
            }
            self.intern_cohort(workers, fingerprint)
        };
        Some(Arc::new(EndpointCertificate {
            cohort,
            table: self.publication.clone(),
            publication: AtomicU64::new(0),
        }))
    }
}

fn tier_index(tier: StorageTier) -> usize {
    match tier {
        StorageTier::Device => 0,
        StorageTier::HostPinned => 1,
        StorageTier::Disk => 2,
        StorageTier::External => 3,
    }
}

fn tier_bit(tier: StorageTier) -> u8 {
    1 << tier_index(tier)
}

#[derive(Debug, Default)]
struct OwnerPresence {
    groups: GroupPresence,
    group_counts: FxHashMap<u64, [u32; 4]>,
    complete_checkpoints: FxHashMap<u64, u8>,
    declared: bool,
    expected_group_count: Option<usize>,
    alignment_blocks: usize,
    generation: u64,
    retired: bool,
    layout_snapshot: Arc<OwnerLayoutHandle>,
}

impl OwnerPresence {
    fn publish_layout(&self) {
        debug_assert_eq!(
            self.layout_snapshot
                .eligible_generation
                .load(Ordering::Relaxed),
            0
        );
        let layout = self.layout();
        self.layout_snapshot
            .snapshot
            .store(Arc::new(CapturedLayout {
                layout,
                generation: self.generation,
            }));
        let eligible = layout.complete_layout && layout.alignment_blocks <= 1;
        self.layout_snapshot.eligible_generation.store(
            if eligible { self.generation } else { 0 },
            Ordering::Release,
        );
    }

    fn layout(&self) -> CheckpointLayout {
        CheckpointLayout {
            alignment_blocks: self.alignment_blocks.max(1),
            complete_layout: !self.retired
                && !self.groups.is_empty()
                && self.expected_group_count == Some(self.groups.len()),
        }
    }

    fn refresh_checkpoint(&mut self, hash: u64) {
        let mask = if self.layout().complete_layout {
            self.group_counts
                .get(&hash)
                .map_or(0, |counts| complete_tiers(counts, self.groups.len()))
        } else {
            0
        };
        if mask == 0 {
            self.complete_checkpoints.remove(&hash);
        } else {
            self.complete_checkpoints.insert(hash, mask);
        }
    }

    fn rebuild_complete_checkpoints(&mut self) {
        self.complete_checkpoints.clear();
        if self.layout().complete_layout {
            for (&hash, counts) in &self.group_counts {
                let mask = complete_tiers(counts, self.groups.len());
                if mask != 0 {
                    self.complete_checkpoints.insert(hash, mask);
                }
            }
        }
    }

    fn rebuild_group_counts(&mut self) {
        self.group_counts.clear();
        for group in self.groups.values() {
            for (&hash, &mask) in group {
                let counts = self.group_counts.entry(hash).or_default();
                for (index, count) in counts.iter_mut().enumerate() {
                    *count += u32::from(mask & (1 << index) != 0);
                }
            }
        }
        self.rebuild_complete_checkpoints();
    }

    fn store(&mut self, group_idx: u32, hash: u64, tier: StorageTier) {
        let bit = tier_bit(tier);
        let mask = self
            .groups
            .entry(group_idx)
            .or_default()
            .entry(hash)
            .or_default();
        if *mask & bit != 0 {
            return;
        }
        *mask |= bit;
        self.group_counts.entry(hash).or_default()[tier_index(tier)] += 1;
        self.refresh_checkpoint(hash);
    }

    fn remove(&mut self, group_idx: u32, hash: u64, tier: StorageTier) {
        let Some(group) = self.groups.get_mut(&group_idx) else {
            return;
        };
        let Some(mask) = group.get_mut(&hash) else {
            return;
        };
        let bit = tier_bit(tier);
        if *mask & bit == 0 {
            return;
        }
        *mask &= !bit;
        if *mask == 0 {
            group.remove(&hash);
        }
        if let Some(counts) = self.group_counts.get_mut(&hash) {
            counts[tier_index(tier)] -= 1;
            if counts.iter().all(|&count| count == 0) {
                self.group_counts.remove(&hash);
            }
        }
        self.refresh_checkpoint(hash);
    }

    fn clear_checkpoints(&mut self) {
        for group in self.groups.values_mut() {
            group.clear();
        }
        self.group_counts.clear();
        self.complete_checkpoints.clear();
    }
}

fn complete_tiers(counts: &[u32; 4], required_groups: usize) -> u8 {
    counts.iter().enumerate().fold(0, |mask, (index, &count)| {
        mask | (u8::from(count as usize == required_groups) << index)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointLayout {
    pub alignment_blocks: usize,
    pub complete_layout: bool,
}

/// Recurrent checkpoints are separate from attention edges and scoped to their owner.
#[derive(Debug, Default, Clone)]
struct OwnerRegistry {
    owners: FxHashMap<ResidencyOwner, Arc<RwLock<OwnerPresence>>>,
    layouts: FxHashMap<ResidencyOwner, Arc<OwnerLayoutHandle>>,
    cache_owners: FxHashMap<ResidencyOwnerKey, ResidencyOwner>,
}

/// Immutable owner layout captured for one query. Keep this token associated
/// with the same owner when validating membership or seeding legacy lookups.
#[derive(Debug, Clone, Copy)]
pub struct CapturedLayout {
    layout: CheckpointLayout,
    generation: u64,
}

impl CapturedLayout {
    pub fn layout(self) -> CheckpointLayout {
        self.layout
    }

    /// Immutable query token for conservative membership intersection. A token
    /// match alone grants no credit without exact owner and layout validation.
    pub fn generation(self) -> u64 {
        self.generation
    }

    pub fn is_eligible_generation(self, generation: u64) -> bool {
        self.generation == generation && self.layout.complete_layout
    }
}

impl Default for CapturedLayout {
    fn default() -> Self {
        Self {
            layout: CheckpointLayout {
                alignment_blocks: 1,
                complete_layout: false,
            },
            generation: 0,
        }
    }
}

/// Candidate layouts are captured once. Legacy membership lookups retain shards
/// lazily; revision-fenced lookups only hold a shard during each call.
#[derive(Debug)]
pub struct CheckpointReadView {
    registry: Arc<OwnerRegistry>,
    layouts: FxHashMap<ResidencyOwner, Option<CapturedLayout>>,
    memberships: Arc<CheckpointMembershipIndex>,
    shards: OnceCell<Box<CapturedCheckpointShards>>,
}

impl CheckpointReadView {
    /// Reuses any prior legacy capture, including an unknown owner. Otherwise
    /// loads the registry's immutable layout without allocating a query entry.
    /// The caller must retain the result to preserve its first observation.
    pub fn capture_layout(&self, owner: ResidencyOwner) -> Option<CapturedLayout> {
        if let Some(&captured) = self.layouts.get(&owner) {
            return captured;
        }
        self.registry
            .layouts
            .get(&owner)
            .map(|handle| **handle.snapshot.load())
    }

    /// Seeds a legacy lookup with the caller's first capture for this same
    /// owner. An existing capture, including `None`, is never replaced.
    pub fn remember_layout(&mut self, owner: ResidencyOwner, captured: Option<CapturedLayout>) {
        self.layouts.entry(owner).or_insert(captured);
    }

    pub fn track_owner(&mut self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        let registry = &self.registry;
        self.layouts
            .entry(owner)
            .or_insert_with(|| {
                registry
                    .layouts
                    .get(&owner)
                    .map(|handle| **handle.snapshot.load())
            })
            .map(CapturedLayout::layout)
    }

    pub fn layout(&self, owner: ResidencyOwner) -> Option<CheckpointLayout> {
        self.layouts
            .get(&owner)
            .copied()
            .flatten()
            .map(|captured| captured.layout)
    }

    pub fn owner_for_key(&self, key: ResidencyOwnerKey) -> Option<ResidencyOwner> {
        self.registry.cache_owners.get(&key).copied()
    }

    pub fn track_cache_owner(&mut self, key: ResidencyOwnerKey) -> Option<ResidencyOwner> {
        let owner = self.owner_for_key(key)?;
        self.track_owner(owner);
        Some(owner)
    }

    #[inline]
    fn membership(&self, hash: ExternalSequenceBlockHash) -> Option<&CompleteMembership> {
        let index = CheckpointMembershipIndex::shard(hash.0);
        let shards = self
            .shards
            .get_or_init(|| Box::new(std::array::from_fn(|_| OnceCell::new())));
        let shard =
            shards[index].get_or_init(|| self.memberships.shards[index].entries.load_full());
        shard.get(&hash.0).map(Arc::as_ref)
    }

    /// Returns a live hint for any owner in this tier, without retaining a shard.
    /// This is not proof of owner or generation eligibility. A revision visitor
    /// must validate this hint and the owner's captured layout before credit.
    #[inline]
    pub fn checkpoint_hint_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
    ) -> Option<u64> {
        let shard = self.memberships.shards[CheckpointMembershipIndex::shard(hash.0)]
            .entries
            .load();
        let membership = shard.get(&hash.0)?;
        (membership.revision != 0 && !membership.owners[tier_index(tier)].is_empty())
            .then_some(membership.revision)
    }

    /// Borrows indexed raw membership only for the exact nonzero hash revision.
    /// The view cannot outlive this call's immutable shard guard. A fact alone
    /// grants no credit: validate the owner's captured layout and generation.
    ///
    /// ```compile_fail
    /// use dynamo_kv_router::indexer::KvCacheGroupResidency;
    /// use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, StorageTier};
    /// let residency = KvCacheGroupResidency::new();
    /// let view = residency.read_view();
    /// let mut escaped = None;
    /// view.with_membership_at_revision(
    ///     ExternalSequenceBlockHash(1), StorageTier::Device, 1,
    ///     |membership| escaped = Some(membership),
    /// );
    /// ```
    pub fn with_membership_at_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
        mut visit: impl for<'a> FnMut(CheckpointMembershipView<'a>),
    ) {
        if revision == 0 {
            return;
        }
        let shard = self.memberships.shards[CheckpointMembershipIndex::shard(hash.0)]
            .entries
            .load();
        let Some(membership) = shard.get(&hash.0) else {
            return;
        };
        if membership.revision != revision {
            return;
        }
        visit(CheckpointMembershipView::new(membership, tier_index(tier)));
    }

    /// Visits raw membership facts only while this hash has the exact observed
    /// revision. A fact alone grants no credit: the caller must match the owner
    /// to its first captured layout and validate generation and completeness
    /// with `CapturedLayout::is_eligible_generation` before granting credit.
    /// The immutable shard guard is scoped to this call and is never cached.
    pub fn for_each_membership_at_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
        mut visit: impl FnMut(ResidencyOwner, u64),
    ) {
        self.with_membership_at_revision(hash, tier, revision, |membership| {
            for (owner, generation) in membership.iter() {
                visit(owner, generation);
            }
        });
    }

    /// Visits eligible owners only if this hash still has the observed revision.
    /// A store, removal, or lifecycle change after the hint cannot introduce
    /// credit. Eligibility uses layouts already captured by this view.
    pub fn for_each_owner_at_revision(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
        mut visit: impl FnMut(ResidencyOwner),
    ) {
        self.for_each_membership_at_revision(hash, tier, revision, |owner, generation| {
            if self
                .layouts
                .get(&owner)
                .copied()
                .flatten()
                .is_some_and(|captured| captured.is_eligible_generation(generation))
            {
                visit(owner);
            }
        });
    }

    pub fn owners_at(
        &self,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
    ) -> impl Iterator<Item = ResidencyOwner> + '_ {
        self.membership(hash)
            .into_iter()
            .flat_map(move |membership| membership.owners[tier_index(tier)].iter())
            .filter_map(|&(owner, generation)| {
                let captured = self.layouts.get(&owner)?.as_ref()?;
                (captured.generation == generation && captured.layout.complete_layout)
                    .then_some(owner)
            })
    }
}

#[derive(Debug, Default)]
pub struct KvCacheGroupResidency {
    registry: ArcSwap<OwnerRegistry>,
    registry_writer: Mutex<()>,
    active: AtomicBool,
    memberships: Arc<CheckpointMembershipIndex>,
    next_generation: AtomicU64,
}

impl KvCacheGroupResidency {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Freeze the publication frontier at the speculative query's first worker
    /// observation, before it decides to skip legacy per-worker layout capture.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn capture_cutoff(&self) -> u64 {
        self.memberships.publication.load(Ordering::Acquire)
    }

    /// Return only a proof of facts published before the captured frontier.
    /// A later proof must never learn a newly eligible owner for this attempt.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn endpoint_certificate_read(
        &self,
        hash: ExternalSequenceBlockHash,
        cutoff: u64,
    ) -> Option<EndpointCertificateRead> {
        let snapshot = self.memberships.shards[CheckpointMembershipIndex::shard(hash.0)]
            .entries
            .load();
        let certificate = snapshot.get(&hash.0)?.endpoint_certificate.as_ref()?;
        let born = certificate.publication_id();
        (born != 0 && born <= cutoff && self.is_certificate_valid(certificate)).then_some(
            EndpointCertificateRead {
                snapshot,
                hash: hash.0,
            },
        )
    }

    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn is_certificate_read_valid(&self, read: &EndpointCertificateRead) -> bool {
        self.is_certificate_valid(read.certificate())
    }

    #[cfg(test)]
    fn endpoint_certificate(
        &self,
        hash: ExternalSequenceBlockHash,
        cutoff: u64,
    ) -> Option<Arc<EndpointCertificate>> {
        let read = self.endpoint_certificate_read(hash, cutoff)?;
        read.snapshot.get(&hash.0)?.endpoint_certificate.clone()
    }

    /// Recheck after result construction. Invalidation is irreversible, and an
    /// unrelated residency table's coincident epoch must not validate this proof.
    #[cfg_attr(not(feature = "standalone-indexer"), allow(dead_code))]
    pub(crate) fn is_certificate_valid(&self, certificate: &EndpointCertificate) -> bool {
        Arc::ptr_eq(&self.memberships.publication, &certificate.table) && certificate.is_valid()
    }

    pub fn read_view(&self) -> CheckpointReadView {
        CheckpointReadView {
            registry: self.registry.load_full(),
            layouts: FxHashMap::default(),
            memberships: self.memberships.clone(),
            shards: OnceCell::new(),
        }
    }

    fn owner(&self, owner: ResidencyOwner) -> Option<Arc<RwLock<OwnerPresence>>> {
        self.registry.load().owners.get(&owner).cloned()
    }

    fn fresh_generation(&self) -> u64 {
        self.next_generation.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn publish_memberships(&self, owner: ResidencyOwner, updates: Vec<(u64, u64, u8)>) {
        let registry = self.registry.load_full();
        self.memberships.publish(owner, updates, Some(&registry));
    }

    fn close_layout_certificates(&self, state: &OwnerPresence) {
        state
            .layout_snapshot
            .eligible_generation
            .store(0, Ordering::Release);
        self.memberships
            .invalidate_certificates(state.complete_checkpoints.keys().copied());
    }

    fn publish_reset(
        &self,
        owner: ResidencyOwner,
        previous_generation: u64,
        previous: FxHashMap<u64, u8>,
        state: &OwnerPresence,
    ) {
        let mut updates = Vec::with_capacity(previous.len() + state.complete_checkpoints.len());
        updates.extend(
            previous
                .into_keys()
                .map(|hash| (hash, previous_generation, 0)),
        );
        updates.extend(
            state
                .complete_checkpoints
                .iter()
                .map(|(&hash, &mask)| (hash, state.generation, mask)),
        );
        self.publish_memberships(owner, updates);
    }

    fn clear_owner(&self, owner: ResidencyOwner, state: &mut OwnerPresence) {
        self.close_layout_certificates(state);
        let previous = std::mem::take(&mut state.complete_checkpoints);
        let previous_generation = state.generation;
        state.clear_checkpoints();
        state.generation = self.fresh_generation();
        state.publish_layout();
        self.publish_reset(owner, previous_generation, previous, state);
    }

    fn get_or_create(&self, owner: ResidencyOwner) -> Arc<RwLock<OwnerPresence>> {
        if let Some(state) = self.owner(owner) {
            return state;
        }
        let _writer = self.registry_writer.lock();
        let current = self.registry.load_full();
        if let Some(state) = current.owners.get(&owner) {
            return state.clone();
        }
        let mut next = (*current).clone();
        let state = OwnerPresence {
            generation: self.fresh_generation(),
            ..OwnerPresence::default()
        };
        state.publish_layout();
        next.layouts.insert(owner, state.layout_snapshot.clone());
        let state = Arc::new(RwLock::new(state));
        next.owners.insert(owner, state.clone());
        if matches!(owner, ResidencyOwner::CacheOwner(_)) {
            next.cache_owners.insert(owner.compact_key(), owner);
        }
        self.registry.store(Arc::new(next));
        self.active.store(true, Ordering::Relaxed);
        state
    }

    /// Install the complete recurrent layout before accepting routing credit.
    pub fn declare_groups(&self, owner: ResidencyOwner, groups: &[u32]) {
        self.install_layout(owner, groups, None);
    }

    pub fn declare_layout(&self, owner: ResidencyOwner, groups: &[u32], alignment_blocks: u32) {
        self.install_layout(owner, groups, Some(alignment_blocks.max(1) as usize));
    }

    fn install_layout(&self, owner: ResidencyOwner, groups: &[u32], alignment: Option<usize>) {
        if groups.is_empty() {
            if let ResidencyOwner::Worker(worker) = owner {
                self.remove_worker_rank(worker);
            }
            return;
        }
        let state = self.get_or_create(owner);
        let mut state = state.write();
        if state.retired {
            return;
        }
        let previous_group_count = state.groups.len();
        let previous_expected_count = state.expected_group_count;
        let removed_groups = state.groups.keys().any(|group| !groups.contains(group));
        let layout_changed = previous_group_count != groups.len()
            || !groups.iter().all(|group| state.groups.contains_key(group))
            || previous_expected_count != Some(groups.len())
            || alignment.is_some_and(|alignment| state.alignment_blocks.max(1) != alignment);
        if layout_changed {
            self.close_layout_certificates(&state);
        }
        let previous = layout_changed.then(|| std::mem::take(&mut state.complete_checkpoints));
        let previous_generation = state.generation;
        let changed = state.declared
            && (state.groups.len() != groups.len()
                || !groups.iter().all(|group| state.groups.contains_key(group))
                || alignment.is_some_and(|alignment| state.alignment_blocks.max(1) != alignment));
        if changed {
            state.groups.clear();
        }
        state.groups.retain(|group, _| groups.contains(group));
        for &group in groups {
            state.groups.entry(group).or_default();
        }
        state.declared = true;
        state.expected_group_count = Some(groups.len());
        if let Some(alignment) = alignment {
            state.alignment_blocks = alignment;
        }
        if changed || removed_groups {
            state.rebuild_group_counts();
        } else if layout_changed {
            state.rebuild_complete_checkpoints();
        }
        if let Some(previous) = previous {
            state.generation = self.fresh_generation();
            state.publish_layout();
            self.publish_reset(owner, previous_generation, previous, &state);
        }
    }

    /// Tagged events are consumed even if unsupported; never send them to attention.
    pub fn apply(&self, event: &RouterEvent) -> bool {
        let Some(tag) = event.kv_cache_group else {
            return false;
        };
        if tag.kind != KvCacheGroupKind::Mamba {
            return true;
        }
        let Ok(owner) = event.residency_owner() else {
            return true;
        };
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                let state = self.get_or_create(owner);
                let mut state = state.write();
                if state.retired {
                    return true;
                }
                if state.declared && !state.groups.contains_key(&tag.group_idx) {
                    return true;
                }
                if let Some(count) = tag.expected_group_count
                    && (count == 0
                        || state
                            .expected_group_count
                            .is_some_and(|expected| expected != count as usize))
                {
                    return true;
                }
                if let Some(alignment) = tag.alignment_blocks {
                    let alignment = alignment.max(1) as usize;
                    if state.alignment_blocks != 0 && state.alignment_blocks != alignment {
                        return true;
                    }
                }
                let layout_changed = !state.groups.contains_key(&tag.group_idx)
                    || tag
                        .expected_group_count
                        .is_some_and(|count| state.expected_group_count != Some(count as usize))
                    || tag.alignment_blocks.is_some_and(|alignment| {
                        state.alignment_blocks.max(1) != alignment.max(1) as usize
                    });
                if layout_changed {
                    self.close_layout_certificates(&state);
                }
                let previous =
                    layout_changed.then(|| std::mem::take(&mut state.complete_checkpoints));
                let previous_generation = state.generation;
                if let Some(count) = tag.expected_group_count {
                    state.expected_group_count = Some(count as usize);
                }
                if let Some(alignment) = tag.alignment_blocks {
                    state.alignment_blocks = alignment.max(1) as usize;
                }
                state.groups.entry(tag.group_idx).or_default();
                if layout_changed {
                    state.rebuild_complete_checkpoints();
                }
                if layout_changed {
                    state.generation = self.fresh_generation();
                }
                let mut updates = Vec::new();
                for block in &data.blocks {
                    let hash = block.block_hash.0;
                    let old = state.complete_checkpoints.get(&hash).copied().unwrap_or(0);
                    state.store(tag.group_idx, block.block_hash.0, event.storage_tier);
                    let new = state.complete_checkpoints.get(&hash).copied().unwrap_or(0);
                    if !layout_changed && old != new {
                        updates.push((hash, state.generation, new));
                    }
                }
                if let Some(previous) = previous {
                    state.publish_layout();
                    self.publish_reset(owner, previous_generation, previous, &state);
                } else {
                    self.publish_memberships(owner, updates);
                }
            }
            KvCacheEventData::Removed(data) => {
                if let Some(state) = self.owner(owner) {
                    let mut state = state.write();
                    if state.retired {
                        return true;
                    }
                    let mut updates = Vec::new();
                    for hash in &data.block_hashes {
                        let old = state
                            .complete_checkpoints
                            .get(&hash.0)
                            .copied()
                            .unwrap_or(0);
                        state.remove(tag.group_idx, hash.0, event.storage_tier);
                        let new = state
                            .complete_checkpoints
                            .get(&hash.0)
                            .copied()
                            .unwrap_or(0);
                        if old != new {
                            updates.push((hash.0, state.generation, new));
                        }
                    }
                    self.publish_memberships(owner, updates);
                }
            }
            KvCacheEventData::Cleared => self.clear_event(event),
        }
        true
    }

    pub fn clear_event(&self, event: &RouterEvent) {
        let Ok(Some(scope)) = event.reset_scope() else {
            return;
        };
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let registry = self.registry.load();
        for (owner, state) in &registry.owners {
            let matches = match (scope, *owner) {
                (
                    ResetScope::All | ResetScope::Domain(ResidencyDomain::Worker),
                    ResidencyOwner::Worker(w),
                ) => w == worker,
                (
                    ResetScope::All | ResetScope::Domain(ResidencyDomain::CacheOwner),
                    ResidencyOwner::CacheOwner(source),
                ) => event.state_source == Some(source),
                _ => false,
            };
            if matches {
                // A cache clear preserves the declared layout; it only invalidates checkpoints.
                self.clear_owner(*owner, &mut state.write());
            }
        }
    }

    pub fn clear_worker(&self, worker: WorkerWithDpRank) {
        if let Some(state) = self.owner(ResidencyOwner::worker(worker)) {
            self.clear_owner(ResidencyOwner::worker(worker), &mut state.write());
        }
    }

    pub fn remove_worker_rank(&self, worker: WorkerWithDpRank) {
        self.retire_workers(|known| known == worker);
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.retire_workers(|worker| worker.worker_id == worker_id);
    }

    fn retire_workers(&self, matches: impl Fn(WorkerWithDpRank) -> bool) {
        let _writer = self.registry_writer.lock();
        let current = self.registry.load_full();
        let mut next = (*current).clone();
        next.owners.retain(|owner, state| {
            if matches!(owner, ResidencyOwner::Worker(worker) if matches(*worker)) {
                let mut state = state.write();
                state.retired = true;
                self.clear_owner(*owner, &mut state);
                false
            } else {
                true
            }
        });
        next.layouts.retain(
            |owner, _| !matches!(owner, ResidencyOwner::Worker(worker) if matches(*worker)),
        );
        self.active
            .store(!next.owners.is_empty(), Ordering::Relaxed);
        self.registry.store(Arc::new(next));
    }

    /// Dump empty groups too: absence of a checkpoint must survive recovery.
    pub fn dump_events(&self) -> Vec<RouterEvent> {
        let registry = self.registry.load();
        let mut out = Vec::new();
        for (&owner, state) in &registry.owners {
            let state = state.read();
            for (&group_idx, group) in &state.groups {
                for tier in [
                    StorageTier::Device,
                    StorageTier::HostPinned,
                    StorageTier::Disk,
                    StorageTier::External,
                ] {
                    if matches!(owner, ResidencyOwner::CacheOwner(_)) && tier == StorageTier::Device
                    {
                        continue;
                    }
                    let blocks: Vec<_> = group
                        .iter()
                        .filter(|(_, mask)| **mask & tier_bit(tier) != 0)
                        .map(|(&hash, _)| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(0),
                            mm_extra_info: None,
                        })
                        .collect();
                    if blocks.is_empty()
                        && tier
                            != match owner {
                                ResidencyOwner::Worker(_) => StorageTier::Device,
                                _ => StorageTier::HostPinned,
                            }
                    {
                        continue;
                    }
                    let event = KvCacheEvent {
                        event_id: 0,
                        dp_rank: match owner {
                            ResidencyOwner::Worker(w) => w.dp_rank,
                            _ => 0,
                        },
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: None,
                            start_position: None,
                            blocks,
                        }),
                    };
                    let event = match owner {
                        ResidencyOwner::Worker(w) => {
                            RouterEvent::with_storage_tier(w.worker_id, event, tier)
                        }
                        ResidencyOwner::CacheOwner(source) => {
                            RouterEvent::with_cache_owner(0, event, tier, source)
                        }
                    };
                    out.push(event.with_kv_cache_group(KvCacheGroupTag {
                        group_idx,
                        kind: KvCacheGroupKind::Mamba,
                        expected_group_count: state.expected_group_count.map(|count| count as u32),
                        alignment_blocks: Some(state.alignment_blocks.max(1) as u32),
                    }));
                }
            }
        }
        out
    }

    pub fn is_hybrid(&self, worker: WorkerWithDpRank) -> bool {
        self.owner(ResidencyOwner::worker(worker)).is_some()
    }

    #[cfg(test)]
    pub(crate) fn groups_for(&self, worker: WorkerWithDpRank) -> rustc_hash::FxHashSet<u32> {
        self.owner(ResidencyOwner::worker(worker))
            .map(|state| state.read().groups.keys().copied().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::KvCacheRemoveData;

    fn checkpoint(
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group: u32,
        hash: u64,
        count: Option<u32>,
    ) -> RouterEvent {
        RouterEvent::with_storage_tier(
            worker.worker_id,
            KvCacheEvent {
                event_id: 1,
                dp_rank: worker.dp_rank,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
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
            expected_group_count: count,
            alignment_blocks: Some(1),
        })
    }

    fn owners_at_revision(
        view: &CheckpointReadView,
        hash: ExternalSequenceBlockHash,
        tier: StorageTier,
        revision: u64,
    ) -> Vec<ResidencyOwner> {
        let mut owners = Vec::new();
        view.for_each_owner_at_revision(hash, tier, revision, |owner| owners.push(owner));
        owners
    }

    fn endpoint_proof(
        table: &KvCacheGroupResidency,
        hash: ExternalSequenceBlockHash,
    ) -> Arc<EndpointCertificate> {
        table
            .endpoint_certificate(hash, table.capture_cutoff())
            .expect("an eligible endpoint has a published proof")
    }

    #[test]
    fn cohort_interning_requires_exact_workers_even_with_identical_fingerprints() {
        let index = CheckpointMembershipIndex::default();
        let first = WorkerWithDpRank::new(7, 0);
        let other_rank = WorkerWithDpRank::new(7, 1);
        // Zero is a legal selector, and deliberately collides here.
        let initial = index.intern_cohort(FxHashSet::from_iter([first]), 0);
        let identical = index.intern_cohort(FxHashSet::from_iter([first]), 0);
        assert!(Arc::ptr_eq(&initial, &identical));
        let different = index.intern_cohort(FxHashSet::from_iter([other_rank]), 0);
        assert!(!Arc::ptr_eq(&initial, &different));
        assert_eq!(different.workers, FxHashSet::from_iter([other_rank]));
        let original_again = index.intern_cohort(FxHashSet::from_iter([first]), 0);
        assert_eq!(original_again.workers, initial.workers);
        assert!(!Arc::ptr_eq(&original_again, &different));
    }

    #[test]
    fn cohort_transitions_validate_collisions_and_share_exact_add_remove_results() {
        let index = CheckpointMembershipIndex::default();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(7, 1);
        let wrong = WorkerWithDpRank::new(8, 0);
        let original = index.intern_cohort(FxHashSet::from_iter([first]), mix_worker(first));
        let fingerprint = original.fingerprint ^ mix_worker(second);
        let collision = index.intern_cohort(FxHashSet::from_iter([first, wrong]), fingerprint);
        let added = index.transition_cohort(&original, second, true).unwrap();
        assert!(!Arc::ptr_eq(&added, &collision));
        assert_eq!(added.workers, FxHashSet::from_iter([first, second]));
        let repeated = index.transition_cohort(&original, second, true).unwrap();
        assert!(Arc::ptr_eq(&added, &repeated));
        let unchanged = index.transition_cohort(&added, second, true).unwrap();
        assert!(Arc::ptr_eq(&added, &unchanged));

        // A different prior cohort reaches the same result through exact set
        // comparison, without the cached previous-allocation transition proof.
        let other_prior = Arc::new(CheckpointCohort {
            workers: FxHashSet::from_iter([second]),
            fingerprint: mix_worker(second),
        });
        let same_result = index.transition_cohort(&other_prior, first, true).unwrap();
        assert!(Arc::ptr_eq(&added, &same_result));

        let removed = index.transition_cohort(&added, first, false).unwrap();
        assert_eq!(removed.workers, FxHashSet::from_iter([second]));
        let repeated_remove = index.transition_cohort(&added, first, false).unwrap();
        assert!(Arc::ptr_eq(&removed, &repeated_remove));
        assert!(index.transition_cohort(&removed, second, false).is_none());
        assert_eq!(added.workers, FxHashSet::from_iter([first, second]));
    }

    #[test]
    fn cohort_cache_is_lazy_bounded_and_keeps_only_weak_references() {
        let index = CheckpointMembershipIndex::default();
        assert!(index.cohorts.get().is_none());
        let mut retired = Vec::new();
        for id in 0..(COHORT_CACHE_CELLS * 4) as u64 {
            let worker = WorkerWithDpRank::new(id, 0);
            let cohort = index.intern_cohort(FxHashSet::from_iter([worker]), mix_worker(worker));
            let next = index
                .transition_cohort(&cohort, WorkerWithDpRank::new(id, 1), true)
                .unwrap();
            retired.push(Arc::downgrade(&cohort));
            retired.push(Arc::downgrade(&next));
        }
        assert!(retired.iter().all(|weak| weak.upgrade().is_none()));
        let cells = index.cohorts.get().unwrap();
        assert_eq!(cells.len(), COHORT_CACHE_CELLS);
        for cell in cells.iter() {
            let cell = cell.lock();
            assert!(cell.cohort.upgrade().is_none());
            assert!(
                cell.transition
                    .as_ref()
                    .is_none_or(|transition| transition.previous.upgrade().is_none())
            );
        }
        let worker = WorkerWithDpRank::new(1, 0);
        let live = index.intern_cohort(FxHashSet::from_iter([worker]), mix_worker(worker));
        assert_eq!(live.workers, FxHashSet::from_iter([worker]));
    }

    #[test]
    fn endpoint_cohorts_share_workers_without_sharing_endpoint_validity() {
        let table = KvCacheGroupResidency::new();
        let hashes = [ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(12)];
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(7, 1);
        for worker in [first, second] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0, 1], 1);
            for hash in hashes {
                for group in [0, 1] {
                    table.apply(&checkpoint(
                        worker,
                        StorageTier::Device,
                        group,
                        hash.0,
                        Some(2),
                    ));
                }
            }
        }
        let previous = hashes.map(|hash| endpoint_proof(&table, hash));
        assert!(Arc::ptr_eq(&previous[0].cohort, &previous[1].cohort));
        assert!(!Arc::ptr_eq(&previous[0], &previous[1]));
        assert_ne!(previous[0].publication_id(), previous[1].publication_id());

        let mut remove = checkpoint(first, StorageTier::Device, 1, hashes[0].0, Some(2));
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hashes[0]],
        });
        table.apply(&remove);
        assert!(!previous[0].is_valid());
        assert!(previous[1].is_valid());
        assert_eq!(
            previous[0].workers(),
            &FxHashSet::from_iter([first, second])
        );
        assert_eq!(
            endpoint_proof(&table, hashes[0]).workers(),
            &FxHashSet::from_iter([second])
        );

        table.apply(&checkpoint(
            first,
            StorageTier::Device,
            1,
            hashes[0].0,
            Some(2),
        ));
        let restored = endpoint_proof(&table, hashes[0]);
        assert!(Arc::ptr_eq(&restored.cohort, &previous[1].cohort));
        assert!(previous[1].is_valid());

        // A new owner generation may reuse immutable workers, but every H's
        // old certificate must still close independently before publication.
        table.clear_worker(first);
        assert!(!restored.is_valid());
        assert!(!previous[1].is_valid());
        for hash in hashes {
            for group in [0, 1] {
                table.apply(&checkpoint(
                    first,
                    StorageTier::Device,
                    group,
                    hash.0,
                    Some(2),
                ));
            }
        }
        let current = hashes.map(|hash| endpoint_proof(&table, hash));
        assert!(Arc::ptr_eq(&current[0].cohort, &current[1].cohort));
        assert_eq!(current[0].workers(), &FxHashSet::from_iter([first, second]));
        assert!(current.iter().all(|proof| proof.is_valid()));
        assert!(previous.iter().all(|proof| !proof.is_valid()));
    }

    #[test]
    fn endpoint_read_is_send_and_protects_invalidation_without_cloning_certificate() {
        fn assert_send<T: Send>() {}
        assert_send::<EndpointCertificateRead>();
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let hash = ExternalSequenceBlockHash(11);
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.apply(&event);
        let certificate = endpoint_proof(&table, hash);
        let references = Arc::strong_count(&certificate);
        let cutoff = table.capture_cutoff();
        let read = table.endpoint_certificate_read(hash, cutoff).unwrap();
        assert_eq!(Arc::strong_count(&certificate), references);
        assert_eq!(read.publication_id(), certificate.publication_id());
        assert_eq!(read.workers(), &FxHashSet::from_iter([worker]));
        assert!(table.is_certificate_read_valid(&read));
        assert!(!KvCacheGroupResidency::new().is_certificate_read_valid(&read));
        assert!(read.is_valid());
        let (ready_send, ready_receive) = std::sync::mpsc::channel();
        let (resume_send, resume_receive) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            ready_send.send(()).unwrap();
            resume_receive.recv().unwrap();
            assert!(!read.is_valid());
            assert_eq!(read.workers(), &FxHashSet::from_iter([worker]));
        });
        ready_receive.recv().unwrap();
        let mut remove = event.clone();
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        assert!(!certificate.is_valid());
        table.apply(&event);
        assert!(table.endpoint_certificate_read(hash, cutoff).is_none());
        assert!(
            table
                .endpoint_certificate_read(hash, table.capture_cutoff())
                .is_some()
        );
        resume_send.send(()).unwrap();
        reader.join().unwrap();
    }

    #[test]
    fn endpoint_certificate_contains_only_exact_current_complete_device_workers() {
        let table = KvCacheGroupResidency::new();
        let full = WorkerWithDpRank::new(7, 2);
        let incomplete = WorkerWithDpRank::new(8, 0);
        let aligned = WorkerWithDpRank::new(9, 0);
        let host = WorkerWithDpRank::new(10, 0);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(ResidencyOwner::worker(full), &[0], 1);
        table.declare_layout(ResidencyOwner::worker(incomplete), &[0, 1], 1);
        table.declare_layout(ResidencyOwner::worker(aligned), &[0], 2);
        table.declare_layout(ResidencyOwner::worker(host), &[0], 1);
        table.apply(&checkpoint(
            incomplete,
            StorageTier::Device,
            0,
            hash.0,
            Some(2),
        ));
        let mut event = checkpoint(aligned, StorageTier::Device, 0, hash.0, Some(1));
        event.kv_cache_group.as_mut().unwrap().alignment_blocks = Some(2);
        table.apply(&event);
        table.apply(&checkpoint(
            host,
            StorageTier::HostPinned,
            0,
            hash.0,
            Some(1),
        ));
        table.apply(&checkpoint(full, StorageTier::Device, 0, hash.0, Some(1)));
        let proof = endpoint_proof(&table, hash);
        assert_eq!(proof.workers(), &FxHashSet::from_iter([full]));
        assert!(!proof.workers().contains(&WorkerWithDpRank::new(7, 0)));
        assert!(proof.is_valid());
        assert!(table.is_certificate_valid(&proof));
        assert!(!KvCacheGroupResidency::new().is_certificate_valid(&proof));
        assert_ne!(proof.publication_id(), 0);
        assert!(
            table
                .endpoint_certificate(hash, proof.publication_id() - 1)
                .is_none()
        );
    }

    #[test]
    fn endpoint_certificate_cutoff_and_same_hash_mutations_fail_closed() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let hash = ExternalSequenceBlockHash(11);
        let before_registration = table.capture_cutoff();
        let mut old_view = table.read_view();
        assert!(
            old_view
                .track_owner(ResidencyOwner::worker(first))
                .is_none()
        );
        for worker in [first, second] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        }
        table.apply(&checkpoint(first, StorageTier::Device, 0, hash.0, Some(1)));
        let old = endpoint_proof(&table, hash);
        let cutoff = table.capture_cutoff();
        assert!(
            table
                .endpoint_certificate(hash, before_registration)
                .is_none()
        );
        assert!(
            old_view
                .track_owner(ResidencyOwner::worker(first))
                .is_none()
        );
        table.apply(&checkpoint(second, StorageTier::Device, 0, hash.0, Some(1)));
        assert!(!old.is_valid());
        assert!(table.endpoint_certificate(hash, cutoff).is_none());
        let current = endpoint_proof(&table, hash);
        let current_publication = current.publication_id();
        assert_eq!(current.workers(), &FxHashSet::from_iter([first, second]));
        let mut remove = checkpoint(second, StorageTier::Device, 0, hash.0, Some(1));
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        assert!(!current.is_valid());
        assert_eq!(
            endpoint_proof(&table, hash).workers(),
            &FxHashSet::from_iter([first])
        );
        table.apply(&checkpoint(second, StorageTier::Device, 0, hash.0, Some(1)));
        assert!(!current.is_valid());
        assert!(endpoint_proof(&table, hash).publication_id() > current_publication);
    }

    #[test]
    fn endpoint_certificate_invalidates_only_affected_layout_endpoints() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let hash = ExternalSequenceBlockHash(11);
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.apply(&event);
        let original = endpoint_proof(&table, hash);
        table.declare_layout(ResidencyOwner::worker(WorkerWithDpRank::new(8, 0)), &[0], 1);
        assert!(original.is_valid());
        assert!(Arc::ptr_eq(&original, &endpoint_proof(&table, hash)));
        // Neither unrelated registration nor duplicate storage changes this H.
        table.apply(&event);
        assert!(Arc::ptr_eq(&original, &endpoint_proof(&table, hash)));
        table.clear_worker(worker);
        assert!(!original.is_valid());
        table.apply(&event);
        let cleared = endpoint_proof(&table, hash);
        table.declare_layout(ResidencyOwner::worker(worker), &[3], 1);
        assert!(!cleared.is_valid());
        table.apply(&checkpoint(worker, StorageTier::Device, 3, hash.0, Some(1)));
        let remapped = endpoint_proof(&table, hash);
        table.remove_worker_rank(worker);
        assert!(!remapped.is_valid());
        table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        table.apply(&event);
        let readded = endpoint_proof(&table, hash);
        assert!(readded.is_valid());
        assert!(!original.is_valid());
        assert!(!cleared.is_valid());
        assert!(!remapped.is_valid());
    }

    #[test]
    fn endpoint_certificate_counters_exhaust_without_aba_or_legacy_loss() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let hash = ExternalSequenceBlockHash(11);
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        table
            .memberships
            .publication
            .store(u64::MAX, Ordering::Release);
        table.apply(&event);
        assert!(table.endpoint_certificate(hash, u64::MAX).is_none());
        assert!(has_checkpoint(
            &table,
            ResidencyOwner::worker(worker),
            hash.0,
            StorageTier::Device
        ));
        table.clear_worker(worker);
        table.apply(&event);
        assert_eq!(table.capture_cutoff(), u64::MAX);
        assert!(table.endpoint_certificate(hash, u64::MAX).is_none());
        assert!(has_checkpoint(
            &table,
            ResidencyOwner::worker(worker),
            hash.0,
            StorageTier::Device
        ));
    }

    #[test]
    fn concurrent_checkpoint_writer_cannot_recertify_an_owner_being_cleared() {
        use std::time::{Duration, Instant};
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        let hash = ExternalSequenceBlockHash(11);
        for worker in [first, second] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        }
        for worker in [first, second] {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        }
        let old = endpoint_proof(&table, hash);
        let handle = table.registry.load().layouts[&ResidencyOwner::worker(first)].clone();
        let generation = handle.snapshot.load().generation;
        let done = AtomicU64::new(0);
        std::thread::scope(|scope| {
            let shard = &table.memberships.shards[CheckpointMembershipIndex::shard(hash.0)];
            let hold_publication = shard.writer.lock();
            scope.spawn(|| {
                table.clear_worker(first);
                done.fetch_add(1, Ordering::Release);
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            let transitioned = loop {
                if handle.eligible_generation.load(Ordering::Acquire) == 0 {
                    break true;
                }
                if Instant::now() >= deadline {
                    break false;
                }
                std::thread::yield_now();
            };
            assert_eq!(handle.snapshot.load().generation, generation);
            scope.spawn(|| {
                let mut removed = checkpoint(second, StorageTier::Device, 0, hash.0, Some(1));
                removed.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![hash],
                });
                table.apply(&removed);
                table.apply(&checkpoint(second, StorageTier::Device, 0, hash.0, Some(1)));
                done.fetch_add(1, Ordering::Release);
            });
            drop(hold_publication);
            assert!(
                transitioned,
                "clear must close eligibility before waiting for the shard"
            );
            while done.load(Ordering::Acquire) != 2 {
                if let Some(proof) = table.endpoint_certificate(hash, table.capture_cutoff()) {
                    // Before all affected Hs are invalidated the old layout is
                    // still published. After publication no old proof may win.
                    if handle.snapshot.load().generation != generation && proof.is_valid() {
                        assert!(!proof.workers().contains(&first));
                    }
                }
                std::thread::yield_now();
            }
        });
        assert!(!old.is_valid());
        assert_eq!(
            endpoint_proof(&table, hash).workers(),
            &FxHashSet::from_iter([second])
        );
    }

    #[test]
    fn concurrent_layout_publishers_leave_only_current_generation_endpoint_facts() {
        let table = KvCacheGroupResidency::new();
        let hash = ExternalSequenceBlockHash(11);
        let workers: Vec<_> = (1..=8).map(|id| WorkerWithDpRank::new(id, 0)).collect();
        for &worker in &workers {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        }
        std::thread::scope(|scope| {
            for &worker in &workers {
                let table = &table;
                scope.spawn(move || {
                    for _ in 0..12 {
                        table.clear_worker(worker);
                        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
                    }
                });
            }
        });
        assert_eq!(
            endpoint_proof(&table, hash).workers(),
            &workers.into_iter().collect()
        );
    }

    #[test]
    fn incremental_endpoint_update_uses_post_mutation_generation_for_stale_removal() {
        let table = KvCacheGroupResidency::new();
        let hash = ExternalSequenceBlockHash(11);
        let workers = [WorkerWithDpRank::new(7, 0), WorkerWithDpRank::new(8, 0)];
        for worker in workers {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        }
        for worker in workers {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        }
        let previous = endpoint_proof(&table, hash);
        let owner = ResidencyOwner::worker(workers[0]);
        let state = table.owner(owner).unwrap();
        let state = state.write();
        assert!(state.generation > 0);
        table.publish_memberships(owner, vec![(hash.0, state.generation - 1, 0)]);
        assert!(!previous.is_valid());
        assert_eq!(
            endpoint_proof(&table, hash).workers(),
            &FxHashSet::from_iter(workers)
        );
    }

    #[test]
    fn closed_owner_cannot_reenter_a_rebuilt_proof_before_new_layout_publication() {
        let table = KvCacheGroupResidency::new();
        let hash = ExternalSequenceBlockHash(11);
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        for worker in [first, second] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        }
        for worker in [first, second] {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        }
        let old = endpoint_proof(&table, hash);
        let owner = ResidencyOwner::worker(first);
        let state = table.owner(owner).unwrap();
        let state = state.write();
        table.close_layout_certificates(&state);
        assert!(!old.is_valid());
        assert!(
            state
                .layout_snapshot
                .snapshot
                .load()
                .layout()
                .complete_layout
        );
        assert_eq!(
            state
                .layout_snapshot
                .eligible_generation
                .load(Ordering::Acquire),
            0
        );
        // A different owner rebuilds while the legacy payload still looks old
        // and complete. The closed atomic epoch, not that payload, rejects first.
        let mut remove = checkpoint(second, StorageTier::Device, 0, hash.0, Some(1));
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        table.apply(&checkpoint(second, StorageTier::Device, 0, hash.0, Some(1)));
        assert_eq!(
            endpoint_proof(&table, hash).workers(),
            &FxHashSet::from_iter([second])
        );
    }

    #[test]
    fn closing_owner_invalidates_a_replacement_published_by_the_old_shard_writer() {
        use std::time::{Duration, Instant};
        let table = KvCacheGroupResidency::new();
        let hash = ExternalSequenceBlockHash(11);
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(8, 0);
        for worker in [first, second] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
        }
        for worker in [first, second] {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        }
        let previous = endpoint_proof(&table, hash);
        let first_handle = table.registry.load().layouts[&ResidencyOwner::worker(first)].clone();
        let old_generation = first_handle.snapshot.load().generation;
        let second_state = table.owner(ResidencyOwner::worker(second)).unwrap();
        let mut second_writer = second_state.write();
        std::thread::scope(|scope| {
            let shard = &table.memberships.shards[CheckpointMembershipIndex::shard(hash.0)];
            let mut shard_writer = shard.writer.lock();
            let clearing = scope.spawn(|| table.clear_worker(first));
            let deadline = Instant::now() + Duration::from_secs(5);
            let closed = loop {
                if first_handle.eligible_generation.load(Ordering::Acquire) == 0 {
                    break true;
                }
                if Instant::now() >= deadline {
                    break false;
                }
                std::thread::yield_now();
            };
            // Force the already-running other-owner publisher to finish after
            // first closed, but before its invalidation can obtain this shard.
            let replacement = if closed {
                assert!(previous.is_valid());
                assert_eq!(first_handle.snapshot.load().generation, old_generation);
                let registry = table.registry.load_full();
                let mut next = (**shard.entries.load()).clone();
                let mut membership = (**next.get(&hash.0).unwrap()).clone();
                second_writer.store(0, hash.0, StorageTier::HostPinned);
                membership.update(
                    ResidencyOwner::worker(second),
                    second_writer.generation,
                    second_writer.complete_checkpoints[&hash.0],
                );
                let replacement = table
                    .memberships
                    .build_certificate(
                        &membership,
                        &registry,
                        ResidencyOwner::worker(second),
                        Some(&previous),
                    )
                    .unwrap();
                assert!(replacement.workers().contains(&first));
                previous.invalidate();
                *shard_writer += 1;
                membership.revision = *shard_writer;
                membership.endpoint_certificate = Some(replacement.clone());
                next.insert(hash.0, Arc::new(membership));
                shard.entries.store(Arc::new(next));
                let born = table.memberships.publication.fetch_add(1, Ordering::AcqRel) + 1;
                replacement.publication.store(born, Ordering::Release);
                assert!(replacement.is_valid());
                Some(replacement)
            } else {
                None
            };
            drop(shard_writer);
            clearing.join().unwrap();
            assert!(
                closed,
                "clear must close before waiting for shard invalidation"
            );
            assert!(!replacement.unwrap().is_valid());
            assert_ne!(first_handle.snapshot.load().generation, old_generation);
        });
        assert_eq!(
            endpoint_proof(&table, hash).workers(),
            &FxHashSet::from_iter([second])
        );
    }

    #[test]
    fn indexed_membership_sorts_exact_owner_identity_and_dp_rank() {
        use crate::identity::{
            CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };
        use IdentitySource::{DefaultDerived, Explicit};

        let cache_owner = |semantics_source, routing_source, cluster, slot, slot_source| {
            ResidencyOwner::cache_owner(CacheOwnerId::new(
                PoolId::new(
                    IndexerDomainId::new(
                        CacheSemanticsId::new([1; 16], semantics_source),
                        RoutingScopeId::new([2; 16], routing_source),
                    ),
                    ClusterId::new(cluster),
                ),
                StableDpSlotId::new([slot; 16], slot_source),
            ))
        };
        let rank0 = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let rank2 = ResidencyOwner::worker(WorkerWithDpRank::new(7, 2));
        let later_worker = ResidencyOwner::worker(WorkerWithDpRank::new(u64::MAX, 0));
        let semantics_derived = cache_owner(DefaultDerived, Explicit, 3, 4, Explicit);
        let routing_derived = cache_owner(Explicit, DefaultDerived, 3, 4, Explicit);
        let slot_derived = cache_owner(Explicit, Explicit, 3, 4, DefaultDerived);
        let explicit = cache_owner(Explicit, Explicit, 3, 4, Explicit);
        let later_slot = cache_owner(Explicit, Explicit, 3, 5, Explicit);
        let later_cluster = cache_owner(Explicit, Explicit, 4, 0, Explicit);
        let expected = [
            rank0,
            rank2,
            later_worker,
            semantics_derived,
            routing_derived,
            slot_derived,
            explicit,
            later_slot,
            later_cluster,
        ];
        let insertion_order = [
            later_cluster,
            rank2,
            explicit,
            semantics_derived,
            later_worker,
            later_slot,
            rank0,
            routing_derived,
            slot_derived,
        ];
        let mut membership = CompleteMembership::default();
        for (index, owner) in insertion_order.into_iter().enumerate() {
            membership.update(owner, index as u64 + 10, 0b1111);
        }
        for tier in 0..4 {
            let indexed = CheckpointMembershipView::new(&membership, tier);
            assert_eq!(indexed.len(), expected.len());
            assert!(!indexed.is_empty());
            assert_eq!(
                indexed.iter().map(|(owner, _)| owner).collect::<Vec<_>>(),
                expected,
            );
            for (index, owner) in insertion_order.into_iter().enumerate() {
                assert_eq!(indexed.generation_for(owner), Some(index as u64 + 10));
            }
            assert_eq!(
                indexed.generation_for(ResidencyOwner::worker(WorkerWithDpRank::new(7, 1))),
                None,
            );
            assert_eq!(
                indexed.generation_for(cache_owner(Explicit, Explicit, 3, 3, Explicit)),
                None,
            );
        }
    }

    #[test]
    fn indexed_membership_removal_requires_the_matching_generation() {
        let first = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let middle = ResidencyOwner::worker(WorkerWithDpRank::new(8, 0));
        let last = ResidencyOwner::worker(WorkerWithDpRank::new(9, 0));
        let mut membership = CompleteMembership::default();
        for owner in [last, first, middle] {
            membership.update(owner, 11, 0b0011);
        }
        membership.update(middle, 12, 0b0011);
        membership.update(middle, 11, 0);
        for tier in 0..2 {
            let indexed = CheckpointMembershipView::new(&membership, tier);
            assert_eq!(indexed.len(), 3);
            assert_eq!(indexed.generation_for(middle), Some(12));
        }
        membership.update(middle, 12, 0);
        for tier in 0..2 {
            let indexed = CheckpointMembershipView::new(&membership, tier);
            assert_eq!(indexed.len(), 2);
            assert_eq!(indexed.generation_for(middle), None);
            assert_eq!(indexed.generation_for(first), Some(11));
            assert_eq!(indexed.generation_for(last), Some(11));
        }
        membership.update(middle, 13, 0b0001);
        assert_eq!(
            membership.owners[0]
                .iter()
                .map(|&(owner, _)| owner)
                .collect::<Vec<_>>(),
            [first, middle, last],
        );
        let device = CheckpointMembershipView::new(&membership, 0);
        assert_eq!(device.generation_for(middle), Some(13));
        assert!(
            CheckpointMembershipView::new(&membership, 1)
                .generation_for(middle)
                .is_none()
        );
    }

    fn assert_generation_words_match_facts(membership: &CompleteMembership) {
        let mut expected_rows = 0;
        for (tier, owners) in membership.owners.iter().enumerate() {
            let mut expected = std::collections::BTreeMap::<u64, u64>::new();
            for &(_, generation) in owners {
                *expected.entry(generation >> 6).or_default() |= 1_u64 << (generation & 63);
            }
            let indexed = CheckpointMembershipView::new(membership, tier);
            let expected = expected.into_iter().collect::<Vec<_>>();
            let mut groups = std::collections::BTreeMap::<u64, u64>::new();
            for &(word, _) in &expected {
                *groups.entry(word >> 6).or_default() |= 1_u64 << (word & 63);
            }
            let groups = groups.into_iter().collect::<Vec<_>>();
            expected_rows += expected.len() + groups.len();
            assert_eq!(indexed.generation_words().collect::<Vec<_>>(), expected);
            assert_eq!(indexed.generation_words().len(), expected.len());
            assert_eq!(indexed.generation_word_groups().collect::<Vec<_>>(), groups);
            for &(group, bits) in &groups {
                assert_eq!(indexed.generation_word_group_mask(group), bits);
            }
            assert_eq!(indexed.generation_word_group_mask(u64::MAX), 0);
            for &(word, mask) in &expected {
                assert_ne!(mask, 0);
                assert_eq!(indexed.generation_mask(word), mask);
            }
            for word in [0, 1, 2, GENERATION_WORD_MASK, GENERATION_WORD_MASK + 1] {
                let mask = expected
                    .iter()
                    .find_map(|&(key, mask)| (key == word).then_some(mask))
                    .unwrap_or(0);
                assert_eq!(indexed.generation_mask(word), mask);
            }
        }
        assert_eq!(membership.generation_words.len(), expected_rows);
        assert!(
            membership
                .generation_words
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0)
        );
    }

    #[test]
    fn generation_words_preserve_tiers_duplicate_tokens_and_full_u64_range() {
        let first = ResidencyOwner::worker(WorkerWithDpRank::new(7, 0));
        let same_worker_other_rank = ResidencyOwner::worker(WorkerWithDpRank::new(7, 1));
        let third = ResidencyOwner::worker(WorkerWithDpRank::new(9, 0));
        let mut membership = CompleteMembership::default();
        membership.update(first, 0, 0b1111);
        membership.update(same_worker_other_rank, 0, 0b1111);
        membership.update(third, u64::MAX, 0b1111);
        assert_generation_words_match_facts(&membership);
        for tier in 0..4 {
            assert_eq!(
                CheckpointMembershipView::new(&membership, tier)
                    .generation_words()
                    .collect::<Vec<_>>(),
                [(0, 1), (GENERATION_WORD_MASK, 1 << 63)],
            );
        }
        // Moving one duplicate to a new generation must retain the old bit.
        membership.update(first, 63, 0b0101);
        assert_generation_words_match_facts(&membership);
        assert_eq!(
            CheckpointMembershipView::new(&membership, 0).generation_mask(0),
            1 | (1 << 63),
        );
        // A stale removal cannot remove the remapped fact or its sidecar bit.
        membership.update(first, 0, 0);
        assert_generation_words_match_facts(&membership);
        assert_eq!(membership.owners[0][0], (first, 63));
        membership.update(same_worker_other_rank, 0, 0);
        assert_generation_words_match_facts(&membership);
        assert_eq!(
            CheckpointMembershipView::new(&membership, 0).generation_mask(0),
            1 << 63,
        );
        membership.update(first, 64, 0b1010);
        assert_generation_words_match_facts(&membership);
        assert_eq!(
            CheckpointMembershipView::new(&membership, 1).generation_mask(1),
            1,
        );
        // Remove exact old-tier generations as lifecycle publication does.
        membership.update(first, 63, 0b1010);
        membership.update(first, 63, 0);
        membership.update(third, u64::MAX, 0);
        assert_generation_words_match_facts(&membership);
        assert!(membership.is_empty());
        assert!(membership.generation_words.is_empty());
    }

    #[test]
    fn scoped_generation_words_support_indexed_and_reverse_sparse_range_access() {
        let table = KvCacheGroupResidency::new();
        let hash = ExternalSequenceBlockHash(11);
        let words = [0, 15, 16, 47, 48, GENERATION_WORD_MASK];
        // Use distinct owners for each sparse word, with identical membership
        // on two tiers and empty membership between them.
        for (index, &word) in words.iter().enumerate() {
            table.memberships.publish(
                ResidencyOwner::worker(WorkerWithDpRank::new(index as u64 + 8, 0)),
                vec![(hash.0, word << 6, 0b0101)],
                None,
            );
        }
        let view = table.read_view();
        let revision = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        for tier in [StorageTier::Device, StorageTier::Disk] {
            view.with_membership_at_revision(hash, tier, revision, |membership| {
                let indexed = membership.generation_words();
                assert_eq!(indexed.len(), words.len());
                assert_eq!(indexed.clone().next(), Some((0, 1)));
                assert_eq!(indexed.clone().next_back(), Some((GENERATION_WORD_MASK, 1)));
                for (index, &word) in words.iter().enumerate() {
                    assert_eq!(indexed.clone().nth(index), Some((word, 1)));
                    assert_eq!(
                        indexed.clone().nth_back(words.len() - index - 1),
                        Some((word, 1))
                    );
                }
                assert_eq!(indexed.clone().nth(words.len()), None);
                assert_eq!(indexed.clone().nth(usize::MAX), None);
                assert_eq!(indexed.len(), words.len());
            });
        }
        view.with_membership_at_revision(hash, StorageTier::HostPinned, revision, |membership| {
            let indexed = membership.generation_words();
            assert_eq!(indexed.len(), 0);
            assert_eq!(indexed.clone().next_back(), None);
            assert_eq!(indexed.clone().nth(indexed.len()), None);
        });
    }

    #[test]
    fn incremental_generation_words_match_exact_facts_through_mixed_updates() {
        let generations = [
            0,
            1,
            2,
            63,
            64,
            65,
            127,
            128,
            4095,
            4096,
            4097,
            u64::MAX - 63,
            u64::MAX,
        ];
        let mut random = 0xa175_c9e3_f62b_840d_u64;
        let mut membership = CompleteMembership::default();
        for step in 0..1024 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let owner = ResidencyOwner::worker(WorkerWithDpRank::new(
                random % 8,
                ((random >> 8) % 3) as u32,
            ));
            let generation = generations[(random >> 16) as usize % generations.len()];
            let mask = ((random >> 24) as u8) & 0b1111;
            membership.update(owner, generation, mask);
            assert_generation_words_match_facts(&membership);
            if step % 17 == 0 {
                let before = membership.generation_words.clone();
                membership.update(owner, generation, mask);
                assert_eq!(membership.generation_words, before);
                assert_generation_words_match_facts(&membership);
            }
        }
    }

    #[test]
    fn published_generation_words_follow_tier_expiry_and_layout_remapping() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        let tiers = [
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::Disk,
            StorageTier::External,
        ];
        for tier in tiers {
            table.apply(&checkpoint(worker, tier, 0, hash.0, Some(1)));
        }
        let view = table.read_view();
        let generation = view.capture_layout(owner).unwrap().generation();
        let revision = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        for tier in tiers {
            view.with_membership_at_revision(hash, tier, revision, |membership| {
                assert_eq!(
                    membership.generation_words().collect::<Vec<_>>(),
                    [(generation >> 6, 1 << (generation & 63))],
                );
            });
        }
        let mut removed = checkpoint(worker, StorageTier::HostPinned, 0, hash.0, Some(1));
        removed.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&removed);
        view.with_membership_at_revision(hash, StorageTier::Device, revision, |_| {
            panic!("tier mutation invalidates the old hash revision")
        });
        let current = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        view.with_membership_at_revision(hash, StorageTier::HostPinned, current, |membership| {
            assert_eq!(membership.generation_words().len(), 0);
        });
        table.declare_layout(owner, &[0, 1], 1);
        assert!(
            view.checkpoint_hint_revision(hash, StorageTier::Device)
                .is_none()
        );
        for group in [0, 1] {
            table.apply(&checkpoint(
                worker,
                StorageTier::Device,
                group,
                hash.0,
                Some(2),
            ));
        }
        let remapped = view.capture_layout(owner).unwrap().generation();
        assert_ne!(remapped, generation);
        let current = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        view.with_membership_at_revision(hash, StorageTier::Device, current, |membership| {
            assert_eq!(
                membership.generation_mask(generation >> 6) & (1 << (generation & 63)),
                0
            );
            assert_eq!(membership.generation_for(owner), Some(remapped));
            assert_eq!(
                membership.generation_words().collect::<Vec<_>>(),
                [(remapped >> 6, 1 << (remapped & 63))],
            );
        });
        table.clear_worker(worker);
        assert!(
            view.checkpoint_hint_revision(hash, StorageTier::Device)
                .is_none()
        );
        view.with_membership_at_revision(hash, StorageTier::Device, current, |_| {
            panic!("clear removes the sidecar with the membership")
        });
    }

    #[test]
    fn indexed_revision_view_fences_updates_and_preserves_cached_snapshots() {
        let table = KvCacheGroupResidency::new();
        let first = WorkerWithDpRank::new(7, 0);
        let second = WorkerWithDpRank::new(9, 2);
        let first_owner = ResidencyOwner::worker(first);
        let second_owner = ResidencyOwner::worker(second);
        let hash = ExternalSequenceBlockHash(11);
        for worker in [second, first] {
            table.declare_layout(ResidencyOwner::worker(worker), &[0], 1);
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        }
        let mut view = table.read_view();
        for owner in [first_owner, second_owner] {
            view.track_owner(owner);
        }
        let original = view
            .owners_at(hash, StorageTier::Device)
            .collect::<Vec<_>>();
        assert_eq!(original, [first_owner, second_owner]);
        let revision = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut first_generation = None;
        view.with_membership_at_revision(hash, StorageTier::Device, revision, |membership| {
            assert_eq!(membership.len(), 2);
            first_generation = membership.generation_for(first_owner);
            assert_eq!(
                membership
                    .iter()
                    .map(|(owner, _)| owner)
                    .collect::<Vec<_>>(),
                original,
            );
        });
        assert!(first_generation.is_some());
        view.with_membership_at_revision(hash, StorageTier::HostPinned, revision, |membership| {
            assert!(membership.is_empty());
            assert_eq!(membership.generation_for(first_owner), None);
        });
        for invalid in [0, revision + 1] {
            view.with_membership_at_revision(hash, StorageTier::Device, invalid, |_| {
                panic!("invalid revision must not expose membership")
            });
        }
        view.with_membership_at_revision(
            ExternalSequenceBlockHash(12),
            StorageTier::Device,
            revision,
            |_| panic!("unknown hash must not expose membership"),
        );

        table.clear_worker(first);
        view.with_membership_at_revision(hash, StorageTier::Device, revision, |_| {
            panic!("old revision must not survive a clear")
        });
        let cleared = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        view.with_membership_at_revision(hash, StorageTier::Device, cleared, |membership| {
            assert_eq!(membership.len(), 1);
            assert_eq!(membership.generation_for(first_owner), None);
            assert!(membership.generation_for(second_owner).is_some());
        });
        table.apply(&checkpoint(first, StorageTier::Device, 0, hash.0, Some(1)));
        let readded = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_ne!(readded, cleared);
        view.with_membership_at_revision(hash, StorageTier::Device, cleared, |_| {
            panic!("clear/readd must not reuse a revision")
        });
        view.with_membership_at_revision(hash, StorageTier::Device, readded, |membership| {
            assert_eq!(membership.len(), 2);
            assert_ne!(membership.generation_for(first_owner), first_generation);
        });
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            original,
        );
    }

    #[test]
    fn uncached_layouts_seed_legacy_lookups_without_recapturing_new_lifecycles() {
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        for lifecycle in 0..3 {
            let table = KvCacheGroupResidency::new();
            table.declare_layout(owner, &[0], 1);
            table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
            let mut view = table.read_view();
            let captured = view.capture_layout(owner).unwrap();
            assert!(captured.layout().complete_layout);
            assert!(view.layouts.is_empty());
            match lifecycle {
                0 => table.clear_worker(worker),
                1 => table.declare_layout(owner, &[3], 1),
                _ => {
                    table.remove_worker_rank(worker);
                    table.declare_layout(owner, &[0], 1);
                }
            }
            let group = if lifecycle == 1 { 3 } else { 0 };
            table.apply(&checkpoint(
                worker,
                StorageTier::Device,
                group,
                hash.0,
                Some(1),
            ));
            let fresh = table.read_view().capture_layout(owner).unwrap();
            assert_ne!(captured.generation, fresh.generation);
            assert!(!captured.is_eligible_generation(fresh.generation));

            view.remember_layout(owner, Some(captured));
            view.remember_layout(owner, Some(fresh));
            assert_eq!(
                view.capture_layout(owner).unwrap().generation,
                captured.generation
            );
            assert_eq!(view.track_owner(owner), Some(captured.layout()));
            assert!(view.owners_at(hash, StorageTier::Device).next().is_none());
            let revision = view
                .checkpoint_hint_revision(hash, StorageTier::Device)
                .unwrap();
            assert!(owners_at_revision(&view, hash, StorageTier::Device, revision).is_empty());
        }
    }

    #[test]
    fn layout_capture_reuses_prior_legacy_some_and_none() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, &[0], 1);
        let mut legacy = table.read_view();
        legacy.track_owner(owner);
        let original = legacy.capture_layout(owner).unwrap();
        table.clear_worker(worker);
        let current = table.read_view().capture_layout(owner).unwrap();
        assert_ne!(original.generation, current.generation);
        assert_eq!(
            legacy.capture_layout(owner).unwrap().generation,
            original.generation
        );
        legacy.remember_layout(owner, Some(current));
        assert_eq!(
            legacy.capture_layout(owner).unwrap().generation,
            original.generation
        );

        let unknown_worker = WorkerWithDpRank::new(8, 0);
        let unknown_owner = ResidencyOwner::worker(unknown_worker);
        let mut unknown = table.read_view();
        let captured = unknown.capture_layout(unknown_owner);
        assert!(captured.is_none());
        assert!(unknown.layouts.is_empty());
        table.declare_layout(unknown_owner, &[0], 1);
        table.apply(&checkpoint(
            unknown_worker,
            StorageTier::Device,
            0,
            11,
            Some(1),
        ));
        unknown.remember_layout(unknown_owner, captured);
        unknown.remember_layout(
            unknown_owner,
            table.read_view().capture_layout(unknown_owner),
        );
        assert!(unknown.layouts.contains_key(&unknown_owner));
        assert!(unknown.capture_layout(unknown_owner).is_none());
        assert!(unknown.track_owner(unknown_owner).is_none());
        let hash = ExternalSequenceBlockHash(11);
        let revision = unknown
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert!(owners_at_revision(&unknown, hash, StorageTier::Device, revision).is_empty());
    }

    #[test]
    fn raw_revision_facts_need_captured_owner_and_complete_generation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        let unknown = table.read_view();
        let unknown_capture = unknown.capture_layout(owner);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let revision = unknown
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut facts = Vec::new();
        unknown.for_each_membership_at_revision(
            hash,
            StorageTier::Device,
            revision,
            |fact_owner, generation| facts.push((fact_owner, generation)),
        );
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].0, owner);
        assert!(!unknown_capture.is_some_and(|token| token.is_eligible_generation(facts[0].1)));
        assert!(owners_at_revision(&unknown, hash, StorageTier::Device, revision).is_empty());
        assert!(unknown.layouts.is_empty());
        assert!(unknown.shards.get().is_none());

        let current = table.read_view();
        let captured = current.capture_layout(owner).unwrap();
        assert!(captured.is_eligible_generation(facts[0].1));
        let incomplete = WorkerWithDpRank::new(8, 0);
        table.apply(&checkpoint(
            incomplete,
            StorageTier::Device,
            0,
            hash.0,
            None,
        ));
        let incomplete = table
            .read_view()
            .capture_layout(ResidencyOwner::worker(incomplete))
            .unwrap();
        assert!(!incomplete.layout().complete_layout);
        assert!(!incomplete.is_eligible_generation(incomplete.generation));

        table.clear_worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let revision = current
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut credit = Vec::new();
        current.for_each_membership_at_revision(
            hash,
            StorageTier::Device,
            revision,
            |fact_owner, generation| {
                if fact_owner == owner && captured.is_eligible_generation(generation) {
                    credit.push(fact_owner);
                }
            },
        );
        assert!(credit.is_empty());
    }

    #[test]
    fn live_hints_fence_changed_memberships_without_caching_shards() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let other_worker = WorkerWithDpRank::new(8, 0);
        let owner = ResidencyOwner::worker(worker);
        let other = ResidencyOwner::worker(other_worker);
        let hash = ExternalSequenceBlockHash(11);
        let sibling = ExternalSequenceBlockHash(266);
        assert_eq!(
            CheckpointMembershipIndex::shard(hash.0),
            CheckpointMembershipIndex::shard(sibling.0)
        );
        table.declare_layout(owner, &[0], 1);
        table.declare_layout(other, &[0], 1);
        let mut view = table.read_view();
        view.track_owner(owner);
        view.track_owner(other);
        let generation = view.layouts[&other].unwrap().generation;
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );

        let mut event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        let KvCacheEventData::Stored(data) = &mut event.event.data else {
            unreachable!()
        };
        data.blocks.push(KvCacheStoredBlockData {
            block_hash: sibling,
            tokens_hash: LocalBlockHash(0),
            mm_extra_info: None,
        });
        table.apply(&event);
        let revision = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_eq!(
            view.checkpoint_hint_revision(sibling, StorageTier::Device),
            Some(revision)
        );
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::HostPinned),
            None
        );
        assert_eq!(
            view.checkpoint_hint_revision(ExternalSequenceBlockHash(12), StorageTier::Device),
            None
        );
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, revision),
            [owner]
        );
        assert!(owners_at_revision(&view, hash, StorageTier::HostPinned, revision).is_empty());
        table.apply(&event);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            Some(revision)
        );

        // A later owner cannot acquire credit using an earlier tree hint.
        table.apply(&checkpoint(
            other_worker,
            StorageTier::Device,
            0,
            hash.0,
            Some(1),
        ));
        let changed = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_ne!(changed, revision);
        assert!(owners_at_revision(&view, hash, StorageTier::Device, revision).is_empty());
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, changed),
            [owner, other]
        );
        // Publishing the same shard does not invalidate unrelated hashes.
        assert_eq!(
            view.checkpoint_hint_revision(sibling, StorageTier::Device),
            Some(revision)
        );
        let mut current = table.read_view();
        current.track_owner(other);
        assert_eq!(current.layouts[&other].unwrap().generation, generation);
        assert!(view.shards.get().is_none());
    }

    #[test]
    fn live_revision_rejects_remove_readd_aba_without_changing_layout_generation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.apply(&event);
        let mut view = table.read_view();
        view.track_owner(owner);
        let generation = view.layouts[&owner].unwrap().generation;
        let before = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        let mut remove = event.clone();
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );
        assert!(owners_at_revision(&view, hash, StorageTier::Device, before).is_empty());
        table.apply(&event);
        let after = view
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert_ne!(before, after);
        assert!(owners_at_revision(&view, hash, StorageTier::Device, before).is_empty());
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, after),
            [owner]
        );
        let mut current = table.read_view();
        current.track_owner(owner);
        assert_eq!(current.layouts[&owner].unwrap().generation, generation);
        assert!(view.shards.get().is_none());
    }

    #[test]
    fn live_revision_still_requires_tracked_current_layout_generation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0], 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));
        let mut old = table.read_view();
        let revision = old
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert!(owners_at_revision(&old, hash, StorageTier::Device, revision).is_empty());
        old.track_owner(owner);

        table.declare_layout(owner, &[3], 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 3, hash.0, Some(1)));
        let revision = old
            .checkpoint_hint_revision(hash, StorageTier::Device)
            .unwrap();
        assert!(owners_at_revision(&old, hash, StorageTier::Device, revision).is_empty());
        let mut current = table.read_view();
        current.track_owner(owner);
        assert_eq!(
            owners_at_revision(&current, hash, StorageTier::Device, revision),
            [owner]
        );
        assert!(old.shards.get().is_none());
        assert!(current.shards.get().is_none());
    }

    #[test]
    fn membership_revision_exhaustion_disables_live_credit_without_wrapping() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0], 1);
        let shard = &table.memberships.shards[CheckpointMembershipIndex::shard(hash.0)];
        *shard.writer.lock() = u64::MAX - 1;
        let event = checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1));
        table.apply(&event);
        let mut view = table.read_view();
        view.track_owner(owner);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            Some(u64::MAX)
        );
        assert_eq!(
            owners_at_revision(&view, hash, StorageTier::Device, u64::MAX),
            [owner]
        );
        table.apply(&checkpoint(
            worker,
            StorageTier::HostPinned,
            0,
            hash.0,
            Some(1),
        ));
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );
        assert!(owners_at_revision(&view, hash, StorageTier::Device, u64::MAX).is_empty());
        assert!(owners_at_revision(&view, hash, StorageTier::Device, 0).is_empty());
        let mut remove = event.clone();
        remove.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash],
        });
        table.apply(&remove);
        table.apply(&event);
        assert_eq!(*shard.writer.lock(), u64::MAX);
        assert_eq!(
            view.checkpoint_hint_revision(hash, StorageTier::Device),
            None
        );
        // Cached lower-tier lookups continue to expose the actual membership.
        assert_eq!(
            view.owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>(),
            [owner]
        );
    }

    #[test]
    fn short_query_snapshots_only_visited_shards_without_copying_checkpoints() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let mut event = checkpoint(worker, StorageTier::Device, 0, 1, Some(1));
        let KvCacheEventData::Stored(data) = &mut event.event.data else {
            unreachable!()
        };
        data.blocks = (1..=1024)
            .map(|hash| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(hash),
                tokens_hash: LocalBlockHash(0),
                mm_extra_info: None,
            })
            .collect();
        table.apply(&event);

        let mut view = table.read_view();
        view.track_owner(owner);
        assert!(view.shards.get().is_none());
        assert_eq!(view.layouts.len(), 1);
        for hash in 1..=8 {
            assert_eq!(
                view.owners_at(ExternalSequenceBlockHash(hash), StorageTier::Device)
                    .collect::<Vec<_>>(),
                [owner]
            );
        }
        assert_eq!(
            view.shards
                .get()
                .unwrap()
                .iter()
                .filter(|shard| shard.get().is_some())
                .count(),
            8
        );

        // Duplicate events do not publish new shards. Removing one checkpoint
        // only replaces its shard; all unrelated shard snapshots stay shared.
        let before: Vec<_> = table
            .memberships
            .shards
            .iter()
            .map(|shard| shard.entries.load_full())
            .collect();
        table.apply(&event);
        assert!(
            before
                .iter()
                .zip(&table.memberships.shards)
                .all(|(old, shard)| Arc::ptr_eq(old, &shard.entries.load_full()))
        );
        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1)],
        });
        table.apply(&event);
        // Legacy lookups keep the original per-shard snapshot across writes.
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(1), StorageTier::Device)
                .collect::<Vec<_>>(),
            [owner]
        );
        assert_eq!(
            before
                .iter()
                .zip(&table.memberships.shards)
                .filter(|(old, shard)| !Arc::ptr_eq(old, &shard.entries.load_full()))
                .count(),
            1
        );
        let mut removed = table.read_view();
        removed.track_owner(owner);
        assert!(
            removed
                .owners_at(ExternalSequenceBlockHash(1), StorageTier::Device)
                .next()
                .is_none()
        );
        assert_eq!(
            removed
                .owners_at(ExternalSequenceBlockHash(2), StorageTier::Device)
                .collect::<Vec<_>>(),
            [owner]
        );
    }

    #[test]
    fn lazy_snapshots_reject_membership_from_new_owner_lifecycles() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0], 1);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, hash.0, Some(1)));

        for lifecycle in 0..3 {
            let mut old = table.read_view();
            old.track_owner(owner);
            let old_generation = old.layouts[&owner].unwrap().generation;
            match lifecycle {
                0 => table.clear_worker(worker),
                1 => table.declare_layout(owner, &[0, 1], 1),
                _ => {
                    table.remove_worker_rank(worker);
                    table.declare_layout(owner, &[0, 1], 1);
                }
            }
            let groups = if lifecycle == 0 { 1 } else { 2 };
            for group in 0..groups {
                table.apply(&checkpoint(
                    worker,
                    StorageTier::Device,
                    group,
                    hash.0,
                    Some(groups),
                ));
            }
            let mut fresh = table.read_view();
            fresh.track_owner(owner);
            assert!(fresh.layouts[&owner].unwrap().generation > old_generation);
            assert!(old.owners_at(hash, StorageTier::Device).next().is_none());
            assert_eq!(
                fresh
                    .owners_at(hash, StorageTier::Device)
                    .collect::<Vec<_>>(),
                [owner]
            );
        }
    }

    #[test]
    fn concurrent_owner_batches_preserve_all_complete_memberships() {
        let table = KvCacheGroupResidency::new();
        std::thread::scope(|scope| {
            for id in 1..=8 {
                let table = &table;
                scope.spawn(move || {
                    let worker = WorkerWithDpRank::new(id, 0);
                    for group in [0, 1] {
                        let mut event = checkpoint(worker, StorageTier::Device, group, 1, Some(2));
                        let KvCacheEventData::Stored(data) = &mut event.event.data else {
                            unreachable!()
                        };
                        data.blocks = (1..=128)
                            .map(|hash| KvCacheStoredBlockData {
                                block_hash: ExternalSequenceBlockHash(hash),
                                tokens_hash: LocalBlockHash(0),
                                mm_extra_info: None,
                            })
                            .collect();
                        table.apply(&event);
                    }
                });
            }
        });
        let mut view = table.read_view();
        for id in 1..=8 {
            view.track_owner(ResidencyOwner::worker(WorkerWithDpRank::new(id, 0)));
        }
        for hash in 1..=128 {
            let mut owners = view
                .owners_at(ExternalSequenceBlockHash(hash), StorageTier::Device)
                .collect::<Vec<_>>();
            owners.sort_by_key(|owner| match owner {
                ResidencyOwner::Worker(worker) => worker.worker_id,
                _ => unreachable!(),
            });
            assert_eq!(
                owners,
                (1..=8)
                    .map(|id| ResidencyOwner::worker(WorkerWithDpRank::new(id, 0)))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn read_view_does_not_wait_for_owner_or_membership_writer_locks() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        let mut view = table.read_view();
        let state = table.owner(owner).unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let owner_writer = state.write();
            let shard_writer = table.memberships.shards[CheckpointMembershipIndex::shard(11)]
                .writer
                .lock();
            scope.spawn(move || {
                assert!(view.track_owner(owner).unwrap().complete_layout);
                send.send(
                    view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            });
            let result = receive.recv_timeout(std::time::Duration::from_secs(5));
            // Drop guards before asserting so a regression fails instead of
            // leaving a blocked scoped thread behind during panic unwinding.
            drop(shard_writer);
            drop(owner_writer);
            assert_eq!(result.unwrap(), [owner]);
        });
    }

    fn has_checkpoint(
        table: &KvCacheGroupResidency,
        owner: ResidencyOwner,
        hash: u64,
        tier: StorageTier,
    ) -> bool {
        let mut view = table.read_view();
        view.track_owner(owner);
        view.owners_at(ExternalSequenceBlockHash(hash), tier)
            .any(|candidate| candidate == owner)
    }

    #[test]
    fn malformed_zero_group_count_keeps_an_incomplete_layout() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(0)));
        assert!(
            !table
                .read_view()
                .track_owner(owner)
                .unwrap()
                .complete_layout
        );
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn incomplete_membership_and_empty_groups_survive_dump() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        table.declare_groups(owner, &[0, 1, 2]);
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(3)));
        }
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        assert_eq!(restored.groups_for(worker).len(), 3);
        assert!(!has_checkpoint(&restored, owner, 11, StorageTier::Device));
        restored.apply(&checkpoint(worker, StorageTier::Device, 2, 11, Some(3)));
        assert!(has_checkpoint(&restored, owner, 11, StorageTier::Device));
    }

    #[test]
    fn tier_membership_and_removal_are_independent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let chain = [ExternalSequenceBlockHash(11)];
        let mut event = checkpoint(worker, StorageTier::HostPinned, 0, 11, Some(1));
        table.apply(&event);
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::HostPinned));
        event.storage_tier = StorageTier::Device;
        table.apply(&event);
        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: chain.to_vec(),
        });
        table.apply(&event);
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        assert!(has_checkpoint(&table, owner, 11, StorageTier::HostPinned));
    }

    #[test]
    fn missing_completeness_metadata_fails_closed() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, None));
        assert!(!has_checkpoint(&table, owner, 11, StorageTier::Device));
        table.declare_groups(ResidencyOwner::worker(worker), &[0]);
        assert!(has_checkpoint(&table, owner, 11, StorageTier::Device));
    }

    #[test]
    fn cache_clear_preserves_layout_but_worker_retirement_removes_it() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        table.declare_groups(ResidencyOwner::worker(worker), &[0, 1]);
        table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(2)));
        let clear = RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id: 2,
                dp_rank: 0,
                data: KvCacheEventData::Cleared,
            },
        );
        table.clear_event(&clear);
        assert_eq!(table.groups_for(worker).len(), 2);
        assert!(!has_checkpoint(
            &table,
            ResidencyOwner::worker(worker),
            11,
            StorageTier::Device
        ));
        table.remove_worker(7);
        assert!(!table.is_active());
    }

    #[test]
    fn unknown_group_kind_is_consumed_without_mutation() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let mut event = checkpoint(worker, StorageTier::Device, 0, 11, Some(1));
        event.kv_cache_group.as_mut().unwrap().kind = KvCacheGroupKind::Unknown;
        assert!(table.apply(&event));
        assert!(!table.is_active());
    }

    #[test]
    fn read_view_only_exposes_complete_tiers_and_duplicate_events_are_idempotent() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        table.declare_layout(owner, &[0, 1], 1);
        let mut event = checkpoint(worker, StorageTier::Device, 0, 11, Some(2));
        table.apply(&event);
        table.apply(&event);
        let mut incomplete = table.read_view();
        assert!(incomplete.track_owner(owner).unwrap().complete_layout);
        assert!(
            incomplete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );

        table.apply(&checkpoint(worker, StorageTier::Device, 1, 11, Some(2)));
        for group in [0, 1] {
            table.apply(&checkpoint(
                worker,
                StorageTier::HostPinned,
                group,
                11,
                Some(2),
            ));
        }
        let mut complete = table.read_view();
        complete.track_owner(owner);
        complete.track_owner(owner);
        assert_eq!(
            complete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert_eq!(
            complete
                .owners_at(hash, StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        event.event.data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![hash, hash],
        });
        table.apply(&event);
        table.apply(&event);
        let mut removed = table.read_view();
        removed.track_owner(owner);
        assert!(
            removed
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        assert_eq!(
            removed
                .owners_at(hash, StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert_eq!(
            complete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
    }

    #[test]
    fn layout_declaration_and_changes_rebuild_complete_membership() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let owner = ResidencyOwner::worker(worker);
        let hash = ExternalSequenceBlockHash(11);
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, None));
        }
        let mut incomplete = table.read_view();
        assert!(!incomplete.track_owner(owner).unwrap().complete_layout);
        assert!(
            incomplete
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );

        table.declare_layout(owner, &[0, 1], 1);
        table.declare_layout(owner, &[0, 1], 1);
        let mut declared = table.read_view();
        declared.track_owner(owner);
        assert_eq!(
            declared
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        table.declare_layout(owner, &[0, 1, 2], 1);
        let mut changed = table.read_view();
        assert!(changed.track_owner(owner).unwrap().complete_layout);
        assert!(
            changed
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        for group in [0, 1, 2] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(3)));
        }
        table.apply(&checkpoint(worker, StorageTier::Device, 9, 11, Some(3)));
        let mut repopulated = table.read_view();
        repopulated.track_owner(owner);
        assert_eq!(
            repopulated
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );

        table.declare_layout(owner, &[0, 1, 2], 2);
        let mut realigned = table.read_view();
        assert_eq!(realigned.track_owner(owner).unwrap().alignment_blocks, 2);
        assert!(
            realigned
                .owners_at(hash, StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
    }

    #[test]
    fn read_view_tracks_only_requested_owners_from_its_registry_snapshot() {
        let table = KvCacheGroupResidency::new();
        let a = WorkerWithDpRank::new(7, 0);
        let b = WorkerWithDpRank::new(8, 0);
        let later = WorkerWithDpRank::new(9, 0);
        for worker in [a, b] {
            table.apply(&checkpoint(worker, StorageTier::Device, 0, 11, Some(1)));
        }
        let mut view = table.read_view();
        assert!(view.layout(ResidencyOwner::worker(a)).is_none());
        view.track_owner(ResidencyOwner::worker(a));
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[ResidencyOwner::worker(a)]
        );
        assert!(view.layout(ResidencyOwner::worker(b)).is_none());
        view.track_owner(ResidencyOwner::worker(b));
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[ResidencyOwner::worker(a), ResidencyOwner::worker(b)]
        );
        table.apply(&checkpoint(later, StorageTier::Device, 0, 11, Some(1)));
        assert!(view.track_owner(ResidencyOwner::worker(later)).is_none());
        assert!(view.track_owner(ResidencyOwner::worker(later)).is_none());
    }

    #[test]
    fn complete_membership_survives_dump_and_clear_preserves_only_layout() {
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 2);
        let owner = ResidencyOwner::worker(worker);
        table.declare_layout(owner, &[0, 1], 1);
        for group in [0, 1] {
            table.apply(&checkpoint(worker, StorageTier::Device, group, 11, Some(2)));
            table.apply(&checkpoint(
                worker,
                StorageTier::HostPinned,
                group,
                22,
                Some(2),
            ));
        }
        table.apply(&checkpoint(worker, StorageTier::Disk, 0, 33, Some(2)));
        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        let mut view = restored.read_view();
        view.track_owner(owner);
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert_eq!(
            view.owners_at(ExternalSequenceBlockHash(22), StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[owner]
        );
        assert!(
            view.owners_at(ExternalSequenceBlockHash(33), StorageTier::Disk)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );

        restored.clear_worker(worker);
        let mut cleared = restored.read_view();
        assert!(cleared.track_owner(owner).unwrap().complete_layout);
        assert!(
            cleared
                .owners_at(ExternalSequenceBlockHash(11), StorageTier::Device)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        assert!(
            cleared
                .owners_at(ExternalSequenceBlockHash(22), StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice()
                .is_empty()
        );
        restored.remove_worker_rank(worker);
        assert!(restored.read_view().track_owner(owner).is_none());
        assert!(!restored.is_active());
    }

    #[test]
    fn compact_cache_owner_lookup_and_clear_do_not_change_worker_membership() {
        use crate::identity::{
            CacheOwnerId, CacheSemanticsId, ClusterId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };

        let source = CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                ClusterId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        );
        let table = KvCacheGroupResidency::new();
        let worker = WorkerWithDpRank::new(7, 0);
        let worker_owner = ResidencyOwner::worker(worker);
        let cache_owner = ResidencyOwner::cache_owner(source);
        for group in [0, 1] {
            let event = checkpoint(worker, StorageTier::HostPinned, group, 11, Some(2));
            table.apply(&event);
            let mut event =
                RouterEvent::with_cache_owner(7, event.event, StorageTier::HostPinned, source);
            event.kv_cache_group = Some(KvCacheGroupTag {
                group_idx: group,
                kind: KvCacheGroupKind::Mamba,
                expected_group_count: Some(2),
                alignment_blocks: Some(1),
            });
            table.apply(&event);
        }
        let mut view = table.read_view();
        assert_eq!(
            view.owner_for_key(cache_owner.compact_key()),
            Some(cache_owner)
        );
        assert_eq!(
            view.track_cache_owner(cache_owner.compact_key()),
            Some(cache_owner)
        );
        assert_eq!(
            view.track_cache_owner(cache_owner.compact_key()),
            Some(cache_owner)
        );
        view.track_owner(worker_owner);
        let owners = view
            .owners_at(ExternalSequenceBlockHash(11), StorageTier::HostPinned)
            .collect::<Vec<_>>();
        assert_eq!(owners.len(), 2);
        assert!(owners.contains(&cache_owner));
        assert!(owners.contains(&worker_owner));

        let restored = KvCacheGroupResidency::new();
        for event in table.dump_events() {
            restored.apply(&event);
        }
        restored.clear_event(&RouterEvent::with_cache_owner(
            0,
            KvCacheEvent {
                event_id: 2,
                dp_rank: 0,
                data: KvCacheEventData::Cleared,
            },
            StorageTier::HostPinned,
            source,
        ));
        let mut cleared = restored.read_view();
        cleared.track_cache_owner(cache_owner.compact_key());
        cleared.track_owner(worker_owner);
        assert!(cleared.layout(cache_owner).unwrap().complete_layout);
        assert_eq!(
            cleared
                .owners_at(ExternalSequenceBlockHash(11), StorageTier::HostPinned)
                .collect::<Vec<_>>()
                .as_slice(),
            &[worker_owner]
        );
    }
}
