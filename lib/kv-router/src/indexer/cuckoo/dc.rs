// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact, lane-free CKF aggregation for one indexer pool.
//!
//! Event replay is logically idempotent: applying an ordered history through the
//! same watermark converges to the same member ownership and DC refcounts. It is
//! not physically idempotent. Remove/reinsert churn and reconstruction may choose
//! another valid cuckoo layout because relocation depends on occupancy and RNG.
//! A snapshot therefore establishes one physical base, and only that producer's
//! ordered absolute-image deltas may extend it byte-for-byte.

use std::collections::HashMap;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheStoreData, RouterEvent,
    WorkerId, WorkerWithDpRank,
};

use super::addressing::CkfAddressing;
use super::bucket::{CuckooBucketStore, OwnedPackedCkfLane, PackedBucket};
use super::canonical::CanonicalSequenceBlockHash;
use super::global::GlobalCkfBucketImage;
use super::mutator::{CuckooInsertionScratch, CuckooMutator, lane_rng_seed};
use super::{CkfBuildError, CkfConfig, bucket_count, validate_config};

const FORMAT_VERSION: u16 = 1;
const FINGERPRINT_BITS: u8 = 16;
const SLOTS_PER_BUCKET: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DcCkfFormatIdentity {
    format_version: u16,
    seed: u64,
    bucket_count: usize,
    fingerprint_bits: u8,
    slots_per_bucket: u8,
}

impl DcCkfFormatIdentity {
    pub(super) const fn new(seed: u64, bucket_count: usize) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            seed,
            bucket_count,
            fingerprint_bits: FINGERPRINT_BITS,
            slots_per_bucket: SLOTS_PER_BUCKET,
        }
    }

    pub const fn format_version(&self) -> u16 {
        self.format_version
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub const fn bucket_count(&self) -> usize {
        self.bucket_count
    }

    pub const fn fingerprint_bits(&self) -> u8 {
        self.fingerprint_bits
    }

    pub const fn slots_per_bucket(&self) -> u8 {
        self.slots_per_bucket
    }
}

/// Canonical absolute bucket images produced by the actor core before stream sequencing.
///
/// This is deliberately not a transport envelope. The logically serialized publisher owns the
/// lease and sequence and consumes this batch after a complete actor command. Cadence may
/// coalesce several commands, but a batch never splits one event, Clear, or relocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcCkfPublicationBatch {
    images: Vec<GlobalCkfBucketImage>,
}

impl DcCkfPublicationBatch {
    pub fn images(&self) -> &[GlobalCkfBucketImage] {
        &self.images
    }

    pub(crate) fn into_images(self) -> Vec<GlobalCkfBucketImage> {
        self.images
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DcCkfStats {
    aggregation: DcCkfAggregationStats,
    publication: DcCkfPublicationStats,
    memory: DcCkfMemoryStats,
}

impl DcCkfStats {
    pub const fn aggregation(&self) -> DcCkfAggregationStats {
        self.aggregation
    }

    pub const fn publication(&self) -> DcCkfPublicationStats {
        self.publication
    }

    pub const fn memory(&self) -> DcCkfMemoryStats {
        self.memory
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DcCkfAggregationStats {
    member_count: usize,
    contribution_count: usize,
    unique_block_count: usize,
    unknown_removals: u64,
    parent_not_found: u64,
    capacity_failures: u64,
    occupied_bucket_count: usize,
    occupied_slot_count: usize,
}

impl DcCkfAggregationStats {
    pub const fn member_count(&self) -> usize {
        self.member_count
    }

    pub const fn contribution_count(&self) -> usize {
        self.contribution_count
    }

    pub const fn unique_block_count(&self) -> usize {
        self.unique_block_count
    }

    pub const fn unknown_removals(&self) -> u64 {
        self.unknown_removals
    }

    pub const fn parent_not_found(&self) -> u64 {
        self.parent_not_found
    }

    pub const fn capacity_failures(&self) -> u64 {
        self.capacity_failures
    }

    pub const fn occupied_bucket_count(&self) -> usize {
        self.occupied_bucket_count
    }

    pub const fn occupied_slot_count(&self) -> usize {
        self.occupied_slot_count
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DcCkfPublicationStats {
    pending_events: usize,
    physical_touches: u64,
    distinct_touched_buckets: u64,
    emitted_images: u64,
    net_reverted_buckets: u64,
}

impl DcCkfPublicationStats {
    pub const fn pending_events(&self) -> usize {
        self.pending_events
    }

    pub const fn physical_touches(&self) -> u64 {
        self.physical_touches
    }

    pub const fn distinct_touched_buckets(&self) -> u64 {
        self.distinct_touched_buckets
    }

    pub const fn emitted_images(&self) -> u64 {
        self.emitted_images
    }

    pub const fn net_reverted_buckets(&self) -> u64 {
        self.net_reverted_buckets
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DcCkfMemoryStats {
    source_lineage_capacity: usize,
    canonical_owner_capacity: usize,
    filter_bytes: usize,
    dirty_tracking_bytes: usize,
    insertion_scratch_capacity: usize,
}

impl DcCkfMemoryStats {
    pub const fn source_lineage_capacity(&self) -> usize {
        self.source_lineage_capacity
    }

    pub const fn canonical_owner_capacity(&self) -> usize {
        self.canonical_owner_capacity
    }

    pub const fn filter_bytes(&self) -> usize {
        self.filter_bytes
    }

    pub const fn dirty_tracking_bytes(&self) -> usize {
        self.dirty_tracking_bytes
    }

    pub const fn insertion_scratch_capacity(&self) -> usize {
        self.insertion_scratch_capacity
    }
}

#[derive(Debug)]
pub struct DcCkfEventOutcome {
    first_error: Option<KvCacheEventError>,
    publication: Option<DcCkfPublicationBatch>,
    unknown_removals: usize,
    publication_boundary: bool,
}

impl DcCkfEventOutcome {
    pub fn first_error(&self) -> Option<&KvCacheEventError> {
        self.first_error.as_ref()
    }

    pub fn publication(&self) -> Option<&DcCkfPublicationBatch> {
        self.publication.as_ref()
    }

    pub fn into_publication(self) -> Option<DcCkfPublicationBatch> {
        self.publication
    }

    pub fn unknown_removals(&self) -> usize {
        self.unknown_removals
    }

    pub fn publication_boundary(&self) -> bool {
        self.publication_boundary
    }
}

#[derive(Debug)]
struct DirtyWindow {
    words: Box<[u64]>,
    buckets: Vec<usize>,
    originals: Vec<u64>,
}

impl DirtyWindow {
    fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        let word_count = bucket_count.div_ceil(u64::BITS as usize);
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        words.resize(word_count, 0);
        Ok(Self {
            words: words.into_boxed_slice(),
            buckets: Vec::new(),
            originals: Vec::new(),
        })
    }

    fn try_reserve(&mut self, additional: usize) -> Result<(), KvCacheEventError> {
        self.buckets
            .try_reserve(additional)
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        self.originals
            .try_reserve(additional)
            .map_err(|_| KvCacheEventError::AllocationFailed)
    }

    fn mark(&mut self, bucket: usize, original: PackedBucket) {
        let word = bucket / u64::BITS as usize;
        let bit = 1u64 << (bucket % u64::BITS as usize);
        if self.words[word] & bit != 0 {
            return;
        }
        self.words[word] |= bit;
        self.buckets.push(bucket);
        self.originals.push(original.0);
    }

    fn contains(&self, bucket: usize) -> bool {
        let word = bucket / u64::BITS as usize;
        let bit = 1u64 << (bucket % u64::BITS as usize);
        self.words[word] & bit != 0
    }

    fn touched_with_originals(&self) -> impl Iterator<Item = (usize, PackedBucket)> + '_ {
        self.buckets
            .iter()
            .copied()
            .zip(self.originals.iter().copied().map(PackedBucket))
    }

    fn clear(&mut self) {
        for &bucket in &self.buckets {
            let word = bucket / u64::BITS as usize;
            self.words[word] &= !(1u64 << (bucket % u64::BITS as usize));
        }
        self.buckets.clear();
        self.originals.clear();
    }

    fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.words.as_ref())
            + self.buckets.capacity() * std::mem::size_of::<usize>()
            + self.originals.capacity() * std::mem::size_of::<u64>()
    }
}

#[derive(Debug)]
struct PublicationWindow {
    dirty: DirtyWindow,
    pending_events: usize,
}

impl PublicationWindow {
    fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        Ok(Self {
            dirty: DirtyWindow::new(bucket_count)?,
            pending_events: 0,
        })
    }
}

#[derive(Debug, Default)]
struct DcCkfTelemetry {
    unknown_removals: u64,
    parent_not_found: u64,
    capacity_failures: u64,
    physical_touches: u64,
    distinct_touched_buckets: u64,
    emitted_images: u64,
    net_reverted_buckets: u64,
}

/// Reusable per-event working memory for the store path.
///
/// One Stored event needs several transient containers; renting them from the allocator on
/// every event serializes ingest actors on the allocator at saturation, so they live for the
/// lifetime of their owner and are wiped between events, mirroring `remove_scratch`.
#[derive(Debug, Default)]
struct StoreScratch {
    new_mappings: Vec<(ExternalSequenceBlockHash, CanonicalSequenceBlockHash)>,
    candidates: Vec<CanonicalSequenceBlockHash>,
    /// Already-mapped blocks the source still claims. These carry no new exact state, so they
    /// are re-offered for admission only on the relocation-free path.
    reoffered: Vec<CanonicalSequenceBlockHash>,
    batch_mappings: FxHashMap<ExternalSequenceBlockHash, CanonicalSequenceBlockHash>,
    candidate_set: FxHashSet<CanonicalSequenceBlockHash>,
    reoffer_set: FxHashSet<CanonicalSequenceBlockHash>,
    owner_increments: FxHashMap<CanonicalSequenceBlockHash, u32>,
}

impl StoreScratch {
    fn clear(&mut self) {
        self.new_mappings.clear();
        self.candidates.clear();
        self.reoffered.clear();
        self.batch_mappings.clear();
        self.candidate_set.clear();
        self.reoffer_set.clear();
        self.owner_increments.clear();
    }
}

/// Owner count and physical residency for one canonical hash.
///
/// Residency lives in the ownership entry because it is only meaningful for an owned hash, and
/// the padding in a `(hash, u32)` map entry already pays for the flag. Dropping the last owner
/// therefore clears residency by construction rather than through a second synchronized removal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Ownership {
    owners: u32,
    resident: bool,
}

/// Exact source state reconstructed from one replay-ordered rank tree dump.
#[derive(Debug, Default)]
pub struct DcCkfRankReplacement {
    source_lineage: FxHashMap<ExternalSequenceBlockHash, CanonicalSequenceBlockHash>,
    canonical_candidates: Vec<CanonicalSequenceBlockHash>,
    scratch: StoreScratch,
}

impl DcCkfRankReplacement {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one replay-ordered store to this rank replacement.
    ///
    /// Parent-addressed entries must follow their parent. Positional entries with a
    /// non-zero start and no known parent are not a canonical replay form.
    pub fn push_store(&mut self, store: &KvCacheStoreData) -> Result<(), KvCacheEventError> {
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.push_store_with_scratch(store, &mut scratch);
        self.scratch = scratch;
        result
    }

    fn push_store_with_scratch(
        &mut self,
        store: &KvCacheStoreData,
        scratch: &mut StoreScratch,
    ) -> Result<(), KvCacheEventError> {
        prepare_store(&self.source_lineage, store, scratch)?;
        self.source_lineage
            .try_reserve(scratch.new_mappings.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        self.canonical_candidates
            .try_reserve(scratch.candidates.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        for &(external, canonical) in &scratch.new_mappings {
            self.source_lineage.insert(external, canonical);
        }
        self.canonical_candidates
            .extend_from_slice(&scratch.candidates);
        Ok(())
    }

    /// Append one replay-ordered event when it targets primary GPU residency.
    ///
    /// Valid non-primary events are irrelevant to this Device CKF and are ignored. Unknown or
    /// invalid residency encodings fail closed before any replacement state is changed.
    pub fn push_event(&mut self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        match event.targets_primary() {
            Ok(true) => {}
            Ok(false) => return Ok(()),
            Err(_) => return Err(KvCacheEventError::UnsupportedResidencyDomain),
        }
        let KvCacheEventData::Stored(store) = event.event.data else {
            return Err(KvCacheEventError::InvalidBlockSequence);
        };
        self.push_store(&store)
    }
}

/// Exact and physical CKF state for one DC-local indexer pool.
#[derive(Debug)]
pub struct DcCkfState {
    /// Per-source engine vocabulary. Lineage and ownership commit even when physical admission
    /// is capacity-omitted, so CKF capacity does not bound this memory; the operative bound is
    /// the source engine's own KV-cache block count, because the engine emits `Removed` when it
    /// evicts and `clear_worker`/rank replacement drop a source wholesale.
    source_lineage: FxHashMap<
        WorkerWithDpRank,
        FxHashMap<ExternalSequenceBlockHash, CanonicalSequenceBlockHash>,
    >,
    canonical_owners: FxHashMap<CanonicalSequenceBlockHash, Ownership>,
    resident_count: usize,
    filter: OwnedPackedCkfLane,
    addressing: CkfAddressing,
    config: CkfConfig,
    format: DcCkfFormatIdentity,
    rng: u64,
    insertion_scratch: CuckooInsertionScratch,
    publication: PublicationWindow,
    telemetry: DcCkfTelemetry,
    remove_scratch: Vec<ExternalSequenceBlockHash>,
    store_scratch: StoreScratch,
    #[cfg(test)]
    fail_next_snapshot_allocation: bool,
    #[cfg(test)]
    fail_next_store_reserve: bool,
    #[cfg(test)]
    fail_replacement_install_after: Option<usize>,
}

impl DcCkfState {
    pub fn new(config: CkfConfig) -> Result<Self, CkfBuildError> {
        validate_config(config)?;
        let bucket_count = bucket_count(config.expected_blocks_per_dc)?;
        Ok(Self {
            source_lineage: FxHashMap::default(),
            canonical_owners: FxHashMap::default(),
            resident_count: 0,
            filter: OwnedPackedCkfLane::new(bucket_count)?,
            addressing: CkfAddressing::new(bucket_count, config.seed),
            config,
            format: DcCkfFormatIdentity::new(config.seed, bucket_count),
            rng: lane_rng_seed(config.seed, 0),
            insertion_scratch: CuckooInsertionScratch::new(config.max_kicks)
                .map_err(|_| CkfBuildError::AllocationFailed)?,
            publication: PublicationWindow::new(bucket_count)?,
            telemetry: DcCkfTelemetry::default(),
            remove_scratch: Vec::new(),
            store_scratch: StoreScratch::default(),
            #[cfg(test)]
            fail_next_snapshot_allocation: false,
            #[cfg(test)]
            fail_next_store_reserve: false,
            #[cfg(test)]
            fail_replacement_install_after: None,
        })
    }

    pub fn format(&self) -> DcCkfFormatIdentity {
        self.format
    }

    pub fn apply_event(&mut self, event: RouterEvent) -> DcCkfEventOutcome {
        match event.targets_primary() {
            Ok(true) => {}
            Ok(false) => {
                return DcCkfEventOutcome {
                    first_error: None,
                    publication: None,
                    unknown_removals: 0,
                    publication_boundary: false,
                };
            }
            Err(_) => {
                return DcCkfEventOutcome {
                    first_error: Some(KvCacheEventError::UnsupportedResidencyDomain),
                    publication: None,
                    unknown_removals: 0,
                    publication_boundary: false,
                };
            }
        }
        self.publication.pending_events = self.publication.pending_events.saturating_add(1);
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut first_error = None;
        let mut unknown_removals = 0usize;
        match event.event.data {
            KvCacheEventData::Stored(store) => match self.store_batch(worker, &store) {
                Ok(capacity_failures) => {
                    if capacity_failures != 0 {
                        self.telemetry.capacity_failures = self
                            .telemetry
                            .capacity_failures
                            .saturating_add(capacity_failures as u64);
                        first_error = Some(KvCacheEventError::CapacityExhausted);
                    }
                }
                Err(error) => {
                    if matches!(error, KvCacheEventError::ParentBlockNotFound) {
                        self.telemetry.parent_not_found =
                            self.telemetry.parent_not_found.saturating_add(1);
                    }
                    retain_first_error(&mut first_error, error);
                }
            },
            KvCacheEventData::Removed(remove) => {
                for hash in remove.block_hashes {
                    match self.remove(worker, hash) {
                        Ok(false) => unknown_removals += 1,
                        Ok(true) => {}
                        Err(error) => retain_first_error(&mut first_error, error),
                    }
                }
            }
            KvCacheEventData::Cleared => {
                if let Err(error) = self.remove_member(worker) {
                    retain_first_error(&mut first_error, error);
                }
            }
        }
        self.telemetry.unknown_removals = self
            .telemetry
            .unknown_removals
            .saturating_add(unknown_removals as u64);
        // Actor ownership makes this a producer cut after the complete command, including every
        // successful sibling block and any rolled-back capacity omission.
        let publication_boundary = self.publication_due();
        let publication = publication_boundary
            .then(|| self.drain_publication())
            .flatten();
        DcCkfEventOutcome {
            first_error,
            publication,
            unknown_removals,
            publication_boundary,
        }
    }

    pub fn remove_rank(
        &mut self,
        worker: WorkerWithDpRank,
    ) -> Result<Option<DcCkfPublicationBatch>, KvCacheEventError> {
        self.remove_member(worker)?;
        Ok(self.drain_publication())
    }

    pub fn remove_worker(
        &mut self,
        worker_id: WorkerId,
    ) -> Result<Option<DcCkfPublicationBatch>, KvCacheEventError> {
        let mut members = Vec::new();
        members
            .try_reserve(self.source_lineage.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        members.extend(
            self.source_lineage
                .keys()
                .copied()
                .filter(|member| member.worker_id == worker_id),
        );
        let mapping_count = members.iter().try_fold(0usize, |count, member| {
            count.checked_add(self.source_lineage.get(member).map_or(0, FxHashMap::len))
        });
        let mapping_count = mapping_count.ok_or(KvCacheEventError::AllocationFailed)?;
        self.remove_scratch
            .try_reserve(mapping_count)
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        self.reserve_publication_scratch(mapping_count.min(self.format.bucket_count))?;
        for member in members {
            self.remove_member(member)?;
        }
        Ok(self.drain_publication())
    }

    pub fn replace_rank(
        &mut self,
        worker: WorkerWithDpRank,
        replacement: DcCkfRankReplacement,
    ) -> Result<Option<DcCkfPublicationBatch>, KvCacheEventError> {
        let mut replacements = HashMap::new();
        replacements
            .try_reserve(1)
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        replacements.insert(worker, replacement);
        self.replace_ranks(replacements)
    }

    /// Transactionally replace several ranks through one off-side pool rebuild.
    pub fn replace_ranks(
        &mut self,
        replacements: HashMap<WorkerWithDpRank, DcCkfRankReplacement>,
    ) -> Result<Option<DcCkfPublicationBatch>, KvCacheEventError> {
        let mut replacement = Self::new(self.config).map_err(|error| match error {
            CkfBuildError::AllocationFailed => KvCacheEventError::AllocationFailed,
            _ => KvCacheEventError::IndexerInvariantViolation,
        })?;
        let mut retained = Vec::new();
        retained
            .try_reserve(self.source_lineage.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        retained.extend(
            self.source_lineage
                .iter()
                .filter(|(member, _)| !replacements.contains_key(member)),
        );
        retained.sort_unstable_by_key(|(member, _)| **member);
        for (&member, lineage) in retained {
            replacement.install_replacement(member, replacement_from_lineage(lineage)?)?;
        }
        let mut ordered_replacements = Vec::new();
        ordered_replacements
            .try_reserve(replacements.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        ordered_replacements.extend(replacements);
        ordered_replacements.sort_unstable_by_key(|(member, _)| *member);
        // The index feeds the test-only install failpoint; release builds discard it.
        #[allow(clippy::unused_enumerate_index)]
        for (_index, (member, rank_replacement)) in ordered_replacements.into_iter().enumerate() {
            #[cfg(test)]
            if self.fail_replacement_install_after == Some(_index) {
                return Err(KvCacheEventError::AllocationFailed);
            }
            replacement.install_replacement(member, rank_replacement)?;
        }

        replacement.publication.dirty.clear();
        replacement
            .publication
            .dirty
            .try_reserve(self.format.bucket_count)?;
        for (bucket, published) in self.publication.dirty.touched_with_originals() {
            if replacement.filter.load_bucket(bucket) != published {
                replacement.publication.dirty.mark(bucket, published);
            }
        }
        for bucket in 0..self.format.bucket_count {
            if self.publication.dirty.contains(bucket) {
                continue;
            }
            let before = self.filter.load_bucket(bucket);
            let after = replacement.filter.load_bucket(bucket);
            if before != after {
                replacement.publication.dirty.mark(bucket, before);
            }
        }
        replacement.publication.pending_events = 1;
        replacement.telemetry.unknown_removals = self.telemetry.unknown_removals;
        replacement.telemetry.parent_not_found = self.telemetry.parent_not_found;
        replacement.telemetry.capacity_failures = self
            .telemetry
            .capacity_failures
            .saturating_add(replacement.telemetry.capacity_failures);
        replacement.telemetry.physical_touches = self
            .telemetry
            .physical_touches
            .saturating_add(replacement.telemetry.physical_touches);
        replacement.telemetry.distinct_touched_buckets = self.telemetry.distinct_touched_buckets;
        replacement.telemetry.emitted_images = self.telemetry.emitted_images;
        replacement.telemetry.net_reverted_buckets = self.telemetry.net_reverted_buckets;
        *self = replacement;
        Ok(self.drain_publication())
    }

    pub fn flush(&mut self) -> Option<DcCkfPublicationBatch> {
        self.drain_publication()
    }

    pub fn has_pending_publication(&self) -> bool {
        self.publication.pending_events != 0 || !self.publication.dirty.buckets.is_empty()
    }

    pub fn pending_event_count(&self) -> usize {
        self.publication.pending_events
    }

    /// Copy one actor-serialized barrier snapshot, then drain its pending absolute-image tail.
    ///
    /// The core deliberately does not tag the snapshot with a lease or sequence. Its publisher
    /// must first enqueue the returned tail, record its terminal sequence, then install the copy.
    pub fn barrier_snapshot(
        &mut self,
    ) -> Result<(Option<DcCkfPublicationBatch>, Box<[u64]>), CkfBuildError> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_snapshot_allocation) {
            return Err(CkfBuildError::AllocationFailed);
        }
        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(self.format.bucket_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        buckets
            .extend((0..self.format.bucket_count).map(|bucket| self.filter.load_bucket(bucket).0));
        // Keep the dirty tail owned by the producer until every fallible snapshot allocation is
        // complete. Otherwise an allocation failure could discard unpublished images while the
        // old lease and sequence remain valid.
        let publication = self.drain_publication();
        Ok((publication, buckets.into_boxed_slice()))
    }

    #[cfg(test)]
    fn fail_next_snapshot_allocation(&mut self) {
        self.fail_next_snapshot_allocation = true;
    }

    pub fn stats(&self) -> DcCkfStats {
        DcCkfStats {
            aggregation: DcCkfAggregationStats {
                member_count: self.source_lineage.len(),
                contribution_count: self.source_lineage.values().map(FxHashMap::len).sum(),
                unique_block_count: self.canonical_owners.len(),
                unknown_removals: self.telemetry.unknown_removals,
                parent_not_found: self.telemetry.parent_not_found,
                capacity_failures: self.telemetry.capacity_failures,
                occupied_bucket_count: self.current_occupied_bucket_count(),
                occupied_slot_count: self.resident_count,
            },
            publication: DcCkfPublicationStats {
                pending_events: self.publication.pending_events,
                physical_touches: self.telemetry.physical_touches,
                distinct_touched_buckets: self.telemetry.distinct_touched_buckets,
                emitted_images: self.telemetry.emitted_images,
                net_reverted_buckets: self.telemetry.net_reverted_buckets,
            },
            memory: DcCkfMemoryStats {
                source_lineage_capacity: self
                    .source_lineage
                    .values()
                    .map(FxHashMap::capacity)
                    .sum(),
                canonical_owner_capacity: self.canonical_owners.capacity(),
                filter_bytes: self.filter.byte_len(),
                dirty_tracking_bytes: self.publication.dirty.byte_len(),
                insertion_scratch_capacity: self.insertion_scratch.capacity(),
            },
        }
    }

    /// Lifetime count of physical admissions omitted for capacity, including omissions that
    /// happen while installing rank replacements. Exposed unconditionally so ingest hosts can
    /// surface recovery omissions the way they surface live ones.
    pub const fn lifetime_capacity_omissions(&self) -> u64 {
        self.telemetry.capacity_failures
    }

    pub fn member_block_count(&self, worker: WorkerWithDpRank) -> usize {
        self.source_lineage.get(&worker).map_or(0, FxHashMap::len)
    }

    pub fn member_counts(&self) -> Vec<(WorkerWithDpRank, usize)> {
        let mut counts: Vec<_> = self
            .source_lineage
            .iter()
            .map(|(&worker, blocks)| (worker, blocks.len()))
            .collect();
        counts.sort_unstable_by_key(|(worker, _)| *worker);
        counts
    }

    pub fn contains(&self, hash: CanonicalSequenceBlockHash) -> bool {
        let probe = self.addressing.prepare(hash.as_u64());
        self.filter
            .load_bucket(probe.bucket_a)
            .contains(probe.fingerprint)
            || self
                .filter
                .load_bucket(probe.bucket_b)
                .contains(probe.fingerprint)
    }

    fn publication_due(&self) -> bool {
        self.publication.pending_events >= self.config.publish_every_n_events
    }

    fn store_batch(
        &mut self,
        worker: WorkerWithDpRank,
        store: &KvCacheStoreData,
    ) -> Result<usize, KvCacheEventError> {
        // The scratch moves out for the duration of the call so its buffers can be filled
        // while `self` is mutated, and moves back even on the error path so the retained
        // capacity survives.
        let mut scratch = std::mem::take(&mut self.store_scratch);
        let result = self.store_batch_with_scratch(worker, store, &mut scratch);
        self.store_scratch = scratch;
        result
    }

    fn store_batch_with_scratch(
        &mut self,
        worker: WorkerWithDpRank,
        store: &KvCacheStoreData,
        scratch: &mut StoreScratch,
    ) -> Result<usize, KvCacheEventError> {
        let empty_lineage = FxHashMap::default();
        let lineage = self.source_lineage.get(&worker).unwrap_or(&empty_lineage);
        prepare_store(lineage, store, scratch)?;
        if scratch.new_mappings.is_empty()
            && scratch.candidates.is_empty()
            && scratch.reoffered.is_empty()
        {
            return Ok(0);
        }
        #[cfg(test)]
        if self.fail_next_store_reserve {
            self.fail_next_store_reserve = false;
            return Err(KvCacheEventError::AllocationFailed);
        }

        scratch
            .owner_increments
            .try_reserve(scratch.new_mappings.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        for &(_, canonical) in &scratch.new_mappings {
            let increment = scratch.owner_increments.entry(canonical).or_default();
            *increment = increment
                .checked_add(1)
                .ok_or(KvCacheEventError::OwnershipDegreeOverflow)?;
        }
        for (&canonical, &increment) in &scratch.owner_increments {
            self.canonical_owners
                .get(&canonical)
                .map_or(0, |ownership| ownership.owners)
                .checked_add(increment)
                .ok_or(KvCacheEventError::OwnershipDegreeOverflow)?;
        }

        if !scratch.new_mappings.is_empty() {
            if let Some(lineage) = self.source_lineage.get_mut(&worker) {
                lineage
                    .try_reserve(scratch.new_mappings.len())
                    .map_err(|_| KvCacheEventError::AllocationFailed)?;
            } else {
                self.source_lineage
                    .try_reserve(1)
                    .map_err(|_| KvCacheEventError::AllocationFailed)?;
            }
            let missing_owner_entries = scratch
                .owner_increments
                .keys()
                .filter(|canonical| !self.canonical_owners.contains_key(canonical))
                .count();
            self.canonical_owners
                .try_reserve(missing_owner_entries)
                .map_err(|_| KvCacheEventError::AllocationFailed)?;
        }

        let admission_candidates = scratch
            .candidates
            .iter()
            .filter(|canonical| !self.is_resident(canonical))
            .count();
        // A relocation-free admission writes exactly one bucket, so re-offers add one touch each.
        let reoffered_admissions = scratch
            .reoffered
            .iter()
            .filter(|canonical| !self.is_resident(canonical))
            .count();
        let maximum_new_touches = admission_candidates
            .checked_mul(self.config.max_kicks.saturating_add(1))
            .and_then(|touches| touches.checked_add(reoffered_admissions))
            .ok_or(KvCacheEventError::AllocationFailed)?
            .min(self.format.bucket_count);
        self.reserve_publication_scratch(maximum_new_touches)?;

        if !scratch.new_mappings.is_empty() {
            if let Some(lineage) = self.source_lineage.get_mut(&worker) {
                for &(external, canonical) in &scratch.new_mappings {
                    let previous = lineage.insert(external, canonical);
                    debug_assert!(previous.is_none());
                }
            } else {
                let mut lineage = FxHashMap::default();
                lineage
                    .try_reserve(scratch.new_mappings.len())
                    .map_err(|_| KvCacheEventError::AllocationFailed)?;
                for &(external, canonical) in &scratch.new_mappings {
                    lineage.insert(external, canonical);
                }
                self.source_lineage.insert(worker, lineage);
            }
            for (&canonical, &increment) in &scratch.owner_increments {
                let ownership = self.canonical_owners.entry(canonical).or_default();
                ownership.owners += increment;
            }
        }

        let mut capacity_failures = 0usize;
        for &canonical in &scratch.candidates {
            if self.is_resident(&canonical) {
                continue;
            }
            match self.insert_canonical(canonical) {
                Ok(()) => {
                    self.mark_resident(canonical);
                }
                Err(KvCacheEventError::CapacityExhausted) => {
                    capacity_failures = capacity_failures.saturating_add(1);
                }
                Err(error) => return Err(error),
            }
        }

        // A block omitted for capacity has no other retry path while it keeps a single owner, so
        // re-offer it whenever the source repeats it. This is deliberately relocation-free: the
        // regime that produces omissions is a saturated filter, where a full kick search would
        // burn `max_kicks` relocations and a rollback per repeat. A failure here is not reported
        // as a new capacity omission because the first store already reported it.
        for &canonical in &scratch.reoffered {
            if self.is_resident(&canonical) {
                continue;
            }
            match self.insert_canonical_without_relocation(canonical) {
                Ok(()) => self.mark_resident(canonical),
                Err(KvCacheEventError::CapacityExhausted) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(capacity_failures)
    }

    fn install_replacement(
        &mut self,
        worker: WorkerWithDpRank,
        replacement: DcCkfRankReplacement,
    ) -> Result<(), KvCacheEventError> {
        if self.source_lineage.contains_key(&worker) {
            return Err(KvCacheEventError::IndexerInvariantViolation);
        }
        if replacement.source_lineage.is_empty() {
            return Ok(());
        }

        let mut owner_increments = FxHashMap::<CanonicalSequenceBlockHash, u32>::default();
        owner_increments
            .try_reserve(replacement.source_lineage.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        for &canonical in replacement.source_lineage.values() {
            let increment = owner_increments.entry(canonical).or_default();
            *increment = increment
                .checked_add(1)
                .ok_or(KvCacheEventError::OwnershipDegreeOverflow)?;
        }
        for (&canonical, &increment) in &owner_increments {
            self.canonical_owners
                .get(&canonical)
                .map_or(0, |ownership| ownership.owners)
                .checked_add(increment)
                .ok_or(KvCacheEventError::OwnershipDegreeOverflow)?;
        }

        self.source_lineage
            .try_reserve(1)
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        self.canonical_owners
            .try_reserve(owner_increments.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        let maximum_new_touches = replacement
            .canonical_candidates
            .len()
            .checked_mul(self.config.max_kicks.saturating_add(1))
            .ok_or(KvCacheEventError::AllocationFailed)?
            .min(self.format.bucket_count);
        self.reserve_publication_scratch(maximum_new_touches)?;

        self.source_lineage
            .insert(worker, replacement.source_lineage);
        for (canonical, increment) in owner_increments {
            let ownership = self.canonical_owners.entry(canonical).or_default();
            ownership.owners += increment;
        }
        for canonical in replacement.canonical_candidates {
            if self.is_resident(&canonical) {
                continue;
            }
            match self.insert_canonical(canonical) {
                Ok(()) => {
                    self.mark_resident(canonical);
                }
                Err(KvCacheEventError::CapacityExhausted) => {
                    self.telemetry.capacity_failures =
                        self.telemetry.capacity_failures.saturating_add(1);
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    fn is_resident(&self, canonical: &CanonicalSequenceBlockHash) -> bool {
        self.canonical_owners
            .get(canonical)
            .is_some_and(|ownership| ownership.resident)
    }

    /// Record a successful admission. The owner entry always exists here: every admission
    /// candidate comes from a mapping whose ownership was committed earlier in the same operation.
    fn mark_resident(&mut self, canonical: CanonicalSequenceBlockHash) {
        match self.canonical_owners.get_mut(&canonical) {
            Some(ownership) => {
                debug_assert!(!ownership.resident);
                ownership.resident = true;
                self.resident_count += 1;
            }
            None => debug_assert!(false, "admitted a canonical hash with no owner"),
        }
    }

    fn insert_canonical(
        &mut self,
        hash: CanonicalSequenceBlockHash,
    ) -> Result<(), KvCacheEventError> {
        let dirty = &mut self.publication.dirty;
        let physical_touches = &mut self.telemetry.physical_touches;
        CuckooMutator::new(&self.filter, &self.addressing, self.config.max_kicks)
            .insert_with_originals(
                hash,
                &mut self.rng,
                &mut self.insertion_scratch,
                |bucket, original| {
                    *physical_touches = physical_touches.saturating_add(1);
                    dirty.mark(bucket, original);
                },
            )
    }

    /// Admit a hash only if one of its two home buckets already has a free slot.
    ///
    /// `max_kicks` of zero makes [`CuckooMutator`] skip the relocation search entirely, so this
    /// costs two bucket loads and never rolls back speculative writes.
    fn insert_canonical_without_relocation(
        &mut self,
        hash: CanonicalSequenceBlockHash,
    ) -> Result<(), KvCacheEventError> {
        let dirty = &mut self.publication.dirty;
        let physical_touches = &mut self.telemetry.physical_touches;
        CuckooMutator::new(&self.filter, &self.addressing, 0).insert_with_originals(
            hash,
            &mut self.rng,
            &mut self.insertion_scratch,
            |bucket, original| {
                *physical_touches = physical_touches.saturating_add(1);
                dirty.mark(bucket, original);
            },
        )
    }

    fn remove(
        &mut self,
        worker: WorkerWithDpRank,
        external: ExternalSequenceBlockHash,
    ) -> Result<bool, KvCacheEventError> {
        let Some(canonical) = self
            .source_lineage
            .get(&worker)
            .and_then(|lineage| lineage.get(&external))
            .copied()
        else {
            return Ok(false);
        };
        let ownership = self
            .canonical_owners
            .get(&canonical)
            .copied()
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let current = ownership.owners;
        let remove_resident = current == 1 && ownership.resident;
        if remove_resident {
            self.reserve_publication_scratch(1)?;
        }
        let mut lineage = self
            .source_lineage
            .remove(&worker)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        if lineage.get(&external) != Some(&canonical) {
            self.source_lineage.insert(worker, lineage);
            return Err(KvCacheEventError::IndexerInvariantViolation);
        }
        if remove_resident {
            let dirty = &mut self.publication.dirty;
            let physical_touches = &mut self.telemetry.physical_touches;
            if let Err(error) =
                CuckooMutator::new(&self.filter, &self.addressing, self.config.max_kicks)
                    .remove_with_original(canonical, |bucket, original| {
                        *physical_touches = physical_touches.saturating_add(1);
                        dirty.mark(bucket, original);
                    })
            {
                self.source_lineage.insert(worker, lineage);
                return Err(error);
            }
        }

        if current == 1 {
            if self.canonical_owners.remove(&canonical).is_none() {
                self.source_lineage.insert(worker, lineage);
                return Err(KvCacheEventError::IndexerInvariantViolation);
            }
            if remove_resident {
                self.resident_count -= 1;
            }
        } else {
            let Some(ownership) = self.canonical_owners.get_mut(&canonical) else {
                self.source_lineage.insert(worker, lineage);
                return Err(KvCacheEventError::IndexerInvariantViolation);
            };
            ownership.owners = current - 1;
        }
        let removed = lineage.remove(&external);
        debug_assert_eq!(removed, Some(canonical));
        if !lineage.is_empty() {
            self.source_lineage.insert(worker, lineage);
        }
        Ok(true)
    }

    fn remove_member(&mut self, worker: WorkerWithDpRank) -> Result<(), KvCacheEventError> {
        self.remove_scratch.clear();
        let Some(lineage) = self.source_lineage.get(&worker) else {
            return Ok(());
        };
        self.remove_scratch
            .try_reserve(lineage.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        // Every mapping can become the source's last owner after earlier mappings in this clear
        // are removed, including aliases of one canonical hash. Reserve before the first mutation.
        let maximum_new_touches = lineage.len().min(self.format.bucket_count);
        self.remove_scratch.extend(lineage.keys().copied());
        self.reserve_publication_scratch(maximum_new_touches)?;
        for index in 0..self.remove_scratch.len() {
            let hash = self.remove_scratch[index];
            self.remove(worker, hash)?;
        }
        self.remove_scratch.clear();
        self.source_lineage.remove(&worker);
        Ok(())
    }

    fn drain_publication(&mut self) -> Option<DcCkfPublicationBatch> {
        if self.publication.pending_events == 0 && self.publication.dirty.buckets.is_empty() {
            return None;
        }
        let distinct_touched = self.publication.dirty.buckets.len() as u64;
        let mut emitted_images = 0usize;
        for (index, &bucket) in self.publication.dirty.buckets.iter().enumerate() {
            if self.filter.load_bucket(bucket).0 != self.publication.dirty.originals[index] {
                emitted_images += 1;
            }
        }
        // The dirty scratch is reserved for worst-case rollback (candidates * (max_kicks + 1)
        // touches), so the payload is materialized into an exact-size vector in a second pass
        // instead of letting that capacity escape into the fan-out queue with the batch.
        let images = (emitted_images > 0).then(|| {
            let mut images = Vec::with_capacity(emitted_images);
            for (index, &bucket) in self.publication.dirty.buckets.iter().enumerate() {
                let value = self.filter.load_bucket(bucket).0;
                if value != self.publication.dirty.originals[index] {
                    images.push(GlobalCkfBucketImage::new(bucket, value));
                }
            }
            images
        });
        self.publication.dirty.clear();
        self.telemetry.distinct_touched_buckets = self
            .telemetry
            .distinct_touched_buckets
            .saturating_add(distinct_touched);
        self.telemetry.emitted_images = self
            .telemetry
            .emitted_images
            .saturating_add(emitted_images as u64);
        self.telemetry.net_reverted_buckets = self
            .telemetry
            .net_reverted_buckets
            .saturating_add(distinct_touched.saturating_sub(emitted_images as u64));
        self.publication.pending_events = 0;
        Some(DcCkfPublicationBatch { images: images? })
    }

    fn reserve_publication_scratch(
        &mut self,
        maximum_new_touches: usize,
    ) -> Result<(), KvCacheEventError> {
        let maximum_new_touches = maximum_new_touches.min(
            self.format
                .bucket_count
                .saturating_sub(self.publication.dirty.buckets.len()),
        );
        self.publication.dirty.try_reserve(maximum_new_touches)
    }

    fn current_occupied_bucket_count(&self) -> usize {
        // This full scan is diagnostic-only. Never cache it by adding work to mutation or
        // publication; Relay calls stats only behind its diagnostics/test surface.
        (0..self.format.bucket_count)
            .filter(|&bucket| self.filter.load_bucket(bucket) != PackedBucket::default())
            .count()
    }
}

fn prepare_store(
    lineage: &FxHashMap<ExternalSequenceBlockHash, CanonicalSequenceBlockHash>,
    store: &KvCacheStoreData,
    scratch: &mut StoreScratch,
) -> Result<(), KvCacheEventError> {
    scratch.clear();
    if store.blocks.is_empty() {
        return Ok(());
    }

    let parent = if let Some(parent) = store.parent_hash {
        Some(
            lineage
                .get(&parent)
                .copied()
                .ok_or(KvCacheEventError::ParentBlockNotFound)?,
        )
    } else {
        if store.start_position.is_some_and(|position| position != 0) {
            return Err(KvCacheEventError::InvalidBlockSequence);
        }
        None
    };

    // try_reserve on a reused container only reserves the shortfall, so the fail-closed
    // AllocationFailed contract is unchanged while steady-state events never touch the
    // allocator.
    scratch
        .new_mappings
        .try_reserve(store.blocks.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    scratch
        .candidates
        .try_reserve(store.blocks.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    scratch
        .reoffered
        .try_reserve(store.blocks.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    scratch
        .batch_mappings
        .try_reserve(store.blocks.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    scratch
        .candidate_set
        .try_reserve(store.blocks.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    scratch
        .reoffer_set
        .try_reserve(store.blocks.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;

    let mut previous = parent;
    for block in &store.blocks {
        let canonical = previous.map_or_else(
            || CanonicalSequenceBlockHash::root(block.tokens_hash),
            |parent| parent.child(block.tokens_hash),
        );
        previous = Some(canonical);

        let existing = lineage
            .get(&block.block_hash)
            .copied()
            .or_else(|| scratch.batch_mappings.get(&block.block_hash).copied());
        match existing {
            Some(existing) if existing != canonical => {
                return Err(KvCacheEventError::InvalidBlockSequence);
            }
            Some(_) => {
                if scratch.reoffer_set.insert(canonical) {
                    scratch.reoffered.push(canonical);
                }
            }
            None => {
                scratch.batch_mappings.insert(block.block_hash, canonical);
                scratch.new_mappings.push((block.block_hash, canonical));
                if scratch.candidate_set.insert(canonical) {
                    scratch.candidates.push(canonical);
                }
            }
        }
    }

    // A canonical hash introduced by this batch is admitted with full relocation, so it must not
    // also be re-offered on the cheap path.
    let StoreScratch {
        reoffered,
        candidate_set,
        ..
    } = scratch;
    reoffered.retain(|canonical| !candidate_set.contains(canonical));
    Ok(())
}

fn replacement_from_lineage(
    lineage: &FxHashMap<ExternalSequenceBlockHash, CanonicalSequenceBlockHash>,
) -> Result<DcCkfRankReplacement, KvCacheEventError> {
    // Retained lineage is trusted state copied straight into the off-side rebuild. Admission
    // order does not need canonicalizing: publication ships absolute bucket images taken from
    // the final filter, so consumer parity is order-independent, and replay equivalence is
    // logical rather than byte-level (`replay_churn_requires_logical_and_membership_not_byte_parity`).
    let mut replacement = DcCkfRankReplacement::new();
    replacement
        .source_lineage
        .try_reserve(lineage.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    replacement
        .canonical_candidates
        .try_reserve(lineage.len())
        .map_err(|_| KvCacheEventError::AllocationFailed)?;
    for (&external, &canonical) in lineage {
        replacement.source_lineage.insert(external, canonical);
        replacement.canonical_candidates.push(canonical);
    }
    Ok(replacement)
}

fn retain_first_error(slot: &mut Option<KvCacheEventError>, error: KvCacheEventError) {
    if slot.is_none() {
        *slot = Some(error);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::protocols::{
        BlockHashOptions, KvCacheEvent, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, ResidencyDomain, StorageTier, WireResidencyDomain,
        compute_block_hash_for_seq,
    };

    use super::*;

    const EXTERNAL_MASK: u64 = 0xA5A5_5A5A_D3C4_B2E1;

    fn external(hash: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(hash ^ EXTERNAL_MASK)
    }

    fn canonical_root(hash: u64) -> CanonicalSequenceBlockHash {
        CanonicalSequenceBlockHash::root(LocalBlockHash(hash))
    }

    fn canonical_chain(hashes: &[u64]) -> Vec<CanonicalSequenceBlockHash> {
        let Some((&first, rest)) = hashes.split_first() else {
            return Vec::new();
        };
        let mut current = canonical_root(first);
        let mut canonical = vec![current];
        canonical.extend(rest.iter().map(|&hash| {
            current = current.child(LocalBlockHash(hash));
            current
        }));
        canonical
    }

    fn store_data(
        parent_hash: Option<ExternalSequenceBlockHash>,
        hashes: &[u64],
    ) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash,
            start_position: None,
            blocks: hashes
                .iter()
                .copied()
                .map(|hash| KvCacheStoredBlockData {
                    block_hash: external(hash),
                    tokens_hash: LocalBlockHash(hash),
                    mm_extra_info: None,
                })
                .collect(),
        }
    }

    fn stored(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(store_data(None, hashes)),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn stored_with_parent(
        worker: WorkerWithDpRank,
        event_id: u64,
        parent: u64,
        hashes: &[u64],
    ) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(store_data(Some(external(parent)), hashes)),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn stored_exact(
        worker: WorkerWithDpRank,
        event_id: u64,
        parent_hash: Option<ExternalSequenceBlockHash>,
        start_position: Option<u32>,
        blocks: &[(ExternalSequenceBlockHash, LocalBlockHash)],
    ) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position,
                    blocks: blocks
                        .iter()
                        .map(|&(block_hash, tokens_hash)| KvCacheStoredBlockData {
                            block_hash,
                            tokens_hash,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn removed(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes.iter().copied().map(external).collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn removed_exact(
        worker: WorkerWithDpRank,
        event_id: u64,
        hashes: &[ExternalSequenceBlockHash],
    ) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes.to_vec(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn replacement(hashes: &[u64]) -> DcCkfRankReplacement {
        let mut replacement = DcCkfRankReplacement::new();
        replacement.push_store(&store_data(None, hashes)).unwrap();
        replacement
    }

    fn cleared(worker_id: WorkerId, dp_rank: u32, event_id: u64) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank,
            },
        )
    }

    #[test]
    fn shared_ownership_survives_nonfinal_then_final_removal() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let hash = canonical_root(11);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored(first, 1, &[11]));
        state.apply_event(stored(second, 1, &[11]));
        assert_eq!(state.stats().aggregation().contribution_count(), 2);
        assert_eq!(state.stats().aggregation().unique_block_count(), 1);

        state.apply_event(removed(first, 2, &[11]));
        assert!(state.contains(hash));
        assert_eq!(state.stats().aggregation().unique_block_count(), 1);

        state.apply_event(removed(second, 2, &[11]));
        assert!(!state.contains(hash));
        assert_eq!(state.stats().aggregation().member_count(), 0);
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
    }

    #[test]
    fn rank_clear_preserves_sibling_rank_and_another_workers_shared_hash() {
        let first_rank = WorkerWithDpRank::new(1, 0);
        let second_rank = WorkerWithDpRank::new(1, 1);
        let other_worker = WorkerWithDpRank::new(2, 0);
        let shared = canonical_root(11);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored(first_rank, 1, &[11]));
        state.apply_event(stored(second_rank, 1, &[12]));
        state.apply_event(stored(other_worker, 1, &[11]));
        state.apply_event(cleared(first_rank.worker_id, first_rank.dp_rank, 2));

        assert_eq!(state.member_block_count(first_rank), 0);
        assert_eq!(state.member_block_count(second_rank), 1);
        assert_eq!(state.member_block_count(other_worker), 1);
        assert_eq!(state.stats().aggregation().unique_block_count(), 2);
        assert!(state.contains(shared));
    }

    #[test]
    fn non_device_store_is_ignored_but_unknown_device_remove_is_counted() {
        let worker = WorkerWithDpRank::new(1, 0);
        let resident = canonical_root(7);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut host_store = stored(worker, 1, &[7]);
        host_store.storage_tier = StorageTier::HostPinned;

        let ignored = state.apply_event(host_store);
        assert!(!ignored.publication_boundary());
        assert_eq!(state.stats().aggregation().contribution_count(), 0);

        state.apply_event(stored(worker, 2, &[7]));
        let unknown = state.apply_event(removed(worker, 3, &[99]));
        assert_eq!(unknown.unknown_removals(), 1);
        assert_eq!(state.stats().aggregation().unknown_removals(), 1);
        assert_eq!(state.stats().aggregation().contribution_count(), 1);
        assert!(state.contains(resident));
    }

    #[test]
    fn cache_owner_clear_is_ignored_without_mutating_primary_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let canonical = canonical_root(7);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        state.apply_event(stored(worker, 1, &[7]));
        state.barrier_snapshot().unwrap();
        let before_stats = state.stats();
        let before_lineage = state.source_lineage.clone();
        let before_owners = state.canonical_owners.clone();
        let mut cache_owner_clear = cleared(worker.worker_id, worker.dp_rank, 2);
        cache_owner_clear.residency_domain =
            WireResidencyDomain::explicit(ResidencyDomain::CacheOwner);

        let outcome = state.apply_event(cache_owner_clear);

        assert!(outcome.first_error().is_none());
        assert!(!outcome.publication_boundary());
        assert!(outcome.publication().is_none());
        assert_eq!(state.stats(), before_stats);
        assert_eq!(state.source_lineage, before_lineage);
        assert_eq!(state.canonical_owners, before_owners);
        assert!(state.is_resident(&canonical));
        assert!(!state.has_pending_publication());
    }

    #[test]
    fn unknown_residency_fails_closed_without_mutating_primary_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let canonical = canonical_root(7);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        state.apply_event(stored(worker, 1, &[7]));
        state.barrier_snapshot().unwrap();
        let before_stats = state.stats();
        let before_lineage = state.source_lineage.clone();
        let before_owners = state.canonical_owners.clone();
        let mut invalid = cleared(worker.worker_id, worker.dp_rank, 2);
        invalid.residency_domain = WireResidencyDomain::Unknown("future_domain".into());

        let outcome = state.apply_event(invalid);

        assert_eq!(
            outcome.first_error(),
            Some(&KvCacheEventError::UnsupportedResidencyDomain)
        );
        assert!(!outcome.publication_boundary());
        assert!(outcome.publication().is_none());
        assert_eq!(state.stats(), before_stats);
        assert_eq!(state.source_lineage, before_lineage);
        assert_eq!(state.canonical_owners, before_owners);
        assert!(state.is_resident(&canonical));
        assert!(!state.has_pending_publication());
    }

    #[test]
    fn recovery_ignores_cache_owner_clear_and_rejects_unknown_residency_atomically() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut replacement = DcCkfRankReplacement::new();
        replacement.push_event(stored(worker, 1, &[7])).unwrap();
        let before_lineage = replacement.source_lineage.clone();
        let before_candidates = replacement.canonical_candidates.clone();

        let mut cache_owner_clear = cleared(worker.worker_id, worker.dp_rank, 2);
        cache_owner_clear.residency_domain =
            WireResidencyDomain::explicit(ResidencyDomain::CacheOwner);
        replacement.push_event(cache_owner_clear).unwrap();
        assert_eq!(replacement.source_lineage, before_lineage);
        assert_eq!(replacement.canonical_candidates, before_candidates);

        let mut invalid = stored(worker, 3, &[8]);
        invalid.residency_domain = WireResidencyDomain::Unknown("future_domain".into());
        assert_eq!(
            replacement.push_event(invalid),
            Err(KvCacheEventError::UnsupportedResidencyDomain)
        );
        assert_eq!(replacement.source_lineage, before_lineage);
        assert_eq!(replacement.canonical_candidates, before_candidates);
    }

    #[test]
    fn capacity_failure_keeps_successful_blocks_and_exact_filter_state_consistent() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        let hashes: Vec<_> = (1..=32).collect();

        let outcome = state.apply_event(stored(worker, 1, &hashes));
        assert!(matches!(
            outcome.first_error(),
            Some(KvCacheEventError::CapacityExhausted)
        ));
        let lineage = state.source_lineage.get(&worker).unwrap();
        assert_eq!(lineage.len(), hashes.len());
        assert_eq!(lineage.len(), state.canonical_owners.len());
        assert!(state.resident_count < hashes.len());
        assert!(
            lineage
                .values()
                .all(|canonical| state.canonical_owners[canonical].owners == 1)
        );
        assert!(
            state
                .canonical_owners
                .iter()
                .filter(|(_, ownership)| ownership.resident)
                .map(|(&hash, _)| hash)
                .all(|hash| state.contains(hash))
        );
    }

    #[test]
    fn root_and_batched_chain_use_canonical_token_hashes_not_external_ids() {
        let worker = WorkerWithDpRank::new(1, 0);
        let tokens = [101, 202, 303];
        let canonical = canonical_chain(&tokens);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        let outcome = state.apply_event(stored(worker, 1, &tokens));

        assert!(outcome.first_error().is_none());
        for (&token, &canonical) in tokens.iter().zip(&canonical) {
            assert_ne!(external(token).0, canonical.as_u64());
            assert!(state.contains(canonical));
            assert_eq!(state.source_lineage[&worker][&external(token)], canonical);
        }
    }

    #[test]
    fn parent_addressed_chain_matches_batched_canonical_sequence() {
        let worker = WorkerWithDpRank::new(1, 0);
        let root_external = ExternalSequenceBlockHash(90_001);
        let child_external = ExternalSequenceBlockHash(90_002);
        let grandchild_external = ExternalSequenceBlockHash(90_003);
        let tokens = [11, 22, 33];
        let canonical = canonical_chain(&tokens);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored_exact(
            worker,
            1,
            None,
            None,
            &[(root_external, LocalBlockHash(tokens[0]))],
        ));
        state.apply_event(stored_exact(
            worker,
            2,
            Some(root_external),
            None,
            &[(child_external, LocalBlockHash(tokens[1]))],
        ));
        state.apply_event(stored_exact(
            worker,
            3,
            Some(child_external),
            None,
            &[(grandchild_external, LocalBlockHash(tokens[2]))],
        ));

        assert_eq!(state.source_lineage[&worker][&root_external], canonical[0]);
        assert_eq!(state.source_lineage[&worker][&child_external], canonical[1]);
        assert_eq!(
            state.source_lineage[&worker][&grandchild_external],
            canonical[2]
        );
        assert!(canonical.into_iter().all(|hash| state.contains(hash)));
    }

    #[test]
    fn captured_vllm_and_sglang_ids_map_to_the_same_dynamo_root() {
        let vllm = WorkerWithDpRank::new(1, 0);
        let sglang = WorkerWithDpRank::new(2, 0);
        let canonical = canonical_root(3_727_940_790_060_946_254);
        let vllm_external = ExternalSequenceBlockHash(5_151_625_331_053_002_286);
        let sglang_external = ExternalSequenceBlockHash(16_481_235_979_017_058_908);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        for (worker, external) in [(vllm, vllm_external), (sglang, sglang_external)] {
            state.apply_event(stored_exact(
                worker,
                1,
                None,
                None,
                &[(external, LocalBlockHash(canonical.as_u64()))],
            ));
        }

        assert_ne!(vllm_external.0, canonical.as_u64());
        assert_ne!(sglang_external.0, canonical.as_u64());
        assert_eq!(state.canonical_owners[&canonical].owners, 2);
        assert!(state.is_resident(&canonical));
    }

    #[test]
    fn unknown_parent_rejects_source_event_without_partial_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        let outcome = state.apply_event(stored_exact(
            worker,
            1,
            Some(ExternalSequenceBlockHash(404)),
            None,
            &[(ExternalSequenceBlockHash(405), LocalBlockHash(7))],
        ));

        assert_eq!(
            outcome.first_error(),
            Some(&KvCacheEventError::ParentBlockNotFound)
        );
        assert_eq!(state.member_block_count(worker), 0);
        assert!(state.canonical_owners.is_empty());
        assert_eq!(state.resident_count, 0);
        assert_eq!(state.stats().aggregation().parent_not_found(), 1);
    }

    #[test]
    fn conflicting_external_remap_rejects_the_entire_batch() {
        let worker = WorkerWithDpRank::new(1, 0);
        let existing = ExternalSequenceBlockHash(7001);
        let new_external = ExternalSequenceBlockHash(7002);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        state.apply_event(stored_exact(
            worker,
            1,
            None,
            None,
            &[(existing, LocalBlockHash(11))],
        ));
        let before_lineage = state.source_lineage[&worker].clone();
        let before_owners = state.canonical_owners.clone();

        let outcome = state.apply_event(stored_exact(
            worker,
            2,
            None,
            None,
            &[
                (existing, LocalBlockHash(99)),
                (new_external, LocalBlockHash(22)),
            ],
        ));

        assert_eq!(
            outcome.first_error(),
            Some(&KvCacheEventError::InvalidBlockSequence)
        );
        assert_eq!(state.source_lineage[&worker], before_lineage);
        assert_eq!(state.canonical_owners, before_owners);
        assert!(!state.source_lineage[&worker].contains_key(&new_external));
    }

    #[test]
    fn different_external_ids_share_one_canonical_owner_domain() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let first_external = ExternalSequenceBlockHash(80_001);
        let second_external = ExternalSequenceBlockHash(90_001);
        let canonical = canonical_root(42);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored_exact(
            first,
            1,
            None,
            None,
            &[(first_external, LocalBlockHash(42))],
        ));
        state.apply_event(stored_exact(
            second,
            1,
            None,
            None,
            &[(second_external, LocalBlockHash(42))],
        ));
        assert_eq!(state.canonical_owners[&canonical].owners, 2);

        state.apply_event(removed_exact(first, 2, &[first_external]));
        assert_eq!(state.canonical_owners[&canonical].owners, 1);
        assert!(state.is_resident(&canonical));
        state.apply_event(removed_exact(second, 2, &[second_external]));
        assert!(!state.canonical_owners.contains_key(&canonical));
        assert!(!state.is_resident(&canonical));
    }

    #[test]
    fn clear_removes_all_same_source_aliases_of_one_canonical_hash() {
        let worker = WorkerWithDpRank::new(1, 0);
        let first_external = ExternalSequenceBlockHash(81_001);
        let second_external = ExternalSequenceBlockHash(81_002);
        let canonical = canonical_root(42);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        for (event_id, external) in [(1, first_external), (2, second_external)] {
            state.apply_event(stored_exact(
                worker,
                event_id,
                None,
                None,
                &[(external, LocalBlockHash(42))],
            ));
        }
        assert_eq!(state.canonical_owners[&canonical].owners, 2);

        let outcome = state.apply_event(cleared(worker.worker_id, worker.dp_rank, 3));

        assert!(outcome.first_error().is_none());
        assert!(state.source_lineage.is_empty());
        assert!(state.canonical_owners.is_empty());
        assert_eq!(state.resident_count, 0);
    }

    #[test]
    fn omitted_owner_keeps_lineage_and_later_owner_retries_admission() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        let omitted_token = (1..10_000)
            .find(|&token| {
                matches!(
                    state
                        .apply_event(stored(first, token, &[token]))
                        .first_error(),
                    Some(KvCacheEventError::CapacityExhausted)
                )
            })
            .expect("tiny filter must produce a deterministic omission");
        let omitted = canonical_root(omitted_token);
        assert!(!state.is_resident(&omitted));
        assert_eq!(state.canonical_owners[&omitted].owners, 1);
        let duplicate = state.apply_event(stored(first, 10_000, &[omitted_token]));
        assert!(duplicate.first_error().is_none());
        assert_eq!(state.canonical_owners[&omitted].owners, 1);
        assert!(!state.is_resident(&omitted));

        let freed = *state
            .canonical_owners
            .iter()
            .find(|(_, ownership)| ownership.resident)
            .map(|(hash, _)| hash)
            .unwrap();
        state.apply_event(removed(first, 10_001, &[freed.as_u64()]));
        assert!(!state.is_resident(&freed));

        let second_external = ExternalSequenceBlockHash(omitted_token ^ 0x55AA_00FF);
        let admitted = state.apply_event(stored_exact(
            second,
            1,
            None,
            None,
            &[(second_external, LocalBlockHash(omitted_token))],
        ));
        assert!(admitted.first_error().is_none());
        assert!(state.is_resident(&omitted));
        assert_eq!(state.canonical_owners[&omitted].owners, 2);

        state.apply_event(removed_exact(second, 2, &[second_external]));
        assert_eq!(state.canonical_owners[&omitted].owners, 1);
        assert!(state.is_resident(&omitted));
        state.apply_event(removed(first, 10_002, &[omitted_token]));
        assert!(!state.canonical_owners.contains_key(&omitted));
        assert!(!state.is_resident(&omitted));
    }

    #[test]
    fn repeated_store_readmits_an_omitted_block_once_its_home_bucket_frees() {
        let filler = WorkerWithDpRank::new(1, 0);
        let claimant = WorkerWithDpRank::new(2, 0);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        let omitted_token = (1..10_000)
            .find(|&token| {
                matches!(
                    state
                        .apply_event(stored(filler, token, &[token]))
                        .first_error(),
                    Some(KvCacheEventError::CapacityExhausted)
                )
            })
            .expect("tiny filter must produce a deterministic omission");
        let omitted = canonical_root(omitted_token);
        state.apply_event(stored(claimant, 1, &[omitted_token]));
        assert_eq!(state.canonical_owners[&omitted].owners, 2);
        assert!(!state.is_resident(&omitted));

        // Free the filter without touching the claimant's lineage.
        state.apply_event(cleared(filler.worker_id, filler.dp_rank, 2));
        assert_eq!(state.canonical_owners[&omitted].owners, 1);
        assert!(!state.is_resident(&omitted));

        // The claimant is the only owner, so a repeat is the sole remaining retry path.
        let repeat = state.apply_event(stored(claimant, 3, &[omitted_token]));

        assert!(repeat.first_error().is_none());
        assert_eq!(state.canonical_owners[&omitted].owners, 1);
        assert!(state.is_resident(&omitted));
        assert!(state.contains(omitted));
        assert_eq!(state.member_block_count(claimant), 1);
    }

    #[test]
    fn repeated_store_of_a_saturated_block_reports_no_new_capacity_omission() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        let omitted_token = (1..10_000)
            .find(|&token| {
                matches!(
                    state
                        .apply_event(stored(worker, token, &[token]))
                        .first_error(),
                    Some(KvCacheEventError::CapacityExhausted)
                )
            })
            .expect("tiny filter must produce a deterministic omission");
        let failures_after_first_store = state.stats().aggregation().capacity_failures();

        let repeat = state.apply_event(stored(worker, 10_000, &[omitted_token]));

        assert!(repeat.first_error().is_none());
        assert_eq!(
            state.stats().aggregation().capacity_failures(),
            failures_after_first_store
        );
        assert!(!state.is_resident(&canonical_root(omitted_token)));
    }

    #[test]
    fn omitted_parent_still_canonicalizes_child_and_removes_by_external_id() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        let omitted_token = (1..10_000)
            .find(|&token| {
                matches!(
                    state
                        .apply_event(stored(worker, token, &[token]))
                        .first_error(),
                    Some(KvCacheEventError::CapacityExhausted)
                )
            })
            .expect("tiny filter must produce a deterministic omission");
        let parent = canonical_root(omitted_token);
        let child_token = 20_001;
        let child = parent.child(LocalBlockHash(child_token));

        let outcome = state.apply_event(stored_with_parent(
            worker,
            10_001,
            omitted_token,
            &[child_token],
        ));

        assert!(!matches!(
            outcome.first_error(),
            Some(KvCacheEventError::ParentBlockNotFound)
        ));
        assert_eq!(state.source_lineage[&worker][&external(child_token)], child);
        state.apply_event(removed(worker, 10_002, &[omitted_token]));
        assert!(!state.source_lineage[&worker].contains_key(&external(omitted_token)));
        assert!(state.source_lineage[&worker].contains_key(&external(child_token)));
    }

    #[test]
    fn recovery_rebuild_preserves_query_state_and_live_external_removal() {
        let worker = WorkerWithDpRank::new(1, 0);
        let root_external = ExternalSequenceBlockHash(5001);
        let child_external = ExternalSequenceBlockHash(5002);
        let root_store = KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: root_external,
                tokens_hash: LocalBlockHash(31),
                mm_extra_info: None,
            }],
        };
        let child_store = KvCacheStoreData {
            parent_hash: Some(root_external),
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: child_external,
                tokens_hash: LocalBlockHash(32),
                mm_extra_info: None,
            }],
        };
        let canonical = canonical_chain(&[31, 32]);
        let mut replacement = DcCkfRankReplacement::new();
        replacement.push_store(&root_store).unwrap();
        replacement.push_store(&child_store).unwrap();
        let mut live = DcCkfState::new(CkfConfig::new(32)).unwrap();
        live.apply_event(stored_exact(
            worker,
            1,
            None,
            None,
            &[(root_external, LocalBlockHash(31))],
        ));
        live.apply_event(stored_exact(
            worker,
            2,
            Some(root_external),
            None,
            &[(child_external, LocalBlockHash(32))],
        ));
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.replace_rank(worker, replacement).unwrap();
        for &hash in &canonical {
            assert_eq!(live.contains(hash), state.contains(hash));
            assert!(state.contains(hash));
        }

        state.apply_event(removed_exact(worker, 3, &[child_external]));
        assert!(state.is_resident(&canonical[0]));
        assert!(!state.is_resident(&canonical[1]));
        assert!(!state.source_lineage[&worker].contains_key(&child_external));
    }

    #[test]
    fn recovery_rejects_child_before_parent_and_nonzero_orphan_position() {
        let parent = ExternalSequenceBlockHash(6001);
        let child_store = KvCacheStoreData {
            parent_hash: Some(parent),
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(6002),
                tokens_hash: LocalBlockHash(2),
                mm_extra_info: None,
            }],
        };
        let orphan_positional = KvCacheStoreData {
            parent_hash: None,
            start_position: Some(7),
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(6003),
                tokens_hash: LocalBlockHash(3),
                mm_extra_info: None,
            }],
        };
        let mut replacement = DcCkfRankReplacement::new();

        assert_eq!(
            replacement.push_store(&child_store),
            Err(KvCacheEventError::ParentBlockNotFound)
        );
        assert_eq!(
            replacement.push_store(&orphan_positional),
            Err(KvCacheEventError::InvalidBlockSequence)
        );
        assert!(replacement.source_lineage.is_empty());
        assert!(replacement.canonical_candidates.is_empty());
    }

    #[test]
    fn lora_salted_block_removal_uses_external_lineage_only() {
        let worker = WorkerWithDpRank::new(1, 0);
        let local = compute_block_hash_for_seq(
            &(0..8).collect::<Vec<u32>>(),
            4,
            BlockHashOptions {
                lora_name: Some("tenant-adapter"),
                ..Default::default()
            },
        );
        assert_eq!(local.len(), 2);
        let root = CanonicalSequenceBlockHash::root(local[0]);
        let child = root.child(local[1]);
        let root_external = ExternalSequenceBlockHash(0x1111_2222);
        let child_external = ExternalSequenceBlockHash(0x3333_4444);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored_exact(
            worker,
            1,
            None,
            None,
            &[(root_external, local[0]), (child_external, local[1])],
        ));
        assert!(state.is_resident(&root));
        assert!(state.is_resident(&child));

        state.apply_event(removed_exact(worker, 2, &[child_external]));
        assert!(state.is_resident(&root));
        assert!(!state.is_resident(&child));
        assert!(!state.source_lineage[&worker].contains_key(&child_external));
    }

    #[test]
    fn publication_window_suppresses_store_remove_net_reversion() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 2;
        let mut state = DcCkfState::new(config).unwrap();

        let first = state.apply_event(stored(worker, 1, &[7]));
        assert!(first.publication().is_none());
        let second = state.apply_event(removed(worker, 2, &[7]));
        assert!(second.publication().is_none());
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
    }

    #[test]
    fn valid_sibling_before_conflicting_remap_commits_nothing() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(64)).unwrap();
        assert!(
            state
                .apply_event(stored(worker, 1, &[7]))
                .first_error()
                .is_none()
        );
        state.barrier_snapshot().unwrap();
        let lineage_before = state.source_lineage.clone();
        let owners_before = state.canonical_owners.clone();

        // A valid new sibling precedes a remap of block 7's external id onto a different chain.
        let conflict = stored_exact(
            worker,
            2,
            None,
            None,
            &[
                (external(8), LocalBlockHash(8)),
                (external(7), LocalBlockHash(9)),
            ],
        );
        let outcome = state.apply_event(conflict);

        assert_eq!(
            outcome.first_error(),
            Some(&KvCacheEventError::InvalidBlockSequence)
        );
        assert_eq!(state.source_lineage, lineage_before);
        assert_eq!(state.canonical_owners, owners_before);
        assert!(!state.is_resident(&canonical_root(8)));
        assert!(state.drain_publication().is_none());
    }

    #[test]
    fn replacement_failure_after_valid_prefix_leaves_state_untouched() {
        let existing = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(64)).unwrap();
        assert!(
            state
                .apply_event(stored(existing, 1, &[7]))
                .first_error()
                .is_none()
        );
        state.barrier_snapshot().unwrap();
        let lineage_before = state.source_lineage.clone();
        let owners_before = state.canonical_owners.clone();

        let mut replacements = HashMap::new();
        replacements.insert(WorkerWithDpRank::new(2, 0), replacement(&[11]));
        replacements.insert(WorkerWithDpRank::new(3, 0), replacement(&[13]));
        state.fail_replacement_install_after = Some(1);

        let error = state.replace_ranks(replacements).unwrap_err();
        assert_eq!(error, KvCacheEventError::AllocationFailed);
        assert_eq!(state.source_lineage, lineage_before);
        assert_eq!(state.canonical_owners, owners_before);
        assert!(state.is_resident(&canonical_root(7)));
        assert!(!state.is_resident(&canonical_root(11)));
        assert!(!state.is_resident(&canonical_root(13)));
        assert!(state.drain_publication().is_none());
    }

    #[test]
    fn store_reserve_failure_commits_nothing() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(64)).unwrap();
        assert!(
            state
                .apply_event(stored(worker, 1, &[7]))
                .first_error()
                .is_none()
        );
        state.barrier_snapshot().unwrap();
        let lineage_before = state.source_lineage.clone();
        let owners_before = state.canonical_owners.clone();
        state.fail_next_store_reserve = true;

        let outcome = state.apply_event(stored(worker, 2, &[8, 9]));

        assert_eq!(
            outcome.first_error(),
            Some(&KvCacheEventError::AllocationFailed)
        );
        assert_eq!(state.source_lineage, lineage_before);
        assert_eq!(state.canonical_owners, owners_before);
        assert!(!state.is_resident(&canonical_root(8)));
        assert!(state.drain_publication().is_none());
    }

    #[test]
    fn saturated_replacement_commits_lineage_and_omits_only_residency() {
        let worker = WorkerWithDpRank::new(1, 0);
        let original = canonical_root(1);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        state.apply_event(stored(worker, 1, &[1]));
        let hashes: Vec<_> = (100..200).collect();

        state.replace_rank(worker, replacement(&hashes)).unwrap();

        assert_eq!(state.member_block_count(worker), hashes.len());
        assert_eq!(state.canonical_owners.len(), hashes.len());
        assert!(state.resident_count < hashes.len());
        assert!(!state.contains(original));
        assert!(state.stats().aggregation().capacity_failures() > 0);
    }

    #[test]
    fn replacing_one_shared_owner_preserves_the_other_without_an_intermediate_reset() {
        let replaced = WorkerWithDpRank::new(1, 0);
        let survivor = WorkerWithDpRank::new(2, 0);
        let shared = canonical_root(77);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        state.apply_event(stored(replaced, 1, &[77]));
        state.apply_event(stored(survivor, 1, &[77]));
        let (_, before) = state.barrier_snapshot().unwrap();

        let publication = state
            .replace_rank(replaced, DcCkfRankReplacement::new())
            .unwrap();

        assert_eq!(state.member_block_count(replaced), 0);
        assert_eq!(state.member_block_count(survivor), 1);
        assert_eq!(state.stats().aggregation().unique_block_count(), 1);
        assert!(state.contains(shared));
        assert!(
            publication.is_none(),
            "shared ownership needs no physical CKF change"
        );
        let (_, after) = state.barrier_snapshot().unwrap();
        assert_eq!(before.as_ref(), after.as_ref());
    }

    #[test]
    fn replacement_diff_includes_the_pending_publication_window() {
        let worker = WorkerWithDpRank::new(1, 0);
        let hash = canonical_root(7);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();
        let (_, base) = state.barrier_snapshot().unwrap();

        assert!(
            state
                .apply_event(stored(worker, 1, &[7]))
                .publication()
                .is_none()
        );
        let publication = state
            .replace_rank(worker, replacement(&[7]))
            .unwrap()
            .expect("replacement must publish the pending state");
        let mut reconstructed = base.to_vec();
        for image in publication.images() {
            reconstructed[image.bucket()] = image.value();
        }
        let (_, current) = state.barrier_snapshot().unwrap();

        assert_eq!(reconstructed, current.as_ref());
        assert!(state.contains(hash));
    }

    #[test]
    fn replacement_suppresses_a_pending_change_that_returns_to_published_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();

        assert!(
            state
                .apply_event(stored(worker, 1, &[7]))
                .publication()
                .is_none()
        );
        assert!(
            state
                .replace_rank(worker, DcCkfRankReplacement::new())
                .unwrap()
                .is_none()
        );
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
    }

    #[test]
    fn batched_rank_replacement_rebuilds_the_pool_once_transactionally() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let retained = WorkerWithDpRank::new(3, 0);
        let mut state = DcCkfState::new(CkfConfig::new(64)).unwrap();
        state.apply_event(stored(first, 1, &[1]));
        state.apply_event(stored(second, 1, &[2]));
        state.apply_event(stored(retained, 1, &[3]));

        let replacements = [
            (first, replacement(&[10, 11])),
            (second, replacement(&[20])),
        ]
        .into_iter()
        .collect();
        state.replace_ranks(replacements).unwrap();

        assert!(!state.contains(canonical_root(1)));
        assert!(!state.contains(canonical_root(2)));
        assert!(state.contains(canonical_root(3)));
        assert!(state.contains(canonical_root(10)));
        assert!(state.contains(canonical_chain(&[10, 11])[1]));
        assert!(state.contains(canonical_root(20)));
        assert_eq!(state.member_block_count(first), 2);
        assert_eq!(state.member_block_count(second), 1);
        assert_eq!(state.member_block_count(retained), 1);
    }

    #[test]
    fn barrier_snapshot_and_absolute_batches_reconstruct_producer_bytes() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let (_, base) = state.barrier_snapshot().unwrap();
        let publication = state
            .apply_event(stored(worker, 1, &[1, 2, 3]))
            .into_publication()
            .unwrap();
        let mut reconstructed = base.to_vec();
        for image in publication.images() {
            reconstructed[image.bucket()] = image.value();
        }
        let (_, current) = state.barrier_snapshot().unwrap();

        assert_eq!(reconstructed, current.as_ref());
    }

    #[test]
    fn barrier_snapshot_allocation_failure_preserves_dirty_tail() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();
        assert!(
            state
                .apply_event(stored(worker, 1, &[1, 2, 3]))
                .publication()
                .is_none()
        );

        state.fail_next_snapshot_allocation();
        assert!(matches!(
            state.barrier_snapshot(),
            Err(CkfBuildError::AllocationFailed)
        ));
        assert!(state.has_pending_publication());

        let (tail, _) = state.barrier_snapshot().unwrap();
        assert!(tail.is_some_and(|batch| !batch.images().is_empty()));
        assert!(!state.has_pending_publication());
    }

    #[test]
    fn replay_churn_requires_logical_and_membership_not_byte_parity() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut direct = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut replay = DcCkfState::new(CkfConfig::new(32)).unwrap();
        direct.apply_event(stored(worker, 1, &[1, 2, 3]));

        replay.apply_event(stored(worker, 1, &[1, 2, 3]));
        replay.apply_event(removed(worker, 2, &[2]));
        replay.apply_event(stored_with_parent(worker, 3, 1, &[2]));

        assert_eq!(direct.source_lineage, replay.source_lineage);
        assert_eq!(direct.canonical_owners, replay.canonical_owners);
        for hash in canonical_chain(&[1, 2, 3]) {
            assert!(direct.contains(hash));
            assert!(replay.contains(hash));
        }
    }

    #[test]
    fn representation_collision_remains_present_until_both_owners_are_removed() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(64)).unwrap();
        let mut seen = HashMap::new();
        let (first, second) = (1u64..)
            .find_map(|hash| {
                let probe = state.addressing.prepare(hash);
                let representation = (probe.fingerprint, probe.bucket_a, probe.bucket_b);
                seen.insert(representation, hash)
                    .filter(|previous| *previous != hash)
                    .map(|previous| (previous, hash))
            })
            .unwrap();

        state.apply_event(stored(worker, 1, &[first]));
        state.apply_event(stored(worker, 2, &[second]));
        state.apply_event(removed(worker, 3, &[first]));
        assert!(state.contains(canonical_root(second)));
        state.apply_event(removed(worker, 4, &[second]));
        assert!(!state.contains(canonical_root(second)));
    }
}
