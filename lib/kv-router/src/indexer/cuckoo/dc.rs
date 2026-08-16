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

use std::collections::{HashMap, HashSet};

use dynamo_tokens::SequenceHash;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, RouterEvent, StorageTier,
    WorkerId, WorkerWithDpRank, chain_seq_hash,
};

use super::addressing::CkfAddressing;
use super::bucket::{CuckooBucketStore, OwnedPackedCkfLane, PackedBucket};
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
    unchainable_batches: u64,
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

    /// Stored batches dropped because this pool does not track their parent block.
    ///
    /// Expected while a source is live-only, since no tree dump seeds the chain. Against a source
    /// that does recover, a rising count means prefix coverage is silently short: the blocks are
    /// resident in the engine but absent from the published filter.
    pub const fn unchainable_batches(&self) -> u64 {
        self.unchainable_batches
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
    member_set_capacity: usize,
    refcount_capacity: usize,
    filter_bytes: usize,
    dirty_tracking_bytes: usize,
    insertion_scratch_capacity: usize,
}

impl DcCkfMemoryStats {
    pub const fn member_set_capacity(&self) -> usize {
        self.member_set_capacity
    }

    pub const fn refcount_capacity(&self) -> usize {
        self.refcount_capacity
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
    images: Vec<GlobalCkfBucketImage>,
    pending_events: usize,
}

impl PublicationWindow {
    fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        Ok(Self {
            dirty: DirtyWindow::new(bucket_count)?,
            images: Vec::new(),
            pending_events: 0,
        })
    }
}

#[derive(Debug, Default)]
struct DcCkfTelemetry {
    unknown_removals: u64,
    unchainable_batches: u64,
    /// Latches once an unchainable batch has been reported, so a live-only source cannot turn
    /// every event into a log line. The counter above still tracks the rate the log cannot.
    unchainable_batch_reported: bool,
    capacity_failures: u64,
    physical_touches: u64,
    distinct_touched_buckets: u64,
    emitted_images: u64,
    net_reverted_buckets: u64,
}

/// One block as a caller supplies it, carrying both identities the pool needs.
///
/// These are separate values and neither substitutes for the other. `external` is the engine's
/// own block hash: the only identity a `Removed` event carries, so it is what ownership and
/// refcounts are keyed by. `sequence` is Dynamo's rolling chain over the block's token hash, and
/// it is the only thing the cuckoo table can be addressed by, because that is what a prefix query
/// probes with (see [`GlobalCkfIndexer::find_prefix_matches`]).
///
/// [`GlobalCkfIndexer::find_prefix_matches`]: super::GlobalCkfIndexer::find_prefix_matches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DcBlockKey {
    pub external: ExternalSequenceBlockHash,
    pub sequence: SequenceHash,
}

impl DcBlockKey {
    pub const fn new(external: ExternalSequenceBlockHash, sequence: SequenceHash) -> Self {
        Self { external, sequence }
    }
}

/// What the pool tracks per distinct block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DcBlockEntry {
    refcount: u32,
    /// Retained because removal arrives keyed by the engine's hash, and clearing the block's
    /// fingerprint requires the address it was inserted under.
    sequence: SequenceHash,
}

/// Exact and physical CKF state for one DC-local indexer pool.
#[derive(Debug)]
pub struct DcCkfState {
    member_blocks: FxHashMap<WorkerWithDpRank, FxHashSet<ExternalSequenceBlockHash>>,
    dc_refcounts: FxHashMap<ExternalSequenceBlockHash, DcBlockEntry>,
    filter: OwnedPackedCkfLane,
    addressing: CkfAddressing,
    config: CkfConfig,
    format: DcCkfFormatIdentity,
    rng: u64,
    insertion_scratch: CuckooInsertionScratch,
    publication: PublicationWindow,
    telemetry: DcCkfTelemetry,
    remove_scratch: Vec<ExternalSequenceBlockHash>,
    #[cfg(test)]
    fail_next_snapshot_allocation: bool,
}

impl DcCkfState {
    pub fn new(config: CkfConfig) -> Result<Self, CkfBuildError> {
        validate_config(config)?;
        let bucket_count = bucket_count(config.expected_blocks_per_dc)?;
        Ok(Self {
            member_blocks: FxHashMap::default(),
            dc_refcounts: FxHashMap::default(),
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
            #[cfg(test)]
            fail_next_snapshot_allocation: false,
        })
    }

    pub fn format(&self) -> DcCkfFormatIdentity {
        self.format
    }

    pub fn apply_event(&mut self, event: RouterEvent) -> DcCkfEventOutcome {
        if event.storage_tier != StorageTier::Device
            && !matches!(event.event.data, KvCacheEventData::Cleared)
        {
            return DcCkfEventOutcome {
                first_error: None,
                publication: None,
                unknown_removals: 0,
                publication_boundary: false,
            };
        }
        self.publication.pending_events = self.publication.pending_events.saturating_add(1);
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut first_error = None;
        let mut unknown_removals = 0usize;
        match event.event.data {
            KvCacheEventData::Stored(store) => match self.batch_root(store.parent_hash) {
                Some(root) => self.store_chain(worker, root, store.blocks, &mut first_error),
                None => {
                    // Inserting anyway would place blocks at addresses no query can reach, which
                    // buys nothing over dropping them but does spend capacity and add false
                    // positives.
                    self.telemetry.unchainable_batches =
                        self.telemetry.unchainable_batches.saturating_add(1);
                    if !self.telemetry.unchainable_batch_reported {
                        self.telemetry.unchainable_batch_reported = true;
                        tracing::warn!(
                            worker_id = worker.worker_id,
                            dp_rank = worker.dp_rank,
                            "dropped a stored batch whose parent block this pool does not track; \
                             its prefixes are absent from the published filter. Expected while a \
                             source is live-only, since no tree dump seeds the chain. Further \
                             occurrences on this pool are not logged."
                        );
                    }
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
        let members: Vec<_> = self
            .member_blocks
            .keys()
            .copied()
            .filter(|member| member.worker_id == worker_id)
            .collect();
        for member in members {
            self.remove_member(member)?;
        }
        Ok(self.drain_publication())
    }

    pub fn replace_rank(
        &mut self,
        worker: WorkerWithDpRank,
        keys: HashSet<DcBlockKey>,
    ) -> Result<Option<DcCkfPublicationBatch>, KvCacheEventError> {
        let mut replacements = HashMap::new();
        replacements.insert(worker, keys);
        self.replace_ranks(replacements)
    }

    /// Transactionally replace several ranks through one off-side pool rebuild.
    ///
    /// Replacement blocks carry their own sequence hashes because the caller is the only party
    /// that can derive them: a rank's authoritative block set arrives as a tree dump whose order
    /// supplies the parent chain, and that structure is gone by the time it reaches this pool.
    pub fn replace_ranks(
        &mut self,
        replacements: HashMap<WorkerWithDpRank, HashSet<DcBlockKey>>,
    ) -> Result<Option<DcCkfPublicationBatch>, KvCacheEventError> {
        let mut replacement = Self::new(self.config).map_err(|error| match error {
            CkfBuildError::AllocationFailed => KvCacheEventError::AllocationFailed,
            _ => KvCacheEventError::IndexerInvariantViolation,
        })?;
        for (&member, blocks) in &self.member_blocks {
            if replacements.contains_key(&member) {
                continue;
            }
            for &hash in blocks {
                // Retained ranks are rebuilt from this pool's own entries, so their addresses
                // come back unchanged rather than being recomputed.
                let entry = self
                    .dc_refcounts
                    .get(&hash)
                    .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
                replacement.store(member, DcBlockKey::new(hash, entry.sequence))?;
            }
        }
        for (member, keys) in replacements {
            for key in keys {
                replacement.store(member, key)?;
            }
        }

        replacement.publication.dirty.clear();
        replacement
            .publication
            .dirty
            .try_reserve(self.format.bucket_count)?;
        replacement
            .publication
            .images
            .try_reserve(self.format.bucket_count)
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
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
        replacement.telemetry.unchainable_batches = self.telemetry.unchainable_batches;
        replacement.telemetry.unchainable_batch_reported =
            self.telemetry.unchainable_batch_reported;
        replacement.telemetry.capacity_failures = self.telemetry.capacity_failures;
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
                member_count: self.member_blocks.len(),
                contribution_count: self.member_blocks.values().map(FxHashSet::len).sum(),
                unique_block_count: self.dc_refcounts.len(),
                unknown_removals: self.telemetry.unknown_removals,
                unchainable_batches: self.telemetry.unchainable_batches,
                capacity_failures: self.telemetry.capacity_failures,
                occupied_bucket_count: self.current_occupied_bucket_count(),
                occupied_slot_count: self.dc_refcounts.len(),
            },
            publication: DcCkfPublicationStats {
                pending_events: self.publication.pending_events,
                physical_touches: self.telemetry.physical_touches,
                distinct_touched_buckets: self.telemetry.distinct_touched_buckets,
                emitted_images: self.telemetry.emitted_images,
                net_reverted_buckets: self.telemetry.net_reverted_buckets,
            },
            memory: DcCkfMemoryStats {
                member_set_capacity: self.member_blocks.values().map(FxHashSet::capacity).sum(),
                refcount_capacity: self.dc_refcounts.capacity(),
                filter_bytes: self.filter.byte_len(),
                dirty_tracking_bytes: self.publication.dirty.byte_len(),
                insertion_scratch_capacity: self.insertion_scratch.capacity(),
            },
        }
    }

    pub fn member_block_count(&self, worker: WorkerWithDpRank) -> usize {
        self.member_blocks.get(&worker).map_or(0, FxHashSet::len)
    }

    pub fn member_counts(&self) -> Vec<(WorkerWithDpRank, usize)> {
        let mut counts: Vec<_> = self
            .member_blocks
            .iter()
            .map(|(&worker, blocks)| (worker, blocks.len()))
            .collect();
        counts.sort_unstable_by_key(|(worker, _)| *worker);
        counts
    }

    /// Whether the filter carries a fingerprint for a block, named by the engine's hash.
    ///
    /// The address is resolved through the block's entry rather than derived from `hash`, so an
    /// untracked block is reported absent without probing: there is nothing to probe with.
    pub fn contains(&self, hash: ExternalSequenceBlockHash) -> bool {
        let Some(entry) = self.dc_refcounts.get(&hash) else {
            return false;
        };
        let probe = self.addressing.prepare(entry.sequence);
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

    /// The sequence hash a stored batch extends.
    ///
    /// `Some(None)` roots the batch at the start of a sequence, `Some(Some(_))` extends a block
    /// this pool holds, and `None` means the parent is untracked — which makes every block in the
    /// batch unaddressable, because the chain has no value to fold them into.
    ///
    /// An untracked parent is expected while a pool is live-only (no recovery target, so no tree
    /// dump seeds the chain) and after event loss. It is not expected on a normal attach: the
    /// recovery tree dump replays parents before children.
    fn batch_root(
        &self,
        parent_hash: Option<ExternalSequenceBlockHash>,
    ) -> Option<Option<SequenceHash>> {
        match parent_hash {
            None => Some(None),
            Some(parent) => self
                .dc_refcounts
                .get(&parent)
                .map(|entry| Some(entry.sequence)),
        }
    }

    /// Store one batch, extending `root` block by block.
    fn store_chain(
        &mut self,
        worker: WorkerWithDpRank,
        root: Option<SequenceHash>,
        blocks: Vec<crate::protocols::KvCacheStoredBlockData>,
        first_error: &mut Option<KvCacheEventError>,
    ) {
        let mut parent = root;
        for block in blocks {
            // `chain_seq_hash` is the per-block step of the `compute_seq_hash_for_block` that a
            // prefix query applies to the same token hashes. Both sides have to derive it
            // identically, so neither may reimplement it.
            let sequence = chain_seq_hash(parent, block.tokens_hash);
            parent = Some(sequence);
            if let Err(error) = self.store(worker, DcBlockKey::new(block.block_hash, sequence)) {
                let exhausted = matches!(error, KvCacheEventError::CapacityExhausted);
                if exhausted {
                    self.telemetry.capacity_failures =
                        self.telemetry.capacity_failures.saturating_add(1);
                }
                retain_first_error(first_error, error);
                if exhausted {
                    // Every later block in this batch chains through the one just omitted, so a
                    // prefix query stops at the omission and cannot reach them. Storing them
                    // anyway spends capacity that is already exhausted on addresses nothing looks
                    // up. Truncating also keeps the batch's stored blocks a contiguous prefix,
                    // which is what the omission had already reduced them to.
                    break;
                }
            }
        }
    }

    fn store(
        &mut self,
        worker: WorkerWithDpRank,
        key: DcBlockKey,
    ) -> Result<bool, KvCacheEventError> {
        let hash = key.external;
        if self
            .member_blocks
            .get(&worker)
            .is_some_and(|member| member.contains(&hash))
        {
            return Ok(false);
        }
        let new_member = if self.member_blocks.contains_key(&worker) {
            self.member_blocks
                .get_mut(&worker)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::AllocationFailed)?;
            None
        } else {
            self.member_blocks
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::AllocationFailed)?;
            let mut member = FxHashSet::default();
            member
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::AllocationFailed)?;
            Some(member)
        };

        let current = self
            .dc_refcounts
            .get(&hash)
            .map_or(0, |entry| entry.refcount);
        if current == u32::MAX {
            return Err(KvCacheEventError::OwnershipDegreeOverflow);
        }
        if current == 0 {
            self.dc_refcounts
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::AllocationFailed)?;
            self.reserve_publication_scratch(self.config.max_kicks.saturating_add(1))?;
            let dirty = &mut self.publication.dirty;
            let physical_touches = &mut self.telemetry.physical_touches;
            CuckooMutator::new(&self.filter, &self.addressing, self.config.max_kicks)
                .insert_with_originals(
                    key.sequence,
                    &mut self.rng,
                    &mut self.insertion_scratch,
                    |bucket, original| {
                        *physical_touches = physical_touches.saturating_add(1);
                        dirty.mark(bucket, original);
                    },
                )?;
            // NOTE: Capacity failure returns above before any exact-state write. The omitted edge
            // is intentionally untracked, so a later Remove is an idempotent unknown removal.
            self.dc_refcounts.insert(
                hash,
                DcBlockEntry {
                    refcount: 1,
                    sequence: key.sequence,
                },
            );
        } else {
            // The first sequence hash recorded for a block wins. A second owner reaching the same
            // block by a different path would chain to a different address, and the entry has to
            // keep naming the one the fingerprint was actually inserted under or the eventual
            // removal clears the wrong bucket.
            self.dc_refcounts
                .get_mut(&hash)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?
                .refcount = current + 1;
        }
        let inserted = if let Some(mut member) = new_member {
            let inserted = member.insert(hash);
            self.member_blocks.insert(worker, member);
            inserted
        } else {
            self.member_blocks
                .get_mut(&worker)
                .expect("validated member must remain present through actor-owned commit")
                .insert(hash)
        };
        debug_assert!(inserted);
        Ok(true)
    }

    fn remove(
        &mut self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
    ) -> Result<bool, KvCacheEventError> {
        let Some(member) = self.member_blocks.get(&worker) else {
            return Ok(false);
        };
        if !member.contains(&hash) {
            return Ok(false);
        }
        // Removal names the block by the engine's hash, so the address its fingerprint occupies
        // has to come back out of the entry — it cannot be derived from `hash`.
        let entry = self
            .dc_refcounts
            .get(&hash)
            .copied()
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let current = entry.refcount;
        if current == 1 {
            self.reserve_publication_scratch(1)?;
            let dirty = &mut self.publication.dirty;
            let physical_touches = &mut self.telemetry.physical_touches;
            CuckooMutator::new(&self.filter, &self.addressing, self.config.max_kicks)
                .remove_with_original(entry.sequence, |bucket, original| {
                    *physical_touches = physical_touches.saturating_add(1);
                    dirty.mark(bucket, original);
                })?;
        }
        let member = self
            .member_blocks
            .get_mut(&worker)
            .expect("validated member must remain present through actor-owned commit");
        let removed = member.remove(&hash);
        let member_is_empty = member.is_empty();
        debug_assert!(removed);
        if current == 1 {
            self.dc_refcounts.remove(&hash);
        } else {
            self.dc_refcounts
                .get_mut(&hash)
                .expect("validated refcount must remain present through actor-owned commit")
                .refcount = current - 1;
        }
        if member_is_empty {
            self.member_blocks.remove(&worker);
        }
        Ok(true)
    }

    fn remove_member(&mut self, worker: WorkerWithDpRank) -> Result<(), KvCacheEventError> {
        self.remove_scratch.clear();
        let Some(member) = self.member_blocks.get(&worker) else {
            return Ok(());
        };
        self.remove_scratch
            .try_reserve(member.len())
            .map_err(|_| KvCacheEventError::AllocationFailed)?;
        self.remove_scratch.extend(member.iter().copied());
        for index in 0..self.remove_scratch.len() {
            let hash = self.remove_scratch[index];
            self.remove(worker, hash)?;
        }
        self.remove_scratch.clear();
        self.member_blocks.remove(&worker);
        Ok(())
    }

    fn drain_publication(&mut self) -> Option<DcCkfPublicationBatch> {
        if self.publication.pending_events == 0 && self.publication.dirty.buckets.is_empty() {
            return None;
        }
        let distinct_touched = self.publication.dirty.buckets.len() as u64;
        self.publication.images.clear();
        for (index, &bucket) in self.publication.dirty.buckets.iter().enumerate() {
            let value = self.filter.load_bucket(bucket).0;
            if value != self.publication.dirty.originals[index] {
                self.publication
                    .images
                    .push(GlobalCkfBucketImage::new(bucket, value));
            }
        }
        // NOTE: Publication intentionally transfers ownership of this image buffer. Defer buffer
        // pooling or shared payloads until the non-local transport and fanout shape are fixed and
        // profiles show that allocation or copying is material.
        let images = std::mem::take(&mut self.publication.images);
        self.publication.dirty.clear();
        self.telemetry.distinct_touched_buckets = self
            .telemetry
            .distinct_touched_buckets
            .saturating_add(distinct_touched);
        self.telemetry.emitted_images = self
            .telemetry
            .emitted_images
            .saturating_add(images.len() as u64);
        self.telemetry.net_reverted_buckets = self
            .telemetry
            .net_reverted_buckets
            .saturating_add(distinct_touched.saturating_sub(images.len() as u64));
        self.publication.pending_events = 0;
        if images.is_empty() {
            return None;
        }
        Some(DcCkfPublicationBatch { images })
    }

    fn reserve_publication_scratch(
        &mut self,
        maximum_new_touches: usize,
    ) -> Result<(), KvCacheEventError> {
        self.publication.dirty.try_reserve(maximum_new_touches)?;
        let maximum_images = self
            .publication
            .dirty
            .buckets
            .len()
            .saturating_add(maximum_new_touches);
        self.publication
            .images
            .try_reserve(maximum_images)
            .map_err(|_| KvCacheEventError::AllocationFailed)
    }

    fn current_occupied_bucket_count(&self) -> usize {
        // This full scan is diagnostic-only. Never cache it by adding work to mutation or
        // publication; Relay calls stats only behind its diagnostics/test surface.
        (0..self.format.bucket_count)
            .filter(|&bucket| self.filter.load_bucket(bucket) != PackedBucket::default())
            .count()
    }
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
        KvCacheEvent, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };

    use super::*;

    /// Replacement keys for blocks that are all sequence roots, matching what [`stored`] builds.
    fn keys(hashes: impl IntoIterator<Item = u64>) -> HashSet<DcBlockKey> {
        hashes
            .into_iter()
            .map(|hash| DcBlockKey::new(ExternalSequenceBlockHash(hash), hash))
            .collect()
    }

    fn stored(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        stored_under(worker, event_id, None, hashes)
    }

    fn stored_under(
        worker: WorkerWithDpRank,
        event_id: u64,
        parent: Option<u64>,
        hashes: &[u64],
    ) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent.map(ExternalSequenceBlockHash),
                    start_position: None,
                    blocks: hashes
                        .iter()
                        .copied()
                        .map(|hash| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(hash),
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
                    block_hashes: hashes
                        .iter()
                        .copied()
                        .map(ExternalSequenceBlockHash)
                        .collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
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
        let hash = ExternalSequenceBlockHash(11);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored(first, 1, &[hash.0]));
        state.apply_event(stored(second, 1, &[hash.0]));
        assert_eq!(state.stats().aggregation().contribution_count(), 2);
        assert_eq!(state.stats().aggregation().unique_block_count(), 1);

        state.apply_event(removed(first, 2, &[hash.0]));
        assert!(state.contains(hash));
        assert_eq!(state.stats().aggregation().unique_block_count(), 1);

        state.apply_event(removed(second, 2, &[hash.0]));
        assert!(!state.contains(hash));
        assert_eq!(state.stats().aggregation().member_count(), 0);
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
    }

    #[test]
    fn rank_clear_preserves_sibling_rank_and_another_workers_shared_hash() {
        let first_rank = WorkerWithDpRank::new(1, 0);
        let second_rank = WorkerWithDpRank::new(1, 1);
        let other_worker = WorkerWithDpRank::new(2, 0);
        let shared = ExternalSequenceBlockHash(11);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored(first_rank, 1, &[shared.0]));
        state.apply_event(stored(second_rank, 1, &[12]));
        state.apply_event(stored(other_worker, 1, &[shared.0]));
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
        let resident = ExternalSequenceBlockHash(7);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut host_store = stored(worker, 1, &[resident.0]);
        host_store.storage_tier = StorageTier::HostPinned;

        let ignored = state.apply_event(host_store);
        assert!(!ignored.publication_boundary());
        assert_eq!(state.stats().aggregation().contribution_count(), 0);

        state.apply_event(stored(worker, 2, &[resident.0]));
        let unknown = state.apply_event(removed(worker, 3, &[99]));
        assert_eq!(unknown.unknown_removals(), 1);
        assert_eq!(state.stats().aggregation().unknown_removals(), 1);
        assert_eq!(state.stats().aggregation().contribution_count(), 1);
        assert!(state.contains(resident));
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
        let member = state.member_blocks.get(&worker).unwrap();
        assert!(!member.is_empty());
        assert!(member.len() < hashes.len());
        assert_eq!(member.len(), state.dc_refcounts.len());
        assert!(
            member
                .iter()
                .all(|hash| state.dc_refcounts[hash].refcount == 1)
        );
        assert!(member.iter().copied().all(|hash| state.contains(hash)));
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
    fn failed_transactional_replacement_preserves_previous_rank() {
        let worker = WorkerWithDpRank::new(1, 0);
        let original = ExternalSequenceBlockHash(1);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        state.apply_event(stored(worker, 1, &[original.0]));
        let before_counts = state.member_counts();
        let replacement = keys(100..200);

        assert!(state.replace_rank(worker, replacement).is_err());
        assert_eq!(state.member_counts(), before_counts);
        assert!(state.contains(original));
    }

    #[test]
    fn replacing_one_shared_owner_preserves_the_other_without_an_intermediate_reset() {
        let replaced = WorkerWithDpRank::new(1, 0);
        let survivor = WorkerWithDpRank::new(2, 0);
        let shared = ExternalSequenceBlockHash(77);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        state.apply_event(stored(replaced, 1, &[shared.0]));
        state.apply_event(stored(survivor, 1, &[shared.0]));
        let (_, before) = state.barrier_snapshot().unwrap();

        let publication = state.replace_rank(replaced, HashSet::new()).unwrap();

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
    fn a_batch_extends_its_parents_chain_rather_than_restarting_it() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        state.apply_event(stored(worker, 1, &[1, 2]));

        // Same token hash, reached under a parent: it must land at the chained address, not at the
        // one it would occupy as a sequence root.
        state.apply_event(stored_under(worker, 2, Some(1), &[3]));

        // Derived through the same helper the query side's `compute_seq_hash_for_block` folds,
        // so this pins agreement between the two rather than restating the arithmetic.
        let rooted = chain_seq_hash(None, LocalBlockHash(3));
        let chained = chain_seq_hash(Some(1), LocalBlockHash(3));
        assert_eq!(
            state.dc_refcounts[&ExternalSequenceBlockHash(3)].sequence,
            chained
        );
        assert_ne!(chained, rooted);
    }

    #[test]
    fn a_batch_whose_parent_is_untracked_is_dropped_and_counted() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored_under(worker, 1, Some(999), &[1, 2]));

        // Storing at an unchained address would be worse than dropping: the blocks are
        // unreachable either way, but they would occupy capacity and answer false positives.
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
        assert_eq!(state.stats().aggregation().unchainable_batches(), 1);
        assert!(!state.contains(ExternalSequenceBlockHash(1)));
    }

    #[test]
    fn replacement_diff_includes_the_pending_publication_window() {
        let worker = WorkerWithDpRank::new(1, 0);
        let hash = ExternalSequenceBlockHash(7);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();
        let (_, base) = state.barrier_snapshot().unwrap();

        assert!(
            state
                .apply_event(stored(worker, 1, &[hash.0]))
                .publication()
                .is_none()
        );
        let publication = state
            .replace_rank(worker, keys([hash.0]))
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
                .replace_rank(worker, HashSet::new())
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

        let replacements = [(first, keys([10, 11])), (second, keys([20]))]
            .into_iter()
            .collect();
        state.replace_ranks(replacements).unwrap();

        assert!(!state.contains(ExternalSequenceBlockHash(1)));
        assert!(!state.contains(ExternalSequenceBlockHash(2)));
        assert!(state.contains(ExternalSequenceBlockHash(3)));
        assert!(state.contains(ExternalSequenceBlockHash(10)));
        assert!(state.contains(ExternalSequenceBlockHash(11)));
        assert!(state.contains(ExternalSequenceBlockHash(20)));
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
        // Re-store carries the parent it did originally, or the block chains to a different
        // address than the one the removal cleared and replay stops converging.
        replay.apply_event(stored_under(worker, 3, Some(1), &[2]));

        assert_eq!(direct.member_blocks, replay.member_blocks);
        assert_eq!(direct.dc_refcounts, replay.dc_refcounts);
        for hash in [1, 2, 3].map(ExternalSequenceBlockHash) {
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

        state.apply_event(stored(worker, 1, &[first, second]));
        state.apply_event(removed(worker, 2, &[first]));
        assert!(state.contains(ExternalSequenceBlockHash(second)));
        state.apply_event(removed(worker, 3, &[second]));
        assert!(!state.contains(ExternalSequenceBlockHash(second)));
    }
}
