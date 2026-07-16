// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact, lane-free CKF aggregation for one model in one data center.
//!
//! Event replay is logically idempotent: applying an ordered history through the
//! same watermark converges to the same member ownership and DC refcounts. It is
//! not physically idempotent. Remove/reinsert churn and reconstruction may choose
//! another valid cuckoo layout because relocation depends on occupancy and RNG.
//! A snapshot therefore establishes one physical base, and only that producer's
//! ordered absolute-image deltas may extend it byte-for-byte.

use derive_getters::Getters;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, RouterEvent, StorageTier,
    WorkerId, WorkerWithDpRank,
};

use super::addressing::CkfAddressing;
use super::bucket::{CuckooBucketStore, OwnedPackedCkfLane, PackedBucket};
use super::mutator::{CuckooInsertionScratch, CuckooMutator, lane_rng_seed};
use super::{CkfBuildError, CkfConfig, bucket_count, validate_config};

const FORMAT_VERSION: u16 = 1;
const FINGERPRINT_BITS: u8 = 16;
const SLOTS_PER_BUCKET: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Getters)]
pub struct DcCkfFormatIdentity {
    #[getter(copy)]
    format_version: u16,
    #[getter(copy)]
    seed: u64,
    #[getter(copy)]
    bucket_count: usize,
    #[getter(copy)]
    fingerprint_bits: u8,
    #[getter(copy)]
    slots_per_bucket: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Getters)]
pub struct DcCkfBucketImage {
    #[getter(copy)]
    bucket: usize,
    #[getter(copy)]
    value: u64,
}

/// Complete physical CKF base for one producer sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcCkfSnapshot {
    sequence: u64,
    format: DcCkfFormatIdentity,
    buckets: Box<[u64]>,
}

impl DcCkfSnapshot {
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn format(&self) -> DcCkfFormatIdentity {
        self.format
    }

    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }
}

/// Ordered absolute-image update extending one [`DcCkfSnapshot`] layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcCkfDelta {
    base_sequence: u64,
    sequence: u64,
    reset: bool,
    images: Vec<DcCkfBucketImage>,
}

impl DcCkfDelta {
    pub fn base_sequence(&self) -> u64 {
        self.base_sequence
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn reset(&self) -> bool {
        self.reset
    }

    pub fn images(&self) -> &[DcCkfBucketImage] {
        &self.images
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Getters)]
#[non_exhaustive]
pub struct DcCkfStats {
    #[getter(copy)]
    aggregation: DcCkfAggregationStats,
    #[getter(copy)]
    publication: DcCkfPublicationStats,
    #[getter(copy)]
    memory: DcCkfMemoryStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Getters)]
#[non_exhaustive]
pub struct DcCkfAggregationStats {
    #[getter(copy)]
    member_count: usize,
    #[getter(copy)]
    contribution_count: usize,
    #[getter(copy)]
    unique_block_count: usize,
    #[getter(copy)]
    unknown_removals: u64,
    #[getter(copy)]
    capacity_failures: u64,
    #[getter(copy)]
    occupied_bucket_count: usize,
    #[getter(copy)]
    occupied_slot_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Getters)]
#[non_exhaustive]
pub struct DcCkfPublicationStats {
    #[getter(copy)]
    sequence: u64,
    #[getter(copy)]
    pending_events: usize,
    #[getter(copy)]
    physical_touches: u64,
    #[getter(copy)]
    distinct_touched_buckets: u64,
    #[getter(copy)]
    emitted_images: u64,
    #[getter(copy)]
    net_reverted_buckets: u64,
    #[getter(copy)]
    reset_count: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Getters)]
#[non_exhaustive]
pub struct DcCkfMemoryStats {
    #[getter(copy)]
    member_set_capacity: usize,
    #[getter(copy)]
    refcount_capacity: usize,
    #[getter(copy)]
    filter_bytes: usize,
    #[getter(copy)]
    dirty_tracking_bytes: usize,
    #[getter(copy)]
    insertion_scratch_capacity: usize,
}

#[derive(Debug)]
pub struct DcCkfEventOutcome {
    first_error: Option<KvCacheEventError>,
    delta: Option<DcCkfDelta>,
    unknown_removals: usize,
    publication_boundary: bool,
}

impl DcCkfEventOutcome {
    pub fn first_error(&self) -> Option<&KvCacheEventError> {
        self.first_error.as_ref()
    }

    pub fn delta(&self) -> Option<&DcCkfDelta> {
        self.delta.as_ref()
    }

    pub fn into_delta(self) -> Option<DcCkfDelta> {
        self.delta
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
        let mut buckets = Vec::new();
        let mut originals = Vec::new();
        buckets
            .try_reserve_exact(bucket_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        originals
            .try_reserve_exact(bucket_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        Ok(Self {
            words: words.into_boxed_slice(),
            buckets,
            originals,
        })
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
    sequence: u64,
    pending_events: usize,
    published_nonempty: bool,
}

impl PublicationWindow {
    fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        Ok(Self {
            dirty: DirtyWindow::new(bucket_count)?,
            sequence: 0,
            pending_events: 0,
            published_nonempty: false,
        })
    }
}

#[derive(Debug, Default)]
struct DcCkfTelemetry {
    unknown_removals: u64,
    capacity_failures: u64,
    physical_touches: u64,
    distinct_touched_buckets: u64,
    emitted_images: u64,
    net_reverted_buckets: u64,
    reset_count: u64,
}

/// Exact and physical CKF state for one model-local DC aggregation domain.
#[derive(Debug)]
pub struct DcCkfState {
    member_blocks: FxHashMap<WorkerWithDpRank, FxHashSet<ExternalSequenceBlockHash>>,
    dc_refcounts: FxHashMap<ExternalSequenceBlockHash, u32>,
    filter: OwnedPackedCkfLane,
    addressing: CkfAddressing,
    config: CkfConfig,
    format: DcCkfFormatIdentity,
    rng: u64,
    insertion_scratch: CuckooInsertionScratch,
    publication: PublicationWindow,
    telemetry: DcCkfTelemetry,
    remove_scratch: Vec<ExternalSequenceBlockHash>,
}

impl DcCkfState {
    pub fn new(config: CkfConfig) -> Result<Self, CkfBuildError> {
        validate_config(config)?;
        let bucket_count = bucket_count(config.expected_blocks_per_dc)?;
        let mut dc_refcounts = FxHashMap::default();
        dc_refcounts
            .try_reserve(config.expected_blocks_per_dc)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let mut remove_scratch = Vec::new();
        remove_scratch
            .try_reserve_exact(config.expected_blocks_per_dc)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        Ok(Self {
            member_blocks: FxHashMap::default(),
            dc_refcounts,
            filter: OwnedPackedCkfLane::new(bucket_count)?,
            addressing: CkfAddressing::new(bucket_count, config.seed),
            config,
            format: DcCkfFormatIdentity {
                format_version: FORMAT_VERSION,
                seed: config.seed,
                bucket_count,
                fingerprint_bits: FINGERPRINT_BITS,
                slots_per_bucket: SLOTS_PER_BUCKET,
            },
            rng: lane_rng_seed(config.seed, 0),
            insertion_scratch: CuckooInsertionScratch::new(config.max_kicks)
                .map_err(|_| CkfBuildError::AllocationFailed)?,
            publication: PublicationWindow::new(bucket_count)?,
            telemetry: DcCkfTelemetry::default(),
            remove_scratch,
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
                delta: None,
                unknown_removals: 0,
                publication_boundary: false,
            };
        }
        self.publication.pending_events = self.publication.pending_events.saturating_add(1);
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut first_error = None;
        let mut unknown_removals = 0usize;
        match event.event.data {
            KvCacheEventData::Stored(store) => {
                for block in store.blocks {
                    if let Err(error) = self.store(worker, block.block_hash) {
                        if matches!(error, KvCacheEventError::CapacityExhausted) {
                            self.telemetry.capacity_failures =
                                self.telemetry.capacity_failures.saturating_add(1);
                        }
                        retain_first_error(&mut first_error, error);
                    }
                }
            }
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
                let members: Vec<_> = self
                    .member_blocks
                    .keys()
                    .copied()
                    .filter(|member| member.worker_id == event.worker_id)
                    .collect();
                for member in members {
                    if let Err(error) = self.remove_member(member) {
                        retain_first_error(&mut first_error, error);
                    }
                }
            }
        }
        self.telemetry.unknown_removals = self
            .telemetry
            .unknown_removals
            .saturating_add(unknown_removals as u64);
        let publication_boundary = self.publication_due();
        let delta = publication_boundary
            .then(|| self.drain_publication())
            .flatten();
        DcCkfEventOutcome {
            first_error,
            delta,
            unknown_removals,
            publication_boundary,
        }
    }

    pub fn remove_rank(
        &mut self,
        worker: WorkerWithDpRank,
    ) -> Result<Option<DcCkfDelta>, KvCacheEventError> {
        self.remove_member(worker)?;
        Ok(self.drain_publication())
    }

    pub fn remove_worker(
        &mut self,
        worker_id: WorkerId,
    ) -> Result<Option<DcCkfDelta>, KvCacheEventError> {
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
        hashes: FxHashSet<ExternalSequenceBlockHash>,
    ) -> Result<Option<DcCkfDelta>, KvCacheEventError> {
        let mut replacement =
            Self::new(self.config).map_err(|_| KvCacheEventError::CapacityExhausted)?;
        replacement.publication.sequence = self.publication.sequence;
        replacement.publication.published_nonempty = self.publication.published_nonempty;
        for (&member, blocks) in &self.member_blocks {
            if member == worker {
                continue;
            }
            for &hash in blocks {
                replacement.store(member, hash)?;
            }
        }
        for hash in hashes {
            replacement.store(worker, hash)?;
        }

        replacement.publication.dirty.clear();
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
        replacement.telemetry.capacity_failures = self.telemetry.capacity_failures;
        replacement.telemetry.physical_touches = self
            .telemetry
            .physical_touches
            .saturating_add(replacement.telemetry.physical_touches);
        replacement.telemetry.distinct_touched_buckets = self.telemetry.distinct_touched_buckets;
        replacement.telemetry.emitted_images = self.telemetry.emitted_images;
        replacement.telemetry.net_reverted_buckets = self.telemetry.net_reverted_buckets;
        replacement.telemetry.reset_count = self.telemetry.reset_count;
        *self = replacement;
        Ok(self.drain_publication())
    }

    pub fn flush(&mut self) -> Option<DcCkfDelta> {
        self.drain_publication()
    }

    pub fn snapshot(&mut self) -> Result<(Option<DcCkfDelta>, DcCkfSnapshot), CkfBuildError> {
        let delta = self.drain_publication();
        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(self.format.bucket_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        buckets
            .extend((0..self.format.bucket_count).map(|bucket| self.filter.load_bucket(bucket).0));
        Ok((
            delta,
            DcCkfSnapshot {
                sequence: self.publication.sequence,
                format: self.format,
                buckets: buckets.into_boxed_slice(),
            },
        ))
    }

    pub fn stats(&self) -> DcCkfStats {
        let mut occupied_bucket_count = 0;
        let mut occupied_slot_count = 0;
        for bucket in 0..self.format.bucket_count {
            let value = self.filter.load_bucket(bucket);
            if value != PackedBucket::default() {
                occupied_bucket_count += 1;
            }
            occupied_slot_count += (0..SLOTS_PER_BUCKET as usize)
                .filter(|&slot| value.slot(slot) != 0)
                .count();
        }
        DcCkfStats {
            aggregation: DcCkfAggregationStats {
                member_count: self.member_blocks.len(),
                contribution_count: self.member_blocks.values().map(FxHashSet::len).sum(),
                unique_block_count: self.dc_refcounts.len(),
                unknown_removals: self.telemetry.unknown_removals,
                capacity_failures: self.telemetry.capacity_failures,
                occupied_bucket_count,
                occupied_slot_count,
            },
            publication: DcCkfPublicationStats {
                sequence: self.publication.sequence,
                pending_events: self.publication.pending_events,
                physical_touches: self.telemetry.physical_touches,
                distinct_touched_buckets: self.telemetry.distinct_touched_buckets,
                emitted_images: self.telemetry.emitted_images,
                net_reverted_buckets: self.telemetry.net_reverted_buckets,
                reset_count: self.telemetry.reset_count,
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

    pub fn contains(&self, hash: ExternalSequenceBlockHash) -> bool {
        let probe = self.addressing.prepare(hash.0);
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

    fn store(
        &mut self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
    ) -> Result<bool, KvCacheEventError> {
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
                .map_err(|_| KvCacheEventError::CapacityExhausted)?;
            None
        } else {
            self.member_blocks
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::CapacityExhausted)?;
            let mut member = FxHashSet::default();
            member
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::CapacityExhausted)?;
            Some(member)
        };

        let current = self.dc_refcounts.get(&hash).copied().unwrap_or(0);
        if current == u32::MAX {
            return Err(KvCacheEventError::IndexerInvariantViolation);
        }
        if current == 0 {
            self.dc_refcounts
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::CapacityExhausted)?;
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
                )?;
            self.dc_refcounts.insert(hash, 1);
        } else {
            *self
                .dc_refcounts
                .get_mut(&hash)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)? = current + 1;
        }
        let inserted = if let Some(mut member) = new_member {
            let inserted = member.insert(hash);
            self.member_blocks.insert(worker, member);
            inserted
        } else {
            self.member_blocks
                .get_mut(&worker)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?
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
        let current = self
            .dc_refcounts
            .get(&hash)
            .copied()
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        if current == 1 {
            let dirty = &mut self.publication.dirty;
            let physical_touches = &mut self.telemetry.physical_touches;
            CuckooMutator::new(&self.filter, &self.addressing, self.config.max_kicks)
                .remove_with_original(hash, |bucket, original| {
                    *physical_touches = physical_touches.saturating_add(1);
                    dirty.mark(bucket, original);
                })?;
        }
        let member = self
            .member_blocks
            .get_mut(&worker)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let removed = member.remove(&hash);
        let member_is_empty = member.is_empty();
        debug_assert!(removed);
        if current == 1 {
            self.dc_refcounts.remove(&hash);
        } else {
            *self
                .dc_refcounts
                .get_mut(&hash)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)? = current - 1;
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
            .map_err(|_| KvCacheEventError::CapacityExhausted)?;
        self.remove_scratch.extend(member.iter().copied());
        for index in 0..self.remove_scratch.len() {
            let hash = self.remove_scratch[index];
            self.remove(worker, hash)?;
        }
        self.remove_scratch.clear();
        self.member_blocks.remove(&worker);
        Ok(())
    }

    fn drain_publication(&mut self) -> Option<DcCkfDelta> {
        if self.publication.pending_events == 0 && self.publication.dirty.buckets.is_empty() {
            return None;
        }
        let reset = self.dc_refcounts.is_empty() && self.publication.published_nonempty;
        let distinct_touched = self.publication.dirty.buckets.len() as u64;
        let mut images = Vec::with_capacity(self.publication.dirty.buckets.len());
        if !reset {
            for (index, &bucket) in self.publication.dirty.buckets.iter().enumerate() {
                let value = self.filter.load_bucket(bucket).0;
                if value != self.publication.dirty.originals[index] {
                    images.push(DcCkfBucketImage { bucket, value });
                }
            }
        }
        self.publication.dirty.clear();
        self.telemetry.distinct_touched_buckets = self
            .telemetry
            .distinct_touched_buckets
            .saturating_add(distinct_touched);
        self.telemetry.emitted_images = self
            .telemetry
            .emitted_images
            .saturating_add(images.len() as u64);
        if reset {
            self.telemetry.reset_count = self.telemetry.reset_count.saturating_add(1);
        } else {
            self.telemetry.net_reverted_buckets = self
                .telemetry
                .net_reverted_buckets
                .saturating_add(distinct_touched.saturating_sub(images.len() as u64));
        }
        self.publication.pending_events = 0;
        self.publication.published_nonempty = !self.dc_refcounts.is_empty();
        if !reset && images.is_empty() {
            return None;
        }
        let base_sequence = self.publication.sequence;
        self.publication.sequence = self.publication.sequence.saturating_add(1);
        Some(DcCkfDelta {
            base_sequence,
            sequence: self.publication.sequence,
            reset,
            images,
        })
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

    fn stored(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
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
    fn worker_clear_spans_ranks_and_preserves_another_workers_shared_hash() {
        let first_rank = WorkerWithDpRank::new(1, 0);
        let second_rank = WorkerWithDpRank::new(1, 1);
        let other_worker = WorkerWithDpRank::new(2, 0);
        let shared = ExternalSequenceBlockHash(11);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();

        state.apply_event(stored(first_rank, 1, &[shared.0]));
        state.apply_event(stored(second_rank, 1, &[12]));
        state.apply_event(stored(other_worker, 1, &[shared.0]));
        state.apply_event(cleared(first_rank.worker_id, 99, 2));

        assert_eq!(state.member_block_count(first_rank), 0);
        assert_eq!(state.member_block_count(second_rank), 0);
        assert_eq!(state.member_block_count(other_worker), 1);
        assert_eq!(state.stats().aggregation().unique_block_count(), 1);
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
        assert!(member.iter().all(|hash| state.dc_refcounts[hash] == 1));
        assert!(member.iter().copied().all(|hash| state.contains(hash)));
    }

    #[test]
    fn publication_window_suppresses_store_remove_net_reversion() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 2;
        let mut state = DcCkfState::new(config).unwrap();

        let first = state.apply_event(stored(worker, 1, &[7]));
        assert!(first.delta().is_none());
        let second = state.apply_event(removed(worker, 2, &[7]));
        assert!(second.delta().is_none());
        assert_eq!(state.stats().publication().sequence(), 0);
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
    }

    #[test]
    fn failed_transactional_replacement_preserves_previous_rank() {
        let worker = WorkerWithDpRank::new(1, 0);
        let original = ExternalSequenceBlockHash(1);
        let mut state = DcCkfState::new(CkfConfig::new(1)).unwrap();
        state.apply_event(stored(worker, 1, &[original.0]));
        let before_counts = state.member_counts();
        let replacement = (100..200).map(ExternalSequenceBlockHash).collect();

        assert!(state.replace_rank(worker, replacement).is_err());
        assert_eq!(state.member_counts(), before_counts);
        assert!(state.contains(original));
    }

    #[test]
    fn replacement_diff_includes_the_pending_publication_window() {
        let worker = WorkerWithDpRank::new(1, 0);
        let hash = ExternalSequenceBlockHash(7);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();
        let (_, base) = state.snapshot().unwrap();

        assert!(
            state
                .apply_event(stored(worker, 1, &[hash.0]))
                .delta()
                .is_none()
        );
        let delta = state
            .replace_rank(worker, [hash].into_iter().collect())
            .unwrap()
            .expect("replacement must publish the pending state");
        let mut reconstructed = base.buckets().to_vec();
        for image in delta.images() {
            reconstructed[image.bucket()] = image.value();
        }
        let (_, current) = state.snapshot().unwrap();

        assert_eq!(reconstructed, current.buckets());
        assert!(state.contains(hash));
    }

    #[test]
    fn replacement_suppresses_a_pending_change_that_returns_to_published_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();

        assert!(state.apply_event(stored(worker, 1, &[7])).delta().is_none());
        assert!(
            state
                .replace_rank(worker, FxHashSet::default())
                .unwrap()
                .is_none()
        );
        assert_eq!(state.stats().publication().sequence(), 0);
        assert_eq!(state.stats().aggregation().unique_block_count(), 0);
    }

    #[test]
    fn snapshot_and_absolute_deltas_reconstruct_producer_bytes() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let (_, base) = state.snapshot().unwrap();
        let delta = state
            .apply_event(stored(worker, 1, &[1, 2, 3]))
            .into_delta()
            .unwrap();
        let mut reconstructed = base.buckets().to_vec();
        if delta.reset() {
            reconstructed.fill(0);
        }
        for image in delta.images() {
            reconstructed[image.bucket()] = image.value();
        }
        let (_, current) = state.snapshot().unwrap();

        assert_eq!(delta.base_sequence(), base.sequence());
        assert_eq!(delta.sequence(), current.sequence());
        assert_eq!(reconstructed, current.buckets());
    }

    #[test]
    fn replay_churn_requires_logical_and_membership_not_byte_parity() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut direct = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut replay = DcCkfState::new(CkfConfig::new(32)).unwrap();
        direct.apply_event(stored(worker, 1, &[1, 2, 3]));

        replay.apply_event(stored(worker, 1, &[1, 2, 3]));
        replay.apply_event(removed(worker, 2, &[2]));
        replay.apply_event(stored(worker, 3, &[2]));

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
