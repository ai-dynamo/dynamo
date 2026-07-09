// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-local validation of the eventual Relay-to-router CKF boundary.
//!
//! Publication windows currently close after a configured number of normalized events per
//! DC lane. Event count is not a wall-clock freshness bound; a distributed publisher will also
//! need a maximum-time trigger for low-traffic lanes. Its transport envelope must carry a Relay
//! instance ID, model key, authoritative generation, state sequence, format identity, and a
//! snapshot/delta/heartbeat kind. Heartbeats do not advance state sequence, gaps require a fresh
//! complete snapshot, and recovery must validate and atomically publish a fully initialized table.
//! This module deliberately implements none of that transport, fencing, or recovery machinery.

use std::sync::atomic::AtomicU64;
#[cfg(feature = "bench")]
use std::sync::atomic::Ordering;
#[cfg(feature = "bench")]
use std::time::Instant;

use parking_lot::{Mutex, MutexGuard};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::indexer::WorkerLookupStats;
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, RouterEvent, StorageTier,
    WorkerId, WorkerWithDpRank,
};

use super::addressing::{CkfAddressing, CkfProbe};
use super::bucket::{CuckooBucketStore, OwnedPackedCkfLane, PackedBucket, TransposedCkfTable};
use super::mutator::{CuckooInsertionScratch, CuckooMutator, lane_rng_seed};
use super::{CkfBuildError, CkfConfig, DC_COUNT, bucket_count, validate_config};

const FORMAT_VERSION: u16 = 1;
const FINGERPRINT_BITS: u8 = 16;
const SLOTS_PER_BUCKET: u8 = 4;

#[derive(Debug, Clone)]
pub struct RelayLaneConfig {
    members: Vec<WorkerWithDpRank>,
    expected_contributions: usize,
}

impl RelayLaneConfig {
    pub fn new(members: Vec<WorkerWithDpRank>, expected_contributions: usize) -> Self {
        Self {
            members,
            expected_contributions,
        }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new(), 0)
    }

    pub fn members(&self) -> &[WorkerWithDpRank] {
        &self.members
    }

    pub fn expected_contributions(&self) -> usize {
        self.expected_contributions
    }
}

#[derive(Debug, Clone)]
pub struct RelayManifest {
    lanes: [RelayLaneConfig; DC_COUNT],
    worker_to_lane: FxHashMap<WorkerWithDpRank, usize>,
    worker_lane_masks: FxHashMap<WorkerId, u16>,
    active_lanes: u16,
}

impl RelayManifest {
    pub fn new(lanes: [RelayLaneConfig; DC_COUNT]) -> Result<Self, CkfBuildError> {
        let member_count = lanes.iter().try_fold(0usize, |count, lane| {
            count
                .checked_add(lane.members.len())
                .ok_or(CkfBuildError::CapacityOverflow)
        })?;
        let mut worker_to_lane = FxHashMap::default();
        worker_to_lane
            .try_reserve(member_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let mut worker_lane_masks = FxHashMap::default();
        worker_lane_masks
            .try_reserve(member_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let mut active_lanes = 0u16;

        for (lane, config) in lanes.iter().enumerate() {
            if !config.members.is_empty() {
                active_lanes |= 1u16 << lane;
            }
            for &worker in &config.members {
                if worker_to_lane.insert(worker, lane).is_some() {
                    return Err(CkfBuildError::DuplicateWorker { worker });
                }
                *worker_lane_masks.entry(worker.worker_id).or_insert(0) |= 1u16 << lane;
            }
        }

        Ok(Self {
            lanes,
            worker_to_lane,
            worker_lane_masks,
            active_lanes,
        })
    }

    pub fn one_worker_per_lane(
        workers: [WorkerWithDpRank; DC_COUNT],
        expected_contributions: usize,
    ) -> Result<Self, CkfBuildError> {
        let mut lanes = Vec::new();
        lanes
            .try_reserve_exact(DC_COUNT)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        for worker in workers {
            let mut members = Vec::new();
            members
                .try_reserve_exact(1)
                .map_err(|_| CkfBuildError::AllocationFailed)?;
            members.push(worker);
            lanes.push(RelayLaneConfig::new(members, expected_contributions));
        }
        let lanes = lanes
            .try_into()
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        Self::new(lanes)
    }

    pub fn lanes(&self) -> &[RelayLaneConfig; DC_COUNT] {
        &self.lanes
    }

    fn lane_for(&self, worker: WorkerWithDpRank) -> Option<usize> {
        self.worker_to_lane.get(&worker).copied()
    }

    fn lane_mask_for_worker(&self, worker_id: WorkerId) -> Option<u16> {
        self.worker_lane_masks.get(&worker_id).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CkfFormatIdentity {
    format_version: u16,
    seed: u64,
    bucket_count: usize,
    fingerprint_bits: u8,
    slots_per_bucket: u8,
    lane_count: u8,
}

impl CkfFormatIdentity {
    pub fn format_version(self) -> u16 {
        self.format_version
    }

    pub fn seed(self) -> u64 {
        self.seed
    }

    pub fn bucket_count(self) -> usize {
        self.bucket_count
    }

    pub fn fingerprint_bits(self) -> u8 {
        self.fingerprint_bits
    }

    pub fn slots_per_bucket(self) -> u8 {
        self.slots_per_bucket
    }

    pub fn lane_count(self) -> u8 {
        self.lane_count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CkfBucketImage {
    lane: u8,
    bucket: usize,
    value: u64,
}

impl CkfBucketImage {
    pub fn lane(self) -> u8 {
        self.lane
    }

    pub fn bucket(self) -> usize {
        self.bucket
    }

    pub fn value(self) -> u64 {
        self.value
    }
}

#[derive(Debug)]
pub struct CkfDeltaBatch {
    format: CkfFormatIdentity,
    reset_lanes: u16,
    images: Vec<CkfBucketImage>,
}

impl CkfDeltaBatch {
    pub fn format(&self) -> CkfFormatIdentity {
        self.format
    }

    pub fn reset_lanes(&self) -> u16 {
        self.reset_lanes
    }

    pub fn images(&self) -> &[CkfBucketImage] {
        &self.images
    }

    fn new(format: CkfFormatIdentity) -> Self {
        Self {
            format,
            reset_lanes: 0,
            images: Vec::new(),
        }
    }

    fn reset(&mut self, format: CkfFormatIdentity) {
        self.format = format;
        self.reset_lanes = 0;
        self.images.clear();
    }

    fn try_reserve(&mut self, additional: usize) -> Result<(), KvCacheEventError> {
        self.images
            .try_reserve(additional)
            .map_err(|_| KvCacheEventError::CapacityExhausted)
    }
}

#[derive(Debug)]
pub struct CkfEventOutcome {
    first_error: Option<KvCacheEventError>,
    batch_applied: bool,
    unknown_removals: u64,
    capacity_failures: u64,
    duplicate_warning: bool,
    #[cfg(feature = "bench")]
    bench: CkfBenchLocalTelemetry,
}

impl CkfEventOutcome {
    pub fn first_error(&self) -> Option<&KvCacheEventError> {
        self.first_error.as_ref()
    }

    pub fn batch_applied(&self) -> bool {
        self.batch_applied
    }

    pub fn into_result(self) -> Result<(), KvCacheEventError> {
        match self.first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) fn unknown_removals(&self) -> u64 {
        self.unknown_removals
    }

    pub(super) fn capacity_failures(&self) -> u64 {
        self.capacity_failures
    }

    pub(super) fn duplicate_warning(&self) -> bool {
        self.duplicate_warning
    }

    fn success(batch_applied: bool) -> Self {
        Self {
            first_error: None,
            batch_applied,
            unknown_removals: 0,
            capacity_failures: 0,
            duplicate_warning: false,
            #[cfg(feature = "bench")]
            bench: CkfBenchLocalTelemetry::default(),
        }
    }

    fn failure(error: KvCacheEventError) -> Self {
        Self {
            first_error: Some(error),
            batch_applied: false,
            unknown_removals: 0,
            capacity_failures: 0,
            duplicate_warning: false,
            #[cfg(feature = "bench")]
            bench: CkfBenchLocalTelemetry::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CkfMemorySnapshot {
    pub configured_unique_capacity: usize,
    pub configured_contribution_capacity: usize,
    pub actual_contributions: usize,
    pub member_set_capacity: usize,
    pub dc_refcount_capacity: usize,
    pub owned_filter_bytes: usize,
    pub replica_bytes: usize,
    pub dirty_tracking_bytes: usize,
    pub aggregator_occupancy_bytes: usize,
    pub replica_occupancy_bytes: usize,
    pub insertion_scratch_capacity: usize,
}

#[derive(Debug)]
struct DirtyBatchScratch {
    words: Box<[u64]>,
    buckets: Vec<usize>,
}

impl DirtyBatchScratch {
    fn new(bucket_count: usize) -> Result<Self, CkfBuildError> {
        let mut buckets = Vec::new();
        buckets
            .try_reserve_exact(bucket_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        Ok(Self {
            words: zeroed_boxed_slice(bucket_count.div_ceil(u64::BITS as usize))?,
            buckets,
        })
    }

    fn begin_window(&self) {
        debug_assert!(self.buckets.is_empty());
    }

    fn mark(&mut self, bucket: usize) {
        let word = bucket / u64::BITS as usize;
        let bit = 1u64 << (bucket % u64::BITS as usize);
        if self.words[word] & bit == 0 {
            self.words[word] |= bit;
            self.buckets.push(bucket);
        }
    }

    fn len(&self) -> usize {
        self.buckets.len()
    }

    fn bucket(&self, index: usize) -> usize {
        self.buckets[index]
    }

    fn clear(&mut self) {
        for &bucket in &self.buckets {
            let word = bucket / u64::BITS as usize;
            self.words[word] &= !(1u64 << (bucket % u64::BITS as usize));
        }
        self.buckets.clear();
    }

    fn byte_len(&self) -> usize {
        std::mem::size_of_val(self.words.as_ref())
            + self.buckets.capacity() * std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
struct RelayLaneState {
    member_blocks: FxHashMap<WorkerWithDpRank, FxHashSet<ExternalSequenceBlockHash>>,
    dc_refcounts: FxHashMap<ExternalSequenceBlockHash, u32>,
    filter: OwnedPackedCkfLane,
    rng: u64,
    insertion_scratch: CuckooInsertionScratch,
    dirty_scratch: DirtyBatchScratch,
    member_remove_scratch: Vec<ExternalSequenceBlockHash>,
    events_since_publish: usize,
    published_nonempty: bool,
    physical_touches_pending: u64,
    #[cfg(feature = "bench")]
    window_started_at: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreMutation {
    Duplicate,
    Changed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoveMutation {
    Unknown,
    Changed,
}

#[derive(Debug, Default)]
struct FinalizeImageStats {
    distinct_touched: u64,
    emitted_images: u64,
    net_reverted: u64,
}

impl RelayLaneState {
    fn new(
        lane: usize,
        lane_config: &RelayLaneConfig,
        config: CkfConfig,
        bucket_count: usize,
    ) -> Result<Self, CkfBuildError> {
        let member_count = lane_config.members.len();
        let mut member_blocks = FxHashMap::default();
        member_blocks
            .try_reserve(member_count)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let share = lane_config
            .expected_contributions
            .checked_div(member_count)
            .unwrap_or(0);
        let remainder = lane_config
            .expected_contributions
            .checked_rem(member_count)
            .unwrap_or(0);
        for (member_index, &member) in lane_config.members.iter().enumerate() {
            let mut blocks = FxHashSet::default();
            blocks
                .try_reserve(share + usize::from(member_index < remainder))
                .map_err(|_| CkfBuildError::AllocationFailed)?;
            member_blocks.insert(member, blocks);
        }

        let mut dc_refcounts = FxHashMap::default();
        dc_refcounts
            .try_reserve(config.expected_blocks_per_dc)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let mut member_remove_scratch = Vec::new();
        member_remove_scratch
            .try_reserve_exact(lane_config.expected_contributions)
            .map_err(|_| CkfBuildError::AllocationFailed)?;

        Ok(Self {
            member_blocks,
            dc_refcounts,
            filter: OwnedPackedCkfLane::new(bucket_count)?,
            rng: lane_rng_seed(config.seed, lane),
            insertion_scratch: CuckooInsertionScratch::new(config.max_kicks)
                .map_err(|_| CkfBuildError::AllocationFailed)?,
            dirty_scratch: DirtyBatchScratch::new(bucket_count)?,
            member_remove_scratch,
            events_since_publish: 0,
            published_nonempty: false,
            physical_touches_pending: 0,
            #[cfg(feature = "bench")]
            window_started_at: None,
        })
    }

    fn advance_event(&mut self) {
        if self.events_since_publish == 0 {
            self.dirty_scratch.begin_window();
            #[cfg(feature = "bench")]
            {
                self.window_started_at = Some(Instant::now());
            }
        }
        self.events_since_publish = self.events_since_publish.saturating_add(1);
    }

    fn publication_due(&self, every_n_events: usize) -> bool {
        self.events_since_publish >= every_n_events
    }

    fn has_pending_publication(&self) -> bool {
        self.events_since_publish != 0
    }

    fn store(
        &mut self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        addressing: &CkfAddressing,
        max_kicks: usize,
    ) -> Result<StoreMutation, KvCacheEventError> {
        let Some(member) = self.member_blocks.get_mut(&worker) else {
            return Err(KvCacheEventError::InvalidBlockSequence);
        };
        if member.contains(&hash) {
            return Ok(StoreMutation::Duplicate);
        }

        member
            .try_reserve(1)
            .map_err(|_| KvCacheEventError::CapacityExhausted)?;
        let current = self.dc_refcounts.get(&hash).copied().unwrap_or(0);
        if current == u32::MAX {
            return Err(KvCacheEventError::IndexerInvariantViolation);
        }
        if current == 0 {
            self.dc_refcounts
                .try_reserve(1)
                .map_err(|_| KvCacheEventError::CapacityExhausted)?;
            let mutator = CuckooMutator::new(&self.filter, addressing, max_kicks);
            let dirty = &mut self.dirty_scratch;
            let physical_touches = &mut self.physical_touches_pending;
            mutator.insert_touched(hash, &mut self.rng, &mut self.insertion_scratch, |bucket| {
                *physical_touches = physical_touches.saturating_add(1);
                dirty.mark(bucket);
            })?;
            self.dc_refcounts.insert(hash, 1);
        } else {
            let Some(refcount) = self.dc_refcounts.get_mut(&hash) else {
                return Err(KvCacheEventError::IndexerInvariantViolation);
            };
            *refcount = current + 1;
        }
        let inserted = member.insert(hash);
        debug_assert!(inserted);
        Ok(StoreMutation::Changed)
    }

    fn remove(
        &mut self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        addressing: &CkfAddressing,
        max_kicks: usize,
    ) -> Result<RemoveMutation, KvCacheEventError> {
        let Some(member) = self.member_blocks.get(&worker) else {
            return Err(KvCacheEventError::InvalidBlockSequence);
        };
        if !member.contains(&hash) {
            return Ok(RemoveMutation::Unknown);
        }
        let Some(current) = self.dc_refcounts.get(&hash).copied() else {
            return Err(KvCacheEventError::IndexerInvariantViolation);
        };
        if current == 0 {
            return Err(KvCacheEventError::IndexerInvariantViolation);
        }

        if current == 1 {
            let mutator = CuckooMutator::new(&self.filter, addressing, max_kicks);
            let dirty = &mut self.dirty_scratch;
            let physical_touches = &mut self.physical_touches_pending;
            mutator.remove_touched(hash, |bucket| {
                *physical_touches = physical_touches.saturating_add(1);
                dirty.mark(bucket);
            })?;
        }

        let Some(member) = self.member_blocks.get_mut(&worker) else {
            return Err(KvCacheEventError::IndexerInvariantViolation);
        };
        let removed = member.remove(&hash);
        debug_assert!(removed);
        if current == 1 {
            self.dc_refcounts.remove(&hash);
        } else {
            let Some(refcount) = self.dc_refcounts.get_mut(&hash) else {
                return Err(KvCacheEventError::IndexerInvariantViolation);
            };
            *refcount = current - 1;
        }
        Ok(RemoveMutation::Changed)
    }

    fn remove_member(
        &mut self,
        worker: WorkerWithDpRank,
        addressing: &CkfAddressing,
        max_kicks: usize,
        first_error: &mut Option<KvCacheEventError>,
    ) {
        self.member_remove_scratch.clear();
        let Some(member) = self.member_blocks.get(&worker) else {
            return;
        };
        if self
            .member_remove_scratch
            .try_reserve(member.len())
            .is_err()
        {
            retain_first_error(first_error, KvCacheEventError::CapacityExhausted);
            return;
        }
        self.member_remove_scratch.extend(member.iter().copied());

        for index in 0..self.member_remove_scratch.len() {
            let hash = self.member_remove_scratch[index];
            if let Err(error) = self.remove(worker, hash, addressing, max_kicks) {
                retain_first_error(first_error, error);
            }
        }
        self.member_remove_scratch.clear();
    }

    fn finalize_images(
        &mut self,
        lane: usize,
        replica: &TransposedCkfReplica,
        batch: &mut CkfDeltaBatch,
    ) -> FinalizeImageStats {
        let mut stats = FinalizeImageStats {
            distinct_touched: self.dirty_scratch.len() as u64,
            ..FinalizeImageStats::default()
        };
        for index in 0..self.dirty_scratch.len() {
            let bucket = self.dirty_scratch.bucket(index);
            let value = self.filter.load_bucket(bucket);
            if replica.table.load_image(bucket, lane) == value {
                stats.net_reverted += 1;
            } else {
                batch.images.push(CkfBucketImage {
                    lane: lane as u8,
                    bucket,
                    value: value.0,
                });
                stats.emitted_images += 1;
            }
        }
        self.dirty_scratch.clear();
        stats
    }

    fn reset_filter(&mut self) {
        self.filter.clear();
        self.dirty_scratch.clear();
    }

    fn finish_publication(&mut self) {
        self.events_since_publish = 0;
        self.published_nonempty = !self.dc_refcounts.is_empty();
    }

    fn contribution_count(&self) -> usize {
        self.member_blocks.values().map(FxHashSet::len).sum()
    }
}

#[derive(Debug)]
pub struct CkfRelayAggregator {
    manifest: RelayManifest,
    lanes: [Option<Mutex<RelayLaneState>>; DC_COUNT],
    addressing: CkfAddressing,
    config: CkfConfig,
    bucket_count: usize,
    format: CkfFormatIdentity,
}

impl CkfRelayAggregator {
    fn new(manifest: RelayManifest, config: CkfConfig) -> Result<Self, CkfBuildError> {
        validate_config(config)?;
        let bucket_count = bucket_count(config.expected_blocks_per_dc)?;
        for (lane, lane_config) in manifest.lanes.iter().enumerate() {
            if lane_config.members.is_empty() {
                if lane_config.expected_contributions != 0 {
                    return Err(CkfBuildError::InvalidEmptyLaneContributionCapacity {
                        lane,
                        value: lane_config.expected_contributions,
                    });
                }
            } else if lane_config.expected_contributions < config.expected_blocks_per_dc {
                return Err(CkfBuildError::InvalidContributionCapacity {
                    lane,
                    value: lane_config.expected_contributions,
                    minimum: config.expected_blocks_per_dc,
                });
            }
        }

        let mut lane_states = Vec::new();
        lane_states
            .try_reserve_exact(DC_COUNT)
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        for (lane, lane_config) in manifest.lanes.iter().enumerate() {
            let state = if lane_config.members.is_empty() {
                None
            } else {
                Some(Mutex::new(RelayLaneState::new(
                    lane,
                    lane_config,
                    config,
                    bucket_count,
                )?))
            };
            lane_states.push(state);
        }
        let lanes = lane_states
            .try_into()
            .map_err(|_| CkfBuildError::AllocationFailed)?;
        let format = CkfFormatIdentity {
            format_version: FORMAT_VERSION,
            seed: config.seed,
            bucket_count,
            fingerprint_bits: FINGERPRINT_BITS,
            slots_per_bucket: SLOTS_PER_BUCKET,
            lane_count: DC_COUNT as u8,
        };

        Ok(Self {
            manifest,
            lanes,
            addressing: CkfAddressing::new(bucket_count, config.seed),
            config,
            bucket_count,
            format,
        })
    }

    fn lock_lanes(&self, mask: u16) -> [Option<MutexGuard<'_, RelayLaneState>>; DC_COUNT] {
        let mut guards = std::array::from_fn(|_| None);
        for (lane, slot) in guards.iter_mut().enumerate() {
            if mask & (1u16 << lane) == 0 {
                continue;
            }
            *slot = self.lanes[lane].as_ref().map(Mutex::lock);
        }
        guards
    }

    fn lifecycle_batch_capacity(&self, mask: u16) -> Result<usize, KvCacheEventError> {
        let lane_count = mask.count_ones() as usize;
        self.bucket_count
            .checked_mul(lane_count)
            .ok_or(KvCacheEventError::CapacityExhausted)
    }

    fn stats_for_worker_ids(&self, worker_ids: &FxHashSet<WorkerId>) -> WorkerLookupStats {
        let mut counts = Vec::new();
        for lane in 0..DC_COUNT {
            let Some(state) = self.lanes[lane].as_ref() else {
                continue;
            };
            let state = state.lock();
            counts.extend(state.member_blocks.iter().filter_map(|(&worker, blocks)| {
                worker_ids
                    .contains(&worker.worker_id)
                    .then_some((worker, blocks.len()))
            }));
        }
        WorkerLookupStats::from_worker_block_counts(counts)
    }

    fn memory_snapshot(&self, replica: &TransposedCkfReplica) -> CkfMemorySnapshot {
        let mut snapshot = CkfMemorySnapshot {
            configured_unique_capacity: self
                .config
                .expected_blocks_per_dc
                .saturating_mul(self.manifest.active_lanes.count_ones() as usize),
            configured_contribution_capacity: self
                .manifest
                .lanes
                .iter()
                .map(|lane| lane.expected_contributions)
                .fold(0usize, usize::saturating_add),
            replica_bytes: replica.byte_len(),
            replica_occupancy_bytes: replica.occupancy_byte_len(),
            ..CkfMemorySnapshot::default()
        };
        for state in self.lanes.iter().flatten() {
            let state = state.lock();
            snapshot.actual_contributions = snapshot
                .actual_contributions
                .saturating_add(state.contribution_count());
            snapshot.member_set_capacity = snapshot.member_set_capacity.saturating_add(
                state
                    .member_blocks
                    .values()
                    .map(FxHashSet::capacity)
                    .fold(0usize, usize::saturating_add),
            );
            snapshot.dc_refcount_capacity = snapshot
                .dc_refcount_capacity
                .saturating_add(state.dc_refcounts.capacity());
            snapshot.owned_filter_bytes = snapshot
                .owned_filter_bytes
                .saturating_add(state.filter.byte_len());
            snapshot.dirty_tracking_bytes = snapshot
                .dirty_tracking_bytes
                .saturating_add(state.dirty_scratch.byte_len());
            snapshot.insertion_scratch_capacity = snapshot
                .insertion_scratch_capacity
                .saturating_add(state.insertion_scratch.capacity());
        }
        snapshot
    }
}

#[derive(Debug)]
pub struct TransposedCkfReplica {
    pub(super) table: TransposedCkfTable<DC_COUNT>,
    pub(super) addressing: CkfAddressing,
    pub(super) config: CkfConfig,
    format: CkfFormatIdentity,
}

impl TransposedCkfReplica {
    fn new(format: CkfFormatIdentity, config: CkfConfig) -> Result<Self, CkfBuildError> {
        Ok(Self {
            table: TransposedCkfTable::new(format.bucket_count)?,
            addressing: CkfAddressing::new(format.bucket_count, format.seed),
            config,
            format,
        })
    }

    pub fn format(&self) -> CkfFormatIdentity {
        self.format
    }

    /// Return one prefix depth for every configured DC lane.
    pub fn find_prefix_depths(
        &self,
        sequence: &[crate::protocols::LocalBlockHash],
    ) -> [u32; DC_COUNT] {
        if sequence.is_empty() {
            return [0; DC_COUNT];
        }
        let probes = self.prepared_probes(sequence);
        #[cfg(not(feature = "metrics"))]
        let depths = super::search::find_prefix_depths::<DC_COUNT>(
            probes.len(),
            u16::MAX,
            self.config.search.verification_window,
            |position| self.table.prefetch_probe(probes[position]),
            |position| self.table.probe(probes[position]),
        );
        #[cfg(feature = "metrics")]
        let depths = super::search::find_prefix_depths_with_stats::<DC_COUNT>(
            probes.len(),
            u16::MAX,
            self.config.search.verification_window,
            |position| self.table.prefetch_probe(probes[position]),
            |position| self.table.probe(probes[position]),
        )
        .depths;
        depths
    }

    pub(super) fn prepared_probes(
        &self,
        sequence: &[crate::protocols::LocalBlockHash],
    ) -> Vec<CkfProbe> {
        crate::protocols::compute_seq_hash_for_block(sequence)
            .into_iter()
            .map(|hash| self.addressing.prepare(hash))
            .collect()
    }

    fn apply(&self, batch: &CkfDeltaBatch) {
        debug_assert_eq!(batch.format, self.format);
        for lane in 0..DC_COUNT {
            if batch.reset_lanes & (1u16 << lane) == 0 {
                continue;
            }
            self.table.clear_lane(lane);
        }
        for image in &batch.images {
            let lane = usize::from(image.lane);
            let value = PackedBucket(image.value);
            self.table.store_image(image.bucket, lane, value);
        }
    }

    fn byte_len(&self) -> usize {
        self.table.bucket_count() * DC_COUNT * std::mem::size_of::<AtomicU64>()
    }

    fn occupancy_byte_len(&self) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct RouterLocalCkfPipeline {
    aggregator: CkfRelayAggregator,
    replica: TransposedCkfReplica,
    #[cfg(feature = "bench")]
    bench_counters: CkfBenchCounters,
}

impl RouterLocalCkfPipeline {
    pub fn new(manifest: RelayManifest, config: CkfConfig) -> Result<Self, CkfBuildError> {
        let aggregator = CkfRelayAggregator::new(manifest, config)?;
        let replica = TransposedCkfReplica::new(aggregator.format, config)?;
        Ok(Self {
            aggregator,
            replica,
            #[cfg(feature = "bench")]
            bench_counters: CkfBenchCounters::default(),
        })
    }

    pub fn replica(&self) -> &TransposedCkfReplica {
        &self.replica
    }

    pub fn new_batch(&self) -> CkfDeltaBatch {
        CkfDeltaBatch::new(self.aggregator.format)
    }

    pub fn apply_event(&self, event: RouterEvent, batch: &mut CkfDeltaBatch) -> CkfEventOutcome {
        batch.reset(self.aggregator.format);
        if event.storage_tier != StorageTier::Device
            && !matches!(event.event.data, KvCacheEventData::Cleared)
        {
            return CkfEventOutcome::success(false);
        }

        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mask = match &event.event.data {
            KvCacheEventData::Stored(_) | KvCacheEventData::Removed(_) => {
                let Some(lane) = self.aggregator.manifest.lane_for(worker) else {
                    tracing::warn!(
                        worker_id = worker.worker_id,
                        dp_rank = worker.dp_rank,
                        event_id = event.event.event_id,
                        "CKF event references an unknown worker/rank"
                    );
                    return CkfEventOutcome::failure(KvCacheEventError::InvalidBlockSequence);
                };
                1u16 << lane
            }
            KvCacheEventData::Cleared => {
                let Some(mask) = self
                    .aggregator
                    .manifest
                    .lane_mask_for_worker(event.worker_id)
                else {
                    tracing::warn!(
                        worker_id = event.worker_id,
                        event_id = event.event.event_id,
                        "CKF clear references an unknown worker"
                    );
                    return CkfEventOutcome::failure(KvCacheEventError::InvalidBlockSequence);
                };
                mask
            }
        };

        let Some(capacity) = self
            .aggregator
            .bucket_count
            .checked_mul(mask.count_ones() as usize)
        else {
            return CkfEventOutcome::failure(KvCacheEventError::CapacityExhausted);
        };
        if let Err(error) = batch.try_reserve(capacity) {
            return CkfEventOutcome::failure(error);
        }

        #[cfg(feature = "bench")]
        let event_started = Instant::now();
        #[cfg(feature = "bench")]
        let lock_started = Instant::now();
        let event_id = event.event.event_id;
        let mut outcome = CkfEventOutcome::success(false);
        {
            let mut guards = self.aggregator.lock_lanes(mask);
            #[cfg(feature = "bench")]
            let lock_acquired = Instant::now();
            for state in guards.iter_mut().flatten() {
                state.advance_event();
            }
            #[cfg(feature = "bench")]
            let aggregation_started = Instant::now();

            match event.event.data {
                KvCacheEventData::Stored(store) => {
                    let lane = mask.trailing_zeros() as usize;
                    if let Some(state) = guards[lane].as_mut() {
                        let mut entirely_duplicate = !store.blocks.is_empty();
                        for block in store.blocks {
                            match state.store(
                                worker,
                                block.block_hash,
                                &self.aggregator.addressing,
                                self.aggregator.config.max_kicks,
                            ) {
                                Ok(StoreMutation::Duplicate) => {}
                                Ok(StoreMutation::Changed) => entirely_duplicate = false,
                                Err(error) => {
                                    entirely_duplicate = false;
                                    if matches!(error, KvCacheEventError::CapacityExhausted) {
                                        outcome.capacity_failures += 1;
                                    }
                                    retain_first_error(&mut outcome.first_error, error);
                                }
                            }
                        }
                        outcome.duplicate_warning = entirely_duplicate;
                    } else {
                        retain_first_error(
                            &mut outcome.first_error,
                            KvCacheEventError::IndexerInvariantViolation,
                        );
                    }
                }
                KvCacheEventData::Removed(remove) => {
                    let lane = mask.trailing_zeros() as usize;
                    if let Some(state) = guards[lane].as_mut() {
                        for hash in remove.block_hashes {
                            match state.remove(
                                worker,
                                hash,
                                &self.aggregator.addressing,
                                self.aggregator.config.max_kicks,
                            ) {
                                Ok(RemoveMutation::Unknown) => outcome.unknown_removals += 1,
                                Ok(RemoveMutation::Changed) => {}
                                Err(error) => retain_first_error(&mut outcome.first_error, error),
                            }
                        }
                    } else {
                        retain_first_error(
                            &mut outcome.first_error,
                            KvCacheEventError::IndexerInvariantViolation,
                        );
                    }
                }
                KvCacheEventData::Cleared => {
                    for (lane, state) in guards.iter_mut().enumerate() {
                        let Some(state) = state.as_mut() else {
                            continue;
                        };
                        for member in self.aggregator.manifest.lanes[lane]
                            .members
                            .iter()
                            .copied()
                            .filter(|member| member.worker_id == event.worker_id)
                        {
                            state.remove_member(
                                member,
                                &self.aggregator.addressing,
                                self.aggregator.config.max_kicks,
                                &mut outcome.first_error,
                            );
                        }
                    }
                }
            }

            #[cfg(feature = "bench")]
            let aggregation_ns = elapsed_ns(aggregation_started);

            let due_mask =
                due_lane_mask(&guards, mask, self.aggregator.config.publish_every_n_events);
            if due_mask != 0 {
                #[cfg(feature = "bench")]
                let finalization_started = Instant::now();
                let publication = publish_locked_lanes(&mut guards, due_mask, &self.replica, batch);
                #[cfg(feature = "bench")]
                let finalization_ns = elapsed_ns(finalization_started);
                let has_replica_update = batch.reset_lanes != 0 || !batch.images.is_empty();
                #[cfg(feature = "bench")]
                let apply_started = Instant::now();
                if has_replica_update {
                    self.replica.apply(batch);
                }
                #[cfg(feature = "bench")]
                outcome.bench.record_publication(
                    &publication,
                    has_replica_update
                        .then(|| elapsed_ns(apply_started))
                        .unwrap_or(0),
                    !has_replica_update,
                    finalization_ns,
                );
                #[cfg(not(feature = "bench"))]
                let _ = publication;
                outcome.batch_applied = has_replica_update;
            }
            #[cfg(feature = "bench")]
            outcome.bench.record_event(
                u64::from(mask.count_ones()),
                elapsed_ns(event_started),
                lock_acquired
                    .saturating_duration_since(lock_started)
                    .as_nanos()
                    .min(u64::MAX as u128) as u64,
                elapsed_ns(lock_acquired),
                aggregation_ns,
            );
        }

        if outcome.unknown_removals != 0 {
            tracing::warn!(
                worker_id = event.worker_id,
                event_id,
                unknown_removals = outcome.unknown_removals,
                "Ignoring CKF removals for blocks not owned by the worker"
            );
        }
        outcome
    }

    pub(super) fn remove_worker(
        &self,
        worker_id: WorkerId,
        batch: &mut CkfDeltaBatch,
    ) -> CkfEventOutcome {
        let Some(mask) = self.aggregator.manifest.lane_mask_for_worker(worker_id) else {
            batch.reset(self.aggregator.format);
            return CkfEventOutcome::success(false);
        };
        self.remove_members(mask, batch, |member| member.worker_id == worker_id)
    }

    pub(super) fn remove_worker_rank(
        &self,
        worker: WorkerWithDpRank,
        batch: &mut CkfDeltaBatch,
    ) -> CkfEventOutcome {
        let Some(lane) = self.aggregator.manifest.lane_for(worker) else {
            batch.reset(self.aggregator.format);
            return CkfEventOutcome::success(false);
        };
        self.remove_members(1u16 << lane, batch, |member| member == worker)
    }

    fn remove_members(
        &self,
        mask: u16,
        batch: &mut CkfDeltaBatch,
        mut selected: impl FnMut(WorkerWithDpRank) -> bool,
    ) -> CkfEventOutcome {
        batch.reset(self.aggregator.format);
        let Ok(capacity) = self.aggregator.lifecycle_batch_capacity(mask) else {
            return CkfEventOutcome::failure(KvCacheEventError::CapacityExhausted);
        };
        if let Err(error) = batch.try_reserve(capacity) {
            return CkfEventOutcome::failure(error);
        }

        let mut outcome = CkfEventOutcome::success(false);
        {
            let mut guards = self.aggregator.lock_lanes(mask);
            for state in guards.iter_mut().flatten() {
                state.advance_event();
            }
            for (lane, state) in guards.iter_mut().enumerate() {
                let Some(state) = state.as_mut() else {
                    continue;
                };
                for member in self.aggregator.manifest.lanes[lane]
                    .members
                    .iter()
                    .copied()
                    .filter(|&member| selected(member))
                {
                    state.remove_member(
                        member,
                        &self.aggregator.addressing,
                        self.aggregator.config.max_kicks,
                        &mut outcome.first_error,
                    );
                }
            }
            #[cfg(feature = "bench")]
            let finalization_started = Instant::now();
            let publication = publish_locked_lanes(&mut guards, mask, &self.replica, batch);
            #[cfg(feature = "bench")]
            let finalization_ns = elapsed_ns(finalization_started);
            let has_replica_update = batch.reset_lanes != 0 || !batch.images.is_empty();
            #[cfg(feature = "bench")]
            let apply_started = Instant::now();
            if has_replica_update {
                self.replica.apply(batch);
            }
            #[cfg(feature = "bench")]
            outcome.bench.record_publication(
                &publication,
                has_replica_update
                    .then(|| elapsed_ns(apply_started))
                    .unwrap_or(0),
                !has_replica_update,
                finalization_ns,
            );
            #[cfg(not(feature = "bench"))]
            let _ = publication;
            outcome.batch_applied = has_replica_update;
        }
        outcome
    }

    #[cfg(any(test, not(feature = "bench")))]
    pub(super) fn flush_pending(
        &self,
        batch: &mut CkfDeltaBatch,
    ) -> Result<bool, KvCacheEventError> {
        Ok(self.flush_pending_core(batch)?.applied)
    }

    #[cfg(feature = "bench")]
    pub(super) fn flush_pending_with_telemetry(
        &self,
        batch: &mut CkfDeltaBatch,
        telemetry: &mut CkfBenchLocalTelemetry,
    ) -> Result<bool, KvCacheEventError> {
        let result = self.flush_pending_core(batch)?;
        if let Some(publication) = result.publication.as_ref() {
            telemetry.record_publication(
                publication,
                result.replica_apply_ns,
                !result.applied,
                result.finalization_ns,
            );
        }
        Ok(result.applied)
    }

    fn flush_pending_core(
        &self,
        batch: &mut CkfDeltaBatch,
    ) -> Result<FlushPendingResult, KvCacheEventError> {
        batch.reset(self.aggregator.format);
        let mask = self.aggregator.manifest.active_lanes;
        let capacity = self.aggregator.lifecycle_batch_capacity(mask)?;
        batch.try_reserve(capacity)?;

        let mut result = FlushPendingResult::default();
        {
            let mut guards = self.aggregator.lock_lanes(mask);
            let pending_mask = pending_lane_mask(&guards, mask);
            if pending_mask != 0 {
                #[cfg(feature = "bench")]
                let finalization_started = Instant::now();
                let publication =
                    publish_locked_lanes(&mut guards, pending_mask, &self.replica, batch);
                #[cfg(feature = "bench")]
                let finalization_ns = elapsed_ns(finalization_started);
                let has_replica_update = batch.reset_lanes != 0 || !batch.images.is_empty();
                #[cfg(feature = "bench")]
                let apply_started = Instant::now();
                if has_replica_update {
                    self.replica.apply(batch);
                }
                #[cfg(feature = "bench")]
                {
                    result.replica_apply_ns = has_replica_update
                        .then(|| elapsed_ns(apply_started))
                        .unwrap_or(0);
                    result.finalization_ns = finalization_ns;
                }
                result.applied = has_replica_update;
                result.publication = Some(publication);
            }
        }
        Ok(result)
    }

    pub(super) fn worker_stats(&self, worker_ids: &FxHashSet<WorkerId>) -> WorkerLookupStats {
        self.aggregator.stats_for_worker_ids(worker_ids)
    }

    pub fn memory_snapshot(&self) -> CkfMemorySnapshot {
        self.aggregator.memory_snapshot(&self.replica)
    }

    #[cfg(feature = "bench")]
    pub(super) fn timing_report(&self) -> String {
        self.bench_counters.report(
            self.aggregator.config.publish_every_n_events,
            self.aggregator.bucket_count,
            self.memory_snapshot(),
        )
    }

    #[cfg(feature = "bench")]
    pub(super) fn merge_bench_telemetry(&self, telemetry: &mut CkfBenchLocalTelemetry) {
        self.bench_counters.merge(telemetry);
    }

    #[cfg(test)]
    pub(super) fn member_contains(
        &self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
    ) -> bool {
        let Some(lane) = self.aggregator.manifest.lane_for(worker) else {
            return false;
        };
        self.aggregator.lanes[lane]
            .as_ref()
            .expect("active lane")
            .lock()
            .member_blocks
            .get(&worker)
            .is_some_and(|blocks| blocks.contains(&hash))
    }

    #[cfg(test)]
    pub(super) fn clear_owned_representation(
        &self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
    ) {
        let lane = self
            .aggregator
            .manifest
            .lane_for(worker)
            .expect("test worker is manifested");
        let state = self.aggregator.lanes[lane]
            .as_ref()
            .expect("test lane is active")
            .lock();
        let probe = self.aggregator.addressing.prepare(hash.0);
        state
            .filter
            .store_bucket(probe.bucket_a, PackedBucket::default());
        state
            .filter
            .store_bucket(probe.bucket_b, PackedBucket::default());
    }
}

fn due_lane_mask(
    guards: &[Option<MutexGuard<'_, RelayLaneState>>; DC_COUNT],
    mask: u16,
    every_n_events: usize,
) -> u16 {
    guards.iter().enumerate().fold(0u16, |due, (lane, state)| {
        if mask & (1u16 << lane) != 0
            && state
                .as_ref()
                .is_some_and(|state| state.publication_due(every_n_events))
        {
            due | (1u16 << lane)
        } else {
            due
        }
    })
}

fn pending_lane_mask(
    guards: &[Option<MutexGuard<'_, RelayLaneState>>; DC_COUNT],
    mask: u16,
) -> u16 {
    guards
        .iter()
        .enumerate()
        .fold(0u16, |pending, (lane, state)| {
            if mask & (1u16 << lane) != 0
                && state
                    .as_ref()
                    .is_some_and(|state| state.has_pending_publication())
            {
                pending | (1u16 << lane)
            } else {
                pending
            }
        })
}

#[derive(Debug, Default)]
struct PublicationStats {
    lane_windows: u64,
    lane_events: u64,
    physical_touches: u64,
    distinct_touched: u64,
    final_images: u64,
    net_reverted: u64,
    reset_lanes: u64,
    #[cfg(feature = "bench")]
    staleness_ns: u64,
    #[cfg(feature = "bench")]
    maximum_staleness_ns: u64,
}

#[derive(Debug, Default)]
struct FlushPendingResult {
    applied: bool,
    publication: Option<PublicationStats>,
    #[cfg(feature = "bench")]
    replica_apply_ns: u64,
    #[cfg(feature = "bench")]
    finalization_ns: u64,
}

fn publish_locked_lanes(
    guards: &mut [Option<MutexGuard<'_, RelayLaneState>>; DC_COUNT],
    mask: u16,
    replica: &TransposedCkfReplica,
    batch: &mut CkfDeltaBatch,
) -> PublicationStats {
    let mut stats = PublicationStats::default();
    for (lane, state) in guards.iter_mut().enumerate() {
        let Some(state) = state.as_mut() else {
            continue;
        };
        if mask & (1u16 << lane) == 0 {
            continue;
        }
        let images_before = batch.images.len();
        let distinct_touched = state.dirty_scratch.len() as u64;
        if state.dc_refcounts.is_empty() && state.published_nonempty {
            state.reset_filter();
            batch.reset_lanes |= 1u16 << lane;
            stats.reset_lanes += 1;
        } else if state.dc_refcounts.is_empty() {
            stats.net_reverted = stats.net_reverted.saturating_add(distinct_touched);
            state.dirty_scratch.clear();
        } else {
            let finalized = state.finalize_images(lane, replica, batch);
            debug_assert_eq!(finalized.distinct_touched, distinct_touched);
            stats.net_reverted = stats.net_reverted.saturating_add(finalized.net_reverted);
            debug_assert_eq!(
                finalized.emitted_images,
                (batch.images.len() - images_before) as u64
            );
        }
        stats.lane_windows += 1;
        stats.lane_events = stats
            .lane_events
            .saturating_add(state.events_since_publish as u64);
        stats.physical_touches = stats
            .physical_touches
            .saturating_add(state.physical_touches_pending);
        stats.distinct_touched = stats.distinct_touched.saturating_add(distinct_touched);
        stats.final_images = stats
            .final_images
            .saturating_add((batch.images.len() - images_before) as u64);
        #[cfg(feature = "bench")]
        if let Some(started_at) = state.window_started_at.take() {
            let staleness = started_at.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            stats.staleness_ns = stats.staleness_ns.saturating_add(staleness);
            stats.maximum_staleness_ns = stats.maximum_staleness_ns.max(staleness);
        }
        state.physical_touches_pending = 0;
        state.finish_publication();
    }
    debug_assert_eq!(batch.reset_lanes & !mask, 0);
    stats
}

#[cfg(feature = "bench")]
#[derive(Debug, Default)]
pub(super) struct CkfBenchLocalTelemetry {
    normalized_events: u64,
    affected_lane_events: u64,
    event_ns: u64,
    aggregation_ns: u64,
    lock_wait_ns: u64,
    lock_hold_ns: u64,
    publications: u64,
    unchanged_publications: u64,
    published_lane_windows: u64,
    published_lane_events: u64,
    physical_touches: u64,
    distinct_touched: u64,
    final_images: u64,
    net_reverted: u64,
    reset_lanes: u64,
    replica_apply_ns: u64,
    finalization_ns: u64,
    staleness_ns: u64,
    maximum_staleness_ns: u64,
}

#[cfg(feature = "bench")]
impl CkfBenchLocalTelemetry {
    fn record_event(
        &mut self,
        affected_lanes: u64,
        event_ns: u64,
        wait_ns: u64,
        hold_ns: u64,
        aggregation_ns: u64,
    ) {
        self.normalized_events += 1;
        self.affected_lane_events = self.affected_lane_events.saturating_add(affected_lanes);
        self.event_ns = self.event_ns.saturating_add(event_ns);
        self.aggregation_ns = self.aggregation_ns.saturating_add(aggregation_ns);
        self.lock_wait_ns = self.lock_wait_ns.saturating_add(wait_ns);
        self.lock_hold_ns = self.lock_hold_ns.saturating_add(hold_ns);
    }

    fn record_publication(
        &mut self,
        stats: &PublicationStats,
        replica_apply_ns: u64,
        unchanged: bool,
        finalization_ns: u64,
    ) {
        self.publications += 1;
        self.unchanged_publications = self
            .unchanged_publications
            .saturating_add(u64::from(unchanged));
        self.published_lane_windows = self
            .published_lane_windows
            .saturating_add(stats.lane_windows);
        self.published_lane_events = self.published_lane_events.saturating_add(stats.lane_events);
        self.physical_touches = self.physical_touches.saturating_add(stats.physical_touches);
        self.distinct_touched = self.distinct_touched.saturating_add(stats.distinct_touched);
        self.final_images = self.final_images.saturating_add(stats.final_images);
        self.net_reverted = self.net_reverted.saturating_add(stats.net_reverted);
        self.reset_lanes = self.reset_lanes.saturating_add(stats.reset_lanes);
        self.replica_apply_ns = self.replica_apply_ns.saturating_add(replica_apply_ns);
        self.finalization_ns = self.finalization_ns.saturating_add(finalization_ns);
        self.staleness_ns = self.staleness_ns.saturating_add(stats.staleness_ns);
        self.maximum_staleness_ns = self.maximum_staleness_ns.max(stats.maximum_staleness_ns);
    }

    pub(super) fn absorb_outcome(&mut self, outcome: &CkfEventOutcome) {
        self.absorb(&outcome.bench);
    }

    fn absorb(&mut self, other: &Self) {
        self.normalized_events = self
            .normalized_events
            .saturating_add(other.normalized_events);
        self.affected_lane_events = self
            .affected_lane_events
            .saturating_add(other.affected_lane_events);
        self.event_ns = self.event_ns.saturating_add(other.event_ns);
        self.aggregation_ns = self.aggregation_ns.saturating_add(other.aggregation_ns);
        self.lock_wait_ns = self.lock_wait_ns.saturating_add(other.lock_wait_ns);
        self.lock_hold_ns = self.lock_hold_ns.saturating_add(other.lock_hold_ns);
        self.publications = self.publications.saturating_add(other.publications);
        self.unchanged_publications = self
            .unchanged_publications
            .saturating_add(other.unchanged_publications);
        self.published_lane_windows = self
            .published_lane_windows
            .saturating_add(other.published_lane_windows);
        self.published_lane_events = self
            .published_lane_events
            .saturating_add(other.published_lane_events);
        self.physical_touches = self.physical_touches.saturating_add(other.physical_touches);
        self.distinct_touched = self.distinct_touched.saturating_add(other.distinct_touched);
        self.final_images = self.final_images.saturating_add(other.final_images);
        self.net_reverted = self.net_reverted.saturating_add(other.net_reverted);
        self.reset_lanes = self.reset_lanes.saturating_add(other.reset_lanes);
        self.replica_apply_ns = self.replica_apply_ns.saturating_add(other.replica_apply_ns);
        self.finalization_ns = self.finalization_ns.saturating_add(other.finalization_ns);
        self.staleness_ns = self.staleness_ns.saturating_add(other.staleness_ns);
        self.maximum_staleness_ns = self.maximum_staleness_ns.max(other.maximum_staleness_ns);
    }
}

#[cfg(feature = "bench")]
#[derive(Debug, Default)]
struct CkfBenchCounters {
    normalized_events: AtomicU64,
    affected_lane_events: AtomicU64,
    event_ns: AtomicU64,
    aggregation_ns: AtomicU64,
    lock_wait_ns: AtomicU64,
    lock_hold_ns: AtomicU64,
    publications: AtomicU64,
    unchanged_publications: AtomicU64,
    published_lane_windows: AtomicU64,
    published_lane_events: AtomicU64,
    physical_touches: AtomicU64,
    distinct_touched: AtomicU64,
    final_images: AtomicU64,
    net_reverted: AtomicU64,
    reset_lanes: AtomicU64,
    replica_apply_ns: AtomicU64,
    finalization_ns: AtomicU64,
    staleness_ns: AtomicU64,
    maximum_staleness_ns: AtomicU64,
}

#[cfg(feature = "bench")]
impl CkfBenchCounters {
    fn merge(&self, local: &mut CkfBenchLocalTelemetry) {
        self.normalized_events
            .fetch_add(local.normalized_events, Ordering::Relaxed);
        self.affected_lane_events
            .fetch_add(local.affected_lane_events, Ordering::Relaxed);
        self.event_ns.fetch_add(local.event_ns, Ordering::Relaxed);
        self.aggregation_ns
            .fetch_add(local.aggregation_ns, Ordering::Relaxed);
        self.lock_wait_ns
            .fetch_add(local.lock_wait_ns, Ordering::Relaxed);
        self.lock_hold_ns
            .fetch_add(local.lock_hold_ns, Ordering::Relaxed);
        self.publications
            .fetch_add(local.publications, Ordering::Relaxed);
        self.unchanged_publications
            .fetch_add(local.unchanged_publications, Ordering::Relaxed);
        self.published_lane_windows
            .fetch_add(local.published_lane_windows, Ordering::Relaxed);
        self.published_lane_events
            .fetch_add(local.published_lane_events, Ordering::Relaxed);
        self.physical_touches
            .fetch_add(local.physical_touches, Ordering::Relaxed);
        self.distinct_touched
            .fetch_add(local.distinct_touched, Ordering::Relaxed);
        self.final_images
            .fetch_add(local.final_images, Ordering::Relaxed);
        self.net_reverted
            .fetch_add(local.net_reverted, Ordering::Relaxed);
        self.reset_lanes
            .fetch_add(local.reset_lanes, Ordering::Relaxed);
        self.replica_apply_ns
            .fetch_add(local.replica_apply_ns, Ordering::Relaxed);
        self.finalization_ns
            .fetch_add(local.finalization_ns, Ordering::Relaxed);
        self.staleness_ns
            .fetch_add(local.staleness_ns, Ordering::Relaxed);
        self.maximum_staleness_ns
            .fetch_max(local.maximum_staleness_ns, Ordering::Relaxed);
        *local = CkfBenchLocalTelemetry::default();
    }

    fn report(
        &self,
        publish_every_n_events: usize,
        bucket_count: usize,
        memory: CkfMemorySnapshot,
    ) -> String {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let events = load(&self.normalized_events);
        let publications = load(&self.publications);
        let unchanged_publications = load(&self.unchanged_publications);
        let lane_windows = load(&self.published_lane_windows);
        let lane_events = load(&self.published_lane_events);
        let touches = load(&self.physical_touches);
        let distinct_touched = load(&self.distinct_touched);
        let images = load(&self.final_images);
        let net_reverted = load(&self.net_reverted);
        format!(
            "ckf_publish_every_n_events={publish_every_n_events} normalized_events={events} affected_lane_events={} publications={publications} unchanged_publications={unchanged_publications} published_lane_windows={lane_windows} events_per_publication={:.3} lane_events_per_lane_window={:.3} physical_touches={touches} distinct_touched={distinct_touched} final_images={images} cross_event_coalesced={} net_reverted={net_reverted} dirty_density={:.6} image_bytes={} bytes_per_event={:.3} bytes_per_publication={:.3} event_ns_per_event={:.3} aggregation_ns_per_event={:.3} finalization_ns_per_publication={:.3} lock_wait_ns_per_event={:.3} lock_hold_ns_per_event={:.3} replica_apply_ns={} staleness_ns_per_lane_window={:.3} maximum_staleness_ns={} reset_lanes={} actual_contributions={} member_set_capacity={} dc_refcount_capacity={} owned_filter_bytes={} replica_bytes={} dirty_tracking_bytes={} aggregator_occupancy_bytes={} replica_occupancy_bytes={}",
            load(&self.affected_lane_events),
            ratio(events, publications),
            ratio(lane_events, lane_windows),
            touches.saturating_sub(distinct_touched),
            if lane_windows == 0 {
                0.0
            } else {
                distinct_touched as f64 / (lane_windows as f64 * bucket_count.max(1) as f64)
            },
            images.saturating_mul(std::mem::size_of::<CkfBucketImage>() as u64),
            ratio(
                images.saturating_mul(std::mem::size_of::<CkfBucketImage>() as u64),
                events,
            ),
            ratio(
                images.saturating_mul(std::mem::size_of::<CkfBucketImage>() as u64),
                publications,
            ),
            ratio(load(&self.event_ns), events),
            ratio(load(&self.aggregation_ns), events),
            ratio(load(&self.finalization_ns), publications),
            ratio(load(&self.lock_wait_ns), events),
            ratio(load(&self.lock_hold_ns), events),
            load(&self.replica_apply_ns),
            ratio(load(&self.staleness_ns), lane_windows),
            load(&self.maximum_staleness_ns),
            load(&self.reset_lanes),
            memory.actual_contributions,
            memory.member_set_capacity,
            memory.dc_refcount_capacity,
            memory.owned_filter_bytes,
            memory.replica_bytes,
            memory.dirty_tracking_bytes,
            memory.aggregator_occupancy_bytes,
            memory.replica_occupancy_bytes,
        )
    }
}

#[cfg(feature = "bench")]
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(feature = "bench")]
fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn retain_first_error(slot: &mut Option<KvCacheEventError>, error: KvCacheEventError) {
    if slot.is_none() {
        *slot = Some(error);
    }
}

fn zeroed_boxed_slice<T: Default>(len: usize) -> Result<Box<[T]>, CkfBuildError> {
    let allocation_bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(CkfBuildError::CapacityOverflow)?;
    if allocation_bytes > isize::MAX as usize {
        return Err(CkfBuildError::CapacityOverflow);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| CkfBuildError::AllocationFailed)?;
    values.resize_with(len, T::default);
    Ok(values.into_boxed_slice())
}
