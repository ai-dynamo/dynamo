// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Indexer-domain-scoped storage and ingestion for Relay-published DC pool lanes.
//!
//! The indexer shares only immutable addressing, atomic packed buckets, and an atomic ready mask
//! with query threads. Each [`GlobalCkfLaneIngestor`] is owned by one logically serialized
//! ingestion lane and keeps its lease, validation scratch, and sequence as ordinary worker-local
//! state. Snapshot activation uses a Release ready-bit store paired with the query's single
//! Acquire ready-mask load. Deltas intentionally remain weak multi-bucket updates: there is no
//! lane seqlock, double buffer, reader retry, or publication-wide barrier. A query that captured
//! readiness before retirement may finish against a cross-bucket mixture.

use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};

use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};

use crate::identity::{DcId, IdentitySource, IndexerDomainId, PoolId};
use crate::protocols::LocalBlockHash;

use super::addressing::{CkfAddressing, CkfProbe};
use super::bucket::{PackedBucket, TransposedCkfTable};
use super::failure::{CkfFailureDisposition, CkfFailurePoint};
use super::{
    CkfBuildError, DC_COUNT, DcCkfFormatIdentity, MAX_VERIFICATION_WINDOW, PrefixSearchConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConsumerInstanceId(u64);

impl ConsumerInstanceId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerIdentity {
    pool_id: PoolId,
    producer_incarnation: u64,
    layout_generation: u64,
    format: DcCkfFormatIdentity,
}

impl ProducerIdentity {
    pub const fn new(
        pool_id: PoolId,
        producer_incarnation: u64,
        layout_generation: u64,
        format: DcCkfFormatIdentity,
    ) -> Self {
        Self {
            pool_id,
            producer_incarnation,
            layout_generation,
            format,
        }
    }

    pub const fn pool_id(self) -> PoolId {
        self.pool_id
    }

    pub const fn indexer_domain(self) -> IndexerDomainId {
        self.pool_id.indexer_domain()
    }

    pub const fn dc_id(self) -> DcId {
        self.pool_id.dc_id()
    }

    pub const fn producer_incarnation(self) -> u64 {
        self.producer_incarnation
    }

    pub const fn layout_generation(self) -> u64 {
        self.layout_generation
    }

    pub const fn format(self) -> DcCkfFormatIdentity {
        self.format
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneLease {
    consumer_instance: ConsumerInstanceId,
    physical_lane: u8,
    assignment_epoch: u64,
}

impl LaneLease {
    pub const fn new(
        consumer_instance: ConsumerInstanceId,
        physical_lane: u8,
        assignment_epoch: u64,
    ) -> Self {
        Self {
            consumer_instance,
            physical_lane,
            assignment_epoch,
        }
    }

    pub const fn consumer_instance(self) -> ConsumerInstanceId {
        self.consumer_instance
    }

    pub const fn physical_lane(self) -> u8 {
        self.physical_lane
    }

    pub const fn assignment_epoch(self) -> u64 {
        self.assignment_epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalCkfManifest {
    consumer_instance: ConsumerInstanceId,
    indexer_domain: IndexerDomainId,
    format: DcCkfFormatIdentity,
    lanes: [Option<PoolId>; DC_COUNT],
    configured_lanes: u16,
}

impl GlobalCkfManifest {
    pub fn new(
        consumer_instance: ConsumerInstanceId,
        indexer_domain: IndexerDomainId,
        format: DcCkfFormatIdentity,
        lanes: [Option<PoolId>; DC_COUNT],
    ) -> Result<Self, GlobalCkfBuildError> {
        let mut configured_lanes = 0u16;
        let mut pools = FxHashSet::default();
        pools
            .try_reserve(lanes.len())
            .map_err(|_| GlobalCkfBuildError::AllocationFailed)?;
        let mut dc_ids = FxHashSet::default();
        dc_ids
            .try_reserve(lanes.len())
            .map_err(|_| GlobalCkfBuildError::AllocationFailed)?;

        for (lane, pool_id) in lanes.iter().copied().enumerate() {
            let Some(pool_id) = pool_id else {
                continue;
            };
            if pool_id.indexer_domain() != indexer_domain {
                return Err(GlobalCkfBuildError::MixedIndexerDomain {
                    lane: lane as u8,
                    expected: indexer_domain,
                    actual: pool_id.indexer_domain(),
                });
            }
            if !pools.insert(pool_id) {
                // Within one immutable IndexerDomainId, PoolId is exactly (domain, DC), so a
                // duplicate pool is also the duplicate-DC-lane case. Keep both checks explicit
                // because the DC check remains a guard if PoolId gains another dimension later.
                return Err(GlobalCkfBuildError::DuplicatePool { pool_id });
            }
            if !dc_ids.insert(pool_id.dc_id()) {
                return Err(GlobalCkfBuildError::DuplicateDcLane {
                    dc_id: pool_id.dc_id(),
                });
            }
            configured_lanes |= 1u16 << lane;
        }
        if configured_lanes == 0 {
            return Err(GlobalCkfBuildError::NoConfiguredLanes);
        }

        // NOTE: One global indexer is exactly one logical domain. DC remains an orthogonal pool
        // dimension: lanes differ by DcId, never by endpoint identity or routing scope.
        if dc_ids.len() > 1 && indexer_domain.relies_on_defaults() {
            let defaulted_dimensions = match (
                indexer_domain.cache_semantics().source(),
                indexer_domain.routing_scope().source(),
            ) {
                (IdentitySource::DefaultDerived, IdentitySource::DefaultDerived) => {
                    "cache_semantics,routing_scope"
                }
                (IdentitySource::DefaultDerived, IdentitySource::Explicit) => "cache_semantics",
                (IdentitySource::Explicit, IdentitySource::DefaultDerived) => "routing_scope",
                (IdentitySource::Explicit, IdentitySource::Explicit) => unreachable!(),
            };
            tracing::warn!(
                %indexer_domain,
                dc_count = dc_ids.len(),
                defaulted_dimensions,
                "joining multiple DC CKF pools using default-derived identity"
            );
        }

        Ok(Self {
            consumer_instance,
            indexer_domain,
            format,
            lanes,
            configured_lanes,
        })
    }

    pub const fn consumer_instance(&self) -> ConsumerInstanceId {
        self.consumer_instance
    }

    pub const fn indexer_domain(&self) -> IndexerDomainId {
        self.indexer_domain
    }

    pub const fn format(&self) -> DcCkfFormatIdentity {
        self.format
    }

    pub const fn configured_lanes(&self) -> u16 {
        self.configured_lanes
    }

    pub fn pool_id(&self, lane: usize) -> Option<PoolId> {
        self.lanes.get(lane).copied().flatten()
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GlobalCkfBuildError {
    #[error("global CKF manifest must configure at least one lane")]
    NoConfiguredLanes,

    #[error("lane {lane} has indexer domain {actual}, expected {expected}")]
    MixedIndexerDomain {
        lane: u8,
        expected: IndexerDomainId,
        actual: IndexerDomainId,
    },

    #[error("duplicate global CKF pool {pool_id}")]
    DuplicatePool { pool_id: PoolId },

    #[error("global CKF manifest contains more than one lane for DC {dc_id}")]
    DuplicateDcLane { dc_id: DcId },

    #[error("global CKF lane {lane} is outside 0..{DC_COUNT}")]
    InvalidLane { lane: usize },

    #[error("global CKF lane {lane} is not configured")]
    UnconfiguredLane { lane: usize },

    #[error("global CKF lane {lane} already has an ingestion owner")]
    LaneAlreadyClaimed { lane: usize },

    #[error("verification_window {value} is outside 1..={MAX_VERIFICATION_WINDOW}")]
    InvalidVerificationWindow { value: usize },

    #[error("failed to allocate global CKF storage")]
    AllocationFailed,

    #[error(transparent)]
    Ckf(#[from] CkfBuildError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalCkfBucketImage {
    bucket: usize,
    value: u64,
}

impl GlobalCkfBucketImage {
    pub const fn new(bucket: usize, value: u64) -> Self {
        Self { bucket, value }
    }

    pub const fn bucket(self) -> usize {
        self.bucket
    }

    pub const fn value(self) -> u64 {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalCkfSnapshot {
    identity: ProducerIdentity,
    lease: LaneLease,
    sequence: u64,
    buckets: Box<[u64]>,
}

impl GlobalCkfSnapshot {
    pub fn new(
        identity: ProducerIdentity,
        lease: LaneLease,
        sequence: u64,
        buckets: Box<[u64]>,
    ) -> Self {
        Self {
            identity,
            lease,
            sequence,
            buckets,
        }
    }

    pub const fn identity(&self) -> ProducerIdentity {
        self.identity
    }

    pub const fn lease(&self) -> LaneLease {
        self.lease
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn buckets(&self) -> &[u64] {
        &self.buckets
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalCkfDelta {
    identity: ProducerIdentity,
    lease: LaneLease,
    base_sequence: u64,
    sequence: u64,
    images: Vec<GlobalCkfBucketImage>,
}

/// Canonical producer-to-consumer wire names. The `GlobalCkf*` aliases remain for source
/// compatibility, but there is only one serialized representation and one validation path.
pub type DcCkfBucketImage = GlobalCkfBucketImage;
pub type DcCkfSnapshot = GlobalCkfSnapshot;
pub type DcCkfDelta = GlobalCkfDelta;

impl GlobalCkfDelta {
    pub fn new(
        identity: ProducerIdentity,
        lease: LaneLease,
        base_sequence: u64,
        sequence: u64,
        images: Vec<GlobalCkfBucketImage>,
    ) -> Self {
        Self {
            identity,
            lease,
            base_sequence,
            sequence,
            images,
        }
    }

    pub const fn identity(&self) -> ProducerIdentity {
        self.identity
    }

    pub const fn lease(&self) -> LaneLease {
        self.lease
    }

    pub const fn base_sequence(&self) -> u64 {
        self.base_sequence
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn images(&self) -> &[GlobalCkfBucketImage] {
        &self.images
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsumerDrainMarker {
    lease: LaneLease,
    expected_sequence: u64,
}

impl ConsumerDrainMarker {
    pub const fn new(lease: LaneLease, expected_sequence: u64) -> Self {
        Self {
            lease,
            expected_sequence,
        }
    }

    pub const fn lease(self) -> LaneLease {
        self.lease
    }

    pub const fn expected_sequence(self) -> u64 {
        self.expected_sequence
    }
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum GlobalCkfAssignmentError {
    #[error("lease targets consumer {actual:?}, expected {expected:?}")]
    WrongConsumerInstance {
        expected: ConsumerInstanceId,
        actual: ConsumerInstanceId,
    },

    #[error("lease targets lane {actual}, expected {expected}")]
    WrongPhysicalLane { expected: u8, actual: u8 },

    #[error("producer identity does not own lane {lane}")]
    WrongLaneOwner { lane: u8 },

    #[error("producer CKF format differs from the immutable consumer format")]
    WrongFormat,

    #[error("assignment epoch {actual} must be newer than {minimum_exclusive}")]
    StaleAssignmentEpoch { minimum_exclusive: u64, actual: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalCkfLaneFault {
    WrongIdentity,
    InvalidSnapshotBucketCount {
        expected: usize,
        actual: usize,
    },
    EmptyDelta,
    BaseSequenceMismatch {
        expected: u64,
        actual: u64,
    },
    NonContiguousSequence {
        base: u64,
        sequence: u64,
    },
    SequenceExhausted,
    InvalidImageIndex {
        bucket: usize,
    },
    DuplicateImageIndex {
        bucket: usize,
    },
    SnapshotWhileReady {
        installed_sequence: u64,
        snapshot_sequence: u64,
    },
    TerminalSequenceMismatch {
        expected: u64,
        actual: Option<u64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalCkfIngestOutcome {
    SnapshotInstalled { sequence: u64 },
    DeltaApplied { sequence: u64, images: usize },
    IgnoredForeignLease,
    IgnoredStaleOrDuplicate { installed_sequence: u64 },
    DrainAcknowledged { installed_sequence: u64 },
    AwaitingSnapshot,
    LaneDeactivated { fault: GlobalCkfLaneFault },
}

#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum GlobalCkfQueryError {
    #[error("no global CKF lanes are ready")]
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalCkfLaneMatch {
    physical_lane: u8,
    pool_id: PoolId,
    prefix_depth: u32,
}

impl GlobalCkfLaneMatch {
    pub const fn physical_lane(self) -> u8 {
        self.physical_lane
    }

    pub const fn pool_id(self) -> PoolId {
        self.pool_id
    }

    pub const fn prefix_depth(self) -> u32 {
        self.prefix_depth
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalCkfQueryResult {
    captured_ready_lanes: u16,
    lanes: [Option<GlobalCkfLaneMatch>; DC_COUNT],
}

impl GlobalCkfQueryResult {
    pub const fn captured_ready_lanes(&self) -> u16 {
        self.captured_ready_lanes
    }

    pub const fn lanes(&self) -> &[Option<GlobalCkfLaneMatch>; DC_COUNT] {
        &self.lanes
    }
}

#[derive(Debug)]
struct GlobalCkfShared {
    manifest: GlobalCkfManifest,
    table: TransposedCkfTable<DC_COUNT>,
    addressing: CkfAddressing,
    search: PrefixSearchConfig,
    ready_lanes: AtomicU16,
    claimed_lanes: AtomicU16,
}

/// Endpoint-scoped CKF query storage shared by query threads and lane ingestors.
#[derive(Debug, Clone)]
pub struct GlobalCkfIndexer {
    shared: Arc<GlobalCkfShared>,
}

impl GlobalCkfIndexer {
    pub fn new(
        manifest: GlobalCkfManifest,
        search: PrefixSearchConfig,
    ) -> Result<Self, GlobalCkfBuildError> {
        if !(1..=MAX_VERIFICATION_WINDOW).contains(&search.verification_window) {
            return Err(GlobalCkfBuildError::InvalidVerificationWindow {
                value: search.verification_window,
            });
        }
        let bucket_count = manifest.format.bucket_count();
        let table = TransposedCkfTable::new(bucket_count)?;
        let addressing = CkfAddressing::new(bucket_count, manifest.format.seed());
        Ok(Self {
            shared: Arc::new(GlobalCkfShared {
                manifest,
                table,
                addressing,
                search,
                ready_lanes: AtomicU16::new(0),
                claimed_lanes: AtomicU16::new(0),
            }),
        })
    }

    pub fn manifest(&self) -> &GlobalCkfManifest {
        &self.shared.manifest
    }

    pub(super) fn validate_assignment(
        &self,
        lane: usize,
        identity: ProducerIdentity,
        lease: LaneLease,
        previous_assignment_epoch: Option<u64>,
    ) -> Result<(), GlobalCkfAssignmentError> {
        validate_assignment(
            &self.shared.manifest,
            lane,
            identity,
            lease,
            previous_assignment_epoch,
        )
    }

    pub fn ready_lanes(&self) -> u16 {
        self.shared.ready_lanes.load(Ordering::Acquire)
    }

    /// Retire one lane before its ingestion FIFO is discarded.
    ///
    /// This is used by the bounded ingestion transport when admission saturates. The Release
    /// operation makes retirement visible without involving query locks; a query that already
    /// captured the ready bit may still finish under the documented weak-read contract.
    pub(super) fn retire_lane_readiness(&self, lane: usize) {
        debug_assert!(lane < DC_COUNT);
        self.shared
            .ready_lanes
            .fetch_and(!(1u16 << lane), Ordering::Release);
    }

    pub fn claim_lane(&self, lane: usize) -> Result<GlobalCkfLaneIngestor, GlobalCkfBuildError> {
        if lane >= DC_COUNT {
            return Err(GlobalCkfBuildError::InvalidLane { lane });
        }
        if self.shared.manifest.pool_id(lane).is_none() {
            return Err(GlobalCkfBuildError::UnconfiguredLane { lane });
        }

        let lane_bit = 1u16 << lane;
        let previous = self
            .shared
            .claimed_lanes
            .fetch_or(lane_bit, Ordering::AcqRel);
        if previous & lane_bit != 0 {
            return Err(GlobalCkfBuildError::LaneAlreadyClaimed { lane });
        }

        let bucket_count = self.shared.table.bucket_count();
        let mut validation_words = Vec::new();
        if validation_words
            .try_reserve_exact(bucket_count.div_ceil(u64::BITS as usize))
            .is_err()
        {
            self.shared
                .claimed_lanes
                .fetch_and(!lane_bit, Ordering::Release);
            return Err(GlobalCkfBuildError::AllocationFailed);
        }
        validation_words.resize(bucket_count.div_ceil(u64::BITS as usize), 0);
        let mut validation_touched_words = Vec::new();
        if validation_touched_words
            .try_reserve_exact(validation_words.len())
            .is_err()
        {
            self.shared
                .claimed_lanes
                .fetch_and(!lane_bit, Ordering::Release);
            return Err(GlobalCkfBuildError::AllocationFailed);
        }

        Ok(GlobalCkfLaneIngestor {
            shared: Arc::clone(&self.shared),
            lane,
            assignment: None,
            last_assignment_epoch: None,
            installed_sequence: 0,
            ready: false,
            last_failure_disposition: None,
            validation_words: validation_words.into_boxed_slice(),
            validation_touched_words,
        })
    }

    /// Search the immutable indexer domain's currently ready DC pool lanes.
    ///
    /// Readiness is captured exactly once with Acquire ordering. A query may therefore finish
    /// after a lane is retired, and it never retries to manufacture a multi-bucket table cut.
    pub fn find_prefix_matches(
        &self,
        sequence: &[LocalBlockHash],
    ) -> Result<GlobalCkfQueryResult, GlobalCkfQueryError> {
        // This is the query's only readiness load. A lane retired immediately afterward may
        // finish under the documented weak-read contract.
        let captured_ready = self.shared.ready_lanes.load(Ordering::Acquire);
        self.find_prefix_matches_with_captured_ready(sequence, captured_ready)
    }

    fn find_prefix_matches_with_captured_ready(
        &self,
        sequence: &[LocalBlockHash],
        captured_ready: u16,
    ) -> Result<GlobalCkfQueryResult, GlobalCkfQueryError> {
        let captured_ready = captured_ready & self.shared.manifest.configured_lanes;
        if captured_ready == 0 {
            return Err(GlobalCkfQueryError::Unavailable);
        }

        let depths = if sequence.is_empty() {
            [0; DC_COUNT]
        } else {
            let probes = self.prepared_probes(sequence);
            super::search::find_prefix_depths::<DC_COUNT>(
                probes.len(),
                captured_ready,
                self.shared.search.verification_window,
                |position| self.shared.table.prefetch_probe(probes[position]),
                |position| self.shared.table.probe(probes[position]),
            )
        };

        let lanes = std::array::from_fn(|lane| {
            if captured_ready & (1u16 << lane) == 0 {
                return None;
            }
            let pool_id = self
                .shared
                .manifest
                .pool_id(lane)
                .expect("captured ready lane must have an immutable pool");
            Some(GlobalCkfLaneMatch {
                physical_lane: lane as u8,
                pool_id,
                prefix_depth: depths[lane],
            })
        });
        Ok(GlobalCkfQueryResult {
            captured_ready_lanes: captured_ready,
            lanes,
        })
    }

    fn prepared_probes(&self, sequence: &[LocalBlockHash]) -> Vec<CkfProbe> {
        crate::protocols::compute_seq_hash_for_block(sequence)
            .into_iter()
            .map(|hash| self.shared.addressing.prepare(hash))
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct LaneAssignment {
    identity: ProducerIdentity,
    lease: LaneLease,
}

/// Direct ingestion state for one physical lane.
///
/// Production dispatch must keep this value on one lane-sticky worker. Its sequence is an
/// ordinary `u64`; it is never shared with mutation or query threads.
#[derive(Debug)]
pub struct GlobalCkfLaneIngestor {
    shared: Arc<GlobalCkfShared>,
    lane: usize,
    assignment: Option<LaneAssignment>,
    last_assignment_epoch: Option<u64>,
    installed_sequence: u64,
    ready: bool,
    last_failure_disposition: Option<CkfFailureDisposition>,
    validation_words: Box<[u64]>,
    validation_touched_words: Vec<usize>,
}

impl GlobalCkfLaneIngestor {
    pub const fn lane(&self) -> usize {
        self.lane
    }

    pub const fn installed_sequence(&self) -> Option<u64> {
        if self.ready {
            Some(self.installed_sequence)
        } else {
            None
        }
    }

    pub const fn lease(&self) -> Option<LaneLease> {
        match self.assignment {
            Some(assignment) => Some(assignment.lease),
            None => None,
        }
    }

    /// Commit-domain classification for the most recent current-lane failure.
    ///
    /// Ignored stale or foreign traffic leaves this as `None`. Recovery is selected from the
    /// state whose commit became uncertain—not merely from the error's name.
    pub const fn last_failure_disposition(&self) -> Option<CkfFailureDisposition> {
        self.last_failure_disposition
    }

    pub fn assign(
        &mut self,
        identity: ProducerIdentity,
        lease: LaneLease,
    ) -> Result<(), GlobalCkfAssignmentError> {
        self.last_failure_disposition = None;
        validate_assignment(
            &self.shared.manifest,
            self.lane,
            identity,
            lease,
            self.last_assignment_epoch,
        )?;

        // Retirement is published before lease state changes. Queries that captured the old bit
        // may finish, while later queries cannot observe the lane as ready under the new lease.
        self.clear_ready();
        self.assignment = Some(LaneAssignment { identity, lease });
        self.last_assignment_epoch = Some(lease.assignment_epoch);
        self.installed_sequence = 0;
        self.clear_validation();
        Ok(())
    }

    pub fn retire(&mut self) {
        self.last_failure_disposition = None;
        self.clear_ready();
        self.assignment = None;
        self.installed_sequence = 0;
        self.clear_validation();
    }

    pub fn install_snapshot(&mut self, snapshot: &GlobalCkfSnapshot) -> GlobalCkfIngestOutcome {
        self.install_snapshot_guarded(snapshot, |ingestor, sequence| {
            ingestor.activate_snapshot(sequence);
            true
        })
    }

    pub(super) fn install_snapshot_guarded<F>(
        &mut self,
        snapshot: &GlobalCkfSnapshot,
        activate: F,
    ) -> GlobalCkfIngestOutcome
    where
        F: FnOnce(&mut Self, u64) -> bool,
    {
        self.last_failure_disposition = None;
        let Some(assignment) = self.assignment else {
            return GlobalCkfIngestOutcome::IgnoredForeignLease;
        };
        if snapshot.lease != assignment.lease {
            return GlobalCkfIngestOutcome::IgnoredForeignLease;
        }
        if snapshot.identity != assignment.identity {
            return self.deactivate_snapshot(GlobalCkfLaneFault::WrongIdentity);
        }
        if self.ready && snapshot.sequence <= self.installed_sequence {
            // A delayed snapshot from the current lease is stale traffic. Ignore it before
            // validating its payload so it cannot retire a newer installed state.
            return GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate {
                installed_sequence: self.installed_sequence,
            };
        }
        let expected = self.shared.table.bucket_count();
        if snapshot.buckets.len() != expected {
            return self.deactivate_snapshot(GlobalCkfLaneFault::InvalidSnapshotBucketCount {
                expected,
                actual: snapshot.buckets.len(),
            });
        }
        if self.ready {
            return self.deactivate(
                GlobalCkfLaneFault::SnapshotWhileReady {
                    installed_sequence: self.installed_sequence,
                    snapshot_sequence: snapshot.sequence,
                },
                CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite,
            );
        }

        self.clear_ready();
        for (bucket, &value) in snapshot.buckets.iter().enumerate() {
            self.shared
                .table
                .store_image(bucket, self.lane, PackedBucket(value));
        }
        if !activate(self, snapshot.sequence) {
            return GlobalCkfIngestOutcome::AwaitingSnapshot;
        }
        GlobalCkfIngestOutcome::SnapshotInstalled {
            sequence: snapshot.sequence,
        }
    }

    /// Apply one contiguous absolute-image delta without making it atomic to readers.
    ///
    /// Foreign or superseded leases are ignored. A current-stream gap, reordering, malformed
    /// image set, or identity mismatch clears readiness and requires a new snapshot; deltas are
    /// never buffered for later repair.
    pub fn apply_delta(&mut self, delta: &GlobalCkfDelta) -> GlobalCkfIngestOutcome {
        self.last_failure_disposition = None;
        let Some(assignment) = self.assignment else {
            return GlobalCkfIngestOutcome::IgnoredForeignLease;
        };
        if delta.lease != assignment.lease {
            // Queued traffic from a retired assignment cannot invalidate the newer lane.
            return GlobalCkfIngestOutcome::IgnoredForeignLease;
        }
        if delta.identity != assignment.identity {
            return self.deactivate(
                GlobalCkfLaneFault::WrongIdentity,
                CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite,
            );
        }
        if !self.ready {
            return GlobalCkfIngestOutcome::AwaitingSnapshot;
        }
        if delta.sequence <= self.installed_sequence {
            return GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate {
                installed_sequence: self.installed_sequence,
            };
        }
        if delta.base_sequence != self.installed_sequence {
            // Do not buffer across a gap: fail closed until a barrier snapshot is installed.
            return self.deactivate(
                GlobalCkfLaneFault::BaseSequenceMismatch {
                    expected: self.installed_sequence,
                    actual: delta.base_sequence,
                },
                CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite,
            );
        }
        let Some(expected_sequence) = delta.base_sequence.checked_add(1) else {
            return self.deactivate(
                GlobalCkfLaneFault::SequenceExhausted,
                CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite,
            );
        };
        if delta.sequence != expected_sequence {
            return self.deactivate(
                GlobalCkfLaneFault::NonContiguousSequence {
                    base: delta.base_sequence,
                    sequence: delta.sequence,
                },
                CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite,
            );
        }
        if let Err(fault) = self.validate_images(&delta.images) {
            return self.deactivate(fault, CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite);
        }

        for &image in &delta.images {
            self.shared
                .table
                .store_image(image.bucket, self.lane, PackedBucket(image.value));
        }
        self.installed_sequence = delta.sequence;
        GlobalCkfIngestOutcome::DeltaApplied {
            sequence: delta.sequence,
            images: delta.images.len(),
        }
    }

    /// Complete an exact drain only at the publisher's expected terminal sequence.
    ///
    /// The production ingestion FIFO must place this marker after all preceding deltas. A
    /// mismatch catches a dropped final delta that would otherwise have no successor to expose
    /// the gap.
    pub fn complete_drain(&mut self, marker: ConsumerDrainMarker) -> GlobalCkfIngestOutcome {
        self.last_failure_disposition = None;
        let Some(assignment) = self.assignment else {
            return GlobalCkfIngestOutcome::IgnoredForeignLease;
        };
        if marker.lease != assignment.lease {
            return GlobalCkfIngestOutcome::IgnoredForeignLease;
        }
        if self.ready && self.installed_sequence == marker.expected_sequence {
            return GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: self.installed_sequence,
            };
        }

        let actual = self.ready.then_some(self.installed_sequence);
        self.deactivate(
            GlobalCkfLaneFault::TerminalSequenceMismatch {
                expected: marker.expected_sequence,
                actual,
            },
            CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite,
        )
    }

    fn validate_images(
        &mut self,
        images: &[GlobalCkfBucketImage],
    ) -> Result<(), GlobalCkfLaneFault> {
        self.clear_validation();
        if images.is_empty() {
            return Err(GlobalCkfLaneFault::EmptyDelta);
        }
        let bucket_count = self.shared.table.bucket_count();
        for image in images {
            if image.bucket >= bucket_count {
                self.clear_validation();
                return Err(GlobalCkfLaneFault::InvalidImageIndex {
                    bucket: image.bucket,
                });
            }
            let word = image.bucket / u64::BITS as usize;
            let bit = 1u64 << (image.bucket % u64::BITS as usize);
            if self.validation_words[word] & bit != 0 {
                self.clear_validation();
                return Err(GlobalCkfLaneFault::DuplicateImageIndex {
                    bucket: image.bucket,
                });
            }
            if self.validation_words[word] == 0 {
                self.validation_touched_words.push(word);
            }
            self.validation_words[word] |= bit;
        }
        self.clear_validation();
        Ok(())
    }

    fn clear_validation(&mut self) {
        for word in self.validation_touched_words.drain(..) {
            self.validation_words[word] = 0;
        }
    }

    fn deactivate_snapshot(&mut self, fault: GlobalCkfLaneFault) -> GlobalCkfIngestOutcome {
        let point = if self.ready {
            CkfFailurePoint::ConsumerGapOrMalformedBeforeWrite
        } else {
            CkfFailurePoint::InactiveSnapshotInstallFailure
        };
        self.deactivate(fault, point)
    }

    fn deactivate(
        &mut self,
        fault: GlobalCkfLaneFault,
        point: CkfFailurePoint,
    ) -> GlobalCkfIngestOutcome {
        self.last_failure_disposition = Some(point.disposition());
        self.clear_ready();
        GlobalCkfIngestOutcome::LaneDeactivated { fault }
    }

    pub(super) fn activate_snapshot(&mut self, sequence: u64) {
        self.installed_sequence = sequence;
        self.ready = true;
        self.shared
            .ready_lanes
            .fetch_or(1u16 << self.lane, Ordering::Release);
    }

    fn clear_ready(&mut self) {
        self.shared
            .ready_lanes
            .fetch_and(!(1u16 << self.lane), Ordering::AcqRel);
        self.ready = false;
    }

    #[cfg(test)]
    fn bucket(&self, bucket: usize) -> u64 {
        self.shared.table.load_image(bucket, self.lane).0
    }
}

fn validate_assignment(
    manifest: &GlobalCkfManifest,
    lane: usize,
    identity: ProducerIdentity,
    lease: LaneLease,
    previous_assignment_epoch: Option<u64>,
) -> Result<(), GlobalCkfAssignmentError> {
    if lease.consumer_instance != manifest.consumer_instance {
        return Err(GlobalCkfAssignmentError::WrongConsumerInstance {
            expected: manifest.consumer_instance,
            actual: lease.consumer_instance,
        });
    }
    if usize::from(lease.physical_lane) != lane {
        return Err(GlobalCkfAssignmentError::WrongPhysicalLane {
            expected: lane as u8,
            actual: lease.physical_lane,
        });
    }
    let pool_id = manifest
        .pool_id(lane)
        .expect("an ingestor can only claim a configured lane");
    if pool_id != identity.pool_id {
        return Err(GlobalCkfAssignmentError::WrongLaneOwner { lane: lane as u8 });
    }
    if identity.format != manifest.format {
        return Err(GlobalCkfAssignmentError::WrongFormat);
    }
    if let Some(previous) = previous_assignment_epoch
        && lease.assignment_epoch <= previous
    {
        return Err(GlobalCkfAssignmentError::StaleAssignmentEpoch {
            minimum_exclusive: previous,
            actual: lease.assignment_epoch,
        });
    }
    Ok(())
}

impl Drop for GlobalCkfLaneIngestor {
    fn drop(&mut self) {
        self.clear_ready();
        self.shared
            .claimed_lanes
            .fetch_and(!(1u16 << self.lane), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{CacheSemanticsId, RoutingScopeId};
    use crate::indexer::cuckoo::{
        CkfCommitState, CkfConfig, CkfFailureAction, CkfFailureDomain, DcCkfState,
    };

    struct Fixture {
        indexer: GlobalCkfIndexer,
        ingestor: GlobalCkfLaneIngestor,
        identity: ProducerIdentity,
        lease: LaneLease,
        bucket_count: usize,
    }

    impl Fixture {
        fn new() -> Self {
            let producer = DcCkfState::new(CkfConfig::new(64)).unwrap();
            let format = producer.format();
            let consumer = ConsumerInstanceId::new(17);
            let domain = IndexerDomainId::new(
                CacheSemanticsId::new([31; 16], IdentitySource::Explicit),
                RoutingScopeId::new([29; 16], IdentitySource::Explicit),
            );
            let pool_id = PoolId::new(domain, DcId::new(41));
            let mut lanes = [None; DC_COUNT];
            lanes[3] = Some(pool_id);
            let manifest = GlobalCkfManifest::new(consumer, domain, format, lanes).unwrap();
            let indexer = GlobalCkfIndexer::new(manifest, PrefixSearchConfig::default()).unwrap();
            let mut ingestor = indexer.claim_lane(3).unwrap();
            let identity = ProducerIdentity::new(pool_id, 43, 47, format);
            let lease = LaneLease::new(consumer, 3, 1);
            ingestor.assign(identity, lease).unwrap();
            Self {
                indexer,
                ingestor,
                identity,
                lease,
                bucket_count: format.bucket_count(),
            }
        }

        fn snapshot(&self, sequence: u64) -> GlobalCkfSnapshot {
            GlobalCkfSnapshot::new(
                self.identity,
                self.lease,
                sequence,
                vec![0; self.bucket_count].into_boxed_slice(),
            )
        }

        fn install(&mut self, sequence: u64) {
            assert_eq!(
                self.ingestor.install_snapshot(&self.snapshot(sequence)),
                GlobalCkfIngestOutcome::SnapshotInstalled { sequence }
            );
        }

        fn delta(
            &self,
            base_sequence: u64,
            sequence: u64,
            images: Vec<GlobalCkfBucketImage>,
        ) -> GlobalCkfDelta {
            GlobalCkfDelta::new(self.identity, self.lease, base_sequence, sequence, images)
        }
    }

    #[test]
    fn manifest_enforces_one_domain_and_one_pool_per_dc() {
        let producer = DcCkfState::new(CkfConfig::new(8)).unwrap();
        let format = producer.format();
        let consumer = ConsumerInstanceId::new(1);
        let domain = IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([2; 16], IdentitySource::Explicit),
        );
        let foreign_domain = IndexerDomainId::new(
            CacheSemanticsId::new([3; 16], IdentitySource::Explicit),
            RoutingScopeId::new([4; 16], IdentitySource::Explicit),
        );

        let mut mixed = [None; DC_COUNT];
        mixed[0] = Some(PoolId::new(domain, DcId::new(10)));
        mixed[1] = Some(PoolId::new(foreign_domain, DcId::new(11)));
        assert!(matches!(
            GlobalCkfManifest::new(consumer, domain, format, mixed),
            Err(GlobalCkfBuildError::MixedIndexerDomain { lane: 1, .. })
        ));

        let pool = PoolId::new(domain, DcId::new(10));
        let mut duplicate = [None; DC_COUNT];
        duplicate[0] = Some(pool);
        duplicate[1] = Some(pool);
        assert_eq!(
            GlobalCkfManifest::new(consumer, domain, format, duplicate),
            Err(GlobalCkfBuildError::DuplicatePool { pool_id: pool })
        );
    }

    #[test]
    fn default_derived_domains_may_join_across_dcs() {
        let producer = DcCkfState::new(CkfConfig::new(8)).unwrap();
        let format = producer.format();
        let domain = IndexerDomainId::new(
            CacheSemanticsId::new([5; 16], IdentitySource::DefaultDerived),
            RoutingScopeId::new([6; 16], IdentitySource::Explicit),
        );
        let mut lanes = [None; DC_COUNT];
        lanes[0] = Some(PoolId::new(domain, DcId::new(1)));
        lanes[1] = Some(PoolId::new(domain, DcId::new(2)));

        let manifest =
            GlobalCkfManifest::new(ConsumerInstanceId::new(2), domain, format, lanes).unwrap();

        assert_eq!(manifest.configured_lanes(), 0b11);
        assert_eq!(manifest.indexer_domain(), domain);
    }

    #[test]
    fn validation_scratch_scales_with_bitmap_words_not_bucket_count() {
        let fixture = Fixture::new();
        let word_count = fixture.bucket_count.div_ceil(u64::BITS as usize);
        assert_eq!(fixture.ingestor.validation_words.len(), word_count);
        assert!(fixture.ingestor.validation_touched_words.capacity() < fixture.bucket_count);
    }

    #[test]
    fn snapshot_activates_and_assignment_clears_readiness_before_replacement() {
        let mut fixture = Fixture::new();
        fixture.install(9);
        assert_eq!(fixture.indexer.ready_lanes(), 1 << 3);
        assert!(fixture.indexer.find_prefix_matches(&[]).is_ok());

        let captured = fixture.indexer.ready_lanes();
        let replacement_identity = ProducerIdentity::new(
            fixture.identity.pool_id(),
            fixture.identity.producer_incarnation() + 1,
            fixture.identity.layout_generation() + 1,
            fixture.identity.format(),
        );
        let replacement_lease = LaneLease::new(fixture.lease.consumer_instance(), 3, 2);
        fixture
            .ingestor
            .assign(replacement_identity, replacement_lease)
            .unwrap();

        assert_eq!(fixture.indexer.ready_lanes(), 0);
        assert_eq!(
            fixture.indexer.find_prefix_matches(&[]),
            Err(GlobalCkfQueryError::Unavailable)
        );
        let old_query = fixture
            .indexer
            .find_prefix_matches_with_captured_ready(&[], captured)
            .unwrap();
        assert_eq!(old_query.captured_ready_lanes(), 1 << 3);
    }

    #[test]
    fn valid_delta_advances_worker_local_sequence() {
        let mut fixture = Fixture::new();
        fixture.install(7);
        let delta = fixture.delta(7, 8, vec![GlobalCkfBucketImage::new(1, 0x1234)]);
        assert_eq!(
            fixture.ingestor.apply_delta(&delta),
            GlobalCkfIngestOutcome::DeltaApplied {
                sequence: 8,
                images: 1
            }
        );
        assert_eq!(fixture.ingestor.installed_sequence(), Some(8));
        assert_eq!(fixture.ingestor.bucket(1), 0x1234);
    }

    #[test]
    fn stale_same_lease_snapshot_cannot_roll_back_active_lane() {
        let mut fixture = Fixture::new();
        fixture.install(7);
        let delta = fixture.delta(7, 8, vec![GlobalCkfBucketImage::new(1, 0x1234)]);
        assert!(matches!(
            fixture.ingestor.apply_delta(&delta),
            GlobalCkfIngestOutcome::DeltaApplied { sequence: 8, .. }
        ));
        let stale = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            7,
            vec![0xFFFF; fixture.bucket_count].into_boxed_slice(),
        );

        assert_eq!(
            fixture.ingestor.install_snapshot(&stale),
            GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate {
                installed_sequence: 8
            }
        );
        assert_eq!(fixture.ingestor.installed_sequence(), Some(8));
        assert_eq!(fixture.ingestor.bucket(1), 0x1234);
        assert_eq!(fixture.indexer.ready_lanes(), 1 << 3);
    }

    #[test]
    fn unexpected_forward_snapshot_deactivates_active_lane_without_writes() {
        let mut fixture = Fixture::new();
        fixture.install(7);
        let forward = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            8,
            vec![0xFFFF; fixture.bucket_count].into_boxed_slice(),
        );

        assert_eq!(
            fixture.ingestor.install_snapshot(&forward),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::SnapshotWhileReady {
                    installed_sequence: 7,
                    snapshot_sequence: 8,
                }
            }
        );
        assert_eq!(fixture.ingestor.bucket(0), 0);
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn query_result_preserves_pool_and_captured_mask() {
        let mut fixture = Fixture::new();
        fixture.install(7);

        let result = fixture.indexer.find_prefix_matches(&[]).unwrap();
        assert_eq!(result.captured_ready_lanes(), 1 << 3);
        let lane = result.lanes()[3].unwrap();
        assert_eq!(lane.physical_lane(), 3);
        assert_eq!(lane.pool_id(), fixture.identity.pool_id());
        assert_eq!(lane.prefix_depth(), 0);
    }

    #[test]
    fn duplicate_and_stale_sequences_are_ignored_without_writes() {
        let mut fixture = Fixture::new();
        fixture.install(4);
        let duplicate = fixture.delta(3, 4, vec![GlobalCkfBucketImage::new(0, 77)]);
        assert_eq!(
            fixture.ingestor.apply_delta(&duplicate),
            GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate {
                installed_sequence: 4
            }
        );
        let stale = fixture.delta(1, 2, vec![GlobalCkfBucketImage::new(0, 88)]);
        assert_eq!(
            fixture.ingestor.apply_delta(&stale),
            GlobalCkfIngestOutcome::IgnoredStaleOrDuplicate {
                installed_sequence: 4
            }
        );
        assert_eq!(fixture.ingestor.bucket(0), 0);
        assert_eq!(fixture.indexer.ready_lanes(), 1 << 3);
    }

    #[test]
    fn forward_gap_deactivates_lane() {
        let mut fixture = Fixture::new();
        fixture.install(4);
        let gap = fixture.delta(5, 6, vec![GlobalCkfBucketImage::new(0, 1)]);
        assert_eq!(
            fixture.ingestor.apply_delta(&gap),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::BaseSequenceMismatch {
                    expected: 4,
                    actual: 5
                }
            }
        );
        assert_eq!(fixture.indexer.ready_lanes(), 0);
        assert_eq!(fixture.ingestor.installed_sequence(), None);
        assert_eq!(fixture.ingestor.bucket(0), 0);
        let disposition = fixture.ingestor.last_failure_disposition().unwrap();
        assert_eq!(disposition.domain, CkfFailureDomain::ConsumerLane);
        assert_eq!(disposition.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(disposition.action, CkfFailureAction::DeactivateAndSnapshot);
    }

    #[test]
    fn inactive_snapshot_validation_failure_is_retryable_on_same_assignment() {
        let mut fixture = Fixture::new();
        let malformed = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            7,
            vec![99; fixture.bucket_count - 1].into_boxed_slice(),
        );

        assert!(matches!(
            fixture.ingestor.install_snapshot(&malformed),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::InvalidSnapshotBucketCount { .. }
            }
        ));
        let disposition = fixture.ingestor.last_failure_disposition().unwrap();
        assert_eq!(disposition.domain, CkfFailureDomain::ConsumerLane);
        assert_eq!(disposition.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(disposition.action, CkfFailureAction::RetrySnapshot);
        assert_eq!(fixture.ingestor.bucket(0), 0);

        fixture.install(7);
        assert_eq!(fixture.ingestor.last_failure_disposition(), None);
        assert_eq!(fixture.indexer.ready_lanes(), 1 << 3);
    }

    #[test]
    fn noncontiguous_sequence_deactivates_lane() {
        let mut fixture = Fixture::new();
        fixture.install(4);
        let reordered = fixture.delta(4, 7, vec![GlobalCkfBucketImage::new(0, 1)]);
        assert_eq!(
            fixture.ingestor.apply_delta(&reordered),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::NonContiguousSequence {
                    base: 4,
                    sequence: 7
                }
            }
        );
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn malformed_images_are_rejected_before_any_write() {
        let mut fixture = Fixture::new();
        fixture.install(1);
        let duplicate = fixture.delta(
            1,
            2,
            vec![
                GlobalCkfBucketImage::new(0, 11),
                GlobalCkfBucketImage::new(0, 22),
            ],
        );
        assert_eq!(
            fixture.ingestor.apply_delta(&duplicate),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::DuplicateImageIndex { bucket: 0 }
            }
        );
        assert_eq!(fixture.ingestor.bucket(0), 0);

        fixture.install(2);
        let invalid = fixture.delta(
            2,
            3,
            vec![
                GlobalCkfBucketImage::new(1, 33),
                GlobalCkfBucketImage::new(fixture.bucket_count, 44),
            ],
        );
        assert_eq!(
            fixture.ingestor.apply_delta(&invalid),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::InvalidImageIndex {
                    bucket: fixture.bucket_count
                }
            }
        );
        assert_eq!(fixture.ingestor.bucket(1), 0);
    }

    #[test]
    fn empty_current_stream_delta_is_malformed() {
        let mut fixture = Fixture::new();
        fixture.install(1);
        let empty = fixture.delta(1, 2, Vec::new());
        assert_eq!(
            fixture.ingestor.apply_delta(&empty),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::EmptyDelta
            }
        );
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn wrong_identity_for_current_lease_deactivates_lane() {
        let mut fixture = Fixture::new();
        fixture.install(1);
        let wrong_identity = ProducerIdentity::new(
            fixture.identity.pool_id(),
            fixture.identity.producer_incarnation() + 1,
            fixture.identity.layout_generation(),
            fixture.identity.format(),
        );
        let delta = GlobalCkfDelta::new(
            wrong_identity,
            fixture.lease,
            1,
            2,
            vec![GlobalCkfBucketImage::new(0, 1)],
        );
        assert_eq!(
            fixture.ingestor.apply_delta(&delta),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::WrongIdentity
            }
        );
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn superseded_lease_is_ignored_without_invalidating_replacement() {
        let mut fixture = Fixture::new();
        fixture.install(1);
        let old_delta = fixture.delta(1, 2, vec![GlobalCkfBucketImage::new(0, 1)]);
        let replacement_lease = LaneLease::new(fixture.lease.consumer_instance(), 3, 2);
        fixture
            .ingestor
            .assign(fixture.identity, replacement_lease)
            .unwrap();
        let replacement_snapshot = GlobalCkfSnapshot::new(
            fixture.identity,
            replacement_lease,
            10,
            vec![0; fixture.bucket_count].into_boxed_slice(),
        );
        fixture.ingestor.install_snapshot(&replacement_snapshot);

        assert_eq!(
            fixture.ingestor.apply_delta(&old_delta),
            GlobalCkfIngestOutcome::IgnoredForeignLease
        );
        assert_eq!(fixture.ingestor.installed_sequence(), Some(10));
        assert_eq!(fixture.indexer.ready_lanes(), 1 << 3);
    }

    #[test]
    fn malformed_snapshot_deactivates_before_writing() {
        let mut fixture = Fixture::new();
        fixture.install(1);
        let snapshot = GlobalCkfSnapshot::new(
            fixture.identity,
            fixture.lease,
            2,
            vec![99; fixture.bucket_count - 1].into_boxed_slice(),
        );
        assert_eq!(
            fixture.ingestor.install_snapshot(&snapshot),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::InvalidSnapshotBucketCount {
                    expected: fixture.bucket_count,
                    actual: fixture.bucket_count - 1
                }
            }
        );
        assert_eq!(fixture.ingestor.bucket(0), 0);
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn wrong_identity_snapshot_for_current_lease_deactivates_before_writing() {
        let mut fixture = Fixture::new();
        fixture.install(1);
        let wrong_identity = ProducerIdentity::new(
            fixture.identity.pool_id(),
            fixture.identity.producer_incarnation(),
            fixture.identity.layout_generation() + 1,
            fixture.identity.format(),
        );
        let snapshot = GlobalCkfSnapshot::new(
            wrong_identity,
            fixture.lease,
            2,
            vec![99; fixture.bucket_count].into_boxed_slice(),
        );
        assert_eq!(
            fixture.ingestor.install_snapshot(&snapshot),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::WrongIdentity
            }
        );
        assert_eq!(fixture.ingestor.bucket(0), 0);
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn terminal_drain_detects_dropped_final_delta() {
        let mut fixture = Fixture::new();
        fixture.install(7);
        let marker = ConsumerDrainMarker::new(fixture.lease, 8);
        assert_eq!(
            fixture.ingestor.complete_drain(marker),
            GlobalCkfIngestOutcome::LaneDeactivated {
                fault: GlobalCkfLaneFault::TerminalSequenceMismatch {
                    expected: 8,
                    actual: Some(7)
                }
            }
        );
        assert_eq!(fixture.indexer.ready_lanes(), 0);
    }

    #[test]
    fn terminal_drain_acknowledges_exact_sequence_only() {
        let mut fixture = Fixture::new();
        fixture.install(7);
        let marker = ConsumerDrainMarker::new(fixture.lease, 7);
        assert_eq!(
            fixture.ingestor.complete_drain(marker),
            GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: 7
            }
        );
        assert_eq!(fixture.indexer.ready_lanes(), 1 << 3);
    }

    #[test]
    fn assignment_epoch_must_increase() {
        let mut fixture = Fixture::new();
        assert_eq!(
            fixture.ingestor.assign(fixture.identity, fixture.lease),
            Err(GlobalCkfAssignmentError::StaleAssignmentEpoch {
                minimum_exclusive: 1,
                actual: 1
            })
        );

        fixture.ingestor.retire();
        assert_eq!(
            fixture.ingestor.assign(fixture.identity, fixture.lease),
            Err(GlobalCkfAssignmentError::StaleAssignmentEpoch {
                minimum_exclusive: 1,
                actual: 1
            })
        );
    }

    #[test]
    fn lane_can_only_have_one_ingestion_owner() {
        let fixture = Fixture::new();
        assert!(matches!(
            fixture.indexer.claim_lane(3),
            Err(GlobalCkfBuildError::LaneAlreadyClaimed { lane: 3 })
        ));
    }
}
