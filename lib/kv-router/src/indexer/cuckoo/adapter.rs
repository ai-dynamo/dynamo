// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-process adapter between one actor-owned CKF producer and the global ingestion pool.
//!
//! This is the production boundary used when Relay and the endpoint-scoped global indexer share a
//! process. The adapter does not weaken either side's ownership model: [`DcCkfState`] remains
//! actor-owned, [`DcCkfPublisher`] alone owns the stream sequence, and the lane-sticky ingestion
//! worker owns consumer validation and its installed sequence. Producer batches are sampled only
//! after complete actor commands. The weak/torn behavior is on the consumer side: applying those
//! batches remains atomic per packed bucket, not across all buckets. Barrier snapshot installation
//! provides the full-state rebootstrap point for a lease.

use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::protocols::{KvCacheEventError, RouterEvent};

use super::global::{GlobalCkfIndexer, GlobalCkfIngestOutcome, LaneLease, ProducerIdentity};
use super::ingestion_pool::{GlobalCkfIngestionError, GlobalCkfIngestionPool};
use super::publication::{
    DcCkfDeltaSink, DcCkfPublishError, DcCkfPublisher, PublisherEmitOutcome, PublisherFenceReason,
    PublisherSnapshotError,
};
use super::{CkfBuildError, CkfConfig, DcCkfState};

pub const DEFAULT_LOCAL_CKF_RECOVERY_ATTEMPTS: usize = 8;
pub const DEFAULT_LOCAL_CKF_RECOVERY_BACKOFF: Duration = Duration::from_millis(200);

/// Same-generation snapshot-recovery policy.
///
/// The adapter exposes the exponential schedule but deliberately does not sleep. A lifecycle
/// coordinator can apply the returned delays without blocking the actor or an ingestion worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalCkfRecoveryPolicy {
    pub max_attempts: usize,
    pub initial_backoff: Duration,
}

impl Default for LocalCkfRecoveryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_LOCAL_CKF_RECOVERY_ATTEMPTS,
            initial_backoff: DEFAULT_LOCAL_CKF_RECOVERY_BACKOFF,
        }
    }
}

impl LocalCkfRecoveryPolicy {
    pub fn delay_after_failure(self, failed_attempt: usize) -> Option<Duration> {
        if failed_attempt == 0 || failed_attempt >= self.max_attempts {
            return None;
        }
        let exponent = u32::try_from(failed_attempt - 1).unwrap_or(u32::MAX);
        Some(
            self.initial_backoff
                .checked_mul(2u32.checked_pow(exponent).unwrap_or(u32::MAX))
                .unwrap_or(Duration::MAX),
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LocalCkfAdapterBuildError {
    #[error(transparent)]
    Producer(#[from] CkfBuildError),
    #[error("producer identity format does not match the local CKF state")]
    ProducerFormatMismatch,
    #[error("local CKF recovery max_attempts must be nonzero")]
    ZeroRecoveryAttempts,
    #[error("initial lane assignment failed: {0}")]
    Assignment(#[source] GlobalCkfIngestionError),
    #[error("initial actor barrier snapshot failed: {0:?}")]
    Snapshot(PublisherSnapshotError<LocalCkfDeltaSinkError>),
    #[error("initial snapshot ingestion failed: {0}")]
    SnapshotIngestion(#[source] GlobalCkfIngestionError),
    #[error("initial snapshot installation returned {0:?}")]
    SnapshotInstallation(GlobalCkfIngestOutcome),
}

#[derive(Debug, thiserror::Error)]
pub enum LocalCkfAdapterError {
    #[error("local CKF adapter admission is closed")]
    AdmissionClosed,
    #[error("local CKF publisher fenced the stream: {0:?}")]
    PublisherFenced(PublisherFenceReason),
    #[error("local CKF delta delivery failed: {0}")]
    DeltaDelivery(#[source] LocalCkfDeltaSinkError),
    #[error("consumer exact drain requires an active publisher lease")]
    MissingLease,
    #[error("consumer exact drain failed: {0}")]
    ConsumerDrain(#[source] GlobalCkfIngestionError),
    #[error("consumer exact drain returned {0:?}")]
    ConsumerDrainOutcome(GlobalCkfIngestOutcome),
    #[error("lane assignment epoch exhausted")]
    AssignmentEpochExhausted,
    #[error("same-generation lane assignment failed: {0}")]
    Assignment(#[source] GlobalCkfIngestionError),
    #[error("actor barrier snapshot failed: {0:?}")]
    Snapshot(PublisherSnapshotError<LocalCkfDeltaSinkError>),
    #[error("snapshot ingestion failed: {0}")]
    SnapshotIngestion(#[source] GlobalCkfIngestionError),
    #[error("snapshot installation returned {0:?}")]
    SnapshotInstallation(GlobalCkfIngestOutcome),
    #[error("snapshot recovery failed after {attempts} attempts: {last}")]
    RecoveryExhausted {
        attempts: usize,
        #[source]
        last: Box<LocalCkfAdapterError>,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum LocalCkfDeltaSinkError {
    #[error(transparent)]
    Ingestion(#[from] GlobalCkfIngestionError),
    #[cfg(test)]
    #[error("injected local CKF delta-delivery failure")]
    InjectedFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalCkfEventResult {
    pub first_error: Option<KvCacheEventError>,
    pub unknown_removals: usize,
    pub publication: Option<PublisherEmitOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalCkfDrainReport {
    pub terminal_sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalCkfRecoveryReport {
    pub attempts: usize,
    pub lease: LaneLease,
    pub sequence: u64,
}

struct IngestionDeltaSink {
    pool: Arc<GlobalCkfIngestionPool>,
    #[cfg(test)]
    faults: Arc<TestSinkFaults>,
}

impl DcCkfDeltaSink for IngestionDeltaSink {
    type Error = LocalCkfDeltaSinkError;

    fn enqueue(&mut self, delta: super::global::DcCkfDelta) -> Result<(), Self::Error> {
        #[cfg(test)]
        {
            if self.faults.fail_next.swap(false, Ordering::AcqRel) {
                return Err(LocalCkfDeltaSinkError::InjectedFailure);
            }
            if self.faults.drop_next.swap(false, Ordering::AcqRel) {
                return Ok(());
            }
        }
        self.pool.submit_delta(delta).map_err(Into::into)
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TestSinkFaults {
    fail_next: AtomicBool,
    drop_next: AtomicBool,
}

/// Actor-backed local producer-to-consumer composition for one `(DC, endpoint)` lane.
pub struct LocalCkfAdapter {
    state: DcCkfState,
    publisher: DcCkfPublisher<IngestionDeltaSink>,
    ingestion: Arc<GlobalCkfIngestionPool>,
    recovery: LocalCkfRecoveryPolicy,
    consumer_instance: super::global::ConsumerInstanceId,
    physical_lane: u8,
    next_assignment_epoch: u64,
    admission_open: bool,
    #[cfg(test)]
    faults: Arc<TestSinkFaults>,
}

impl LocalCkfAdapter {
    /// Create an empty producer, assign the supplied lease, and activate it from a barrier
    /// snapshot before opening actor admission.
    pub fn new(
        config: CkfConfig,
        identity: ProducerIdentity,
        lease: LaneLease,
        ingestion: Arc<GlobalCkfIngestionPool>,
        recovery: LocalCkfRecoveryPolicy,
    ) -> Result<Self, LocalCkfAdapterBuildError> {
        if recovery.max_attempts == 0 {
            return Err(LocalCkfAdapterBuildError::ZeroRecoveryAttempts);
        }
        let mut state = DcCkfState::new(config)?;
        if state.format() != identity.format() {
            return Err(LocalCkfAdapterBuildError::ProducerFormatMismatch);
        }

        #[cfg(test)]
        let faults = Arc::new(TestSinkFaults::default());
        let sink = IngestionDeltaSink {
            pool: Arc::clone(&ingestion),
            #[cfg(test)]
            faults: Arc::clone(&faults),
        };
        let mut publisher = DcCkfPublisher::new(identity, 0, sink);

        ingestion
            .assign(identity, lease)
            .map_err(LocalCkfAdapterBuildError::Assignment)?;
        let snapshot = publisher
            .snapshot_after_barrier(&mut state, lease)
            .map_err(LocalCkfAdapterBuildError::Snapshot)?;
        let expected_sequence = snapshot.sequence();
        let outcome = ingestion
            .install_snapshot(snapshot)
            .map_err(LocalCkfAdapterBuildError::SnapshotIngestion)?;
        if outcome
            != (GlobalCkfIngestOutcome::SnapshotInstalled {
                sequence: expected_sequence,
            })
        {
            return Err(LocalCkfAdapterBuildError::SnapshotInstallation(outcome));
        }

        Ok(Self {
            state,
            publisher,
            ingestion,
            recovery,
            consumer_instance: lease.consumer_instance(),
            physical_lane: lease.physical_lane(),
            next_assignment_epoch: lease.assignment_epoch(),
            admission_open: true,
            #[cfg(test)]
            faults,
        })
    }

    pub fn state(&self) -> &DcCkfState {
        &self.state
    }

    pub fn indexer(&self) -> &GlobalCkfIndexer {
        self.ingestion.indexer()
    }

    pub fn ingestion_pool(&self) -> &Arc<GlobalCkfIngestionPool> {
        &self.ingestion
    }

    pub fn identity(&self) -> ProducerIdentity {
        self.publisher.identity()
    }

    pub fn lease(&self) -> Option<LaneLease> {
        self.publisher.lease()
    }

    pub fn producer_sequence(&self) -> u64 {
        self.publisher.last_sequence()
    }

    pub fn publisher_fence_reason(&self) -> Option<PublisherFenceReason> {
        self.publisher.fence_reason()
    }

    /// Apply one actor-serialized event and synchronously hand any resulting publication to the
    /// bounded ingestion FIFO. Capacity exhaustion remains the event's observable pre-commit
    /// omission; it neither retires the lease nor starts snapshot recovery.
    pub fn apply_event(
        &mut self,
        event: RouterEvent,
    ) -> Result<LocalCkfEventResult, LocalCkfAdapterError> {
        if !self.admission_open {
            return Err(LocalCkfAdapterError::AdmissionClosed);
        }
        let outcome = self.state.apply_event(event);
        let first_error = outcome.first_error().copied();
        let unknown_removals = outcome.unknown_removals();
        let publication = outcome
            .into_publication()
            .map(|batch| self.publisher.publish(batch))
            .transpose()
            .map_err(map_publish_error)?;
        Ok(LocalCkfEventResult {
            first_error,
            unknown_removals,
            publication,
        })
    }

    /// Emit the actor's pending absolute-image tail into the lane-sticky ingestion FIFO.
    pub fn flush(&mut self) -> Result<Option<PublisherEmitOutcome>, LocalCkfAdapterError> {
        self.state
            .flush()
            .map(|batch| self.publisher.publish(batch))
            .transpose()
            .map_err(map_publish_error)
    }

    /// Emit the producer tail and wait for the consumer worker to observe the exact terminal
    /// stream sequence. The FIFO marker detects a missing final delta even if no later delta
    /// exists to expose the gap.
    pub fn exact_consumer_drain(&mut self) -> Result<LocalCkfDrainReport, LocalCkfAdapterError> {
        self.admission_open = false;
        self.flush()?;
        let marker = self
            .publisher
            .exact_drain_marker()
            .ok_or(LocalCkfAdapterError::MissingLease)?;
        let expected_sequence = marker.expected_sequence();
        let outcome = self
            .ingestion
            .complete_drain(marker)
            .map_err(LocalCkfAdapterError::ConsumerDrain)?;
        if outcome
            != (GlobalCkfIngestOutcome::DrainAcknowledged {
                installed_sequence: expected_sequence,
            })
        {
            return Err(LocalCkfAdapterError::ConsumerDrainOutcome(outcome));
        }
        self.admission_open = true;
        Ok(LocalCkfDrainReport {
            terminal_sequence: expected_sequence,
        })
    }

    /// Replace the current lease and recover the same producer generation from a new barrier
    /// snapshot. Retry delays are exposed by [`LocalCkfRecoveryPolicy`] and intentionally not
    /// slept here so this method never blocks a shared runtime worker on backoff.
    pub fn recover_snapshot(&mut self) -> Result<LocalCkfRecoveryReport, LocalCkfAdapterError> {
        self.admission_open = false;
        let mut last = None;
        for attempt in 1..=self.recovery.max_attempts {
            match self.recover_snapshot_once() {
                Ok((lease, sequence)) => {
                    self.admission_open = true;
                    return Ok(LocalCkfRecoveryReport {
                        attempts: attempt,
                        lease,
                        sequence,
                    });
                }
                Err(
                    error @ LocalCkfAdapterError::Snapshot(
                        PublisherSnapshotError::SequenceExhausted,
                    ),
                ) => return Err(error),
                Err(error) => last = Some(error),
            }
        }
        Err(LocalCkfAdapterError::RecoveryExhausted {
            attempts: self.recovery.max_attempts,
            last: Box::new(last.expect("nonzero recovery attempts always record an error")),
        })
    }

    fn recover_snapshot_once(&mut self) -> Result<(LaneLease, u64), LocalCkfAdapterError> {
        let assignment_epoch = self
            .next_assignment_epoch
            .checked_add(1)
            .ok_or(LocalCkfAdapterError::AssignmentEpochExhausted)?;
        self.next_assignment_epoch = assignment_epoch;
        let lease = LaneLease::new(self.consumer_instance, self.physical_lane, assignment_epoch);

        self.publisher.retire_lease();
        self.ingestion
            .assign(self.publisher.identity(), lease)
            .map_err(LocalCkfAdapterError::Assignment)?;
        let snapshot = self
            .publisher
            .snapshot_after_barrier(&mut self.state, lease)
            .map_err(LocalCkfAdapterError::Snapshot)?;
        let sequence = snapshot.sequence();
        let outcome = self
            .ingestion
            .install_snapshot(snapshot)
            .map_err(LocalCkfAdapterError::SnapshotIngestion)?;
        if outcome != (GlobalCkfIngestOutcome::SnapshotInstalled { sequence }) {
            return Err(LocalCkfAdapterError::SnapshotInstallation(outcome));
        }
        Ok((lease, sequence))
    }

    #[cfg(test)]
    fn fail_next_delta_delivery(&self) {
        self.faults.fail_next.store(true, Ordering::Release);
    }

    #[cfg(test)]
    fn drop_next_delta_delivery(&self) {
        self.faults.drop_next.store(true, Ordering::Release);
    }
}

fn map_publish_error(error: DcCkfPublishError<LocalCkfDeltaSinkError>) -> LocalCkfAdapterError {
    match error {
        DcCkfPublishError::Fenced(reason) => LocalCkfAdapterError::PublisherFenced(reason),
        DcCkfPublishError::Enqueue(error) => LocalCkfAdapterError::DeltaDelivery(error),
    }
}

#[cfg(test)]
mod tests {
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };

    use super::super::global::{
        CacheDomainId, CacheDomainIdentity, ConsumerInstanceId, DcId, EndpointId,
        GlobalCkfLaneOwner, GlobalCkfManifest,
    };
    use super::super::ingestion_pool::GlobalCkfIngestionPoolConfig;
    use super::*;

    const LANE: u8 = 3;

    fn build_adapter(mut config: CkfConfig) -> LocalCkfAdapter {
        config.publish_every_n_events = 1;
        let format = DcCkfState::new(config).unwrap().format();
        let domain = CacheDomainIdentity::new(CacheDomainId::new(7), 512, 1);
        let dc = DcId::new(11);
        let endpoint = EndpointId::new(13);
        let consumer = ConsumerInstanceId::new(17);
        let identity = ProducerIdentity::new(domain, dc, endpoint, 19, 1, format);
        let lease = LaneLease::new(consumer, LANE, 1);
        let mut lanes = [None; super::super::DC_COUNT];
        lanes[usize::from(LANE)] = Some(GlobalCkfLaneOwner::new(domain, dc, endpoint));
        let manifest = GlobalCkfManifest::new(consumer, endpoint, format, lanes).unwrap();
        let indexer = GlobalCkfIndexer::new(manifest, config.search).unwrap();
        let ingestion = Arc::new(
            GlobalCkfIngestionPool::new(
                indexer,
                GlobalCkfIngestionPoolConfig {
                    worker_count: 1,
                    queue_capacity: 32,
                    control_timeout: Duration::from_secs(1),
                    max_outstanding_images_per_lane: Some(format.bucket_count() * 64),
                    max_dirty_to_applied_age: Duration::from_secs(1),
                },
            )
            .unwrap(),
        );
        LocalCkfAdapter::new(
            config,
            identity,
            lease,
            ingestion,
            LocalCkfRecoveryPolicy {
                max_attempts: 8,
                initial_backoff: Duration::ZERO,
            },
        )
        .unwrap()
    }

    fn stored(event_id: u64, hash: u64) -> RouterEvent {
        RouterEvent::new(
            1,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        )
    }

    fn removed(event_id: u64, hash: u64) -> RouterEvent {
        RouterEvent::new(
            1,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(hash)],
                }),
                dp_rank: 0,
            },
        )
    }

    #[test]
    fn actor_publication_and_exact_consumer_drain_converge() {
        let mut adapter = build_adapter(CkfConfig::new(128));
        let result = adapter.apply_event(stored(1, 41)).unwrap();
        assert!(result.first_error.is_none());
        assert!(matches!(
            result.publication,
            Some(PublisherEmitOutcome::Published { sequence: 1, .. })
        ));
        assert_eq!(adapter.exact_consumer_drain().unwrap().terminal_sequence, 1);
        assert_ne!(adapter.indexer().ready_lanes(), 0);
    }

    #[test]
    fn delivery_failure_fences_publisher_and_recovers_with_new_lease_snapshot() {
        let mut adapter = build_adapter(CkfConfig::new(128));
        adapter.fail_next_delta_delivery();
        assert!(matches!(
            adapter.apply_event(stored(1, 41)),
            Err(LocalCkfAdapterError::DeltaDelivery(
                LocalCkfDeltaSinkError::InjectedFailure
            ))
        ));
        assert_eq!(
            adapter.publisher_fence_reason(),
            Some(PublisherFenceReason::DeliveryUncertain)
        );

        let recovery = adapter.recover_snapshot().unwrap();
        assert_eq!(recovery.attempts, 1);
        assert_eq!(recovery.sequence, 0);
        assert_eq!(recovery.lease.assignment_epoch(), 2);
        assert_ne!(adapter.indexer().ready_lanes(), 0);
        adapter.exact_consumer_drain().unwrap();
    }

    #[test]
    fn missing_final_delta_is_detected_by_terminal_marker_and_recovers() {
        let mut adapter = build_adapter(CkfConfig::new(128));
        adapter.drop_next_delta_delivery();
        adapter.apply_event(stored(1, 41)).unwrap();
        assert!(matches!(
            adapter.exact_consumer_drain(),
            Err(LocalCkfAdapterError::ConsumerDrainOutcome(
                GlobalCkfIngestOutcome::LaneDeactivated { .. }
            ))
        ));
        let recovery = adapter.recover_snapshot().unwrap();
        assert_eq!(recovery.sequence, 1);
        adapter.exact_consumer_drain().unwrap();
    }

    #[test]
    fn capacity_omission_does_not_retire_the_consumer_lane() {
        let mut config = CkfConfig::new(1);
        config.max_kicks = 1;
        let mut adapter = build_adapter(config);
        let mut omitted = None;
        let mut inserted = Vec::new();
        for hash in 1..=64 {
            let result = adapter.apply_event(stored(hash, hash)).unwrap();
            if result.first_error == Some(KvCacheEventError::CapacityExhausted) {
                omitted = Some(hash);
                break;
            }
            inserted.push(hash);
        }
        let omitted = omitted.expect("tiny actor producer must exhaust bounded capacity");
        assert_ne!(adapter.indexer().ready_lanes(), 0);
        adapter.exact_consumer_drain().unwrap();

        let unknown = adapter.apply_event(removed(100, omitted)).unwrap();
        assert_eq!(unknown.unknown_removals, 1);
        assert_ne!(adapter.indexer().ready_lanes(), 0);
        for (offset, hash) in inserted.into_iter().enumerate() {
            adapter
                .apply_event(removed(101 + offset as u64, hash))
                .unwrap();
        }
        assert!(
            adapter
                .apply_event(stored(200, omitted))
                .unwrap()
                .first_error
                .is_none()
        );
        adapter.exact_consumer_drain().unwrap();
    }

    #[test]
    fn recovery_policy_exposes_backoff_without_sleeping() {
        let policy = LocalCkfRecoveryPolicy::default();
        assert_eq!(
            policy.delay_after_failure(1),
            Some(Duration::from_millis(200))
        );
        assert_eq!(
            policy.delay_after_failure(2),
            Some(Duration::from_millis(400))
        );
        assert_eq!(policy.delay_after_failure(8), None);
    }
}
