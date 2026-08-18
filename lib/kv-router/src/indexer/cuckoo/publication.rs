// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Logically serialized publication for one actor-owned [`PoolId`](crate::identity::PoolId).
//!
//! The actor core emits unsequenced absolute bucket-image batches. This publisher alone owns the
//! ordinary `last_sequence`, current consumer lease, and reliable FIFO handoff. Mutation handling
//! never reads or increments a sequence, and queries never touch publisher state.
//!
//! Each ordinary batch is sampled after one complete actor command, so it represents one
//! actor-serialized producer cut and cannot split an event, Clear, or relocation. Cadence may
//! coalesce several completed commands into that cut. Consumer application remains deliberately
//! weaker: packed buckets are installed one `u64` at a time, so live queries may observe a
//! cross-bucket mixture while a delta is applied. Sequence numbers detect missing or reordered
//! batches; they are not event numbers or reader seqlocks.
//!
//! A proposed synchronization fix must identify a reachable interleaving that violates a
//! required invariant. Strengthening an explicit non-invariant is a contract change, not a
//! correctness fix.

use super::CkfBuildError;
use super::dc::{DcCkfPublicationBatch, DcCkfState};
use super::global::{ConsumerDrainMarker, DcCkfDelta, DcCkfSnapshot, LaneLease, ProducerIdentity};

/// Reliable FIFO ownership transfer for one complete delta envelope.
///
/// `Ok(())` means the next owner accepted the complete delta. An error means the publisher cannot
/// prove whether continuation is safe, so it leaves the sequence unchanged and fences the lease.
pub trait DcCkfDeltaSink {
    type Error;

    fn enqueue(&mut self, delta: DcCkfDelta) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublisherFenceReason {
    SequenceExhausted,
    DeliveryUncertain,
    LeaseStalled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublisherEmitOutcome {
    NoSubscriber { sequence: u64 },
    Published { sequence: u64, image_count: usize },
}

#[derive(Debug, PartialEq, Eq)]
pub enum DcCkfPublishError<E> {
    Fenced(PublisherFenceReason),
    Enqueue(E),
}

#[derive(Debug, PartialEq, Eq)]
pub enum PublisherSnapshotError<E> {
    Build(CkfBuildError),
    Publish(DcCkfPublishError<E>),
    SequenceExhausted,
}

/// Serialized stream owner paired with one actor-owned producer core.
#[derive(Debug)]
pub struct DcCkfPublisher<S> {
    identity: ProducerIdentity,
    lease: Option<LaneLease>,
    last_sequence: u64,
    fenced: Option<PublisherFenceReason>,
    sink: S,
}

impl<S> DcCkfPublisher<S>
where
    S: DcCkfDeltaSink,
{
    pub fn new(identity: ProducerIdentity, initial_sequence: u64, sink: S) -> Self {
        Self {
            identity,
            lease: None,
            last_sequence: initial_sequence,
            fenced: None,
            sink,
        }
    }

    pub const fn identity(&self) -> ProducerIdentity {
        self.identity
    }

    pub const fn lease(&self) -> Option<LaneLease> {
        self.lease
    }

    pub const fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub const fn fence_reason(&self) -> Option<PublisherFenceReason> {
        self.fenced
    }

    pub fn sink(&self) -> &S {
        &self.sink
    }

    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    pub fn into_sink(self) -> S {
        self.sink
    }

    pub fn publish(
        &mut self,
        batch: DcCkfPublicationBatch,
    ) -> Result<PublisherEmitOutcome, DcCkfPublishError<S::Error>> {
        if let Some(reason) = self.fenced {
            return Err(DcCkfPublishError::Fenced(reason));
        }
        let Some(lease) = self.lease else {
            return Ok(PublisherEmitOutcome::NoSubscriber {
                sequence: self.last_sequence,
            });
        };
        let Some(next_sequence) = self.last_sequence.checked_add(1) else {
            self.fenced = Some(PublisherFenceReason::SequenceExhausted);
            return Err(DcCkfPublishError::Fenced(
                PublisherFenceReason::SequenceExhausted,
            ));
        };
        let images = batch.into_images();
        let image_count = images.len();
        let delta = DcCkfDelta::new(
            self.identity,
            lease,
            self.last_sequence,
            next_sequence,
            images,
        );
        if let Err(error) = self.sink.enqueue(delta) {
            self.fenced = Some(PublisherFenceReason::DeliveryUncertain);
            return Err(DcCkfPublishError::Enqueue(error));
        }
        self.last_sequence = next_sequence;
        Ok(PublisherEmitOutcome::Published {
            sequence: next_sequence,
            image_count,
        })
    }

    pub fn retire_lease(&mut self) {
        self.lease = None;
    }

    pub fn retire_stalled_lease(&mut self) {
        self.lease = None;
        self.fenced = Some(PublisherFenceReason::LeaseStalled);
    }

    /// Install a new lease from an actor barrier snapshot.
    ///
    /// A healthy old lease first receives its pending tail. The replacement snapshot then records
    /// that terminal sequence `N`, so its first continuation is exactly `N -> N+1`. If prior
    /// delivery was uncertain, the replacement snapshot covers the already-drained producer tail
    /// without attempting to extend the retired stream. Checked sequence exhaustion requires a
    /// new producer generation.
    pub fn snapshot_after_barrier(
        &mut self,
        state: &mut DcCkfState,
        lease: LaneLease,
    ) -> Result<DcCkfSnapshot, PublisherSnapshotError<S::Error>> {
        if self.last_sequence == u64::MAX
            || self.fenced == Some(PublisherFenceReason::SequenceExhausted)
        {
            return Err(PublisherSnapshotError::SequenceExhausted);
        }
        let (pending_tail, buckets) = state
            .barrier_snapshot()
            .map_err(PublisherSnapshotError::Build)?;
        if self.fenced.is_none()
            && let Some(batch) = pending_tail
        {
            self.publish(batch).map_err(|error| match error {
                DcCkfPublishError::Fenced(PublisherFenceReason::SequenceExhausted) => {
                    PublisherSnapshotError::SequenceExhausted
                }
                error => PublisherSnapshotError::Publish(error),
            })?;
        }
        self.lease = Some(lease);
        self.fenced = None;
        Ok(DcCkfSnapshot::new(
            self.identity,
            lease,
            self.last_sequence,
            buckets,
        ))
    }

    pub fn exact_drain_marker(&self) -> Option<ConsumerDrainMarker> {
        self.lease
            .map(|lease| ConsumerDrainMarker::new(lease, self.last_sequence))
    }
}

#[cfg(test)]
mod tests {
    use crate::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
    };
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
    };

    use super::super::global::ConsumerInstanceId;
    use super::super::{CkfConfig, DcCkfFormatIdentity};
    use super::*;

    #[derive(Debug, Default)]
    struct RecordingSink {
        deltas: Vec<DcCkfDelta>,
        fail: bool,
    }

    impl DcCkfDeltaSink for RecordingSink {
        type Error = &'static str;

        fn enqueue(&mut self, delta: DcCkfDelta) -> Result<(), Self::Error> {
            if self.fail {
                return Err("disconnected");
            }
            self.deltas.push(delta);
            Ok(())
        }
    }

    fn identity(format: DcCkfFormatIdentity, generation: u64) -> ProducerIdentity {
        let domain = IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([3; 16], IdentitySource::Explicit),
        );
        ProducerIdentity::new(PoolId::new(domain, DcId::new(2)), 4, generation, format)
    }

    fn lease(epoch: u64) -> LaneLease {
        LaneLease::new(ConsumerInstanceId::new(5), 0, epoch)
    }

    fn stored(worker: WorkerWithDpRank, event_id: u64, hash: u64) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
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
                dp_rank: worker.dp_rank,
            },
        )
    }

    #[test]
    fn publisher_owns_checked_sequence_and_snapshot_continuation() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut publisher =
            DcCkfPublisher::new(identity(state.format(), 1), 7, RecordingSink::default());
        let snapshot = publisher
            .snapshot_after_barrier(&mut state, lease(1))
            .unwrap();
        assert_eq!(snapshot.sequence(), 7);

        let batch = state
            .apply_event(stored(worker, 1, 11))
            .into_publication()
            .unwrap();
        assert_eq!(
            publisher.publish(batch).unwrap(),
            PublisherEmitOutcome::Published {
                sequence: 8,
                image_count: 1,
            }
        );
        let delta = &publisher.sink().deltas[0];
        assert_eq!((delta.base_sequence(), delta.sequence()), (7, 8));
        assert_eq!(
            publisher.exact_drain_marker().unwrap().expected_sequence(),
            8
        );
    }

    #[test]
    fn final_remove_is_an_absolute_zero_image_not_an_empty_reset() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut publisher =
            DcCkfPublisher::new(identity(state.format(), 1), 0, RecordingSink::default());
        publisher
            .snapshot_after_barrier(&mut state, lease(1))
            .unwrap();
        publisher
            .publish(
                state
                    .apply_event(stored(worker, 1, 11))
                    .into_publication()
                    .unwrap(),
            )
            .unwrap();
        publisher
            .publish(state.remove_rank(worker).unwrap().unwrap())
            .unwrap();
        assert!(
            publisher.sink().deltas[1]
                .images()
                .iter()
                .any(|image| image.value() == 0)
        );
    }

    #[test]
    fn delivery_uncertainty_does_not_advance_and_recovers_with_new_lease_snapshot() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut publisher =
            DcCkfPublisher::new(identity(state.format(), 1), 9, RecordingSink::default());
        publisher
            .snapshot_after_barrier(&mut state, lease(1))
            .unwrap();
        publisher.sink_mut().fail = true;
        let error = publisher
            .publish(
                state
                    .apply_event(stored(worker, 1, 11))
                    .into_publication()
                    .unwrap(),
            )
            .unwrap_err();
        assert_eq!(error, DcCkfPublishError::Enqueue("disconnected"));
        assert_eq!(publisher.last_sequence(), 9);

        publisher.sink_mut().fail = false;
        let snapshot = publisher
            .snapshot_after_barrier(&mut state, lease(2))
            .unwrap();
        assert_eq!(snapshot.sequence(), 9);
        assert_eq!(snapshot.lease(), lease(2));
    }

    #[test]
    fn replacement_snapshot_emits_healthy_old_lease_tail_before_recording_sequence() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();
        let mut publisher =
            DcCkfPublisher::new(identity(state.format(), 1), 0, RecordingSink::default());
        publisher
            .snapshot_after_barrier(&mut state, lease(1))
            .unwrap();
        assert!(
            state
                .apply_event(stored(worker, 1, 11))
                .publication()
                .is_none()
        );

        let snapshot = publisher
            .snapshot_after_barrier(&mut state, lease(2))
            .unwrap();

        assert_eq!(snapshot.sequence(), 1);
        assert_eq!(snapshot.lease(), lease(2));
        assert_eq!(publisher.sink().deltas.len(), 1);
        assert_eq!(publisher.sink().deltas[0].lease(), lease(1));
        assert_eq!(
            (
                publisher.sink().deltas[0].base_sequence(),
                publisher.sink().deltas[0].sequence()
            ),
            (0, 1)
        );
    }

    #[test]
    fn failed_old_lease_tail_requires_retry_then_snapshot_covers_it() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let mut state = DcCkfState::new(config).unwrap();
        let mut publisher =
            DcCkfPublisher::new(identity(state.format(), 1), 4, RecordingSink::default());
        publisher
            .snapshot_after_barrier(&mut state, lease(1))
            .unwrap();
        assert!(
            state
                .apply_event(stored(worker, 1, 11))
                .publication()
                .is_none()
        );
        publisher.sink_mut().fail = true;

        assert!(matches!(
            publisher.snapshot_after_barrier(&mut state, lease(2)),
            Err(PublisherSnapshotError::Publish(DcCkfPublishError::Enqueue(
                "disconnected"
            )))
        ));
        assert_eq!(publisher.last_sequence(), 4);

        publisher.sink_mut().fail = false;
        let snapshot = publisher
            .snapshot_after_barrier(&mut state, lease(2))
            .unwrap();
        assert_eq!(snapshot.sequence(), 4);
        assert_eq!(snapshot.lease(), lease(2));
        assert!(snapshot.buckets().iter().any(|&bucket| bucket != 0));
    }

    #[test]
    fn checked_rollover_fences_without_enqueueing() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut state = DcCkfState::new(CkfConfig::new(32)).unwrap();
        let mut publisher = DcCkfPublisher::new(
            identity(state.format(), 1),
            u64::MAX - 1,
            RecordingSink::default(),
        );
        publisher
            .snapshot_after_barrier(&mut state, lease(1))
            .unwrap();
        publisher
            .publish(
                state
                    .apply_event(stored(worker, 1, 11))
                    .into_publication()
                    .unwrap(),
            )
            .unwrap();
        let error = publisher
            .publish(
                state
                    .apply_event(stored(worker, 2, 12))
                    .into_publication()
                    .unwrap(),
            )
            .unwrap_err();
        assert_eq!(
            error,
            DcCkfPublishError::Fenced(PublisherFenceReason::SequenceExhausted)
        );
        assert_eq!(publisher.sink().deltas.len(), 1);
        assert_eq!(publisher.last_sequence(), u64::MAX);
        assert!(matches!(
            publisher.snapshot_after_barrier(&mut state, lease(2)),
            Err(PublisherSnapshotError::SequenceExhausted)
        ));
    }
}
