// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
};

use tokio::sync::{Notify, Semaphore};
use tokio_util::sync::CancellationToken;

use super::*;
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, PlacementEvent,
};

use super::super::{PlacementEventSender, PublisherInput};

fn test_event() -> RouterEvent {
    RouterEvent::with_storage_tier(
        7,
        KvCacheEvent {
            event_id: 1,
            dp_rank: 0,
            data: KvCacheEventData::Cleared,
        },
        StorageTier::Device,
    )
}

fn stored_test_event(event_id: u64, dp_rank: u32) -> RouterEvent {
    RouterEvent::with_storage_tier(
        7,
        KvCacheEvent {
            event_id,
            dp_rank,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(event_id),
                    tokens_hash: LocalBlockHash(event_id),
                    mm_extra_info: None,
                }],
            }),
        },
        StorageTier::Device,
    )
}

const FAKE_APPLY_OK: u8 = 0;
const FAKE_APPLY_ERROR: u8 = 1;
const FAKE_APPLY_BLOCKED: u8 = 2;
const FAKE_APPLY_CONTROLLED: u8 = 3;

struct FakeWorkerEventOps {
    operations: StdMutex<Vec<&'static str>>,
    apply_mode: AtomicU8,
    apply_started: Notify,
    apply_release: Semaphore,
    apply_in_flight: AtomicUsize,
    max_apply_in_flight: AtomicUsize,
    applied_batch_sizes: StdMutex<Vec<usize>>,
    gc_fails: AtomicBool,
}

impl FakeWorkerEventOps {
    fn new(apply_mode: u8) -> Self {
        Self {
            operations: StdMutex::new(Vec::new()),
            apply_mode: AtomicU8::new(apply_mode),
            apply_started: Notify::new(),
            apply_release: Semaphore::new(0),
            apply_in_flight: AtomicUsize::new(0),
            max_apply_in_flight: AtomicUsize::new(0),
            applied_batch_sizes: StdMutex::new(Vec::new()),
            gc_fails: AtomicBool::new(false),
        }
    }

    fn operations(&self) -> Vec<&'static str> {
        self.operations.lock().unwrap().clone()
    }
}

impl ValkeyWorkerEventOps for FakeWorkerEventOps {
    fn apply_owned_batch<'a>(&'a self, events: &'a [RouterEvent]) -> ValkeyOperation<'a> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("apply");
            self.applied_batch_sizes.lock().unwrap().push(events.len());
            self.apply_started.notify_one();
            match self.apply_mode.load(Ordering::Relaxed) {
                FAKE_APPLY_OK => Ok(()),
                FAKE_APPLY_ERROR => bail!("permanent module rejection"),
                FAKE_APPLY_BLOCKED => std::future::pending().await,
                FAKE_APPLY_CONTROLLED => {
                    let in_flight = self.apply_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
                    self.max_apply_in_flight
                        .fetch_max(in_flight, Ordering::AcqRel);
                    let permit = self
                        .apply_release
                        .acquire()
                        .await
                        .expect("test semaphore remains open");
                    permit.forget();
                    self.apply_in_flight.fetch_sub(1, Ordering::AcqRel);
                    Ok(())
                }
                _ => unreachable!(),
            }
        })
    }

    fn unregister(&self) -> ValkeyOperation<'_> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("unregister");
            Ok(())
        })
    }

    fn renew(&self) -> ValkeyOperation<'_> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("renew");
            Ok(())
        })
    }

    fn gc_step(&self, inspection_budget: u32) -> ValkeyGcOperation<'_> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("gc");
            if self.gc_fails.load(Ordering::Relaxed) {
                bail!("synthetic GC failure");
            }
            Ok([u64::from(inspection_budget), 0, 0, 0, 0, 0, 0, 0])
        })
    }
}

fn fake_integrity(apply_mode: u8) -> (ValkeyPublisherIntegrity, Arc<FakeWorkerEventOps>) {
    let backend = Arc::new(FakeWorkerEventOps::new(apply_mode));
    (
        ValkeyPublisherIntegrity::with_backend(7, 11, backend.clone()),
        backend,
    )
}

async fn wait_for_in_flight(backend: &FakeWorkerEventOps, expected: usize) {
    tokio::time::timeout(Duration::from_secs(1), async {
        while backend.apply_in_flight.load(Ordering::Acquire) != expected {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expected controlled APPLY operations did not start");
}

#[tokio::test]
async fn concurrent_rank_events_collapse_into_one_integrity_batch() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_CONTROLLED);
    let first_integrity = integrity.clone();
    let first = tokio::spawn(async move {
        first_integrity
            .publish_event(&stored_test_event(1, 0))
            .await
    });
    let second =
        tokio::spawn(async move { integrity.publish_event(&stored_test_event(2, 1)).await });

    wait_for_in_flight(&backend, 1).await;
    assert_eq!(*backend.applied_batch_sizes.lock().unwrap(), vec![2]);
    assert_eq!(backend.max_apply_in_flight.load(Ordering::Acquire), 1);
    backend.apply_release.add_permits(1);
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
}

#[tokio::test]
async fn queued_rank_events_fill_one_worker_batch_without_waiting_per_event() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_OK);
    let publisher = QueuedValkeyPublisher::new(ValkeyEventPublisher::new(integrity.clone()));

    // Empty-channel reservations complete without yielding, so a full
    // ordered pipeline is visible when the worker collector first runs.
    for event_id in 1..=DIRECT_EVENT_BATCH_MAX_EVENTS as u64 {
        publisher
            .publish_event(&stored_test_event(event_id, 0))
            .await
            .unwrap();
    }
    integrity.wait_for_idle().await;

    assert_eq!(
        *backend.applied_batch_sizes.lock().unwrap(),
        vec![DIRECT_EVENT_BATCH_MAX_EVENTS]
    );
    assert_eq!(integrity.state(), IntegrityState::Healthy);
}

#[tokio::test]
async fn direct_valkey_events_stay_batched() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_OK);
    let publisher = QueuedValkeyPublisher::new(ValkeyEventPublisher::new(integrity.clone()));

    for event_id in 1..=DIRECT_EVENT_BATCH_MAX_EVENTS as u64 {
        publisher
            .publish_event(&stored_test_event(event_id, 0))
            .await
            .unwrap();
    }
    integrity.wait_for_idle().await;

    assert_eq!(
        *backend.applied_batch_sizes.lock().unwrap(),
        vec![DIRECT_EVENT_BATCH_MAX_EVENTS]
    );
    assert_eq!(integrity.state(), IntegrityState::Healthy);
}

#[tokio::test]
async fn worker_wide_clear_fails_closed_without_applying_queued_events() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_OK);
    let store_integrity = integrity.clone();
    let store = tokio::spawn(async move {
        store_integrity
            .publish_event(&stored_test_event(1, 0))
            .await
    });
    let clear_integrity = integrity.clone();
    let clear = tokio::spawn(async move { clear_integrity.publish_event(&test_event()).await });

    assert!(store.await.unwrap().is_err());
    assert!(clear.await.unwrap().is_err());
    assert_eq!(backend.operations(), Vec::<&'static str>::new());
    assert_eq!(integrity.state(), IntegrityState::Faulted);
}

#[tokio::test]
async fn fence_waits_for_in_flight_apply_before_unregister() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_CONTROLLED);
    let apply_integrity = integrity.clone();
    let apply = tokio::spawn(async move {
        apply_integrity
            .publish_event(&stored_test_event(1, 0))
            .await
    });
    wait_for_in_flight(&backend, 1).await;

    integrity.mark_fault("test_fault");
    let fence_integrity = integrity.clone();
    let fence = tokio::spawn(async move { fence_integrity.fence_once().await });
    tokio::task::yield_now().await;
    assert_eq!(backend.operations(), vec!["apply"]);

    backend.apply_release.add_permits(1);
    assert!(apply.await.unwrap().is_err());
    assert_eq!(fence.await.unwrap(), FenceAttempt::Confirmed);
    assert_eq!(backend.operations(), vec!["apply", "unregister"]);
}

#[tokio::test]
async fn cancelled_caller_does_not_release_batch_integrity_permit() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_CONTROLLED);
    let apply_integrity = integrity.clone();
    let apply = tokio::spawn(async move {
        apply_integrity
            .publish_event(&stored_test_event(1, 0))
            .await
    });
    wait_for_in_flight(&backend, 1).await;
    apply.abort();
    assert!(apply.await.unwrap_err().is_cancelled());

    integrity.mark_fault("test_fault");
    let fence_integrity = integrity.clone();
    let fence = tokio::spawn(async move { fence_integrity.fence_once().await });
    tokio::task::yield_now().await;
    assert_eq!(backend.operations(), vec!["apply"]);

    backend.apply_release.add_permits(1);
    assert_eq!(fence.await.unwrap(), FenceAttempt::Confirmed);
    assert_eq!(backend.operations(), vec!["apply", "unregister"]);
}

#[tokio::test]
async fn apply_waiting_on_gate_rechecks_health_after_fault() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_OK);
    let gate = integrity.inner.mutation_gate.write().await;
    let apply_integrity = integrity.clone();
    let apply = tokio::spawn(async move {
        apply_integrity
            .publish_event(&stored_test_event(1, 0))
            .await
    });
    tokio::task::yield_now().await;
    integrity.mark_fault("test_fault");
    let fence_integrity = integrity.clone();
    let fence = tokio::spawn(async move { fence_integrity.fence_once().await });
    drop(gate);

    assert!(apply.await.unwrap().is_err());
    assert_eq!(fence.await.unwrap(), FenceAttempt::Confirmed);
    assert_eq!(backend.operations(), vec!["unregister"]);
}

#[tokio::test]
async fn outage_apply_is_time_bounded_and_enters_integrity_fence() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_BLOCKED);
    let started = std::time::Instant::now();

    let error = integrity
        .publish_event(&stored_test_event(1, 0))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("integrity deadline"));
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(integrity.state(), IntegrityState::Faulted);
    assert_eq!(backend.operations(), vec!["apply"]);
}

#[tokio::test]
async fn outage_saturation_keeps_direct_ingress_bounded_and_fences() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_BLOCKED);
    let publisher = ValkeyEventPublisher::new(integrity.clone());
    let (tx, rx) = mpsc::channel::<PublisherInput>(2);
    let ingress = PlacementEventSender::direct_valkey(tx, integrity.clone());
    let processor = tokio::spawn(
        super::super::event_processor::start_direct_valkey_event_processor(
            publisher.clone(),
            publisher,
            7,
            CancellationToken::new(),
            super::super::PlacementEventReceiver::DirectValkey(rx),
            None,
            None,
        ),
    );

    ingress
        .send(PlacementEvent::local_gpu(7, stored_test_event(1, 0).event))
        .unwrap();
    backend.apply_started.notified().await;

    let mut rejected = 0;
    for event_id in 2..=100 {
        let event = PlacementEvent::local_gpu(7, stored_test_event(event_id, 0).event);
        if ingress.send(event).is_err() {
            rejected += 1;
        }
    }
    assert!(rejected >= 97, "bounded queue accepted too many events");
    drop(ingress);

    tokio::time::timeout(Duration::from_secs(1), processor)
        .await
        .expect("faulted processor should stop after its input closes")
        .expect("processor task should not panic");
    assert_eq!(backend.operations(), vec!["apply", "unregister"]);
    assert_eq!(integrity.state(), IntegrityState::Fenced);
}

#[tokio::test]
async fn lifecycle_gc_runs_only_when_healthy_and_idle() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_OK);

    assert_eq!(
        integrity.gc_step_if_idle(257).await.unwrap(),
        GcStepOutcome::Completed([257, 0, 0, 0, 0, 0, 0, 0])
    );
    {
        let _gate = integrity.inner.mutation_gate.read().await;
        assert_eq!(
            integrity.gc_step_if_idle(257).await.unwrap(),
            GcStepOutcome::SkippedBusy
        );
    }
    integrity.mark_fault("test_fault");
    assert_eq!(
        integrity.gc_step_if_idle(257).await.unwrap(),
        GcStepOutcome::SkippedFenced
    );
    assert_eq!(backend.operations(), vec!["gc"]);
}

#[tokio::test]
async fn lifecycle_gc_failure_does_not_fault_worker_integrity() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_OK);
    backend.gc_fails.store(true, Ordering::Relaxed);

    let error = integrity.gc_step_if_idle(256).await.unwrap_err();

    assert!(error.to_string().contains("synthetic GC failure"));
    assert_eq!(integrity.state(), IntegrityState::Healthy);
    assert_eq!(backend.operations(), vec!["gc"]);
}

#[tokio::test]
async fn permanent_apply_error_is_visible_and_never_auto_resumes() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_ERROR);

    let error = integrity
        .publish_event(&stored_test_event(1, 0))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("permanent module rejection"));
    assert_eq!(integrity.state(), IntegrityState::Faulted);

    // No event can sneak through after the permanent rejection. Even an
    // acknowledged unregister never reopens this process.
    assert!(
        integrity
            .publish_event(&stored_test_event(2, 0))
            .await
            .is_err()
    );
    backend.apply_mode.store(FAKE_APPLY_OK, Ordering::Relaxed);
    assert_eq!(integrity.fence_once().await, FenceAttempt::Confirmed);
    assert!(
        integrity
            .publish_event(&stored_test_event(3, 0))
            .await
            .is_err()
    );

    assert_eq!(backend.operations(), vec!["apply", "unregister"]);
    assert_eq!(integrity.state(), IntegrityState::Fenced);
}

#[tokio::test]
async fn fenced_publisher_cleanup_skips_duplicate_unregister() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_ERROR);
    assert!(
        integrity
            .publish_event(&stored_test_event(1, 0))
            .await
            .is_err()
    );
    assert_eq!(integrity.fence_once().await, FenceAttempt::Confirmed);

    integrity.unregister_for_shutdown().await.unwrap();

    assert_eq!(backend.operations(), vec!["apply", "unregister"]);
    assert_eq!(integrity.state(), IntegrityState::Fenced);
}

#[tokio::test]
async fn faulted_publisher_cleanup_claims_the_single_unregister_fence() {
    let (integrity, backend) = fake_integrity(FAKE_APPLY_ERROR);
    assert!(
        integrity
            .publish_event(&stored_test_event(1, 0))
            .await
            .is_err()
    );

    integrity.unregister_for_shutdown().await.unwrap();

    assert_eq!(backend.operations(), vec!["apply", "unregister"]);
    assert_eq!(integrity.state(), IntegrityState::Fenced);
}

#[tokio::test]
async fn direct_ingress_overflow_is_bounded_and_trips_worker_fence() {
    let (integrity, _backend) = fake_integrity(FAKE_APPLY_OK);
    let (tx, mut rx) = mpsc::channel::<PublisherInput>(1);
    let ingress = PlacementEventSender::direct_valkey(tx, integrity.clone());
    let event = PlacementEvent::local_gpu(7, test_event().event);

    ingress.send(event.clone()).unwrap();
    assert!(matches!(
        ingress.send(event),
        Err(super::super::PlacementEventSendError::Full(_))
    ));
    assert_eq!(integrity.state(), IntegrityState::Faulted);
    assert!(rx.try_recv().is_ok());
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn direct_zmq_connect_failure_trips_worker_fence() {
    let (integrity, _backend) = fake_integrity(FAKE_APPLY_OK);
    let (tx, _rx) = mpsc::channel::<PublisherInput>(1);
    let ingress = PlacementEventSender::direct_valkey(tx, integrity.clone());

    super::super::zmq_listener::start_zmq_listener(
        "not-a-valid-zmq-endpoint".to_string(),
        String::new(),
        7,
        ingress,
        CancellationToken::new(),
        4,
        Arc::new(AtomicU64::new(0)),
        None,
    )
    .await;

    assert_eq!(integrity.state(), IntegrityState::Faulted);
}

#[tokio::test]
async fn direct_zmq_cancellation_does_not_trip_worker_fence() {
    let (integrity, _backend) = fake_integrity(FAKE_APPLY_OK);
    let (tx, _rx) = mpsc::channel::<PublisherInput>(1);
    let ingress = PlacementEventSender::direct_valkey(tx, integrity.clone());
    let cancellation = CancellationToken::new();
    cancellation.cancel();

    super::super::zmq_listener::start_zmq_listener(
        "not-a-valid-zmq-endpoint".to_string(),
        String::new(),
        7,
        ingress,
        cancellation,
        4,
        Arc::new(AtomicU64::new(0)),
        None,
    )
    .await;

    assert_eq!(integrity.state(), IntegrityState::Healthy);
}
