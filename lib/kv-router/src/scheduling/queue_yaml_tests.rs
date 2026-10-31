// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Queue-actor tests whose policy profiles come from the router's YAML policy file: the
//! kernel owns the queue, the composed router owns the file format, so these live here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rustc_hash::FxHashMap;
use tokio::sync::{oneshot, watch};
use tokio::time::Instant;

use dynamo_admission::{PolicyProfile, RouterQueuePolicy};
use dynamo_scheduler::OverlapSignals;
use dynamo_scheduler::protocols::{WorkerConfigLike, WorkerWithDpRank};
use dynamo_scheduler::queue::*;
use dynamo_scheduler::selector::{DefaultWorkerSelector, WorkerCapacityExt, WorkerInputs};
use dynamo_scheduler::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};
use dynamo_scheduler::types::{
    KvSchedulerError, RejectThresholdsProvider, ScheduleMode, SchedulingRequest, SchedulingResponse,
};
use dynamo_slot_tracker::ActiveSequencesMultiWorker;
use dynamo_worker_selection::WorkerCapacity;

use crate::scheduling::policy_config::RouterPolicyConfig;

fn policy_profile(yaml: &str) -> PolicyProfile {
    RouterPolicyConfig::from_yaml(yaml)
        .unwrap()
        .resolve_profile(None, None, dynamo_admission::RouterQueuePolicy::Fcfs)
}

fn decay_now() -> Instant {
    Instant::now()
}

#[allow(clippy::type_complexity)]
fn make_queue_with_profile(
    num_workers: usize,
    block_size: u32,
    max_num_batched_tokens: usize,
    profile: PolicyProfile,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
) {
    let (queue, slots, _cfg_tx) = make_queue_with_profile_and_sender(
        num_workers,
        block_size,
        max_num_batched_tokens,
        profile,
    );
    (queue, slots)
}

#[allow(clippy::type_complexity)]
fn make_queue_with_profile_and_sender(
    num_workers: usize,
    block_size: u32,
    max_num_batched_tokens: usize,
    profile: PolicyProfile,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
) {
    make_queue_with_profile_sender_and_reject(
        num_workers,
        block_size,
        max_num_batched_tokens,
        profile,
        None,
    )
}

#[allow(clippy::type_complexity)]
fn make_queue_with_profile_sender_and_reject(
    num_workers: usize,
    block_size: u32,
    max_num_batched_tokens: usize,
    profile: PolicyProfile,
    reject_thresholds: Option<RejectThresholdsProvider>,
) -> (
    Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>>,
    Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
) {
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "test",
    ));
    let configs = (0..num_workers as u64)
        .map(|id| {
            (
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(max_num_batched_tokens as u64),
                    ..Default::default()
                },
            )
        })
        .collect();
    let (cfg_tx, cfg_rx) = watch::channel(configs);
    set_worker_capacities_from_configs(&slots, &cfg_rx.borrow());
    let queue = Arc::new(SchedulerQueue::new(
        Arc::clone(&slots),
        cfg_rx,
        profile,
        block_size,
        DefaultWorkerSelector::new(None, "test"),
        None,
        None,
        None,
        None,
        reject_thresholds,
    ));
    (queue, slots, cfg_tx)
}

/// What `LocalScheduler::reconcile_worker_configs` does after a config publish: the
/// registry learns each rank's reported capacity so the projection pass can judge it.
fn set_worker_capacities_from_configs(
    slots: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    configs: &HashMap<u64, SimpleWorkerConfig>,
) {
    slots.set_worker_capacities(configs.iter().map(|(&id, cfg)| {
        (
            dynamo_slot_tracker::topology::WorkerDpRange::new(
                id,
                cfg.data_parallel_start_rank(),
                cfg.data_parallel_size(),
            ),
            WorkerCapacity::of(cfg),
        )
    }));
}

fn make_request(
    request_id: &str,
    isl_tokens: usize,
) -> (
    SchedulingRequest,
    tokio::sync::oneshot::Receiver<
        Result<SchedulingResponse, dynamo_scheduler::types::KvSchedulerError>,
    >,
) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let req = SchedulingRequest {
        mode: ScheduleMode::Tracked {
            request_id: request_id.to_string(),
        },
        token_seq: None,
        isl_tokens,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        worker_loads: FxHashMap::default(),
        track_prefill_tokens: true,
        router_config_override: None,
        lora_name: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: dynamo_scheduler::protocols::RoutingConstraints::default(),
        shared_cache_hits: None,
        resp_tx: Some(tx),
    };
    (req, rx)
}

#[tokio::test]
async fn classifiers_read_class_and_reject_thresholds() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: short
  - min_tokens: 32
    bucket: long
policy_classes:
  - name: latency_short
    policy_family: latency
    cache_bucket: short
    quantum: 1
    decode_busy_threshold_frac: 0.5
  - name: latency_long
    policy_family: latency
    cache_bucket: long
    quantum: 1
  - name: bulk_short
    policy_family: bulk
    cache_bucket: short
    quantum: 1
    prefill_busy_threshold_frac: 0.25
  - name: bulk_long
    policy_family: bulk
    cache_bucket: long
    quantum: 1
"#,
    );
    let reject = dynamo_slot_tracker::busy::BusyThresholds {
        prefill_busy_threshold_frac: Some(1.0),
        ..Default::default()
    };
    let (queue, _slots, _cfg_tx) = make_queue_with_profile_sender_and_reject(
        1,
        16,
        64,
        profile,
        Some(Arc::new(move || reject)),
    );
    let (mut request, _rx) = make_request("classified", 16);
    request.policy_class = Some("latency".to_string());
    let (mut classified, _workers) =
        queue.build_classify_request(&request, Instant::now(), WorkerInputs::NONE);

    let latency = dynamo_slot_tracker::busy::BusyThresholds {
        decode_busy_threshold_frac: Some(0.5),
        ..Default::default()
    };
    let bulk = dynamo_slot_tracker::busy::BusyThresholds {
        prefill_busy_threshold_frac: Some(0.25),
        ..Default::default()
    };
    // 16 uncached tokens fall in the `short` bucket.
    assert_eq!(classified.queue_thresholds(), latency);
    assert_eq!(classified.queue_thresholds_for("bulk"), Some(bulk));
    assert_eq!(classified.queue_thresholds_for("missing"), None);
    assert_eq!(classified.reject_thresholds(), reject);

    classified.set_policy_class("bulk");
    assert_eq!(classified.queue_thresholds(), bulk);
}

#[tokio::test]
async fn classifier_overrides_are_validated_before_enqueue() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
  - name: bulk_cached
    policy_family: bulk
    cache_bucket: cached
    quantum: 1
  - name: bulk_uncached
    policy_family: bulk
    cache_bucket: uncached
    quantum: 1
  - name: custom_priority
    quantum: 1
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
    let (mut request, _rx) = make_request("classified", 16);
    request.policy_class = Some("latency".to_string());
    let ingress_at = Instant::now();
    let due_at = ingress_at + Duration::from_secs(20);
    let mut classified = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    assert_eq!(classified.ingress_at(), ingress_at);
    classified.set_policy_class("bulk");
    classified.set_due_at(due_at);
    classified.set_scheduling_cost_tokens(7);

    let metadata = queue
        .validate_classification(&mut request, classified, ingress_at)
        .unwrap();
    assert_eq!(
        queue.profile().class(metadata.class_index).name,
        "bulk_cached"
    );
    assert_eq!(metadata.snapshot.scheduling_cost_tokens, 7);
    assert_eq!(metadata.due_at, Some(due_at));

    let mut invalid = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    invalid.set_policy_class("missing");
    assert!(matches!(
        queue.validate_classification(&mut request, invalid, ingress_at),
        Err(KvSchedulerError::InvalidClassificationMetadata(_))
    ));

    let mut physical = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    physical.set_policy_class("latency_cached");
    assert!(matches!(
        queue.validate_classification(&mut request, physical, ingress_at),
        Err(KvSchedulerError::InvalidClassificationMetadata(_))
    ));

    let mut zero_cost = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;
    zero_cost.set_scheduling_cost_tokens(0);
    assert!(matches!(
        queue.validate_classification(&mut request, zero_cost, ingress_at),
        Err(KvSchedulerError::InvalidClassificationMetadata(_))
    ));
}

#[tokio::test]
async fn validation_recomputes_cache_eligibility_at_enqueue() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
"#,
    );
    let (queue, _slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 64, profile);
    let (mut request, _rx) = make_request("classified", 48);
    request
        .overlap
        .effective_cached_tokens
        .insert(WorkerWithDpRank::new(0, 0), 32);

    let ingress_at = Instant::now();
    let classified = queue
        .build_classify_request(&request, ingress_at, WorkerInputs::NONE)
        .0;

    // The only cache-bearing worker disappears while the classification is
    // pending; enqueue-time state governs, so the request buckets as
    // uncached — exactly as an unclassified request admitted now would.
    cfg_tx.send(HashMap::new()).unwrap();
    let metadata = queue
        .validate_classification(&mut request, classified, ingress_at)
        .unwrap();
    assert_eq!(
        queue.profile().class(metadata.class_index).name,
        "latency_uncached"
    );
}

#[tokio::test]
async fn mixed_classes_recheck_capacity_after_each_admission() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: uncapped
    policy_family: uncapped
    cache_bucket: all
    quantum: 1
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);
    let worker = WorkerWithDpRank::new(0, 0);
    let (mut active, active_rx) = make_request("active", 64);
    active.policy_class = Some("uncapped".into());
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (mut first, mut first_rx) = make_request("first", 64);
    first.pinned_worker = Some(worker);
    queue.enqueue(first).await;
    let (second, mut second_rx) = make_request("second", 64);
    queue.enqueue(second).await;
    assert_eq!(queue.pending_count(), 2);

    // The uncapped class can still run while the capped class waits.
    let (mut uncapped, uncapped_rx) = make_request("uncapped", 64);
    uncapped.policy_class = Some("uncapped".into());
    queue.enqueue(uncapped).await;
    assert_eq!(uncapped_rx.await.unwrap().unwrap().best_worker, worker);
    assert_eq!(queue.pending_count(), 2);
    assert!(matches!(
        first_rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert!(matches!(
        second_rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));

    for id in ["active", "uncapped"] {
        slots
            .mark_prefill_completed(&id.to_string(), decay_now())
            .unwrap();
        slots.free(&id.to_string(), decay_now()).unwrap();
    }
    queue.update().await;
    assert_eq!(first_rx.try_recv().unwrap().unwrap().best_worker, worker);
    // Reusing the pre-admission snapshot would also admit the second request.
    assert!(matches!(
        second_rx.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    assert_eq!(queue.pending_count(), 1);

    slots
        .mark_prefill_completed(&"first".to_string(), decay_now())
        .unwrap();
    slots.free(&"first".to_string(), decay_now()).unwrap();
    queue.update().await;
    assert_eq!(second_rx.try_recv().unwrap().unwrap().best_worker, worker);
    assert_eq!(queue.pending_count(), 0);
    slots
        .mark_prefill_completed(&"second".to_string(), decay_now())
        .unwrap();
    slots.free(&"second".to_string(), decay_now()).unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test]
async fn zero_class_queue_limit_never_queues() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 0
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

    // An idle worker admits a zero-limit request directly, bypassing the
    // policy queue.
    let (admitted, admitted_rx) = make_request("admitted-idle", 64);
    queue.enqueue(admitted).await;
    admitted_rx.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0);

    // With every worker busy the request would have to queue, so the zero
    // limit rejects it instead of parking it.
    let (rejected, rejected_rx) = make_request("rejected-busy", 64);
    queue.enqueue(rejected).await;
    assert!(matches!(
        rejected_rx.await.unwrap(),
        Err(KvSchedulerError::QueueRejected(_))
    ));
    assert_eq!(queue.pending_count(), 0);

    slots
        .mark_prefill_completed(&"admitted-idle".to_string(), decay_now())
        .unwrap();
    slots
        .free(&"admitted-idle".to_string(), decay_now())
        .unwrap();
    slots.assert_completely_drained(decay_now());
}

#[tokio::test]
async fn zero_token_queue_limit_admits_idle_workers() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    raw_isl_token_queue_limit_per_worker: 0
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);

    // A zero token limit means the class can never hold a queued entry, so
    // an idle worker must still admit directly instead of tripping the
    // limit-zero rejection.
    let (admitted, admitted_rx) = make_request("admitted-idle-token-limit", 64);
    queue.enqueue(admitted).await;
    admitted_rx.await.unwrap().unwrap();
    assert_eq!(queue.pending_count(), 0);

    let (rejected, rejected_rx) = make_request("rejected-busy-token-limit", 64);
    queue.enqueue(rejected).await;
    assert!(matches!(
        rejected_rx.await.unwrap(),
        Err(KvSchedulerError::QueueRejected(_))
    ));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test]
async fn zero_workers_report_no_endpoints_not_queue_rejection() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 4
"#,
    );
    let (queue, _slots) = make_queue_with_profile(0, 16, 64, profile);

    // With zero discovered workers every per-worker limit scales to zero;
    // the request must surface the real condition instead of a
    // `QueueRejected { limit: 0 }`.
    let (request, response_rx) = make_request("no-workers", 64);
    queue.enqueue(request).await;
    assert!(matches!(
        response_rx.await.unwrap(),
        Err(KvSchedulerError::NoEndpoints)
    ));
    assert_eq!(queue.pending_count(), 0);
}

#[tokio::test]
async fn direct_release_wakes_queued_and_mixed_classes_with_full_admission_channel() {
    let mixed = policy_profile(
        r#"
default_policy_family: queued
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: direct
    policy_family: direct
    cache_bucket: all
    quantum: 1
  - name: queued
    policy_family: queued
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 32
"#,
    );
    for profile in [
        PolicyProfile::synthetic(Some(0.5), RouterQueuePolicy::Fcfs),
        mixed,
    ] {
        for overflow in [false, true] {
            let (queue, slots) = make_queue_with_profile(1, 16, 64, profile.clone());
            let (mut active, active_rx) = make_request("active", 64);
            // In the mixed profile, capacity released by a direct class
            // must wake requests waiting in the queued class.
            active.policy_class = Some("direct".into());
            let (attempt_tx, attempt_rx) = oneshot::channel();
            queue
                .enqueue_admitted_with_block_hashes_and_lease(
                    active,
                    None,
                    None,
                    Some(attempt_tx),
                    None,
                    Instant::now(),
                )
                .await;
            let worker = active_rx.await.unwrap().unwrap().best_worker;
            let booking = SchedulerBookingDescriptor {
                request_id: "active".into(),
                worker,
                attempt_id: attempt_rx.await.unwrap(),
            };
            let (mut queued, queued_rx) = make_request("queued", 64);
            queued.pinned_worker = Some(worker);
            queue.enqueue(queued).await;
            assert_eq!(queue.pending_count(), 1);

            // Hold every admission permit. Capacity progress must not need
            // space in the command channel, or a periodic update task.
            let permits = queue.hold_admission_permits().await;
            let released = queue.booking_cleanup().enqueue_acknowledged(booking);
            assert!(released.response().is_ok());
            assert_eq!(slots.active_tokens(Instant::now())[&worker], 0);
            if overflow {
                for _ in 0..300 {
                    queue.capacity_changed(Some(worker));
                }
            }
            let response = tokio::time::timeout(Duration::from_secs(1), queued_rx)
                .await
                .expect("capacity wake must drain without command permits")
                .unwrap()
                .unwrap();
            assert_eq!(response.best_worker, worker);
            assert_eq!(queue.pending_count(), 0);
            drop(permits);
            slots.free(&"queued".to_owned(), Instant::now()).unwrap();
            queue.update().await;
            slots.assert_completely_drained(Instant::now());
        }
    }
}

#[tokio::test]
async fn decode_limit_holds_new_work_until_held_kv_frees() {
    let profile = policy_profile(
        r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: decode-bound
    policy_family: standard
    cache_bucket: all
    quantum: 1
    decode_busy_threshold_frac: 0.5
"#,
    );
    let (queue, slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 1_000_000, profile);
    cfg_tx
        .send(HashMap::from([(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(1_000_000),
                total_kv_blocks: Some(8),
                ..Default::default()
            },
        )]))
        .unwrap();

    // A decoding request holds 6 of the worker's 8 KV blocks; its prefill load is gone.
    let (mut held, held_rx) = make_request("held", 96);
    held.token_seq = Some((0..6).collect());
    queue.enqueue(held).await;
    held_rx.await.unwrap().unwrap();
    slots
        .mark_prefill_completed(&"held".to_string(), decay_now())
        .unwrap();

    let (next, mut next_rx) = make_request("next", 16);
    queue.enqueue(next).await;
    queue.update().await;
    assert!(
        next_rx.try_recv().is_err(),
        "held KV above the decode limit must keep new work queued"
    );

    slots.free(&"held".to_string(), decay_now()).unwrap();
    queue.update().await;
    next_rx
        .await
        .unwrap()
        .expect("freed KV must release the queued request");
}

#[tokio::test(flavor = "multi_thread")]
async fn policy_classes_apply_independent_thresholds_and_preserve_backlog_order() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
  - name: bulk
    policy_family: bulk
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 1024
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

    let (mut active, active_rx) = make_request("active", 64);
    active.policy_class = Some("latency".to_string());
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (mut bulk, bulk_rx) = make_request("bulk", 64);
    bulk.policy_class = Some("bulk".to_string());
    queue.enqueue(bulk).await;
    bulk_rx.await.unwrap().unwrap();

    let (mut queued_first, mut queued_first_rx) = make_request("queued-first", 64);
    queued_first.policy_class = Some("latency".to_string());
    queue.enqueue(queued_first).await;
    assert_eq!(queue.pending_count(), 1);

    for request_id in ["active", "bulk"] {
        slots
            .mark_prefill_completed(&request_id.to_string(), decay_now())
            .unwrap();
        slots.free(&request_id.to_string(), decay_now()).unwrap();
    }

    let (mut queued_second, mut queued_second_rx) = make_request("queued-second", 64);
    queued_second.policy_class = Some("latency".to_string());
    queue.enqueue(queued_second).await;
    assert_eq!(
        queue.pending_count(),
        1,
        "enqueue should admit the existing head before the new arrival"
    );
    queued_first_rx
        .try_recv()
        .expect("first queued request should be admitted")
        .expect("first queued request failed");
    assert!(
        queued_second_rx.try_recv().is_err(),
        "second request should remain behind the admitted head"
    );

    slots
        .mark_prefill_completed(&"queued-first".to_string(), decay_now())
        .unwrap();
    slots
        .free(&"queued-first".to_string(), decay_now())
        .unwrap();
    queue.update().await;
    queued_second_rx.await.unwrap().unwrap();
}

// No-bypass ordering per DEP #13891: freed capacity goes to the queued
// head, never to the arriving request.
#[tokio::test]
async fn arrival_drains_other_class_backlog_instead_of_bypassing() {
    let profile = policy_profile(
        r#"
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 1024
  - name: bulk
    policy_family: bulk
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
"#,
    );
    let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

    let (mut active, active_rx) = make_request("active", 64);
    active.policy_class = Some("latency".to_string());
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    // The bulk class sees the worker as busy, so its request parks.
    let (mut bulk_head, mut bulk_head_rx) = make_request("bulk-head", 64);
    bulk_head.policy_class = Some("bulk".to_string());
    queue.enqueue(bulk_head).await;
    assert_eq!(queue.pending_count(), 1);

    // Free the worker without emitting a capacity update; the parked head
    // stays queued until something drains.
    slots
        .mark_prefill_completed(&"active".to_string(), decay_now())
        .unwrap();
    slots.free(&"active".to_string(), decay_now()).unwrap();
    assert_eq!(queue.pending_count(), 1);

    // A latency arrival must not bypass the bulk backlog: its enqueue-turn
    // drain admits the parked bulk head alongside the arrival itself.
    let (mut arrival, arrival_rx) = make_request("latency-arrival", 64);
    arrival.policy_class = Some("latency".to_string());
    queue.enqueue(arrival).await;
    assert_eq!(queue.pending_count(), 0);
    bulk_head_rx
        .try_recv()
        .expect("parked bulk head should drain on the arrival's turn")
        .expect("bulk head failed");
    arrival_rx.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn policy_families_and_cache_buckets_select_physical_queues() {
    let profile = policy_profile(
        r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 0
  - name: custom_priority
    quantum: 1
    prefill_busy_threshold: 0
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
    let worker = WorkerWithDpRank::new(0, 0);

    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (mut latency_cached, _latency_cached_rx) = make_request("latency-cached", 64);
    latency_cached.policy_class = Some("latency".to_string());
    latency_cached
        .overlap
        .effective_cached_tokens
        .insert(worker, 64);
    queue.enqueue(latency_cached).await;

    let (mut latency_uncached, _latency_uncached_rx) = make_request("latency-uncached", 64);
    latency_uncached.policy_class = Some("latency".to_string());
    queue.enqueue(latency_uncached).await;

    let (mut unknown_cached, _unknown_cached_rx) = make_request("unknown-cached", 64);
    unknown_cached.policy_class = Some("unknown".to_string());
    unknown_cached
        .overlap
        .effective_cached_tokens
        .insert(worker, 64);
    queue.enqueue(unknown_cached).await;

    let (mut ordinary_class_name, _ordinary_class_name_rx) =
        make_request("ordinary-class-name", 64);
    ordinary_class_name.policy_class = Some("latency_cached".to_string());
    queue.enqueue(ordinary_class_name).await;

    let (mut custom, _custom_rx) = make_request("custom", 64);
    custom.policy_class = Some("custom_priority".to_string());
    queue.enqueue(custom).await;

    assert_eq!(
        queue.class_queue_stats(0),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 64,
        })
    );
    assert_eq!(
        queue.class_queue_stats(1),
        Some(ClassQueueStats {
            received_total: 2,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
    assert_eq!(
        queue.class_queue_stats(2),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 64,
        })
    );
    assert_eq!(
        queue.class_queue_stats(3),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
    assert_eq!(
        queue.class_queue_stats(4),
        Some(ClassQueueStats {
            received_total: 1,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn class_local_limit_rejection_is_typed_and_not_overload() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
    );
    let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);

    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (queued, _queued_rx) = make_request("queued", 64);
    queue.enqueue(queued).await;

    let (rejected, rejected_rx) = make_request("rejected", 64);
    queue.enqueue(rejected).await;
    let error = rejected_rx.await.unwrap().unwrap_err();
    let KvSchedulerError::QueueRejected(rejection) = &error else {
        panic!("expected queue rejection, got {error:?}");
    };
    assert_eq!(rejection.policy_class, "capped");
    assert_eq!(
        rejection.limit_kind,
        dynamo_admission::queue::QueueLimitKind::Requests
    );
    assert_eq!(rejection.current, 1);
    assert_eq!(rejection.limit, 1);
    assert!(!error.is_overload());

    assert_eq!(
        queue.class_queue_stats(0),
        Some(ClassQueueStats {
            received_total: 3,
            rejected_due_time_passed_total: 0,
            pending_count: 1,
            pending_isl_tokens: 64,
            pending_cached_tokens: 0,
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn per_worker_limit_tracks_discovered_worker_count_without_evicting() {
    let profile = policy_profile(
        r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
    );
    let (queue, _slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 64, profile);

    let (active, active_rx) = make_request("active", 64);
    queue.enqueue(active).await;
    active_rx.await.unwrap().unwrap();

    let (first, mut first_rx) = make_request("first", 64);
    queue.enqueue(first).await;

    cfg_tx.send_modify(|configs| {
        configs.insert(
            1,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
    });
    let (second, _second_rx) = make_request("second", 64);
    queue.enqueue(second).await;
    first_rx
        .try_recv()
        .expect("new worker capacity should admit the existing head")
        .expect("first queued request failed");
    assert_eq!(queue.pending_count(), 1);

    cfg_tx.send_modify(|configs| {
        configs.remove(&1);
    });
    let (rejected, rejected_rx) = make_request("rejected", 64);
    queue.enqueue(rejected).await;
    let error = rejected_rx.await.unwrap().unwrap_err();
    let KvSchedulerError::QueueRejected(rejection) = error else {
        panic!("expected queue rejection, got {error:?}");
    };
    assert_eq!(rejection.current, 1);
    assert_eq!(rejection.limit, 1);
    assert_eq!(queue.pending_count(), 1);
}
