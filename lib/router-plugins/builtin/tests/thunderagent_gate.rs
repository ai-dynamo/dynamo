// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The host-side pieces this test drives (the local scheduler, its queue, the selector, and the
// slot tracker) are reached through the pre-split paths the composition crate keeps.
#![allow(deprecated)]

//! ThunderAgent through the router's queue: the classifier records turns, the queue holds them
//! in per-session lanes, and the dispatch gate releases them as the program ledger allows.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use dynamo_kv_router::plugins::request_classifier::RequestClassifierContext;
use dynamo_kv_router::plugins::worker_selection::{WorkerCapacity, WorkerInfo};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
use dynamo_kv_router::scheduling::config::RouterQueuePolicy;
use dynamo_kv_router::scheduling::overlap_refresh::NoopOverlapScoresRefresh;
use dynamo_kv_router::scheduling::selector::DefaultWorkerSelector;
use dynamo_kv_router::scheduling::{
    AdmittedSchedulingResponse, KvSchedulerError, LocalScheduler, OverlapSignals, PolicyProfile,
    RequestLifecycle, ScheduleMode, ScheduleRequest, SessionContext,
};
use dynamo_kv_router::test_utils::SimpleWorkerConfig;
use dynamo_kv_router::{ActiveSequencesMultiWorker, KvRouterConfig, NoopSequencePublisher};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

type Scheduler = LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, DefaultWorkerSelector>;
type Admission =
    Pin<Box<dyn Future<Output = Result<AdmittedSchedulingResponse, KvSchedulerError>>>>;

const BLOCK_SIZE: u32 = 64;
fn worker() -> WorkerWithDpRank {
    WorkerWithDpRank::new(0, 0)
}

/// One worker whose KV capacity is `kv_blocks * BLOCK_SIZE` tokens, with the ThunderAgent
/// classifier and selector installed from router-policy YAML.
fn scheduler(kv_blocks: u64) -> (Arc<Scheduler>, CancellationToken) {
    scheduler_with_threshold(kv_blocks, Some(0.5))
}

/// As [`scheduler`], with the router queue's busy threshold; `None` is the documented launch,
/// where no class queues on capacity and only the gate holds turns.
fn scheduler_with_threshold(
    kv_blocks: u64,
    busy_threshold: Option<f64>,
) -> (Arc<Scheduler>, CancellationToken) {
    let config = SimpleWorkerConfig {
        max_num_batched_tokens: Some(10_000),
        total_kv_blocks: Some(kv_blocks),
        ..Default::default()
    };
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        BLOCK_SIZE as usize,
        HashMap::from([(0, (0, 1))]),
        false,
        0,
        "test",
    ));
    let (_configs, config_rx) = watch::channel(HashMap::from([(0, config)]));
    // Keep the sender alive for the scheduler's lifetime.
    std::mem::forget(_configs);
    let cancel = CancellationToken::new();
    let scheduler = Arc::new(LocalScheduler::new(
        slots,
        config_rx,
        PolicyProfile::synthetic(busy_threshold, RouterQueuePolicy::Fcfs),
        BLOCK_SIZE,
        DefaultWorkerSelector::new(None, "test"),
        None,
        None::<Arc<NoopOverlapScoresRefresh>>,
        None,
        None,
        None,
        Duration::from_secs(60),
        true,
        cancel.clone(),
        "test",
        false,
    ));

    let policy = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        policy.path(),
        r#"
request_classifier:
  type: thunderagent
  parameters:
    buffer_per_program: 100
    resume_timeout_seconds: 600
worker_selection:
  aggregated: thunderagent
  instances:
    - name: thunderagent
      type: thunderagent
"#,
    )
    .unwrap();
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).unwrap();
    let plugins = registry
        .resolve_plugins(&KvRouterConfig {
            router_policy_config: Some(policy.path().display().to_string()),
            ..Default::default()
        })
        .unwrap();
    let context = RequestClassifierContext::new(BLOCK_SIZE, move || {
        vec![WorkerInfo::new(
            worker(),
            WorkerCapacity::new(Some(10_000), Some(kv_blocks)),
        )]
    });
    let classifier = plugins.request_classifier().unwrap()(context);
    assert!(scheduler.install_request_classifier(classifier, cancel.clone()));
    (scheduler, cancel)
}

fn turn(request_id: &str, session_id: &str, isl_tokens: usize, last: bool) -> ScheduleRequest {
    ScheduleRequest {
        mode: ScheduleMode::TrackedWithLifecycle {
            request_id: request_id.into(),
        },
        token_seq: None,
        block_hashes: None,
        isl_tokens,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        accounting_attribution: Default::default(),
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: Some(SessionContext::new(
            session_id.into(),
            None,
            Some(last),
            None,
        )),
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
    }
}

fn poll_once(admission: &mut Admission) -> Poll<Option<WorkerWithDpRank>> {
    match admission
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(Ok(admitted)) => Poll::Ready(Some(admitted.response.best_worker)),
        Poll::Ready(Err(error)) => panic!("admission failed: {error}"),
        Poll::Pending => Poll::Pending,
    }
}

struct Harness {
    scheduler: Arc<Scheduler>,
    lifecycles: HashMap<String, RequestLifecycle>,
}

impl Harness {
    /// Registers the lifecycle, submits the turn, and lets the actor process it.
    async fn submit(&mut self, request: ScheduleRequest) -> Admission {
        let request_id = request.mode.request_id().unwrap().to_owned();
        let lifecycle = self
            .scheduler
            .begin_request_lifecycle(&request_id)
            .unwrap()
            .unwrap();
        self.lifecycles.insert(request_id, lifecycle);
        let scheduler = Arc::clone(&self.scheduler);
        let mut admission: Admission =
            Box::pin(async move { scheduler.schedule_request_admitted(request).await });
        assert!(poll_once(&mut admission).is_pending());
        self.scheduler.admission_barrier().await.unwrap();
        admission
    }

    /// The turn reached its worker and finished; the lifecycle's events reach the plugin on
    /// its delivery task.
    fn finish(&mut self, request_id: &str) {
        let lifecycle = self.lifecycles.get_mut(request_id).unwrap();
        lifecycle.sent(worker());
        lifecycle.complete();
    }

    /// Polls `admission` across actor turns until it resolves, or fails after `rounds`.
    async fn admitted(&self, admission: &mut Admission, rounds: usize) -> WorkerWithDpRank {
        for _ in 0..rounds {
            if let Poll::Ready(Some(worker)) = poll_once(admission) {
                return worker;
            }
            tokio::task::yield_now().await;
            self.scheduler.admission_barrier().await.unwrap();
        }
        panic!("the turn was not admitted");
    }

    async fn still_held(&self, admission: &mut Admission) -> bool {
        self.scheduler.admission_barrier().await.unwrap();
        poll_once(admission).is_pending()
    }
}

#[tokio::test]
async fn held_turns_wait_in_their_own_lane_and_go_when_the_ledger_frees_room() {
    // 8 blocks of 64 tokens: room for one 200-token program plus its 100-token buffer.
    let (scheduler, cancel) = scheduler(8);
    let mut harness = Harness {
        scheduler,
        lifecycles: HashMap::new(),
    };

    // Session a's first turn goes at once, placed by the gate on the only worker.
    let mut a1 = harness.submit(turn("a1", "a", 200, false)).await;
    assert_eq!(harness.admitted(&mut a1, 1).await, worker());

    // Session b does not fit beside a; session a's final turn waits for a's first to finish.
    let mut b1 = harness.submit(turn("b1", "b", 200, false)).await;
    let mut a2 = harness.submit(turn("a2", "a", 50, true)).await;
    assert!(harness.still_held(&mut b1).await);
    assert!(harness.still_held(&mut a2).await);
    assert_eq!(harness.scheduler.pending_count(), 2);

    // a1 completes: a's lane is released first, and its final turn frees the program, so b's
    // lane is released next. Neither needed a timer.
    harness.finish("a1");
    assert_eq!(harness.admitted(&mut a2, 64).await, worker());
    assert_eq!(harness.admitted(&mut b1, 64).await, worker());
    assert_eq!(harness.scheduler.pending_count(), 0);

    harness.finish("a2");
    harness.finish("b1");
    cancel.cancel();
}

#[tokio::test]
async fn a_held_session_does_not_block_a_fitting_one_queued_behind_it() {
    // Room for two programs (300 each) out of three.
    let (scheduler, cancel) = scheduler(10);
    let mut harness = Harness {
        scheduler,
        lifecycles: HashMap::new(),
    };

    let mut a1 = harness.submit(turn("a1", "a", 200, false)).await;
    assert_eq!(harness.admitted(&mut a1, 1).await, worker());
    // b arrives before c and is held: it needs 500 of the 340 left. c is smaller and fits;
    // without session lanes b's head would have blocked it.
    let mut b1 = harness.submit(turn("b1", "b", 400, false)).await;
    let mut c1 = harness.submit(turn("c1", "c", 100, false)).await;
    assert!(harness.still_held(&mut b1).await);
    assert_eq!(harness.admitted(&mut c1, 4).await, worker());
    assert!(harness.still_held(&mut b1).await);
    assert_eq!(harness.scheduler.pending_count(), 1);

    harness.finish("a1");
    harness.finish("c1");
    cancel.cancel();
}

#[tokio::test]
async fn the_gate_holds_turns_without_a_queue_threshold() {
    let (scheduler, cancel) = scheduler_with_threshold(8, None);
    let mut harness = Harness {
        scheduler,
        lifecycles: HashMap::new(),
    };

    // One turn per session in flight: a's second turn waits for its first, and the gate's
    // recheck releases it when a1 completes.
    let mut a1 = harness.submit(turn("a1", "a", 200, false)).await;
    assert_eq!(harness.admitted(&mut a1, 1).await, worker());
    let mut a2 = harness.submit(turn("a2", "a", 50, true)).await;
    assert!(harness.still_held(&mut a2).await);
    assert_eq!(harness.scheduler.pending_count(), 1);

    harness.finish("a1");
    assert_eq!(harness.admitted(&mut a2, 64).await, worker());
    assert_eq!(harness.scheduler.pending_count(), 0);

    harness.finish("a2");
    cancel.cancel();
}
