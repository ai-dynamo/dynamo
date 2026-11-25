// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request cost of admission under a held backlog, through the kernel's `LocalScheduler`:
//! the request classifier, the queue actor, the policy queue, selection, and booking.
//!
//! `selection_core` times a request that is admitted at once; `policy_queue` times the queue's
//! data structure alone. This bench holds every worker full with `backlog` requests waiting, and
//! times one steady-state cycle as a router under load runs it:
//!
//! 1. a new request arrives, is classified, and joins the queue;
//! 2. a booked request completes and frees its worker;
//! 3. the capacity change makes the actor re-ask its blocked heads, dispatch one, select a worker
//!    for it, and book it.
//!
//! Every worker holds one request and a request fills its worker, so each cycle admits exactly
//! one request and the backlog stays at `backlog`.
//!
//! Read a time as a floor, and compare two cases by their difference in microseconds, not
//! their ratio. Every request carries the same four block hashes and the fleet is 64 workers,
//! so the part of a cycle that does not depend on the backlog (storing and removing the
//! request's blocks, the overlap walk, the projection of each rank's load) is far below a real
//! request's. A free also admits exactly one request, so the rechecks per admission are a
//! lower bound.
//!
//! The cases differ in where the backlog waits:
//!
//! - `shared`: no classifier; one class, one shared heap.
//! - `session_lanes`: a classifier keys each request's lane by its session, one session per
//!   request, so the backlog is `backlog` keyed lanes that share one capacity answer.
//! - `pinned`: every request is pinned to a worker, so the backlog is one lane per worker.
//! - `classes`: four classes of weights 1:2:3:4 share the backlog and take deficit-round-robin
//!   turns. One slot frees per cycle and the ring moves to the next class after every
//!   dispatch, so the classes are served in turn whatever their weights: the case times a
//!   backlog spread over four class queues, not weighted shares.
//! - `constrained_lanes`: as `session_lanes`, but each session may run on one worker only, as a
//!   caller allow-list, a LoRA, or a required taint restricts it. Such a lane has its own
//!   capacity answer, so a worker that frees re-asks every one of them.
//! - `gate_held`: as `session_lanes`, with a dispatch gate that holds `backlog` further lanes
//!   for good. A worker that frees must not re-ask them, so this should cost what
//!   `session_lanes` does.
//! - `gate_wake`: as `gate_held`, but the gate wakes the queue on every completion. Each wake
//!   re-asks every held lane and every waiting lane, so this is the cost that grows with the
//!   backlog. The wake follows the free, so it finds the freed slot taken and every waiting
//!   lane stays blocked, as a wake at a request's dispatch does. ThunderAgent wakes the queue
//!   whenever its ledger changes, which one request can cause three times: at its dispatch,
//!   when it is sent, and when it completes or aborts. This case issues one wake per request,
//!   so it is a lower bound for that plugin.
//! - `gate_wake_room`: as `gate_wake`, but a request's lifecycle ends before its slot frees,
//!   as the frontend ends it. On this bench's one thread the wake and the free then reach the
//!   actor together, so the recheck finds a slot free: every waiting lane may go, the best of
//!   them is admitted, and the rest stay where they are. On the frontend's runtime the wake
//!   can also arrive after the free, which is `gate_wake`: the two cases bound what a wake
//!   costs.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench admission_core --features bench`

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_admission::{
    ClassifierError, ClassifyEvent, ClassifyRequest, DispatchDecision, DispatchGate, DispatchHead,
    DispatchRecheck, PolicyProfile, RequestClassifier, RequestLifecycle, RouterQueuePolicy,
};
use dynamo_kv_router::scheduling::policy_config::RouterPolicyConfig;
use dynamo_router_protocol::session::SessionContext;
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_scheduler::test_utils::{
    SimpleWorkerConfig, TestScheduler, local_test_scheduler, test_schedule_request,
};
use dynamo_scheduler::{ScheduleMode, ScheduleRequest};
use dynamo_worker_selection::WorkerCandidates;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

const BLOCK_SIZE: u32 = 64;
/// A request's prompt, and a worker's whole batched-token budget: one booking fills a worker.
const REQUEST_TOKENS: usize = 64;
const WORKERS: u64 = 64;
/// A class is busy once a worker's booked prefill passes half its budget.
const BUSY_THRESHOLD: f64 = 0.5;
const CLASSES: [&str; 4] = ["weight1", "weight2", "weight3", "weight4"];

/// Keys each request's lane by its session, as ThunderAgent does, and installs `gate` if any.
struct SessionLanes {
    gate: Option<Arc<HoldsHeldSessions>>,
}

#[async_trait]
impl RequestClassifier for SessionLanes {
    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        _workers: WorkerCandidates<'_>,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        let session = request.info().session_context();
        if let Some(session) = session.map(|s| s.session_id().to_owned()) {
            request.set_lane_key(&session);
        }
        Ok(request)
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        if let (Some(gate), ClassifyEvent::Completed { .. }) = (&self.gate, &event) {
            gate.completed();
        }
    }

    fn dispatch_gate(&self) -> Option<Arc<dyn DispatchGate>> {
        self.gate.clone().map(|gate| gate as Arc<dyn DispatchGate>)
    }
}

/// Holds every session whose id starts with `held`. With `wakes` it asks the queue to re-ask
/// its held heads whenever a request completes, and counts the times it asked.
struct HoldsHeldSessions {
    wakes: bool,
    woke: AtomicU64,
    recheck: OnceLock<DispatchRecheck>,
}

impl HoldsHeldSessions {
    fn new(wakes: bool) -> Arc<Self> {
        Arc::new(Self {
            wakes,
            woke: AtomicU64::new(0),
            recheck: OnceLock::new(),
        })
    }

    fn completed(&self) {
        if self.wakes
            && let Some(recheck) = self.recheck.get()
        {
            recheck.notify();
            self.woke.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl DispatchGate for HoldsHeldSessions {
    fn check(&self, head: &DispatchHead<'_>) -> DispatchDecision {
        let held = head
            .info()
            .session_context()
            .is_some_and(|session| session.session_id().starts_with("held"));
        if held {
            DispatchDecision::Hold { until: None }
        } else {
            DispatchDecision::Go { target: None }
        }
    }

    fn attach(&self, recheck: DispatchRecheck) {
        let _ = self.recheck.set(recheck);
    }
}

#[derive(Clone, Copy)]
enum Case {
    Shared,
    SessionLanes,
    Pinned,
    Classes,
    ConstrainedLanes,
    GateHeld,
    GateWake,
    GateWakeRoom,
}

impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::SessionLanes => "session_lanes",
            Self::Pinned => "pinned",
            Self::Classes => "classes",
            Self::ConstrainedLanes => "constrained_lanes",
            Self::GateHeld => "gate_held",
            Self::GateWake => "gate_wake",
            Self::GateWakeRoom => "gate_wake_room",
        }
    }

    fn profile(self) -> PolicyProfile {
        match self {
            Self::Classes => {
                let classes: String = CLASSES
                    .iter()
                    .enumerate()
                    .map(|(index, name)| {
                        format!(
                            "  - name: {name}\n    queue_policy: fcfs\n    quantum: {}\n    \
                             prefill_busy_threshold_frac: {BUSY_THRESHOLD}\n",
                            (index + 1) * REQUEST_TOKENS
                        )
                    })
                    .collect();
                RouterPolicyConfig::from_yaml(&format!(
                    "default_policy_class: {}\npolicy_classes:\n{classes}",
                    CLASSES[0]
                ))
                .expect("policy yaml")
                .resolve_profile(None, None, RouterQueuePolicy::Fcfs)
            }
            _ => PolicyProfile::synthetic(Some(BUSY_THRESHOLD), RouterQueuePolicy::Fcfs),
        }
    }

    /// Whether the case installs a classifier.
    fn classified(self) -> bool {
        !matches!(self, Self::Shared | Self::Pinned | Self::Classes)
    }

    fn classifier(self) -> Option<SessionLanes> {
        let gate = self
            .holds_lanes()
            .then(|| HoldsHeldSessions::new(self.wakes()));
        self.classified().then_some(SessionLanes { gate })
    }

    /// Whether the gate holds `backlog` lanes on top of the `backlog` that waits for capacity.
    fn holds_lanes(self) -> bool {
        matches!(self, Self::GateHeld | Self::GateWake | Self::GateWakeRoom)
    }

    /// Whether the gate wakes the queue on every completion.
    fn wakes(self) -> bool {
        matches!(self, Self::GateWake | Self::GateWakeRoom)
    }

    fn request(self, sequence: u64) -> ScheduleRequest {
        let mut request = request(format!("request-{sequence}"), self.classified());
        match self {
            Self::Shared => {}
            Self::SessionLanes | Self::GateHeld | Self::GateWake | Self::GateWakeRoom => {
                request.session_context = Some(session(format!("session-{sequence}")));
            }
            // Frees go round the workers in booking order, so each lane's head is next in turn.
            Self::Pinned => {
                request.pinned_worker = Some(WorkerWithDpRank::new(sequence % WORKERS, 0));
            }
            Self::ConstrainedLanes => {
                request.session_context = Some(session(format!("session-{sequence}")));
                request.allowed_worker_ids = Some(HashSet::from([sequence % WORKERS]));
            }
            Self::Classes => {
                request.policy_class = Some(CLASSES[sequence as usize % CLASSES.len()].to_string());
            }
        }
        request
    }
}

fn session(id: String) -> SessionContext {
    SessionContext::new(id, None, None, None)
}

fn request(request_id: String, classified: bool) -> ScheduleRequest {
    // Only a request with a lifecycle reaches the classifier.
    let mode = if classified {
        ScheduleMode::TrackedWithLifecycle { request_id }
    } else {
        ScheduleMode::Tracked { request_id }
    };
    ScheduleRequest {
        token_seq: Some(vec![1, 2, 3, 4]),
        ..test_schedule_request(mode, REQUEST_TOKENS)
    }
}

/// A request the scheduler admitted and booked on a worker.
struct Booked {
    request_id: String,
    lifecycle: Option<RequestLifecycle>,
}

/// A scheduler whose workers are all full, with `backlog` requests waiting.
struct Loaded {
    case: Case,
    scheduler: Arc<TestScheduler>,
    gate: Option<Arc<HoldsHeldSessions>>,
    booked: VecDeque<Booked>,
    admitted_tx: mpsc::UnboundedSender<Booked>,
    admitted_rx: mpsc::UnboundedReceiver<Booked>,
    sequence: u64,
    _configs: watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    _cancel: CancellationToken,
}

impl Loaded {
    async fn new(case: Case, backlog: u64) -> Self {
        let configs: HashMap<u64, SimpleWorkerConfig> = (0..WORKERS)
            .map(|worker_id| {
                let config = SimpleWorkerConfig {
                    max_num_batched_tokens: Some(REQUEST_TOKENS as u64),
                    ..Default::default()
                };
                (worker_id, config)
            })
            .collect();
        let cancel = CancellationToken::new();
        let (scheduler, config_tx) =
            local_test_scheduler(configs, case.profile(), BLOCK_SIZE, cancel.clone());
        let classifier = case.classifier();
        let gate = classifier.as_ref().and_then(|lanes| lanes.gate.clone());
        if let Some(classifier) = classifier {
            assert!(scheduler.install_request_classifier(Box::new(classifier), cancel.clone()));
        }

        let (admitted_tx, admitted_rx) = mpsc::unbounded_channel();
        let mut loaded = Self {
            case,
            scheduler,
            gate,
            booked: VecDeque::new(),
            admitted_tx,
            admitted_rx,
            sequence: 0,
            _configs: config_tx,
            _cancel: cancel,
        };

        // Fill every worker, in worker order, so the bookings free in that order. A fill has a
        // lifecycle where the case has a classifier, so its completion is an event like any other.
        for worker_id in 0..WORKERS {
            let mut fill = request(format!("fill-{worker_id}"), case.classified());
            fill.pinned_worker = Some(WorkerWithDpRank::new(worker_id, 0));
            loaded.submit(fill);
            let booked = loaded.admitted_rx.recv().await.expect("fill admitted");
            loaded.booked.push_back(booked);
        }
        if case.holds_lanes() {
            for held in 0..backlog {
                let mut request = request(format!("held-{held}"), true);
                request.session_context = Some(session(format!("held-{held}")));
                loaded.submit(request);
            }
        }
        for _ in 0..backlog {
            loaded.submit_next();
        }
        // The admission channel is bounded, so a backlog takes several actor turns to queue.
        let waiting = backlog * if case.holds_lanes() { 2 } else { 1 };
        for _ in 0..waiting {
            if loaded.scheduler.pending_count() as u64 == waiting {
                break;
            }
            loaded.settle().await;
        }
        assert_eq!(loaded.scheduler.pending_count() as u64, waiting);
        assert!(
            loaded.admitted_rx.try_recv().is_err(),
            "a full router admitted a request"
        );
        loaded
    }

    fn submit_next(&mut self) {
        let request = self.case.request(self.sequence);
        self.sequence += 1;
        self.submit(request);
    }

    /// Starts one admission as its own task, as a frontend's request task does. The task
    /// reports the booking once the scheduler admits the request.
    fn submit(&self, request: ScheduleRequest) {
        let request_id = request.mode.request_id().expect("tracked").to_owned();
        let mut lifecycle = match &request.mode {
            ScheduleMode::TrackedWithLifecycle { request_id } => self
                .scheduler
                .begin_request_lifecycle(request_id)
                .expect("lifecycle begins"),
            _ => None,
        };
        let scheduler = Arc::clone(&self.scheduler);
        let admitted = self.admitted_tx.clone();
        tokio::spawn(async move {
            let response = scheduler.schedule_request(request).await.expect("admitted");
            if let Some(lifecycle) = lifecycle.as_mut() {
                lifecycle.sent(response.best_worker);
            }
            let _ = admitted.send(Booked {
                request_id,
                lifecycle,
            });
        });
    }

    /// Lets the submitted tasks reach the queue actor and the actor take their commands. Setup
    /// only: the barrier makes the actor re-ask every blocked lane, which an arrival or a
    /// completion does only when a dispatch gate wakes the queue (`gate_wake`).
    async fn settle(&self) {
        tokio::task::yield_now().await;
        self.scheduler.admission_barrier().await.expect("barrier");
    }

    async fn cycle(&mut self) {
        self.submit_next();
        // The arrival's task runs up to its hand-off to the actor.
        tokio::task::yield_now().await;
        let mut done = self.booked.pop_front().expect("a booked request");
        // In `gate_wake_room` the completion's wake is on its way before the slot frees.
        let complete_first = matches!(self.case, Case::GateWakeRoom);
        if complete_first && let Some(lifecycle) = done.lifecycle.as_mut() {
            lifecycle.complete();
        }
        self.scheduler.free(&done.request_id).await.expect("free");
        if !complete_first && let Some(lifecycle) = done.lifecycle.as_mut() {
            lifecycle.complete();
        }
        let booked = self.admitted_rx.recv().await.expect("an admission");
        self.booked.push_back(booked);
    }
}

fn bench_admission(c: &mut Criterion) {
    let mut group = c.benchmark_group("admission_core/arrive_free_admit");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));
    for case in [
        Case::Shared,
        Case::SessionLanes,
        Case::Pinned,
        Case::Classes,
        Case::ConstrainedLanes,
        Case::GateHeld,
        Case::GateWake,
        Case::GateWakeRoom,
    ] {
        for backlog in [1_000u64, 10_000] {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            let mut loaded = runtime.block_on(Loaded::new(case, backlog));
            group.bench_function(BenchmarkId::new(case.name(), backlog), |b| {
                b.iter(|| runtime.block_on(loaded.cycle()));
            });
            // The backlog is still there: every cycle admitted one request and queued one.
            runtime.block_on(loaded.settle());
            let held = if case.holds_lanes() { 2 } else { 1 };
            assert_eq!(loaded.scheduler.pending_count() as u64, backlog * held);
            // Without its wakes `gate_wake` is `gate_held`: each completion must reach the gate
            // as `Completed`, and the gate must hold the scheduler's recheck handle. A name
            // filter that skips the case still sets it up, and then no cycle ran.
            if case.wakes() && loaded.sequence > backlog {
                let gate = loaded.gate.as_ref().expect("a waking case installs a gate");
                assert!(
                    gate.woke.load(Ordering::Relaxed) > 0,
                    "no completion woke the queue"
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_admission);
criterion_main!(benches);
