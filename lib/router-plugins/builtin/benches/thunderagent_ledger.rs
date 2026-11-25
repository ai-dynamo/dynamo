// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The ThunderAgent ledger under the dispatch gate, at program counts the defaults allow.
//!
//! `check` runs once per blocked session lane on every gate wake, and `dispatched` plus the
//! turn's terminal event run once per admitted turn, all on the scheduler actor. Each group holds
//! `programs` acting programs that fill every worker exactly, so a new session's head is held.
//!
//! - `check_held`: one gate check of a held head while no retention deadline has passed.
//! - `recheck_1000_held`: what one gate wake costs the actor: a check per held lane.
//! - `check_held_retention_passed`: the same check once every program's retention deadline has
//!   passed. Built on a paused clock so every program expires at once: the first check settles
//!   them all, the rest size room from the settled charges.
//! - `turn`: `dispatched` for a continuing turn, then its `Completed` event.
//! - `turn_with_paused`: the same turn while programs are paused and every rank has a sliver of
//!   room under the resume ceiling that fits none of them, so resume has nothing to do.
//!
//! Groups run at 4 workers, and a 1,024-program group at 64, since a check for a new or
//! paused program sizes room across ranks.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::experimental::dispatch_gate::{
    DispatchDecision, DispatchGate, DispatchHead,
};
use dynamo_kv_router::plugins::host::RouterPluginRegistryHostExt;
use dynamo_kv_router::plugins::request_classifier::*;
use dynamo_kv_router::plugins::testing::classify_request;

const BLOCK_SIZE: u32 = 64;
const ISL: usize = 1_000;
const HELD: usize = 1_000;
const RETENTION: Duration = Duration::from_secs(3_600);

fn worker(index: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(index, 0)
}

fn session(index: usize) -> SessionContext {
    SessionContext::new(format!("session-{index}"), None, Some(false), None)
}

/// How the ranks are filled relative to the pause threshold.
#[derive(Clone, Copy)]
enum Fill {
    /// Ranks exactly full at a pause threshold of 1.0: nothing pauses, a new session holds.
    Exact,
    /// Ranks over the default threshold, so setup pauses programs down to a target that leaves
    /// room under the resume ceiling smaller than one program.
    OverThreshold,
}

fn classifier(programs: usize, workers: u64, fill: Fill) -> Box<dyn RequestClassifier> {
    let policy = tempfile::NamedTempFile::new().unwrap();
    let thresholds = match fill {
        Fill::Exact => {
            "    pause_threshold: 1.0\n    pause_target: 1.0\n    resume_hysteresis: 0.0\n"
        }
        // Ceiling 0.85, target 0.848: 0.2% of a rank is free under the ceiling.
        Fill::OverThreshold => {
            "    pause_threshold: 0.95\n    pause_target: 0.848\n    resume_hysteresis: 0.10\n"
        }
    };
    std::fs::write(
        policy.path(),
        format!(
            r#"
request_classifier:
  type: thunderagent
  parameters:
    buffer_per_program: 0
    resume_timeout_seconds: 600
    session_retention_seconds: {}
    max_tracked_requests: 1000000
{}worker_selection:
  aggregated: thunderagent
  instances:
    - name: thunderagent
      type: thunderagent
"#,
            RETENTION.as_secs(),
            thresholds
        ),
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
    // Every worker holds exactly its share of the programs: at an exact fill a new program has
    // no room; over the threshold the programs exceed the rank by a ninth.
    let per_worker_tokens = programs.div_ceil(workers as usize) * ISL;
    let per_worker_tokens = match fill {
        Fill::Exact => per_worker_tokens,
        Fill::OverThreshold => per_worker_tokens * 9 / 10,
    };
    let kv_blocks = (per_worker_tokens as u64).div_ceil(u64::from(BLOCK_SIZE));
    let context = RequestClassifierContext::new(BLOCK_SIZE, move || {
        (0..workers)
            .map(|index| {
                WorkerInfo::new(
                    worker(index),
                    WorkerCapacity::new(Some(10_000), Some(kv_blocks)),
                )
            })
            .collect()
    });
    plugins.request_classifier().unwrap()(context)
}

fn classify(classifier: &mut dyn RequestClassifier, request_id: &str, session: &SessionContext) {
    let request = classify_request(ISL, Some(request_id), Some(session.clone()), None);
    dynamo_kv_router::plugins::testing::classify(classifier, request)
        .expect("classification succeeds");
}

fn head<'a>(request_id: &'a str, session: &'a SessionContext) -> DispatchHead<'a> {
    DispatchHead::new(
        Some(request_id),
        Some(session),
        ISL,
        None,
        "default",
        Instant::now(),
    )
}

/// Registers a turn for `session_index` and commits it through the gate.
fn dispatch_turn(
    classifier: &mut dyn RequestClassifier,
    gate: &dyn DispatchGate,
    request_id: &str,
    session_index: usize,
    workers: u64,
) {
    let session = session(session_index);
    classify(classifier, request_id, &session);
    gate.dispatched(
        &head(request_id, &session),
        Some(worker(session_index as u64 % workers)),
    );
}

async fn complete(
    classifier: &mut dyn RequestClassifier,
    request_id: &str,
    session_index: usize,
    workers: u64,
) {
    let worker = worker(session_index as u64 % workers);
    classifier
        .on_event(ClassifyEvent::Sent {
            request_id: request_id.to_owned(),
            worker,
        })
        .await;
    classifier
        .on_event(ClassifyEvent::Completed {
            request_id: request_id.to_owned(),
            worker,
            context_tokens: Some(ISL),
        })
        .await;
}

/// New sessions whose first turn waits at the gate for room that never comes.
fn hold_new_sessions(classifier: &mut dyn RequestClassifier, programs: usize) {
    for index in 0..HELD {
        classify(
            classifier,
            &format!("held-{index}"),
            &session(programs + index),
        );
    }
}

fn check_held(gate: &dyn DispatchGate, programs: usize, index: usize) {
    let id = format!("held-{index}");
    let session = session(programs + index);
    criterion::black_box(gate.check(&head(&id, &session)));
}

struct Ledger {
    classifier: Box<dyn RequestClassifier>,
    gate: Arc<dyn DispatchGate>,
    runtime: tokio::runtime::Runtime,
    workers: u64,
    next_request: usize,
}

impl Ledger {
    fn new(programs: usize, workers: u64, fill: Fill) -> Self {
        let classifier = classifier(programs, workers, fill);
        let gate = classifier
            .dispatch_gate()
            .expect("ThunderAgent returns a gate");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut ledger = Self {
            classifier,
            gate,
            runtime,
            workers,
            next_request: 0,
        };
        for index in 0..programs {
            ledger.turn(index);
        }
        hold_new_sessions(ledger.classifier.as_mut(), programs);
        if matches!(fill, Fill::Exact) {
            // The state the module doc describes: every rank is exactly full, so a new
            // session's first turn is held, not admitted.
            let id = "held-0";
            let session = session(programs);
            assert!(
                matches!(
                    ledger.gate.check(&head(id, &session)),
                    DispatchDecision::Hold { .. }
                ),
                "an exactly full pool must hold a new session"
            );
        }
        ledger
    }

    /// One continuing turn of `session_index`: dispatched, then completed.
    fn turn(&mut self, session_index: usize) {
        let id = format!("turn-{}", self.next_request);
        self.next_request += 1;
        dispatch_turn(
            self.classifier.as_mut(),
            self.gate.as_ref(),
            &id,
            session_index,
            self.workers,
        );
        self.runtime.block_on(complete(
            self.classifier.as_mut(),
            &id,
            session_index,
            self.workers,
        ));
    }
}

/// A ledger whose every program's retention has passed, on a paused clock: all programs acted
/// at the same instant, then the clock jumped past the retention window.
struct ExpiredLedger {
    gate: Arc<dyn DispatchGate>,
    runtime: tokio::runtime::Runtime,
    // Owns the programs the gate reads.
    _classifier: Box<dyn RequestClassifier>,
}

impl ExpiredLedger {
    fn new(programs: usize, workers: u64) -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .unwrap();
        let (classifier, gate) = runtime.block_on(async {
            let mut classifier = classifier(programs, workers, Fill::Exact);
            let gate = classifier
                .dispatch_gate()
                .expect("ThunderAgent returns a gate");
            for index in 0..programs {
                let id = format!("turn-{index}");
                dispatch_turn(classifier.as_mut(), gate.as_ref(), &id, index, workers);
                complete(classifier.as_mut(), &id, index, workers).await;
            }
            hold_new_sessions(classifier.as_mut(), programs);
            tokio::time::advance(RETENTION + Duration::from_secs(1)).await;
            (classifier, gate)
        });
        Self {
            gate,
            runtime,
            _classifier: classifier,
        }
    }

    /// The check reads the paused clock, so it runs inside the runtime.
    fn check_held(&self, programs: usize, index: usize) {
        self.runtime
            .block_on(async { check_held(self.gate.as_ref(), programs, index) });
    }
}

fn bench(c: &mut Criterion) {
    for (programs, workers) in [(256usize, 4u64), (1_000, 4), (10_000, 4), (1_024, 64)] {
        let mut group = c.benchmark_group(format!(
            "thunderagent_ledger/{programs}_programs_{workers}_workers"
        ));
        group
            .warm_up_time(Duration::from_millis(300))
            .measurement_time(Duration::from_secs(2))
            .sample_size(20);

        let ledger = Ledger::new(programs, workers, Fill::Exact);
        group.bench_function("check_held", |b| {
            b.iter(|| check_held(ledger.gate.as_ref(), programs, 0))
        });
        group.throughput(Throughput::Elements(HELD as u64));
        group.bench_function("recheck_1000_held", |b| {
            b.iter(|| {
                for index in 0..HELD {
                    check_held(ledger.gate.as_ref(), programs, index);
                }
            })
        });
        group.throughput(Throughput::Elements(1));

        let expired = ExpiredLedger::new(programs, workers);
        group.bench_function("check_held_retention_passed", |b| {
            b.iter(|| expired.check_held(programs, 0))
        });

        let mut live = Ledger::new(programs, workers, Fill::Exact);
        let mut cursor = 0usize;
        group.bench_function("turn", |b| {
            b.iter_batched(
                || {
                    cursor = (cursor + 1) % programs;
                    cursor
                },
                |index| live.turn(index),
                BatchSize::PerIteration,
            )
        });

        let mut paused = Ledger::new(programs, workers, Fill::OverThreshold);
        let mut cursor = 0usize;
        group.bench_function("turn_with_paused", |b| {
            b.iter_batched(
                || {
                    cursor = (cursor + 1) % programs;
                    cursor
                },
                |index| paused.turn(index),
                BatchSize::PerIteration,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
