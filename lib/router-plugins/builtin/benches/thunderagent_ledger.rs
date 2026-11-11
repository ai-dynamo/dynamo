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

use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_kv_router::plugins::prelude::*;

const BLOCK_SIZE: u32 = 64;
const WORKERS: u64 = 4;
const ISL: usize = 1_000;
const HELD: usize = 1_000;
const RETENTION: Duration = Duration::from_secs(3_600);

fn worker(index: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(index, 0)
}

fn session(index: usize) -> SessionContext {
    SessionContext::new(format!("session-{index}"), None, Some(false), None)
}

fn classifier(programs: usize) -> Box<dyn RequestClassifier> {
    let policy = tempfile::NamedTempFile::new().unwrap();
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
worker_selection:
  aggregated: thunderagent
  instances:
    - name: thunderagent
      type: thunderagent
"#,
            RETENTION.as_secs()
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
    // Every worker holds exactly its share of the programs, so a new program has no room.
    let per_worker_tokens = programs.div_ceil(WORKERS as usize) * ISL;
    let kv_blocks = (per_worker_tokens as u64).div_ceil(u64::from(BLOCK_SIZE));
    let context = RequestClassifierContext::new(BLOCK_SIZE, move || {
        (0..WORKERS)
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
    let request = ClassifyRequest::for_test(ISL, Some(request_id), Some(session.clone()));
    classifier
        .classify(request, &NoWorkers)
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
) {
    let session = session(session_index);
    classify(classifier, request_id, &session);
    gate.dispatched(
        &head(request_id, &session),
        Some(worker(session_index as u64 % WORKERS)),
    );
}

async fn complete(classifier: &mut dyn RequestClassifier, request_id: &str, session_index: usize) {
    let worker = worker(session_index as u64 % WORKERS);
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
    next_request: usize,
}

impl Ledger {
    fn new(programs: usize) -> Self {
        let classifier = classifier(programs);
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
            next_request: 0,
        };
        for index in 0..programs {
            ledger.turn(index);
        }
        hold_new_sessions(ledger.classifier.as_mut(), programs);
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
        );
        self.runtime
            .block_on(complete(self.classifier.as_mut(), &id, session_index));
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
    fn new(programs: usize) -> Self {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .unwrap();
        let (classifier, gate) = runtime.block_on(async {
            let mut classifier = classifier(programs);
            let gate = classifier
                .dispatch_gate()
                .expect("ThunderAgent returns a gate");
            for index in 0..programs {
                let id = format!("turn-{index}");
                dispatch_turn(classifier.as_mut(), gate.as_ref(), &id, index);
                complete(classifier.as_mut(), &id, index).await;
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
    for programs in [256usize, 1_000, 10_000] {
        let mut group = c.benchmark_group(format!("thunderagent_ledger/{programs}_programs"));
        group
            .warm_up_time(Duration::from_millis(300))
            .measurement_time(Duration::from_secs(2))
            .sample_size(20);

        let ledger = Ledger::new(programs);
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

        let expired = ExpiredLedger::new(programs);
        group.bench_function("check_held_retention_passed", |b| {
            b.iter(|| expired.check_held(programs, 0))
        });

        let mut live = Ledger::new(programs);
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
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
