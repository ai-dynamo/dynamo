// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Queue admission policy scaling benchmark.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench policy_queue_admission --features bench`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, ScheduleRequest};
use dynamo_kv_router::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, KvRouterConfig, LocalScheduler, QueueAdmissionDecision,
    QueueAdmissionPolicy, QueueAdmissionRequest, RouterQueuePolicy, WorkerInputView, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};
use tokio::runtime::Runtime;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

struct BypassPolicy;

impl QueueAdmissionPolicy for BypassPolicy {
    fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
        black_box(request.workers());
        QueueAdmissionDecision::Bypass
    }
}

struct FirstPicker;

impl WorkerPicker for FirstPicker {
    fn pick(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        assert!(!input.candidates().is_empty());
        Ok(0)
    }
}

type BenchScheduler =
    LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, WorkerSelectionPolicy>;

fn scheduler(worker_count: usize, admission_enabled: bool) -> (BenchScheduler, CancellationToken) {
    let dp_ranges = (0..worker_count as u64)
        .map(|worker_id| (worker_id, (0, 1)))
        .collect();
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        16,
        dp_ranges,
        false,
        0,
        "bench",
    ));
    let workers = (0..worker_count as u64)
        .map(|worker_id| {
            (
                worker_id,
                SimpleWorkerConfig {
                    total_kv_blocks: Some(1_000_000),
                    ..Default::default()
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let (_workers_tx, workers_rx) = watch::channel(workers);
    let selector = WorkerSelectionPolicy::new(
        KvRouterConfig::default(),
        "bench",
        Vec::new(),
        Box::new(FirstPicker),
    );
    let selector = if admission_enabled {
        selector.with_admission_policy(Box::new(BypassPolicy))
    } else {
        selector
    };
    let cancellation_token = CancellationToken::new();
    let scheduler = LocalScheduler::new(
        slots,
        workers_rx,
        None,
        16,
        selector,
        RouterQueuePolicy::Fcfs,
        None,
        Duration::from_secs(60),
        true,
        cancellation_token.clone(),
        "bench",
        false,
    );
    (scheduler, cancellation_token)
}

fn request(request_id: String) -> ScheduleRequest {
    ScheduleRequest {
        mode: ScheduleMode::TrackedWithLifecycle { request_id },
        token_seq: None,
        block_hashes: None,
        isl_tokens: 32,
        lora_name: None,
        expected_output_tokens: Some(32),
        pinned_worker: Some(WorkerWithDpRank::from_worker_id(0)),
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        router_hint_candidates: None,
        retain_router_hint_chain: false,
        shared_cache_hits: None,
    }
}

fn policy_queue_admission(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("policy_queue/admit_complete_pinned");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for admission_enabled in [false, true] {
        let mode = if admission_enabled { "on" } else { "off" };
        for worker_count in [32, 1_024, 10_000] {
            let (scheduler, cancellation_token) =
                runtime.block_on(async { scheduler(worker_count, admission_enabled) });
            let mut request_seq = 0_u64;
            group.bench_with_input(
                BenchmarkId::new(mode, worker_count),
                &worker_count,
                |b, _| {
                    b.iter(|| {
                        request_seq += 1;
                        let request_id = request_seq.to_string();
                        runtime.block_on(async {
                            let response = scheduler
                                .schedule_request(request(request_id.clone()))
                                .await
                                .unwrap();
                            scheduler
                                .complete_if_worker(&request_id, response.best_worker, 64)
                                .await
                                .unwrap();
                            black_box(response);
                        });
                    });
                },
            );
            cancellation_token.cancel();
        }
    }
    group.finish();
}

criterion_group!(benches, policy_queue_admission);
criterion_main!(benches);
