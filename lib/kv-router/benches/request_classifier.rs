// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler overhead with classification disabled or using a pass-through classifier.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench request_classifier`

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
use dynamo_kv_router::scheduling::{
    OverlapSignals, PolicyProfile, RequestClassifier, ScheduleMode, ScheduleRequest,
};
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, DefaultWorkerSelector, LocalScheduler, NoopSequencePublisher,
    RouterQueuePolicy,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

const REQUESTS_PER_BATCH: usize = 128;
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

type BenchScheduler = LocalScheduler<NoopSequencePublisher, BenchWorkerConfig>;

#[derive(Clone, Default, PartialEq, Eq)]
struct BenchWorkerConfig {
    max_num_batched_tokens: Option<u64>,
    taints: HashSet<String>,
}

impl WorkerConfigLike for BenchWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }

    fn data_parallel_size(&self) -> u32 {
        1
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        None
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }
}

struct PassThroughClassifier;

impl RequestClassifier for PassThroughClassifier {}

async fn make_scheduler(classifier_enabled: bool) -> (Arc<BenchScheduler>, CancellationToken) {
    let workers = HashMap::from([(
        0,
        BenchWorkerConfig {
            max_num_batched_tokens: Some(4_096),
            ..Default::default()
        },
    )]);
    let slots = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        64,
        HashMap::from([(0, (0, 1))]),
        false,
        0,
        "request-classifier-benchmark",
    ));
    let (_workers_tx, workers_rx) = watch::channel(workers);
    let cancellation = CancellationToken::new();
    let profile = PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs);

    let scheduler = if classifier_enabled {
        LocalScheduler::new_with_policy_profile_and_request_classifier(
            slots,
            workers_rx,
            profile,
            64,
            DefaultWorkerSelector::new(None, "request-classifier-benchmark"),
            None,
            None,
            None,
            None,
            None,
            Duration::from_secs(60),
            true,
            cancellation.clone(),
            "request-classifier-benchmark",
            false,
            Box::new(PassThroughClassifier),
        )
        .unwrap()
    } else {
        LocalScheduler::new_without_overlap_refresh_with_policy_profile(
            slots,
            workers_rx,
            profile,
            64,
            DefaultWorkerSelector::new(None, "request-classifier-benchmark"),
            None,
            None,
            None,
            Duration::from_secs(60),
            true,
            cancellation.clone(),
            "request-classifier-benchmark",
            false,
        )
        .unwrap()
    };

    (Arc::new(scheduler), cancellation)
}

fn request(mode: ScheduleMode) -> ScheduleRequest {
    ScheduleRequest {
        mode,
        token_seq: None,
        block_hashes: None,
        isl_tokens: 64,
        lora_name: None,
        expected_output_tokens: None,
        pinned_worker: None,
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

async fn route_batch(scheduler: &BenchScheduler, lifecycle: bool) {
    for _ in 0..REQUESTS_PER_BATCH {
        let request_id = format!(
            "request-classifier-benchmark-{}",
            NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
        );
        let mut request_lifecycle = lifecycle
            .then(|| scheduler.begin_request_lifecycle(&request_id))
            .transpose()
            .unwrap()
            .flatten();
        let mode = if request_lifecycle.is_some() {
            ScheduleMode::TrackedWithLifecycle {
                request_id: request_id.clone(),
            }
        } else {
            ScheduleMode::Tracked {
                request_id: request_id.clone(),
            }
        };
        let response = scheduler.schedule_request(request(mode)).await.unwrap();
        if let Some(request_lifecycle) = request_lifecycle.as_mut() {
            request_lifecycle.selected(response.best_worker);
            request_lifecycle.sent(response.best_worker);
            request_lifecycle.received();
            request_lifecycle.responding();
            request_lifecycle.complete(Some(65));
        }
        black_box(response.best_worker);
        scheduler.free(&request_id).await.unwrap();
    }
}

fn request_classifier(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (disabled, disabled_cancel) = runtime.block_on(make_scheduler(false));
    let (pass_through, pass_through_cancel) = runtime.block_on(make_scheduler(true));

    let mut group = c.benchmark_group("request_classifier/schedule");
    group.throughput(Throughput::Elements(REQUESTS_PER_BATCH as u64));
    group.bench_function("disabled", |b| {
        b.iter(|| runtime.block_on(route_batch(&disabled, false)));
    });
    group.bench_function("pass_through", |b| {
        b.iter(|| runtime.block_on(route_batch(&pass_through, false)));
    });
    group.finish();

    let mut group = c.benchmark_group("request_classifier/full_lifecycle");
    group.throughput(Throughput::Elements(REQUESTS_PER_BATCH as u64));
    group.bench_function("disabled", |b| {
        b.iter(|| runtime.block_on(route_batch(&disabled, true)));
    });
    group.bench_function("pass_through", |b| {
        b.iter(|| runtime.block_on(route_batch(&pass_through, true)));
    });
    group.finish();

    disabled_cancel.cancel();
    pass_through_cancel.cancel();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .noise_threshold(0.03);
    targets = request_classifier
}
criterion_main!(benches);
