// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `SelectionCore::potential_loads` empty-token fast path vs. a non-empty
//! control, across representative worker counts.
//!
//! Run with: `cargo bench -p dynamo-kv-router --bench potential_loads`
//!
//! To compare before/after a change to the empty-token fast path, run this
//! same file at two revisions (it only depends on the stable
//! `SelectionCore::potential_loads` public entrypoint) and diff with
//! criterion's baselines:
//!   cargo bench -p dynamo-kv-router --bench potential_loads -- --save-baseline before
//!   # checkout the other revision
//!   cargo bench -p dynamo-kv-router --bench potential_loads -- --baseline before

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::services::selection::{PotentialLoadsRequest, SelectionCore, WorkerRequest};
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

const WORKER_COUNTS: [u64; 4] = [1, 8, 32, 128];
const CONTROL_TOKEN_COUNT: usize = 64;

fn router_config() -> KvRouterConfig {
    KvRouterConfig {
        use_kv_events: false,
        router_queue_threshold: None,
        ..Default::default()
    }
}

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime")
}

fn build_core(rt: &Runtime, num_workers: u64) -> SelectionCore {
    let core = SelectionCore::new_local(router_config(), 1, CancellationToken::new());
    rt.block_on(async {
        for worker_id in 0..num_workers {
            let req: WorkerRequest = serde_json::from_value(serde_json::json!({
                "worker_id": worker_id,
                "model_name": "model",
                "endpoint": format!("http://worker-{worker_id}:8000"),
                "block_size": 4,
            }))
            .expect("valid worker request");
            core.upsert_worker(req).await.expect("register worker");
        }
    });
    core
}

fn empty_tokens_request() -> PotentialLoadsRequest {
    serde_json::from_value(serde_json::json!({
        "model_name": "model",
        "token_ids": [],
    }))
    .expect("valid potential-loads request")
}

fn control_tokens_request() -> PotentialLoadsRequest {
    let token_ids: Vec<u32> = (0..CONTROL_TOKEN_COUNT as u32).collect();
    serde_json::from_value(serde_json::json!({
        "model_name": "model",
        "token_ids": token_ids,
    }))
    .expect("valid potential-loads request")
}

fn bench_group(c: &mut Criterion, group_name: &str, request: fn() -> PotentialLoadsRequest) {
    let rt = build_runtime();
    let mut group = c.benchmark_group(group_name);
    for &workers in &WORKER_COUNTS {
        let core = build_core(&rt, workers);
        group.bench_with_input(BenchmarkId::from_parameter(workers), &workers, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    black_box(core.potential_loads(request()).await.expect("ok"));
                });
            });
        });
    }
    group.finish();
}

fn bench_empty_tokens(c: &mut Criterion) {
    bench_group(c, "potential_loads/empty_tokens", empty_tokens_request);
}

fn bench_control_tokens(c: &mut Criterion) {
    bench_group(
        c,
        "potential_loads/control_64_tokens",
        control_tokens_request,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default()
    .sample_size(200)
    .warm_up_time(Duration::from_secs(2))
    .measurement_time(Duration::from_secs(8))
    .noise_threshold(0.03);
    targets = bench_empty_tokens, bench_control_tokens
}

criterion_main!(benches);
