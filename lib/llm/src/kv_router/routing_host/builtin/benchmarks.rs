// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Microbenchmarks for built-in `RoutingHost` policy overhead.
//!
//! Run an individual case with:
//!
//! ```text
//! DYN_BENCH_POLICY=round-robin DYN_BENCH_WORKERS=256 \
//! DYN_BENCH_ITERS=100000 cargo test --release -p dynamo-llm \
//! --features bench routing_host_microbench -- --ignored --nocapture
//! ```

use std::{hint::black_box, sync::Arc, time::Instant};

use dynamo_kv_router::DefaultWorkerSelector;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::Instance,
    distributed::DistributedConfig,
    pipeline::{
        AddressedRequest, Context, ManyIn, PushRouter, RouterMode, StreamingDispatch, async_trait,
    },
};

use super::*;

#[derive(Default)]
struct BenchmarkDispatch;

#[async_trait]
impl StreamingDispatch<PreprocessedRequest, Annotated<LLMEngineOutput>> for BenchmarkDispatch {
    async fn generate(
        &self,
        _request: SingleIn<AddressedRequest<PreprocessedRequest>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("RoutingHost microbenchmark excludes transport")
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        unreachable!("RoutingHost microbenchmark excludes transport")
    }
}

fn benchmark_mode(policy: &str) -> RouterMode {
    match policy {
        "round-robin" => RouterMode::RoundRobin,
        "random" => RouterMode::Random,
        "power-of-two" => RouterMode::PowerOfTwoChoices,
        "least-loaded" => RouterMode::LeastLoaded,
        other => panic!("unsupported built-in benchmark policy: {other}"),
    }
}

fn benchmark_env() -> (String, usize, usize, usize) {
    let policy = std::env::var("DYN_BENCH_POLICY").unwrap_or_else(|_| "round-robin".into());
    let workers = std::env::var("DYN_BENCH_WORKERS")
        .unwrap_or_else(|_| "32".into())
        .parse()
        .unwrap();
    let iterations: usize = std::env::var("DYN_BENCH_ITERS")
        .unwrap_or_else(|_| "100000".into())
        .parse()
        .unwrap();
    let warmup = std::env::var("DYN_BENCH_WARMUP")
        .unwrap_or_else(|_| (iterations / 20).max(1000).to_string())
        .parse()
        .unwrap();
    assert!(workers > 0 && iterations > 0);
    (policy, workers, iterations, warmup)
}

fn benchmark_request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("benchmark".to_string())
        .token_ids(vec![1])
        .stop_conditions(Default::default())
        .sampling_options(Default::default())
        .output_options(Default::default())
        .build()
        .unwrap()
}

async fn lifecycle_iteration(host: &RoutingHost) {
    let request = Context::new(benchmark_request());
    let HostedSelection {
        initial_worker,
        occupancy_reservation,
        ..
    } = host.select_hosted_worker(&request, None).unwrap();
    black_box(initial_worker);
    let mut guard: RequestGuard<DefaultWorkerSelector> = RequestGuard::new_builtin(
        host.request_metrics.clone(),
        initial_worker,
        occupancy_reservation,
        None,
        &request,
    );
    guard.abort().await;
}

async fn benchmark_builtin(
    policy: &str,
    worker_count: usize,
    iterations: usize,
    warmup: usize,
) -> f64 {
    let mode = benchmark_mode(policy);
    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace(format!("routing-host-microbench-{}", std::process::id()))
        .unwrap()
        .component("workers".to_string())
        .unwrap()
        .endpoint("generate".to_string());
    let client = endpoint.client().await.unwrap();
    let worker_ids = (1..=worker_count as u64).collect::<Vec<_>>();
    client.override_discovered_instances(worker_ids.clone());
    client.override_instance_avail(worker_ids);
    let dispatch = Arc::new(BenchmarkDispatch);
    let inner = PushRouter::from_client_with_dispatch(
        client,
        mode,
        dispatch as Arc<dyn StreamingDispatch<_, _>>,
    )
    .await
    .unwrap();
    let host = RoutingHost::<DefaultWorkerSelector>::new_builtin(inner).unwrap();

    for _ in 0..warmup {
        lifecycle_iteration(&host).await;
    }
    let started = Instant::now();
    for _ in 0..iterations {
        lifecycle_iteration(&host).await;
    }
    let ns_per_op = started.elapsed().as_nanos() as f64 / iterations as f64;
    black_box(&host);
    runtime.shutdown();
    ns_per_op
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "microbenchmark; run explicitly"]
#[serial_test::serial]
async fn routing_host_microbench() {
    let (policy, workers, iterations, warmup) = benchmark_env();
    let ns_per_op = benchmark_builtin(&policy, workers, iterations, warmup).await;
    println!(
        "ROUTING_HOST_BENCH\tpolicy={policy}\tworkers={workers}\titerations={iterations}\twarmup={warmup}\tns_per_op={ns_per_op:.3}"
    );
}
