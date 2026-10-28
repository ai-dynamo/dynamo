// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishing the engine's load.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_backend_common::metrics::{ComponentGauges, EngineMetrics, TestHierarchy};
use dynamo_backend_common::{MetricsBindings, MetricsCtx, SnapshotPublisher};

use super::fake::Load;
use super::*;

/// A started sidecar for an engine whose `GetLoad` answers as `load`, with
/// the metrics bindings the Worker would receive.
async fn sidecar(
    load: Load,
) -> (
    FakeServer,
    LpxSidecarEngine,
    WorkerConfig,
    EngineMetrics,
    MetricsBindings,
) {
    let fake = FakeEngine::new(AGG);
    fake.set_load(load);
    let server = FakeServer::start(fake).await;
    let (engine, config) = discover(argv(&server.endpoint, &[]))
        .await
        .expect("bootstrap");
    engine.start(0).await.expect("start");
    let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
    let bindings = engine
        .setup_metrics(MetricsCtx {
            model: SERVED,
            component: "backend",
            model_load_time_seconds: 0.0,
            metrics: &metrics,
        })
        .await
        .expect("setup_metrics");
    (server, engine, config, metrics, bindings)
}

/// Hands `bindings` a publisher as the Worker builds one, without the
/// router's per-rank publishers, which need a runtime.
fn start_publishing(metrics: &EngineMetrics, bindings: MetricsBindings) {
    let gauges = Arc::new(ComponentGauges::new(metrics, &bindings.dp_ranks).expect("gauges"));
    let publisher = Arc::new(SnapshotPublisher::new(gauges, HashMap::new()));
    (bindings.on_publisher_ready.expect("a publisher callback"))(publisher).expect("publishing");
}

/// Rank 0's `gpu_cache_usage_percent`.
fn usage(metrics: &EngineMetrics) -> Option<f64> {
    let text = metrics
        .hierarchy()
        .get_metrics_registry()
        .prometheus_expfmt_combined()
        .expect("render the metrics");
    text.lines()
        .find(|line| line.starts_with("dynamo_component_gpu_cache_usage_percent{"))
        .and_then(|line| line.rsplit(' ').next()?.parse().ok())
}

#[tokio::test]
async fn an_engine_without_get_load_publishes_no_load() {
    let (_server, engine, config, _, bindings) = sidecar(Load::Unimplemented).await;
    assert!(!config.enable_kv_routing);
    assert!(bindings.dp_ranks.is_empty() && bindings.on_publisher_ready.is_none());
    engine.cleanup().await.expect("cleanup");
}

/// The startup load is published at once, and each poll after it.
#[tokio::test]
async fn reported_load_is_polled_into_the_publisher() {
    let (server, engine, config, metrics, bindings) = sidecar(Load::Used(64)).await;
    assert!(config.enable_kv_routing && !config.enable_local_indexer);
    assert_eq!(bindings.dp_ranks, [0]);
    start_publishing(&metrics, bindings);
    assert_eq!(usage(&metrics), Some(0.25));

    // Past the first poll, so only a repeated one can see the change.
    tokio::time::sleep(Duration::from_millis(250)).await;
    server.fake.set_load(Load::Used(128));
    tokio::time::timeout(Duration::from_secs(5), async {
        while usage(&metrics) != Some(0.5) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("a poll publishes the new load");

    engine.cleanup().await.expect("cleanup");
    server.fake.set_load(Load::Used(192));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(usage(&metrics), Some(0.5), "cleanup stops the poll");
}

/// A load that stays unavailable is given up on at the staleness limit, which
/// `serve` turns into the worker's withdrawal.
#[tokio::test]
async fn stale_load_is_given_up_at_the_staleness_limit() {
    let (server, engine, _, metrics, bindings) = sidecar(Load::Used(64)).await;
    server.fake.set_load(Load::Hangs);
    // Every poll times out; paused time skips the waits.
    tokio::time::pause();
    start_publishing(&metrics, bindings);
    let lost = engine.load_lost();
    tokio::pin!(lost);
    let just_before = crate::load::STALENESS_LIMIT - Duration::from_secs(1);
    tokio::time::timeout(just_before, &mut lost)
        .await
        .expect_err("the load is kept until the staleness limit");
    let error = tokio::time::timeout(Duration::from_secs(2), lost)
        .await
        .expect("the load is given up on at the staleness limit");
    assert_backend(&error, BackendError::EngineShutdown);
    engine.cleanup().await.expect("cleanup");
}

#[test]
fn only_a_fresh_valid_rank_0_load_is_accepted() {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let load = |at: u64, ranks: &[(u32, u64, u64)]| crate::pb::LoadInfo {
        timestamp_unix_nanos: Some(at),
        ranks: ranks
            .iter()
            .map(|&(rank, used, total)| crate::pb::RankLoadInfo {
                data_parallel_rank: Some(rank),
                used_kv_blocks: Some(used),
                total_kv_blocks: Some(total),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let snapshot = |load| crate::load::snapshot("lpx-test", load);
    assert_eq!(
        snapshot(load(now, &[(0, 64, 256)])).unwrap().kv_used_blocks,
        64
    );
    let second = 1_000_000_000;
    for invalid in [
        crate::pb::LoadInfo::default(),
        load(now - 6 * second, &[(0, 64, 256)]),
        load(now + 2 * second, &[(0, 64, 256)]),
        load(now, &[]),
        load(now, &[(0, 64, 256), (0, 64, 256)]),
        load(now, &[(1, 64, 256)]),
        load(now, &[(0, 257, 256)]),
        load(now, &[(0, 0, 0)]),
    ] {
        snapshot(invalid).expect_err("an invalid load is rejected");
    }
}
