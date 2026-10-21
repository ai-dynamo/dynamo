// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Polling `Control.GetLoad` for router KV load, component gauges, and planner
//! forward-pass metrics.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use dynamo_backend_common::metrics::TestHierarchy;
use dynamo_backend_common::{
    ComponentGauges, ComponentSnapshot, EngineMetrics, ForwardPassSnapshot, FpmPublisher,
    MetricsCtx, SnapshotPublisher,
};
use dynamo_mocker::common::protocols::FpmSink;
use prost_types::value::Kind;

use super::*;
use crate::load::{advances, parse_forward_pass, snapshot_from_load};
use crate::pb;

#[derive(Default)]
struct RecordingFpm(Mutex<Vec<ForwardPassSnapshot>>);

impl FpmSink for RecordingFpm {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(snapshot);
        Ok(())
    }
}

impl RecordingFpm {
    fn snapshots(&self) -> Vec<ForwardPassSnapshot> {
        self.0.lock().unwrap().clone()
    }
}

fn val(kind: Kind) -> prost_types::Value {
    prost_types::Value { kind: Some(kind) }
}

fn num_struct(fields: &[(&str, f64)]) -> prost_types::Value {
    val(Kind::StructValue(prost_types::Struct {
        fields: fields
            .iter()
            .map(|&(k, v)| (k.to_string(), val(Kind::NumberValue(v))))
            .collect(),
    }))
}

fn load_sample(
    used_kv_blocks: Option<u64>,
    total_kv_blocks: Option<u64>,
    counter_id: f64,
    wall_time: f64,
) -> pb::LoadInfo {
    let scheduled = num_struct(&[
        ("num_prefill_requests", 3.0),
        ("sum_prefill_tokens", 1536.0),
        ("var_prefill_length", 12.5),
        ("sum_prefill_kv_tokens", 512.0),
        ("num_decode_requests", 9.0),
        ("sum_decode_kv_tokens", 18432.0),
        ("var_decode_kv_tokens", 64.25),
    ]);
    let queued = num_struct(&[
        ("num_prefill_requests", 2.0),
        ("sum_prefill_tokens", 1024.0),
        ("var_prefill_length", 8.75),
        ("num_decode_requests", 4.0),
        ("sum_decode_kv_tokens", 4096.0),
        ("var_decode_kv_tokens", 32.125),
    ]);
    let forward_pass = prost_types::Struct {
        fields: BTreeMap::from([
            ("counter_id".to_string(), val(Kind::NumberValue(counter_id))),
            ("wall_time".to_string(), val(Kind::NumberValue(wall_time))),
            ("scheduled_requests".to_string(), scheduled),
            ("queued_requests".to_string(), queued),
        ]),
    };
    pb::LoadInfo {
        instance_id: "inst-1".to_string(),
        used_kv_blocks,
        total_kv_blocks,
        attributes: Some(prost_types::Struct {
            fields: BTreeMap::from([(
                "forward_pass".to_string(),
                val(Kind::StructValue(forward_pass)),
            )]),
        }),
        ..Default::default()
    }
}

fn metrics_ctx<'a>(metrics: &'a EngineMetrics) -> MetricsCtx<'a> {
    MetricsCtx {
        model: SERVED,
        component: "backend",
        model_load_time_seconds: 0.0,
        metrics,
    }
}

async fn start_polling(
    initial: Result<pb::LoadInfo, tonic::Code>,
    card_total_kv_blocks: Option<u64>,
) -> (
    FakeServer,
    LpxSidecarEngine,
    EngineMetrics,
    Arc<RecordingFpm>,
) {
    let fake = FakeEngine::new(AGG);
    fake.set_load(initial);
    let server = FakeServer::start(fake, None).await;
    let engine = started_with_capacity(&server, AGG, card_total_kv_blocks).await;

    let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
    let bindings = engine
        .setup_metrics(metrics_ctx(&metrics))
        .await
        .expect("setup_metrics");
    let fpm = engine
        .forward_pass_metrics()
        .await
        .expect("forward_pass_metrics");
    assert_eq!(fpm.dp_size, 1, "FPM binding must be declared");
    let fpm_sink = Arc::new(RecordingFpm::default());
    fpm.on_ready.expect("fpm on_ready")(vec![FpmPublisher::new(Some(
        Arc::clone(&fpm_sink) as Arc<dyn FpmSink>
    ))])
    .expect("fpm ready");

    if card_total_kv_blocks.is_some() {
        assert_eq!(bindings.dp_ranks, [0], "snapshot binding must be declared");
        let gauges = Arc::new(ComponentGauges::new(&metrics, &bindings.dp_ranks).expect("gauges"));
        let publisher = Arc::new(SnapshotPublisher::new(gauges, HashMap::new()));
        bindings
            .on_publisher_ready
            .expect("snapshot on_publisher_ready")(publisher)
        .expect("snapshot publisher ready");
    } else {
        assert!(bindings.dp_ranks.is_empty() && bindings.on_publisher_ready.is_none());
    }
    (server, engine, metrics, fpm_sink)
}

async fn consume_sample(server: &FakeServer) {
    server
        .fake
        .wait_for_load_calls(server.fake.load_calls() + 2)
        .await;
}

async fn poll_with(server: &FakeServer, load: Result<pb::LoadInfo, tonic::Code>) {
    server.fake.set_load(load);
    consume_sample(server).await;
}

fn expfmt(metrics: &EngineMetrics) -> String {
    metrics
        .hierarchy()
        .get_metrics_registry()
        .prometheus_expfmt_combined()
        .expect("expfmt")
}

fn assert_gauge(metrics: &EngineMetrics, metric: &str, expected: f32) {
    let expected_line = format!(r#"{metric}{{dp_rank="0"}} {}"#, f64::from(expected));
    let rendered = expfmt(metrics);
    assert!(
        rendered.lines().any(|line| line == expected_line),
        "missing line {expected_line:?} in:\n{rendered}"
    );
}

#[test]
fn snapshot_from_load_validates_capacity() {
    for (used, total, card_total, expected) in [
        (Some(64), None, 256, Some((64, 256, 0.25))),
        (None, Some(100), 256, None),
        (Some(10), Some(0), 256, None),
        (Some(10), None, 0, None),
    ] {
        let sample = load_sample(used, total, 0.0, 0.0);
        let expected = expected.map(|(used, total, usage)| ComponentSnapshot {
            kv_used_blocks: used,
            kv_total_blocks: total,
            gpu_cache_usage: usage,
            kv_cache_hit_rate: None,
            dp_rank: 0,
        });
        assert_eq!(snapshot_from_load(&sample, card_total), expected);
    }
}

#[test]
fn advances_tracks_counter_and_instance_resets() {
    let mut last = None;
    for (instance, counter, expected) in [
        ("inst-a", 0, false),
        ("inst-a", 1, true),
        ("inst-a", 1, false),
        ("inst-a", 2, true),
        ("inst-a", 1, false),
        ("inst-b", 0, false),
        ("inst-b", 1, true),
        ("inst-c", 5, true),
    ] {
        assert_eq!(advances(&mut last, instance, counter), expected);
    }
}

#[tokio::test]
async fn fpm_parses_every_field_and_runs_without_card_kv_blocks() {
    let (server, engine, _metrics, fpm) =
        start_polling(Ok(load_sample(None, None, 7.0, 0.0425)), None).await;
    consume_sample(&server).await;

    assert_eq!(
        fpm.snapshots(),
        vec![ForwardPassSnapshot {
            counter_id: 7,
            num_prefill_requests: 3,
            sum_prefill_tokens: 1536,
            var_prefill_length: 12.5,
            sum_prefill_kv_tokens: 512,
            num_decode_requests: 9,
            sum_decode_kv_tokens: 18432,
            var_decode_kv_tokens: 64.25,
            num_queued_prefill: 2,
            sum_queued_prefill_tokens: 1024,
            var_queued_prefill_length: 8.75,
            num_queued_decode: 4,
            sum_queued_decode_kv_tokens: 4096,
            var_queued_decode_kv_tokens: 32.125,
            wall_time_secs: 0.0425,
            ..Default::default()
        }]
    );
    engine.cleanup().await.expect("cleanup");
}

#[tokio::test]
async fn load_polling_validates_independently_and_recovers() {
    assert!(parse_forward_pass(&load_sample(None, None, -1.0, 0.05)).is_none());

    let (server, engine, metrics, fpm) =
        start_polling(Err(tonic::Code::Unavailable), Some(256)).await;
    consume_sample(&server).await;
    assert!(fpm.snapshots().is_empty());

    poll_with(&server, Ok(load_sample(Some(7), Some(100), 1.5, 0.05))).await;
    assert_gauge(&metrics, "dynamo_component_total_blocks", 100.0);
    assert_gauge(&metrics, "dynamo_component_gpu_cache_usage_percent", 0.07);
    assert!(!expfmt(&metrics).contains("dynamo_component_kv_cache_hit_rate"));
    assert!(fpm.snapshots().is_empty());

    poll_with(&server, Ok(load_sample(Some(200), Some(100), 1.0, 0.08))).await;
    assert_gauge(&metrics, "dynamo_component_gpu_cache_usage_percent", 0.07);
    assert_eq!(fpm.snapshots().len(), 1);
    assert_eq!(fpm.snapshots()[0].wall_time_secs, 0.08);

    poll_with(&server, Err(tonic::Code::Unavailable)).await;
    assert_gauge(&metrics, "dynamo_component_gpu_cache_usage_percent", 0.07);
    assert_eq!(fpm.snapshots().len(), 1);

    poll_with(&server, Ok(load_sample(Some(0), Some(100), 2.0, 0.04))).await;
    assert_gauge(&metrics, "dynamo_component_gpu_cache_usage_percent", 0.0);
    assert_eq!(fpm.snapshots().len(), 2);
    engine.cleanup().await.expect("cleanup");
}

#[tokio::test]
async fn unimplemented_at_startup_and_after_start() {
    let unimpl = FakeServer::start(FakeEngine::new(AGG), None).await;
    let engine = started(&unimpl, AGG).await;
    let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
    let bindings = engine
        .setup_metrics(metrics_ctx(&metrics))
        .await
        .expect("setup_metrics");
    let fpm = engine
        .forward_pass_metrics()
        .await
        .expect("forward_pass_metrics");
    assert!(bindings.dp_ranks.is_empty() && bindings.on_publisher_ready.is_none());
    assert!(fpm.dp_size == 0 && fpm.on_ready.is_none());
    engine.cleanup().await.expect("cleanup");

    let (server, engine, metrics, fpm) =
        start_polling(Ok(load_sample(Some(10), Some(100), 0.0, 0.0)), Some(256)).await;
    consume_sample(&server).await;

    let before = server.fake.load_calls();
    server.fake.set_load(Err(tonic::Code::Unimplemented));
    server.fake.wait_for_load_calls(before + 1).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while Arc::strong_count(&fpm) > 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("poller exits on UNIMPLEMENTED");

    let calls_at_exit = server.fake.load_calls();
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert_eq!(server.fake.load_calls(), calls_at_exit);
    assert_gauge(&metrics, "dynamo_component_gpu_cache_usage_percent", 0.1);
    engine.cleanup().await.expect("cleanup");
}

#[tokio::test]
async fn stalled_call_times_out_and_cleanup_cancels_and_joins_immediately() {
    let (server, engine, metrics, fpm) =
        start_polling(Ok(load_sample(Some(10), Some(100), 0.0, 0.0)), Some(256)).await;
    consume_sample(&server).await;

    let before_stall = server.fake.load_calls();
    server.fake.stall_next_load();
    server
        .fake
        .set_load(Ok(load_sample(Some(20), Some(100), 0.0, 0.0)));
    server.fake.wait_for_load_calls(before_stall + 1).await;
    let start = Instant::now();
    server.fake.wait_for_load_calls(before_stall + 3).await;
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_gauge(&metrics, "dynamo_component_gpu_cache_usage_percent", 0.2);

    let stalled_call = server.fake.load_calls() + 1;
    server.fake.stall_next_load();
    server.fake.wait_for_load_calls(stalled_call).await;
    assert_eq!(Arc::strong_count(&fpm), 2);

    engine.cleanup().await.expect("cleanup");
    assert_eq!(
        Arc::strong_count(&fpm),
        1,
        "cleanup must join the poller task before returning"
    );
}
