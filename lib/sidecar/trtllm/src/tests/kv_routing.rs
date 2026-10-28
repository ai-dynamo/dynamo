// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenEngine routing, affinity, and telemetry failure regressions.

use super::*;
use crate::convert::build_generate_request;

#[tokio::test]
async fn sidecar_signs_parent_context_affinity_without_replacing_child() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let client = TrtllmClient::connect(
        &GrpcEndpoint::parse(&server.endpoint, "test endpoint").unwrap(),
        transport(1),
    )
    .await
    .expect("connect");
    let mut req = request();
    req.agent_context = Some(
        serde_json::from_value(json!({"session_id": "child-a", "parent_session_id": "parent-1"}))
            .unwrap(),
    );
    let proto = build_generate_request(
        &req,
        "child-request",
        "test-model",
        None,
        DisaggregationMode::Prefill,
    )
    .expect("context request");
    assert_eq!(
        proto.extra.as_ref().unwrap().fields["conversation_id"].kind,
        Some(prost_types::value::Kind::StringValue("child-a".to_string()))
    );
    client
        .generate(proto, Some(2), Some(("parent-1", "shared-secret")))
        .await
        .expect("send context request");
    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        &[Some("2".to_string())]
    );
    assert_eq!(
        server.service.affinity_headers.lock().await.as_slice(),
        &[(
            "parent-1".to_string(),
            "sha256=e98db0ec0d6c724ac9426801469a0c9fdce4149553a0d8b52a3adf8506f4d617".to_string()
        )]
    );
}

#[tokio::test]
async fn configured_prefill_sidecar_requires_capability_and_sends_parent_hint() {
    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service.clone()).await;
    let args = || {
        vec![
            "dynamo-trtllm-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            server.endpoint.clone(),
            "--model-path".to_string(),
            "model-source".to_string(),
            "--disaggregation-mode".to_string(),
            "prefill".to_string(),
            "--subagent-affinity-auth-key".to_string(),
            "shared-secret".to_string(),
        ]
    };
    let (unsupported, _) = TrtllmSidecarEngine::from_args(args()).expect("valid sidecar args");
    let error = unsupported
        .start(0)
        .await
        .expect_err("server lacks signed affinity");
    assert!(
        error
            .to_string()
            .contains("does not advertise signed subagent affinity")
    );

    service.subagent_affinity.store(true, Ordering::SeqCst);
    let (engine, _) = TrtllmSidecarEngine::from_args(args()).expect("valid sidecar args");
    engine
        .start(0)
        .await
        .expect("server supports signed affinity");
    let mut req = request();
    req.agent_context = Some(
        serde_json::from_value(json!({"session_id": "child-a", "parent_session_id": "parent-1"}))
            .expect("valid agent context"),
    );
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        prefill_dp_rank: Some(0),
        ..Default::default()
    });
    collect(&engine, req).await;
    let requests = service.requests.lock().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].extra.as_ref().unwrap().fields["conversation_id"].kind,
        Some(prost_types::value::Kind::StringValue("child-a".to_string()))
    );
    assert_eq!(service.affinity_headers.lock().await.len(), 1);
    assert_eq!(service.affinity_headers.lock().await[0].0, "parent-1");
}

#[tokio::test]
async fn kv_routing_discovers_contract_and_forwards_selected_rank() {
    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    service.dp_size.store(2, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with_kv_routing(&server.endpoint);

    let config = engine.start(0).await.expect("start with KV routing");
    let registration = config.llm.expect("LLM registration");
    assert_eq!(registration.kv_cache_block_size, Some(32));
    assert_eq!(registration.total_kv_blocks, Some(50));

    let sources = engine.kv_event_sources().await.expect("KV event sources");
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[1].dp_rank(), 1);

    let mut req = request();
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        dp_rank: Some(1),
        ..Default::default()
    });
    let _ = collect(&engine, req).await;
    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        [Some("1".to_string())]
    );
    let requests = server.service.requests.lock().await;
    let sent = requests.last().unwrap();
    assert_eq!(
        sent.extra.as_ref().unwrap().fields["conversation_id"].kind,
        Some(prost_types::value::Kind::StringValue(
            sent.request_id.clone()
        ))
    );
}

#[tokio::test]
async fn single_rank_kv_routing_preserves_only_real_conversation_ids() {
    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    service.dp_size.store(1, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with_kv_routing(&server.endpoint);
    engine.start(0).await.expect("start with KV routing");

    for session_id in [None, Some("real-session")] {
        let mut req = request();
        req.routing = Some(dynamo_backend_common::engine::RoutingHints {
            dp_rank: Some(0),
            ..Default::default()
        });
        req.agent_context = session_id
            .map(|session_id| serde_json::from_value(json!({"session_id": session_id})).unwrap());
        collect(&engine, req).await;
    }

    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        [None, None],
        "single-rank engines do not need an explicit rank hint"
    );
    let requests = server.service.requests.lock().await;
    assert_eq!(requests.len(), 2);
    assert!(
        !requests[0]
            .extra
            .as_ref()
            .unwrap()
            .fields
            .contains_key("conversation_id"),
        "anonymous requests must not create native per-conversation state"
    );
    assert_eq!(
        requests[1].extra.as_ref().unwrap().fields["conversation_id"].kind,
        Some(prost_types::value::Kind::StringValue(
            "real-session".to_string()
        ))
    );
}

#[tokio::test]
async fn cleanup_cancels_a_hung_load_poll() {
    use std::collections::HashMap;

    use dynamo_backend_common::metrics::{ComponentGauges, EngineMetrics, TestHierarchy};
    use dynamo_backend_common::{MetricsCtx, SnapshotPublisher};

    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service.clone()).await;
    let engine = engine_with_kv_routing(&server.endpoint);
    engine.start(0).await.expect("start with KV routing");

    let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
    let mut bindings = engine
        .setup_metrics(MetricsCtx {
            model: "test-model",
            component: "test-component",
            model_load_time_seconds: 0.0,
            metrics: &metrics,
        })
        .await
        .expect("set up metrics");
    let gauges = Arc::new(ComponentGauges::new(&metrics, &[0]).expect("component gauges"));
    let publisher = Arc::new(SnapshotPublisher::new(gauges, HashMap::new()));
    service.hang_load.store(true, Ordering::SeqCst);
    bindings
        .on_publisher_ready
        .take()
        .expect("metrics publisher callback")(publisher)
    .expect("start metrics publisher");
    tokio::time::sleep(Duration::from_millis(150)).await;

    tokio::time::timeout(Duration::from_millis(500), engine.cleanup())
        .await
        .expect("cleanup must cancel the in-flight GetLoad")
        .expect("cleanup succeeds");
}

#[tokio::test]
async fn stale_load_withdraws_worker_without_node_metadata() {
    use std::collections::HashMap;

    use dynamo_backend_common::metrics::{ComponentGauges, EngineMetrics, TestHierarchy};
    use dynamo_backend_common::{MetricsCtx, SnapshotPublisher};

    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service.clone()).await;
    let engine = engine_with_kv_routing(&server.endpoint);
    engine.start(0).await.expect("start with KV routing");

    let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
    let mut bindings = engine
        .setup_metrics(MetricsCtx {
            model: "test-model",
            component: "test-component",
            model_load_time_seconds: 0.0,
            metrics: &metrics,
        })
        .await
        .expect("set up metrics");
    let gauges = Arc::new(ComponentGauges::new(&metrics, &[0]).expect("component gauges"));
    let publisher = Arc::new(SnapshotPublisher::new(gauges, HashMap::new()));
    service.hang_load.store(true, Ordering::SeqCst);
    bindings
        .on_publisher_ready
        .take()
        .expect("metrics publisher callback")(publisher)
    .expect("start metrics publisher");
    // This legacy server has no node metadata: load health must still be supervised.
    let error = tokio::time::timeout(Duration::from_secs(35), engine.monitor_routing())
        .await
        .expect("bounded load staleness")
        .expect_err("stale routing load must withdraw the worker");
    assert!(
        error
            .to_string()
            .contains("Routing load remained unavailable")
    );

    tokio::time::timeout(Duration::from_millis(500), engine.cleanup())
        .await
        .expect("cleanup must cancel the in-flight GetLoad")
        .expect("cleanup succeeds");
}

#[tokio::test]
async fn discovery_keeps_events_load_and_dp_targeting_independent() {
    use dynamo_backend_common::MetricsCtx;
    use dynamo_backend_common::metrics::{EngineMetrics, TestHierarchy};

    for events in [false, true] {
        for load in [false, true] {
            let service = FakeTrtllm::default();
            service.kv_routing.store(true, Ordering::SeqCst);
            service.dp_size.store(2, Ordering::SeqCst);
            service.disable_events.store(!events, Ordering::SeqCst);
            service.disable_load.store(!load, Ordering::SeqCst);
            let server = FakeServer::start(service).await;
            let engine = engine_with_kv_routing(&server.endpoint);
            engine
                .start(0)
                .await
                .expect("discover independently enabled capabilities");
            assert_eq!(
                engine.kv_event_sources().await.unwrap().len(),
                if events { 2 } else { 0 }
            );

            let metrics = EngineMetrics::from_hierarchy(TestHierarchy::new());
            let bindings = engine
                .setup_metrics(MetricsCtx {
                    model: "test-model",
                    component: "test-component",
                    model_load_time_seconds: 0.0,
                    metrics: &metrics,
                })
                .await
                .unwrap();
            assert_eq!(bindings.on_publisher_ready.is_some(), load);
            assert_eq!(bindings.dp_ranks.len(), if load { 2 } else { 0 });

            let mut req = request();
            req.routing = Some(dynamo_backend_common::engine::RoutingHints {
                dp_rank: Some(1),
                ..Default::default()
            });
            collect(&engine, req).await;
            assert_eq!(
                server.service.target_dp_ranks.lock().await.as_slice(),
                [Some("1".to_string())]
            );
            engine.cleanup().await.unwrap();
        }
    }
}

#[tokio::test]
async fn leader_monitor_rejects_restarted_native_engine() {
    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    *service.node_metadata.lock().await = Some(crate::node::NodeMetadata {
        version: 1,
        engine_id: "original".into(),
        node_id: "leader".into(),
        leader: true,
        node_count: 1,
        kv_block_size: 32,
        local_dp_ranks: vec![0],
        source_owners: [(0, "leader".into())].into(),
    });
    let server = FakeServer::start(service.clone()).await;
    let engine = engine_with_kv_routing(&server.endpoint);
    engine.start(0).await.expect("start original engine");
    // Either metadata RPC may transiently reject work without withdrawing the
    // unchanged engine. Recovery must not suppress actual identity changes.
    let monitor = engine.monitor_routing();
    tokio::pin!(monitor);
    for failure in [&service.server_info_error, &service.kv_sources_error] {
        for code in [tonic::Code::Unavailable, tonic::Code::ResourceExhausted] {
            *failure.lock().await = Some(code);
            assert!(
                tokio::time::timeout(Duration::from_secs(6), &mut monitor)
                    .await
                    .is_err(),
                "transient {code:?} must not withdraw the worker"
            );
            *failure.lock().await = None;
            assert!(
                tokio::time::timeout(Duration::from_secs(6), &mut monitor)
                    .await
                    .is_err()
            );
        }
    }
    service
        .node_metadata
        .lock()
        .await
        .as_mut()
        .unwrap()
        .engine_id = "restarted".into();
    let error = tokio::time::timeout(Duration::from_secs(6), &mut monitor)
        .await
        .expect("detect changed incarnation")
        .expect_err("must withdraw stale discovery");
    assert!(error.to_string().contains("Local engine changed"));
}

#[tokio::test]
async fn follower_shutdown_cancels_initial_reconnect() {
    use clap::Parser;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use tokio_util::sync::CancellationToken;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let args = crate::args::Args::try_parse_from([
        "sidecar",
        "--grpc-endpoint",
        &format!("http://{address}"),
        "--model-path",
        "model-source",
    ])
    .unwrap();
    let drt = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let node = crate::node::NodeMetadata {
        version: 1,
        engine_id: "engine".into(),
        node_id: "follower".into(),
        leader: false,
        node_count: 2,
        kv_block_size: 32,
        local_dp_ranks: vec![],
        source_owners: Default::default(),
    };
    let cancel = CancellationToken::new();
    let shutdown = cancel.clone();
    let relay = crate::headless::relay(drt, args, node, cancel);
    let signal = async move {
        tokio::time::sleep(Duration::from_millis(25)).await;
        shutdown.cancel();
    };
    tokio::time::timeout(Duration::from_secs(1), async {
        let (result, ()) = tokio::join!(relay, signal);
        result.expect("clean cancellation");
    })
    .await
    .expect("follower must stop promptly");
}

#[tokio::test]
async fn sustained_metadata_outage_withdraws_worker() {
    let service = FakeTrtllm::default();
    service.kv_routing.store(true, Ordering::SeqCst);
    *service.node_metadata.lock().await = Some(crate::node::NodeMetadata {
        version: 1,
        engine_id: "original".into(),
        node_id: "leader".into(),
        leader: true,
        node_count: 1,
        kv_block_size: 32,
        local_dp_ranks: vec![0],
        source_owners: [(0, "leader".into())].into(),
    });
    let server = FakeServer::start(service.clone()).await;
    let engine = engine_with_kv_routing(&server.endpoint);
    engine.start(0).await.expect("start original engine");
    *service.kv_sources_error.lock().await = Some(tonic::Code::ResourceExhausted);
    let error = tokio::time::timeout(Duration::from_secs(35), engine.monitor_routing())
        .await
        .expect("bounded metadata recovery")
        .expect_err("sustained outage must withdraw worker");
    assert!(
        error
            .to_string()
            .contains("Native metadata remained unavailable")
    );
}
