// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine driven against an in-process fake OpenEngine server.

use super::*;

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
async fn aggregated_generation_streams_delta_then_terminal() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 2);
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.model, "model-source");
    // GetModelInfo reports max_context_length 4096.
    assert_eq!(config.llm.unwrap().context_length, Some(4096));

    let outputs = collect(&engine, request()).await;
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].token_ids, [42]);
    assert!(outputs[0].finish_reason.is_none());
    assert_eq!(outputs[0].log_probs.as_deref(), Some(&[-0.25][..]));

    let terminal = &outputs[1];
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(2)));
    let usage = terminal.completion_usage.as_ref().expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 1));

    let requests = server.service.requests.lock().await;
    let sent = requests.first().expect("recorded request");
    assert_eq!(sent.model, "model-source");
    match sent.input.as_ref().expect("input") {
        pb::generate_request::Input::TokenIds(tokens) => assert_eq!(tokens.ids, [11, 22, 33]),
        other => panic!("expected token IDs input, got {other:?}"),
    }
}

#[tokio::test]
async fn grpc_request_errors_are_propagated() {
    let service = FakeTrtllm::default();
    service.reject.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    // TRT-LLM surfaces an invalid-argument on the initial response header, so
    // opening the stream fails rather than yielding an error item.
    let context = dynamo_backend_common::testing::mock_context();
    let result = engine
        .generate(request(), GenerateContext::new(context, None))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn cancellation_yields_a_cancelled_terminal() {
    let service = FakeTrtllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(request(), GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token_ids, [42]);
    context.stop_generating();
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("terminal within deadline")
        .unwrap()
        .unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

/// A request cancelled before dispatch must not reach the engine.
#[tokio::test]
async fn an_aggregated_request_cancelled_before_dispatch_never_reaches_the_engine() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    context.stop_generating();

    let mut stream = engine
        .generate(request(), GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("terminal within deadline")
        .unwrap()
        .unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    assert!(
        server.service.requests.lock().await.is_empty(),
        "a request cancelled before dispatch must not be sent to the engine"
    );
}

/// Nonzero logical ranks are not advertised and must be rejected before dispatch.
#[tokio::test]
async fn a_data_parallel_rank_hint_is_rejected_before_dispatch() {
    let server = FakeServer::start(FakeTrtllm::default()).await;

    for (mode, hints) in [
        (
            DisaggregationMode::Prefill,
            dynamo_backend_common::engine::RoutingHints {
                prefill_dp_rank: Some(7),
                ..Default::default()
            },
        ),
        (
            DisaggregationMode::Decode,
            dynamo_backend_common::engine::RoutingHints {
                dp_rank: Some(3),
                ..Default::default()
            },
        ),
        (
            AGG,
            dynamo_backend_common::engine::RoutingHints {
                dp_rank: Some(3),
                ..Default::default()
            },
        ),
    ] {
        let engine = engine_in_mode(&server.endpoint, 1, mode);
        engine.start(0).await.expect("start");
        let mut req = request();
        req.routing = Some(hints);
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(req, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err(), "{mode:?} must refuse a rank hint");
    }
    assert!(
        server.service.requests.lock().await.is_empty(),
        "nothing may reach the engine"
    );
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

/// A decode request holding transferred KV blocks must outlive its client until
/// the first token proves the transfer landed -- dropping the stream earlier
/// strands the prefill worker's blocks. It must not outlive it any longer than
/// that, or a cancelled request generates its whole budget with no consumer.
#[tokio::test]
async fn a_cancelled_decode_request_survives_only_until_the_transfer_lands() {
    let service = FakeTrtllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Decode);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session()).expect("handoff"),
        prompt_tokens_details: None,
    });
    let mut stream = engine
        .generate(req, GenerateContext::new(context.clone(), None))
        .await
        .expect("generate");

    context.stop_generating();
    // Still deferring: nothing has confirmed the KV transfer yet.
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(
        first.finish_reason, None,
        "the deferral outlasts the client"
    );
    assert!(!first.token_ids.is_empty());

    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), stream.next())
        .await
        .expect("the deferral lifts once a token lands")
        .unwrap()
        .unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

#[tokio::test]
async fn abort_sends_the_abort_rpc_to_the_server() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let request_id = context.id().to_string();
    engine.abort(context).await;

    // The cancelled generation's ID must reach TensorRT-LLM, not just produce a
    // local terminal, or the server keeps generating.
    assert_eq!(server.service.aborts.lock().await.as_slice(), [request_id]);
}

#[tokio::test]
async fn unsupported_features_fail_before_rpc_submission() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");

    let mut multiple = request();
    multiple.sampling_options.n = Some(2);

    let mut beam = request();
    beam.sampling_options.use_beam_search = Some(true);

    let mut embeds = request();
    embeds.prompt_embeds = Some("encoded".to_string());

    let mut prompt_logprobs = request();
    prompt_logprobs.output_options.prompt_logprobs = Some(1);

    let mut visible_stops = request();
    visible_stops.stop_conditions.stop_token_ids_visible = Some(vec![7]);

    for unsupported in [multiple, beam, embeds, prompt_logprobs, visible_stops] {
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(unsupported, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err());
    }
    assert!(server.service.requests.lock().await.is_empty());
}

#[tokio::test]
async fn pool_uses_each_configured_connection() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let endpoint =
        GrpcEndpoint::parse(&server.endpoint, "--grpc-endpoint").expect("valid endpoint");
    let client = TrtllmClient::connect(&endpoint, transport(2))
        .await
        .expect("connect pool");
    assert_eq!(client.connection_count(), 2);

    for index in 0..4 {
        let mut stream = client
            .generate(
                pb::GenerateRequest {
                    request_id: format!("request-{index}"),
                    model: "model-source".to_string(),
                    input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                        ids: vec![1, 2],
                    })),
                    ..Default::default()
                },
                None,
                None,
            )
            .await
            .expect("start stream");
        while stream.message().await.expect("message").is_some() {}
    }

    let ports: BTreeSet<_> = server
        .service
        .peers
        .lock()
        .await
        .iter()
        .map(SocketAddr::port)
        .collect();
    assert_eq!(ports.len(), 2);
}

/// A server with no Control service leaves the window unknown. Registering
/// anyway would advertise capacity and then reject, with a non-migratable 4xx,
/// every request that omits `max_tokens` -- most chat traffic. Fail at startup
/// instead, naming the flag that fixes it.
#[tokio::test]
async fn start_without_a_context_length_fails() {
    let service = FakeTrtllm::default();
    service.no_control.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let error = engine
        .start(0)
        .await
        .expect_err("a worker with no window must not register");
    assert!(
        error.to_string().contains("--context-length"),
        "the error must name the flag that supplies a window: {error}"
    );
}

/// An engine that answers has finished loading, so an answer carrying no
/// context length means it was started without `--max_seq_len` and will not
/// change. Fail on the first answer rather than polling to the deadline and
/// then blaming a slow load.
#[tokio::test]
async fn start_fails_at_once_when_the_server_answers_without_a_context_length() {
    let service = FakeTrtllm::default();
    service.empty_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let error = engine
        .start(0)
        .await
        .expect_err("a window is required to register");
    assert!(
        error.to_string().contains("--max_seq_len"),
        "the error must name the fix: {error}"
    );
    assert_eq!(
        server.service.model_info_calls.load(Ordering::SeqCst),
        1,
        "an answer that cannot change must not be retried"
    );
}

/// The wait exists for an engine that is not serving yet: it keeps asking
/// through transport failures and picks up the window once the engine answers.
#[tokio::test]
async fn start_waits_for_a_server_that_is_not_serving_yet() {
    let service = FakeTrtllm::default();
    service.unavailable_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let ready = Arc::clone(&server.service.unavailable_model_info);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        ready.store(false, Ordering::SeqCst);
    });
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let config = engine.start(0).await.expect("start once the engine serves");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        Some(4096)
    );
}

/// `--context-length` wins over what the engine reports. The engine is still
/// asked, so the disagreement can be logged and its output cap picked up, but
/// its answer does not decide the window.
#[tokio::test]
async fn a_configured_context_length_outranks_the_servers() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine_with(&server.endpoint, transport(1), Some(8192), AGG);

    let config = engine
        .start(0)
        .await
        .expect("start with a configured window");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        Some(8192),
        "the configured window must outrank the server's 4096"
    );
    assert_eq!(
        server.service.model_info_calls.load(Ordering::SeqCst),
        1,
        "the engine is still consulted, to cross-check and to learn its output cap"
    );
}

/// A configured window also means a server with no Control service is no longer
/// a startup problem: there is nothing left to ask it for.
#[tokio::test]
async fn a_configured_context_length_survives_a_server_without_control() {
    let service = FakeTrtllm::default();
    service.no_control.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, transport(1), Some(8192), AGG);

    let config = engine
        .start(0)
        .await
        .expect("start with a configured window");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        Some(8192)
    );
}

/// Shutdown has to terminate in-flight requests: the worker cannot drain if a
/// stream waits forever on an engine that will never answer.
#[tokio::test]
async fn cleanup_terminates_an_in_flight_request() {
    let service = FakeTrtllm::default();
    service.hang.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = Arc::new(engine(&server.endpoint, 1));
    engine.start(0).await.expect("start");

    let context = dynamo_backend_common::testing::mock_context();
    let mut stream = engine
        .generate(request(), GenerateContext::new(context, None))
        .await
        .expect("generate");
    let first = stream.next().await.expect("first item").expect("delta");
    assert_eq!(first.token_ids, [42]);

    engine.cleanup().await.expect("cleanup");
    let terminal = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("shutdown must terminate the stream")
        .expect("terminal item")
        .expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

/// A decode leg's cancellation deferral has to cover the dispatch itself, not
/// just the streaming loop. `generate` sends the request and then awaits
/// response headers, so a cancellation that wins that race abandons a request
/// the engine has already accepted and begun pulling KV for -- stranding the
/// prefill worker's blocks with no leg left to claim them. Shutdown still wins.
#[tokio::test]
async fn a_cancelled_decode_dispatch_is_not_abandoned_mid_flight() {
    let service = FakeTrtllm::default();
    service.hang_before_stream.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = Arc::new(engine_in_mode(
        &server.endpoint,
        1,
        DisaggregationMode::Decode,
    ));
    engine.start(0).await.expect("start");

    let mut decode_request = request();
    decode_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session()).expect("handoff"),
        prompt_tokens_details: None,
    });
    let context = dynamo_backend_common::testing::mock_context();
    let mut dispatch = tokio::spawn({
        let engine = Arc::clone(&engine);
        let context = context.clone();
        async move {
            engine
                .generate(decode_request, GenerateContext::new(context, None))
                .await
        }
    });

    // The fake records the request before it withholds response headers, which
    // is exactly the window this test is about: the engine has the request and
    // the sidecar does not know it yet.
    while server.service.requests.lock().await.is_empty() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    context.stop_generating();
    assert!(
        tokio::time::timeout(Duration::from_millis(200), &mut dispatch)
            .await
            .is_err(),
        "the dispatch must outlive the client's cancellation"
    );

    engine.cleanup().await.expect("cleanup");
    let mut stream = tokio::time::timeout(Duration::from_secs(5), dispatch)
        .await
        .expect("shutdown must release the dispatch")
        .expect("dispatch task")
        .expect("generate");
    let terminal = stream
        .next()
        .await
        .expect("a terminal item")
        .expect("terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
}

/// Argument parsing decides where each worker registers. A disaggregated leg
/// that lands on the operator-configured component instead of its role
/// component is invisible to the frontend's prefill router, and no amount of
/// request-level testing would show it.
#[test]
fn parsed_arguments_map_onto_the_worker_registration() {
    let parse = |extra: &[&str]| {
        let mut argv = vec![
            "dynamo-trtllm-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            "127.0.0.1:50051".to_string(),
            "--model-path".to_string(),
            "model-source".to_string(),
        ];
        argv.extend(extra.iter().map(|arg| arg.to_string()));
        TrtllmSidecarEngine::from_args(argv)
    };

    let (_, aggregated) = parse(&[
        "--component",
        "operator-chosen",
        "--dyn-default-thinking-mode",
        "disabled",
    ])
    .expect("aggregated parses");
    assert_eq!(aggregated.component, "operator-chosen");
    assert_eq!(aggregated.disaggregation_mode, AGG);
    assert_eq!(aggregated.model_name, "model-source");
    assert_eq!(
        aggregated.default_thinking_mode.as_deref(),
        Some("disabled")
    );
    assert!(
        aggregated.enable_kv_routing,
        "source discovery controls whether publishers start"
    );

    for (mode, expected) in [
        (DisaggregationMode::Prefill, "prefill"),
        (DisaggregationMode::Decode, "backend"),
    ] {
        let (_, config) = parse(&[
            "--component",
            "operator-chosen",
            "--disaggregation-mode",
            &mode.to_string(),
        ])
        .expect("a disaggregated leg parses");
        assert_eq!(
            config.component,
            mode.discovery_component(),
            "{mode} must register under its own component, not the configured one"
        );
        assert_eq!(config.component, expected);
        assert_eq!(config.disaggregation_mode, mode);
    }

    for rejected in [
        vec!["--disaggregation-mode", "encode"],
        vec!["--route-to-encoder"],
    ] {
        assert!(
            parse(&rejected).is_err(),
            "{rejected:?} is not supported by this sidecar"
        );
    }
    assert!(
        TrtllmSidecarEngine::from_args(vec![
            "dynamo-trtllm-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            "127.0.0.1:50051".to_string(),
            "--model-path".to_string(),
            "   ".to_string(),
        ])
        .is_err(),
        "an empty model path has nothing to tokenize with"
    );
}
