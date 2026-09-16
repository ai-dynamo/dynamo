// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine driven against an in-process fake OpenEngine server.

use super::*;

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

/// The server answers `openengine-target-dp-rank` with UNIMPLEMENTED, so a
/// rank hint has to be refused before dispatch: sending it anyway fails the
/// whole request with a non-migratable 5xx. `nvext.dp_rank` and the
/// `x-dynamo-dp-rank` header both land in these fields, so this is reachable
/// without a KV router.
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
            .generate(pb::GenerateRequest {
                request_id: format!("request-{index}"),
                model: "model-source".to_string(),
                input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
                    ids: vec![1, 2],
                })),
                ..Default::default()
            })
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
/// anyway is still useful -- requests that carry their own `max_tokens` are
/// served -- so this warns rather than refusing to start, and only the requests
/// that omit `max_tokens` are rejected.
#[tokio::test]
async fn start_without_a_context_length_registers_a_window_less_worker() {
    let service = FakeTrtllm::default();
    service.no_control.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine(&server.endpoint, 1);

    let config = engine.start(0).await.expect("start is not blocked");
    assert_eq!(
        config.llm.expect("llm registration").context_length,
        None,
        "an unknown window must not be registered as a real one"
    );

    let mut request = request();
    request.stop_conditions.max_tokens = None;
    let Err(error) = engine
        .generate(
            request,
            GenerateContext::new(dynamo_backend_common::testing::mock_context(), None),
        )
        .await
    else {
        panic!("a request that omits max_tokens has no budget to derive");
    };
    assert!(
        error.to_string().contains("specify max_tokens explicitly"),
        "unexpected error: {error}"
    );
}

/// A server that answers GetModelInfo but reports no context length is not
/// ready yet -- TensorRT-LLM binds its port before the model finishes loading
/// -- so the sidecar keeps asking until the operator's startup deadline.
#[tokio::test]
async fn start_retries_until_the_deadline_when_the_server_reports_no_context_length() {
    let service = FakeTrtllm::default();
    service.empty_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let config = engine.start(0).await.expect("start is not blocked");
    assert_eq!(config.llm.expect("llm registration").context_length, None);
    assert!(
        server.service.model_info_calls.load(Ordering::SeqCst) > 1,
        "a server that is still loading must be asked more than once"
    );
}

/// Once the model finishes loading, the same retry loop picks up the context
/// length -- the case the deadline exists to allow.
#[tokio::test]
async fn start_waits_for_a_server_that_is_still_loading_its_model() {
    let service = FakeTrtllm::default();
    service.empty_model_info.store(true, Ordering::SeqCst);
    let server = FakeServer::start(service).await;
    let ready = Arc::clone(&server.service.empty_model_info);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        ready.store(false, Ordering::SeqCst);
    });
    let engine = engine_with(&server.endpoint, impatient_transport(), None, AGG);

    let config = engine.start(0).await.expect("start once the model loads");
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

    let (_, aggregated) = parse(&["--component", "operator-chosen"]).expect("aggregated parses");
    assert_eq!(aggregated.component, "operator-chosen");
    assert_eq!(aggregated.disaggregation_mode, AGG);
    assert_eq!(aggregated.model_name, "model-source");
    assert!(
        !aggregated.enable_kv_routing,
        "the sidecar has no KV events"
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
