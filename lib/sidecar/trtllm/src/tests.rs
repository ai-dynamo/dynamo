// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dynamo_backend_common::{
    DisaggregationMode, ErrorType, FinishReason, GenerateContext, LLMEngine, MultimodalData,
    OutputOptions, PrefillResult, PreprocessedRequest, SamplingOptions, StopConditions, StopReason,
};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig};
use futures::StreamExt;
use serde_json::json;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;

use crate::client::TrtllmClient;
use crate::engine::{RoutingSupervision, TrtllmSidecarEngine};
use crate::model::ConfiguredModel;
use crate::proto as pb;

use crate::test_fixtures::*;

mod e2e;
mod encoder;
mod encoder_cleanup;
mod kv_routing;
mod support;

use support::*;

/// `PrefillReady` is the prefill worker's terminal chunk: no tokens, and the
/// handoff the decode worker will replay.
#[tokio::test]
async fn prefill_ready_is_the_terminal_handoff() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    engine.start(0).await.expect("start");

    let mut req = request();
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        prefill_dp_rank: Some(0),
        dp_rank: Some(7),
        ..Default::default()
    });
    let outputs = collect(&engine, req).await;
    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        [None]
    );
    let terminal = outputs
        .iter()
        .find(|output| output.finish_reason.is_some())
        .expect("a terminal chunk");
    // The frontend's prefill router chains into decode only for `Length`; any
    // other terminal reason is returned to the caller as a finished request.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));

    assert!(
        outputs.iter().all(|output| output.token_ids.is_empty()),
        "a prefill worker must not stream tokens to the client"
    );
    let handoff = terminal
        .disaggregated_params
        .as_ref()
        .expect("terminal carries the prefill handoff");
    assert_eq!(handoff["session_id"], json!("12345"));
    assert_eq!(handoff["transfer_backend"], json!("NIXL"));
    assert_eq!(handoff["endpoints"][0]["port"], json!(5601));
    assert_eq!(handoff["attributes"]["first_gen_tokens"], json!([42]));
}

/// The decode worker replays the prefill handoff verbatim in `kv.session`.
#[tokio::test]
async fn decode_request_replays_the_prefill_session() {
    let server = FakeServer::start(FakeTrtllm::default()).await;

    let mut agent_request = request();
    agent_request.agent_context = Some(
        serde_json::from_value(json!({"session_id": "agentx-session"}))
            .expect("valid agent context"),
    );

    // Phase 1: prefill produces the handoff.
    let prefill = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    prefill.start(0).await.expect("start prefill");
    let handoff = collect(&prefill, agent_request.clone())
        .await
        .into_iter()
        .find_map(|output| output.disaggregated_params.map(|params| *params))
        .expect("prefill handoff");

    // Phase 2: decode replays it.
    let decode = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Decode);
    decode.start(0).await.expect("start decode");
    let mut req = agent_request;
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        dp_rank: Some(0),
        prefill_dp_rank: Some(7),
        ..Default::default()
    });
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let outputs = collect(&decode, req).await;
    assert_eq!(
        server.service.target_dp_ranks.lock().await.as_slice(),
        [None, None]
    );
    assert!(
        outputs.iter().any(|output| !output.token_ids.is_empty()),
        "the decode worker streams the completion"
    );

    let requests = server.service.requests.lock().await;
    for request in requests.iter() {
        let extra = request.extra.as_ref().expect("request extensions");
        assert_eq!(
            extra.fields["detokenize"].kind,
            Some(prost_types::value::Kind::BoolValue(false))
        );
        assert_eq!(
            extra.fields["conversation_id"].kind,
            Some(prost_types::value::Kind::StringValue(
                "agentx-session".into()
            )),
            "both RPC legs must retain the same conversation identity"
        );
    }
    let session = requests[1]
        .kv
        .as_ref()
        .and_then(|kv| kv.session.as_ref())
        .expect("decode request carries kv.session")
        .clone();
    assert_eq!(
        session,
        fake_session(),
        "the handoff must round-trip intact"
    );
}

#[tokio::test]
async fn aggregated_generation_streams_delta_then_terminal() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 2);
    let config = engine.start(0).await.expect("start");
    assert_eq!(config.model, "model-source");
    // GetModelInfo reports max_context_length 4096.
    assert_eq!(config.llm.unwrap().context_length, Some(4096));

    let mut req = request();
    req.routing = Some(dynamo_backend_common::engine::RoutingHints {
        dp_rank: Some(0),
        ..Default::default()
    });
    let outputs = collect(&engine, req).await;
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
async fn cancellation_dispatches_only_staged_kv_handoffs() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for mode in [DisaggregationMode::Decode, AGG] {
            let service = FakeTrtllm::default();
            service.hang.store(true, Ordering::SeqCst);
            if !mode.is_decode() {
                *service.encoder_role.lock().await = Some("consume".into());
            }
            let server = FakeServer::start(service).await;
            let (engine, req) = if mode.is_decode() {
                let mut req = request();
                req.prefill_result = Some(PrefillResult {
                    disaggregated_params: crate::disagg::session_to_json(fake_session()).unwrap(),
                    prompt_tokens_details: None,
                });
                (engine_in_mode(&server.endpoint, 1, mode), req)
            } else {
                (
                    encoder_consumer_engine(&server.endpoint),
                    request_with_encoder_result(),
                )
            };
            engine.start(0).await.unwrap();
            let context = dynamo_backend_common::testing::mock_context();
            context.stop_generating();
            let mut stream = engine
                .generate(req, GenerateContext::new(context, None))
                .await
                .unwrap();

            let first = stream.next().await.unwrap().unwrap();
            if mode.is_decode() {
                assert_eq!(first.finish_reason, None);
                assert_eq!(first.token_ids, [42]);
                let terminal = stream.next().await.unwrap().unwrap();
                assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
                assert_eq!(server.service.requests.lock().await.len(), 1);
            } else {
                assert_eq!(first.finish_reason, Some(FinishReason::Cancelled));
                assert!(server.service.requests.lock().await.is_empty());
            }
            engine.cleanup().await.unwrap();
        }
    })
    .await
    .expect("cancellation must resume after the transfer lands");
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

    for unsupported in [multiple, beam, embeds, prompt_logprobs] {
        let context = dynamo_backend_common::testing::mock_context();
        let result = engine
            .generate(unsupported, GenerateContext::new(context, None))
            .await;
        assert!(result.is_err());
    }
    assert!(server.service.requests.lock().await.is_empty());
}

#[tokio::test]
async fn visible_stop_token_is_returned_and_hidden_stop_token_is_not() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine(&server.endpoint, 1);
    engine.start(0).await.expect("start");
    let token_ids = |outputs: &[dynamo_backend_common::LLMEngineOutput]| {
        outputs
            .iter()
            .flat_map(|output| output.token_ids.clone())
            .collect::<Vec<_>>()
    };

    let mut visible = request();
    visible.stop_conditions.stop_token_ids_visible = Some(vec![42]);
    let outputs = collect(&engine, visible).await;
    assert_eq!(token_ids(&outputs), [42]);
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.token_ids, [42], "held for the terminal's usage");
    assert!(terminal.completion_usage.is_some());
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(42)));

    // Retaining a visible stop makes the engine return hidden stops too.
    let mut hidden = request();
    hidden.stop_conditions.stop_token_ids_hidden = Some(vec![42]);
    hidden.stop_conditions.stop_token_ids_visible = Some(vec![7]);
    let outputs = collect(&engine, hidden).await;
    assert!(token_ids(&outputs).is_empty());
    let terminal = outputs.last().unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(42)));
}

#[test]
fn hidden_stop_waits_for_authoritative_terminal_usage() {
    let mut request = request();
    request.output_options.logprobs = None;
    request.stop_conditions.stop_token_ids_hidden = Some(vec![42]);
    let mut state = crate::convert::ResponseState::new(&request, DisaggregationMode::Aggregated);
    let output = state
        .convert(pb::GenerateResponse {
            event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
                tokens: vec![pb::TokenInfo {
                    token_id: 42,
                    ..Default::default()
                }],
                output_index: Some(0),
                ..Default::default()
            })),
            ..Default::default()
        })
        .expect("token");
    assert!(
        output.is_none(),
        "EOS must not cancel the stream before final usage"
    );
    let terminal = state
        .convert(pb::GenerateResponse {
            event: Some(pb::generate_response::Event::Finished(
                pb::GenerationFinished {
                    output_index: Some(0),
                    reason: pb::FinishReason::Stop as i32,
                    stop_match: Some(pb::StopMatch {
                        r#match: Some(pb::stop_match::Match::StopTokenId(42)),
                    }),
                    ..Default::default()
                },
            )),
            usage: Some(pb::Usage {
                prompt_tokens: 88,
                completion_tokens: 23,
                total_tokens: 111,
                ..Default::default()
            }),
            ..Default::default()
        })
        .expect("finished")
        .expect("terminal");
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(terminal.completion_usage.unwrap().prompt_tokens, 88);
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
        error.to_string().contains("GetModelInfo"),
        "the error must name the required RPC: {error}"
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

/// Transient metadata failures must not register an incorrect model name.
#[tokio::test]
async fn start_waits_for_a_server_that_is_not_serving_yet() {
    for configured_length in [None, Some(8192)] {
        let service = FakeTrtllm::default();
        service.unavailable_model_info.store(true, Ordering::SeqCst);
        let server = FakeServer::start(service).await;
        let ready = Arc::clone(&server.service.unavailable_model_info);
        let unavailable = Arc::clone(&server.service.model_info_unavailable);
        let release = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(5), unavailable.notified())
                .await
                .expect("first GetModelInfo was not observed");
            ready.store(false, Ordering::SeqCst);
        });
        let engine = engine_with(
            &server.endpoint,
            GrpcTransportConfig {
                startup_deadline: Duration::from_secs(5),
                ..impatient_transport()
            },
            configured_length,
            AGG,
        );
        let config = engine.start(0).await.expect("start once the engine serves");
        release.await.expect("release discovery");
        assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
        assert!(server.service.model_info_calls.load(Ordering::SeqCst) > 1);
    }
}

/// The public model name must not replace the path used to load artifacts.
#[tokio::test]
async fn engine_registers_reported_name_separately_from_model_path() {
    for configured_length in [None, Some(8192)] {
        let service = FakeTrtllm::default();
        *service.served_model_name.lock().await = Some("org/served-model".into());
        let server = FakeServer::start(service).await;
        let engine = engine_with(&server.endpoint, transport(1), configured_length, AGG);
        let config = engine.start(0).await.expect("start");
        assert_eq!(config.model, "model-source");
        assert_eq!(
            config.served_model_name.as_deref(),
            Some("org/served-model")
        );
    }
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

/// The model name must come from the engine, even with a configured window.
#[tokio::test]
async fn a_configured_context_length_requires_model_info_and_a_served_name() {
    for missing_control in [true, false] {
        let service = FakeTrtllm::default();
        service.no_control.store(missing_control, Ordering::SeqCst);
        if !missing_control {
            *service.served_model_name.lock().await = Some(String::new());
        }
        let server = FakeServer::start(service).await;
        let engine = engine_with(&server.endpoint, impatient_transport(), Some(8192), AGG);
        let error = engine
            .start(0)
            .await
            .expect_err("model identity is required");
        let expected = if missing_control {
            "GetModelInfo"
        } else {
            "served_model_name"
        };
        assert!(
            error.to_string().contains(expected),
            "unexpected error: {error}"
        );
    }
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

#[tokio::test]
async fn cancellation_during_dispatch_defers_only_staged_kv() {
    tokio::time::timeout(Duration::from_secs(10), async {
        for mode in [DisaggregationMode::Decode, AGG] {
            let service = FakeTrtllm::default();
            service.hang_before_stream.store(true, Ordering::SeqCst);
            if !mode.is_decode() {
                *service.encoder_role.lock().await = Some("consume".into());
            }
            let server = FakeServer::start(service).await;
            let (engine, req) = if mode.is_decode() {
                let mut req = request();
                req.prefill_result = Some(PrefillResult {
                    disaggregated_params: crate::disagg::session_to_json(fake_session()).unwrap(),
                    prompt_tokens_details: None,
                });
                (engine_in_mode(&server.endpoint, 1, mode), req)
            } else {
                (
                    encoder_consumer_engine(&server.endpoint),
                    request_with_encoder_result(),
                )
            };
            let engine = Arc::new(engine);
            engine.start(0).await.unwrap();
            let context = dynamo_backend_common::testing::mock_context();
            let mut dispatch = tokio::spawn({
                let engine = Arc::clone(&engine);
                let context = context.clone();
                async move {
                    engine
                        .generate(req, GenerateContext::new(context, None))
                        .await
                }
            });

            // The engine accepted the request but has withheld response headers.
            while server.service.requests.lock().await.is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            context.stop_generating();
            let mut stream = if mode.is_decode() {
                assert!(
                    tokio::time::timeout(Duration::from_millis(200), &mut dispatch)
                        .await
                        .is_err()
                );
                engine.cleanup().await.unwrap();
                dispatch.await.unwrap().unwrap()
            } else {
                tokio::time::timeout(Duration::from_secs(5), &mut dispatch)
                    .await
                    .expect("encoder consumer cancellation must not wait for headers")
                    .unwrap()
                    .unwrap()
            };
            let terminal = stream.next().await.unwrap().unwrap();
            assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
            engine.cleanup().await.unwrap();
        }
    })
    .await
    .expect("shutdown must release deferred dispatch");
}

// Regression: eager discovery must neither bypass Worker start nor reconnect
// during that non-cancellable phase after bootstrap has already succeeded.
#[tokio::test]
async fn bootstrap_discovers_once_but_requires_start_before_serving() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let bootstrap = TrtllmSidecarEngine::try_from_args_async(vec![
        "dynamo-trtllm-sidecar".into(),
        "--grpc-endpoint".into(),
        server.endpoint.clone(),
        "--model-path".into(),
        "model-source".into(),
    ])
    .unwrap();
    let (engine, _) = bootstrap.1.await.unwrap();
    assert_eq!(server.service.model_info_calls.load(Ordering::SeqCst), 1);
    let context = dynamo_backend_common::testing::mock_context();
    let result = engine
        .generate(request(), GenerateContext::new(context, None))
        .await;
    let error = result.err().expect("bootstrap must not enable generation");
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(dynamo_backend_common::BackendError::EngineShutdown)
    );
    assert!(server.service.requests.lock().await.is_empty());

    // Metadata was resolved already; a transient Control failure cannot make
    // Worker repeat discovery and hang in its non-cancellable start phase.
    server
        .service
        .unavailable_model_info
        .store(true, Ordering::SeqCst);
    let config = engine.start(0).await.unwrap();
    assert_eq!(config.llm.unwrap().context_length, Some(4096));
    assert_eq!(server.service.model_info_calls.load(Ordering::SeqCst), 1);
    let outputs = collect(&engine, request()).await;
    assert_eq!(outputs[0].token_ids, [42]);
    assert_eq!(
        outputs.last().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
    engine.cleanup().await.unwrap();
}
