// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_mocker::common::protocols::MockEngineArgsBuilder;
use dynamo_trtllm_sidecar::disagg::context_only_extra;
use futures::StreamExt;
use pb::control_server::Control;
use pb::inference_server::Inference;
use prost_types::{Value, value::Kind};
use tonic::Code;

use super::*;

fn admitting_args() -> MockEngineArgs {
    MockEngineArgsBuilder::default()
        .engine_type(EngineType::Trtllm)
        .num_gpu_blocks(4_096usize)
        .block_size(4usize)
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

fn config() -> MockerServerConfig {
    MockerServerConfig {
        context_length: 1_024,
        ..Default::default()
    }
}

fn service() -> TrtllmMockerService {
    TrtllmMockerService::new(config(), admitting_args()).unwrap()
}

/// Neither the service nor the response stream implements `Debug`, so
/// `unwrap_err` is unavailable on these results.
fn construction_error(config: MockerServerConfig, args: MockEngineArgs) -> String {
    match TrtllmMockerService::new(config, args) {
        Ok(_) => panic!("expected the constructor to fail"),
        Err(error) => error.to_string(),
    }
}

async fn generate_error(service: &TrtllmMockerService, request: pb::GenerateRequest) -> Status {
    match service.generate(Request::new(request)).await {
        Ok(_) => panic!("expected Generate to fail"),
        Err(status) => status,
    }
}

fn request(request_id: &str, max_tokens: u32) -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: "mocker-model".to_string(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: vec![1, 2, 3, 4],
        })),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }),
        ..Default::default()
    }
}

async fn drain(
    service: &TrtllmMockerService,
    request: pb::GenerateRequest,
) -> Result<Vec<pb::GenerateResponse>, Status> {
    let mut stream = service.generate(Request::new(request)).await?.into_inner();
    let mut responses = Vec::new();
    while let Some(item) = stream.next().await {
        responses.push(item?);
    }
    Ok(responses)
}

fn prefill_service() -> TrtllmMockerService {
    TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..config()
        },
        admitting_args(),
    )
    .unwrap()
}

fn decode_service() -> TrtllmMockerService {
    TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Decode,
            ..config()
        },
        admitting_args(),
    )
    .unwrap()
}

async fn prefill_session(request_id: &str) -> pb::KvSessionRef {
    let prefill = prefill_service();
    let mut prefill_request = request(request_id, 1);
    prefill_request.extra = Some(context_only_extra());
    let responses = drain(&prefill, prefill_request).await.unwrap();
    events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::PrefillReady(ready) => ready.kv_session.clone(),
            _ => None,
        })
        .expect("prefill must emit a session")
}

fn events(responses: &[pb::GenerateResponse]) -> Vec<&pb::generate_response::Event> {
    responses
        .iter()
        .map(|response| {
            response
                .event
                .as_ref()
                .expect("every response must carry an event")
        })
        .collect()
}

#[tokio::test]
async fn preparation_is_deterministic() {
    let config = config();
    let first = PreparedRequest::new(request("req-1", 4), &config).unwrap();
    let second = PreparedRequest::new(request("req-1", 4), &config).unwrap();
    assert_eq!(first.uuid, second.uuid);
    assert_eq!(first.session_id, second.session_id);
    let tokens =
        |prepared: &PreparedRequest| (0..4).map(|i| prepared.output_token(i)).collect::<Vec<_>>();
    assert_eq!(tokens(&first), tokens(&second));
}

#[tokio::test]
async fn service_requires_a_trtllm_single_rank_aggregated_engine() {
    let vllm = MockEngineArgsBuilder::default()
        .engine_type(EngineType::Vllm)
        .build()
        .unwrap();
    let error = construction_error(config(), vllm);
    assert!(error.contains("engine_type"), "{error}");

    let multi_rank = MockEngineArgsBuilder::default()
        .engine_type(EngineType::Trtllm)
        .dp_size(2u32)
        .build()
        .unwrap();
    let error = construction_error(config(), multi_rank);
    assert!(error.contains("dp_size"), "{error}");

    let error = construction_error(
        MockerServerConfig {
            max_concurrent_requests: 0,
            ..config()
        },
        admitting_args(),
    );
    assert!(error.contains("max_concurrent_requests"), "{error}");

    let error = construction_error(
        MockerServerConfig {
            context_length: 0,
            ..config()
        },
        admitting_args(),
    );
    assert!(error.contains("context_length"), "{error}");

    let error = construction_error(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            kv_port: 0,
            ..config()
        },
        admitting_args(),
    );
    assert!(error.contains("kv_port"), "{error}");
}

/// The sidecar reads exactly one field out of `GetModelInfo` and refuses to
/// start if it is absent or zero.
#[tokio::test]
async fn model_info_always_reports_a_positive_context_length() {
    let service = service();
    let info = service
        .get_model_info(Request::new(pb::GetModelInfoRequest::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.max_context_length, Some(1_024));

    let error = service
        .get_model_info(Request::new(pb::GetModelInfoRequest {
            model: "other".to_string(),
        }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::NotFound);
}

#[tokio::test]
async fn text_prompts_fail_with_an_actionable_status() {
    let mut request = request("req-text", 2);
    request.input = Some(pb::generate_request::Input::Prompt("hi".to_string()));
    let error = generate_error(&service(), request).await;
    assert_eq!(error.code(), Code::Unimplemented);
    assert!(error.message().contains("token_ids"), "{error}");
}

#[tokio::test]
async fn oversized_generation_is_rejected_before_token_planning() {
    let error = generate_error(&service(), request("req-big", request::MAX_NEW_TOKENS + 1)).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    // Without this the context-window check would satisfy the test instead.
    assert!(error.message().contains("Mocker limit"), "{error}");
}

#[tokio::test]
async fn prompt_plus_output_must_fit_the_context_window() {
    let service = TrtllmMockerService::new(
        MockerServerConfig {
            context_length: 8,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    let error = generate_error(&service, request("req-ctx", 8)).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("context length"), "{error}");

    // The same request fits once the budget does.
    assert!(drain(&service, request("req-ctx-ok", 2)).await.is_ok());
}

/// The sidecar fails the whole request if a single `TokenInfo` is missing its
/// logprob when logprobs were asked for.
#[tokio::test]
async fn every_streamed_token_carries_a_logprob_when_requested() {
    let service = service();
    let mut with = request("req-lp", 8);
    with.response = Some(pb::ResponseOptions {
        return_output_logprobs: Some(true),
        output_candidates: Some(pb::CandidateTokenSelection {
            selection: Some(pb::candidate_token_selection::Selection::TopN(3)),
        }),
        ..Default::default()
    });
    let responses = drain(&service, with).await.unwrap();
    let mut tokens = 0;
    for event in events(&responses) {
        if let pb::generate_response::Event::Token(token) = event {
            for info in &token.tokens {
                assert!(
                    info.logprob.is_some(),
                    "token {} lost its logprob",
                    info.token_id
                );
                assert!(info.rank.is_some());
                assert_eq!(info.candidates.len(), 3);
            }
            tokens += 1;
        }
    }
    assert_eq!(tokens, 8);

    let responses = drain(&service, request("req-nolp", 4)).await.unwrap();
    for event in events(&responses) {
        if let pb::generate_response::Event::Token(token) = event {
            assert!(token.tokens.iter().all(|info| info.logprob.is_none()));
        }
    }
}

/// Pins the terminal shape the sidecar requires: exactly one `finished`, never
/// `UNSPECIFIED`, `output_index` always set, no prompt event, usage only at the
/// end. A clean end without a terminal fails the request outright.
#[tokio::test]
async fn aggregated_stream_ends_with_exactly_one_finished_and_no_prompt_event() {
    let responses = drain(&service(), request("req-shape", 5)).await.unwrap();
    let events = events(&responses);

    let finished: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            pb::generate_response::Event::Finished(finished) => Some(finished),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 1);
    assert!(matches!(
        events.last().unwrap(),
        pb::generate_response::Event::Finished(_)
    ));
    assert_eq!(finished[0].reason, pb::FinishReason::Length as i32);
    assert_eq!(finished[0].output_index, Some(0));

    assert!(!events.iter().any(|event| matches!(
        event,
        pb::generate_response::Event::Prompt(_) | pb::generate_response::Event::PrefillReady(_)
    )));
    for event in &events {
        if let pb::generate_response::Event::Token(token) = event {
            assert_eq!(token.output_index, Some(0));
        }
    }

    let with_usage: Vec<_> = responses.iter().filter(|r| r.usage.is_some()).collect();
    assert_eq!(with_usage.len(), 1);
    let usage = with_usage[0].usage.as_ref().unwrap();
    assert_eq!(usage.prompt_tokens, 4);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 9);
}

#[tokio::test]
async fn duplicate_request_ids_are_rejected() {
    let service = TrtllmMockerService::new(config(), {
        MockEngineArgsBuilder::default()
            .engine_type(EngineType::Trtllm)
            .num_gpu_blocks(4_096usize)
            .block_size(4usize)
            .speedup_ratio(0.01)
            .build()
            .unwrap()
    })
    .unwrap();
    let _first = service
        .generate(Request::new(request("req-dup", 64)))
        .await
        .unwrap();
    let error = generate_error(&service, request("req-dup", 64)).await;
    assert_eq!(error.code(), Code::AlreadyExists);
}

#[tokio::test]
async fn concurrent_request_limit_rejects_an_extra_stream() {
    let service = TrtllmMockerService::new(
        MockerServerConfig {
            max_concurrent_requests: 2,
            ..config()
        },
        MockEngineArgsBuilder::default()
            .engine_type(EngineType::Trtllm)
            .num_gpu_blocks(4_096usize)
            .block_size(4usize)
            .speedup_ratio(0.01)
            .build()
            .unwrap(),
    )
    .unwrap();
    let _first = service
        .generate(Request::new(request("a", 64)))
        .await
        .unwrap();
    let _second = service
        .generate(Request::new(request("b", 64)))
        .await
        .unwrap();
    let error = generate_error(&service, request("c", 64)).await;
    assert_eq!(error.code(), Code::ResourceExhausted);
}

/// A stalled consumer must not trip LiveEngine's slow-consumer shedding.
#[tokio::test]
async fn streaming_survives_a_producer_that_outruns_a_stalled_consumer() {
    let service = service();
    let mut stream = service
        .generate(Request::new(request("req-slow", 50)))
        .await
        .unwrap()
        .into_inner();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut tokens = 0;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.unwrap().event.unwrap() {
            pb::generate_response::Event::Token(_) => tokens += 1,
            pb::generate_response::Event::Finished(finished) => terminal = Some(finished),
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!(tokens, 50);
    assert_eq!(terminal.unwrap().reason, pb::FinishReason::Length as i32);
}

#[tokio::test]
async fn capacity_rejection_is_an_in_band_overloaded_error() {
    // One 4-token block cannot hold a 5-token prompt, so the scheduler rejects
    // the request after it was admitted.
    let service = TrtllmMockerService::new(
        config(),
        MockEngineArgsBuilder::default()
            .engine_type(EngineType::Trtllm)
            .num_gpu_blocks(1usize)
            .block_size(4usize)
            .max_num_seqs(Some(8))
            .max_num_batched_tokens(Some(64))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
    .unwrap();
    let mut oversized = request("req-cap", 4);
    oversized.input = Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
        ids: vec![1, 2, 3, 4, 5],
    }));

    // The RPC itself must succeed: an accepted request reports failure in-band
    // and the stream still closes OK, per the OpenEngine error contract.
    let responses = drain(&service, oversized).await.unwrap();
    let error = events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::Error(error) => Some(error),
            _ => None,
        })
        .expect("expected an in-band EngineError");
    assert_eq!(error.code, pb::ErrorCode::Overloaded as i32);
    assert!(error.retryable);
    assert!(!responses.iter().any(|response| matches!(
        response.event,
        Some(pb::generate_response::Event::Finished(_))
    )));
}

#[tokio::test]
async fn abort_reports_aborted_then_already_finished() {
    let service = TrtllmMockerService::new(
        config(),
        MockEngineArgsBuilder::default()
            .engine_type(EngineType::Trtllm)
            .num_gpu_blocks(4_096usize)
            .block_size(4usize)
            .speedup_ratio(0.01)
            .build()
            .unwrap(),
    )
    .unwrap();
    let mut stream = service
        .generate(Request::new(request("req-abort", 512)))
        .await
        .unwrap()
        .into_inner();
    let _first = stream.next().await.unwrap().unwrap();

    let abort = |target| {
        let service = service.clone();
        async move {
            service
                .abort(Request::new(pb::AbortRequest {
                    target: Some(target),
                }))
                .await
                .unwrap()
                .into_inner()
                .status
        }
    };

    let status = abort(pb::abort_request::Target::RequestId("req-abort".into())).await;
    assert_eq!(status, pb::AbortStatus::Aborted as i32);

    // Drain what is left; the request must still end with one terminal event.
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let Some(pb::generate_response::Event::Finished(finished)) = item.unwrap().event {
            terminal = Some(finished);
        }
    }
    assert_eq!(
        terminal
            .expect("aborted request must still terminate")
            .reason,
        pb::FinishReason::Cancelled as i32
    );

    let status = abort(pb::abort_request::Target::RequestId("req-abort".into())).await;
    assert_eq!(status, pb::AbortStatus::AlreadyFinished as i32);
    let status = abort(pb::abort_request::Target::RequestId("never-existed".into())).await;
    assert_eq!(status, pb::AbortStatus::AlreadyFinished as i32);

    let error = service
        .abort(Request::new(pb::AbortRequest { target: None }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
}

/// The real TensorRT-LLM server leaves these unimplemented. A mocker that
/// answered them would let a KV-routing test pass here and fail in production.
#[tokio::test]
async fn kv_event_rpcs_are_unimplemented() {
    let service = service();
    let error = service
        .get_kv_event_sources(Request::new(pb::GetKvEventSourcesRequest::default()))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::Unimplemented);
    match service
        .subscribe_kv_events(Request::new(pb::SubscribeKvEventsRequest::default()))
        .await
    {
        Ok(_) => panic!("SubscribeKvEvents must be unimplemented"),
        Err(error) => assert_eq!(error.code(), Code::Unimplemented),
    }
}

#[tokio::test]
async fn lora_rpcs_are_unimplemented() {
    let service = service();
    assert_eq!(
        service
            .list_loras(Request::new(pb::ListLorasRequest::default()))
            .await
            .unwrap_err()
            .code(),
        Code::Unimplemented
    );
}

#[tokio::test]
async fn health_is_ready_but_the_inference_probe_is_not_simulated() {
    let service = service();
    let health = service
        .health(Request::new(pb::HealthRequest::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(health.state, pb::HealthState::Ready as i32);
    assert_eq!(health.checks.len(), 3);

    let error = service
        .health(Request::new(pb::HealthRequest {
            include_inference_probe: true,
            ..Default::default()
        }))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::Unimplemented);
}

#[tokio::test]
async fn role_validation_rejects_mismatched_disaggregation_payloads() {
    let aggregated = service();
    let mut context_only = request("agg-ctx", 4);
    context_only.extra = Some(context_only_extra());
    assert_eq!(
        generate_error(&aggregated, context_only).await.code(),
        Code::FailedPrecondition
    );

    let prefill = TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    assert_eq!(
        generate_error(&prefill, request("pf-plain", 4))
            .await
            .code(),
        Code::FailedPrecondition
    );

    let decode = TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Decode,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    assert_eq!(
        generate_error(&decode, request("dc-plain", 4)).await.code(),
        Code::FailedPrecondition
    );
}

#[tokio::test]
async fn prefill_stream_ends_with_prefill_ready_and_no_finished() {
    let prefill = TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    let mut request = request("pf-1", 1);
    request.extra = Some(context_only_extra());
    let responses = drain(&prefill, request).await.unwrap();
    let events = events(&responses);

    assert!(
        !events
            .iter()
            .any(|event| matches!(event, pb::generate_response::Event::Finished(_)))
    );
    let ready = events
        .iter()
        .filter_map(|event| match event {
            pb::generate_response::Event::PrefillReady(ready) => Some(ready),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ready.len(), 1);
    let session = ready[0].kv_session.as_ref().unwrap();
    assert!(session.session_id.starts_with(handoff::SESSION_PREFIX));
    assert_eq!(session.transfer_backend, handoff::TRANSFER_BACKEND);
    assert_eq!(session.endpoints.len(), 1);
    assert!(matches!(
        events.last().unwrap(),
        pb::generate_response::Event::PrefillReady(_)
    ));
}

/// Each case mutates one leg of the handoff the way a lossy relay would. All of
/// them must be caught, or the round trip proves nothing.
#[tokio::test]
async fn decode_rejects_a_handoff_the_sidecar_mangled() {
    let session = prefill_session("pf-mangle").await;
    let decode = decode_service();

    let without = |key: &str| {
        let mut mutated = session.clone();
        let mut attributes = mutated.attributes_struct.clone().unwrap();
        attributes.fields.remove(key);
        mutated.attributes_struct = Some(attributes);
        mutated
    };
    let with_attribute = |key: &str, value: Kind| {
        let mut mutated = session.clone();
        let mut attributes = mutated.attributes_struct.clone().unwrap();
        attributes
            .fields
            .insert(key.to_string(), Value { kind: Some(value) });
        mutated.attributes_struct = Some(attributes);
        mutated
    };

    let mut no_attributes = session.clone();
    no_attributes.attributes_struct = None;
    let mut bad_backend = session.clone();
    bad_backend.transfer_backend = "NIXL".to_string();
    let mut no_endpoints = session.clone();
    no_endpoints.endpoints.clear();
    let mut lost_port = session.clone();
    lost_port.endpoints[0].port = 0;
    let mut bad_rank = session.clone();
    bad_rank.dp_rank = 3;

    // One case per distinct thing the JSON codec could do to the payload.
    let mutations = [
        ("dropped attribute", without(handoff::ATTR_REQUEST_ID)),
        ("dropped attributes", no_attributes),
        (
            "rounded fractional",
            with_attribute(handoff::ATTR_TTFT_MS, Kind::NumberValue(12.0)),
        ),
        (
            "flattened list",
            with_attribute(handoff::ATTR_FIRST_GEN_TOKENS, Kind::NumberValue(7.0)),
        ),
        ("defaulted string", bad_backend),
        ("dropped repeated", no_endpoints),
        ("defaulted number", lost_port),
        ("altered scalar", bad_rank),
    ];

    for (label, mutated) in mutations {
        let mut decode_request = request("dc-mangle", 4);
        decode_request.kv = Some(pb::KvOptions {
            session: Some(mutated),
            ..Default::default()
        });
        let error = generate_error(&decode, decode_request).await;
        assert_eq!(error.code(), Code::InvalidArgument, "mutation '{label}'");
    }

    // The untouched session still works, so the mutations above are what fail.
    let mut decode_request = request("dc-ok", 4);
    decode_request.kv = Some(pb::KvOptions {
        session: Some(session),
        ..Default::default()
    });
    assert!(drain(&decode, decode_request).await.is_ok());
}

/// A context request that does not ask for exactly one token is a client bug the
/// prefill role must surface rather than silently normalize.
#[tokio::test]
async fn prefill_requires_a_single_token_budget() {
    let prefill = prefill_service();
    let mut oversized = request("pf-budget", 8);
    oversized.extra = Some(context_only_extra());
    let error = generate_error(&prefill, oversized).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("exactly one token"), "{error}");
}

/// The decode leg replays the context phase's first token, so the two legs'
/// accounting matches a real engine's instead of inventing a fresh stream.
#[tokio::test]
async fn decode_replays_the_prefill_first_token() {
    let session = prefill_session("pf-replay").await;
    let handed_off = handoff::first_gen_token(&session).unwrap();

    let decode = decode_service();
    let mut decode_request = request("dc-replay", 3);
    decode_request.kv = Some(pb::KvOptions {
        session: Some(session),
        ..Default::default()
    });
    let responses = drain(&decode, decode_request).await.unwrap();
    let first = events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::Token(token) => Some(token.tokens[0].token_id),
            _ => None,
        })
        .unwrap();
    assert_eq!(first, handed_off);
}

/// Prompt logprobs are their own switch; gating them on the output flag would
/// emit a PromptOutput whose tokens all carry `logprob: None`.
#[tokio::test]
async fn prompt_logprobs_do_not_depend_on_the_output_flag() {
    let service = service();
    let mut request = request("req-prompt-lp", 2);
    request.response = Some(pb::ResponseOptions {
        return_prompt_logprobs: Some(true),
        return_output_logprobs: Some(false),
        ..Default::default()
    });
    let responses = drain(&service, request).await.unwrap();
    let prompt = events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::Prompt(prompt) => Some(prompt.clone()),
            _ => None,
        })
        .expect("a prompt event was requested");
    assert!(!prompt.tokens.is_empty());
    assert!(prompt.tokens.iter().all(|info| info.logprob.is_some()));
}

/// An explicit `max_tokens: 0` is a real request, not an omitted field.
#[tokio::test]
async fn zero_max_tokens_is_rejected_rather_than_defaulted() {
    let error = generate_error(&service(), request("req-zero", 0)).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("greater than zero"), "{error}");
}

/// The server records what the client put on the wire, so a test can assert the
/// request path and not only the response path.
#[tokio::test]
async fn accepted_requests_are_recorded() {
    let service = service();
    drain(&service, request("req-recorded", 2)).await.unwrap();
    let received = service.received_requests();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].request_id, "req-recorded");
    assert_eq!(received[0].stopping.as_ref().unwrap().max_tokens, Some(2));
}

/// Once the engine has finished, an abort must not claim it cancelled the
/// request -- the stream is about to report LENGTH, and the two answers would
/// contradict each other. The pump makes this window wide: the engine can run
/// to completion while the client has read nothing.
#[tokio::test]
async fn abort_after_the_engine_finished_reports_already_finished() {
    let service = service();
    let mut stream = service
        .generate(Request::new(request("req-raced", 4)))
        .await
        .unwrap()
        .into_inner();

    // Let the (instant) engine run ahead of this consumer.
    while service.active_request_count() > 0 {
        tokio::task::yield_now().await;
    }

    let status = service
        .abort(Request::new(pb::AbortRequest {
            target: Some(pb::abort_request::Target::RequestId("req-raced".into())),
        }))
        .await
        .unwrap()
        .into_inner()
        .status;
    assert_eq!(status, pb::AbortStatus::AlreadyFinished as i32);

    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let Some(pb::generate_response::Event::Finished(finished)) = item.unwrap().event {
            terminal = Some(finished);
        }
    }
    assert_eq!(
        terminal.expect("the request still terminates").reason,
        pb::FinishReason::Length as i32,
        "a finished request must report LENGTH, matching the ALREADY_FINISHED abort"
    );
}
