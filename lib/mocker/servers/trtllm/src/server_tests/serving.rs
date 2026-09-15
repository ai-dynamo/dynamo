// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission, validation, and the shape of an aggregated response stream.

use super::*;
use dynamo_mocker::live::deterministic_token_id;

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
    let service = slow_service();
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
        slow_args(),
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

/// The mocker must refuse what a real engine refuses, or a test passes here and
/// fails in production.
#[tokio::test]
async fn unsupported_request_features_are_refused() {
    let service = service();
    type Mutate = fn(&mut pb::GenerateRequest);
    let cases: [(&str, Mutate); 4] = [
        ("multimodal media", |r| {
            r.media.push(pb::MediaItem::default());
        }),
        ("LoRA selection", |r| r.lora_name = "adapter".to_string()),
        ("num_sequences", |r| {
            r.sampling = Some(pb::SamplingParams {
                num_sequences: Some(2),
                ..Default::default()
            })
        }),
        ("cache_salt", |r| {
            r.kv = Some(pb::KvOptions {
                cache_salt: Some("tenant".to_string()),
                ..Default::default()
            })
        }),
    ];
    for (label, mutate) in cases {
        let mut request = request("req-unsupported", 2);
        mutate(&mut request);
        let error = generate_error(&service, request).await;
        assert_eq!(error.code(), Code::Unimplemented, "case '{label}'");
    }
}

/// A request cannot be both a context request and carry a session; that arm is
/// unreachable from the role-mismatch cases.
#[tokio::test]
async fn context_only_and_a_session_together_are_rejected() {
    let session = prefill_session("pf-both").await;
    let decode = decode_service();
    let mut request = request("dc-both", 4);
    request.extra = Some(context_only_extra());
    request.kv = Some(pb::KvOptions {
        session: Some(session),
        ..Default::default()
    });
    let error = generate_error(&decode, request).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("cannot be both"), "{error}");
}

/// Only requests the server accepted are recorded, and the window is bounded.
#[tokio::test]
async fn only_accepted_requests_are_recorded() {
    let service = service();
    drain(&service, request("req-ok", 2)).await.unwrap();
    let mut rejected = request("req-rejected", 2);
    rejected.model = String::new();
    let _ = generate_error(&service, rejected).await;

    let received = service.received_requests();
    assert_eq!(received.len(), 1, "a rejected request must not be recorded");
    assert_eq!(received[0].request_id, "req-ok");
}

/// `GetServerInfo` and `GetLoad` are part of the surface a real server answers,
/// so pin the fields a client would read rather than leaving them untested.
#[tokio::test]
async fn control_reports_server_identity_and_load() {
    let service = service();
    let info = service
        .get_server_info(Request::new(pb::GetServerInfoRequest::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.engine_name, "tensorrt_llm");
    assert_eq!(info.engine_role, pb::EngineRole::Aggregated as i32);
    assert_ne!(info.schema_revision, 0, "zero is invalid per the proto");
    assert_eq!(info.supported_models, vec!["mocker-model".to_string()]);

    let load = service
        .get_load(Request::new(pb::GetLoadRequest::default()))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(load.instance_id, info.instance_id);
    // KV sessions only exist in the disaggregated roles.
    assert_eq!(load.active_kv_sessions, None);
    assert!(load.total_kv_blocks.is_some());
}

/// A real TensorRT-LLM server loads one model and serves it under whatever
/// non-empty name a request carries -- verified against 1.3.0rc26, whose
/// `Generate` and `GetModelInfo` both answer for an unrelated name. Rejecting a
/// mismatch here would fail requests the real engine serves.
#[tokio::test]
async fn any_non_empty_model_name_is_served() {
    let service = service();

    let mut renamed = request("req-renamed", 2);
    renamed.model = "some-other-model".to_string();
    let responses = drain(&service, renamed).await.unwrap();
    assert!(
        events(&responses)
            .iter()
            .any(|event| matches!(event, pb::generate_response::Event::Finished(_))),
        "a request naming another model must still be served"
    );

    let mut anonymous = request("req-anonymous", 2);
    anonymous.model = String::new();
    let error = generate_error(&service, anonymous).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("non-empty"), "{error}");
}

/// A real engine ends on a stop condition with `STOP` and a `stop_match`, which
/// is the only input to the sidecar's `stop_reason` mapping. Ending every
/// request at `LENGTH` would leave that mapping unexercised on the wire.
#[tokio::test]
async fn a_matched_stop_token_ends_the_request_with_stop_and_a_match() {
    let service = service();
    // The plan is deterministic, so the third token is knowable up front.
    let stop_token = deterministic_token_id(config().seed, "req-stop", 2);
    let mut stopping = request("req-stop", 8);
    stopping.stopping = Some(pb::StoppingOptions {
        max_tokens: Some(8),
        conditions: vec![pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopTokenId(stop_token)),
        }],
        ..Default::default()
    });

    let responses = drain(&service, stopping).await.unwrap();
    let events = events(&responses);
    let tokens = events
        .iter()
        .filter(|event| matches!(event, pb::generate_response::Event::Token(_)))
        .count();
    assert_eq!(tokens, 2, "the matched token is not streamed");

    let pb::generate_response::Event::Finished(finished) = events.last().unwrap() else {
        panic!("the stream must end with a terminal");
    };
    assert_eq!(finished.reason, pb::FinishReason::Stop as i32);
    assert_eq!(
        finished.stop_match,
        Some(pb::StopMatch {
            r#match: Some(pb::stop_match::Match::StopTokenId(stop_token)),
        })
    );
}

/// `min_tokens` outranks a stop condition, as it does on a real engine.
#[tokio::test]
async fn a_stop_token_below_min_tokens_does_not_fire() {
    let service = service();
    let stop_token = deterministic_token_id(config().seed, "req-min", 1);
    let mut stopping = request("req-min", 6);
    stopping.stopping = Some(pb::StoppingOptions {
        max_tokens: Some(6),
        min_tokens: Some(4),
        conditions: vec![pb::StopCondition {
            condition: Some(pb::stop_condition::Condition::StopTokenId(stop_token)),
        }],
        ..Default::default()
    });

    let responses = drain(&service, stopping).await.unwrap();
    let pb::generate_response::Event::Finished(finished) = events(&responses).pop().unwrap() else {
        panic!("the stream must end with a terminal");
    };
    assert_eq!(finished.reason, pb::FinishReason::Length as i32);
}

/// A real engine recomputes the last prompt token however warm the cache is, so
/// a cache hit never covers the whole prompt. The sidecar's disaggregated cache
/// accounting reads this number.
#[tokio::test]
async fn a_cache_hit_never_covers_the_whole_prompt() {
    let service = TrtllmMockerService::new(config(), admitting_args()).unwrap();

    let cached_of = |responses: &[pb::GenerateResponse]| {
        responses
            .iter()
            .find_map(|response| response.usage.as_ref())
            .expect("a terminal carries usage")
            .cached_prompt_tokens
    };

    let first = drain(&service, request("cache-cold", 2)).await.unwrap();
    assert_eq!(cached_of(&first), Some(0), "a cold prompt has no hits");

    let second = drain(&service, request("cache-warm", 2)).await.unwrap();
    let prompt_tokens = 4;
    assert_eq!(
        cached_of(&second),
        Some(prompt_tokens - 1),
        "the last prompt token is always recomputed"
    );
}
