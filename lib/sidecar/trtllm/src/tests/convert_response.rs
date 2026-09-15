// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reducing the server's streamed events into `LLMEngineOutput`s.

use super::*;

#[test]
fn token_then_finished_produces_delta_then_terminal_usage() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);

    let delta = state
        .convert(token_response(vec![
            logprob_token(7, -0.1),
            logprob_token(8, -0.2),
        ]))
        .expect("convert token")
        .expect("delta");
    assert_eq!(delta.token_ids, [7, 8]);
    assert!(delta.finish_reason.is_none());
    let log_probs = delta.log_probs.as_ref().expect("log_probs");
    assert_eq!(log_probs, &[-0.1, -0.2]);
    assert_eq!(delta.top_logprobs.as_ref().unwrap().len(), 2);

    let finished = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Length as i32,
                message: String::new(),
                stop_match: None,
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
            cached_prompt_tokens: None,
            reasoning_tokens: None,
        }),
    };
    let terminal = state
        .convert(finished)
        .expect("convert finished")
        .expect("terminal");
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
    let usage = terminal.completion_usage.as_ref().expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 2));
}

#[test]
fn unsupported_output_index_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let response = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(1),
            tokens: vec![logprob_token(1, -0.1)],
            text: String::new(),
        })),
        usage: None,
    };
    assert!(state.convert(response).is_err());
}

#[test]
fn missing_event_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let empty = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: None,
        usage: None,
    };
    assert!(state.convert(empty).is_err());
}

#[test]
fn unspecified_finish_reason_is_rejected() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let finished = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Unspecified as i32,
                message: String::new(),
                stop_match: None,
            },
        )),
        usage: None,
    };
    assert!(state.convert(finished).is_err());
}

#[test]
fn engine_error_event_is_surfaced_as_error() {
    let req = request();
    let mut state = ResponseState::new(&req, AGG);
    let error = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Error(pb::EngineError {
            code: pb::ErrorCode::Internal as i32,
            message: "boom".to_string(),
            retryable: false,
        })),
        usage: None,
    };
    let error = state.convert(error).expect_err("engine error must surface");
    assert!(error.to_string().contains("boom"));
}

/// Cache hits are measured during the context phase, so a decode worker can
/// only report them by carrying the handoff's count through to its terminal.
/// An aggregated worker reads its own engine's count off the same terminal.
#[test]
fn cached_prompt_tokens_reach_the_client_on_both_paths() {
    let terminal = |mut state: ResponseState, reported: Option<u32>| {
        let response = pb::GenerateResponse {
            request_id: "req".to_string(),
            event: Some(pb::generate_response::Event::Finished(
                pb::GenerationFinished {
                    output_index: Some(0),
                    reason: pb::FinishReason::Stop as i32,
                    ..Default::default()
                },
            )),
            usage: Some(pb::Usage {
                prompt_tokens: 11,
                completion_tokens: 3,
                total_tokens: 14,
                cached_prompt_tokens: reported,
                reasoning_tokens: None,
            }),
        };
        state
            .convert(response)
            .expect("finished converts")
            .expect("finished yields a terminal")
            .completion_usage
            .expect("usage is set")
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens)
    };

    assert_eq!(
        terminal(ResponseState::new(&request(), AGG), Some(7)),
        Some(7)
    );

    let mut decode_request = request();
    decode_request.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session()).expect("handoff"),
        prompt_tokens_details: Some(dynamo_backend_common::PromptTokensDetails {
            audio_tokens: None,
            cached_tokens: Some(5),
        }),
    });
    let state = ResponseState::new(&decode_request, DisaggregationMode::Decode);
    assert_eq!(
        terminal(state, None),
        Some(5),
        "the decode leg reports the prefill leg's cache hits"
    );

    // The decode engine counts the blocks transferred into it as cache hits,
    // which would read as a ~100% prefix-cache hit rate on every disaggregated
    // request. The handoff's count is the real one.
    let state = ResponseState::new(&decode_request, DisaggregationMode::Decode);
    assert_eq!(
        terminal(state, Some(11)),
        Some(5),
        "the decode engine's own count must not overwrite the handoff's"
    );
    let state = ResponseState::new(&decode_request, DisaggregationMode::Decode);
    assert_eq!(
        terminal(state, Some(0)),
        Some(5),
        "a decode engine reporting zero must not erase the handoff's count"
    );

    // A handoff that carried no count means the context phase measured no
    // hits. The decode engine's own count is about transferred blocks, so it
    // must not fill the gap.
    let mut uncounted = decode_request.clone();
    uncounted.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: crate::disagg::session_to_json(fake_session()).expect("handoff"),
        prompt_tokens_details: None,
    });
    let state = ResponseState::new(&uncounted, DisaggregationMode::Decode);
    assert_eq!(terminal(state, Some(11)), None);

    // A decode leg that ran its own context phase (conditional disaggregation
    // bypassed the prefill worker) has no handoff, so its engine is the only
    // source.
    let state = ResponseState::new(&request(), DisaggregationMode::Decode);
    assert_eq!(terminal(state, Some(4)), Some(4));

    // A count measured against an expanded prompt must not exceed the prompt
    // the client sent.
    assert_eq!(
        terminal(ResponseState::new(&request(), AGG), Some(9_999)),
        Some(11)
    );
}

/// The router sheds and migrates on an overload. Flattening it to a generic
/// engine error turns a retryable condition into an opaque 500.
#[test]
fn engine_error_codes_the_router_acts_on_are_preserved() {
    let overloaded = engine_error(pb::EngineError {
        code: pb::ErrorCode::Overloaded as i32,
        message: "at capacity".to_string(),
        retryable: true,
    });
    assert!(matches!(
        overloaded.error_type(),
        ErrorType::WorkerOverloaded
    ));

    let cancelled = engine_error(pb::EngineError {
        code: pb::ErrorCode::Cancelled as i32,
        message: "aborted".to_string(),
        retryable: false,
    });
    assert!(matches!(cancelled.error_type(), ErrorType::Cancelled));
}

/// The OpenEngine contract requires an explicit output index. Defaulting an
/// absent one to zero hides exactly the drift the index check exists to catch.
#[test]
fn a_response_without_an_output_index_is_rejected() {
    let mut state = ResponseState::new(&request(), AGG);
    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: None,
            tokens: vec![pb::TokenInfo {
                token_id: 7,
                ..Default::default()
            }],
            ..Default::default()
        })),
        usage: None,
    };
    assert!(state.convert(response).is_err());
}
