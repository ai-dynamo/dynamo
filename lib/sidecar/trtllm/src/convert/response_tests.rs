// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reducing the server's streamed events into `LLMEngineOutput`s.

use super::*;
use crate::test_fixtures::*;
use dynamo_backend_common::ErrorType;

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
            ..Default::default()
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

/// `Cancelled` must survive as itself; `Overloaded` must not become
/// `WorkerOverloaded`. The only server site that emits `Overloaded` is the
/// consumer-stall watchdog, so migrating on it would re-dispatch to a second
/// worker that stalls the same way.
#[test]
fn engine_error_codes_the_router_acts_on_are_preserved() {
    let overloaded = engine_error(pb::EngineError {
        code: pb::ErrorCode::Overloaded as i32,
        message: "at capacity".to_string(),
        retryable: true,
    });
    assert!(!matches!(
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

/// A role mismatch means the frontend routed to a worker whose engine is in
/// the other disaggregation role, so every request to it fails identically.
/// An opaque 500 gives the operator nothing to search for, and the class must
/// stay client-side: retrying elsewhere cannot fix a misconfigured deployment.
#[test]
fn a_role_mismatch_names_the_misconfiguration() {
    let error = engine_error(pb::EngineError {
        code: pb::ErrorCode::RoleMismatch as i32,
        message: "generation_only request on a context engine".to_string(),
        retryable: false,
    });
    assert!(
        matches!(
            error.error_type(),
            ErrorType::Backend(dynamo_backend_common::BackendError::InvalidArgument)
        ),
        "a deployment mistake is a client error, not an opaque engine failure"
    );
    assert!(
        error.to_string().contains("--disaggregation-mode"),
        "the error must name the flag the operator has to check: {error}"
    );
}

/// A handoff the decode engine cannot resolve is not migratable: the retry
/// would replay the same dead session and fail the same way on the next
/// worker. It still has to say what to look at, which is why these two codes
/// no longer fall into the catch-all.
#[test]
fn an_unresolvable_handoff_is_named_but_not_migratable() {
    for code in [
        pb::ErrorCode::KvSessionNotFound,
        pb::ErrorCode::KvTransferFailed,
    ] {
        let error = engine_error(pb::EngineError {
            code: code as i32,
            message: "session 12345 is gone".to_string(),
            retryable: false,
        });
        assert!(
            !matches!(error.error_type(), ErrorType::WorkerOverloaded),
            "{code:?} must not be migratable"
        );
        assert!(
            error.to_string().contains("cache transceiver"),
            "{code:?} must point at the transceiver: {error}"
        );
    }
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

/// An aggregated worker never asks for a context handoff, so receiving one is
/// protocol drift rather than a silent no-op.
#[test]
fn unexpected_prefill_ready_on_an_aggregated_worker_is_rejected() {
    let mut state = ResponseState::new(&request(), AGG);
    let error = state
        .convert(pb::GenerateResponse {
            request_id: "req".to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady {
                    kv_session: Some(fake_session()),
                },
            )),
            usage: None,
        })
        .expect_err("prefill_ready must be rejected");
    assert!(
        error.to_string().contains("not running in prefill mode"),
        "unexpected error: {error}"
    );
}

/// The OpenEngine servicer suppresses `finished` on a context request only
/// when it already sent a `PrefillReady`; otherwise -- a stop condition during
/// the one-token context phase, or a cancellation before transmission -- it
/// deliberately emits the real terminal, because there is no decode leg to
/// answer the caller. Context tokens are held back on the normal path so the
/// decode leg's replay cannot duplicate them, which makes this terminal the
/// only place they can surface.
#[test]
fn a_prefill_terminal_without_a_handoff_carries_its_tokens() {
    let mut state = ResponseState::new(&request(), DisaggregationMode::Prefill);
    let token = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(0),
            tokens: vec![pb::TokenInfo {
                token_id: 99,
                logprob: Some(-0.5),
                rank: Some(1),
                candidates: vec![pb::LogProb {
                    token_id: 100,
                    logprob: -1.5,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        })),
        usage: None,
    };
    assert!(
        state.convert(token).expect("token converts").is_none(),
        "a context token is held back, not streamed"
    );

    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Stop as i32,
                message: String::new(),
                stop_match: None,
            },
        )),
        usage: None,
    };
    let terminal = state
        .convert(response)
        .expect("a handoff-less prefill terminal is serviceable")
        .expect("it yields a terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        terminal.token_ids,
        [99],
        "the caller gets what the context phase produced"
    );
    // The held token is the whole answer here, so dropping to bare IDs would
    // silently strip logprobs a caller asked for and cannot get anywhere else.
    assert_eq!(terminal.log_probs.as_deref(), Some(&[-0.5][..]));
    let candidates = terminal.top_logprobs.expect("candidates");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0][0].token_id, 100);
}

/// `PrefillReady` is the final response for a context request, so its usage is
/// authoritative -- the frontend forwards `cached_tokens` from it to the decode
/// leg and cannot reconstruct it.
#[test]
fn prefill_ready_keeps_the_engines_reported_usage() {
    let mut state = ResponseState::new(&request(), DisaggregationMode::Prefill);
    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::PrefillReady(
            pb::PrefillReady {
                kv_session: Some(fake_session()),
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 11,
            completion_tokens: 0,
            total_tokens: 11,
            cached_prompt_tokens: Some(7),
            reasoning_tokens: None,
        }),
    };
    let output = state
        .convert(response)
        .expect("prefill_ready converts")
        .expect("prefill_ready yields a terminal");
    let usage = output.completion_usage.expect("usage is set");
    assert_eq!(usage.prompt_tokens, 11);
    assert_eq!(
        usage
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens),
        Some(7),
        "cached prompt tokens must survive to the decode leg"
    );
}

/// A context request that ends at its one-token budget never transmitted its
/// KV, so there is no decode leg to finish the completion. Returning that
/// single token would silently truncate the answer, and passing the `Length`
/// terminal through makes the frontend's prefill router report a missing
/// handoff instead of the real cause.
#[test]
fn a_prefill_terminal_that_ran_out_of_budget_is_rejected() {
    let mut state = ResponseState::new(&request(), DisaggregationMode::Prefill);
    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Length as i32,
                ..Default::default()
            },
        )),
        usage: None,
    };
    let error = state
        .convert(response)
        .expect_err("a budget-exhausted context request has no answer to return");
    assert!(
        error.to_string().contains("without a kv_session handoff"),
        "the error must name the missing handoff: {error}"
    );
}

#[test]
fn held_hidden_stop_token_is_released_unless_it_ended_the_sequence() {
    let mut req = request();
    req.stop_conditions.stop_token_ids_visible = Some(vec![9]);
    let mut state = ResponseState::new(&req, AGG);
    let mut token_ids = |tokens: Vec<pb::TokenInfo>| {
        state
            .convert(token_response(tokens))
            .expect("convert token")
            .map(|delta| delta.token_ids)
    };

    assert_eq!(
        token_ids(vec![logprob_token(5, -0.1), logprob_token(2, -0.2)]),
        Some(vec![5])
    );
    assert_eq!(
        token_ids(vec![logprob_token(6, -0.3), logprob_token(2, -0.4)]),
        Some(vec![2, 6])
    );

    let finished = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Stop as i32,
                message: String::new(),
                stop_match: Some(pb::StopMatch {
                    r#match: Some(pb::stop_match::Match::StopTokenId(2)),
                }),
            },
        )),
        usage: None,
    };
    let terminal = state
        .convert(finished)
        .expect("convert finished")
        .expect("terminal");
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.completion_usage.unwrap().completion_tokens, 4);
}

#[test]
fn visible_stop_token_arrives_with_the_terminal_usage() {
    let mut req = request();
    req.stop_conditions.stop_token_ids_visible = Some(vec![9]);
    let mut state = ResponseState::new(&req, AGG);
    let delta = state
        .convert(token_response(vec![
            logprob_token(5, -0.1),
            logprob_token(9, -0.2),
        ]))
        .expect("convert token")
        .expect("delta");
    assert_eq!(delta.token_ids, [5]);

    let finished = pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Stop as i32,
                message: String::new(),
                stop_match: Some(pb::StopMatch {
                    r#match: Some(pb::stop_match::Match::StopTokenId(9)),
                }),
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            total_tokens: 5,
            cached_prompt_tokens: Some(2),
            reasoning_tokens: None,
        }),
    };
    let terminal = state
        .convert(finished)
        .expect("convert finished")
        .expect("terminal");
    assert_eq!(terminal.token_ids, [9]);
    assert_eq!(terminal.log_probs.as_deref(), Some(&[-0.2][..]));
    assert_eq!(terminal.stop_reason, Some(StopReason::Int(9)));
    let usage = terminal.completion_usage.expect("usage");
    assert_eq!(
        usage
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens),
        Some(2)
    );
}
