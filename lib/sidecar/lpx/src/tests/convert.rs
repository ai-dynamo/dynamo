// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request and response conversion, and the error mapping.

use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{FinishReason, GuidedDecodingOptions, MultimodalData, StopReason};

use super::*;
use crate::convert::{ResponseState, build_generate_request};
use crate::pb;

fn build(request: &PreprocessedRequest) -> Result<pb::GenerateRequest, DynamoError> {
    build_generate_request(request, "request-1", &discovered(AGG))
}

fn stop_id(id: u32) -> pb::StopCondition {
    pb::StopCondition {
        condition: Some(pb::stop_condition::Condition::StopTokenId(id)),
    }
}

#[test]
fn every_supported_field_is_forwarded() {
    let mut request = request();
    request.routing = Some(RoutingHints {
        lora_name: Some("adapter".to_string()),
        dp_rank: Some(0),
        priority: Some(0),
        // Targets the prefill worker, not this one.
        prefill_dp_rank: Some(3),
        ..Default::default()
    });
    let sent = build(&request).expect("a supported request");

    assert_eq!(sent.request_id, "request-1");
    assert_eq!(sent.model, SERVED);
    assert_eq!(
        sent.input,
        Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: vec![128000, 9906, 1917],
        }))
    );
    assert_eq!(
        sent.sampling,
        Some(pb::SamplingParams {
            temperature: Some(0.5),
            top_p: Some(0.75),
            top_k: Some(4),
            min_p: Some(0.25),
            frequency_penalty: Some(0.375),
            presence_penalty: Some(0.125),
            repetition_penalty: Some(1.5),
            seed: Some(123),
            num_sequences: Some(1),
        })
    );
    assert_eq!(
        sent.stopping,
        Some(pb::StoppingOptions {
            max_tokens: Some(16),
            min_tokens: Some(1),
            // Both hidden lists, deduplicated. The stop string stays with the
            // frontend.
            conditions: vec![stop_id(7), stop_id(128001), stop_id(128009)],
            ignore_eos: Some(false),
            include_stop_in_output: None,
        })
    );
    assert_eq!(sent.lora_name, "adapter", "the engine rejects LoRA itself");
    assert!(sent.response.is_none());
    assert!(sent.kv.is_none());
    assert!(sent.guided.is_none());
    assert!(sent.media.is_empty());
    assert!(sent.extra.is_none());
}

/// The budget sent for a `len`-token prompt and `max_tokens` to an engine
/// with a 4096-token context window. The fill depends only on the window the
/// engine reports, not on which engine or role reports it.
fn sent_max_tokens(len: usize, max_tokens: Option<u32>) -> Option<u32> {
    let mut model = fake::model_info();
    model.max_context_length = Some(4096);
    let engine = DiscoveredEngine::check(fake::server_info(AGG), model, AGG).expect("accepted");
    let mut request = request();
    request.token_ids = std::sync::Arc::new(vec![1; len]);
    request.stop_conditions.max_tokens = max_tokens;
    build_generate_request(&request, "request-1", &engine)
        .expect("accepted")
        .stopping
        .expect("stopping")
        .max_tokens
}

/// The LPU-only engine rejects a request without a budget, so an absent one is
/// filled the way the hybrid engine fills it, which sees no difference. The
/// vectors are that engine's own for a 4096-token window: the rest of the
/// window, and one boundary token once the prompt fills or overflows it.
/// An explicit budget passes through, including zero and oversized values.
#[test]
fn max_tokens_is_filled_only_when_absent() {
    for (prompt, max_tokens, expected) in [
        (96, None, 4000),
        (4096, None, 1),
        (5000, None, 1),
        (96, Some(100_000), 100_000),
        (96, Some(0), 0),
    ] {
        assert_eq!(
            sent_max_tokens(prompt, max_tokens),
            Some(expected),
            "a {prompt}-token prompt"
        );
    }
}

/// -1 and 0 mean "all tokens"; any other value reaches the engine, which
/// rejects a negative one itself.
#[test]
fn top_k_means_all_tokens_at_minus_one_and_zero() {
    for (top_k, expected) in [
        (None, None),
        (Some(-1), None),
        (Some(0), None),
        (Some(-2), Some(-2)),
    ] {
        let mut request = request();
        request.sampling_options.top_k = top_k;
        let sent = build(&request).expect("accepted");
        assert_eq!(sent.sampling.expect("sampling").top_k, expected);
    }
}

type Mutation = fn(&mut PreprocessedRequest);

/// One row per sidecar rejection; each must name what was refused.
#[test]
fn unsupported_requests_are_rejected_before_dispatch() {
    let rows: &[(Mutation, &str)] = &[
        (
            |r| r.token_ids = std::sync::Arc::new(Vec::new()),
            "token_ids must not be empty",
        ),
        (
            |r| r.prompt_embeds = Some("embeds".into()),
            "prompt embeddings",
        ),
        (
            |r| {
                r.multi_modal_data = Some(
                    [(
                        "image_url".to_string(),
                        vec![MultimodalData::RawUrl("data:,".to_string())],
                    )]
                    .into_iter()
                    .collect(),
                )
            },
            "multimodal",
        ),
        (
            |r| r.mm_routing_info = Some(Default::default()),
            "multimodal",
        ),
        (|r| r.encoder_result = Some(json!({})), "multimodal"),
        (
            |r| r.bootstrap_info = Some(bootstrap()),
            "bootstrap handoff",
        ),
        (
            |r| r.sampling_options.best_of = Some(2),
            "best_of must be 1",
        ),
        (
            |r| r.sampling_options.use_beam_search = Some(true),
            "beam search",
        ),
        (
            |r| {
                r.sampling_options.logit_bias = Some(std::collections::HashMap::from([(18, -10.0)]))
            },
            "logit_bias",
        ),
        (
            |r| r.output_options.prompt_logprobs = Some(1),
            "prompt logprobs",
        ),
        (|r| r.output_options.logprobs = Some(0), "logprobs"),
        (
            |r| r.extra_args = Some(json!({"sampling_options": {"allowed_token_ids": []}})),
            "allowed_token_ids",
        ),
        (
            |r| r.extra_args = Some(json!({"sampling_options": {"bad_words_token_ids": [[2, 3]]}})),
            "bad_words_token_ids",
        ),
        (
            |r| r.extra_args = Some(json!({"sampling_options": {"logprob_token_ids": []}})),
            "logprob_token_ids",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    json: Some(json!({"type": "object"})),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    regex: Some("a+".to_string()),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    choice: Some(vec!["yes".to_string(), "no".to_string()]),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    grammar: Some("root ::= \"yes\"".to_string()),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    structural_tag: Some(json!({"type": "structural_tag"})),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    backend: Some("xgrammar".to_string()),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| {
                r.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
                    whitespace_pattern: Some(" *".to_string()),
                    ..Default::default()
                })
            },
            "guided decoding",
        ),
        (
            |r| r.stop_conditions.stop_token_ids_visible = Some(vec![5]),
            "visible stop token IDs",
        ),
        (
            |r| r.output_options.no_stop_trim = Some(true),
            "no_stop_trim",
        ),
        (
            |r| {
                r.output_options.no_stop_trim = Some(true);
                r.output_options.skip_special_tokens = Some(false);
                r.stop_conditions.stop_token_ids = None;
            },
            "no_stop_trim",
        ),
        (
            |r| r.sampling_options.include_stop_str_in_output = Some(true),
            "include_stop_str_in_output",
        ),
        (
            |r| r.stop_conditions.max_thinking_tokens = Some(8),
            "max_thinking_tokens",
        ),
        (
            |r| {
                r.routing = Some(RoutingHints {
                    cache_namespace: Some("tenant".to_string()),
                    ..Default::default()
                })
            },
            "cache namespace",
        ),
        (
            |r| {
                r.routing = Some(RoutingHints {
                    priority: Some(1),
                    ..Default::default()
                })
            },
            "priority",
        ),
        (
            |r| {
                r.routing = Some(RoutingHints {
                    dp_rank: Some(1),
                    ..Default::default()
                })
            },
            "data-parallel rank",
        ),
        (
            |r| r.sampling_options.seed = Some(-1),
            "seed must be non-negative",
        ),
    ];
    for &(mutate, expect) in rows {
        let mut request = request();
        mutate(&mut request);
        let error = build(&request).expect_err(expect);
        assert_backend(&error, BackendError::InvalidArgument);
        assert!(
            error.to_string().contains(expect),
            "error {error} should mention {expect:?}"
        );
    }
}

fn bootstrap() -> dynamo_backend_common::BootstrapInfo {
    serde_json::from_value(json!({
        "bootstrap_host": "10.0.0.2",
        "bootstrap_port": 8998,
        "bootstrap_room": 7
    }))
    .expect("bootstrap info")
}

#[test]
fn neutral_options_are_accepted() {
    let rows: &[Mutation] = &[
        |r| r.sampling_options.logit_bias = Some(Default::default()),
        |r| r.sampling_options.guided_decoding = Some(GuidedDecodingOptions::default()),
        |r| r.extra_args = Some(json!({"sampling_options": {"detokenize": false}})),
        |r| r.extra_args = Some(json!({"sampling_options": {"detokenize": true}})),
        |r| r.extra_args = Some(json!({"sampling_options": {"bad_words_token_ids": []}})),
        |r| {
            r.output_options.no_stop_trim = Some(true);
            r.stop_conditions.stop_token_ids = None;
        },
        |r| {
            r.output_options.no_stop_trim = Some(true);
            r.output_options.skip_special_tokens = Some(true);
            r.stop_conditions.stop_token_ids = Some(vec![]);
        },
        |r| {
            r.output_options.no_stop_trim = Some(false);
            r.output_options.skip_special_tokens = Some(false);
        },
    ];
    for mutate in rows {
        let mut request = request();
        mutate(&mut request);
        build(&request).expect("no unsupported constraint or output option");
    }
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

fn token(ids: &[u32], output_index: Option<u32>) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index,
            tokens: ids
                .iter()
                .map(|id| pb::TokenInfo {
                    token_id: *id,
                    ..Default::default()
                })
                .collect(),
            text: String::new(),
        })),
        usage: None,
    }
}

fn finished(
    reason: pb::FinishReason,
    stop_match: Option<pb::stop_match::Match>,
    usage: Option<pb::Usage>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: reason as i32,
                message: String::new(),
                stop_match: stop_match.map(|matched| pb::StopMatch {
                    r#match: Some(matched),
                }),
            },
        )),
        usage,
    }
}

fn engine_usage(prompt: u32, completion: u32, cached: Option<u32>) -> pb::Usage {
    pb::Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cached_prompt_tokens: cached,
        reasoning_tokens: None,
    }
}

#[test]
fn tokens_stream_and_the_terminal_carries_authoritative_usage() {
    let mut state = ResponseState::new(&request(), &discovered(AGG));
    let output = state
        .convert(token(&[11, 12], Some(0)))
        .expect("token")
        .expect("a delta");
    assert_eq!(output.token_ids, [11, 12]);
    assert_eq!(output.index, Some(0));
    assert!(output.finish_reason.is_none());
    assert!(
        state.convert(token(&[], Some(0))).expect("empty").is_none(),
        "an empty delta is skipped"
    );

    let terminal = state
        .convert(finished(
            pb::FinishReason::Stop,
            None,
            Some(engine_usage(5, 3, Some(2))),
        ))
        .expect("finished")
        .expect("terminal");
    assert!(terminal.token_ids.is_empty());
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    let usage = terminal.completion_usage.expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (5, 3));
    assert_eq!(usage.total_tokens, 8);
    // Aggregated: the engine's own cache count stands.
    assert_eq!(
        usage
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens),
        Some(2)
    );
}

#[test]
fn finish_reasons_map_and_unspecified_fails_closed() {
    use pb::stop_match::Match;

    let terminal = ResponseState::new(&request(), &discovered(AGG))
        .convert(finished(pb::FinishReason::Cancelled, None, None))
        .unwrap()
        .unwrap();
    assert_eq!(terminal.finish_reason, Some(FinishReason::Cancelled));
    // Without reported usage, the streamed counts stand.
    let usage = terminal.completion_usage.expect("usage");
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (3, 0));
    for (matched, expected) in [
        (Match::EosTokenId(128009), StopReason::Int(128009)),
        (
            Match::StopText("done".into()),
            StopReason::String("done".into()),
        ),
    ] {
        let terminal = ResponseState::new(&request(), &discovered(AGG))
            .convert(finished(pb::FinishReason::Stop, Some(matched), None))
            .expect("finished")
            .expect("terminal");
        assert_eq!(terminal.stop_reason, Some(expected));
    }

    let error = ResponseState::new(&request(), &discovered(AGG))
        .convert(finished(pb::FinishReason::Unspecified, None, None))
        .expect_err("unspecified");
    assert!(error.to_string().contains("unspecified finish reason"));
}

/// A decode engine counts the transferred blocks as cache hits; on a decode
/// leg only the prefill worker's count means anything, clamped to the prompt.
#[test]
fn a_decode_leg_reports_the_prefill_cache_hits() {
    let mut request = decode_request();
    let state_with = |cached: Option<u32>, request: &mut PreprocessedRequest| {
        request.prefill_result = Some(prefill_result(wrapped(vllm_params()), cached));
        ResponseState::new(request, &discovered(DECODE))
    };
    let cached = |mut state: ResponseState| {
        state
            .convert(finished(
                pb::FinishReason::Length,
                None,
                Some(engine_usage(3, 2, Some(3))),
            ))
            .unwrap()
            .unwrap()
            .completion_usage
            .unwrap()
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens)
    };
    assert_eq!(cached(state_with(Some(1), &mut request)), Some(1));
    assert_eq!(
        cached(state_with(Some(99), &mut request)),
        Some(3),
        "clamped"
    );
    assert_eq!(
        cached(state_with(None, &mut request)),
        None,
        "never the decode engine's count"
    );
}

#[test]
fn protocol_drift_is_an_error() {
    let drift = |response: pb::GenerateResponse, expect: &str| {
        let error = ResponseState::new(&request(), &discovered(AGG))
            .convert(response)
            .expect_err(expect);
        assert_backend(&error, BackendError::Unknown);
        assert!(error.to_string().contains("invalid lpx-test gRPC response"));
        assert!(error.to_string().contains(expect), "{error}");
    };
    drift(
        pb::GenerateResponse {
            request_id: "r".into(),
            event: None,
            usage: None,
        },
        "no event",
    );
    drift(token(&[1], None), "no output index");
    drift(token(&[1], Some(1)), "output index 1");
    drift(
        pb::GenerateResponse {
            request_id: "r".into(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady { kv_session: None },
            )),
            usage: None,
        },
        "prefill_ready",
    );
    drift(
        pb::GenerateResponse {
            request_id: "r".into(),
            event: Some(pb::generate_response::Event::Prompt(pb::PromptOutput {
                tokens: Vec::new(),
            })),
            usage: None,
        },
        "prompt event",
    );
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failure the engine reports in-band on a request it accepted, mapped by
/// code and retryability: one row per mapping.
#[test]
fn in_band_errors_map_by_code_and_retryability() {
    let error = |code: pb::ErrorCode, retryable: bool, message: &str| {
        ResponseState::new(&request(), &discovered(AGG))
            .convert(pb::GenerateResponse {
                request_id: "r".to_string(),
                event: Some(pb::generate_response::Event::Error(pb::EngineError {
                    code: code as i32,
                    message: message.to_string(),
                    retryable,
                })),
                usage: None,
            })
            .expect_err("an in-band error fails the request")
    };
    for (code, retryable, expected) in [
        // Retryable is migratable whatever the code.
        (
            pb::ErrorCode::InvalidArgument,
            true,
            BackendError::EngineShutdown,
        ),
        (
            pb::ErrorCode::UnsupportedFeature,
            false,
            BackendError::InvalidArgument,
        ),
        // How a decode engine reports a failed KV pull: its message passes through.
        (pb::ErrorCode::Internal, false, BackendError::Unknown),
    ] {
        let error = error(code, retryable, "boom");
        assert_backend(&error, expected);
        assert!(
            error.to_string().contains("lpx-test engine error: boom"),
            "{error}"
        );
    }
}
