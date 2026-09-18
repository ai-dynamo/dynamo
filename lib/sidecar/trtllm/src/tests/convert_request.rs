// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building an OpenEngine `GenerateRequest` from a Dynamo request.

use super::*;

#[test]
fn request_maps_sampling_stop_and_output_fields() {
    let proto = build_generate_request(&request(), "req-1", "served-model", None, AGG)
        .expect("build request");
    assert_eq!(proto.request_id, "req-1");
    assert_eq!(proto.model, "served-model");
    match proto.input.as_ref().expect("input") {
        pb::generate_request::Input::TokenIds(tokens) => assert_eq!(tokens.ids, [11, 22, 33]),
        other => panic!("expected token IDs input, got {other:?}"),
    }

    let stopping = proto.stopping.as_ref().unwrap();
    assert_eq!(stopping.max_tokens, Some(16));
    assert_eq!(stopping.min_tokens, Some(1));
    assert_eq!(stopping.ignore_eos, Some(true));
    // Stop-string retention is never enabled from `include_stop_str_in_output`.
    assert_eq!(stopping.include_stop_in_output, None);
    let conditions: Vec<_> = stopping
        .conditions
        .iter()
        .map(|condition| condition.condition.clone().unwrap())
        .collect();
    assert!(
        conditions.contains(&pb::stop_condition::Condition::StopText("done".to_string())),
        "missing stop text: {conditions:?}"
    );
    assert!(
        conditions.contains(&pb::stop_condition::Condition::StopTokenId(2)),
        "missing stop token id: {conditions:?}"
    );

    let sampling = proto.sampling.as_ref().unwrap();
    assert_eq!(sampling.top_k, Some(4));
    assert_eq!(sampling.top_p, Some(f64::from(0.9_f32)));
    assert_eq!(sampling.min_p, Some(f64::from(0.1_f32)));
    assert_eq!(sampling.temperature, Some(f64::from(0.2_f32)));
    assert_eq!(sampling.seed, Some(123));
    assert_eq!(sampling.repetition_penalty, Some(f64::from(1.1_f32)));
    assert_eq!(sampling.num_sequences, Some(1));

    let response = proto.response.as_ref().unwrap();
    assert_eq!(response.return_output_logprobs, Some(true));
    match response
        .output_candidates
        .as_ref()
        .unwrap()
        .selection
        .as_ref()
        .unwrap()
    {
        pb::candidate_token_selection::Selection::TopN(n) => assert_eq!(*n, 1),
        other => panic!("expected top_n selection, got {other:?}"),
    }

    match proto.guided.as_ref().unwrap().guide.as_ref().unwrap() {
        pb::guided_decoding::Guide::JsonSchema(guide) => assert!(guide.contains("object")),
        other => panic!("expected JSON schema guide, got {other:?}"),
    }
}

#[test]
fn omitted_max_tokens_without_context_length_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let error = build_generate_request(&req, "req", "model", None, AGG)
        .expect_err("must require max_tokens");
    assert!(error.to_string().contains("max_tokens"));
}

#[test]
fn omitted_max_tokens_defaults_to_remaining_context() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // request() carries three prompt tokens ([11, 22, 33]); the default fills the
    // remaining context: max(1, context_length - prompt_len).
    let proto = build_generate_request(&req, "req", "model", limits(100), AGG)
        .expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(97));
}

#[test]
fn omitted_max_tokens_default_is_floored_at_one() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    // Prompt already fills (or exceeds) the context: default must not underflow to 0.
    let proto =
        build_generate_request(&req, "req", "model", limits(2), AGG).expect("build with fallback");
    assert_eq!(proto.stopping.unwrap().max_tokens, Some(1));
}

#[test]
fn top_k_all_tokens_is_left_unset() {
    let mut req = request();
    req.sampling_options.top_k = Some(-1);
    let proto = build_generate_request(&req, "req", "model", None, AGG).expect("build");
    assert_eq!(proto.sampling.unwrap().top_k, None);
}

#[test]
fn unsupported_request_controls_are_rejected() {
    // Controls the OpenEngine contract can neither forward nor faithfully honor:
    // reject rather than fail open.
    assert_rejected(
        |r| r.sampling_options.include_stop_str_in_output = Some(true),
        "include_stop_str_in_output",
    );
    assert_rejected(
        |r| r.stop_conditions.max_thinking_tokens = Some(32),
        "max_thinking_tokens",
    );
    assert_rejected(
        |r| {
            r.routing = Some(dynamo_backend_common::engine::RoutingHints {
                cache_namespace: Some("tenant-a".to_string()),
                ..Default::default()
            })
        },
        "cache namespace",
    );
    assert_rejected(
        |r| {
            r.routing = Some(dynamo_backend_common::engine::RoutingHints {
                priority: Some(5),
                ..Default::default()
            })
        },
        "priority",
    );
    // A negative top_k other than -1/0 (the "all tokens" sentinels) is invalid,
    // not a silent widening to "all tokens".
    assert_rejected(|r| r.sampling_options.top_k = Some(-5), "top_k must be");
    assert_rejected(
        |r| r.sampling_options.seed = Some(-1),
        "seed must be non-negative",
    );
}

#[test]
fn logprobs_zero_keeps_selected_without_alternatives() {
    let mut req = request();
    req.output_options.logprobs = Some(0);
    // The wire request must still ask TRT-LLM for one candidate so the selected
    // token's logprob is computed.
    let proto = build_generate_request(&req, "req", "model", None, AGG).expect("build");
    match proto
        .response
        .unwrap()
        .output_candidates
        .unwrap()
        .selection
        .unwrap()
    {
        pb::candidate_token_selection::Selection::TopN(n) => assert_eq!(n, 1),
        other => panic!("expected top_n selection, got {other:?}"),
    }

    let mut state = ResponseState::new(&req, AGG);
    let delta = state
        .convert(token_response(vec![logprob_token(7, -0.1)]))
        .expect("convert")
        .expect("delta");
    assert_eq!(delta.log_probs.as_deref(), Some(&[-0.1_f64][..]));
    // logprobs=0 surfaces the selected-token logprob but no top alternatives.
    assert!(delta.top_logprobs.is_none());
}

/// The engine's own output cap can be smaller than the context window leaves
/// room for, and a request derived past it is rejected by the engine.
#[test]
fn the_derived_max_tokens_respects_the_engines_output_cap() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    let proto = build_generate_request(
        &req,
        "req",
        "model",
        Some(ModelLimits {
            context_length: Some(4096),
            max_output_tokens: Some(256),
        }),
        AGG,
    )
    .expect("build with an output cap");
    assert_eq!(
        proto.stopping.expect("stopping").max_tokens,
        Some(256),
        "the derived default must not exceed what the engine will generate"
    );
}

/// A `max_tokens` derived from the context window is clamped by the engine's
/// output cap and floored at 1, so it can land under an explicit `min_tokens`.
/// TensorRT-LLM does not cross-validate the pair, so a request carrying
/// `min_tokens > max_tokens` is one the engine resolves however it likes --
/// reject it here, naming both values.
#[test]
fn a_derived_budget_below_min_tokens_is_rejected() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    req.stop_conditions.min_tokens = Some(64);
    // A 40-token window against this request's prompt leaves fewer than 64.
    let prompt_len = req.token_ids.len() as u32;
    let window = prompt_len + 8;

    let error = build_generate_request(&req, "req", "model", limits(window), AGG)
        .expect_err("a minimum above the remaining budget cannot be served");
    let message = error.to_string();
    assert!(
        message.contains("64") && message.contains('8'),
        "the error must name both the minimum and what is left: {message}"
    );
}

/// The same request is served when the window leaves room for the minimum --
/// the guard must reject the contradiction, not every request that has one.
#[test]
fn a_minimum_within_the_derived_budget_is_served() {
    let mut req = request();
    req.stop_conditions.max_tokens = None;
    req.stop_conditions.min_tokens = Some(8);
    let window = req.token_ids.len() as u32 + 64;

    let proto = build_generate_request(&req, "req", "model", limits(window), AGG)
        .expect("a minimum that fits is not a conflict");
    let stopping = proto.stopping.expect("stopping options");
    assert_eq!(stopping.min_tokens, Some(8));
    assert_eq!(stopping.max_tokens, Some(64));
}
