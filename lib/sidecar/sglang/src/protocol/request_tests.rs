// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    BackendError, BootstrapInfo, ErrorType, GuidedDecodingOptions, KvHintAction, OutputOptions,
    PrefillResult, SamplingOptions,
};
use serde_json::json;

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("Qwen/Qwen3-0.6B".to_string())
        .token_ids(vec![1, 2, 3])
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .stop_conditions(StopConditions {
            max_tokens: Some(8),
            ..Default::default()
        })
        .build()
        .unwrap()
}

/// A single-rank worker's target with the bootstrap address `host:port`.
fn target<'a>(host: Option<&'a str>, port: Option<u16>) -> Target<'a> {
    Target {
        bootstrap_host: host,
        bootstrap_port: port,
        prefill_rank: None,
        global_dp_size: 1,
    }
}

fn build(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
) -> Result<pb::GenerateRequest, DynamoError> {
    build_generate_request(request, "rid", mode, target(Some("prefill"), Some(5000)))
        .map(|(generate, _)| generate)
}

fn assert_invalid(error: DynamoError, message: &str) {
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(error.to_string().contains(message), "{error}");
}

fn sampling(request: &pb::GenerateRequest) -> &pb::SamplingParams {
    match request.sampling_params.as_ref().unwrap().value.as_ref() {
        Some(pb::sampling_params_or_list::Value::One(params)) => params,
        other => panic!("expected one sampling-params object: {other:?}"),
    }
}

fn input_ids(request: &pb::GenerateRequest) -> &[i64] {
    match request.input_ids.as_ref().unwrap().value.as_ref() {
        Some(pb::token_ids_or_list::Value::One(ids)) => &ids.ids,
        other => panic!("expected one token-id list: {other:?}"),
    }
}

fn rid(request: &pb::GenerateRequest) -> &str {
    match request.rid.as_ref().unwrap().value.as_ref() {
        Some(pb::string_or_list::Value::One(rid)) => rid,
        other => panic!("expected one rid: {other:?}"),
    }
}

/// `(return_logprob, top_logprobs_num, logprob_start_len)`.
fn logprob_controls(request: &pb::GenerateRequest) -> (bool, i64, i64) {
    let int = |value: Option<&pb::Int64OrList>| match value.unwrap().value.as_ref() {
        Some(pb::int64_or_list::Value::One(value)) => *value,
        other => panic!("expected one integer: {other:?}"),
    };
    let enabled = match request.return_logprob.as_ref().unwrap().value.as_ref() {
        Some(pb::bool_or_list::Value::One(value)) => *value,
        other => panic!("expected one boolean: {other:?}"),
    };
    (
        enabled,
        int(request.top_logprobs_num.as_ref()),
        int(request.logprob_start_len.as_ref()),
    )
}

fn lora_path(request: &pb::GenerateRequest) -> Option<&str> {
    match request.lora_path.as_ref()?.value.as_ref() {
        Some(pb::optional_string_or_list::Value::One(path)) => path.value.as_deref(),
        other => panic!("expected one adapter: {other:?}"),
    }
}

/// `(bootstrap_host, bootstrap_port, bootstrap_room)` of a single request.
fn rendezvous(request: &pb::GenerateRequest) -> Option<(String, i64, i64)> {
    let one = |value: Option<&pb::OptionalInt64OrList>| match value?.value.as_ref()? {
        pb::optional_int64_or_list::Value::One(value) => value.value,
        pb::optional_int64_or_list::Value::Many(_) => panic!("expected one integer"),
    };
    let host = match request.bootstrap_host.as_ref()?.value.as_ref()? {
        pb::optional_string_or_list::Value::One(host) => host.value.clone()?,
        pb::optional_string_or_list::Value::Many(_) => panic!("expected one host"),
    };
    Some((
        host,
        one(request.bootstrap_port.as_ref())?,
        one(request.bootstrap_room.as_ref())?,
    ))
}

#[test]
fn request_maps_native_fields_and_full_width_room() {
    let mut request = request();
    request.bootstrap_info = Some(BootstrapInfo {
        bootstrap_host: "prefill".to_string(),
        bootstrap_port: 5000,
        bootstrap_room: i64::MAX as u64,
        handoff_id: None,
    });
    let (mapped, handoff) = build_generate_request(
        &request,
        "rid-1",
        DisaggregationMode::Decode,
        target(None, None),
    )
    .unwrap();
    assert_eq!(input_ids(&mapped), [1, 2, 3]);
    assert_eq!(rid(&mapped), "rid-1");
    assert_eq!(mapped.stream, Some(true));
    assert_eq!(sampling(&mapped).max_new_tokens, Some(8));
    assert_eq!(
        rendezvous(&mapped),
        Some(("prefill".to_string(), 5000, i64::MAX))
    );
    assert_eq!(
        handoff,
        Some(
            json!({"bootstrap_host": "prefill", "bootstrap_port": 5000, "bootstrap_room": i64::MAX})
        )
    );
}

#[test]
fn prefill_clamps_generation_and_disables_decode_only_options() {
    let mut request = request();
    request.stop_conditions.min_tokens = Some(4);
    request.output_options = OutputOptions {
        logprobs: Some(2),
        prompt_logprobs: Some(3),
        ..Default::default()
    };
    let mapped = build(&request, DisaggregationMode::Prefill).unwrap();
    assert_eq!(sampling(&mapped).max_new_tokens, Some(1));
    assert_eq!(sampling(&mapped).min_new_tokens, None);
    assert_eq!(logprob_controls(&mapped), (false, 0, -1));
    let (host, port, room) = rendezvous(&mapped).unwrap();
    assert_eq!((host.as_str(), port), ("prefill", 5000));
    assert!(room >= 0);
}

#[test]
fn prefill_handoff_round_trips_to_decode_request() {
    let (prefill, handoff) = build_generate_request(
        &request(),
        "rid-prefill",
        DisaggregationMode::Prefill,
        target(Some("prefill.internal"), Some(5001)),
    )
    .unwrap();
    let handoff = handoff.unwrap();
    assert_eq!(handoff["bootstrap_host"], "prefill.internal");
    assert_eq!(handoff["bootstrap_port"], 5001);

    let mut decode_request = request();
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let decode = build(&decode_request, DisaggregationMode::Decode).unwrap();
    assert_eq!(rendezvous(&decode), rendezvous(&prefill));
}

#[test]
fn decode_requires_rendezvous_params() {
    let error = build_generate_request(
        &request(),
        "rid-3",
        DisaggregationMode::Decode,
        target(None, None),
    )
    .unwrap_err();
    assert_invalid(error, "neither bootstrap_info nor prefill_result");
}

#[test]
fn request_refusal_is_public() {
    let mut refused = request();
    refused.mm_processor_kwargs = Some(json!({}));
    let error = build(&refused, DisaggregationMode::Aggregated).unwrap_err();
    assert_eq!(
        error.public_message(),
        Some("multimodal payloads are not supported by SGLang's native Generate RPC")
    );

    let mut embeds = request();
    embeds.token_ids = Vec::new().into();
    embeds.prompt_embeds = Some("embeds".to_string());
    let error = build(&embeds, DisaggregationMode::Aggregated).unwrap_err();
    assert_eq!(
        error.public_message(),
        Some("prompt_embeds are not supported by SGLang's native gRPC proto")
    );
}

#[test]
#[allow(deprecated)]
fn sampling_and_stopping_fields_preserve_native_values() {
    let mut request = request();
    request.sampling_options = SamplingOptions {
        temperature: Some(0.7),
        top_p: Some(0.9),
        top_k: Some(17),
        min_p: Some(0.05),
        frequency_penalty: Some(0.2),
        presence_penalty: Some(-0.3),
        repetition_penalty: Some(1.1),
        n: Some(1),
        seed: Some(42),
        ..Default::default()
    };
    request.stop_conditions = StopConditions {
        max_tokens: Some(11),
        min_tokens: Some(2),
        stop: Some(vec!["end".to_string(), "done".to_string()]),
        stop_token_ids: Some(vec![19, 19, 20]),
        stop_token_ids_hidden: Some(vec![20, 21]),
        ignore_eos: Some(true),
        ..Default::default()
    };
    let mapped = build(&request, DisaggregationMode::Aggregated).unwrap();
    // The native fields are doubles; the canonical request carries f32.
    assert_eq!(
        sampling(&mapped),
        &pb::SamplingParams {
            temperature: Some(f64::from(0.7_f32)),
            top_p: Some(f64::from(0.9_f32)),
            top_k: Some(17),
            min_p: Some(f64::from(0.05_f32)),
            frequency_penalty: Some(f64::from(0.2_f32)),
            presence_penalty: Some(f64::from(-0.3_f32)),
            repetition_penalty: Some(f64::from(1.1_f32)),
            max_new_tokens: Some(11),
            min_new_tokens: Some(2),
            stop: Some(pb::StringOrList {
                value: Some(pb::string_or_list::Value::Many(pb::StringList {
                    items: vec!["end".to_string(), "done".to_string()],
                })),
            }),
            stop_token_ids: Some(pb::Int64List {
                items: vec![19, 20, 21],
            }),
            ignore_eos: Some(true),
            n: Some(1),
            sampling_seed: Some(42),
            ..Default::default()
        }
    );
    assert!(mapped.session_params.is_none());
    assert!(mapped.text.is_none());
}

#[test]
fn visible_stop_tokens_stop_in_engine_and_return_only_ids() {
    let mut request = request();
    request.stop_conditions.stop_token_ids = Some(vec![19]);
    request.stop_conditions.stop_token_ids_hidden = Some(vec![200002, 199999]);
    request.stop_conditions.stop_token_ids_visible = Some(vec![200012, 19]);
    let mapped = build(&request, DisaggregationMode::Aggregated).unwrap();
    // SGLang keeps a matched stop token in output_ids; the sidecar never asks
    // for text, which SGLang trims.
    assert_eq!(
        sampling(&mapped).stop_token_ids.as_ref().unwrap().items,
        [19, 200002, 199999, 200012]
    );
    assert!(mapped.text.is_none());
}

#[test]
fn optional_controls_distinguish_absence_from_explicit_zero_or_false() {
    for explicit in [false, true] {
        let mut request = request();
        request.stop_conditions = StopConditions {
            max_tokens: explicit.then_some(0),
            min_tokens: explicit.then_some(0),
            ignore_eos: explicit.then_some(false),
            stop: explicit.then(Vec::new),
            stop_token_ids: explicit.then(Vec::new),
            ..Default::default()
        };
        request.sampling_options = SamplingOptions {
            temperature: explicit.then_some(0.0),
            top_p: explicit.then_some(0.0),
            top_k: explicit.then_some(-1),
            min_p: explicit.then_some(0.0),
            frequency_penalty: explicit.then_some(0.0),
            presence_penalty: explicit.then_some(0.0),
            repetition_penalty: explicit.then_some(0.0),
            n: explicit.then_some(1),
            best_of: explicit.then_some(1),
            use_beam_search: explicit.then_some(false),
            length_penalty: explicit.then_some(1.0),
            include_stop_str_in_output: explicit.then_some(false),
            ..Default::default()
        };
        let mapped = build(&request, DisaggregationMode::Aggregated).unwrap();
        assert_eq!(
            sampling(&mapped),
            &pb::SamplingParams {
                temperature: explicit.then_some(0.0),
                top_p: explicit.then_some(0.0),
                top_k: explicit.then_some(-1),
                min_p: explicit.then_some(0.0),
                frequency_penalty: explicit.then_some(0.0),
                presence_penalty: explicit.then_some(0.0),
                repetition_penalty: explicit.then_some(0.0),
                max_new_tokens: explicit.then_some(0),
                min_new_tokens: explicit.then_some(0),
                ignore_eos: explicit.then_some(false),
                n: explicit.then_some(1),
                ..Default::default()
            }
        );
    }
}

#[test]
fn logprob_requests_preserve_selected_only_and_prompt_opt_in() {
    for (output, prompt, enabled, count, start) in [
        (None, None, false, 0, -1),
        (Some(0), None, true, 0, -1),
        (None, Some(0), true, 0, 0),
        (Some(2), Some(3), true, 3, 0),
        (Some(4), Some(1), true, 4, 0),
    ] {
        let mut request = request();
        request.output_options.logprobs = output;
        request.output_options.prompt_logprobs = prompt;
        let mapped = build(&request, DisaggregationMode::Aggregated).unwrap();
        assert_eq!(logprob_controls(&mapped), (enabled, count, start));
        // Logprob entries carry token ids, not decoded text.
        assert!(mapped.return_text_in_logprobs.is_none());
    }
}

#[test]
fn unsupported_sampling_controls_are_rejected() {
    for (options, label) in [
        (
            SamplingOptions {
                n: Some(0),
                ..Default::default()
            },
            "n must be 1",
        ),
        (
            SamplingOptions {
                n: Some(2),
                ..Default::default()
            },
            "n must be 1",
        ),
        (
            SamplingOptions {
                best_of: Some(2),
                ..Default::default()
            },
            "best_of",
        ),
        (
            SamplingOptions {
                use_beam_search: Some(true),
                ..Default::default()
            },
            "beam search",
        ),
        (
            SamplingOptions {
                length_penalty: Some(0.5),
                ..Default::default()
            },
            "length_penalty",
        ),
        (
            SamplingOptions {
                include_stop_str_in_output: Some(true),
                ..Default::default()
            },
            "include_stop_str_in_output",
        ),
    ] {
        let mut request = request();
        request.sampling_options = options;
        assert_invalid(
            build(&request, DisaggregationMode::Aggregated).unwrap_err(),
            label,
        );
    }
}

#[test]
fn unsupported_payload_stopping_and_priority_controls_are_rejected() {
    for field in [
        "token_ids",
        "prompt_embeds",
        "multimodal",
        "mm_processor_kwargs",
    ] {
        let mut request = request();
        let message = match field {
            "token_ids" => {
                request.token_ids = Vec::new().into();
                "token_ids"
            }
            "prompt_embeds" => {
                request.prompt_embeds = Some("encoded".to_string());
                "prompt_embeds"
            }
            "multimodal" => {
                request.multi_modal_data = Some(Default::default());
                "multimodal"
            }
            "mm_processor_kwargs" => {
                request.mm_processor_kwargs = Some(json!({}));
                "multimodal"
            }
            _ => unreachable!(),
        };
        assert_invalid(
            build(&request, DisaggregationMode::Aggregated).unwrap_err(),
            message,
        );
    }
}

#[test]
fn guided_decoding_lowers_to_one_flat_constraint() {
    for (guided, expected) in [
        (
            GuidedDecodingOptions {
                json: Some(json!({"type": "string"})),
                ..Default::default()
            },
            pb::SamplingParams {
                json_schema: Some(r#"{"type":"string"}"#.to_string()),
                ..Default::default()
            },
        ),
        (
            GuidedDecodingOptions {
                json: Some(json!(r#"{"type":"string"}"#)),
                ..Default::default()
            },
            pb::SamplingParams {
                json_schema: Some(r#"{"type":"string"}"#.to_string()),
                ..Default::default()
            },
        ),
        (
            GuidedDecodingOptions {
                regex: Some("[a-z]+".to_string()),
                ..Default::default()
            },
            pb::SamplingParams {
                regex: Some("[a-z]+".to_string()),
                ..Default::default()
            },
        ),
        (
            GuidedDecodingOptions {
                choice: Some(vec!["a".to_string(), "b".to_string()]),
                ..Default::default()
            },
            pb::SamplingParams {
                choice: vec!["a".to_string(), "b".to_string()],
                ..Default::default()
            },
        ),
        (
            GuidedDecodingOptions {
                grammar: Some("root ::= 'a'".to_string()),
                ..Default::default()
            },
            pb::SamplingParams {
                ebnf: Some("root ::= 'a'".to_string()),
                ..Default::default()
            },
        ),
        (
            GuidedDecodingOptions {
                structural_tag: Some(json!({"type": "object"})),
                ..Default::default()
            },
            pb::SamplingParams {
                structural_tag: Some(r#"{"type":"object"}"#.to_string()),
                ..Default::default()
            },
        ),
        // An empty choice list is no constraint.
        (
            GuidedDecodingOptions {
                choice: Some(Vec::new()),
                ..Default::default()
            },
            pb::SamplingParams::default(),
        ),
    ] {
        let mut request = request();
        request.stop_conditions.max_tokens = None;
        request.sampling_options.guided_decoding = Some(guided);
        let mapped = build(&request, DisaggregationMode::Aggregated).unwrap();
        assert_eq!(sampling(&mapped), &expected);
    }

    let mut both = request();
    both.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
        json: Some(json!({"type": "string"})),
        regex: Some("[a-z]+".to_string()),
        ..Default::default()
    });
    assert_invalid(
        build(&both, DisaggregationMode::Aggregated).unwrap_err(),
        "only one guided-decoding constraint",
    );
    for guided in [
        GuidedDecodingOptions {
            backend: Some("xgrammar".to_string()),
            ..Default::default()
        },
        GuidedDecodingOptions {
            whitespace_pattern: Some(" *".to_string()),
            ..Default::default()
        },
    ] {
        let mut request = request();
        request.sampling_options.guided_decoding = Some(guided);
        assert_invalid(
            build(&request, DisaggregationMode::Aggregated).unwrap_err(),
            "backend and whitespace modifiers",
        );
    }
}

#[test]
fn empty_guided_constraints_are_rejected_publicly() {
    for (guided, message) in [
        (
            GuidedDecodingOptions {
                json: Some(json!("")),
                ..Default::default()
            },
            "constraint must not be empty",
        ),
        (
            GuidedDecodingOptions {
                regex: Some(String::new()),
                ..Default::default()
            },
            "constraint must not be empty",
        ),
        (
            GuidedDecodingOptions {
                choice: Some(vec!["a".to_string(), String::new()]),
                ..Default::default()
            },
            "choice must contain only non-empty values",
        ),
    ] {
        let mut request = request();
        request.sampling_options.guided_decoding = Some(guided);
        let error = build(&request, DisaggregationMode::Aggregated).unwrap_err();
        assert!(
            error
                .public_message()
                .is_some_and(|public| public.contains(message)),
            "{error}"
        );
        assert_invalid(error, message);
    }
}

#[test]
fn sglang_0521_request_controls_are_forwarded() {
    let mut controlled = request();
    controlled.stop_conditions.max_thinking_tokens = Some(17);
    controlled.require_reasoning = true;
    controlled.routing = Some(RoutingHints {
        priority: Some(-3),
        ..Default::default()
    });
    controlled.kv_hint = Some(KvHint::new(
        "message-1",
        vec![KvHintAction::new(
            "action-1",
            "kv.fetch",
            "1.0",
            std::collections::BTreeMap::from([("source".to_string(), json!("worker-7"))]),
        )],
    ));

    let mapped = build(&controlled, DisaggregationMode::Aggregated).unwrap();
    assert_eq!(mapped.priority, Some(-3));
    assert_eq!(mapped.require_reasoning, Some(true));
    assert_eq!(mapped.max_thinking_tokens, Some(17));
    let Some(pb::kv_hints_envelope_or_list::Value::One(hints)) = mapped.kv_hints.unwrap().value
    else {
        panic!("expected one KV-hint envelope");
    };
    assert_eq!(hints.protocol_version, "0.1");
    assert_eq!(hints.message_id, "message-1");
    assert_eq!(hints.actions.len(), 1);
    let action = &hints.actions[0];
    assert_eq!(action.action_id, "action-1");
    assert_eq!(action.action_type, "kv.fetch");
    assert_eq!(action.action_version, "1.0");
    assert_eq!(
        serde_json::from_str::<Value>(action.payload.as_deref().unwrap()).unwrap(),
        json!({"source": "worker-7"})
    );

    // The Rust server rejects any request that carries max_thinking_tokens
    // unless it runs with --enable-strict-thinking, so an unset limit stays
    // off the wire.
    let mapped = build(&request(), DisaggregationMode::Aggregated).unwrap();
    assert!(mapped.max_thinking_tokens.is_none());
}

#[test]
fn adapter_and_cache_identity_are_forwarded_without_routed_rank() {
    let mut request = request();
    request.mdc_sum = Some("model-cache-key".to_string());
    request.routing = Some(RoutingHints {
        lora_name: Some("adapter-a".to_string()),
        dp_rank: Some(0),
        ..Default::default()
    });
    let mapped = build(&request, DisaggregationMode::Aggregated).unwrap();
    assert_eq!(lora_path(&mapped), Some("adapter-a"));
    assert_eq!(mapped.routing_key.as_deref(), Some("model-cache-key"));
    // The rank chose the listener, so it is not forwarded.
    assert!(mapped.routed_dp_rank.is_none());
}

/// A request routed to a rank this worker does not serve fails.
#[test]
fn nonzero_router_rank_is_invalid_argument() {
    let routed = |dp_rank: Option<u32>, prefill_dp_rank: Option<u32>| {
        let mut request = request();
        request.routing = Some(RoutingHints {
            dp_rank,
            prefill_dp_rank,
            ..Default::default()
        });
        request
    };
    for (mode, dp_rank, prefill_dp_rank) in [
        (DisaggregationMode::Aggregated, Some(1), None),
        (DisaggregationMode::Prefill, None, Some(1)),
        (DisaggregationMode::Decode, Some(1), None),
    ] {
        let error = validate_dp_rank(&routed(dp_rank, prefill_dp_rank), mode, 0..1).unwrap_err();
        assert_invalid(error, "DP rank 1 was requested");
    }
    for (mode, dp_rank, prefill_dp_rank) in [
        (DisaggregationMode::Aggregated, Some(0), None),
        // Prefill names its own rank first.
        (DisaggregationMode::Prefill, Some(1), Some(0)),
        // Decode names the request's rank, not prefill's.
        (DisaggregationMode::Decode, Some(0), Some(1)),
    ] {
        validate_dp_rank(&routed(dp_rank, prefill_dp_rank), mode, 0..1).unwrap();
    }
}

#[test]
fn room_above_signed_int64_is_rejected() {
    let mut request = request();
    request.bootstrap_info = Some(BootstrapInfo {
        bootstrap_host: "prefill".to_string(),
        bootstrap_port: 5000,
        bootstrap_room: i64::MAX as u64 + 1,
        handoff_id: None,
    });
    assert_invalid(
        build(&request, DisaggregationMode::Decode).unwrap_err(),
        "bootstrap_room must fit",
    );
}

#[test]
fn bootstrap_info_precedes_prefill_result_and_aggregated_ignores_both() {
    let mut request = request();
    request.bootstrap_info = Some(BootstrapInfo {
        bootstrap_host: "router-prefill".to_string(),
        bootstrap_port: 5000,
        bootstrap_room: 23,
        handoff_id: None,
    });
    request.prefill_result = Some(PrefillResult {
        disaggregated_params: json!({"bootstrap_host": "other-prefill", "bootstrap_port": 5001, "bootstrap_room": 24}),
        prompt_tokens_details: None,
    });
    for mode in [DisaggregationMode::Decode, DisaggregationMode::Prefill] {
        let params =
            resolve_disaggregated_params(&request, mode, target(Some("discovery"), Some(5002)))
                .unwrap()
                .unwrap();
        assert_eq!(
            params,
            Rendezvous {
                bootstrap_host: "router-prefill".to_string(),
                bootstrap_port: 5000,
                bootstrap_room: 23,
                disagg_prefill_dp_rank: None,
            }
        );
    }
    assert!(
        resolve_disaggregated_params(&request, DisaggregationMode::Aggregated, target(None, None))
            .unwrap()
            .is_none()
    );
}

#[test]
fn prefill_requires_discovery_address_and_generates_signed_room() {
    for (host, port, message) in [
        (None, Some(5000), "bootstrap host"),
        (Some("prefill"), None, "bootstrap port"),
        (Some("  "), Some(5000), "bootstrap_host"),
    ] {
        assert_invalid(
            resolve_disaggregated_params(
                &request(),
                DisaggregationMode::Prefill,
                target(host, port),
            )
            .unwrap_err(),
            message,
        );
    }
    let params = resolve_disaggregated_params(
        &request(),
        DisaggregationMode::Prefill,
        target(Some("prefill"), Some(5000)),
    )
    .unwrap()
    .unwrap();
    assert_eq!(params.bootstrap_host, "prefill");
    assert_eq!(params.bootstrap_port, 5000);
    assert!(params.bootstrap_room >= 0);
}

#[test]
fn decode_handoff_rejects_missing_wrong_type_and_out_of_range_fields() {
    for (field, value) in [
        ("bootstrap_host", json!(null)),
        ("bootstrap_host", json!(" ")),
        ("bootstrap_port", json!("5000")),
        ("bootstrap_port", json!(-1)),
        ("bootstrap_port", json!(i64::from(i32::MAX) + 1)),
        ("bootstrap_room", json!(null)),
        ("bootstrap_room", json!(-1)),
        ("bootstrap_room", json!(i64::MAX as u64 + 1)),
    ] {
        let mut params =
            json!({"bootstrap_host": "prefill", "bootstrap_port": 5000, "bootstrap_room": 42});
        params[field] = value;
        let mut request = request();
        request.prefill_result = Some(PrefillResult {
            disaggregated_params: params,
            prompt_tokens_details: None,
        });
        assert_invalid(
            resolve_disaggregated_params(&request, DisaggregationMode::Decode, target(None, None))
                .unwrap_err(),
            field,
        );
    }
}

/// The target of a prefill dispatched to rank `rank` of a `size`-rank
/// deployment, on that listener's bootstrap port.
fn ranked(rank: Option<u32>, port: u16, size: u32) -> Target<'static> {
    Target {
        bootstrap_host: Some("prefill"),
        bootstrap_port: Some(port),
        prefill_rank: rank,
        global_dp_size: size,
    }
}

fn routed_bootstrap(prefill_dp_rank: Option<u32>) -> PreprocessedRequest {
    let mut request = request();
    request.bootstrap_info = Some(BootstrapInfo {
        bootstrap_host: "router-prefill".to_string(),
        bootstrap_port: 31_001,
        bootstrap_room: 41,
        handoff_id: None,
    });
    request.routing = Some(RoutingHints {
        prefill_dp_rank,
        ..Default::default()
    });
    request
}

/// A prefill keeps the router's rendezvous and names the router's rank, on
/// the wire and in its handoff; decode names the router's prefill rank.
#[test]
fn prefill_names_dispatched_rank() {
    let (prefill, handoff) = build_generate_request(
        &routed_bootstrap(Some(3)),
        "rid",
        DisaggregationMode::Prefill,
        ranked(Some(3), 31_003, 4),
    )
    .unwrap();
    assert_eq!(prefill.disagg_prefill_dp_rank, Some(3));
    assert!(prefill.routed_dp_rank.is_none());
    // The router's rendezvous is kept as it is.
    assert_eq!(
        rendezvous(&prefill),
        Some(("router-prefill".to_string(), 31_001, 41))
    );
    assert_eq!(handoff.unwrap()["disagg_prefill_dp_rank"], 3);

    let decode = build(&routed_bootstrap(Some(3)), DisaggregationMode::Decode).unwrap();
    assert_eq!(decode.disagg_prefill_dp_rank, Some(3));
    assert!(decode.routed_dp_rank.is_none());
    let decode = build(&routed_bootstrap(None), DisaggregationMode::Decode).unwrap();
    assert_eq!(decode.disagg_prefill_dp_rank, None);
}

/// A rendezvous prefill builds itself has the dispatched listener's port and
/// a room congruent to its rank modulo the DP size. Prefill and its handoff
/// name the rank only when the router named it, and decode forwards exactly
/// that; a malformed rank is refused.
#[test]
fn handoff_names_rank_only_when_prefill_named_it() {
    for (routed, dispatched, size, named) in [
        (Some(3), Some(3), 4, Some(3)),
        (None, Some(3), 4, None),
        (None, None, 1, None),
    ] {
        let mut prefill_request = request();
        prefill_request.routing = Some(RoutingHints {
            prefill_dp_rank: routed,
            ..Default::default()
        });
        let (prefill, handoff) = build_generate_request(
            &prefill_request,
            "rid",
            DisaggregationMode::Prefill,
            ranked(dispatched, 31_003, size),
        )
        .unwrap();
        let (_, port, room) = rendezvous(&prefill).unwrap();
        assert_eq!(port, 31_003);
        if let Some(rank) = dispatched {
            assert_eq!(room % i64::from(size), i64::from(rank));
        }
        let named = named.map(i64::from);
        assert_eq!(prefill.disagg_prefill_dp_rank, named);
        let handoff = handoff.unwrap();
        assert_eq!(
            handoff
                .get("disagg_prefill_dp_rank")
                .and_then(Value::as_i64),
            named
        );
        let mut decode_request = request();
        decode_request.prefill_result = Some(PrefillResult {
            disaggregated_params: handoff,
            prompt_tokens_details: None,
        });
        let decode = build(&decode_request, DisaggregationMode::Decode).unwrap();
        assert_eq!(rendezvous(&decode), rendezvous(&prefill));
        assert_eq!(decode.disagg_prefill_dp_rank, named);
    }
    let room = room_for_rank(3, 4, i64::MAX as u64);
    assert!(room <= i64::MAX as u64 && room % 4 == 3);
    let mut decode_request = request();
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: json!({
            "bootstrap_host": "prefill",
            "bootstrap_port": 5000,
            "bootstrap_room": 42,
            "disagg_prefill_dp_rank": "1",
        }),
        prompt_tokens_details: None,
    });
    assert_invalid(
        build(&decode_request, DisaggregationMode::Decode).unwrap_err(),
        "disagg_prefill_dp_rank must be a DP rank",
    );
}
