// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    BackendError, BootstrapInfo, ErrorType, GuidedDecodingOptions, KvHint, KvHintAction,
    KvSourceLocationsPayload, OutputOptions, PrefillResult, SamplingOptions, StopConditions,
};
use dynamo_kv_router::protocols::ExternalSequenceBlockHash;
use prost::Message;
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

fn assert_invalid(error: DynamoError, message: &str) {
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(error.to_string().contains(message), "{error}");
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
    let mapped =
        build_generate_request(&request, "rid-1", DisaggregationMode::Decode, None, None).unwrap();
    assert_eq!(mapped.input_ids, vec![1, 2, 3]);
    assert_eq!(mapped.rid.as_deref(), Some("rid-1"));
    assert_eq!(mapped.sampling_params.unwrap().max_new_tokens, Some(8));
    assert_eq!(
        mapped.disaggregated_params.unwrap().bootstrap_room,
        i64::MAX
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
    let mapped = build_generate_request(
        &request,
        "rid-2",
        DisaggregationMode::Prefill,
        Some("prefill"),
        Some(5001),
    )
    .unwrap();
    let sampling = mapped.sampling_params.unwrap();
    assert_eq!(sampling.max_new_tokens, Some(1));
    assert_eq!(sampling.min_new_tokens, None);
    assert_eq!(mapped.return_logprob, Some(false));
    assert_eq!(mapped.top_logprobs_num, Some(0));
    assert_eq!(mapped.logprob_start_len, Some(-1));
    assert_eq!(mapped.disaggregated_params.unwrap().bootstrap_port, 5001);
}

#[test]
fn prefill_uses_selected_prefill_dp_rank() {
    let mut request = request();
    request.routing = Some(RoutingHints {
        dp_rank: Some(7),
        prefill_dp_rank: Some(3),
        ..Default::default()
    });

    assert_eq!(
        routed_dp_rank(&request, DisaggregationMode::Prefill),
        Some(3)
    );
    assert_eq!(
        routed_dp_rank(&request, DisaggregationMode::Aggregated),
        Some(7)
    );

    request.routing.as_mut().unwrap().prefill_dp_rank = None;
    assert_eq!(
        routed_dp_rank(&request, DisaggregationMode::Prefill),
        Some(7)
    );
}

#[test]
fn prefill_handoff_round_trips_to_decode_request() {
    let prefill = build_generate_request(
        &request(),
        "rid-prefill",
        DisaggregationMode::Prefill,
        Some("prefill.internal"),
        Some(5001),
    )
    .unwrap();
    let handoff = prefill.disaggregated_params.unwrap();

    let mut decode_request = request();
    decode_request.prefill_result = Some(PrefillResult {
        disaggregated_params: disaggregated_params_to_json(&handoff),
        prompt_tokens_details: None,
    });
    let decode = build_generate_request(
        &decode_request,
        "rid-decode",
        DisaggregationMode::Decode,
        None,
        None,
    )
    .unwrap();

    assert_eq!(decode.disaggregated_params, Some(handoff));
}

#[test]
fn decode_requires_rendezvous_params() {
    let error = build_generate_request(&request(), "rid-3", DisaggregationMode::Decode, None, None)
        .unwrap_err();
    assert_eq!(error.public_message(), None);
}

#[test]
fn request_refusal_is_public() {
    let mut refused = request();
    refused.mm_processor_kwargs = Some(json!({}));
    let error = build_generate_request(
        &refused,
        "rid-5",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(
        error.public_message(),
        Some("multimodal payloads are not supported by SGLang's native Generate RPC")
    );

    let mut embeds = request();
    embeds.token_ids = Vec::new().into();
    embeds.prompt_embeds = Some("embeds".to_string());
    let error =
        build_generate_request(&embeds, "rid-6", DisaggregationMode::Aggregated, None, None)
            .unwrap_err();
    assert_eq!(
        error.public_message(),
        Some("prompt_embeds are not supported by SGLang's native gRPC proto")
    );

    let mut stop = request();
    stop.stop_conditions.stop_token_ids = Some(vec![u32::MAX]);
    let error = build_generate_request(&stop, "rid-7", DisaggregationMode::Aggregated, None, None)
        .unwrap_err();
    assert_eq!(
        error.public_message(),
        Some("stop token ids must fit in i32")
    );
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
    assert!(
        build_generate_request(&request, "rid-4", DisaggregationMode::Decode, None, None,).is_err()
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
    let mapped = build_generate_request(
        &request,
        "sampling",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        mapped.sampling_params,
        Some(pb::SamplingParams {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(17),
            min_p: Some(0.05),
            frequency_penalty: Some(0.2),
            presence_penalty: Some(-0.3),
            repetition_penalty: Some(1.1),
            max_new_tokens: Some(11),
            min_new_tokens: Some(2),
            stop: vec!["end".to_string(), "done".to_string()],
            stop_token_ids: vec![19, 20, 21],
            ignore_eos: Some(true),
            n: Some(1),
            seed: None,
            json_schema: None,
            regex: None,
            guided_decoding: None,
        })
    );
    assert_eq!(mapped.stream, Some(true));
    assert!(mapped.disaggregated_params.is_none());
    assert!(mapped.session_id.is_none());
}

#[test]
fn optional_controls_distinguish_absence_from_explicit_zero_or_false() {
    for explicit in [false, true] {
        let mut request = request();
        request.stop_conditions = StopConditions {
            max_tokens: explicit.then_some(0),
            min_tokens: explicit.then_some(0),
            ignore_eos: explicit.then_some(false),
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
        let mapped = build_generate_request(
            &request,
            "optional",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            mapped.sampling_params,
            Some(pb::SamplingParams {
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
            })
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
        let mapped = build_generate_request(
            &request,
            "logprobs",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap();
        assert_eq!(mapped.return_logprob, Some(enabled));
        assert_eq!(mapped.top_logprobs_num, Some(count));
        assert_eq!(mapped.logprob_start_len, Some(start));
    }
}

#[test]
fn values_outside_native_signed_fields_are_rejected() {
    for (field, message) in [
        ("token", "token ids must fit in i32"),
        ("stop", "stop token ids must fit in i32"),
        ("hidden_stop", "stop token ids must fit in i32"),
        ("max_tokens", "max_tokens does not fit in i32"),
        ("min_tokens", "min_tokens does not fit in i32"),
        ("logprobs", "requested logprobs does not fit in i32"),
        ("prompt_logprobs", "requested logprobs does not fit in i32"),
        ("dp_rank", "routed dp_rank does not fit in i32"),
    ] {
        let mut request = request();
        let overflow = i32::MAX as u32 + 1;
        match field {
            "token" => request.token_ids = vec![overflow].into(),
            "stop" => request.stop_conditions.stop_token_ids = Some(vec![overflow]),
            "hidden_stop" => request.stop_conditions.stop_token_ids_hidden = Some(vec![overflow]),
            "max_tokens" => request.stop_conditions.max_tokens = Some(overflow),
            "min_tokens" => request.stop_conditions.min_tokens = Some(overflow),
            "logprobs" => request.output_options.logprobs = Some(overflow),
            "prompt_logprobs" => request.output_options.prompt_logprobs = Some(overflow),
            "dp_rank" => {
                request.routing = Some(RoutingHints {
                    dp_rank: Some(overflow),
                    ..Default::default()
                })
            }
            _ => unreachable!(),
        }
        let error =
            build_generate_request(&request, field, DisaggregationMode::Aggregated, None, None)
                .unwrap_err();
        assert_eq!(error.public_message(), Some(message));
        assert_invalid(error, message);
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
        let error =
            build_generate_request(&request, label, DisaggregationMode::Aggregated, None, None)
                .unwrap_err();
        assert_invalid(error, label);
    }
}

#[test]
fn unsupported_payload_stopping_and_priority_controls_are_rejected() {
    for field in [
        "token_ids",
        "prompt_embeds",
        "multimodal",
        "mm_processor_kwargs",
        "visible",
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
            "visible" => {
                request.stop_conditions.stop_token_ids_visible = Some(vec![42]);
                "visible stop-token"
            }
            _ => unreachable!(),
        };
        let error =
            build_generate_request(&request, field, DisaggregationMode::Aggregated, None, None)
                .unwrap_err();
        assert_invalid(error, message);
    }
}

#[test]
#[allow(deprecated)]
fn json_and_regex_guides_use_typed_constraints() {
    for schema in [json!({"type": "string"}), json!(r#"{"type":"string"}"#)] {
        let mut request = request();
        request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
            json: Some(schema),
            ..Default::default()
        });
        let mapped =
            build_generate_request(&request, "json", DisaggregationMode::Aggregated, None, None)
                .unwrap()
                .sampling_params
                .unwrap();
        assert_eq!(mapped.json_schema, None);
        assert!(mapped.regex.is_none());
        assert_eq!(
            mapped.guided_decoding.unwrap().constraint,
            Some(pb::guided_decoding::Constraint::JsonSchema(
                r#"{"type":"string"}"#.to_string()
            ))
        );
    }
    let mut request = request();
    request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
        regex: Some("[a-z]+".to_string()),
        ..Default::default()
    });
    let mapped = build_generate_request(
        &request,
        "regex",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap()
    .sampling_params
    .unwrap();
    assert_eq!(mapped.regex, None);
    assert!(mapped.json_schema.is_none());
    assert_eq!(
        mapped.guided_decoding.unwrap().constraint,
        Some(pb::guided_decoding::Constraint::Regex("[a-z]+".to_string()))
    );

    request
        .sampling_options
        .guided_decoding
        .as_mut()
        .unwrap()
        .json = Some(json!({"type": "string"}));
    assert_invalid(
        build_generate_request(
            &request,
            "both-guides",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap_err(),
        "only one guided-decoding constraint",
    );
}

#[test]
fn additional_guides_use_typed_constraints() {
    for (guided, expected) in [
        (
            GuidedDecodingOptions {
                choice: Some(vec!["a".to_string(), "b".to_string()]),
                ..Default::default()
            },
            pb::guided_decoding::Constraint::Choice(pb::ChoiceConstraint {
                values: vec!["a".to_string(), "b".to_string()],
            }),
        ),
        (
            GuidedDecodingOptions {
                grammar: Some("root ::= 'a'".to_string()),
                ..Default::default()
            },
            pb::guided_decoding::Constraint::Ebnf("root ::= 'a'".to_string()),
        ),
        (
            GuidedDecodingOptions {
                structural_tag: Some(json!({"type": "object"})),
                ..Default::default()
            },
            pb::guided_decoding::Constraint::StructuralTag(r#"{"type":"object"}"#.to_string()),
        ),
    ] {
        let mut request = request();
        request.sampling_options.guided_decoding = Some(guided);
        let mapped = build_generate_request(
            &request,
            "guide",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            mapped
                .sampling_params
                .unwrap()
                .guided_decoding
                .unwrap()
                .constraint,
            Some(expected)
        );
    }
}

#[test]
fn unsupported_guide_modifiers_are_rejected() {
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
            build_generate_request(
                &request,
                "guide",
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap_err(),
            "backend and whitespace modifiers",
        );
    }
}

#[test]
fn sglang_0521_request_controls_are_forwarded() {
    let mut request = request();
    request.sampling_options.seed = Some(42);
    request.stop_conditions.max_thinking_tokens = Some(17);
    request.require_reasoning = true;
    request.routing = Some(RoutingHints {
        priority: Some(-3),
        ..Default::default()
    });
    request.kv_hint = Some(KvHint::new(
        "message-1",
        vec![KvHintAction::new(
            "action-1",
            "kv.fetch",
            "1.0",
            std::collections::BTreeMap::from([("source".to_string(), json!("worker-7"))]),
        )],
    ));

    let mapped = build_generate_request(
        &request,
        "controls",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    assert_eq!(mapped.sampling_params.unwrap().seed, Some(42));
    assert_eq!(mapped.priority, Some(-3));
    assert_eq!(mapped.require_reasoning, Some(true));
    assert_eq!(mapped.max_thinking_tokens, Some(17));
    let hints = mapped.kv_hints.unwrap();
    assert_eq!(hints.protocol_version, "0.1");
    assert_eq!(hints.message_id, "message-1");
    assert_eq!(hints.actions.len(), 1);
    let action = &hints.actions[0];
    assert_eq!(action.action_id, "action-1");
    assert_eq!(action.action_type, "kv.fetch");
    assert_eq!(action.action_version, "1.0");
    assert_eq!(
        dynamo_sidecar_common::struct_to_json(
            action.payload.clone().unwrap(),
            "SGLang",
            "kv hint action payload",
        )
        .unwrap(),
        json!({"source": "worker-7"})
    );
}

fn fetch_hint_request(hashes: Value) -> PreprocessedRequest {
    let mut request = request();
    request.kv_hint = Some(KvHint::new(
        "fetch-message",
        vec![KvHintAction::new(
            "fetch-action",
            "kv.fetch",
            "1.0",
            std::collections::BTreeMap::from([
                ("block_hashes".to_string(), hashes),
                (
                    "source_control_endpoint".to_string(),
                    json!("tcp://peer:12000"),
                ),
                ("extension".to_string(), json!({"mode": "copy", "count": 2})),
            ]),
        )],
    ));
    request
}

#[test]
fn fetch_hashes_survive_generate_request_protobuf_round_trip() {
    let hashes = [
        0,
        1,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        i64::MAX as u64,
        u64::MAX,
    ];
    let mut request = request();
    let mut action = KvHintAction::fetch(
        "fetch-action",
        KvSourceLocationsPayload {
            source_control_endpoint: "tcp://peer:12000".to_string(),
            block_hashes: hashes.into_iter().map(ExternalSequenceBlockHash).collect(),
        },
    );
    action
        .payload
        .insert("extension".into(), json!({"mode": "copy", "count": 2}));
    request.kv_hint = Some(KvHint::new("fetch-message", vec![action]));
    let original_hint = request.kv_hint.clone();
    let mapped = build_generate_request(
        &request,
        "fetch",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    // Exercise the actual protobuf wire, not just the JSON conversion helper.
    let decoded = pb::GenerateRequest::decode(mapped.encode_to_vec().as_slice()).unwrap();
    let hints = decoded.kv_hints.unwrap();
    assert_eq!(hints.protocol_version, "0.1");
    assert_eq!(hints.message_id, "fetch-message");
    let action = &hints.actions[0];
    assert_eq!(action.action_id, "fetch-action");
    assert_eq!(action.action_type, "kv.fetch");
    assert_eq!(action.action_version, "1.0");
    let payload = dynamo_sidecar_common::struct_to_json(
        action.payload.clone().unwrap(),
        "SGLang",
        "kv hint action payload",
    )
    .unwrap();
    assert_eq!(payload["source_control_endpoint"], "tcp://peer:12000");
    assert_eq!(payload["extension"], json!({"mode": "copy", "count": 2}));
    for (wire_hash, expected) in payload["block_hashes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(hashes)
    {
        let wire_hash = wire_hash.as_str().unwrap();
        assert_eq!(wire_hash.len(), 16);
        assert_eq!(u64::from_str_radix(wire_hash, 16).unwrap(), expected);
    }
    assert_eq!(
        request.kv_hint, original_hint,
        "lowering must not mutate the router request"
    );
}

#[test]
fn fetch_hashes_preserve_signed_bit_patterns_and_existing_strings() {
    let request = fetch_hint_request(json!([
        -1,
        i64::MIN,
        "0000000000001234",
        "ABCDEF0123456789"
    ]));
    let mapped = build_generate_request(
        &request,
        "fetch",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    let decoded = pb::GenerateRequest::decode(mapped.encode_to_vec().as_slice()).unwrap();
    let payload = dynamo_sidecar_common::struct_to_json(
        decoded.kv_hints.unwrap().actions.remove(0).payload.unwrap(),
        "SGLang",
        "kv hint action payload",
    )
    .unwrap();
    assert_eq!(
        payload["block_hashes"],
        json!([
            "ffffffffffffffff",
            "8000000000000000",
            "0000000000001234",
            "ABCDEF0123456789",
        ])
    );
}

#[test]
fn malformed_fetch_hashes_are_rejected_without_float_coercion() {
    for hashes in [
        json!(42),
        json!([1.5]),
        json!([1.0]),
        json!([true]),
        json!([null]),
        json!([{}]),
    ] {
        let request = fetch_hint_request(hashes);
        assert_invalid(
            build_generate_request(
                &request,
                "fetch",
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap_err(),
            "kv.fetch block_hashes",
        );
    }
}

#[test]
fn unknown_hint_actions_and_versions_are_not_rewritten() {
    let mut request = fetch_hint_request(json!([23]));
    request.kv_hint.as_mut().unwrap().actions.extend([
        KvHintAction::new(
            "unknown-action",
            "future.action",
            "1.0",
            std::collections::BTreeMap::from([("block_hashes".to_string(), json!([23]))]),
        ),
        KvHintAction::new(
            "future-fetch",
            "kv.fetch",
            "2.0",
            std::collections::BTreeMap::from([("block_hashes".to_string(), json!([23]))]),
        ),
    ]);
    let mapped = build_generate_request(
        &request,
        "fetch",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    let hints = pb::GenerateRequest::decode(mapped.encode_to_vec().as_slice())
        .unwrap()
        .kv_hints
        .unwrap();
    assert_eq!(hints.actions.len(), 3);
    for action in &hints.actions[1..] {
        let payload = dynamo_sidecar_common::struct_to_json(
            action.payload.clone().unwrap(),
            "SGLang",
            "kv hint action payload",
        )
        .unwrap();
        assert_eq!(payload, json!({"block_hashes": [23]}));
    }
}

#[test]
fn unknown_hint_protocol_is_forwarded_without_known_action_lowering() {
    let mut request = fetch_hint_request(json!([23]));
    request.kv_hint.as_mut().unwrap().protocol_version = "0.2".into();
    let mapped = build_generate_request(
        &request,
        "fetch",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    let hints = pb::GenerateRequest::decode(mapped.encode_to_vec().as_slice())
        .unwrap()
        .kv_hints
        .unwrap();
    assert_eq!(hints.protocol_version, "0.2");
    let payload = dynamo_sidecar_common::struct_to_json(
        hints.actions[0].payload.clone().unwrap(),
        "SGLang",
        "kv hint action payload",
    )
    .unwrap();
    assert_eq!(payload["block_hashes"], json!([23]));
}

#[test]
fn unrelated_payload_integers_still_require_exact_struct_representation() {
    let mut request = fetch_hint_request(json!([u64::MAX]));
    request.kv_hint.as_mut().unwrap().actions[0]
        .payload
        .insert("extension".into(), json!(u64::MAX));
    assert_invalid(
        build_generate_request(
            &request,
            "fetch",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap_err(),
        "cannot be represented exactly",
    );
    for (action_type, action_version) in [("future.action", "1.0"), ("kv.fetch", "2.0")] {
        let mut request = fetch_hint_request(json!([u64::MAX]));
        let action = &mut request.kv_hint.as_mut().unwrap().actions[0];
        action.action_type = action_type.into();
        action.action_version = action_version.into();
        assert_invalid(
            build_generate_request(
                &request,
                "fetch",
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap_err(),
            "cannot be represented exactly",
        );
    }
}

#[test]
fn selected_adapter_cache_identity_and_role_rank_are_forwarded() {
    let mut request = request();
    request.mdc_sum = Some("model-cache-key".to_string());
    request.routing = Some(RoutingHints {
        lora_name: Some("adapter-a".to_string()),
        dp_rank: Some(0),
        prefill_dp_rank: Some(3),
        priority: Some(0),
        ..Default::default()
    });
    for (mode, rank) in [
        (DisaggregationMode::Aggregated, 0),
        (DisaggregationMode::Prefill, 3),
    ] {
        let mapped =
            build_generate_request(&request, "routed", mode, Some("prefill"), Some(5000)).unwrap();
        assert_eq!(mapped.lora_path.as_deref(), Some("adapter-a"));
        assert_eq!(mapped.routing_key.as_deref(), Some("model-cache-key"));
        assert_eq!(mapped.routed_dp_rank, Some(rank));
    }
    request.routing = None;
    request.mdc_sum = None;
    let mapped = build_generate_request(
        &request,
        "unrouted",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap();
    assert!(mapped.lora_path.is_none());
    assert!(mapped.routing_key.is_none());
    assert!(mapped.routed_dp_rank.is_none());
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
        let params = resolve_disaggregated_params(&request, mode, Some("discovery"), Some(5002))
            .unwrap()
            .unwrap();
        assert_eq!(params.bootstrap_host, "router-prefill");
        assert_eq!(params.bootstrap_port, 5000);
        assert_eq!(params.bootstrap_room, 23);
    }
    assert!(
        resolve_disaggregated_params(&request, DisaggregationMode::Aggregated, None, None)
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
            resolve_disaggregated_params(&request(), DisaggregationMode::Prefill, host, port)
                .unwrap_err(),
            message,
        );
    }
    let params = resolve_disaggregated_params(
        &request(),
        DisaggregationMode::Prefill,
        Some("prefill"),
        Some(5000),
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
            resolve_disaggregated_params(&request, DisaggregationMode::Decode, None, None)
                .unwrap_err(),
            field,
        );
    }
}
