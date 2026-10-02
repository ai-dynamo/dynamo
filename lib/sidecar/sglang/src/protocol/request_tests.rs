// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    BackendError, BootstrapInfo, ErrorType, GuidedDecodingOptions, OutputOptions, PrefillResult,
    SamplingOptions, StopConditions,
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
            json_schema: None,
            regex: None,
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
                seed: Some(0),
                ..Default::default()
            },
            "seed",
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
        "max_thinking_tokens",
        "visible",
        "priority",
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
            "max_thinking_tokens" => {
                request.stop_conditions.max_thinking_tokens = Some(0);
                "max_thinking_tokens"
            }
            "visible" => {
                request.stop_conditions.stop_token_ids_visible = Some(vec![42]);
                "visible stop-token"
            }
            "priority" => {
                request.routing = Some(RoutingHints {
                    priority: Some(-1),
                    ..Default::default()
                });
                "engine priority"
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
fn json_and_regex_guides_preserve_native_payloads() {
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
        assert_eq!(mapped.json_schema.as_deref(), Some(r#"{"type":"string"}"#));
        assert!(mapped.regex.is_none());
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
    assert_eq!(mapped.regex.as_deref(), Some("[a-z]+"));
    assert!(mapped.json_schema.is_none());

    request
        .sampling_options
        .guided_decoding
        .as_mut()
        .unwrap()
        .json = Some(json!({"type": "string"}));
    let mapped = build_generate_request(
        &request,
        "both-guides",
        DisaggregationMode::Aggregated,
        None,
        None,
    )
    .unwrap()
    .sampling_params
    .unwrap();
    assert_eq!(mapped.regex.as_deref(), Some("[a-z]+"));
    assert_eq!(mapped.json_schema.as_deref(), Some(r#"{"type":"string"}"#));
}

#[test]
fn unsupported_guides_and_modifiers_are_rejected() {
    for guided in [
        GuidedDecodingOptions {
            choice: Some(vec!["a".to_string()]),
            ..Default::default()
        },
        GuidedDecodingOptions {
            grammar: Some("root ::= 'a'".to_string()),
            ..Default::default()
        },
        GuidedDecodingOptions {
            backend: Some("xgrammar".to_string()),
            ..Default::default()
        },
        GuidedDecodingOptions {
            whitespace_pattern: Some(" *".to_string()),
            ..Default::default()
        },
        GuidedDecodingOptions {
            structural_tag: Some(json!({})),
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
            "only JSON-schema and regex",
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
