// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_fixtures::*;
use dynamo_backend_common::{
    BackendError, ErrorType, OutputOptions, SamplingOptions, StopConditions,
};
use serde_json::json;

#[test]
fn oversized_logprob_counts_are_rejected() {
    let oversized = i32::MAX as u32 + 1;

    let mut output_request = request();
    output_request.output_options.logprobs = Some(oversized);
    let output_error = build_generate_request(
        output_request,
        "output-logprobs".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("oversized output logprobs must fail");
    assert!(output_error.to_string().contains("must fit in i32"));

    let mut prompt_request = request();
    prompt_request.output_options.prompt_logprobs = Some(oversized);
    let prompt_error = build_generate_request(
        prompt_request,
        "prompt-logprobs".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect_err("oversized prompt logprobs must fail");
    assert!(prompt_error.to_string().contains("must fit in i32"));
}

#[test]
fn skip_special_tokens_is_forwarded_without_compatibility_envelope() {
    let mut request = request();
    request.output_options.skip_special_tokens = Some(false);
    let wire = build_generate_request(
        request,
        "request-1".to_string(),
        DisaggregationMode::Aggregated,
    )
    .expect("native controls should be forwarded");

    assert_eq!(
        wire.response
            .and_then(|response| response.skip_special_tokens),
        Some(false)
    );
}

#[test]
fn compatibility_envelope_preserves_typed_controls() {
    for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Decode] {
        let request = PreprocessedRequest::builder()
            .model("served-model".to_string())
            .token_ids(vec![11, 22, 33])
            .stop_conditions(StopConditions {
                max_tokens: Some(8),
                min_tokens: Some(2),
                ignore_eos: Some(true),
                ..Default::default()
            })
            .sampling_options(SamplingOptions {
                n: Some(1),
                ..Default::default()
            })
            .output_options(OutputOptions::default())
            .prefill_result(if mode.is_decode() {
                decode_request().prefill_result
            } else {
                None
            })
            .extra_args(Some(json!({
                "vllm_tito": {
                    "sampling_params": {
                        "max_tokens": 8,
                        "min_tokens": 2,
                        "ignore_eos": true,
                        "logprobs": 2,
                        "prompt_logprobs": 3,
                        "skip_special_tokens": false
                    }
                }
            })))
            .build()
            .expect("v1.4 request");
        let wire = build_generate_request(request, "legacy".to_string(), mode)
            .expect("legacy typed controls should be preserved");
        let stopping = wire.stopping.expect("stopping");
        assert_eq!(stopping.max_new_tokens, 8);
        assert_eq!(stopping.min_new_tokens, 2);
        assert!(stopping.ignore_eos);
        let response = wire.response.expect("response");
        assert!(response.output_logprobs);
        assert_eq!(
            response.output_candidates.and_then(|tokens| tokens.select),
            Some(pb::candidate_tokens::Select::TopN(2))
        );
        assert!(response.prompt_logprobs);
        assert_eq!(
            response.prompt_candidates.and_then(|tokens| tokens.select),
            Some(pb::candidate_tokens::Select::TopN(3))
        );
        assert_eq!(response.skip_special_tokens, Some(false));
    }
}

#[test]
fn native_sampling_is_rejected_instead_of_silently_discarded() {
    for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Decode] {
        let mut request = request();
        request.extra_args = Some(json!({
            "vllm_tito": {"sampling_params": {"temperature": 0.0}}
        }));
        let error = build_generate_request(request, "native".to_string(), mode)
            .expect_err("released protocol cannot preserve native sampling semantics");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(
            error
                .to_string()
                .contains("sampling_params.temperature is not supported")
        );
    }
}

#[test]
fn prefill_uses_canonical_controls_without_decode_sampling_json() {
    let mut request = request();
    request.extra_args = Some(json!({
        "vllm_tito": {"sampling_params": {"skip_special_tokens": false, "max_tokens": 100}}
    }));
    let wire = build_generate_request(request, "prefill".to_string(), DisaggregationMode::Prefill)
        .expect("prefill does not require native decode sampling");
    let stopping = wire.stopping.expect("stopping");
    assert_eq!(stopping.max_new_tokens, 1);
    assert_eq!(stopping.min_new_tokens, 1);
    assert_eq!(wire.response.unwrap().skip_special_tokens, Some(false));
}

#[test]
fn released_envelope_hydrates_kv_transfer_with_canonical_precedence() {
    let mut legacy = request();
    legacy.extra_args = Some(json!({
        "vllm_tito": {
            "sampling_params": {},
            "kv_transfer_params": {"source": "legacy"}
        }
    }));
    let legacy = normalize_response_options(legacy).expect("normalize legacy KV transfer");
    assert_eq!(
        legacy.extra_args.as_ref().unwrap()["kv_transfer_params"],
        json!({"source": "legacy"})
    );

    let mut canonical = request();
    canonical.extra_args = Some(json!({
        "kv_transfer_params": {"source": "canonical"},
        "vllm_tito": {
            "sampling_params": {},
            "kv_transfer_params": {"source": "legacy"}
        }
    }));
    let canonical = normalize_response_options(canonical).expect("normalize canonical KV transfer");
    assert_eq!(
        canonical.extra_args.as_ref().unwrap()["kv_transfer_params"],
        json!({"source": "canonical"})
    );
}

#[test]
fn canonical_dynamo_priority_is_converted_for_vllm() {
    for (dynamo_priority, vllm_priority) in [(-7, 7), (7, -7), (i32::MIN, i32::MAX)] {
        let mut request = request();
        request.routing.as_mut().expect("routing").priority = Some(dynamo_priority);

        let wire = build_generate_request(
            request,
            "request-1".to_string(),
            DisaggregationMode::Aggregated,
        )
        .expect("canonical priority should be converted");

        assert_eq!(wire.priority, vllm_priority);
    }
}

#[test]
fn unsafe_media_uuids_are_rejected() {
    for uuid in [
        "/tmp/escape",
        "../escape",
        "nested/item",
        "nested\\item",
        ".",
        "..",
        "nul\0item",
    ] {
        let mut request = epd_image_request();
        request
            .multi_modal_uuids
            .as_mut()
            .and_then(|by_modality| by_modality.get_mut("image_url"))
            .expect("image UUIDs")[0] = Some(uuid.to_string());
        let error = build_generate_request(
            request,
            "unsafe-media-uuid".to_string(),
            DisaggregationMode::Encode,
        )
        .expect_err("unsafe UUID must be rejected");
        assert!(error.to_string().contains("safe identifier"), "uuid={uuid}");
    }
}

#[test]
fn encode_requests_reject_non_image_media() {
    let mut request = epd_image_request();
    request.multi_modal_data.as_mut().unwrap().insert(
        "audio_url".to_string(),
        vec![MultimodalData::RawUrl(
            "https://example.com/sample.wav".to_string(),
        )],
    );

    let error = build_generate_request(
        request,
        "encode-audio".to_string(),
        DisaggregationMode::Encode,
    )
    .expect_err("Encode must remain image-only");
    assert!(error.to_string().contains("image media only"));
}

#[test]
fn absent_and_explicit_zero_controls_preserve_native_sentinels() {
    for explicit in [false, true] {
        let mut request = request();
        request.sampling_options = SamplingOptions::default();
        request.stop_conditions = StopConditions::default();
        request.output_options = OutputOptions::default();
        request.extra_args = None;
        request.routing = None;
        if explicit {
            request.sampling_options.temperature = Some(0.0);
            request.sampling_options.top_p = Some(0.0);
            request.sampling_options.min_p = Some(0.0);
            request.sampling_options.seed = Some(0);
            request.sampling_options.presence_penalty = Some(0.0);
            request.sampling_options.frequency_penalty = Some(0.0);
            request.sampling_options.repetition_penalty = Some(0.0);
            request.sampling_options.include_stop_str_in_output = Some(false);
            request.stop_conditions.max_tokens = Some(0);
            request.stop_conditions.min_tokens = Some(0);
            request.stop_conditions.ignore_eos = Some(false);
            request.output_options.logprobs = Some(0);
            request.output_options.prompt_logprobs = Some(0);
            request.output_options.skip_special_tokens = Some(false);
        }
        let wire = build_generate_request(
            request,
            "zero-controls".into(),
            DisaggregationMode::Aggregated,
        )
        .unwrap();
        assert_eq!(wire.request_id, "zero-controls");
        assert_eq!(
            wire.prompt,
            Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![11, 22, 33]
            }))
        );
        assert_eq!(wire.temperature, explicit.then_some(0.0));
        assert_eq!(
            wire.sampling,
            Some(pb::RandomSampling {
                num_sequences: 1,
                top_k: 0,
                top_p: 0.0,
                min_p: 0.0,
                seed: explicit.then_some(0)
            })
        );
        assert_eq!(
            wire.decoding,
            Some(pb::DecodingParameters {
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                repetition_penalty: 0.0,
                ..Default::default()
            })
        );
        assert_eq!(wire.stopping, Some(pb::StoppingCriteria::default()));
        let response = wire.response.unwrap();
        assert_eq!(response.output_logprobs, explicit);
        assert_eq!(response.prompt_logprobs, explicit);
        assert_eq!(response.prompt_token_ids, explicit);
        assert_eq!(response.skip_special_tokens, explicit.then_some(false));
        assert_eq!(
            response.output_candidates.and_then(|value| value.select),
            explicit.then_some(pb::candidate_tokens::Select::TopN(0))
        );
        assert_eq!(
            response.prompt_candidates.and_then(|value| value.select),
            explicit.then_some(pb::candidate_tokens::Select::TopN(0))
        );
        assert!(wire.kv.unwrap().cache_salt.is_empty());
    }
}

#[test]
fn request_boundaries_reject_unsupported_controls_before_submission() {
    type RequestMutation = fn(&mut PreprocessedRequest);
    let cases: &[(&str, RequestMutation)] = &[
        ("token_ids", |r| r.token_ids = Arc::new(Vec::new())),
        ("n must be 1", |r| r.sampling_options.n = Some(2)),
        ("prompt embeddings", |r| {
            r.prompt_embeds = Some("encoded".into())
        }),
        ("best_of", |r| r.sampling_options.best_of = Some(2)),
        ("beam search", |r| {
            r.sampling_options.use_beam_search = Some(true)
        }),
        ("length_penalty", |r| {
            r.sampling_options.length_penalty = Some(0.5)
        }),
        ("top_k", |r| r.sampling_options.top_k = Some(-2)),
        ("visible stop", |r| {
            r.stop_conditions.stop_token_ids_visible = Some(vec![42])
        }),
        ("max_thinking_tokens", |r| {
            r.stop_conditions.max_thinking_tokens = Some(5)
        }),
        ("multimodal features", |r| {
            r.mm_processor_kwargs = Some(json!({}))
        }),
        ("without multi_modal_data", |r| {
            r.multi_modal_uuids = Some(std::collections::HashMap::from([(
                "image_url".into(),
                vec![Some("image-a".into())],
            )]))
        }),
        ("extra_args must be", |r| r.extra_args = Some(json!([]))),
        ("extra_args.unknown", |r| {
            r.extra_args = Some(json!({"unknown": true}))
        }),
        ("must be a boolean", |r| {
            r.extra_args = Some(json!({"bypass_prefix_cache": 0}))
        }),
        ("does not match", |r| {
            r.extra_args = Some(json!({"nvext": {"cache_salt": "other"}}))
        }),
        ("token_in must be true", |r| {
            r.extra_args = Some(json!({"nvext": {"token_in": false}}))
        }),
    ];
    assert!(
        build_generate_request(
            request(),
            "supported".into(),
            DisaggregationMode::Aggregated
        )
        .is_ok()
    );
    for (expected, change) in cases {
        let mut request = request();
        change(&mut request);
        let error =
            build_generate_request(request, "rejected".into(), DisaggregationMode::Aggregated)
                .expect_err(expected);
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument),
            "{expected}"
        );
        assert!(error.to_string().contains(expected), "{expected}: {error}");
    }
}

#[test]
fn guide_variants_preserve_exact_type_and_payload() {
    use pb::decoding_parameters::StructuredOutput;
    for (guide, expected) in [
        (
            GuidedDecodingOptions {
                json: Some(
                    json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}),
                ),
                ..Default::default()
            },
            StructuredOutput::Json(
                r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}"#
                    .into(),
            ),
        ),
        (
            GuidedDecodingOptions {
                regex: Some("[a-z]+".into()),
                ..Default::default()
            },
            StructuredOutput::Regex("[a-z]+".into()),
        ),
        (
            GuidedDecodingOptions {
                grammar: Some("root ::= 'yes'".into()),
                ..Default::default()
            },
            StructuredOutput::Grammar("root ::= 'yes'".into()),
        ),
        (
            GuidedDecodingOptions {
                choice: Some(vec!["yes".into(), "no".into()]),
                ..Default::default()
            },
            StructuredOutput::Choice(pb::decoding_parameters::StringChoices {
                choices: vec!["yes".into(), "no".into()],
            }),
        ),
        (
            GuidedDecodingOptions {
                structural_tag: Some(json!("<answer>")),
                ..Default::default()
            },
            StructuredOutput::StructuralTag("<answer>".into()),
        ),
        (
            GuidedDecodingOptions {
                structural_tag: Some(json!({"tag": "answer"})),
                ..Default::default()
            },
            StructuredOutput::StructuralTag(r#"{"tag":"answer"}"#.into()),
        ),
    ] {
        let mut request = request();
        request.sampling_options.guided_decoding = Some(guide);
        let wire = build_generate_request(request, "guided".into(), DisaggregationMode::Aggregated)
            .unwrap();
        assert_eq!(wire.decoding.unwrap().structured_output, Some(expected));
    }
    for guide in [
        GuidedDecodingOptions {
            json: Some(json!({})),
            regex: Some(".*".into()),
            ..Default::default()
        },
        GuidedDecodingOptions {
            backend: Some("xgrammar".into()),
            ..Default::default()
        },
        GuidedDecodingOptions {
            whitespace_pattern: Some(" *".into()),
            ..Default::default()
        },
    ] {
        let mut request = request();
        request.sampling_options.guided_decoding = Some(guide);
        let error = build_generate_request(
            request,
            "unsupported-guide".into(),
            DisaggregationMode::Aggregated,
        )
        .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
}

#[test]
fn stops_top_k_and_adapter_rank_hints_preserve_supported_selection() {
    for (top_k, expected) in [
        (None, 0),
        (Some(-1), 0),
        (Some(0), 0),
        (Some(7), 7),
        (Some(i32::MAX), i32::MAX as u32),
    ] {
        let mut request = request();
        request.sampling_options.top_k = top_k;
        request.stop_conditions.stop_token_ids = Some(vec![3, 2, 3]);
        request.stop_conditions.stop_token_ids_hidden = Some(vec![2, 4]);
        let routing = request.routing.as_mut().unwrap();
        routing.lora_name = Some("adapter-a".into());
        routing.dp_rank = Some(5);
        routing.prefill_dp_rank = Some(3);
        for (mode, expected_rank) in [
            (DisaggregationMode::Aggregated, Some(5)),
            (DisaggregationMode::Decode, Some(5)),
            (DisaggregationMode::Prefill, Some(3)),
            (DisaggregationMode::Encode, None),
        ] {
            assert_eq!(data_parallel_rank(&request, mode), expected_rank);
        }
        request.routing.as_mut().unwrap().prefill_dp_rank = None;
        assert_eq!(
            data_parallel_rank(&request, DisaggregationMode::Prefill),
            Some(5)
        );
        let wire =
            build_generate_request(request, "selected".into(), DisaggregationMode::Aggregated)
                .unwrap();
        assert_eq!(wire.sampling.unwrap().top_k, expected);
        assert_eq!(wire.stopping.unwrap().stop_token_ids, vec![2, 3, 4]);
        assert_eq!(wire.lora_name, "adapter-a");
    }
}

#[test]
fn canonical_cache_identity_and_bypass_alias_precedence_are_preserved() {
    for (extra, expected) in [
        (
            json!({"bypass_prefix_cache": false, "skip_reading_prefix_cache": true}),
            false,
        ),
        (json!({"skip_reading_prefix_cache": true}), true),
        (json!({}), false),
    ] {
        let mut request = request();
        request.extra_args = Some(extra);
        let wire = build_generate_request(request, "cache".into(), DisaggregationMode::Aggregated)
            .unwrap();
        let kv = wire.kv.unwrap();
        assert_eq!(kv.cache_salt, "dynamo-cache-salt:cache-salt");
        assert_eq!(kv.bypass_prefix_cache, expected);
    }
    let mut prefixed = request();
    prefixed.routing.as_mut().unwrap().cache_namespace = Some("dynamo-cache-salt:caller".into());
    prefixed.extra_args = None;
    let wire = build_generate_request(prefixed, "prefixed".into(), DisaggregationMode::Aggregated)
        .unwrap();
    assert_eq!(
        wire.kv.unwrap().cache_salt,
        "dynamo-cache-salt:dynamo-cache-salt:caller"
    );
    let mut no_namespace = request();
    no_namespace.routing = None;
    no_namespace.extra_args = None;
    let wire = build_generate_request(
        no_namespace,
        "no-cache-identity".into(),
        DisaggregationMode::Aggregated,
    )
    .unwrap();
    assert!(wire.kv.unwrap().cache_salt.is_empty());
}

#[test]
fn handoff_payload_precedence_ports_and_malformed_inputs_are_checked() {
    for (port, expected) in [
        (json!(5600), json!("5600")),
        (json!(5600.0), json!("5600")),
        (json!("5600"), json!("5600")),
    ] {
        let mut request = decode_request();
        let handoff =
            json!({"remote_port": port, "opaque": {"flags": [true, null, {"ids": [1, 2]}]}});
        request
            .prefill_result
            .as_mut()
            .unwrap()
            .disaggregated_params = handoff.clone();
        let mut expected_handoff = handoff;
        expected_handoff["remote_port"] = expected;
        for _ in 0..2 {
            let wire = build_generate_request(
                request.clone(),
                "decode".into(),
                DisaggregationMode::Decode,
            )
            .unwrap();
            assert_eq!(
                struct_to_json(wire.kv.unwrap().kv_transfer_params.unwrap()).unwrap(),
                expected_handoff
            );
        }
    }
    for value in [None, Some(json!([])), Some(json!("invalid"))] {
        let mut request = request();
        request.prefill_result = value.map(|disaggregated_params| PrefillResult {
            disaggregated_params,
            prompt_tokens_details: None,
        });
        let error =
            build_generate_request(request, "bad-handoff".into(), DisaggregationMode::Decode)
                .unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
}

#[test]
fn representative_request_preserves_all_supported_native_fields() {
    let sent = build_generate_request(
        request(),
        "native-fields".into(),
        DisaggregationMode::Aggregated,
    )
    .unwrap();
    assert_eq!(sent.priority, 0);
    assert_eq!(sent.temperature, Some(0.2));
    let sampling = sent.sampling.as_ref().unwrap();
    assert_eq!(
        (sampling.top_k, sampling.top_p, sampling.min_p),
        (4, 0.9, 0.1)
    );
    assert_eq!(sampling.seed, Some(123));
    let decoding = sent.decoding.as_ref().unwrap();
    assert_eq!(
        (
            decoding.presence_penalty,
            decoding.frequency_penalty,
            decoding.repetition_penalty,
        ),
        (0.3, 0.4, 1.1)
    );
    assert!(matches!(
        decoding.structured_output,
        Some(pb::decoding_parameters::StructuredOutput::Json(_))
    ));
    let stopping = sent.stopping.as_ref().unwrap();
    assert_eq!((stopping.max_new_tokens, stopping.min_new_tokens), (1, 1));
    assert_eq!(stopping.stop_strings, ["done"]);
    assert!(stopping.include_stop_strings);
    assert!(stopping.ignore_eos);
    let kv = sent.kv.as_ref().unwrap();
    assert!(kv.bypass_prefix_cache);
    assert_eq!(kv.cache_salt, "dynamo-cache-salt:cache-salt");
    assert_eq!(
        struct_to_json(kv.kv_transfer_params.clone().unwrap()).unwrap(),
        json!({"connector_data": {"values": [1, true, null]}})
    );
}
