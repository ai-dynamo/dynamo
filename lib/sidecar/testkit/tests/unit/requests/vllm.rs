// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::unit_vllm_fixtures::*;
use dynamo_backend_common::{
    BackendError, ErrorType, OutputOptions, SamplingOptions, StopConditions,
};
use serde_json::json;

#[path = "shared.rs"]
mod shared;

use shared::{Feature, Guide, InvalidRequest, RequestAdapter, RequestObservation, Support};

struct Backend;

impl RequestAdapter for Backend {
    const PREFILL_MIN_TOKENS: Option<u32> = Some(1);
    const TOP_K_EXPECTED: [Option<i64>; 5] =
        [Some(0), Some(0), Some(0), Some(7), Some(i32::MAX as i64)];
    const PRIORITY_EXPECTED: [i32; 3] = [7, -7, i32::MAX];
    const STOP_TOKEN_IDS: &'static [u32] = &[2, 3, 4];
    const CACHE_IDENTITY: &'static str = "dynamo-cache-salt:cache-salt";

    fn lower(
        request: PreprocessedRequest,
        request_id: &str,
        mode: DisaggregationMode,
    ) -> Result<RequestObservation, dynamo_backend_common::DynamoError> {
        let wire = build_generate_request(request, request_id.into(), mode)?;
        let Some(pb::generate_request::Prompt::TokenIds(tokens)) = wire.prompt else {
            panic!("expected native token prompt");
        };
        let sampling = wire.sampling.expect("sampling");
        let decoding = wire.decoding.expect("decoding");
        let stopping = wire.stopping.expect("stopping");
        let response = wire.response.expect("response");
        let kv = wire.kv.expect("KV controls");
        use pb::decoding_parameters::StructuredOutput;
        let guide = decoding.structured_output.map(|guide| match guide {
            StructuredOutput::Json(value) => Guide::Json(value),
            StructuredOutput::Regex(value) => Guide::Regex(value),
            StructuredOutput::Grammar(value) => Guide::Grammar(value),
            StructuredOutput::Choice(value) => Guide::Choice(value.choices),
            StructuredOutput::StructuralTag(value) => Guide::StructuralTag(value),
            StructuredOutput::JsonObject(_) => panic!("unexpected native JSON-object guide"),
        });
        Ok(RequestObservation {
            request_id: wire.request_id,
            token_ids: tokens.ids,
            temperature: wire.temperature,
            top_k: Some(i64::from(sampling.top_k)),
            top_p: Some(sampling.top_p),
            min_p: Some(sampling.min_p),
            presence_penalty: Some(decoding.presence_penalty),
            frequency_penalty: Some(decoding.frequency_penalty),
            repetition_penalty: Some(decoding.repetition_penalty),
            max_tokens: Some(stopping.max_new_tokens),
            min_tokens: Some(stopping.min_new_tokens),
            stop_strings: stopping.stop_strings,
            stop_token_ids: stopping.stop_token_ids,
            ignore_eos: Some(stopping.ignore_eos),
            output_logprobs: response.output_logprobs,
            prompt_logprobs: response.prompt_logprobs,
            skip_special_tokens: response.skip_special_tokens,
            priority: Some(wire.priority),
            lora_name: Some(wire.lora_name),
            cache_identity: Some(kv.cache_salt),
            bypass_prefix_cache: Some(kv.bypass_prefix_cache),
            guide,
            guide_backend: None,
            guide_whitespace: None,
            handoff: kv.kv_transfer_params.map(struct_to_json).transpose()?,
        })
    }

    fn support(feature: Feature) -> Support {
        match feature {
            Feature::SkipSpecialTokens
            | Feature::Priority
            | Feature::CacheIdentity
            | Feature::CacheBypass
            | Feature::JsonGuide
            | Feature::RegexGuide
            | Feature::GrammarGuide
            | Feature::ChoiceGuide
            | Feature::StructuralTagGuide => Support::Supported,
            Feature::GuideBackend => {
                Support::Rejected("the vLLM protocol cannot select a guide backend")
            }
            Feature::GuideWhitespace => {
                Support::Rejected("the vLLM protocol cannot set guide whitespace")
            }
        }
    }

    fn invalid_message(case: InvalidRequest) -> &'static str {
        match case {
            InvalidRequest::EmptyTokens => "token_ids",
            InvalidRequest::MultipleSequences => "n must be 1",
            InvalidRequest::PromptEmbeddings => "prompt embeddings",
            InvalidRequest::BestOf => "best_of",
            InvalidRequest::BeamSearch => "beam search",
            InvalidRequest::LengthPenalty => "length_penalty",
            InvalidRequest::TopK => "top_k",
            InvalidRequest::VisibleStop => "visible stop",
            InvalidRequest::ThinkingTokens => "max_thinking_tokens",
            InvalidRequest::MultimodalProcessor => "multimodal features",
            InvalidRequest::OrphanMediaIds => "without multi_modal_data",
        }
    }

    fn rank(request: &PreprocessedRequest, mode: DisaggregationMode) -> Option<u32> {
        data_parallel_rank(request, mode)
    }

    fn handoff_fixture() -> (serde_json::Value, serde_json::Value) {
        (
            json!({"remote_engine_id": "prefill-0", "remote_host": "127.0.0.1", "remote_port": 20097, "remote_block_ids": [7, 8]}),
            json!({"remote_engine_id": "prefill-0", "remote_host": "127.0.0.1", "remote_port": "20097", "remote_block_ids": [7, 8]}),
        )
    }
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn absent_and_explicit_zero_controls_preserve_native_sentinels() {
        for explicit in [false, true] {
            let mut request = crate::unit_fixtures::minimal_request();
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
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn native_envelope_controls_are_validated() {
        type RequestMutation = fn(&mut PreprocessedRequest);
        let cases: &[(&str, RequestMutation)] = &[
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
}

sidecar_test! {
    lane: pre_merge;
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
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn native_handoff_ports_and_opaque_payloads_are_preserved() {
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
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn vllm_extensions_preserve_native_fields() {
        let sent = build_generate_request(
            request(),
            "native-fields".into(),
            DisaggregationMode::Aggregated,
        )
        .unwrap();
        assert_eq!(sent.priority, 0);
        let sampling = sent.sampling.as_ref().unwrap();
        assert_eq!(sampling.seed, Some(123));
        let decoding = sent.decoding.as_ref().unwrap();
        assert!(matches!(
            decoding.structured_output,
            Some(pb::decoding_parameters::StructuredOutput::Json(_))
        ));
        let stopping = sent.stopping.as_ref().unwrap();
        assert!(stopping.include_stop_strings);
        let kv = sent.kv.as_ref().unwrap();
        assert!(kv.bypass_prefix_cache);
        assert_eq!(kv.cache_salt, "dynamo-cache-salt:cache-salt");
        assert_eq!(
            struct_to_json(kv.kv_transfer_params.clone().unwrap()).unwrap(),
            json!({"connector_data": {"values": [1, true, null]}})
        );
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn encode_ignores_routing_rank() {
        let mut request = request();
        let routing = request.routing.as_mut().unwrap();
        routing.dp_rank = Some(5);
        routing.prefill_dp_rank = Some(3);
        assert_eq!(
            data_parallel_rank(&request, DisaggregationMode::Encode),
            None
        );
    }
}
