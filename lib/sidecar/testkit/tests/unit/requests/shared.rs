// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::Backend;
use crate::unit_fixtures::minimal_request;
use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    BackendError, DisaggregationMode, DynamoError, ErrorType, GuidedDecodingOptions, PrefillResult,
    PreprocessedRequest, SamplingOptions, StopConditions,
};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) enum Feature {
    SkipSpecialTokens,
    Priority,
    CacheIdentity,
    CacheBypass,
    JsonGuide,
    RegexGuide,
    GrammarGuide,
    ChoiceGuide,
    StructuralTagGuide,
    GuideBackend,
    GuideWhitespace,
}

pub(super) enum Support {
    Supported,
    Rejected(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum InvalidRequest {
    EmptyTokens,
    MultipleSequences,
    PromptEmbeddings,
    BestOf,
    BeamSearch,
    LengthPenalty,
    TopK,
    VisibleStop,
    ThinkingTokens,
    MultimodalProcessor,
    OrphanMediaIds,
}

#[derive(Debug, PartialEq)]
pub(super) enum Guide {
    Json(String),
    Regex(String),
    Grammar(String),
    Choice(Vec<String>),
    StructuralTag(String),
}

#[derive(Debug)]
pub(super) struct RequestObservation {
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub min_tokens: Option<u32>,
    pub stop_strings: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: Option<bool>,
    pub output_logprobs: bool,
    pub prompt_logprobs: bool,
    pub skip_special_tokens: Option<bool>,
    pub priority: Option<i32>,
    pub lora_name: Option<String>,
    pub cache_identity: Option<String>,
    pub bypass_prefix_cache: Option<bool>,
    pub guide: Option<Guide>,
    pub guide_backend: Option<String>,
    pub guide_whitespace: Option<String>,
    pub handoff: Option<Value>,
}

pub(super) trait RequestAdapter {
    const PREFILL_MIN_TOKENS: Option<u32>;
    const TOP_K_EXPECTED: [Option<i64>; 5];
    const PRIORITY_EXPECTED: [i32; 3];
    const STOP_TOKEN_IDS: &'static [u32];
    const CACHE_IDENTITY: &'static str;

    fn lower(
        request: PreprocessedRequest,
        request_id: &str,
        mode: DisaggregationMode,
    ) -> Result<RequestObservation, DynamoError>;
    fn support(feature: Feature) -> Support;
    fn invalid_message(case: InvalidRequest) -> &'static str;
    fn rank(request: &PreprocessedRequest, mode: DisaggregationMode) -> Option<u32>;
    fn handoff_fixture() -> (Value, Value);
}

fn assert_invalid(error: DynamoError) {
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}

fn check_feature(
    feature: Feature,
    result: Result<RequestObservation, DynamoError>,
    check: impl FnOnce(RequestObservation),
) {
    match Backend::support(feature) {
        Support::Supported => check(result.expect("supported control must lower")),
        Support::Rejected(reason) => {
            assert!(!reason.is_empty(), "unsupported features need a reason");
            assert_invalid(result.expect_err(reason));
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn oversized_logprob_counts_are_rejected() {
        for prompt in [false, true] {
            let mut request = minimal_request();
            let count = i32::MAX as u32 + 1;
            if prompt {
                request.output_options.prompt_logprobs = Some(count);
            } else {
                request.output_options.logprobs = Some(count);
            }
            let error = Backend::lower(request, "logprobs", DisaggregationMode::Aggregated)
                .expect_err("oversized logprob count");
            assert!(error.to_string().contains("fit in i32"));
            assert_invalid(error);
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn special_token_policy_is_forwarded_or_explicitly_rejected() {
        let mut request = minimal_request();
        request.output_options.skip_special_tokens = Some(false);
        check_feature(
            Feature::SkipSpecialTokens,
            Backend::lower(request, "special-tokens", DisaggregationMode::Aggregated),
            |observed| assert_eq!(observed.skip_special_tokens, Some(false)),
        );
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn canonical_priority_preserves_native_ordering_or_is_rejected() {
        for (priority, expected) in [-7, 7, i32::MIN]
            .into_iter()
            .zip(Backend::PRIORITY_EXPECTED)
        {
            let mut request = minimal_request();
            request.routing = Some(RoutingHints {
                priority: Some(priority),
                ..Default::default()
            });
            check_feature(
                Feature::Priority,
                Backend::lower(request, "priority", DisaggregationMode::Aggregated),
                |observed| assert_eq!(observed.priority, Some(expected)),
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn prefill_limits_generation() {
        let mut request = minimal_request();
        request.stop_conditions.max_tokens = Some(100);
        request.stop_conditions.min_tokens = Some(4);
        let observed = Backend::lower(request, "prefill", DisaggregationMode::Prefill).unwrap();
        assert_eq!(observed.max_tokens, Some(1));
        assert_eq!(observed.min_tokens, Backend::PREFILL_MIN_TOKENS);
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn absent_and_explicit_zero_options_remain_distinct() {
        for explicit in [false, true] {
            let mut request = minimal_request();
            request.sampling_options.temperature = explicit.then_some(0.0);
            request.output_options.logprobs = explicit.then_some(0);
            request.output_options.prompt_logprobs = explicit.then_some(0);
            let observed = Backend::lower(request, "zero", DisaggregationMode::Aggregated).unwrap();
            assert_eq!(observed.temperature, explicit.then_some(0.0));
            assert_eq!(observed.output_logprobs, explicit);
            assert_eq!(observed.prompt_logprobs, explicit);
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn invalid_canonical_controls_are_rejected_before_submission() {
        type Mutation = fn(&mut PreprocessedRequest);
        let cases: &[(InvalidRequest, Mutation)] = &[
            (InvalidRequest::EmptyTokens, |r| {
                r.token_ids = Arc::new(Vec::new())
            }),
            (InvalidRequest::MultipleSequences, |r| {
                r.sampling_options.n = Some(2)
            }),
            (InvalidRequest::PromptEmbeddings, |r| {
                r.prompt_embeds = Some("encoded".into())
            }),
            (InvalidRequest::BestOf, |r| {
                r.sampling_options.best_of = Some(2)
            }),
            (InvalidRequest::BeamSearch, |r| {
                r.sampling_options.use_beam_search = Some(true)
            }),
            (InvalidRequest::LengthPenalty, |r| {
                r.sampling_options.length_penalty = Some(0.5)
            }),
            (InvalidRequest::TopK, |r| {
                r.sampling_options.top_k = Some(-2)
            }),
            (InvalidRequest::VisibleStop, |r| {
                r.stop_conditions.stop_token_ids_visible = Some(vec![42])
            }),
            (InvalidRequest::ThinkingTokens, |r| {
                r.stop_conditions.max_thinking_tokens = Some(5)
            }),
            (InvalidRequest::MultimodalProcessor, |r| {
                r.mm_processor_kwargs = Some(json!({}))
            }),
            (InvalidRequest::OrphanMediaIds, |r| {
                r.multi_modal_uuids = Some(std::collections::HashMap::from([(
                    "image_url".into(),
                    vec![Some("image-a".into())],
                )]))
            }),
        ];
        Backend::lower(
            minimal_request(),
            "supported",
            DisaggregationMode::Aggregated,
        )
        .unwrap();
        for &(case, mutate) in cases {
            let mut request = minimal_request();
            mutate(&mut request);
            let error = Backend::lower(request, "invalid", DisaggregationMode::Aggregated)
                .expect_err("invalid canonical control");
            let expected = Backend::invalid_message(case);
            assert!(!expected.is_empty());
            assert!(error.to_string().contains(expected), "{case:?}: {error}");
            assert_invalid(error);
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn guide_variants_preserve_type_and_payload_or_are_rejected() {
        for (feature, guide, expected) in [
            (
                Feature::JsonGuide,
                GuidedDecodingOptions {
                    json: Some(
                        json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}),
                    ),
                    ..Default::default()
                },
                Guide::Json(
                    r#"{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}"#
                        .into(),
                ),
            ),
            (
                Feature::RegexGuide,
                GuidedDecodingOptions {
                    regex: Some("[a-z]+".into()),
                    ..Default::default()
                },
                Guide::Regex("[a-z]+".into()),
            ),
            (
                Feature::GrammarGuide,
                GuidedDecodingOptions {
                    grammar: Some("root ::= 'yes'".into()),
                    ..Default::default()
                },
                Guide::Grammar("root ::= 'yes'".into()),
            ),
            (
                Feature::ChoiceGuide,
                GuidedDecodingOptions {
                    choice: Some(vec!["yes".into(), "no".into()]),
                    ..Default::default()
                },
                Guide::Choice(vec!["yes".into(), "no".into()]),
            ),
            (
                Feature::StructuralTagGuide,
                GuidedDecodingOptions {
                    structural_tag: Some(json!("<answer>")),
                    ..Default::default()
                },
                Guide::StructuralTag("<answer>".into()),
            ),
            (
                Feature::StructuralTagGuide,
                GuidedDecodingOptions {
                    structural_tag: Some(json!({"tag": "answer"})),
                    ..Default::default()
                },
                Guide::StructuralTag(r#"{"tag":"answer"}"#.into()),
            ),
        ] {
            let mut request = minimal_request();
            request.sampling_options.guided_decoding = Some(guide);
            check_feature(
                feature,
                Backend::lower(request, "guide", DisaggregationMode::Aggregated),
                |observed| assert_eq!(observed.guide, Some(expected)),
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn guide_modifiers_are_preserved_or_explicitly_rejected() {
        for backend in [true, false] {
            let mut request = minimal_request();
            request.sampling_options.guided_decoding = Some(if backend {
                GuidedDecodingOptions {
                    backend: Some("xgrammar".into()),
                    ..Default::default()
                }
            } else {
                GuidedDecodingOptions {
                    whitespace_pattern: Some(" *".into()),
                    ..Default::default()
                }
            });
            check_feature(
                if backend {
                    Feature::GuideBackend
                } else {
                    Feature::GuideWhitespace
                },
                Backend::lower(request, "guide-modifier", DisaggregationMode::Aggregated),
                |observed| {
                    if backend {
                        assert_eq!(observed.guide_backend.as_deref(), Some("xgrammar"));
                    } else {
                        assert_eq!(observed.guide_whitespace.as_deref(), Some(" *"));
                    }
                },
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn conflicting_guides_are_rejected() {
        let mut request = minimal_request();
        request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
            json: Some(json!({})),
            regex: Some(".*".into()),
            ..Default::default()
        });
        assert_invalid(
            Backend::lower(
                request,
                "conflicting-guides",
                DisaggregationMode::Aggregated,
            )
            .unwrap_err(),
        );
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn stopping_tokens_are_merged_and_deduplicated() {
        let mut request = minimal_request();
        request.stop_conditions.stop_token_ids = Some(vec![3, 2, 3]);
        request.stop_conditions.stop_token_ids_hidden = Some(vec![2, 4]);
        let observed = Backend::lower(request, "stops", DisaggregationMode::Aggregated).unwrap();
        assert_eq!(observed.stop_token_ids, Backend::STOP_TOKEN_IDS);
        let mut tokens = observed.stop_token_ids;
        tokens.sort_unstable();
        assert_eq!(tokens, [2, 3, 4]);
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn top_k_preserves_native_sentinels_and_explicit_limits() {
        for (top_k, expected) in [None, Some(-1), Some(0), Some(7), Some(i32::MAX)]
            .into_iter()
            .zip(Backend::TOP_K_EXPECTED)
        {
            let mut request = minimal_request();
            request.sampling_options.top_k = top_k;
            let observed = Backend::lower(request, "top-k", DisaggregationMode::Aggregated).unwrap();
            assert_eq!(observed.top_k, expected);
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn selected_lora_adapter_is_forwarded() {
        let mut request = minimal_request();
        request.routing = Some(RoutingHints {
            lora_name: Some("adapter-a".into()),
            ..Default::default()
        });
        let observed = Backend::lower(request, "adapter", DisaggregationMode::Aggregated).unwrap();
        assert_eq!(observed.lora_name.as_deref(), Some("adapter-a"));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn prefill_rank_overrides_decode_rank_with_fallback() {
        let mut request = minimal_request();
        request.routing = Some(RoutingHints {
            dp_rank: Some(5),
            prefill_dp_rank: Some(3),
            ..Default::default()
        });
        for (mode, expected) in [
            (DisaggregationMode::Aggregated, Some(5)),
            (DisaggregationMode::Decode, Some(5)),
            (DisaggregationMode::Prefill, Some(3)),
        ] {
            assert_eq!(Backend::rank(&request, mode), expected);
        }
        request.routing.as_mut().unwrap().prefill_dp_rank = None;
        assert_eq!(
            Backend::rank(&request, DisaggregationMode::Prefill),
            Some(5)
        );
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn canonical_cache_controls_are_preserved_or_rejected() {
        let mut identity = minimal_request();
        identity.routing = Some(RoutingHints {
            cache_namespace: Some("cache-salt".into()),
            ..Default::default()
        });
        check_feature(
            Feature::CacheIdentity,
            Backend::lower(identity, "cache-identity", DisaggregationMode::Aggregated),
            |observed| {
                assert_eq!(
                    observed.cache_identity.as_deref(),
                    Some(Backend::CACHE_IDENTITY)
                )
            },
        );
        for bypass in [false, true] {
            let mut request = minimal_request();
            request.extra_args = Some(json!({"bypass_prefix_cache": bypass}));
            check_feature(
                Feature::CacheBypass,
                Backend::lower(request, "cache-bypass", DisaggregationMode::Aggregated),
                |observed| assert_eq!(observed.bypass_prefix_cache, Some(bypass)),
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn decode_requires_and_preserves_valid_handoff_metadata() {
        let (input, expected) = Backend::handoff_fixture();
        let mut request = minimal_request();
        request.prefill_result = Some(PrefillResult {
            disaggregated_params: input,
            prompt_tokens_details: None,
        });
        let observed = Backend::lower(request, "decode", DisaggregationMode::Decode).unwrap();
        assert_eq!(observed.handoff, Some(expected));
        for value in [None, Some(json!([])), Some(json!("invalid"))] {
            let mut request = minimal_request();
            request.prefill_result = value.map(|disaggregated_params| PrefillResult {
                disaggregated_params,
                prompt_tokens_details: None,
            });
            assert_invalid(
                Backend::lower(request, "bad-handoff", DisaggregationMode::Decode).unwrap_err(),
            );
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn canonical_sampling_and_stopping_fields_are_preserved() {
        let mut request = minimal_request();
        request.sampling_options = SamplingOptions {
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: Some(4),
            min_p: Some(0.1),
            presence_penalty: Some(0.3),
            frequency_penalty: Some(0.4),
            repetition_penalty: Some(1.1),
            ..Default::default()
        };
        request.stop_conditions = StopConditions {
            max_tokens: Some(1),
            min_tokens: Some(1),
            stop: Some(vec!["done".into()]),
            ignore_eos: Some(true),
            ..Default::default()
        };
        let observed =
            Backend::lower(request, "canonical-fields", DisaggregationMode::Aggregated).unwrap();
        assert_eq!(observed.request_id, "canonical-fields");
        assert_eq!(observed.token_ids, [11, 22, 33]);
        assert_eq!(observed.temperature, Some(0.2));
        assert_eq!(
            (observed.top_k, observed.top_p, observed.min_p),
            (Some(4), Some(0.9), Some(0.1))
        );
        assert_eq!(
            (
                observed.presence_penalty,
                observed.frequency_penalty,
                observed.repetition_penalty
            ),
            (Some(0.3), Some(0.4), Some(1.1))
        );
        assert_eq!(
            (observed.max_tokens, observed.min_tokens),
            (Some(1), Some(1))
        );
        assert_eq!(observed.stop_strings, ["done"]);
        assert_eq!(observed.ignore_eos, Some(true));
    }
}
