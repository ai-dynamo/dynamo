// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine as _;
use dynamo_backend_common::{
    BackendError, ErrorType, MultimodalData, MultimodalDataMap, PreprocessedRequest, RoutingHints,
    preprocessed_mm_cache_identifier,
};

use super::request;
use crate::proto as pb;
use crate::request::build_generate_request;
use crate::wire::{
    GenerateEvent, json_to_prost_struct, prost_struct_to_json, validate_generate_response,
    validate_generate_response_with_routed_start,
};

fn request_with_media(map: MultimodalDataMap) -> PreprocessedRequest {
    let mut req = request(Some(16));
    req.multi_modal_data = Some(map);
    req
}

fn url_media(value: &str) -> MultimodalData {
    serde_json::from_value(serde_json::json!({ "Url": value })).expect("parse MultimodalData::Url")
}

fn decoded_media() -> MultimodalData {
    serde_json::from_value(serde_json::json!({
        "Decoded": {
            "nixl_metadata": "",
            "nixl_descriptor": { "addr": 0, "size": 0, "mem_type": "Dram", "device_id": 0 },
            "shape": [1, 1, 1],
            "dtype": "UINT8",
            "metadata": null,
        }
    }))
    .expect("synthesize MultimodalData::Decoded")
}

#[test]
fn canonical_preprocessed_mm_identity_matches_native_grpc_vector() {
    assert_eq!(
        preprocessed_mm_cache_identifier("image", &[0x80]),
        "grpc-mm:835d2213e413e78e540c88905d36dfa708aa0eb02e9d546026a832c6f5ac5825"
    );
}

#[test]
fn routed_experts_tensor_becomes_prime_compact_engine_data() {
    let response = pb::GenerateResponse {
        outputs: Some(pb::SequenceOutput {
            token_ids: vec![10],
            routed_experts: Some(pb::RoutedExpertsTensor {
                // Python/NumPy's canonical dtype string for uint8. The sidecar also
                // accepts the semantic alias `uint8` used by hand-authored clients.
                dtype: "|u1".to_string(),
                shape: vec![1, 2, 2],
                data: vec![1, 2, 3, 4],
            }),
            ..Default::default()
        }),
        ..Default::default()
    };

    let validated = validate_generate_response_with_routed_start(response, false, 7).unwrap();
    let GenerateEvent::Token {
        routed_experts: Some(encoded),
        ..
    } = &validated.events[0]
    else {
        panic!("expected routed-experts token event");
    };
    assert_eq!(encoded["shape"], serde_json::json!([1, 2, 2]));
    assert_eq!(encoded["start"], 7);
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded["data"].as_str().unwrap())
        .unwrap();
    assert_eq!(data, [1, 2, 3, 4]);
}

#[test]
fn build_media_passes_through_http_url_as_url_source() {
    let map = MultimodalDataMap::from([(
        "image_url".to_string(),
        vec![url_media("http://example.com/cat.png")],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-1", false).unwrap();

    assert_eq!(req.media.len(), 1);
    let item = &req.media[0];
    assert_eq!(item.modality, pb::Modality::Image as i32);
    assert_eq!(
        item.source,
        Some(pb::media_item::Source::Url(
            "http://example.com/cat.png".to_string()
        ))
    );
}

#[test]
fn build_media_passes_through_data_uri_as_data_uri_source() {
    let data_uri = "data:image/png;base64,AAAA";
    let map = MultimodalDataMap::from([(
        "image_url".to_string(),
        vec![MultimodalData::RawUrl(data_uri.to_string())],
    )]);
    let req = build_generate_request(&request_with_media(map), "req-2", false).unwrap();

    assert_eq!(req.media.len(), 1);
    assert_eq!(
        req.media[0].source,
        Some(pb::media_item::Source::DataUri(data_uri.to_string()))
    );
}

#[test]
fn build_media_maps_modality_from_map_key() {
    let map = MultimodalDataMap::from([
        (
            "image_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/a.png".to_string())],
        ),
        (
            "video_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/b.mp4".to_string())],
        ),
        (
            "audio_url".to_string(),
            vec![MultimodalData::RawUrl("http://h/c.wav".to_string())],
        ),
    ]);
    let req = build_generate_request(&request_with_media(map), "req-3", false).unwrap();

    let modalities: Vec<i32> = req.media.iter().map(|media| media.modality).collect();
    assert_eq!(
        modalities,
        vec![
            pb::Modality::Image as i32,
            pb::Modality::Video as i32,
            pb::Modality::Audio as i32,
        ]
    );
}

#[test]
fn build_media_is_empty_for_text_only_request() {
    let req = build_generate_request(&request(Some(8)), "req-4", false).unwrap();
    assert!(req.media.is_empty());
}

#[test]
fn build_media_rejects_decoded_rdma_descriptor() {
    let map = MultimodalDataMap::from([("image_url".to_string(), vec![decoded_media()])]);
    let err = build_generate_request(&request_with_media(map), "req-5", false).unwrap_err();

    assert_eq!(
        err.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(err.message().contains("media_decoder"));
}

#[test]
fn disagg_json_round_trips_through_protobuf_struct() {
    let original = serde_json::json!({
        "remote_engine_id": "engine-7",
        "remote_port": 20097,
        "tp_size": 1,
        "do_remote_prefill": true,
        "remote_block_ids": [4, 5, 6],
    });
    let encoded = json_to_prost_struct(&original).unwrap();
    assert_eq!(prost_struct_to_json(&encoded), original);
}

#[test]
fn generate_response_validation_fails_closed() {
    let invalid = [
        validate_generate_response(pb::GenerateResponse::default(), false),
        validate_generate_response(
            pb::GenerateResponse {
                outputs: Some(pb::SequenceOutput {
                    index: 1,
                    token_ids: vec![1],
                    ..Default::default()
                }),
                ..Default::default()
            },
            false,
        ),
        validate_generate_response(
            pb::GenerateResponse {
                outputs: Some(pb::SequenceOutput {
                    finish_info: Some(pb::FinishInfo {
                        finish_reason: pb::finish_info::FinishReason::NotFinished as i32,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            false,
        ),
        validate_generate_response(
            pb::GenerateResponse {
                outputs: Some(pb::SequenceOutput {
                    finish_info: Some(pb::FinishInfo {
                        finish_reason: pb::finish_info::FinishReason::Stop as i32,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            true,
        ),
        validate_generate_response(
            pb::GenerateResponse {
                outputs: Some(pb::SequenceOutput {
                    finish_info: Some(pb::FinishInfo {
                        finish_reason: 999,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            false,
        ),
    ];

    for result in invalid {
        let error = result.err().expect("malformed response must fail");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
    }
}

#[test]
fn generate_response_reconstructs_positionally_aligned_top_logprobs() {
    let response = validate_generate_response(
        pb::GenerateResponse {
            outputs: Some(pb::SequenceOutput {
                token_ids: vec![11, 22],
                logprobs: vec![-0.1, -0.2],
                ranks: vec![1, 3],
                candidate_tokens: vec![
                    pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 12,
                            logprob: -0.3,
                            rank: 2,
                        }],
                    },
                    pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 23,
                            logprob: -0.4,
                            rank: 1,
                        }],
                    },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        false,
    )
    .expect("aligned logprob response");

    let GenerateEvent::Token {
        token_ids,
        logprobs,
        top_logprobs,
        ..
    } = &response.events[0]
    else {
        panic!("expected token event");
    };
    assert_eq!(token_ids, &[11, 22]);
    assert_eq!(
        logprobs.as_deref(),
        Some(&[f64::from(-0.1_f32), f64::from(-0.2_f32)][..])
    );

    let positions = top_logprobs.as_ref().expect("top logprobs");
    assert_eq!(positions.len(), 2);
    assert_eq!(positions[0][0].token_id, 11);
    assert_eq!(positions[0][0].rank, 1);
    assert_eq!(positions[0][1].token_id, 12);
    assert_eq!(positions[0][1].rank, 2);
    assert_eq!(positions[1][0].token_id, 22);
    assert_eq!(positions[1][0].rank, 3);
    assert_eq!(positions[1][1].token_id, 23);
    assert_eq!(positions[1][1].rank, 1);
}

#[test]
fn generate_response_reconstructs_prompt_logprobs_with_candidates_and_ranks() {
    let response = validate_generate_response(
        pb::GenerateResponse {
            prompt_info: Some(pb::PromptInfo {
                num_prompt_tokens: 3,
                token_ids: vec![10, 11, 12],
                logprobs: vec![0.0, -0.1, -0.2],
                ranks: vec![0, 1, 3],
                candidate_tokens: vec![
                    Default::default(),
                    pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 21,
                            logprob: -0.3,
                            rank: 2,
                        }],
                    },
                    pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 22,
                            logprob: -0.4,
                            rank: 1,
                        }],
                    },
                ],
            }),
            ..Default::default()
        },
        false,
    )
    .expect("aligned prompt logprob response");

    let GenerateEvent::PromptLogprobs(prompt_logprobs) = &response.events[0] else {
        panic!("expected prompt-logprobs event");
    };
    assert_eq!(
        prompt_logprobs,
        &serde_json::json!([
            null,
            {
                "11": {"logprob": f64::from(-0.1_f32), "rank": 1},
                "21": {"logprob": f64::from(-0.3_f32), "rank": 2}
            },
            {
                "12": {"logprob": f64::from(-0.2_f32), "rank": 3},
                "22": {"logprob": f64::from(-0.4_f32), "rank": 1}
            }
        ])
    );
}

#[test]
fn generate_response_clamps_non_finite_prompt_logprobs() {
    let response = validate_generate_response(
        pb::GenerateResponse {
            prompt_info: Some(pb::PromptInfo {
                num_prompt_tokens: 2,
                token_ids: vec![10, 11],
                logprobs: vec![0.0, f32::NEG_INFINITY],
                ranks: vec![0, 1],
                candidate_tokens: vec![
                    Default::default(),
                    pb::CandidateTokenInfo {
                        tokens: vec![pb::candidate_token_info::TokenInfo {
                            id: 21,
                            logprob: f32::NEG_INFINITY,
                            rank: 2,
                        }],
                    },
                ],
            }),
            ..Default::default()
        },
        false,
    )
    .expect("non-finite prompt logprobs use Dynamo's finite sentinel");

    let GenerateEvent::PromptLogprobs(prompt_logprobs) = &response.events[0] else {
        panic!("expected prompt-logprobs event");
    };
    assert_eq!(
        prompt_logprobs,
        &serde_json::json!([
            null,
            {
                "11": {"logprob": -1e30, "rank": 1},
                "21": {"logprob": -1e30, "rank": 2}
            }
        ])
    );
}

#[test]
fn generate_response_rejects_nan_and_positive_infinite_logprobs() {
    for invalid in [f32::NAN, f32::INFINITY] {
        let prompt = pb::GenerateResponse {
            prompt_info: Some(pb::PromptInfo {
                num_prompt_tokens: 2,
                token_ids: vec![10, 11],
                logprobs: vec![0.0, invalid],
                ranks: vec![0, 1],
                candidate_tokens: vec![Default::default(), Default::default()],
            }),
            ..Default::default()
        };
        assert!(validate_generate_response(prompt, false).is_err());

        let output = pb::GenerateResponse {
            outputs: Some(pb::SequenceOutput {
                token_ids: vec![11],
                logprobs: vec![-0.1],
                ranks: vec![1],
                candidate_tokens: vec![pb::CandidateTokenInfo {
                    tokens: vec![pb::candidate_token_info::TokenInfo {
                        id: 21,
                        logprob: invalid,
                        rank: 2,
                    }],
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_generate_response(output, false).is_err());
    }
}

#[test]
fn generate_response_rejects_duplicate_output_logprob_token_ids() {
    for candidate_id in [11, 21] {
        let duplicate_candidates = if candidate_id == 11 {
            vec![candidate_id]
        } else {
            vec![candidate_id, candidate_id]
        };
        let response = pb::GenerateResponse {
            outputs: Some(pb::SequenceOutput {
                token_ids: vec![11],
                logprobs: vec![-0.1],
                ranks: vec![1],
                candidate_tokens: vec![pb::CandidateTokenInfo {
                    tokens: duplicate_candidates
                        .into_iter()
                        .map(|id| pb::candidate_token_info::TokenInfo {
                            id,
                            logprob: -0.2,
                            rank: 2,
                        })
                        .collect(),
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_generate_response(response, false).is_err());
    }
}

#[test]
fn generate_response_rejects_misaligned_prompt_logprob_metadata() {
    for prompt_info in [
        pb::PromptInfo {
            num_prompt_tokens: 2,
            token_ids: vec![10],
            logprobs: vec![0.0, -0.1],
            ranks: vec![0, 1],
            candidate_tokens: vec![Default::default(), Default::default()],
        },
        pb::PromptInfo {
            num_prompt_tokens: 2,
            token_ids: vec![10, 11],
            logprobs: vec![0.0],
            ranks: vec![0, 1],
            candidate_tokens: vec![Default::default(), Default::default()],
        },
        pb::PromptInfo {
            num_prompt_tokens: 2,
            token_ids: vec![10, 11],
            logprobs: vec![0.0, -0.1],
            ranks: vec![0],
            candidate_tokens: vec![Default::default(), Default::default()],
        },
        pb::PromptInfo {
            num_prompt_tokens: 2,
            token_ids: vec![10, 11],
            logprobs: vec![0.0, -0.1],
            ranks: vec![0, 1],
            candidate_tokens: vec![Default::default()],
        },
    ] {
        let error = validate_generate_response(
            pb::GenerateResponse {
                prompt_info: Some(prompt_info),
                ..Default::default()
            },
            false,
        )
        .err()
        .expect("misaligned prompt metadata must fail closed");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
    }
}

#[test]
fn generate_response_rejects_misaligned_logprob_metadata() {
    for output in [
        pb::SequenceOutput {
            token_ids: vec![11, 22],
            logprobs: vec![-0.1, -0.2],
            ranks: vec![1],
            candidate_tokens: vec![Default::default(), Default::default()],
            ..Default::default()
        },
        pb::SequenceOutput {
            token_ids: vec![11, 22],
            logprobs: vec![-0.1, -0.2],
            ranks: vec![1, 2],
            candidate_tokens: vec![Default::default()],
            ..Default::default()
        },
        pb::SequenceOutput {
            token_ids: vec![11, 22],
            logprobs: vec![],
            ranks: vec![1, 2],
            candidate_tokens: vec![Default::default(), Default::default()],
            ..Default::default()
        },
    ] {
        let result = validate_generate_response(
            pb::GenerateResponse {
                outputs: Some(output),
                ..Default::default()
            },
            false,
        );
        let error = result.err().expect("misaligned response must fail closed");
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
    }
}

#[test]
fn build_request_forwards_hidden_stops_and_priority() {
    let mut request = request(Some(8));
    request.stop_conditions.stop_token_ids_hidden = Some(vec![17, 19]);
    request.routing = Some(RoutingHints {
        priority: Some(-3),
        ..Default::default()
    });

    let wire = build_generate_request(&request, "req-priority", false).unwrap();
    assert_eq!(wire.priority, -3);
    assert_eq!(wire.stopping.unwrap().stop_token_ids, vec![17, 19]);
}

#[test]
fn build_request_preserves_prime_tito_sampling_logprobs_and_cache_salt() {
    let mut request = request(Some(8));
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {
            "sampling_params": {
                "max_tokens": 17,
                "min_tokens": 2,
                "temperature": 0.7,
                "top_p": 0.91,
                "top_k": 23,
                "min_p": 0.05,
                "seed": 42,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.3,
                "repetition_penalty": 1.1,
                "stop": ["done"],
                "stop_token_ids": [99],
                "ignore_eos": true,
                "logprobs": 1,
                "skip_special_tokens": false,
                "routed_experts_prompt_start": 1
            },
            "cache_salt": "rollout-7",
            "priority": -7
        }
    }));

    let wire = build_generate_request(&request, "req-tito", false).unwrap();
    assert_eq!(wire.routed_experts_prompt_start, 1);
    let sampling = wire.sampling.unwrap();
    let decoding = wire.decoding.unwrap();
    let stopping = wire.stopping.unwrap();
    let response = wire.response.unwrap();
    assert_eq!(wire.temperature, Some(0.7));
    assert_eq!(sampling.top_k, Some(23));
    assert_eq!(sampling.top_p, Some(0.91));
    assert_eq!(sampling.min_p, Some(0.05));
    assert_eq!(sampling.seed, Some(42));
    assert_eq!(decoding.presence_penalty, Some(0.2));
    assert_eq!(decoding.frequency_penalty, Some(0.3));
    assert_eq!(decoding.repetition_penalty, Some(1.1));
    assert_eq!(stopping.max_new_tokens, 17);
    assert_eq!(stopping.min_new_tokens, 2);
    assert_eq!(stopping.stop_strings, vec!["done"]);
    assert_eq!(stopping.stop_token_ids, vec![99]);
    assert!(stopping.ignore_eos);
    assert!(response.output_logprobs);
    assert!(response.output_candidates.is_some());
    assert_eq!(wire.kv.unwrap().cache_salt, "dynamo-cache-salt:rollout-7");
    assert_eq!(wire.priority, -7);
}

#[test]
fn build_request_maps_extended_sampling_contract_losslessly() {
    let mut request = request(Some(8));
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {
            "sampling_params": {
                "thinking_token_budget": 64,
                "logit_bias": {"7": -1.25},
                "allowed_token_ids": [7, 8],
                "bad_words": ["blocked"],
                "logprobs": 2,
                "logprob_token_ids": [7, 8],
                "structured_outputs": {
                    "regex": "[a-z]+",
                    "disable_any_whitespace": true,
                    "disable_additional_properties": true,
                    "whitespace_pattern": "\\s*"
                },
                "skip_reading_prefix_cache": true,
                "vllm_xargs": {"exact_integer": 9007199254740993_u64}
            }
        }
    }));

    let wire = build_generate_request(&request, "req-extended", false).unwrap();
    let decoding = wire.decoding.unwrap();
    let stopping = wire.stopping.unwrap();
    let response = wire.response.unwrap();
    assert_eq!(decoding.logit_bias[&7], -1.25);
    assert_eq!(decoding.allowed_token_ids, vec![7, 8]);
    assert_eq!(decoding.bad_words, vec!["blocked"]);
    assert!(decoding.structured_output_disable_any_whitespace);
    assert!(decoding.structured_output_disable_additional_properties);
    assert_eq!(
        decoding.structured_output_whitespace_pattern.as_deref(),
        Some("\\s*")
    );
    assert_eq!(stopping.thinking_token_budget, Some(64));
    assert!(wire.kv.unwrap().bypass_prefix_cache);
    assert_eq!(
        response.output_candidates.and_then(|value| value.select),
        Some(pb::candidate_tokens::Select::TokenIds(pb::TokenIds {
            ids: vec![7, 8]
        }))
    );
    let xargs: serde_json::Value =
        serde_json::from_slice(wire.vllm_xargs_json.as_deref().unwrap()).unwrap();
    assert_eq!(xargs["exact_integer"], 9_007_199_254_740_993_u64);
}

#[test]
fn build_request_forwards_validated_preprocessed_multimodal_features() {
    let mut request = request(Some(8));
    request.token_ids = vec![10, 99, 99, 20];
    let encoded = base64::engine::general_purpose::STANDARD.encode([0x81, 0xa1, b'x', 0x01]);
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-hash"]},
                "mm_placeholders": {
                    "image": [{"offset": 1, "length": 2, "is_embed": [true, false]}]
                },
                "kwargs_data": {"image": [encoded]}
            }
        }
    }));

    let wire = build_generate_request(&request, "req-mm", false).unwrap();
    assert!(wire.media.is_empty());
    assert_eq!(wire.mm_features.len(), 1);
    let feature = &wire.mm_features[0];
    assert_eq!(feature.modality, "image");
    let identifier = preprocessed_mm_cache_identifier("image", &[0x81, 0xa1, b'x', 0x01]);
    assert_eq!(feature.mm_hash, identifier);
    assert_eq!(feature.cache_identifier, feature.mm_hash);
    assert_eq!(
        feature.kwargs_msgpack.as_deref(),
        Some(&[0x81, 0xa1, b'x', 0x01][..])
    );
    assert_eq!(
        feature.position.as_ref().unwrap().is_embed,
        vec![true, false]
    );
}

#[test]
fn build_request_rejects_preprocessed_multimodal_lora_without_tower_contract() {
    let mut request = request(Some(8));
    request.token_ids = vec![10, 99, 20];
    request.routing = Some(RoutingHints {
        lora_name: Some("adapter-a".to_string()),
        ..Default::default()
    });
    let encoded = base64::engine::general_purpose::STANDARD.encode([0x81, 0xa1, b'x', 0x01]);
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["routing-hash"]},
                "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
                "kwargs_data": {"image": [encoded]}
            }
        }
    }));

    let error = build_generate_request(&request, "req-mm-lora", false).unwrap_err();
    assert!(error.message().contains("tower-LoRA"));
}

#[test]
fn build_request_rejects_unverified_multimodal_cache_hit() {
    let mut request = request(Some(8));
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {
            "sampling_params": {},
            "features": {
                "mm_hashes": {"image": ["image-hash"]},
                "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
                "kwargs_data": null
            }
        }
    }));

    let error = build_generate_request(&request, "req-mm-cache", false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(error.message().contains("cache hits"));
}

#[test]
fn build_request_forwards_prompt_logprob_selectors() {
    let mut input = request(Some(8));
    input.output_options.prompt_logprobs = Some(3);
    let response = build_generate_request(&input, "req-prompt", false)
        .expect("prompt logprobs request")
        .response
        .expect("response options");
    assert!(response.prompt_token_ids);
    assert!(response.prompt_logprobs);
    assert_eq!(
        response.prompt_candidates.and_then(|value| value.select),
        Some(pb::candidate_tokens::Select::TopN(3))
    );

    let mut input = request(Some(8));
    input.extra_args = Some(serde_json::json!({
        "vllm_tito": {"sampling_params": {"prompt_logprobs": -1}}
    }));
    let response = build_generate_request(&input, "req-prompt-all", false)
        .expect("all prompt logprobs request")
        .response
        .expect("response options");
    assert_eq!(
        response.prompt_candidates.and_then(|value| value.select),
        Some(pb::candidate_tokens::Select::All(true))
    );
}

#[test]
fn build_request_accepts_prime_cache_salt_in_sampling_params() {
    let mut request = request(Some(8));
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {
            "sampling_params": {
                "cache_salt": "rollout-sampling-9"
            }
        }
    }));

    let wire = build_generate_request(&request, "req-tito-sampling-salt", false).unwrap();
    assert_eq!(
        wire.kv.unwrap().cache_salt,
        "dynamo-cache-salt:rollout-sampling-9"
    );
}

#[test]
fn build_request_rejects_unsupported_tito_sampling_fields() {
    let mut request = request(Some(8));
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {"sampling_params": {"beam_width": 4}}
    }));
    let error = build_generate_request(&request, "req-unsupported", false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
    assert!(error.message().contains("beam_width"));
}

#[test]
fn build_request_accepts_prime_return_token_ids_hint() {
    let mut request = request(Some(8));
    request.extra_args = Some(serde_json::json!({
        "vllm_tito": {"sampling_params": {"return_token_ids": true}}
    }));

    let wire = build_generate_request(&request, "req-return-token-ids", false).unwrap();
    assert!(wire.response.unwrap().output_token_ids);
}

#[test]
fn prefill_clamps_min_and_max_tokens_to_one() {
    let mut request = request(Some(8));
    request.stop_conditions.min_tokens = Some(7);

    let wire = build_generate_request(&request, "req-prefill", true).unwrap();
    let stopping = wire.stopping.unwrap();
    assert_eq!(stopping.max_new_tokens, 1);
    assert_eq!(stopping.min_new_tokens, 1);
}

#[test]
fn build_request_rejects_visible_stop_ids() {
    let mut request = request(Some(8));
    request.stop_conditions.stop_token_ids_visible = Some(vec![17]);

    let error = build_generate_request(&request, "req-visible-stop", false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}

#[test]
fn build_request_rejects_prompt_embeddings() {
    let mut request = request(Some(8));
    request.prompt_embeds = Some("opaque-embedding-handle".to_string());

    let error = build_generate_request(&request, "req-embeds", false).unwrap_err();
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(BackendError::InvalidArgument)
    );
}
