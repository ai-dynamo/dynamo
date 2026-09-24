// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::proto as pb;
use dynamo_backend_common::engine::RoutingHints;
use dynamo_backend_common::{
    MultimodalData, OutputOptions, PrefillResult, PreprocessedRequest, SamplingOptions,
    StopConditions,
};
use prost_types_v14 as prost_types;
use serde_json::json;

pub(crate) fn model_info() -> pb::ModelInfo {
    pb::ModelInfo {
        model_id: "model-source".to_string(),
        served_model_name: "served-model".to_string(),
        served_model_aliases: vec!["model-alias".to_string()],
        supports_text_input: true,
        supports_token_ids_input: true,
        supports_lora: true,
        supports_multimodal: false,
        reasoning_parser: "deepseek_r1".to_string(),
        tool_call_parser: "hermes".to_string(),
    }
}

pub(crate) fn server_info() -> pb::ServerInfo {
    pb::ServerInfo {
        engine_version: "test-vllm".to_string(),
        api_version: "vllm".to_string(),
        instance_id: "test-instance".to_string(),
        parallelism: Some(pb::ParallelismInfo {
            tensor_parallel_size: 2,
            pipeline_parallel_size: 1,
            data_parallel_size: 2,
            data_parallel_rank: 0,
            decode_context_parallel_size: 1,
            world_size: 2,
            data_parallel_size_local: 0,
        }),
        max_model_len: 8192,
        kv_block_size: 16,
        total_kv_blocks: 4096,
        max_running_requests: 128,
        max_batched_tokens: 2048,
        max_loras: 4,
        effective_attention_block_size: None,
        rl_capabilities: Some(pb::RlCapabilities {
            weight_transfer_enabled: true,
            weight_transfer_backend: "nccl".to_string(),
            sleep_mode_enabled: true,
            draft_weight_updates_enabled: true,
        }),
    }
}

pub(crate) fn sequence_response(
    terminal: bool,
    logprobs: bool,
    kv_transfer_params: Option<prost_types::Struct>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        prompt_info: None,
        outputs: Some(pb::SequenceOutput {
            index: 0,
            text: " token".to_string(),
            num_tokens: 1,
            token_ids: vec![42],
            logprobs: logprobs.then_some(vec![-0.25]).unwrap_or_default(),
            ranks: logprobs.then_some(vec![1]).unwrap_or_default(),
            candidate_tokens: logprobs
                .then_some(vec![pb::CandidateTokenInfo {
                    tokens: vec![pb::candidate_token_info::TokenInfo {
                        id: 43,
                        logprob: -0.5,
                        rank: 2,
                    }],
                }])
                .unwrap_or_default(),
            finish_info: terminal.then_some(pb::FinishInfo {
                num_output_tokens: 1,
                finish_reason: pb::finish_info::FinishReason::Stop as i32,
                stop_reason: Some(pb::finish_info::StopReason::StopTokenId(2)),
                kv_transfer_params,
                ec_transfer_params: None,
            }),
        }),
    }
}

pub(crate) fn encoder_handoff() -> serde_json::Value {
    json!({
        "request_id": "encode-0",
        "ec_items": [
            {"key": "image-a", "shape": [1, 729, 2048]},
            {"key": "image-b", "shape": [1, 441, 2048]},
        ],
        "nested": {"flags": [true, null, "opaque"]},
    })
}

pub(crate) fn encode_response(
    ec_transfer_params: Option<prost_types::Struct>,
) -> pb::GenerateResponse {
    pb::GenerateResponse {
        prompt_info: None,
        outputs: Some(pb::SequenceOutput {
            index: 0,
            text: String::new(),
            num_tokens: 0,
            token_ids: Vec::new(),
            logprobs: Vec::new(),
            ranks: Vec::new(),
            candidate_tokens: Vec::new(),
            finish_info: Some(pb::FinishInfo {
                num_output_tokens: 0,
                finish_reason: pb::finish_info::FinishReason::Stop as i32,
                stop_reason: None,
                kv_transfer_params: None,
                ec_transfer_params,
            }),
        }),
    }
}

pub(crate) fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("served-model".to_string())
        .token_ids(vec![11, 22, 33])
        .stop_conditions(StopConditions {
            max_tokens: Some(1),
            min_tokens: Some(1),
            stop: Some(vec!["done".to_string()]),
            stop_token_ids_hidden: Some(vec![2]),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: Some(4),
            min_p: Some(0.1),
            seed: Some(123),
            presence_penalty: Some(0.3),
            frequency_penalty: Some(0.4),
            repetition_penalty: Some(1.1),
            include_stop_str_in_output: Some(true),
            guided_decoding: Some(dynamo_backend_common::GuidedDecodingOptions {
                json: Some(json!({"type": "object"})),
                ..Default::default()
            }),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(1),
            prompt_logprobs: Some(1),
            ..Default::default()
        })
        .mdc_sum(Some("model-checksum".to_string()))
        .routing(Some(RoutingHints {
            cache_namespace: Some("cache-salt".to_string()),
            ..Default::default()
        }))
        .extra_args(Some(json!({
            "nvext": {"cache_salt": "cache-salt", "token_in": true},
            "bypass_prefix_cache": true,
            "kv_transfer_params": {
                "connector_data": {"values": [1, true, null]}
            }
        })))
        .build()
        .expect("request")
}

pub(crate) fn epd_image_request() -> PreprocessedRequest {
    let mut request = request();
    request.output_options.prompt_logprobs = None;
    request.multi_modal_data = Some(std::collections::HashMap::from([(
        "image_url".to_string(),
        vec![
            MultimodalData::RawUrl("data:image/png;base64,aW1hZ2UtYQ==".to_string()),
            MultimodalData::RawUrl("data:image/png;base64,aW1hZ2UtYg==".to_string()),
        ],
    )]));
    request.multi_modal_uuids = Some(std::collections::HashMap::from([(
        "image_url".to_string(),
        vec![Some("image-a".to_string()), Some("image-b".to_string())],
    )]));
    request
}

pub(crate) fn decode_request() -> PreprocessedRequest {
    let mut request = request();
    request.prefill_result = Some(PrefillResult {
        disaggregated_params: json!({
            "do_remote_decode": false,
            "do_remote_prefill": true,
            "remote_engine_id": "prefill-0",
            "remote_host": "127.0.0.1",
            "remote_port": 20097,
            "remote_block_ids": [7, 8],
        }),
        prompt_tokens_details: None,
    });
    request
}

pub(crate) fn terminal_response(
    reason: dynamo_backend_common::FinishReason,
) -> pb::GenerateResponse {
    let mut response = sequence_response(true, false, None);
    response
        .outputs
        .as_mut()
        .unwrap()
        .finish_info
        .as_mut()
        .unwrap()
        .finish_reason = match reason {
        dynamo_backend_common::FinishReason::Stop => pb::finish_info::FinishReason::Stop,
        dynamo_backend_common::FinishReason::Length => pb::finish_info::FinishReason::Length,
        dynamo_backend_common::FinishReason::Cancelled => pb::finish_info::FinishReason::Aborted,
        other => panic!("unsupported fixture finish reason: {other:?}"),
    } as i32;
    response
}

pub(crate) fn prompt_logprob_response(
    token_ids: &[u32],
    selected: &[f32],
    candidates: &[Vec<(u32, f32)>],
) -> pb::GenerateResponse {
    let mut response = sequence_response(true, false, None);
    response.prompt_info = Some(pb::PromptInfo {
        num_prompt_tokens: token_ids.len() as u32,
        token_ids: token_ids.to_vec(),
        logprobs: selected.to_vec(),
        ranks: (0..selected.len() as u32).collect(),
        candidate_tokens: candidates
            .iter()
            .map(|entries| pb::CandidateTokenInfo {
                tokens: entries
                    .iter()
                    .enumerate()
                    .map(
                        |(index, &(id, logprob))| pb::candidate_token_info::TokenInfo {
                            id,
                            logprob,
                            rank: index as u32 + 2,
                        },
                    )
                    .collect(),
            })
            .collect(),
    });
    response
}
