// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::client::ModelLimits;
use crate::proto as pb;
use dynamo_backend_common::{
    DisaggregationMode, OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions,
};
use serde_json::json;

pub(crate) const AGG: DisaggregationMode = DisaggregationMode::Aggregated;
pub(crate) const PREFILL: DisaggregationMode = DisaggregationMode::Prefill;

pub(crate) fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("served-model".to_string())
        .token_ids(vec![11, 22, 33])
        .stop_conditions(StopConditions {
            max_tokens: Some(16),
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
            guided_decoding: Some(dynamo_backend_common::GuidedDecodingOptions {
                json: Some(json!({"type": "object"})),
                ..Default::default()
            }),
            ..Default::default()
        })
        .output_options(OutputOptions {
            logprobs: Some(1),
            ..Default::default()
        })
        .build()
        .expect("request")
}

/// Engine limits as `Control.GetModelInfo` would report them, with no
/// separate output cap.
pub(crate) fn limits(context_length: u32) -> Option<ModelLimits> {
    Some(ModelLimits {
        context_length: Some(context_length),
        max_output_tokens: None,
    })
}

/// The handoff a context worker returns, shaped like TensorRT-LLM's: the
/// session id is the context request id and the engine-specific state rides in
/// `attributes_struct`.
pub(crate) fn fake_session() -> pb::KvSessionRef {
    pb::KvSessionRef {
        session_id: "12345".to_string(),
        transfer_backend: "NIXL".to_string(),
        endpoints: vec![pb::KvEndpoint {
            host: "10.0.0.7".to_string(),
            port: 5601,
            protocol: "grpc".to_string(),
        }],
        dp_rank: 0,
        attributes_struct: Some(prost_types::Struct {
            fields: [
                (
                    "opaque_state".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue(
                            "c3RhdGU=".to_string(),
                        )),
                    },
                ),
                (
                    "first_gen_tokens".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::ListValue(
                            prost_types::ListValue {
                                values: vec![prost_types::Value {
                                    kind: Some(prost_types::value::Kind::NumberValue(42.0)),
                                }],
                            },
                        )),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }),
    }
}

/// Mirrors the OpenEngine servicer: a request is prefill-only when its `extra`
/// Struct carries `request_type = "context_only"`.
pub(crate) fn is_context_only(request: &pb::GenerateRequest) -> bool {
    request
        .extra
        .as_ref()
        .and_then(|extra| extra.fields.get("request_type"))
        .and_then(|value| value.kind.as_ref())
        .is_some_and(|kind| {
            matches!(kind, prost_types::value::Kind::StringValue(value) if value == "context_only")
        })
}

pub(crate) fn token_response(tokens: Vec<pb::TokenInfo>) -> pb::GenerateResponse {
    pb::GenerateResponse {
        request_id: "r".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(0),
            tokens,
            text: String::new(),
        })),
        usage: None,
    }
}

pub(crate) fn logprob_token(token_id: u32, logprob: f64) -> pb::TokenInfo {
    pb::TokenInfo {
        token_id,
        token: String::new(),
        logprob: Some(logprob),
        rank: Some(1),
        candidates: Vec::new(),
    }
}
