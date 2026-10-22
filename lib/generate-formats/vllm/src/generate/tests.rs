// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

fn aggregate(
    chunks: impl IntoIterator<Item = LLMEngineOutput>,
    request_id: String,
    options: GenerateResponseOptions,
) -> Result<GenerateResponse> {
    let mut aggregator = GenerateAggregator::new(request_id);
    for output in chunks {
        aggregator.apply_output(output, options)?;
    }
    aggregator.finish(options)
}

#[test]
fn generate_request_deserializes_from_vllm_json() {
    let raw = json!({
        "request_id": "req-123",
        "token_ids": [1, 2, 3, 4],
        "sampling_params": {"temperature": 0.7, "max_tokens": 16},
        "model": "test-model",
        "stream": false,
        "cache_salt": "salt",
        "priority": 0,
        "kv_transfer_params": null
    });
    let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
    assert_eq!(req.request_id.as_deref(), Some("req-123"));
    assert_eq!(req.token_ids, vec![1, 2, 3, 4]);
    assert!(!req.stream);
    assert_eq!(req.model.as_deref(), Some("test-model"));
    assert_eq!(req.cache_salt.as_deref(), Some("salt"));
    assert_eq!(req.priority, 0);
    assert!(req.kv_transfer_params.is_none());
    assert_eq!(req.sampling_params.max_tokens(), Some(16));
}

#[test]
fn generate_request_captures_unknown_fields() {
    // Untyped + future fields are CAPTURED in `passthrough` (forwarded
    // verbatim), not dropped — so a field a newer vLLM adds flows through.
    let raw = json!({
        "token_ids": [5, 6],
        "sampling_params": {},
        "priority": 7,
        "future_field": "kept"
    });
    let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
    assert_eq!(req.token_ids, vec![5, 6]);
    assert!(!req.stream);
    assert_eq!(req.request_id, None);
    assert_eq!(req.priority, 7);
    // Future fields remain captured in `passthrough`.
    assert!(!req.passthrough.contains_key("priority"));
    assert_eq!(req.passthrough.get("future_field"), Some(&json!("kept")));
    // Round-trip: flattened fields serialize back at the top level.
    let back = serde_json::to_value(&req).expect("serialize");
    assert_eq!(back.get("priority"), Some(&json!(7)));
    assert_eq!(back.get("future_field"), Some(&json!("kept")));
}

#[test]
fn generate_request_preserves_unknown_sampling_fields() {
    let raw_sampling = json!({
        "temperature": 0.5,
        "top_k": 12,
        "max_tokens": 8,
        "min_tokens": null,
        "bad_words": ["blocked"],
        "future_sampling_field": {"nested": true}
    });
    let req: GenerateRequest = serde_json::from_value(json!({
        "token_ids": [5, 6],
        "sampling_params": raw_sampling.clone()
    }))
    .expect("deserialize");

    assert_eq!(req.sampling_params.max_tokens(), Some(8));
    assert_eq!(req.sampling_params.temperature, Some(0.5));
    assert_eq!(req.sampling_params.top_k, Some(12));
    assert_eq!(
        req.sampling_params.bad_words.as_deref(),
        Some(["blocked".to_string()].as_slice())
    );
    assert_eq!(req.sampling_params.min_tokens, None);
    assert_eq!(req.sampling_params.as_value()["min_tokens"], Value::Null);
    assert_eq!(req.sampling_params.as_value(), &raw_sampling);
    let back = serde_json::to_value(&req).expect("serialize");
    assert_eq!(back["sampling_params"], raw_sampling);
}

#[test]
fn generate_request_accepts_disabled_top_k_sentinel() {
    let request: GenerateRequest = serde_json::from_value(json!({
        "token_ids": [1],
        "sampling_params": {"top_k": -1}
    }))
    .expect("deserialize");

    request.validate().expect("validate");
    assert_eq!(
        request.sampling_params.project_sampling_options().top_k,
        Some(-1)
    );
}

#[test]
fn generate_request_matches_rust_integer_types() {
    for raw in [
        json!({
            "token_ids": [1],
            "sampling_params": {},
            "priority": i64::from(i32::MAX) + 1
        }),
        json!({
            "token_ids": [1],
            "sampling_params": {"max_tokens": -1}
        }),
        json!({
            "token_ids": [1],
            "sampling_params": {"top_k": i64::from(i32::MAX) + 1}
        }),
        json!({
            "token_ids": [1],
            "sampling_params": {"ignore_eos": null}
        }),
    ] {
        assert!(serde_json::from_value::<GenerateRequest>(raw).is_err());
    }
}

#[test]
fn generate_request_validates_rust_route_rules() {
    let invalid = [
        (
            json!({
                "token_ids": [1],
                "sampling_params": {},
                "stream_options": {"include_usage": true}
            }),
            "stream_options",
        ),
        (
            json!({
                "token_ids": [1],
                "sampling_params": {"max_tokens": 0}
            }),
            "max_tokens",
        ),
        (
            json!({
                "token_ids": [1],
                "sampling_params": {"thinking_token_budget": 32}
            }),
            "thinking_token_budget",
        ),
        (
            json!({
                "token_ids": [1],
                "sampling_params": {"prompt_logprobs": -2}
            }),
            "prompt_logprobs",
        ),
        (
            json!({
                "token_ids": [1],
                "sampling_params": {"top_k": -2}
            }),
            "Top_k",
        ),
        (
            json!({
                "token_ids": [1],
                "sampling_params": {"min_tokens": 3, "max_tokens": 2}
            }),
            "min_tokens",
        ),
    ];

    for (raw, expected) in invalid {
        let req: GenerateRequest = serde_json::from_value(raw).expect("deserialize");
        let error = req.validate().expect_err("must reject");
        assert!(error.contains(expected), "unexpected error: {error}");
    }
}

#[test]
fn generate_request_accepts_vllm_repetition_penalty_above_two() {
    let request: GenerateRequest = serde_json::from_value(json!({
        "token_ids": [1],
        "sampling_params": {"repetition_penalty": 2.5}
    }))
    .expect("deserialize");

    request
        .validate()
        .expect("vLLM accepts finite positive repetition penalties");
}

#[test]
fn generate_request_preserves_opaque_structured_outputs() {
    let raw = json!({
        "token_ids": [1],
        "sampling_params": {
            "structured_outputs": {
                "future_constraint": {"version": 2},
                "future_option": true
            }
        }
    });
    let req: GenerateRequest = serde_json::from_value(raw.clone()).expect("deserialize");

    assert_eq!(
        req.sampling_params.structured_outputs.as_ref(),
        Some(&raw["sampling_params"]["structured_outputs"])
    );
    assert_eq!(
        serde_json::to_value(req).expect("serialize")["sampling_params"],
        raw["sampling_params"]
    );
}

#[test]
fn generate_request_rejects_empty_token_ids() {
    let request = serde_json::from_value::<GenerateRequest>(json!({
        "token_ids": [],
        "sampling_params": {}
    }))
    .expect("vLLM Rust validates empty token IDs after JSON parsing");
    let error = request
        .validate()
        .expect_err("empty token_ids must be rejected");

    assert!(error.contains("token_ids must contain at least one token"));
}

#[test]
fn generate_response_matches_vllm_null_metadata_shape() {
    let resp = GenerateResponse {
        request_id: "req-123".to_string(),
        choices: vec![GenerateResponseChoice {
            index: 0,
            token_ids: None,
            logprobs: None,
            finish_reason: None,
            routed_experts: None,
            sampling_mask: None,
        }],
        prompt_logprobs: None,
        kv_transfer_params: None,
    };

    let value = serde_json::to_value(&resp).expect("serialize");
    assert!(
        value.get("usage").is_none(),
        "GenerateResponse must not emit a `usage` key"
    );
    assert_eq!(value["prompt_logprobs"], Value::Null);
    assert_eq!(value["kv_transfer_params"], Value::Null);
    assert_eq!(value["choices"][0]["token_ids"], Value::Null);
    assert_eq!(value["choices"][0]["logprobs"], Value::Null);
    assert_eq!(value["choices"][0]["sampling_mask"], Value::Null);
    assert_eq!(value["choices"][0]["finish_reason"], Value::Null);
    assert_eq!(value["choices"][0]["routed_experts"], Value::Null);

    let round: GenerateResponse = serde_json::from_value(value).expect("round-trip deserialize");
    assert_eq!(round.request_id, "req-123");
    assert_eq!(round.choices.len(), 1);
    assert_eq!(round.choices[0].token_ids, None);
    assert_eq!(round.choices[0].finish_reason, None);
}

#[test]
fn completion_logprobs_match_typed_wire_shape() {
    let output = LLMEngineOutput {
        token_ids: vec![100],
        log_probs: Some(vec![-1.0e30]),
        top_logprobs: Some(vec![vec![
            dynamo_llm::protocols::common::llm_backend::TopLogprob {
                rank: 1,
                token_id: 7,
                token: None,
                logprob: -1.0e30,
                bytes: None,
            },
        ]]),
        ..Default::default()
    };
    let mut content = Vec::new();

    assert!(
        append_completion_logprobs_from_output(&output, &mut content)
            .expect("build completion logprobs")
    );

    let expected = dynamo_protocols::types::ChatCompletionTokenLogprob {
        token: "token_id:100".to_string(),
        logprob: -9999.0,
        token_id: None,
        bytes: Some(b"token_id:100".to_vec()),
        top_logprobs: vec![
            dynamo_protocols::types::TopLogprobs {
                token: "token_id:7".to_string(),
                logprob: -9999.0,
                bytes: Some(b"token_id:7".to_vec()),
            },
            dynamo_protocols::types::TopLogprobs {
                token: "token_id:100".to_string(),
                logprob: -9999.0,
                bytes: Some(b"token_id:100".to_vec()),
            },
        ],
    };

    assert_eq!(
        serde_json::to_value(content).expect("serialize accumulated logprobs"),
        serde_json::to_value(vec![expected]).expect("serialize typed logprob")
    );
}

#[test]
fn completion_logprobs_reject_malformed_output() {
    let cases = [
        (
            LLMEngineOutput {
                token_ids: vec![100],
                top_logprobs: Some(vec![vec![]]),
                ..Default::default()
            },
            "top_logprobs without selected-token logprobs",
        ),
        (
            LLMEngineOutput {
                token_ids: vec![100, 101],
                log_probs: Some(vec![-0.25]),
                ..Default::default()
            },
            "1 selected-token logprobs for 2 tokens",
        ),
        (
            LLMEngineOutput {
                token_ids: vec![100, 101],
                log_probs: Some(vec![-0.25, -0.5]),
                top_logprobs: Some(vec![vec![]]),
                ..Default::default()
            },
            "1 top-logprob positions for 2 tokens",
        ),
    ];

    for (output, expected_error) in cases {
        let existing =
            json!({"token": "existing", "logprob": 0.0, "bytes": null, "top_logprobs": []});
        let mut content = vec![serde_json::from_value(existing.clone()).unwrap()];

        let error = append_completion_logprobs_from_output(&output, &mut content)
            .expect_err("malformed output must fail");

        assert!(
            error.to_string().contains(expected_error),
            "unexpected error: {error}"
        );
        assert_eq!(serde_json::to_value(content).unwrap(), json!([existing]));
    }
}

#[test]
fn empty_completion_logprobs_do_not_modify_content() {
    let output = LLMEngineOutput {
        log_probs: Some(vec![]),
        top_logprobs: Some(vec![]),
        ..Default::default()
    };
    let existing = json!({"token": "existing", "logprob": 0.0, "bytes": null, "top_logprobs": []});
    let mut content = vec![serde_json::from_value(existing.clone()).unwrap()];

    assert!(
        !append_completion_logprobs_from_output(&output, &mut content)
            .expect("empty output is valid")
    );
    assert_eq!(serde_json::to_value(content).unwrap(), json!([existing]));
}

#[test]
fn generate_response_accumulates_engine_deltas() {
    let chunks = vec![
        LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            log_probs: Some(vec![-0.25]),
            top_logprobs: Some(vec![vec![
                dynamo_llm::protocols::common::llm_backend::TopLogprob {
                    rank: 1,
                    token_id: 100,
                    token: None,
                    logprob: -0.25,
                    bytes: None,
                },
                dynamo_llm::protocols::common::llm_backend::TopLogprob {
                    rank: 2,
                    token_id: 7,
                    token: None,
                    logprob: -1.5,
                    bytes: None,
                },
            ]]),
            engine_data: Some(json!({
                "prompt_logprobs": [
                    null,
                    {
                        "22": {
                            "logprob": -0.125,
                            "rank": 1
                        },
                        "23": {
                            "logprob": -0.5,
                            "rank": 2,
                            "decoded_token": "worker label"
                        }
                    }
                ]
            })),
            ..Default::default()
        },
        LLMEngineOutput {
            token_ids: vec![101],
            index: Some(0),
            log_probs: Some(vec![-0.5]),
            top_logprobs: Some(vec![vec![
                dynamo_llm::protocols::common::llm_backend::TopLogprob {
                    rank: 1,
                    token_id: 101,
                    token: None,
                    logprob: -0.5,
                    bytes: None,
                },
            ]]),
            finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
            engine_data: Some(json!({
                "routed_experts": "encoded-experts",
                "sampling_mask": [[7, 11], [13, 17, 19]],
                "kv_transfer_params": {"connector": "x"}
            })),
            ..Default::default()
        },
    ];
    let response = aggregate(
        chunks,
        "req-agg".to_string(),
        GenerateResponseOptions {
            include_logprobs: true,
            include_prompt_logprobs: true,
        },
    )
    .expect("aggregate engine deltas");

    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].token_ids, Some(vec![100, 101]));
    assert_eq!(response.choices[0].finish_reason.as_deref(), Some("length"));
    assert_eq!(
        response.choices[0].routed_experts.as_ref(),
        Some(&GenerateRoutedExperts::Legacy(
            "encoded-experts".to_string()
        ))
    );
    assert_eq!(
        response.choices[0].sampling_mask,
        Some(vec![vec![7, 11], vec![13, 17, 19]])
    );
    let logprobs = response.choices[0]
        .logprobs
        .as_ref()
        .expect("completion logprobs");
    assert_eq!(logprobs["content"][0]["token"], "token_id:100");
    assert_eq!(logprobs["content"][1]["token"], "token_id:101");
    assert!(logprobs.get("refusal").is_none());
    assert_eq!(
        logprobs["content"][0]["bytes"],
        serde_json::json!(b"token_id:100")
    );
    assert_eq!(
        logprobs["content"][0]["top_logprobs"][0]["bytes"],
        serde_json::json!(b"token_id:100")
    );
    assert_eq!(
        logprobs["content"][0]["top_logprobs"][1]["token"],
        "token_id:7"
    );
    assert_eq!(
        logprobs["content"][0]["top_logprobs"][1]["bytes"],
        serde_json::json!(b"token_id:7")
    );
    assert_eq!(
        response.prompt_logprobs.as_ref().expect("prompt logprobs")[1]["22"]["decoded_token"],
        "token_id:22"
    );
    assert_eq!(
        response.prompt_logprobs.as_ref().unwrap()[1]["23"]["decoded_token"],
        "worker label"
    );
    assert_eq!(
        response
            .kv_transfer_params
            .as_ref()
            .expect("kv transfer params"),
        &json!({"connector": "x"})
    );
    assert!(response.is_complete_unary());
}

#[test]
fn generate_response_associates_routed_experts_with_choices() {
    let chunks = [
        LLMEngineOutput {
            token_ids: vec![201],
            index: Some(1),
            finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
            engine_data: Some(json!({"routed_experts": "experts-1"})),
            ..Default::default()
        },
        LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
            engine_data: Some(json!({"routed_experts": "experts-0"})),
            ..Default::default()
        },
    ];

    let response = aggregate(
        chunks,
        "req-routed".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("aggregate routed experts");

    assert_eq!(response.choices[0].index, 0);
    assert_eq!(
        response.choices[0].routed_experts.as_ref(),
        Some(&GenerateRoutedExperts::Legacy("experts-0".to_string()))
    );
    assert_eq!(response.choices[1].index, 1);
    assert_eq!(
        response.choices[1].routed_experts.as_ref(),
        Some(&GenerateRoutedExperts::Legacy("experts-1".to_string()))
    );
}

#[test]
fn generate_response_accepts_structured_routed_experts() {
    let payload = json!({
        "data": "AQI=",
        "shape": [2, 1, 1],
        "start": 3,
        "dtype": "uint8"
    });
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
        engine_data: Some(json!({"routed_experts": payload})),
        ..Default::default()
    }];

    let response = aggregate(
        chunks,
        "req-routed-structured".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("aggregate structured routed experts");

    assert_eq!(
        response.choices[0].routed_experts,
        Some(GenerateRoutedExperts::Tensor(GenerateRoutedExpertsTensor {
            data: "AQI=".to_string(),
            shape: vec![2, 1, 1],
            start: 3,
            dtype: "uint8".to_string(),
        }))
    );
}

#[test]
fn generate_response_rejects_structured_routed_expert_byte_mismatch() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
        engine_data: Some(json!({
            "routed_experts": {
                "data": "AQI=",
                "shape": [3, 1, 1],
                "start": 0,
                "dtype": "uint8"
            }
        })),
        ..Default::default()
    }];

    let error = aggregate(
        chunks,
        "req-routed-structured".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect_err("routed expert byte mismatch must fail");

    assert!(error.to_string().contains("decoded byte count"));
}

#[test]
fn generate_response_rejects_malformed_routed_experts() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
        engine_data: Some(json!({"routed_experts": {"unexpected": "object"}})),
        ..Default::default()
    }];

    let error = aggregate(
        chunks,
        "req-routed".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect_err("malformed routed experts must fail");

    assert!(
        error
            .to_string()
            .contains("invalid generate routed_experts payload")
    );
}

#[test]
fn generate_response_rejects_malformed_sampling_mask() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
        engine_data: Some(json!({"sampling_mask": "invalid"})),
        ..Default::default()
    }];

    let error = aggregate(
        chunks,
        "req-mask".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect_err("malformed sampling mask must fail");

    assert!(
        error
            .to_string()
            .contains("invalid generate sampling_mask payload")
    );
}

#[test]
fn generate_response_rejects_sampling_mask_length_mismatch() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
        engine_data: Some(json!({"sampling_mask": [[1], [2]]})),
        ..Default::default()
    }];

    let error = aggregate(
        chunks,
        "req-mask".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect_err("misaligned sampling mask must fail");

    assert!(error.to_string().contains("sampling-mask positions"));
}

#[test]
fn generate_response_uses_terminal_kv_transfer_fallback() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
        disaggregated_params: Some(json!({"source": "terminal-fallback"})),
        ..Default::default()
    }];

    let response = aggregate(
        chunks,
        "req-kv".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("aggregate terminal KV metadata");

    assert_eq!(
        response.kv_transfer_params,
        Some(json!({"source": "terminal-fallback"}))
    );
}

#[test]
fn generate_response_prefers_engine_kv_metadata_over_later_fallback() {
    let chunks = [
        LLMEngineOutput {
            token_ids: vec![100],
            index: Some(0),
            engine_data: Some(json!({
                "kv_transfer_params": {"source": "engine-data"}
            })),
            ..Default::default()
        },
        LLMEngineOutput {
            index: Some(0),
            finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Stop),
            disaggregated_params: Some(json!({"source": "terminal-fallback"})),
            ..Default::default()
        },
    ];

    let response = aggregate(
        chunks,
        "req-kv".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("aggregate KV metadata with precedence");

    assert_eq!(
        response.kv_transfer_params,
        Some(json!({"source": "engine-data"}))
    );
}

#[test]
fn incomplete_engine_streams_are_not_complete_unary_responses() {
    let empty = Vec::<LLMEngineOutput>::new();
    let empty_response = aggregate(
        empty,
        "req-empty".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("aggregate empty stream");
    assert!(!empty_response.is_complete_unary());

    let partial = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        ..Default::default()
    }];
    let partial_response = aggregate(
        partial,
        "req-partial".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("aggregate partial stream");
    assert!(!partial_response.is_complete_unary());
}

#[test]
fn generate_response_rejects_error_and_cancelled_finish_reasons() {
    for finish_reason in [
        dynamo_llm::protocols::common::FinishReason::Error("worker failed".to_string()),
        dynamo_llm::protocols::common::FinishReason::Cancelled,
    ] {
        let chunks = [LLMEngineOutput {
            index: Some(0),
            finish_reason: Some(finish_reason),
            ..Default::default()
        }];

        aggregate(
            chunks,
            "req-failed".to_string(),
            GenerateResponseOptions::default(),
        )
        .expect_err("error-like finish reasons must not produce HTTP-success responses");
    }
}

#[test]
fn generate_response_rejects_incomplete_completion_logprobs() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100, 101],
        log_probs: Some(vec![-0.25]),
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
        ..Default::default()
    }];

    let error = aggregate(
        chunks,
        "req-logprobs".to_string(),
        GenerateResponseOptions {
            include_logprobs: true,
            ..Default::default()
        },
    )
    .expect_err("misaligned logprobs must fail");

    assert!(
        error
            .to_string()
            .contains("1 selected-token logprobs for 2 tokens")
    );
}

#[test]
fn generate_response_rejects_malformed_prompt_logprobs() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
        engine_data: Some(json!({"prompt_logprobs": "invalid"})),
        ..Default::default()
    }];

    let error = aggregate(
        chunks,
        "req-prompt".to_string(),
        GenerateResponseOptions {
            include_prompt_logprobs: true,
            ..Default::default()
        },
    )
    .expect_err("malformed prompt logprobs must fail");

    assert!(
        error
            .to_string()
            .contains("invalid generate prompt_logprobs payload")
    );
}

#[test]
fn generate_request_derives_response_metadata_options() {
    let request: GenerateRequest = serde_json::from_value(json!({
        "token_ids": [1],
        "sampling_params": {"logprobs": 0, "prompt_logprobs": 0}
    }))
    .expect("deserialize request");

    let options = request.response_options();

    assert!(options.include_logprobs);
    assert!(options.include_prompt_logprobs);
}

#[test]
fn generate_response_rejects_requested_but_missing_metadata() {
    let output = || LLMEngineOutput {
        token_ids: vec![100],
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
        ..Default::default()
    };

    let completion_error = aggregate(
        [output()],
        "req-missing-completion".to_string(),
        GenerateResponseOptions {
            include_logprobs: true,
            ..Default::default()
        },
    )
    .expect_err("missing requested completion logprobs must fail");
    assert!(completion_error.to_string().contains("returned none"));

    let prompt_error = aggregate(
        [output()],
        "req-missing-prompt".to_string(),
        GenerateResponseOptions {
            include_prompt_logprobs: true,
            ..Default::default()
        },
    )
    .expect_err("missing requested prompt logprobs must fail");
    assert!(prompt_error.to_string().contains("prompt_logprobs"));
}

#[test]
fn generate_response_does_not_leak_unrequested_metadata() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        log_probs: Some(vec![-0.25, -0.5]),
        engine_data: Some(json!({"prompt_logprobs": "invalid"})),
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
        ..Default::default()
    }];

    let response = aggregate(
        chunks,
        "req-no-metadata".to_string(),
        GenerateResponseOptions::default(),
    )
    .expect("unrequested metadata must be ignored");

    assert!(response.choices[0].logprobs.is_none());
    assert!(response.prompt_logprobs.is_none());
}

#[test]
fn generate_response_clamps_logprobs_to_vllm_floor() {
    let chunks = [LLMEngineOutput {
        token_ids: vec![100],
        log_probs: Some(vec![-1.0e30]),
        top_logprobs: Some(vec![vec![
            dynamo_llm::protocols::common::llm_backend::TopLogprob {
                rank: 1,
                token_id: 100,
                token: None,
                logprob: -1.0e30,
                bytes: None,
            },
        ]]),
        engine_data: Some(json!({
            "prompt_logprobs": [
                null,
                {
                    "22": {
                        "logprob": -1.0e30,
                        "rank": 1,
                        "decoded_token": "token_id:22"
                    }
                }
            ]
        })),
        index: Some(0),
        finish_reason: Some(dynamo_llm::protocols::common::FinishReason::Length),
        ..Default::default()
    }];

    let response = aggregate(
        chunks,
        "req-clamped".to_string(),
        GenerateResponseOptions {
            include_logprobs: true,
            include_prompt_logprobs: true,
        },
    )
    .expect("aggregate clamped logprobs");
    let logprobs = response.choices[0].logprobs.as_ref().unwrap();

    assert_eq!(logprobs["content"][0]["logprob"], json!(-9999.0));
    assert_eq!(
        logprobs["content"][0]["top_logprobs"][0]["logprob"],
        json!(-9999.0)
    );
    assert_eq!(
        response.prompt_logprobs.as_ref().unwrap()[1]["22"]["logprob"],
        json!(-9999.0)
    );
}
