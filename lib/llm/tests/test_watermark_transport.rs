// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::engines::ValidateRequest;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::{SamplingOptions, SamplingOptionsProvider};
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::protocols::openai::completions::NvCreateCompletionRequest;
use serde_json::{Value, json};

#[test]
fn watermarking_survives_chat_and_completion_worker_transport() {
    for value in [Value::Null, json!(false), json!(true)] {
        let expected = value.as_bool();
        for stream in [false, true] {
            let chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
                "watermarking": value,
            }))
            .unwrap();
            let completion: NvCreateCompletionRequest = serde_json::from_value(json!({
                "model": "test-model", "prompt": "hello", "stream": stream,
                "watermarking": value,
            }))
            .unwrap();
            chat.validate().unwrap();
            completion.validate().unwrap();
            for options in [
                chat.extract_sampling_options().unwrap(),
                completion.extract_sampling_options().unwrap(),
            ] {
                let request = PreprocessedRequest::builder()
                    .model("test-model".to_string())
                    .token_ids(vec![1])
                    .stop_conditions(Default::default())
                    .sampling_options(options)
                    .output_options(Default::default())
                    .build()
                    .unwrap();
                let wire = serde_json::to_value(request).unwrap();
                match expected {
                    Some(flag) => assert_eq!(wire["sampling_options"]["watermarking"], flag),
                    None => assert!(wire["sampling_options"].get("watermarking").is_none()),
                }
                let received: PreprocessedRequest = serde_json::from_value(wire).unwrap();
                let received = serde_json::to_value(received).unwrap();
                assert_eq!(
                    received["sampling_options"].get("watermarking"),
                    expected.map(Value::Bool).as_ref()
                );
            }
        }
    }
}

#[test]
fn omitted_watermarking_keeps_legacy_worker_payload() {
    let legacy = serde_json::to_value(SamplingOptions::default()).unwrap();
    assert!(legacy.get("watermarking").is_none());
    let received: SamplingOptions = serde_json::from_value(legacy).unwrap();
    assert!(
        serde_json::to_value(received)
            .unwrap()
            .get("watermarking")
            .is_none()
    );
    let chat: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }))
    .unwrap();
    let completion: NvCreateCompletionRequest = serde_json::from_value(json!({
        "model": "test-model", "prompt": "hello",
    }))
    .unwrap();
    chat.validate().unwrap();
    completion.validate().unwrap();
    for options in [
        chat.extract_sampling_options().unwrap(),
        completion.extract_sampling_options().unwrap(),
    ] {
        assert!(
            serde_json::to_value(options)
                .unwrap()
                .get("watermarking")
                .is_none()
        );
    }
}

#[test]
fn malformed_watermarking_is_rejected_by_both_endpoints() {
    for value in [
        json!(1),
        json!("false"),
        json!([]),
        json!({"enabled": false}),
    ] {
        assert!(
            serde_json::from_value::<NvCreateChatCompletionRequest>(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "watermarking": value,
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<NvCreateCompletionRequest>(json!({
                "model": "test-model", "prompt": "hello", "watermarking": value,
            }))
            .is_err()
        );
    }
}
