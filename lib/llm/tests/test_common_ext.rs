// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::{
    common::StopConditionsProvider,
    openai::{
        chat_completions::NvCreateChatCompletionRequest,
        common_ext::CommonExt,
        completions::NvCreateCompletionRequest,
        nvext::NvExt,
    },
};

#[test]
fn test_chat_completions_ignore_eos_from_common() {
    // Test that ignore_eos can be specified at root level
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "ignore_eos": true,
        "min_tokens": 100
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, Some(true));
    assert_eq!(request.common.min_tokens, Some(100));

    // Verify through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(true));
    assert_eq!(stop_conditions.min_tokens, Some(100));
}

#[test]
fn test_chat_completions_nvext_overrides_common() {
    // Test that nvext ignore_eos overrides root-level ignore_eos
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "ignore_eos": false,
        "min_tokens": 50,
        "nvext": {
            "ignore_eos": true
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, Some(false));
    assert_eq!(
        request.nvext.as_ref().and_then(|nv| nv.ignore_eos),
        Some(true)
    );
    // Verify precedence through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(true)); // nvext value takes precedence
    assert_eq!(stop_conditions.min_tokens, Some(50));
}

#[test]
fn test_chat_completions_backward_compatibility() {
    // Test backward compatibility - ignore_eos only in nvext
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "nvext": {
            "ignore_eos": true
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, None);
    assert_eq!(
        request.nvext.as_ref().and_then(|nv| nv.ignore_eos),
        Some(true)
    );
    // Verify through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(true));
    assert_eq!(stop_conditions.min_tokens, None);
}

#[test]
fn test_completions_ignore_eos_from_common() {
    // Test that ignore_eos can be specified at root level for completions
    let json_str = r#"{
        "model": "test-model",
        "prompt": "Hello world",
        "ignore_eos": true,
        "min_tokens": 200
    }"#;

    let request: NvCreateCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, Some(true));
    assert_eq!(request.common.min_tokens, Some(200));

    // Verify through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(true));
    assert_eq!(stop_conditions.min_tokens, Some(200));
}

#[test]
fn test_completions_nvext_overrides_common() {
    // Test that nvext ignore_eos overrides root-level ignore_eos for completions
    let json_str = r#"{
        "model": "test-model",
        "prompt": "Hello world",
        "ignore_eos": false,
        "min_tokens": 75,
        "nvext": {
            "ignore_eos": true
        }
    }"#;

    let request: NvCreateCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, Some(false));
    assert_eq!(
        request.nvext.as_ref().and_then(|nv| nv.ignore_eos),
        Some(true)
    );
    // Verify precedence through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(true)); // nvext value takes precedence
    assert_eq!(stop_conditions.min_tokens, Some(75));
}

#[test]
fn test_serialization_preserves_structure() {
    // Test that serialization preserves the flattened structure
    let request = NvCreateChatCompletionRequest {
        inner: async_openai::types::CreateChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hello".to_string(),
                    ),
                    ..Default::default()
                },
            )],
            ..Default::default()
        },
        common: CommonExt {
            ignore_eos: Some(true),
            min_tokens: Some(100),
        },
        nvext: Some(NvExt {
            ignore_eos: Some(false),
            ..Default::default()
        }),
    };

    let json = serde_json::to_value(&request).unwrap();

    // Check that fields are at the expected levels
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["ignore_eos"], true); // From common (flattened)
    assert_eq!(json["min_tokens"], 100); // From common (flattened)
    assert_eq!(json["nvext"]["ignore_eos"], false); // From nvext

    // Verify precedence through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(false)); // nvext overrides common
    assert_eq!(stop_conditions.min_tokens, Some(100));
}

#[test]
fn test_min_tokens_only_at_root_level() {
    // Test that min_tokens is only available at root level, not in nvext
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "min_tokens": 150
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.min_tokens, Some(150));

    // Verify through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.min_tokens, Some(150));
}
