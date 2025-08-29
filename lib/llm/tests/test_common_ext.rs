// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::{
    common::StopConditionsProvider,
    openai::{
        chat_completions::NvCreateChatCompletionRequest,
        common_ext::{CommonExt, CommonExtProvider},
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
}

#[test]
fn test_chat_completions_guided_decoding_from_common() {
    // Test that guided_json can be specified at root level
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "guided_json": {"key": "value"}
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(
        request.common.guided_json,
        Some(serde_json::json!({"key": "value"}))
    );
    assert_eq!(
        request.get_guided_json(),
        Some(&serde_json::json!({"key": "value"}))
    );

    // Test guided_regex can be specified at root level
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "guided_regex": "*"
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.guided_regex, Some("*".to_string()));
    assert_eq!(request.get_guided_regex(), Some("*".to_string()));

    // Test guided_grammar can be specified at root level
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "guided_grammar": "::=[1-9]"
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.guided_grammar, Some("::=[1-9]".to_string()));
    assert_eq!(request.get_guided_grammar(), Some("::=[1-9]".to_string()));

    // Test guided_choice can be specified at root level
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "guided_choice": ["choice1", "choice2"]
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(
        request.common.guided_choice,
        Some(vec!["choice1".to_string(), "choice2".to_string()])
    );
    assert_eq!(
        request.get_guided_choice(),
        Some(vec!["choice1".to_string(), "choice2".to_string()])
    );

    // Test guided_decoding_backend can be specified at root level
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "guided_decoding_backend": "backend"
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(
        request.common.guided_decoding_backend,
        Some("backend".to_string())
    );
    assert_eq!(
        request.get_guided_decoding_backend(),
        Some("backend".to_string())
    );
}

#[test]
fn test_chat_completions_common_overrides_nvext() {
    // Test that root-level ignore_eos overrides nvext ignore_eos
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "ignore_eos": false,
        "guided_regex": ".*",
        "min_tokens": 50,
        "nvext": {
            "ignore_eos": true,
            "guided_regex": "./*"
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, Some(false));
    assert_eq!(request.common.guided_regex, Some(".*".to_string()));
    assert_eq!(
        request.nvext.as_ref().and_then(|nv| nv.ignore_eos),
        Some(true)
    );
    assert_eq!(request.get_guided_regex(), Some(".*".to_string())); // common value takes precedence
    // Verify precedence through stop conditions extraction
    let stop_conditions = request.extract_stop_conditions().unwrap();
    assert_eq!(stop_conditions.ignore_eos, Some(false)); // common value takes precedence
    assert_eq!(stop_conditions.min_tokens, Some(50));
}

#[test]
fn test_chat_completions_backward_compatibility() {
    // Test backward compatibility - ignore_eos and guided_json only in nvext
    let json_str = r#"{
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "nvext": {
            "ignore_eos": true,
            "guided_json": {"key": "value"}
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_str).unwrap();

    assert_eq!(request.common.ignore_eos, None);
    assert_eq!(request.common.guided_json, None);
    assert_eq!(
        request.nvext.as_ref().and_then(|nv| nv.ignore_eos),
        Some(true)
    );
    assert_eq!(
        request
            .nvext
            .as_ref()
            .and_then(|nv| nv.guided_json.as_ref()),
        Some(&serde_json::json!({"key": "value"}))
    );
    assert_eq!(
        request.get_guided_json(),
        Some(&serde_json::json!({"key": "value"}))
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
fn test_completions_common_overrides_nvext() {
    // Test that root-level ignore_eos overrides nvext ignore_eos for completions
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
    assert_eq!(stop_conditions.ignore_eos, Some(false)); // common value takes precedence
    assert_eq!(stop_conditions.min_tokens, Some(75));
}

#[test]
fn test_serialization_preserves_structure() {
    // Test that serialization preserves the flattened structure
    let request = NvCreateChatCompletionRequest {
        inner: dynamo_async_openai::types::CreateChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![dynamo_async_openai::types::ChatCompletionRequestMessage::User(
                dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                    content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
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
            ..Default::default()
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
    assert_eq!(stop_conditions.ignore_eos, Some(true)); // common overrides nvext
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

#[test]
fn test_sampling_parameters_extraction() {
    use dynamo_llm::protocols::common::SamplingOptionsProvider;
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use dynamo_llm::protocols::openai::common_ext::CommonExt;

    // Test that top_k and repetition_penalty are extracted in sampling options when passed a top level
    let request = NvCreateChatCompletionRequest {
        inner: Default::default(),
        common: CommonExt::builder()
            .top_k(42)
            .repetition_penalty(1.3)
            .build()
            .unwrap(),
        nvext: None,
    };

    let sampling_options = request.extract_sampling_options().unwrap();

    assert_eq!(sampling_options.top_k, Some(42));
    assert_eq!(sampling_options.repetition_penalty, Some(1.3));
}
