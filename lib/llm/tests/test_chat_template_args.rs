// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

#[test]
fn test_both_fields_fails() {
    // Test that when both are present, serde will fail with duplicate field error
    // This test documents that behavior
    let json_with_both = r#"{
        "model": "test-model",
        "messages": [],
        "chat_template_args": {
            "enable_thinking": true
        },
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }"#;

    // This will fail with duplicate field error
    let result: Result<NvCreateChatCompletionRequest, _> = serde_json::from_str(json_with_both);
    assert!(result.is_err());
}

#[test]
fn test_chat_template_kwargs_alias() {
    // Test that chat_template_kwargs is accepted as an alias for chat_template_args
    let json_with_kwargs = r#"{
        "model": "test-model",
        "messages": [],
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_with_kwargs).unwrap();
    assert!(request.chat_template_args.is_some());
    assert_eq!(
        request.chat_template_args.unwrap().get("enable_thinking"),
        Some(&serde_json::json!(false))
    );
}

#[test]
fn test_chat_template_args() {
    // Test that chat_template_args still works as the primary field name
    let json_with_args = r#"{
        "model": "test-model",
        "messages": [],
        "chat_template_args": {
            "enable_thinking": true
        }
    }"#;

    let request: NvCreateChatCompletionRequest = serde_json::from_str(json_with_args).unwrap();
    assert!(request.chat_template_args.is_some());
    assert_eq!(
        request.chat_template_args.unwrap().get("enable_thinking"),
        Some(&serde_json::json!(true))
    );
}

// TODO: Add template rendering test that verifies chat_template_args/chat_template_kwargs
// values are actually passed to the Jinja template context during rendering.
// This would require setting up PromptFormatter with ChatTemplate/ContextMixins.
