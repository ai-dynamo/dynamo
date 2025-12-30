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
fn test_chat_template_args_rendering() {
    // Test that both chat_template_args and chat_template_kwargs work correctly
    // This test covers all the alias functionality cases

    // Test cases: both field names with enable_thinking true/false
    let test_cases = [
        (
            r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }"#,
            None,
            "no template args",
        ),
        (
            r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_args": {
                "enable_thinking": false
            }
        }"#,
            Some(false),
            "chat_template_args with false",
        ),
        (
            r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_args": {
                "enable_thinking": true
            }
        }"#,
            Some(true),
            "chat_template_args with true",
        ),
        (
            r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "enable_thinking": false
            }
        }"#,
            Some(false),
            "chat_template_kwargs with false",
        ),
        (
            r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "enable_thinking": true
            }
        }"#,
            Some(true),
            "chat_template_kwargs with true",
        ),
    ];

    for (json_request, expected_enable_thinking, description) in test_cases {
        // Parse request from JSON
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_request)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", description, e));

        // Verify that chat_template_args is populated correctly (this is the key test)
        match expected_enable_thinking {
            None => {
                assert!(
                    request.chat_template_args.is_none(),
                    "Expected no chat_template_args for {}, got: {:?}",
                    description,
                    request.chat_template_args
                );
            }
            Some(expected_value) => {
                let args = request
                    .chat_template_args
                    .as_ref()
                    .expect(&format!("Expected chat_template_args for {}", description));

                let enable_thinking = args.get("enable_thinking").expect(&format!(
                    "Expected enable_thinking in args for {}",
                    description
                ));

                let actual_value = enable_thinking.as_bool().expect(&format!(
                    "Expected enable_thinking to be boolean for {}",
                    description
                ));

                assert_eq!(
                    actual_value, expected_value,
                    "Expected enable_thinking={} for {}, got: {}",
                    expected_value, description, actual_value
                );
            }
        }

        println!(
            "✓ {} - chat_template_args: {:?}",
            description, request.chat_template_args
        );
    }
}
