// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_llm::preprocessor::prompt::PromptFormatter;
use dynamo_llm::model_card::ModelDeploymentCard;

// Common test template that uses enable_thinking variable
const TEST_TEMPLATE: &str = r#"{%- for message in messages %}
{%- if message['role'] == 'user' %}
TEMPLATE_TEST{% if enable_thinking is defined and enable_thinking %}_THINKING{% endif %}| {{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
<|assistant|>{{ message['content'] }}
{%- endif %}
{%- endfor %}
<|assistant|>"#;

#[test]
fn test_chat_template_kwargs_alias() {
    // Test that chat_template_kwargs is accepted as an alias
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
fn test_chat_template_args_still_works() {
    // Test that chat_template_args still works
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
fn test_template_rendering() {
    // Test that both chat_template_args and chat_template_kwargs work in template rendering
    
    // Create formatter with common template
    let mut mdc = ModelDeploymentCard::new("test-model".to_string());
    mdc.set_chat_template(TEST_TEMPLATE.to_string());
    
    let formatter = match PromptFormatter::from_mdc(&mdc) {
        Ok(PromptFormatter::OAI(formatter)) => formatter,
        _ => {
            println!("Skipping template test - unable to create OAI formatter");
            return;
        }
    };

    // Test cases: both field names with enable_thinking true/false
    let test_cases = [
        (r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }"#, false, "no template args"),
        
        (r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_args": {
                "enable_thinking": false
            }
        }"#, false, "chat_template_args with false"),
        
        (r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_args": {
                "enable_thinking": true
            }
        }"#, true, "chat_template_args with true"),
        
        (r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "enable_thinking": false
            }
        }"#, false, "chat_template_kwargs with false"),
        
        (r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": {
                "enable_thinking": true
            }
        }"#, true, "chat_template_kwargs with true"),
    ];

    for (json_request, expect_thinking, description) in test_cases {
        // Parse request from JSON
        let request: NvCreateChatCompletionRequest = serde_json::from_str(json_request)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", description, e));
        
        // Render template
        let output = formatter.render(&request)
            .unwrap_or_else(|e| panic!("Failed to render {}: {}", description, e));
        
        // Verify output based on enable_thinking value
        if expect_thinking {
            assert!(output.contains("TEMPLATE_TEST_THINKING|"), 
                "Expected _THINKING in output for {}, got: {}", description, output);
        } else {
            assert!(output.contains("TEMPLATE_TEST|"), 
                "Expected base template marker for {}, got: {}", description, output);
            assert!(!output.contains("_THINKING"), 
                "Unexpected _THINKING in output for {}, got: {}", description, output);
        }
        
        // All should contain the message content
        assert!(output.contains("Hello"), 
            "Expected 'Hello' in output for {}, got: {}", description, output);
        
        println!("✓ {} - output: {}", description, output.trim());
    }
}