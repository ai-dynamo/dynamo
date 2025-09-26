// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for parallel tool calling functionality
//!
//! This test simulates a complete chat completion request with parallel tool calls,
//! mocking the response and testing the tool call parsing functionality.
//!
//! The test covers:
//! - Creating a mock NvCreateChatCompletionRequest based on a curl request
//! - Mocking a chat completion response with parallel tool calls in <tool_call> format
//! - Parsing the tool calls using the hermes parser
//! - Validating OpenAI API compatibility
//! - Testing error handling with malformed content
//! - Ensuring tool call IDs are unique and properly formatted

use dynamo_llm::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, common_ext::CommonExt,
};
use dynamo_parsers::{ToolCallResponse, ToolCallType, detect_and_parse_tool_call};
use serde_json::json;

/// Creates a mock NvCreateChatCompletionRequest based on the curl request
fn create_mock_chat_completion_request() -> NvCreateChatCompletionRequest {
    let messages = vec![
        dynamo_async_openai::types::ChatCompletionRequestMessage::System(
            dynamo_async_openai::types::ChatCompletionRequestSystemMessage {
                content: dynamo_async_openai::types::ChatCompletionRequestSystemMessageContent::Text(
                    "You MUST use two tools in parallel to resolve the user request: call get_current_weather for each city AND call is_holiday_today to check if today is a holiday. Do not answer without using both tools.".to_string()
                ),
                name: None,
            }
        ),
        dynamo_async_openai::types::ChatCompletionRequestMessage::User(
            dynamo_async_openai::types::ChatCompletionRequestUserMessage {
                content: dynamo_async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    "What is the weather in Dallas, Texas? Is today a holiday?".to_string()
                ),
                name: None,
            }
        ),
    ];

    let tools = vec![
        dynamo_async_openai::types::ChatCompletionTool {
            r#type: dynamo_async_openai::types::ChatCompletionToolType::Function,
            function: dynamo_async_openai::types::FunctionObject {
                name: "get_current_weather".to_string(),
                description: Some("Get weather for a city/state in specified units".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name, e.g., Dallas" },
                        "state": { "type": "string", "description": "Two-letter state code, e.g., TX" },
                        "unit": { "type": "string", "enum": ["fahrenheit", "celsius"] }
                    },
                    "required": ["city", "state", "unit"],
                    "additionalProperties": false
                })),
                strict: None,
            },
        },
        dynamo_async_openai::types::ChatCompletionTool {
            r#type: dynamo_async_openai::types::ChatCompletionToolType::Function,
            function: dynamo_async_openai::types::FunctionObject {
                name: "is_holiday_today".to_string(),
                description: Some("Return whether today is a public holiday".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })),
                strict: None,
            },
        },
    ];

    let inner = dynamo_async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("Qwen/Qwen3-0.6B")
        .temperature(0.0)
        .max_tokens(3000u32)
        .stream(false)
        .messages(messages)
        .tools(tools)
        .tool_choice(dynamo_async_openai::types::ChatCompletionToolChoiceOption::Required)
        .build()
        .expect("Failed to build chat completion request");

    NvCreateChatCompletionRequest {
        inner,
        common: CommonExt::default(),
        nvext: None,
        chat_template_args: None,
    }
}

/// Mock response content that contains parallel tool calls
fn get_mock_response_content() -> String {
    r#"<think>Okay, the user is asking two things: the weather in Dallas, Texas, and whether today is a holiday. I need to use both tools here. First, I'll check the weather using get_current_weather with city Dallas and state Texas. Then, I'll use is_holiday_today to see if today is a public holiday. I have to make sure to call both functions in parallel. Let me structure the tool calls properly.</think>

<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
</tool_call>

<tool_call>
{"name": "is_holiday_today", "arguments": {}}
</tool_call>"#.to_string()
}

/// Validates that a tool call response matches expected values
fn validate_weather_tool_call(tool_call: &ToolCallResponse) {
    assert_eq!(tool_call.function.name, "get_current_weather");

    let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
        .expect("Arguments should be valid JSON");
    let args_obj = args.as_object().expect("Arguments should be an object");
    assert_eq!(args_obj.get("city").unwrap().as_str().unwrap(), "Dallas");
    assert_eq!(args_obj.get("state").unwrap().as_str().unwrap(), "TX");
    assert_eq!(
        args_obj.get("unit").unwrap().as_str().unwrap(),
        "fahrenheit"
    );

    // Validate OpenAI compatibility
    assert!(!tool_call.id.is_empty(), "Tool call should have an ID");
    assert_eq!(tool_call.tp, ToolCallType::Function);
}

/// Validates that a holiday tool call response matches expected values
fn validate_holiday_tool_call(tool_call: &ToolCallResponse) {
    assert_eq!(tool_call.function.name, "is_holiday_today");

    let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
        .expect("Arguments should be valid JSON");
    let args_obj = args.as_object().expect("Arguments should be an object");
    assert!(
        args_obj.is_empty(),
        "Holiday tool should have empty arguments"
    );

    // Validate OpenAI compatibility
    assert!(!tool_call.id.is_empty(), "Tool call should have an ID");
    assert_eq!(tool_call.tp, ToolCallType::Function);
}

/// Validates that tool call IDs are unique
fn validate_unique_tool_call_ids(tool_calls: &[ToolCallResponse]) {
    let mut ids = std::collections::HashSet::new();
    for tool_call in tool_calls {
        assert!(
            ids.insert(tool_call.id.clone()),
            "Tool call IDs should be unique: {}",
            tool_call.id
        );
    }
}

#[tokio::test]
async fn test_parallel_tool_call_integration() {
    // Create the mock request
    let request = create_mock_chat_completion_request();

    // Validate request structure
    assert_eq!(request.inner.model, "Qwen/Qwen3-0.6B");
    assert_eq!(request.inner.temperature, Some(0.0));
    #[allow(deprecated)]
    {
        assert_eq!(request.inner.max_tokens, Some(3000));
    }
    assert_eq!(request.inner.stream, Some(false));
    assert_eq!(request.inner.messages.len(), 2);
    assert_eq!(request.inner.tools.as_ref().unwrap().len(), 2);

    // Verify tool choice is required
    match request.inner.tool_choice.as_ref().unwrap() {
        dynamo_async_openai::types::ChatCompletionToolChoiceOption::Required => {
            // This is expected
        }
        _ => panic!("Tool choice should be Required"),
    }

    // Get the mock response content
    let response_content = get_mock_response_content();

    // Verify the response contains both tool calls
    assert!(response_content.contains("get_current_weather"));
    assert!(response_content.contains("is_holiday_today"));
    assert!(response_content.contains("Dallas"));
    assert!(response_content.contains("Texas"));
    assert!(response_content.contains("fahrenheit"));
}

#[tokio::test]
async fn test_parallel_tool_call_parsing() {
    let response_content = get_mock_response_content();

    // Parse the tool calls using the hermes parser (works well with <tool_call> format)
    let (tool_calls, remaining_content) =
        detect_and_parse_tool_call(&response_content, Some("hermes"))
            .await
            .expect("Should successfully parse tool calls");

    // Validate we got exactly 2 tool calls
    assert_eq!(
        tool_calls.len(),
        2,
        "Should parse exactly 2 parallel tool calls"
    );

    // Validate remaining content (should be the thinking part)
    assert!(remaining_content.is_some());
    let remaining = remaining_content.unwrap();
    assert!(remaining.contains("<think>"));
    assert!(remaining.contains("</think>"));

    // Sort tool calls by name for consistent testing
    let mut sorted_calls = tool_calls;
    sorted_calls.sort_by(|a, b| a.function.name.cmp(&b.function.name));

    // Validate the weather tool call (first alphabetically)
    validate_weather_tool_call(&sorted_calls[0]);

    // Validate the holiday tool call (second alphabetically)
    validate_holiday_tool_call(&sorted_calls[1]);

    // Validate tool call IDs are unique
    validate_unique_tool_call_ids(&sorted_calls);
}

#[tokio::test]
async fn test_parallel_tool_call_with_explicit_parser() {
    let response_content = get_mock_response_content();

    // Test with explicit parser selection
    let parsers_to_test = vec![
        "hermes", // Should work well with this format
    ];

    for parser in parsers_to_test {
        let (tool_calls, remaining_content) =
            detect_and_parse_tool_call(&response_content, Some(parser))
                .await
                .unwrap_or_else(|e| panic!("Should successfully parse with {parser} parser: {e}"));

        // Should get 2 tool calls regardless of parser
        assert_eq!(
            tool_calls.len(),
            2,
            "Parser {parser} should find 2 tool calls"
        );

        // Validate remaining content exists
        assert!(remaining_content.is_some());

        // Sort and validate calls
        let mut sorted_calls = tool_calls;
        sorted_calls.sort_by(|a, b| a.function.name.cmp(&b.function.name));

        validate_weather_tool_call(&sorted_calls[0]);
        validate_holiday_tool_call(&sorted_calls[1]);
        validate_unique_tool_call_ids(&sorted_calls);
    }
}

#[tokio::test]
async fn test_tool_call_json_structure() {
    let response_content = get_mock_response_content();

    let (tool_calls, _) = detect_and_parse_tool_call(&response_content, Some("hermes"))
        .await
        .expect("Should parse tool calls");

    // Test JSON serialization
    for tool_call in &tool_calls {
        let json_str =
            serde_json::to_string(tool_call).expect("Tool call should serialize to JSON");

        // Verify the JSON contains expected fields
        assert!(json_str.contains("\"id\""));
        assert!(json_str.contains("\"type\""));
        assert!(json_str.contains("\"function\""));
        assert!(json_str.contains(&tool_call.function.name));
    }
}

#[tokio::test]
async fn test_openai_compatibility_structure() {
    let response_content = get_mock_response_content();

    let (tool_calls, _) = detect_and_parse_tool_call(&response_content, Some("hermes"))
        .await
        .expect("Should parse tool calls");

    // Validate OpenAI API compatibility
    for tool_call in &tool_calls {
        // Should have all required OpenAI fields
        assert!(!tool_call.id.is_empty(), "Missing required 'id' field");
        assert_eq!(
            tool_call.tp,
            ToolCallType::Function,
            "Type should be 'function'"
        );
        assert!(
            !tool_call.function.name.is_empty(),
            "Function name should not be empty"
        );

        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
            .expect("Arguments should be valid JSON");
        assert!(args.is_object(), "Arguments should be an object");

        // ID should follow expected format (call-XXXXXXXX or call_XXXXXXXX)
        assert!(
            tool_call.id.starts_with("call-") || tool_call.id.starts_with("call_"),
            "ID should start with 'call-' or 'call_': {}",
            tool_call.id
        );
        assert!(
            tool_call.id.len() > 5,
            "ID should be longer than just 'call': {}",
            tool_call.id
        );
    }
}

#[tokio::test]
async fn test_parallel_tool_call_error_handling() {
    // Test with malformed content
    let malformed_content = r#"<tool_call>
{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}
</tool_call>

<tool_call>
{"invalid_json": }
</tool_call>"#;

    let result = detect_and_parse_tool_call(malformed_content, Some("hermes")).await;

    // Should handle partial parsing gracefully
    match result {
        Ok((tool_calls, _)) => {
            // May parse valid tool calls and ignore malformed ones, or return empty
            println!(
                "Parsed {} tool calls from malformed content",
                tool_calls.len()
            );

            if !tool_calls.is_empty() {
                // If any were parsed, verify they're valid
                for call in &tool_calls {
                    assert!(
                        !call.function.name.is_empty(),
                        "Parsed tool call should have valid name"
                    );
                }
            }
        }
        Err(e) => {
            // Error handling is also acceptable for malformed input
            println!("Expected error for malformed input: {}", e);
        }
    }
}

#[tokio::test]
async fn test_empty_tool_calls() {
    let content_without_tools = "This is just a regular response without any tool calls.";

    let (tool_calls, remaining_content) =
        detect_and_parse_tool_call(content_without_tools, Some("hermes"))
            .await
            .expect("Should handle content without tool calls");

    assert!(
        tool_calls.is_empty(),
        "Should return empty tool calls array"
    );
    assert!(
        remaining_content.is_some(),
        "Should return the original content"
    );
    assert_eq!(remaining_content.unwrap(), content_without_tools);
}
