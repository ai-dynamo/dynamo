// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::ChatChoiceStream;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt, stream};
use std::pin::Pin;

const DATA_ROOT_PATH: &str = "tests/data/";

/// Test data structure containing expected results and stream data
struct TestData {
    expected_normal_content: String,
    expected_reasoning_content: String,
    expected_tool_calls: Vec<serde_json::Value>,
    stream_chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
}

/// Helper function to load test data from a test data file
fn load_test_data(file_path: &str) -> TestData {
    // Read the data from file
    let data = std::fs::read_to_string(file_path).unwrap();

    // Parse the file as JSON
    let parsed_json: serde_json::Value = serde_json::from_str(&data).unwrap();

    // Extract expected values
    let expected_normal_content = parsed_json
        .get("normal_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let expected_reasoning_content = parsed_json
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let expected_tool_calls = parsed_json
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Extract the data chunks with choices
    let data_chunks = parsed_json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("No 'data' array found in JSON");

    let stream_chunks = data_chunks
        .iter()
        .map(|chunk| {
            let inner_data = chunk.get("data").expect("No 'data' field in chunk");

            let id = inner_data
                .get("id")
                .and_then(|v| v.as_str())
                .expect("No 'id' field")
                .to_string();

            let choices: Vec<ChatChoiceStream> = serde_json::from_value(
                inner_data
                    .get("choices")
                    .cloned()
                    .expect("No 'choices' field"),
            )
            .expect("Failed to parse choices");

            let response = NvCreateChatCompletionStreamResponse {
                id: id.clone(),
                choices,
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
            };

            Annotated {
                id: Some(id),
                data: Some(response),
                event: None,
                comment: None,
            }
        })
        .collect();

    TestData {
        expected_normal_content,
        expected_reasoning_content,
        expected_tool_calls,
        stream_chunks,
    }
}

/// Helper function to parse response stream with optional reasoning and tool parsing
async fn parse_response_stream(
    stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send + 'static,
    tool_parse_enable: bool,
    reasoning_enable: bool,
    tool_parser_str: Option<String>,
    reasoning_parser_str: Option<String>,
) -> Vec<Annotated<NvCreateChatCompletionStreamResponse>> {
    // Apply reasoning parser if enabled
    let stream: Pin<
        Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>,
    > = if reasoning_enable {
        if let Some(reasoning_parser) = reasoning_parser_str {
            Box::pin(OpenAIPreprocessor::parse_reasoning_content_from_stream(
                stream,
                reasoning_parser,
            ))
        } else {
            Box::pin(stream)
        }
    } else {
        Box::pin(stream)
    };

    // Apply tool calling parser if enabled
    let stream: Pin<
        Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>,
    > = if tool_parse_enable {
        if let Some(tool_parser) = tool_parser_str {
            Box::pin(OpenAIPreprocessor::apply_tool_calling_jail(
                tool_parser,
                stream,
            ))
        } else {
            Box::pin(stream)
        }
    } else {
        Box::pin(stream)
    };

    // Collect all output chunks
    let mut stream = std::pin::pin!(stream);
    let mut output_chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        output_chunks.push(chunk);
    }

    output_chunks
}

/// Structure to hold aggregated results from chunks
struct AggregatedContent {
    reasoning_content: String,
    normal_content: String,
    has_tool_calls: bool,
    tool_calls: Vec<serde_json::Value>,
}

/// Helper function to assert tool calls match expected (ignoring random IDs)
fn assert_tool_calls(
    actual_tool_calls: &[serde_json::Value],
    expected_tool_calls: &[serde_json::Value],
) {
    assert_eq!(actual_tool_calls.len(), expected_tool_calls.len());

    if !expected_tool_calls.is_empty() {
        let actual_fn = &actual_tool_calls[0]["function"];
        let expected_fn = &expected_tool_calls[0]["function"];

        let actual_name = actual_fn["name"].as_str().unwrap();
        let expected_name = expected_fn["name"].as_str().unwrap();
        assert_eq!(actual_name, expected_name);

        let actual_args: serde_json::Value =
            serde_json::from_str(actual_fn["arguments"].as_str().unwrap()).unwrap();
        let expected_args: serde_json::Value =
            serde_json::from_str(expected_fn["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(actual_args, expected_args);
    }
}

/// Helper function to aggregate all content types from chunks
fn aggregate_content_from_chunks(
    chunks: &[Annotated<NvCreateChatCompletionStreamResponse>],
) -> AggregatedContent {
    let mut reasoning_content = String::new();
    let mut normal_content = String::new();
    let mut has_tool_calls = false;
    let mut tool_calls = Vec::new();

    for chunk in chunks.iter() {
        if let Some(ref response_data) = chunk.data {
            for choice in &response_data.choices {
                // Collect reasoning content
                if let Some(ref reasoning) = choice.delta.reasoning_content {
                    reasoning_content.push_str(reasoning);
                }

                // Collect normal content
                if let Some(ref content) = choice.delta.content {
                    normal_content.push_str(content);
                }

                // Collect tool calls
                if let Some(ref chunk_tool_calls) = choice.delta.tool_calls {
                    has_tool_calls = true;
                    if let Ok(json_array) = serde_json::to_value(chunk_tool_calls)
                        && let Some(array) = json_array.as_array()
                    {
                        tool_calls.extend(array.iter().cloned());
                    }
                }
            }
        }
    }

    AggregatedContent {
        reasoning_content,
        normal_content,
        has_tool_calls,
        tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_no_tool_calls_vllm() {
        // E2E Parsing test for GPT-OSS. The input stream does not contain tool calls.
        // Just content and reasoning content.
        // Test will call both reasoning parsing logic and tool calling parsing logic and verify the output

        // Load test data from file
        let file_path = format!(
            "{}/vllm/gpt-oss-20b/chat_completion_stream_49f581c1-no-tool.json",
            DATA_ROOT_PATH
        );
        let test_data = load_test_data(&file_path);

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing enabled
        let output_chunks = parse_response_stream(
            input_stream,
            true,
            true,
            Some("harmony".to_string()),
            Some("gpt_oss".to_string()),
        )
        .await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from all chunks
        let aggregated = aggregate_content_from_chunks(&output_chunks);

        // Verify against expected content from test file
        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Reasoning content should match expected value"
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value"
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );
    }

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_tool_calls_vllm() {
        // E2E Parsing test for GPT-OSS. The input stream contains tool calls.
        // Test will call both reasoning parsing logic and tool calling parsing logic and verify the output

        // Load test data from file
        let file_path = format!(
            "{}/vllm/gpt-oss-20b/chat_completion_stream_f0c86d72-tool.json",
            DATA_ROOT_PATH
        );
        let test_data = load_test_data(&file_path);

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing enabled
        let output_chunks = parse_response_stream(
            input_stream,
            true,
            true,
            Some("harmony".to_string()),
            Some("gpt_oss".to_string()),
        )
        .await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from all chunks
        let aggregated = aggregate_content_from_chunks(&output_chunks);

        // Assert reasoning content was parsed
        assert!(
            !aggregated.reasoning_content.is_empty(),
            "Should have extracted reasoning content from analysis channel. Got: '{}'",
            aggregated.reasoning_content
        );

        // Assert normal content was parsed
        assert!(
            aggregated.normal_content.is_empty(),
            "Normal content should be empty. Got: '{}'",
            aggregated.normal_content
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );

        // Verify tool calls
        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);
    }

    #[tokio::test]
    async fn test_qwen_e2e_with_no_tools_vllm() {
        // E2E Parsing test for Qwen with no tools.

        let file_path = format!(
            "{}/vllm/qwen3-0.6B/chat_completion_stream_5627a4c6-no-tool.json",
            DATA_ROOT_PATH
        );
        let test_data = load_test_data(&file_path);

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing disabled
        let output_chunks = parse_response_stream(
            input_stream,
            true,
            true,
            Some("hermes".to_string()),
            Some("qwen".to_string()),
        )
        .await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from output chunks
        let aggregated = aggregate_content_from_chunks(&output_chunks);

        // Assert that output content matches input content exactly (no parsing applied)
        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "When parsing is disabled, output should match input exactly"
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );
    }

    #[tokio::test]
    async fn test_qwen_e2e_with_tools_vllm() {
        // E2E Parsing test for Qwen with tools.
        // Test will call both reasoning parsing logic and tool calling parsing logic and verify the output

        let file_path = format!(
            "{}/vllm/qwen3-0.6B/chat_completion_stream_8f33c28b-tool.json",
            DATA_ROOT_PATH
        );
        let test_data = load_test_data(&file_path);

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing enabled
        let output_chunks = parse_response_stream(
            input_stream,
            true,
            true,
            Some("hermes".to_string()),
            Some("qwen".to_string()),
        )
        .await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from output chunks
        let aggregated = aggregate_content_from_chunks(&output_chunks);

        // Assert reasoning content was parsed
        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Should have extracted reasoning content.",
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value.",
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );

        // Verify tool calls
        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);
    }

    #[tokio::test]
    async fn test_nemotron_e2e_with_tools_vllm() {
        // E2E Parsing test for Nemotron with tools.
        // Test will call both reasoning parsing logic and tool calling parsing logic and verify the output

        let file_path = format!(
            "{}/vllm/nemotron-49b/chat_completion_stream_3d40f925-tool.json",
            DATA_ROOT_PATH
        );
        let test_data = load_test_data(&file_path);

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing enabled
        let output_chunks = parse_response_stream(
            input_stream,
            true,
            true,
            Some("nemotron_deci".to_string()),
            Some("nemotron_deci".to_string()),
        )
        .await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from output chunks
        let aggregated = aggregate_content_from_chunks(&output_chunks);

        // Assert reasoning content was parsed
        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Should have extracted reasoning content.",
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value.",
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );

        // Verify tool calls
        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);
    }
}
