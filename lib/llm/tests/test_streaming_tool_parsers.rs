// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*

- Primary reason these tests were added is because we wanted to iterate quickly
  with concrete examples (rather than speculative fixtures), these tests catch
  regressions caused by backend chunk boundaries or minor field differences even
  when the overall protocol is the same.

- The "vllm" / "sglang" labels are not parser-specific logic. They only indicate
  the recorded source of the streaming chunks under tests/data. Different serving
  frameworks can vary chunk granularity and some envelope details (e.g., TRT-LLM
  often emits bigger deltas). Our parsing must be robust to these variations, so
  we validate against multiple real-world backends.

- These tests run through our full streaming parsing pipeline. We feed captured,
  production-like chunks into tool call parsing, then assert the aggregated
  reasoning content, final content, and tool-calls. This provides broader
  coverage than narrowly scoped unit tests of helpers and gives quick confidence
  when we tweak parsers (Harmony/Hermes/Qwen/Nemotron, etc.).

- To add another backend (e.g., trt-llm), record its streams under
tests/data/<backend>/... and mirror one of the existing tests so invariants hold
across backends.

*/

use dynamo_async_openai::types::{ChatChoiceStream, FinishReason};
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

    // Extract expected values (supports both new and legacy formats)
    let expected = parsed_json
        .get("expected_output")
        .expect("No 'expected_output' object found in JSON");

    let expected_normal_content = expected
        .get("normal_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let expected_reasoning_content = expected
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let expected_tool_calls = expected
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Extract the data chunks with choices from new `input_stream`
    let data_chunks = parsed_json
        .get("input_stream")
        .and_then(|v| v.as_array())
        .expect("No 'input_stream' array found in JSON");

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
                nvext: None,
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

/// Helper function to validate finish_reason in the stream
/// Returns true if:
/// 1. There is exactly one finish_reason in the entire stream
/// 2. The finish_reason is in the last chunk
/// 3. The finish_reason matches the expected value
fn validate_finish_reason(
    chunks: &[Annotated<NvCreateChatCompletionStreamResponse>],
    expected_finish_reason: FinishReason,
) -> bool {
    let mut finish_reason_count = 0;
    let mut last_chunk_index = None;
    let mut finish_reason_value = None;

    // Count finish_reason occurrences and track position
    for (idx, chunk) in chunks.iter().enumerate() {
        if let Some(ref response_data) = chunk.data {
            for choice in &response_data.choices {
                if let Some(reason) = choice.finish_reason {
                    finish_reason_count += 1;
                    last_chunk_index = Some(idx);
                    finish_reason_value = Some(reason);
                }
            }
        }
    }

    // Validate:
    // 1. Exactly one finish_reason in the stream
    if finish_reason_count != 1 {
        eprintln!(
            "Expected exactly 1 finish_reason, but found {}",
            finish_reason_count
        );
        return false;
    }

    // 2. finish_reason is in the last chunk
    if let Some(idx) = last_chunk_index {
        if idx != chunks.len() - 1 {
            eprintln!(
                "Expected finish_reason in last chunk (index {}), but found at index {}",
                chunks.len() - 1,
                idx
            );
            return false;
        }
    } else {
        eprintln!("No finish_reason found in stream");
        return false;
    }

    // 3. finish_reason matches expected value
    if let Some(reason) = finish_reason_value
        && reason != expected_finish_reason
    {
        eprintln!(
            "Expected finish_reason {:?}, but found {:?}",
            expected_finish_reason, reason
        );
        return false;
    }

    true
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Stop
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::Stop),
            "finish_reason validation failed for non-tool call case"
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Stop
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::Stop),
            "finish_reason validation failed for non-tool call case"
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
    }

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_no_tool_calls_sglang() {
        // SGLang Parsing test for GPT-OSS without tool calls.

        let file_path = format!(
            "{}/sglang/gpt-oss-20b/chat_completion_stream_675195a8-no-tool.json",
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

        // Expect content and reasoning present, no tool calls
        assert!(
            !aggregated.normal_content.is_empty(),
            "Should have normal content for no-tool case"
        );
        assert!(
            !aggregated.reasoning_content.is_empty(),
            "Should have reasoning content parsed from analysis channel"
        );
        assert!(
            !aggregated.has_tool_calls,
            "Should not have tool calls in no-tool case"
        );

        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Reasoning content should match expected value.",
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value.",
        );

        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Stop
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::Stop),
            "finish_reason validation failed for non-tool call case"
        );
    }

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_tool_calls_sglang() {
        // SGLang Parsing test for GPT-OSS with tool calls.

        let file_path = format!(
            "{}/sglang/gpt-oss-20b/chat_completion_stream_19c97899-tool.json",
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

        // Expect reasoning parsed, no normal content, and tool calls present
        assert!(
            !aggregated.reasoning_content.is_empty(),
            "Should have extracted reasoning content from analysis channel. Got: '{}'",
            aggregated.reasoning_content
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value.",
        );

        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Reasoning content should match expected value.",
        );

        // Verify tool calls presence and values
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );

        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
    }

    #[tokio::test]
    async fn test_qwen_e2e_with_no_tools_sglang() {
        // SGLang Parsing test for Qwen with no tools.

        let file_path = format!(
            "{}/sglang/qwen3-0.6B/chat_completion_stream_f121d1ca-no-tool.json",
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

        // Expect both reasoning and normal content (final answer) present, and no tool calls
        assert!(
            !aggregated.reasoning_content.is_empty(),
            "Should have extracted reasoning content."
        );
        assert!(
            !aggregated.normal_content.is_empty(),
            "Should have final normal content."
        );
        assert!(!aggregated.has_tool_calls, "Tool calls should be absent");

        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Reasoning content should match expected value.",
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value.",
        );

        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Stop
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::Stop),
            "finish_reason validation failed for non-tool call case"
        );
    }

    #[tokio::test]
    async fn test_qwen_e2e_with_tools_sglang() {
        // SGLang Parsing test for Qwen with tools.

        let file_path = format!(
            "{}/sglang/qwen3-0.6B/chat_completion_stream_c42ba578-tool.json",
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

        // Expect reasoning parsed, no normal content, and tool calls present
        assert!(
            !aggregated.reasoning_content.is_empty(),
            "Should have extracted reasoning content."
        );

        assert_eq!(
            aggregated.reasoning_content, test_data.expected_reasoning_content,
            "Reasoning content should match expected value.",
        );

        assert_eq!(
            aggregated.normal_content, test_data.expected_normal_content,
            "Normal content should match expected value.",
        );

        // Verify tool calls presence and values
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            aggregated.has_tool_calls, expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );
        assert_tool_calls(&aggregated.tool_calls, &test_data.expected_tool_calls);

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
    }

    #[tokio::test]
    async fn test_qwen_finish_reason_length_vllm() {
        let file_paths = vec![
            format!(
                "{}/vllm/qwen3-0.6B/chat_completion_stream_finish_length.json",
                DATA_ROOT_PATH
            ),
            format!(
                "{}/vllm/qwen3-0.6B/chat_completion_incomplete_tool.json",
                DATA_ROOT_PATH
            ),
        ];

        for file_path in file_paths {
            let test_data = load_test_data(&file_path);

            // Create a stream from the mock chunks
            let input_stream = stream::iter(test_data.stream_chunks);

            // Parse the response stream with tool parsing enabled
            let output_chunks =
                parse_response_stream(input_stream, true, false, Some("hermes".to_string()), None)
                    .await;

            // Verify we got output chunks
            assert!(!output_chunks.is_empty(), "Should have output chunks");

            // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Length
            assert!(
                validate_finish_reason(&output_chunks, FinishReason::Length),
                "finish_reason validation failed for length finish case"
            );
        }
    }

    #[tokio::test]
    async fn test_deepseek_v3_e2e_with_tools_vllm() {
        // E2E Parsing test for DeepSeek V3 with tools.
        let file_path = format!(
            "{}/vllm/deepseek-v3/chat_completion_stream_tool.json",
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
            Some("deepseek_v3".to_string()),
            Some("deepseek_v3".to_string()),
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
    }

    #[tokio::test]
    async fn test_deepseek_v3_1_e2e_with_tools_vllm() {
        // E2E Parsing test for DeepSeek V3.1 with tools.
        let file_path = format!(
            "{}/vllm/deepseek-v3.1/chat_completion_stream_tool.json",
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
            Some("deepseek_v3_1".to_string()),
            Some("deepseek_v3_1".to_string()),
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

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is ToolCalls
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::ToolCalls),
            "finish_reason validation failed for tool call case"
        );
    }

    #[tokio::test]
    async fn test_deepseek_v3_e2e_with_no_tools_vllm() {
        // E2E Parsing test for DeepSeek V3 without tools.
        let file_path = format!(
            "{}/vllm/deepseek-v3/chat_completion_stream_no_tool.json",
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
            Some("deepseek_v3".to_string()),
            Some("deepseek_v3".to_string()),
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

        // Verify no tool calls
        assert!(!aggregated.has_tool_calls, "Should not have any tool calls");

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Stop
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::Stop),
            "finish_reason validation failed for non-tool call case"
        );
    }

    #[tokio::test]
    async fn test_deepseek_v3_1_e2e_with_no_tools_vllm() {
        // E2E Parsing test for DeepSeek V3.1 without tools.
        let file_path = format!(
            "{}/vllm/deepseek-v3.1/chat_completion_stream_no_tool.json",
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
            Some("deepseek_v3_1".to_string()),
            Some("deepseek_v3_1".to_string()),
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

        // Verify no tool calls
        assert!(!aggregated.has_tool_calls, "Should not have any tool calls");

        // Verify finish_reason is valid: exactly one occurrence, in last chunk, and is Stop
        assert!(
            validate_finish_reason(&output_chunks, FinishReason::Stop),
            "finish_reason validation failed for non-tool call case"
        );
    }
}
