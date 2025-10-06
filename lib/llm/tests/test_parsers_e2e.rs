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
                inner_data.get("choices").cloned().expect("No 'choices' field")
            ).expect("Failed to parse choices");

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
    let stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> = 
        if reasoning_enable {
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
    let stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> = 
        if tool_parse_enable {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_no_tool_calls_vllm() {
        // E2E Parsing test for GPT-OSS. The input stream does not contain tool calls. 
        // Just content and reasoning content.    
        // Test will call both reasoning parsing logic and tool calling parsing logic and verify the output
        
        // Load test data from file
        let file_path = format!("{}/vllm/gpt-oss-20b/chat_completion_stream_49f581c1-no-tool.json", DATA_ROOT_PATH);
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
        ).await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from all chunks
        let mut all_reasoning = String::new();
        let mut all_normal_content = String::new();
        let mut found_tool_calls = false;

        for chunk in output_chunks.iter() {
            if let Some(ref response_data) = chunk.data {
                for choice in &response_data.choices {
                    // Collect reasoning content
                    if let Some(ref reasoning) = choice.delta.reasoning_content {
                        all_reasoning.push_str(reasoning);
                    }

                    // Collect normal content
                    if let Some(ref content) = choice.delta.content {
                        all_normal_content.push_str(content);
                    }

                    // Check for tool calls
                    if let Some(ref tool_calls) = choice.delta.tool_calls
                        && !tool_calls.is_empty()
                    {
                        found_tool_calls = true;
                    }
                }
            }
        }

        // Assert reasoning content was parsed (GPT-OSS has reasoning in analysis channel)
        assert!(
            !all_reasoning.is_empty(),
            "Should have extracted reasoning content from analysis channel. Got: '{}'",
            all_reasoning
        );

        // Assert normal content was parsed (from final channel)
        assert!(
            !all_normal_content.is_empty(),
            "Should have extracted normal content from final channel. Got: '{}'",
            all_normal_content
        );

        // Verify against expected content from test file
        assert_eq!(
            all_reasoning.trim(),
            test_data.expected_reasoning_content.trim(),
            "Reasoning content should match expected value"
        );

        assert_eq!(
            all_normal_content.trim(),
            test_data.expected_normal_content.trim(),
            "Normal content should match expected value"
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            found_tool_calls,
            expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );
    }

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_tool_calls_vllm() {
        // E2E Parsing test for GPT-OSS. The input stream contains tool calls. 
        // Just content and reasoning content.    
        // Test will call both reasoning parsing logic and tool calling parsing logic and verify the output
        
        // Load test data from file
        let file_path = format!("{}/vllm/gpt-oss-20b/chat_completion_stream_f0c86d72-tool.json", DATA_ROOT_PATH);
        let test_data = load_test_data(&file_path);

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing enabled
        let output_chunks = parse_response_stream(
            input_stream,
            true,  // tool_parse_enable
            true,  // reasoning_enable
            Some("harmony".to_string()),  // tool_parser_str
            Some("gpt_oss".to_string()),  // reasoning_parser_str
        ).await;

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from all chunks
        let mut all_reasoning = String::new();
        let mut all_normal_content = String::new();
        let mut found_tool_calls = false;

        for chunk in output_chunks.iter() {
            if let Some(ref response_data) = chunk.data {
                for choice in &response_data.choices {
                    // Collect reasoning content
                    if let Some(ref reasoning) = choice.delta.reasoning_content {
                        all_reasoning.push_str(reasoning);
                    }
        
                    // Collect normal content
                    if let Some(ref content) = choice.delta.content {
                        all_normal_content.push_str(content);
                    }

                    // Check for tool calls
                    if let Some(ref tool_calls) = choice.delta.tool_calls
                        && !tool_calls.is_empty()
                    {
                        found_tool_calls = true;
                    }
                }
            }
        }
        
        // Assert reasoning content was parsed (GPT-OSS has reasoning in analysis channel)
        assert!(
            !all_reasoning.is_empty(),
            "Should have extracted reasoning content from analysis channel. Got: '{}'",
            all_reasoning
        );

        // Assert normal content was parsed (from final channel)
        assert!(
            all_normal_content.is_empty(),
            "Normal content should be empty. Got: '{}'",
            all_normal_content
        );

        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            found_tool_calls,
            expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );
    }

    #[tokio::test]
    async fn test_gpt_oss_e2e_with_no_parsing_vllm(){
        // E2E Parsing test for GPT-OSS with no parsing enabled.
        // When parsing is disabled, output should match input exactly.

        let file_path = format!("{}/vllm/gpt-oss-20b/chat_completion_stream_49f581c1-no-tool.json", DATA_ROOT_PATH);
        let test_data = load_test_data(&file_path);

        // First, accumulate content from input chunks to get expected raw content
        let mut expected_raw_content = String::new();
        for chunk in test_data.stream_chunks.iter() {
            if let Some(ref response_data) = chunk.data {
                for choice in &response_data.choices {
                    if let Some(ref content) = choice.delta.content {
                        expected_raw_content.push_str(content);
                    }
                }
            }
        }

        // Create a stream from the mock chunks
        let input_stream = stream::iter(test_data.stream_chunks);

        // Parse the response stream with reasoning and tool parsing disabled
        let output_chunks = parse_response_stream(
            input_stream,
            false, 
            false, 
            None,
            None,  
        ).await;    

        // Verify we got output chunks
        assert!(!output_chunks.is_empty(), "Should have output chunks");

        // Aggregate content from output chunks
        let mut actual_output_content = String::new();
        let mut found_tool_calls = false;

        for chunk in output_chunks.iter() {
            if let Some(ref response_data) = chunk.data {
                for choice in &response_data.choices {
                    if let Some(ref content) = choice.delta.content {
                        actual_output_content.push_str(content);
                    }

                    // Check for tool calls
                    if let Some(ref tool_calls) = choice.delta.tool_calls
                        && !tool_calls.is_empty()
                    {
                        found_tool_calls = true;
                    }
                }
            }
        }
        
        // Assert that output content matches input content exactly (no parsing applied)
        assert_eq!(
            actual_output_content,
            expected_raw_content,
            "When parsing is disabled, output should match input exactly"
        );
        
        // Verify tool calls match expectations
        let expected_has_tool_calls = !test_data.expected_tool_calls.is_empty();
        assert_eq!(
            found_tool_calls,
            expected_has_tool_calls,
            "Tool calls presence should match expected value"
        );
    }

}