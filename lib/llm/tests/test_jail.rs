// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionStreamResponseDelta, FinishReason, Role,
};
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_llm::protocols::openai::chat_completions::jail::JailedStream;
use dynamo_runtime::protocols::annotated::Annotated;

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;

    // Test utilities module - shared test infrastructure
    pub(crate) mod test_utils {
        use super::*;

        /// Helper function to create a mock chat response chunk
        pub fn create_mock_response_chunk(
            content: String,
            index: u32,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(content),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
                nvext: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper function to create a final response chunk with finish reason
        pub fn create_final_response_chunk(
            index: u32,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
                nvext: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper function to create a mock chat response chunk with metadata
        pub fn create_annotated_chunk(
            content: String,
            index: u32,
            id: Option<String>,
            event: Option<String>,
            comment: Option<Vec<String>>,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            #[allow(deprecated)]
            let choice = ChatChoiceStream {
                index,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(content),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            };

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices: vec![choice],
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
                nvext: None,
            };

            Annotated {
                data: Some(response),
                id,
                event,
                comment,
            }
        }

        /// Helper function to create a multi-choice chunk
        pub fn create_multi_choice_chunk(
            choices_content: Vec<(String, u32)>, // (content, index)
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let choices: Vec<ChatChoiceStream> = choices_content
                .into_iter()
                .map(|(content, index)| {
                    #[allow(deprecated)]
                    ChatChoiceStream {
                        index,
                        delta: ChatCompletionStreamResponseDelta {
                            role: Some(Role::Assistant),
                            content: Some(content),
                            tool_calls: None,
                            function_call: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }
                })
                .collect();

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices,
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
                nvext: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper function to create a multi-choice finish_reason chunk
        pub fn create_multi_choice_finish_chunk(
            choice_indices: Vec<u32>,
        ) -> Annotated<NvCreateChatCompletionStreamResponse> {
            let choices: Vec<ChatChoiceStream> = choice_indices
                .into_iter()
                .map(|index| {
                    #[allow(deprecated)]
                    ChatChoiceStream {
                        index,
                        delta: ChatCompletionStreamResponseDelta {
                            role: None,
                            content: None,
                            tool_calls: None,
                            function_call: None,
                            refusal: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some(FinishReason::Stop),
                        logprobs: None,
                    }
                })
                .collect();

            let response = NvCreateChatCompletionStreamResponse {
                id: "test-id".to_string(),
                choices,
                created: 1234567890,
                model: "test-model".to_string(),
                system_fingerprint: Some("test-fingerprint".to_string()),
                object: "chat.completion.chunk".to_string(),
                usage: None,
                service_tier: None,
                nvext: None,
            };

            Annotated {
                data: Some(response),
                id: None,
                event: None,
                comment: None,
            }
        }

        /// Helper to assert content in a result
        pub fn assert_content(
            result: &Annotated<NvCreateChatCompletionStreamResponse>,
            expected: &str,
        ) {
            let content = result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .expect("Expected content in result");

            assert_eq!(
                content, expected,
                "Content mismatch: expected '{}', got '{}'",
                expected, content
            );
        }

        /// Helper to assert a tool call in a result
        pub fn assert_tool_call(
            result: &Annotated<NvCreateChatCompletionStreamResponse>,
            name: &str,
            args: serde_json::Value,
        ) {
            let tool_calls = result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .expect("Expected tool calls in result");

            assert!(!tool_calls.is_empty(), "Expected at least one tool call");

            let tool_call = &tool_calls[0];
            let function = tool_call
                .function
                .as_ref()
                .expect("Expected function in tool call");

            assert_eq!(
                function.name.as_deref(),
                Some(name),
                "Tool call name mismatch: expected '{}', got '{:?}'",
                name,
                function.name
            );

            if let Some(arguments_str) = &function.arguments {
                let parsed_args: serde_json::Value = serde_json::from_str(arguments_str)
                    .expect("Tool call arguments should be valid JSON");
                assert_eq!(
                    parsed_args, args,
                    "Tool call arguments mismatch: expected {}, got {}",
                    args, parsed_args
                );
            } else if !args.is_null() {
                panic!("Expected tool call arguments {} but got None", args);
            }
        }

        /// Helper to assert no content or tool calls (for accumulated chunks)
        #[allow(dead_code)]
        pub fn assert_empty_emission(result: &Annotated<NvCreateChatCompletionStreamResponse>) {
            if let Some(data) = &result.data
                && let Some(choice) = data.choices.first()
            {
                assert!(
                    choice.delta.content.is_none()
                        || choice.delta.content.as_ref().unwrap().is_empty(),
                    "Expected no content but got: {:?}",
                    choice.delta.content
                );
                assert!(
                    choice.delta.tool_calls.is_none()
                        || choice.delta.tool_calls.as_ref().unwrap().is_empty(),
                    "Expected no tool calls but got: {:?}",
                    choice.delta.tool_calls
                );
            }
        }

        /// Helper to reconstruct all content from results
        pub fn reconstruct_content(
            results: &[Annotated<NvCreateChatCompletionStreamResponse>],
        ) -> String {
            results
                .iter()
                .filter_map(|r| {
                    r.data
                        .as_ref()
                        .and_then(|d| d.choices.first())
                        .and_then(|c| c.delta.content.as_ref())
                })
                .cloned()
                .collect::<Vec<_>>()
                .join("")
        }

        /// Helper to extract content from a single result (for negative assertions)
        pub fn extract_content(result: &Annotated<NvCreateChatCompletionStreamResponse>) -> String {
            result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .cloned()
                .unwrap_or_default()
        }

        /// Helper to check if result contains a tool call
        pub fn has_tool_call(result: &Annotated<NvCreateChatCompletionStreamResponse>) -> bool {
            result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false)
        }

        /// Helper to check if result contains content
        #[allow(dead_code)]
        pub fn has_content(result: &Annotated<NvCreateChatCompletionStreamResponse>) -> bool {
            result
                .data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| !content.is_empty())
                .unwrap_or(false)
        }
    }

    use serde_json::json;
    use test_utils::*;

    #[tokio::test]
    async fn test_jailed_stream_with_start_end_sequences() {
        // Create chunks with jail start/end markers
        let chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), 0),
            create_mock_response_chunk("<jail>".to_string(), 0),
            create_mock_response_chunk("This is jailed ".to_string(), 0),
            create_mock_response_chunk("content".to_string(), 0),
            create_mock_response_chunk("</jail>".to_string(), 0),
            create_mock_response_chunk(" World".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with start/end sequences
        let jail = JailedStream::builder()
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // We should only get 3 chunks now:
        // 1. "Hello " (before jail)
        // 2. Accumulated jailed content when jail ends
        // 3. " World" (after jail)
        assert_eq!(results.len(), 3);

        // First chunk should pass through
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Hello ")
        );

        // When jail ends, accumulated content should be released
        let unjailed_content = &results[1].data.as_ref().unwrap().choices[0].delta.content;
        assert!(unjailed_content.is_some());
        assert!(
            unjailed_content
                .as_ref()
                .unwrap()
                .contains("<jail>This is jailed content</jail>")
        );

        // Last chunk should pass through normally
        assert_eq!(
            results[2].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some(" World")
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_with_tool_calls() {
        // Create chunks representing a tool call
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with tool call parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have jailed the content and parsed tool calls at the end
        assert!(!results.is_empty());

        // Check if tool calls were parsed
        if let Some(last_result) = results.last()
            && let Some(ref response_data) = last_result.data
            && let Some(ref tool_calls) = response_data.choices[0].delta.tool_calls
        {
            assert!(!tool_calls.as_slice().is_empty());
            assert_eq!(
                tool_calls[0].function.as_ref().unwrap().name.as_deref(),
                Some("get_weather")
            );
        }
    }

    #[tokio::test]
    async fn test_jailed_stream_dual_entry_paths() {
        // Test that BOTH sequence AND tool call detection can trigger jail
        let chunks = vec![
            create_mock_response_chunk("Normal text ".to_string(), 0),
            create_mock_response_chunk("<jail><TOOLCALL>".to_string(), 0), // Both triggers
            create_mock_response_chunk("Jailed content".to_string(), 0),
            create_mock_response_chunk("</jail>".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Configure with both sequences AND tool call parser
        let jail = JailedStream::builder()
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // We should get 2 chunks:
        // 1. "Normal text " (before jail)
        // 2. Accumulated jailed content when jail ends via </jail>
        assert_eq!(results.len(), 2);

        // First chunk should pass through
        assert_eq!(
            results[0].data.as_ref().unwrap().choices[0]
                .delta
                .content
                .as_deref(),
            Some("Normal text ")
        );

        // Second chunk should contain the accumulated jailed content
        let jailed = results[1].data.as_ref().unwrap().choices[0]
            .delta
            .content
            .as_ref()
            .expect("Expected accumulated jailed content");
        assert!(jailed.contains("<jail><TOOLCALL>Jailed content</jail>"));
    }

    #[tokio::test]
    async fn test_jailed_stream_early_exit() {
        // Tests detection of complete tool call with unjail in same chunk as the end marker
        // Input: "<TOOLCALL>" + "[{\"name\": \"test\", " + "\"arguments\": {}}]" + "</TOOLCALL>More text"
        // Expected output: 2 chunks [ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"test\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {}}]".to_string(), 0),
            create_mock_response_chunk("</TOOLCALL>More text".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 2 chunks: tool call + trailing content
        assert_eq!(
            results.len(),
            2,
            "Should have tool call and trailing content"
        );

        // Verify exact output structure: [ToolCall(), Content()]
        test_utils::assert_tool_call(&results[0], "test", serde_json::json!({}));
        test_utils::assert_content(&results[1], "More text");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "More text");
    }

    #[tokio::test]
    async fn test_jailed_stream_no_jailing() {
        // Input chunks:
        // [0] "Hello "
        // [1] "World"
        // [2] [final chunk]
        //
        // Expected output (pass-through):
        // [0] Content("Hello ")
        // [1] Content("World")
        // [2] [final chunk]
        let chunks = vec![
            create_mock_response_chunk("Hello ".to_string(), 0),
            create_mock_response_chunk("World".to_string(), 0),
            create_final_response_chunk(0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with sequences that won't match
        let jail = JailedStream::builder()
            .jail_start_sequence("<NOTPRESENT>")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            3,
            "Should pass through all 3 chunks unchanged"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "Hello ");
        assert_content(&results[1], "World");
        // results[2] is the final chunk - no content to verify

        // === Verify negative assertions ===
        for (i, result) in results.iter().take(2).enumerate() {
            assert!(
                !has_tool_call(result),
                "Chunk {} should not contain tool calls when no patterns match",
                i
            );
        }

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "Hello World",
            "Content should pass through unchanged when no jailing occurs"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_hermes_parser() {
        // Tests Hermes format tool call parsing with <tool_call> markers
        // Input: "I'll help you with that. " + "<tool_call>{\"name\": \"search_web\", \"arguments\": {\"query\": \"weather today\"}}</tool_call>" + " Let me search for that."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("I'll help you with that. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"search_web\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"query\": \"weather today\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            create_mock_response_chunk(" Let me search for that.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "I'll help you with that. ");
        test_utils::assert_tool_call(
            &results[1],
            "search_web",
            serde_json::json!({"query": "weather today"}),
        );
        test_utils::assert_content(&results[2], " Let me search for that.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "I'll help you with that.  Let me search for that."
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_parser() {
        // Tests Mistral format tool call parsing with [{ pattern
        // Input: "Sure, I can help. " + "[{\"name\": \"calculate\", \"arguments\": {\"expression\": \"2+2\"}}]" + " The calculation is done."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Sure, I can help. ".to_string(), 0),
            create_mock_response_chunk("[{".to_string(), 0),
            create_mock_response_chunk("\"name\": \"calculate\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"expression\": \"2+2\"}".to_string(), 0),
            create_mock_response_chunk("}]".to_string(), 0),
            create_mock_response_chunk(" The calculation is done.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Mistral parser
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Sure, I can help. ");
        test_utils::assert_tool_call(
            &results[1],
            "calculate",
            serde_json::json!({"expression": "2+2"}),
        );
        test_utils::assert_content(&results[2], " The calculation is done.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "Sure, I can help.  The calculation is done.");
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_parser_with_tool_calls_marker() {
        // Tests Mistral format tool call parsing with explicit [TOOL_CALLS] marker
        // Input: "Let me check that for you. " + "[TOOL_CALLS][{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}]" + " Here's the time."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Let me check that for you. ".to_string(), 0),
            create_mock_response_chunk("[TOOL_CALLS]".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"get_time\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"timezone\": \"UTC\"}}]".to_string(), 0),
            create_mock_response_chunk(" Here's the time.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Mistral parser
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Let me check that for you. ");
        test_utils::assert_tool_call(
            &results[1],
            "get_time",
            serde_json::json!({"timezone": "UTC"}),
        );
        test_utils::assert_content(&results[2], " Here's the time.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Let me check that for you.  Here's the time."
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_phi4_parser() {
        // Tests Phi4 format tool call parsing with functools[ pattern
        // Input: "I'll analyze this data. " + "functools[{\"name\": \"analyze_data\", \"arguments\": {\"dataset\": \"sales_data\"}}]" + " Analysis complete."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("I'll analyze this data. ".to_string(), 0),
            create_mock_response_chunk("functools[".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"analyze_data\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"dataset\": \"sales_data\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk("]".to_string(), 0),
            create_mock_response_chunk(" Analysis complete.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Phi4 parser
        let jail = JailedStream::builder().tool_call_parser("phi4").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "I'll analyze this data. ");
        test_utils::assert_tool_call(
            &results[1],
            "analyze_data",
            serde_json::json!({"dataset": "sales_data"}),
        );
        test_utils::assert_content(&results[2], " Analysis complete.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "I'll analyze this data.  Analysis complete.");
    }

    #[tokio::test]
    async fn test_jailed_stream_llama3_json_parser() {
        // Tests Llama3 JSON format tool call parsing with <|python_tag|> pattern
        // Input: "Let me run some code. " + "<|python_tag|>{\"name\": \"execute_code\", \"arguments\": {\"code\": \"print('Hello')\"}}" + " Done executing."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Let me run some code. ".to_string(), 0),
            create_mock_response_chunk("<|python_tag|>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"execute_code\", ".to_string(), 0),
            create_mock_response_chunk(
                "\"arguments\": {\"code\": \"print('Hello')\"}}".to_string(),
                0,
            ),
            create_mock_response_chunk(" Done executing.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with llama3_json parser
        let jail = JailedStream::builder()
            .tool_call_parser("llama3_json")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 3 chunks: content + tool call + content
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Let me run some code. ");
        test_utils::assert_tool_call(
            &results[1],
            "execute_code",
            serde_json::json!({"code": "print('Hello')"}),
        );
        test_utils::assert_content(&results[2], " Done executing.");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(reconstructed, "Let me run some code.  Done executing.");
    }

    #[tokio::test]
    async fn test_jailed_stream_false_positive_json() {
        // Tests that JSON-like content doesn't trigger false positive tool call detection
        // Input: "I can explain JSON format. " + "Here's an example: { \"key\": \"value\" }" + " is a simple JSON object. " + "Hope that helps!"
        // Expected output: 4 chunks [Content(), Content(), Content(), Content()] - no jailing
        let chunks = vec![
            create_mock_response_chunk("I can explain JSON format. ".to_string(), 0),
            create_mock_response_chunk("Here's an example: { \"key\": \"value\" }".to_string(), 0),
            create_mock_response_chunk(" is a simple JSON object. ".to_string(), 0),
            create_mock_response_chunk("Hope that helps!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with mistral parser (which specifically looks for [{ or [TOOL_CALLS] patterns)
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // The "{" pattern triggers jailing, so some chunks get combined
        assert_eq!(results.len(), 2);

        // Verify exact output structure: content chunks
        test_utils::assert_content(&results[0], "I can explain JSON format. ");
        test_utils::assert_content(
            &results[1],
            "Here's an example: { \"key\": \"value\" } is a simple JSON object. Hope that helps!",
        );

        // Verify no tool calls were detected and all content preserved
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "I can explain JSON format. Here's an example: { \"key\": \"value\" } is a simple JSON object. Hope that helps!"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_malformed_tool_call() {
        // Tests graceful handling of malformed JSON within tool call markers
        // Input: "Let me call a function. " + "<TOOLCALL>[{\"name\": \"broken_func\", \"arguments\": {\"param\": incomplete</TOOLCALL>" + " Function call attempt finished."
        // Expected output: 3 chunks [Content(), Content(malformed), Content()] - parser fails gracefully
        let chunks = vec![
            create_mock_response_chunk("Let me call a function. ".to_string(), 0),
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"broken_func\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"param\": incomplete".to_string(), 0), // Malformed JSON
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Function call attempt finished.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with nemotron_deci parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Jailing combines the tool call content into fewer chunks
        assert_eq!(
            results.len(),
            3,
            "Should handle malformed JSON gracefully and jail appropriately"
        );

        // Verify exact output structure: [Content(), Content(complete jailed content)]
        test_utils::assert_content(&results[0], "Let me call a function. ");
        test_utils::assert_content(
            &results[1],
            "<TOOLCALL>[{\"name\": \"broken_func\", \"arguments\": {\"param\": incomplete</TOOLCALL>",
        );

        // Verify malformed content is preserved as text (including markers when parsing fails)
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Let me call a function. <TOOLCALL>[{\"name\": \"broken_func\", \"arguments\": {\"param\": incomplete</TOOLCALL> Function call attempt finished."
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_partial_tool_call() {
        // Tests handling of incomplete tool call when stream ends abruptly
        // Input: "Starting function call. " + "<TOOLCALL>[{\"name\": \"incomplete_func\", \"arguments\": {" (no end marker)
        // Expected output: 2 chunks [Content(), Content(partial)] - partial accumulated content released on stream end
        let chunks = vec![
            create_mock_response_chunk("Starting function call. ".to_string(), 0),
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk("[{\"name\": \"incomplete_func\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {".to_string(), 0),
            // Stream ends abruptly without closing the tool call
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with nemotron_deci parser
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should handle partial tool call gracefully - releases accumulated content on stream end
        assert_eq!(
            results.len(),
            2,
            "Should handle partial tool call and release content"
        );

        // Verify exact output structure: [Content(), Content(accumulated partial)]
        test_utils::assert_content(&results[0], "Starting function call. ");
        test_utils::assert_content(
            &results[1],
            "<TOOLCALL>[{\"name\": \"incomplete_func\", \"arguments\": {",
        );

        // Verify partial content is preserved as text since no valid tool call could be parsed
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Starting function call. <TOOLCALL>[{\"name\": \"incomplete_func\", \"arguments\": {"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_empty_stream() {
        // Input chunks: []
        //
        // Expected output: []
        let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];
        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .jail_start_sequence("<jail>")
            .jail_end_sequence("</jail>")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            0,
            "Empty stream should produce exactly 0 results"
        );

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "",
            "Empty stream should reconstruct to empty string"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_multiple_tool_calls() {
        // Input chunks: 9 chunks for 2 tool calls with content between
        //
        // Expected output:
        // [0] Content("I'll help with multiple tasks. ")
        // [1] ToolCall("get_weather", {"city": "NYC"})
        // [2] Content(" Now let me get the time. ")
        // [3] ToolCall("get_time", {"timezone": "EST"})
        // [4] Content(" Both tasks completed!")
        let chunks = vec![
            create_mock_response_chunk("I'll help with multiple tasks. ".to_string(), 0),
            // First tool call
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Now let me get the time. ".to_string(), 0),
            // Second tool call
            create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"EST\"}}]".to_string(),
                0,
            ),
            create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
            create_mock_response_chunk(" Both tasks completed!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            5,
            "Should emit exactly 5 chunks as documented above"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "I'll help with multiple tasks. ");
        assert_tool_call(&results[1], "get_weather", json!({"city": "NYC"}));
        assert_content(&results[2], " Now let me get the time. ");
        assert_tool_call(&results[3], "get_time", json!({"timezone": "EST"}));
        assert_content(&results[4], " Both tasks completed!");

        // === Verify content reconstruction ===
        let expected_content =
            "I'll help with multiple tasks.  Now let me get the time.  Both tasks completed!";
        assert_eq!(
            reconstruct_content(&results),
            expected_content,
            "Content reconstruction should exclude tool calls and preserve text flow"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_tool_call_across_many_chunks() {
        // Tests extreme fragmentation: tool call split across 65 individual character chunks
        // Input: "I'll process your request. " + "<TOOLCALL>[{"name": "process_data", "arguments": {}}]</TOOLCALL>" + " Processing complete!"
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("I'll process your request. ".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk("T".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("C".to_string(), 0),
            create_mock_response_chunk("A".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk(">".to_string(), 0),
            create_mock_response_chunk("[".to_string(), 0),
            create_mock_response_chunk("{".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("n".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("m".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(":".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("p".to_string(), 0),
            create_mock_response_chunk("r".to_string(), 0),
            create_mock_response_chunk("o".to_string(), 0),
            create_mock_response_chunk("c".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("_".to_string(), 0),
            create_mock_response_chunk("d".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("t".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(",".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk("a".to_string(), 0),
            create_mock_response_chunk("r".to_string(), 0),
            create_mock_response_chunk("g".to_string(), 0),
            create_mock_response_chunk("u".to_string(), 0),
            create_mock_response_chunk("m".to_string(), 0),
            create_mock_response_chunk("e".to_string(), 0),
            create_mock_response_chunk("n".to_string(), 0),
            create_mock_response_chunk("t".to_string(), 0),
            create_mock_response_chunk("s".to_string(), 0),
            create_mock_response_chunk("\"".to_string(), 0),
            create_mock_response_chunk(":".to_string(), 0),
            create_mock_response_chunk(" ".to_string(), 0),
            create_mock_response_chunk("{".to_string(), 0),
            create_mock_response_chunk("}".to_string(), 0),
            create_mock_response_chunk("}".to_string(), 0),
            create_mock_response_chunk("]".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk("/".to_string(), 0),
            create_mock_response_chunk("T".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("O".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("C".to_string(), 0),
            create_mock_response_chunk("A".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk("L".to_string(), 0),
            create_mock_response_chunk(">".to_string(), 0),
            create_mock_response_chunk(" Processing complete!".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should consolidate extreme fragmentation into 3 clean chunks
        // Input: "I'll process your request. " + 54-char tool call + " Processing complete!"
        // Expected output: [Content(), ToolCall(), Content()]
        assert_eq!(
            results.len(),
            3,
            "Should consolidate fragments into 3 chunks"
        );

        // Verify exact output structure
        test_utils::assert_content(&results[0], "I'll process your request. ");
        test_utils::assert_tool_call(&results[1], "process_data", serde_json::json!({}));
        test_utils::assert_content(&results[2], " Processing complete!");

        // Verify content reconstruction excludes tool calls
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "I'll process your request.  Processing complete!"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_preserves_metadata() {
        // Test metadata preservation through jail processing
        let test_id = Some("correlation-id-123".to_string());
        let test_event = Some("request-processing".to_string());
        let test_comment = Some(vec![
            "upstream-correlation".to_string(),
            "debug-info".to_string(),
        ]);

        // Create chunks with specific metadata for the jail trigger
        let chunks = vec![
            create_annotated_chunk(
                "I'll help you with that. ".to_string(),
                0,
                None, // No metadata on first chunk
                None,
                None,
            ),
            create_annotated_chunk(
                "<tool_call>".to_string(),
                0,
                test_id.clone(), // Metadata on jail trigger chunk
                test_event.clone(),
                test_comment.clone(),
            ),
            create_mock_response_chunk("{\"name\": \"search_web\", ".to_string(), 0),
            create_mock_response_chunk("\"arguments\": {\"query\": \"test\"}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            create_mock_response_chunk(" Processing complete.".to_string(), 0),
            test_utils::create_final_response_chunk(0), // Backend finish_reason chunk
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should get 3 chunks: before jail, tool call response, after jail
        assert!(
            results.len() >= 3,
            "Should have at least 3 chunks, got {}",
            results.len()
        );

        // Find the tool call chunk (the one with tool_calls, not the finish_reason chunk)
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.delta.tool_calls.is_some())
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify metadata is preserved
        assert_eq!(
            tool_call_chunk.id, test_id,
            "ID should be preserved from jail trigger chunk"
        );
        assert_eq!(
            tool_call_chunk.event, test_event,
            "Event should be preserved from jail trigger chunk"
        );
        assert_eq!(
            tool_call_chunk.comment, test_comment,
            "Comment should be preserved from jail trigger chunk"
        );

        // Verify tool call was parsed correctly
        let tool_calls = &tool_call_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls;
        assert!(tool_calls.is_some(), "Should have tool calls");
        let tool_calls = tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1, "Should have exactly one tool call");
        assert_eq!(
            tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .name
                .as_ref()
                .unwrap(),
            "search_web"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_preserves_metadata_on_stream_end() {
        // Test metadata preservation when stream ends while jailed
        let test_id = Some("end-correlation-456".to_string());
        let test_event = Some("stream-termination".to_string());
        let test_comment = Some(vec!["incomplete-processing".to_string()]);

        // Create chunks that end while jailed (no explicit end marker)
        let chunks = vec![
            create_mock_response_chunk("Starting function call: ".to_string(), 0),
            create_annotated_chunk(
                "<tool_call>".to_string(), // This chunk triggers jail and has metadata
                0,
                test_id.clone(),
                test_event.clone(),
                test_comment.clone(),
            ),
            create_mock_response_chunk(
                "{\"name\": \"incomplete_call\"".to_string(), // No closing brace
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        // Create JailedStream with Hermes parser
        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should get 2 chunks: first chunk passes through, stream end releases accumulated
        assert_eq!(results.len(), 2, "Should have exactly 2 chunks");

        // The second chunk is the accumulated content released when stream ended
        let accumulated_chunk = &results[1];

        // Verify metadata is preserved from the jail trigger
        assert_eq!(
            accumulated_chunk.id, test_id,
            "ID should be preserved when stream ends while jailed"
        );
        assert_eq!(
            accumulated_chunk.event, test_event,
            "Event should be preserved when stream ends while jailed"
        );
        assert_eq!(
            accumulated_chunk.comment, test_comment,
            "Comment should be preserved when stream ends while jailed"
        );

        // Verify accumulated content is returned
        let content = &accumulated_chunk.data.as_ref().unwrap().choices[0]
            .delta
            .content;
        assert!(content.is_some(), "Should have accumulated content");
        let content = content.as_ref().unwrap();
        assert!(
            content.contains("<tool_call>"),
            "Should contain jail start marker in accumulated content"
        );
        assert!(
            content.contains("incomplete_call"),
            "Should contain accumulated incomplete content"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_metadata_edge_cases() {
        // Test edge cases: empty metadata, partial metadata, etc.
        let chunks = vec![
            create_annotated_chunk(
                "Text with ".to_string(),
                0,
                Some("".to_string()), // Empty string ID
                None,                 // No event
                Some(vec![]),         // Empty comment vector
            ),
            create_annotated_chunk(
                "<tool_call>".to_string(),
                0,
                None,                                 // No ID
                Some("partial-metadata".to_string()), // Only event
                None,                                 // No comment
            ),
            create_mock_response_chunk("{\"name\": \"test\", \"arguments\": {}}".to_string(), 0),
            create_mock_response_chunk("</tool_call>".to_string(), 0),
            test_utils::create_final_response_chunk(0), // Backend finish_reason chunk
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Find the tool call chunk (the one with tool_calls, not the finish_reason chunk)
        let tool_call_chunk = results
            .iter()
            .find(|r| {
                r.data
                    .as_ref()
                    .and_then(|d| d.choices.first())
                    .map(|c| c.delta.tool_calls.is_some())
                    .unwrap_or(false)
            })
            .expect("Should have a tool call response chunk");

        // Verify partial metadata is preserved correctly
        assert_eq!(tool_call_chunk.id, None, "Should preserve None ID");
        assert_eq!(
            tool_call_chunk.event,
            Some("partial-metadata".to_string()),
            "Should preserve event"
        );
        assert_eq!(
            tool_call_chunk.comment, None,
            "Should preserve None comment"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_trailing_content_same_chunk() {
        // Input chunks:
        // [0] "I'll help you. "
        // [1] "<tool_call>"
        // [2] "{\"name\": \"search\", \"arguments\": {}}"
        // [3] "</tool_call>trailing text that should not be lost"
        //
        // Expected output:
        // [0] Content("I'll help you. ")
        // [1] ToolCall("search", {})
        // [2] Content("trailing text that should not be lost")
        let chunks = vec![
            create_mock_response_chunk("I'll help you. ".to_string(), 0),
            create_mock_response_chunk("<tool_call>".to_string(), 0),
            create_mock_response_chunk("{\"name\": \"search\", \"arguments\": {}}".to_string(), 0),
            // This chunk contains both the end marker AND trailing content
            create_mock_response_chunk(
                "</tool_call>trailing text that should not be lost".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            3,
            "Should emit exactly 3 chunks as documented above"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "I'll help you. ");
        assert_tool_call(&results[1], "search", json!({}));
        assert_content(&results[2], "trailing text that should not be lost");

        // === Verify content reconstruction ===
        let expected_content = "I'll help you. trailing text that should not be lost";
        assert_eq!(
            reconstruct_content(&results),
            expected_content,
            "Content reconstruction should preserve initial and trailing text"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_early_exit_with_trailing() {
        // Tests early exit when complete tool call is detected in chunk that also contains trailing content
        // Input: "Starting task: " + "<tool_call>{\"name\": \"complete_task\", \"arguments\": {}}" + "</tool_call> Task completed successfully."
        // Expected output: 3 chunks [Content(), ToolCall(), Content()]
        let chunks = vec![
            create_mock_response_chunk("Starting task: ".to_string(), 0),
            create_mock_response_chunk(
                "<tool_call>{\"name\": \"complete_task\", \"arguments\": {}}".to_string(),
                0,
            ),
            // Early exit should happen here, but we also have trailing content
            create_mock_response_chunk("</tool_call> Task completed successfully.".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have exactly 3 chunks: content + tool call + trailing
        assert_eq!(
            results.len(),
            3,
            "Should have content, tool call, and trailing content"
        );

        // Verify exact output structure: [Content(), ToolCall(), Content()]
        test_utils::assert_content(&results[0], "Starting task: ");
        test_utils::assert_tool_call(&results[1], "complete_task", serde_json::json!({}));
        test_utils::assert_content(&results[2], " Task completed successfully.");

        // Verify content reconstruction excludes tool calls but preserves trailing
        let reconstructed = test_utils::reconstruct_content(&results);
        assert_eq!(
            reconstructed,
            "Starting task:  Task completed successfully."
        );
    }

    #[tokio::test]
    async fn test_multiple_choices_independent_jailing() {
        // Test that different choices can jail and unjail independently
        // This test will FAIL with the current HashMap-based implementation
        let chunks = vec![
            // Chunk 1: All choices start normally
            create_multi_choice_chunk(vec![
                ("Starting task A. ".to_string(), 0),
                ("Starting task B. ".to_string(), 1),
                ("Starting task C. ".to_string(), 2),
            ]),
            // Chunk 2: Choice 0 starts tool call (gets jailed), others continue
            create_multi_choice_chunk(vec![
                ("<tool_call>".to_string(), 0),    // Choice 0 jailed
                ("Continuing B. ".to_string(), 1), // Choice 1 continues
                ("Continuing C. ".to_string(), 2), // Choice 2 continues
            ]),
            // Chunk 3: Choice 0 still jailed, Choice 2 starts tool call
            create_multi_choice_chunk(vec![
                ("{\"name\": \"tool_a\"".to_string(), 0), // Choice 0 still jailed
                ("More B content. ".to_string(), 1),      // Choice 1 continues
                ("<tool_call>".to_string(), 2),           // Choice 2 now jailed
            ]),
            // Chunk 4: Choice 0 finishes tool call, Choice 2 continues tool call
            create_multi_choice_chunk(vec![
                (", \"arguments\": {}}</tool_call>".to_string(), 0), // Choice 0 unjails
                ("Final B. ".to_string(), 1),                        // Choice 1 continues
                ("{\"name\": \"tool_c\", \"arguments\": {}}".to_string(), 2), // Choice 2 still jailed
            ]),
            // Chunk 5: Choice 2 finishes tool call
            create_multi_choice_chunk(vec![
                ("After tool A. ".to_string(), 0), // Choice 0 continues after unjail
                ("Done with B. ".to_string(), 1),  // Choice 1 continues
                ("</tool_call>".to_string(), 2),   // Choice 2 unjails
            ]),
            // Chunk 6: Backend finish_reason chunks for all choices
            test_utils::create_multi_choice_finish_chunk(vec![0, 1, 2]),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // EXPECTED BEHAVIOR (will fail with current implementation):
        // - Choice 1 should stream continuously (never jailed)
        // - Choice 0 should jail from chunk 2 until chunk 4
        // - Choice 2 should jail from chunk 3 until chunk 5
        // - Each choice should emit independently

        // Verify choice 1 was never interrupted (should have ~5 chunks of content)
        let choice_1_chunks: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 1 && c.delta.content.is_some())
            .collect();

        assert!(
            choice_1_chunks.len() >= 4,
            "Choice 1 should have multiple continuous chunks, got {}",
            choice_1_chunks.len()
        );

        // Verify choice 0 has a tool call response
        let choice_0_tool_calls: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 0 && c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        assert!(
            !choice_0_tool_calls.is_empty(),
            "Choice 0 should have tool call response"
        );

        // Verify choice 2 has a tool call response
        let choice_2_tool_calls: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.index == 2 && c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        assert!(
            !choice_2_tool_calls.is_empty(),
            "Choice 2 should have tool call response"
        );
    }

    #[tokio::test]
    async fn test_deterministic_choice_ordering() {
        // Test that choices are processed in deterministic order (0, 1, 2...)
        // This test will FAIL with the current HashMap implementation
        let chunks = vec![
            // All choices have tool calls that complete at the same time
            create_multi_choice_chunk(vec![
                (
                    "<tool_call>{\"name\": \"tool_0\", \"arguments\": {}}</tool_call>".to_string(),
                    0,
                ),
                (
                    "<tool_call>{\"name\": \"tool_1\", \"arguments\": {}}</tool_call>".to_string(),
                    1,
                ),
                (
                    "<tool_call>{\"name\": \"tool_2\", \"arguments\": {}}</tool_call>".to_string(),
                    2,
                ),
            ]),
            test_utils::create_multi_choice_finish_chunk(vec![0, 1, 2]),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("hermes").build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Find all tool call responses
        let mut tool_call_responses: Vec<_> = results
            .iter()
            .filter_map(|r| r.data.as_ref())
            .flat_map(|d| &d.choices)
            .filter(|c| c.finish_reason == Some(FinishReason::ToolCalls))
            .collect();

        // Sort by the order they appear in the results
        // With HashMap, this order will be non-deterministic
        // With Vec, this should always be [0, 1, 2]
        tool_call_responses.sort_by_key(|c| c.index);

        assert_eq!(
            tool_call_responses.len(),
            3,
            "Should have 3 tool call responses"
        );

        // Run this test multiple times to verify determinism
        for run in 0..5 {
            let chunks = vec![
                create_multi_choice_chunk(vec![
                    (
                        "<tool_call>{\"name\": \"tool_0\", \"arguments\": {}}</tool_call>"
                            .to_string(),
                        0,
                    ),
                    (
                        "<tool_call>{\"name\": \"tool_1\", \"arguments\": {}}</tool_call>"
                            .to_string(),
                        1,
                    ),
                    (
                        "<tool_call>{\"name\": \"tool_2\", \"arguments\": {}}</tool_call>"
                            .to_string(),
                        2,
                    ),
                ]),
                test_utils::create_multi_choice_finish_chunk(vec![0, 1, 2]),
            ];

            let input_stream = stream::iter(chunks);
            let jail = JailedStream::builder().tool_call_parser("hermes").build();
            let run_results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

            let run_responses: Vec<_> = run_results
                .iter()
                .filter_map(|r| r.data.as_ref())
                .flat_map(|d| &d.choices)
                .filter(|c| c.finish_reason == Some(FinishReason::ToolCalls))
                .collect();

            // The order should be consistent across runs
            // This will fail with HashMap due to non-deterministic iteration
            let indices: Vec<u32> = run_responses.iter().map(|c| c.index).collect();
            assert_eq!(
                indices,
                vec![0, 1, 2],
                "Choice processing order should be deterministic on run {}",
                run
            );
        }
    }

    #[tokio::test]
    async fn test_multiple_choices_usage_aggregation() {
        // Test that usage is correctly aggregated across multiple choices
        // This test demonstrates how usage should work with n>1

        // For now, this test just documents expected behavior
        // It will need to be expanded once usage aggregation is implemented

        let chunks = vec![create_multi_choice_chunk(vec![
            ("Response A with many tokens".to_string(), 0), // ~5 tokens
            ("Response B".to_string(), 1),                  // ~2 tokens
            ("Response C has even more tokens than A".to_string(), 2), // ~8 tokens
        ])];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // TODO: Once usage aggregation is implemented, verify:
        // - Usage chunk has choices: [] (empty array)
        // - completion_tokens = sum of all choices (~15 total)
        // - prompt_tokens counted once
        // - total_tokens = prompt_tokens + completion_tokens

        // For now, just verify we got some results
        assert!(!results.is_empty(), "Should have some results");
    }

    #[tokio::test]
    async fn test_partial_matching_false_positive_prevention() {
        // Input chunks:
        // [0] "n "
        // [1] "<"
        // [2] " 5"
        //
        // Expected output:
        // [0] Content("n ")
        // [1] Content("< 5")  // "<" held as partial, then combined with " 5" when pattern doesn't match
        let chunks = vec![
            create_mock_response_chunk("n ".to_string(), 0),
            create_mock_response_chunk("<".to_string(), 0),
            create_mock_response_chunk(" 5".to_string(), 0),
        ];

        let input_stream = stream::iter(chunks);

        // Use nemotron parser which has <TOOLCALL> as a pattern
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            2,
            "Should emit exactly 2 chunks: 'n ' and '< 5'"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "n ");
        assert_content(&results[1], "< 5");

        // === Verify negative assertions ===
        // Verify NO tool calls were detected
        for (i, result) in results.iter().enumerate() {
            assert!(
                !has_tool_call(result),
                "Chunk {} should not contain tool calls in mathematical expression",
                i
            );
        }

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "n < 5",
            "Content reconstruction should preserve the complete mathematical expression"
        );
    }

    #[tokio::test]
    async fn test_partial_matching_suffix_detection() {
        // Input chunks:
        // [0] "text<TO"
        // [1] "OLCALL>[{\"name\": \"test\", \"arguments\": {}}]</TOOLCALL>"
        //
        // Expected output:
        // [0] Content("text")  // "<TO" held as partial
        // [1] ToolCall("test", {})
        let chunks = vec![
            create_mock_response_chunk("text<TO".to_string(), 0),
            create_mock_response_chunk(
                "OLCALL>[{\"name\": \"test\", \"arguments\": {}}]</TOOLCALL>".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .jail_end_sequence("</TOOLCALL>")
            .build();

        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // === Verify chunk count ===
        assert_eq!(
            results.len(),
            2,
            "Should emit exactly 2 chunks: [0] 'text' content, [1] tool call"
        );

        // === Verify individual chunks ===
        assert_content(&results[0], "text");
        assert_tool_call(&results[1], "test", json!({}));

        // === Verify negative assertions ===
        // Verify '<' was not emitted in first chunk (held as partial)
        let first_content = extract_content(&results[0]);
        assert!(
            !first_content.contains('<'),
            "First chunk should not contain '<' as it's part of partial match '<TO'"
        );

        // === Verify content reconstruction ===
        assert_eq!(
            reconstruct_content(&results),
            "text",
            "Content reconstruction should only include 'text' (tool call parsed separately)"
        );
    }

    #[tokio::test]
    async fn test_jailed_stream_harmony_parser() {
        // Harmony format with analysis text and a tool call encoded in special tags
        let chunks = vec![
            create_mock_response_chunk(
                "<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>"
                    .to_string(),
                0,
            ),
            create_mock_response_chunk("<|start|>".to_string(), 0),
            create_mock_response_chunk("assistant".to_string(), 0),
            create_mock_response_chunk("<|channel|>".to_string(), 0),
            create_mock_response_chunk(
                "commentary to=functions.get_current_weather <|constrain|>json".to_string(),
                0,
            ),
            create_mock_response_chunk(
                "<|message|>{\"location\":\"San Francisco\"}<|call|>".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder().tool_call_parser("harmony").build();
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have at least one output containing both analysis text and parsed tool call
        assert!(!results.is_empty());

        // Verify the analysis text appears as content in one of the outputs
        let has_analysis_text = results.iter().any(|r| {
            r.data
                .as_ref()
                .and_then(|d| d.choices.first())
                .and_then(|c| c.delta.content.as_ref())
                .map(|content| content.contains("Need to use function get_current_weather."))
                .unwrap_or(false)
        });
        assert!(has_analysis_text, "Should contain extracted analysis text");

        // Verify a tool call was parsed with expected name and args
        let tool_call_idx = results
            .iter()
            .position(test_utils::has_tool_call)
            .expect("Should have a tool call result");
        test_utils::assert_tool_call(
            &results[tool_call_idx],
            "get_current_weather",
            json!({"location": "San Francisco"}),
        );
    }

    #[tokio::test]
    async fn test_deepseek_v3_1() {
        // DeepSeek v3.1 format with two tool calls encoded in special tags
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Berlin", "units": "metric"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather_forecast<｜tool▁sep｜>{"location": "Berlin", "days": 7, "units": "imperial"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_air_quality<｜tool▁sep｜>{"location": "Berlin", "radius": 50}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;

        let chunks = vec![create_mock_response_chunk(text.to_string(), 0)];

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("deepseek_v3_1")
            .build();
        let jailed_stream = jail.apply_with_finish_reason(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have at least one output containing both analysis text and parsed tool call
        assert!(!results.is_empty());

        // Verify a tool call was parsed with expected name and args
        let tool_call_idx = results
            .iter()
            .position(test_utils::has_tool_call)
            .expect("Should have a tool call result");
        test_utils::assert_tool_call(
            &results[tool_call_idx],
            "get_current_weather",
            json!({"location": "Berlin", "units": "metric"}),
        );
        for result in results {
            let Some(data) = result.data else {
                continue;
            };
            for choice in data.choices {
                if let Some(content) = choice.delta.content {
                    assert!(
                        !content.contains("<｜tool▁calls▁end｜>"),
                        "Should not contain deepseek special tokens in content"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_deepseek_v3_1_chunk() {
        // DeepSeek v3.1 format with two tool calls encoded in special tags
        let text = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Berlin", "units": "metric"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather_forecast<｜tool▁sep｜>{"location": "Berlin", "days": 7, "units": "imperial"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_air_quality<｜tool▁sep｜>{"location": "Berlin", "radius": 50}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;

        // Split text into words, treating angle-bracketed tokens as one word
        let mut words = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        while i < chars.len() {
            if chars[i] == '<' {
                // Find the next '>'
                if let Some(end) = chars[i..].iter().position(|&c| c == '>') {
                    let word: String = chars[i..=i + end].iter().collect();
                    words.push(word);
                    i += end + 1;
                } else {
                    // Malformed, just push the rest
                    words.push(chars[i..].iter().collect());
                    break;
                }
            } else if chars[i].is_whitespace() {
                i += 1;
            } else {
                // Collect until next whitespace or '<'
                let start = i;
                while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '<' {
                    i += 1;
                }
                words.push(chars[start..i].iter().collect());
            }
        }

        let chunks = words
            .into_iter()
            .map(|word| create_mock_response_chunk(word, 0))
            .collect::<Vec<_>>();

        let input_stream = stream::iter(chunks);

        let jail = JailedStream::builder()
            .tool_call_parser("deepseek_v3_1")
            .build();
        let jailed_stream = jail.apply_with_finish_reason(input_stream);
        let results: Vec<_> = jailed_stream.collect().await;

        // Should have at least one output containing both analysis text and parsed tool call
        assert!(!results.is_empty());

        // Verify a tool call was parsed with expected name and args
        let tool_call_idx = results
            .iter()
            .position(test_utils::has_tool_call)
            .expect("Should have a tool call result");
        test_utils::assert_tool_call(
            &results[tool_call_idx],
            "get_current_weather",
            json!({"location": "Berlin", "units": "metric"}),
        );
        for result in results {
            let Some(data) = result.data else {
                continue;
            };
            for choice in data.choices {
                if let Some(content) = choice.delta.content {
                    assert!(
                        !content.contains("<｜tool▁calls▁end｜>"),
                        "Should not contain deepseek special tokens in content"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_jailed_stream_mistral_false_positive_curly() {
        // Curly brace in normal text should not trigger tool call detection for mistral
        let chunks = vec![
            create_mock_response_chunk("Hey How".to_string(), 0),
            create_mock_response_chunk("are { you? ".to_string(), 0),
            create_final_response_chunk(0),
        ];

        let input_stream = stream::iter(chunks);
        let jail = JailedStream::builder().tool_call_parser("mistral").build();
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        assert!(results.len() >= 2);
        assert_content(&results[0], "Hey How");
        assert!(
            results.iter().any(|r| extract_content(r) == "are { you? "),
            "Should preserve the literal text with curly brace"
        );
        for (i, r) in results.iter().enumerate() {
            assert!(
                !has_tool_call(r),
                "Result {} should not contain tool calls for false-positive text",
                i
            );
        }
    }

    #[tokio::test]
    #[ignore]
    // TODO: This needs to be fixed in parser library. P1 priority.
    async fn test_jailed_stream_mistral_false_positive_then_tool_calls_marker() {
        // Normal text with curly brace followed by explicit [TOOL_CALLS] marker should parse tool call
        let chunks = vec![
            create_mock_response_chunk("Hey How".to_string(), 0),
            create_mock_response_chunk("are { you? ".to_string(), 0),
            create_mock_response_chunk("[TOOL_CALLS]".to_string(), 0),
            create_mock_response_chunk(
                "[{\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\", \"unit\": \"fahrenheit\"}}]"
                    .to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(chunks);
        let jail = JailedStream::builder().tool_call_parser("mistral").build();
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should preserve earlier content and also produce a tool call
        assert!(results.len() >= 2);

        assert!(
            results.iter().any(|r| extract_content(r) == "Hey How"),
            "Should include initial content"
        );
        assert!(
            results.iter().any(|r| extract_content(r) == "{ you? "),
            "Should include content preceding the marker"
        );

        let tool_call_idx = results
            .iter()
            .position(test_utils::has_tool_call)
            .expect("Should have a tool call result");
        test_utils::assert_tool_call(
            &results[tool_call_idx],
            "get_weather",
            json!({"location": "San Francisco", "unit": "fahrenheit"}),
        );
    }
}

// Comprehensive parallel tool calling jail tests
#[cfg(test)]
mod parallel_jail_tests {
    use super::tests::test_utils;
    use super::*;
    use futures::StreamExt;
    use futures::stream;
    use serde_json::json;

    /// Helper function to create a mock response chunk with multiple choices
    fn create_multi_choice_response_chunk(
        contents: Vec<String>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let choices: Vec<ChatChoiceStream> = contents
            .into_iter()
            .enumerate()
            .map(|(i, content)| {
                #[allow(deprecated)]
                ChatChoiceStream {
                    index: i as u32,
                    delta: ChatCompletionStreamResponseDelta {
                        role: Some(Role::Assistant),
                        content: Some(content),
                        tool_calls: None,
                        function_call: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }
            })
            .collect();

        let response = NvCreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices,
            created: 1234567890,
            model: "test-model".to_string(),
            system_fingerprint: Some("test-fingerprint".to_string()),
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
            nvext: None,
        };

        Annotated {
            data: Some(response),
            id: None,
            event: None,
            comment: None,
        }
    }

    /// Helper function to validate parallel tool call results in streaming format
    fn validate_parallel_streaming_tool_calls(
        results: &[Annotated<NvCreateChatCompletionStreamResponse>],
        expected_tool_calls: &[(&str, serde_json::Value)],
    ) {
        // Find results with tool calls
        let tool_call_results: Vec<_> = results
            .iter()
            .filter(|r| {
                r.data
                    .as_ref()
                    .is_some_and(|d| d.choices.iter().any(|c| c.delta.tool_calls.is_some()))
            })
            .collect();

        assert!(
            !tool_call_results.is_empty(),
            "Should have at least one tool call result"
        );

        // Collect all tool calls from all results
        let mut all_tool_calls = Vec::new();
        for result in &tool_call_results {
            if let Some(ref data) = result.data {
                for choice in &data.choices {
                    if let Some(ref tool_calls) = choice.delta.tool_calls {
                        all_tool_calls.extend(tool_calls.iter());
                    }
                }
            }
        }

        assert_eq!(
            all_tool_calls.len(),
            expected_tool_calls.len(),
            "Expected {} tool calls, got {}",
            expected_tool_calls.len(),
            all_tool_calls.len()
        );

        // Validate each tool call
        for (i, (expected_name, expected_args)) in expected_tool_calls.iter().enumerate() {
            let tool_call = &all_tool_calls[i];
            assert!(tool_call.id.is_some(), "Tool call {} should have an ID", i);
            assert_eq!(
                tool_call.r#type,
                Some(dynamo_async_openai::types::ChatCompletionToolType::Function),
                "Tool call {} should be of type 'function'",
                i
            );

            if let Some(ref function) = tool_call.function {
                assert_eq!(
                    function.name.as_deref(),
                    Some(*expected_name),
                    "Tool call {} name should be {}",
                    i,
                    expected_name
                );

                if let Some(ref args_str) = function.arguments {
                    let parsed_args: serde_json::Value =
                        serde_json::from_str(args_str).expect("Arguments should be valid JSON");
                    assert_eq!(
                        parsed_args, *expected_args,
                        "Tool call {} arguments should match expected",
                        i
                    );
                }
            }
        }
    }

    // =============================================================================
    // 1. PARALLEL TOOL CALLS IN SINGLE CHUNK
    // =============================================================================

    #[tokio::test]
    async fn test_parallel_tool_calls_single_chunk_nemotron() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk(
                r#"<TOOLCALL>[
    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}
]</TOOLCALL>"#.to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should have tool call results
        assert!(!results.is_empty(), "Should have results");

        let expected_calls = [
            (
                "get_current_weather",
                json!({"city": "Dallas", "state": "TX", "unit": "fahrenheit"}),
            ),
            (
                "get_current_weather",
                json!({"city": "Orlando", "state": "FL", "unit": "fahrenheit"}),
            ),
        ];

        validate_parallel_streaming_tool_calls(&results, &expected_calls);
    }

    #[tokio::test]
    async fn test_parallel_tool_calls_single_chunk_mistral() {
        let jail = JailedStream::builder().tool_call_parser("mistral").build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk(
                r#"[TOOL_CALLS][{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}, {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}][/TOOL_CALLS]"#.to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        let expected_calls = [
            (
                "get_current_weather",
                json!({"city": "Dallas", "state": "TX", "unit": "fahrenheit"}),
            ),
            (
                "get_current_weather",
                json!({"city": "Orlando", "state": "FL", "unit": "fahrenheit"}),
            ),
        ];

        validate_parallel_streaming_tool_calls(&results, &expected_calls);
    }

    // =============================================================================
    // 2. PARALLEL TOOL CALLS ACROSS MULTIPLE CHUNKS (STREAMING)
    // =============================================================================

    #[tokio::test]
    async fn test_parallel_tool_calls_streaming_chunks() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk("<TOOLCALL>[".to_string(), 0),
            test_utils::create_mock_response_chunk(
                r#"    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},"#.to_string(),
                0,
            ),
            test_utils::create_mock_response_chunk(
                r#"    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}"#.to_string(),
                0,
            ),
            test_utils::create_mock_response_chunk("]</TOOLCALL>".to_string(), 0),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        let expected_calls = [
            (
                "get_current_weather",
                json!({"city": "Dallas", "state": "TX", "unit": "fahrenheit"}),
            ),
            (
                "get_current_weather",
                json!({"city": "Orlando", "state": "FL", "unit": "fahrenheit"}),
            ),
        ];

        validate_parallel_streaming_tool_calls(&results, &expected_calls);
    }

    #[tokio::test]
    async fn test_parallel_tool_calls_with_normal_text_before_and_after() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk("I'll check the weather for both cities. ".to_string(), 0),
            test_utils::create_mock_response_chunk(
                r#"<TOOLCALL>[
    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}
]</TOOLCALL>"#.to_string(),
                0,
            ),
            test_utils::create_mock_response_chunk(" Let me get that information for you.".to_string(), 0),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        // Should have normal text before tool calls
        let normal_text_before = results.iter().find(|r| {
            r.data.as_ref().is_some_and(|d| {
                d.choices.iter().any(|c| {
                    c.delta
                        .content
                        .as_ref()
                        .is_some_and(|content| content.contains("I'll check the weather"))
                })
            })
        });
        assert!(
            normal_text_before.is_some(),
            "Should have normal text before tool calls"
        );

        // Should have tool calls
        let expected_calls = [
            (
                "get_current_weather",
                json!({"city": "Dallas", "state": "TX", "unit": "fahrenheit"}),
            ),
            (
                "get_current_weather",
                json!({"city": "Orlando", "state": "FL", "unit": "fahrenheit"}),
            ),
        ];

        validate_parallel_streaming_tool_calls(&results, &expected_calls);

        // Should have normal text after tool calls
        let normal_text_after = results.iter().find(|r| {
            r.data.as_ref().is_some_and(|d| {
                d.choices.iter().any(|c| {
                    c.delta
                        .content
                        .as_ref()
                        .is_some_and(|content| content.contains("Let me get that information"))
                })
            })
        });
        assert!(
            normal_text_after.is_some(),
            "Should have normal text after tool calls"
        );
    }

    // =============================================================================
    // 3. MULTIPLE CHOICES WITH PARALLEL TOOL CALLS
    // =============================================================================

    #[tokio::test]
    async fn test_multiple_choices_with_parallel_tool_calls() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .emission_mode(dynamo_llm::protocols::openai::chat_completions::jail::EmissionMode::SingleChoicePerChunk)
            .build();

        let input_chunks = vec![
            create_multi_choice_response_chunk(vec![
                r#"<TOOLCALL>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]</TOOLCALL>"#.to_string(),
                r#"<TOOLCALL>[{"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}]</TOOLCALL>"#.to_string(),
            ]),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        // Should have tool calls from both choices
        let tool_call_count = results
            .iter()
            .map(|r| {
                r.data.as_ref().map_or(0, |d| {
                    d.choices
                        .iter()
                        .map(|c| c.delta.tool_calls.as_ref().map_or(0, |tc| tc.len()))
                        .sum::<usize>()
                })
            })
            .sum::<usize>();

        assert!(
            tool_call_count >= 2,
            "Should have at least 2 tool calls from different choices"
        );
    }

    // =============================================================================
    // 4. MIXED TOOL TYPES IN PARALLEL CALLS
    // =============================================================================

    #[tokio::test]
    async fn test_parallel_mixed_tool_types_streaming() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk(
                r#"<TOOLCALL>[
    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},
    {"name": "web_search", "arguments": {"query": "Orlando Florida attractions", "max_results": 5}},
    {"name": "get_user_location", "arguments": {"ip_address": "192.168.1.1"}}
]</TOOLCALL>"#.to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        let expected_calls = [
            (
                "get_current_weather",
                json!({"city": "Dallas", "state": "TX", "unit": "fahrenheit"}),
            ),
            (
                "web_search",
                json!({"query": "Orlando Florida attractions", "max_results": 5}),
            ),
            ("get_user_location", json!({"ip_address": "192.168.1.1"})),
        ];

        validate_parallel_streaming_tool_calls(&results, &expected_calls);
    }

    // =============================================================================
    // 5. LARGE SCALE PARALLEL CALLS (5+ TOOLS)
    // =============================================================================

    #[tokio::test]
    async fn test_large_scale_parallel_tool_calls() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk(
                r#"<TOOLCALL>[
    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Seattle", "state": "WA", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Denver", "state": "CO", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Miami", "state": "FL", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Phoenix", "state": "AZ", "unit": "fahrenheit"}},
    {"name": "get_current_weather", "arguments": {"city": "Chicago", "state": "IL", "unit": "fahrenheit"}}
]</TOOLCALL>"#.to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        // Should have 7 tool calls
        let tool_call_count = results
            .iter()
            .map(|r| {
                r.data.as_ref().map_or(0, |d| {
                    d.choices
                        .iter()
                        .map(|c| c.delta.tool_calls.as_ref().map_or(0, |tc| tc.len()))
                        .sum::<usize>()
                })
            })
            .sum::<usize>();

        assert_eq!(tool_call_count, 7, "Should have exactly 7 tool calls");
    }

    // =============================================================================
    // 6. COMPLEX NESTED ARGUMENTS IN PARALLEL CALLS
    // =============================================================================

    #[tokio::test]
    async fn test_parallel_complex_nested_arguments() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![test_utils::create_mock_response_chunk(
            r#"<TOOLCALL>[
    {
        "name": "get_weather_forecast",
        "arguments": {
            "location": {
                "city": "Dallas",
                "state": "TX",
                "country": "USA",
                "coordinates": {"lat": 32.7767, "lon": -96.7970}
            },
            "options": {
                "days": 7,
                "units": "fahrenheit",
                "include_hourly": true,
                "include_alerts": true,
                "metrics": ["temperature", "humidity", "wind_speed", "precipitation"]
            }
        }
    },
    {
        "name": "get_air_quality_data",
        "arguments": {
            "location": {
                "coordinates": {"lat": 32.7767, "lon": -96.7970},
                "radius_km": 25
            },
            "pollutants": ["pm2.5", "pm10", "ozone", "no2", "so2", "co"],
            "time_range": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-07T23:59:59Z"
            }
        }
    }
]</TOOLCALL>"#
                .to_string(),
            0,
        )];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        // Should have 2 tool calls with complex nested arguments
        let tool_call_count = results
            .iter()
            .map(|r| {
                r.data.as_ref().map_or(0, |d| {
                    d.choices
                        .iter()
                        .map(|c| c.delta.tool_calls.as_ref().map_or(0, |tc| tc.len()))
                        .sum::<usize>()
                })
            })
            .sum::<usize>();

        assert_eq!(tool_call_count, 2, "Should have exactly 2 tool calls");

        // Validate that complex nested structures are preserved
        let tool_call_results: Vec<_> = results
            .iter()
            .filter(|r| {
                r.data
                    .as_ref()
                    .is_some_and(|d| d.choices.iter().any(|c| c.delta.tool_calls.is_some()))
            })
            .collect();

        if let Some(result) = tool_call_results.first()
            && let Some(ref data) = result.data
        {
            for choice in &data.choices {
                if let Some(ref tool_calls) = choice.delta.tool_calls {
                    for tool_call in tool_calls {
                        if let Some(ref function) = tool_call.function
                            && let Some(args_str) = &function.arguments
                        {
                            let parsed_args: serde_json::Value = serde_json::from_str(args_str)
                                .expect("Arguments should be valid JSON");

                            // Verify nested structure is preserved
                            if function.name.as_deref() == Some("get_weather_forecast") {
                                assert!(parsed_args["location"]["coordinates"]["lat"].is_number());
                                assert!(parsed_args["options"]["metrics"].is_array());
                            } else if function.name.as_deref() == Some("get_air_quality_data") {
                                assert!(parsed_args["pollutants"].is_array());
                                assert!(parsed_args["time_range"]["start"].is_string());
                            }
                        }
                    }
                }
            }
        }
    }

    // =============================================================================
    // 7. ERROR HANDLING AND EDGE CASES
    // =============================================================================

    #[tokio::test]
    async fn test_parallel_partial_malformed_calls() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk(
                r#"<TOOLCALL>[
    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},
    {"invalid": "malformed_call"},
    {"name": "get_current_weather", "arguments": {"city": "Orlando", "state": "FL", "unit": "fahrenheit"}}
]</TOOLCALL>"#.to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        // Should still parse the valid tool calls despite the malformed one
        let tool_call_count = results
            .iter()
            .map(|r| {
                r.data.as_ref().map_or(0, |d| {
                    d.choices
                        .iter()
                        .map(|c| c.delta.tool_calls.as_ref().map_or(0, |tc| tc.len()))
                        .sum::<usize>()
                })
            })
            .sum::<usize>();

        // Should have at least the valid tool calls
        assert!(
            tool_call_count >= 1,
            "Should have at least 1 valid tool call"
        );
    }

    #[tokio::test]
    async fn test_parallel_streaming_interrupted() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        // Simulate a stream that gets cut off mid-tool-call
        let input_chunks = vec![
            test_utils::create_mock_response_chunk("<TOOLCALL>[".to_string(), 0),
            test_utils::create_mock_response_chunk(
                r#"    {"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}},"#.to_string(),
                0,
            ),
            test_utils::create_mock_response_chunk(
                r#"    {"name": "get_current_weather", "arguments": {"city": "Orlando""#.to_string(),
                0,
            ),
            // Stream ends abruptly without closing the JSON array or TOOLCALL tag
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply_with_finish_reason(input_stream).collect().await;

        // Should still handle the incomplete stream gracefully
        assert!(
            !results.is_empty(),
            "Should have results even with incomplete stream"
        );

        // Should try to parse whatever content was accumulated
        let has_some_content = results.iter().any(|r| {
            r.data.as_ref().is_some_and(|d| {
                d.choices
                    .iter()
                    .any(|c| c.delta.content.is_some() || c.delta.tool_calls.is_some())
            })
        });

        assert!(
            has_some_content,
            "Should have some content despite incomplete stream"
        );
    }

    #[tokio::test]
    async fn test_parallel_empty_tool_calls_array() {
        let jail = JailedStream::builder()
            .tool_call_parser("nemotron_deci")
            .build();

        let input_chunks = vec![
            test_utils::create_mock_response_chunk("I'll help you with that. ".to_string(), 0),
            test_utils::create_mock_response_chunk("<TOOLCALL>[]</TOOLCALL>".to_string(), 0),
            test_utils::create_mock_response_chunk(
                " Actually, I don't need any tools for this.".to_string(),
                0,
            ),
        ];

        let input_stream = stream::iter(input_chunks);
        let results: Vec<_> = jail.apply(input_stream).collect().await;

        assert!(!results.is_empty(), "Should have results");

        // Should have normal text content but no tool calls
        let has_normal_text = results.iter().any(|r| {
            r.data.as_ref().is_some_and(|d| {
                d.choices.iter().any(|c| {
                    c.delta.content.as_ref().is_some_and(|content| {
                        content.contains("I'll help you")
                            || content.contains("don't need any tools")
                    })
                })
            })
        });

        assert!(has_normal_text, "Should have normal text content");

        let tool_call_count = results
            .iter()
            .map(|r| {
                r.data.as_ref().map_or(0, |d| {
                    d.choices
                        .iter()
                        .map(|c| c.delta.tool_calls.as_ref().map_or(0, |tc| tc.len()))
                        .sum::<usize>()
                })
            })
            .sum::<usize>();

        assert_eq!(
            tool_call_count, 0,
            "Should have no tool calls for empty array"
        );
    }
}
