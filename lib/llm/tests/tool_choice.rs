// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionNamedToolChoice,
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionToolChoiceOption,
    ChatCompletionToolType, CreateChatCompletionRequest, FunctionCallStream, FunctionName,
};
use dynamo_llm::protocols::common;
use dynamo_llm::protocols::common::llm_backend::BackendOutput;
use dynamo_llm::protocols::openai::DeltaGeneratorExt;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

fn create_test_request() -> NvCreateChatCompletionRequest {
    let messages = vec![ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text("test".to_string()),
            name: None,
        },
    )];

    NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            model: "test-model".to_string(),
            messages,
            stream: Some(false),
            stream_options: None,
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
        unsupported_fields: Default::default(),
    }
}

fn build_backend_output(text: &str) -> BackendOutput {
    BackendOutput {
        token_ids: vec![],
        tokens: vec![],
        text: Some(text.to_string()),
        cum_log_probs: None,
        log_probs: None,
        top_logprobs: None,
        finish_reason: Some(common::FinishReason::Stop),
        index: Some(0),
        completion_usage: None,
        disaggregated_params: None,
    }
}

#[test]
fn test_named_tool_choice_parses_json() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));

    let mut generator = request.response_generator("req-1".to_string());
    let backend_output = build_backend_output(r#"{"location":"Paris"}"#);
    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let choice = &response.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    let delta = &choice.delta;
    assert!(delta.content.is_none());
    let tool_calls = delta.tool_calls.as_ref().unwrap();

    // In streaming mode, we emit 2 chunks: first with id/name, second with arguments
    assert!(
        !tool_calls.is_empty(),
        "Should have at least 1 tool call chunk"
    );

    // Find the chunk with the name (first chunk)
    let name_chunk = tool_calls
        .iter()
        .find(|tc| tc.function.as_ref().and_then(|f| f.name.as_ref()).is_some());
    assert!(name_chunk.is_some(), "Should have chunk with name");
    let name_chunk = name_chunk.unwrap();

    assert_eq!(name_chunk.index, 0);
    assert_eq!(name_chunk.id.as_deref(), Some("call_1"));
    assert_eq!(
        name_chunk.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );

    // Arguments may be in the same chunk or a subsequent one
    let has_arguments = tool_calls.iter().any(|tc| {
        tc.function
            .as_ref()
            .and_then(|f| f.arguments.as_ref())
            .is_some_and(|args| !args.is_empty())
    });
    assert!(has_arguments, "Should have arguments in some chunk");
}

#[test]
fn test_required_tool_choice_parses_json_array() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-2".to_string());
    let backend_output = build_backend_output(
        r#"[{"name":"search","parameters":{"query":"rust"}},
            {"name":"summarize","parameters":{"topic":"memory"}}]"#,
    );
    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let choice = &response.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
    let delta = &choice.delta;
    assert!(delta.content.is_none());
    let tool_calls = delta.tool_calls.as_ref().unwrap();

    // With incremental streaming, we emit separate chunks for name and arguments
    // Expected: 4 chunks total (2 per tool: name chunk + arguments chunk)
    assert_eq!(tool_calls.len(), 4);

    // First tool: name chunk
    assert_eq!(tool_calls[0].index, 0);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert!(tool_calls[0].id.is_some());

    // First tool: arguments chunk
    assert_eq!(tool_calls[1].index, 0);
    assert!(tool_calls[1].function.as_ref().unwrap().name.is_none());
    assert!(
        tool_calls[1]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap()
            .contains("rust")
    );

    // Second tool: name chunk
    assert_eq!(tool_calls[2].index, 1);
    assert_eq!(
        tool_calls[2].function.as_ref().unwrap().name.as_deref(),
        Some("summarize")
    );
    assert!(tool_calls[2].id.is_some());

    // Second tool: arguments chunk
    assert_eq!(tool_calls[3].index, 1);
    assert!(tool_calls[3].function.as_ref().unwrap().name.is_none());
    assert!(
        tool_calls[3]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap()
            .contains("memory")
    );
}

#[test]
fn test_tool_choice_parse_failure_suppresses_text() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-3".to_string());
    let backend_output = build_backend_output("not-json");
    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let delta = &response.choices[0].delta;
    // When tool_choice is active but parsing fails, we suppress the text output
    assert!(delta.content.is_none());
    assert!(delta.tool_calls.is_none());
}

#[test]
fn test_streaming_named_tool_incremental() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));

    let mut generator = request.response_generator("req-stream-1".to_string());

    // Simulate streaming chunks
    // For simplicity in testing, send complete JSON in final chunk
    let chunks = [r#"{"location":"Paris","unit":"celsius"}"#];

    let mut all_responses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(chunk.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if i == chunks.len() - 1 {
                Some(common::FinishReason::Stop)
            } else {
                None
            },
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("streaming chunk");
        all_responses.push(response);
    }

    // Last response should have finish_reason
    let last_response = all_responses.last().unwrap();
    assert_eq!(
        last_response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );

    // Should have tool_calls somewhere in the stream
    let has_tool_calls = all_responses
        .iter()
        .any(|r| r.choices[0].delta.tool_calls.is_some());
    assert!(has_tool_calls, "No tool calls found in any response");
}

#[test]
fn test_streaming_required_tool_parallel() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-stream-2".to_string());

    // Simulate streaming array of tool calls
    let chunks = [
        r#"[{"name":"search","parameters":{"query":"rust"}},"#,
        r#"{"name":"summarize","parameters":{"topic":"memory"}}]"#,
    ];

    let mut all_responses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(chunk.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if i == chunks.len() - 1 {
                Some(common::FinishReason::Stop)
            } else {
                None
            },
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("streaming chunk");
        all_responses.push(response);
    }

    // Final chunk should have finish_reason = ToolCalls
    let last_response = all_responses.last().unwrap();
    assert_eq!(
        last_response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );

    // Should have detected both tools
    let mut found_search = false;
    let mut found_summarize = false;
    for resp in &all_responses {
        if let Some(tool_calls) = &resp.choices[0].delta.tool_calls {
            for tc in tool_calls {
                if let Some(func) = &tc.function
                    && let Some(name) = &func.name
                {
                    if name == "search" {
                        found_search = true;
                    }
                    if name == "summarize" {
                        found_summarize = true;
                    }
                }
            }
        }
    }
    assert!(found_search, "Should detect search tool");
    assert!(found_summarize, "Should detect summarize tool");
}

#[test]
fn test_streaming_with_incremental_arguments() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "search".to_string(),
            },
        },
    ));

    let mut generator = request.response_generator("req-stream-3".to_string());

    // Character-by-character streaming
    let full_json = r#"{"query":"rust programming"}"#;
    let mut responses = Vec::new();

    for ch in full_json.chars() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(ch.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("char chunk");
        responses.push(response);
    }

    // Should have suppressed raw text output
    for resp in &responses {
        assert!(resp.choices[0].delta.content.is_none());
    }
}

#[test]
fn test_no_streaming_when_tool_choice_none() {
    let request = create_test_request();
    // tool_choice = None (default)

    let mut generator = request.response_generator("req-stream-4".to_string());

    let backend_output = BackendOutput {
        token_ids: vec![],
        tokens: vec![],
        text: Some("Hello world".to_string()),
        cum_log_probs: None,
        log_probs: None,
        top_logprobs: None,
        finish_reason: None,
        index: Some(0),
        completion_usage: None,
        disaggregated_params: None,
    };

    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("normal text");

    // Should have text content, not tool_calls
    assert_eq!(
        response.choices[0].delta.content.as_deref(),
        Some("Hello world")
    );
    assert!(response.choices[0].delta.tool_calls.is_none());
}

#[test]
fn test_true_incremental_streaming_named() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));

    let mut generator = request.response_generator("req-stream-inc-1".to_string());

    // Simulate realistic token-by-token streaming
    let chunks = vec![
        r#"{"#,
        r#""location""#,
        r#":"#,
        r#""Paris""#,
        r#","#,
        r#""unit""#,
        r#":"#,
        r#""celsius""#,
        r#"}"#,
    ];

    let mut responses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(chunk.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if i == chunks.len() - 1 {
                Some(common::FinishReason::Stop)
            } else {
                None
            },
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("chunk");
        responses.push(response);
    }

    // Should have emitted tool_calls in one of the early chunks
    let first_tool_call_idx = responses
        .iter()
        .position(|r| r.choices[0].delta.tool_calls.is_some())
        .expect("Should find tool_calls in stream");

    // First tool call should have id, type, name
    let first_tc = &responses[first_tool_call_idx].choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap()[0];
    assert!(first_tc.id.is_some());
    assert_eq!(first_tc.r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        first_tc.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );

    // Should have multiple chunks with arguments deltas
    let args_chunks: Vec<_> = responses
        .iter()
        .filter_map(|r| r.choices[0].delta.tool_calls.as_ref())
        .flat_map(|tcs| tcs.iter())
        .filter_map(|tc| tc.function.as_ref()?.arguments.as_ref())
        .collect();

    assert!(
        args_chunks.len() > 1,
        "Should have multiple argument delta chunks"
    );
}

#[test]
fn test_true_incremental_streaming_parallel() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-stream-inc-2".to_string());

    // Simulate streaming: array with two tool calls
    let chunks = [
        r#"["#,
        r#"{"name":"search","#,
        r#""parameters":{"query":"rust"}"#,
        r#"},"#,
        r#"{"name":"summarize","#,
        r#""parameters":{"topic":"memory"}"#,
        r#"}]"#,
    ];

    let mut responses = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(chunk.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if i == chunks.len() - 1 {
                Some(common::FinishReason::Stop)
            } else {
                None
            },
            index: Some(0),
            completion_usage: None,
            disaggregated_params: None,
        };

        let response = generator
            .choice_from_postprocessor(backend_output)
            .expect("chunk");
        responses.push(response);
    }

    // Count tool call initializations (first chunks with names)
    let mut tool_names_seen = std::collections::HashSet::new();
    for resp in &responses {
        if let Some(tool_calls) = &resp.choices[0].delta.tool_calls {
            for tc in tool_calls {
                if let Some(func) = &tc.function
                    && let Some(name) = &func.name
                {
                    tool_names_seen.insert(name.clone());
                }
            }
        }
    }

    assert_eq!(tool_names_seen.len(), 2, "Should detect both tool calls");
    assert!(tool_names_seen.contains("search"));
    assert!(tool_names_seen.contains("summarize"));

    // Verify that tool calls are streamed incrementally, not just at the end
    let chunks_with_tool_calls: Vec<_> = responses
        .iter()
        .enumerate()
        .filter(|(_, r)| r.choices[0].delta.tool_calls.is_some())
        .map(|(i, _)| i)
        .collect();

    assert!(
        chunks_with_tool_calls.len() > 1,
        "Should have multiple chunks with tool_calls (not just final)"
    );
}

/// Helper function to create a streaming chunk
fn create_chunk(
    index: u32,
    role: Option<dynamo_async_openai::types::Role>,
    tool_call_chunk: Option<dynamo_async_openai::types::ChatCompletionMessageToolCallChunk>,
    finish_reason: Option<dynamo_async_openai::types::FinishReason>,
) -> dynamo_async_openai::types::CreateChatCompletionStreamResponse {
    use dynamo_async_openai::types::{
        ChatCompletionStreamResponseDelta, CreateChatCompletionStreamResponse,
    };

    CreateChatCompletionStreamResponse {
        id: "test".to_string(),
        choices: vec![ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role,
                content: None,
                function_call: None,
                tool_calls: tool_call_chunk.map(|chunk| vec![chunk]),
                refusal: None,
                reasoning_content: None,
            },
            finish_reason,
            logprobs: None,
        }],
        created: 1234567890,
        model: "test-model".to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        service_tier: None,
        usage: None,
        nvext: None,
    }
}

#[tokio::test]
async fn test_aggregator_named_tool_accumulates_arguments() {
    use dynamo_llm::protocols::Annotated;
    use dynamo_llm::protocols::openai::ParsingOptions;
    use dynamo_llm::protocols::openai::chat_completions::aggregator::DeltaAggregator;
    use futures::stream;

    // Simulate streaming chunks for named tool choice: get_weather
    let chunks = vec![
        // Chunk 1: role
        create_chunk(
            0,
            Some(dynamo_async_openai::types::Role::Assistant),
            None,
            None,
        ),
        // Chunk 2: tool call start (id, type, name, empty arguments)
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call_1".to_string()),
                r#type: Some(ChatCompletionToolType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".to_string()),
                    arguments: Some(String::new()),
                }),
            }),
            None,
        ),
        // Chunk 3: first part of arguments (raw JSON fragment from buffer)
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: None,
                r#type: None,
                function: Some(FunctionCallStream {
                    name: None,
                    arguments: Some(r#"{"location":"Paris""#.to_string()),
                }),
            }),
            None,
        ),
        // Chunk 4: second part of arguments (continuation)
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: None,
                r#type: None,
                function: Some(FunctionCallStream {
                    name: None,
                    arguments: Some(r#","unit":"celsius"}"#.to_string()),
                }),
            }),
            None,
        ),
        // Chunk 5: finish
        create_chunk(
            0,
            None,
            None,
            Some(dynamo_async_openai::types::FinishReason::Stop),
        ),
    ];

    // Convert to Annotated stream
    let annotated_chunks: Vec<Annotated<_>> = chunks
        .into_iter()
        .map(|chunk| Annotated {
            data: Some(chunk),
            id: None,
            event: None,
            comment: None,
        })
        .collect();

    let stream = Box::pin(stream::iter(annotated_chunks));

    // Aggregate the stream
    let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();

    // Verify aggregated response
    assert_eq!(response.choices.len(), 1);
    let choice = &response.choices[0];

    // Check tool calls
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);

    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.id, "call_1");
    assert_eq!(tool_call.function.name, "get_weather");
    // THIS IS THE KEY ASSERTION - arguments should be accumulated!
    assert_eq!(
        tool_call.function.arguments, r#"{"location":"Paris","unit":"celsius"}"#,
        "Arguments should be fully accumulated from all chunks"
    );
}

#[tokio::test]
async fn test_aggregator_required_tool_parallel_calls() {
    use dynamo_async_openai::types::{
        ChatCompletionMessageToolCallChunk, ChatCompletionToolType, FunctionCallStream,
    };
    use dynamo_llm::protocols::Annotated;
    use dynamo_llm::protocols::openai::ParsingOptions;
    use dynamo_llm::protocols::openai::chat_completions::aggregator::DeltaAggregator;
    use futures::stream;

    // Simulate streaming chunks for required tool choice with parallel calls
    let chunks = vec![
        // Chunk 1: role
        create_chunk(
            0,
            Some(dynamo_async_openai::types::Role::Assistant),
            None,
            None,
        ),
        // Chunk 2: first tool call start
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call_1".to_string()),
                r#type: Some(ChatCompletionToolType::Function),
                function: Some(FunctionCallStream {
                    name: Some("search".to_string()),
                    arguments: Some(String::new()),
                }),
            }),
            None,
        ),
        // Chunk 3: first tool arguments
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: None,
                r#type: None,
                function: Some(FunctionCallStream {
                    name: None,
                    arguments: Some(r#"{"query":"rust"}"#.to_string()),
                }),
            }),
            None,
        ),
        // Chunk 4: second tool call start
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 1,
                id: Some("call_2".to_string()),
                r#type: Some(ChatCompletionToolType::Function),
                function: Some(FunctionCallStream {
                    name: Some("summarize".to_string()),
                    arguments: Some(String::new()),
                }),
            }),
            None,
        ),
        // Chunk 5: second tool arguments (partial)
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 1,
                id: None,
                r#type: None,
                function: Some(FunctionCallStream {
                    name: None,
                    arguments: Some(r#"{"text":"#.to_string()),
                }),
            }),
            None,
        ),
        // Chunk 6: second tool arguments (rest)
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 1,
                id: None,
                r#type: None,
                function: Some(FunctionCallStream {
                    name: None,
                    arguments: Some(r#""long article"}"#.to_string()),
                }),
            }),
            None,
        ),
        // Chunk 7: finish
        create_chunk(
            0,
            None,
            None,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls),
        ),
    ];

    // Convert to Annotated stream
    let annotated_chunks: Vec<Annotated<_>> = chunks
        .into_iter()
        .map(|chunk| Annotated {
            data: Some(chunk),
            id: None,
            event: None,
            comment: None,
        })
        .collect();

    let stream = Box::pin(stream::iter(annotated_chunks));

    // Aggregate the stream
    let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();

    // Verify aggregated response
    assert_eq!(response.choices.len(), 1);
    let choice = &response.choices[0];

    // Check tool calls
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2, "Should have 2 tool calls");

    // Verify first tool call
    let tool_call_1 = &tool_calls[0];
    assert_eq!(tool_call_1.id, "call_1");
    assert_eq!(tool_call_1.function.name, "search");
    assert_eq!(
        tool_call_1.function.arguments, r#"{"query":"rust"}"#,
        "First tool arguments should be complete"
    );

    // Verify second tool call - THIS IS THE CRITICAL TEST
    let tool_call_2 = &tool_calls[1];
    assert_eq!(tool_call_2.id, "call_2");
    assert_eq!(tool_call_2.function.name, "summarize");
    assert_eq!(
        tool_call_2.function.arguments, r#"{"text":"long article"}"#,
        "Second tool arguments should be accumulated from multiple chunks"
    );

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
}
