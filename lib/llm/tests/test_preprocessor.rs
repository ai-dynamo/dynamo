// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use dynamo_async_openai::types::ChatCompletionToolChoiceOption;
use dynamo_async_openai::types::CreateChatCompletionRequest;
use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionStreamResponseDelta, FinishReason as OAIFinishReason, Role,
};
use dynamo_llm::preprocessor::{
    ANNOTATION_POSSIBLE_TOOL_CALL, PossibleToolCallAnnotation, apply_tool_calling_jail_internal,
    maybe_enable_tool_call,
};
use dynamo_llm::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use dynamo_parsers::tool_calling::parsers::detect_tool_call_start;
use dynamo_runtime::pipeline::ResponseStream;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::stream::{self, StreamExt};
use std::sync::Arc;

#[allow(deprecated)]
// Helper function to create a mock chat response chunk
fn create_mock_response_chunk(
    content: String,
    index: u32,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
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
    };

    Annotated {
        data: Some(response),
        id: None,
        event: None,
        comment: None,
    }
}

#[allow(deprecated)]
// Helper function to create a final response chunk with finish reason
fn create_final_response_chunk(index: u32) -> Annotated<NvCreateChatCompletionStreamResponse> {
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
        finish_reason: Some(OAIFinishReason::Stop),
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
    };

    Annotated {
        data: Some(response),
        id: None,
        event: None,
        comment: None,
    }
}

// Mock async engine context for testing
#[derive(Debug)]
struct MockAsyncEngineContext {
    id: String,
    stopped: std::sync::atomic::AtomicBool,
}

impl MockAsyncEngineContext {
    fn new(id: String) -> Self {
        Self {
            id,
            stopped: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl dynamo_runtime::pipeline::AsyncEngineContext for MockAsyncEngineContext {
    fn id(&self) -> &str {
        &self.id
    }

    fn stop(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn stop_generating(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn kill(&self) {
        self.stopped
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn is_killed(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Relaxed)
    }

    async fn stopped(&self) {
        // No-op for testing
    }

    async fn killed(&self) {
        // No-op for testing
    }

    fn link_child(&self, _: Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>) {
        // No-op for testing
    }
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_with_tool_call_detection() {
    // Create a stream with tool call content that SHOULD trigger jailing
    let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id".to_string()));

    // Create chunks that represent a tool call being generated
    let chunks = vec![
        create_mock_response_chunk("<TOOLCALL>".to_string(), 0),
        create_mock_response_chunk("[{\"name\": \"get_weather\", ".to_string(), 0),
        create_mock_response_chunk(
            "\"arguments\": {\"location\": \"San Francisco\"}}]".to_string(),
            0,
        ),
        create_mock_response_chunk("</TOOLCALL>".to_string(), 0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    // Apply the jail with nemotron_deci parser - should trigger jailing on first chunk
    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("nemotron_deci".to_string())).await;

    // Collect all results
    let results: Vec<_> = jailed_stream.collect().await;

    // Verify that jailing was triggered
    assert!(!results.is_empty(), "Should have some results");

    // Results should be of length 1
    // First Stream: [{"name": "get_weather", "arguments":"{"location": "San Francisco"}}]"

    assert_eq!(results.len(), 1);
    assert!(
        results[0].data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls
            .is_some()
    );
    let tools = results[0].data.as_ref().unwrap().choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(tools.len(), 1);
    let name = tools[0].function.as_ref().unwrap().name.as_ref().unwrap();
    let arguments = serde_json::from_str::<serde_json::Value>(
        tools[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(name, "get_weather");
    assert_eq!(arguments["location"], "San Francisco");
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_no_tool_calls() {
    // Create a stream with regular content that should NOT trigger jailing
    let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-2".to_string()));

    let chunks = vec![
        create_mock_response_chunk("Hello, ".to_string(), 0),
        create_mock_response_chunk("how can I ".to_string(), 0),
        create_mock_response_chunk("help you today?".to_string(), 0),
        create_final_response_chunk(0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    // Apply the jail with nemotron_deci parser - regular text should NOT be jailed
    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("nemotron_deci".to_string())).await;

    // Collect all results
    let results: Vec<_> = jailed_stream.collect().await;

    // Should have results and they should NOT be jailed (content should be preserved)
    assert!(!results.is_empty(), "Should have results");
    assert_eq!(results.len(), 4, "Should have all 4 chunks");

    // Verify that content is NOT jailed - first few chunks should have their original content
    for (i, result) in results.iter().take(3).enumerate() {
        if let Some(ref response_data) = result.data {
            let expected_content = match i {
                0 => "Hello, ",
                1 => "how can I ",
                2 => "help you today?",
                _ => unreachable!(),
            };
            assert_eq!(
                response_data.choices[0].delta.content.as_deref(),
                Some(expected_content),
                "Chunk {} should have original content, not be jailed",
                i
            );
            // Should NOT have annotation events for regular content
            assert!(
                result.event.is_none(),
                "Regular content should not have annotation events"
            );
        }
    }

    // Last chunk should be the final response with finish reason
    if let Some(last_result) = results.last()
        && let Some(ref response_data) = last_result.data
    {
        assert_eq!(
            response_data.choices[0].finish_reason,
            Some(OAIFinishReason::Stop)
        );
    }
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_with_empty_stream() {
    let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-3".to_string()));

    let chunks: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = vec![];
    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream = apply_tool_calling_jail_internal(response_stream, None).await;
    let results: Vec<_> = jailed_stream.collect().await;

    assert!(results.is_empty(), "Empty stream should produce no results");
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_with_different_parsers() {
    let mock_context = Arc::new(MockAsyncEngineContext::new("test-request-id-4".to_string()));

    // Test with hermes parser format
    let chunks = vec![
        create_mock_response_chunk("<tool_call>".to_string(), 0),
        create_mock_response_chunk(
            "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}".to_string(),
            0,
        ),
        create_mock_response_chunk("</tool_call>".to_string(), 0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("hermes".to_string())).await;
    let results: Vec<_> = jailed_stream.collect().await;

    assert!(!results.is_empty(), "Should have results for hermes parser");
}

#[tokio::test]
async fn test_detect_tool_call_start_different_parsers() {
    // Test nemotron_deci parser
    assert!(detect_tool_call_start("<TOOLCALL>", Some("nemotron_deci")).unwrap());
    assert!(!detect_tool_call_start("Hello world", Some("nemotron_deci")).unwrap());
    assert!(!detect_tool_call_start("<tool_call>", Some("nemotron_deci")).unwrap()); // Wrong format

    // Test hermes parser - now also detects JSON patterns
    assert!(detect_tool_call_start("<tool_call>", Some("hermes")).unwrap());
    assert!(detect_tool_call_start("{\"name\": \"test\"}", Some("hermes")).unwrap()); // JSON detection
    assert!(!detect_tool_call_start("Hello world", Some("hermes")).unwrap());
    assert!(!detect_tool_call_start("<TOOLCALL>", Some("hermes")).unwrap()); // Wrong format

    // Test phi4 parser
    assert!(detect_tool_call_start("functools[", Some("phi4")).unwrap());
    assert!(detect_tool_call_start("{\"name\": \"test\"}", Some("phi4")).unwrap()); // JSON detection
    assert!(!detect_tool_call_start("Hello world", Some("phi4")).unwrap());

    // Test mistral parser
    assert!(detect_tool_call_start("[{", Some("mistral")).unwrap());
    assert!(detect_tool_call_start("[TOOL_CALLS]", Some("mistral")).unwrap());
    assert!(!detect_tool_call_start("Hello world", Some("mistral")).unwrap());

    // Test llama3_json parser
    assert!(detect_tool_call_start("<|python_tag|>", Some("llama3_json")).unwrap());
    assert!(detect_tool_call_start("{\"name\": \"test\"}", Some("llama3_json")).unwrap()); // JSON detection

    // Test default parser (should behave like nemotron_deci)
    assert!(detect_tool_call_start("<TOOLCALL>", None).unwrap());
    assert!(detect_tool_call_start("{\"name\": \"test\"}", None).unwrap()); // JSON detection
    assert!(!detect_tool_call_start("Hello world", None).unwrap());
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_hermes_parser() {
    // Test with hermes parser format
    let mock_context = Arc::new(MockAsyncEngineContext::new(
        "test-request-id-hermes".to_string(),
    ));

    let chunks = vec![
        create_mock_response_chunk("I'll help you with that. ".to_string(), 0),
        create_mock_response_chunk("<tool_call>".to_string(), 0), // This should trigger jailing
        create_mock_response_chunk(
            "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}".to_string(),
            0,
        ),
        create_mock_response_chunk("</tool_call>".to_string(), 0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("hermes".to_string())).await;
    let results: Vec<_> = jailed_stream.collect().await;

    assert!(!results.is_empty(), "Should have results for hermes parser");

    // Results should be of length 2
    // First Stream : I'll help you with that.
    // Second Stream : [{"name": "get_weather", "arguments":"{"location": "Tokyo"}}]" (jailed)
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].data.as_ref().unwrap().choices[0].delta.content,
        Some("I'll help you with that. ".to_string())
    );
    assert!(
        results[1].data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls
            .is_some()
    );
    let tools = results[1].data.as_ref().unwrap().choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(tools.len(), 1);
    let name = tools[0].function.as_ref().unwrap().name.as_ref().unwrap();
    let arguments = serde_json::from_str::<serde_json::Value>(
        tools[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(name, "get_weather");
    assert_eq!(arguments["location"], "Tokyo");
}

#[tokio::test]
async fn test_possible_tool_call_annotation_serialization() {
    let annotation = PossibleToolCallAnnotation {
        possible_tokens: 5,
        possible_content: "test content".to_string(),
        parser_used: Some("nemotron_deci".to_string()),
    };

    let annotated_result = annotation.to_annotation::<NvCreateChatCompletionStreamResponse>();
    assert!(
        annotated_result.is_ok(),
        "Should be able to create annotation"
    );

    let annotated = annotated_result.unwrap();
    assert_eq!(
        annotated.event,
        Some(ANNOTATION_POSSIBLE_TOOL_CALL.to_string())
    );
    assert!(annotated.comment.is_some(), "Should have comment");

    // Test deserialization
    let parsed_annotation = PossibleToolCallAnnotation::from_annotation(&annotated);
    assert!(
        parsed_annotation.is_ok(),
        "Should be able to parse annotation"
    );

    let parsed = parsed_annotation.unwrap();
    assert!(parsed.is_some(), "Should have parsed annotation");

    let parsed = parsed.unwrap();
    assert_eq!(parsed.possible_tokens, 5);
    assert_eq!(parsed.possible_content, "test content");
    assert_eq!(parsed.parser_used, Some("nemotron_deci".to_string()));
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_mistral_parser_with_no_tool_call_start_token() {
    let mock_context = Arc::new(MockAsyncEngineContext::new(
        "test-request-id-mistral".to_string(),
    ));

    let chunks = vec![
        create_mock_response_chunk("Hey How".to_string(), 0),
        create_mock_response_chunk("are you? ".to_string(), 0),
        create_mock_response_chunk(r#"[{"name": "get_weather", "arguments":"#.to_string(), 0),
        create_mock_response_chunk(
            r#"{"location": "San Francisco", "unit": "fahrenheit"}}]"#.to_string(),
            0,
        ),
        create_final_response_chunk(0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("mistral".to_string())).await;

    let results: Vec<_> = jailed_stream.collect().await;

    assert!(
        !results.is_empty(),
        "Should have results for mistral parser"
    );
    // Results should be of length 4
    // First Stream : Hey How
    // Second Stream : are you?
    // Third Stream : None (final response chunk)
    // Fourth Stream : [{"name": "get_weather", "arguments":"{"location": "San Francisco", "unit": "fahrenheit"}}]" (jailed)
    assert_eq!(results.len(), 4);

    // First two normal text
    assert_eq!(
        results[0].data.as_ref().unwrap().choices[0].delta.content,
        Some("Hey How".to_string())
    );
    assert_eq!(
        results[1].data.as_ref().unwrap().choices[0].delta.content,
        Some("are you? ".to_string())
    );
    assert_eq!(
        results[2].data.as_ref().unwrap().choices[0].delta.content,
        None
    );

    // Final tool call
    assert!(
        results[3].data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls
            .is_some()
    );
    let tools = results[3].data.as_ref().unwrap().choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(tools.len(), 1);
    let name = tools[0].function.as_ref().unwrap().name.as_ref().unwrap();
    let arguments = serde_json::from_str::<serde_json::Value>(
        tools[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(name, "get_weather");
    assert_eq!(arguments["location"], "San Francisco");
    assert_eq!(arguments["unit"], "fahrenheit");
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_mistral_parser_with_false_positive_tool_start() {
    let mock_context = Arc::new(MockAsyncEngineContext::new(
        "test-request-id-mistral".to_string(),
    ));

    let chunks = vec![
        create_mock_response_chunk("Hey How".to_string(), 0),
        create_mock_response_chunk("are { you? ".to_string(), 0),
        create_final_response_chunk(0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("mistral".to_string())).await;
    let results: Vec<_> = jailed_stream.collect().await;

    assert!(
        !results.is_empty(),
        "Should have results for mistral parser"
    );
    // Results should be of length 3
    // First Stream : Hey How
    // Second Stream : None (final response chunk)
    // Third Stream : are { you? (normal text field from tool-call-parse-aggregate)
    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0].data.as_ref().unwrap().choices[0].delta.content,
        Some("Hey How".to_string())
    );
    assert_eq!(
        results[1].data.as_ref().unwrap().choices[0].delta.content,
        None
    );
    assert_eq!(
        results[2].data.as_ref().unwrap().choices[0].delta.content,
        Some("are { you?".to_string())
    );
}

#[tokio::test]
async fn test_apply_tool_calling_jail_internal_mistral_parser_with_false_positive_tool_start_and_tool_call_token()
 {
    let mock_context = Arc::new(MockAsyncEngineContext::new(
        "test-request-id-mistral".to_string(),
    ));

    let chunks = vec![
        create_mock_response_chunk("Hey How".to_string(), 0),
        create_mock_response_chunk("are { you? ".to_string(), 0),
        create_mock_response_chunk(
            r#"[TOOL_CALLS][{"name": "get_weather", "arguments":"#.to_string(),
            0,
        ),
        create_mock_response_chunk(
            r#"{"location": "San Francisco", "unit": "fahrenheit"}}]"#.to_string(),
            0,
        ),
        create_final_response_chunk(0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("mistral".to_string())).await;
    let results: Vec<_> = jailed_stream.collect().await;

    assert!(
        !results.is_empty(),
        "Should have results for mistral parser"
    );

    // Results should be of length 3
    // First Stream : Hey How
    // Second Stream : None (final response chunk)
    // Third Stream : Content: are { you? , Tool Calls: [{"name": "get_weather", "arguments":"{"location": "San Francisco", "unit": "fahrenheit"}}]"
    assert_eq!(results.len(), 3);
    assert_eq!(
        results[0].data.as_ref().unwrap().choices[0].delta.content,
        Some("Hey How".to_string())
    );
    assert_eq!(
        results[1].data.as_ref().unwrap().choices[0].delta.content,
        None
    );
    assert_eq!(
        results[2].data.as_ref().unwrap().choices[0].delta.content,
        Some("are { you?".to_string())
    );
    assert!(
        results[2].data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls
            .is_some()
    );
    let tools = results[2].data.as_ref().unwrap().choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(tools.len(), 1);
    let name = tools[0].function.as_ref().unwrap().name.as_ref().unwrap();
    let arguments = serde_json::from_str::<serde_json::Value>(
        tools[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(name, "get_weather");
    assert_eq!(arguments["location"], "San Francisco");
    assert_eq!(arguments["unit"], "fahrenheit");
}

#[tokio::test]
async fn test_tool_calling_jail_internal_with_harmony_parser() {
    let mock_context = Arc::new(MockAsyncEngineContext::new(
        "test-request-id-harmony".to_string(),
    ));

    // Harmony Format:
    // <|channel|>analysis<|message|>Need to use function get_current_weather.<|end|>
    // <|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json
    // <|message|>{"location":"San Francisco"}<|call|>
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
        create_final_response_chunk(0),
    ];

    let input_stream = stream::iter(chunks);
    let response_stream = ResponseStream::new(Box::pin(input_stream), mock_context.clone());

    let jailed_stream =
        apply_tool_calling_jail_internal(response_stream, Some("harmony".to_string())).await;
    let results: Vec<_> = jailed_stream.collect().await;

    assert!(
        !results.is_empty(),
        "Should have results for harmony parser"
    );

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[1].data.as_ref().unwrap().choices[0].delta.content,
        Some("Need to use function get_current_weather.".to_string())
    );
    assert!(
        results[1].data.as_ref().unwrap().choices[0]
            .delta
            .tool_calls
            .is_some()
    );
    let tools = results[1].data.as_ref().unwrap().choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap();
    assert_eq!(tools.len(), 1);
    let name = tools[0].function.as_ref().unwrap().name.as_ref().unwrap();
    let arguments = serde_json::from_str::<serde_json::Value>(
        tools[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(name, "get_current_weather");
    assert_eq!(arguments["location"], "San Francisco");
}

#[test]
fn test_enable_tool_call() {
    let request = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            tool_choice: Some(ChatCompletionToolChoiceOption::Auto),
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };
    assert!(maybe_enable_tool_call(Some("nemotron_deci"), &request));

    let request = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            tool_choice: Some(ChatCompletionToolChoiceOption::None),
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };
    assert!(!maybe_enable_tool_call(Some("nemotron_deci"), &request));

    let request = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            tool_choice: Some(ChatCompletionToolChoiceOption::Required),
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };
    assert!(maybe_enable_tool_call(Some("nemotron_deci"), &request));

    let request = NvCreateChatCompletionRequest {
        inner: CreateChatCompletionRequest {
            tool_choice: Some(ChatCompletionToolChoiceOption::Auto),
            ..Default::default()
        },
        common: Default::default(),
        nvext: None,
        chat_template_args: None,
    };
    assert!(!maybe_enable_tool_call(None, &request));
}
