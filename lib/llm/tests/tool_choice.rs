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

    assert_eq!(tool_calls.len(), 1);

    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.index, 0);
    assert_eq!(tool_call.id.as_deref(), Some("call-1"));
    assert_eq!(tool_call.r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_call.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        tool_call.function.as_ref().unwrap().arguments.as_deref(),
        Some(r#"{"location":"Paris"}"#)
    );
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

    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].index, 0);
    assert_eq!(tool_calls[0].id.as_deref(), Some("call-1"));
    assert_eq!(tool_calls[0].r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().arguments.as_deref(),
        Some(r#"{"query":"rust"}"#)
    );

    assert_eq!(tool_calls[1].index, 1);
    assert_eq!(tool_calls[1].id.as_deref(), Some("call-2"));
    assert_eq!(tool_calls[1].r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_calls[1].function.as_ref().unwrap().name.as_deref(),
        Some("summarize")
    );
    assert_eq!(
        tool_calls[1].function.as_ref().unwrap().arguments.as_deref(),
        Some(r#"{"topic":"memory"}"#)
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
    assert!(delta.content.is_none());
    assert!(delta.tool_calls.is_none());
}

#[test]
fn test_streaming_named_tool_buffers_until_finish() {
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

    let chunks = [
        r#"{"location":""#,
        r#"Paris","unit":""#,
        r#"celsius"}"#,
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

    for i in 0..all_responses.len() - 1 {
        assert!(all_responses[i].choices[0].delta.tool_calls.is_none());
    }

    let last_response = all_responses.last().unwrap();
    assert_eq!(
        last_response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );

    let tool_calls = last_response.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.as_ref().unwrap().name.as_deref(), Some("get_weather"));
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().arguments.as_deref(),
        Some(r#"{"location":"Paris","unit":"celsius"}"#)
    );
}

#[test]
fn test_streaming_required_tool_parallel() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-stream-2".to_string());

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

    for i in 0..all_responses.len() - 1 {
        assert!(all_responses[i].choices[0].delta.tool_calls.is_none());
    }

    let last_response = all_responses.last().unwrap();
    assert_eq!(
        last_response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );

    let tool_calls = last_response.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].function.as_ref().unwrap().name.as_deref(), Some("search"));
    assert_eq!(tool_calls[0].function.as_ref().unwrap().arguments.as_deref(), Some(r#"{"query":"rust"}"#));

    assert_eq!(tool_calls[1].function.as_ref().unwrap().name.as_deref(), Some("summarize"));
    assert_eq!(tool_calls[1].function.as_ref().unwrap().arguments.as_deref(), Some(r#"{"topic":"memory"}"#));
}

#[test]
fn test_streaming_buffers_until_finish() {
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

    let full_json = r#"{"query":"rust programming"}"#;
    let mut responses = Vec::new();

    for (i, ch) in full_json.chars().enumerate() {
        let is_last = i == full_json.len() - 1;
        let backend_output = BackendOutput {
            token_ids: vec![],
            tokens: vec![],
            text: Some(ch.to_string()),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: if is_last {
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
            .expect("char chunk");
        responses.push(response);
    }

    for resp in &responses {
        assert!(resp.choices[0].delta.content.is_none());
    }

    for i in 0..responses.len() - 1 {
        assert!(responses[i].choices[0].delta.tool_calls.is_none());
    }

    let last = responses.last().unwrap();
    assert!(last.choices[0].delta.tool_calls.is_some());
    assert_eq!(
        last.choices[0].delta.tool_calls.as_ref().unwrap()[0]
            .function.as_ref().unwrap().arguments.as_deref(),
        Some(r#"{"query":"rust programming"}"#)
    );
}

#[test]
fn test_no_tool_choice_outputs_normal_text() {
    let request = create_test_request();

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

    assert_eq!(
        response.choices[0].delta.content.as_deref(),
        Some("Hello world")
    );
    assert!(response.choices[0].delta.tool_calls.is_none());
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
async fn test_aggregator_named_tool() {
    use dynamo_llm::protocols::Annotated;
    use dynamo_llm::protocols::openai::ParsingOptions;
    use dynamo_llm::protocols::openai::chat_completions::aggregator::DeltaAggregator;
    use futures::stream;

    let chunks = vec![
        create_chunk(
            0,
            Some(dynamo_async_openai::types::Role::Assistant),
            None,
            None,
        ),
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call-1".to_string()),
                r#type: Some(ChatCompletionToolType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".to_string()),
                    arguments: Some(r#"{"location":"Paris","unit":"celsius"}"#.to_string()),
                }),
            }),
            None,
        ),
        create_chunk(
            0,
            None,
            None,
            Some(dynamo_async_openai::types::FinishReason::Stop),
        ),
    ];

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
    let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();

    assert_eq!(response.choices.len(), 1);
    let choice = &response.choices[0];

    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);

    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.id, "call-1");
    assert_eq!(tool_call.function.name, "get_weather");
    assert_eq!(
        tool_call.function.arguments, r#"{"location":"Paris","unit":"celsius"}"#
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

    let chunks = vec![
        create_chunk(
            0,
            Some(dynamo_async_openai::types::Role::Assistant),
            None,
            None,
        ),
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call-1".to_string()),
                r#type: Some(ChatCompletionToolType::Function),
                function: Some(FunctionCallStream {
                    name: Some("search".to_string()),
                    arguments: Some(r#"{"query":"rust"}"#.to_string()),
                }),
            }),
            None,
        ),
        create_chunk(
            0,
            None,
            Some(ChatCompletionMessageToolCallChunk {
                index: 1,
                id: Some("call-2".to_string()),
                r#type: Some(ChatCompletionToolType::Function),
                function: Some(FunctionCallStream {
                    name: Some("summarize".to_string()),
                    arguments: Some(r#"{"text":"long article"}"#.to_string()),
                }),
            }),
            None,
        ),
        create_chunk(
            0,
            None,
            None,
            Some(dynamo_async_openai::types::FinishReason::ToolCalls),
        ),
    ];

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
    let result = DeltaAggregator::apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();

    assert_eq!(response.choices.len(), 1);
    let choice = &response.choices[0];

    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].id, "call-1");
    assert_eq!(tool_calls[0].function.name, "search");
    assert_eq!(tool_calls[0].function.arguments, r#"{"query":"rust"}"#);

    assert_eq!(tool_calls[1].id, "call-2");
    assert_eq!(tool_calls[1].function.name, "summarize");
    assert_eq!(tool_calls[1].function.arguments, r#"{"text":"long article"}"#);

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
}
