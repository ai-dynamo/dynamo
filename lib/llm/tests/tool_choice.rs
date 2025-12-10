// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::{
    ChatCompletionNamedToolChoice, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionToolChoiceOption,
    ChatCompletionToolType, CreateChatCompletionRequest, FunctionName,
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

async fn apply_jail_transformation(
    raw_response: dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
) -> dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse {
    use dynamo_llm::protocols::openai::chat_completions::jail::JailedStream;
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    use futures::stream;

    let input_stream = stream::iter(vec![Annotated {
        data: Some(raw_response),
        id: None,
        event: None,
        comment: None,
    }]);

    let mut builder = JailedStream::builder();

    match tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(ref named)) => {
            builder = builder.tool_choice_named(named.function.name.clone());
        }
        Some(ChatCompletionToolChoiceOption::Required) => {
            builder = builder.tool_choice_required();
        }
        _ => {}
    }

    let jail = builder.build();
    let output_stream = jail.apply_with_finish_reason(input_stream);

    tokio::pin!(output_stream);
    output_stream.next().await.unwrap().data.unwrap()
}

async fn apply_jail_transformation_streaming(
    raw_responses: Vec<
        dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
    >,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
) -> Vec<dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse> {
    use dynamo_llm::protocols::openai::chat_completions::jail::JailedStream;
    use dynamo_runtime::protocols::annotated::Annotated;
    use futures::StreamExt;
    use futures::stream;

    let input_stream = stream::iter(raw_responses.into_iter().map(|r| Annotated {
        data: Some(r),
        id: None,
        event: None,
        comment: None,
    }));

    let mut builder = JailedStream::builder();

    match tool_choice {
        Some(ChatCompletionToolChoiceOption::Named(ref named)) => {
            builder = builder.tool_choice_named(named.function.name.clone());
        }
        Some(ChatCompletionToolChoiceOption::Required) => {
            builder = builder.tool_choice_required();
        }
        _ => {}
    }

    let jail = builder.build();
    let output_stream = jail.apply_with_finish_reason(input_stream);

    tokio::pin!(output_stream);
    output_stream
        .filter_map(|ann| async move { ann.data })
        .collect()
        .await
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

#[tokio::test]
async fn test_named_tool_choice_parses_json() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-1".to_string());
    let backend_output = build_backend_output(r#"{"location":"Paris"}"#);
    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;
    let choice = &response.choices[0];

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );
    let delta = &choice.delta;
    assert!(delta.content.is_none() || delta.content.as_deref() == Some(""));
    let tool_calls = delta.tool_calls.as_ref().unwrap();

    assert_eq!(tool_calls.len(), 1);

    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.index, 0);
    assert!(tool_call.id.as_ref().unwrap().starts_with("call-"));
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

#[tokio::test]
async fn test_required_tool_choice_parses_json_array() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-2".to_string());
    let backend_output = build_backend_output(
        r#"[{"name":"search","parameters":{"query":"rust"}},
            {"name":"summarize","parameters":{"topic":"memory"}}]"#,
    );
    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;
    let choice = &response.choices[0];

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );
    let delta = &choice.delta;
    assert!(delta.content.is_none() || delta.content.as_deref() == Some(""));
    let tool_calls = delta.tool_calls.as_ref().unwrap();

    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].index, 0);
    assert!(tool_calls[0].id.as_ref().unwrap().starts_with("call-"));
    assert_eq!(tool_calls[0].r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"query":"rust"}"#)
    );

    assert_eq!(tool_calls[1].index, 1);
    assert!(tool_calls[1].id.as_ref().unwrap().starts_with("call-"));
    assert_eq!(tool_calls[1].r#type, Some(ChatCompletionToolType::Function));
    assert_eq!(
        tool_calls[1].function.as_ref().unwrap().name.as_deref(),
        Some("summarize")
    );
    assert_eq!(
        tool_calls[1]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"topic":"memory"}"#)
    );
}

#[tokio::test]
async fn test_tool_choice_parse_failure_returns_as_content() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-3".to_string());
    let backend_output = build_backend_output("not-json");
    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;
    let delta = &response.choices[0].delta;

    // Jail stream behavior: if parsing fails, return accumulated content as-is
    // This matches marker-based FC behavior
    assert_eq!(delta.content.as_deref(), Some("not-json"));
    assert!(delta.tool_calls.is_none());
}

#[tokio::test]
async fn test_streaming_named_tool_buffers_until_finish() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-stream-1".to_string());

    let chunks = [r#"{"location":""#, r#"Paris","unit":""#, r#"celsius"}"#];

    let mut raw_responses = Vec::new();
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
        raw_responses.push(response);
    }

    let all_responses = apply_jail_transformation_streaming(raw_responses, tool_choice).await;

    // Jail stream buffers content until valid JSON, then emits once
    assert_eq!(all_responses.len(), 1);

    let response = &all_responses[0];
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop)
    );

    let tool_calls = response.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"location":"Paris","unit":"celsius"}"#)
    );
}

#[tokio::test]
async fn test_streaming_required_tool_parallel() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-stream-2".to_string());

    let chunks = [
        r#"[{"name":"search","parameters":{"query":"rust"}},"#,
        r#"{"name":"summarize","parameters":{"topic":"memory"}}]"#,
    ];

    let mut raw_responses = Vec::new();
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
        raw_responses.push(response);
    }

    let all_responses = apply_jail_transformation_streaming(raw_responses, tool_choice).await;

    // Jail stream buffers until complete JSON array
    assert_eq!(all_responses.len(), 1);

    let response = &all_responses[0];
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls)
    );

    let tool_calls = response.choices[0].delta.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 2);

    assert_eq!(
        tool_calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("search")
    );
    assert_eq!(
        tool_calls[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"query":"rust"}"#)
    );

    assert_eq!(
        tool_calls[1].function.as_ref().unwrap().name.as_deref(),
        Some("summarize")
    );
    assert_eq!(
        tool_calls[1]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some(r#"{"topic":"memory"}"#)
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
