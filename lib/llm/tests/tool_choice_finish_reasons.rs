// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for tool_choice finish_reason handling.

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

fn build_backend_output_with_finish(text: &str, finish: common::FinishReason) -> BackendOutput {
    BackendOutput {
        token_ids: vec![],
        tokens: vec![],
        text: Some(text.to_string()),
        cum_log_probs: None,
        log_probs: None,
        top_logprobs: None,
        finish_reason: Some(finish),
        index: Some(0),
        completion_usage: None,
        disaggregated_params: None,
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

#[tokio::test]
async fn test_named_tool_choice_preserves_length_finish_reason() {
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

    let mut generator = request.response_generator("req-length-1".to_string());
    let backend_output = build_backend_output_with_finish(
        r#"{"location":"Par"#, // Incomplete due to length limit
        common::FinishReason::Length,
    );

    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;

    // Critical: Length finish reason should be preserved, NOT replaced with Stop
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Length),
        "Length finish reason must be preserved for tool_choice=named"
    );
}

#[test]
fn test_required_tool_choice_preserves_length_finish_reason() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-length-2".to_string());
    let backend_output = build_backend_output_with_finish(
        r#"[{"name":"search","parameters":{"query":"incomplete"#, // Incomplete due to length
        common::FinishReason::Length,
    );

    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    // Critical: Length finish reason should be preserved, NOT replaced with ToolCalls
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Length),
        "Length finish reason must be preserved for tool_choice=required"
    );
}

#[test]
fn test_named_tool_choice_preserves_content_filter() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "search".to_string(),
            },
        },
    ));

    let mut generator = request.response_generator("req-filter-1".to_string());
    let backend_output = build_backend_output_with_finish(
        r#"{"query":"filtered content"#,
        common::FinishReason::ContentFilter,
    );

    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    // Critical: ContentFilter finish reason should be preserved
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ContentFilter),
        "ContentFilter finish reason must be preserved for tool_choice=named"
    );
}

#[test]
fn test_required_tool_choice_preserves_content_filter() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Required);

    let mut generator = request.response_generator("req-filter-2".to_string());
    let backend_output = build_backend_output_with_finish(
        r#"[{"name":"harmful_action"#,
        common::FinishReason::ContentFilter,
    );

    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    // Critical: ContentFilter finish reason should be preserved
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ContentFilter),
        "ContentFilter finish reason must be preserved for tool_choice=required"
    );
}

#[test]
fn test_named_tool_choice_normal_stop_becomes_stop() {
    let mut request = create_test_request();
    request.inner.tool_choice = Some(ChatCompletionToolChoiceOption::Named(
        ChatCompletionNamedToolChoice {
            r#type: ChatCompletionToolType::Function,
            function: FunctionName {
                name: "get_weather".to_string(),
            },
        },
    ));

    let mut generator = request.response_generator("req-stop-1".to_string());
    let backend_output = build_backend_output_with_finish(
        r#"{"location":"Paris","unit":"celsius"}"#,
        common::FinishReason::Stop,
    );

    let response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    // Normal completion: Stop should remain Stop for named tool choice
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::Stop),
    );
}

#[tokio::test]
async fn test_required_tool_choice_normal_stop_becomes_tool_calls() {
    let mut request = create_test_request();
    let tool_choice = Some(ChatCompletionToolChoiceOption::Required);
    request.inner.tool_choice = tool_choice.clone();

    let mut generator = request.response_generator("req-stop-2".to_string());
    let backend_output = build_backend_output_with_finish(
        r#"[{"name":"search","parameters":{"query":"rust"}}]"#,
        common::FinishReason::Stop,
    );

    let raw_response = generator
        .choice_from_postprocessor(backend_output)
        .expect("choice generation");

    let response = apply_jail_transformation(raw_response, tool_choice).await;

    // Normal completion: Stop should become ToolCalls for required tool choice
    assert_eq!(
        response.choices[0].finish_reason,
        Some(dynamo_async_openai::types::FinishReason::ToolCalls),
    );
}
