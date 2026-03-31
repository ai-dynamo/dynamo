// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_async_openai::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
};
use dynamo_llm::protocols::{
    Annotated, ContentProvider, DataStream,
    codec::{Message, SseCodecError, create_message_stream},
    openai::{
        ParsingOptions,
        chat_completions::{
            NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
            aggregator::ChatCompletionAggregator,
        },
        completions::NvCreateCompletionResponse,
    },
};
use futures::StreamExt;

fn get_text(content: &ChatCompletionMessageContent) -> &str {
    match content {
        ChatCompletionMessageContent::Text(text) => text.as_str(),
        ChatCompletionMessageContent::Parts(_) => "",
    }
}

const CMPL_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/completions";
const CHAT_ROOT_PATH: &str = "tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions";

fn create_stream(root_path: &str, file_name: &str) -> DataStream<Result<Message, SseCodecError>> {
    let data = std::fs::read_to_string(format!("{}/{}", root_path, file_name)).unwrap();
    create_message_stream(&data)
}

#[tokio::test]
async fn test_openai_chat_stream() {
    let data = std::fs::read_to_string("tests/data/replays/meta/llama-3.1-8b-instruct/chat_completions/chat-completion.streaming.1").unwrap();

    // note: we are only taking the first 16 messages to keep the size of the response small
    let stream = create_message_stream(&data).take(16);
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    // todo: provide a cleaner way to extract the content from choices
    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .expect("there to be content")
        ),
        "Deep learning is a subfield of machine learning that involves the use of artificial"
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_multi_line_data() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-multi-line-data");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .expect("there to be content")
        ),
        "Deep learning"
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_comments_per_response() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/valid-comments_per_response");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .expect("there to be content")
        ),
        "Deep learning"
    );
}

#[tokio::test]
async fn test_openai_chat_edge_case_invalid_deserialize_error() {
    let stream = create_stream(CHAT_ROOT_PATH, "edge_cases/invalid-deserialize_error");
    let result = NvCreateChatCompletionResponse::from_sse_stream(
        Box::pin(stream),
        ParsingOptions::default(),
    )
    .await;

    assert!(result.is_err());
    // insta::assert_debug_snapshot!(result);
}

// =============================
// Completions (/v1/completions)
// =============================

#[tokio::test]
async fn test_openai_cmpl_stream() {
    let stream = create_stream(CMPL_ROOT_PATH, "completion.streaming.1").take(16);
    let result =
        NvCreateCompletionResponse::from_sse_stream(Box::pin(stream), ParsingOptions::default())
            .await
            .unwrap();

    // todo: provide a cleaner way to extract the content from choices
    assert_eq!(
        result.inner.choices.first().unwrap().content(),
        " This is a question that is often asked by those outside of AI research and development"
    );
}

// ===================================
// nvext aggregation regression tests
// ===================================

#[allow(deprecated)]
fn make_stream_delta(
    content: Option<&str>,
    nvext: Option<serde_json::Value>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    Annotated::from_data(NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: if let Some(text) = content {
                vec![ChatChoiceStream {
                    index: 0,
                    delta: ChatCompletionStreamResponseDelta {
                        content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                        function_call: None,
                        tool_calls: None,
                        role: Some(Role::Assistant),
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                    stop_reason: None,
                    logprobs: None,
                }]
            } else {
                vec![]
            },
            created: 1234567890,
            model: "test-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        },
        nvext,
    })
}

/// Verify that nvext set on a stream delta survives aggregation into the final response.
#[tokio::test]
async fn test_nvext_passthrough_aggregation() {
    let nvext_value = serde_json::json!({"custom_field": "test_value"});

    let deltas = vec![
        make_stream_delta(Some("Hello"), None),
        make_stream_delta(Some(" world"), Some(nvext_value.clone())),
        make_stream_delta(Some("!"), None),
    ];

    let stream = futures::stream::iter(deltas);
    let result =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, ParsingOptions::default())
            .await
            .unwrap();

    assert_eq!(result.nvext, Some(nvext_value));
    assert_eq!(
        get_text(
            result
                .inner
                .choices
                .first()
                .unwrap()
                .message
                .content
                .as_ref()
                .unwrap()
        ),
        "Hello world!"
    );
}

/// Verify that the last non-None nvext wins when multiple deltas carry nvext.
#[tokio::test]
async fn test_nvext_last_value_wins() {
    let first_nvext = serde_json::json!({"version": 1});
    let last_nvext = serde_json::json!({"version": 2});

    let deltas = vec![
        make_stream_delta(Some("a"), Some(first_nvext)),
        make_stream_delta(Some("b"), None),
        make_stream_delta(Some("c"), Some(last_nvext.clone())),
    ];

    let stream = futures::stream::iter(deltas);
    let result =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, ParsingOptions::default())
            .await
            .unwrap();

    assert_eq!(result.nvext, Some(last_nvext));
}

/// Verify that nvext remains None when no delta carries it.
#[tokio::test]
async fn test_nvext_none_when_absent() {
    let deltas = vec![make_stream_delta(Some("hello"), None)];

    let stream = futures::stream::iter(deltas);
    let result =
        NvCreateChatCompletionResponse::from_annotated_stream(stream, ParsingOptions::default())
            .await
            .unwrap();

    assert_eq!(result.nvext, None);
}
