// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, Role,
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
        llm_metrics: None,
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

// ===================================
// Muse unified batch finalize (topology B)
// ===================================

/// Topology B: raw muse markup reaches the aggregator un-split (the frontend
/// holds the tool_call_parser, but the worker did not run the streaming hook, so
/// the deltas carry raw text). The finalize block must route muse through the
/// UNIFIED parser and populate reasoning_content + content + tool_calls on the
/// final message — the reasoning channel is the one thing the v2 tool-only
/// `parse_complete` cannot recover.
#[tokio::test]
async fn test_muse_unified_batch_finalize_splits_all_channels() {
    let deltas = vec![
        make_stream_delta(
            Some("<|start|>assistant to=self<|message|>Look it up.<|eom|>"),
            None,
        ),
        make_stream_delta(
            Some(
                "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"location\">Paris</atem:parameter></atem:invoke><|eom|>",
            ),
            None,
        ),
        make_stream_delta(
            Some("<|start|>assistant to=user<|message|>It's 18C.<|eot|>"),
            None,
        ),
    ];

    let stream = futures::stream::iter(deltas);
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        // The HTTP handlers derive this from the request's tool_choice via
        // `batch_tool_choice_eligible`, which admits unset/auto. Setting it here is
        // what an auto request really carries; leaving it default-false would be
        // asserting against a shape no served request has.
        ParsingOptions::new(Some("muse_glimmer".to_string()), None)
            .with_experimental_v2_batch_eligible(true),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up."),
        "reasoning_content must come from the unified batch finalize"
    );
    assert_eq!(
        get_text(choice.message.content.as_ref().expect("content")),
        "It's 18C."
    );
    let tool_calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(tool_calls.len(), 1, "expected one tool call");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"location": "Paris"}));
}

/// Reasoning-only card (topology B): `reasoning_parser=muse_glimmer`, NO
/// `tool_call_parser` — the card shape the streaming guard, `unified_family`, and
/// `parser_requires_special_tokens` all support on either name. The batch finalize
/// must key on the reasoning name too, so raw channel markup never survives into the
/// non-streaming response.
#[tokio::test]
async fn test_muse_unified_batch_finalize_routes_on_reasoning_name_only() {
    let deltas = vec![
        make_stream_delta(
            Some("<|start|>assistant to=self<|message|>Look it up.<|eom|>"),
            None,
        ),
        make_stream_delta(
            Some(
                "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"location\">Paris</atem:parameter></atem:invoke><|eom|>",
            ),
            None,
        ),
        make_stream_delta(
            Some("<|start|>assistant to=user<|message|>It's 18C.<|eot|>"),
            None,
        ),
    ];

    let stream = futures::stream::iter(deltas);
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        ParsingOptions::new(None, Some("muse_glimmer".to_string()))
            .with_experimental_v2_batch_eligible(true),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    let content = get_text(choice.message.content.as_ref().expect("content"));
    assert!(
        !content.contains("<|start|>") && !content.contains("<atem:invoke"),
        "raw markup leaked into content: {content:?}"
    );
    assert_eq!(content, "It's 18C.");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up."),
        "reasoning_content must be split even when only the reasoning name is set"
    );
    let tool_calls = choice.message.tool_calls.as_ref().expect("tool_calls");
    assert_eq!(tool_calls.len(), 1, "expected one tool call");
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

/// Batch counterpart of the streaming `none` pin. A caller that disabled tool calling
/// still needs the reasoning/content split and the marker stripping — muse has no v1
/// reasoning parser to fall back on — but must not receive `tool_calls`. The handlers
/// express that by deriving `experimental_v2_batch_eligible` from the request, which
/// `batch_tool_choice_eligible` sets false for an explicit `none`.
#[tokio::test]
async fn test_muse_unified_batch_finalize_suppresses_calls_when_not_eligible() {
    let deltas = vec![
        make_stream_delta(
            Some("<|start|>assistant to=self<|message|>Look it up.<|eom|>"),
            None,
        ),
        make_stream_delta(
            Some(
                "<|start|>assistant to=get_weather<|message|><atem:invoke name=\"get_weather\"><atem:parameter name=\"location\">Paris</atem:parameter></atem:invoke><|eom|>",
            ),
            None,
        ),
        make_stream_delta(
            Some("<|start|>assistant to=user<|message|>It's 18C.<|eot|>"),
            None,
        ),
    ];

    let stream = futures::stream::iter(deltas);
    let result = NvCreateChatCompletionResponse::from_annotated_stream(
        stream,
        ParsingOptions::new(Some("muse_glimmer".to_string()), None)
            .with_experimental_v2_batch_eligible(false),
    )
    .await
    .unwrap();

    let choice = result.inner.choices.first().expect("one choice");
    assert!(
        choice
            .message
            .tool_calls
            .as_ref()
            .is_none_or(|calls| calls.is_empty()),
        "an ineligible tool_choice must not return tool_calls: {:?}",
        choice.message.tool_calls
    );
    // The split and the stripping still have to happen.
    let content = get_text(choice.message.content.as_ref().expect("content"));
    assert_eq!(content, "It's 18C.");
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Look it up.")
    );
    for marker in ["<|start|>", "<|message|>", "<atem:invoke"] {
        assert!(
            !content.contains(marker),
            "marker {marker:?} leaked: {content:?}"
        );
    }
}
