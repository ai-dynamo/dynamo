// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream-level coverage for `event: tool_call_dispatch`.
//!
//! These tests drive the real chat-completions streaming handler over HTTP and
//! read the SSE frames a client actually receives, so they observe where the
//! handler places the dispatch event relative to the ordinary data frames
//! produced by `EventConverter`, and that no ordinary frame is lost.

use std::sync::Arc;
use std::time::Duration;

use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionRequestMessage,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionStreamResponseDelta, CreateChatCompletionRequestArgs,
    CreateChatCompletionStreamResponse, FinishReason, FunctionCallStream, FunctionType,
};
use dynamo_runtime::CancellationToken;

// The shared modules are included by several test binaries; this one does not
// use every helper they offer.
#[allow(dead_code)]
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use ports::bind_random_port;
use scripted_chat_engine::{Script, ScriptedChatEngine};

const MODEL: &str = "tool-dispatch-model";

/// A delta carrying one tool-call fragment for choice 0, inner index 0.
fn fragment(
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> NvCreateChatCompletionStreamResponse {
    chunk(choice(
        Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: id.map(str::to_string),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
        }]),
        None,
    ))
}

/// A delta whose only content is the terminal `finish_reason`.
fn finish(reason: FinishReason) -> NvCreateChatCompletionStreamResponse {
    chunk(choice(None, Some(reason)))
}

#[allow(deprecated)]
fn choice(
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    finish_reason: Option<FinishReason>,
) -> ChatChoiceStream {
    ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            content: None,
            function_call: None,
            tool_calls,
            role: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason,
        logprobs: None,
    }
}

fn chunk(choice: ChatChoiceStream) -> NvCreateChatCompletionStreamResponse {
    NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "tool-dispatch-stream".to_string(),
            choices: vec![choice],
            created: 1234567890,
            model: MODEL.to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    }
}

/// Run `script` through a real streaming request and label every SSE frame the
/// client receives, in wire order.
///
/// Ordinary frames are labelled by what the client can see in them, so a lost or
/// reordered frame changes the label sequence rather than only its length.
async fn wire_labels(script: Script) -> Vec<String> {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder()
        .port(port)
        .host("127.0.0.1")
        .enable_chat_endpoints(true)
        .enable_streaming_tool_dispatch(true)
        .build()
        .expect("failed to build HTTP service");

    let card = ModelDeploymentCard::with_name_only(MODEL);
    service
        .model_manager()
        .add_chat_completions_model(
            MODEL,
            card.mdcsum(),
            Arc::new(ScriptedChatEngine::new([script])),
        )
        .expect("failed to register scripted model");

    let cancel = CancellationToken::new();
    let task = service.spawn_with_listener(cancel.clone(), listener).await;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("failed to build HTTP client");
    let base_url = format!("http://127.0.0.1:{port}");
    tokio::time::timeout(Duration::from_secs(5), async {
        while client
            .get(format!("{base_url}/health"))
            .send()
            .await
            .is_err()
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("HTTP service did not become ready");

    let mut request = CreateChatCompletionRequestArgs::default()
        .model(MODEL)
        .messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text("hi".to_string()),
                name: None,
            },
        )])
        .build()
        .expect("failed to build request");
    request.stream = Some(true);

    let response = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&request)
        .send()
        .await
        .expect("streaming request failed");
    assert!(response.status().is_success(), "{response:?}");
    let body = tokio::time::timeout(Duration::from_secs(10), response.text())
        .await
        .expect("stream did not finish")
        .expect("failed to read SSE body");

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(2), task).await;

    let labels = label_frames(&body);
    assert!(
        !labels.is_empty(),
        "no SSE frames were received. Body:\n{body}"
    );
    labels
}

fn label_frames(body: &str) -> Vec<String> {
    let mut labels = Vec::new();
    for raw in body.split("\n\n") {
        let mut event = None;
        let mut data: Option<String> = None;
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data.get_or_insert_with(String::new).push_str(rest.trim());
            }
        }
        // Keep-alive frames are bare `:` comments and carry no data.
        let Some(data) = data else { continue };
        labels.push(match event.as_deref() {
            Some(name) => format!("{name}:{}", dispatched_arguments(&data)),
            None if data == "[DONE]" => "done".to_string(),
            None => format!("data:{}", delta_summary(&data)),
        });
    }
    labels
}

/// The assembled argument string a dispatch event carries.
fn dispatched_arguments(data: &str) -> String {
    let json: serde_json::Value =
        serde_json::from_str(data).unwrap_or_else(|e| panic!("dispatch frame is not JSON: {e}"));
    json["tool_call"]["function"]["arguments"]
        .as_str()
        .unwrap_or("<missing>")
        .to_string()
}

/// What an ordinary chat-completion chunk shows the client.
fn delta_summary(data: &str) -> String {
    let json: serde_json::Value =
        serde_json::from_str(data).unwrap_or_else(|e| panic!("data frame is not JSON: {e}"));
    let choice = &json["choices"][0];
    if let Some(reason) = choice["finish_reason"].as_str() {
        return format!("finish={reason}");
    }
    let call = &choice["delta"]["tool_calls"][0];
    match call["function"]["arguments"].as_str() {
        Some(arguments) => format!("args={arguments}"),
        None => format!("open={}", call["function"]["name"].as_str().unwrap_or("?")),
    }
}

/// Argument fragments that only become valid JSON on the last one. The dispatch
/// event must reach the client before the ordinary frame carrying that last
/// fragment, and every ordinary frame must still be delivered, in order.
#[tokio::test]
async fn dispatch_event_precedes_the_completing_frame_on_the_wire() {
    let labels = wire_labels(vec![
        fragment(Some("call_1"), Some("create_file"), None),
        fragment(None, None, Some(r#"{"path""#)),
        fragment(None, None, Some(r#":"/a/very"#)),
        fragment(None, None, Some(r#"/long/file"}"#)),
        finish(FinishReason::ToolCalls),
    ])
    .await;

    assert_eq!(
        labels,
        vec![
            "data:open=create_file",
            r#"data:args={"path""#,
            r#"data:args=:"/a/very"#,
            r#"tool_call_dispatch:{"path":"/a/very/long/file"}"#,
            r#"data:args=/long/file"}"#,
            "data:finish=tool_calls",
            "done",
        ],
    );
}

/// Truncated arguments never become valid JSON, so no dispatch event may appear,
/// while the ordinary frames carrying the fragments are still all delivered.
#[tokio::test]
async fn truncated_arguments_never_dispatch_on_the_wire() {
    let labels = wire_labels(vec![
        fragment(Some("call_1"), Some("create_file"), None),
        fragment(None, None, Some(r#"{"path""#)),
        finish(FinishReason::Length),
    ])
    .await;

    assert_eq!(
        labels,
        vec![
            "data:open=create_file",
            r#"data:args={"path""#,
            "data:finish=length",
            "done",
        ],
    );
}
