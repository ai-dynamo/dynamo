// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP integration coverage for the Anthropic Messages compatibility surface.

use std::collections::BTreeMap;
use std::time::Duration;

use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessageContent,
};
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
};
use futures::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;

#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{
    HarnessService, IncrementalSseParser, MODEL, canonicalize, load_agent_fixture, parse_json_sse,
};
use scripted_chat_engine::Script;

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];

fn user_text(
    request: &dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest,
) -> &str {
    match &request.inner.messages[..] {
        [ChatCompletionRequestMessage::User(user)] => match &user.content {
            ChatCompletionRequestUserMessageContent::Text(text) => text,
            other => panic!("expected text user content, got {other:?}"),
        },
        other => panic!("expected one translated user message, got {other:#?}"),
    }
}

async fn post_messages(svc: &HarnessService, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/messages", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/messages failed")
}

fn tool(name: &str) -> Value {
    json!({
        "name": name,
        "description": "test tool",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    })
}

#[tokio::test]
#[serial]
async fn unary_text_baseline() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 64,
                "stream": false,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        insta::assert_json_snapshot!("anthropic_unary_text", canonicalize(body));

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(user_text(&requests[0]), "ping");
        assert_eq!(requests[0].inner.max_completion_tokens, Some(64));
        assert_eq!(requests[0].inner.stream, Some(true));
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn streaming_text_baseline() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 64,
                "stream": true,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let raw = response.text().await.unwrap();
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        let events = parse_json_sse(&raw).await.unwrap();
        insta::assert_json_snapshot!(
            "anthropic_streaming_text",
            canonicalize(serde_json::to_value(events).unwrap())
        );

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn fragmented_tool_arguments_close_after_all_deltas() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start([load_agent_fixture("fragmented-tool.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();

        let start = events
            .iter()
            .find(|event| event.event == "content_block_start")
            .expect("missing tool block start");
        assert_eq!(start.data["content_block"]["id"], "call_list_directory");
        assert_eq!(start.data["content_block"]["name"], "list_directory");

        let deltas: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event.event == "content_block_delta"
                    && event.data["delta"]["type"] == "input_json_delta"
            })
            .map(|(index, event)| (index, event.data["delta"]["partial_json"].as_str().unwrap()))
            .collect();
        assert_eq!(
            deltas.iter().map(|(_, part)| *part).collect::<String>(),
            r#"{"path":"/tmp"}"#
        );
        let stop_positions: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.event == "content_block_stop")
            .map(|(index, _)| index)
            .collect();
        assert_eq!(stop_positions.len(), 1);
        assert!(stop_positions[0] > deltas.last().unwrap().0);
        assert_eq!(
            events
                .iter()
                .find(|event| event.event == "message_delta")
                .unwrap()
                .data["delta"]["stop_reason"],
            "tool_use"
        );

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn finish_signal_publishes_tool_block_before_usage_tail() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        let split_at = script
            .iter()
            .position(|chunk| chunk.inner.usage.is_some())
            .expect("fragmented-tool fixture has no usage chunk");
        let (svc, gate) = HarnessService::start_with_gated_tail(script, split_at).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let mut body = response.bytes_stream();
        let mut parser = IncrementalSseParser::default();
        let mut saw_tool_stop = false;
        let mut saw_message_delta = false;

        tokio::time::timeout(Duration::from_secs(2), async {
            while !saw_tool_stop {
                let bytes = body
                    .next()
                    .await
                    .expect("response ended before tool block completion")
                    .expect("failed to read response SSE bytes");
                for event in parser.push(&bytes).expect("failed to parse response SSE") {
                    saw_tool_stop |= event == "content_block_stop";
                    saw_message_delta |= event == "message_delta";
                }
            }
        })
        .await
        .expect("tool block completion did not arrive before the gated usage tail");

        assert!(!saw_message_delta);
        gate.release();
        while let Some(bytes) = body.next().await {
            let bytes = bytes.expect("failed to drain response SSE bytes");
            parser.push(&bytes).expect("failed to parse response SSE");
        }

        let raw = parser.into_body().expect("response SSE was not UTF-8");
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        let events = parse_json_sse(&raw).await.unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "content_block_stop")
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "message_delta")
                .count(),
            1
        );

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn parallel_tools_preserve_identity_and_arguments() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start([load_agent_fixture("parallel-tools.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("read_file")],
                "messages": [{"role": "user", "content": "Read /a and /b"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();

        let starts: Vec<_> = events
            .iter()
            .filter(|event| event.event == "content_block_start")
            .map(|event| {
                (
                    event.data["index"].as_u64().unwrap(),
                    event.data["content_block"]["id"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                    event.data["content_block"]["name"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                )
            })
            .collect();
        assert_eq!(
            starts,
            vec![
                (0, "call_read_a".into(), "read_file".into()),
                (1, "call_read_b".into(), "read_file".into())
            ]
        );

        let mut arguments = BTreeMap::<u64, String>::new();
        for event in &events {
            if event.event == "content_block_delta"
                && event.data["delta"]["type"] == "input_json_delta"
            {
                arguments
                    .entry(event.data["index"].as_u64().unwrap())
                    .or_default()
                    .push_str(event.data["delta"]["partial_json"].as_str().unwrap());
            }
        }
        assert_eq!(arguments.get(&0).unwrap(), r#"{"path":"/a"}"#);
        assert_eq!(arguments.get(&1).unwrap(), r#"{"path":"/b"}"#);

        let mut open_block = None;
        for event in &events {
            match event.event.as_str() {
                "content_block_start" if event.data["content_block"]["type"] == "tool_use" => {
                    assert!(open_block.is_none(), "tool blocks must not overlap");
                    open_block = event.data["index"].as_u64();
                }
                "content_block_delta" if event.data["delta"]["type"] == "input_json_delta" => {
                    assert_eq!(event.data["index"].as_u64(), open_block);
                }
                "content_block_stop" => {
                    assert_eq!(event.data["index"].as_u64(), open_block);
                    open_block = None;
                }
                _ => {}
            }
        }
        assert!(open_block.is_none());

        let stops: Vec<_> = events
            .iter()
            .filter(|event| event.event == "content_block_stop")
            .map(|event| event.data["index"].as_u64().unwrap())
            .collect();
        assert_eq!(stops, vec![0, 1]);

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn tool_result_round_trip_reaches_the_chat_engine() {
    temp_env::async_with_vars(ENV, async {
        let first_script = load_agent_fixture("thinking-tool.sse").await.unwrap();
        let second_script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([first_script, second_script]).await;

        let first_response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": false,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(first_response.status(), reqwest::StatusCode::OK);
        let first_body: Value = first_response.json().await.unwrap();
        let prior_content = first_body["content"].clone();
        assert!(prior_content.as_array().is_some_and(|blocks| {
            blocks.iter().any(|block| block["type"] == "thinking")
                && blocks.iter().any(|block| block["type"] == "tool_use")
        }));

        let second_response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 64,
                "stream": false,
                "tools": [tool("list_directory")],
                "messages": [
                    {"role": "user", "content": "List /tmp"},
                    {"role": "assistant", "content": prior_content},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_list_directory",
                        "content": "a.txt"
                    }]}
                ]
            }),
        )
        .await;
        assert_eq!(second_response.status(), reqwest::StatusCode::OK);
        let second_body: Value = second_response.json().await.unwrap();
        assert_eq!(second_body["content"][0]["text"], "Pong.");

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 2);
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        match &requests[1].inner.messages[..] {
            [
                ChatCompletionRequestMessage::User(user),
                ChatCompletionRequestMessage::Assistant(assistant),
                ChatCompletionRequestMessage::Tool(tool_result),
            ] => {
                assert!(matches!(
                    &user.content,
                    ChatCompletionRequestUserMessageContent::Text(text) if text == "List /tmp"
                ));
                assert!(matches!(
                    assistant.content.as_ref(),
                    Some(ChatCompletionRequestAssistantMessageContent::Text(text))
                        if text == "I will list it."
                ));
                assert_eq!(
                    assistant
                        .reasoning_content
                        .as_ref()
                        .expect("thinking must reach the chat request")
                        .to_flat_string(),
                    "I should inspect the directory."
                );
                let calls = assistant.tool_calls.as_deref().expect("tool calls missing");
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_list_directory");
                assert_eq!(calls[0].function.name, "list_directory");
                assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp"}"#);
                assert_eq!(tool_result.tool_call_id, "call_list_directory");
                assert!(matches!(
                    &tool_result.content,
                    ChatCompletionRequestToolMessageContent::Text(text) if text == "a.txt"
                ));
            }
            other => panic!("unexpected translated round-trip messages: {other:#?}"),
        }

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn count_tokens_returns_exact_estimate_without_calling_engine() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start(Vec::<Script>::new()).await;
        let response = svc
            .client
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "system": "You are helpful.",
                "messages": [{
                    "role": "user",
                    "content": "Hello, world! This is a test message."
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            json!({"input_tokens": 19})
        );
        assert!(svc.engine.take_requests().await.is_empty());

        svc.shutdown().await;
    })
    .await;
}
