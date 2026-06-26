// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP integration coverage for the OpenAI Responses compatibility surface.

use std::collections::BTreeMap;
use std::time::Duration;

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestUserMessageContent,
};
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
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

const ENV: [(&str, Option<&str>); 1] = [(DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0"))];

async fn post_responses(svc: &HarnessService, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/responses failed")
}

fn tool(name: &str) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": "test tool",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    })
}

fn event_position(events: &[http_harness::JsonSseEvent], event_type: &str) -> usize {
    events
        .iter()
        .position(|event| event.event == event_type)
        .unwrap_or_else(|| panic!("missing {event_type} event"))
}

#[tokio::test]
#[serial]
async fn unary_text_baseline() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        let response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "ping",
                "stream": false,
                "max_output_tokens": 64
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        insta::assert_json_snapshot!("responses_unary_text", canonicalize(body));

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].inner.max_completion_tokens, Some(64));
        assert_eq!(requests[0].inner.stream, Some(true));
        match &requests[0].inner.messages[..] {
            [ChatCompletionRequestMessage::User(user)] => assert!(matches!(
                &user.content,
                ChatCompletionRequestUserMessageContent::Text(text) if text == "ping"
            )),
            other => panic!("unexpected translated request: {other:#?}"),
        }
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
        let response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "ping",
                "stream": true,
                "max_output_tokens": 64
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let raw = response.text().await.unwrap();
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        let events = parse_json_sse(&raw).await.unwrap();
        insta::assert_json_snapshot!(
            "responses_streaming_text",
            canonicalize(serde_json::to_value(events).unwrap())
        );

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn empty_first_arguments_do_not_finish_function_call_early() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start([load_agent_fixture("fragmented-tool.sse").await.unwrap()]).await;
        let response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "List /tmp",
                "stream": true,
                "max_output_tokens": 128,
                "tools": [tool("list_directory")]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();

        let added = events
            .iter()
            .find(|event| event.event == "response.output_item.added")
            .expect("missing function-call item");
        assert_eq!(added.data["item"]["call_id"], "call_list_directory");
        assert_eq!(added.data["item"]["name"], "list_directory");

        let deltas: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.event == "response.function_call_arguments.delta")
            .map(|(index, event)| (index, event.data["delta"].as_str().unwrap()))
            .collect();
        assert_eq!(
            deltas.iter().map(|(_, part)| *part).collect::<String>(),
            r#"{"path":"/tmp"}"#
        );
        let args_done_position = event_position(&events, "response.function_call_arguments.done");
        let item_done_position = event_position(&events, "response.output_item.done");
        assert!(args_done_position > deltas.last().unwrap().0);
        assert!(item_done_position > args_done_position);

        let args_done = &events[args_done_position].data;
        assert_eq!(args_done["arguments"], r#"{"path":"/tmp"}"#);
        assert_eq!(args_done["name"], "list_directory");
        let item_done = &events[item_done_position].data["item"];
        assert_eq!(item_done["call_id"], "call_list_directory");
        assert_eq!(item_done["arguments"], r#"{"path":"/tmp"}"#);
        assert_eq!(item_done["id"], added.data["item"]["id"]);

        let completed = events
            .iter()
            .find(|event| event.event == "response.completed")
            .unwrap();
        let output = completed.data["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["id"], added.data["item"]["id"]);
        assert_eq!(output[0]["arguments"], r#"{"path":"/tmp"}"#);

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn finish_signal_publishes_function_call_before_usage_tail() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        let split_at = script
            .iter()
            .position(|chunk| chunk.inner.usage.is_some())
            .expect("fragmented-tool fixture has no usage chunk");
        let (svc, gate) = HarnessService::start_with_gated_tail(script, split_at).await;
        let response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "List /tmp",
                "stream": true,
                "max_output_tokens": 128,
                "tools": [tool("list_directory")]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let mut body = response.bytes_stream();
        let mut parser = IncrementalSseParser::default();
        let mut saw_arguments_done = false;
        let mut saw_item_done = false;
        let mut saw_response_completed = false;

        tokio::time::timeout(Duration::from_secs(2), async {
            while !(saw_arguments_done && saw_item_done) {
                let bytes = body
                    .next()
                    .await
                    .expect("response ended before function-call completion")
                    .expect("failed to read response SSE bytes");
                for event in parser.push(&bytes).expect("failed to parse response SSE") {
                    saw_arguments_done |= event == "response.function_call_arguments.done";
                    saw_item_done |= event == "response.output_item.done";
                    saw_response_completed |= event == "response.completed";
                }
            }
        })
        .await
        .expect("function-call completion did not arrive before the gated usage tail");

        assert!(!saw_response_completed);
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
                .filter(|event| event.event == "response.function_call_arguments.done")
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "response.output_item.done")
                .count(),
            1
        );
        assert!(
            events
                .iter()
                .any(|event| event.event == "response.completed")
        );

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn parallel_function_calls_preserve_identity_and_arguments() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start([load_agent_fixture("parallel-tools.sse").await.unwrap()]).await;
        let response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "Read /a and /b",
                "stream": true,
                "max_output_tokens": 128,
                "parallel_tool_calls": true,
                "tools": [tool("read_file")]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();

        let mut calls = BTreeMap::<u64, (String, String, String, String)>::new();
        let mut last_delta_positions = BTreeMap::<u64, usize>::new();
        let mut completion_counts = BTreeMap::<u64, (usize, usize)>::new();
        for (position, event) in events.iter().enumerate() {
            match event.event.as_str() {
                "response.output_item.added" => {
                    let index = event.data["output_index"].as_u64().unwrap();
                    let item = &event.data["item"];
                    calls.insert(
                        index,
                        (
                            item["id"].as_str().unwrap().to_string(),
                            item["call_id"].as_str().unwrap().to_string(),
                            item["name"].as_str().unwrap().to_string(),
                            String::new(),
                        ),
                    );
                    completion_counts.insert(index, (0, 0));
                }
                "response.function_call_arguments.delta" => {
                    let index = event.data["output_index"].as_u64().unwrap();
                    calls
                        .get_mut(&index)
                        .unwrap()
                        .3
                        .push_str(event.data["delta"].as_str().expect("arguments delta"));
                    assert_eq!(event.data["item_id"], calls.get(&index).unwrap().0.as_str());
                    last_delta_positions.insert(index, position);
                }
                "response.function_call_arguments.done" => {
                    let index = event.data["output_index"].as_u64().unwrap();
                    assert_eq!(event.data["item_id"], calls.get(&index).unwrap().0.as_str());
                    assert!(position > last_delta_positions[&index]);
                    completion_counts.get_mut(&index).unwrap().0 += 1;
                }
                "response.output_item.done" => {
                    let index = event.data["output_index"].as_u64().unwrap();
                    assert_eq!(
                        event.data["item"]["id"],
                        calls.get(&index).unwrap().0.as_str()
                    );
                    assert!(position > last_delta_positions[&index]);
                    completion_counts.get_mut(&index).unwrap().1 += 1;
                }
                _ => {}
            }
        }
        assert_eq!(
            calls,
            BTreeMap::from([
                (
                    0,
                    (
                        calls[&0].0.clone(),
                        "call_read_a".into(),
                        "read_file".into(),
                        r#"{"path":"/a"}"#.into()
                    )
                ),
                (
                    1,
                    (
                        calls[&1].0.clone(),
                        "call_read_b".into(),
                        "read_file".into(),
                        r#"{"path":"/b"}"#.into()
                    )
                )
            ])
        );
        assert_eq!(
            completion_counts,
            BTreeMap::from([(0, (1, 1)), (1, (1, 1))])
        );

        let completed = events
            .iter()
            .find(|event| event.event == "response.completed")
            .expect("missing response.completed");
        let output = completed.data["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        for (index, item) in output.iter().enumerate() {
            let call = &calls[&(index as u64)];
            assert_eq!(item["id"], call.0);
            assert_eq!(item["call_id"], call.1);
            assert_eq!(item["name"], call.2);
            assert_eq!(item["arguments"], call.3);
        }

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn function_call_output_round_trip_reaches_the_chat_engine() {
    temp_env::async_with_vars(ENV, async {
        let first_script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        let second_script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([first_script, second_script]).await;

        let first_response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": "List /tmp",
                "stream": false,
                "max_output_tokens": 128,
                "tools": [tool("list_directory")]
            }),
        )
        .await;
        assert_eq!(first_response.status(), reqwest::StatusCode::OK);
        let first_body: Value = first_response.json().await.unwrap();
        let function_call = first_body["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("first response did not contain a function call")
            .clone();

        let second_response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "stream": false,
                "max_output_tokens": 64,
                "tools": [tool("list_directory")],
                "input": [
                    {"role": "user", "content": "List /tmp"},
                    function_call,
                    {
                        "type": "function_call_output",
                        "call_id": "call_list_directory",
                        "output": "[\"a.txt\"]"
                    }
                ]
            }),
        )
        .await;
        assert_eq!(second_response.status(), reqwest::StatusCode::OK);
        let second_body: Value = second_response.json().await.unwrap();
        assert_eq!(second_body["output"][0]["content"][0]["text"], "Pong.");

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
                assert!(assistant.content.is_none());
                let calls = assistant.tool_calls.as_deref().expect("tool calls missing");
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_list_directory");
                assert_eq!(calls[0].function.name, "list_directory");
                assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp"}"#);
                assert_eq!(tool_result.tool_call_id, "call_list_directory");
                assert!(matches!(
                    &tool_result.content,
                    ChatCompletionRequestToolMessageContent::Text(text)
                        if text == r#"["a.txt"]"#
                ));
            }
            other => panic!("unexpected translated round-trip messages: {other:#?}"),
        }

        svc.shutdown().await;
    })
    .await;
}
