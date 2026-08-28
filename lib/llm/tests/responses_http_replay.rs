// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP integration coverage for the OpenAI Responses compatibility surface.

use std::collections::BTreeMap;
use std::time::Duration;

use dynamo_llm::http::service::metrics::{Endpoint, ErrorType, RequestType, Status};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestUserMessageContent,
};
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
use dynamo_runtime::error::{BackendError, DynamoError, ErrorType as DynamoErrorType};
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
async fn streaming_backend_error_closes_partial_output_and_counts_failure() {
    temp_env::async_with_vars(ENV, async {
        const ERROR_MESSAGE: &str =
            "ValueError: Received multimodal data but multimodal processing is not enabled. Use --enable-multimodal flag to enable multimodal processing.";
        let mut script = load_agent_fixture("text.sse").await.unwrap();
        let finish_position = script
            .iter()
            .position(|chunk| {
                chunk
                    .inner
                    .choices
                    .iter()
                    .any(|choice| choice.finish_reason.is_some())
            })
            .expect("text fixture has no finish-reason chunk");
        script.truncate(finish_position);
        let typed_error = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::InvalidArgument))
            .message(ERROR_MESSAGE)
            .build();
        // The tool-enabled Python adapter path serializes the typed error into
        // a generic error message before it reaches the HTTP frontend.
        let error = DynamoError::msg(typed_error.to_string());
        let svc = HarnessService::start_with_backend_error(script, error).await;

        let response = post_responses(
            &svc,
            &json!({
                "model": MODEL,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is this?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                    ]
                }],
                "tools": [{
                    "type": "function",
                    "name": "noop",
                    "description": "No-op tool",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }],
                "parallel_tool_calls": false,
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();
        let text_done_position = event_position(&events, "response.output_text.done");
        let part_done_position = event_position(&events, "response.content_part.done");
        let item_done_position = event_position(&events, "response.output_item.done");
        let failed_position = event_position(&events, "response.failed");
        assert!(text_done_position < part_done_position);
        assert!(part_done_position < item_done_position);
        assert!(item_done_position < failed_position);

        let completed_item = &events[item_done_position].data["item"];
        let failed = &events[failed_position];
        assert_eq!(completed_item["status"], "incomplete");
        assert_eq!(failed.data["response"]["status"], "failed");
        assert_eq!(failed.data["response"]["output"], json!([completed_item]));
        assert_eq!(
            failed.data["response"]["error"]["code"],
            "invalid_prompt"
        );
        assert_eq!(
            failed.data["response"]["error"]["message"],
            ERROR_MESSAGE
        );
        assert!(
            events
                .iter()
                .all(|event| event.event != "response.completed")
        );
        assert_eq!(
            svc.metrics.get_request_counter(
                MODEL,
                &Endpoint::Responses,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Internal,
            ),
            1
        );
        assert_eq!(
            svc.metrics.get_request_counter(
                MODEL,
                &Endpoint::Responses,
                &RequestType::Stream,
                &Status::Success,
                &ErrorType::None,
            ),
            0
        );

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
        let call_id = function_call["call_id"]
            .as_str()
            .expect("function call missing call_id")
            .to_string();

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
                        "call_id": call_id,
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

// ---------------------------------------------------------------------------
// POST /v1/responses/input_tokens
// ---------------------------------------------------------------------------

async fn post_input_tokens(svc: &HarnessService, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/responses/input_tokens", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/responses/input_tokens failed")
}

#[tokio::test]
#[serial]
async fn input_tokens_counts_input() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let response =
            post_input_tokens(&svc, &json!({"model": MODEL, "input": "Hello, world!"})).await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let body: Value = response.json().await.unwrap();
        assert_eq!(body["object"], "response.input_tokens");
        assert!(
            body["input_tokens"].as_u64().unwrap() > 0,
            "expected a non-zero count, got {body}"
        );

        // Counting is pre-flight only; it must never reach a backend.
        assert_eq!(svc.engine.take_requests().await.len(), 0);
        svc.shutdown().await;
    })
    .await;
}

/// The endpoint must answer for models this frontend does not serve.
///
#[tokio::test]
#[serial]
async fn input_tokens_does_not_gate_on_model() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let response = post_input_tokens(
            &svc,
            &json!({
                "model": "dynamo/deepseek-ai/deepseek-v4-pro-sglang",
                "input": "Hello, world!"
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            response.json::<Value>().await.unwrap()["object"],
            "response.input_tokens"
        );

        // `model` is optional entirely.
        let response = post_input_tokens(&svc, &json!({"input": "Hello, world!"})).await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn input_tokens_counts_instructions_and_tools() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let input = json!({"model": MODEL, "input": "List /tmp"});
        let baseline = post_input_tokens(&svc, &input).await;
        assert_eq!(baseline.status(), reqwest::StatusCode::OK);
        let baseline = baseline.json::<Value>().await.unwrap()["input_tokens"]
            .as_u64()
            .unwrap();

        let augmented = post_input_tokens(
            &svc,
            &json!({
                "model": MODEL,
                "input": "List /tmp",
                "instructions": "You are a careful filesystem assistant.",
                "tools": [tool("list_directory")]
            }),
        )
        .await;
        assert_eq!(augmented.status(), reqwest::StatusCode::OK);
        let augmented = augmented.json::<Value>().await.unwrap()["input_tokens"]
            .as_u64()
            .unwrap();

        assert!(
            augmented > baseline,
            "instructions and tools should add tokens: {augmented} vs {baseline}"
        );
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn input_tokens_rejects_malformed_json() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let response = svc
            .client
            .post(format!("{}/v1/responses/input_tokens", svc.base_url))
            .header("content-type", "application/json")
            .body("{\"input\": ")
            .send()
            .await
            .expect("POST /v1/responses/input_tokens failed");
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);

        // Same error envelope `/v1/responses` produces for a malformed body.
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["code"], 400);
        assert_eq!(body["type"], "Bad Request");
        assert!(
            body["message"]
                .as_str()
                .unwrap()
                .contains("Failed to deserialize the JSON body"),
            "unexpected error message: {body}"
        );
        svc.shutdown().await;
    })
    .await;
}

/// A tool-using conversation, in the exact item shapes a chat-to-Responses
/// converter emits: `function_call` / `function_call_output` items carrying
/// neither `id` nor `status`. Agent clients send these on every turn after the
/// first, so they need to survive the full HTTP path, not just the estimator's
/// own unit tests.
#[tokio::test]
#[serial]
async fn input_tokens_accepts_tool_call_conversation_items() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let response = post_input_tokens(
            &svc,
            &json!({
                "model": "dynamo/deepseek-ai/deepseek-v4-pro-sglang",
                "instructions": "You are a careful filesystem assistant.",
                "input": [
                    {"role": "user", "content": "List /tmp"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "list_directory",
                        "arguments": r#"{"path":"/tmp"}"#
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": r#"["a.txt"]"#
                    }
                ],
                "tools": [tool("list_directory")]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let body: Value = response.json().await.unwrap();
        assert_eq!(body["object"], "response.input_tokens");
        assert!(
            body["input_tokens"].as_u64().unwrap() > 0,
            "tool-call items should contribute to the count, got {body}"
        );
        svc.shutdown().await;
    })
    .await;
}

/// Pins the estimator's coalescing model against the converter that defines it.
///
/// `CountInputTokensRequest::estimate_tokens` charges one assistant role marker
/// per *flushed* pending assistant message, not one per assistant-side item,
/// because `convert_input_items_to_messages` accumulates them. That equivalence
/// is asserted here against the real converter: the same items go to both
/// endpoints, and the chat messages the engine actually receives are the ground
/// truth for how many role markers the count should have paid.
#[tokio::test]
#[serial]
async fn parallel_tool_calls_are_one_assistant_message_for_both_endpoints() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        let input = json!([
            {"role": "user", "content": "List /tmp and /var"},
            {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{\"p\":\"/tmp\"}"},
            {"type": "function_call", "call_id": "c2", "name": "ls", "arguments": "{\"p\":\"/var\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "a"},
            {"type": "function_call_output", "call_id": "c2", "output": "b"}
        ]);

        let counted = post_input_tokens(&svc, &json!({"model": MODEL, "input": input})).await;
        assert_eq!(counted.status(), reqwest::StatusCode::OK);
        assert!(counted.json::<Value>().await.unwrap()["input_tokens"].as_u64().unwrap() > 0);

        let generated = post_responses(
            &svc,
            &json!({"model": MODEL, "input": input, "stream": false}),
        )
        .await;
        assert_eq!(generated.status(), reqwest::StatusCode::OK);

        // Two parallel calls collapse into ONE assistant message; each output
        // is its own tool message. Four items, three messages — which is why
        // the estimate pays one assistant marker here, not two.
        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        match &requests[0].inner.messages[..] {
            [
                ChatCompletionRequestMessage::User(_),
                ChatCompletionRequestMessage::Assistant(assistant),
                ChatCompletionRequestMessage::Tool(_),
                ChatCompletionRequestMessage::Tool(_),
            ] => {
                assert_eq!(
                    assistant.tool_calls.as_deref().map(<[_]>::len),
                    Some(2),
                    "both calls should ride on the single assistant message"
                );
            }
            other => panic!("unexpected coalescing: {other:#?}"),
        }

        svc.shutdown().await;
    })
    .await;
}

/// A tool shape the pinned `async-openai` cannot model must not fail the count.
///
/// Callers forward tool definitions verbatim, including Chat-Completions-style
/// `custom` tools and types newer than the pin. Tools contribute almost nothing
/// to the estimate, so rejecting the body over one would hand back the very
/// error this endpoint exists to stop — and silently, since callers treat a
/// failed count as "fall back to a local tokenizer".
#[tokio::test]
#[serial]
async fn input_tokens_tolerates_tool_shapes_it_cannot_model() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let count = async |body: Value| {
            let response = post_input_tokens(&svc, &body).await;
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.json::<Value>().await.unwrap()["input_tokens"]
                .as_u64()
                .unwrap()
        };

        let baseline = count(json!({"model": "m", "input": "Hello, world!"})).await;

        // Unmodelled tools are dropped, so the count matches the same body with
        // no `tools` key at all — they were worth nothing either way.
        for tools in [
            json!([{"type": "custom", "custom": {"name": "x"}}]),
            json!([{"type": "totally_new_tool", "whatever": {"a": 1}}]),
            json!([42, "nonsense", null]),
        ] {
            assert_eq!(
                count(json!({"model": "m", "input": "Hello, world!", "tools": tools})).await,
                baseline
            );
        }

        // Dropping is per-entry: a usable function tool alongside an unusable
        // one still counts.
        assert!(
            count(json!({
                "model": "m",
                "input": "Hello, world!",
                "tools": [{"type": "custom", "custom": {"name": "x"}}, tool("list_directory")]
            }))
            .await
                > baseline
        );

        svc.shutdown().await;
    })
    .await;
}

/// A trailing slash on `DYN_HTTP_SVC_RESPONSES_PATH` must not leak into the
/// derived subroute.
///
/// `/custom/` is a working parent configuration — axum matches `POST /custom/`
/// — but appending naively would register `/custom//input_tokens`, which axum
/// does not treat as equivalent to the `/custom/input_tokens` a client calls.
#[tokio::test]
#[serial]
async fn input_tokens_path_normalizes_a_trailing_slash_parent() {
    temp_env::async_with_vars(
        [
            (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
            ("DYN_HTTP_SVC_RESPONSES_PATH", Some("/custom/")),
        ],
        async {
            let svc = HarnessService::start([]).await;

            let response = svc
                .client
                .post(format!("{}/custom/input_tokens", svc.base_url))
                .json(&json!({"model": "m", "input": "Hello, world!"}))
                .send()
                .await
                .expect("POST /custom/input_tokens failed");
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            assert_eq!(
                response.json::<Value>().await.unwrap()["object"],
                "response.input_tokens"
            );

            // The doubled-slash form is what the bug produced; it must not be
            // what got registered instead.
            let doubled = svc
                .client
                .post(format!("{}/custom//input_tokens", svc.base_url))
                .json(&json!({"model": "m", "input": "Hello, world!"}))
                .send()
                .await
                .expect("POST /custom//input_tokens failed");
            assert_eq!(doubled.status(), reqwest::StatusCode::NOT_FOUND);

            svc.shutdown().await;
        },
    )
    .await;
}
