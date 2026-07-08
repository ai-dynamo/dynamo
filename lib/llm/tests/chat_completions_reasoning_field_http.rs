// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP coverage for configurable Chat Completions reasoning field names.

use dynamo_llm::frontend_config::ChatCompletionsReasoningField;
use serde_json::{Value, json};

#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_json_sse};

async fn post(svc: &HarnessService, path: &str, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}{path}", svc.base_url))
        .json(body)
        .send()
        .await
        .unwrap_or_else(|error| panic!("POST {path} failed: {error}"))
}

#[tokio::test]
async fn unary_override_is_scoped_to_chat_completions() {
    let fixture = load_agent_fixture("thinking-tool.sse").await.unwrap();
    let svc = HarnessService::start_with_reasoning_field(
        [fixture.clone(), fixture.clone(), fixture],
        ChatCompletionsReasoningField::Reasoning,
        false,
    )
    .await;

    let chat = post(
        &svc,
        "/v1/chat/completions",
        &json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "List /tmp"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(chat.status(), reqwest::StatusCode::OK);
    let chat: Value = chat.json().await.unwrap();
    assert_eq!(
        chat["choices"][0]["message"]["reasoning"],
        "I should inspect the directory."
    );
    assert!(
        chat["choices"][0]["message"]
            .get("reasoning_content")
            .is_none()
    );

    let responses = post(
        &svc,
        "/v1/responses",
        &json!({"model": MODEL, "input": "List /tmp", "stream": false}),
    )
    .await;
    assert_eq!(responses.status(), reqwest::StatusCode::OK);
    let responses: Value = responses.json().await.unwrap();
    let reasoning_item = responses["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("Responses API reasoning item");
    assert_eq!(
        reasoning_item["summary"][0]["text"],
        "I should inspect the directory."
    );

    let messages = post(
        &svc,
        "/v1/messages",
        &json!({
            "model": MODEL,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "List /tmp"}]
        }),
    )
    .await;
    assert_eq!(messages.status(), reqwest::StatusCode::OK);
    let messages: Value = messages.json().await.unwrap();
    let thinking_block = messages["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|block| block["type"] == "thinking")
        .expect("Anthropic thinking block");
    assert_eq!(
        thinking_block["thinking"],
        "I should inspect the directory."
    );

    assert_eq!(svc.engine.remaining_scripts().await, 0);
    svc.shutdown().await;
}

#[tokio::test]
async fn streaming_override_applies_to_deltas_and_reasoning_dispatch() {
    let svc = HarnessService::start_with_reasoning_field(
        [load_agent_fixture("thinking-tool.sse").await.unwrap()],
        ChatCompletionsReasoningField::Reasoning,
        true,
    )
    .await;

    let response = post(
        &svc,
        "/v1/chat/completions",
        &json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "List /tmp"}],
            "stream": true
        }),
    )
    .await;
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let raw = response.text().await.unwrap();
    let events = parse_json_sse(&raw).await.unwrap();

    let reasoning_delta = events
        .iter()
        .find(|event| event.data["choices"][0]["delta"]["reasoning"].is_string())
        .expect("reasoning delta");
    assert_eq!(
        reasoning_delta.data["choices"][0]["delta"]["reasoning"],
        "I should inspect the directory."
    );
    assert!(
        reasoning_delta.data["choices"][0]["delta"]
            .get("reasoning_content")
            .is_none()
    );

    let dispatch = events
        .iter()
        .find(|event| event.event == "reasoning_dispatch")
        .expect("reasoning_dispatch event");
    assert_eq!(dispatch.data["index"], 0);
    assert_eq!(
        dispatch.data["reasoning"],
        "I should inspect the directory."
    );
    assert!(dispatch.data.get("reasoning_content").is_none());

    assert_eq!(svc.engine.remaining_scripts().await, 0);
    svc.shutdown().await;
}
