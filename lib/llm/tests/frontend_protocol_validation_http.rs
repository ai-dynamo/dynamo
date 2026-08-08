// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP regressions for validation performed by protocol adapters.

use dynamo_llm::http::service::metrics::{Endpoint, ErrorType, RequestType, Status};
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
    DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS,
};
use serde_json::{Value, json};
use serial_test::serial;

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture};

const BASE_ENV: [(&str, Option<&str>); 3] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS, None),
];

async fn assert_openai_400(response: reqwest::Response, message: &str) {
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["code"], 400);
    assert!(
        body["message"].as_str().is_some_and(|actual| actual
            .to_ascii_lowercase()
            .contains(&message.to_ascii_lowercase())),
        "unexpected OpenAI error body: {body}"
    );
}

async fn assert_openai_501(response: reqwest::Response, message: &str) {
    assert_eq!(response.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["code"], 501);
    assert!(
        body["message"].as_str().is_some_and(|actual| actual
            .to_ascii_lowercase()
            .contains(&message.to_ascii_lowercase())),
        "unexpected OpenAI error body: {body}"
    );
}

async fn assert_anthropic_400(response: reqwest::Response, message: &str) {
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|actual| actual.contains(message)),
        "unexpected Anthropic error body: {body}"
    );
}

async fn assert_anthropic_501(response: reqwest::Response, message: &str) {
    assert_eq!(response.status(), reqwest::StatusCode::NOT_IMPLEMENTED);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "api_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|actual| actual.contains(message)),
        "unexpected Anthropic error body: {body}"
    );
}

#[tokio::test]
#[serial]
async fn responses_conversion_distinguishes_invalid_from_unsupported() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for stream in [false, true] {
            for (content, message) in [
                (
                    json!({"type": "input_image", "file_id": "file_123"}),
                    "image input by file_id",
                ),
                (
                    json!({
                        "type": "input_file",
                        "file_url": "https://example.com/report.pdf"
                    }),
                    "file input content",
                ),
            ] {
                let response = svc
                    .client
                    .post(format!("{}/v1/responses", svc.base_url))
                    .json(&json!({
                        "model": MODEL,
                        "stream": stream,
                        "input": [{"role": "user", "content": [content]}]
                    }))
                    .send()
                    .await
                    .unwrap();
                assert_openai_501(response, message).await;
            }

            for (content, message) in [
                (
                    json!({"type": "input_image"}),
                    "requires exactly one of file_id or image_url",
                ),
                (
                    json!({"type": "input_file"}),
                    "requires exactly one of file_data, file_id, or file_url",
                ),
            ] {
                let response = svc
                    .client
                    .post(format!("{}/v1/responses", svc.base_url))
                    .json(&json!({
                        "model": MODEL,
                        "stream": stream,
                        "input": [{"role": "user", "content": [content]}]
                    }))
                    .send()
                    .await
                    .unwrap();
                assert_openai_400(response, message).await;
            }
        }

        for request_type in [RequestType::Unary, RequestType::Stream] {
            for (error_type, expected) in [
                (ErrorType::NotImplemented, 2),
                (ErrorType::Validation, 2),
                (ErrorType::Internal, 0),
            ] {
                assert_eq!(
                    svc.metrics.get_request_counter(
                        MODEL,
                        &Endpoint::Responses,
                        &request_type,
                        &Status::Error,
                        &error_type,
                    ),
                    expected,
                    "unexpected {error_type:?} count for {request_type}"
                );
            }
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
// `reqwest::send` completes when response headers arrive. Asserting 4xx/5xx
// status for `stream: true` proves the adapter rejected the request before
// committing the HTTP 200 SSE response.
async fn protocol_adapter_errors_are_returned_before_streaming_headers() {
    temp_env::async_with_vars(BASE_ENV, async {
        let valid_script = load_agent_fixture("text.sse").await.unwrap();
        let svc =
            HarnessService::start([valid_script.clone(), valid_script.clone(), valid_script]).await;

        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/responses", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "input": [],
                    "max_tokens": 10,
                    "stream": stream
                }))
                .send()
                .await
                .unwrap();
            assert_openai_400(response, "messages").await;
        }
        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/responses", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "input": "ping",
                    "stream": stream,
                    "tools": [],
                    "tool_choice": "required"
                }))
                .send()
                .await
                .unwrap();
            assert_openai_400(response, "tool_choice is \"required\"").await;
        }
        for field in [json!({"temperature": 3.0}), json!({"top_p": 2.0})] {
            for stream in [false, true] {
                let mut body = json!({"model": MODEL, "input": "ping", "stream": stream});
                body.as_object_mut()
                    .unwrap()
                    .extend(field.as_object().unwrap().clone());
                let response = svc
                    .client
                    .post(format!("{}/v1/responses", svc.base_url))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_openai_400(response, "must be").await;
            }
        }

        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .header("x-api-key", "dummy")
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": MODEL,
                    "max_tokens": 10,
                    "stream": stream,
                    "messages": [{"role": "user", "content": ["hello"]}]
                }))
                .send()
                .await
                .unwrap();
            assert_anthropic_400(response, "content blocks must be objects").await;
        }
        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .header("x-api-key", "dummy")
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": MODEL,
                    "max_tokens": 10,
                    "stream": stream,
                    "messages": [{"role": "user", "content": []}]
                }))
                .send()
                .await
                .unwrap();
            assert_anthropic_400(response, "must contain at least one content block").await;
        }
        for field in [json!({"temperature": 3.0}), json!({"top_p": 2.0})] {
            for stream in [false, true] {
                let mut body = json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "ping"}]
                });
                body.as_object_mut()
                    .unwrap()
                    .extend(field.as_object().unwrap().clone());
                let response = svc
                    .client
                    .post(format!("{}/v1/messages", svc.base_url))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_anthropic_400(response, "must be").await;
            }
        }

        let response = svc
            .client
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": ["hello"]}]
            }))
            .send()
            .await
            .unwrap();
        assert_anthropic_400(response, "content blocks must be objects").await;

        for endpoint in [Endpoint::Responses, Endpoint::AnthropicMessages] {
            for request_type in [RequestType::Unary, RequestType::Stream] {
                assert_eq!(
                    svc.metrics.get_request_counter(
                        MODEL,
                        &endpoint,
                        &request_type,
                        &Status::Error,
                        &ErrorType::Validation,
                    ),
                    4,
                    "validation errors were not metered for {endpoint}/{request_type}"
                );
                assert_eq!(
                    svc.metrics.get_request_counter(
                        MODEL,
                        &endpoint,
                        &request_type,
                        &Status::Error,
                        &ErrorType::Internal,
                    ),
                    0,
                    "validation errors were misclassified for {endpoint}/{request_type}"
                );
            }
        }

        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "stream": stream,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "future_block_type", "value": 1},
                            {"type": "text", "text": "ping"}
                        ]
                    }]
                }))
                .send()
                .await
                .unwrap();
            assert_anthropic_501(response, "content block type \"future_block_type\"").await;
        }

        let response = svc
            .client
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [{"type": "future_block_type", "value": 1}]
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_anthropic_501(response, "content block type \"future_block_type\"").await;

        for request_type in [RequestType::Unary, RequestType::Stream] {
            assert_eq!(
                svc.metrics.get_request_counter(
                    MODEL,
                    &Endpoint::AnthropicMessages,
                    &request_type,
                    &Status::Error,
                    &ErrorType::NotImplemented,
                ),
                1,
                "unsupported content blocks were not metered for {request_type}"
            );
        }

        let max_length_tool_name = "a".repeat(128);
        let response = svc
            .client
            .post(format!("{}/v1/messages", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "name": max_length_tool_name.clone(),
                    "input_schema": {"type": "object", "properties": {}}
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let response = svc
            .client
            .post(format!("{}/v1/responses", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "input": "ping",
                "tools": [{
                    "type": "function",
                    "name": max_length_tool_name.clone(),
                    "parameters": {"type": "object", "properties": {}}
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let response = svc
            .client
            .post(format!("{}/v1/chat/completions", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": max_length_tool_name,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let too_long_tool_name = "a".repeat(129);
        let response = svc
            .client
            .post(format!("{}/v1/messages", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "name": too_long_tool_name.clone(),
                    "input_schema": {"type": "object", "properties": {}}
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_anthropic_400(response, "128 character limit").await;

        let response = svc
            .client
            .post(format!("{}/v1/responses", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "input": "ping",
                "tools": [{
                    "type": "function",
                    "name": too_long_tool_name.clone(),
                    "parameters": {"type": "object", "properties": {}}
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_openai_400(response, "128 character limit").await;

        let response = svc
            .client
            .post(format!("{}/v1/chat/completions", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": too_long_tool_name,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_openai_400(response, "128 character limit").await;

        assert_eq!(svc.engine.take_requests().await.len(), 3);
        svc.shutdown().await;
    })
    .await;
}
