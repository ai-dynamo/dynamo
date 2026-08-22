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

async fn post_json(svc: &HarnessService, path: &str, body: Value) -> reqwest::Response {
    svc.client
        .post(format!("{}{path}", svc.base_url))
        .json(&body)
        .send()
        .await
        .unwrap()
}

#[derive(Clone, Copy)]
enum ExpectedError {
    Validation,
    NotImplemented,
}

impl ExpectedError {
    fn status(self) -> reqwest::StatusCode {
        match self {
            Self::Validation => reqwest::StatusCode::BAD_REQUEST,
            Self::NotImplemented => reqwest::StatusCode::NOT_IMPLEMENTED,
        }
    }

    fn anthropic_type(self) -> &'static str {
        match self {
            Self::Validation => "invalid_request_error",
            Self::NotImplemented => "api_error",
        }
    }
}

async fn assert_openai_error(response: reqwest::Response, expected: ExpectedError, message: &str) {
    let status = expected.status();
    assert_eq!(response.status(), status);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["code"].as_u64(), Some(u64::from(status.as_u16())));
    assert!(
        body["message"].as_str().is_some_and(|actual| actual
            .to_ascii_lowercase()
            .contains(&message.to_ascii_lowercase())),
        "unexpected OpenAI error body: {body}"
    );
}

async fn assert_anthropic_error(
    response: reqwest::Response,
    expected: ExpectedError,
    message: &str,
) {
    assert_eq!(response.status(), expected.status());
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], expected.anthropic_type());
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|actual| actual.contains(message)),
        "unexpected Anthropic error body: {body}"
    );
}

fn assert_error_metrics(
    svc: &HarnessService,
    endpoint: &Endpoint,
    request_type: &RequestType,
    expected: &[(ErrorType, u64)],
) {
    for (error_type, expected) in expected {
        assert_eq!(
            svc.metrics.get_request_counter(
                MODEL,
                endpoint,
                request_type,
                &Status::Error,
                error_type,
            ),
            *expected,
            "unexpected {error_type:?} count for {endpoint}/{request_type}"
        );
    }
}

fn tool_name_requests(name: &str) -> [(&'static str, Value, bool); 3] {
    [
        (
            "/v1/messages",
            json!({
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "name": name,
                    "input_schema": {"type": "object", "properties": {}}
                }]
            }),
            true,
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL,
                "input": "ping",
                "tools": [{
                    "type": "function",
                    "name": name,
                    "parameters": {"type": "object", "properties": {}}
                }]
            }),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": name,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            }),
            false,
        ),
    ]
}

#[tokio::test]
#[serial]
async fn responses_conversion_distinguishes_invalid_from_unsupported() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (stream, content, expected, message) in [
            (
                true,
                json!({"type": "input_image", "file_id": "file_123"}),
                ExpectedError::NotImplemented,
                "image input by file_id",
            ),
            (
                false,
                json!({
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf"
                }),
                ExpectedError::NotImplemented,
                "file input content",
            ),
            (
                false,
                json!({"type": "input_image"}),
                ExpectedError::Validation,
                "requires file_id or image_url",
            ),
            (
                true,
                json!({"type": "input_file"}),
                ExpectedError::Validation,
                "requires exactly one of file_data, file_id, or file_url",
            ),
        ] {
            let response = post_json(
                &svc,
                "/v1/responses",
                json!({
                    "model": MODEL,
                    "stream": stream,
                    "input": [{"role": "user", "content": [content]}]
                }),
            )
            .await;
            assert_openai_error(response, expected, message).await;
        }

        for request_type in [RequestType::Unary, RequestType::Stream] {
            assert_error_metrics(
                &svc,
                &Endpoint::Responses,
                &request_type,
                &[
                    (ErrorType::NotImplemented, 1),
                    (ErrorType::Validation, 1),
                    (ErrorType::Internal, 0),
                ],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_tools_reject_unsupported_and_malformed_definitions() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (stream, tool_choice) in [
            (false, None),
            (true, Some(json!({"type": "tool", "name": "web_search"}))),
        ] {
            let mut body = json!({
                "model": MODEL,
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "type": "web_search_20260209",
                    "name": "web_search"
                }]
            });
            if let Some(tool_choice) = tool_choice {
                body["tool_choice"] = tool_choice;
            }
            let response = post_json(&svc, "/v1/messages", body).await;
            assert_anthropic_error(
                response,
                ExpectedError::NotImplemented,
                "server tool type \"web_search_20260209\" is not supported",
            )
            .await;
        }

        let response = post_json(
            &svc,
            "/v1/messages",
            json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "stream": false,
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [{"name": "get_weather"}]
            }),
        )
        .await;
        assert_anthropic_error(
            response,
            ExpectedError::Validation,
            "tools[0].input_schema: field required",
        )
        .await;

        for request_type in [RequestType::Unary, RequestType::Stream] {
            let validation = match &request_type {
                RequestType::Unary => 1,
                RequestType::Stream => 0,
            };
            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[
                    (ErrorType::NotImplemented, 1),
                    (ErrorType::Validation, validation),
                    (ErrorType::Internal, 0),
                ],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

// `reqwest::send` completes when response headers arrive. A 400 for `stream: true`
// proves converted-request validation ran before the HTTP 200 SSE response was committed.
#[tokio::test]
#[serial]
async fn converted_validation_errors_are_returned_before_streaming_headers() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (stream, field) in [
            (false, json!({"top_p": 2.0})),
            (true, json!({"temperature": 3.0})),
        ] {
            let mut body = json!({"model": MODEL, "input": "ping", "stream": stream});
            body.as_object_mut()
                .unwrap()
                .extend(field.as_object().unwrap().clone());
            let response = post_json(&svc, "/v1/responses", body).await;
            assert_openai_error(response, ExpectedError::Validation, "must be").await;
        }

        for (stream, field) in [
            (false, json!({"temperature": 3.0})),
            (true, json!({"top_p": 2.0})),
        ] {
            let mut body = json!({
                "model": MODEL,
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "ping"}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(field.as_object().unwrap().clone());
            let response = post_json(&svc, "/v1/messages", body).await;
            assert_anthropic_error(response, ExpectedError::Validation, "must be").await;
        }

        for endpoint in [Endpoint::Responses, Endpoint::AnthropicMessages] {
            for request_type in [RequestType::Unary, RequestType::Stream] {
                assert_error_metrics(
                    &svc,
                    &endpoint,
                    &request_type,
                    &[(ErrorType::Validation, 1), (ErrorType::Internal, 0)],
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
async fn chat_tool_argument_type_errors_are_rejected_before_backend_dispatch() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for stream in [false, true] {
            let response = post_json(
                &svc,
                "/v1/chat/completions",
                json!({
                    "model": MODEL,
                    "stream": stream,
                    "messages": [
                        {"role": "user", "content": "weather?"},
                        {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\":123}"
                                }
                            }]
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_weather",
                            "content": "sunny"
                        },
                        {"role": "user", "content": "thanks"}
                    ],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                },
                                "required": ["location"]
                            }
                        }
                    }]
                }),
            )
            .await;
            assert_openai_error(response, ExpectedError::Validation, "must be of type").await;
        }

        for request_type in [RequestType::Unary, RequestType::Stream] {
            assert_error_metrics(
                &svc,
                &Endpoint::ChatCompletions,
                &request_type,
                &[(ErrorType::Validation, 1), (ErrorType::Internal, 0)],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn responses_reject_empty_input_and_required_tool_choice_without_tools() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (body, message) in [
            (
                json!({"model": MODEL, "input": [], "max_tokens": 10}),
                "messages",
            ),
            (
                json!({
                    "model": MODEL,
                    "input": "ping",
                    "tools": [],
                    "tool_choice": "required"
                }),
                "tool_choice is \"required\"",
            ),
        ] {
            let response = post_json(&svc, "/v1/responses", body).await;
            assert_openai_error(response, ExpectedError::Validation, message).await;
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_content_validation_applies_to_messages_and_count_tokens() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (path, body, expected, message) in [
            (
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 10,
                    "stream": true,
                    "messages": [{"role": "user", "content": ["hello"]}]
                }),
                ExpectedError::Validation,
                "content blocks must be objects",
            ),
            (
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": []}]
                }),
                ExpectedError::Validation,
                "must contain at least one content block",
            ),
            (
                "/v1/messages/count_tokens",
                json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": ["hello"]}]
                }),
                ExpectedError::Validation,
                "content blocks must be objects",
            ),
            (
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "future_block_type", "value": 1},
                            {"type": "text", "text": "ping"}
                        ]
                    }]
                }),
                ExpectedError::NotImplemented,
                "content block type \"future_block_type\"",
            ),
            (
                "/v1/messages/count_tokens",
                json!({
                    "model": MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [{"type": "future_block_type", "value": 1}]
                    }]
                }),
                ExpectedError::NotImplemented,
                "content block type \"future_block_type\"",
            ),
        ] {
            let response = post_json(&svc, path, body).await;
            assert_anthropic_error(response, expected, message).await;
        }

        for request_type in [RequestType::Unary, RequestType::Stream] {
            let not_implemented = match &request_type {
                RequestType::Unary => 1,
                RequestType::Stream => 0,
            };
            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[
                    (ErrorType::Validation, 1),
                    (ErrorType::NotImplemented, not_implemented),
                    (ErrorType::Internal, 0),
                ],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn tool_name_limit_is_shared_across_protocols() {
    temp_env::async_with_vars(BASE_ENV, async {
        let valid_script = load_agent_fixture("text.sse").await.unwrap();
        let svc =
            HarnessService::start([valid_script.clone(), valid_script.clone(), valid_script]).await;

        let max_length_tool_name = "a".repeat(128);
        for (path, body, _) in tool_name_requests(&max_length_tool_name) {
            let response = post_json(&svc, path, body).await;
            assert_eq!(response.status(), reqwest::StatusCode::OK);
        }

        let too_long_tool_name = "a".repeat(129);
        for (path, body, anthropic) in tool_name_requests(&too_long_tool_name) {
            let response = post_json(&svc, path, body).await;
            if anthropic {
                assert_anthropic_error(response, ExpectedError::Validation, "128 character limit")
                    .await;
            } else {
                assert_openai_error(response, ExpectedError::Validation, "128 character limit")
                    .await;
            }
        }

        assert_eq!(svc.engine.take_requests().await.len(), 3);
        svc.shutdown().await;
    })
    .await;
}
