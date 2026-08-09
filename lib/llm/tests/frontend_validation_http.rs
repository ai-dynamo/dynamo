// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP regressions for validation failures shared by the frontend protocols.

use std::time::Duration;

use dynamo_llm::protocols::{
    Annotated,
    openai::{
        chat_completions::NvCreateChatCompletionStreamResponse,
        completions::NvCreateCompletionResponse,
    },
};
use dynamo_runtime::{
    config::environment_names::llm::{
        DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
        DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS,
    },
    error::{BackendError, DynamoError, ErrorType},
};
use serde_json::{Value, json};
use serial_test::serial;

#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_json_sse};
use scripted_chat_engine::{AnnotatedScript, CompletionScript};

const BASE_ENV: [(&str, Option<&str>); 3] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS, None),
];
const PEEK_ENABLED_ENV: [(&str, Option<&str>); 3] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS, Some("500")),
];

fn backend_error<T>(error_type: ErrorType, message: &str) -> Annotated<T> {
    Annotated {
        data: None,
        id: None,
        event: Some("error".to_string()),
        comment: None,
        error: Some(
            DynamoError::builder()
                .error_type(error_type)
                .message(message)
                .build(),
        ),
    }
}

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

#[tokio::test]
#[serial]
async fn completions_backend_validation_is_400_for_unary_and_streaming() {
    temp_env::async_with_vars(PEEK_ENABLED_ENV, async {
        const MAX_TOKENS_WITHIN_FRONTEND_LIMIT: u32 = 1_048_576;
        let max_tokens_message = "backend rejected max_tokens";
        let logprobs_message = "Dynamo's SGLang backend does not currently support logprobs >= 1";
        let scripts: Vec<CompletionScript> = [
            max_tokens_message,
            max_tokens_message,
            logprobs_message,
            logprobs_message,
        ]
        .into_iter()
        .map(|message| {
            vec![backend_error::<NvCreateCompletionResponse>(
                ErrorType::Backend(BackendError::InvalidArgument),
                message,
            )]
        })
        .collect();
        let svc = HarnessService::start_with_completion_scripts(scripts).await;

        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/completions", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "prompt": "test",
                    "max_tokens": MAX_TOKENS_WITHIN_FRONTEND_LIMIT,
                    "stream": stream
                }))
                .send()
                .await
                .unwrap();
            assert_openai_400(response, "max_tokens").await;
        }

        for stream in [false, true] {
            let response = svc
                .client
                .post(format!("{}/v1/completions", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "prompt": "hello",
                    "max_tokens": 16,
                    "logprobs": 1,
                    "stream": stream
                }))
                .send()
                .await
                .unwrap();
            assert_openai_400(response, "logprobs").await;
        }

        let requests = svc.completion_engine.take_requests().await;
        assert_eq!(requests.len(), 4);
        assert_eq!(
            requests[0].inner.max_tokens,
            Some(MAX_TOKENS_WITHIN_FRONTEND_LIMIT)
        );
        assert_eq!(
            requests[1].inner.max_tokens,
            Some(MAX_TOKENS_WITHIN_FRONTEND_LIMIT)
        );
        assert_eq!(requests[2].inner.logprobs, Some(1));
        assert_eq!(requests[3].inner.logprobs, Some(1));
        svc.shutdown().await;
    })
    .await;
}

fn partial_then_error(
    partial: &NvCreateChatCompletionStreamResponse,
    error_type: ErrorType,
    message: &str,
) -> AnnotatedScript {
    vec![
        Annotated::from_data(partial.clone()),
        backend_error(error_type, message),
    ]
}

#[tokio::test]
#[serial]
async fn unary_and_late_stream_errors_keep_classification_and_envelopes() {
    temp_env::async_with_vars(BASE_ENV, async {
        let partial = load_agent_fixture("text.sse").await.unwrap()[0].clone();
        let invalid = ErrorType::Backend(BackendError::InvalidArgument);
        let scripts = vec![
            partial_then_error(&partial, invalid, "invalid response input"),
            partial_then_error(
                &partial,
                ErrorType::Backend(BackendError::Unknown),
                "secret /srv/worker.py",
            ),
            partial_then_error(&partial, invalid, "invalid late response input"),
            partial_then_error(
                &partial,
                ErrorType::Backend(BackendError::Unknown),
                "secret /srv/late.py",
            ),
            partial_then_error(&partial, invalid, "invalid late Anthropic input"),
            partial_then_error(
                &partial,
                ErrorType::Backend(BackendError::Unknown),
                "secret /srv/anthropic.py",
            ),
            partial_then_error(&partial, invalid, "invalid late chat input"),
            partial_then_error(
                &partial,
                ErrorType::Backend(BackendError::Unknown),
                "secret /srv/chat.py",
            ),
        ];
        let partial_completion: NvCreateCompletionResponse = serde_json::from_value(json!({
            "id": "cmpl-partial",
            "object": "text_completion",
            "created": 0,
            "model": MODEL,
            "choices": [{
                "text": "partial",
                "index": 0,
                "logprobs": null,
                "finish_reason": null
            }]
        }))
        .unwrap();
        let completion_scripts = vec![
            vec![
                Annotated::from_data(partial_completion.clone()),
                backend_error(invalid, "invalid late completion input"),
            ],
            vec![
                Annotated::from_data(partial_completion),
                backend_error(
                    ErrorType::Backend(BackendError::Unknown),
                    "secret /srv/completion.py",
                ),
            ],
        ];
        let svc = HarnessService::start_with_scripts(scripts, completion_scripts).await;

        let response = svc
            .client
            .post(format!("{}/v1/responses", svc.base_url))
            .json(&json!({"model": MODEL, "input": "ping", "stream": false}))
            .send()
            .await
            .unwrap();
        assert_openai_400(response, "invalid response input").await;

        let response = svc
            .client
            .post(format!("{}/v1/responses", svc.base_url))
            .json(&json!({"model": MODEL, "input": "ping", "stream": false}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["message"], "Failed to fold responses stream");
        assert!(!body.to_string().contains("/srv/worker.py"));

        for (error_code, expected_message) in [
            ("invalid_prompt", "invalid late response input"),
            ("server_error", "Internal server error"),
        ] {
            let response = svc
                .client
                .post(format!("{}/v1/responses", svc.base_url))
                .json(&json!({"model": MODEL, "input": "ping", "stream": true}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let events = parse_json_sse(&response.text().await.unwrap())
                .await
                .unwrap();
            let failed = events
                .iter()
                .find(|event| event.event == "response.failed")
                .expect("missing response.failed event");
            assert_eq!(failed.data["response"]["error"]["code"], error_code);
            assert_eq!(
                failed.data["response"]["error"]["message"],
                expected_message
            );
        }

        for (error_type, expected_message) in [
            ("invalid_request_error", "invalid late Anthropic input"),
            ("api_error", "Internal server error"),
        ] {
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": true
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let events = parse_json_sse(&response.text().await.unwrap())
                .await
                .unwrap();
            let error = events
                .iter()
                .find(|event| event.event == "error")
                .expect("missing Anthropic error event");
            assert_eq!(error.data["error"]["type"], error_type);
            assert_eq!(error.data["error"]["message"], expected_message);
        }

        for (endpoint, client_message) in [
            ("chat/completions", "invalid late chat input"),
            ("completions", "invalid late completion input"),
        ] {
            for (code, error_type, expected_message) in [
                (400, "invalid_request_error", client_message.to_string()),
                (
                    500,
                    "internal_server_error",
                    "Internal server error".to_string(),
                ),
            ] {
                let body = if endpoint == "chat/completions" {
                    json!({
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": true
                    })
                } else {
                    json!({"model": MODEL, "prompt": "ping", "stream": true})
                };
                let response = svc
                    .client
                    .post(format!("{}/v1/{endpoint}", svc.base_url))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                let events = parse_json_sse(&response.text().await.unwrap())
                    .await
                    .unwrap();
                let error = events
                    .iter()
                    .find_map(|event| event.data.get("error"))
                    .expect("missing OpenAI inline error event");
                assert_eq!(error["code"], code);
                assert_eq!(error["type"], error_type);
                assert_eq!(error["message"], expected_message);
            }
        }

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_overload_statuses_use_overloaded_error() {
    temp_env::async_with_vars(PEEK_ENABLED_ENV, async {
        let cases = [
            (
                ErrorType::Unavailable,
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                "Service temporarily unavailable",
            ),
            (
                ErrorType::ResourceExhausted,
                reqwest::StatusCode::from_u16(529).unwrap(),
                "Service temporarily overloaded",
            ),
        ];
        let scripts = cases.iter().map(|(error_type, _, _)| {
            vec![backend_error(
                *error_type,
                "private backend overload detail",
            )]
        });
        let svc = HarnessService::start_with_scripts(scripts, []).await;

        for (_, expected_status, expected_message) in cases {
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .json(&json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": true
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status);
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["type"], "error");
            assert_eq!(body["error"]["type"], "overloaded_error");
            assert_eq!(body["error"]["message"], expected_message);
            assert!(!body.to_string().contains("private backend overload detail"));
        }

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn responses_operational_error_kinds_use_protocol_codes() {
    temp_env::async_with_vars(BASE_ENV, async {
        let cases = [
            (
                ErrorType::Unavailable,
                "server_error",
                "Service temporarily unavailable",
            ),
            (
                ErrorType::ResourceExhausted,
                "server_error",
                "Service temporarily overloaded",
            ),
            (
                ErrorType::Cancelled,
                "request_cancelled",
                "Request cancelled",
            ),
        ];
        let scripts = cases.iter().map(|(error_type, _, _)| {
            vec![backend_error(*error_type, "private backend error detail")]
        });
        let svc = HarnessService::start_with_scripts(scripts, []).await;

        for (_, expected_code, expected_message) in cases {
            let response = svc
                .client
                .post(format!("{}/v1/responses", svc.base_url))
                .json(&json!({"model": MODEL, "input": "ping", "stream": true}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let events = parse_json_sse(&response.text().await.unwrap())
                .await
                .unwrap();
            let failed = events
                .iter()
                .find(|event| event.event == "response.failed")
                .expect("missing response.failed event");
            assert_eq!(failed.data["response"]["error"]["code"], expected_code);
            assert_eq!(
                failed.data["response"]["error"]["message"],
                expected_message
            );
            assert!(
                !failed
                    .data
                    .to_string()
                    .contains("private backend error detail")
            );
        }

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn native_stream_errors_are_terminal_and_recorded_as_failures() {
    temp_env::async_with_vars(BASE_ENV, async {
        let partial = load_agent_fixture("text.sse").await.unwrap()[0].clone();
        let invalid = ErrorType::Backend(BackendError::InvalidArgument);
        let svc = HarnessService::start_with_pending_annotated_scripts([
            partial_then_error(&partial, invalid, "terminal Responses validation"),
            partial_then_error(&partial, invalid, "terminal Anthropic validation"),
        ])
        .await;

        let response = svc
            .client
            .post(format!("{}/v1/responses", svc.base_url))
            .json(&json!({"model": MODEL, "input": "ping", "stream": true}))
            .send()
            .await
            .unwrap();
        let body = tokio::time::timeout(Duration::from_secs(1), response.text())
            .await
            .expect("Responses stream waited for backend EOF")
            .unwrap();
        let events = parse_json_sse(&body).await.unwrap();
        let failed = events
            .iter()
            .find(|event| event.event == "response.failed")
            .expect("missing terminal response.failed event");
        assert_eq!(
            failed.data["response"]["error"]["message"],
            "terminal Responses validation"
        );

        let response = svc
            .client
            .post(format!("{}/v1/messages", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": true
            }))
            .send()
            .await
            .unwrap();
        let body = tokio::time::timeout(Duration::from_secs(1), response.text())
            .await
            .expect("Anthropic stream waited for backend EOF")
            .unwrap();
        let events = parse_json_sse(&body).await.unwrap();
        let error = events
            .iter()
            .find(|event| event.event == "error")
            .expect("missing terminal Anthropic error event");
        assert_eq!(
            error.data["error"]["message"],
            "terminal Anthropic validation"
        );

        let metrics = svc
            .client
            .get(format!("{}/metrics", svc.base_url))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        for endpoint in ["responses", "anthropic_messages"] {
            let line = metrics
                .lines()
                .find(|line| {
                    line.starts_with("dynamo_frontend_requests_total{")
                        && line.contains(&format!("endpoint=\"{endpoint}\""))
                        && line.contains("request_type=\"stream\"")
                        && line.contains("status=\"error\"")
                        && line.contains("error_type=\"validation\"")
                })
                .unwrap_or_else(|| panic!("missing validation failure metric for {endpoint}"));
            assert!(line.ends_with(" 1"), "unexpected request metric: {line}");
        }
        assert!(
            metrics.lines().any(|line| {
                line == format!("dynamo_frontend_inflight_requests{{model=\"{MODEL}\"}} 0")
            }),
            "inflight gauge did not return to zero"
        );

        svc.shutdown().await;
    })
    .await;
}
