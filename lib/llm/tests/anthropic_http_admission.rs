// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request admission and response envelopes through the real Messages HTTP routes.

use std::time::Duration;

use dynamo_protocols::types::ChatCompletionToolChoiceOption;
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_BODY_LIMIT_MB, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
};
use reqwest::StatusCode;
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

const ENV: [(&str, Option<&str>); 3] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (DYN_HTTP_BODY_LIMIT_MB, Some("1")),
];

fn request() -> Value {
    json!({
        "model": MODEL, "max_tokens": 2048,
        "messages": [{"role": "user", "content": "ping"}]
    })
}

async fn assert_error(response: reqwest::Response, status: StatusCode, error_type: &str) -> Value {
    assert_eq!(response.status(), status);
    assert_eq!(response.headers()["content-type"], "application/json");
    let request_id = response.headers()["request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert!(!request_id.is_empty());
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], error_type);
    assert!(!body["error"]["message"].as_str().unwrap().is_empty());
    assert_eq!(body["request_id"], request_id);
    body
}

#[tokio::test]
#[serial]
async fn extractor_errors_have_anthropic_envelopes_and_request_ids() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let mut oversized = request();
        oversized["messages"][0]["content"] = "x".repeat(1024 * 1024).into();
        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            for (content_type, body, status, error_type) in [
                (
                    Some("application/json"),
                    "{".into(),
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                ),
                (
                    Some("application/json"),
                    "{}".into(),
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                ),
                (
                    Some("text/plain"),
                    request().to_string(),
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                ),
                (
                    None,
                    request().to_string(),
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                ),
                (
                    Some("application/json"),
                    oversized.to_string(),
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "request_too_large",
                ),
            ] {
                let mut post = svc
                    .client
                    .post(format!("{}{path}", svc.base_url))
                    .timeout(Duration::from_secs(10))
                    .header("x-request-id", "client-correlation")
                    .body(body);
                if let Some(content_type) = content_type {
                    post = post.header("content-type", content_type);
                }
                let response = post.send().await.unwrap();
                assert_eq!(response.headers()["x-request-id"], "client-correlation");
                let body = assert_error(response, status, error_type).await;
                assert_ne!(body["request_id"], "client-correlation");
            }
        }
        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn duplicate_request_fields_are_rejected_before_generation() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([script]).await;
        let model = serde_json::to_string(MODEL).unwrap();
        // Keep raw JSON: constructing a Value would discard the duplicate keys.
        let fields = [
            format!(
                r#""model":{model},"model":{model},"messages":[{{"role":"user","content":"ping"}}]"#
            ),
            format!(
                r#""model":{model},"mo\u0064el":{model},"messages":[{{"role":"user","content":"ping"}}]"#
            ),
            format!(
                r#""model":{model},"messages":[{{"role":"user","role":"user","content":"ping"}}]"#
            ),
            format!(
                r#""model":{model},"messages":[{{"role":"user","content":"ping"}}],"system":[{{"type":"text","text":"first","text":"second"}}]"#
            ),
            format!(
                r#""model":{model},"messages":[{{"role":"user","content":"ping"}}],"system":null,"system":null"#
            ),
        ];
        for path in ["/v1/messages", "/v1/messages/count_tokens"] {
            for stream in [false, true] {
                for fields in &fields {
                    let body = format!("{{{fields},\"max_tokens\":2048,\"stream\":{stream}}}");
                    let response = svc
                        .client
                        .post(format!("{}{path}", svc.base_url))
                        .timeout(Duration::from_secs(10))
                        .header("content-type", "application/json")
                        .body(body)
                        .send()
                        .await
                        .unwrap();
                    assert_error(response, StatusCode::BAD_REQUEST, "invalid_request_error").await;
                }
            }
        }
        assert!(svc.engine.take_requests().await.is_empty());
        assert_eq!(svc.engine.remaining_scripts().await, 1);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn malformed_request_fields_are_rejected_before_generation() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        let invalid = [
            ("tool_choice", json!({"type": "none", "name": "read"})),
            ("tool_choice", json!({"type": "auto", "name": "read"})),
            ("tool_choice", json!({"type": "any", "name": "read"})),
            ("tool_choice", json!({"type": "auto", "name": null})),
            ("tool_choice", json!({"type": "tool"})),
            ("tool_choice", json!({"type": "tool", "name": ""})),
            ("tool_choice", json!({"type": "tool", "name": 1})),
            ("tool_choice", json!({"type": "tool", "name": null})),
            ("tool_choice", json!({"type": "unknown"})),
            ("tool_choice", json!({"name": "read"})),
            (
                "tool_choice",
                json!({"type": "auto", "disable_parallel_tool_use": 1}),
            ),
            (
                "tool_choice",
                json!({"type": "none", "disable_parallel_tool_use": false}),
            ),
            ("tool_choice", json!("auto")),
            ("metadata", json!(123)),
            ("metadata", json!(true)),
            ("metadata", json!("user")),
            ("metadata", json!([])),
            ("metadata", json!({"user_id": 1})),
            ("metadata", json!({"user_id": "é".repeat(513)})),
            ("system", json!([{"type": "image", "text": "instructions"}])),
            ("system", json!([{"text": "instructions"}])),
            ("system", json!([{"type": null, "text": "instructions"}])),
            ("system", json!([{"type": "text", "text": 123}])),
            ("system", json!([{"type": "text", "text": ""}])),
            ("system", json!(["instructions"])),
            ("system", json!(true)),
            ("thinking", json!({"type": "enabled"})),
            (
                "thinking",
                json!({"type": "enabled", "budget_tokens": 1023}),
            ),
            ("thinking", json!({"type": "enabled", "budget_tokens": 0})),
            ("thinking", json!({"type": "enabled", "budget_tokens": -1})),
            (
                "thinking",
                json!({"type": "enabled", "budget_tokens": 1024.5}),
            ),
            (
                "thinking",
                json!({"type": "enabled", "budget_tokens": null}),
            ),
            (
                "thinking",
                json!({"type": "enabled", "budget_tokens": 2048}),
            ),
            (
                "thinking",
                json!({"type": "enabled", "budget_tokens": 2049}),
            ),
            ("thinking", json!({"type": "unknown"})),
            (
                "thinking",
                json!({"type": "disabled", "budget_tokens": 1024}),
            ),
            (
                "thinking",
                json!({"type": "adaptive", "budget_tokens": 1024}),
            ),
            ("thinking", json!(true)),
        ];
        for stream in [false, true] {
            for (field, value) in &invalid {
                let mut body = request();
                body["stream"] = stream.into();
                body["tools"] = json!([{"name": "read", "input_schema": {"type": "object"}}]);
                body[*field] = value.clone();
                let response = svc
                    .client
                    .post(format!("{}/v1/messages", svc.base_url))
                    .timeout(Duration::from_secs(10))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                let error =
                    assert_error(response, StatusCode::BAD_REQUEST, "invalid_request_error").await;
                assert!(
                    error["error"]["message"].as_str().unwrap().contains(field),
                    "{field}={value}: {error}"
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
async fn valid_choices_and_shapes_preserve_backend_requests_and_assigned_ids() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(std::iter::repeat_n(script, 8)).await;
        let mut response_ids = Vec::new();
        for stream in [false, true] {
            for choice_type in ["none", "auto", "any", "tool"] {
                let mut body = request();
                body["stream"] = stream.into();
                body["max_tokens"] = 1025.into();
                body["thinking"] = if matches!(choice_type, "auto" | "none") {
                    json!({"type": "enabled", "budget_tokens": 1024})
                } else {
                    json!({"type": "disabled"})
                };
                body["system"] = json!([{"type": "text", "text": "instructions"}]);
                body["metadata"] = json!({"user_id": "é".repeat(512)});
                body["tools"] = json!([{"name": "read", "input_schema": {"type": "object"}}]);
                body["tool_choice"] = json!({"type": choice_type});
                if choice_type == "tool" {
                    body["tool_choice"]["name"] = "read".into();
                }
                let response = svc.client.post(format!("{}/v1/messages", svc.base_url)).timeout(Duration::from_secs(10))
                    .json(&body).send().await.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                response_ids.push(response.headers()["request-id"].to_str().unwrap().to_owned());
                let text = response.text().await.unwrap();
                if stream {
                    assert!(!text.contains("[DONE]"));
                    assert!(text.trim_end().ends_with("data: {\"type\":\"message_stop\"}"));
                }
            }
        }
        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 8);
        for group in requests.chunks_exact(4) {
            assert_eq!(group[0].inner.tool_choice, Some(ChatCompletionToolChoiceOption::None));
            assert_eq!(group[1].inner.tool_choice, Some(ChatCompletionToolChoiceOption::Auto));
            assert_eq!(group[2].inner.tool_choice, Some(ChatCompletionToolChoiceOption::Required));
            assert!(matches!(&group[3].inner.tool_choice, Some(ChatCompletionToolChoiceOption::Named(named)) if named.function.name == "read"));
        }
        let contexts = svc.engine.take_contexts().await;
        let backend_ids: Vec<_> = contexts.iter().map(|context| context.id().to_owned()).collect();
        assert_eq!(response_ids, backend_ids);
        response_ids.sort();
        response_ids.dedup();
        assert_eq!(response_ids.len(), 8);
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    }).await;
}

#[tokio::test]
#[serial]
async fn handler_errors_and_token_counts_have_request_ids() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([]).await;
        for (field, value, status, error_type) in [
            (
                "max_tokens",
                json!(0),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                "model",
                json!("missing-model"),
                StatusCode::NOT_FOUND,
                "not_found_error",
            ),
        ] {
            let mut body = request();
            body[field] = value;
            let response = svc
                .client
                .post(format!("{}/v1/messages", svc.base_url))
                .timeout(Duration::from_secs(10))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_error(response, status, error_type).await;
        }
        let response = svc
            .client
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .timeout(Duration::from_secs(10))
            .json(&request())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            !response.headers()["request-id"]
                .to_str()
                .unwrap()
                .is_empty()
        );
        let body: Value = response.json().await.unwrap();
        assert!(body["input_tokens"].is_number());

        let mut invalid = request();
        invalid["system"] = json!([{"type": "image", "text": "instructions"}]);
        let response = svc
            .client
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .timeout(Duration::from_secs(10))
            .json(&invalid)
            .send()
            .await
            .unwrap();
        assert_error(response, StatusCode::BAD_REQUEST, "invalid_request_error").await;
        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}
