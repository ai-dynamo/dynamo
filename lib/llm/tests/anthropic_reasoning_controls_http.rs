// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Messages reasoning controls reach the existing canonical backend fields.

use std::time::Duration;

use dynamo_llm::protocols::common::StopConditionsProvider;

use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
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

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];

#[tokio::test]
#[serial]
async fn malformed_reasoning_controls_reject_before_generation() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([script]).await;
        let mut cases: Vec<_> = ["unknown", "none", "minimal"]
            .into_iter()
            .map(|effort| {
                (
                    "output_config",
                    json!({"effort": effort}),
                    "output_config.effort",
                )
            })
            .collect();
        cases.push((
            "output_config",
            json!({"effort": 42}),
            "output_config.effort",
        ));
        for budget in [0, 1023, 4096] {
            cases.push((
                "thinking",
                json!({"type": "enabled", "budget_tokens": budget}),
                "thinking.budget_tokens",
            ));
        }
        for stream in [false, true] {
            for (field, value, diagnostic) in &cases {
                let mut body = json!({"model": MODEL, "max_tokens": 4096, "stream": stream,
                    "messages": [{"role": "user", "content": "ping"}]});
                body[*field] = value.clone();
                let response = svc
                    .client
                    .post(format!("{}/v1/messages", svc.base_url))
                    .timeout(Duration::from_secs(10))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                assert_eq!(response.headers()["content-type"], "application/json");
                let error: Value = response.json().await.unwrap();
                assert_eq!(error["type"], "error");
                assert_eq!(error["error"]["type"], "invalid_request_error");
                assert!(
                    error["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains(diagnostic),
                    "{error}"
                );
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
async fn omitted_and_supported_thinking_modes_still_reach_generation() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(std::iter::repeat_n(script, 6)).await;
        for stream in [false, true] {
            for (thinking, enabled) in [
                (Value::Null, None),
                (json!({"type": "adaptive"}), None),
                (json!({"type": "disabled"}), Some(false)),
            ] {
                let body = json!({"model": MODEL, "max_tokens": 4096, "stream": stream,
                    "messages": [{"role": "user", "content": "ping"}],
                    "thinking": thinking, "output_config": {}});
                let response = svc
                    .client
                    .post(format!("{}/v1/messages", svc.base_url))
                    .timeout(Duration::from_secs(10))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                let _ = response.text().await.unwrap();
                let requests = svc.engine.take_requests().await;
                assert_eq!(requests.len(), 1);
                assert_eq!(
                    requests[0]
                        .chat_template_args
                        .as_ref()
                        .and_then(|args| args.get("enable_thinking"))
                        .and_then(Value::as_bool),
                    enabled
                );
            }
        }
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn effort_and_manual_budgets_reach_canonical_fields() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(std::iter::repeat_n(script, 20)).await;
        for stream in [false, true] {
            for effort in ["low", "medium", "high", "xhigh", "max"] {
                for budget in [1024, 2048] {
                    let body = json!({"model": MODEL, "max_tokens": 4096, "stream": stream,
                        "messages": [{"role": "user", "content": "ping"}],
                        "thinking": {"type": "enabled", "budget_tokens": budget},
                        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": {"type": "object"}}}});
                    let response = svc.client.post(format!("{}/v1/messages", svc.base_url)).json(&body).send().await.unwrap();
                    assert_eq!(response.status(), reqwest::StatusCode::OK, "{}", response.text().await.unwrap());
                    let requests = svc.engine.take_requests().await;
                    assert_eq!(requests.len(), 1);
                    let request = &requests[0];
                    assert_eq!(serde_json::to_value(&request.inner.reasoning_effort).unwrap(), effort);
                    assert_eq!(request.thinking_token_budget, Some(budget));
                    assert_eq!(request.extract_stop_conditions().unwrap().max_thinking_tokens, Some(budget));
                    assert!(request.inner.response_format.is_some());
                    let args = request.chat_template_args.as_ref().unwrap();
                    assert_eq!(args["reasoning_effort"], effort);
                    assert_eq!(args["enable_thinking"], true);
                }
            }
        }
        svc.shutdown().await;
    }).await;
}

#[tokio::test]
#[serial]
async fn claude_code_default_effort_shape_is_accepted() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([script]).await;
        // Captured Claude Code request shape for a non-Claude model; no live inference.
        let body = json!({"model": MODEL, "max_tokens": 32000, "stream": true,
            "messages": [{"role": "user", "content": "ping"}],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "high"}});
        let response = svc
            .client
            .post(format!("{}/v1/messages", svc.base_url))
            .header(
                "anthropic-beta",
                "interleaved-thinking-2025-05-14,effort-2025-11-24",
            )
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(response.text().await.unwrap().contains("message_stop"));
        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(
            serde_json::to_value(&request.inner.reasoning_effort).unwrap(),
            "high"
        );
        let args = request.chat_template_args.as_ref().unwrap();
        assert_eq!(args["reasoning_effort"], "high");
        assert_eq!(args["thinking_mode"], "adaptive");
        assert!(!args.contains_key("enable_thinking"));
        svc.shutdown().await;
    })
    .await;
}
