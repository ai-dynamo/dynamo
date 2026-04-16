// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harness integration tests for Dynamo's Anthropic `/v1/messages` surface.
//!
//! Each test loads a handwritten JSONL replay, spins up the real HTTP service
//! with a [`ReplayEngine`] wired in as the backing chat-completions engine,
//! drives a request through `reqwest`, and snapshots the SSE event sequence
//! (or the unary JSON body) with `insta`.
//!
//! The env var `DYN_ENABLE_ANTHROPIC_API=1` is required for the `/v1/messages`
//! route to be mounted — all tests set it via `temp_env::async_with_vars` and
//! run with `serial_test` so the process-global variable doesn't race.

use std::time::Duration;

use dynamo_runtime::config::environment_names::llm::DYN_ENABLE_ANTHROPIC_API;
use futures::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;

#[path = "common/harness.rs"]
mod harness;
#[path = "common/replay_engine.rs"]
mod replay_engine;

use harness::{HarnessService, MODEL, ParsedEvent, parse_sse, redact};

const ENV: [(&str, Option<&str>); 1] = [(DYN_ENABLE_ANTHROPIC_API, Some("1"))];

async fn post_messages(svc: &HarnessService, body: &Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{}/v1/messages", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/messages failed")
}

fn redacted_events(body: &str) -> Vec<ParsedEvent> {
    parse_sse(body)
        .into_iter()
        .map(|ev| ParsedEvent {
            event: ev.event,
            data: redact(ev.data),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// A1 — text only, non-streaming
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn hello_unary() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start("claude_code__hello.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": false,
            "messages": [{"role": "user", "content": "Say hi"}],
        });
        let resp = post_messages(&svc, &body).await;
        assert!(resp.status().is_success(), "status: {:?}", resp.status());
        let json: Value = resp.json().await.unwrap();
        insta::assert_json_snapshot!("hello_unary", redact(json));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A2 — text only, streaming SSE
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn hello_streaming() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start("claude_code__hello.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "Say hi"}],
        });
        let resp = post_messages(&svc, &body).await;
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        insta::assert_json_snapshot!("hello_streaming", redacted_events(&text));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A3 — single tool_use block
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn tool_use_single() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start("claude_code__tool_use.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 256,
            "stream": true,
            "tools": [{
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city"]
                }
            }],
            "messages": [{
                "role": "user",
                "content": "What is the weather in Paris?"
            }]
        });
        let resp = post_messages(&svc, &body).await;
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        insta::assert_json_snapshot!("tool_use_single", redacted_events(&text));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A4 — parallel tool_use
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn tool_use_parallel() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start(
            "claude_code__multi_tool_parallel.chunks.jsonl",
            Duration::ZERO,
        )
        .await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 256,
            "stream": true,
            "tools": [
                {"name": "get_weather", "description": "weather",
                 "input_schema": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
                {"name": "get_time", "description": "time",
                 "input_schema": {"type":"object","properties":{"tz":{"type":"string"}}}}
            ],
            "messages": [{"role":"user","content":"Paris weather and UTC time?"}]
        });
        let resp = post_messages(&svc, &body).await;
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        insta::assert_json_snapshot!("tool_use_parallel", redacted_events(&text));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A5 — tool_result round-trip (request accepted, response streams)
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn tool_result_roundtrip() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start("claude_code__multiturn.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_abc123", "name": "get_weather",
                     "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_abc123",
                     "content": "14C clear"}
                ]}
            ]
        });
        let resp = post_messages(&svc, &body).await;
        assert!(
            resp.status().is_success(),
            "status: {} body: {:?}",
            resp.status(),
            resp.text().await
        );
        let text = resp.text().await.unwrap();
        insta::assert_json_snapshot!("tool_result_roundtrip", redacted_events(&text));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A6 — client disconnect mid-stream
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn cancel_midstream() {
    temp_env::async_with_vars(ENV, async {
        // Long inter-chunk interval so the replay outlives the client.
        // Mirrors the timeout+drop pattern in
        // `test_client_disconnect_cancellation_streaming` in http-service.rs.
        let svc = HarnessService::start(
            "claude_code__cancel_midstream.chunks.jsonl",
            Duration::from_millis(500),
        )
        .await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role":"user","content":"tell me a long story"}]
        });

        let base_url = svc.base_url.clone();
        let request_future = async move {
            let resp = reqwest::Client::new()
                .post(format!("{}/v1/messages", base_url))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert!(resp.status().is_success());
            let mut stream = resp.bytes_stream();
            let first = stream.next().await.expect("expected at least one chunk");
            assert!(first.is_ok());
            // Stream (and response) dropped when this future is cancelled
            // by the outer timeout — forces the TCP connection closed.
            let _ = stream.next().await;
        };
        let _ = tokio::time::timeout(Duration::from_millis(800), request_future).await;

        // Poll for the engine to observe the cancellation. Axum takes
        // up to ~1s to notice the TCP disconnect.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while svc.engine.observed_cancels().await == 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            svc.engine.observed_cancels().await >= 1,
            "ReplayEngine never observed the client-side cancellation"
        );

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A7 — count_tokens endpoint shape
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn count_tokens_shape() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start("claude_code__hello.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "messages": [{"role":"user","content":"hello world"}]
        });
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(
            resp.status().is_success() || resp.status().as_u16() == 501,
            "status: {}",
            resp.status()
        );
        let value: Value = resp.json().await.unwrap_or(Value::Null);
        insta::assert_json_snapshot!("count_tokens", redact(value));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A8 (extra) — stop_sequences termination
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn stop_sequence_streaming() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start("claude_code__stop_sequence.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "stop_sequences": ["###END"],
            "messages": [{"role": "user", "content": "print lines"}]
        });
        let resp = post_messages(&svc, &body).await;
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        insta::assert_json_snapshot!("stop_sequence", redacted_events(&text));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A9 (extra) — multimodal input: image block + text, unary response
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn multimodal_image_input() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start("claude_code__hello.chunks.jsonl", Duration::ZERO).await;

        // 1x1 transparent PNG as a base64 image payload.
        let png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAeImBZsAAAAASUVORK5CYII=";
        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": false,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": png_b64
                    }},
                    {"type": "text", "text": "Describe this image"}
                ]
            }]
        });
        let resp = post_messages(&svc, &body).await;
        assert!(
            resp.status().is_success(),
            "status: {} body: {:?}",
            resp.status(),
            resp.text().await
        );
        let json: Value = resp.json().await.unwrap();
        insta::assert_json_snapshot!("multimodal_image", redact(json));

        svc.shutdown().await;
    })
    .await;
}

// ---------------------------------------------------------------------------
// A10 (extra) — multi-turn conversation continuation
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn multiturn_continuation() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start("claude_code__multiturn.chunks.jsonl", Duration::ZERO).await;

        let body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "system": "You are a terse weather bot.",
            "messages": [
                {"role": "user", "content": "Paris weather?"},
                {"role": "assistant", "content": "Let me check."},
                {"role": "user", "content": "Now in Celsius."}
            ]
        });
        let resp = post_messages(&svc, &body).await;
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        insta::assert_json_snapshot!("multiturn", redacted_events(&text));

        svc.shutdown().await;
    })
    .await;
}
