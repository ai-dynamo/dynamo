// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Harness integration tests for Dynamo's OpenAI `/v1/responses` surface —
//! the Codex-facing endpoint.
//!
//! Same pattern as `anthropic_replay.rs`: JSONL chat-completion chunks are
//! replayed through the real HTTP service, which routes them through the
//! Responses stream converter. The resulting SSE event sequence is parsed,
//! redacted, and snapshotted with `insta`.

use std::time::Duration;

use futures::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;

#[path = "common/harness.rs"]
mod harness;
#[path = "common/replay_engine.rs"]
mod replay_engine;

use harness::{HarnessService, MODEL, ParsedEvent, parse_sse, redact};

async fn post_responses(svc: &HarnessService, body: &Value) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{}/v1/responses", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/responses failed")
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
// R1 — plain text streaming
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn hello_streaming() {
    let svc = HarnessService::start("codex__hello.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "input": "ping",
        "stream": true,
        "max_output_tokens": 64
    });
    let resp = post_responses(&svc, &body).await;
    assert!(resp.status().is_success());
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("hello_streaming", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R2 — reasoning summary
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn reasoning_summary() {
    let svc = HarnessService::start("codex__reasoning_summary.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "input": "hi",
        "stream": true,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 128
    });
    let resp = post_responses(&svc, &body).await;
    assert!(resp.status().is_success());
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("reasoning_summary", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R3 — single function_call
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn function_call_single() {
    let svc = HarnessService::start("codex__function_call.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "input": "list files under /tmp",
        "stream": true,
        "max_output_tokens": 128,
        "tools": [{
            "type": "function",
            "name": "list_files",
            "description": "List files at a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }]
    });
    let resp = post_responses(&svc, &body).await;
    assert!(resp.status().is_success());
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("function_call_single", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R4 — parallel function_calls
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn function_call_parallel() {
    let svc = HarnessService::start(
        "codex__parallel_function_calls.chunks.jsonl",
        Duration::ZERO,
    )
    .await;

    let body = json!({
        "model": MODEL,
        "input": "read both /a and /b",
        "stream": true,
        "max_output_tokens": 256,
        "parallel_tool_calls": true,
        "tools": [{
            "type": "function",
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }]
    });
    let resp = post_responses(&svc, &body).await;
    assert!(resp.status().is_success());
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("function_call_parallel", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R5 — FunctionCallOutput follow-up turn
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn function_call_output_followup() {
    let svc =
        HarnessService::start("codex__reasoning_multiturn.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "stream": true,
        "max_output_tokens": 64,
        "input": [
            {"role": "user", "content": "What is 6 * 7?"},
            {"type": "function_call",
             "id": "fc_1", "call_id": "call_fc_1",
             "name": "multiply",
             "arguments": "{\"a\":6,\"b\":7}"},
            {"type": "function_call_output",
             "call_id": "call_fc_1",
             "output": "42"}
        ]
    });
    let resp = post_responses(&svc, &body).await;
    assert!(
        resp.status().is_success(),
        "status: {} body: {:?}",
        resp.status(),
        resp.text().await
    );
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("function_call_output_followup", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R6 — client disconnect mid-stream
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn cancel_midstream() {
    let svc = HarnessService::start(
        "claude_code__cancel_midstream.chunks.jsonl",
        Duration::from_millis(500),
    )
    .await;

    let body = json!({
        "model": MODEL,
        "input": "tell me a long story",
        "stream": true,
        "max_output_tokens": 128
    });

    let base_url = svc.base_url.clone();
    let request_future = async move {
        let resp = reqwest::Client::new()
            .post(format!("{}/v1/responses", base_url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let mut stream = resp.bytes_stream();
        let first = stream.next().await.expect("expected at least one chunk");
        assert!(first.is_ok());
        let _ = stream.next().await;
    };
    let _ = tokio::time::timeout(Duration::from_millis(800), request_future).await;

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while svc.engine.observed_cancels().await == 0 && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        svc.engine.observed_cancels().await >= 1,
        "ReplayEngine never observed the client-side cancellation"
    );

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R7 (extra) — deep tool-chain (3 sequential function calls in one turn)
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn deep_tool_chain() {
    let svc = HarnessService::start("codex__deep_tool_chain.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "input": "survey the repo",
        "stream": true,
        "max_output_tokens": 256,
        "parallel_tool_calls": true,
        "tools": [
            {"type": "function", "name": "ls",
             "parameters": {"type":"object","properties":{"path":{"type":"string"}}}},
            {"type": "function", "name": "grep",
             "parameters": {"type":"object","properties":{"pattern":{"type":"string"}}}},
            {"type": "function", "name": "git_diff",
             "parameters": {"type":"object","properties":{"branch":{"type":"string"}}}}
        ]
    });
    let resp = post_responses(&svc, &body).await;
    assert!(resp.status().is_success());
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("deep_tool_chain", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R8 (extra) — structured output (JSON schema text.format)
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn structured_output() {
    let svc = HarnessService::start("codex__structured_output.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "input": "return a name and version",
        "stream": true,
        "max_output_tokens": 64,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "Pkg",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "integer"}
                    },
                    "required": ["name", "version"],
                    "additionalProperties": false
                }
            }
        }
    });
    let resp = post_responses(&svc, &body).await;
    assert!(
        resp.status().is_success(),
        "status: {} body: {:?}",
        resp.status(),
        resp.text().await
    );
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("structured_output", redacted_events(&text));

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// R9 (extra) — multi-turn reasoning continuation
// ---------------------------------------------------------------------------
#[tokio::test]
#[serial]
async fn multiturn_reasoning() {
    let svc =
        HarnessService::start("codex__reasoning_multiturn.chunks.jsonl", Duration::ZERO).await;

    let body = json!({
        "model": MODEL,
        "stream": true,
        "max_output_tokens": 128,
        "reasoning": {"effort": "medium"},
        "input": [
            {"role": "user", "content": "What did you compute last time?"},
            {"role": "assistant", "content": "I computed 6 * 7 = 42."},
            {"role": "user", "content": "What is the answer?"}
        ]
    });
    let resp = post_responses(&svc, &body).await;
    assert!(resp.status().is_success());
    let text = resp.text().await.unwrap();
    insta::assert_json_snapshot!("multiturn_reasoning", redacted_events(&text));

    svc.shutdown().await;
}
