// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU HTTP regressions for chosen-token logprobs without top alternatives.
//!
//! Backend output is injected deterministically; these tests exercise the real
//! Rust delta converter, HTTP aggregation, and SSE serialization, not inference.

use std::time::Duration;

use dynamo_llm::protocols::{
    common::{FinishReason, llm_backend::BackendOutput},
    openai::{DeltaGeneratorExt, chat_completions::NvCreateChatCompletionRequest},
};
use dynamo_runtime::config::environment_names::llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS;
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

use http_harness::{HarnessService, MODEL, parse_json_sse};
use scripted_chat_engine::Script;

const TOKENS: [(&str, u32, f64); 2] = [("Hello", 42, -0.125), ("!", 99, -0.75)];

fn request_body(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_completion_tokens": TOKENS.len(),
        "stream": stream,
        "logprobs": true,
        "top_logprobs": 0,
    })
}

fn converted_backend_script(body: &Value) -> Script {
    let mut request: NvCreateChatCompletionRequest =
        serde_json::from_value(body.clone()).expect("invalid regression request");
    // Backend preprocessing enables final usage for nonstreaming requests;
    // the backend always returns chunks for the HTTP handler to aggregate.
    request.enable_usage_for_nonstreaming(request.inner.stream.unwrap_or(false));
    let mut generator = request.response_generator("zero-top-regression".to_string());
    generator.update_isl(3);

    let mut chunks: Script = TOKENS
        .iter()
        .enumerate()
        .map(|(index, &(token, token_id, logprob))| {
            generator
                .choice_from_postprocessor(BackendOutput {
                    token_ids: vec![token_id],
                    tokens: vec![Some(token.to_string())],
                    text: Some(token.to_string()),
                    cum_log_probs: None,
                    log_probs: Some(vec![logprob]),
                    top_logprobs: None,
                    finish_reason: (index + 1 == TOKENS.len()).then_some(FinishReason::Stop),
                    stop_reason: None,
                    index: Some(0),
                    completion_usage: None,
                    disaggregated_params: None,
                    encoder_result: None,
                    worker_trace_link: None,
                    engine_data: None,
                    routing_data: None,
                })
                .expect("backend output conversion failed")
        })
        .collect();
    if generator.is_usage_enabled() {
        chunks.push(generator.create_usage_chunk());
    }
    chunks
}

fn assert_chosen_logprobs(content: &Value, expected: &[(&str, u32, f64)]) {
    let entries = content
        .as_array()
        .expect("HTTP logprobs.content must contain chosen-token entries, not null");
    assert_eq!(entries.len(), expected.len());
    for (entry, &(token, token_id, logprob)) in entries.iter().zip(expected) {
        assert_eq!(entry["token"], token);
        assert_eq!(entry["token_id"], token_id);
        assert_eq!(entry["bytes"], json!(token.as_bytes()));
        let actual = entry["logprob"]
            .as_f64()
            .expect("chosen-token logprob must be numeric");
        assert!(actual.is_finite());
        assert_eq!(actual, logprob);
        assert_eq!(entry["top_logprobs"], json!([]));
    }
}

async fn assert_http_response(stream: bool) {
    let body = request_body(stream);
    // Do not assert on the generated chunks before sending the HTTP request:
    // the regression must be observable in the actual HTTP response body.
    let svc = HarnessService::start([converted_backend_script(&body)]).await;
    let response = svc
        .client
        .post(format!("{}/v1/chat/completions", svc.base_url))
        .timeout(Duration::from_secs(5))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let content_type = response.headers()[reqwest::header::CONTENT_TYPE]
        .to_str()
        .unwrap()
        .to_string();
    let raw = response.text().await.expect("failed to read HTTP response");
    println!("stream={stream}, top_logprobs=0, HTTP response:\n{raw}");

    if stream {
        assert!(content_type.starts_with("text/event-stream"));
        let events = parse_json_sse(&raw).await.expect("invalid SSE response");
        // The shared message codec consumes the terminal [DONE] sentinel.
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        let chunks: Vec<&Value> = events.iter().map(|event| &event.data).collect();
        assert_eq!(chunks.len(), TOKENS.len());
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk["choices"].as_array().unwrap().len(), 1);
            let choice = &chunk["choices"][0];
            assert_eq!(choice["delta"]["content"], TOKENS[index].0);
            assert_chosen_logprobs(&choice["logprobs"]["content"], &TOKENS[index..=index]);
        }
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    } else {
        assert!(content_type.starts_with("application/json"));
        let response: Value = serde_json::from_str(&raw).expect("invalid JSON response");
        assert_eq!(response["choices"].as_array().unwrap().len(), 1);
        let choice = &response["choices"][0];
        assert_eq!(choice["message"]["content"], "Hello!");
        assert_eq!(choice["finish_reason"], "stop");
        assert_chosen_logprobs(&choice["logprobs"]["content"], &TOKENS);
    }

    // The shared harness uses precomputed chunks. Check that the real incoming
    // request retained the same converter options used to prepare that script.
    let requests = svc.engine.take_requests().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].inner.logprobs, Some(true));
    assert_eq!(requests[0].inner.top_logprobs, Some(0));
    assert_eq!(requests[0].inner.stream, Some(stream));
    assert_eq!(svc.engine.remaining_scripts().await, 0);
    svc.shutdown().await;
}

async fn run_case(stream: bool) {
    temp_env::async_with_vars(
        [(DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0"))],
        async {
            tokio::time::timeout(Duration::from_secs(15), assert_http_response(stream))
                .await
                .expect("logprobs HTTP regression timed out");
        },
    )
    .await;
}

#[tokio::test]
#[serial]
async fn nonstreaming_top_zero_preserves_chosen_logprobs() {
    run_case(false).await;
}

#[tokio::test]
#[serial]
async fn streaming_top_zero_preserves_chosen_logprobs() {
    run_case(true).await;
}
