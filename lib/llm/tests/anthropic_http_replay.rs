// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-HTTP integration coverage for the Anthropic Messages compatibility surface.

use std::collections::BTreeMap;
use std::time::Duration;

use dynamo_llm::http::service::metrics::{Endpoint, ErrorType, RequestType, Status};
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessageContent,
};
use dynamo_runtime::config::environment_names::llm::{
    DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
    DYN_HTTP_STREAM_MAX_DURATION_MS,
};
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
use scripted_chat_engine::Script;

const ENV: [(&str, Option<&str>); 2] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
];

fn user_text(
    request: &dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest,
) -> &str {
    match &request.inner.messages[..] {
        [ChatCompletionRequestMessage::User(user)] => match &user.content {
            ChatCompletionRequestUserMessageContent::Text(text) => text,
            other => panic!("expected text user content, got {other:?}"),
        },
        other => panic!("expected one translated user message, got {other:#?}"),
    }
}

async fn post_messages(svc: &HarnessService, body: &Value) -> reqwest::Response {
    svc.client
        .post(format!("{}/v1/messages", svc.base_url))
        .json(body)
        .send()
        .await
        .expect("POST /v1/messages failed")
}

fn tool(name: &str) -> Value {
    json!({
        "name": name,
        "description": "test tool",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    })
}

#[tokio::test]
#[serial]
async fn unary_text_baseline() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start([load_agent_fixture("text.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 64,
                "stream": false,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        insta::assert_json_snapshot!("anthropic_unary_text", canonicalize(body));

        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(user_text(&requests[0]), "ping");
        assert_eq!(requests[0].inner.max_completion_tokens, Some(64));
        assert_eq!(requests[0].inner.stream, Some(true));
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
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 64,
                "stream": true,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let raw = response.text().await.unwrap();
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        let events = parse_json_sse(&raw).await.unwrap();
        insta::assert_json_snapshot!(
            "anthropic_streaming_text",
            canonicalize(serde_json::to_value(events).unwrap())
        );

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn fragmented_tool_arguments_close_after_all_deltas() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start([load_agent_fixture("fragmented-tool.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();

        let start = events
            .iter()
            .find(|event| event.event == "content_block_start")
            .expect("missing tool block start");
        let block_id = start.data["content_block"]["id"].as_str().unwrap();
        assert!(
            block_id.starts_with("toolu_") && block_id.len() > "toolu_".len(),
            "streamed tool_use id must be Anthropic-native, got {block_id:?}"
        );
        assert_eq!(start.data["content_block"]["name"], "list_directory");

        let deltas: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                event.event == "content_block_delta"
                    && event.data["delta"]["type"] == "input_json_delta"
            })
            .map(|(index, event)| (index, event.data["delta"]["partial_json"].as_str().unwrap()))
            .collect();
        assert_eq!(
            deltas.iter().map(|(_, part)| *part).collect::<String>(),
            r#"{"path":"/tmp"}"#
        );
        let stop_positions: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.event == "content_block_stop")
            .map(|(index, _)| index)
            .collect();
        assert_eq!(stop_positions.len(), 1);
        assert!(stop_positions[0] > deltas.last().unwrap().0);
        assert_eq!(
            events
                .iter()
                .find(|event| event.event == "message_delta")
                .unwrap()
                .data["delta"]["stop_reason"],
            "tool_use"
        );

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn finish_signal_publishes_tool_block_before_usage_tail() {
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        let split_at = script
            .iter()
            .position(|chunk| chunk.inner.usage.is_some())
            .expect("fragmented-tool fixture has no usage chunk");
        let (svc, gate) = HarnessService::start_with_gated_tail(script, split_at).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let mut body = response.bytes_stream();
        let mut parser = IncrementalSseParser::default();
        let mut saw_tool_stop = false;
        let mut saw_message_delta = false;

        tokio::time::timeout(Duration::from_secs(2), async {
            while !saw_tool_stop {
                let bytes = body
                    .next()
                    .await
                    .expect("response ended before tool block completion")
                    .expect("failed to read response SSE bytes");
                for event in parser.push(&bytes).expect("failed to parse response SSE") {
                    saw_tool_stop |= event == "content_block_stop";
                    saw_message_delta |= event == "message_delta";
                }
            }
        })
        .await
        .expect("tool block completion did not arrive before the gated usage tail");

        assert!(!saw_message_delta);
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
                .filter(|event| event.event == "content_block_stop")
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "message_delta")
                .count(),
            1
        );

        svc.shutdown().await;
    })
    .await;
}

/// Vars for the wall-clock-deadline tests: base Anthropic env plus a short
/// `DYN_HTTP_STREAM_MAX_DURATION_MS` so the frontend deadline fires well before
/// any real timeout while the backend stream is gated (and never released).
const DEADLINE_ENV: [(&str, Option<&str>); 3] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (DYN_HTTP_STREAM_MAX_DURATION_MS, Some("300")),
];

/// When the wall-clock deadline expires mid-stream (backend still open), the
/// frontend must close the response as a spec-valid short turn:
/// `content_block_stop` for the open text block, `message_delta` with
/// `stop_reason:"max_tokens"`, then `message_stop` + `[DONE]` — and NOT an
/// `event: error`. The `text.sse` head (`role`, `"Pong."`) streams immediately;
/// the finish/usage tail is gated and never released, so only the deadline can
/// terminate the stream.
#[tokio::test]
#[serial]
async fn stream_deadline_truncates_open_text_block_with_max_tokens() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        // Split after the two content chunks (role + "Pong.") so the finish and
        // usage chunks are gated; the text block is left open.
        let split_at = 2;
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, split_at).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        // The deadline is 300ms; give it generous slack. Without the fix this
        // would hang until the test timeout.
        let raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("stream did not terminate at the wall-clock deadline")
            .expect("failed to read truncated SSE body");

        let events = parse_json_sse(&raw).await.unwrap();

        // Exactly one terminal message_delta, carrying max_tokens.
        let deltas: Vec<_> = events
            .iter()
            .filter(|event| event.event == "message_delta")
            .collect();
        assert_eq!(deltas.len(), 1, "expected one message_delta: {raw}");
        assert_eq!(deltas[0].data["delta"]["stop_reason"], "max_tokens");

        // A spec-valid terminal turn: message_stop + [DONE], no error frame.
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "message_stop")
                .count(),
            1
        );
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        assert!(
            !events.iter().any(|event| event.event == "error"),
            "truncation must not surface an error event: {raw}"
        );

        svc.shutdown().await;
    })
    .await;
}

/// Opt-in guarantee: with `DYN_HTTP_STREAM_MAX_DURATION_MS` unset, a gated stream
/// does NOT self-terminate — no `message_delta` arrives within a window that is
/// comfortably longer than the deadline used above. This proves the deadline is
/// what produced the terminal turn in the test above, not some other timer.
#[tokio::test]
#[serial]
async fn stream_without_deadline_does_not_self_truncate() {
    // Base ENV only: no DYN_HTTP_STREAM_MAX_DURATION_MS.
    temp_env::async_with_vars(ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let (svc, gate) = HarnessService::start_with_gated_tail(script, 2).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let mut body = response.bytes_stream();
        let mut parser = IncrementalSseParser::default();
        let mut saw_message_delta = false;
        // Read for well past the 300ms deadline used above; nothing should
        // finalize because no deadline is configured.
        let _ = tokio::time::timeout(Duration::from_millis(800), async {
            while let Some(bytes) = body.next().await {
                let bytes = bytes.expect("failed to read SSE bytes");
                for event in parser.push(&bytes).expect("failed to parse SSE") {
                    saw_message_delta |= event == "message_delta";
                }
            }
        })
        .await;
        assert!(
            !saw_message_delta,
            "stream self-truncated without a configured deadline"
        );

        gate.release();
        svc.shutdown().await;
    })
    .await;
}

/// A deadline that fires while a tool call's arguments are still mid-JSON must not
/// flush a malformed `tool_use` (which the client would replay as HTTP 400). The
/// `fragmented-tool.sse` head streams the tool id/name and the first arg fragment
/// (`{"path"` — incomplete JSON); the fragment that closes the JSON is gated. On
/// truncation the incomplete block is dropped: no `tool_use` content block is
/// emitted, only the spec-valid `max_tokens` terminal turn.
#[tokio::test]
#[serial]
async fn stream_deadline_drops_incomplete_tool_use() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        let script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        // Immediate: role, tool id/name, and the first arg fragment leaving args
        // at `{"path"` (incomplete JSON). Gated: the fragment that completes the
        // JSON, plus finish/usage. So at the deadline the buffered tool args do
        // not parse and the block must be dropped.
        let split_at = 3;
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, split_at).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("stream did not terminate at the wall-clock deadline")
            .expect("failed to read truncated SSE body");
        let events = parse_json_sse(&raw).await.unwrap();

        // No tool_use content block was flushed (the args JSON was incomplete).
        assert!(
            !events.iter().any(|event| {
                event.event == "content_block_start"
                    && event.data["content_block"]["type"] == "tool_use"
            }),
            "an incomplete tool_use must not be emitted on truncation: {raw}"
        );
        // Still a spec-valid truncated turn.
        let deltas: Vec<_> = events
            .iter()
            .filter(|event| event.event == "message_delta")
            .collect();
        assert_eq!(deltas.len(), 1, "expected one message_delta: {raw}");
        assert_eq!(deltas[0].data["delta"]["stop_reason"], "max_tokens");
        assert_eq!(raw.matches("data: [DONE]").count(), 1);

        svc.shutdown().await;
    })
    .await;
}

/// A deadline that fires before the backend has ever produced a single chunk must
/// still close the response as a spec-valid, non-empty turn: `message_start` (which
/// is emitted unconditionally before the backend is first polled) followed directly
/// by the terminal `message_delta{stop_reason:"max_tokens"}` + `message_stop`. No
/// content block was ever opened, so none should appear closed either. This is the
/// edge case the upstream contribution plan (docs/issues §10.5) calls out as
/// required in addition to the mid-block case: the finalizer must not assume at
/// least one block was ever started.
#[tokio::test]
#[serial]
async fn stream_deadline_before_any_backend_chunk_still_emits_valid_terminal_turn() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        // Gate everything: not even the role/text head chunk is released, so the
        // deadline must fire while the converter has emitted nothing but the
        // unconditional message_start.
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, 0).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("stream did not terminate at the wall-clock deadline")
            .expect("failed to read truncated SSE body");
        let events = parse_json_sse(&raw).await.unwrap();

        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "message_start")
                .count(),
            1,
            "message_start must still be emitted when the deadline fires before any backend chunk: {raw}"
        );
        assert!(
            !events
                .iter()
                .any(|event| event.event.starts_with("content_block")),
            "no content block was ever opened, so none should appear in the truncated turn: {raw}"
        );
        let deltas: Vec<_> = events
            .iter()
            .filter(|event| event.event == "message_delta")
            .collect();
        assert_eq!(deltas.len(), 1, "expected one message_delta: {raw}");
        assert_eq!(deltas[0].data["delta"]["stop_reason"], "max_tokens");
        assert_eq!(
            events
                .iter()
                .filter(|event| event.event == "message_stop")
                .count(),
            1
        );
        assert_eq!(raw.matches("data: [DONE]").count(), 1);
        assert!(
            !events.iter().any(|event| event.event == "error"),
            "truncation must not surface an error event: {raw}"
        );

        svc.shutdown().await;
    })
    .await;
}

/// Truncation must be observable through `stream_truncated_total`, not just through
/// the SSE body — otherwise a deployment cannot distinguish a proxy-deadline
/// truncation from a genuine `max_tokens` turn (docs/issues §7 "Observability nên
/// thêm").
#[tokio::test]
#[serial]
async fn stream_deadline_increments_truncation_metric() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, 2).await;
        assert_eq!(
            svc.metrics
                .get_stream_truncated_count(MODEL, Endpoint::AnthropicMessages),
            0
        );

        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let _raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("stream did not terminate at the wall-clock deadline")
            .expect("failed to read truncated SSE body");

        assert_eq!(
            svc.metrics
                .get_stream_truncated_count(MODEL, Endpoint::AnthropicMessages),
            1,
            "wall-clock truncation must increment stream_truncated_total"
        );

        svc.shutdown().await;
    })
    .await;
}

/// The whole point of truncating server-side is to stop the backend from wasting
/// GPU cycles decoding tokens the client can never receive (docs/issues TL;DR: "vẫn
/// decode — GPU cháy vô ích"). Assert `stop_generating()` actually reaches the
/// backend's engine context, not just that the client-visible SSE looks correct.
#[tokio::test]
#[serial]
async fn stream_deadline_calls_stop_generating_on_backend_context() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, 2).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let _raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("stream did not terminate at the wall-clock deadline")
            .expect("failed to read truncated SSE body");

        assert!(
            svc.engine.last_context_stopped().await,
            "wall-clock deadline must call stop_generating() on the backend context \
             so the worker stops producing tokens the client can never receive"
        );

        svc.shutdown().await;
    })
    .await;
}

/// A genuine client disconnect must still be recorded as `cancelled`, never folded
/// into the `truncated` path, even when a wall-clock deadline is configured and has
/// not yet expired. This pins the invariant the implementation comment calls out
/// directly: "cancelled" parks on `pending()` so the outer `monitor_for_disconnects`
/// records the request as cancelled, while "truncated" ends normally and is recorded
/// OK — the two must never be conflated (docs/issues §6.3(d)).
#[tokio::test]
#[serial]
async fn client_disconnect_is_not_misclassified_as_truncation_when_deadline_configured() {
    // A deadline generous enough that it cannot fire before this test's assertions
    // run, so any termination we observe must come from the disconnect path.
    const GENEROUS_DEADLINE_ENV: [(&str, Option<&str>); 3] = [
        (DYN_ENABLE_ANTHROPIC_API, Some("1")),
        (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
        (DYN_HTTP_STREAM_MAX_DURATION_MS, Some("60000")),
    ];
    temp_env::async_with_vars(GENEROUS_DEADLINE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, 2).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        // Read the immediately-available head chunk, then drop the response body to
        // simulate a client disconnect (same technique as the existing
        // client-disconnect coverage in tests/http-service.rs).
        let mut body = response.bytes_stream();
        let _ = body.next().await;
        drop(body);

        // Give the connection monitor time to detect the drop and record it.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if svc.metrics.get_request_counter(
                    MODEL,
                    &Endpoint::AnthropicMessages,
                    &RequestType::Stream,
                    &Status::Error,
                    &ErrorType::Cancelled,
                ) == 1
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("client disconnect was not recorded as cancelled");

        assert_eq!(
            svc.metrics
                .get_stream_truncated_count(MODEL, Endpoint::AnthropicMessages),
            0,
            "a client disconnect must never be recorded as a wall-clock truncation"
        );

        svc.shutdown().await;
    })
    .await;
}

/// Regression: the deadline must not relabel a turn the backend already finished.
///
/// `text.sse` chunk 2 carries `finish_reason:"stop"`, so by the time the gated
/// usage-only tail (chunk 3) is still in flight the converter already holds a real
/// terminal reason (`end_turn`). If the deadline fires in that window the handler
/// still takes the truncation path — but overwriting the recorded reason with
/// `max_tokens` tells the client a complete answer was cut short, and a client such
/// as Claude Code will continue a turn that already ended. Truncation bookkeeping
/// (metric, warn) is still correct here; only the client-visible label must be
/// preserved.
#[tokio::test]
#[serial]
async fn stream_deadline_preserves_finish_reason_already_reported_by_backend() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        // Head = role + content + finish_reason chunks; only the usage-only chunk
        // is gated, so the deadline fires *after* a genuine finish was recorded.
        let (svc, _gate) = HarnessService::start_with_gated_tail(script, 3).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("stream did not terminate at the wall-clock deadline")
            .expect("failed to read SSE body");

        let events = parse_json_sse(&raw).await.unwrap();
        let deltas: Vec<_> = events
            .iter()
            .filter(|event| event.event == "message_delta")
            .collect();
        assert_eq!(deltas.len(), 1, "expected one message_delta: {raw}");
        assert_eq!(
            deltas[0].data["delta"]["stop_reason"], "end_turn",
            "a turn the backend already finished must not be relabelled max_tokens: {raw}"
        );

        svc.shutdown().await;
    })
    .await;
}

/// Regression: a backend that is always ready must not starve the deadline.
///
/// The select loop is `biased`, so the backend-chunk arm is polled first by design
/// (an already-generated token must never be dropped in favour of a cancel). With a
/// backend whose `next()` is *always* immediately ready, that ordering means the
/// budget arm is never reached and `DYN_HTTP_STREAM_MAX_DURATION_MS` is never
/// enforced — exactly the fast-backend load the cap exists to bound. Multi-threaded
/// so the client side can still make progress while the server task spins.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_deadline_fires_against_a_continuously_ready_backend() {
    temp_env::async_with_vars(DEADLINE_ENV, async {
        // Reuse the fixture's first chunk (a role-only delta) as filler: it keeps
        // the select loop permanently fed without accumulating output, so this test
        // isolates scheduling starvation.
        let script = load_agent_fixture("text.sse").await.unwrap();
        let filler = script
            .into_iter()
            .next()
            .expect("text.sse must contain at least one chunk");
        let svc = HarnessService::start_endless(filler).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "messages": [{"role": "user", "content": "Ping"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        // The deadline is 300ms. Without enforcement ahead of chunk processing the
        // backend arm wins every poll and this never returns.
        let raw = tokio::time::timeout(Duration::from_secs(10), response.text())
            .await
            .expect("deadline was starved by a continuously-ready backend")
            .expect("failed to read truncated SSE body");

        let events = parse_json_sse(&raw).await.unwrap();
        let deltas: Vec<_> = events
            .iter()
            .filter(|event| event.event == "message_delta")
            .collect();
        assert_eq!(deltas.len(), 1, "expected one message_delta: {raw}");
        assert_eq!(deltas[0].data["delta"]["stop_reason"], "max_tokens");
        assert_eq!(
            svc.metrics
                .get_stream_truncated_count(MODEL, Endpoint::AnthropicMessages),
            1,
            "deadline truncation must be recorded"
        );

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn parallel_tools_preserve_identity_and_arguments() {
    temp_env::async_with_vars(ENV, async {
        let svc =
            HarnessService::start([load_agent_fixture("parallel-tools.sse").await.unwrap()]).await;
        let response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": true,
                "tools": [tool("read_file")],
                "messages": [{"role": "user", "content": "Read /a and /b"}]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let events = parse_json_sse(&response.text().await.unwrap())
            .await
            .unwrap();

        let starts: Vec<_> = events
            .iter()
            .filter(|event| event.event == "content_block_start")
            .map(|event| {
                (
                    event.data["index"].as_u64().unwrap(),
                    event.data["content_block"]["id"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                    event.data["content_block"]["name"]
                        .as_str()
                        .unwrap()
                        .to_string(),
                )
            })
            .collect();
        assert_eq!(
            starts
                .iter()
                .map(|(index, _, name)| (*index, name.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "read_file"), (1, "read_file")]
        );
        for (index, id, _) in &starts {
            assert!(
                id.starts_with("toolu_") && id.len() > "toolu_".len(),
                "tool_use id at block {index} must be Anthropic-native, got {id:?}"
            );
        }
        assert_ne!(
            starts[0].1, starts[1].1,
            "parallel tool calls must receive distinct generated ids"
        );

        let mut arguments = BTreeMap::<u64, String>::new();
        for event in &events {
            if event.event == "content_block_delta"
                && event.data["delta"]["type"] == "input_json_delta"
            {
                arguments
                    .entry(event.data["index"].as_u64().unwrap())
                    .or_default()
                    .push_str(event.data["delta"]["partial_json"].as_str().unwrap());
            }
        }
        assert_eq!(arguments.get(&0).unwrap(), r#"{"path":"/a"}"#);
        assert_eq!(arguments.get(&1).unwrap(), r#"{"path":"/b"}"#);

        let mut open_block = None;
        for event in &events {
            match event.event.as_str() {
                "content_block_start" if event.data["content_block"]["type"] == "tool_use" => {
                    assert!(open_block.is_none(), "tool blocks must not overlap");
                    open_block = event.data["index"].as_u64();
                }
                "content_block_delta" if event.data["delta"]["type"] == "input_json_delta" => {
                    assert_eq!(event.data["index"].as_u64(), open_block);
                }
                "content_block_stop" => {
                    assert_eq!(event.data["index"].as_u64(), open_block);
                    open_block = None;
                }
                _ => {}
            }
        }
        assert!(open_block.is_none());

        let stops: Vec<_> = events
            .iter()
            .filter(|event| event.event == "content_block_stop")
            .map(|event| event.data["index"].as_u64().unwrap())
            .collect();
        assert_eq!(stops, vec![0, 1]);

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn tool_result_round_trip_reaches_the_chat_engine() {
    temp_env::async_with_vars(ENV, async {
        let first_script = load_agent_fixture("thinking-tool.sse").await.unwrap();
        let second_script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start([first_script, second_script]).await;

        let first_response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 128,
                "stream": false,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "tools": [tool("list_directory")],
                "messages": [{"role": "user", "content": "List /tmp"}]
            }),
        )
        .await;
        assert_eq!(first_response.status(), reqwest::StatusCode::OK);
        let first_body: Value = first_response.json().await.unwrap();
        let prior_content = first_body["content"].clone();
        let prior_blocks = prior_content.as_array().expect("content must be an array");
        assert!(prior_blocks.iter().any(|block| block["type"] == "thinking"));

        let tool_use = prior_blocks
            .iter()
            .find(|block| block["type"] == "tool_use")
            .expect("missing tool_use block");
        let generated_id = tool_use["id"].as_str().unwrap().to_string();
        assert!(
            generated_id.starts_with("toolu_") && generated_id.len() > "toolu_".len(),
            "non-streamed tool_use id must be Anthropic-native, got {generated_id:?}"
        );
        assert_eq!(tool_use["name"], "list_directory");
        assert_eq!(tool_use["input"], json!({"path": "/tmp"}));

        let second_response = post_messages(
            &svc,
            &json!({
                "model": MODEL,
                "max_tokens": 64,
                "stream": false,
                "tools": [tool("list_directory")],
                "messages": [
                    {"role": "user", "content": "List /tmp"},
                    {"role": "assistant", "content": prior_content},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": generated_id,
                        "content": "a.txt"
                    }]}
                ]
            }),
        )
        .await;
        assert_eq!(second_response.status(), reqwest::StatusCode::OK);
        let second_body: Value = second_response.json().await.unwrap();
        assert_eq!(second_body["content"][0]["text"], "Pong.");

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
                assert!(matches!(
                    assistant.content.as_ref(),
                    Some(ChatCompletionRequestAssistantMessageContent::Text(text))
                        if text == "I will list it."
                ));
                assert_eq!(
                    assistant
                        .reasoning_content
                        .as_ref()
                        .expect("thinking must reach the chat request")
                        .to_flat_string(),
                    "I should inspect the directory."
                );
                let calls = assistant.tool_calls.as_deref().expect("tool calls missing");
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, generated_id);
                assert_eq!(calls[0].function.name, "list_directory");
                assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp"}"#);
                assert_eq!(tool_result.tool_call_id, generated_id);
                assert!(matches!(
                    &tool_result.content,
                    ChatCompletionRequestToolMessageContent::Text(text) if text == "a.txt"
                ));
            }
            other => panic!("unexpected translated round-trip messages: {other:#?}"),
        }

        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn count_tokens_returns_exact_estimate_without_calling_engine() {
    temp_env::async_with_vars(ENV, async {
        let svc = HarnessService::start(Vec::<Script>::new()).await;
        let response = svc
            .client
            .post(format!("{}/v1/messages/count_tokens", svc.base_url))
            .json(&json!({
                "model": MODEL,
                "system": "You are helpful.",
                "messages": [{
                    "role": "user",
                    "content": "Hello, world! This is a test message."
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            response.json::<Value>().await.unwrap(),
            json!({"input_tokens": 19})
        );
        assert!(svc.engine.take_requests().await.is_empty());

        svc.shutdown().await;
    })
    .await;
}
