// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the experimental `/v1/realtime` WebSocket endpoint.
//!
//! Verifies the slice's acceptance criteria: a WebSocket client can connect,
//! receive the spec-mandated `session.created` server frame, exchange OpenAI
//! Realtime client/server events with the mock bidirectional engine, and
//! cleanly terminate the session in both client- and server-initiated paths.

use std::time::Duration;

use dynamo_llm::http::service::{realtime, service_v2::HttpService};
use dynamo_runtime::CancellationToken;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

/// Engine slot is process-global; ensure we install the echo mock at most once
/// across all tests in this binary.
///
/// #9174 (2/N) will replace `OnceLock`-based engine registration with proper
/// `ModelManager`-keyed lookups; this helper goes away then.
fn ensure_echo_engine_installed() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = realtime::install_echo_engine();
    });
}

async fn wait_for_health(port: u16) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if reqwest::get(format!("http://127.0.0.1:{port}/health"))
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("frontend never became healthy on port {port}");
}

/// Read one Text frame off the socket and parse it as JSON, asserting the
/// `type` field matches `expected_type`. Returns the parsed value so callers
/// can drill into the payload.
async fn expect_text_event(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    expected_type: &str,
) -> Value {
    let frame = tokio::time::timeout(Duration::from_secs(2), ws.next())
        .await
        .expect("frame within 2s")
        .expect("stream not closed")
        .expect("no transport error");
    let Message::Text(text) = frame else {
        panic!("expected Text frame, got {frame:?}");
    };
    let v: Value = serde_json::from_str(&text).expect("response is valid JSON");
    assert_eq!(
        v.get("type").and_then(|t| t.as_str()),
        Some(expected_type),
        "unexpected event type in {v}"
    );
    v
}

/// On connect, the server emits `session.created` as the first wire frame —
/// per the OpenAI Realtime spec, before any client event arrives. This is the
/// merge-blocker the rest of the slice gates on.
#[tokio::test]
async fn realtime_websocket_emits_session_created_on_connect() {
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    let event = expect_text_event(&mut ws, "session.created").await;
    let event_id = event
        .get("event_id")
        .and_then(|s| s.as_str())
        .expect("event_id is a string");
    assert!(!event_id.is_empty(), "event_id should not be empty");
    assert!(
        event.pointer("/session/type").is_some(),
        "session payload should include the discriminator"
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

/// `session.update` round-trips through the engine: client sends the event,
/// server replies with `session.updated` carrying the same `Session` payload.
/// Demonstrates end-to-end realtime-event plumbing through the WebSocket.
#[tokio::test]
async fn realtime_websocket_session_update_echoes_session_updated() {
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Drain the on-connect session.created frame.
    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": "gpt-realtime" }
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send session.update");

    let event = expect_text_event(&mut ws, "session.updated").await;
    assert_eq!(
        event.pointer("/session/type").and_then(|s| s.as_str()),
        Some("realtime")
    );
    assert_eq!(
        event.pointer("/session/model").and_then(|s| s.as_str()),
        Some("gpt-realtime")
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

/// `input_audio_buffer.append` produces a spec-shaped response envelope on
/// the wire: `response.created` (`status: in_progress`) → one or more
/// `response.output_audio.delta` frames echoing the input audio →
/// `response.output_audio.done` → `response.done` (`status: completed`).
/// End-to-end check that streamed multi-frame engine output reaches the wire
/// in the right order with stable response_id across the turn.
#[tokio::test]
async fn realtime_websocket_audio_append_streams_response_envelope() {
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    let audio = "A".repeat(200);
    let body = serde_json::json!({
        "type": "input_audio_buffer.append",
        "audio": audio.clone(),
    });
    ws.send(Message::Text(body.to_string()))
        .await
        .expect("send append");

    let mut deltas = String::new();
    let mut response_id: Option<String> = None;
    let mut saw_audio_done = false;
    let mut response_done_status: Option<String> = None;
    let mut events_seen: Vec<String> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while response_done_status.is_none() && tokio::time::Instant::now() < deadline {
        let frame = tokio::time::timeout(Duration::from_secs(2), ws.next())
            .await
            .expect("frame within 2s")
            .expect("stream not closed")
            .expect("no transport error");
        let Message::Text(text) = frame else {
            panic!("expected Text frame, got {frame:?}");
        };
        let event: Value = serde_json::from_str(&text).expect("response is valid JSON");
        let event_type = event
            .get("type")
            .and_then(|t| t.as_str())
            .expect("event has type")
            .to_string();
        events_seen.push(event_type.clone());
        match event_type.as_str() {
            "response.created" => {
                let id = event
                    .pointer("/response/id")
                    .and_then(|s| s.as_str())
                    .expect("response.created carries response.id")
                    .to_string();
                response_id = Some(id);
            }
            "response.output_audio.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(|d| d.as_str())
                    .expect("delta is a string");
                deltas.push_str(delta);
            }
            "response.output_audio.done" => saw_audio_done = true,
            "response.done" => {
                response_done_status = event
                    .pointer("/response/status")
                    .and_then(|s| s.as_str())
                    .map(String::from);
            }
            other => panic!("unexpected event type {other:?} in audio echo stream: {event}"),
        }
    }

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;

    assert_eq!(
        events_seen.first().map(String::as_str),
        Some("response.created")
    );
    assert_eq!(
        events_seen.last().map(String::as_str),
        Some("response.done")
    );
    assert!(
        saw_audio_done,
        "engine should emit response.output_audio.done"
    );
    assert_eq!(response_done_status.as_deref(), Some("completed"));
    assert!(response_id.is_some());
    assert_eq!(
        deltas, audio,
        "concatenated deltas should reproduce the input audio"
    );
}

/// After a client-initiated close, the server should let the engine drain
/// (sender side dropped → `req_rx` returns None → engine response stream ends)
/// and emit its own Close frame as part of the cleanup. Covers the
/// "Send a normal close once the engine finishes" path in `realtime.rs`'s outbound
/// task, where the trigger is client disconnect rather than natural completion
/// (the echo test) or server-side rejection (the binary-frame test).
#[tokio::test]
async fn realtime_websocket_emits_close_after_client_close() {
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Drain the on-connect session.created frame.
    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime" }
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send");

    // Confirm the engine emitted at least one frame before we close, so we know
    // the round-trip path is live (not just the on-connect synthesis).
    expect_text_event(&mut ws, "session.updated").await;

    // Client-initiated close.
    ws.close(None).await.expect("client close");

    // Server should now clean up by sending an explicit Close frame back. The
    // outbound task drives the sink to completion (`ws_tx.close().await`)
    // after writing the Close frame, which ensures it fully drains before the
    // transport is dropped — so the client must observe `Message::Close`, not
    // a bare EOF. Drain any residual frames already in flight along the way.
    let mut got_close = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        let frame = tokio::time::timeout(Duration::from_secs(2), ws.next()).await;
        let Ok(maybe) = frame else { break };
        match maybe {
            Some(Ok(Message::Close(_))) => {
                got_close = true;
                break;
            }
            None => break, // EOF without an in-band Close — treated as a regression below
            _ => {}        // residual frame from the in-flight response — drain
        }
    }

    token.cancel();
    let _ = handle.await;

    assert!(
        got_close,
        "server should send an explicit Close frame after client-initiated close"
    );
}

#[tokio::test]
async fn realtime_websocket_rejects_binary_frame() {
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Drain the on-connect session.created frame so we don't false-pass on it.
    expect_text_event(&mut ws, "session.created").await;

    ws.send(Message::Binary(vec![0u8, 1, 2, 3].into()))
        .await
        .expect("send binary");

    let mut got_close = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while tokio::time::Instant::now() < deadline {
        let frame = tokio::time::timeout(Duration::from_secs(2), ws.next()).await;
        let Ok(maybe) = frame else { break };
        match maybe {
            Some(Ok(Message::Close(_))) => {
                got_close = true;
                break;
            }
            None => break,
            _ => {}
        }
    }

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;

    assert!(
        got_close,
        "server should close the connection on a binary frame"
    );
}
