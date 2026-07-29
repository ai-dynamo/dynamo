// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the experimental `/v1/realtime` WebSocket endpoint.
//!
//! Verifies the slice's acceptance criteria: a WebSocket client can connect,
//! receive the spec-mandated `session.created` server frame, exchange OpenAI
//! Realtime client/server events with the mock bidirectional engine, and
//! cleanly terminate the session in both client- and server-initiated paths.

use std::time::Duration;

use std::sync::Arc;

use anyhow::Result;
use dynamo_llm::endpoint_type::EndpointType;
use dynamo_llm::engines::EchoBidirectionalEngine;
use dynamo_llm::http::service::service_v2::HttpService;
use dynamo_runtime::CancellationToken;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

#[path = "common/ports.rs"]
mod ports;
use ports::bind_random_port;

/// Default model name used by tests; the fixture's first `session.update`
/// frame must carry this in `session.model` so the handler's
/// `get_realtime_engine` lookup hits the registered echo engine.
const ECHO_MODEL: &str = "echo";

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

fn register_echo(service: &HttpService) {
    service
        .model_manager()
        .add_realtime_model(ECHO_MODEL, "0", Arc::new(EchoBidirectionalEngine))
        .expect("register echo realtime engine");
}

async fn spawn_test_service(
    realtime_enabled: bool,
    register_echo_engine: bool,
) -> (u16, CancellationToken, JoinHandle<Result<()>>) {
    let (listener, port) = bind_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    if realtime_enabled {
        service.enable_model_endpoint(EndpointType::Realtime, true);
    }
    if register_echo_engine {
        register_echo(&service);
    }
    let token = CancellationToken::new();
    let handle = service.spawn_with_listener(token.clone(), listener).await;
    wait_for_health(port).await;
    (port, token, handle)
}

#[tokio::test]
async fn realtime_websocket_route_gated_by_endpoint_flag() {
    let (port, token, handle) = spawn_test_service(false, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let result = tokio_tungstenite::connect_async(&url).await;
    assert!(
        result.is_err(),
        "/v1/realtime upgrade must fail when EndpointType::Realtime is disabled"
    );

    token.cancel();
    let _ = handle.await;
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
/// per the OpenAI Realtime spec, before any client event arrives.
#[tokio::test]
async fn realtime_websocket_emits_session_created_on_connect() {
    let (port, token, handle) = spawn_test_service(true, true).await;

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
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Drain the on-connect session.created frame.
    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": ECHO_MODEL }
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
        Some(ECHO_MODEL)
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
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    // Initialize the session before sending audio — handler requires
    // session.update first to pick the engine.
    let session_update = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": ECHO_MODEL }
    });
    ws.send(Message::Text(session_update.to_string().into()))
        .await
        .expect("send session.update");
    expect_text_event(&mut ws, "session.updated").await;

    let audio = "A".repeat(200);
    let body = serde_json::json!({
        "type": "input_audio_buffer.append",
        "audio": audio.clone(),
    });
    ws.send(Message::Text(body.to_string().into()))
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
                let id = event
                    .pointer("/response_id")
                    .and_then(|s| s.as_str())
                    .expect("delta carries response_id");
                assert_eq!(
                    Some(id),
                    response_id.as_deref(),
                    "delta response_id should match response.created"
                );
            }
            "response.output_audio.done" => {
                saw_audio_done = true;
                let id = event
                    .pointer("/response_id")
                    .and_then(|s| s.as_str())
                    .expect("audio.done carries response_id");
                assert_eq!(
                    Some(id),
                    response_id.as_deref(),
                    "audio.done response_id should match response.created"
                );
            }
            "response.done" => {
                response_done_status = event
                    .pointer("/response/status")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                let id = event
                    .pointer("/response/id")
                    .and_then(|s| s.as_str())
                    .expect("response.done carries response.id");
                assert_eq!(
                    Some(id),
                    response_id.as_deref(),
                    "response.done id should match response.created"
                );
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
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Drain the on-connect session.created frame.
    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": ECHO_MODEL }
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

/// Wait for an `error` event, then assert no `Close` frame arrives within a
/// short bounded window. The lenient engine-selection loop emits error events
/// for intermediate failures (wrong event type, missing model, unknown model,
/// binary frames) without closing the connection so a well-behaved client can
/// recover by sending another `session.update`.
async fn expect_error_event_no_close(
    ws: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Value {
    let event = expect_text_event(ws, "error").await;
    let probe = tokio::time::timeout(Duration::from_millis(500), ws.next()).await;
    match probe {
        Ok(Some(Ok(Message::Close(_)))) => {
            panic!("server should not close on an intermediate error during engine selection")
        }
        Ok(None) => {
            panic!(
                "server should not end the stream on an intermediate error during engine selection"
            )
        }
        _ => {} // timeout (no close) or non-close frame → connection kept open
    }
    event
}

#[tokio::test]
async fn realtime_websocket_binary_frame_during_selection_emits_error() {
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    // Drain the on-connect session.created frame so we don't false-pass on it.
    expect_text_event(&mut ws, "session.created").await;

    ws.send(Message::Binary(vec![0u8, 1, 2, 3].into()))
        .await
        .expect("send binary");

    let event = expect_error_event_no_close(&mut ws).await;
    assert_eq!(
        event.pointer("/error/code").and_then(|s| s.as_str()),
        Some("invalid_request"),
        "error event should report invalid_request for binary frame: {event}"
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn realtime_websocket_unknown_model_emits_error() {
    let (port, token, handle) = spawn_test_service(true, false).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": "does-not-exist" }
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send");

    let event = expect_error_event_no_close(&mut ws).await;
    assert_eq!(
        event.pointer("/error/code").and_then(|s| s.as_str()),
        Some("model_not_found"),
        "error event should report model_not_found: {event}"
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn realtime_websocket_non_session_update_first_frame_emits_error() {
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "input_audio_buffer.append",
        "audio": "AAAA",
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send");

    let event = expect_error_event_no_close(&mut ws).await;
    assert_eq!(
        event.pointer("/error/code").and_then(|s| s.as_str()),
        Some("invalid_request"),
        "error event should report invalid_request: {event}"
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

#[tokio::test]
async fn realtime_websocket_session_update_missing_model_emits_error() {
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    let body = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime" }
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send");

    let event = expect_error_event_no_close(&mut ws).await;
    assert_eq!(
        event.pointer("/error/code").and_then(|s| s.as_str()),
        Some("invalid_request"),
        "error event should report invalid_request when model is absent: {event}"
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

/// `session.update` with `session.type = "transcription"` is the OpenAI Realtime
/// transcription mode, which this slice doesn't yet wire up to an engine.
/// Surface a specific `unsupported_session_type` error instead of letting the
/// `Session::RealtimeTranscriptionSession` arm fall through to the generic
/// "session.model required" path (which would be misleading — the issue isn't
/// a missing field, it's an unsupported session subtype). The connection stays
/// open so the client can recover by sending a `realtime` session.
#[tokio::test]
async fn realtime_websocket_transcription_session_emits_unsupported_error() {
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    // Minimum valid `RealtimeTranscriptionSession` JSON: requires
    // `audio.input.format` (unit-variant `audio/pcmu` is shortest) and
    // `audio.input.turn_detection`. `server_vad` carries three required
    // numeric fields. The handler rejects the session subtype before any of
    // these values matter — they exist only to get serde past
    // deserialization.
    let body = serde_json::json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": { "type": "audio/pcmu" },
                    "turn_detection": {
                        "type": "server_vad",
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "threshold": 0.5
                    }
                }
            }
        }
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send");

    let event = expect_error_event_no_close(&mut ws).await;
    assert_eq!(
        event.pointer("/error/code").and_then(|s| s.as_str()),
        Some("unsupported_session_type"),
        "error event should report unsupported_session_type for transcription: {event}"
    );

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}

/// After an intermediate error during engine selection, the loop stays open
/// and a follow-up `session.update` with a registered model selects the engine
/// normally. Demonstrates the recovery property the lenient loop is designed
/// for.
#[tokio::test]
async fn realtime_websocket_recovers_after_unknown_model() {
    let (port, token, handle) = spawn_test_service(true, true).await;

    let url = format!("ws://127.0.0.1:{port}/v1/realtime");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    expect_text_event(&mut ws, "session.created").await;

    // First attempt: bogus model → error event, connection stays open.
    let bad = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": "does-not-exist" }
    });
    ws.send(Message::Text(bad.to_string().into()))
        .await
        .expect("send bad");
    let err = expect_error_event_no_close(&mut ws).await;
    assert_eq!(
        err.pointer("/error/code").and_then(|s| s.as_str()),
        Some("model_not_found")
    );

    // Recovery: valid session.update with the registered echo model. The
    // engine round-trips it back as session.updated.
    let good = serde_json::json!({
        "type": "session.update",
        "session": { "type": "realtime", "model": ECHO_MODEL }
    });
    ws.send(Message::Text(good.to_string().into()))
        .await
        .expect("send good");
    expect_text_event(&mut ws, "session.updated").await;

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;
}
