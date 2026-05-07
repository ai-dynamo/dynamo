// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for the experimental `/v1/realtime` WebSocket endpoint.
//!
//! Verifies the slice's acceptance criteria: a WebSocket client can connect,
//! send chat-completion JSON frames, and receive streamed echo response frames
//! end-to-end through the new endpoint and the mock bidirectional engine.

use std::time::Duration;

use dynamo_llm::http::service::{realtime, service_v2::HttpService};
use dynamo_runtime::CancellationToken;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

/// Engine slot is process-global; ensure we install it at most once across all
/// tests in this binary, with `DYN_TOKEN_ECHO_DELAY_MS` pinned before the
/// engine's `LazyLock` captures it. Wrapped in `Once::call_once` so concurrent
/// `#[tokio::test]`s synchronize on the single env mutation rather than racing
/// against `set_var`'s safety preconditions.
///
/// #9174 (2/N) will replace `OnceLock`-based engine registration with proper
/// `ModelManager`-keyed lookups; this helper goes away then.
fn ensure_echo_engine_installed() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // SAFETY: runs at most once, before any worker thread reads the env;
        // the engine's `LazyLock` reads it at most once per process.
        unsafe {
            std::env::set_var("DYN_TOKEN_ECHO_DELAY_MS", "0");
        }
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

#[tokio::test]
async fn realtime_websocket_echoes_per_char_and_finishes_per_request() {
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

    // Send two requests back-to-back. The engine processes them sequentially
    // (each one fully echoed and terminated with `FinishReason::Stop` before
    // the next is dequeued), so the response stream is "hi" + Stop + "ok" + Stop.
    let body1 = serde_json::json!({
        "model": "echo",
        "messages": [{ "role": "user", "content": "hi" }],
    });
    ws.send(Message::Text(body1.to_string().into()))
        .await
        .expect("send first request");
    let body2 = serde_json::json!({
        "model": "echo",
        "messages": [{ "role": "user", "content": "ok" }],
    });
    ws.send(Message::Text(body2.to_string().into()))
        .await
        .expect("send second request");

    // Read until we see two finish_reason="stop" or a normal close.
    let mut text = String::new();
    let mut stops = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        let frame = tokio::time::timeout(Duration::from_secs(2), ws.next()).await;
        let Ok(Some(Ok(msg))) = frame else { break };
        match msg {
            Message::Text(t) => {
                // Server frames are JSON-serialized `Annotated<NvCreateChatCompletionStreamResponse>`,
                // so the response payload lives under `.data`.
                let v: Value = serde_json::from_str(&t).expect("response is valid JSON");
                let choices = v
                    .pointer("/data/choices")
                    .and_then(|c| c.as_array())
                    .cloned()
                    .unwrap_or_default();
                for choice in choices {
                    if let Some(content) = choice
                        .get("delta")
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        text.push_str(content);
                    }
                    if choice
                        .get("finish_reason")
                        .and_then(|f| f.as_str())
                        .map(|s| s == "stop")
                        .unwrap_or(false)
                    {
                        stops += 1;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
        if stops >= 2 {
            break;
        }
    }

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;

    assert_eq!(text, "hiok", "echoed text from both requests");
    assert_eq!(stops, 2, "expected one finish_reason=stop per request");
}

/// After a client-initiated close, the server should let the engine drain
/// (sender side dropped → `req_rx` returns None → engine response stream ends)
/// and emit its own Close frame as part of the cleanup. Covers the
/// "Send a normal close once the engine finishes" path in `realtime.rs`'s outbound
/// task, where the trigger is client disconnect rather than natural completion
/// (the other test) or server-side rejection (the binary-frame test).
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

    let body = serde_json::json!({
        "model": "echo",
        "messages": [{ "role": "user", "content": "hi" }],
    });
    ws.send(Message::Text(body.to_string().into()))
        .await
        .expect("send");

    // Read one Text frame to confirm the engine started emitting before we
    // close. We don't need to time the close to the middle of an emission —
    // the cleanup path under test triggers the same way regardless of whether
    // the engine is mid-emission or already done.
    let first = tokio::time::timeout(Duration::from_secs(2), ws.next()).await;
    assert!(
        matches!(first, Ok(Some(Ok(Message::Text(_))))),
        "expected at least one delta from the engine before closing, got {first:?}"
    );

    // Client-initiated close.
    ws.close(None).await.expect("client close");

    // Server should now clean up by sending an explicit Close frame back. The
    // outbound task drives the sink to completion (`ws_tx.close().await`)
    // after writing the Close frame, which ensures it fully drains before the
    // transport is dropped — so the client must observe `Message::Close`, not
    // a bare EOF. Drain any residual delta frames already in flight along the
    // way.
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
            _ => {}        // residual Text frame from the in-flight response — drain
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
