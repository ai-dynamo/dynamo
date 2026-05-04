// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test for the experimental `/v1/asr` WebSocket endpoint (DIS-1858).
//!
//! Verifies the slice's acceptance criteria: a WebSocket client can connect,
//! send chat-completion JSON frames, and receive streamed echo response frames
//! end-to-end through the new endpoint and the mock bidirectional engine.

use std::time::Duration;

use dynamo_llm::http::service::{asr, service_v2::HttpService};
use dynamo_runtime::CancellationToken;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

#[path = "common/ports.rs"]
mod ports;
use ports::get_random_port;

/// Engine slot is process-global; ensure we install it at most once across all
/// tests in this binary. Subsequent calls are silent no-ops.
fn ensure_echo_engine_installed() {
    let _ = asr::install_echo_engine();
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
async fn asr_websocket_echoes_per_char_and_finishes_per_request() {
    // Keep echo delay short so the test is fast.
    unsafe {
        std::env::set_var("DYN_TOKEN_ECHO_DELAY_MS", "0");
    }
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/asr");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    let body = serde_json::json!({
        "model": "echo",
        "messages": [{ "role": "user", "content": "hi" }],
    });
    ws.send(Message::Text(body.to_string()))
        .await
        .expect("send");

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
        if stops >= 1 {
            // For a single inbound request, the engine emits chars then one Stop;
            // we can also wait for the server-side Close frame, but the assertion
            // below is sufficient.
            break;
        }
    }

    let _ = ws.close(None).await;
    token.cancel();
    let _ = handle.await;

    assert_eq!(text, "hi", "echoed text");
    assert_eq!(stops, 1, "expected exactly one finish_reason=stop");
}

#[tokio::test]
async fn asr_websocket_rejects_binary_frame() {
    unsafe {
        std::env::set_var("DYN_TOKEN_ECHO_DELAY_MS", "0");
    }
    ensure_echo_engine_installed();

    let port = get_random_port().await;
    let service = HttpService::builder().port(port).build().unwrap();
    let token = CancellationToken::new();
    let handle = service.spawn(token.clone()).await;
    wait_for_health(port).await;

    let url = format!("ws://127.0.0.1:{port}/v1/asr");
    let (mut ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    ws.send(Message::Binary(vec![0u8, 1, 2, 3]))
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
