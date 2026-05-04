// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental WebSocket endpoint at `/v1/asr` — first slice of the bidirectional
//! streaming-input feature (DIS-1858, parent design "Streaming Request Support").
//!
//! Wire shape: client sends a sequence of `Message::Text` frames each containing a
//! JSON-encoded `NvCreateChatCompletionRequest`; server forwards each frame onto an
//! engine-bound stream and forwards engine response chunks back as `Message::Text`
//! frames each containing a JSON-encoded `NvCreateChatCompletionStreamResponse`.
//!
//! For this slice the engine is a process-scoped mock (`EchoBidirectionalEngine`)
//! held in a `OnceLock` so tests can install one without going through the
//! `ModelManager` (which has no bidirectional accessor yet — see plan.md). A future
//! slice will replace the static with a `ModelManager` lookup keyed on `model_name`.

use std::sync::{Arc, OnceLock};

use axum::{
    Router,
    extract::{
        State,
        ws::{Message, Utf8Bytes, WebSocket, WebSocketUpgrade, close_code},
    },
    http::Method,
    response::Response,
    routing::get,
};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, RequestStream};
use dynamo_runtime::pipeline::Context;
use futures::{SinkExt, StreamExt};
use tokio::sync::{Mutex, mpsc::unbounded_channel};

use super::{RouteDoc, service_v2};
use crate::engines::EchoBidirectionalEngine;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};

/// Process-scoped registry for the bidirectional engine. Populated by tests and
/// (in production) by whatever wires up the experimental endpoint. If unset when
/// a connection arrives, the handler closes with `INTERNAL_ERROR`.
static BIDIRECTIONAL_ENGINE: OnceLock<EchoBidirectionalEngine> = OnceLock::new();

/// Install the bidirectional engine to be used by `/v1/asr`. Returns `Err` if an
/// engine is already installed (the static can only be set once per process).
pub fn install_engine(engine: EchoBidirectionalEngine) -> Result<(), &'static str> {
    BIDIRECTIONAL_ENGINE
        .set(engine)
        .map_err(|_| "asr bidirectional engine already installed")
}

/// Convenience installer for tests/dev: registers the echo mock engine.
pub fn install_echo_engine() -> Result<(), &'static str> {
    install_engine(EchoBidirectionalEngine {})
}

pub fn asr_router(state: Arc<service_v2::State>, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let asr_path = path.unwrap_or_else(|| "/v1/asr".to_string());
    let docs = vec![RouteDoc::new(Method::GET, &asr_path)];
    let router = Router::new()
        .route(&asr_path, get(asr_ws_handler))
        .with_state(state);
    (docs, router)
}

async fn asr_ws_handler(
    State(state): State<Arc<service_v2::State>>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, _state: Arc<service_v2::State>) {
    let Some(engine) = BIDIRECTIONAL_ENGINE.get() else {
        tracing::error!("/v1/asr connection rejected: bidirectional engine not installed");
        let _ = close_with(
            socket,
            close_code::ERROR,
            "bidirectional engine not installed",
        )
        .await;
        return;
    };

    let (mut ws_tx, mut ws_rx) = socket.split();
    let (req_tx, req_rx) = unbounded_channel::<NvCreateChatCompletionRequest>();

    // Wrap the receiver in `Arc<Mutex>` so the engine-bound stream can be `Send + Sync`,
    // which `RequestStream` requires (its inner `SyncDataStream` carries `+ Sync`).
    let rx_handle = Arc::new(Mutex::new(req_rx));
    let rx_clone = rx_handle.clone();
    let request_stream = Box::pin(async_stream::stream! {
        let mut rx = rx_clone.lock().await;
        while let Some(item) = rx.recv().await {
            yield item;
        }
    });

    let input = RequestStream::new(request_stream, Context::new(()).context());

    let mut response_stream = match engine.generate(input).await {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(%err, "/v1/asr engine.generate() failed");
            let mut sink = ws_tx;
            let _ = send_error(&mut sink, &err.to_string()).await;
            let _ = sink
                .send(Message::Close(Some(close_frame(
                    close_code::ERROR,
                    "engine error",
                ))))
                .await;
            return;
        }
    };
    let resp_ctx = response_stream.context();

    // Outbound task: drain the engine response stream onto the WebSocket.
    let outbound = tokio::spawn(async move {
        while let Some(annotated) = response_stream.next().await {
            let frame_payload = match serde_json::to_string(&annotated) {
                Ok(s) => s,
                Err(err) => {
                    tracing::warn!(%err, "/v1/asr serializing response chunk failed");
                    continue;
                }
            };
            if ws_tx
                .send(Message::Text(Utf8Bytes::from(frame_payload)))
                .await
                .is_err()
            {
                tracing::debug!("/v1/asr client disconnected during response");
                break;
            }
        }
        // Send a normal close once the engine finishes.
        let _ = ws_tx
            .send(Message::Close(Some(close_frame(
                close_code::NORMAL,
                "stream complete",
            ))))
            .await;
    });

    // Inbound loop: parse client frames into request stream items.
    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                tracing::debug!(%err, "/v1/asr inbound frame error; treating as disconnect");
                break;
            }
        };
        match msg {
            Message::Text(text) => {
                match serde_json::from_str::<NvCreateChatCompletionRequest>(text.as_str()) {
                    Ok(req) => {
                        if req_tx.send(req).is_err() {
                            tracing::debug!("/v1/asr engine receiver dropped; ending inbound");
                            break;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(%err, "/v1/asr malformed JSON frame; closing");
                        close_request_side(&req_tx);
                        resp_ctx.stop_generating();
                        break;
                    }
                }
            }
            Message::Binary(_) => {
                tracing::warn!("/v1/asr received binary frame; not supported in this slice");
                close_request_side(&req_tx);
                resp_ctx.stop_generating();
                break;
            }
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => {} // axum handles ping replies
        }
    }

    // Inbound loop ended (client close, error, or unsupported frame).
    // Drop the sender so the engine's input stream completes naturally.
    drop(req_tx);

    // Wait for outbound to finish flushing.
    let _ = outbound.await;
}

#[allow(unused)]
type FailableSink = futures::stream::SplitSink<WebSocket, Message>;

fn close_request_side(_tx: &tokio::sync::mpsc::UnboundedSender<NvCreateChatCompletionRequest>) {
    // sender is dropped by the caller via `drop(req_tx)` after this returns;
    // helper exists so call sites express intent ("stop accepting input").
}

fn close_frame(code: u16, reason: &str) -> axum::extract::ws::CloseFrame {
    axum::extract::ws::CloseFrame {
        code,
        reason: Utf8Bytes::from(reason.to_string()),
    }
}

async fn close_with(socket: WebSocket, code: u16, reason: &str) -> Result<(), axum::Error> {
    let mut socket = socket;
    socket
        .send(Message::Close(Some(close_frame(code, reason))))
        .await
}

async fn send_error<S>(sink: &mut S, message: &str) -> Result<(), axum::Error>
where
    S: SinkExt<Message, Error = axum::Error> + Unpin,
{
    let payload = serde_json::json!({
        "error": { "message": message, "code": "internal_error" }
    });
    sink.send(Message::Text(Utf8Bytes::from(payload.to_string())))
        .await
}

// `EchoBidirectionalEngine` must be `Send + Sync + 'static` to live in `OnceLock`.
// Compile-time assertion:
const _: fn() = || {
    fn _assert_send_sync<T: Send + Sync + 'static>() {}
    _assert_send_sync::<EchoBidirectionalEngine>();
};

#[allow(unused)]
fn _ensure_response_unused(_: NvCreateChatCompletionStreamResponse) {}
