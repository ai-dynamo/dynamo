// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental WebSocket endpoint at `/v1/realtime` for bidirectional streaming input.
//!
//! Wire shape: client sends a sequence of `Message::Text` frames each containing a
//! JSON-encoded `NvCreateChatCompletionRequest`; server forwards each frame onto an
//! engine-bound stream and forwards engine response chunks back as `Message::Text`
//! frames each containing a JSON-encoded `NvCreateChatCompletionStreamResponse`.
//!
//! For now the engine is a process-scoped mock (`EchoBidirectionalEngine`) held in
//! a `OnceLock` so tests can install one without going through the `ModelManager`,
//! which has no bidirectional accessor yet. A future revision will replace the
//! static with a `ModelManager` lookup keyed on `model_name`.

use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use axum::{
    Router,
    extract::{
        State,
        ws::{CloseFrame, Message, Utf8Bytes, WebSocket, WebSocketUpgrade, close_code},
    },
    http::Method,
    response::Response,
    routing::get,
};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, RequestStream};
use dynamo_runtime::pipeline::Context;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Bound on the per-connection request queue. Picks backpressure over
/// unbounded growth so a fast client cannot drive memory exhaustion against
/// a slow engine.
const REQUEST_CHANNEL_CAPACITY: usize = 64;

use super::{RouteDoc, service_v2};
use crate::engines::EchoBidirectionalEngine;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

/// Process-scoped registry for the bidirectional engine. Populated by tests and
/// (in production) by whatever wires up the experimental endpoint. If unset when
/// a connection arrives, the handler closes with `INTERNAL_ERROR`.
///
/// **Placeholder.** Tracked in #9174 (2/N). The proper registration path is
/// through `ModelManager` keyed on `model_name`, parallel to chat / completions
/// / embeddings engines, but no bidirectional accessor exists on `ModelManager`
/// yet. When that lands, replace this static and the `install_*` helpers below
/// with `state.manager().get_realtime_engine(model_name)` lookups in `handle_socket`,
/// and remove the install-time API entirely.
static BIDIRECTIONAL_ENGINE: OnceLock<EchoBidirectionalEngine> = OnceLock::new();

/// Install the bidirectional engine to be used by `/v1/realtime`. Returns `Err` if an
/// engine is already installed (the static can only be set once per process).
/// See [`BIDIRECTIONAL_ENGINE`] for why this install-time API exists.
pub fn install_engine(engine: EchoBidirectionalEngine) -> Result<(), &'static str> {
    BIDIRECTIONAL_ENGINE
        .set(engine)
        .map_err(|_| "realtime bidirectional engine already installed")
}

/// Convenience installer for tests/dev: registers the echo mock engine.
pub fn install_echo_engine() -> Result<(), &'static str> {
    install_engine(EchoBidirectionalEngine {})
}

pub fn realtime_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let realtime_path = path.unwrap_or_else(|| "/v1/realtime".to_string());
    let docs = vec![RouteDoc::new(Method::GET, &realtime_path)];
    let router = Router::new()
        .route(&realtime_path, get(realtime_ws_handler))
        .with_state(state);
    (docs, router)
}

async fn realtime_ws_handler(
    State(state): State<Arc<service_v2::State>>,
    upgrade: WebSocketUpgrade,
) -> Response {
    upgrade.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, _state: Arc<service_v2::State>) {
    // TODO (#9175): read a session-init frame first so we can route /
    // look up the model before forwarding inference frames to the engine.
    let Some(engine) = BIDIRECTIONAL_ENGINE.get() else {
        tracing::error!("/v1/realtime connection rejected: bidirectional engine not installed");
        let _ = socket
            .send(close_message(
                close_code::ERROR,
                "bidirectional engine not installed",
            ))
            .await;
        return;
    };

    let (mut ws_tx, mut ws_rx) = socket.split();
    let (req_tx, req_rx) = mpsc::channel::<NvCreateChatCompletionRequest>(REQUEST_CHANNEL_CAPACITY);

    let request_stream = Box::pin(ReceiverStream::new(req_rx));
    let input = RequestStream::new(request_stream, Context::new(()).context());

    // Inbound writes a non-NORMAL close message here on protocol errors
    // before cancelling the engine; outbound takes it after the response
    // stream ends. Empty slot ⇒ NORMAL completion.
    let close_reason: Arc<Mutex<Option<Message>>> = Arc::new(Mutex::new(None));

    let mut response_stream = match engine.generate(input).await {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(%err, "/v1/realtime engine.generate() failed");
            let _ = ws_tx
                .send(close_message(
                    close_code::ERROR,
                    &format!("engine error: {err}"),
                ))
                .await;
            return;
        }
    };
    let resp_ctx = response_stream.context();

    // Outbound task: drain the engine response stream onto the WebSocket.
    let outbound_close_reason = close_reason.clone();
    let outbound = tokio::spawn(async move {
        while let Some(annotated) = response_stream.next().await {
            let frame_payload = match serde_json::to_string(&annotated) {
                Ok(s) => s,
                Err(err) => {
                    tracing::warn!(%err, "/v1/realtime serializing response chunk failed");
                    continue;
                }
            };
            if ws_tx
                .send(Message::Text(Utf8Bytes::from(frame_payload)))
                .await
                .is_err()
            {
                tracing::debug!("/v1/realtime client disconnected during response");
                break;
            }
        }
        // Pick the close message inbound left behind on protocol errors;
        // otherwise the engine ended naturally (or via client cancellation)
        // → NORMAL.
        let msg = outbound_close_reason
            .lock()
            .take()
            .unwrap_or_else(|| close_message(close_code::NORMAL, "stream complete"));
        let _ = ws_tx.send(msg).await;
        // Drive the sink to completion so the Close frame drains before the
        // transport is dropped — otherwise axum can tear down the TCP socket
        // mid-frame and the client sees EOF instead of an in-band Close. Bound
        // the wait so a half-broken peer can't park this task indefinitely.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ws_tx.close()).await;
    });

    // Inbound loop: parse client frames into request stream items.
    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                tracing::debug!(%err, "/v1/realtime inbound frame error; treating as disconnect");
                break;
            }
        };
        match msg {
            Message::Text(text) => {
                match serde_json::from_str::<NvCreateChatCompletionRequest>(text.as_str()) {
                    Ok(req) => {
                        if req_tx.send(req).await.is_err() {
                            tracing::debug!("/v1/realtime engine receiver dropped; ending inbound");
                            break;
                        }
                    }
                    Err(err) => {
                        tracing::warn!(%err, "/v1/realtime malformed JSON frame; closing");
                        *close_reason.lock() =
                            Some(close_message(close_code::INVALID, "malformed JSON frame"));
                        break;
                    }
                }
            }
            Message::Binary(_) => {
                tracing::warn!("/v1/realtime received binary frame; not supported in this slice");
                *close_reason.lock() = Some(close_message(
                    close_code::UNSUPPORTED,
                    "binary frames not supported",
                ));
                break;
            }
            Message::Close(_) => break,
            Message::Ping(_) | Message::Pong(_) => {} // axum handles ping replies
        }
    }

    // Inbound loop ended (client close, EOF, error, or unsupported frame).
    // Cancel any in-flight engine work, then drop the sender so the engine's
    // input stream completes; outbound picks up the close-reason left in the
    // shared slot (or NORMAL on natural completion).
    resp_ctx.stop_generating();
    drop(req_tx);

    // Wait for outbound to finish flushing.
    let _ = outbound.await;
}

fn close_message(code: u16, reason: &str) -> Message {
    Message::Close(Some(CloseFrame {
        code,
        reason: Utf8Bytes::from(reason.to_string()),
    }))
}
