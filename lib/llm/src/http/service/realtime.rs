// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental WebSocket endpoint at `/v1/realtime` for the OpenAI Realtime API.
//!
//! Wire shape: client sends a sequence of `Message::Text` frames each containing a
//! JSON-encoded [`RealtimeClientEvent`]; server forwards each frame onto an
//! engine-bound stream and forwards engine [`RealtimeServerEvent`] chunks back as
//! `Message::Text` frames. Per the OpenAI Realtime spec, audio is base64-encoded
//! inside the JSON envelope (`input_audio_buffer.append`); binary WebSocket frames
//! are rejected.
//!
//! On connect the handler synthesizes a `session.created` server event before any
//! engine event flows — the spec requires it to be the first server event on the
//! wire. The handler then waits for the first client frame; it must be a
//! `session.update`, and its `session.model` field selects the engine via
//! [`ModelManager::get_realtime_engine`]. The selected engine handles everything
//! subsequent (including `session.updated` echoes, audio-buffer state, and
//! response generation). The first `session.update` is forwarded onto the
//! engine's input stream verbatim — the handler used it only to pick the
//! engine, the rest of the carried session config is the engine's to apply.

use std::sync::Arc;
use std::time::Duration;

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
use dynamo_runtime::engine::{AsyncEngineContextProvider, RequestStream};
use dynamo_runtime::pipeline::Context;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Bound on the per-connection request queue. Picks backpressure over
/// unbounded growth so a fast client cannot drive memory exhaustion against
/// a slow engine.
const REQUEST_CHANNEL_CAPACITY: usize = 64;

/// Bound on the time the outbound task waits for the WebSocket sink to
/// drain a final Close frame before tearing down the transport. Keeps a
/// half-broken peer from parking the connection indefinitely.
const CLOSE_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

use super::{RouteDoc, service_v2};
use crate::discovery::ModelManagerError;
use dynamo_protocols::types::realtime::{
    EventType, RealtimeAPIError, RealtimeClientEvent, RealtimeServerEvent,
    RealtimeServerEventError, RealtimeServerEventSessionCreated, Session,
};
use uuid::Uuid;

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

async fn handle_socket(mut socket: WebSocket, state: Arc<service_v2::State>) {
    // OpenAI Realtime spec requires `session.created` to be the first server
    // frame on the wire, before any client event arrives. The handler synthesizes
    // it here so the connection handshake works regardless of which engine is
    // selected later.
    let session_created = RealtimeServerEvent::SessionCreated(RealtimeServerEventSessionCreated {
        event_id: format!("event_{}", Uuid::new_v4()),
        session: Session::RealtimeSession(Box::default()),
    });
    let session_created_payload = match serde_json::to_string(&session_created) {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(%err, "/v1/realtime serializing session.created failed");
            let _ = socket
                .send(close_message(
                    close_code::ERROR,
                    "internal error preparing session.created",
                ))
                .await;
            return;
        }
    };
    if let Err(err) = socket
        .send(Message::Text(Utf8Bytes::from(session_created_payload)))
        .await
    {
        tracing::debug!(%err, "/v1/realtime client disconnected before session.created");
        return;
    }

    let (mut ws_tx, mut ws_rx) = socket.split();

    let session_update = match expect_session_update(&mut ws_rx).await {
        Ok(req) => req,
        Err(close) => {
            if let Some((code, reason)) = close {
                send_error_event(
                    &mut ws_tx,
                    "invalid_request",
                    &reason,
                    Some("session.update"),
                )
                .await;
                let _ = ws_tx.send(close_message(code, &reason)).await;
                let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
            }
            return;
        }
    };

    let model = match &session_update.session {
        Session::RealtimeSession(s) => s.model.as_deref().filter(|m| !m.is_empty()),
        Session::RealtimeTranscriptionSession(_) => None,
    };
    let Some(model_name) = model else {
        send_error_event(
            &mut ws_tx,
            "invalid_request",
            "session.model required",
            Some("session.model"),
        )
        .await;
        let _ = ws_tx
            .send(close_message(close_code::POLICY, "session.model required"))
            .await;
        let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
        return;
    };

    let engine = match state.manager().get_realtime_engine(model_name) {
        Ok(engine) => engine,
        Err(ModelManagerError::ModelNotFound(_)) => {
            send_error_event(
                &mut ws_tx,
                "model_not_found",
                &format!("unknown model: {model_name}"),
                Some("session.model"),
            )
            .await;
            let _ = ws_tx
                .send(close_message(close_code::POLICY, "unknown model"))
                .await;
            let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
            return;
        }
        Err(ModelManagerError::ModelUnavailable(_)) => {
            send_error_event(
                &mut ws_tx,
                "model_unavailable",
                &format!("model unavailable: {model_name}"),
                Some("session.model"),
            )
            .await;
            let _ = ws_tx
                .send(close_message(close_code::AGAIN, "model unavailable"))
                .await;
            let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
            return;
        }
        Err(err) => {
            tracing::error!(%err, "/v1/realtime engine lookup failed");
            send_error_event(
                &mut ws_tx,
                "server_error",
                &err.to_string(),
                Some("session.model"),
            )
            .await;
            let _ = ws_tx
                .send(close_message(close_code::ERROR, "engine lookup failed"))
                .await;
            let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
            return;
        }
    };

    let (req_tx, req_rx) = mpsc::channel::<RealtimeClientEvent>(REQUEST_CHANNEL_CAPACITY);

    // Forward the session.update verbatim — it carries the engine's
    // generation config (instructions, voice, audio formats, turn-detection,
    // max_output_tokens, tools, output_modalities). The handler only used
    // it to pick the engine; the rest is the engine's to apply.
    if req_tx
        .send(RealtimeClientEvent::SessionUpdate(session_update))
        .await
        .is_err()
    {
        tracing::debug!("/v1/realtime engine receiver dropped before session.update delivered");
        return;
    }

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
    // Peels off the Dynamo-side `Annotated` envelope so clients receive bare
    // `RealtimeServerEvent` frames as the OpenAI Realtime spec requires. Engine
    // errors surfaced via `Annotated::error` are mapped to a synthesized
    // `RealtimeServerEvent::Error` so they remain visible on the wire.
    let outbound_close_reason = close_reason.clone();
    let outbound = tokio::spawn(async move {
        while let Some(annotated) = response_stream.next().await {
            let event = if let Some(event) = annotated.data {
                event
            } else if let Some(err) = annotated.error {
                RealtimeServerEvent::Error(RealtimeServerEventError {
                    event_id: format!("event_{}", Uuid::new_v4()),
                    error: RealtimeAPIError {
                        r#type: "server_error".to_string(),
                        code: None,
                        message: err.to_string(),
                        param: None,
                        event_id: None,
                    },
                })
            } else {
                continue;
            };
            let frame_payload = match serde_json::to_string(&event) {
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
        let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
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
                match serde_json::from_str::<RealtimeClientEvent>(text.as_str()) {
                    Ok(event) => {
                        if req_tx.send(event).await.is_err() {
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

/// Drain the inbound socket until a `session.update` arrives; that's the
/// only event the handler accepts before engine selection. Returns
/// `Err(Some((code, reason)))` for protocol violations the handler should
/// signal back to the client, or `Err(None)` for a silent client-initiated
/// close / EOF (no Close frame to send).
async fn expect_session_update<S>(
    ws_rx: &mut S,
) -> Result<
    dynamo_protocols::types::realtime::RealtimeClientEventSessionUpdate,
    Option<(u16, String)>,
>
where
    S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
{
    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                tracing::debug!(%err, "/v1/realtime inbound error before session.update");
                return Err(None);
            }
        };
        match msg {
            Message::Text(text) => match serde_json::from_str::<RealtimeClientEvent>(text.as_str())
            {
                Ok(RealtimeClientEvent::SessionUpdate(req)) => return Ok(req),
                Ok(other) => {
                    tracing::warn!(
                        event = other.event_type(),
                        "/v1/realtime first frame must be session.update"
                    );
                    return Err(Some((
                        close_code::INVALID,
                        "expected session.update as first client event".to_string(),
                    )));
                }
                Err(err) => {
                    tracing::warn!(%err, "/v1/realtime malformed JSON before session.update");
                    return Err(Some((
                        close_code::INVALID,
                        "malformed JSON frame".to_string(),
                    )));
                }
            },
            Message::Binary(_) => {
                tracing::warn!("/v1/realtime binary frame before session.update; rejecting");
                return Err(Some((
                    close_code::UNSUPPORTED,
                    "binary frames not supported".to_string(),
                )));
            }
            Message::Close(_) => return Err(None),
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
    Err(None)
}

async fn send_error_event<S>(ws_tx: &mut S, code: &str, message: &str, param: Option<&str>)
where
    S: futures::Sink<Message, Error = axum::Error> + Unpin,
{
    let event = RealtimeServerEvent::Error(RealtimeServerEventError {
        event_id: format!("event_{}", Uuid::new_v4()),
        error: RealtimeAPIError {
            r#type: "invalid_request_error".to_string(),
            code: Some(code.to_string()),
            message: message.to_string(),
            param: param.map(|s| s.to_string()),
            event_id: None,
        },
    });
    let payload = match serde_json::to_string(&event) {
        Ok(s) => s,
        Err(err) => {
            tracing::warn!(%err, "/v1/realtime serializing error event failed");
            return;
        }
    };
    let _ = ws_tx.send(Message::Text(Utf8Bytes::from(payload))).await;
}
