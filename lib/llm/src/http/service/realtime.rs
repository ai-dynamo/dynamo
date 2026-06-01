// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebSocket endpoint at `/v1/realtime` for the OpenAI Realtime API.
//!
//! Wire shape: client sends a sequence of `Message::Text` frames each containing a
//! JSON-encoded [`RealtimeClientEvent`]; server forwards each frame onto an
//! engine-bound stream and forwards engine [`RealtimeServerEvent`] chunks back as
//! `Message::Text` frames. Per the OpenAI Realtime spec, audio is base64-encoded
//! inside the JSON envelope (`input_audio_buffer.append`); binary WebSocket frames
//! are rejected.
//!
//! Workflow:
//!
//! On connect:
//! - The handler sends a `session.created` server event before any
//!   engine events flow.
//! - The handler loops over inbound client frames in
//!   [`select_engine`] until a `session.update` arrives with a usable
//!   `session.model` and [`ModelManager::get_realtime_engine`] returns Ok.
//! - Non-conforming frames (wrong event type, missing model, unknown / unavailable
//!   model, malformed JSON, binary frames) emit a spec-shaped
//!   `RealtimeServerEvent::Error` while the loop continues so a well-behaved client
//!   can recover.
//!
//! On selected engine:
//! - The handler forwards all frames including `session.update` onto the engine's input stream.
//! - The handler drains the engine's response stream onto the WebSocket.
//! - WebSocket stream close procedure is encapsulated in the `ScopedWsWriter` wrapper.

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
use futures::{SinkExt, StreamExt, stream::SplitSink};
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
use crate::types::RealtimeBidirectionalEngine;
use dynamo_protocols::types::realtime::{
    EventType, RealtimeAPIError, RealtimeClientEvent, RealtimeClientEventSessionUpdate,
    RealtimeServerEvent, RealtimeServerEventError, RealtimeServerEventSessionCreated, Session,
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

async fn handle_socket(socket: WebSocket, state: Arc<service_v2::State>) {
    // Inbound writes a non-NORMAL close message to `close_reason` on protocol errors
    // before cancelling the engine; the ScopedWsWriter takes it on Drop.
    // Empty slot ⇒ NORMAL completion.
    let close_reason: Arc<Mutex<Option<Message>>> = Arc::new(Mutex::new(None));
    let (ws_tx, mut ws_rx) = socket.split();
    let mut writer = ScopedWsWriter::new(ws_tx, close_reason.clone());

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
            *close_reason.lock() = Some(close_message(
                close_code::ERROR,
                "internal error preparing session.created",
            ));
            return;
        }
    };
    if let Err(err) = writer
        .send(Message::Text(Utf8Bytes::from(session_created_payload)))
        .await
    {
        tracing::debug!(%err, "/v1/realtime client disconnected before session.created");
        return;
    }

    let Some((engine, session_update)) =
        select_engine(&mut ws_rx, &mut *writer, state.as_ref()).await
    else {
        // Client disconnected before an engine was selected.
        return;
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

    let mut response_stream = match engine.generate(input).await {
        Ok(s) => s,
        Err(err) => {
            tracing::error!(%err, "/v1/realtime engine.generate() failed");
            *close_reason.lock() = Some(close_message(
                close_code::ERROR,
                &format!("engine error: {err}"),
            ));
            return;
        }
    };
    let resp_ctx = response_stream.context();

    // Outbound task: drain the engine response stream onto the WebSocket.
    // Peels off the Dynamo-side `Annotated` envelope so clients receive bare
    // `RealtimeServerEvent` frames as the OpenAI Realtime spec requires. Engine
    // errors surfaced via `Annotated::error` are mapped to a synthesized
    // `RealtimeServerEvent::Error` so they remain visible on the wire.
    let outbound = tokio::spawn(async move {
        let mut writer = writer;
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
            if writer
                .send(Message::Text(Utf8Bytes::from(frame_payload)))
                .await
                .is_err()
            {
                tracing::debug!("/v1/realtime client disconnected during response");
                break;
            }
        }
        // writer dropped at end of scope → spawned cleanup sends Close + drains.
    });

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

/// RAII wrapper around the outbound side of the WebSocket. Owns the
/// [`SplitSink`] plus the inbound-supplied close-reason slot, and on `Drop`
/// spawns a detached cleanup that sends the Close frame (inbound-supplied
/// reason, or `NORMAL`) and drives the sink to completion under
/// [`CLOSE_DRAIN_TIMEOUT`]. Without that drain step axum can tear down the TCP
/// socket mid-frame and the client sees EOF instead of an in-band Close.
///
/// The wrapper [`Deref`]s to its inner sink so callers use it as if it were
/// the sink: `writer.send(...).await`, `&mut *writer` for fns that take
/// `&mut Sink<Message>`. There is no explicit close API — the close sequence
/// is bound to the wrapper's lifecycle, including panic / early-return paths.
///
/// `Drop` is sync and the close sequence is async, so the cleanup runs on
/// a spawned task that outlives the dropping scope. The task owns `ws_tx`
/// exclusively, so the WebSocket's underlying TCP socket stays open until
/// the cleanup writes the Close frame, drains the sink under
/// `CLOSE_DRAIN_TIMEOUT`, and drops `ws_tx`. `handle_socket`'s
/// `outbound.await` returns *before* the Close frame is on the wire, but
/// the client still sees the in-band Close — the only scenario the cleanup
/// can't complete is runtime shutdown racing connection teardown, which
/// doesn't happen on axum's long-lived global runtime.
struct ScopedWsWriter {
    ws_tx: Option<SplitSink<WebSocket, Message>>,
    close_reason: Arc<Mutex<Option<Message>>>,
}

impl ScopedWsWriter {
    fn new(
        ws_tx: SplitSink<WebSocket, Message>,
        close_reason: Arc<Mutex<Option<Message>>>,
    ) -> Self {
        Self {
            ws_tx: Some(ws_tx),
            close_reason,
        }
    }
}

impl std::ops::Deref for ScopedWsWriter {
    type Target = SplitSink<WebSocket, Message>;
    fn deref(&self) -> &Self::Target {
        self.ws_tx
            .as_ref()
            .expect("ScopedWsWriter sink only taken by Drop; no other consumer should exist")
    }
}

impl std::ops::DerefMut for ScopedWsWriter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ws_tx
            .as_mut()
            .expect("ScopedWsWriter sink only taken by Drop; no other consumer should exist")
    }
}

impl Drop for ScopedWsWriter {
    fn drop(&mut self) {
        let Some(mut ws_tx) = self.ws_tx.take() else {
            return;
        };
        let close_reason = self.close_reason.clone();
        // Spawn detached because Drop is sync but the close sequence is
        // async. Tokio's runtime outlives the per-connection task so the
        // cleanup task gets a chance to run; if the runtime is shutting
        // down we get a best-effort attempt.
        tokio::spawn(async move {
            let msg = close_reason
                .lock()
                .take()
                .unwrap_or_else(|| close_message(close_code::NORMAL, "stream complete"));
            let _ = ws_tx.send(msg).await;
            let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, ws_tx.close()).await;
        });
    }
}

/// Drive engine selection by looping over inbound client frames until either
/// a usable `session.update` lands and `ModelManager::get_realtime_engine`
/// returns Ok — `Some((engine, session_update))` — or the client disconnects
/// (Close frame, EOF, or transport error) — `None`.
///
/// Every other frame the loop sees (non-`session.update` event type, malformed
/// JSON, binary frame, missing/empty `session.model`, model not found,
/// model unavailable, other lookup errors) emits a spec-shaped
/// `RealtimeServerEvent::Error` to the client and the loop continues — a
/// well-behaved client can recover by sending another `session.update` with
/// corrected fields.
async fn select_engine<S, T>(
    ws_rx: &mut S,
    ws_tx: &mut T,
    state: &service_v2::State,
) -> Option<(
    RealtimeBidirectionalEngine,
    RealtimeClientEventSessionUpdate,
)>
where
    S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
    T: futures::Sink<Message, Error = axum::Error> + Unpin,
{
    while let Some(msg) = ws_rx.next().await {
        // Parse socket messages.
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                tracing::debug!(%err, "/v1/realtime inbound error during engine selection");
                return None;
            }
        };
        let event = match msg {
            Message::Text(text) => {
                match serde_json::from_str::<RealtimeClientEvent>(text.as_str()) {
                    Ok(e) => e,
                    Err(err) => {
                        // Client-driven and repeatable; debug! so a misbehaving
                        // peer can't amplify the log channel.
                        tracing::debug!(%err, "/v1/realtime malformed JSON during engine selection");
                        send_error_event(ws_tx, "invalid_request", "malformed JSON frame", None)
                            .await;
                        continue;
                    }
                }
            }
            Message::Binary(_) => {
                tracing::debug!("/v1/realtime binary frame during engine selection");
                send_error_event(
                    ws_tx,
                    "invalid_request",
                    "binary frames not supported",
                    None,
                )
                .await;
                continue;
            }
            Message::Close(_) => return None,
            Message::Ping(_) | Message::Pong(_) => continue, // axum handles ping replies
        };
        let session_update = match event {
            RealtimeClientEvent::SessionUpdate(req) => req,
            other => {
                tracing::debug!(
                    event = other.event_type(),
                    "/v1/realtime expected session.update before engine selection"
                );
                send_error_event(
                    ws_tx,
                    "invalid_request",
                    "expected session.update before engine is selected",
                    Some("session.update"),
                )
                .await;
                continue;
            }
        };

        // Extract model name from session update.
        let model_name = match &session_update.session {
            Session::RealtimeSession(s) => s.model.as_deref().filter(|m| !m.is_empty()),
            Session::RealtimeTranscriptionSession(_) => {
                // Transcription sessions need their own engine wiring (audio →
                // text via /audio/transcriptions) that this slice doesn't
                // implement. Surface that directly instead of letting the
                // empty `model` fall through to the generic "session.model
                // required" error from the realtime path below.
                send_error_event(
                    ws_tx,
                    "unsupported_session_type",
                    "session.type 'transcription' is not yet supported (only 'realtime' is supported)",
                    Some("session.type"),
                )
                .await;
                continue;
            }
        };
        let Some(model_name) = model_name else {
            send_error_event(
                ws_tx,
                "invalid_request",
                "session.model required",
                Some("session.model"),
            )
            .await;
            continue;
        };
        match state.manager().get_realtime_engine(model_name) {
            Ok(engine) => return Some((engine, session_update)),
            Err(ModelManagerError::ModelNotFound(_)) => {
                send_error_event(
                    ws_tx,
                    "model_not_found",
                    &format!("unknown model: {model_name}"),
                    Some("session.model"),
                )
                .await;
                continue;
            }
            Err(ModelManagerError::ModelUnavailable(_)) => {
                send_error_event(
                    ws_tx,
                    "model_unavailable",
                    &format!("model unavailable: {model_name}"),
                    Some("session.model"),
                )
                .await;
                continue;
            }
            Err(err) => {
                tracing::error!(%err, "/v1/realtime engine lookup failed");
                send_error_event(
                    ws_tx,
                    "server_error",
                    &err.to_string(),
                    Some("session.model"),
                )
                .await;
                continue;
            }
        }
    }
    None
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
