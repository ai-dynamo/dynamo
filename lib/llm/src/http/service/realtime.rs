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
//! wire. The handler then waits for the first client frame; if it's a
//! `session.update`, the carried `session.model` field is the routing input for
//! engine selection. The selected engine handles everything subsequent
//! (including `session.updated` echoes, audio-buffer state, and response
//! generation).
//!
//! For now the engine is a process-scoped mock ([`EchoBidirectionalEngine`]) held
//! in a `OnceLock` and `select_engine` ignores the model field — there's only
//! one engine to return. The architectural shape (model in, engine out) is in
//! place for #9174 to swap the static for a `ModelManager`-keyed lookup.

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
use dynamo_protocols::types::realtime::{
    RealtimeAPIError, RealtimeClientEvent, RealtimeServerEvent, RealtimeServerEventError,
    RealtimeServerEventSessionCreated, Session,
};
use uuid::Uuid;

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
    install_engine(EchoBidirectionalEngine)
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
    // OpenAI Realtime spec requires `session.created` to be the first server
    // frame on the wire, before any client event arrives. The handler synthesizes
    // it here so the connection handshake works regardless of which engine is
    // installed; engine selection happens once we observe the client's first
    // frame (typically `session.update` carrying the desired model).
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

    // Wait for the first client frame to drive engine selection. If it's a
    // `session.update`, its model field is the routing input; otherwise the
    // model is unspecified and selection falls back to default.
    let first_event = match wait_for_first_client_event(&mut ws_rx).await {
        FirstEvent::Event(event) => *event,
        FirstEvent::ClosedByClient => return,
        FirstEvent::ProtocolError { code, reason } => {
            let _ = ws_tx.send(close_message(code, &reason)).await;
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ws_tx.close()).await;
            return;
        }
    };
    let model = session_update_model(&first_event);
    let Some(engine) = select_engine(model.as_deref()) else {
        tracing::error!(
            ?model,
            "/v1/realtime connection rejected: no engine available"
        );
        let _ = ws_tx
            .send(close_message(
                close_code::ERROR,
                "no realtime engine available",
            ))
            .await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ws_tx.close()).await;
        return;
    };

    let (req_tx, req_rx) = mpsc::channel::<RealtimeClientEvent>(REQUEST_CHANNEL_CAPACITY);

    // Forward the first event into the engine. For `session.update` in
    // particular this is load-bearing, not just echo plumbing: the payload
    // carries per-session generation config the engine actually consumes —
    // instructions, voice, audio formats, turn-detection / VAD parameters,
    // max_output_tokens, tools, output_modalities. The handler's only
    // interest in the event was extracting the routing key (`model`); the
    // rest of the config is the engine's to apply, so we hand the whole
    // event over verbatim.
    if req_tx.send(first_event).await.is_err() {
        tracing::debug!("/v1/realtime engine receiver dropped before first event delivered");
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

/// Outcome of waiting for the first client frame on a freshly-upgraded
/// connection — either a parsed event ready to drive engine selection, a
/// clean client-initiated close, or a protocol-level reason to close the
/// connection ourselves.
///
/// `Event` is boxed so the enum size doesn't track the largest variant of
/// `RealtimeClientEvent` (~400 bytes due to the upstream `Session` payload),
/// which is the rest-of-handler-irrelevant detail clippy flags as a
/// large-variant difference.
enum FirstEvent {
    Event(Box<RealtimeClientEvent>),
    ClosedByClient,
    ProtocolError { code: u16, reason: String },
}

/// Drain the inbound socket until the first `RealtimeClientEvent` arrives.
/// Skips ping/pong frames (axum handles ping replies on its own); rejects
/// binary and malformed-JSON frames per the rest of the handler's posture.
async fn wait_for_first_client_event<S>(ws_rx: &mut S) -> FirstEvent
where
    S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
{
    while let Some(msg) = ws_rx.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                tracing::debug!(%err, "/v1/realtime inbound frame error before first event");
                return FirstEvent::ClosedByClient;
            }
        };
        match msg {
            Message::Text(text) => match serde_json::from_str::<RealtimeClientEvent>(text.as_str())
            {
                Ok(event) => return FirstEvent::Event(Box::new(event)),
                Err(err) => {
                    tracing::warn!(%err, "/v1/realtime malformed JSON before first event");
                    return FirstEvent::ProtocolError {
                        code: close_code::INVALID,
                        reason: "malformed JSON frame".to_string(),
                    };
                }
            },
            Message::Binary(_) => {
                tracing::warn!("/v1/realtime binary frame before first event; rejecting");
                return FirstEvent::ProtocolError {
                    code: close_code::UNSUPPORTED,
                    reason: "binary frames not supported".to_string(),
                };
            }
            Message::Close(_) => return FirstEvent::ClosedByClient,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
    FirstEvent::ClosedByClient
}

/// Extract the `model` routing key from a client event. Today only
/// `session.update` carries one (`session.session.model`); other events
/// return `None`, which signals "use default" to [`select_engine`].
fn session_update_model(event: &RealtimeClientEvent) -> Option<String> {
    let RealtimeClientEvent::SessionUpdate(req) = event else {
        return None;
    };
    match &req.session {
        Session::RealtimeSession(s) => s.model.clone(),
        Session::RealtimeTranscriptionSession(_) => None,
    }
}

/// Pick the realtime engine for this connection.
///
/// `model` comes from a client `session.update` (or `None` if the first
/// client frame didn't specify one). Today the slot is a single
/// `OnceLock<EchoBidirectionalEngine>` and the model is logged but not used
/// for routing — there's only one engine to return. #9174 will replace this
/// with a `ModelManager`-keyed lookup; the function signature is the
/// architectural seam.
fn select_engine(model: Option<&str>) -> Option<&'static EchoBidirectionalEngine> {
    if let Some(m) = model {
        tracing::debug!(model = m, "/v1/realtime engine selection by model");
    }
    BIDIRECTIONAL_ENGINE.get()
}
