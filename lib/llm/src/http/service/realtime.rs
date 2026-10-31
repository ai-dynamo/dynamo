// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebSocket endpoint at `/v1/realtime` for the OpenAI Realtime API.
//!
//! Wire shape: client sends a sequence of `Message::Text` frames each containing a
//! JSON-encoded [`DynamoRealtimeClientEvent`]; server forwards each frame onto an
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
//!   `select_engine` until a `session.update` arrives with a usable model
//!   (`session.model` for realtime sessions or
//!   `session.audio.input.transcription.model` for transcription sessions)
//!   and `ModelManager::get_realtime_engine` returns Ok.
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
    response::{IntoResponse, Response},
    routing::get,
};
use dynamo_runtime::engine::AsyncEngineContextProvider;
use dynamo_runtime::pipeline::{Context, ManyIn, ManyOut, RequestStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{SinkExt, StreamExt, stream::SplitSink};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

#[cfg(test)]
mod tests;

const WORKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
const SOCKET_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// Bound on the per-connection request queue. Picks backpressure over
/// unbounded growth so a fast client cannot drive memory exhaustion against
/// a slow engine.
const REQUEST_CHANNEL_CAPACITY: usize = 64;

/// Bound on the time the outbound task waits for the WebSocket sink to
/// drain a final Close frame before tearing down the transport. Keeps a
/// half-broken peer from parking the connection indefinitely.
const CLOSE_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

use super::{RouteDoc, error::SanitizedError, service_v2};
use crate::discovery::ModelManagerError;
use crate::migration::realtime::{RealtimeRecovery, UnexpectedEnd};
use crate::types::RealtimeBidirectionalEngine;
use dynamo_protocols::types::realtime::{
    DynamoRealtimeClientEvent, EventType, RealtimeAPIError, RealtimeClientEvent,
    RealtimeClientEventSessionUpdate, RealtimeServerEvent, RealtimeServerEventError,
    RealtimeServerEventSessionCreated, Session,
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
    let permit = state.acquire_inflight();
    // Count the upgrade attempt before the readiness gate so shutdown cannot
    // observe zero inflight while this WebSocket admission is in progress.
    if !state.accepts_inference() {
        drop(permit);
        return super::openai::ErrorMessage::_service_unavailable().into_response();
    }

    upgrade
        .on_upgrade(move |socket| handle_socket(socket, state, permit))
        .into_response()
}

async fn handle_socket(
    socket: WebSocket,
    state: Arc<service_v2::State>,
    // Held for the full WebSocket task lifetime so realtime sessions are
    // visible to graceful shutdown after the 101 upgrade response completes.
    _permit: service_v2::InflightPermit,
) {
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

    let selected = tokio::select! {
        _ = state.cancel_token().cancelled() => return,
        selected = select_engine(&mut ws_rx, &mut *writer, state.as_ref()) => selected,
    };
    let Some((engine, session_update)) = selected else {
        // Client disconnected before an engine was selected.
        return;
    };

    let mut session = RealtimeSession::new(
        engine,
        &session_update,
        state.migration_limit,
        state.cancel_token().clone(),
    );
    if let Err(error) = session
        .run(&mut ws_rx, &mut *writer, &close_reason, session_update)
        .await
    {
        tracing::warn!(%error, "Realtime session failed");
        *close_reason.lock() = Some(close_message(
            close_code::ERROR,
            &SanitizedError::Internal.to_string(),
        ));
        let _ = tokio::time::timeout(
            CLOSE_DRAIN_TIMEOUT,
            send_error_event(
                &mut *writer,
                "server_error",
                &SanitizedError::Internal.to_string(),
                None,
            ),
        )
        .await;
    }
}

struct StopGeneration(dynamo_runtime::engine::Context);

impl Drop for StopGeneration {
    fn drop(&mut self) {
        self.0.stop_generating();
    }
}

async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

struct RealtimeSession {
    engine: RealtimeBidirectionalEngine,
    recovery: RealtimeRecovery,
    incoming: Option<DynamoRealtimeClientEvent>,
    cancel: CancellationToken,
}

impl RealtimeSession {
    fn new(
        engine: RealtimeBidirectionalEngine,
        session: &RealtimeClientEventSessionUpdate,
        migration_limit: u32,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            engine,
            recovery: RealtimeRecovery::new(session, migration_limit),
            incoming: None,
            cancel,
        }
    }

    /// Keep the client connection across worker attempts, retrying only eligible failures.
    async fn run<S, T>(
        &mut self,
        ws_rx: &mut S,
        ws_tx: &mut T,
        close_reason: &Mutex<Option<Message>>,
        session: RealtimeClientEventSessionUpdate,
    ) -> anyhow::Result<()>
    where
        S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
        T: futures::Sink<Message, Error = axum::Error> + Unpin,
    {
        let mut setup = RealtimeClientEvent::SessionUpdate(session).into();
        loop {
            self.recovery.start_attempt()?;
            match self.run_attempt(ws_rx, ws_tx, close_reason, setup).await {
                Ok(()) => return Ok(()),
                Err(error) => setup = self.recovery.retry(error)?,
            }
        }
    }

    /// Send setup first and keep the worker's stop guard alive through connection and forwarding.
    async fn run_attempt<S, T>(
        &mut self,
        ws_rx: &mut S,
        ws_tx: &mut T,
        close_reason: &Mutex<Option<Message>>,
        setup: DynamoRealtimeClientEvent,
    ) -> anyhow::Result<()>
    where
        S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
        T: futures::Sink<Message, Error = axum::Error> + Unpin,
    {
        let (req_tx, req_rx) = mpsc::channel(REQUEST_CHANNEL_CAPACITY);
        req_tx.try_send(setup)?;
        let input = Context::new(RequestStream::new(Box::pin(ReceiverStream::new(req_rx))));
        let _stop = StopGeneration(input.context());
        let Some(responses) = self.connect_worker(ws_rx, close_reason, input).await? else {
            return Ok(());
        };
        self.forward_events(ws_rx, ws_tx, close_reason, &req_tx, responses)
            .await
    }

    /// Wait for a worker while buffering at most one client event and observing recovery's deadline.
    /// `Ok(None)` ends the session on client disconnect or frontend shutdown without a retry.
    async fn connect_worker<S>(
        &mut self,
        ws_rx: &mut S,
        close_reason: &Mutex<Option<Message>>,
        input: ManyIn<DynamoRealtimeClientEvent>,
    ) -> anyhow::Result<Option<ManyOut<Annotated<RealtimeServerEvent>>>>
    where
        S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
    {
        let connecting = self.engine.generate(input);
        tokio::pin!(connecting);
        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => return Ok(None),
                _ = wait_for_deadline(self.recovery.deadline()) => anyhow::bail!("realtime recovery timed out"),
                response = &mut connecting => return response.map(Some),
                msg = ws_rx.next(), if self.incoming.is_none() => {
                    if !queue_client_message(msg, &mut self.incoming, close_reason) { return Ok(None); }
                }
            }
        }
    }

    /// Forward saved input before live input and reconcile worker output before delivery.
    /// If the worker stops reading, drain terminal output before deciding whether to retry.
    async fn forward_events<S, T>(
        &mut self,
        ws_rx: &mut S,
        ws_tx: &mut T,
        close_reason: &Mutex<Option<Message>>,
        req_tx: &mpsc::Sender<DynamoRealtimeClientEvent>,
        mut responses: ManyOut<Annotated<RealtimeServerEvent>>,
    ) -> anyhow::Result<()>
    where
        S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
        T: futures::Sink<Message, Error = axum::Error> + Unpin,
    {
        let mut pending = None;
        let mut is_sender_closed = false;
        let mut sender_deadline = None;
        loop {
            if pending.is_none() {
                pending = self.recovery.next_replay()?;
                if pending.is_none()
                    && !self.recovery.is_replaying()
                    && let Some(event) = self.incoming.take()
                {
                    self.recovery.record(&event);
                    pending = Some(event);
                }
            }
            tokio::select! {
                _ = self.cancel.cancelled() => return Ok(()),
                _ = wait_for_deadline(self.recovery.deadline()) => anyhow::bail!("realtime recovery timed out"),
                _ = wait_for_deadline(sender_deadline) => return Err(UnexpectedEnd.into()),
                response = responses.next() => {
                    let Some(annotated) = response else {
                        return self.recovery.stream_ended();
                    };
                    if let Some(error) = annotated.error { return Err(error.into()); }
                    let Some(mut event) = annotated.data else { continue; };
                    if !self.recovery.reconcile(&mut event)? { continue; }
                    if !self.write_client_event(ws_rx, ws_tx, close_reason, &event).await? {
                        return Ok(());
                    }
                }
                permit = req_tx.reserve(), if pending.is_some() && !is_sender_closed => {
                    match permit {
                        Ok(permit) => {
                            if let Some(event) = pending.take() { permit.send(event); }
                        }
                        Err(_) => {
                            // Drain terminal output before classifying a closed sender: a pending
                            // application error must not be mistaken for a transport failure.
                            is_sender_closed = true;
                            sender_deadline = Some(Instant::now() + WORKER_DRAIN_TIMEOUT);
                        }
                    }
                }
                msg = ws_rx.next(), if self.incoming.is_none() => {
                    if !queue_client_message(msg, &mut self.incoming, close_reason) { return Ok(()); }
                }
            }
        }
    }

    /// Bound each client write while still observing shutdown, recovery timeout, and client input.
    /// `Ok(false)` ends the session; a client write failure must never trigger worker recovery.
    async fn write_client_event<S, T>(
        &mut self,
        ws_rx: &mut S,
        ws_tx: &mut T,
        close_reason: &Mutex<Option<Message>>,
        event: &RealtimeServerEvent,
    ) -> anyhow::Result<bool>
    where
        S: futures::Stream<Item = Result<Message, axum::Error>> + Unpin,
        T: futures::Sink<Message, Error = axum::Error> + Unpin,
    {
        let payload = serde_json::to_string(event)?;
        // Bounded input buffering can hide a Close behind a text frame
        // while output is blocked, so every write needs a deadline.
        let writing = tokio::time::timeout(
            SOCKET_WRITE_TIMEOUT,
            ws_tx.send(Message::Text(payload.into())),
        );
        tokio::pin!(writing);
        loop {
            tokio::select! {
                _ = self.cancel.cancelled() => return Ok(false),
                _ = wait_for_deadline(self.recovery.deadline()) => anyhow::bail!("realtime recovery timed out"),
                result = &mut writing => return Ok(matches!(result, Ok(Ok(())))),
                msg = ws_rx.next(), if self.incoming.is_none() => {
                    if !queue_client_message(msg, &mut self.incoming, close_reason) { return Ok(false); }
                }
            }
        }
    }
}

fn queue_client_message(
    msg: Option<Result<Message, axum::Error>>,
    incoming: &mut Option<DynamoRealtimeClientEvent>,
    close_reason: &Mutex<Option<Message>>,
) -> bool {
    match msg {
        Some(Ok(Message::Text(text))) => match serde_json::from_str(text.as_str()) {
            Ok(event) => *incoming = Some(event),
            Err(_) => {
                *close_reason.lock() =
                    Some(close_message(close_code::INVALID, "malformed JSON frame"));
                return false;
            }
        },
        Some(Ok(Message::Binary(_))) => {
            *close_reason.lock() = Some(close_message(
                close_code::UNSUPPORTED,
                "binary frames not supported",
            ));
            return false;
        }
        Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
        _ => return false,
    }
    true
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
/// `CLOSE_DRAIN_TIMEOUT`, and drops `ws_tx`. `handle_socket` can return
/// before the Close frame is on the wire, but
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
            let _ = tokio::time::timeout(CLOSE_DRAIN_TIMEOUT, async {
                // Receiving a Close queues tungstenite's automatic reply; another
                // send can fail with SendAfterClosing, but that reply still needs flushing.
                let _ = ws_tx.send(msg).await;
                let _ = ws_tx.close().await;
            })
            .await;
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
                match serde_json::from_str::<DynamoRealtimeClientEvent>(text.as_str()) {
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
            DynamoRealtimeClientEvent::OpenAI(RealtimeClientEvent::SessionUpdate(req)) => req,
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
        let (model_name, model_param) = match &session_update.session {
            Session::RealtimeSession(s) => (
                s.model.as_deref().filter(|m| !m.is_empty()),
                "session.model",
            ),
            Session::RealtimeTranscriptionSession(s) => (
                s.audio
                    .input
                    .transcription
                    .as_ref()
                    .and_then(|t| t.model.as_deref())
                    .filter(|m| !m.is_empty()),
                "session.audio.input.transcription.model",
            ),
        };
        let Some(model_name) = model_name else {
            send_error_event(
                ws_tx,
                "invalid_request",
                &format!("{model_param} required"),
                Some(model_param),
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
                    Some(model_param),
                )
                .await;
                continue;
            }
            Err(ModelManagerError::ModelUnavailable(_)) => {
                send_error_event(
                    ws_tx,
                    "model_unavailable",
                    &format!("model unavailable: {model_name}"),
                    Some(model_param),
                )
                .await;
                continue;
            }
            Err(err) => {
                tracing::error!(%err, "/v1/realtime engine lookup failed");
                send_error_event(
                    ws_tx,
                    "server_error",
                    &SanitizedError::Internal.to_string(),
                    Some(model_param),
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
            r#type: if code == "server_error" {
                "server_error"
            } else {
                "invalid_request_error"
            }
            .to_string(),
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
