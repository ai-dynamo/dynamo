// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `disconnect` module provides a mechanism for our axum http services to monitoring and responding
//! to disconnects from the client.
//!
//! There are two potential phases in any request where we need to handle the disconnect.
//!
//! For unary, request-response, there is just a single phase where the primary task that axum kicks off
//! to handle the request will be dropped if the client disconnects. In order for us to have a long running
//! task, like an LLM request, we need to spawn our long running task in a separate task and then spawn
//! a second task that will monitor for disconnects from the client. The primary task which spawned the
//! two tasks will hold an "armed" [`ConnectionHandle`] which will issue a [`ConnectionStatus::ClosedUnexpectedly`]
//! if the task is dropped before it is [`ConnectionHandle::disarm`]ed.
//!
//! For the streaming case, request in - stream out, we need a second [`ConnectionHandle`] which will be owned
//! by the stream. A streaming response is when the [`axum::response::Response]] is a [axum::response::Sse] stream.
//! This means the primary task handle will go out of scope when it returns the stream. When we create our
//! SSE stream, we capture the second [`ConnectionHandle`] and arm it. If the stream closes gracefully, the
//! second handle will be disarmed, otherwise, the stream was dropped and the [`Drop`] trait on the [`ConnectionHandle`]
//! triggers a [`ConnectionStatus::ClosedUnexpectedly`] signal.
//!
//! The [`ConnectionHandle`] is a simple wrapper around a [`tokio::sync::oneshot::Sender`] which will send a
//! [`ConnectionStatus`] enum to the primary task. The primary task will then use this to determine if it should
//! cancel the request or not.
//!
//! The [`ConnectionHandle`] is also used to signal to the client that the request has been cancelled. This is
//! done by sending a [`axum::response::sse::Event`] with the event type "error" and the data "[DONE]".
//!

use axum::response::sse::Event;
use dynamo_runtime::engine::AsyncEngineContext;
use futures::{Stream, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::http::service::metrics::{InflightGuard, Metrics};

#[derive(Clone, Copy)]
pub enum ConnectionStatus {
    Disabled,
    ClosedUnexpectedly,
    ClosedGracefully,
}

pub struct ConnectionHandle {
    sender: Option<tokio::sync::oneshot::Sender<ConnectionStatus>>,
    on_drop: ConnectionStatus,
}

impl ConnectionHandle {
    /// Handle which by default will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
    pub fn create_disarmed(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::ClosedGracefully,
        }
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
    pub fn create_armed(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::ClosedUnexpectedly,
        }
    }

    /// Handle which will not issue a signal when dropped.
    pub fn create_disabled(sender: tokio::sync::oneshot::Sender<ConnectionStatus>) -> Self {
        Self {
            sender: Some(sender),
            on_drop: ConnectionStatus::Disabled,
        }
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
    pub fn disarm(&mut self) {
        self.on_drop = ConnectionStatus::ClosedGracefully;
    }

    /// Handle which will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
    pub fn arm(&mut self) {
        self.on_drop = ConnectionStatus::ClosedUnexpectedly;
    }
}

impl Drop for ConnectionHandle {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(self.on_drop);
        }
    }
}

/// Creates a pair of handles which will monitor for disconnects from the client.
///
/// The first handle is armed and will issue a [`ConnectionStatus::ClosedUnexpectedly`] signal when dropped.
/// The second handle is disarmed and will issue a [`ConnectionStatus::ClosedGracefully`] signal when dropped.
///
/// The handles are returned in the order of the first being armed and the second being disarmed.
pub async fn create_connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    metrics: Option<Arc<Metrics>>,
) -> (ConnectionHandle, ConnectionHandle) {
    // these oneshot channels monitor possible disconnects from the client in two different scopes:
    // - the local task (connection_handle)
    // - an optionally streaming response (stream_handle)
    let (connection_tx, connection_rx) = tokio::sync::oneshot::channel();
    let (stream_tx, stream_rx) = tokio::sync::oneshot::channel();

    // detached task that will naturally close when both handles are dropped
    tokio::spawn(connection_monitor(
        engine_context.clone(),
        connection_rx,
        stream_rx,
        metrics,
    ));

    // Two handles, the first is armed, the second is disarmed
    (
        ConnectionHandle::create_armed(connection_tx),
        ConnectionHandle::create_disabled(stream_tx),
    )
}

#[tracing::instrument(level = "trace", skip_all, fields(request_id = %engine_context.id()))]
async fn connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    connection_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    stream_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    metrics: Option<Arc<Metrics>>,
) {
    // Per-request cancellation state - ensures cancel path runs only once per request
    let mut cancel_handled = false;
    
    async fn handle_client_cancellation(
        cancel_handled: &mut bool,
        engine_context: &Arc<dyn AsyncEngineContext>,
        metrics: &Option<Arc<Metrics>>,
        cancel_reason: &str,
    ) {
        // Guard idempotency so cancel handler runs once per request
        if *cancel_handled {
            debug!("Cancellation already handled for request {}", engine_context.id());
            return;
        }
        *cancel_handled = true;

        tracing::trace!("{} closed unexpectedly; issuing cancellation", cancel_reason);
        if let Some(metrics) = metrics {
            metrics.inc_client_disconnect();
        }

        // Check if this is SGLang backend that requires two-phase cancellation
        if is_sglang_backend(engine_context) {
            info!("sglang_cancel_sent: Starting SGLang two-phase cancellation for request {}", engine_context.id());
            
            // Phase 1: Best-effort send_cancel() to SGLang adapter
            // The stop_generating() call notifies SGLang to begin cleanup
            engine_context.stop_generating();
            
            // Phase 2: Wait for terminal or timeout with tokio::select! and pinned sleep
            let grace_duration = cancel_grace_duration();
            let deadline = tokio::time::sleep(grace_duration);
            tokio::pin!(deadline);
            
            let start_time = Instant::now();
            
            tokio::select! {
                // Wait for context to report stopped (terminal condition from SGLang)
                _ = engine_context.stopped() => {
                    let handshake_ms = start_time.elapsed().as_millis() as u64;
                    info!("sglang_cancel_ack: SGLang graceful termination completed in {}ms for request {}", 
                         handshake_ms, engine_context.id());
                    
                    // TODO: Record metrics when repo's metrics system is available
                    // metrics.record_histogram("cancel.sglang.handshake_ms", handshake_ms);
                    // metrics.inc_counter("cancel.sglang.sent");
                    
                    // Context already stopped gracefully, no need to kill
                }
                // Timeout after grace period
                _ = &mut deadline => {
                    let timeout_ms = grace_duration.as_millis() as u64;
                    warn!("cancel_grace_timeout: SGLang cancellation timed out after {}ms for request {}", 
                         timeout_ms, engine_context.id());
                    
                    // TODO: Record metrics when repo's metrics system is available  
                    // metrics.inc_counter("cancel.sglang.timeout");
                    
                    // Force kill after timeout
                    engine_context.kill();
                }
            }
        } else {
            // Keep existing immediate drop path for non-SGLang backends (vLLM, TensorRT-LLM, etc.)
            debug!("Using immediate cancellation for non-SGLang backend");
            engine_context.kill();
        }
    }

    match connection_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            handle_client_cancellation(&mut cancel_handled, &engine_context, &metrics, "Connection").await;
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Connection closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }

    match stream_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            handle_client_cancellation(&mut cancel_handled, &engine_context, &metrics, "Stream").await;
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Stream closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }
}

/// This method will consume a stream of SSE events and monitor for disconnects or context cancellation.
///
/// Uses `tokio::select!` to choose between receiving events from the source stream or detecting when
/// the context is stopped. If the context is stopped, we break the stream. If the source stream ends
/// naturally, we mark the request as successful and send the final `[DONE]` event.
pub fn monitor_for_disconnects(
    stream: impl Stream<Item = Result<Event, axum::Error>>,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight_guard: InflightGuard,
    mut stream_handle: ConnectionHandle,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    stream_handle.arm();
    async_stream::try_stream! {
        tokio::pin!(stream);
        loop {
            tokio::select! {
                event = stream.next() => {
                    match event {
                        Some(Ok(event)) => {
                            yield event;
                        }
                        Some(Err(err)) => {
                            yield Event::default().event("error").comment(err.to_string());
                        }
                        None => {
                            // Stream ended normally
                            inflight_guard.mark_ok();
                            stream_handle.disarm();

                            // todo: if we yield a dynamo sentinel event, we need to do it before the done or the
                            // async-openai client will chomp it.
                            yield Event::default().data("[DONE]");
                            break;
                        }
                    }
                }
                _ = context.stopped() => {
                    tracing::trace!("Context stopped; breaking stream");
                    break;
                }
            }
        }
    }
}

/// Configuration helper for SGLang cancellation grace period.
/// 
/// This function provides a configurable grace period for SGLang backend cancellation
/// to prevent race conditions between Rust runtime and SGLang cleanup processes.
/// 
/// The grace period can be configured via the `CANCEL_GRACE_MS` environment variable.
/// Default is 300ms as recommended by project leaders.
fn cancel_grace_ms() -> u64 {
    std::env::var("CANCEL_GRACE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

/// Returns the cancel grace period as a Duration for use with tokio::time operations
fn cancel_grace_duration() -> Duration {
    Duration::from_millis(cancel_grace_ms())
}

/// Detect if this is an SGLang backend by examining the engine context.
/// Uses standardized ID prefix pattern following the existing engine type system.
fn is_sglang_backend(engine_context: &Arc<dyn AsyncEngineContext>) -> bool {
    engine_context.id().starts_with("sglang:")
}
