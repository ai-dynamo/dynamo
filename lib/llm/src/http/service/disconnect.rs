// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use std::{sync::Arc, time::Duration};

use crate::http::service::metrics::{ErrorType, InflightGuard, Metrics};

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

async fn handle_client_cancellation(
    cancel_handled: &mut bool,
    engine_context: &Arc<dyn AsyncEngineContext>,
    metrics: Option<&Arc<Metrics>>,
    cancel_reason: &str,
    cancel_grace: Duration,
) {
    if *cancel_handled {
        return;
    }
    *cancel_handled = true;

    tracing::trace!(
        request_id = %engine_context.id(),
        cancel_reason,
        "client disconnected; requesting graceful cancellation"
    );
    if let Some(metrics) = metrics {
        metrics.inc_client_disconnect();
    }

    engine_context.stop_generating();

    tokio::select! {
        _ = engine_context.stopped() => {
            tracing::debug!(
                request_id = %engine_context.id(),
                cancel_reason,
                "request stopped before disconnect grace deadline"
            );
        }
        _ = tokio::time::sleep(cancel_grace) => {
            tracing::warn!(
                request_id = %engine_context.id(),
                cancel_reason,
                timeout_ms = cancel_grace.as_millis(),
                "request did not stop before disconnect grace deadline; forcing kill"
            );
            engine_context.kill();
        }
    }
}

#[tracing::instrument(level = "trace", skip_all, fields(request_id = %engine_context.id()))]
async fn connection_monitor(
    engine_context: Arc<dyn AsyncEngineContext>,
    connection_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    stream_rx: tokio::sync::oneshot::Receiver<ConnectionStatus>,
    metrics: Option<Arc<Metrics>>,
) {
    let mut cancel_handled = false;

    match connection_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            handle_client_cancellation(
                &mut cancel_handled,
                &engine_context,
                metrics.as_ref(),
                "connection",
                cancel_grace_duration(),
            )
            .await;
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Connection closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }

    match stream_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            handle_client_cancellation(
                &mut cancel_handled,
                &engine_context,
                metrics.as_ref(),
                "stream",
                cancel_grace_duration(),
            )
            .await;
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

    // Default to Cancelled: if the stream is dropped unexpectedly (e.g. client
    // disconnect causing a broken-pipe on the SSE write), the guard will report
    // "cancelled" instead of "internal". The happy path overrides this via mark_ok().
    inflight_guard.mark_error(ErrorType::Cancelled);

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
                            // Mark error as internal since it's a streaming error
                            inflight_guard.mark_error(ErrorType::Internal);
                            yield Event::default().event("error").comment(err.to_string());
                            // Break to prevent any subsequent mark_ok() from overwriting the error
                            break;
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
                    // Mark as cancelled when context is stopped (client disconnect or timeout)
                    inflight_guard.mark_error(ErrorType::Cancelled);
                    break;
                }
            }
        }
    }
}

fn cancel_grace_ms() -> u64 {
    std::env::var("CANCEL_GRACE_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .map(|value: u64| value.min(10_000))
        .unwrap_or(300)
}

fn cancel_grace_duration() -> Duration {
    Duration::from_millis(cancel_grace_ms())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::Notify;

    #[derive(Debug)]
    struct TestContext {
        id: &'static str,
        auto_stop_on_cancel: bool,
        stop_calls: AtomicUsize,
        kill_calls: AtomicUsize,
        stopped: AtomicBool,
        killed: AtomicBool,
        stopped_notify: Notify,
        killed_notify: Notify,
    }

    impl TestContext {
        fn new(auto_stop_on_cancel: bool) -> Self {
            Self {
                id: "test-request",
                auto_stop_on_cancel,
                stop_calls: AtomicUsize::new(0),
                kill_calls: AtomicUsize::new(0),
                stopped: AtomicBool::new(false),
                killed: AtomicBool::new(false),
                stopped_notify: Notify::new(),
                killed_notify: Notify::new(),
            }
        }

        fn stop_calls(&self) -> usize {
            self.stop_calls.load(Ordering::SeqCst)
        }

        fn kill_calls(&self) -> usize {
            self.kill_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl AsyncEngineContext for TestContext {
        fn id(&self) -> &str {
            self.id
        }

        fn is_stopped(&self) -> bool {
            self.stopped.load(Ordering::SeqCst)
        }

        fn is_killed(&self) -> bool {
            self.killed.load(Ordering::SeqCst)
        }

        async fn stopped(&self) {
            while !self.is_stopped() {
                self.stopped_notify.notified().await;
            }
        }

        async fn killed(&self) {
            while !self.is_killed() {
                self.killed_notify.notified().await;
            }
        }

        fn stop_generating(&self) {
            self.stop_calls.fetch_add(1, Ordering::SeqCst);
            if self.auto_stop_on_cancel {
                self.stopped.store(true, Ordering::SeqCst);
                self.stopped_notify.notify_waiters();
            }
        }

        fn stop(&self) {
            self.stop_generating();
        }

        fn kill(&self) {
            self.kill_calls.fetch_add(1, Ordering::SeqCst);
            self.killed.store(true, Ordering::SeqCst);
            self.stopped.store(true, Ordering::SeqCst);
            self.killed_notify.notify_waiters();
            self.stopped_notify.notify_waiters();
        }

        fn link_child(&self, _child: Arc<dyn AsyncEngineContext>) {}
    }

    #[tokio::test]
    async fn disconnect_uses_graceful_stop_before_kill() {
        let raw_context = Arc::new(TestContext::new(true));
        let context: Arc<dyn AsyncEngineContext> = raw_context.clone();
        let mut cancel_handled = false;

        handle_client_cancellation(
            &mut cancel_handled,
            &context,
            None,
            "connection",
            Duration::from_millis(10),
        )
        .await;

        assert_eq!(raw_context.stop_calls(), 1);
        assert_eq!(raw_context.kill_calls(), 0);
    }

    #[tokio::test]
    async fn disconnect_forces_kill_after_grace_timeout() {
        let raw_context = Arc::new(TestContext::new(false));
        let context: Arc<dyn AsyncEngineContext> = raw_context.clone();

        let task = tokio::spawn(async move {
            let mut cancel_handled = false;
            handle_client_cancellation(
                &mut cancel_handled,
                &context,
                None,
                "stream",
                Duration::from_millis(10),
            )
            .await;
        });

        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("disconnect task should finish")
            .unwrap();

        assert_eq!(raw_context.stop_calls(), 1);
        assert_eq!(raw_context.kill_calls(), 1);
    }

    #[tokio::test]
    async fn duplicate_disconnect_only_handles_cancellation_once() {
        let raw_context = Arc::new(TestContext::new(true));
        let context: Arc<dyn AsyncEngineContext> = raw_context.clone();
        let mut cancel_handled = false;

        handle_client_cancellation(
            &mut cancel_handled,
            &context,
            None,
            "connection",
            Duration::from_millis(10),
        )
        .await;
        handle_client_cancellation(
            &mut cancel_handled,
            &context,
            None,
            "stream",
            Duration::from_millis(10),
        )
        .await;

        assert_eq!(raw_context.stop_calls(), 1);
        assert_eq!(raw_context.kill_calls(), 0);
    }
}
