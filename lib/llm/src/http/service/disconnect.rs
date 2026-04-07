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
use std::sync::Arc;
use std::time::Duration;

use crate::http::service::metrics::{CancellationLabels, ErrorType, InflightGuard, Metrics};

/// Environment variable name for configuring the backend stream inactivity timeout.
///
/// When set to a positive integer, `monitor_for_disconnects` will kill the engine context
/// and drop the inflight guard if no SSE event is received from the backend within this
/// many seconds. This acts as a circuit breaker for zombie workers that hold a live TCP
/// connection but never produce output, which would otherwise permanently inflate the
/// `dynamo_frontend_inflight_requests` gauge.
///
/// Set to `0` or leave unset to disable the timeout (default: disabled).
pub const BACKEND_STREAM_TIMEOUT_ENV: &str = "DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS";

/// Read the backend stream inactivity timeout from the environment.
/// Returns `None` if unset or zero (timeout disabled).
pub fn backend_stream_timeout() -> Option<Duration> {
    std::env::var(BACKEND_STREAM_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map(Duration::from_secs)
}

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
    cancellation_labels: CancellationLabels,
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
        cancellation_labels,
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
    cancellation_labels: CancellationLabels,
) {
    match connection_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            // the client has disconnected, no need to gracefully cancel, just kill the context
            tracing::warn!("Connection closed unexpectedly; issuing cancellation");
            if let Some(metrics) = &metrics {
                metrics.inc_client_disconnect();
                metrics.inc_cancellation(&cancellation_labels);
            }
            engine_context.kill();
        }
        Ok(ConnectionStatus::ClosedGracefully) => {
            tracing::trace!("Connection closed gracefully");
        }
        Ok(ConnectionStatus::Disabled) => {}
    }

    match stream_rx.await {
        Err(_) | Ok(ConnectionStatus::ClosedUnexpectedly) => {
            tracing::warn!("Stream closed unexpectedly; issuing cancellation");
            if let Some(metrics) = &metrics {
                metrics.inc_client_disconnect();
                metrics.inc_cancellation(&cancellation_labels);
            }
            engine_context.kill();
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
///
/// A configurable inactivity timeout (see [`BACKEND_STREAM_TIMEOUT_ENV`]) adds a third arm: if no
/// SSE event is received from the backend within the timeout window, the engine context is killed and
/// the inflight guard is dropped, preventing permanent gauge inflation caused by zombie workers that
/// hold a live TCP connection but produce no output.
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

    // Read the backend inactivity timeout once at stream construction time.
    // None means the timeout arm in select! will never fire (std::future::pending).
    let inactivity_timeout = backend_stream_timeout();

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
                    // Mark as cancelled when context is stopped (client disconnect or timeout)
                    inflight_guard.mark_error(ErrorType::Cancelled);
                    // Token counts (input_tokens, output_tokens) are recorded on
                    // the enclosing span by ResponseMetricCollector::Drop.
                    tracing::warn!(
                        request_id = %inflight_guard.request_id(),
                        model = %inflight_guard.model(),
                        endpoint = %inflight_guard.endpoint(),
                        request_type = %inflight_guard.request_type(),
                        error_type = "cancelled",
                        elapsed_ms = %inflight_guard.elapsed_ms(),
                        "request cancelled"
                    );
                    break;
                }
                // Circuit breaker for zombie backend workers: if the backend holds a live TCP
                // connection but produces no output for `inactivity_timeout`, kill the engine
                // context so that InflightGuard::drop() fires and dec() corrects the gauge.
                // The sleep is re-created each iteration so it acts as an *inactivity* timeout
                // (resets whenever a token is received), not a hard total-request deadline.
                // When inactivity_timeout is None the pending() future never resolves.
                _ = async {
                    match inactivity_timeout {
                        Some(d) => tokio::time::sleep(d).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    inflight_guard.mark_error(ErrorType::Cancelled);
                    tracing::warn!(
                        request_id = %inflight_guard.request_id(),
                        model = %inflight_guard.model(),
                        endpoint = %inflight_guard.endpoint(),
                        request_type = %inflight_guard.request_type(),
                        error_type = "cancelled",
                        elapsed_ms = %inflight_guard.elapsed_ms(),
                        timeout_secs = ?inactivity_timeout.map(|d| d.as_secs()),
                        "backend stream inactivity timeout; killing engine context to release inflight gauge"
                    );
                    context.kill();
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::client::HttpRequestContext;
    use crate::http::service::metrics::Endpoint;
    use futures::StreamExt;
    use serial_test::serial;

    /// Returns a stream that never yields any items, simulating a zombie backend
    /// that holds a live TCP connection but produces no SSE output.
    fn hanging_stream()
    -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            // This future never resolves; the stream will block indefinitely.
            std::future::pending::<()>().await;
            // Unreachable but required to give the closure the correct item type.
            yield axum::response::sse::Event::default().data("unreachable");
        }
    }

    /// Regression test for issue #7545:
    ///
    /// A zombie backend worker that holds a live TCP connection but never produces
    /// output would cause `monitor_for_disconnects` to block forever.  As a result
    /// `InflightGuard::drop` never fired, `dec_inflight_gauge` was never called,
    /// and `dynamo_frontend_inflight_requests` accumulated indefinitely.
    ///
    /// This test verifies the fix: with `DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS` set,
    /// the hung stream is terminated by the inactivity timeout, and the inflight
    /// gauge is decremented back to zero.
    ///
    /// Without the third `select!` arm added by the fix, this test would stall until
    /// the outer `tokio::time::timeout` fires and the test would fail with
    /// "Stream did not terminate within deadline".
    #[tokio::test(start_paused = true)]
    #[serial]
    async fn test_backend_inactivity_timeout_releases_inflight_gauge() {
        // Use an isolated Metrics object; no registry needed for gauge inc/dec.
        let metrics = Arc::new(Metrics::new());
        let model = "zombie-model";

        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "gauge should start at zero"
        );

        let inflight_guard = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            /*streaming=*/ true,
            "test-req-7545",
        );

        assert_eq!(
            metrics.get_inflight_count(model),
            1,
            "gauge should be 1 after guard creation"
        );

        // A real context whose `stopped()` blocks until `cancel_token` is cancelled.
        let context: Arc<dyn AsyncEngineContext> = Arc::new(HttpRequestContext::new());

        // Dummy oneshot — we only need the sender side to construct ConnectionHandle.
        let (stream_tx, _stream_rx) = tokio::sync::oneshot::channel();
        let stream_handle = ConnectionHandle::create_disabled(stream_tx);

        // Set a 1-second inactivity timeout. Because the test runtime starts paused,
        // tokio::time::sleep(1s) resolves as soon as we advance time — no real delay.
        // SAFETY: env mutation is safe in single-threaded tokio tests.
        // SAFETY: test is single-threaded (tokio::test), no concurrent env access
        unsafe {
            std::env::set_var(BACKEND_STREAM_TIMEOUT_ENV, "1");
        }

        let monitored =
            monitor_for_disconnects(hanging_stream(), context, inflight_guard, stream_handle);
        tokio::pin!(monitored);

        // Advance virtual time past the timeout so the sleep arm fires.
        tokio::time::advance(Duration::from_secs(2)).await;

        // Drive the stream to exhaustion. The timeout arm should have broken the loop.
        let completed = tokio::time::timeout(Duration::from_secs(1), async move {
            while monitored.next().await.is_some() {}
        })
        .await;

        // SAFETY: test cleanup, single-threaded
        unsafe {
            std::env::remove_var(BACKEND_STREAM_TIMEOUT_ENV);
        }

        completed.expect(
            "Stream did not terminate after inactivity timeout — \
             the backend stream timeout circuit breaker is not working",
        );

        // InflightGuard was dropped by the stream combinator; gauge must return to 0.
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge leaked — InflightGuard was not dropped on timeout"
        );
    }

    /// Returns a stream that yields `count` tokens, pausing `interval` between each.
    /// Useful for simulating a healthy backend that produces output at a regular cadence.
    fn timed_token_stream(
        count: usize,
        interval: Duration,
    ) -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            for i in 0..count {
                tokio::time::sleep(interval).await;
                yield axum::response::sse::Event::default().data(format!("token-{i}"));
            }
        }
    }

    /// Verify that the inactivity timeout resets each time a token is received.
    ///
    /// The stream produces tokens with gaps SHORTER than the timeout, so the
    /// timeout should never fire.  After the stream finishes normally, we then
    /// feed a hanging stream and confirm the timeout DOES fire -- proving the
    /// timeout is per-gap (inactivity), not a hard total-request deadline.
    #[tokio::test(start_paused = true)]
    #[serial]
    async fn test_inactivity_timeout_resets_on_each_token() {
        let metrics = Arc::new(Metrics::new());
        let model = "reset-model";

        // -- Phase 1: tokens arrive faster than the timeout -> stream completes normally --

        let inflight_guard_1 = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            /*streaming=*/ true,
            "test-reset-phase1",
        );
        assert_eq!(metrics.get_inflight_count(model), 1);

        let context_1: Arc<dyn AsyncEngineContext> = Arc::new(HttpRequestContext::new());
        let (stream_tx_1, _stream_rx_1) = tokio::sync::oneshot::channel();
        let stream_handle_1 = ConnectionHandle::create_disabled(stream_tx_1);

        // Timeout = 5s, but tokens arrive every 2s. The total wall time (5 tokens x 2s = 10s)
        // exceeds the timeout, proving it is an inactivity timer, not a hard deadline.
        // SAFETY: test is single-threaded (tokio::test), no concurrent env access
        unsafe {
            std::env::set_var(BACKEND_STREAM_TIMEOUT_ENV, "5");
        }

        let token_count = 5;
        let token_interval = Duration::from_secs(2);
        let monitored_1 = monitor_for_disconnects(
            timed_token_stream(token_count, token_interval),
            context_1,
            inflight_guard_1,
            stream_handle_1,
        );
        tokio::pin!(monitored_1);

        // Advance time step by step to let each token + the final [DONE] flow through.
        // Total virtual time: 5 tokens x 2s = 10s, plus a bit extra for the [DONE].
        let mut received = Vec::new();
        let phase1 = tokio::time::timeout(Duration::from_secs(30), async {
            while let Some(event) = monitored_1.next().await {
                received.push(event);
            }
        })
        .await;

        assert!(
            phase1.is_ok(),
            "Phase 1 timed out -- inactivity timeout incorrectly fired as a hard deadline"
        );

        // 5 token events + 1 [DONE] sentinel = 6 events total
        assert_eq!(
            received.len(),
            token_count + 1,
            "Expected {expected} events (tokens + [DONE]), got {actual}",
            expected = token_count + 1,
            actual = received.len()
        );

        // Guard was dropped normally; gauge back to zero.
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge should be zero after normal stream completion"
        );

        // -- Phase 2: hanging stream -> timeout DOES fire --

        let inflight_guard_2 = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            /*streaming=*/ true,
            "test-reset-phase2",
        );
        assert_eq!(metrics.get_inflight_count(model), 1);

        let context_2: Arc<dyn AsyncEngineContext> = Arc::new(HttpRequestContext::new());
        let (stream_tx_2, _stream_rx_2) = tokio::sync::oneshot::channel();
        let stream_handle_2 = ConnectionHandle::create_disabled(stream_tx_2);

        let monitored_2 = monitor_for_disconnects(
            hanging_stream(),
            context_2,
            inflight_guard_2,
            stream_handle_2,
        );
        tokio::pin!(monitored_2);

        // Advance past the 5s inactivity timeout so the sleep arm fires.
        tokio::time::advance(Duration::from_secs(6)).await;

        // Drive the stream; the timeout arm should have already broken the loop.
        let phase2 = tokio::time::timeout(Duration::from_secs(10), async {
            while monitored_2.next().await.is_some() {}
        })
        .await;

        // SAFETY: test cleanup, single-threaded
        unsafe {
            std::env::remove_var(BACKEND_STREAM_TIMEOUT_ENV);
        }

        assert!(
            phase2.is_ok(),
            "Phase 2: hanging stream was not terminated by inactivity timeout"
        );
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge leaked in phase 2 -- timeout did not release guard"
        );
    }

    /// Verify that when `DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS` is unset, the
    /// inactivity timeout is disabled and a slow stream completes normally.
    ///
    /// This ensures the opt-out path works: operators who do not set the env var
    /// should never see healthy (but slow) streams killed by the timeout.
    #[tokio::test(start_paused = true)]
    #[serial]
    async fn test_no_timeout_when_env_unset() {
        let metrics = Arc::new(Metrics::new());
        let model = "no-timeout-model";

        // Ensure the env var is NOT set.
        // SAFETY: test is single-threaded (tokio::test), no concurrent env access
        unsafe {
            std::env::remove_var(BACKEND_STREAM_TIMEOUT_ENV);
        }

        let inflight_guard = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            /*streaming=*/ true,
            "test-no-timeout",
        );
        assert_eq!(metrics.get_inflight_count(model), 1);

        let context: Arc<dyn AsyncEngineContext> = Arc::new(HttpRequestContext::new());
        let (stream_tx, _stream_rx) = tokio::sync::oneshot::channel();
        let stream_handle = ConnectionHandle::create_disabled(stream_tx);

        // Each token takes 30 seconds -- far longer than any reasonable timeout.
        // Without the env var, the timeout arm should never fire.
        let token_count = 3;
        let token_interval = Duration::from_secs(30);
        let monitored = monitor_for_disconnects(
            timed_token_stream(token_count, token_interval),
            context,
            inflight_guard,
            stream_handle,
        );
        tokio::pin!(monitored);

        let mut received = Vec::new();
        let result = tokio::time::timeout(Duration::from_secs(300), async {
            while let Some(event) = monitored.next().await {
                received.push(event);
            }
        })
        .await;

        assert!(
            result.is_ok(),
            "Stream was terminated even though timeout env var is unset --              the opt-out path is broken"
        );

        // 3 token events + 1 [DONE] sentinel
        assert_eq!(
            received.len(),
            token_count + 1,
            "Expected {expected} events, got {actual}",
            expected = token_count + 1,
            actual = received.len()
        );

        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge should be zero after normal completion"
        );
    }
}
