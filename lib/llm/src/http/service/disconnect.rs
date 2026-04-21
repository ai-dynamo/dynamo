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

use dynamo_runtime::config::environment_names::llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS as BACKEND_STREAM_TIMEOUT_ENV;

/// Read the backend stream inactivity timeout from the environment.
/// Returns `None` if unset or zero (timeout disabled).
///
/// The HTTP-layer timeout uses a 2x multiplier over the configured value so that
/// the request-plane timeout in `push_router` (which uses the raw value) always
/// fires first and triggers `report_instance_down()` for worker quarantine.
/// This layer is strictly a safety net for gauge cleanup.
pub fn backend_stream_timeout() -> Option<Duration> {
    std::env::var(BACKEND_STREAM_TIMEOUT_ENV)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map(|secs| Duration::from_secs(secs.saturating_mul(2)))
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
                            // We're terminating the stream intentionally here with a
                            // structured error + [DONE]; disarm so the stream handle
                            // doesn't later record this as ClosedUnexpectedly (which
                            // would mis-attribute the fault as a client disconnect).
                            stream_handle.disarm();
                            // DIS-1768: emit structured OpenAI-style error frame + `data: [DONE]`
                            // so naive `data:`-line parsers see both the error and a stream terminator.
                            let err_json = serde_json::json!({
                                "error": {
                                    "message": err.to_string(),
                                    "type": "internal_server_error",
                                    "code": 500,
                                }
                            });
                            yield Event::default().data(err_json.to_string());
                            yield Event::default().data("[DONE]");
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
                    inflight_guard.mark_error(ErrorType::ResponseTimeout);
                    stream_handle.disarm();
                    tracing::warn!(
                        request_id = %inflight_guard.request_id(),
                        model = %inflight_guard.model(),
                        endpoint = %inflight_guard.endpoint(),
                        request_type = %inflight_guard.request_type(),
                        error_type = "response_timeout",
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
    use crate::http::service::metrics::{Endpoint, ErrorType, RequestType, Status};
    use futures::StreamExt;
    use serial_test::serial;

    #[derive(Debug)]
    struct MockContext;
    impl MockContext {
        fn new() -> Self {
            Self
        }
    }
    #[async_trait::async_trait]
    impl dynamo_runtime::engine::AsyncEngineContext for MockContext {
        fn id(&self) -> &str {
            "test"
        }
        fn stop(&self) {}
        fn stop_generating(&self) {}
        fn kill(&self) {}
        fn is_stopped(&self) -> bool {
            false
        }
        fn is_killed(&self) -> bool {
            false
        }
        async fn stopped(&self) {
            std::future::pending::<()>().await;
        }
        async fn killed(&self) {
            std::future::pending::<()>().await;
        }
        fn link_child(&self, _: Arc<dyn dynamo_runtime::engine::AsyncEngineContext>) {}
    }

    fn hanging_stream()
    -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            std::future::pending::<()>().await;
            yield axum::response::sse::Event::default().data("unreachable");
        }
    }

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

    // SAFETY: env mutation is safe — all tests are single-threaded (#[serial] + tokio::test).
    fn setup_test(
        model: &str,
        req_id: &str,
        timeout_secs: &str,
    ) -> (
        Arc<Metrics>,
        InflightGuard,
        Arc<dyn AsyncEngineContext>,
        ConnectionHandle,
    ) {
        let metrics = Arc::new(Metrics::new());
        let guard =
            metrics
                .clone()
                .create_inflight_guard(model, Endpoint::ChatCompletions, true, req_id);
        let context: Arc<dyn AsyncEngineContext> = Arc::new(MockContext::new());
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let handle = ConnectionHandle::create_disabled(tx);
        unsafe { std::env::set_var(BACKEND_STREAM_TIMEOUT_ENV, timeout_secs) };
        (metrics, guard, context, handle)
    }

    fn cleanup_env() {
        unsafe { std::env::remove_var(BACKEND_STREAM_TIMEOUT_ENV) };
    }

    /// Zombie backend with hanging stream is terminated by inactivity timeout.
    #[tokio::test(start_paused = true)]
    #[serial]
    async fn test_backend_inactivity_timeout_releases_inflight_gauge() {
        let model = "zombie-model";
        // Config value "1" → HTTP-layer timeout is 2s (2x safety-net multiplier)
        let (metrics, guard, context, handle) = setup_test(model, "req-zombie", "1");
        assert_eq!(metrics.get_inflight_count(model), 1);

        let monitored = monitor_for_disconnects(hanging_stream(), context, guard, handle);
        tokio::pin!(monitored);

        tokio::time::advance(Duration::from_secs(3)).await;

        let completed = tokio::time::timeout(Duration::from_secs(2), async move {
            while monitored.next().await.is_some() {}
        })
        .await;

        cleanup_env();

        completed.expect("stream did not terminate — backend inactivity timeout is broken");
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge leaked"
        );

        // Verify the error was categorized as ResponseTimeout, not Cancelled
        assert_eq!(
            metrics.get_request_counter(
                model,
                &Endpoint::ChatCompletions,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::ResponseTimeout,
            ),
            1,
            "inactivity timeout should be recorded as ResponseTimeout"
        );
        assert_eq!(
            metrics.get_request_counter(
                model,
                &Endpoint::ChatCompletions,
                &RequestType::Stream,
                &Status::Error,
                &ErrorType::Cancelled,
            ),
            0,
            "inactivity timeout should NOT be recorded as Cancelled"
        );
    }

    /// Inactivity timeout resets on each token; only fires after a true gap.
    #[tokio::test(start_paused = true)]
    #[serial]
    async fn test_inactivity_timeout_resets_on_each_token() {
        let model = "reset-model";

        // Phase 1: tokens arrive every 2s with a 5s config (10s HTTP timeout after 2x multiplier)
        // — stream completes normally because each token resets the timer.
        let (metrics, guard_1, ctx_1, handle_1) = setup_test(model, "phase1", "5");
        assert_eq!(metrics.get_inflight_count(model), 1);

        let token_count = 5;
        let monitored_1 = monitor_for_disconnects(
            timed_token_stream(token_count, Duration::from_secs(2)),
            ctx_1,
            guard_1,
            handle_1,
        );
        tokio::pin!(monitored_1);

        let mut received = Vec::new();
        let phase1 = tokio::time::timeout(Duration::from_secs(30), async {
            while let Some(event) = monitored_1.next().await {
                received.push(event);
            }
        })
        .await;

        assert!(
            phase1.is_ok(),
            "inactivity timeout incorrectly fired as a hard deadline"
        );
        assert_eq!(received.len(), token_count + 1); // tokens + [DONE]
        assert_eq!(metrics.get_inflight_count(model), 0);

        // Phase 2: hanging stream — timeout DOES fire.
        let guard_2 =
            metrics
                .clone()
                .create_inflight_guard(model, Endpoint::ChatCompletions, true, "phase2");
        assert_eq!(metrics.get_inflight_count(model), 1);

        let ctx_2: Arc<dyn AsyncEngineContext> = Arc::new(MockContext::new());
        let (tx_2, _rx_2) = tokio::sync::oneshot::channel();
        let handle_2 = ConnectionHandle::create_disabled(tx_2);

        let monitored_2 = monitor_for_disconnects(hanging_stream(), ctx_2, guard_2, handle_2);
        tokio::pin!(monitored_2);

        // Config "5" → HTTP timeout 10s (2x multiplier). Advance past it.
        tokio::time::advance(Duration::from_secs(11)).await;

        let phase2 = tokio::time::timeout(Duration::from_secs(10), async {
            while monitored_2.next().await.is_some() {}
        })
        .await;

        cleanup_env();

        assert!(
            phase2.is_ok(),
            "hanging stream was not terminated by inactivity timeout"
        );
        assert_eq!(
            metrics.get_inflight_count(model),
            0,
            "inflight gauge leaked in phase 2"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // DIS-1768: mid-stream fault SSE contract
    //
    // When the upstream stream yields `Err(_)` mid-stream — e.g. an upstream
    // worker dies and the mpsc channel reports
    // `Disconnected: Stream ended before generation completed`, or the Python
    // chat-processor raises and the Rust→Python `tx.send()` fails with
    // `Failed to send response: SendError { .. }` — the client MUST receive:
    //   1. a structured `data: {"error":{"message":..., "type":... or "code":...}}` frame, then
    //   2. a `data: [DONE]` terminator.
    // Before the fix, the code emitted the bare SSE trailer
    // `event: error\n: <comment>\n\n` with no `[DONE]`, which violates the
    // OpenAI SSE contract and is silently skipped by naive `data:`-line parsers.
    // The two tests below pin the post-fix contract.
    // ─────────────────────────────────────────────────────────────────────────────

    /// Builds a stream that yields `data_chunks` successful events, then yields an
    /// `Err` carrying `err_msg`, simulating a mid-stream upstream fault.
    fn simulate_mid_stream_error(
        data_chunks: usize,
        err_msg: &'static str,
    ) -> impl futures::Stream<Item = Result<axum::response::sse::Event, axum::Error>> {
        async_stream::try_stream! {
            for i in 0..data_chunks {
                yield axum::response::sse::Event::default().data(format!("chunk-{i}"));
            }
            Err(axum::Error::new(err_msg))?;
        }
    }

    /// Collect the wire-format SSE body from a monitored stream.
    async fn collect_sse_body(
        stream: impl Stream<Item = Result<Event, axum::Error>> + Send + 'static,
    ) -> String {
        use axum::body::to_bytes;
        use axum::response::{IntoResponse, Sse};
        let response = Sse::new(stream).into_response();
        let body = to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("body bytes");
        String::from_utf8(body.to_vec()).expect("utf8 body")
    }

    /// Assert the post-fix SSE fault contract: parsed structured error frame with exact
    /// message/type/code, positioned before `[DONE]`, and no bare `event: error` trailer.
    fn assert_fault_contract(case: &str, text: &str, expected_message: &str) {
        let done_pos = text.find("data: [DONE]").unwrap_or_else(|| {
            panic!("[{case}] body does not terminate with `data: [DONE]`. Body:\n{text}")
        });

        let (error_line, error_frame) = text
            .lines()
            .find_map(|line| {
                let payload = line.strip_prefix("data: ")?;
                serde_json::from_str::<serde_json::Value>(payload)
                    .ok()
                    .filter(|v| v.get("error").is_some())
                    .map(|v| (line, v))
            })
            .unwrap_or_else(|| {
                panic!(
                    "[{case}] body missing structured JSON `data: {{\"error\":{{...}}}}` frame. Body:\n{text}"
                )
            });

        let error_pos = text.find(error_line).unwrap_or_default();
        assert!(
            error_pos < done_pos,
            "[{case}] structured error frame must precede `data: [DONE]`. Body:\n{text}"
        );

        let error = error_frame
            .get("error")
            .and_then(|v| v.as_object())
            .unwrap_or_else(|| panic!("[{case}] `error` field is not an object. Body:\n{text}"));
        assert_eq!(
            error.get("message").and_then(|v| v.as_str()),
            Some(expected_message),
            "[{case}] structured error `message` mismatch. Body:\n{text}"
        );
        assert_eq!(
            error.get("type").and_then(|v| v.as_str()),
            Some("internal_server_error"),
            "[{case}] structured error `type` mismatch. Body:\n{text}"
        );
        assert_eq!(
            error.get("code").and_then(|v| v.as_i64()),
            Some(500),
            "[{case}] structured error `code` mismatch. Body:\n{text}"
        );
        assert!(
            !text.contains("event: error\n: "),
            "[{case}] body contains bare `event: error\\n: <comment>` trailer (pre-fix bug). Body:\n{text}"
        );
    }

    /// Upstream worker killed mid-stream → mpsc channel reports `Disconnected` to the
    /// HTTP layer. Client MUST receive structured error + `[DONE]`.
    #[tokio::test]
    #[serial]
    async fn test_simulate_worker_kill_emits_structured_error_and_done() {
        let (_metrics, guard, ctx, handle) = setup_test("worker-kill-model", "req-wk", "0");
        let expected_message = "Disconnected: Stream ended before generation completed";
        let stream = simulate_mid_stream_error(3, expected_message);
        let monitored = monitor_for_disconnects(stream, ctx, guard, handle);
        let body = collect_sse_body(monitored).await;
        cleanup_env();
        assert_fault_contract("worker_kill", &body, expected_message);
    }

    /// Python chat-processor raises mid-stream → Rust→Python `tx.send()` fails with
    /// `SendError`. Client MUST receive structured error + `[DONE]`.
    #[tokio::test]
    #[serial]
    async fn test_simulate_python_consumer_drop_emits_structured_error_and_done() {
        let (_metrics, guard, ctx, handle) = setup_test("py-drop-model", "req-py", "0");
        let expected_message = "Failed to send response: SendError { .. }";
        let stream = simulate_mid_stream_error(3, expected_message);
        let monitored = monitor_for_disconnects(stream, ctx, guard, handle);
        let body = collect_sse_body(monitored).await;
        cleanup_env();
        assert_fault_contract("python_consumer_drop", &body, expected_message);
    }
}
