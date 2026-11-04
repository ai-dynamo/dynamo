// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod kserve;
pub mod openai;
pub mod tensor;

use std::sync::Arc;

use dynamo_runtime::engine::AsyncEngineContext;
use futures::{Stream, StreamExt};
use tonic::Status;

use crate::http::service::disconnect::ConnectionHandle;
use crate::http::service::error::SanitizedError;
use crate::http::service::metrics::{InflightGuard, request_was_unavailable};
use crate::types::Annotated;

/// Map a dispatch error that is not a client-visible rejection onto its gRPC status.
///
/// Worker-scoped and pool-scoped unavailability both mean the request found no
/// servable worker, so both answer `UNAVAILABLE` (14) and let the client retry.
/// Anything else is a server fault and keeps `INTERNAL` (13). The unavailable
/// message is sanitized to match the HTTP frontends; only the internal arm keeps
/// the caller's context string.
pub(crate) fn dispatch_error_status(
    error: &(dyn std::error::Error + 'static),
    internal_context: &str,
) -> Status {
    if request_was_unavailable(error) {
        return Status::unavailable(SanitizedError::Unavailable.to_string());
    }
    Status::internal(format!("{internal_context}: {error}"))
}

/// This method will consume an AsyncEngineStream and monitor for disconnects or context cancellation.
/// This is gRPC variant of `monitor_for_disconnects` as that implementation has SSE specific handling.
/// Should decouple and reuse `monitor_for_disconnects`
///
/// Uses `tokio::select!` to choose between receiving responses from the source stream or detecting when
/// the context is stopped. If the context is stopped, we break the stream. If the source stream ends
/// naturally, we mark the request as successful and disarm the stream handle. No terminal event is
/// sent in either case.
pub fn grpc_monitor_for_disconnects<T>(
    stream: impl Stream<Item = Annotated<T>>,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight_guard: InflightGuard,
    mut stream_handle: ConnectionHandle,
) -> impl Stream<Item = Annotated<T>> {
    stream_handle.arm();
    async_stream::stream! {
        tokio::pin!(stream);
        loop {
            tokio::select! {
                event = stream.next() => {
                    match event {
                        Some(response) => {
                            yield response;
                        }
                        None => {
                            // Stream ended normally
                            inflight_guard.mark_ok();
                            stream_handle.disarm();
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::error::{DynamoError, ErrorType};

    fn error(error_type: ErrorType) -> anyhow::Error {
        DynamoError::builder()
            .error_type(error_type)
            .message("boom")
            .build()
            .into()
    }

    #[test]
    fn unavailable_dispatch_errors_map_to_grpc_unavailable() {
        for error_type in [ErrorType::WorkerUnavailable, ErrorType::Unavailable] {
            let status = dispatch_error_status(error(error_type).as_ref(), "ctx");
            assert_eq!(status.code(), tonic::Code::Unavailable, "{error_type}");
            assert_eq!(status.message(), "Service temporarily unavailable");
        }
    }

    #[test]
    fn other_dispatch_errors_stay_internal_with_context() {
        let status = dispatch_error_status(error(ErrorType::Unknown).as_ref(), "ctx");
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().starts_with("ctx: "));
    }
}

// Kept apart from `mod tests` so the monitor cases stay independent of the
// `dispatch_error_status` cases.
#[cfg(test)]
mod monitor_tests {
    use super::*;
    use crate::http::service::disconnect::create_connection_monitor;
    use crate::http::service::metrics::{
        CancellationLabels, Endpoint, ErrorType, Metrics, RequestType, Status as RequestStatus,
    };
    use crate::protocols::openai::completions::NvCreateCompletionResponse;
    use crate::protocols::tensor::NvCreateTensorResponse;
    use dynamo_runtime::pipeline::context::Controller;
    use std::time::Duration;

    const MODEL: &str = "grpc-monitor-model";

    struct Fixture {
        metrics: Arc<Metrics>,
        context: Arc<Controller>,
        endpoint: Endpoint,
    }

    impl Fixture {
        fn labels(&self) -> CancellationLabels {
            CancellationLabels {
                model: MODEL.to_string(),
                endpoint: self.endpoint.to_string(),
                request_type: "stream".to_string(),
            }
        }

        fn request_count(&self, status: RequestStatus, error_type: ErrorType) -> u64 {
            self.metrics.get_request_counter(
                MODEL,
                &self.endpoint,
                &RequestType::Stream,
                &status,
                &error_type,
            )
        }

        // The connection monitor task reacts to handle drops asynchronously.
        async fn assert_killed(&self) {
            tokio::time::timeout(Duration::from_secs(1), self.context.killed())
                .await
                .expect("dropping an armed stream handle must kill the context");
            assert_eq!(self.metrics.get_client_disconnect_count(), 1);
            assert_eq!(self.metrics.get_cancellation_count(&self.labels()), 1);
        }

        async fn assert_not_killed(&self) {
            assert!(
                tokio::time::timeout(Duration::from_millis(100), self.context.killed())
                    .await
                    .is_err(),
                "a disarmed stream handle must not kill the context"
            );
            assert_eq!(self.metrics.get_client_disconnect_count(), 0);
            assert_eq!(self.metrics.get_cancellation_count(&self.labels()), 0);
        }

        fn assert_counted(&self, success: bool) {
            assert_eq!(self.metrics.get_inflight_count(MODEL), 0);
            let (ok, err) = if success { (1, 0) } else { (0, 1) };
            assert_eq!(
                self.request_count(RequestStatus::Success, ErrorType::None),
                ok
            );
            assert_eq!(
                self.request_count(RequestStatus::Error, ErrorType::Internal),
                err
            );
        }
    }

    // Mirrors the handler setup: real metrics, guard, context and connection
    // monitor; the connection handle is disarmed once the stream is built.
    async fn setup(endpoint: Endpoint) -> (Fixture, InflightGuard, ConnectionHandle) {
        let fixture = Fixture {
            metrics: Arc::new(Metrics::new()),
            context: Arc::new(Controller::default()),
            endpoint,
        };
        let (mut connection_handle, stream_handle) = create_connection_monitor(
            fixture.context.clone(),
            Some(fixture.metrics.clone()),
            fixture.labels(),
        )
        .await;
        let guard =
            fixture
                .metrics
                .clone()
                .create_inflight_guard(MODEL, fixture.endpoint, true, "req");
        connection_handle.disarm();
        drop(connection_handle);
        assert_eq!(fixture.metrics.get_inflight_count(MODEL), 1);
        (fixture, guard, stream_handle)
    }

    // The monitor never reads the payload, so data-less items exercise it for
    // any response type.
    fn items<T>(count: usize) -> impl Stream<Item = Annotated<T>> {
        futures::stream::iter((0..count).map(|i| Annotated {
            data: None,
            id: Some(i.to_string()),
            event: None,
            comment: None,
            error: None,
        }))
    }

    async fn normal_eof<T>(endpoint: Endpoint) {
        let (f, guard, handle) = setup(endpoint).await;
        let ids: Vec<_> =
            grpc_monitor_for_disconnects(items::<T>(3), f.context.clone(), guard, handle)
                .map(|item| item.id)
                .collect()
                .await;
        assert_eq!(ids, ["0", "1", "2"].map(|id| Some(id.to_string())));
        f.assert_counted(true);
        f.assert_not_killed().await;
    }

    async fn context_stop<T>(endpoint: Endpoint) {
        let (f, guard, handle) = setup(endpoint).await;
        let mut monitored = Box::pin(grpc_monitor_for_disconnects(
            futures::stream::pending::<Annotated<T>>(),
            f.context.clone(),
            guard,
            handle,
        ));
        assert!(futures::poll!(monitored.next()).is_pending());
        f.context.stop_generating();
        let next = tokio::time::timeout(Duration::from_secs(1), monitored.next())
            .await
            .expect("a stopped context must end the stream");
        assert!(next.is_none());
        drop(monitored);
        f.assert_counted(false);
        f.assert_killed().await;
    }

    async fn drop_unconsumed<T>(endpoint: Endpoint) {
        let (f, guard, handle) = setup(endpoint).await;
        let monitored =
            grpc_monitor_for_disconnects(items::<T>(3), f.context.clone(), guard, handle);
        drop(monitored);
        f.assert_counted(false);
        f.assert_killed().await;
    }

    async fn drop_partly_consumed<T>(endpoint: Endpoint) {
        let (f, guard, handle) = setup(endpoint).await;
        let mut monitored = Box::pin(grpc_monitor_for_disconnects(
            items::<T>(3),
            f.context.clone(),
            guard,
            handle,
        ));
        let first = monitored.next().await.expect("first item");
        assert_eq!(first.id.as_deref(), Some("0"));
        drop(monitored);
        f.assert_counted(false);
        f.assert_killed().await;
    }

    #[tokio::test]
    async fn completion_normal_eof_marks_success_and_disarms() {
        normal_eof::<NvCreateCompletionResponse>(Endpoint::Completions).await;
    }

    #[tokio::test]
    async fn tensor_normal_eof_marks_success_and_disarms() {
        normal_eof::<NvCreateTensorResponse>(Endpoint::Tensor).await;
    }

    #[tokio::test]
    async fn completion_context_stop_ends_stream_without_success() {
        context_stop::<NvCreateCompletionResponse>(Endpoint::Completions).await;
    }

    #[tokio::test]
    async fn tensor_context_stop_ends_stream_without_success() {
        context_stop::<NvCreateTensorResponse>(Endpoint::Tensor).await;
    }

    #[tokio::test]
    async fn completion_drop_unconsumed_cancels() {
        drop_unconsumed::<NvCreateCompletionResponse>(Endpoint::Completions).await;
    }

    #[tokio::test]
    async fn tensor_drop_unconsumed_cancels() {
        drop_unconsumed::<NvCreateTensorResponse>(Endpoint::Tensor).await;
    }

    #[tokio::test]
    async fn completion_drop_partly_consumed_cancels() {
        drop_partly_consumed::<NvCreateCompletionResponse>(Endpoint::Completions).await;
    }

    #[tokio::test]
    async fn tensor_drop_partly_consumed_cancels() {
        drop_partly_consumed::<NvCreateTensorResponse>(Endpoint::Tensor).await;
    }
}
