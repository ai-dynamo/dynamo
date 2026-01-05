// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response handling for push handler.
//!
//! This module handles creating response channels (publishers) and streaming
//! responses back to clients over TCP.

use std::sync::Arc;

use futures::StreamExt;
use serde::Serialize;
use tokio::sync::Notify;

use crate::engine::{AsyncEngineContext, AsyncEngineContextProvider, Data};
use crate::metrics::prometheus_names::work_handler;
use crate::pipeline::network::{
    tcp, ConnectionInfo, NetworkStreamWrapper, StreamSender, STREAM_ERR_MSG,
};
use crate::pipeline::PipelineError;
use crate::protocols::maybe_error::MaybeError;

use super::metrics::WorkHandlerMetrics;

/// Create a TCP response publisher for sending responses back to the client.
///
/// # Errors
///
/// Returns `PipelineError::Generic` if the TCP connection cannot be established.
pub async fn create_response_publisher(
    context: Arc<dyn AsyncEngineContext>,
    connection_info: ConnectionInfo,
    metrics: Option<&Arc<WorkHandlerMetrics>>,
) -> Result<StreamSender, PipelineError> {
    tracing::trace!("creating tcp response stream");
    tcp::client::TcpClient::create_response_stream(context, connection_info)
        .await
        .map_err(|e| {
            if let Some(m) = metrics {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::RESPONSE_STREAM])
                    .inc();
            }
            PipelineError::Generic(format!("Failed to create response stream: {:?}", e))
        })
}

/// Send prologue to client and handle generate result.
///
/// On success, sends a success prologue and returns the stream.
/// On error, sends an error prologue and propagates the error.
///
/// # Errors
///
/// Returns the generate error after sending the error prologue to the client.
pub async fn send_prologue_and_get_stream<S>(
    publisher: &mut StreamSender,
    generate_result: Result<S, PipelineError>,
) -> Result<S, PipelineError> {
    match generate_result {
        Ok(stream) => {
            tracing::trace!("Successfully generated response stream; sending prologue");
            let _result = publisher.send_prologue(None).await;
            Ok(stream)
        }
        Err(e) => {
            let error_string = e.to_string();

            #[cfg(debug_assertions)]
            {
                tracing::debug!(
                    "Failed to generate response stream (with debug backtrace): {:?}",
                    e
                );
            }
            #[cfg(not(debug_assertions))]
            {
                tracing::error!("Failed to generate response stream: {}", error_string);
            }

            let _result = publisher.send_prologue(Some(error_string)).await;
            Err(e)
        }
    }
}

/// Stream responses from the engine to the client.
///
/// Handles:
/// - Serializing each response with `NetworkStreamWrapper`
/// - Detecting stream errors and early termination
/// - Sending the `complete_final` marker on successful completion
/// - Recording metrics for response bytes and errors
/// - Notifying health check on successful completion
pub async fn stream_responses<S, U>(
    stream: &mut S,
    publisher: &mut StreamSender,
    metrics: Option<&Arc<WorkHandlerMetrics>>,
    health_notifier: Option<&Arc<Notify>>,
) -> Result<(), PipelineError>
where
    S: futures::Stream<Item = U> + AsyncEngineContextProvider + Unpin,
    U: Data + Serialize + MaybeError + std::fmt::Debug,
{
    let context = stream.context();

    // TODO: Detect end-of-stream using Server-Sent Events (SSE)
    let mut send_complete_final = true;

    while let Some(resp) = stream.next().await {
        tracing::trace!("Sending response: {:?}", resp);

        if let Some(err) = resp.err()
            && format!("{:?}", err) == STREAM_ERR_MSG
        {
            tracing::warn!(STREAM_ERR_MSG);
            send_complete_final = false;
            break;
        }

        let resp_wrapper = NetworkStreamWrapper {
            data: Some(resp),
            complete_final: false,
        };
        let resp_bytes = serde_json::to_vec(&resp_wrapper)
            .expect("fatal error: invalid response object - this should never happen");

        if let Some(m) = metrics {
            m.response_bytes.inc_by(resp_bytes.len() as u64);
        }

        if (publisher.send(resp_bytes.into()).await).is_err() {
            // If context is already stopped (e.g., stop word detected), the client likely
            // closed the connection after receiving the complete response. This is expected.
            if !context.is_stopped() {
                tracing::error!("Failed to publish response for stream {}", context.id());
            }
            context.stop_generating();
            send_complete_final = false;
            if let Some(m) = metrics {
                m.error_counter
                    .with_label_values(&[work_handler::error_types::PUBLISH_RESPONSE])
                    .inc();
            }
            break;
        }
    }

    if send_complete_final {
        send_complete_final_marker::<U>(publisher, &context, metrics, health_notifier).await;
    }

    Ok(())
}

/// Send the complete_final marker to signal end of stream.
async fn send_complete_final_marker<U>(
    publisher: &mut StreamSender,
    context: &Arc<dyn AsyncEngineContext>,
    metrics: Option<&Arc<WorkHandlerMetrics>>,
    health_notifier: Option<&Arc<Notify>>,
) where
    U: Serialize,
{
    let resp_wrapper = NetworkStreamWrapper::<U> {
        data: None,
        complete_final: true,
    };
    let resp_bytes = serde_json::to_vec(&resp_wrapper)
        .expect("fatal error: invalid response object - this should never happen");

    if let Some(m) = metrics {
        m.response_bytes.inc_by(resp_bytes.len() as u64);
    }

    if (publisher.send(resp_bytes.into()).await).is_err() {
        tracing::error!(
            "Failed to publish complete final for stream {}",
            context.id()
        );
        if let Some(m) = metrics {
            m.error_counter
                .with_label_values(&[work_handler::error_types::PUBLISH_FINAL])
                .inc();
        }
    }

    // Notify the health check manager that the stream has finished.
    // This resets the timer, delaying the next canary health check.
    if let Some(notifier) = health_notifier {
        notifier.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::context::Controller;
    use crate::pipeline::network::StreamSender;
    use prometheus::{Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts};
    use serde::{Deserialize, Serialize};
    use std::pin::Pin;
    use std::task::{Context as TaskContext, Poll};

    /// Create standalone test metrics (not tied to an Endpoint)
    fn create_test_metrics() -> Arc<WorkHandlerMetrics> {
        Arc::new(WorkHandlerMetrics {
            request_counter: IntCounter::new("test_requests", "test").unwrap(),
            request_duration: Histogram::with_opts(HistogramOpts::new("test_duration", "test"))
                .unwrap(),
            inflight_requests: IntGauge::new("test_inflight", "test").unwrap(),
            request_bytes: IntCounter::new("test_req_bytes", "test").unwrap(),
            response_bytes: IntCounter::new("test_resp_bytes", "test").unwrap(),
            error_counter: IntCounterVec::new(Opts::new("test_errors", "test"), &["error_type"])
                .unwrap(),
        })
    }

    fn create_test_context() -> Arc<dyn AsyncEngineContext> {
        Arc::new(Controller::new("test-stream-id".to_string()))
    }

    // ==================== send_prologue_and_get_stream tests ====================

    #[tokio::test]
    async fn test_send_prologue_on_success_returns_stream() {
        let (mut publisher, _rx) = StreamSender::new_test();
        let stream_value = "my_stream";

        let result =
            send_prologue_and_get_stream(&mut publisher, Ok::<_, PipelineError>(stream_value))
                .await;

        assert_eq!(result.unwrap(), stream_value);
    }

    #[tokio::test]
    async fn test_send_prologue_on_error_propagates_error() {
        let (mut publisher, _rx) = StreamSender::new_test();
        let error = PipelineError::Generic("test error".to_string());

        let result = send_prologue_and_get_stream::<()>(&mut publisher, Err(error)).await;

        assert!(matches!(
            result,
            Err(PipelineError::Generic(msg)) if msg == "test error"
        ));
    }

    #[tokio::test]
    async fn test_send_prologue_success_sends_message() {
        let (mut publisher, mut rx) = StreamSender::new_test();

        let _ =
            send_prologue_and_get_stream(&mut publisher, Ok::<_, PipelineError>("stream")).await;

        // Verify a prologue message was sent
        let msg = rx.try_recv();
        assert!(msg.is_ok(), "Expected prologue message to be sent");
    }

    #[tokio::test]
    async fn test_send_prologue_error_sends_message() {
        let (mut publisher, mut rx) = StreamSender::new_test();
        let error = PipelineError::Generic("something went wrong".to_string());

        let _ = send_prologue_and_get_stream::<()>(&mut publisher, Err(error)).await;

        // Verify error prologue was sent
        let msg = rx.try_recv();
        assert!(msg.is_ok(), "Expected error prologue message to be sent");
    }

    // ==================== Mock types for stream_responses tests ====================

    /// A simple response type for testing
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestResponse {
        text: String,
    }

    // Data is auto-implemented for Send + Sync + 'static

    impl MaybeError for TestResponse {
        fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
            TestResponse {
                text: format!("error: {}", err),
            }
        }

        fn err(&self) -> Option<anyhow::Error> {
            None
        }
    }

    /// A mock stream that wraps a Vec and provides AsyncEngineContext
    #[derive(Debug)]
    struct MockResponseStream<T: Send + Sync + std::fmt::Debug> {
        items: std::collections::VecDeque<T>,
        context: Arc<dyn AsyncEngineContext>,
    }

    impl<T: Send + Sync + std::fmt::Debug> MockResponseStream<T> {
        fn new(items: Vec<T>, context: Arc<dyn AsyncEngineContext>) -> Self {
            Self {
                items: items.into(),
                context,
            }
        }
    }

    impl<T: Send + Sync + Unpin + std::fmt::Debug> futures::Stream for MockResponseStream<T> {
        type Item = T;

        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
        ) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.items.pop_front())
        }
    }

    impl<T: Send + Sync + std::fmt::Debug> AsyncEngineContextProvider for MockResponseStream<T> {
        fn context(&self) -> Arc<dyn AsyncEngineContext> {
            self.context.clone()
        }
    }

    // ==================== stream_responses tests ====================


    #[tokio::test]
    async fn test_stream_responses_sends_all_items() {
        let (mut publisher, mut rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();
        let responses = vec![
            TestResponse {
                text: "hello".to_string(),
            },
            TestResponse {
                text: "world".to_string(),
            },
        ];

        let mut mock_stream = MockResponseStream::new(responses, context);

        let result = stream_responses(&mut mock_stream, &mut publisher, None, None).await;

        assert!(result.is_ok());

        // Skip prologue
        let _prologue = rx.try_recv().expect("Expected prologue");

        // First response
        let msg1 = rx.try_recv().expect("Expected first response");
        let data1 = msg1.data().expect("Expected data");
        let wrapper1: NetworkStreamWrapper<TestResponse> = serde_json::from_slice(data1).unwrap();
        assert_eq!(wrapper1.data.unwrap().text, "hello");
        assert!(!wrapper1.complete_final);

        // Second response
        let msg2 = rx.try_recv().expect("Expected second response");
        let data2 = msg2.data().expect("Expected data");
        let wrapper2: NetworkStreamWrapper<TestResponse> = serde_json::from_slice(data2).unwrap();
        assert_eq!(wrapper2.data.unwrap().text, "world");
        assert!(!wrapper2.complete_final);

        // Complete final marker
        let msg3 = rx.try_recv().expect("Expected complete_final");
        let data3 = msg3.data().expect("Expected data");
        let wrapper3: NetworkStreamWrapper<TestResponse> = serde_json::from_slice(data3).unwrap();
        assert!(wrapper3.data.is_none());
        assert!(wrapper3.complete_final);
    }

    #[tokio::test]
    async fn test_stream_responses_records_response_bytes() {
        let (mut publisher, _rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();
        let metrics = create_test_metrics();
        let responses = vec![TestResponse {
            text: "test".to_string(),
        }];

        let mut mock_stream = MockResponseStream::new(responses, context);

        assert_eq!(metrics.response_bytes.get(), 0);

        let _ = stream_responses(&mut mock_stream, &mut publisher, Some(&metrics), None).await;

        // Should have recorded bytes for response + complete_final
        assert!(
            metrics.response_bytes.get() > 0,
            "Expected response_bytes to be incremented"
        );
    }

    #[tokio::test]
    async fn test_stream_responses_empty_stream_sends_complete_final() {
        let (mut publisher, mut rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();
        let responses: Vec<TestResponse> = vec![];

        let mut mock_stream = MockResponseStream::new(responses, context);

        let result = stream_responses(&mut mock_stream, &mut publisher, None, None).await;

        assert!(result.is_ok());

        // Skip prologue
        let _prologue = rx.try_recv().expect("Expected prologue");

        // Should still send complete_final even with empty stream
        let msg = rx.try_recv().expect("Expected complete_final");
        let data = msg.data().expect("Expected data");
        let wrapper: NetworkStreamWrapper<TestResponse> = serde_json::from_slice(data).unwrap();
        assert!(wrapper.data.is_none());
        assert!(wrapper.complete_final);
    }

    #[tokio::test]
    async fn test_stream_responses_notifies_health_check_on_completion() {
        let (mut publisher, _rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();
        let notifier = Arc::new(Notify::new());
        let responses = vec![TestResponse {
            text: "done".to_string(),
        }];

        let mut mock_stream = MockResponseStream::new(responses, context);

        // Spawn a task that waits for notification
        let notifier_clone = notifier.clone();
        let handle = tokio::spawn(async move {
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                notifier_clone.notified(),
            )
            .await
        });

        let _ = stream_responses(&mut mock_stream, &mut publisher, None, Some(&notifier)).await;

        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Expected health check notification");
    }

    #[tokio::test]
    async fn test_stream_responses_stops_on_publisher_failure() {
        let (mut publisher, rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;
        drop(rx); // Simulate publisher failure

        let context = create_test_context();
        let metrics = create_test_metrics();
        let responses = vec![
            TestResponse {
                text: "first".to_string(),
            },
            TestResponse {
                text: "second".to_string(),
            },
        ];

        let mut mock_stream = MockResponseStream::new(responses, context);

        let result = stream_responses(&mut mock_stream, &mut publisher, Some(&metrics), None).await;

        // Should still return Ok (errors are logged, not propagated)
        assert!(result.is_ok());

        // Should have recorded PUBLISH_RESPONSE error
        assert_eq!(
            metrics
                .error_counter
                .with_label_values(&[work_handler::error_types::PUBLISH_RESPONSE])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn test_stream_responses_stops_generating_on_publisher_failure() {
        let (mut publisher, rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;
        drop(rx); // Simulate publisher failure

        let context = create_test_context();
        let responses = vec![TestResponse {
            text: "item".to_string(),
        }];

        let mut mock_stream = MockResponseStream::new(responses, context.clone());

        assert!(
            !context.is_stopped(),
            "Context should not be stopped initially"
        );

        let _ = stream_responses(&mut mock_stream, &mut publisher, None, None).await;

        assert!(
            context.is_stopped(),
            "Context should be stopped after publisher failure"
        );
    }

    // ==================== send_complete_final_marker tests ====================

    #[tokio::test]
    async fn test_send_complete_final_marker_sends_correct_message() {
        let (mut publisher, mut rx) = StreamSender::new_test();
        // Consume the prologue first (required before send() works)
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();

        send_complete_final_marker::<()>(&mut publisher, &context, None, None).await;

        // First message is the prologue, skip it
        let _prologue = rx.try_recv().expect("Expected prologue message");

        // Second message is the complete_final
        let msg = rx.try_recv().expect("Expected complete_final message to be sent");
        let data = msg.data().expect("Expected message to have data");

        let wrapper: NetworkStreamWrapper<()> =
            serde_json::from_slice(data).expect("Failed to deserialize NetworkStreamWrapper");

        assert!(wrapper.data.is_none(), "Expected data to be None");
        assert!(wrapper.complete_final, "Expected complete_final to be true");
    }

    #[tokio::test]
    async fn test_send_complete_final_marker_increments_response_bytes() {
        let (mut publisher, _rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();
        let metrics = create_test_metrics();

        assert_eq!(metrics.response_bytes.get(), 0);

        send_complete_final_marker::<()>(&mut publisher, &context, Some(&metrics), None).await;

        // The serialized NetworkStreamWrapper { data: None, complete_final: true } has some bytes
        assert!(metrics.response_bytes.get() > 0);
    }

    #[tokio::test]
    async fn test_send_complete_final_marker_notifies_health_check() {
        let (mut publisher, _rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;

        let context = create_test_context();
        let notifier = Arc::new(Notify::new());

        // Spawn a task that waits for notification
        let notifier_clone = notifier.clone();
        let handle = tokio::spawn(async move {
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                notifier_clone.notified(),
            )
            .await
        });

        send_complete_final_marker::<()>(&mut publisher, &context, None, Some(&notifier)).await;

        // The notification should have been sent
        let result = handle.await.unwrap();
        assert!(result.is_ok(), "Expected health check notification");
    }

    #[tokio::test]
    async fn test_send_complete_final_marker_increments_error_on_send_failure() {
        // Create a publisher and drop the receiver to simulate send failure
        let (mut publisher, rx) = StreamSender::new_test();
        let _ = publisher.send_prologue(None).await;
        drop(rx); // Now sends will fail

        let context = create_test_context();
        let metrics = create_test_metrics();

        assert_eq!(
            metrics
                .error_counter
                .with_label_values(&[work_handler::error_types::PUBLISH_FINAL])
                .get(),
            0
        );

        send_complete_final_marker::<()>(&mut publisher, &context, Some(&metrics), None).await;

        assert_eq!(
            metrics
                .error_counter
                .with_label_values(&[work_handler::error_types::PUBLISH_FINAL])
                .get(),
            1
        );
    }
}

