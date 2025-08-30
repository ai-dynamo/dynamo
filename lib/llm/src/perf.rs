// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Performance recording and analysis for streaming LLM responses
//!
//! This module provides mechanisms to record streaming responses with minimal overhead
//! during collection, then analyze the recorded data for performance insights.

pub mod logprobs;

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

// Import the runtime types we need
use dynamo_runtime::engine::{
    AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data, DataStream,
    EngineStream, ResponseStream,
};
use std::sync::Arc;

/// Type alias for a receiver of recorded stream data
pub type RecordedStreamReceiver<R> = oneshot::Receiver<RecordedStream<R>>;

/// Type alias for the return type of recording functions
pub type RecordingResult<R> = (EngineStream<R>, RecordedStreamReceiver<R>);

/// A response wrapper that adds timing information with minimal overhead
#[derive(Debug, Clone)]
pub struct TimestampedResponse<T> {
    /// The actual response data
    pub response: T,
    /// High-resolution timestamp when this response was recorded
    pub timestamp: Instant,
    /// Sequence number in the stream (0-based)
    pub sequence_number: usize,
}

impl<T> TimestampedResponse<T> {
    /// Create a new timestamped response
    pub fn new(response: T, sequence_number: usize) -> Self {
        Self {
            response,
            timestamp: Instant::now(),
            sequence_number,
        }
    }

    /// Get the response data
    pub fn data(&self) -> &T {
        &self.response
    }

    /// Get the elapsed time since stream start
    pub fn elapsed_since(&self, start_time: Instant) -> Duration {
        self.timestamp.duration_since(start_time)
    }
}

/// Trait for requests that can provide hints about expected response count
/// This enables capacity pre-allocation for better performance
pub trait CapacityHint {
    /// Estimate the number of responses this request might generate
    /// Returns None if estimation is not possible
    fn estimated_response_count(&self) -> Option<usize>;
}

/// Recording mode determines how the recorder behaves with the stream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingMode {
    /// Pass responses through while recording (scan mode)
    /// Stream continues to flow to downstream consumers
    Scan,
    /// Consume responses as terminus (sink mode)
    /// Stream ends at the recorder
    Sink,
}

/// Container for recorded streaming responses.
/// This forms the core object on which analysis is performed.
#[derive(Debug, Clone)]
pub struct RecordedStream<T> {
    /// All recorded responses with timestamps
    responses: Vec<TimestampedResponse<T>>,

    /// When recording started
    start_time: Instant,

    /// When recording ended
    end_time: Instant,
}

impl<T> RecordedStream<T> {
    /// Create a new recorded stream from collected responses
    pub fn new(
        responses: Vec<TimestampedResponse<T>>,
        start_time: Instant,
        end_time: Instant,
    ) -> Self {
        Self {
            responses,
            start_time,
            end_time,
        }
    }

    /// Get the number of responses recorded
    pub fn response_count(&self) -> usize {
        self.responses.len()
    }

    /// Get the total duration of the stream
    pub fn total_duration(&self) -> Duration {
        self.end_time.duration_since(self.start_time)
    }

    /// Get the responses recorded
    pub fn responses(&self) -> &[TimestampedResponse<T>] {
        &self.responses
    }

    /// Get the start time of the stream
    pub fn start_time(&self) -> &Instant {
        &self.start_time
    }

    /// Get the end time of the stream
    pub fn end_time(&self) -> &Instant {
        &self.end_time
    }
}

/// Recording stream that wraps an AsyncEngineStream and records responses
/// Following the pattern of ResponseStream for AsyncEngine compatibility
pub struct RecordingStream<R: Data> {
    /// The wrapped stream
    stream: DataStream<R>,
    /// Context from the original stream
    ctx: Arc<dyn AsyncEngineContext>,
    /// Recording mode
    mode: RecordingMode,
    /// Recorded responses
    responses: Vec<TimestampedResponse<R>>,
    /// When recording started
    start_time: Instant,
    /// Channel to send recorded data when stream completes
    recorded_tx: Option<oneshot::Sender<RecordedStream<R>>>,
}

impl<R: Data> Unpin for RecordingStream<R> {}

impl<R: Data + Clone> RecordingStream<R> {
    /// Create a new recording stream from a raw stream and context
    pub fn from_stream_and_context(
        stream: DataStream<R>,
        ctx: Arc<dyn AsyncEngineContext>,
        mode: RecordingMode,
        capacity: Option<usize>,
        recorded_tx: oneshot::Sender<RecordedStream<R>>,
    ) -> Self {
        let mut responses = Vec::new();
        if let Some(cap) = capacity {
            responses.reserve(cap);
        }

        Self {
            stream,
            ctx,
            mode,
            responses,
            start_time: Instant::now(),
            recorded_tx: Some(recorded_tx),
        }
    }

    /// Create a new recording stream from an AsyncEngineStream (private constructor)
    fn from_async_engine_stream(
        stream: EngineStream<R>,
        mode: RecordingMode,
        capacity: Option<usize>,
        recorded_tx: oneshot::Sender<RecordedStream<R>>,
    ) -> Self {
        let ctx = stream.context();
        Self::from_stream_and_context(stream, ctx, mode, capacity, recorded_tx)
    }

    /// Convert to Pin<Box<dyn AsyncEngineStream<R>>>
    pub fn into_async_engine_stream(self) -> EngineStream<R> {
        Box::pin(self)
    }
}

impl<R: Data + Clone> Stream for RecordingStream<R> {
    type Item = R;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();

        match Pin::new(&mut this.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                // Always capture timestamp first (cheap operation)
                let timestamp = Instant::now();
                let sequence_number = this.responses.len();

                match this.mode {
                    RecordingMode::Scan => {
                        // Clone for recording, pass original through
                        let timestamped = TimestampedResponse {
                            response: item.clone(),
                            timestamp,
                            sequence_number,
                        };
                        this.responses.push(timestamped);
                        Poll::Ready(Some(item)) // Pass through original
                    }
                    RecordingMode::Sink => {
                        // Move item directly into recording (no clone needed)
                        let timestamped = TimestampedResponse {
                            response: item, // Move, don't clone
                            timestamp,
                            sequence_number,
                        };
                        this.responses.push(timestamped);

                        // Continue consuming but don't emit
                        // self.poll_next(cx)
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }
            }
            Poll::Ready(None) => {
                // Stream ended - send recorded data
                if let Some(tx) = this.recorded_tx.take() {
                    let recorded = RecordedStream::new(
                        std::mem::take(&mut this.responses),
                        this.start_time,
                        Instant::now(),
                    );
                    let _ = tx.send(recorded); // Ignore if receiver dropped
                }

                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<R: Data + Clone> AsyncEngineStream<R> for RecordingStream<R> {}

impl<R: Data + Clone> AsyncEngineContextProvider for RecordingStream<R> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.ctx.clone()
    }
}

impl<R: Data + Clone> std::fmt::Debug for RecordingStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingStream")
            .field("mode", &self.mode)
            .field("responses_count", &self.responses.len())
            .field("ctx", &self.ctx)
            .finish()
    }
}

/// Create a recording stream that wraps an AsyncEngineStream
/// Returns a pinned stream and a receiver for the recorded data
pub fn record_stream<R: Data + Clone>(
    stream: EngineStream<R>,
    mode: RecordingMode,
) -> RecordingResult<R> {
    let (tx, rx) = oneshot::channel();
    let recording_stream = RecordingStream::from_async_engine_stream(stream, mode, None, tx);
    let boxed_stream = Box::pin(recording_stream);
    (boxed_stream, rx)
}

/// Create a recording stream from a raw stream and context
/// Returns a pinned stream and a receiver for the recorded data
pub fn record_stream_with_context<R: Data + Clone>(
    stream: DataStream<R>,
    ctx: Arc<dyn AsyncEngineContext>,
    mode: RecordingMode,
) -> RecordingResult<R> {
    let (tx, rx) = oneshot::channel();
    let recording_stream = RecordingStream::from_stream_and_context(stream, ctx, mode, None, tx);
    let boxed_stream = Box::pin(recording_stream);
    (boxed_stream, rx)
}

/// Create a recording stream with capacity hint
pub fn record_stream_with_capacity<R: Data + Clone>(
    stream: EngineStream<R>,
    mode: RecordingMode,
    capacity: usize,
) -> RecordingResult<R> {
    let (tx, rx) = oneshot::channel();
    let recording_stream =
        RecordingStream::from_async_engine_stream(stream, mode, Some(capacity), tx);
    let boxed_stream = Box::pin(recording_stream);
    (boxed_stream, rx)
}

/// Create a recording stream with capacity hint from request
pub fn record_stream_with_request_hint<R: Data + Clone, Req: CapacityHint>(
    stream: EngineStream<R>,
    mode: RecordingMode,
    request: &Req,
) -> RecordingResult<R> {
    let capacity = request.estimated_response_count();
    match capacity {
        Some(cap) => record_stream_with_capacity(stream, mode, cap),
        None => record_stream(stream, mode),
    }
}

/// Create a recording stream from a raw stream and context with capacity hint
pub fn record_stream_with_context_and_capacity<R: Data + Clone>(
    stream: DataStream<R>,
    ctx: Arc<dyn AsyncEngineContext>,
    mode: RecordingMode,
    capacity: usize,
) -> RecordingResult<R> {
    let (tx, rx) = oneshot::channel();
    let recording_stream =
        RecordingStream::from_stream_and_context(stream, ctx, mode, Some(capacity), tx);
    let boxed_stream = Box::pin(recording_stream);
    (boxed_stream, rx)
}

/// Create a recording stream from ResponseStream (convenience wrapper)
pub fn record_response_stream<R: Data + Clone>(
    response_stream: Pin<Box<ResponseStream<R>>>,
    mode: RecordingMode,
) -> RecordingResult<R> {
    record_stream(response_stream, mode)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use dynamo_runtime::engine::ResponseStream;
    use futures::stream;
    use std::time::Duration;

    #[test]
    fn test_timestamped_response_creation() {
        let response = "test response";
        let timestamped = TimestampedResponse::new(response, 0);

        assert_eq!(timestamped.response, response);
        assert_eq!(timestamped.sequence_number, 0);
        assert_eq!(timestamped.data(), &response);
    }

    #[test]
    fn test_recorded_stream_analysis() {
        let start_time = Instant::now();

        // Create mock responses with known timing
        let responses = vec![
            TimestampedResponse {
                response: "response1",
                timestamp: start_time,
                sequence_number: 0,
            },
            TimestampedResponse {
                response: "response2",
                timestamp: start_time + Duration::from_millis(100),
                sequence_number: 1,
            },
            TimestampedResponse {
                response: "response3",
                timestamp: start_time + Duration::from_millis(250),
                sequence_number: 2,
            },
        ];

        let end_time = start_time + Duration::from_millis(250);
        let recorded = RecordedStream::new(responses, start_time, end_time);

        assert_eq!(recorded.response_count(), 3);
        assert_eq!(recorded.total_duration(), Duration::from_millis(250));
    }

    #[test]
    fn test_performance_metrics_conversion() {
        let start_time = Instant::now();
        let responses = vec![
            TimestampedResponse {
                response: "test",
                timestamp: start_time + Duration::from_millis(50),
                sequence_number: 0,
            },
            TimestampedResponse {
                response: "test",
                timestamp: start_time + Duration::from_millis(150),
                sequence_number: 1,
            },
        ];

        let end_time = start_time + Duration::from_millis(150);
        let recorded = RecordedStream::new(responses, start_time, end_time);

        assert_eq!(recorded.response_count(), 2);
        assert_eq!(recorded.total_duration(), Duration::from_millis(150));
    }

    #[tokio::test]
    async fn test_recording_stream_scan_mode() {
        use futures::StreamExt;

        // Create a simple test stream
        let test_data = vec!["token1", "token2", "token3"];
        let base_stream = stream::iter(test_data.clone());

        // Create a mock context for the stream
        let ctx = Arc::new(MockContext::new());

        // Record the stream in scan mode using the simplified API
        let (recorded_stream, recording_rx) =
            record_stream_with_context(Box::pin(base_stream), ctx, RecordingMode::Scan);

        // Consume the stream normally (pass-through mode)
        let collected_responses: Vec<_> = recorded_stream.collect().await;

        // Verify the responses passed through unchanged
        assert_eq!(collected_responses, test_data);

        // Get the recorded data
        let recorded = recording_rx.await.unwrap();
        assert_eq!(recorded.response_count(), 3);
        assert_eq!(recorded.responses[0].response, "token1");
        assert_eq!(recorded.responses[1].response, "token2");
        assert_eq!(recorded.responses[2].response, "token3");

        // Verify timing was recorded
        assert!(recorded.total_duration() > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_recording_stream_sink_mode() {
        use futures::StreamExt;

        // Create a simple test stream
        let test_data = vec!["token1", "token2", "token3"];
        let base_stream = stream::iter(test_data.clone());

        // Create a mock context for the stream
        let ctx = Arc::new(MockContext::new());

        // Record the stream in sink mode using the simplified API
        let (recorded_stream, recording_rx) =
            record_stream_with_context(Box::pin(base_stream), ctx, RecordingMode::Sink);

        // In sink mode, the stream should complete without emitting items
        let collected_responses: Vec<_> = recorded_stream.collect().await;
        assert_eq!(collected_responses, Vec::<&str>::new());

        // Get the recorded data - should contain all original items
        let recorded = recording_rx.await.unwrap();
        assert_eq!(recorded.response_count(), 3);
        assert_eq!(recorded.responses[0].response, "token1");
        assert_eq!(recorded.responses[1].response, "token2");
        assert_eq!(recorded.responses[2].response, "token3");

        // Verify timing was recorded
        assert!(recorded.total_duration() > Duration::from_nanos(0));
    }

    #[tokio::test]
    async fn test_recording_stream_from_response_stream() {
        use futures::StreamExt;

        // Create a simple test stream
        let test_data = vec!["token1", "token2", "token3"];
        let base_stream = stream::iter(test_data.clone());

        // Create a ResponseStream (the traditional way)
        let ctx = Arc::new(MockContext::new());
        let response_stream = ResponseStream::new(Box::pin(base_stream), ctx);

        // Use the convenience API for ResponseStream
        let (recorded_stream, recording_rx) =
            record_response_stream(response_stream, RecordingMode::Scan);

        // Consume the stream normally (pass-through mode)
        let collected_responses: Vec<_> = recorded_stream.collect().await;

        // Verify the responses passed through unchanged
        assert_eq!(collected_responses, test_data);

        // Get the recorded data
        let recorded = recording_rx.await.unwrap();
        assert_eq!(recorded.response_count(), 3);
        assert_eq!(recorded.responses[0].response, "token1");
        assert_eq!(recorded.responses[1].response, "token2");
        assert_eq!(recorded.responses[2].response, "token3");

        // Verify timing was recorded
        assert!(recorded.total_duration() > Duration::from_nanos(0));
    }

    // Mock context for testing
    #[derive(Debug)]
    struct MockContext {
        id: String,
    }

    impl MockContext {
        fn new() -> Self {
            Self {
                id: "test-context".to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngineContext for MockContext {
        fn id(&self) -> &str {
            &self.id
        }

        fn stop(&self) {
            // No-op for testing
        }

        fn stop_generating(&self) {
            // No-op for testing
        }

        fn kill(&self) {
            // No-op for testing
        }

        fn is_stopped(&self) -> bool {
            false
        }

        fn is_killed(&self) -> bool {
            false
        }

        async fn stopped(&self) {
            // No-op for testing
        }

        async fn killed(&self) {
            // No-op for testing
        }

        fn link_child(&self, _: Arc<dyn AsyncEngineContext>) {
            // No-op for testing
        }
    }
}
