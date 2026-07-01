// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side tracing for one routed request attempt.

use std::error::Error as StdError;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context as TaskContext, Poll};

use futures::Stream;

use crate::engine::{AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data};
use crate::error::{BackendError, DynamoError, ErrorType};
use crate::pipeline::{Context, ManyOut, ResponseStream};
use crate::protocols::maybe_error::MaybeError;

const ROUTE_TRACE_CONTEXT_KEY: &str = "dynamo.route_trace_context";

/// Retry metadata shared between the migration layer and the selected router.
///
/// The router fills in `selected_worker_id` once routing completes. The migration
/// layer retains the same `Arc`, allowing the next attempt to attribute a
/// disconnect to the worker selected for the previous attempt.
#[derive(Debug)]
pub struct RouteTraceContext {
    attempt: u32,
    migration_reason: Option<ErrorType>,
    from_worker_id: Option<u64>,
    tokens_completed: usize,
    selected_worker_id: AtomicU64,
    has_selected_worker_id: AtomicBool,
}

impl RouteTraceContext {
    pub fn new(
        attempt: u32,
        migration_reason: Option<ErrorType>,
        from_worker_id: Option<u64>,
        tokens_completed: usize,
    ) -> Self {
        Self {
            attempt,
            migration_reason,
            from_worker_id,
            tokens_completed,
            selected_worker_id: AtomicU64::new(0),
            has_selected_worker_id: AtomicBool::new(false),
        }
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    pub fn migration_reason(&self) -> Option<ErrorType> {
        self.migration_reason
    }

    pub fn from_worker_id(&self) -> Option<u64> {
        self.from_worker_id
    }

    pub fn tokens_completed(&self) -> usize {
        self.tokens_completed
    }

    pub fn is_retry(&self) -> bool {
        self.attempt > 0
    }

    pub fn set_selected_worker_id(&self, worker_id: u64) {
        self.selected_worker_id.store(worker_id, Ordering::Relaxed);
        self.has_selected_worker_id.store(true, Ordering::Release);
    }

    pub fn selected_worker_id(&self) -> Option<u64> {
        self.has_selected_worker_id
            .load(Ordering::Acquire)
            .then(|| self.selected_worker_id.load(Ordering::Relaxed))
    }
}

/// Attach route tracing metadata to a request and return the shared handle that
/// will later contain the selected worker ID.
pub fn attach_route_trace_context<T: Data>(
    request: &mut Context<T>,
    trace_context: RouteTraceContext,
) -> Arc<RouteTraceContext> {
    request.insert(ROUTE_TRACE_CONTEXT_KEY, trace_context);
    request
        .get::<RouteTraceContext>(ROUTE_TRACE_CONTEXT_KEY)
        .expect("route trace context was just inserted")
}

pub fn get_route_trace_context<T: Data>(request: &Context<T>) -> Option<Arc<RouteTraceContext>> {
    request
        .get_optional::<RouteTraceContext>(ROUTE_TRACE_CONTEXT_KEY)
        .ok()
        .flatten()
}

/// Populate fields common to the default and KV routing spans.
pub fn record_route_span_start(
    span: &tracing::Span,
    trace_context: Option<&RouteTraceContext>,
    worker_id: u64,
) {
    if let Some(trace_context) = trace_context {
        trace_context.set_selected_worker_id(worker_id);
    }
    if span.is_disabled() {
        return;
    }

    span.record("worker_id", worker_id);
    let attempt = trace_context.map_or(0, RouteTraceContext::attempt);
    span.record("request.attempt", attempt);
    span.record(
        "migration.is_retry",
        trace_context.is_some_and(RouteTraceContext::is_retry),
    );
    if let Some(trace_context) = trace_context {
        span.record(
            "migration.tokens_completed",
            trace_context.tokens_completed(),
        );
        if let Some(reason) = trace_context.migration_reason() {
            span.record("migration.reason", error_type_name(reason));
        }
        if let Some(from_worker_id) = trace_context.from_worker_id() {
            span.record("migration.from_worker_id", from_worker_id);
        }
    }
}

/// Mark a route span when dispatch fails before a response stream is returned.
pub fn record_route_error(span: &tracing::Span, err: &(dyn StdError + 'static)) {
    if span.is_disabled() {
        return;
    }
    let error_type = error_type_from_chain(err);
    if is_cancelled(error_type) {
        finish_cancelled(span, "context");
    } else {
        finish_error(span, error_type);
    }
}

/// Keep the client-side route span alive until the response stream reaches a
/// terminal outcome. This is intentionally outside fault-detection wrappers so
/// synthetic disconnect/timeout items are reflected on the route span.
pub fn wrap_route_span<U>(stream: ManyOut<U>, span: tracing::Span) -> ManyOut<U>
where
    U: Data + MaybeError,
{
    if span.is_disabled() {
        return stream;
    }
    let context = stream.context();
    ResponseStream::new(
        Box::pin(RouteSpanStream {
            inner: stream,
            context: context.clone(),
            span: Some(span),
        }),
        context,
    )
}

pub fn error_type_from_chain(err: &(dyn StdError + 'static)) -> ErrorType {
    let mut current = Some(err);
    let mut first = None;
    while let Some(error) = current {
        if let Some(dynamo_error) = error.downcast_ref::<DynamoError>() {
            let error_type = dynamo_error.error_type();
            first.get_or_insert(error_type);
            if error_type != ErrorType::Unknown {
                return error_type;
            }
        }
        current = error.source();
    }
    first.unwrap_or(ErrorType::Unknown)
}

pub fn error_type_name(error_type: ErrorType) -> &'static str {
    match error_type {
        ErrorType::Unknown => "unknown",
        ErrorType::InvalidArgument => "invalid_argument",
        ErrorType::CannotConnect => "cannot_connect",
        ErrorType::Disconnected => "disconnected",
        ErrorType::ConnectionTimeout => "connection_timeout",
        ErrorType::ResponseTimeout => "response_timeout",
        ErrorType::Cancelled => "cancelled",
        ErrorType::ResourceExhausted => "resource_exhausted",
        ErrorType::Unavailable => "unavailable",
        ErrorType::Backend(BackendError::Unknown) => "backend_unknown",
        ErrorType::Backend(BackendError::InvalidArgument) => "backend_invalid_argument",
        ErrorType::Backend(BackendError::CannotConnect) => "backend_cannot_connect",
        ErrorType::Backend(BackendError::Disconnected) => "backend_disconnected",
        ErrorType::Backend(BackendError::ConnectionTimeout) => "backend_connection_timeout",
        ErrorType::Backend(BackendError::ResponseTimeout) => "backend_response_timeout",
        ErrorType::Backend(BackendError::Cancelled) => "backend_cancelled",
        ErrorType::Backend(BackendError::EngineShutdown) => "engine_shutdown",
        ErrorType::Backend(BackendError::StreamIncomplete) => "stream_incomplete",
    }
}

fn is_cancelled(error_type: ErrorType) -> bool {
    matches!(
        error_type,
        ErrorType::Cancelled | ErrorType::Backend(BackendError::Cancelled)
    )
}

fn error_outcome(error_type: ErrorType) -> &'static str {
    match error_type {
        ErrorType::Disconnected
        | ErrorType::Backend(BackendError::Disconnected)
        | ErrorType::Backend(BackendError::EngineShutdown)
        | ErrorType::Backend(BackendError::StreamIncomplete) => "worker_disconnected",
        ErrorType::CannotConnect | ErrorType::Backend(BackendError::CannotConnect) => {
            "connection_failed"
        }
        ErrorType::ConnectionTimeout
        | ErrorType::ResponseTimeout
        | ErrorType::Backend(BackendError::ConnectionTimeout)
        | ErrorType::Backend(BackendError::ResponseTimeout) => "timeout",
        ErrorType::InvalidArgument | ErrorType::Backend(BackendError::InvalidArgument) => {
            "rejected"
        }
        ErrorType::ResourceExhausted => "rejected",
        ErrorType::Unavailable => "unavailable",
        ErrorType::Cancelled | ErrorType::Backend(BackendError::Cancelled) => "cancelled",
        ErrorType::Unknown | ErrorType::Backend(BackendError::Unknown) => "error",
    }
}

fn finish_success(span: &tracing::Span) {
    span.record("request.outcome", "success");
    tracing::info!(
        target: "request_span",
        parent: span,
        { { "request.outcome" } = "success" },
        "route request completed"
    );
}

fn finish_cancelled(span: &tracing::Span, signal: &'static str) {
    span.record("request.outcome", "cancelled");
    span.record("cancellation.signal", signal);
    tracing::info!(
        target: "request_span",
        parent: span,
        {
            { "request.outcome" } = "cancelled",
            { "cancellation.signal" } = signal
        },
        "route request cancelled"
    );
}

fn finish_error(span: &tracing::Span, error_type: ErrorType) {
    let error_name = error_type_name(error_type);
    let outcome = error_outcome(error_type);
    span.record("request.outcome", outcome);
    span.record("error.type", error_name);
    span.record("otel.status_code", "error");
    span.record("otel.status_description", error_name);
    tracing::warn!(
        target: "request_span",
        parent: span,
        {
            { "request.outcome" } = outcome,
            { "error.type" } = error_name
        },
        "route request failed"
    );
}

struct RouteSpanStream<U: Data> {
    inner: ManyOut<U>,
    context: Arc<dyn AsyncEngineContext>,
    span: Option<tracing::Span>,
}

impl<U: Data> RouteSpanStream<U> {
    fn finish_with(&mut self, finish: impl FnOnce(&tracing::Span)) {
        if let Some(span) = self.span.take() {
            // tracing-opentelemetry uses the most recent span exit as the OTel
            // end timestamp. Re-enter only for terminal bookkeeping so a long
            // response stream does not appear to end after dispatch setup.
            let _guard = span.enter();
            finish(&span);
        }
    }

    fn finish_from_context(&mut self) -> bool {
        if self.context.is_killed() {
            self.finish_with(|span| finish_cancelled(span, "kill"));
            true
        } else if self.context.is_stopped() {
            self.finish_with(|span| finish_cancelled(span, "stop"));
            true
        } else {
            false
        }
    }
}

impl<U> Stream for RouteSpanStream<U>
where
    U: Data + MaybeError,
{
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);

        match &poll {
            Poll::Ready(Some(item)) => {
                if let Some(error) = item.err() {
                    let error_type = error.error_type();
                    if is_cancelled(error_type) {
                        self.finish_with(|span| finish_cancelled(span, "context"));
                    } else {
                        self.finish_with(|span| finish_error(span, error_type));
                    }
                }
            }
            Poll::Ready(None) => {
                if !self.finish_from_context() {
                    self.finish_with(finish_success);
                }
            }
            Poll::Pending => {}
        }
        poll
    }
}

impl<U: Data> Drop for RouteSpanStream<U> {
    fn drop(&mut self) {
        if self.span.is_some() && !self.finish_from_context() {
            // The HTTP disconnect monitor kills the context asynchronously after
            // its response body is dropped, so the context may still look live
            // here. An unfinished consumer drop is nevertheless a cancellation,
            // not a routed-worker failure.
            self.finish_with(|span| finish_cancelled(span, "stream_drop"));
        }
    }
}

impl<U: Data> fmt::Debug for RouteSpanStream<U> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RouteSpanStream").finish()
    }
}

impl<U: Data> AsyncEngineContextProvider for RouteSpanStream<U> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.context.clone()
    }
}

impl<U> AsyncEngineStream<U> for RouteSpanStream<U> where U: Data + MaybeError {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::{StreamExt, stream};
    use tracing::field::{Field, Visit};
    use tracing::span;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::{Context as TraceContext, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::util::SubscriberInitExt;

    use super::*;

    #[derive(Clone, Debug)]
    struct TestItem {
        error: Option<DynamoError>,
    }

    impl TestItem {
        fn ok() -> Self {
            Self { error: None }
        }

        fn error(error_type: ErrorType) -> Self {
            Self {
                error: Some(
                    DynamoError::builder()
                        .error_type(error_type)
                        .message("test failure")
                        .build(),
                ),
            }
        }
    }

    impl MaybeError for TestItem {
        fn from_err(err: impl StdError + 'static) -> Self {
            Self {
                error: Some(DynamoError::from(&err as &(dyn StdError + 'static))),
            }
        }

        fn err(&self) -> Option<DynamoError> {
            self.error.clone()
        }
    }

    #[derive(Default)]
    struct Captured {
        fields: Mutex<HashMap<String, String>>,
        closed: AtomicUsize,
    }

    struct FieldRecorder<'a>(&'a mut HashMap<String, String>);

    impl Visit for FieldRecorder<'_> {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_bool(&mut self, field: &Field, value: bool) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    struct CaptureLayer(Arc<Captured>);

    impl<S> Layer<S> for CaptureLayer
    where
        S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            attrs: &span::Attributes<'_>,
            _id: &span::Id,
            _ctx: TraceContext<'_, S>,
        ) {
            if attrs.metadata().name() == "test.route" {
                attrs.record(&mut FieldRecorder(&mut self.0.fields.lock().unwrap()));
            }
        }

        fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: TraceContext<'_, S>) {
            if ctx.span(id).is_some_and(|span| span.name() == "test.route") {
                values.record(&mut FieldRecorder(&mut self.0.fields.lock().unwrap()));
            }
        }

        fn on_close(&self, id: span::Id, ctx: TraceContext<'_, S>) {
            if ctx
                .span(&id)
                .is_some_and(|span| span.name() == "test.route")
            {
                self.0.closed.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    fn test_span() -> tracing::Span {
        tracing::info_span!(
            "test.route",
            otel.kind = "client",
            worker_id = tracing::field::Empty,
            "request.attempt" = tracing::field::Empty,
            "request.outcome" = tracing::field::Empty,
            "migration.is_retry" = tracing::field::Empty,
            "migration.reason" = tracing::field::Empty,
            "migration.from_worker_id" = tracing::field::Empty,
            "migration.tokens_completed" = tracing::field::Empty,
            "cancellation.signal" = tracing::field::Empty,
            "error.type" = tracing::field::Empty,
            otel.status_code = tracing::field::Empty,
            otel.status_description = tracing::field::Empty,
        )
    }

    fn response_stream(items: Vec<TestItem>) -> ManyOut<TestItem> {
        let context = Context::new(()).context();
        ResponseStream::new(Box::pin(stream::iter(items)), context)
    }

    fn reset(captured: &Captured) {
        captured.fields.lock().unwrap().clear();
        captured.closed.store(0, Ordering::SeqCst);
    }

    fn field(captured: &Captured, name: &str) -> Option<String> {
        captured.fields.lock().unwrap().get(name).cloned()
    }

    #[tokio::test]
    async fn route_span_covers_attempt_lifecycle_and_retry_metadata() {
        let captured = Arc::new(Captured::default());
        let _subscriber = tracing_subscriber::registry()
            .with(CaptureLayer(captured.clone()))
            .set_default();

        let mut request = Context::new(());
        let trace_context = attach_route_trace_context(
            &mut request,
            RouteTraceContext::new(2, Some(ErrorType::Disconnected), Some(17), 42),
        );
        let span = test_span();
        record_route_span_start(&span, Some(&trace_context), 23);
        let mut routed = wrap_route_span(response_stream(vec![TestItem::ok()]), span);

        assert_eq!(trace_context.selected_worker_id(), Some(23));
        assert_eq!(captured.closed.load(Ordering::SeqCst), 0);
        assert!(routed.next().await.is_some());
        assert_eq!(captured.closed.load(Ordering::SeqCst), 0);
        assert!(routed.next().await.is_none());
        assert_eq!(captured.closed.load(Ordering::SeqCst), 1);
        assert_eq!(
            field(&captured, "request.outcome").as_deref(),
            Some("success")
        );
        assert_eq!(field(&captured, "request.attempt").as_deref(), Some("2"));
        assert_eq!(
            field(&captured, "migration.is_retry").as_deref(),
            Some("true")
        );
        assert_eq!(
            field(&captured, "migration.reason").as_deref(),
            Some("disconnected")
        );
        assert_eq!(
            field(&captured, "migration.from_worker_id").as_deref(),
            Some("17")
        );
        assert_eq!(
            field(&captured, "migration.tokens_completed").as_deref(),
            Some("42")
        );

        reset(&captured);
        let mut disconnected = wrap_route_span(
            response_stream(vec![TestItem::error(ErrorType::Disconnected)]),
            test_span(),
        );
        assert!(disconnected.next().await.is_some());
        assert_eq!(captured.closed.load(Ordering::SeqCst), 1);
        assert_eq!(
            field(&captured, "request.outcome").as_deref(),
            Some("worker_disconnected")
        );
        assert_eq!(
            field(&captured, "error.type").as_deref(),
            Some("disconnected")
        );
        assert_eq!(
            field(&captured, "otel.status_code").as_deref(),
            Some("error")
        );

        reset(&captured);
        let mut handled_error = wrap_route_span(
            response_stream(vec![TestItem::error(ErrorType::Backend(
                BackendError::InvalidArgument,
            ))]),
            test_span(),
        );
        assert!(handled_error.next().await.is_some());
        assert_eq!(captured.closed.load(Ordering::SeqCst), 1);
        assert_eq!(
            field(&captured, "request.outcome").as_deref(),
            Some("rejected")
        );
        assert_eq!(
            field(&captured, "error.type").as_deref(),
            Some("backend_invalid_argument")
        );

        reset(&captured);
        let context = Context::new(()).context();
        let inner = ResponseStream::new(Box::pin(stream::pending::<TestItem>()), context.clone());
        let cancelled = wrap_route_span(inner, test_span());
        context.stop();
        drop(cancelled);
        assert_eq!(captured.closed.load(Ordering::SeqCst), 1);
        assert_eq!(
            field(&captured, "request.outcome").as_deref(),
            Some("cancelled")
        );
        assert_eq!(
            field(&captured, "cancellation.signal").as_deref(),
            Some("stop")
        );

        reset(&captured);
        let context = Context::new(()).context();
        let inner = ResponseStream::new(Box::pin(stream::pending::<TestItem>()), context.clone());
        let killed = wrap_route_span(inner, test_span());
        context.kill();
        drop(killed);
        assert_eq!(captured.closed.load(Ordering::SeqCst), 1);
        assert_eq!(
            field(&captured, "request.outcome").as_deref(),
            Some("cancelled")
        );
        assert_eq!(
            field(&captured, "cancellation.signal").as_deref(),
            Some("kill")
        );

        reset(&captured);
        let dropped = wrap_route_span(
            ResponseStream::new(
                Box::pin(stream::pending::<TestItem>()),
                Context::new(()).context(),
            ),
            test_span(),
        );
        drop(dropped);
        assert_eq!(captured.closed.load(Ordering::SeqCst), 1);
        assert_eq!(
            field(&captured, "request.outcome").as_deref(),
            Some("cancelled")
        );
        assert_eq!(
            field(&captured, "cancellation.signal").as_deref(),
            Some("stream_drop")
        );
    }
}
