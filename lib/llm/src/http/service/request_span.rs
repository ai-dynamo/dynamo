// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `http-request` span of inference routes.
//!
//! This does what `TraceLayer::new_for_http().make_span_with(make_inference_request_span)`
//! does, except that the response body holds the span without entering it on every poll.
//! A streamed response's body is polled once or twice per token, and each enter/exit
//! costs two span-registry lookups. The body enters the span once more when it is dropped,
//! so the span's last exit, which tracing-opentelemetry takes as its end time, is at the end
//! of the response.

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::Request;
use axum::http::Response;
use axum::middleware::Next;
use dynamo_runtime::logging::make_inference_request_span;
use http_body::{Frame, SizeHint};
use tracing::{Instrument, Span};

use super::service_v2::log_response;

/// Middleware for the inference routes: their `http-request` span, with `TraceLayer`'s
/// default events. Those keep tower-http's targets, so `DYN_LOG` directives still match.
pub(crate) async fn inference_request_span(request: Request, next: Next) -> Response<Body> {
    let span = make_inference_request_span(&request);
    let start = Instant::now();
    let response = async move {
        tracing::debug!(target: "tower_http::trace::on_request", "started processing request");
        let response = next.run(request).await;
        let latency = start.elapsed();
        log_response(&response, latency);
        if response.status().is_server_error() {
            tracing::error!(
                target: "tower_http::trace::on_failure",
                classification = %format_args!("Status code: {}", response.status()),
                latency = %format_args!("{} ms", latency.as_millis()),
                "response failed"
            );
        }
        response
    }
    .instrument(span.clone())
    .await;
    response.map(|inner| {
        Body::new(SpanBody {
            inner,
            span,
            stream_start: Some(Instant::now()),
        })
    })
}

/// A response body that keeps its request span open until the body is dropped.
struct SpanBody {
    inner: Body,
    span: Span,
    /// Taken by the end-of-stream event.
    stream_start: Option<Instant>,
}

impl HttpBody for SpanBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        let frame = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(None) = frame
            && let Some(stream_start) = this.stream_start.take()
        {
            tracing::debug!(
                target: "tower_http::trace::on_eos",
                parent: &this.span,
                stream_duration = %format_args!("{} ms", stream_start.elapsed().as_millis()),
                "end of stream"
            );
        }
        frame
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for SpanBody {
    fn drop(&mut self) {
        // The response stream's own cleanup (a cancelled request's) runs in the span, and this
        // exit is the span's last.
        let _entered = self.span.enter();
        self.inner = Body::empty();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::routing::get;
    use futures::stream;
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use tracing::span;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::{Context as TraceContext, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    use super::*;

    #[derive(Default)]
    struct Counts {
        enters: AtomicUsize,
        closes: AtomicUsize,
    }

    struct CountLayer(Arc<Counts>);

    impl<S> Layer<S> for CountLayer
    where
        S: tracing::Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn on_enter(&self, id: &span::Id, ctx: TraceContext<'_, S>) {
            if ctx.span(id).is_some_and(|s| s.name() == "http-request") {
                self.0.enters.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn on_close(&self, id: span::Id, ctx: TraceContext<'_, S>) {
            if ctx.span(&id).is_some_and(|s| s.name() == "http-request") {
                self.0.closes.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    #[tokio::test]
    async fn streamed_body_enters_request_span_once_and_closes_it_on_drop() {
        const CHUNKS: usize = 64;
        let counts = Arc::new(Counts::default());
        let _interest = crate::test_utils::tracing_interest_guard();
        let _subscriber = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(CountLayer(counts.clone())),
        );

        let app = Router::new()
            .route(
                "/v1/chat/completions",
                get(|| async {
                    Body::from_stream(stream::iter(
                        (0..CHUNKS).map(|i| Ok::<_, std::io::Error>(format!("data: {i}\n\n"))),
                    ))
                }),
            )
            .layer(axum::middleware::from_fn(inference_request_span));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/chat/completions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let entered_by_handler = counts.enters.load(Ordering::SeqCst);

        let mut body = response.into_body();
        let mut frames = 0;
        while let Some(frame) = body.frame().await {
            frame.unwrap();
            frames += 1;
        }
        assert_eq!(frames, CHUNKS);
        assert_eq!(counts.enters.load(Ordering::SeqCst), entered_by_handler);
        assert_eq!(counts.closes.load(Ordering::SeqCst), 0);

        drop(body);
        assert_eq!(counts.enters.load(Ordering::SeqCst), entered_by_handler + 1);
        assert_eq!(counts.closes.load(Ordering::SeqCst), 1);
    }
}
