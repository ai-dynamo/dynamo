// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use tokio_stream::StreamExt;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use dynamo_llm::entrypoint::PrefillRoutedEngine;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_llm::protocols::common::timing::RequestTracker;
use dynamo_llm::request_trace;
use dynamo_runtime::logging::{DistributedTraceContext, otel_parent_context_from_distributed};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;

use crate::to_pyerr;

#[pyclass]
pub struct RoutedEngine {
    inner: PrefillRoutedEngine,
    kv_cache_block_size: usize,
}

impl RoutedEngine {
    pub fn new(inner: PrefillRoutedEngine, kv_cache_block_size: usize) -> Self {
        Self {
            inner,
            kv_cache_block_size,
        }
    }
}

struct PendingRequestEnd {
    data: Option<RequestEndData>,
}

struct RequestEndData {
    request_id: String,
    tracker: Arc<RequestTracker>,
    token_ids: Arc<Vec<dynamo_llm::protocols::TokenIdType>>,
    kv_cache_block_size: usize,
    replayable: bool,
}

impl PendingRequestEnd {
    fn tracker(&self) -> &RequestTracker {
        self.data
            .as_ref()
            .expect("request end data")
            .tracker
            .as_ref()
    }
}

impl Drop for PendingRequestEnd {
    fn drop(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };
        // The routed stream has been dropped before this guard, so the worker
        // and router have finished writing into the shared tracker. Hashing a
        // long prompt and writing a trace must never delay the client stream.
        if data.tracker.total_time_ms().is_none() {
            data.tracker.record_finish();
        }
        let RequestEndData {
            request_id,
            tracker,
            token_ids,
            kv_cache_block_size,
            replayable,
        } = data;
        let emit = move || {
            request_trace::emit_python_routed_request_end(
                request_id,
                &tracker,
                &token_ids,
                kv_cache_block_size,
                replayable,
            );
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn_blocking(emit);
        } else {
            emit();
        }
    }
}

fn replayable_text_request(request: &PreprocessedRequest, block_size: usize) -> bool {
    block_size != 0
        && request.prompt_embeds.is_none()
        && request.multi_modal_data.is_none()
        && request.multi_modal_uuids.is_none()
        && request.mm_routing_info.is_none()
        && request.media_io_kwargs.is_none()
        && !request.extra_args.as_ref().is_some_and(|args| {
            args.get("mm_placeholders").is_some()
                || args.get("mm_hashes").is_some()
                || args.get("expanded_token_ids").is_some()
        })
        && request.sampling_options.n.unwrap_or(1) == 1
        && request.sampling_options.best_of.unwrap_or(1) == 1
}

#[pymethods]
impl RoutedEngine {
    /// Send a preprocessed request through the Rust prefill-routed pipeline.
    #[pyo3(signature = (preprocessed, context=None))]
    fn generate<'p>(
        &self,
        py: Python<'p>,
        preprocessed: PyObject,
        context: Option<crate::context::Context>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let mut request: PreprocessedRequest =
            depythonize(preprocessed.bind(py)).map_err(to_pyerr)?;
        let trace = if request_trace::config::capture_enabled()
            && request_trace::policy().emit_request_end_records()
        {
            let tracker = request
                .tracker
                .get_or_insert_with(|| Arc::new(RequestTracker::new()))
                .clone();
            let token_ids = request.token_ids.clone();
            let replayable = replayable_text_request(&request, self.kv_cache_block_size);
            Some((tracker, token_ids, replayable))
        } else {
            None
        };
        let request_context = if let Some(parent_context) = context.as_ref() {
            let parent_metadata = parent_context.metadata_snapshot();
            let parent_context = parent_context.inner();
            let child_context = SingleIn::with_id_and_metadata(
                request,
                parent_context.id().to_string(),
                parent_metadata,
            );
            let child_controller = child_context.context();
            parent_context.link_child(child_controller.clone());
            if parent_context.is_killed() {
                child_controller.kill();
            } else if parent_context.is_stopped() {
                child_controller.stop_generating();
            }
            child_context
        } else {
            SingleIn::new(request)
        };
        let trace = trace.map(|(tracker, token_ids, replayable)| {
            (
                request_context.id().to_string(),
                tracker,
                token_ids,
                replayable,
            )
        });
        let kv_cache_block_size = self.kv_cache_block_size;
        let inner = self.inner.clone();

        // Re-parent onto the caller's trace: this future runs on a fresh
        // Tokio task where Span::current() is empty (ai-dynamo/dynamo#11397).
        let dispatch_span = dispatch_span(
            context.as_ref().and_then(|c| c.trace_context()),
            request_context.id(),
        );

        crate::future_into_py(
            py,
            async move {
                let mut stream = inner.generate(request_context).await.map_err(to_pyerr)?;
                let trace =
                    trace.map(
                        |(request_id, tracker, token_ids, replayable)| PendingRequestEnd {
                            data: Some(RequestEndData {
                                request_id,
                                tracker,
                                token_ids,
                                kv_cache_block_size,
                                replayable,
                            }),
                        },
                    );
                let task_context = stream.context();
                let (tx, rx) = tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(32);

                tokio::spawn(async move {
                    let mut output_tokens = 0_usize;
                    loop {
                        let response = tokio::select! {
                            _ = tx.closed() => {
                                task_context.stop_generating();
                                break;
                            }
                            response = stream.next() => response,
                        };

                        let Some(response) = response else {
                            break;
                        };

                        if let (Some(trace), Some(output)) =
                            (trace.as_ref(), response.data.as_ref())
                            && !output.token_ids.is_empty()
                        {
                            trace.tracker().record_first_token();
                            output_tokens = output_tokens.saturating_add(output.token_ids.len());
                        }

                        let py_response = Python::with_gil(|py| {
                            response.map_data(|data| {
                                pythonize(py, &data)
                                    .map(|obj| obj.unbind())
                                    .map_err(|e| format!("pythonize failed: {e}"))
                            })
                        });

                        if tx.send(py_response).await.is_err() {
                            task_context.stop_generating();
                            break;
                        }
                    }
                    drop(stream);
                    if let Some(trace) = trace.as_ref()
                        && trace.tracker().osl_tokens() == 0
                        && output_tokens > 0
                    {
                        trace.tracker().record_osl(output_tokens);
                    }
                    drop(trace);
                });

                Ok(crate::AsyncResponseStream::new(rx, true))
            }
            .instrument(dispatch_span),
        )
    }
}

fn dispatch_span(
    trace_context: Option<&DistributedTraceContext>,
    request_id: &str,
) -> tracing::Span {
    match trace_context {
        Some(tc) => {
            let span = tracing::info_span!(
                target: "request_span",
                "routed_engine.generate",
                request_id = request_id,
                trace_id = tc.trace_id.as_str(),
                parent_id = tc.span_id.as_str(),
                trace_flags = tc.trace_flags.as_str(),
                tracestate = tc.tracestate.as_deref(),
                x_request_id = tc.x_request_id.as_deref(),
            );
            if let Some(context) = otel_parent_context_from_distributed(tc) {
                let _ = span.set_parent(context);
            }
            span
        }
        None => tracing::Span::current(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_runtime::logging::{DistributedTraceIdLayer, inject_trace_headers_into_map};
    use opentelemetry::trace::{TraceContextExt, TraceId, TracerProvider as _};
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn replay_hashes_only_plain_single_choice_text() {
        let mut request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap();
        assert!(replayable_text_request(&request, 16));
        assert!(!replayable_text_request(&request, 0));

        request.multi_modal_data = Some(Default::default());
        assert!(!replayable_text_request(&request, 16));
        request.multi_modal_data = None;
        request.extra_args = Some(serde_json::json!({"mm_placeholders": []}));
        assert!(!replayable_text_request(&request, 16));
        request.extra_args = None;
        request.sampling_options.n = Some(2);
        assert!(!replayable_text_request(&request, 16));
    }

    #[test]
    fn request_end_preserves_router_finish_time() {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_finish();
        let router_elapsed_ms = tracker.total_time_ms();

        drop(PendingRequestEnd {
            data: Some(RequestEndData {
                request_id: "finished-request".to_string(),
                tracker: tracker.clone(),
                token_ids: Arc::new(Vec::new()),
                kv_cache_block_size: 0,
                replayable: false,
            }),
        });

        assert_eq!(tracker.total_time_ms(), router_elapsed_ms);
    }

    fn make_trace_context(
        trace_id: &str,
        span_id: &str,
        trace_flags: &str,
    ) -> DistributedTraceContext {
        serde_json::from_value(serde_json::json!({
            "trace_id": trace_id,
            "span_id": span_id,
            "trace_flags": trace_flags,
            "tracestate": "vendor=dynamo",
            "x_request_id": "xr-1",
        }))
        .expect("DistributedTraceContext deserializes from trace_id + span_id")
    }

    fn with_otel_subscriber<T>(f: impl FnOnce() -> T) -> T {
        let subscriber = tracing_subscriber::registry().with(tracing_opentelemetry::layer());
        tracing::subscriber::with_default(subscriber, f)
    }

    #[test]
    fn dispatch_span_reparents_to_captured_trace_context() {
        with_otel_subscriber(|| {
            let tc =
                make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "01");
            let span = dispatch_span(Some(&tc), "req-1");
            let otel_ctx = span.context();
            let span_context = otel_ctx.span().span_context().clone();
            assert_eq!(
                span_context.trace_id(),
                TraceId::from_hex("0123456789abcdef0123456789abcdef").unwrap(),
                "dispatch span must inherit the captured trace_id"
            );
            assert!(span_context.is_sampled());
        });
    }

    #[test]
    fn dispatch_span_preserves_unsampled_trace_flags() {
        with_otel_subscriber(|| {
            let tc =
                make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "00");
            let span = dispatch_span(Some(&tc), "req-unsampled");
            let otel_ctx = span.context();
            let span_context = otel_ctx.span().span_context().clone();
            assert_eq!(
                span_context.trace_id(),
                TraceId::from_hex("0123456789abcdef0123456789abcdef").unwrap(),
                "trace identity must propagate even when not sampled"
            );
            assert!(
                !span_context.is_sampled(),
                "a non-sampled parent must not be re-sampled across the Python boundary"
            );
        });
    }

    #[test]
    fn dispatch_span_tolerates_malformed_trace_ids() {
        with_otel_subscriber(|| {
            let tc = make_trace_context("not-hex", "also-not-hex", "01");
            let span = dispatch_span(Some(&tc), "req-2");
            let otel_ctx = span.context();
            assert!(!otel_ctx.span().span_context().is_valid());
        });
    }

    #[test]
    fn dispatch_span_falls_back_to_current_span_without_trace_context() {
        with_otel_subscriber(|| {
            let outer = tracing::info_span!("outer");
            let _enter = outer.enter();
            let span = dispatch_span(None, "req-3");
            assert_eq!(
                span.id(),
                tracing::Span::current().id(),
                "without a trace context the previous behavior must be preserved"
            );
        });
    }

    #[test]
    fn dispatch_span_uses_request_span_target() {
        with_otel_subscriber(|| {
            let tc =
                make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "01");
            let span = dispatch_span(Some(&tc), "req-target");
            assert_eq!(
                span.metadata().map(|m| m.target()),
                Some("request_span"),
                "dispatch span must use the always-on request-plane target"
            );
        });
    }

    #[test]
    fn dispatch_span_feeds_distributed_layer_for_header_injection() {
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        // Not SubscriberInitExt::set_default — that installs a global LogTracer.
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .with(DistributedTraceIdLayer),
        );
        let tc = make_trace_context("0123456789abcdef0123456789abcdef", "0123456789abcdef", "00");
        let span = dispatch_span(Some(&tc), "req-inject");
        let _enter = span.enter();
        let mut headers = std::collections::HashMap::new();

        inject_trace_headers_into_map(&mut headers);

        let traceparent = headers
            .get("traceparent")
            .expect("dispatch span must yield a traceparent for downstream injection");
        assert!(
            traceparent.starts_with("00-0123456789abcdef0123456789abcdef-"),
            "injected traceparent must carry the captured trace_id, got {traceparent}"
        );
        assert!(
            traceparent.ends_with("-00"),
            "unsampled trace_flags must survive injection, got {traceparent}"
        );
        assert_eq!(
            headers.get("tracestate").map(String::as_str),
            Some("vendor=dynamo")
        );
        assert_eq!(
            headers.get("x-request-id").map(String::as_str),
            Some("xr-1")
        );
    }
}
