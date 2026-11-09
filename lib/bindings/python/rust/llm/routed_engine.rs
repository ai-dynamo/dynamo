// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::{depythonize, pythonize};
use tokio_stream::StreamExt;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use dynamo_llm::entrypoint::PrefillRoutedEngine;
use dynamo_llm::protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY;
use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
use dynamo_runtime::logging::{DistributedTraceContext, otel_parent_context_from_distributed};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated as RsAnnotated;

use super::py_convert;
use crate::errors::routed_error_to_pyerr;
use crate::to_pyerr;

#[pyclass]
pub struct RoutedEngine {
    inner: PrefillRoutedEngine,
}

impl RoutedEngine {
    pub fn new(inner: PrefillRoutedEngine) -> Self {
        Self { inner }
    }
}

/// The `PreprocessedRequest` a Python engine passes to [`RoutedEngine::generate`].
///
/// `token_ids` and `extra_args` can each hold a prompt-sized list of ints (a chat engine's
/// capture plan, a multimodal prompt's expanded ids), and pythonize costs tens of ns per
/// int on the event loop's thread. [`fast_preprocessed_request`] reads those two fields
/// directly; any request it declines takes the full pythonize conversion, so results and
/// errors for it are unchanged.
fn preprocessed_request_from_py(obj: &Bound<'_, PyAny>) -> PyResult<PreprocessedRequest> {
    if let Some(request) = fast_preprocessed_request(obj) {
        return Ok(request);
    }
    depythonize(obj).map_err(to_pyerr)
}

/// An exact `dict` whose `token_ids` is an exact list of ints and whose `extra_args` is
/// absent, `None` or plain JSON-shaped data, with those two fields read directly and the
/// rest by pythonize. `None` for anything else, or if the rest fails to convert.
fn fast_preprocessed_request(obj: &Bound<'_, PyAny>) -> Option<PreprocessedRequest> {
    let dict = obj.downcast_exact::<PyDict>().ok()?;
    let token_ids = py_convert::read_exact_u32_list(&dict.get_item("token_ids").ok()??)?;
    let extra_args = match dict.get_item("extra_args").ok()? {
        Some(value) if !value.is_none() => Some(py_convert::exact_json_value(&value)?),
        _ => None,
    };
    // `token_ids` is required, so the copy keeps an empty one for pythonize to read.
    let rest = dict.copy().ok()?;
    rest.set_item("token_ids", PyList::empty(obj.py())).ok()?;
    rest.set_item("extra_args", obj.py().None()).ok()?;
    let mut request: PreprocessedRequest = depythonize(rest.as_any()).ok()?;
    request.token_ids = Arc::new(token_ids);
    request.extra_args = extra_args;
    Some(request)
}

/// Test hook: the request [`RoutedEngine::generate`] builds from `preprocessed`, pythonized
/// back. With `fast_path`, only the direct read (`None` when it declines); without, full
/// pythonize. Tests compare the two.
#[pyfunction]
#[pyo3(signature = (preprocessed, fast_path))]
pub fn _convert_preprocessed_request(
    py: Python<'_>,
    preprocessed: &Bound<'_, PyAny>,
    fast_path: bool,
) -> PyResult<Option<PyObject>> {
    let request = if fast_path {
        fast_preprocessed_request(preprocessed)
    } else {
        Some(depythonize::<PreprocessedRequest>(preprocessed).map_err(to_pyerr)?)
    };
    request
        .map(|request| Ok(pythonize(py, &request).map_err(to_pyerr)?.unbind()))
        .transpose()
}

fn copy_session_affinity(
    parent_context: &crate::context::Context,
    child_context: &mut SingleIn<PreprocessedRequest>,
) {
    if let Some(session_affinity) = parent_context.session_affinity() {
        child_context.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity.clone());
    }
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
        let request = preprocessed_request_from_py(preprocessed.bind(py))?;
        let request_context = if let Some(parent_context) = context.as_ref() {
            let parent_metadata = parent_context.metadata_snapshot();
            let parent_controller = parent_context.inner();
            let mut child_context = SingleIn::with_id_and_metadata(
                request,
                parent_controller.id().to_string(),
                parent_metadata,
            );
            copy_session_affinity(parent_context, &mut child_context);
            let child_controller = child_context.context();
            parent_controller.link_child(child_controller.clone());
            if parent_controller.is_killed() {
                child_controller.kill();
            } else if parent_controller.is_stopped() {
                child_controller.stop_generating();
            }
            child_context
        } else {
            SingleIn::new(request)
        };
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
                let mut stream = inner
                    .generate(request_context)
                    .await
                    .map_err(routed_error_to_pyerr)?;
                let task_context = stream.context();
                let (tx, rx) = tokio::sync::mpsc::channel::<RsAnnotated<PyObject>>(32);

                tokio::spawn(async move {
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
    use dynamo_llm::protocols::common::extensions::{
        SessionAffinityId, session_affinity_from_headers,
    };
    use dynamo_runtime::logging::{DistributedTraceIdLayer, inject_trace_headers_into_map};
    use opentelemetry::trace::{TraceContextExt, TraceId, TracerProvider as _};
    use tracing_subscriber::layer::SubscriberExt;

    fn preprocessed_request() -> PreprocessedRequest {
        serde_json::from_value(serde_json::json!({
            "token_ids": []
        }))
        .expect("minimal preprocessed request")
    }

    #[test]
    fn request_context_preserves_session_affinity_across_python_boundary() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-dynamo-session-id", "session-123".parse().unwrap());
        let session_affinity =
            session_affinity_from_headers(&headers).expect("session affinity header");
        let mut source = SingleIn::new(());
        source.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity);
        let python_context =
            crate::context::Context::new(source.context(), None, None, Default::default())
                .with_session_affinity_from(&source)
                .expect("pipeline context converts to Python binding context");
        let mut routed_context = SingleIn::new(preprocessed_request());
        copy_session_affinity(&python_context, &mut routed_context);

        let session_affinity = routed_context
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .expect("session affinity restored after Python boundary");
        assert_eq!(session_affinity.as_str(), "session-123");
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
