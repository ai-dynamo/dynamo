// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `Context` wraps `AsyncEngineContext` for PyO3 — exposes cancellation,
// trace identity, and span access for engine observability.
//
// Engine code reaches the observability surface through the
// `dynamo.common.backend.telemetry` facade, which is itself a thin wrapper
// over [`Context::current_span`] / [`Context::start_span`]. Both return a
// unified [`SpanProxy`] handle whose `set_attribute` / `add_event` /
// `set_status` operations mirror the OTel `Span` API.

use dynamo_runtime::logging::DistributedTraceContext;
pub use dynamo_runtime::pipeline::AsyncEngineContext;
use dynamo_runtime::pipeline::context::Controller;
use opentelemetry::global::BoxedSpan;
use opentelemetry::trace::{Span as OtelSpan, Status, TraceContextExt, Tracer};
use opentelemetry::{KeyValue, global};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use tokio::sync::watch;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Process-wide guard: once-per-process WARN when telemetry calls hit a
/// parent `engine.generate` span that has no OTel context (i.e., the
/// `tracing-opentelemetry` layer isn't installed — non-JSONL deployments).
/// One log line is enough to surface the configuration issue; rate-limiting
/// to once avoids flooding logs in high-QPS workers.
///
/// Visibility level is WARN because telemetry calls silently no-oping is a
/// real operator-relevant misconfiguration — if the engine is recording
/// attributes and getting nothing, the operator needs to see it at default
/// log levels, not just when debug is enabled.
static BRIDGE_MISSING_WARNED: OnceLock<()> = OnceLock::new();

fn warn_bridge_missing_once(method: &str) {
    if BRIDGE_MISSING_WARNED.set(()).is_ok() {
        tracing::warn!(
            method,
            "telemetry call is a no-op: OTel bridge layer not installed \
             (needs DYN_LOGGING_JSONL=1 + OTEL_EXPORT_ENABLED=1). \
             Engine telemetry attributes / events / child spans are NOT \
             being recorded. Further no-ops in this process are silent."
        );
    }
}

/// Per-request handle exposed to Python engine code. Owns cancellation
/// (via `AsyncEngineContext`), trace identity (via `DistributedTraceContext`),
/// the disagg first-token signal, and the captured `engine.generate` span.
///
/// The span is private — engine code reaches it via [`Context::current_span`]
/// (the auto-span proxy) or [`Context::start_span`] (a child span). The
/// facade in `dynamo.common.backend.telemetry` is a one-line wrapper around
/// those methods.
#[derive(Clone)]
#[pyclass]
pub struct Context {
    inner: Arc<dyn AsyncEngineContext>,
    trace_context: Option<DistributedTraceContext>,
    /// First-token signal for decode-mode disagg. `None` on aggregated /
    /// prefill requests.
    first_token: Option<watch::Sender<bool>>,
    metadata: Arc<Mutex<BTreeMap<String, String>>>,
    /// Captured `engine.generate` span. `None` for Python-instantiated test
    /// contexts (where no parent span was plumbed in) — `current_span` /
    /// `start_span` return a no-op `SpanProxy` in that case.
    span: Option<tracing::Span>,
}

#[derive(Clone)]
#[pyclass]
pub struct ContextMetadata {
    inner: Arc<Mutex<BTreeMap<String, String>>>,
}

impl ContextMetadata {
    fn lock_map(&self) -> MutexGuard<'_, BTreeMap<String, String>> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[pymethods]
impl ContextMetadata {
    fn __getitem__(&self, key: &str) -> PyResult<String> {
        self.lock_map()
            .get(key)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string()))
    }

    fn __setitem__(&self, key: String, value: String) {
        self.lock_map().insert(key, value);
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        self.lock_map()
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string()))
    }

    fn __len__(&self) -> usize {
        self.lock_map().len()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.lock_map().contains_key(key)
    }

    fn __iter__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let keys = self.lock_map().keys().cloned().collect::<Vec<_>>();
        PyList::new(py, keys)?.call_method0("__iter__")
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, key: &str, default: Option<String>) -> Option<String> {
        self.lock_map().get(key).cloned().or(default)
    }

    #[pyo3(signature = (key, default = None::<Option<String>>))]
    fn pop(&self, key: &str, default: Option<Option<String>>) -> PyResult<Option<String>> {
        let mut guard = self.lock_map();
        match guard.remove(key) {
            Some(value) => Ok(Some(value)),
            None if default.is_some() => Ok(default.flatten()),
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                key.to_string(),
            )),
        }
    }

    fn keys<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let keys = self.lock_map().keys().cloned().collect::<Vec<_>>();
        PyList::new(py, keys)
    }

    fn values<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let values = self.lock_map().values().cloned().collect::<Vec<_>>();
        PyList::new(py, values)
    }

    fn items<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let items = self
            .lock_map()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        PyList::new(py, items)
    }

    fn clear(&self) {
        self.lock_map().clear();
    }

    fn copy<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let snapshot = self.lock_map().clone();
        let dict = PyDict::new(py);
        for (key, value) in snapshot {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.lock_map().clone())
    }
}

impl Context {
    pub fn new(
        inner: Arc<dyn AsyncEngineContext>,
        trace_context: Option<DistributedTraceContext>,
        first_token: Option<watch::Sender<bool>>,
        metadata: BTreeMap<String, String>,
    ) -> Self {
        Self {
            inner,
            trace_context,
            first_token,
            metadata: Arc::new(Mutex::new(metadata)),
            span: None,
        }
    }

    /// Attach the `engine.generate` span. Called by `PyLLMEngine::generate`
    /// after capturing `Span::current()` outside the spawn_blocking boundary.
    /// See `lib/bindings/python/rust/backend.rs` for the invariant.
    pub fn with_span(mut self, span: tracing::Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn trace_context(&self) -> Option<&DistributedTraceContext> {
        self.trace_context.as_ref()
    }

    pub fn inner(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.clone()
    }

    pub fn metadata_snapshot(&self) -> BTreeMap<String, String> {
        self.metadata
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Build the `traceparent` header value. Prefers the engine.generate
    /// OTel span context when valid so downstream engine spans nest under
    /// `engine.generate`; falls back to the inbound trace context.
    fn format_traceparent(&self) -> Option<String> {
        self.bridged_traceparent()
            .or_else(|| self.fallback_traceparent())
    }

    fn bridged_traceparent(&self) -> Option<String> {
        let span = self.span.as_ref()?;
        let otel_ctx = span.context();
        let otel_span = otel_ctx.span();
        let sc = otel_span.span_context();
        if !sc.is_valid() {
            return None;
        }
        let flags = if sc.trace_flags().is_sampled() {
            "01"
        } else {
            "00"
        };
        Some(format!("00-{}-{}-{}", sc.trace_id(), sc.span_id(), flags))
    }

    fn fallback_traceparent(&self) -> Option<String> {
        let tc = self.trace_context.as_ref()?;
        if tc.trace_id.is_empty() || tc.span_id.is_empty() {
            return None;
        }
        Some(tc.create_traceparent())
    }
}

#[pymethods]
impl Context {
    #[new]
    #[pyo3(signature = (id=None, metadata=None))]
    fn py_new(id: Option<String>, metadata: Option<BTreeMap<String, String>>) -> Self {
        let controller = match id {
            Some(id) => Controller::new(id),
            None => Controller::default(),
        };
        Self {
            inner: Arc::new(controller),
            trace_context: None,
            first_token: None,
            metadata: Arc::new(Mutex::new(metadata.unwrap_or_default())),
            span: None,
        }
    }

    /// Create a context with a fresh cancellation controller and request id.
    ///
    /// The detached context keeps the trace context, captured span, and a
    /// snapshot of metadata so disaggregated handoffs keep observability
    /// parentage without sharing cancellation ownership. The first-token
    /// signal is intentionally dropped.
    #[pyo3(signature = (id))]
    fn detached(&self, id: String) -> Self {
        Self {
            inner: Arc::new(Controller::new(id)),
            trace_context: self.trace_context.clone(),
            first_token: None,
            metadata: Arc::new(Mutex::new(self.metadata_snapshot())),
            span: self.span.clone(),
        }
    }

    fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    fn is_killed(&self) -> bool {
        self.inner.is_killed()
    }

    fn stop_generating(&self) {
        self.inner.stop_generating();
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn async_killed_or_stopped<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::select! {
                _ = inner.killed() => {
                    Ok(true)
                }
                _ = inner.stopped() => {
                    Ok(true)
                }
            }
        })
    }

    /// Fire the first-token signal so the framework can release any
    /// deferred `engine.abort()`. Idempotent; no-op on non-decode
    /// requests. Engines normally don't need this — the framework
    /// auto-fires on the first non-empty chunk in the response stream.
    fn notify_first_token(&self) {
        if let Some(tx) = &self.first_token {
            let _ = tx.send(true);
        }
    }

    #[getter]
    fn metadata(&self) -> ContextMetadata {
        ContextMetadata {
            inner: self.metadata.clone(),
        }
    }

    #[setter]
    fn set_metadata(&mut self, metadata: BTreeMap<String, String>) {
        *self
            .metadata
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = metadata;
    }

    #[getter]
    fn trace_id(&self) -> Option<String> {
        self.trace_context.as_ref().map(|ctx| ctx.trace_id.clone())
    }

    #[getter]
    fn span_id(&self) -> Option<String> {
        self.trace_context.as_ref().map(|ctx| ctx.span_id.clone())
    }

    #[getter]
    fn parent_span_id(&self) -> Option<String> {
        self.trace_context
            .as_ref()
            .and_then(|ctx| ctx.parent_id.clone())
    }

    /// Handle on the framework's `engine.generate` span — the parent span
    /// that engine attributes / events should attach to. Always returns a
    /// `SpanProxy`; when no parent was plumbed in (test contexts) or the
    /// OTel bridge isn't installed, the proxy is a silent no-op.
    ///
    /// Prefer the `dynamo.common.backend.telemetry.current_span(context)`
    /// facade in engine code — it's the documented surface.
    fn current_span(&self) -> SpanProxy {
        match &self.span {
            Some(span) if span.context().span().span_context().is_valid() => SpanProxy {
                inner: SpanProxyInner::Tracing(span.clone()),
            },
            Some(_) => {
                warn_bridge_missing_once("current_span");
                SpanProxy {
                    inner: SpanProxyInner::NoOp,
                }
            }
            None => SpanProxy {
                inner: SpanProxyInner::NoOp,
            },
        }
    }

    /// Open a child span under the `engine.generate` parent. Use this for
    /// dynamic span names (`tracing::info_span!` requires compile-time
    /// names). Returned `SpanProxy` is a context manager; the span ends on
    /// `__exit__` / `close()` / drop.
    ///
    /// Returns a no-op proxy when no parent was plumbed in or the bridge
    /// isn't installed. Prefer the
    /// `dynamo.common.backend.telemetry.start_span(context, name)` facade
    /// in engine code.
    #[pyo3(signature = (name, attrs=None))]
    fn start_span(&self, name: &str, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<SpanProxy> {
        let Some(parent) = &self.span else {
            return Ok(SpanProxy {
                inner: SpanProxyInner::NoOp,
            });
        };
        let parent_ctx = parent.context();
        if !parent_ctx.span().span_context().is_valid() {
            warn_bridge_missing_once("start_span");
            return Ok(SpanProxy {
                inner: SpanProxyInner::NoOp,
            });
        }
        let tracer = global::tracer("dynamo");
        let mut otel_attrs = Vec::new();
        if let Some(d) = attrs {
            for (k, v) in d.iter() {
                otel_attrs.push(KeyValue::new(k.extract::<String>()?, py_to_otel_value(&v)?));
            }
        }
        let mut builder = tracer.span_builder(name.to_string());
        if !otel_attrs.is_empty() {
            builder = builder.with_attributes(otel_attrs);
        }
        let span = builder.start_with_context(&tracer, &parent_ctx);
        Ok(SpanProxy {
            inner: SpanProxyInner::Otel(Some(span)),
        })
    }

    /// Build W3C trace headers for propagating to downstream inference engines.
    /// Returns `None` when no trace context is available (neither a captured
    /// engine.generate span nor inbound trace headers); callers should
    /// forward `None` as-is — inference engines treat it as "no upstream
    /// trace."
    ///
    /// Prefers the OTel context of the `engine.generate` auto-span when the
    /// bridge is installed, so downstream engine internals (vLLM scheduler,
    /// TRT-LLM forward, SGLang KV transfer) nest UNDER `engine.generate`.
    /// Falls back to the inbound `DistributedTraceContext` for legacy
    /// callers, Python-instantiated test contexts, and non-JSONL deployments
    /// without the bridge.
    ///
    /// Always emits `traceparent`. Also emits `tracestate`, `x-request-id`,
    /// and `request-id` when the upstream propagated them.
    fn trace_headers(&self) -> Option<HashMap<String, String>> {
        let mut headers = HashMap::new();
        headers.insert("traceparent".to_string(), self.format_traceparent()?);
        if let Some(tc) = self.trace_context.as_ref() {
            if let Some(ts) = &tc.tracestate {
                headers.insert("tracestate".to_string(), ts.clone());
            }
            if let Some(id) = &tc.x_request_id {
                headers.insert("x-request-id".to_string(), id.clone());
            }
            if let Some(id) = &tc.request_id {
                headers.insert("request-id".to_string(), id.clone());
            }
        }
        Some(headers)
    }
}

/// Unified Python-facing span handle. Returned from both
/// `Context.current_span()` (auto-span) and `Context.start_span()` (child
/// span). Routes `set_attribute` / `add_event` / `set_status` to whichever
/// underlying span the proxy wraps.
///
/// `__enter__` / `__exit__` make the proxy usable as a `with`-block context
/// manager. For child spans, `__exit__` (and `close()`) ends the span; for
/// the auto-span the proxy is a borrow and `close()` is a no-op (the
/// framework owns the span's lifecycle).
#[pyclass]
pub struct SpanProxy {
    inner: SpanProxyInner,
}

enum SpanProxyInner {
    /// Auto-span — borrows the `engine.generate` span via tracing. Writes
    /// go through `OpenTelemetrySpanExt` so attribute names are dynamic
    /// (no pre-declaration constraint).
    Tracing(tracing::Span),
    /// Child span — owns an OTel `BoxedSpan`. Ends on close / drop.
    /// `Option` makes close idempotent.
    Otel(Option<BoxedSpan>),
    /// No parent or bridge missing. All calls are silent no-ops.
    NoOp,
}

#[pymethods]
impl SpanProxy {
    /// Set an attribute. OTel imposes no pre-declaration constraint;
    /// any key is accepted. No-op when the proxy is no-op or already closed.
    fn set_attribute(&mut self, key: &str, value: Bound<'_, PyAny>) -> PyResult<()> {
        match &mut self.inner {
            SpanProxyInner::Tracing(span) => {
                span.set_attribute(key.to_string(), py_to_otel_value(&value)?);
            }
            SpanProxyInner::Otel(Some(span)) => {
                span.set_attribute(KeyValue::new(key.to_string(), py_to_otel_value(&value)?));
            }
            _ => {}
        }
        Ok(())
    }

    /// Emit a structured event. Per-attr key/value (real OTel `SpanEvent`
    /// fields — trace backends can query individual keys).
    #[pyo3(signature = (name, attrs=None))]
    fn add_event(&mut self, name: &str, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let collect = |d: Option<&Bound<'_, PyDict>>| -> PyResult<Vec<KeyValue>> {
            let mut out = Vec::new();
            if let Some(d) = d {
                for (k, v) in d.iter() {
                    out.push(KeyValue::new(k.extract::<String>()?, py_to_otel_value(&v)?));
                }
            }
            Ok(out)
        };
        match &mut self.inner {
            SpanProxyInner::Tracing(span) => {
                span.add_event(name.to_string(), collect(attrs)?);
            }
            SpanProxyInner::Otel(Some(span)) => {
                span.add_event(name.to_string(), collect(attrs)?);
            }
            _ => {}
        }
        Ok(())
    }

    /// Set the span's status. `status` is either `"ok"` or `"error"`;
    /// `description` is optional context (typically a short error name).
    #[pyo3(signature = (status, description=None))]
    fn set_status(&mut self, status: &str, description: Option<String>) -> PyResult<()> {
        let otel_status = match status {
            "ok" => Status::Ok,
            "error" => Status::error(description.unwrap_or_default()),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "set_status: expected \"ok\" or \"error\", got {other:?}"
                )));
            }
        };
        match &mut self.inner {
            SpanProxyInner::Tracing(span) => {
                span.set_status(otel_status);
            }
            SpanProxyInner::Otel(Some(span)) => {
                span.set_status(otel_status);
            }
            _ => {}
        }
        Ok(())
    }

    /// End the underlying span (child spans only). Idempotent; no-op for
    /// the auto-span and no-op proxies. Called automatically by `__exit__`
    /// and on drop.
    fn close(&mut self) {
        if let SpanProxyInner::Otel(slot) = &mut self.inner
            && let Some(mut span) = slot.take()
        {
            span.end();
        }
    }

    fn __enter__(slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}

impl Drop for SpanProxy {
    fn drop(&mut self) {
        self.close();
    }
}

/// Coerce a Python value into an OTel `Value` for `KeyValue` attributes.
/// Primitive types preserve type; everything else renders via `repr()`.
fn py_to_otel_value(v: &Bound<'_, PyAny>) -> PyResult<opentelemetry::Value> {
    use opentelemetry::Value;
    if let Ok(b) = v.downcast::<PyBool>() {
        Ok(Value::Bool(b.is_true()))
    } else if let Ok(i) = v.downcast::<PyInt>() {
        Ok(Value::I64(i.extract::<i64>()?))
    } else if let Ok(f) = v.downcast::<PyFloat>() {
        Ok(Value::F64(f.extract::<f64>()?))
    } else if let Ok(s) = v.downcast::<PyString>() {
        // `to_cow` (not `to_str`) for abi3 compatibility: enabling the
        // `aic-forward-pass` feature pulls in aiconfigurator-core, which sets
        // pyo3 `abi3-py39`; under the <3.10 limited API `PyString::to_str` is
        // compiled out, while `to_cow` is always available. Same conversion.
        Ok(Value::String(s.to_cow()?.into_owned().into()))
    } else {
        Ok(Value::String(v.repr()?.extract::<String>()?.into()))
    }
}

// PyO3 equivalent for verify if signature contains target_name
// def callable_accepts_kwarg(target_name: str):
//      import inspect
//      return target_name in inspect.signature(func).parameters
pub fn callable_accepts_kwarg(
    py: Python,
    callable: &Bound<'_, PyAny>,
    target_name: &str,
) -> PyResult<bool> {
    let inspect: Bound<'_, PyModule> = py.import("inspect")?;
    let signature = inspect.call_method1("signature", (callable,))?;
    let params_any: Bound<'_, PyAny> = signature.getattr("parameters")?;
    params_any
        .call_method1("__contains__", (target_name,))?
        .extract::<bool>()
}
