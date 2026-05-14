// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Context is a wrapper around the AsyncEngineContext to allow for Python bindings.

use dynamo_runtime::logging::DistributedTraceContext;
pub use dynamo_runtime::pipeline::AsyncEngineContext;
use dynamo_runtime::pipeline::context::Controller;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::sync::watch;

// Context is a wrapper around the AsyncEngineContext to allow for Python bindings.
// Not all methods of the AsyncEngineContext are exposed, jsut the primary ones for tracing + cancellation.
// Kept as class, to allow for future expansion if needed.
#[derive(Clone)]
#[pyclass]
pub struct Context {
    inner: Arc<dyn AsyncEngineContext>,
    trace_context: Option<DistributedTraceContext>,
    /// First-token signal for decode-mode disagg. `None` on aggregated /
    /// prefill requests.
    first_token: Option<watch::Sender<bool>>,
    metadata: Arc<Mutex<BTreeMap<String, String>>>,
    /// `engine.generate` span captured before crossing the spawn_blocking
    /// boundary, so PyO3 `record_attribute` / `record_event` can target it
    /// from Python code (where `Span::current()` is the worker thread root,
    /// not the auto-span). `None` for Python-instantiated test contexts.
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
            span: Some(tracing::Span::current()),
        }
    }

    /// Override the span the telemetry methods record onto. Used by
    /// `PyLLMEngine::generate` to plumb the `engine.generate` auto-span
    /// across the spawn_blocking boundary.
    pub fn with_span(mut self, span: tracing::Span) -> Self {
        self.span = Some(span);
        self
    }

    // Get trace context for Rust-side usage
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

    // sync method of `await async_is_stopped()`
    fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    // sync method of `await async_is_killed()`
    fn is_killed(&self) -> bool {
        self.inner.is_killed()
    }
    // issues a stop generating
    fn stop_generating(&self) {
        self.inner.stop_generating();
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    // allows building a async callback.
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

    // Expose trace information to Python for debugging
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

    /// Record an attribute on the `engine.generate` span. Silently no-op when
    /// no span was plumbed in (Python-instantiated test contexts). Used by
    /// `dynamo.common.backend.telemetry.record(...)`.
    fn record_attribute(&self, key: &str, value: Bound<'_, PyAny>) -> PyResult<()> {
        let Some(span) = &self.span else {
            return Ok(());
        };
        // Span::record only accepts declared empty fields. Engines should
        // record canonical attrs (e.g. via `telemetry.record(input_tokens=N)`)
        // — unknown field names are silently dropped by tracing, matching
        // the `tracing::Span::record` semantics.
        if let Ok(b) = value.downcast::<PyBool>() {
            span.record(key, b.is_true());
        } else if let Ok(i) = value.downcast::<PyInt>() {
            span.record(key, i.extract::<i64>()?);
        } else if let Ok(f) = value.downcast::<PyFloat>() {
            span.record(key, f.extract::<f64>()?);
        } else if let Ok(s) = value.downcast::<PyString>() {
            span.record(key, s.to_str()?);
        } else {
            // Fallback: render via Python `repr()` so callers see something
            // useful in trace UIs even for engine-specific objects.
            let repr = value.repr()?.extract::<String>()?;
            span.record(key, repr.as_str());
        }
        Ok(())
    }

    /// Emit a structured event on the `engine.generate` span. `attrs` is an
    /// optional dict of field name → Python value, rendered via `repr()` for
    /// non-primitive types. Used by `dynamo.common.backend.telemetry.event`.
    #[pyo3(signature = (name, attrs=None))]
    fn record_event(&self, name: &str, attrs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let Some(span) = &self.span else {
            return Ok(());
        };
        // Build a comma-separated rendering of the attrs map. We can't emit
        // dynamic fields via the `event!` macro (compile-time names only), so
        // pack into a single `attrs` field. Operators read `event_name` + `attrs`.
        let attrs_str = match attrs {
            Some(d) if !d.is_empty() => {
                let mut parts: Vec<String> = Vec::with_capacity(d.len());
                for (k, v) in d.iter() {
                    let k_str = k.extract::<String>()?;
                    let v_repr = v.repr()?.extract::<String>()?;
                    parts.push(format!("{k_str}={v_repr}"));
                }
                parts.join(", ")
            }
            _ => String::new(),
        };
        let _enter = span.enter();
        tracing::event!(
            target: "request_span",
            tracing::Level::INFO,
            event_name = name,
            attrs = %attrs_str,
        );
        Ok(())
    }

    /// Build W3C trace headers for propagating to downstream inference engines.
    /// Returns `None` when no upstream trace context is present, in which case
    /// callers should forward the value as-is — inference engines treat `None`
    /// as "no upstream trace."
    ///
    /// Always emits `traceparent`. Also emits `tracestate`, `x-request-id`,
    /// and `request-id` when the upstream propagated them. Trace-flags are
    /// hard-coded to `01` (sampled) until we plumb the live span's flags.
    fn trace_headers(&self) -> Option<HashMap<String, String>> {
        let tc = self.trace_context.as_ref()?;
        if tc.trace_id.is_empty() || tc.span_id.is_empty() {
            return None;
        }
        let mut headers = HashMap::new();
        headers.insert(
            "traceparent".to_string(),
            format!("00-{}-{}-01", tc.trace_id, tc.span_id),
        );
        if let Some(ts) = &tc.tracestate {
            headers.insert("tracestate".to_string(), ts.clone());
        }
        if let Some(id) = &tc.x_request_id {
            headers.insert("x-request-id".to_string(), id.clone());
        }
        if let Some(id) = &tc.request_id {
            headers.insert("request-id".to_string(), id.clone());
        }
        Some(headers)
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
