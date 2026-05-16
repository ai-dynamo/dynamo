// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Context is a wrapper around the AsyncEngineContext to allow for Python bindings.

use dynamo_runtime::logging::DistributedTraceContext;
pub use dynamo_runtime::pipeline::AsyncEngineContext;
use dynamo_runtime::pipeline::context::Controller;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
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
}

#[derive(Clone)]
#[pyclass]
pub struct ContextMetadata {
    inner: Arc<Mutex<BTreeMap<String, String>>>,
}

#[pymethods]
impl ContextMetadata {
    fn __getitem__(&self, key: &str) -> PyResult<String> {
        self.inner
            .lock()
            .expect("metadata mutex poisoned")
            .get(key)
            .cloned()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string()))
    }

    fn __setitem__(&self, key: String, value: String) {
        self.inner
            .lock()
            .expect("metadata mutex poisoned")
            .insert(key, value);
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        self.inner
            .lock()
            .expect("metadata mutex poisoned")
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.lock().expect("metadata mutex poisoned").len()
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner
            .lock()
            .expect("metadata mutex poisoned")
            .contains_key(key)
    }

    fn __iter__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let keys = self
            .inner
            .lock()
            .expect("metadata mutex poisoned")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        PyList::new(py, keys)?.call_method0("__iter__")
    }

    fn get(&self, key: &str, default: Option<String>) -> Option<String> {
        self.inner
            .lock()
            .expect("metadata mutex poisoned")
            .get(key)
            .cloned()
            .or(default)
    }

    #[pyo3(signature = (key, default=None))]
    fn pop(&self, key: &str, default: Option<String>) -> PyResult<Option<String>> {
        let mut guard = self.inner.lock().expect("metadata mutex poisoned");
        match guard.remove(key) {
            Some(value) => Ok(Some(value)),
            None if default.is_some() => Ok(default),
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string())),
        }
    }

    fn keys<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let keys = self
            .inner
            .lock()
            .expect("metadata mutex poisoned")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        PyList::new(py, keys)
    }

    fn values<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let values = self
            .inner
            .lock()
            .expect("metadata mutex poisoned")
            .values()
            .cloned()
            .collect::<Vec<_>>();
        PyList::new(py, values)
    }

    fn items<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyList>> {
        let items = self
            .inner
            .lock()
            .expect("metadata mutex poisoned")
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        PyList::new(py, items)
    }

    fn clear(&self) {
        self.inner.lock().expect("metadata mutex poisoned").clear();
    }

    fn copy<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        let snapshot = self
            .inner
            .lock()
            .expect("metadata mutex poisoned")
            .clone();
        let dict = PyDict::new(py);
        for (key, value) in snapshot {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "{:?}",
            self.inner.lock().expect("metadata mutex poisoned").clone()
        )
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
        }
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
            .expect("metadata mutex poisoned")
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
        *self.metadata.lock().expect("metadata mutex poisoned") = metadata;
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
