// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Context is a wrapper around the AsyncEngineContext to allow for Python bindings.

use dynamo_runtime::logging::DistributedTraceContext;
pub use dynamo_runtime::pipeline::AsyncEngineContext;
use dynamo_runtime::pipeline::context::Controller;
use pyo3::prelude::*;
use std::sync::Arc;

// Context is a wrapper around the AsyncEngineContext to allow for Python bindings.
// Not all methods of the AsyncEngineContext are exposed, jsut the primary ones for tracing + cancellation.
// Kept as class, to allow for future expansion if needed.
#[derive(Clone)]
#[pyclass]
pub struct Context {
    inner: Arc<dyn AsyncEngineContext>,
    trace_context: Option<DistributedTraceContext>,
}

impl Context {
    pub fn new(
        inner: Arc<dyn AsyncEngineContext>,
        trace_context: Option<DistributedTraceContext>,
    ) -> Self {
        Self {
            inner,
            trace_context,
        }
    }

    // Get trace context for Rust-side usage
    pub fn trace_context(&self) -> Option<&DistributedTraceContext> {
        self.trace_context.as_ref()
    }

    pub fn inner(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.clone()
    }
}

#[pymethods]
impl Context {
    #[new]
    #[pyo3(signature = (id=None))]
    fn py_new(id: Option<String>) -> Self {
        let controller = match id {
            Some(id) => Controller::new(id),
            None => Controller::default(),
        };
        Self {
            inner: Arc::new(controller),
            trace_context: None,
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
