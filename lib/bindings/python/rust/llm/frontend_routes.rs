// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_llm::http::service::axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use dynamo_llm::http::service::{
    FrontendExtensionContext as RsFrontendExtensionContext, FrontendRouteExtension,
    FrontendRouteSet,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyIterator,
};
use pythonize::depythonize;
use serde_json::{Value, json};

/// A trusted Python-provided frontend route (static-path `GET` only). `handler`
/// is a synchronous callable taking a `FrontendExtensionContext` and returning a
/// JSON body (200) or a `FrontendResponse` for a custom status. Non-GET methods,
/// non-static paths, and async handlers are rejected at construction.
#[pyclass(name = "FrontendRoute")]
#[derive(Clone)]
pub(crate) struct PyFrontendRoute {
    path: String,
    handler: PyObject,
}

#[pymethods]
impl PyFrontendRoute {
    #[new]
    pub fn new(py: Python<'_>, method: String, path: String, handler: PyObject) -> PyResult<Self> {
        let bound_handler = handler.bind(py);
        if !bound_handler.is_callable() {
            return Err(PyTypeError::new_err(
                "FrontendRoute handler must be callable",
            ));
        }
        // Reject async handlers up front, not on first request.
        if is_coroutine_function(py, bound_handler)? {
            return Err(PyValueError::new_err(
                "FrontendRoute handler must be synchronous; async def handlers are not supported",
            ));
        }
        // GET-only initial surface.
        if !method.eq_ignore_ascii_case("GET") {
            return Err(PyValueError::new_err(format!(
                "unsupported FrontendRoute method '{method}'; only GET is supported"
            )));
        }
        validate_static_path(&path)?;
        Ok(Self { path, handler })
    }

    #[getter]
    pub fn method(&self) -> String {
        "GET".to_string()
    }

    #[getter]
    pub fn path(&self) -> String {
        self.path.clone()
    }
}

/// Status-code override returned by a handler: `FrontendResponse(status, body)`.
/// Return a plain value for the default 200.
#[pyclass(name = "FrontendResponse")]
#[derive(Clone)]
pub(crate) struct PyFrontendResponse {
    status_code: u16,
    body: PyObject,
}

#[pymethods]
impl PyFrontendResponse {
    #[new]
    pub fn new(status_code: u16, body: PyObject) -> PyResult<Self> {
        StatusCode::from_u16(status_code).map_err(|e| {
            PyValueError::new_err(format!("invalid status code {status_code}: {e}"))
        })?;
        Ok(Self { status_code, body })
    }
}

/// Read-only live frontend state exposed to Python frontend route handlers.
#[pyclass(name = "FrontendExtensionContext")]
#[derive(Clone)]
pub(crate) struct PyFrontendExtensionContext {
    inner: RsFrontendExtensionContext,
}

#[pymethods]
impl PyFrontendExtensionContext {
    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    pub fn has_any_ready_model(&self) -> bool {
        self.inner.has_any_ready_model()
    }

    pub fn is_model_ready_to_serve(&self, model: &str) -> bool {
        self.inner.is_model_ready_to_serve(model)
    }

    pub fn model_display_names(&self) -> Vec<String> {
        sorted_strings(self.inner.model_display_names().into_iter())
    }

    pub fn serving_ready_display_names(&self) -> Vec<String> {
        sorted_strings(self.inner.serving_ready_display_names().into_iter())
    }
}

pub(crate) fn frontend_route_extensions_from_py(
    py: Python<'_>,
    routes: Option<PyObject>,
) -> PyResult<Vec<FrontendRouteExtension>> {
    let Some(routes) = routes else {
        return Ok(Vec::new());
    };
    if routes.is_none(py) {
        return Ok(Vec::new());
    }

    let bound = routes.bind(py);
    let iter = PyIterator::from_object(bound)?;
    let mut route_specs = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in iter {
        let item = item?;
        let route = item.extract::<PyRef<'_, PyFrontendRoute>>()?;
        // All providers fold into one router; Router::route panics on an
        // overlapping path, so reject duplicates cleanly here (GET-only, so
        // path is the key).
        if !seen.insert(route.path.clone()) {
            return Err(PyValueError::new_err(format!(
                "duplicate frontend route registered: GET {}",
                route.path
            )));
        }
        route_specs.push(route.clone());
    }

    if route_specs.is_empty() {
        Ok(Vec::new())
    } else {
        Ok(vec![frontend_route_extension_from_routes(route_specs)])
    }
}

fn frontend_route_extension_from_routes(routes: Vec<PyFrontendRoute>) -> FrontendRouteExtension {
    let routes = Arc::new(routes);
    Arc::new(move |context: RsFrontendExtensionContext| {
        let mut builder = FrontendRouteSet::builder();
        for route in routes.iter() {
            let path = route.path.clone();
            // Clone the handler under the GIL (`clone_ref` requires it), then
            // share it into the handler closure via `Arc` (GIL-free clone) —
            // the closure runs on tokio workers that don't hold the GIL.
            let handler = Arc::new(Python::with_gil(|py| route.handler.clone_ref(py)));
            let route_context = context.clone();
            builder = builder.get(path, move || {
                call_python_frontend_route(handler.clone(), route_context.clone())
            });
        }
        builder.build()
    })
}

async fn call_python_frontend_route(
    handler: Arc<PyObject>,
    context: RsFrontendExtensionContext,
) -> Response {
    // Run the synchronous Python handler on a blocking thread so a slow
    // extension cannot pin a tokio worker and reduce request concurrency.
    let outcome = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| call_python_frontend_route_inner(py, &handler, context))
    })
    .await;

    match outcome {
        Ok(Ok((status, body))) => (status, Json(body)).into_response(),
        Ok(Err(err)) => {
            tracing::error!(error = %err, "Python frontend route extension failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "frontend route extension failed"})),
            )
                .into_response()
        }
        Err(err) => {
            tracing::error!(error = %err, "Python frontend route extension task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "frontend route extension failed"})),
            )
                .into_response()
        }
    }
}

fn call_python_frontend_route_inner(
    py: Python<'_>,
    handler: &PyObject,
    context: RsFrontendExtensionContext,
) -> PyResult<(StatusCode, Value)> {
    let py_context = Py::new(py, PyFrontendExtensionContext { inner: context })?;
    let result = handler.call1(py, (py_context,))?;
    normalize_route_response(py, result)
}

fn normalize_route_response(py: Python<'_>, result: PyObject) -> PyResult<(StatusCode, Value)> {
    let bound = result.bind(py);

    // Explicit status override via FrontendResponse.
    if let Ok(resp) = bound.extract::<PyRef<'_, PyFrontendResponse>>() {
        let status = StatusCode::from_u16(resp.status_code)
            .map_err(|e| PyValueError::new_err(format!("invalid status code: {e}")))?;
        let body: Value = depythonize(resp.body.bind(py)).map_err(|e| {
            PyValueError::new_err(format!("response body must be JSON-serializable: {e}"))
        })?;
        return Ok((status, body));
    }

    // A sync handler may still return an awaitable; reject and close it to avoid
    // an unawaited-coroutine warning.
    if bound.hasattr("__await__")? {
        let _ = bound.call_method0("close");
        return Err(PyTypeError::new_err(
            "FrontendRoute handler returned an awaitable; return synchronously \
             (a JSON body or a FrontendResponse)",
        ));
    }

    // Otherwise a JSON body with status 200 (tuples serialize as JSON arrays).
    let body: Value = depythonize(bound).map_err(|e| {
        PyValueError::new_err(format!("response body must be JSON-serializable: {e}"))
    })?;
    Ok((StatusCode::OK, body))
}

/// Whether `handler` is an `async def` (a coroutine function).
fn is_coroutine_function(py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<bool> {
    py.import("inspect")?
        .call_method1("iscoroutinefunction", (handler,))?
        .extract()
}

/// Reject non-static paths (params `{...}`, wildcards `*`, whitespace) that would
/// panic `Router::route` or create conflicts the path-string key can't detect.
fn validate_static_path(path: &str) -> PyResult<()> {
    if !path.starts_with('/') {
        return Err(PyValueError::new_err(
            "FrontendRoute path must start with '/'",
        ));
    }
    if path
        .chars()
        .any(|c| matches!(c, '{' | '}' | '*') || c.is_whitespace() || c.is_control())
    {
        return Err(PyValueError::new_err(format!(
            "FrontendRoute path '{path}' must be a static path \
             (no path parameters '{{...}}', wildcards '*', or whitespace)"
        )));
    }
    Ok(())
}

fn sorted_strings(values: impl Iterator<Item = String>) -> Vec<String> {
    let mut values: Vec<String> = values.collect();
    values.sort();
    values
}
