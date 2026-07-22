// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

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
        // Reuse the core validator so Python and the Rust builder can't drift.
        dynamo_llm::http::service::validate_extension_route_path(&path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
            })?;
        }
        Ok(builder.build())
    })
}

const EXTENSION_POOL_THREADS: usize = 2;
const EXTENSION_QUEUE_DEPTH: usize = 64;
const EXTENSION_HANDLER_TIMEOUT: Duration = Duration::from_secs(30);

type ExtensionJob = Box<dyn FnOnce() + Send + 'static>;

/// Dedicated pool for Python extension handlers, kept off tokio's shared
/// `spawn_blocking` pool so a slow handler can't starve inference tokenization.
/// Small and bounded because handlers are GIL-serialized.
fn extension_executor() -> &'static SyncSender<ExtensionJob> {
    static POOL: OnceLock<SyncSender<ExtensionJob>> = OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = sync_channel::<ExtensionJob>(EXTENSION_QUEUE_DEPTH);
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..EXTENSION_POOL_THREADS {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("frontend-ext-{i}"))
                .spawn(move || {
                    loop {
                        let job = rx.lock().unwrap().recv();
                        match job {
                            // catch_unwind so a panicking handler can't kill the worker.
                            Ok(job) => {
                                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
                                    .is_err()
                                {
                                    tracing::error!("Python frontend route extension panicked");
                                }
                            }
                            Err(_) => break,
                        }
                    }
                })
                .expect("spawn frontend extension worker");
        }
        tx
    })
}

fn extension_error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

async fn call_python_frontend_route(
    handler: Arc<PyObject>,
    context: RsFrontendExtensionContext,
) -> Response {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let job: ExtensionJob = Box::new(move || {
        // Caller already gave up (timed out/disconnected): skip before the GIL.
        if tx.is_closed() {
            return;
        }
        let result = Python::with_gil(|py| call_python_frontend_route_inner(py, &handler, context));
        let _ = tx.send(result);
    });

    // Bounded pool: shed with the shared overload code (DYN_HTTP_OVERLOAD_STATUS_CODE).
    if extension_executor().try_send(job).is_err() {
        tracing::warn!("frontend extension pool saturated; shedding request");
        return extension_error_response(
            dynamo_llm::http::service::error::SanitizedError::Overloaded.status(),
            "frontend route extension busy",
        );
    }

    match tokio::time::timeout(EXTENSION_HANDLER_TIMEOUT, rx).await {
        Ok(Ok(Ok((status, body)))) => (status, Json(body)).into_response(),
        Ok(Ok(Err(err))) => {
            tracing::error!(error = %err, "Python frontend route extension failed");
            extension_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "frontend route extension failed",
            )
        }
        Ok(Err(_)) => {
            // Worker dropped the sender without a value (handler panicked).
            tracing::error!("Python frontend route extension worker dropped without a response");
            extension_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "frontend route extension failed",
            )
        }
        Err(_) => {
            tracing::error!(
                timeout_s = EXTENSION_HANDLER_TIMEOUT.as_secs(),
                "Python frontend route extension timed out"
            );
            extension_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "frontend route extension timed out",
            )
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

fn sorted_strings(values: impl Iterator<Item = String>) -> Vec<String> {
    let mut values: Vec<String> = values.collect();
    values.sort();
    values
}
