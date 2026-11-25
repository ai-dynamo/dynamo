// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use dynamo_llm::http::service::axum::{
    Json,
    body::Bytes,
    extract::{DefaultBodyLimit, rejection::BytesRejection},
    handler::Handler,
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use dynamo_llm::http::service::{
    FrontendExtensionContext as RsFrontendExtensionContext, FrontendRouteExtension,
    FrontendRouteSet,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyDict, PyIterator},
};
use pythonize::depythonize;
use serde_json::{Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A trusted Python-provided frontend route on a static path. `method` is one
/// of GET, POST, PUT, PATCH or DELETE. `handler` takes a
/// `FrontendExtensionContext` and returns a JSON body (200) or a
/// `FrontendResponse` for a custom status; it may be `async def`, in which case
/// it runs on the frontend's event loop. `timeout_s` bounds each call and
/// `max_body_bytes` its request body.
#[pyclass(name = "FrontendRoute")]
#[derive(Clone)]
pub(crate) struct PyFrontendRoute {
    method: Method,
    path: String,
    handler: PyObject,
    timeout: Duration,
    max_body_bytes: usize,
}

#[pymethods]
impl PyFrontendRoute {
    #[new]
    #[pyo3(signature = (method, path, handler, timeout_s = 30.0, max_body_bytes = 64 * 1024))]
    pub fn new(
        py: Python<'_>,
        method: String,
        path: String,
        handler: PyObject,
        timeout_s: f64,
        max_body_bytes: usize,
    ) -> PyResult<Self> {
        if !handler.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "FrontendRoute handler must be callable",
            ));
        }
        // Reuse the core validators so Python and the Rust builder can't drift.
        let method = Method::from_bytes(method.to_ascii_uppercase().as_bytes())
            .map_err(|e| PyValueError::new_err(format!("FrontendRoute method {method:?}: {e}")))?;
        dynamo_llm::http::service::validate_extension_route_method(&method)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dynamo_llm::http::service::validate_extension_route_path(&path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // `try_from_secs_f64` rejects negative, NaN and infinite values.
        let timeout = Duration::try_from_secs_f64(timeout_s)
            .ok()
            .filter(|t| !t.is_zero() && *t <= MAX_TIMEOUT)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "FrontendRoute timeout_s must be positive and at most {} seconds, got {timeout_s}",
                    MAX_TIMEOUT.as_secs()
                ))
            })?;
        Ok(Self {
            method,
            path,
            handler,
            timeout,
            max_body_bytes,
        })
    }

    #[getter]
    pub fn method(&self) -> String {
        self.method.to_string()
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

/// The request and live frontend state passed to Python frontend route handlers.
#[pyclass(name = "FrontendExtensionContext")]
pub(crate) struct PyFrontendExtensionContext {
    inner: RsFrontendExtensionContext,
    /// Raw request body; empty when the request has none.
    #[pyo3(get)]
    body: Py<PyBytes>,
    /// Request headers: lower-case names, repeated headers joined with ", ",
    /// values decoded as lossy UTF-8.
    #[pyo3(get)]
    headers: Py<PyDict>,
    /// The frontend's own `DistributedRuntime`.
    #[pyo3(get)]
    runtime: PyObject,
}

impl PyFrontendExtensionContext {
    fn new(
        py: Python<'_>,
        inner: RsFrontendExtensionContext,
        headers: &HeaderMap,
        body: &[u8],
        runtime: &PyObject,
    ) -> PyResult<Self> {
        let dict = PyDict::new(py);
        for name in headers.keys() {
            let values = headers.get_all(name).iter();
            let values = values.map(|v| String::from_utf8_lossy(v.as_bytes()));
            dict.set_item(name.as_str(), values.collect::<Vec<_>>().join(", "))?;
        }
        Ok(Self {
            inner,
            body: PyBytes::new(py, body).unbind(),
            headers: dict.unbind(),
            runtime: runtime.clone_ref(py),
        })
    }
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

struct PyRouteHandler {
    spec: PyFrontendRoute,
    /// The frontend's asyncio loop; awaitable handler results run here.
    event_loop: PyObject,
    /// The frontend's `DistributedRuntime`, handed to handlers as `ctx.runtime`.
    runtime: PyObject,
    /// Calls in flight: queued or running on the pool, or running on the event
    /// loop. A slot is held until its handler is done, so slow async calls are
    /// shed rather than piled onto the loop. Sync routes also share the pool.
    in_flight: Arc<Semaphore>,
}

pub(crate) fn frontend_route_extensions_from_py(
    py: Python<'_>,
    routes: Option<PyObject>,
    runtime: PyObject,
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
        // Fail here, with a `ValueError` from `run_input`. The route builder
        // rejects a duplicate too, but only when the HTTP service is built.
        if !seen.insert((route.method.clone(), route.path.clone())) {
            return Err(PyValueError::new_err(format!(
                "duplicate frontend route registered: {} {}",
                route.method, route.path
            )));
        }
        route_specs.push(route.clone());
    }

    if route_specs.is_empty() {
        return Ok(Vec::new());
    }
    // `run_input` is awaited from the frontend's event loop, so it is running here.
    let event_loop = py.import("asyncio")?.call_method0("get_running_loop")?;
    // Shared into the route closures by `Arc`: they run on tokio workers, which
    // never hold the GIL that cloning a Python reference needs.
    let routes = route_specs.into_iter().map(|spec| {
        Arc::new(PyRouteHandler {
            spec,
            event_loop: event_loop.clone().unbind(),
            runtime: runtime.clone_ref(py),
            in_flight: Arc::new(Semaphore::new(ROUTE_MAX_IN_FLIGHT)),
        })
    });
    Ok(vec![frontend_route_extension_from_routes(routes.collect())])
}

fn frontend_route_extension_from_routes(
    routes: Vec<Arc<PyRouteHandler>>,
) -> FrontendRouteExtension {
    Arc::new(move |context: RsFrontendExtensionContext| {
        let mut builder = FrontendRouteSet::builder();
        for route in &routes {
            let (route, context) = (route.clone(), context.clone());
            let (method, path) = (route.spec.method.clone(), route.spec.path.clone());
            let body_limit = DefaultBodyLimit::max(route.spec.max_body_bytes);
            let handler = move |headers: HeaderMap, body: Result<Bytes, BytesRejection>| {
                call_python_frontend_route(route.clone(), context.clone(), headers, body)
            };
            builder = builder.route(method, path, handler.layer(body_limit))?;
        }
        Ok(builder.build())
    })
}

const EXTENSION_POOL_THREADS: usize = 2;
const EXTENSION_QUEUE_DEPTH: usize = 64;
const ROUTE_MAX_IN_FLIGHT: usize = 64;
/// Far above any real handler timeout, and small enough that adding it to an
/// `Instant` cannot overflow.
const MAX_TIMEOUT: Duration = Duration::from_secs(365 * 24 * 60 * 60);

type ExtensionJob = Box<dyn FnOnce() + Send + 'static>;

enum RouteError {
    /// The error is rendered under the GIL where it occurs: formatting a
    /// `PyErr` takes the GIL, which a tokio worker must not wait for.
    Failed(String),
    /// The handler was still running at the route's deadline.
    TimedOut,
}
type RouteResult = Result<(StatusCode, Value), RouteError>;

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
    route: Arc<PyRouteHandler>,
    context: RsFrontendExtensionContext,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let (method, path) = (&route.spec.method, route.spec.path.as_str());
    let body = match body {
        Ok(body) => body,
        // 413 for a body over the route's `max_body_bytes`.
        Err(rejection) => {
            return extension_error_response(rejection.status(), &rejection.body_text());
        }
    };
    let deadline = tokio::time::Instant::now() + route.spec.timeout;
    // Bounded: shed with the shared overload code (DYN_HTTP_OVERLOAD_STATUS_CODE).
    let shed = |limit: &str| {
        tracing::warn!(
            %method, path, limit,
            "frontend extension handlers saturated; shedding request"
        );
        extension_error_response(
            dynamo_llm::http::service::error::SanitizedError::Overloaded.status(),
            "frontend route extension busy",
        )
    };
    let Ok(permit) = route.in_flight.clone().try_acquire_owned() else {
        return shed("route_in_flight");
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    let reply = Reply {
        route: route.clone(),
        tx: Some(tx),
        permit: Some(permit),
        deadline: deadline.into_std(),
        timer: None,
    };
    let job: ExtensionJob = Box::new(move || {
        // Caller already gave up (timed out/disconnected): skip before the GIL.
        if !reply.is_closed() {
            Python::with_gil(|py| run_handler(py, context, &headers, &body, reply));
        }
    });
    if extension_executor().try_send(job).is_err() {
        return shed("pool_queue");
    }

    let failure = match tokio::time::timeout_at(deadline, rx).await {
        Ok(Ok(Ok((status, body)))) => return (status, Json(body)).into_response(),
        Ok(Ok(Err(failure))) => failure,
        // The reply was dropped without a value (handler panicked).
        Ok(Err(_)) => RouteError::Failed("dropped without a response".to_string()),
        // A sync handler, or an event loop too busy to cancel an async one.
        Err(_) => RouteError::TimedOut,
    };
    match failure {
        RouteError::Failed(err) => {
            tracing::error!(%method, path, error = %err, "Python frontend route extension failed");
            extension_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "frontend route extension failed",
            )
        }
        RouteError::TimedOut => {
            tracing::error!(
                %method, path, timeout_s = route.spec.timeout.as_secs_f64(),
                "Python frontend route extension timed out"
            );
            extension_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "frontend route extension timed out",
            )
        }
    }
}

/// On a pool thread: call the handler. A plain result is answered here; an
/// awaitable is handed to the event loop, where `Reply::start` runs it.
fn run_handler(
    py: Python<'_>,
    context: RsFrontendExtensionContext,
    headers: &HeaderMap,
    body: &[u8],
    mut reply: Reply,
) {
    // The wait for the GIL may have outlasted the deadline.
    if reply.remaining().is_none() {
        return reply.finish(Err(RouteError::TimedOut));
    }
    let route = reply.route.clone();
    let result = PyFrontendExtensionContext::new(py, context, headers, body, &route.runtime)
        .and_then(|ctx| route.spec.handler.call1(py, (ctx,)));
    let result = match result {
        Ok(result) => result.into_bound(py),
        Err(err) => return reply.finish(Err(RouteError::Failed(err.to_string()))),
    };
    if !result.hasattr("__await__").unwrap_or(false) {
        let response = normalize_route_response(py, &result);
        return reply.finish(response.map_err(|e| RouteError::Failed(e.to_string())));
    }
    // On failure `reply` is dropped unanswered; the caller answers 500.
    let Ok(reply) = Bound::new(py, reply) else {
        return;
    };
    let scheduled = reply.getattr("start").and_then(|start| {
        route
            .event_loop
            .call_method1(py, "call_soon_threadsafe", (start, &result))
    });
    if let Err(err) = scheduled {
        Reply::abort(&reply, &result, RouteError::Failed(err.to_string()));
    }
}

/// Answers one request at most once, and holds its in-flight slot until the
/// handler is done rather than until the caller stops waiting.
/// `pyo3_async_runtimes::into_future_with_locals` can do neither that nor
/// cancel the task at the deadline.
#[pyclass]
struct Reply {
    route: Arc<PyRouteHandler>,
    tx: Option<tokio::sync::oneshot::Sender<RouteResult>>,
    permit: Option<OwnedSemaphorePermit>,
    deadline: std::time::Instant,
    /// The event loop's deadline timer, until it fires or the handler is done.
    timer: Option<PyObject>,
}

impl Reply {
    fn is_closed(&self) -> bool {
        self.tx.as_ref().is_none_or(|tx| tx.is_closed())
    }

    /// Time left for the handler, or `None` once the caller stopped waiting.
    fn remaining(&self) -> Option<Duration> {
        let remaining = self
            .deadline
            .saturating_duration_since(std::time::Instant::now());
        (!self.is_closed() && !remaining.is_zero()).then_some(remaining)
    }

    /// Answer the caller. The slot stays held until `finish`.
    fn answer(&mut self, result: RouteResult) {
        let Some(tx) = self.tx.take() else { return };
        // The caller is gone, so its side will never log this failure.
        if let Err(Err(RouteError::Failed(err))) = tx.send(result) {
            let (method, path) = (&self.route.spec.method, self.route.spec.path.as_str());
            tracing::error!(
                %method, path, error = %err,
                "Python frontend route extension failed after its caller stopped waiting"
            );
        }
    }

    fn finish(&mut self, result: RouteResult) {
        self.permit = None;
        self.answer(result);
    }

    /// Close an awaitable that will not run, and answer the caller.
    fn abort(slf: &Bound<'_, Self>, awaitable: &Bound<'_, PyAny>, err: RouteError) {
        let _ = awaitable.call_method0("close");
        slf.borrow_mut().finish(Err(err));
    }
}

#[pymethods]
impl Reply {
    /// On the event loop: run `awaitable` as a task that `expire` cancels at
    /// the deadline. A call the loop picks up after its deadline, or whose
    /// caller is gone, is skipped.
    fn start(slf: &Bound<'_, Self>, awaitable: &Bound<'_, PyAny>) {
        let remaining = slf.borrow().remaining();
        let Some(remaining) = remaining else {
            return Self::abort(slf, awaitable, RouteError::TimedOut);
        };
        let started = slf.py().import("asyncio").and_then(|asyncio| {
            let task = asyncio.call_method1("ensure_future", (awaitable,))?;
            task.call_method1("add_done_callback", (slf.getattr("done")?,))?;
            let timer = asyncio.call_method0("get_running_loop")?.call_method1(
                "call_later",
                (remaining.as_secs_f64(), slf.getattr("expire")?, task),
            )?;
            slf.borrow_mut().timer = Some(timer.unbind());
            Ok(())
        });
        if let Err(err) = started {
            Self::abort(slf, awaitable, RouteError::Failed(err.to_string()));
        }
    }

    /// On the event loop, at the deadline: cancel the handler and answer the
    /// caller. A handler that has just finished keeps its own answer.
    fn expire(&mut self, task: &Bound<'_, PyAny>) -> PyResult<()> {
        self.timer = None;
        if task.call_method0("cancel")?.is_truthy()? {
            self.answer(Err(RouteError::TimedOut));
        }
        Ok(())
    }

    /// On the event loop: the handler's task finished.
    fn done(&mut self, task: &Bound<'_, PyAny>) {
        let py = task.py();
        if let Some(timer) = self.timer.take() {
            let _ = timer.call_method0(py, "cancel");
        }
        let response = task
            .call_method0("result")
            .and_then(|result| normalize_route_response(py, &result));
        self.finish(response.map_err(|e| RouteError::Failed(e.to_string())));
    }
}

fn normalize_route_response(
    py: Python<'_>,
    bound: &Bound<'_, PyAny>,
) -> PyResult<(StatusCode, Value)> {
    // Explicit status override via FrontendResponse.
    if let Ok(resp) = bound.extract::<PyRef<'_, PyFrontendResponse>>() {
        let status = StatusCode::from_u16(resp.status_code)
            .map_err(|e| PyValueError::new_err(format!("invalid status code: {e}")))?;
        let body: Value = depythonize(resp.body.bind(py)).map_err(|e| {
            PyValueError::new_err(format!("response body must be JSON-serializable: {e}"))
        })?;
        return Ok((status, body));
    }

    // Otherwise a JSON body with status 200 (tuples serialize as JSON arrays).
    let body: Value = depythonize(bound).map_err(|e| {
        PyValueError::new_err(format!("response body must be JSON-serializable: {e}"))
    })?;
    Ok((StatusCode::OK, body))
}

fn sorted_strings(values: impl Iterator<Item = String>) -> Vec<String> {
    let mut values: Vec<String> = values.collect();
    values.sort();
    values
}
