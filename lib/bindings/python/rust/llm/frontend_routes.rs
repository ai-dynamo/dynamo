// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use dynamo_llm::http::service::axum::{
    Json,
    body::Bytes,
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
use tokio::sync::{Semaphore, SemaphorePermit};

/// A trusted Python-provided frontend route on a static path. `method` is one
/// of GET, POST, PUT, PATCH or DELETE. `handler` takes a
/// `FrontendExtensionContext` and returns a JSON body (200) or a
/// `FrontendResponse` for a custom status; it may be `async def`, in which case
/// it runs on the frontend's event loop. `timeout_s` bounds each call.
#[pyclass(name = "FrontendRoute")]
#[derive(Clone)]
pub(crate) struct PyFrontendRoute {
    method: Method,
    path: String,
    handler: PyObject,
    timeout: Duration,
}

#[pymethods]
impl PyFrontendRoute {
    #[new]
    #[pyo3(signature = (method, path, handler, timeout_s = 30.0))]
    pub fn new(
        py: Python<'_>,
        method: String,
        path: String,
        handler: PyObject,
        timeout_s: f64,
    ) -> PyResult<Self> {
        if !handler.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "FrontendRoute handler must be callable",
            ));
        }
        // Reuse the core validators so Python and the Rust builder can't drift.
        let method = dynamo_llm::http::service::validate_extension_route_method(&method)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dynamo_llm::http::service::validate_extension_route_path(&path)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let timeout = Some(timeout_s)
            .filter(|t| t.is_finite() && *t > 0.0)
            .and_then(|t| Duration::try_from_secs_f64(t).ok())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "FrontendRoute timeout_s must be a positive number of seconds, got {timeout_s}"
                ))
            })?;
        Ok(Self {
            method,
            path,
            handler,
            timeout,
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

    #[getter]
    pub fn timeout_s(&self) -> f64 {
        self.timeout.as_secs_f64()
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
    body: Bytes,
    headers: HeaderMap,
    runtime: PyObject,
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

    /// Raw request body; empty when the request has none.
    #[getter]
    pub fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.body)
    }

    /// Request headers as a new dict: lower-case names, repeated headers
    /// joined with ", ", values decoded as lossy UTF-8.
    #[getter]
    pub fn headers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in self.headers.keys() {
            let value = self
                .headers
                .get_all(name)
                .iter()
                .map(|v| String::from_utf8_lossy(v.as_bytes()))
                .collect::<Vec<_>>()
                .join(", ");
            dict.set_item(name.as_str(), value)?;
        }
        Ok(dict)
    }

    /// The frontend's own `DistributedRuntime`.
    #[getter]
    pub fn runtime(&self, py: Python<'_>) -> PyObject {
        self.runtime.clone_ref(py)
    }
}

/// Frontend-process objects every route handler shares.
struct PyRouteEnv {
    /// The frontend's asyncio loop; awaitable handler results run here.
    event_loop: PyObject,
    /// The frontend's `DistributedRuntime` Python object.
    runtime: PyObject,
}

struct PyRouteHandler {
    handler: PyObject,
    timeout: Duration,
    env: Arc<PyRouteEnv>,
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
        // All providers fold into one router; Router::route panics on an
        // overlapping method and path, so reject duplicates cleanly here.
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
    let event_loop = py
        .import("asyncio")?
        .call_method0("get_running_loop")
        .map_err(|e| {
            PyValueError::new_err(format!(
                "frontend route extensions must be registered from a running asyncio event loop: {e}"
            ))
        })?
        .unbind();
    let env = Arc::new(PyRouteEnv {
        event_loop,
        runtime,
    });
    Ok(vec![frontend_route_extension_from_routes(route_specs, env)])
}

fn frontend_route_extension_from_routes(
    routes: Vec<PyFrontendRoute>,
    env: Arc<PyRouteEnv>,
) -> FrontendRouteExtension {
    let routes = Arc::new(routes);
    Arc::new(move |context: RsFrontendExtensionContext| {
        let mut builder = FrontendRouteSet::builder();
        for route in routes.iter() {
            // Clone the handler under the GIL (`clone_ref` requires it), then
            // share it into the handler closure via `Arc` (GIL-free clone) —
            // the closure runs on tokio workers that don't hold the GIL.
            let handler = Arc::new(PyRouteHandler {
                handler: Python::with_gil(|py| route.handler.clone_ref(py)),
                timeout: route.timeout,
                env: env.clone(),
            });
            let route_context = context.clone();
            builder = builder.route(
                route.method.clone(),
                route.path.clone(),
                move |headers: HeaderMap, body: Bytes| {
                    call_python_frontend_route(
                        handler.clone(),
                        route_context.clone(),
                        headers,
                        body,
                    )
                },
            )?;
        }
        Ok(builder.build())
    })
}

const EXTENSION_POOL_THREADS: usize = 2;
const EXTENSION_QUEUE_DEPTH: usize = 64;

/// Handlers in flight: queued or running on the pool, or running on the event
/// loop. A permit is held until the Python handler actually finishes, so a
/// flood of slow async calls is shed rather than piled onto the loop.
static IN_FLIGHT: Semaphore = Semaphore::const_new(EXTENSION_QUEUE_DEPTH);

type ExtensionJob = Box<dyn FnOnce() + Send + 'static>;
type RouteResult = Result<(StatusCode, Value), RouteError>;

/// Rendered under the GIL where it occurs: formatting a `PyErr` takes the GIL,
/// which a tokio worker must not wait for.
enum RouteError {
    Failed(String),
    /// `asyncio.wait_for` hit the route's deadline on the event loop.
    TimedOut,
}

impl RouteError {
    fn from_py(py: Python<'_>, err: PyErr) -> Self {
        let timed_out = py
            .import("asyncio")
            .and_then(|asyncio| asyncio.getattr("TimeoutError"))
            .is_ok_and(|timeout| err.is_instance(py, &timeout));
        if timed_out {
            Self::TimedOut
        } else {
            Self::Failed(err.to_string())
        }
    }
}
type RouteSender = tokio::sync::oneshot::Sender<RouteResult>;

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
    body: Bytes,
) -> Response {
    let timeout = route.timeout;
    let deadline = tokio::time::Instant::now() + timeout;
    // Bounded: shed with the shared overload code (DYN_HTTP_OVERLOAD_STATUS_CODE).
    let shed = || {
        tracing::warn!("frontend extension handlers saturated; shedding request");
        extension_error_response(
            dynamo_llm::http::service::error::SanitizedError::Overloaded.status(),
            "frontend route extension busy",
        )
    };
    let Ok(permit) = IN_FLIGHT.try_acquire() else {
        return shed();
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    let job: ExtensionJob = Box::new(move || {
        // Caller already gave up (timed out/disconnected): skip before the GIL.
        if tx.is_closed() {
            return;
        }
        let request = PendingRequest {
            context,
            headers,
            body,
            deadline: deadline.into_std(),
            permit,
        };
        Python::with_gil(|py| start_python_frontend_route(py, &route, request, tx));
    });
    if extension_executor().try_send(job).is_err() {
        return shed();
    }

    match tokio::time::timeout_at(deadline, rx).await {
        Ok(Ok(Ok((status, body)))) => (status, Json(body)).into_response(),
        Ok(Ok(Err(RouteError::Failed(err)))) => {
            tracing::error!(error = %err, "Python frontend route extension failed");
            extension_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "frontend route extension failed",
            )
        }
        Ok(Err(_)) => {
            // The sender was dropped without a value (handler panicked).
            tracing::error!("Python frontend route extension dropped without a response");
            extension_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "frontend route extension failed",
            )
        }
        // The loop's `wait_for` and this timer share one deadline; either may fire first.
        Ok(Ok(Err(RouteError::TimedOut))) | Err(_) => {
            tracing::error!(
                timeout_s = timeout.as_secs_f64(),
                "Python frontend route extension timed out"
            );
            extension_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "frontend route extension timed out",
            )
        }
    }
}

/// A request handed from a tokio worker to an extension pool thread.
struct PendingRequest {
    context: RsFrontendExtensionContext,
    headers: HeaderMap,
    body: Bytes,
    /// When the caller stops waiting; an async handler is cancelled then too.
    deadline: std::time::Instant,
    permit: SemaphorePermit<'static>,
}

/// Call the handler on a pool thread. A plain result is answered now; an
/// awaitable is scheduled on the frontend's event loop and answered by a
/// `RouteCompleter` there, so no tokio worker ever waits on the GIL.
fn start_python_frontend_route(
    py: Python<'_>,
    route: &PyRouteHandler,
    request: PendingRequest,
    tx: RouteSender,
) {
    let PendingRequest {
        context,
        headers,
        body,
        deadline,
        permit,
    } = request;
    let ctx = PyFrontendExtensionContext {
        inner: context,
        body,
        headers,
        runtime: route.env.runtime.clone_ref(py),
    };
    let result = match Py::new(py, ctx).and_then(|ctx| route.handler.call1(py, (ctx,))) {
        Ok(result) => result.into_bound(py),
        Err(err) => {
            let _ = tx.send(Err(RouteError::from_py(py, err)));
            return;
        }
    };
    if !result.hasattr("__await__").unwrap_or(false) {
        let response = normalize_route_response(py, result.unbind());
        let _ = tx.send(response.map_err(|e| RouteError::from_py(py, e)));
        return;
    }

    let completer = RouteCompleter {
        tx: Some(tx),
        permit: Some(permit),
    };
    let completer = match Py::new(py, completer) {
        Ok(completer) => completer,
        // `tx` is dropped with the error; the caller answers 500.
        Err(err) => {
            tracing::error!(error = %err, "failed to create frontend route completer");
            return;
        }
    };
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    if let Err(err) = schedule_on_event_loop(py, route, &result, &completer, remaining) {
        // Close an unscheduled coroutine to avoid an unawaited-coroutine warning.
        let _ = result.call_method0("close");
        completer
            .borrow_mut(py)
            .finish(Err(RouteError::from_py(py, err)));
    }
}

/// Run `awaitable` on the frontend's event loop; `asyncio.wait_for` cancels it
/// there once `remaining` elapses.
fn schedule_on_event_loop(
    py: Python<'_>,
    route: &PyRouteHandler,
    awaitable: &Bound<'_, PyAny>,
    completer: &Py<RouteCompleter>,
    remaining: Duration,
) -> PyResult<()> {
    let asyncio = py.import("asyncio")?;
    let bounded = asyncio.call_method1("wait_for", (awaitable, remaining.as_secs_f64()))?;
    let future = asyncio
        .call_method1(
            "run_coroutine_threadsafe",
            (&bounded, route.env.event_loop.bind(py)),
        )
        .inspect_err(|_| {
            let _ = bounded.call_method0("close");
        })?;
    future.call_method1("add_done_callback", (completer.clone_ref(py),))?;
    Ok(())
}

/// Done-callback for an awaitable handler's `concurrent.futures.Future`. It
/// runs on the event loop thread, which already holds the GIL.
#[pyclass]
struct RouteCompleter {
    tx: Option<RouteSender>,
    /// Released when the handler finishes, not when the caller stops waiting.
    permit: Option<SemaphorePermit<'static>>,
}

#[pymethods]
impl RouteCompleter {
    fn __call__(&mut self, py: Python<'_>, future: &Bound<'_, PyAny>) {
        let result = future
            .call_method0("result")
            .and_then(|value| normalize_route_response(py, value.unbind()))
            .map_err(|e| RouteError::from_py(py, e));
        self.finish(result);
    }
}

impl RouteCompleter {
    fn finish(&mut self, result: RouteResult) {
        self.permit.take();
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
    }
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
