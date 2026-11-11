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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

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
        // `try_from_secs_f64` rejects negative, NaN and infinite values.
        let timeout = Duration::try_from_secs_f64(timeout_s)
            .ok()
            .filter(|t| !t.is_zero())
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
    env: Arc<PyRouteEnv>,
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
        self.env.runtime.clone_ref(py)
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
    /// This route's calls in flight: queued or running on the pool, or running
    /// on the event loop. Per route, so a slow route can't shed the others.
    in_flight: Arc<Semaphore>,
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
        .call_method0("get_running_loop")?
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
                in_flight: Arc::new(Semaphore::new(ROUTE_MAX_IN_FLIGHT)),
                env: env.clone(),
            });
            let context = context.clone();
            builder = builder.route(
                route.method.clone(),
                route.path.clone(),
                move |headers: HeaderMap, body: Bytes| {
                    let ctx = PyFrontendExtensionContext {
                        inner: context.clone(),
                        body,
                        headers,
                        env: handler.env.clone(),
                    };
                    call_python_frontend_route(handler.clone(), ctx)
                },
            )?;
        }
        Ok(builder.build())
    })
}

const EXTENSION_POOL_THREADS: usize = 2;
const EXTENSION_QUEUE_DEPTH: usize = 64;
const ROUTE_MAX_IN_FLIGHT: usize = 64;
/// The event loop's timeout can reach the caller before the deadline: uvloop
/// timers are millisecond-quantized and fire up to 1.5 ms early.
const DEADLINE_SLACK: Duration = Duration::from_millis(10);

type ExtensionJob = Box<dyn FnOnce() + Send + 'static>;
/// The error is rendered under the GIL where it occurs: formatting a `PyErr`
/// takes the GIL, which a tokio worker must not wait for.
type RouteResult = Result<(StatusCode, Value), String>;
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
    ctx: PyFrontendExtensionContext,
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
    // Held until the Python handler actually finishes, so a flood of slow async
    // calls is shed rather than piled onto the loop.
    let Ok(permit) = route.in_flight.clone().try_acquire_owned() else {
        return shed();
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    let reply = Reply {
        tx: Some(tx),
        permit: Some(permit),
        deadline: deadline.into_std(),
    };
    let job: ExtensionJob = Box::new(move || {
        // Caller already gave up (timed out/disconnected): skip before the GIL.
        if !reply.is_closed() {
            Python::with_gil(|py| run_handler(py, &route, ctx, reply));
        }
    });
    if extension_executor().try_send(job).is_err() {
        return shed();
    }

    let result = tokio::time::timeout_at(deadline, rx)
        .await
        // The reply was dropped without a value (handler panicked).
        .map(|reply| reply.unwrap_or_else(|_| Err("dropped without a response".to_string())));
    match result {
        Ok(Ok((status, body))) => (status, Json(body)).into_response(),
        // Before the deadline an error is the handler's own, a `TimeoutError`
        // it raised included. At the deadline it is the event loop cancelling
        // the handler, which shares this timer's deadline.
        Ok(Err(err)) if tokio::time::Instant::now() + DEADLINE_SLACK < deadline => {
            tracing::error!(error = %err, "Python frontend route extension failed");
            extension_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "frontend route extension failed",
            )
        }
        _ => {
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

/// On a pool thread: call the handler. A plain result is answered here; an
/// awaitable is handed to the event loop, where `Reply::start` runs it.
fn run_handler(
    py: Python<'_>,
    route: &PyRouteHandler,
    ctx: PyFrontendExtensionContext,
    mut reply: Reply,
) {
    let result = match Py::new(py, ctx).and_then(|ctx| route.handler.call1(py, (ctx,))) {
        Ok(result) => result.into_bound(py),
        Err(err) => return reply.finish(Err(err.to_string())),
    };
    if !result.hasattr("__await__").unwrap_or(false) {
        let response = normalize_route_response(py, &result);
        return reply.finish(response.map_err(|e| e.to_string()));
    }
    // On failure `reply` is dropped unanswered; the caller answers 500.
    let Ok(reply) = Py::new(py, reply) else {
        return;
    };
    let scheduled = reply.bind(py).getattr("start").and_then(|start| {
        route
            .env
            .event_loop
            .call_method1(py, "call_soon_threadsafe", (start, &result))
    });
    if let Err(err) = scheduled {
        let _ = result.call_method0("close");
        reply.borrow_mut(py).finish(Err(err.to_string()));
    }
}

/// Answers one request at most once, and holds its in-flight slot until the
/// handler is done rather than until the caller stops waiting.
#[pyclass]
struct Reply {
    tx: Option<RouteSender>,
    permit: Option<OwnedSemaphorePermit>,
    deadline: std::time::Instant,
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

    fn finish(&mut self, result: RouteResult) {
        self.permit = None;
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(result);
        }
    }
}

#[pymethods]
impl Reply {
    /// On the event loop: run `awaitable` until the deadline. A call the loop
    /// picks up after its deadline, or whose caller is gone, is skipped.
    fn start(slf: &Bound<'_, Self>, awaitable: &Bound<'_, PyAny>) {
        let abort = |err: String| {
            let _ = awaitable.call_method0("close");
            slf.borrow_mut().finish(Err(err));
        };
        let remaining = slf.borrow().remaining();
        let Some(remaining) = remaining else {
            return abort("deadline passed before the handler started".to_string());
        };
        let started = slf.py().import("asyncio").and_then(|asyncio| {
            let bounded = asyncio.call_method1("wait_for", (awaitable, remaining.as_secs_f64()))?;
            let task = asyncio.call_method1("ensure_future", (bounded,))?;
            task.call_method1("add_done_callback", (slf.getattr("done")?,))
        });
        if let Err(err) = started {
            abort(err.to_string());
        }
    }

    /// On the event loop: the handler's task finished.
    fn done(&mut self, task: &Bound<'_, PyAny>) {
        let py = task.py();
        let response = task
            .call_method0("result")
            .and_then(|result| normalize_route_response(py, &result));
        self.finish(response.map_err(|e| e.to_string()));
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
