// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::{Error, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::{PyAny, PyErr};
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};
pub use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
use dynamo_runtime::logging::get_distributed_tracing_context;
pub use dynamo_runtime::{
    pipeline::{AsyncEngine, AsyncEngineContextProvider, Data, ManyOut, ResponseStream, SingleIn},
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::PyAsyncRequestStream;
use dynamo_runtime::pipeline::ManyIn;

use super::context::{Context, callable_accepts_kwarg};
use super::errors::py_exception_to_backend_error;

/// Add bindings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonAsyncEngine>()?;
    Ok(())
}
// todos:
// - [ ] enable context cancellation
//   - this will likely require a change to the function signature python calling arguments
// - [ ] other `AsyncEngine` implementations will have a similar pattern, i.e. one AsyncEngine
//       implementation per struct

/// Detect whether the Python `generate` callable accepts a `context`
/// keyword argument. Both engines pass the per-request [`Context`] as a
/// `context=` kwarg when the callable opts in, and otherwise fall back to
/// the legacy positional-only call.
fn detect_has_context(generator: &PyObject) -> bool {
    Python::with_gil(|py| {
        let callable = generator.bind(py);
        callable_accepts_kwarg(py, callable, "context").unwrap_or(false)
    })
}

/// Rust/Python bridge that maps to the [`AsyncEngine`] trait
///
/// Currently this is only implemented for the [`SingleIn`] and [`ManyOut`] types; however,
/// more [`AsyncEngine`] implementations can be added in the future.
///
/// For the [`SingleIn`] and [`ManyOut`] case, this implementation will take a Python async
/// generator and convert it to a Rust async stream.
///
/// ```python
/// class ComputeEngine:
///     def __init__(self):
///         self.compute_engine = make_compute_engine()
///
///     def generate(self, request):
///         async generator():
///            async for output in self.compute_engine.generate(request):
///                yield output
///         return generator()
///
/// def main():
///     loop = asyncio.create_event_loop()
///     compute_engine = ComputeEngine()
///     engine = PythonAsyncEngine(compute_engine.generate, loop)
///     service = RustService()
///     service.add_engine("model_name", engine)
///     loop.run_until_complete(service.run())
/// ```
#[pyclass]
#[derive(Clone)]
pub struct PythonAsyncEngine(PythonServerStreamingEngine);

#[pymethods]
impl PythonAsyncEngine {
    /// Create a new instance of the PythonAsyncEngine
    ///
    /// # Arguments
    /// - `generator`: a Python async generator that will be used to generate responses
    /// - `event_loop`: the Python event loop that will be used to run the generator
    ///
    /// Note: In Rust land, the request and the response are both concrete; however, in
    /// Python land, the request and response not strongly typed, meaning the generator
    /// could accept a different type of request or return a different type of response
    /// and we would not know until runtime.
    #[new]
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        Ok(PythonAsyncEngine(PythonServerStreamingEngine::new(
            cancel_token,
            Arc::new(generator),
            Arc::new(event_loop),
        )))
    }
}

#[async_trait::async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error> for PythonAsyncEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        self.0.generate(request).await
    }
}

#[derive(Clone)]
pub struct PythonServerStreamingEngine {
    _cancel_token: CancellationToken,
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    has_context: bool,
}

impl PythonServerStreamingEngine {
    pub fn new(
        cancel_token: CancellationToken,
        generator: Arc<PyObject>,
        event_loop: Arc<PyObject>,
    ) -> Self {
        let has_context = detect_has_context(&generator);

        PythonServerStreamingEngine {
            _cancel_token: cancel_token,
            generator,
            event_loop,
            has_context,
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum ResponseProcessingError {
    #[error("dynamo error")]
    Dynamo(DynamoError),

    #[error("deserialize error: {0}")]
    Deserialize(String),

    #[error("gil offload error: {0}")]
    Offload(String),
}

#[async_trait::async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Annotated<Resp>>, Error>
    for PythonServerStreamingEngine
where
    Req: Data + Serialize,
    Resp: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<Req>) -> Result<ManyOut<Annotated<Resp>>, Error> {
        // Create a context
        let (request, context) = request.transfer(());
        let ctx = context.context();

        let id = context.id().to_string();
        tracing::trace!("processing request: {}", id);

        // Capture current trace context
        let current_trace_context = get_distributed_tracing_context();

        // Clone the PyObject to move into the thread

        // Create a channel to communicate between the Python thread and the Rust async context
        let (tx, rx) = mpsc::channel::<Annotated<Resp>>(128);

        let generator = self.generator.clone();
        let event_loop = self.event_loop.clone();
        let ctx_python = ctx.clone();
        let has_context = self.has_context;
        let metadata = context.metadata().clone();

        // Acquiring the GIL is similar to acquiring a standard lock/mutex
        // Performing this in an tokio async task could block the thread for an undefined amount of time
        // To avoid this, we spawn a blocking task to acquire the GIL and perform the operations needed
        // while holding the GIL.
        //
        // Under low GIL contention, we wouldn't need to do this.
        // However, under high GIL contention, this can lead to significant performance degradation.
        //
        // Since we cannot predict the GIL contention, we will always use the blocking task and pay the
        // cost. The Python GIL is the gift that keeps on giving -- performance hits...
        let stream = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_request = pythonize(py, &request)?;

                // Create context with trace information
                let py_ctx = Py::new(
                    py,
                    Context::new(ctx_python.clone(), current_trace_context, None, metadata),
                )?;

                let gen_result = if has_context {
                    // Pass context as a kwarg
                    let kwarg = PyDict::new(py);
                    kwarg.set_item("context", &py_ctx)?;
                    generator.call(py, (py_request,), Some(&kwarg))
                } else {
                    // Legacy: No `context` arg
                    generator.call1(py, (py_request,))
                }?;

                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(
                    locals,
                    gen_result.into_bound(py),
                )
            })
        })
        .await??;

        let stream = Box::pin(stream);

        // process the stream
        // any error thrown in the stream will be caught and complete the processing task
        // errors are captured by a task that is watching the processing task
        // the error will be emitted as an annotated error
        let request_id = id.clone();

        tokio::spawn(async move {
            tracing::debug!(
                request_id,
                "starting task to process python async generator stream"
            );

            let mut stream = stream;
            let mut count = 0;

            while let Some(item) = stream.next().await {
                count += 1;
                tracing::trace!(
                    request_id,
                    "processing the {}th item from python async generator",
                    count
                );

                let mut done = false;

                let response = match process_item::<Resp>(item).await {
                    Ok(response) => response,
                    Err(e) => {
                        done = true;

                        match e {
                            ResponseProcessingError::Deserialize(e) => {
                                // tell the python async generator to stop generating
                                // right now, this is impossible as we are not passing the context to the python async generator
                                // todo: add task-local context to the python async generator
                                ctx.stop_generating();
                                Annotated::from_error(format!(
                                    "critical error: invalid response object from python async generator; application-logic-mismatch: {}",
                                    e
                                ))
                            }
                            ResponseProcessingError::Dynamo(dynamo_err) => {
                                Annotated::from_err(dynamo_err)
                            }
                            ResponseProcessingError::Offload(e) => Annotated::from_error(format!(
                                "critical error: failed to offload the python async generator to a new thread: {}",
                                e
                            )),
                        }
                    }
                };

                if tx.send(response).await.is_err() {
                    tracing::trace!(
                        request_id,
                        "error forwarding annotated response to channel; channel is closed"
                    );
                    break;
                }

                if done {
                    tracing::debug!(
                        request_id,
                        "early termination of python async generator stream task"
                    );
                    break;
                }
            }

            tracing::debug!(
                request_id,
                "finished processing python async generator stream"
            );
        });

        let stream = ReceiverStream::new(rx);

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

async fn process_item<Resp>(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(|e| {
        Python::with_gil(|py| {
            e.display(py);

            // Check if the Python exception is a Dynamo error type.
            // Wrap as Backend* since this is the backend engine context.
            if let Some((backend_err, message)) = py_exception_to_backend_error(py, &e) {
                return ResponseProcessingError::Dynamo(
                    DynamoError::builder()
                        .error_type(ErrorType::Backend(backend_err))
                        .message(message)
                        .build(),
                );
            }

            // GeneratorExit from Python's generator protocol (e.g., GC closing
            // a generator) is treated as an engine shutdown.
            if e.is_instance_of::<pyo3::exceptions::PyGeneratorExit>(py) {
                return ResponseProcessingError::Dynamo(
                    DynamoError::builder()
                        .error_type(ErrorType::Backend(BackendError::EngineShutdown))
                        .message("engine shutting down")
                        .build(),
                );
            }

            // Map well-known Python exceptions to specific Backend error types.
            // Order matters: check subclasses before their parents
            // (e.g., ConnectionRefusedError before ConnectionError).
            let backend_err = if e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
            {
                BackendError::InvalidArgument
            } else if e.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py) {
                BackendError::ConnectionTimeout
            } else if e.is_instance_of::<pyo3::exceptions::PyConnectionRefusedError>(py) {
                BackendError::CannotConnect
            } else if e.is_instance_of::<pyo3::exceptions::PyConnectionResetError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyBrokenPipeError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyConnectionError>(py)
            {
                BackendError::Disconnected
            } else if e.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py) {
                BackendError::Cancelled
            } else {
                BackendError::Unknown
            };

            ResponseProcessingError::Dynamo(
                DynamoError::builder()
                    .error_type(ErrorType::Backend(backend_err))
                    .message(e.to_string())
                    .build(),
            )
        })
    })?;
    let response = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| depythonize::<Resp>(&item.into_bound(py)))
    })
    .await
    .map_err(|e| ResponseProcessingError::Offload(e.to_string()))?
    .map_err(|e| ResponseProcessingError::Deserialize(e.to_string()))?;

    let response = Annotated::from_data(response);

    Ok(response)
}

/// Channel depth between the inbound forwarder and the Python iterator.
/// Mirrors the depth used by the wire-side bidirectional ingress forwarder
/// in `lib/runtime/src/pipeline/network/ingress/push_handler.rs`.
const BIDIRECTIONAL_INPUT_CHANNEL_DEPTH: usize = 8;

/// Rust-side adapter that bridges a Python `async def generate(request_stream, context)`
/// callable into an [`AsyncEngine`] of the streaming-input / streaming-output
/// shape (`ManyIn<serde_json::Value>` →
/// `ManyOut<Annotated<serde_json::Value>>`).
///
/// The adapter:
///
/// 1. Takes ownership of the inbound `RequestStream<serde_json::Value>` and
///    spawns a forwarder that pythonizes each frame and pushes it onto an
///    mpsc consumed by a [`PyAsyncRequestStream`]. Cancellation is not
///    threaded through the forwarder — when the engine returns and the
///    `PyAsyncRequestStream` is dropped, the receiver closes and the
///    forwarder exits passively on the next `tx.send`. Cancellation
///    observation is the Python engine's responsibility via the `context`
///    argument, matching the unary engine pattern.
/// 2. Invokes the Python generator with `(request_stream, context)`, then
///    wraps the returned async generator with
///    [`pyo3_async_runtimes::tokio::into_stream_with_locals_v1`] to obtain
///    a Rust `Stream<Item = PyResult<PyObject>>`.
/// 3. Depythonizes each item into a `serde_json::Value`, wraps it as
///    `Annotated::from_data(value)`, and forwards it on the response
///    stream. The `Annotated` envelope satisfies the `MaybeError` bound
///    on the response side; the Python user still writes engines that
///    yield plain dicts.
///
/// Wire types are fixed to `serde_json::Value` on the request side and
/// `Annotated<serde_json::Value>` on the response side — the Python user
/// works with dicts on both sides and any schema enforcement lives in
/// Python.
pub struct PythonBidirectionalEngine {
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    has_context: bool,
}

impl PythonBidirectionalEngine {
    /// Build the adapter from a Python callable and an event loop. The
    /// callable should be an `async def generate(request_stream)` or
    /// `async def generate(request_stream, context)` returning an async
    /// generator of JSON-shaped response frames.
    pub fn new(generator: PyObject, event_loop: PyObject) -> PyResult<Self> {
        let has_context = detect_has_context(&generator);
        Ok(Self {
            generator: Arc::new(generator),
            event_loop: Arc::new(event_loop),
            has_context,
        })
    }
}

#[async_trait::async_trait]
impl AsyncEngine<ManyIn<serde_json::Value>, ManyOut<Annotated<serde_json::Value>>, Error>
    for PythonBidirectionalEngine
{
    async fn generate(
        &self,
        input: ManyIn<serde_json::Value>,
    ) -> Result<ManyOut<Annotated<serde_json::Value>>, Error> {
        let request_id = input.context().id().to_string();
        let ctx = input.context();
        let metadata = input.metadata().clone();
        let (request_stream, ctx_unit) = input.into_parts();
        let mut inbound = request_stream
            .take()
            .ok_or_else(|| anyhow::anyhow!("RequestStream::take returned None"))?;

        // Capture trace context once, while we still hold the
        // dispatching task; needed when constructing the Python `Context`.
        let current_trace_context = get_distributed_tracing_context();

        // Forwarder: pull `serde_json::Value` frames off the inbound
        // stream, pythonize each, and hand the resulting `PyObject` to the
        // Python iterator. No cancellation arm; the channel close on the
        // consumer side is the natural exit signal.
        let (frame_tx, frame_rx) = mpsc::channel::<PyObject>(BIDIRECTIONAL_INPUT_CHANNEL_DEPTH);
        let forwarder_request_id = request_id.clone();
        tokio::spawn(async move {
            while let Some(value) = inbound.next().await {
                let pyobj = match Python::with_gil(|py| {
                    pythonize(py, &value).map(|bound| bound.unbind())
                }) {
                    Ok(pyobj) => pyobj,
                    Err(e) => {
                        tracing::error!(
                            request_id = %forwarder_request_id,
                            error = %e,
                            "failed to pythonize bidirectional request frame; \
                             closing input forwarder"
                        );
                        break;
                    }
                };
                if frame_tx.send(pyobj).await.is_err() {
                    tracing::debug!(
                        request_id = %forwarder_request_id,
                        "python engine dropped request stream; input forwarder exiting"
                    );
                    break;
                }
            }
        });

        let py_request_stream = PyAsyncRequestStream::new(frame_rx);

        let generator = self.generator.clone();
        let event_loop = self.event_loop.clone();
        let has_context = self.has_context;
        let ctx_python = ctx_unit.context();

        // Acquire the GIL on a blocking task so the tokio reactor is not
        // parked while waiting on it. Once inside the GIL: build the
        // Python `Context`, build the `PyAsyncRequestStream` py handle,
        // call the user callable, then convert the returned async
        // generator into a Rust stream of `PyObject`.
        let stream = tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let py_request_stream_obj = Py::new(py, py_request_stream)?;
                let py_ctx = Py::new(
                    py,
                    Context::new(ctx_python.clone(), current_trace_context, None, metadata),
                )?;
                let gen_result = if has_context {
                    let kwarg = PyDict::new(py);
                    kwarg.set_item("context", &py_ctx)?;
                    generator.call(py, (py_request_stream_obj,), Some(&kwarg))
                } else {
                    generator.call1(py, (py_request_stream_obj,))
                }?;
                let locals = TaskLocals::new(event_loop.bind(py).clone());
                pyo3_async_runtimes::tokio::into_stream_with_locals_v1(
                    locals,
                    gen_result.into_bound(py),
                )
            })
        })
        .await
        .map_err(|e| anyhow::anyhow!("failed to offload python call to blocking task: {e}"))??;

        let stream = Box::pin(stream);

        let (resp_tx, resp_rx) = mpsc::channel::<Annotated<serde_json::Value>>(128);
        let response_request_id = request_id.clone();
        let response_ctx = ctx.clone();
        tokio::spawn(async move {
            tracing::debug!(
                request_id = %response_request_id,
                "starting bidirectional python response stream task"
            );
            let mut stream = stream;
            while let Some(item) = stream.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(e) => {
                        tracing::error!(
                            request_id = %response_request_id,
                            error = %e,
                            "python bidirectional generator yielded an error; \
                             ending response stream"
                        );
                        break;
                    }
                };

                let value = match tokio::task::spawn_blocking(move || {
                    Python::with_gil(|py| depythonize::<serde_json::Value>(&item.into_bound(py)))
                })
                .await
                {
                    Ok(Ok(value)) => value,
                    Ok(Err(e)) => {
                        tracing::error!(
                            request_id = %response_request_id,
                            error = %e,
                            "failed to depythonize bidirectional response frame; \
                             stopping generator"
                        );
                        response_ctx.stop_generating();
                        break;
                    }
                    Err(e) => {
                        tracing::error!(
                            request_id = %response_request_id,
                            error = %e,
                            "failed to offload depythonize to blocking task"
                        );
                        break;
                    }
                };

                if resp_tx.send(Annotated::from_data(value)).await.is_err() {
                    tracing::debug!(
                        request_id = %response_request_id,
                        "bidirectional response channel closed; exiting response task"
                    );
                    break;
                }
            }
            tracing::debug!(
                request_id = %response_request_id,
                "finished bidirectional python response stream task"
            );
        });

        Ok(ResponseStream::new(
            Box::pin(ReceiverStream::new(resp_rx)),
            ctx,
        ))
    }
}
