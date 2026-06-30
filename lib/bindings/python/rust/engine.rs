// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Error, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::{PyAny, PyErr};
use pyo3_async_runtimes::TaskLocals;
use pythonize::{depythonize, pythonize};
pub use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tokio_util::sync::CancellationToken;

use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
use dynamo_runtime::logging::get_distributed_tracing_context;
pub use dynamo_runtime::{
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Data, ManyOut, ResponseStream,
        SingleIn,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::PyAsyncRequestStream;
use dynamo_runtime::pipeline::ManyIn;

use super::context::{Context, callable_accepts_kwarg};
use super::errors::{extract_http_like_error, py_exception_to_backend_error};
use crate::python_payload::{PythonPayload, PythonResponseItem};

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
/// keyword argument for backwards compatibility with the legacy positional-only call.
fn detect_has_context(generator: &PyObject) -> bool {
    Python::with_gil(|py| {
        let callable = generator.bind(py);
        callable_accepts_kwarg(py, callable, "context").unwrap_or(false)
    })
}

/// Boxed Rust stream of items yielded by a Python async generator. Each
/// item is either a `PyObject` frame or the `PyErr` the generator raised.
type PyItemStream = Pin<Box<dyn Stream<Item = PyResult<Py<PyAny>>> + Send>>;

/// Invoke the Python `generate` callable and convert the async generator it
/// returns into a Rust [`Stream`] of `PyObject` items.
///
/// `to_python_input` converts this engine's `generate` input into the Python
/// object passed as the generator's positional argument.
/// `to_python_context` when `Some`, builds the `context=` keyword argument; `None` selects the
/// legacy positional-only call.
/// Note: functions instead of Python objects are passed because the construction
/// of the Python objects needs to run inside the GIL.
///
/// The GIL is acquired on a blocking task rather than inline: under contention
/// it can block for an unbounded time, which would park the tokio reactor.
async fn invoke_generator<F, G>(
    generator: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    to_python_input: F,
    to_python_context: Option<G>,
) -> Result<PyItemStream>
where
    F: FnOnce(Python) -> PyResult<Py<PyAny>> + Send + 'static,
    G: FnOnce(Python) -> PyResult<Py<PyAny>> + Send + 'static,
{
    let stream = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let python_input = to_python_input(py)?;

            let gen_result = match to_python_context {
                Some(to_python_context) => {
                    let py_ctx = to_python_context(py)?;
                    let kwarg = PyDict::new(py);
                    kwarg.set_item("context", py_ctx)?;
                    generator.call(py, (python_input,), Some(&kwarg))
                }
                // Legacy: no `context` arg.
                None => generator.call1(py, (python_input,)),
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

    Ok(Box::pin(stream))
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

impl PythonAsyncEngine {
    pub(crate) fn network_engine(&self) -> PythonNetworkEngine {
        PythonNetworkEngine(self.0.clone())
    }
}

/// Network-only Python engine. Requests and responses stay as Python-owned
/// objects so the ingress adapter can transcode them directly to and from the
/// configured request-plane codec.
#[derive(Clone)]
pub(crate) struct PythonNetworkEngine(PythonServerStreamingEngine);

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<PythonPayload>, ManyOut<PythonResponseItem>, Error>
    for PythonNetworkEngine
{
    async fn generate(
        &self,
        request: SingleIn<PythonPayload>,
    ) -> Result<ManyOut<PythonResponseItem>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let id = context.id().to_string();
        let current_trace_context = get_distributed_tracing_context();
        let metadata = context.metadata().clone();

        let stream = invoke_generator(
            self.0.generator.clone(),
            self.0.event_loop.clone(),
            move |_py| Ok(request.into_inner()),
            self.0.has_context.then_some({
                let ctx = ctx.clone();
                move |py: Python<'_>| {
                    Py::new(py, Context::new(ctx, current_trace_context, None, metadata))
                        .map(|c| c.into_any())
                }
            }),
        )
        .await?;

        let rx = spawn_raw_response_forwarder(stream, ctx, id);
        Ok(ResponseStream::new(
            Box::pin(ReceiverStream::new(rx)),
            context.context(),
        ))
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
        let metadata = context.metadata().clone();

        let stream = invoke_generator(
            self.generator.clone(),
            self.event_loop.clone(),
            move |py| Ok(pythonize(py, &request)?.unbind()),
            self.has_context.then_some({
                let ctx = ctx.clone();
                move |py: Python<'_>| {
                    Py::new(py, Context::new(ctx, current_trace_context, None, metadata))
                        .map(|c| c.into_any())
                }
            }),
        )
        .await?;

        // Drain the Python response stream on a dedicated task, mapping any
        // generator error to a typed annotated error frame.
        let rx = spawn_response_forwarder::<Resp>(stream, ctx, id);

        Ok(ResponseStream::new(
            Box::pin(ReceiverStream::new(rx)),
            context.context(),
        ))
    }
}

async fn process_item<Resp>(
    item: Result<Py<PyAny>, PyErr>,
) -> Result<Annotated<Resp>, ResponseProcessingError>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let item = item.map_err(|e| ResponseProcessingError::Dynamo(map_python_exception(e)))?;
    let response = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| {
            let bound = item.into_bound(py);
            // Yields tagged with `_dynamo_annotated: True` are wire
            // Annotated<R> envelopes; everything else is plain data.
            let is_envelope = bound
                .downcast::<PyDict>()
                .ok()
                .and_then(|d| d.get_item("_dynamo_annotated").ok().flatten())
                .and_then(|v| v.is_truthy().ok())
                .unwrap_or(false);
            if is_envelope {
                depythonize::<Annotated<Resp>>(&bound)
            } else {
                depythonize::<Resp>(&bound).map(Annotated::from_data)
            }
        })
    })
    .await
    .map_err(|e| ResponseProcessingError::Offload(e.to_string()))?
    .map_err(|e| ResponseProcessingError::Deserialize(e.to_string()))?;

    Ok(response)
}

pub(crate) fn map_python_exception(error: PyErr) -> DynamoError {
    Python::with_gil(|py| {
        error.display(py);

        if let Some((backend_err, message)) = py_exception_to_backend_error(py, &error) {
            return DynamoError::builder()
                .error_type(ErrorType::Backend(backend_err))
                .message(message)
                .build();
        }

        if let Some((code, message)) = extract_http_like_error(py, &error) {
            let backend_err = if (400..500).contains(&code) {
                BackendError::InvalidArgument
            } else {
                BackendError::Unknown
            };
            let json_msg = serde_json::json!({
                "message": message,
                "code": code,
            })
            .to_string();
            return DynamoError::builder()
                .error_type(ErrorType::Backend(backend_err))
                .message(json_msg)
                .build();
        }

        if error.is_instance_of::<pyo3::exceptions::PyGeneratorExit>(py) {
            return DynamoError::builder()
                .error_type(ErrorType::Backend(BackendError::EngineShutdown))
                .message("engine shutting down")
                .build();
        }

        let backend_err = if error.is_instance_of::<pyo3::exceptions::PyValueError>(py)
            || error.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
        {
            BackendError::InvalidArgument
        } else if error.is_instance_of::<pyo3::exceptions::PyTimeoutError>(py) {
            BackendError::ConnectionTimeout
        } else if error.is_instance_of::<pyo3::exceptions::PyConnectionRefusedError>(py) {
            BackendError::CannotConnect
        } else if error.is_instance_of::<pyo3::exceptions::PyConnectionResetError>(py)
            || error.is_instance_of::<pyo3::exceptions::PyBrokenPipeError>(py)
            || error.is_instance_of::<pyo3::exceptions::PyConnectionError>(py)
        {
            BackendError::Disconnected
        } else if error.is_instance_of::<pyo3::exceptions::asyncio::CancelledError>(py) {
            BackendError::Cancelled
        } else {
            BackendError::Unknown
        };

        DynamoError::builder()
            .error_type(ErrorType::Backend(backend_err))
            .message(error.to_string())
            .build()
    })
}

/// Channel depth between the response-forwarding task and the consumer of
/// the engine's output stream.
const RESPONSE_CHANNEL_DEPTH: usize = 128;

/// Drain the Python response stream on a spawned task, deserialize each item
/// into `Resp` via [`process_item`], and forward it as an [`Annotated`] frame
/// over an mpsc channel. Returns the receiver, to be wrapped in a
/// [`ResponseStream`].
///
/// Errors raised by the Python generator are mapped to typed backend errors
/// and emitted as annotated error frames, so a failing generator returns as
/// an error to the client rather than a silently truncated stream.
/// On a deserialize mismatch the request context is told to stop generating.
fn spawn_response_forwarder<Resp>(
    stream: PyItemStream,
    ctx: Arc<dyn AsyncEngineContext>,
    request_id: String,
) -> mpsc::Receiver<Annotated<Resp>>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let (tx, rx) = mpsc::channel::<Annotated<Resp>>(RESPONSE_CHANNEL_DEPTH);

    // any error thrown in the stream will be caught and complete the
    // processing task; the error is emitted as an annotated error frame
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
                // Consumer dropped the response stream — tell the generator to
                // stop rather than relying solely on upstream push_handler.
                ctx.stop_generating();
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

    rx
}

fn spawn_raw_response_forwarder(
    stream: PyItemStream,
    ctx: Arc<dyn AsyncEngineContext>,
    request_id: String,
) -> mpsc::Receiver<PythonResponseItem> {
    let (tx, rx) = mpsc::channel(RESPONSE_CHANNEL_DEPTH);
    tokio::spawn(async move {
        let mut stream = stream;
        while let Some(item) = stream.next().await {
            let stop_after_item = item.is_err();
            if tx.send(PythonResponseItem::new(item)).await.is_err() {
                ctx.stop_generating();
                tracing::trace!(
                    request_id,
                    "direct Python response consumer dropped; stopping generator"
                );
                break;
            }
            if stop_after_item {
                break;
            }
        }
    });
    rx
}

/// Channel depth between the inbound forwarder and the Python iterator.
/// Mirrors the depth used by the wire-side bidirectional ingress forwarder
/// in `lib/runtime/src/pipeline/network/ingress/push_handler.rs`.
const BIDIRECTIONAL_INPUT_CHANNEL_DEPTH: usize = 8;

/// Rust-side adapter that bridges a Python `async def generate(request_stream, context)`
/// callable into an [`AsyncEngine`] of the ManyIn / ManyOut
/// shape (`PythonPayload` request frames and raw Python response items).
///
/// The adapter:
///
/// 1. Transforms the inbound `RequestStream<PythonPayload>` into a
///    `PyAsyncRequestStream` and `context`. Similar to unary engine,
///    cancellation observation is the Python engine's responsibility via the
///    `context` argument. Input stream can end early if no more inputs are expected.
/// 2. Invokes the Python generator with `(request_stream, context)`, then
///    wraps the returned async generator into a Rust `Stream<Item = PyResult<PyObject>>`.
/// 3. Forwards each raw Python response to the ingress payload adapter, which
///    performs annotation handling and wire serialization in one blocking step.
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
impl AsyncEngine<ManyIn<PythonPayload>, ManyOut<PythonResponseItem>, Error>
    for PythonBidirectionalEngine
{
    async fn generate(
        &self,
        input: ManyIn<PythonPayload>,
    ) -> Result<ManyOut<PythonResponseItem>, Error> {
        let (request_stream, ctx_unit) = input.into_parts();
        let ctx = ctx_unit.context();
        let request_id = ctx_unit.id().to_string();
        let metadata = ctx_unit.metadata().clone();
        let mut inbound = request_stream
            .take()
            .ok_or_else(|| anyhow::anyhow!("RequestStream::take returned None"))?;

        // Capture trace context once, while we still hold the
        // dispatching task; needed when constructing the Python `Context`.
        let current_trace_context = get_distributed_tracing_context();

        // Forwarder: move Python-owned frames directly into the Python
        // iterator. The `frame_tx.closed()` arm cancels the forwarder as soon
        // as the Python iterator drops the receiver, so it shuts down promptly
        // instead of blocking on the next inbound frame.
        let (frame_tx, frame_rx) = mpsc::channel::<PyObject>(BIDIRECTIONAL_INPUT_CHANNEL_DEPTH);
        let forwarder_request_id = request_id.clone();
        tokio::spawn(async move {
            loop {
                let value = tokio::select! {
                    _ = frame_tx.closed() => break,
                    value = inbound.next() => value,
                };
                let Some(value) = value else {
                    break;
                };
                if frame_tx.send(value.into_inner()).await.is_err() {
                    tracing::debug!(
                        request_id = %forwarder_request_id,
                        "python engine dropped request stream; input forwarder exiting"
                    );
                    break;
                }
            }
        });

        let py_request_stream = PyAsyncRequestStream::new(frame_rx);

        // The positional argument is the `PyAsyncRequestStream` handle, wrapped
        // inside the GIL.
        let stream = invoke_generator(
            self.generator.clone(),
            self.event_loop.clone(),
            move |py| Ok(Py::new(py, py_request_stream)?.into_any()),
            self.has_context.then_some({
                let ctx = ctx.clone();
                move |py: Python<'_>| {
                    Py::new(py, Context::new(ctx, current_trace_context, None, metadata))
                        .map(|c| c.into_any())
                }
            }),
        )
        .await?;

        let rx = spawn_raw_response_forwarder(stream, ctx.clone(), request_id);

        Ok(ResponseStream::new(Box::pin(ReceiverStream::new(rx)), ctx))
    }
}
