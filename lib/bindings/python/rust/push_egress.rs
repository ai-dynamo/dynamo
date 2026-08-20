// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inverted (push-based) Python -> Rust response egress.
//!
//! # Why
//!
//! The default egress path in [`crate::engine`] is a RUST PULL from a Python
//! async generator (`demand_driven_python_stream`). Per response it costs two
//! independent GIL acquisitions on *tokio* threads:
//!
//! 1. `pybridge.anext_call` — a tokio worker takes the GIL only to call
//!    `__anext__` and hand the actual work to the Python event-loop thread via
//!    `call_soon_threadsafe`; the tokio thread then drops the GIL and parks.
//! 2. `pybridge.decode_response` — a `spawn_blocking` thread takes the GIL
//!    again to `depythonize` the yielded object.
//!
//! Which tokio worker polls the stream is arbitrary, so over a run essentially
//! every worker thread becomes a GIL contender. Measured on the TRT-LLM decode
//! worker: 45 GIL-capable threads and a GIL wait/hold ratio of 23.4, versus 3
//! threads / 0.3 for `trtllm-serve`.
//!
//! # What this module does
//!
//! Inverts the direction. The Python handler — which is *already* holding the
//! GIL while it runs — calls [`ResponseSender::send`] once per response. The
//! conversion to an owned Rust value happens inline on that call, under the
//! caller's existing GIL, and the result is enqueued on a bounded
//! `tokio::sync::mpsc`. The tokio side only ever does `rx.recv().await`: it
//! never acquires the GIL and never touches a Python object.
//!
//! # How the handler gets its sender
//!
//! On the `context` argument, as `context.response_sender`. Not as a dedicated
//! parameter: the Python handlers wrap `generate` in decorators that use
//! `functools.wraps`, which makes `inspect.signature` report the *wrapped*
//! function's parameters, so a `response_sender` parameter added by a wrapper
//! is invisible to the bridge's signature check. `context` is already on every
//! handler's signature and is already the per-request handle, so it is both
//! reliably detectable and the natural carrier.
//!
//! # Feature flag
//!
//! Gated on `DYN_TRTLLM_PUSH_EGRESS=1`, default OFF. With the flag unset the
//! pull path in [`crate::engine`] is used verbatim and nothing here runs, so
//! one image can A/B both paths.

use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Error;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pythonize::depythonize;
use serde::Deserialize;
use tokio_stream::{Stream, StreamExt};

use tokio::sync::mpsc;

use dynamo_runtime::dynamo_nvtx_range;
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::logging::get_distributed_tracing_context;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Data, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;

use crate::context::{Context, callable_accepts_kwarg};
use crate::engine::{self, map_python_exception};
use crate::python_payload::PythonPayload;
#[cfg(not(test))]
use crate::trtllm_egress::OwnedFrameSink;

/// Environment variable that selects the push egress path. `"1"` enables it;
/// anything else (including unset) leaves the pull path in place.
pub(crate) const PUSH_EGRESS_ENV: &str = "DYN_TRTLLM_PUSH_EGRESS";

/// Depth of the Python -> Rust response channel. Matches
/// `engine::RESPONSE_CHANNEL_DEPTH` so the push path buffers exactly as much
/// as the pull path's forwarder does.
pub(crate) const PUSH_CHANNEL_DEPTH: usize = 128;

/// Whether push egress is enabled process-wide. Read once: the flag selects a
/// serving topology at startup and must not change under a running endpoint.
pub(crate) fn push_egress_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(PUSH_EGRESS_ENV)
            .map(|value| value == "1")
            .unwrap_or(false)
    })
}

/// Whether this Python callable can be driven in push mode.
///
/// The sender is delivered on the `context` argument, so a handler that does
/// not take one cannot be reached and must stay on the pull path. Unlike a
/// check for a `response_sender` parameter, this one survives `functools.wraps`
/// decorators: `context` is part of the wrapped function's own signature.
pub(crate) fn handler_supports_push(handler: &PyObject) -> bool {
    // MUST be `response_sender`, not `context`. Every handler accepts `context`,
    // so checking for it makes this test always true and the pull-path fallback
    // unreachable — an undecorated handler would then be driven as a coroutine
    // and fail every request. `push_egress_capable` deletes its own
    // `__wrapped__` (`push_egress.py:259`) precisely so `inspect.signature`
    // reports this parameter through the decorator.
    Python::with_gil(|py| {
        callable_accepts_kwarg(py, handler.bind(py), "response_sender").unwrap_or(false)
    })
}

/// Add this module's classes to the extension module.
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ResponseSender>()?;
    Ok(())
}

// ── GIL-side sink ────────────────────────────────────────────────────────────

/// The GIL-side half of one request's push channel.
///
/// Type-erased because `#[pyclass]` cannot be generic: [`ResponseSender`] must
/// be a single concrete Python type, while the channel's item type is chosen by
/// the (generic) [`response_channel`] factory. Every method here is called with
/// the GIL already held by the Python caller.
pub(crate) trait ResponseSink: Send + Sync {
    /// Convert `obj` to an owned Rust value and enqueue it.
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()>;

    /// Enqueue a response that is already fully owned by Rust.
    fn send_owned(&self, frame: Annotated<serde_json::Value>) -> Result<(), String>;

    /// Normal end of stream.
    fn close(&self);

    /// Terminate the stream with an untyped error frame.
    fn close_with_error(&self, message: String);

    /// Terminate the stream with a typed backend error frame. Not exposed to
    /// Python; used by the Rust-side safety net when the handler coroutine
    /// raises instead of closing the sender itself.
    fn close_with_dynamo_error(&self, error: DynamoError);
}

struct TypedSink<Resp> {
    /// `None` once the stream has been closed. Dropping the last `Sender` is
    /// what ends the receiver stream, so closing is "take the sender".
    tx: Mutex<Option<mpsc::Sender<Annotated<Resp>>>>,
}

impl<Resp> TypedSink<Resp> {
    /// Clone the sender out from under the lock rather than holding the lock
    /// across the enqueue: a blocking send while holding the mutex would let a
    /// concurrent `close()` deadlock against it.
    fn sender(&self) -> Option<mpsc::Sender<Annotated<Resp>>> {
        self.tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn take_sender(&self) -> Option<mpsc::Sender<Annotated<Resp>>> {
        self.tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    /// Enqueue a terminal frame after the sender has already been taken.
    ///
    /// Unlike [`ResponseSink::send`] this runs on either side of the bridge:
    /// from Python (`close_with_error`) or from the Rust driver task's error
    /// path. Those have opposite constraints when the channel is full —
    /// `blocking_send` panics inside an async task, and `spawn` needs a runtime
    /// handle Python's event-loop thread does not have — so pick per caller.
    /// Holding `tx` until the frame lands is what keeps the stream open long
    /// enough to deliver it; the stream ends when the task below drops it.
    fn send_terminal(&self, tx: mpsc::Sender<Annotated<Resp>>, frame: Annotated<Resp>)
    where
        Resp: Data,
    {
        let frame = match tx.try_send(frame) {
            Ok(()) => return,
            Err(mpsc::error::TrySendError::Full(frame)) => frame,
            // Consumer is gone; there is nothing left to report to.
            Err(mpsc::error::TrySendError::Closed(_)) => return,
        };

        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _nvtx_blocked = dynamo_nvtx_range!("pybridge.push_blocked");
                let _ = tx.send(frame).await;
            });
            return;
        }

        // No runtime on this thread: we are on the Python side. Same GIL rule
        // as `send` — release it across the wait.
        let _nvtx_blocked = dynamo_nvtx_range!("pybridge.push_blocked");
        let _ = Python::with_gil(|py| py.allow_threads(|| tx.blocking_send(frame)));
    }
}

impl<Resp> ResponseSink for TypedSink<Resp>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        // Covers the whole Python->Rust crossing for one response: the
        // `depythonize` plus the enqueue. Unlike the pull path there is no
        // GIL *acquisition* inside this range — the Python handler already
        // holds it — so this measures conversion cost only, not GIL wait.
        let _nvtx = dynamo_nvtx_range!("pybridge.push_send");

        let frame = decode_response::<Resp>(obj)?;

        let Some(tx) = self.sender() else {
            return Err(PyRuntimeError::new_err(
                "response stream is closed; send() after close()",
            ));
        };

        // Fast path. `try_send` never blocks, so there is no reason to drop the
        // GIL for it, and dropping/reacquiring it would cost more than the send.
        let frame = match tx.try_send(frame) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(frame)) => frame,
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(PyRuntimeError::new_err(
                    "response stream is closed; the consumer dropped the response stream",
                ));
            }
        };

        // Backpressure path.
        //
        // CRITICAL: the GIL MUST be released across this blocking wait.
        // `allow_threads` here is not an optimization, it is a correctness
        // requirement. This call runs on the asyncio loop thread; blocking it
        // with the GIL held freezes the entire interpreter — the event loop
        // that would run every other request's handler, and every tokio task
        // that needs the GIL. Anything that has to run for the channel to
        // drain then cannot run, and a merely-full channel becomes an
        // interpreter-wide stall instead of local backpressure.
        //
        // `blocking_send` panics if called from inside an async task. Both
        // callers are safe: the Python event-loop thread (no runtime context at
        // all) and the driver task's degradation path (a `spawn_blocking`
        // thread, which is a permitted blocking region). It would panic only if
        // a handler managed to call `send` from an async tokio task while
        // holding the GIL.
        let _nvtx_blocked = dynamo_nvtx_range!("pybridge.push_blocked");
        py.allow_threads(|| tx.blocking_send(frame)).map_err(|_| {
            PyRuntimeError::new_err(
                "response stream is closed; the consumer dropped the response stream",
            )
        })
    }

    fn send_owned(&self, frame: Annotated<serde_json::Value>) -> Result<(), String> {
        let frame = Annotated {
            data: frame
                .data
                .map(serde_json::from_value::<Resp>)
                .transpose()
                .map_err(|error| format!("failed to convert owned response frame: {error}"))?,
            id: frame.id,
            event: frame.event,
            comment: frame.comment,
            error: frame.error,
        };

        let Some(tx) = self.sender() else {
            return Err("response stream is closed; send after close".to_string());
        };
        let frame = match tx.try_send(frame) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(frame)) => frame,
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err("response stream consumer has closed".to_string());
            }
        };

        let _nvtx_blocked = dynamo_nvtx_range!("rust_egress.send_blocked");
        tx.blocking_send(frame)
            .map_err(|_| "response stream consumer has closed".to_string())
    }

    fn close(&self) {
        // Dropping the last sender is what ends the receiver stream. Idempotent
        // so the Rust-side safety net can close a stream the handler already
        // closed itself.
        drop(self.take_sender());
    }

    fn close_with_error(&self, message: String) {
        let Some(tx) = self.take_sender() else {
            return;
        };
        self.send_terminal(tx, Annotated::from_error(message));
    }

    fn close_with_dynamo_error(&self, error: DynamoError) {
        let Some(tx) = self.take_sender() else {
            return;
        };
        self.send_terminal(tx, Annotated::from_err(error));
    }
}

/// Python -> Rust conversion for one response object.
///
/// Mirrors `engine::process_item` exactly, so the push path yields the same
/// Rust value the pull path would have produced for the same Python object:
/// yields tagged with `_dynamo_annotated: True` are wire `Annotated<R>`
/// envelopes, everything else is plain data.
fn decode_response<Resp>(obj: &Bound<'_, PyAny>) -> PyResult<Annotated<Resp>>
where
    Resp: for<'de> Deserialize<'de>,
{
    let py = obj.py();
    let is_envelope = obj
        .downcast::<PyDict>()
        .ok()
        .and_then(|dict| {
            dict.get_item(pyo3::intern!(py, "_dynamo_annotated"))
                .ok()
                .flatten()
        })
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false);

    let decoded = if is_envelope {
        depythonize::<Annotated<Resp>>(obj)
    } else {
        depythonize::<Resp>(obj).map(Annotated::from_data)
    };

    decoded.map_err(|error| {
        PyValueError::new_err(format!(
            "critical error: invalid response object from python handler; \
             application-logic-mismatch: {error}"
        ))
    })
}

// ── Python-facing handle ─────────────────────────────────────────────────────

/// Rust response sink handed to a Python handler running in push mode.
///
/// Delivered as the `response_sender=` keyword argument, and also reachable as
/// `context.response_sender`; `None`/absent on the pull path, so its presence is
/// also how a handler decides which path it is on.
///
/// ```python
/// async def generate(self, request, context, response_sender=None):
///     sender = response_sender or getattr(context, "response_sender", None)
///     if sender is None:              # pull path: unchanged
///         async for chunk in engine.generate(request):
///             yield chunk
///         return
///     try:                            # push path: yields nothing
///         async for chunk in engine.generate(request):
///             sender.send(chunk)
///     except Exception as exc:
///         sender.close_with_error(f"{type(exc).__name__}: {exc}")
///     else:
///         sender.close()
/// ```
///
/// `send` blocks when the consumer is behind; it is safe to call from the
/// asyncio loop thread because the GIL is released across that wait.
#[pyclass]
pub struct ResponseSender {
    sink: Arc<dyn ResponseSink>,
}

#[pymethods]
impl ResponseSender {
    /// Convert one response object and enqueue it for the Rust egress path.
    ///
    /// Raises `RuntimeError` if the stream is already closed and `ValueError`
    /// if the object cannot be converted.
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        self.sink.send(py, obj)
    }

    /// Normal end of stream. Idempotent.
    fn close(&self) -> PyResult<()> {
        self.sink.close();
        Ok(())
    }

    /// Terminate the stream with an error frame. Idempotent; a later `close()`
    /// is a no-op.
    fn close_with_error(&self, msg: String) -> PyResult<()> {
        self.sink.close_with_error(msg);
        Ok(())
    }
}

impl ResponseSender {
    /// Rust-side handle to the same sink, for the safety net that terminates
    /// the stream if the handler coroutine raises.
    pub(crate) fn sink(&self) -> Arc<dyn ResponseSink> {
        self.sink.clone()
    }

    #[cfg(not(test))]
    pub(crate) fn owned_sink(&self) -> Arc<dyn OwnedFrameSink> {
        Arc::new(OwnedSinkAdapter {
            sink: self.sink.clone(),
        })
    }
}

#[cfg(not(test))]
struct OwnedSinkAdapter {
    sink: Arc<dyn ResponseSink>,
}

#[cfg(not(test))]
impl OwnedFrameSink for OwnedSinkAdapter {
    fn send(&self, frame: Annotated<serde_json::Value>) -> Result<(), String> {
        self.sink.send_owned(frame)
    }

    fn close(&self) {
        self.sink.close();
    }

    fn close_with_error(&self, message: String) {
        self.sink.close_with_error(message);
    }
}

// ── Factory ──────────────────────────────────────────────────────────────────

/// Build one request's push channel.
///
/// The returned stream yields `Annotated<Resp>` — the same item type
/// `engine::buffered_typed_response_stream` produces on the pull path — so the
/// ingress side (`lib/runtime/src/pipeline/network/ingress/push_handler.rs`)
/// is unchanged.
pub(crate) fn response_channel<Resp>(
    depth: usize,
) -> (
    ResponseSender,
    impl Stream<Item = Annotated<Resp>> + Send + 'static,
)
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    let (tx, rx) = mpsc::channel::<Annotated<Resp>>(depth);

    let sender = ResponseSender {
        sink: Arc::new(TypedSink {
            tx: Mutex::new(Some(tx)),
        }),
    };

    let stream = futures::stream::unfold(rx, |mut rx| async move {
        // Unlike `pybridge.anext_call` in engine.rs, this range deliberately
        // DOES span the `.await`: there is no Python work inside it to isolate,
        // and the whole point of the push path is that the consumer side is
        // pure Rust. Read it accordingly — the range length is dominated by the
        // idle wait for the engine's next token, and its *end* marks the
        // response arriving. The bridge cost itself is `pybridge.push_send` on
        // the Python thread; a full channel shows up as `pybridge.push_blocked`.
        let _nvtx = dynamo_nvtx_range!("pybridge.push_recv");
        rx.recv().await.map(|item| (item, rx))
    });

    (sender, stream)
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// Response type carried by the push path. The GIL-side `depythonize` produces
/// an owned Rust value, so — unlike the pull path's `PythonPayload` — nothing
/// downstream of the channel can touch a Python object.
pub(crate) type PushResponse = Annotated<serde_json::Value>;

/// Push-mode counterpart of `engine::PythonNetworkEngine`.
///
/// The handler is still called exactly as on the pull path — it is still an
/// async-generator function, and the returned generator is still driven by
/// `engine::invoke_generator` / `demand_driven_python_stream`. What changes is
/// how often: in push mode the generator yields nothing and is advanced ONCE
/// per request (one `__anext__`, which runs the whole request and then raises
/// `StopAsyncIteration`), instead of once per response. Responses arrive out of
/// band on the [`ResponseSender`] the handler finds on its `context`.
///
/// Keeping the generator shape is deliberate: registration
/// (`endpoint.serve_endpoint(handler.generate, ...)`), cancellation, and the
/// handler's own control flow are untouched, and a handler that ignores the
/// sender and yields normally still works (see the driver task below).
pub(crate) struct PythonPushEngine {
    handler: Arc<PyObject>,
    event_loop: Arc<PyObject>,
}

impl PythonPushEngine {
    pub(crate) fn new(handler: PyObject, event_loop: PyObject) -> Self {
        Self {
            handler: Arc::new(handler),
            event_loop: Arc::new(event_loop),
        }
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<PythonPayload>, ManyOut<PushResponse>, Error> for PythonPushEngine {
    async fn generate(
        &self,
        request: SingleIn<PythonPayload>,
    ) -> Result<ManyOut<PushResponse>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let request_id = context.id().to_string();
        let metadata = context.metadata().clone();
        let current_trace_context = get_distributed_tracing_context();

        let (sender, stream) = response_channel::<serde_json::Value>(PUSH_CHANNEL_DEPTH);
        let sink = sender.sink();

        let python_input = request.into_inner();
        let handler_ctx = ctx.clone();

        // The sender rides on the `Context`. `serve_endpoint` has already
        // verified the handler accepts a `context` argument before selecting
        // this engine, so the closure below always runs and the sender is
        // always delivered — if it were dropped here instead, the channel would
        // close and the request would return an empty stream.
        let driver = engine::invoke_generator(
            self.handler.clone(),
            self.event_loop.clone(),
            move |_py| Ok(python_input),
            Some(move |py: Python<'_>| {
                // ONE sender object, delivered TWO ways, because the Python half
                // reads it from a `response_sender=` keyword argument
                // (`push_egress.py:235`) while `context.response_sender` is the
                // documented fallback. Delivering only one of them is the seam
                // that silently drops the worker back to the pull path — or, if
                // Rust has already committed to awaiting a coroutine, fails the
                // request outright. `Py<T>` is refcounted, so the clone is the
                // same Python object, and a handler comparing them sees identity.
                let py_sender = Py::new(py, sender)?;
                let context = Py::new(
                    py,
                    Context::new(handler_ctx, current_trace_context, None, metadata)
                        .with_response_sender(py_sender.clone_ref(py)),
                )?;
                Ok(vec![
                    ("context", context.into_any()),
                    ("response_sender", py_sender.into_any()),
                ])
            }),
        )
        .await?;

        // Driver task. In push mode this exists only to run the handler to
        // completion and observe how it ended; the responses themselves never
        // pass through here.
        tokio::spawn(async move {
            let mut driver = driver;
            let mut yielded = 0usize;

            while let Some(item) = driver.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(error) => {
                        sink.close_with_dynamo_error(map_python_exception(error));
                        return;
                    }
                };

                // Graceful degradation. A push-mode handler yields nothing, so
                // this arm is normally dead code. It is reached when the handler
                // ignored the sender and yielded instead (e.g. a handler that
                // predates the push path, or one whose own flag check disagreed
                // with ours): forward the frame through the same sink so the
                // request still completes correctly, just without the benefit.
                yielded += 1;
                let forward_sink = sink.clone();
                let forwarded = tokio::task::spawn_blocking(move || {
                    // Unlike `pybridge.push_send`, this range DOES include a GIL
                    // acquisition on a tokio thread — it is the pull path's cost,
                    // reappearing precisely because the handler did not push.
                    let _nvtx = dynamo_nvtx_range!("pybridge.push_forward_yield");
                    Python::with_gil(|py| forward_sink.send(py, item.bind(py)))
                })
                .await;

                match forwarded {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        sink.close_with_dynamo_error(map_python_exception(error));
                        return;
                    }
                    Err(error) => {
                        sink.close_with_error(format!(
                            "critical error: failed to offload the python response forward \
                             to a new thread: {error}"
                        ));
                        return;
                    }
                }
            }

            if yielded > 0 {
                tracing::debug!(
                    request_id,
                    yielded,
                    "push egress: handler yielded responses instead of pushing them; \
                     forwarded them over the pull path"
                );
            }

            // Idempotent: a well-behaved handler has already closed the sender.
            sink.close();
        });

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}
