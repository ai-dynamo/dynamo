// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inverted (push-based) Python -> Rust response egress.
//!
//! The pull path in [`crate::engine`] costs two GIL acquisitions on arbitrary
//! tokio threads per response. Here the Python handler — already holding the
//! GIL — calls [`ResponseSender::send`] instead: the conversion happens inline
//! under that existing GIL and the value is enqueued on a bounded
//! `tokio::sync::mpsc`, so the tokio side only ever does `rx.recv().await` and
//! never touches Python. Rationale and measurements: DYN-3703.
//!
//! Selected per handler by signature — [`handler_supports_push`] picks this
//! path for a handler declaring a `response_sender` parameter, which in
//! practice means the TRT-LLM `@push_egress_capable` decorator
//! (`components/src/dynamo/trtllm/request_handlers/push_egress.py`). There is
//! no environment variable: that decorator is the switch, so the two halves
//! cannot disagree about which path an endpoint is on. Every other Python
//! handler keeps the pull path verbatim.
//!
//! The sender reaches the handler as the `response_sender` keyword argument,
//! and only that way — the same parameter the signature check keys on.

use std::sync::{Arc, Mutex};

use anyhow::Error;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Deserialize;
use tokio_stream::{Stream, StreamExt};

use tokio::sync::mpsc;

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

/// Depth of the Python -> Rust response channel. Matches
/// `engine::RESPONSE_CHANNEL_DEPTH` so the push path buffers exactly as much
/// as the pull path's forwarder does.
pub(crate) const PUSH_CHANNEL_DEPTH: usize = 128;

/// Whether this Python callable can be driven in push mode. This is the ONLY
/// switch between the two egress paths.
///
/// The sender is delivered as a `response_sender` keyword argument, so a
/// handler that does not declare one cannot be reached and must stay on the
/// pull path.
pub(crate) fn handler_supports_push(handler: &PyObject) -> bool {
    // MUST be `response_sender`, not `context`. Every handler accepts `context`,
    // so checking for it makes this test always true and the pull-path fallback
    // unreachable — every non-TRT-LLM Python handler in the repo would then be
    // driven in push mode and never terminate its stream.
    // `push_egress_capable` deletes its own `__wrapped__` precisely so
    // `inspect.signature` reports this parameter through the decorator.
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
trait ResponseSink: Send + Sync {
    /// Convert `obj` to an owned Rust value and enqueue it.
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()>;

    /// Normal end of stream.
    fn close(&self);

    /// Terminate the stream with an untyped error frame.
    fn close_with_error(&self, message: String);

    /// Terminate the stream with a typed backend error frame. Not exposed to
    /// Python; used by the Rust-side safety net when the handler's generator
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
                let _ = tx.send(frame).await;
            });
            return;
        }

        // No runtime on this thread: we are on the Python side. Same GIL rule
        // as `send` — release it across the wait.
        let _ = Python::with_gil(|py| py.allow_threads(|| tx.blocking_send(frame)));
    }
}

impl<Resp> ResponseSink for TypedSink<Resp>
where
    Resp: Data + for<'de> Deserialize<'de>,
{
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        // The whole Python->Rust crossing for one response: the `depythonize`
        // plus the enqueue. Unlike the pull path neither step acquires the GIL
        // — the Python handler already holds it.
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
        // `blocking_send` panics if called from inside an async task. The only
        // caller is `ResponseSender::send`, reached from the Python event-loop
        // thread, which has no tokio runtime context at all. It would panic only
        // if a handler managed to call `send` from an async tokio task while
        // holding the GIL.
        py.allow_threads(|| tx.blocking_send(frame)).map_err(|_| {
            PyRuntimeError::new_err(
                "response stream is closed; the consumer dropped the response stream",
            )
        })
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
/// Shares [`engine::depythonize_annotated`] with the pull path, so both paths
/// produce the same Rust value for the same Python object by construction.
/// Only the error mapping differs: here it is raised back into the calling
/// handler, which turns it into a `close_with_error` frame.
fn decode_response<Resp>(obj: &Bound<'_, PyAny>) -> PyResult<Annotated<Resp>>
where
    Resp: for<'de> Deserialize<'de>,
{
    engine::depythonize_annotated(obj).map_err(|error| {
        PyValueError::new_err(format!(
            "critical error: invalid response object from python handler; \
             application-logic-mismatch: {error}"
        ))
    })
}

// ── Python-facing handle ─────────────────────────────────────────────────────

/// Rust response sink handed to a Python handler running in push mode.
///
/// Delivered as the `response_sender=` keyword argument; absent on the pull
/// path, so its presence is also how a handler knows which path it is on.
///
/// ```python
/// async def generate(self, request, context, response_sender=None):
///     if response_sender is None:     # pull path: unchanged
///         async for chunk in engine.generate(request):
///             yield chunk
///         return
///     try:                            # push path: yields nothing
///         async for chunk in engine.generate(request):
///             response_sender.send(chunk)
///     except Exception as exc:
///         response_sender.close_with_error(f"{type(exc).__name__}: {exc}")
///     else:
///         response_sender.close()
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
    /// the stream if the handler's generator raises.
    fn sink(&self) -> Arc<dyn ResponseSink> {
        self.sink.clone()
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

    // The consumer side is pure Rust: no Python work, no GIL, just the wait for
    // whatever the handler pushes next.
    let stream = futures::stream::unfold(rx, |mut rx| async move {
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
/// band on the [`ResponseSender`] passed as the handler's `response_sender`
/// argument.
///
/// Keeping the generator shape is deliberate: registration
/// (`endpoint.serve_endpoint(handler.generate, ...)`), cancellation, and the
/// handler's own control flow are untouched. A handler that yields anyway has
/// broken the contract and its request is failed — see the driver task below.
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

        // Both kwargs are built under one GIL acquisition. `serve_endpoint` has
        // already verified the handler declares a `response_sender` parameter
        // before selecting this engine, so this closure always runs and the
        // sender is always delivered — were it dropped here instead, the channel
        // would close and the request would return an empty stream.
        let driver = engine::invoke_generator(
            self.handler.clone(),
            self.event_loop.clone(),
            move |_py| Ok(python_input),
            Some(move |py: Python<'_>| {
                let context = Py::new(
                    py,
                    Context::new(handler_ctx, current_trace_context, None, metadata),
                )?;
                Ok(vec![
                    ("context", context.into_any()),
                    ("response_sender", Py::new(py, sender)?.into_any()),
                ])
            }),
        )
        .await?;

        // Driver task. In push mode this exists only to run the handler to
        // completion and observe how it ended; the responses themselves never
        // pass through here.
        tokio::spawn(async move {
            let mut driver = driver;

            // Advanced exactly once: a push-mode handler yields nothing, so
            // that single `__anext__` runs the whole request and then ends the
            // generator. Anything else is a broken contract.
            match driver.next().await {
                // Idempotent: a well-behaved handler has already closed.
                None => sink.close(),
                // Forwarding a yielded frame instead would silently reintroduce
                // the per-response GIL acquisition this module exists to
                // remove, so fail the request loudly.
                Some(Ok(_)) => {
                    tracing::error!(
                        request_id,
                        "push egress: handler yielded a response instead of pushing it"
                    );
                    sink.close_with_error(
                        "critical error: push-mode handler yielded a response instead of \
                         pushing it to its response_sender"
                            .to_string(),
                    );
                }
                Some(Err(error)) => sink.close_with_dynamo_error(map_python_exception(error)),
            }
        });

        Ok(ResponseStream::new(Box::pin(stream), context.context()))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Unit tests for the pure-Rust parts of push_egress.rs.
    //!
    //! # Python linkage constraint
    //!
    //! This crate builds with `pyo3/extension-module`.  The linker does NOT
    //! include libpython in the test binary; Python symbols are supplied at
    //! runtime by the Python interpreter loading the `.so`.  Any test that
    //! compiles code which references Python C-API symbols — even unreachable
    //! branches — fails with "undefined symbol: Py_InitializeEx" etc.
    //!
    //! Confirmed: a trivial `prepare_freethreaded_python()` test produced
    //! "undefined symbol: Py_InitializeEx" from rust-lld.
    //!
    //! **Root cause for push_egress.rs**: `TypedSink<Resp>` implements
    //! `ResponseSink`, whose vtable includes `fn send(&self, py: Python<'_>,
    //! obj: &Bound<'_, PyAny>)`.  Instantiating `TypedSink` (even in a test
    //! that never calls `send`) forces the compiler to monomorphize ALL trait
    //! impl methods, pulling in pyo3 and Python C-API symbols.  The same
    //! applies to `ResponseSender` (`#[pyclass]`) and `response_channel`
    //! (which creates a `TypedSink` internally).
    //!
    //! **What compiles**: tests that use only `mpsc` channels, `Annotated`,
    //! `serde_json::Value`, and integer constants — nothing from this module.
    //! Everything else requires Python and must be covered by pytest against
    //! the built `.so`.  Coverage gaps are documented at the bottom.

    use super::PUSH_CHANNEL_DEPTH;
    use dynamo_runtime::protocols::annotated::Annotated;
    use tokio::sync::mpsc;

    // ── PUSH_CHANNEL_DEPTH constant ───────────────────────────────────────

    /// Push and pull channel depths must match.  `engine::RESPONSE_CHANNEL_DEPTH`
    /// is 128; a mismatch makes the push path buffer differently from the pull
    /// path, silently changing backpressure behavior.
    #[test]
    fn push_channel_depth_equals_pull_channel_depth() {
        assert_eq!(
            PUSH_CHANNEL_DEPTH, 128,
            "PUSH_CHANNEL_DEPTH must equal engine::RESPONSE_CHANNEL_DEPTH (128)"
        );
    }

    // ── channel backpressure contract ─────────────────────────────────────
    //
    // TypedSink::send uses a channel of depth PUSH_CHANNEL_DEPTH.  The fast
    // path uses try_send (no blocking); the slow path parks.  These tests
    // verify the channel behaves as the comments describe, using mpsc directly
    // because TypedSink instantiation requires Python symbols.

    /// A channel of PUSH_CHANNEL_DEPTH capacity must accept exactly that many
    /// items without blocking, then report Full on the next one.
    /// Catches drift between the constant and the actual channel constructor call.
    #[test]
    fn channel_accepts_exactly_push_channel_depth_items_then_full() {
        let (tx, _rx) = mpsc::channel::<Annotated<serde_json::Value>>(PUSH_CHANNEL_DEPTH);

        let mut sent = 0usize;
        for i in 0..PUSH_CHANNEL_DEPTH {
            match tx.try_send(Annotated::from_data(serde_json::Value::from(i as i64))) {
                Ok(()) => sent += 1,
                Err(mpsc::error::TrySendError::Full(_)) => break,
                Err(e) => panic!("unexpected channel error: {e}"),
            }
        }
        assert_eq!(
            sent, PUSH_CHANNEL_DEPTH,
            "channel must buffer exactly {PUSH_CHANNEL_DEPTH} items without blocking"
        );

        assert!(
            matches!(
                tx.try_send(Annotated::from_data(serde_json::Value::Null)),
                Err(mpsc::error::TrySendError::Full(_))
            ),
            "item {PUSH_CHANNEL_DEPTH}+1 must fail Full, not Closed"
        );
    }

    // ── error frame contract ──────────────────────────────────────────────
    //
    // TypedSink::close_with_error and close_with_dynamo_error each:
    //   1. Call take_sender() to atomically claim the send side (idempotence)
    //   2. Pass an Annotated error frame to send_terminal
    //   3. send_terminal delivers the frame and then drops the sender
    //
    // Steps 1 and 3 (idempotence via take_sender; stream end via drop) are
    // tested against mpsc directly.  Step 2 (frame structure) is pinned by
    // the from_error shape test.  The full end-to-end path needs Python; see
    // the note at the end of this module.

    #[test]
    fn annotated_from_error_sets_error_field_and_clears_data() {
        // Pins the Annotated frame shape that close_with_error delivers.
        let frame = Annotated::<serde_json::Value>::from_error("some-error");
        assert!(frame.error.is_some(), "from_error must set the error field");
        assert!(frame.data.is_none(), "from_error must not set a data field");
    }

    /// Pins the channel-level protocol: one error frame then end-of-stream.
    /// This is what TypedSink::close_with_error (via send_terminal) is supposed
    /// to produce.  If send_terminal fails to drop the sender after delivering
    /// the error frame, the consumer would wait forever.
    #[tokio::test]
    async fn one_error_frame_then_dropped_sender_ends_stream() {
        let (tx, mut rx) = mpsc::channel::<Annotated<serde_json::Value>>(4);
        tx.send(Annotated::from_error("fatal".to_string()))
            .await
            .unwrap();
        drop(tx); // simulates send_terminal dropping the sender after the frame
        let item = rx.recv().await.expect("one error frame must arrive");
        assert!(item.error.is_some(), "expected an error frame");
        assert!(
            rx.recv().await.is_none(),
            "stream must end after the error frame"
        );
    }

    /// Dropping a sender before sending any data must immediately end the stream.
    /// This pins the close() (no error) contract.
    #[tokio::test]
    async fn dropped_sender_without_data_ends_stream_immediately() {
        let (tx, mut rx) = mpsc::channel::<Annotated<serde_json::Value>>(4);
        drop(tx);
        assert!(
            rx.recv().await.is_none(),
            "stream must end immediately when sender is dropped with no data"
        );
    }

    // The rest of this module's surface -- TypedSink send/close semantics,
    // send_terminal's spawn-vs-blocking_send branches, decode_response, and
    // handler_supports_push -- cannot be unit-tested here: pyo3's
    // `extension-module` is hardcoded in Cargo.toml, so any test whose object
    // graph reaches the Python C API fails to link against libpython. Covering
    // it needs pytest against a maturin-built extension. See DYN-3703.
}
