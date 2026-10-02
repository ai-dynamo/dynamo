// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inverted (push-based) Python -> Rust response egress.
//!
//! The pull path in [`crate::engine`] costs two GIL acquisitions on arbitrary
//! tokio threads per response. Here the Python handler — already holding the
//! GIL — calls [`ResponseSender::send`] instead: the response is encoded to
//! request-plane bytes inline under that existing GIL and the resulting
//! [`PushFrame`] is enqueued on a bounded `tokio::sync::mpsc`, so the tokio
//! side only ever does `rx.recv().await` and never touches Python. Rationale
//! and measurements: #12845.
//!
//! Errors terminate the stream with the type `map_python_exception` assigned
//! them, which is what request migration and worker inhibition key on.
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
use bytes::{BufMut, Bytes, BytesMut};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};

use tokio::sync::mpsc;

use dynamo_runtime::engine::AsyncEngineContext;
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::logging::get_distributed_tracing_context;
use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, NetworkStreamWrapper, RESPONSE_ENCODE_CAPACITY_HINT,
    RequestPlanePayloadCodec,
};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, PipelineError, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;

use crate::context::{Context, callable_accepts_kwarg};
use crate::engine::{self, map_python_exception};
use crate::python_payload::{self, PythonPayload};

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

// ── Encode buffer ────────────────────────────────────────────────────────────

/// One request's reusable frame-encoding buffer.
///
/// Rather than allocating a buffer per frame, every frame is written into one
/// large `BytesMut` chunk and then handed out using [`BytesMut::split`] — the
/// `bytes` crate method that returns the bytes written so far as a separate
/// handle and leaves `buf` empty but still holding the rest of the chunk's
/// capacity. The allocator is therefore touched about once per chunk instead of
/// once per frame.
///
/// `split` is `O(1)` and copies nothing: both handles point into the same
/// allocation and share a reference count. A chunk is only freed once every
/// frame taken from it has been sent and dropped. The response channel bounds
/// how many can be in flight at a time and a chunk is small, so that retention
/// is not worth designing around.
struct EncodeBuffer {
    buf: BytesMut,
    /// Frames within one stream are all about the same size, so the last one is
    /// a good predictor of how much room the next one needs.
    last_frame_len: usize,
}

impl EncodeBuffer {
    /// How much is allocated at a time. Sized to cover a full in-flight window:
    /// `engine::RESPONSE_CHANNEL_DEPTH` frames of a few dozen bytes each is well
    /// under 8 KiB, so a chunk retires at roughly the rate its frames drain.
    /// Raising it trades resident memory per in-flight request for allocator
    /// calls that are already amortized over ~150 frames.
    const CHUNK: usize = 8 * 1024;
    /// Floor on the predicted frame size, so an unusually small frame does not
    /// make the next one ask for less room than any frame could need.
    const MIN_RESERVE: usize = 64;
    /// Starting capacity for the private buffer a re-entrant `send` falls back
    /// to. Shares the constant every other response encoder starts from, so the
    /// fallback does not grow from zero.
    const SCRATCH_CAPACITY: usize = RESPONSE_ENCODE_CAPACITY_HINT;

    fn new() -> Self {
        Self {
            buf: BytesMut::new(),
            last_frame_len: 0,
        }
    }

    /// Make room for another frame, allocating a fresh chunk only once the
    /// leftover capacity in the current one runs short.
    ///
    /// Two things this deliberately does not do:
    ///
    /// It does not commit a chunk for the first frame. Encode, prefill and short
    /// decode requests often produce exactly one frame, and chunking those would
    /// swap a small allocation for an 8 KiB one that is never reused. Until a
    /// second frame proves the request is streaming, `BytesMut` sizes the frame
    /// itself.
    ///
    /// It does not let the predictor exceed one chunk. One oversized frame must
    /// not pin an oversized buffer for the rest of the request. An outsized frame
    /// still encodes fine — `BytesMut` grows on write, and `split` hands that
    /// large allocation away with the frame rather than retaining it.
    fn reserve(&mut self) {
        if self.last_frame_len == 0 {
            return;
        }
        let needed = self.last_frame_len.clamp(Self::MIN_RESERVE, Self::CHUNK);
        if self.buf.capacity() < needed {
            self.buf.reserve(Self::CHUNK);
        }
    }

    /// Hand out the frame just written as its own `Bytes`, and remember its
    /// length so the next [`Self::reserve`] knows what to expect. `buf` is left
    /// empty, with whatever is left of the chunk still available to the frame
    /// after this one.
    fn take(&mut self) -> Bytes {
        self.last_frame_len = self.buf.len();
        self.buf.split().freeze()
    }

    /// Drop a partially written frame after an encode failure, so the next
    /// frame does not inherit its bytes.
    ///
    /// An oversized partial frame also has to give its allocation back. On the
    /// success path `split` hands that memory away with the frame, but a
    /// discarded frame has nothing to hand it to — and `clear` keeps capacity,
    /// which [`Self::reserve`] then sees as "already big enough" and never
    /// replaces. Without this the buffer would stay oversized for the rest of
    /// the request.
    fn discard(&mut self) {
        if self.buf.capacity() > Self::CHUNK {
            self.buf = BytesMut::new();
        } else {
            self.buf.clear();
        }
    }
}

// ── Wire frame ───────────────────────────────────────────────────────────────

/// One response, already encoded for the request plane.
///
/// Encoding happens in `send`, under the GIL the handler already holds, and the
/// payload is still a [`PythonPayload`] at that moment — so it transcodes
/// straight into the codec in one pass, exactly as on the pull path, and
/// nothing the codec can represent is narrowed on the way.
///
/// `codec` records which codec produced `bytes`. `send` has no request in hand
/// to read the codec from, so the sink captures the process-global
/// [`RequestPlanePayloadCodec::configured`] once per request. That is the value
/// this worker advertises in its instance record, which is also where callers
/// look up the codec to address it with — so the two normally agree. The one
/// exception is a caller old enough to predict neither, which falls back to
/// JSON. [`PushFrame::into_encoded`] re-encodes for that case rather than
/// putting the wrong bytes on the wire.
pub(crate) struct PushFrame {
    bytes: Bytes,
    is_error: bool,
    codec: RequestPlanePayloadCodec,
}

impl std::fmt::Debug for PushFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PushFrame")
            .field("len", &self.bytes.len())
            .field("is_error", &self.is_error)
            .field("codec", &self.codec.name())
            .finish()
    }
}

impl PushFrame {
    /// Encode one Python response object, with the GIL held by the caller.
    ///
    /// Both steps are shared with the pull path — `parse_python_response`
    /// interprets the object, `write_annotated_response` produces the bytes — so
    /// the two cannot drift. The only change from the pull path is where those
    /// bytes land: a buffer reused across this request's frames, rather than a
    /// fresh `Vec` each time.
    ///
    /// ## Why `try_lock` rather than `lock`
    ///
    /// [`Self::write`] walks `obj`, and walking a Python object can run arbitrary
    /// Python: `__bool__` when an envelope is tested for truthiness, `__iter__` or
    /// `__len__` during the pythonize walk. That Python is free to call `send`
    /// again on this same sender.
    ///
    /// The mutex is not reentrant, so blocking here would deadlock the event-loop
    /// thread *while it holds the GIL*, stalling every request in the interpreter.
    /// Instead a re-entrant call falls back to its own private buffer — slower for
    /// that one frame, which is a good trade against a hang.
    ///
    /// Either way the lock is released before returning, and so before the
    /// caller's enqueue, which can block.
    fn encode(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        codec: RequestPlanePayloadCodec,
        buffer: &parking_lot::Mutex<EncodeBuffer>,
    ) -> PyResult<Self> {
        let Some(mut pooled) = buffer.try_lock() else {
            // Sized, not `new()`: growing from zero is the cost this whole
            // change exists to remove, and the fallback should not quietly
            // reintroduce it for the frame that takes it.
            let mut scratch = BytesMut::with_capacity(EncodeBuffer::SCRATCH_CAPACITY);
            let is_error = Self::write(py, obj, codec, &mut scratch)?;
            return Ok(Self {
                bytes: scratch.freeze(),
                is_error,
                codec,
            });
        };

        pooled.reserve();
        match Self::write(py, obj, codec, &mut pooled.buf) {
            Ok(is_error) => Ok(Self {
                bytes: pooled.take(),
                is_error,
                codec,
            }),
            Err(error) => {
                // The frame may be half-written. Drop those bytes so the next
                // frame does not inherit them.
                pooled.discard();
                Err(error)
            }
        }
    }

    /// Write one response's frame into `out`, returning whether it is an error
    /// frame. On failure `out` may be left holding a partial frame, so the
    /// caller must discard it.
    fn write(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        codec: RequestPlanePayloadCodec,
        out: &mut BytesMut,
    ) -> PyResult<bool> {
        let annotated =
            python_payload::parse_python_response(obj.clone().unbind(), py).map_err(|error| {
                PyValueError::new_err(format!(
                    "critical error: invalid response object from python handler; \
                     application-logic-mismatch: {error}"
                ))
            })?;
        python_payload::write_annotated_response(codec, annotated, &mut out.writer()).map_err(
            |error| {
                PyValueError::new_err(format!(
                    "critical error: failed serializing python response as {}: {error}",
                    codec.name()
                ))
            },
        )
    }

    /// Encode a terminal error frame. Pure Rust — no Python object involved —
    /// so this is callable from either side of the bridge.
    fn error(annotated: Annotated<serde_json::Value>) -> Self {
        let codec = RequestPlanePayloadCodec::configured();
        // An error frame is a string and three `None`s; neither codec can fail
        // on it. Degrade to an empty frame rather than panic if one somehow does.
        let (bytes, is_error) = python_payload::encode_annotated_response(codec, annotated)
            .unwrap_or_else(|error| {
                tracing::error!(%error, "push egress: failed to encode terminal error frame");
                (Vec::new(), true)
            });
        Self {
            bytes: bytes.into(),
            is_error,
            codec,
        }
    }

    /// Hand the frame to the ingress for the codec this request is addressed
    /// with. Normally a move; see the type docs for when it is not.
    pub(crate) fn into_encoded(
        self,
        target: RequestPlanePayloadCodec,
    ) -> Result<EncodedResponseFrame, PipelineError> {
        if self.codec == target {
            return Ok(EncodedResponseFrame {
                bytes: self.bytes,
                is_error: self.is_error,
                stop_stream: false,
            });
        }

        // Caller addressed us with a different codec than the one we advertise.
        // `rmpv::Value` is the intermediate because it is the wider of the two
        // wire models — it carries binary and non-finite floats, which
        // `serde_json::Value` does not — so nothing is lost that the target
        // codec could still have represented.
        tracing::debug!(
            from = self.codec.name(),
            to = target.name(),
            "push egress: re-encoding response for the caller's request-plane codec"
        );
        let wrapper: NetworkStreamWrapper<Annotated<rmpv::Value>> =
            self.codec.decode(&self.bytes).map_err(|error| {
                PipelineError::SerializationError(format!(
                    "failed decoding push-egress response as {} for re-encoding: {error}",
                    self.codec.name()
                ))
            })?;
        let bytes = target.encode(&wrapper).map_err(|error| {
            PipelineError::SerializationError(format!(
                "Failed serializing {} push-egress response: {error}",
                target.name()
            ))
        })?;
        Ok(EncodedResponseFrame {
            bytes: bytes.into(),
            is_error: self.is_error,
            stop_stream: false,
        })
    }
}

// ── GIL-side sink ────────────────────────────────────────────────────────────

/// The GIL-side half of one request's push channel. Every method is called with
/// the GIL already held by the Python caller, except the two the Rust driver
/// task uses to terminate a stream its handler left open.
struct ResponseSink {
    /// `None` once the stream has been closed. Dropping the last `Sender` is
    /// what ends the receiver stream, so closing is "take the sender".
    tx: Mutex<Option<mpsc::Sender<PushFrame>>>,
    /// Stopped when the consumer goes away, mirroring what the pull path's
    /// forwarder does on a closed channel (`engine.rs`): without it a dropped
    /// response stream would leave the handler generating into nothing until it
    /// noticed the send error on its own.
    ctx: Arc<dyn AsyncEngineContext>,
    /// Read once per request instead of once per frame. It is backed by a
    /// `OnceLock` (see [`RequestPlanePayloadCodec::configured`]), so reading it
    /// again later could only ever return this same value.
    codec: RequestPlanePayloadCodec,
    /// Reused across this request's frames. Behind a mutex because `send` takes
    /// `&self`. Uncontended in practice: one request's handler pushes from a
    /// single event-loop thread.
    encode_buffer: parking_lot::Mutex<EncodeBuffer>,
}

impl ResponseSink {
    /// Clone the sender out from under the lock rather than holding the lock
    /// across the enqueue: a blocking send while holding the mutex would let a
    /// concurrent `close()` deadlock against it.
    fn sender(&self) -> Option<mpsc::Sender<PushFrame>> {
        self.tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn take_sender(&self) -> Option<mpsc::Sender<PushFrame>> {
        self.tx
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
    }

    /// The consumer dropped the response stream. Tell the request context so
    /// the handler stops generating, then report it to the caller.
    fn consumer_gone(&self) -> PyErr {
        self.ctx.stop_generating();
        PyRuntimeError::new_err(
            "response stream is closed; the consumer dropped the response stream",
        )
    }

    /// Convert `obj` to wire bytes and enqueue it.
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        // The whole Python->Rust crossing for one response: the encode plus the
        // enqueue. Unlike the pull path neither step acquires the GIL — the
        // Python handler already holds it.
        let frame = PushFrame::encode(py, obj, self.codec, &self.encode_buffer)?;

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
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(self.consumer_gone()),
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
        py.allow_threads(|| tx.blocking_send(frame))
            .map_err(|_| self.consumer_gone())
    }

    /// Normal end of stream.
    fn close(&self) {
        // Dropping the last sender is what ends the receiver stream. Idempotent
        // so the Rust-side safety net can close a stream the handler already
        // closed itself.
        drop(self.take_sender());
    }

    /// Terminate the stream with an untyped error frame.
    fn close_with_error(&self, message: String) {
        let Some(tx) = self.take_sender() else {
            return;
        };
        self.send_terminal(tx, PushFrame::error(Annotated::from_error(message)));
    }

    /// Terminate the stream with a typed backend error frame. Not exposed to
    /// Python; used by the Rust-side safety net when the handler's generator
    /// raises instead of closing the sender itself. Preserving the type matters
    /// downstream — `BackendError::EngineShutdown` is what triggers request
    /// migration and marks the worker inhibited.
    fn close_with_dynamo_error(&self, error: DynamoError) {
        let Some(tx) = self.take_sender() else {
            return;
        };
        self.send_terminal(tx, PushFrame::error(Annotated::from_err(error)));
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
    fn send_terminal(&self, tx: mpsc::Sender<PushFrame>, frame: PushFrame) {
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
///     async for chunk in engine.generate(request):   # push path: yields nothing
///         response_sender.send(chunk)
///     response_sender.close()
///     # exceptions propagate: Rust classifies them and terminates the stream
/// ```
///
/// `send` blocks when the consumer is behind; it is safe to call from the
/// asyncio loop thread because the GIL is released across that wait.
#[pyclass]
pub struct ResponseSender {
    sink: Arc<ResponseSink>,
}

#[pymethods]
impl ResponseSender {
    /// Encode one response object and enqueue it for the Rust egress path.
    ///
    /// Raises `RuntimeError` if the stream is already closed and `ValueError`
    /// if the object cannot be encoded.
    fn send(&self, py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        self.sink.send(py, obj)
    }

    /// Normal end of stream. Idempotent.
    fn close(&self) -> PyResult<()> {
        self.sink.close();
        Ok(())
    }

    /// Terminate the stream with an untyped error frame. Idempotent; a later
    /// `close()` is a no-op.
    ///
    /// Prefer letting the exception propagate out of the handler instead: Rust
    /// then runs it through `map_python_exception` and terminates the stream
    /// with a *typed* backend error, which is what drives request migration and
    /// worker inhibition downstream. This exists for failures that are not
    /// exceptions.
    fn close_with_error(&self, msg: String) -> PyResult<()> {
        self.sink.close_with_error(msg);
        Ok(())
    }
}

impl ResponseSender {
    /// Rust-side handle to the same sink, for the safety net that terminates
    /// the stream if the handler's generator raises.
    fn sink(&self) -> Arc<ResponseSink> {
        self.sink.clone()
    }
}

// ── Factory ──────────────────────────────────────────────────────────────────

/// Build one request's push channel.
///
/// The returned stream yields already-encoded [`PushFrame`]s; the ingress side
/// (`lib/runtime/src/pipeline/network/ingress/push_handler.rs`) is unchanged and
/// its encoder just forwards them.
pub(crate) fn response_channel(
    depth: usize,
    ctx: Arc<dyn AsyncEngineContext>,
) -> (
    ResponseSender,
    impl Stream<Item = PushFrame> + Send + 'static,
) {
    let (tx, rx) = mpsc::channel::<PushFrame>(depth);

    let sender = ResponseSender {
        sink: Arc::new(ResponseSink {
            tx: Mutex::new(Some(tx)),
            ctx,
            codec: RequestPlanePayloadCodec::configured(),
            encode_buffer: parking_lot::Mutex::new(EncodeBuffer::new()),
        }),
    };

    // The consumer side is pure Rust: no Python work, no GIL, no encoding, just
    // the wait for whatever the handler pushes next.
    let stream = ReceiverStream::new(rx);

    (sender, stream)
}

// ── Engine ───────────────────────────────────────────────────────────────────

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
impl AsyncEngine<SingleIn<PythonPayload>, ManyOut<PushFrame>, Error> for PythonPushEngine {
    async fn generate(
        &self,
        request: SingleIn<PythonPayload>,
    ) -> Result<ManyOut<PushFrame>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        let request_id = context.id().to_string();
        let metadata = context.metadata().clone();
        let current_trace_context = get_distributed_tracing_context();

        // Same depth as the pull path's forwarder, so both paths buffer the
        // same amount and backpressure behaves identically.
        let (sender, stream) = response_channel(engine::RESPONSE_CHANNEL_DEPTH, ctx.clone());
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
    //! **Root cause for push_egress.rs**: `ResponseSink::send` takes
    //! `Python<'_>` / `Bound<'_, PyAny>`, and `ResponseSender` is a
    //! `#[pyclass]`.  A test that instantiates either — `response_channel`
    //! creates both — pulls pyo3 and the Python C API into the test binary's
    //! object graph.
    //!
    //! **What compiles**: [`PushFrame`] and everything downstream of it. The
    //! frame is a `Bytes` plus two scalars, and both the terminal-frame
    //! constructor and the ingress hand-off are pure serde, so the encoding
    //! contract — the part this module actually owns — is testable here.
    //! `send`/`close` and `handler_supports_push` are not; they are covered by
    //! pytest against the built `.so`.  Gaps are documented at the bottom.

    use super::{EncodeBuffer, PushFrame};
    use crate::engine::RESPONSE_CHANNEL_DEPTH;
    use bytes::BufMut;
    use dynamo_runtime::pipeline::network::{NetworkStreamWrapper, RequestPlanePayloadCodec};
    use dynamo_runtime::protocols::annotated::Annotated;
    use tokio::sync::mpsc;

    // ── EncodeBuffer ─────────────────────────────────────────────────────────
    //
    // `PushFrame::encode` itself cannot be tested here — it takes `Python<'_>`
    // and a `Bound<'_, PyAny>`, which would pull the Python C API into the test
    // binary (see the module note above). But the buffer it drives is pure Rust,
    // and the buffer is where every failure mode of this change lives: frames
    // leaking into one another, a partial frame surviving, and the sizing policy.

    /// Write `n` bytes of `fill` as one frame would.
    fn write_frame(buffer: &mut EncodeBuffer, fill: u8, n: usize) {
        buffer.reserve();
        buffer.buf.put_slice(&vec![fill; n]);
    }

    /// Consecutive frames must come out independent and intact. `split` leaves
    /// them sharing one allocation, so a bad index would show up as one frame
    /// carrying another's bytes.
    #[test]
    fn consecutive_frames_are_independent() {
        let mut buffer = EncodeBuffer::new();
        let mut frames = Vec::new();
        for i in 0..64u8 {
            write_frame(&mut buffer, i, 50 + usize::from(i));
            frames.push(buffer.take());
        }
        for (i, frame) in frames.iter().enumerate() {
            let i = u8::try_from(i).expect("fits");
            assert_eq!(frame.len(), 50 + usize::from(i), "frame {i} wrong length");
            assert!(
                frame.iter().all(|b| *b == i),
                "frame {i} carries another frame's bytes"
            );
        }
    }

    /// A frame abandoned partway through must not survive into the next one.
    /// This is the `discard` path taken when encoding fails mid-write.
    #[test]
    fn discard_drops_a_partial_frame() {
        let mut buffer = EncodeBuffer::new();
        write_frame(&mut buffer, 0xAA, 40); // partial frame, then failure
        buffer.discard();

        write_frame(&mut buffer, 0xBB, 30);
        let frame = buffer.take();
        assert_eq!(frame.len(), 30, "next frame inherited the discarded bytes");
        assert!(frame.iter().all(|b| *b == 0xBB));
    }

    /// A request that sends exactly one frame — encode, prefill, short decode —
    /// must not be charged a full chunk it never reuses.
    #[test]
    fn first_frame_does_not_commit_a_chunk() {
        let mut buffer = EncodeBuffer::new();
        buffer.reserve();
        assert!(
            buffer.buf.capacity() < EncodeBuffer::CHUNK,
            "first frame pre-allocated a whole chunk: {}",
            buffer.buf.capacity()
        );
    }

    /// Once a request is known to be streaming, chunking kicks in.
    #[test]
    fn second_frame_commits_a_chunk() {
        let mut buffer = EncodeBuffer::new();
        write_frame(&mut buffer, 1, 52);
        let _ = buffer.take();
        buffer.reserve();
        assert!(
            buffer.buf.capacity() >= EncodeBuffer::CHUNK,
            "streaming request did not chunk: {}",
            buffer.buf.capacity()
        );
    }

    /// One oversized frame must not pin an oversized buffer for the rest of the
    /// request. The large allocation leaves with the frame; the next reserve is
    /// capped at a chunk.
    #[test]
    fn an_oversized_frame_does_not_pin_capacity() {
        let mut buffer = EncodeBuffer::new();
        write_frame(&mut buffer, 1, 52);
        let _ = buffer.take();

        let huge = 4 * 1024 * 1024;
        write_frame(&mut buffer, 2, huge);
        let big = buffer.take();
        assert_eq!(big.len(), huge);

        buffer.reserve();
        assert!(
            buffer.buf.capacity() <= EncodeBuffer::CHUNK,
            "an oversized frame pinned {} bytes for the rest of the request",
            buffer.buf.capacity()
        );
    }

    /// A frame that fails partway through must give its allocation back too,
    /// not just its bytes. `clear()` alone keeps capacity, and `reserve()` reads
    /// a large capacity as "already big enough" — so an oversized *failed* frame
    /// would pin its buffer for the rest of the request, which is the same leak
    /// `an_oversized_frame_does_not_pin_capacity` rules out on the success path.
    #[test]
    fn discard_releases_an_oversized_buffer() {
        let mut buffer = EncodeBuffer::new();
        write_frame(&mut buffer, 1, 52);
        let _ = buffer.take();

        // A large frame that fails mid-encode rather than being handed out.
        write_frame(&mut buffer, 2, 4 * 1024 * 1024);
        buffer.discard();

        assert!(
            buffer.buf.capacity() <= EncodeBuffer::CHUNK,
            "a discarded oversized frame pinned {} bytes for the rest of the request",
            buffer.buf.capacity()
        );

        // And the buffer must still be usable afterwards.
        write_frame(&mut buffer, 3, 40);
        let frame = buffer.take();
        assert_eq!(frame.len(), 40);
        assert!(
            frame.iter().all(|b| *b == 3),
            "discard leaked bytes forward"
        );
    }

    /// The re-entrancy fallback in `PushFrame::encode` depends on `try_lock`
    /// failing rather than blocking when the buffer is already held. Encoding a
    /// frame can run arbitrary Python that re-enters `send`, and blocking there
    /// would deadlock the event-loop thread with the GIL held.
    #[test]
    fn try_lock_fails_instead_of_blocking_when_held() {
        let buffer = parking_lot::Mutex::new(EncodeBuffer::new());
        let outer = buffer.lock();
        assert!(
            buffer.try_lock().is_none(),
            "try_lock must decline a held buffer so encode can fall back"
        );
        drop(outer);
        assert!(buffer.try_lock().is_some(), "lock must be reusable after");
    }

    /// Decode a frame's bytes back to the wire envelope, as the caller does.
    fn decode(
        bytes: &[u8],
        codec: RequestPlanePayloadCodec,
    ) -> NetworkStreamWrapper<Annotated<rmpv::Value>> {
        codec.decode(bytes).expect("frame must decode")
    }

    // ── channel backpressure contract ─────────────────────────────────────
    //
    // ResponseSink::send uses a channel of RESPONSE_CHANNEL_DEPTH -- the pull
    // path's constant, referenced rather than duplicated so the two cannot
    // drift.  The fast path uses try_send (no blocking); the slow path parks.
    // These tests use mpsc directly because instantiating the sink requires
    // Python symbols.

    /// The channel must accept exactly `RESPONSE_CHANNEL_DEPTH` items without
    /// blocking, then report Full. Catches drift between the constant and the
    /// actual channel constructor call.
    #[test]
    fn channel_accepts_exactly_response_channel_depth_items_then_full() {
        let (tx, _rx) = mpsc::channel::<PushFrame>(RESPONSE_CHANNEL_DEPTH);

        let mut sent = 0usize;
        for i in 0..RESPONSE_CHANNEL_DEPTH {
            match tx.try_send(PushFrame::error(Annotated::from_error(format!("{i}")))) {
                Ok(()) => sent += 1,
                Err(mpsc::error::TrySendError::Full(_)) => break,
                Err(e) => panic!("unexpected channel error: {e}"),
            }
        }
        assert_eq!(
            sent, RESPONSE_CHANNEL_DEPTH,
            "channel must buffer exactly {RESPONSE_CHANNEL_DEPTH} items without blocking"
        );

        assert!(
            matches!(
                tx.try_send(PushFrame::error(Annotated::from_error("overflow"))),
                Err(mpsc::error::TrySendError::Full(_))
            ),
            "item {RESPONSE_CHANNEL_DEPTH}+1 must fail Full, not Closed"
        );
    }

    // ── terminal frame contract ───────────────────────────────────────────
    //
    // close_with_error and close_with_dynamo_error each:
    //   1. Call take_sender() to atomically claim the send side (idempotence)
    //   2. Pass a PushFrame::error to send_terminal
    //   3. send_terminal delivers the frame and then drops the sender
    //
    // Step 2 is tested here directly; steps 1 and 3 (idempotence via
    // take_sender; stream end via drop) are pinned against mpsc below.

    /// A terminal frame must reach the caller as a decodable error frame with
    /// no data, and must be marked `is_error` so the ingress does not treat it
    /// as evidence the engine is healthy.
    #[test]
    fn terminal_frame_encodes_an_error_with_no_data() {
        let frame = PushFrame::error(Annotated::from_error("fatal"));
        assert!(frame.is_error, "terminal frames must be flagged is_error");

        let wrapper = decode(&frame.bytes, frame.codec);
        assert!(!wrapper.complete_final, "not the end-of-stream marker");
        let annotated = wrapper.data.expect("terminal frame carries data");
        assert!(annotated.error.is_some(), "expected an error field");
        assert!(annotated.data.is_none(), "error frames carry no data");
    }

    /// Matching codecs are the whole point: the bytes encoded under the GIL go
    /// out untouched, with no second serde pass.
    #[test]
    fn matching_codec_forwards_the_bytes_unchanged() {
        let frame = PushFrame::error(Annotated::from_error("fatal"));
        let (codec, expected) = (frame.codec, frame.bytes.clone());

        let encoded = frame.into_encoded(codec).expect("forward must succeed");
        assert_eq!(encoded.bytes, expected, "bytes must not be re-encoded");
        assert!(encoded.is_error);
        assert!(!encoded.stop_stream);
    }

    /// A caller addressing this worker with a different codec than the one it
    /// advertises must still get a frame it can decode. Sending the bytes
    /// through unchanged would put msgpack on a JSON stream.
    #[test]
    fn mismatched_codec_re_encodes_for_the_caller() {
        let frame = PushFrame {
            bytes: RequestPlanePayloadCodec::Msgpack
                .encode(&NetworkStreamWrapper {
                    data: Some(Annotated::from_data(serde_json::json!({"text": "hi"}))),
                    complete_final: false,
                })
                .expect("encode")
                .into(),
            is_error: false,
            codec: RequestPlanePayloadCodec::Msgpack,
        };

        let encoded = frame
            .into_encoded(RequestPlanePayloadCodec::Json)
            .expect("re-encode must succeed");
        let wrapper = decode(&encoded.bytes, RequestPlanePayloadCodec::Json);
        let data = wrapper.data.expect("data").data.expect("annotated data");
        assert_eq!(
            data.as_map()
                .expect("map")
                .iter()
                .find(|(k, _)| k.as_str() == Some("text"))
                .map(|(_, v)| v.as_str().unwrap_or_default()),
            Some("hi"),
            "payload must survive the re-encode"
        );
        assert!(!encoded.is_error, "the error flag must be preserved");
    }

    /// Pins the channel-level protocol: one error frame then end-of-stream.
    /// If send_terminal fails to drop the sender after delivering the error
    /// frame, the consumer would wait forever.
    #[tokio::test]
    async fn one_error_frame_then_dropped_sender_ends_stream() {
        let (tx, mut rx) = mpsc::channel::<PushFrame>(4);
        tx.send(PushFrame::error(Annotated::from_error("fatal")))
            .await
            .unwrap();
        drop(tx); // simulates send_terminal dropping the sender after the frame
        let item = rx.recv().await.expect("one error frame must arrive");
        assert!(item.is_error, "expected an error frame");
        assert!(
            rx.recv().await.is_none(),
            "stream must end after the error frame"
        );
    }

    /// Dropping a sender before sending any data must immediately end the stream.
    /// This pins the close() (no error) contract.
    #[tokio::test]
    async fn dropped_sender_without_data_ends_stream_immediately() {
        let (tx, mut rx) = mpsc::channel::<PushFrame>(4);
        drop(tx);
        assert!(
            rx.recv().await.is_none(),
            "stream must end immediately when sender is dropped with no data"
        );
    }

    // ── stop_stream divergence ────────────────────────────────────────────
    //
    // INTENTIONAL BEHAVIORAL DIFFERENCE from the pull path:
    //
    // Pull path (encode_python_response): parse errors and Python exceptions
    // produce `stop_stream: true`, telling the ingress to abort the engine
    // stream immediately after this frame.
    //
    // Push path (PushFrame::into_encoded): ALL frames — including error frames
    // from close_with_error / close_with_dynamo_error — have `stop_stream:
    // false`. End-of-stream is signalled by dropping the mpsc sender, not by
    // a flag. If downstream code keys on stop_stream to abort early it will
    // behave differently here; it must handle the channel-close path too.
    //
    // These tests pin the push-path contract so a future change cannot
    // accidentally match the pull path's stop_stream: true without someone
    // noticing and auditing the downstream consumer.

    /// Push-path error frames must have `stop_stream: false`, even though the
    /// pull path sets `stop_stream: true` for the same conditions. End-of-stream
    /// is signalled by dropping the sender, not by this flag.
    #[test]
    fn push_error_frame_stop_stream_is_always_false_matching_codec() {
        let frame = PushFrame::error(Annotated::from_error("fatal"));
        let codec = frame.codec;
        let encoded = frame.into_encoded(codec).expect("forward must succeed");
        assert!(encoded.is_error, "error frame must be flagged is_error");
        assert!(
            !encoded.stop_stream,
            "push-path error frames must have stop_stream: false (stream ends when sender drops)"
        );
    }

    /// The stop_stream: false contract must hold on the re-encode path too,
    /// so a mismatch between the push worker's codec and the caller's codec
    /// cannot accidentally flip the flag.
    #[test]
    fn push_error_frame_stop_stream_is_always_false_mismatched_codec() {
        let frame = PushFrame {
            bytes: RequestPlanePayloadCodec::Msgpack
                .encode(&NetworkStreamWrapper {
                    data: Some(Annotated::<serde_json::Value>::from_error("fatal")),
                    complete_final: false,
                })
                .expect("encode")
                .into(),
            is_error: true,
            codec: RequestPlanePayloadCodec::Msgpack,
        };
        let encoded = frame
            .into_encoded(RequestPlanePayloadCodec::Json)
            .expect("re-encode must succeed");
        assert!(encoded.is_error, "is_error must survive re-encode");
        assert!(
            !encoded.stop_stream,
            "stop_stream must remain false after re-encode"
        );
    }

    // The rest of this module's surface -- ResponseSink send/close semantics,
    // send_terminal's spawn-vs-blocking_send branches, PushFrame::encode, and
    // handler_supports_push -- cannot be unit-tested here: pyo3's
    // `extension-module` is hardcoded in Cargo.toml, so any test whose object
    // graph reaches the Python C API fails to link against libpython. Covering
    // it needs pytest against a maturin-built extension. See #12845.
}
