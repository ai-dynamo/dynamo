// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-request batched egress engine (flag-gated; see DYN-3703).
//!
//! The default `PythonServerStreamingEngine` drives ONE Python async generator
//! per request and crosses the Python->Rust bridge once per token. This engine
//! instead drives a SINGLE multiplexed `drain` generator for the whole worker:
//! each `__anext__` returns a whole engine step's `(request_id, chunk)` pairs,
//! so the batch crosses the bridge in one crossing and is demuxed back to each
//! request's response stream in GIL-free Rust. Analogue of TRT-LLM's
//! `handle_for_ipc_batched`.
//!
//! Python side is `components/src/dynamo/trtllm/request_handlers/batched_egress.py`.
//!
//! BUILD NOTES (this file is a best-effort scaffold; verify at compile):
//!  - `demand_driven_python_stream` and `invoke_generator` in `engine.rs` must
//!    be made `pub(crate)` to reuse them here (or replicate the unfold driver).
//!  - Wire in `lib.rs`: `mod batched_egress;` and an `Endpoint`
//!    `serve_endpoint_batched(submit, drain, ...)` method that builds this
//!    engine + the same `PythonServerStreamingIngress` adapter used by
//!    `serve_endpoint`.
//!  - Confirm the exact `PythonResponseItem` / `PythonPayload` constructors
//!    against `python_payload.rs` (names used below mirror `engine.rs`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Error, Result};
use dashmap::DashMap;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use tokio::sync::mpsc;
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
};

use crate::engine::{PythonServerStreamingEngine, demand_driven_python_stream, map_python_exception};
use crate::python_payload::{PythonPayload, PythonResponseItem};
use super::context::Context;

/// Per-request response sink registered while the request is in flight.
type Route = mpsc::Sender<PythonResponseItem>;

const RESPONSE_CHANNEL_DEPTH: usize = 128;

/// Engine that batches egress across requests. Constructed from the Python
/// `submit` coroutine-fn and the `drain` async-generator-fn.
#[derive(Clone)]
pub(crate) struct PythonBatchedEgressEngine {
    submit: Arc<PyObject>,
    drain: Arc<PyObject>,
    event_loop: Arc<PyObject>,
    routes: Arc<DashMap<String, Route>>,
    forwarder_started: Arc<AtomicBool>,
}

impl PythonBatchedEgressEngine {
    pub(crate) fn new(submit: PyObject, drain: PyObject, event_loop: PyObject) -> Self {
        Self {
            submit: Arc::new(submit),
            drain: Arc::new(drain),
            event_loop: Arc::new(event_loop),
            routes: Arc::new(DashMap::new()),
            forwarder_started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the single drain->demux forwarder exactly once.
    fn ensure_forwarder(&self) {
        if self.forwarder_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let drain = self.drain.clone();
        let event_loop = self.event_loop.clone();
        let routes = self.routes.clone();

        tokio::spawn(async move {
            // Build the drain async generator and wrap it as a demand-driven
            // Rust stream (same machinery the per-request path uses).
            let stream = Python::with_gil(|py| -> PyResult<_> {
                let generator = drain.call0(py)?; // drain() -> async generator
                let locals = pyo3_async_runtimes::TaskLocals::new(event_loop.bind(py).clone());
                demand_driven_python_stream(locals, generator.into_bound(py))
            });
            let mut stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("batched egress: failed to start drain: {e}");
                    return;
                }
            };

            while let Some(item) = stream.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(e) => {
                        tracing::error!("batched egress drain error: {e}");
                        break;
                    }
                };
                // ONE GIL acquisition converts + splits the whole step's batch
                // into owned (request_id, PythonResponseItem, is_done) triples.
                let routed: Vec<(String, Py<PyAny>, bool)> = Python::with_gil(|py| {
                    let mut out = Vec::new();
                    let list = match item.bind(py).downcast::<PyList>() {
                        Ok(list) => list.clone(),
                        Err(_) => {
                            tracing::error!("batched egress: drain yielded non-list");
                            return out;
                        }
                    };
                    for elem in list.iter() {
                        // each elem is a (request_id, chunk_or_None) tuple
                        let Ok(tup) = elem.downcast::<PyTuple>() else {
                            continue;
                        };
                        let Ok(rid_obj) = tup.get_item(0) else { continue };
                        let Ok(chunk) = tup.get_item(1) else { continue };
                        let Ok(rid) = rid_obj.extract::<String>() else {
                            continue;
                        };
                        let is_done = chunk.is_none();
                        out.push((rid, chunk.unbind(), is_done));
                    }
                    out
                });

                for (rid, chunk, is_done) in routed {
                    if is_done {
                        // Drop the sender -> closes this request's ManyOut.
                        routes.remove(&rid);
                        continue;
                    }
                    if let Some(tx) = routes.get(&rid) {
                        let item = PythonResponseItem::new(Ok(chunk));
                        if tx.send(item).await.is_err() {
                            drop(tx);
                            routes.remove(&rid);
                        }
                    }
                }
            }
        });
    }
}

#[async_trait::async_trait]
impl AsyncEngine<SingleIn<PythonPayload>, ManyOut<PythonResponseItem>, Error>
    for PythonBatchedEgressEngine
{
    async fn generate(
        &self,
        request: SingleIn<PythonPayload>,
    ) -> Result<ManyOut<PythonResponseItem>, Error> {
        self.ensure_forwarder();

        let (request, context) = request.transfer(());
        let ctx = context.context();
        let id = context.id().to_string();

        // Register this request's response route BEFORE submitting so the drain
        // can never deliver a token before the route exists.
        let (tx, rx) = mpsc::channel::<PythonResponseItem>(RESPONSE_CHANNEL_DEPTH);
        self.routes.insert(id.clone(), tx);

        // Schedule submit(request, context) as a fire-and-forget coroutine on the
        // worker's event loop. This starts the per-request feeder that runs the
        // existing handler.generate and pushes tagged chunks onto the shared
        // Python queue that `drain` batches.
        let submit = self.submit.clone();
        let event_loop = self.event_loop.clone();
        let metadata = context.metadata().clone();
        let trace = dynamo_runtime::logging::get_distributed_tracing_context();
        let request_py = request.into_inner();
        let schedule = Python::with_gil(|py| -> PyResult<()> {
            let ctx_obj = Py::new(py, Context::new(ctx.clone(), trace, None, metadata))?;
            let coro = submit.call1(py, (request_py, ctx_obj))?;
            // asyncio.run_coroutine_threadsafe(coro, loop) — fire and forget.
            let asyncio = py.import("asyncio")?;
            asyncio.call_method1(
                "run_coroutine_threadsafe",
                (coro, event_loop.bind(py).clone()),
            )?;
            Ok(())
        });
        if let Err(e) = schedule {
            self.routes.remove(&id);
            return Err(map_python_exception(e).into());
        }

        Ok(ResponseStream::new(
            Box::pin(ReceiverStream::new(rx)),
            context.context(),
        ))
    }
}

/// Constructed by `Endpoint::serve_endpoint_batched` in `lib.rs`.
pub(crate) fn new_engine(
    submit: PyObject,
    drain: PyObject,
    event_loop: PyObject,
) -> PythonBatchedEgressEngine {
    PythonBatchedEgressEngine::new(submit, drain, event_loop)
}

// Silence unused import if the ingress adapter is wired directly in lib.rs.
#[allow(unused_imports)]
use PythonServerStreamingEngine as _KeepEngineImport;
