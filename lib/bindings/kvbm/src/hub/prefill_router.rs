// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`PrefillRouterHandler`] is a self-contained worker-side runtime that:
//!
//!   1. stands up its own [`velo::Velo`] participant with a TCP transport,
//!   2. registers a typed unary handler for
//!      [`PREFILL_DISPATCH_HANDLER`](kvbm_hub::PREFILL_DISPATCH_HANDLER)
//!      whose closure invokes a captured Python lambda for each dispatched
//!      request,
//!   3. registers itself with a remote hub via
//!      [`HubClient`](kvbm_hub::HubClient) advertising
//!      `Feature::PrefillRouter(Velo{instance_id})`,
//!   4. holds the registration guard until [`PrefillRouterHandler::shutdown`]
//!      is called (or the pyclass is dropped).
//!
//! The Python lambda is `(req_dict, event) -> None` — it is expected to
//! schedule asynchronous work on the caller's asyncio loop and signal the
//! provided [`CompletionEvent`] when the prefill is complete. The handler
//! awaits the event on a tokio oneshot and returns the outcome as a
//! `PrefillDispatchResponse` to the hub.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use kvbm_hub::{
    Feature, HubClient, PREFILL_DISPATCH_HANDLER, PrefillBackendAdvertisement,
    PrefillDispatchRequest, PrefillDispatchResponse, PrefillRouterConfig, RuntimeConfigSummary,
};
use parking_lot::Mutex;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::oneshot;
use velo::Handler;
use velo::transports::tcp::TcpTransportBuilder;

/// Tokio oneshot wrapped for Python so a lambda's `_run` coroutine can
/// signal completion back into the Rust velo handler awaiting it.
///
/// Python sees only `ok()` and `err(msg)`. The sender is set by the Rust
/// caller at construction time and consumed on the first call; subsequent
/// calls are silently swallowed (double-fire is harmless).
#[pyclass]
pub struct CompletionEvent {
    tx: Mutex<Option<oneshot::Sender<Result<(), String>>>>,
}

impl CompletionEvent {
    fn with_sender(tx: oneshot::Sender<Result<(), String>>) -> Self {
        Self {
            tx: Mutex::new(Some(tx)),
        }
    }
}

#[pymethods]
impl CompletionEvent {
    /// Signal successful completion. Idempotent.
    fn ok(&self) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(Ok(()));
        }
    }

    /// Signal failure with a human-readable reason. Idempotent.
    fn err(&self, msg: String) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(Err(msg));
        }
    }
}

struct Inner {
    /// Held so the velo participant lives until the handler is dropped.
    /// Dropping this `Arc` is what stops serving on the registered
    /// handler.
    #[allow(dead_code)]
    velo: Arc<velo::Velo>,
    hub: Arc<HubClient>,
    hub_velo_id: Option<velo_ext::InstanceId>,
    worker_velo_id: velo_ext::InstanceId,
}

/// Worker-side prefill-router runtime exposed to Python.
#[pyclass]
pub struct PrefillRouterHandler {
    inner: OnceLock<Arc<Inner>>,
    runtime: Arc<tokio::runtime::Runtime>,
    shutdown_done: AtomicBool,
}

#[pymethods]
impl PrefillRouterHandler {
    /// Construct and register with the hub synchronously.
    ///
    /// Arguments:
    /// - `lambda`: a `(req_dict, event) -> None` Python callable. Invoked
    ///   under a brief GIL hold for every dispatched prefill request.
    /// - `hub_url`: the hub's discovery URL, e.g. `http://127.0.0.1:1337`.
    /// - `bind_addr`: optional `host:port` to bind the worker's velo TCP
    ///   transport to. Defaults to `0.0.0.0:0` (OS-assigned port).
    #[new]
    #[pyo3(signature = (lambda, hub_url, bind_addr=None))]
    fn new(lambda: Py<PyAny>, hub_url: String, bind_addr: Option<String>) -> PyResult<Self> {
        let lambda = Arc::new(lambda);
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("kvbm-prefill-router")
                .build()
                .map_err(|e| PyRuntimeError::new_err(format!("build tokio runtime: {e}")))?,
        );

        let bind = bind_addr.unwrap_or_else(|| "0.0.0.0:0".to_string());
        let inner: Arc<Inner> = runtime
            .block_on(async move { build_inner(bind, hub_url, lambda).await })
            .map_err(|e| PyRuntimeError::new_err(format!("prefill router setup: {e:#}")))?;

        let cell = OnceLock::new();
        let _ = cell.set(inner);
        Ok(Self {
            inner: cell,
            runtime,
            shutdown_done: AtomicBool::new(false),
        })
    }

    /// Worker's velo `InstanceId` as a string.
    fn worker_velo_id(&self) -> PyResult<String> {
        let inner = self
            .inner
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("PrefillRouterHandler is uninitialized"))?;
        Ok(inner.worker_velo_id.to_string())
    }

    /// Hub's velo `InstanceId` (if the hub was configured with a transport).
    fn hub_velo_id(&self) -> PyResult<Option<String>> {
        let inner = self
            .inner
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("PrefillRouterHandler is uninitialized"))?;
        Ok(inner.hub_velo_id.as_ref().map(|id| id.to_string()))
    }

    /// Unregister from the hub. Idempotent.
    fn shutdown(&self) -> PyResult<()> {
        if self.shutdown_done.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let Some(inner) = self.inner.get() else {
            return Ok(());
        };
        let hub = Arc::clone(&inner.hub);
        if let Err(e) = self.runtime.block_on(async move { hub.unregister().await }) {
            tracing::warn!(error = %e, "PrefillRouterHandler: hub unregister failed");
        }
        Ok(())
    }
}

impl Drop for PrefillRouterHandler {
    fn drop(&mut self) {
        if !self.shutdown_done.swap(true, Ordering::SeqCst)
            && let Some(inner) = self.inner.get()
        {
            let hub = Arc::clone(&inner.hub);
            if let Err(e) = self
                .runtime
                .block_on(async move { hub.unregister().await })
            {
                tracing::warn!(error = %e, "PrefillRouterHandler: hub unregister on drop failed");
            }
        }
    }
}

async fn build_inner(
    bind_addr: String,
    hub_url: String,
    lambda: Arc<Py<PyAny>>,
) -> anyhow::Result<Arc<Inner>> {
    let listener = std::net::TcpListener::bind(&bind_addr)
        .map_err(|e| anyhow::anyhow!("bind tcp listener {bind_addr}: {e}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| anyhow::anyhow!("set_nonblocking on tcp listener: {e}"))?;
    let transport = TcpTransportBuilder::new()
        .from_listener(listener)
        .map_err(|e| anyhow::anyhow!("tcp transport from_listener: {e}"))?
        .build()
        .map_err(|e| anyhow::anyhow!("tcp transport build: {e}"))?;

    let velo = velo::Velo::builder()
        .add_transport(Arc::new(transport))
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("velo build: {e}"))?;
    let worker_velo_id = velo.instance_id();
    let peer_info = velo.peer_info();

    register_dispatch_handler(velo.messenger(), Arc::clone(&lambda))?;

    let hub = kvbm_hub::HubClientBuilder::from_url(&hub_url)
        .map_err(|e| anyhow::anyhow!("parse hub url {hub_url}: {e}"))?
        .build()
        .map_err(|e| anyhow::anyhow!("build HubClient for {hub_url}: {e}"))?;
    hub.register_handlers_messenger(velo.messenger())
        .map_err(|e| anyhow::anyhow!("register hub handlers on velo: {e}"))?;

    let hub_velo_id = hub
        .register_instance_with_features_and_runtime(
            peer_info,
            vec![Feature::PrefillRouter(PrefillRouterConfig {
                backend: PrefillBackendAdvertisement::Velo {
                    instance_id: worker_velo_id,
                },
            })],
            RuntimeConfigSummary::default(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("register with hub {hub_url}: {e}"))?;

    tracing::info!(
        worker = %worker_velo_id,
        hub_url = %hub_url,
        hub = ?hub_velo_id,
        "PrefillRouterHandler: registered with hub"
    );

    Ok(Arc::new(Inner {
        velo,
        hub,
        hub_velo_id,
        worker_velo_id,
    }))
}

fn register_dispatch_handler(
    messenger: &Arc<velo::Messenger>,
    lambda: Arc<Py<PyAny>>,
) -> anyhow::Result<()> {
    let handler = Handler::typed_unary_async::<
        PrefillDispatchRequest,
        PrefillDispatchResponse,
        _,
        _,
    >(PREFILL_DISPATCH_HANDLER, move |ctx| {
        let lambda = Arc::clone(&lambda);
        async move {
            let (tx, rx) = oneshot::channel::<Result<(), String>>();
            // Brief GIL hold: build CompletionEvent + pythonize request +
            // invoke the lambda. The lambda is expected to schedule its
            // async work onto its captured asyncio loop and return None.
            // Round-trip through `serde_json::Value` BEFORE pythonize so
            // the Python side sees JSON-friendly types only — never Python
            // `bytes`. Background: `kv_transfer_params.remote_prefill.
            // sequence_hashes` is a `Vec<PositionalLineageHash>`, and
            // `PositionalLineageHash` serializes its inner u128 via
            // `serde_bytes_u128` (`serializer.serialize_bytes(&val.
            // to_be_bytes())`). Pythonize honors `serialize_bytes`
            // faithfully and produces a Python `bytes` object — but
            // downstream consumers (the vLLM scheduler ultimately calls
            // `json.dumps(kv_transfer_params)` to ship the params to
            // its Rust side) cannot encode `bytes` to JSON and crash
            // the engine. `serde_json` has no bytes type, so a value-tree
            // round-trip turns the byte arrays into JSON arrays of u8 ints
            // — pythonize then maps those to Python `list[int]`, which is
            // the same shape the HTTP execution backend produces (since
            // its body is also JSON) and which `json.dumps` handles
            // natively. This keeps the Python type identical across both
            // transports and avoids touching every downstream consumer.
            let json_input = serde_json::to_value(&ctx.input)
                .map_err(|e| anyhow::anyhow!("serialize PrefillDispatchRequest: {e}"))?;
            let call_result: Result<(), String> = Python::attach(|py| {
                let evt = CompletionEvent::with_sender(tx);
                let py_evt = Py::new(py, evt)
                    .map_err(|e| format!("Py::new(CompletionEvent): {e}"))?;
                let py_req = pythonize::pythonize(py, &json_input)
                    .map_err(|e| format!("pythonize request: {e}"))?;
                lambda
                    .call1(py, (py_req, py_evt))
                    .map_err(|e| format!("lambda invocation raised: {e}"))?;
                Ok(())
            });

            if let Err(msg) = call_result {
                return Ok(PrefillDispatchResponse {
                    ok: false,
                    error: Some(msg),
                });
            }

            // Await completion outside the GIL.
            match rx.await {
                Ok(Ok(())) => Ok(PrefillDispatchResponse {
                    ok: true,
                    error: None,
                }),
                Ok(Err(msg)) => Ok(PrefillDispatchResponse {
                    ok: false,
                    error: Some(msg),
                }),
                Err(_) => Ok(PrefillDispatchResponse {
                    ok: false,
                    error: Some("CompletionEvent dropped before signal".into()),
                }),
            }
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("register {PREFILL_DISPATCH_HANDLER} handler: {e}"))?;
    Ok(())
}
