// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`PrefillRouterHandler`] is a self-contained worker-side runtime that:
//!
//!   1. stands up its own [`velo::Velo`] participant with a TCP transport,
//!   2. registers a typed unary handler for
//!      [`PREFILL_DISPATCH_HANDLER`](kvbm_hub::PREFILL_DISPATCH_HANDLER)
//!      whose closure invokes a captured Python lambda for each dispatched
//!      request,
//!   3. optionally registers a sibling typed unary handler for
//!      [`CALIBRATE_HANDLER`](kvbm_hub::CALIBRATE_HANDLER) that drives a
//!      separate Python lambda through a single-stream ISL sweep and fits
//!      the two-line performance model in Rust,
//!   4. registers itself with a remote hub via
//!      [`HubClient`](kvbm_hub::HubClient) advertising
//!      `Feature::PrefillRouter(Velo{instance_id})`,
//!   5. holds the registration guard until [`PrefillRouterHandler::shutdown`]
//!      is called (or the pyclass is dropped).
//!
//! The dispatch lambda is `(req_dict, event) -> None`; the calibrate
//! lambda is `(resolved_dict, event) -> None`. Both schedule asynchronous
//! work on the caller's asyncio loop and signal a [`CompletionEvent`] on
//! completion — calibrate signals via `ok_with_payload(json_str)` so the
//! Rust side can deserialize the raw trace payload and run the regression
//! analysis.
//!
//! Calibration and prefill dispatch are mutually exclusive on the same
//! worker via best-effort atomics (no mutex in the prefill hot path):
//! a `calibrating: AtomicBool` flag plus an `inflight_prefill: AtomicUsize`
//! counter. While calibrating, prefill returns `calibration_in_progress`;
//! while any prefill is in flight, calibrate refuses to start.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use kvbm_hub::{
    CALIBRATE_HANDLER, CalibrationDefaults, CalibrationRequest, CalibrationResponse,
    CalibrationSnapshot, Feature, HubClient, PREFILL_DISPATCH_HANDLER,
    PrefillBackendAdvertisement, PrefillDispatchRequest, PrefillDispatchResponse,
    PrefillRouterConfig, RawCalibrationPayload, RuntimeConfigSummary, analyze_calibration,
};
use parking_lot::{Mutex, RwLock};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::oneshot;
use velo::Handler;
use velo::transports::tcp::TcpTransportBuilder;

/// Completion signal sent by a Python lambda back to the Rust handler.
/// `Ok(None)` = dispatch path (no payload), `Ok(Some(json))` = calibrate
/// path (raw trace JSON), `Err(msg)` = lambda raised.
type CompletionResult = Result<Option<String>, String>;

/// Tokio oneshot wrapped for Python so a lambda's `_run` coroutine can
/// signal completion back into the Rust velo handler awaiting it.
///
/// Python sees `ok()`, `ok_with_payload(str)`, and `err(msg)`. The sender
/// is set by the Rust caller at construction time and consumed on the
/// first call; subsequent calls are silently swallowed.
#[pyclass]
pub struct CompletionEvent {
    tx: Mutex<Option<oneshot::Sender<CompletionResult>>>,
}

impl CompletionEvent {
    fn with_sender(tx: oneshot::Sender<CompletionResult>) -> Self {
        Self {
            tx: Mutex::new(Some(tx)),
        }
    }
}

#[pymethods]
impl CompletionEvent {
    /// Signal successful completion with no payload. Idempotent.
    /// Used by the dispatch lambda.
    fn ok(&self) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(Ok(None));
        }
    }

    /// Signal successful completion with a JSON payload string.
    /// Idempotent. Used by the calibrate lambda.
    fn ok_with_payload(&self, payload: String) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(Ok(Some(payload)));
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

/// Decrement `inflight_prefill` when this guard drops. Used so every
/// return path from the dispatch closure releases the counter.
struct InflightGuard(Arc<AtomicUsize>);
impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Release `calibrating` when this guard drops. Wrapped around the
/// calibrate handler body so any error / panic path resets the flag.
struct CalibratingGuard(Arc<AtomicBool>);
impl Drop for CalibratingGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
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
    /// - `lambda`: a `(req_dict, event) -> None` Python callable for the
    ///   prefill dispatch path. Invoked under a brief GIL hold for every
    ///   dispatched prefill request.
    /// - `hub_url`: the hub's discovery URL, e.g. `http://127.0.0.1:1337`.
    /// - `bind_addr`: optional `host:port` to bind the worker's velo TCP
    ///   transport to. Defaults to `0.0.0.0:0` (OS-assigned port).
    /// - `calibrate_lambda`: optional `(resolved_dict, event) -> None`
    ///   Python callable. When supplied, registers the calibrate handler
    ///   alongside dispatch.
    /// - `calibration_defaults`: optional Python dict captured from the
    ///   live framework (vLLM engine + tokenizer). Used by the calibrate
    ///   handler's resolver to fill None knobs and clamp out-of-range
    ///   overrides. Falls back to [`CalibrationDefaults::FALLBACK`] if
    ///   missing — fine for dispatch-only setups, marginal for real
    ///   calibration on large-context models.
    #[new]
    #[pyo3(signature = (
        lambda, hub_url, bind_addr=None,
        calibrate_lambda=None, calibration_defaults=None,
    ))]
    fn new(
        lambda: Py<PyAny>,
        hub_url: String,
        bind_addr: Option<String>,
        calibrate_lambda: Option<Py<PyAny>>,
        calibration_defaults: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let lambda = Arc::new(lambda);
        let calibrate_lambda = calibrate_lambda.map(Arc::new);
        let defaults = match calibration_defaults {
            Some(d) => pythonize::depythonize::<CalibrationDefaults>(d.as_any())
                .map_err(|e| PyRuntimeError::new_err(format!("calibration_defaults: {e}")))?,
            None => CalibrationDefaults::FALLBACK,
        };
        let defaults = Arc::new(defaults);

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
            .block_on(async move {
                build_inner(bind, hub_url, lambda, calibrate_lambda, defaults).await
            })
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
            if let Err(e) = self.runtime.block_on(async move { hub.unregister().await }) {
                tracing::warn!(error = %e, "PrefillRouterHandler: hub unregister on drop failed");
            }
        }
    }
}

async fn build_inner(
    bind_addr: String,
    hub_url: String,
    lambda: Arc<Py<PyAny>>,
    calibrate_lambda: Option<Arc<Py<PyAny>>>,
    calibration_defaults: Arc<CalibrationDefaults>,
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

    let calibrating = Arc::new(AtomicBool::new(false));
    let inflight_prefill = Arc::new(AtomicUsize::new(0));

    register_dispatch_handler(
        velo.messenger(),
        Arc::clone(&lambda),
        Arc::clone(&calibrating),
        Arc::clone(&inflight_prefill),
    )?;

    if let Some(cal_lambda) = calibrate_lambda {
        let cache: Arc<RwLock<Option<CalibrationSnapshot>>> = Arc::new(RwLock::new(None));
        register_calibrate_handler(
            velo.messenger(),
            cal_lambda,
            cache,
            Arc::clone(&calibration_defaults),
            Arc::clone(&calibrating),
            Arc::clone(&inflight_prefill),
        )?;
    }

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
    calibrating: Arc<AtomicBool>,
    inflight_prefill: Arc<AtomicUsize>,
) -> anyhow::Result<()> {
    let handler =
        Handler::typed_unary_async::<PrefillDispatchRequest, PrefillDispatchResponse, _, _>(
            PREFILL_DISPATCH_HANDLER,
            move |ctx| {
                let lambda = Arc::clone(&lambda);
                let calibrating = Arc::clone(&calibrating);
                let inflight = Arc::clone(&inflight_prefill);
                async move {
                    // Best-effort exclusion: a single Relaxed load on the
                    // hot path. Race window with calibrate is acceptable
                    // and documented — caller retries.
                    if calibrating.load(Ordering::Relaxed) {
                        return Ok(PrefillDispatchResponse {
                            ok: false,
                            error: Some("calibration_in_progress".into()),
                        });
                    }
                    inflight.fetch_add(1, Ordering::Relaxed);
                    let _guard = InflightGuard(Arc::clone(&inflight));

                    let (tx, rx) = oneshot::channel::<CompletionResult>();
                    let call_result: Result<(), String> = Python::attach(|py| {
                        let evt = CompletionEvent::with_sender(tx);
                        let py_evt = Py::new(py, evt)
                            .map_err(|e| format!("Py::new(CompletionEvent): {e}"))?;
                        let py_req = pythonize::pythonize(py, &ctx.input)
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

                    match rx.await {
                        Ok(Ok(_payload)) => Ok(PrefillDispatchResponse {
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
            },
        )
        .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("register {PREFILL_DISPATCH_HANDLER} handler: {e}"))?;
    Ok(())
}

fn register_calibrate_handler(
    messenger: &Arc<velo::Messenger>,
    lambda: Arc<Py<PyAny>>,
    cache: Arc<RwLock<Option<CalibrationSnapshot>>>,
    defaults: Arc<CalibrationDefaults>,
    calibrating: Arc<AtomicBool>,
    inflight_prefill: Arc<AtomicUsize>,
) -> anyhow::Result<()> {
    let handler =
        Handler::typed_unary_async::<CalibrationRequest, CalibrationResponse, _, _>(
            CALIBRATE_HANDLER,
            move |ctx| {
                let lambda = Arc::clone(&lambda);
                let cache = Arc::clone(&cache);
                let defaults = Arc::clone(&defaults);
                let calibrating = Arc::clone(&calibrating);
                let inflight = Arc::clone(&inflight_prefill);
                async move {
                    let request: CalibrationRequest = ctx.input;
                    let resolved = request.resolve(&defaults)?;

                    // Cache hit path — bypass all exclusion gates.
                    if !request.force
                        && let Some(snap) = cache.read().as_ref().cloned()
                    {
                        return Ok(CalibrationResponse {
                            results: snap.results,
                            from_cache: true,
                            resolved: snap.resolved,
                            defaults: snap.defaults,
                        });
                    }

                    // Best-effort CAS on the calibrating flag.
                    if calibrating
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                        .is_err()
                    {
                        anyhow::bail!("already_calibrating");
                    }
                    let _cal_guard = CalibratingGuard(Arc::clone(&calibrating));

                    // Wait up to ~250ms for in-flight prefills to drain.
                    for _ in 0..50 {
                        if inflight.load(Ordering::Acquire) == 0 {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    if inflight.load(Ordering::Acquire) != 0 {
                        anyhow::bail!("prefill_busy");
                    }

                    // Invoke the Python lambda with the *resolved* request.
                    let (tx, rx) = oneshot::channel::<CompletionResult>();
                    let call_result: Result<(), String> = Python::attach(|py| {
                        let evt = CompletionEvent::with_sender(tx);
                        let py_evt = Py::new(py, evt)
                            .map_err(|e| format!("Py::new(CompletionEvent): {e}"))?;
                        let py_req = pythonize::pythonize(py, &resolved)
                            .map_err(|e| format!("pythonize resolved request: {e}"))?;
                        lambda
                            .call1(py, (py_req, py_evt))
                            .map_err(|e| format!("calibrate lambda invocation raised: {e}"))?;
                        Ok(())
                    });
                    if let Err(msg) = call_result {
                        anyhow::bail!("calibrate lambda failed: {msg}");
                    }

                    let payload = match rx.await {
                        Ok(Ok(Some(s))) => s,
                        Ok(Ok(None)) => anyhow::bail!(
                            "calibrate lambda signaled ok() without payload — \
                             expected ok_with_payload(json)"
                        ),
                        Ok(Err(msg)) => anyhow::bail!("calibrate lambda errored: {msg}"),
                        Err(_) => anyhow::bail!("CompletionEvent dropped before signal"),
                    };

                    let raw: RawCalibrationPayload = serde_json::from_str(&payload)?;
                    let results = analyze_calibration(raw.traces)?;
                    let snap = CalibrationSnapshot {
                        results: results.clone(),
                        resolved: resolved.clone(),
                        defaults: (*defaults).clone(),
                    };
                    *cache.write() = Some(snap);

                    Ok(CalibrationResponse {
                        results,
                        from_cache: false,
                        resolved,
                        defaults: (*defaults).clone(),
                    })
                }
            },
        )
        .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("register {CALIBRATE_HANDLER} handler: {e}"))?;
    Ok(())
}
