// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Forward Pass Metrics (FPM = ForwardPassMetrics) event plane integration.
//!
//! - `FpmEventRelay`: thin wrapper around `dynamo_llm::fpm_publisher::FpmEventRelay`
//! - `FpmEventSubscriber`: wraps `EventSubscriber::for_component` for the consumer side.
//!   Supports two mutually exclusive modes:
//!   - **recv mode**: call `recv()` to pull one message at a time (existing behaviour).
//!   - **tracking mode**: call `start_tracking()` once, then `get_recent_stats()` to
//!     retrieve the latest FPM bytes keyed by `(worker_id, dp_rank)`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::to_pyerr;
use crate::Endpoint;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryQuery};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;

const FPM_TOPIC: &str = "forward-pass-metrics";

// ---------------------------------------------------------------------------
// Relay: raw ZMQ (child process) -> event plane
// ---------------------------------------------------------------------------

/// Relay that bridges ForwardPassMetrics from a local raw ZMQ PUB socket
/// (InstrumentedScheduler in EngineCore child process) to the Dynamo event
/// plane with automatic discovery registration.
#[pyclass]
pub(crate) struct FpmEventRelay {
    inner: llm_rs::fpm_publisher::FpmEventRelay,
}

#[pymethods]
impl FpmEventRelay {
    /// Create a relay that bridges raw ZMQ to the event plane.
    ///
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    ///     zmq_endpoint: Local ZMQ PUB address to subscribe to
    ///         (e.g., "tcp://127.0.0.1:20380").
    #[new]
    #[pyo3(signature = (endpoint, zmq_endpoint))]
    fn new(endpoint: Endpoint, zmq_endpoint: String) -> PyResult<Self> {
        let component = endpoint.inner.component().clone();
        let inner =
            llm_rs::fpm_publisher::FpmEventRelay::new(component, zmq_endpoint).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Shut down the relay task.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Helpers: partial msgpack decode
// ---------------------------------------------------------------------------

/// Extract `(worker_id, dp_rank)` from a msgspec-encoded `ForwardPassMetrics`.
///
/// msgspec.Struct with positional encoding produces a msgpack **array**:
///   `[worker_id: str, dp_rank: int, wall_time: float, ...]`
///
/// We only decode the first two elements to build the tracking key.
fn extract_fpm_key(data: &[u8]) -> Option<(String, i64)> {
    use rmp::decode::{read_array_len, read_int, read_str_len};

    let mut cursor = std::io::Cursor::new(data);

    let arr_len = read_array_len(&mut cursor).ok()?;
    if arr_len < 2 {
        return None;
    }

    // Index 0: worker_id (str)
    let str_len = read_str_len(&mut cursor).ok()? as usize;
    let pos = cursor.position() as usize;
    if pos + str_len > data.len() {
        return None;
    }
    let worker_id = std::str::from_utf8(&data[pos..pos + str_len]).ok()?.to_owned();
    cursor.set_position((pos + str_len) as u64);

    // Index 1: dp_rank (int)
    let dp_rank: i64 = read_int(&mut cursor).ok()?;

    Some((worker_id, dp_rank))
}

// ---------------------------------------------------------------------------
// Subscriber: event plane -> consumer
// ---------------------------------------------------------------------------

type StatsMap = HashMap<(String, i64), Vec<u8>>;

/// Subscriber for ForwardPassMetrics from the event plane.
///
/// Auto-discovers engine publishers via the discovery plane (K8s CRD / etcd / file).
///
/// Two mutually exclusive usage modes:
///
/// 1. **recv mode** (default): call `recv()` to pull individual messages.
/// 2. **tracking mode**: call `start_tracking()` once, then poll `get_recent_stats()`
///    to retrieve the latest FPM bytes keyed by `(worker_id, dp_rank)`.
///    Stale entries are cleaned up via MDC discovery watch and TTL.
#[pyclass]
pub(crate) struct FpmEventSubscriber {
    component: Component,
    cancel: CancellationToken,

    // recv mode state (lazily initialised on first recv() call)
    recv_started: Arc<AtomicBool>,
    rx: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>>,

    // tracking mode state
    tracking_started: Arc<AtomicBool>,
    latest_stats: Arc<std::sync::RwLock<StatsMap>>,
}

#[pymethods]
impl FpmEventSubscriber {
    /// Create a subscriber that auto-discovers FPM publishers.
    ///
    /// No background tasks are started until `recv()` or `start_tracking()` is called.
    ///
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    #[new]
    #[pyo3(signature = (endpoint,))]
    fn new(endpoint: Endpoint) -> PyResult<Self> {
        let component = endpoint.inner.component().clone();
        Ok(Self {
            component,
            cancel: CancellationToken::new(),
            recv_started: Arc::new(AtomicBool::new(false)),
            rx: Arc::new(std::sync::Mutex::new(None)),
            tracking_started: Arc::new(AtomicBool::new(false)),
            latest_stats: Arc::new(std::sync::RwLock::new(HashMap::new())),
        })
    }

    /// Blocking receive of next message bytes. Releases the GIL while waiting.
    ///
    /// On the first call a background subscriber task is spawned (recv mode).
    /// Cannot be used after `start_tracking()`.
    ///
    /// Returns the raw msgspec payload, or None if the stream is closed.
    fn recv(&self, py: Python) -> PyResult<Option<Vec<u8>>> {
        if self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "Cannot call recv() after start_tracking()",
            ));
        }

        // Lazily start the recv-mode subscriber task on the first call.
        if !self.recv_started.swap(true, Ordering::SeqCst) {
            let component = self.component.clone();
            let cancel = self.cancel.clone();
            let (tx, rx_new) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();

            {
                let mut guard = self.rx.lock().map_err(|e| to_pyerr(format!("{e}")))?;
                *guard = Some(rx_new);
            }

            let rt = component.drt().runtime().secondary();
            rt.spawn(async move {
                let mut subscriber =
                    match EventSubscriber::for_component(&component, FPM_TOPIC).await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("FPM subscriber (recv): failed to create: {e}");
                            return;
                        }
                    };

                tracing::info!("FPM subscriber (recv): listening for forward-pass-metrics events");

                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            tracing::info!("FPM subscriber (recv): shutting down");
                            break;
                        }
                        event = subscriber.next() => {
                            match event {
                                Some(Ok(envelope)) => {
                                    if tx.send(envelope.payload.to_vec()).is_err() {
                                        tracing::info!(
                                            "FPM subscriber (recv): receiver dropped, exiting"
                                        );
                                        break;
                                    }
                                }
                                Some(Err(e)) => {
                                    tracing::warn!("FPM subscriber (recv): event error: {e}");
                                }
                                None => {
                                    tracing::info!("FPM subscriber (recv): stream ended");
                                    break;
                                }
                            }
                        }
                    }
                }
            });
        }

        let rx = self.rx.clone();
        py.allow_threads(move || {
            let mut guard = rx
                .lock()
                .map_err(|e| to_pyerr(format!("lock poisoned: {e}")))?;
            match guard.as_mut() {
                Some(rx) => Ok(rx.blocking_recv()),
                None => Ok(None),
            }
        })
    }

    /// Start background tracking of the latest FPM per `(worker_id, dp_rank)`.
    ///
    /// Spawns two background tasks:
    /// 1. **Event consumption**: subscribes to FPM events, extracts the composite
    ///    key `(worker_id, dp_rank)` from the msgpack payload, and stores the
    ///    latest raw bytes in an internal map.
    /// 2. **MDC discovery watch**: monitors `ComponentModels` for the target
    ///    component.  When a model is removed (engine scaled down / died), all
    ///    entries whose `worker_id == str(removed_instance_id)` are purged.
    ///
    /// After calling this method, `recv()` will raise an error.
    fn start_tracking(&self) -> PyResult<()> {
        if self.recv_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "Cannot call start_tracking() after recv()",
            ));
        }
        if self.tracking_started.swap(true, Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err("Tracking already started"));
        }

        let component = self.component.clone();
        let rt = component.drt().runtime().secondary();
        let cancel = self.cancel.clone();
        let stats = self.latest_stats.clone();

        // Task 1: event consumption
        rt.spawn({
            let cancel = cancel.clone();
            let component = component.clone();
            let stats = stats.clone();
            async move {
                let mut subscriber =
                    match EventSubscriber::for_component(&component, FPM_TOPIC).await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("FPM tracker: failed to create subscriber: {e}");
                            return;
                        }
                    };

                tracing::info!("FPM tracker: listening for forward-pass-metrics events");

                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            tracing::info!("FPM tracker: shutting down event task");
                            break;
                        }
                        event = subscriber.next() => {
                            match event {
                                Some(Ok(envelope)) => {
                                    let payload = envelope.payload.to_vec();
                                    if let Some(key) = extract_fpm_key(&payload) {
                                        if let Ok(mut map) = stats.write() {
                                            map.insert(key, payload);
                                        }
                                    } else {
                                        tracing::warn!(
                                            "FPM tracker: failed to extract key from payload ({} bytes)",
                                            envelope.payload.len()
                                        );
                                    }
                                }
                                Some(Err(e)) => {
                                    tracing::warn!("FPM tracker: event error: {e}");
                                }
                                None => {
                                    tracing::info!("FPM tracker: event stream ended");
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });

        // Task 2: MDC discovery watch for cleanup
        rt.spawn({
            let cancel = cancel.clone();
            let component = component.clone();
            let stats = stats.clone();
            async move {
                let discovery = component.drt().discovery();
                let query = DiscoveryQuery::ComponentModels {
                    namespace: component.namespace().name(),
                    component: component.name().to_string(),
                };

                let stream = match discovery
                    .list_and_watch(query, Some(cancel.clone()))
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!(
                            "FPM tracker: failed to create discovery watch: {e}"
                        );
                        return;
                    }
                };

                tracing::info!("FPM tracker: watching MDC discovery for engine lifecycle");

                let mut stream = stream;
                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            tracing::info!("FPM tracker: shutting down discovery task");
                            break;
                        }
                        event = stream.next() => {
                            match event {
                                Some(Ok(DiscoveryEvent::Removed(id))) => {
                                    let removed_id = id.instance_id().to_string();
                                    if let Ok(mut map) = stats.write() {
                                        let before = map.len();
                                        map.retain(|(worker_id, _), _| *worker_id != removed_id);
                                        let removed = before - map.len();
                                        if removed > 0 {
                                            tracing::info!(
                                                "FPM tracker: removed {removed} entries for \
                                                 worker_id={removed_id} (MDC removed)"
                                            );
                                        }
                                    }
                                }
                                Some(Ok(DiscoveryEvent::Added(_))) => {
                                    // Engine appeared; stats will be populated by the event task.
                                }
                                Some(Err(e)) => {
                                    tracing::warn!("FPM tracker: discovery error: {e}");
                                }
                                None => {
                                    tracing::info!("FPM tracker: discovery stream ended");
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Return the latest FPM bytes for every tracked `(worker_id, dp_rank)`.
    ///
    /// Cleanup of removed engines is handled by the MDC discovery watch task
    /// (spawned by `start_tracking()`).
    ///
    /// Returns:
    ///     dict mapping `(worker_id: str, dp_rank: int)` to raw msgspec bytes.
    fn get_recent_stats(&self) -> PyResult<HashMap<(String, i64), Vec<u8>>> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }

        let map = self
            .latest_stats
            .read()
            .map_err(|e| to_pyerr(format!("lock poisoned: {e}")))?;

        Ok(map.clone())
    }

    /// Shut down the subscriber (all background tasks).
    fn shutdown(&self) {
        self.cancel.cancel();
    }
}

impl Drop for FpmEventSubscriber {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
