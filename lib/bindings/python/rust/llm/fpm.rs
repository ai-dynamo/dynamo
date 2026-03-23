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

use std::collections::{HashMap, HashSet};
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
///   `[version: int, worker_id: str, dp_rank: int, counter_id: int, ...]`
///
/// We skip the version field and decode worker_id (index 1) and dp_rank (index 2).
fn extract_fpm_key(data: &[u8]) -> Option<(String, i64)> {
    use rmp::decode::{read_array_len, read_int, read_str_len};

    let mut cursor = std::io::Cursor::new(data);

    let arr_len = read_array_len(&mut cursor).ok()?;
    if arr_len < 3 {
        return None;
    }

    // Index 0: version (int) -- skip
    let _version: i64 = read_int(&mut cursor).ok()?;

    // Index 1: worker_id (str)
    let str_len = read_str_len(&mut cursor).ok()? as usize;
    let pos = cursor.position() as usize;
    if pos + str_len > data.len() {
        return None;
    }
    let worker_id = std::str::from_utf8(&data[pos..pos + str_len]).ok()?.to_owned();
    cursor.set_position((pos + str_len) as u64);

    // Index 2: dp_rank (int)
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
///
/// # Tracking mode concurrency design
///
/// Three concurrent actors access shared state:
///
/// - **Task 1** (event consumption, tokio): writes to `latest_stats` on every FPM.
/// - **Task 2** (MDC discovery watch, tokio): maintains `known_workers` set and
///   removes dead-worker entries from `latest_stats` on `Removed` events.
/// - **`get_recent_stats()`** (Python thread): reads both `latest_stats` and
///   `known_workers` to produce a filtered snapshot.
///
/// `get_recent_stats()` uses **read locks only** on both collections.  This is
/// deliberate: Task 1 is the hot path (~20k writes/s at 200 engines) and must
/// never be blocked by the planner's polling.  Read locks coexist with other
/// readers and only block on writers, so `get_recent_stats()` never contends
/// with Task 1's writes to `latest_stats`.
///
/// Ghost entries (FPM arriving after its worker's MDC `Removed` event) are
/// filtered out by the `known_workers` check in `get_recent_stats()` but not
/// pruned from `latest_stats`.  This avoids a write lock and the memory is
/// negligible (bounded by in-flight FPMs at the instant of worker death).
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
    // Worker IDs currently registered in MDC.  Maintained by Task 2
    // (insert on Added, remove on Removed).  Used by get_recent_stats()
    // to filter out ghost entries without taking a write lock on latest_stats.
    known_workers: Arc<std::sync::RwLock<HashSet<String>>>,
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
            known_workers: Arc::new(std::sync::RwLock::new(HashSet::new())),
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
    ///
    /// 1. **Event consumption** (Task 1): subscribes to FPM events, extracts
    ///    `(worker_id, dp_rank)` from the msgpack payload, stores the latest
    ///    raw bytes in `latest_stats`.  This is the hot path and only takes a
    ///    write lock on `latest_stats`.
    ///
    /// 2. **MDC discovery watch** (Task 2): monitors `ComponentModels` for the
    ///    target component.  Maintains `known_workers` (the set of currently
    ///    alive worker IDs) and eagerly removes dead-worker entries from
    ///    `latest_stats` on `Removed` events.
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
        let known = self.known_workers.clone();

        // Task 1: event consumption.
        //
        // Blindly inserts every FPM into latest_stats without checking
        // known_workers.  This avoids extra lock contention on the hot path.
        // Ghost entries (from workers that have already been removed) are
        // filtered out by get_recent_stats() at read time.
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

        // Task 2: MDC discovery watch.
        //
        // Maintains known_workers (insert on Added, remove on Removed) and
        // eagerly prunes latest_stats on Removed events.  This handles the
        // normal scale-down path.  Any ghost entries created by the race
        // condition (FPM arriving *after* the Removed event) are caught by the
        // known_workers filter in get_recent_stats().
        rt.spawn({
            let cancel = cancel.clone();
            let component = component.clone();
            let stats = stats.clone();
            let known = known.clone();
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
                                Some(Ok(DiscoveryEvent::Added(instance))) => {
                                    let wid = instance.instance_id().to_string();
                                    if let Ok(mut set) = known.write() {
                                        set.insert(wid.clone());
                                    }
                                    tracing::debug!("FPM tracker: worker {wid} added to known set");
                                }
                                Some(Ok(DiscoveryEvent::Removed(id))) => {
                                    let removed_id = id.instance_id().to_string();

                                    if let Ok(mut set) = known.write() {
                                        set.remove(&removed_id);
                                    }

                                    // Eagerly prune latest_stats for the common case
                                    // (worker removed cleanly before any late FPMs arrive).
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
    /// The returned snapshot is filtered against `known_workers` so that
    /// ghost entries (late FPMs from already-removed workers) are excluded.
    /// Both `latest_stats` and `known_workers` are accessed via **read locks**
    /// to avoid contending with Task 1's high-frequency writes.  Ghost entries
    /// remain in `latest_stats` memory but are never returned to Python; they
    /// are bounded by the number of in-flight FPMs at the instant of worker
    /// death (typically 1-2 per engine) and are harmless.
    ///
    /// Returns:
    ///     dict mapping `(worker_id: str, dp_rank: int)` to raw msgspec bytes.
    fn get_recent_stats(&self) -> PyResult<HashMap<(String, i64), Vec<u8>>> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }

        let known = self
            .known_workers
            .read()
            .map_err(|e| to_pyerr(format!("lock poisoned: {e}")))?;

        let stats = self
            .latest_stats
            .read()
            .map_err(|e| to_pyerr(format!("lock poisoned: {e}")))?;

        let snapshot = stats
            .iter()
            .filter(|((worker_id, _), _)| known.contains(worker_id))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Ok(snapshot)
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
