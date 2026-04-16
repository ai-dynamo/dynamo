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

use dashmap::{DashMap, DashSet};
use futures::StreamExt;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::Endpoint;
use crate::to_pyerr;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryInstance, DiscoveryQuery};
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
/// msgspec.Struct (without `array_like=True`) encodes as a msgpack **map**:
///   `{"version": 1, "worker_id": "...", "dp_rank": 0, ...}`
///
/// We iterate through the map entries, read "worker_id" and "dp_rank",
/// and skip all other values.  Breaks early once both keys are found.
fn extract_fpm_key(data: &[u8]) -> Option<(String, i64)> {
    use rmp::decode::{read_int, read_map_len, read_str_len};

    let mut cursor = std::io::Cursor::new(data);

    let map_len = read_map_len(&mut cursor).ok()?;

    let mut worker_id: Option<String> = None;
    let mut dp_rank: Option<i64> = None;

    for _ in 0..map_len {
        // Read key (always a string in msgspec map encoding)
        let key_len = read_str_len(&mut cursor).ok()? as usize;
        let pos = cursor.position() as usize;
        if pos + key_len > data.len() {
            return None;
        }
        let key = std::str::from_utf8(&data[pos..pos + key_len]).ok()?;
        cursor.set_position((pos + key_len) as u64);

        match key {
            "worker_id" => {
                let str_len = read_str_len(&mut cursor).ok()? as usize;
                let pos = cursor.position() as usize;
                if pos + str_len > data.len() {
                    return None;
                }
                worker_id = Some(
                    std::str::from_utf8(&data[pos..pos + str_len])
                        .ok()?
                        .to_owned(),
                );
                cursor.set_position((pos + str_len) as u64);
            }
            "dp_rank" => {
                dp_rank = Some(read_int(&mut cursor).ok()?);
            }
            _ => {
                skip_msgpack_value(&mut cursor)?;
            }
        }

        if worker_id.is_some() && dp_rank.is_some() {
            break;
        }
    }

    Some((worker_id?, dp_rank?))
}

/// Advance the cursor past one msgpack value of any type.
///
/// Handles all msgpack formats needed for `ForwardPassMetrics` fields:
/// positive/negative fixint, uint/int 8-64, float 32/64, fixstr/str 8-32,
/// bool, nil, fixarray/array 16-32, fixmap/map 16-32, bin 8-32.
fn skip_msgpack_value(cursor: &mut std::io::Cursor<&[u8]>) -> Option<()> {
    use rmp::Marker;

    let marker = rmp::decode::read_marker(cursor).ok()?;
    match marker {
        // Integers
        Marker::FixPos(_) | Marker::FixNeg(_) => {}
        Marker::U8 | Marker::I8 => skip_bytes(cursor, 1)?,
        Marker::U16 | Marker::I16 => skip_bytes(cursor, 2)?,
        Marker::U32 | Marker::I32 | Marker::F32 => skip_bytes(cursor, 4)?,
        Marker::U64 | Marker::I64 | Marker::F64 => skip_bytes(cursor, 8)?,
        // Nil / Bool
        Marker::Null | Marker::True | Marker::False => {}
        // Strings
        Marker::FixStr(len) => skip_bytes(cursor, len as u64)?,
        Marker::Str8 => {
            let len = read_u8(cursor)? as u64;
            skip_bytes(cursor, len)?;
        }
        Marker::Str16 => {
            let len = read_u16(cursor)? as u64;
            skip_bytes(cursor, len)?;
        }
        Marker::Str32 => {
            let len = read_u32(cursor)? as u64;
            skip_bytes(cursor, len)?;
        }
        // Binary
        Marker::Bin8 => {
            let len = read_u8(cursor)? as u64;
            skip_bytes(cursor, len)?;
        }
        Marker::Bin16 => {
            let len = read_u16(cursor)? as u64;
            skip_bytes(cursor, len)?;
        }
        Marker::Bin32 => {
            let len = read_u32(cursor)? as u64;
            skip_bytes(cursor, len)?;
        }
        // Arrays (recurse to skip each element)
        Marker::FixArray(len) => {
            for _ in 0..len {
                skip_msgpack_value(cursor)?;
            }
        }
        Marker::Array16 => {
            let len = read_u16(cursor)?;
            for _ in 0..len {
                skip_msgpack_value(cursor)?;
            }
        }
        Marker::Array32 => {
            let len = read_u32(cursor)?;
            for _ in 0..len {
                skip_msgpack_value(cursor)?;
            }
        }
        // Maps (recurse to skip each key-value pair)
        Marker::FixMap(len) => {
            for _ in 0..len {
                skip_msgpack_value(cursor)?;
                skip_msgpack_value(cursor)?;
            }
        }
        Marker::Map16 => {
            let len = read_u16(cursor)?;
            for _ in 0..len {
                skip_msgpack_value(cursor)?;
                skip_msgpack_value(cursor)?;
            }
        }
        Marker::Map32 => {
            let len = read_u32(cursor)?;
            for _ in 0..len {
                skip_msgpack_value(cursor)?;
                skip_msgpack_value(cursor)?;
            }
        }
        // Ext types
        Marker::FixExt1 => skip_bytes(cursor, 2)?,
        Marker::FixExt2 => skip_bytes(cursor, 3)?,
        Marker::FixExt4 => skip_bytes(cursor, 5)?,
        Marker::FixExt8 => skip_bytes(cursor, 9)?,
        Marker::FixExt16 => skip_bytes(cursor, 17)?,
        Marker::Ext8 => {
            let len = read_u8(cursor)? as u64;
            skip_bytes(cursor, 1 + len)?;
        }
        Marker::Ext16 => {
            let len = read_u16(cursor)? as u64;
            skip_bytes(cursor, 1 + len)?;
        }
        Marker::Ext32 => {
            let len = read_u32(cursor)? as u64;
            skip_bytes(cursor, 1 + len)?;
        }
        Marker::Reserved => return None,
    }
    Some(())
}

fn skip_bytes(cursor: &mut std::io::Cursor<&[u8]>, n: u64) -> Option<()> {
    let new_pos = cursor.position().checked_add(n)?;
    if new_pos > cursor.get_ref().len() as u64 {
        return None;
    }
    cursor.set_position(new_pos);
    Some(())
}

fn read_u8(cursor: &mut std::io::Cursor<&[u8]>) -> Option<u8> {
    use std::io::Read;
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf).ok()?;
    Some(buf[0])
}

fn read_u16(cursor: &mut std::io::Cursor<&[u8]>) -> Option<u16> {
    use std::io::Read;
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf).ok()?;
    Some(u16::from_be_bytes(buf))
}

fn read_u32(cursor: &mut std::io::Cursor<&[u8]>) -> Option<u32> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf).ok()?;
    Some(u32::from_be_bytes(buf))
}

// ---------------------------------------------------------------------------
// Subscriber: event plane -> consumer
// ---------------------------------------------------------------------------

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
/// Both collections use `DashMap`/`DashSet` (sharded concurrent maps) so that
/// `get_recent_stats()` never blocks Task 1's high-frequency writes.  Per-shard
/// locking means readers and writers only contend if they happen to hit the same
/// shard, which is rare in practice.
///
/// Ghost entries (FPM arriving after its worker's MDC `Removed` event) are
/// filtered out by the `known_workers` check in `get_recent_stats()` and eagerly
/// pruned from `latest_stats` on `Removed` events.
#[pyclass]
pub(crate) struct FpmEventSubscriber {
    component: Component,
    cancel: CancellationToken,

    // recv mode state (lazily initialised on first recv() call)
    recv_started: Arc<AtomicBool>,
    rx: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>>,

    // tracking mode state
    tracking_started: Arc<AtomicBool>,
    latest_stats: Arc<DashMap<(String, i64), Vec<u8>>>,
    // Worker IDs currently registered in MDC.  Maintained by Task 2
    // (insert on Added, remove on Removed).  Used by get_recent_stats()
    // to filter out ghost entries without contending with Task 1's writes.
    known_workers: Arc<DashSet<String>>,
    // max_num_batched_tokens per worker_id, extracted from the model
    // deployment card's runtime_config on discovery Added events.
    // Exposed via get_max_num_batched_tokens() so the planner can read
    // the effective (minimum) value without a separate connector query.
    worker_max_batched_tokens: Arc<DashMap<String, u64>>,
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
            latest_stats: Arc::new(DashMap::new()),
            known_workers: Arc::new(DashSet::new()),
            worker_max_batched_tokens: Arc::new(DashMap::new()),
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
    ///    raw bytes in `latest_stats`.  Uses per-shard locking via `DashMap`
    ///    so contention with concurrent readers is minimal.
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
        // Inserts every FPM into latest_stats without checking known_workers.
        // Ghost entries (from workers that have already been removed) are
        // filtered out by get_recent_stats() at read time.  DashMap's
        // per-shard locking keeps contention low but does not eliminate it
        // entirely -- a concurrent reader hitting the same shard will briefly
        // wait for the insert to complete.
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
                                        stats.insert(key, payload);
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
        let max_bt = self.worker_max_batched_tokens.clone();
        rt.spawn({
            let cancel = cancel.clone();
            let component = component.clone();
            let stats = stats.clone();
            let known = known.clone();
            let max_bt = max_bt.clone();
            async move {
                let discovery = component.drt().discovery();
                let query = DiscoveryQuery::ComponentModels {
                    namespace: component.namespace().name(),
                    component: component.name().to_string(),
                };

                let stream = match discovery.list_and_watch(query, Some(cancel.clone())).await {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("FPM tracker: failed to create discovery watch: {e}");
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
                                    // Extract max_num_batched_tokens from the model card's runtime_config.
                                    if let DiscoveryInstance::Model { ref card_json, .. } = instance
                                        && let Some(val) = card_json
                                            .get("runtime_config")
                                            .and_then(|rc| rc.get("max_num_batched_tokens"))
                                            .and_then(|v| v.as_u64())
                                        && val > 0
                                    {
                                        max_bt.insert(wid.clone(), val);
                                    }
                                    known.insert(wid.clone());
                                    tracing::debug!("FPM tracker: worker {wid} added to known set");
                                }
                                Some(Ok(DiscoveryEvent::Removed(id))) => {
                                    let removed_id = id.instance_id().to_string();
                                    known.remove(&removed_id);
                                    max_bt.remove(&removed_id);

                                    // Eagerly prune latest_stats for the common case
                                    // (worker removed cleanly before any late FPMs arrive).
                                    let before = stats.len();
                                    stats.retain(|(worker_id, _), _| *worker_id != removed_id);
                                    let removed = before - stats.len();
                                    if removed > 0 {
                                        tracing::info!(
                                            "FPM tracker: removed {removed} entries for \
                                             worker_id={removed_id} (MDC removed)"
                                        );
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
    /// Uses `DashMap`/`DashSet` with per-shard locking so contention with
    /// the hot-path writer is minimal (but not zero -- a reader and writer
    /// hitting the same shard will briefly contend).
    ///
    /// Returns:
    ///     dict mapping `(worker_id: str, dp_rank: int)` to raw msgspec bytes.
    fn get_recent_stats(&self) -> PyResult<HashMap<(String, i64), Vec<u8>>> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }

        let snapshot = self
            .latest_stats
            .iter()
            .filter(|entry| self.known_workers.contains(&entry.key().0))
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        Ok(snapshot)
    }

    /// Return the effective `max_num_batched_tokens` across all known workers.
    ///
    /// The value is the **minimum** observed across workers so the planner
    /// uses the most constrained capacity, avoiding nondeterminism when
    /// workers report different values during heterogeneous rollouts.
    ///
    /// Returns:
    ///     The minimum `max_num_batched_tokens`, or `None` if no worker has reported one.
    fn get_max_num_batched_tokens(&self) -> PyResult<Option<u64>> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }

        let min_val = self
            .worker_max_batched_tokens
            .iter()
            .filter(|entry| self.known_workers.contains(entry.key()))
            .map(|entry| *entry.value())
            .min();

        Ok(min_val)
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
