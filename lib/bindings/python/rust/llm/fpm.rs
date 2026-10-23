// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Forward Pass Metrics (FPM = ForwardPassMetrics) event plane integration.
//!
//! - `FpmEventRelay`: thin wrapper around `dynamo_llm::fpm_publisher::FpmEventRelay`
//!   (used by the vLLM adapter — ZMQ bridge from EngineCore child process).
//! - `FpmDirectPublisher`: thin wrapper around `dynamo_llm::fpm_publisher::FpmDirectPublisher`
//!   (used by the TRT-LLM adapter — direct event-plane publish, no ZMQ hop).
//! - `FpmEventSubscriber`: wraps `EventSubscriber::for_endpoint` for the consumer side.
//!   Supports two mutually exclusive modes:
//!   - **recv mode**: call `recv()` to pull one message at a time (existing behaviour).
//!   - **tracking mode**: call `start_tracking()` once, then `get_recent_stats()` to
//!     retrieve the latest FPM bytes keyed by `(worker_id, dp_rank)`.

use dashmap::DashMap;
use dynamo_mocker::common::protocols::{ForwardPassSnapshot, FpmPublisher};
use futures::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::Endpoint;
use crate::to_pyerr;
use dynamo_runtime::component::Endpoint as RuntimeEndpoint;
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, ModelCardInstanceId,
    ModelTaintsUpdate,
};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use llm_rs::local_model::runtime_config::ModelRuntimeConfig;

const FPM_TOPIC: &str = "forward-pass-metrics";
const DEFAULT_FPM_MAX_AGE: Duration = Duration::from_secs(15);
const FPM_CLEANUP_INTERVAL: Duration = Duration::from_secs(1);

struct TrackedFpm {
    payload: Vec<u8>,
    received_at: Instant,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum FpmWorkerLifecycle {
    #[default]
    Unjoined,
    Active {
        registered_at: Instant,
    },
    Removed {
        last_activity: Instant,
    },
}

#[derive(Default)]
struct WorkerFpm {
    lifecycle: FpmWorkerLifecycle,
    ranks: HashMap<i64, TrackedFpm>,
    model_cards: WorkerModelCards,
}

impl WorkerFpm {
    fn activate(&mut self, registered_at: Instant) {
        if !matches!(self.lifecycle, FpmWorkerLifecycle::Active { .. }) {
            self.ranks.clear();
            self.lifecycle = FpmWorkerLifecycle::Active { registered_at };
        }
    }

    fn remove(&mut self, removed_at: Instant) {
        self.lifecycle = FpmWorkerLifecycle::Removed {
            last_activity: removed_at,
        };
        self.ranks.clear();
    }

    fn reconcile_ranks(&mut self, previous: Option<Range<u32>>) {
        let expected = self.model_cards.selected_ranks();
        if expected == previous.as_ref() {
            return;
        }
        // Retain already-invalid ranks as evidence of a bad publisher.
        self.ranks.retain(|rank, _| {
            let Ok(rank) = u32::try_from(*rank) else {
                return true;
            };
            !previous.as_ref().is_some_and(|ranks| ranks.contains(&rank))
                || expected.is_some_and(|ranks| ranks.contains(&rank))
        });
    }
}

type FpmSnapshot = HashMap<(String, i64), Vec<u8>>;

/// Fresh samples and completeness diagnostics for the subscribed worker pool.
#[pyclass(get_all)]
#[derive(Default)]
pub(crate) struct FpmHealthSnapshot {
    snapshot: FpmSnapshot,
    unjoined_worker_ids: Vec<String>,
    stale_rank_count: usize,
    missing_rank_count: usize,
}

fn record_fpm(
    workers: &DashMap<String, WorkerFpm>,
    (worker_id, rank): (String, i64),
    payload: Vec<u8>,
    received_at: Instant,
) {
    let mut worker = workers.entry(worker_id).or_default();
    match &mut worker.lifecycle {
        FpmWorkerLifecycle::Active { registered_at } if received_at < *registered_at => return,
        FpmWorkerLifecycle::Removed { last_activity } => {
            *last_activity = (*last_activity).max(received_at);
            return;
        }
        _ => {}
    }
    worker.ranks.insert(
        rank,
        TrackedFpm {
            payload,
            received_at,
        },
    );
}

fn prune_inactive_workers(workers: &DashMap<String, WorkerFpm>, now: Instant, max_age: Duration) {
    workers.retain(|_, worker| match worker.lifecycle {
        FpmWorkerLifecycle::Active { .. } => true,
        FpmWorkerLifecycle::Removed { last_activity } => {
            now.saturating_duration_since(last_activity) <= max_age
        }
        FpmWorkerLifecycle::Unjoined => {
            worker
                .ranks
                .retain(|_, sample| now.saturating_duration_since(sample.received_at) <= max_age);
            !worker.ranks.is_empty()
        }
    });
}

fn fresh_fpm_snapshot(
    workers: &DashMap<String, WorkerFpm>,
    now: Instant,
    max_age: Duration,
) -> FpmHealthSnapshot {
    let mut health = FpmHealthSnapshot::default();
    for worker in workers {
        if matches!(worker.lifecycle, FpmWorkerLifecycle::Removed { .. }) {
            continue;
        }
        let expected = worker.model_cards.selected_ranks();
        let mut fresh_count = 0;
        let mut unjoined = false;
        for (&rank, sample) in &worker.ranks {
            let expired = now.saturating_duration_since(sample.received_at) > max_age;
            let rank_is_expected = expected
                .is_some_and(|ranks| u32::try_from(rank).is_ok_and(|rank| ranks.contains(&rank)));
            if !rank_is_expected {
                unjoined |= !expired;
            } else if expired {
                health.stale_rank_count += 1;
            } else {
                health
                    .snapshot
                    .insert((worker.key().clone(), rank), sample.payload.clone());
                fresh_count += 1;
            }
        }
        if unjoined {
            health.unjoined_worker_ids.push(worker.key().clone());
        }
        if matches!(worker.lifecycle, FpmWorkerLifecycle::Active { .. }) {
            health.missing_rank_count +=
                expected.map_or(1, |ranks| ranks.len().saturating_sub(fresh_count));
        }
    }
    health.unjoined_worker_ids.sort_unstable();
    health
}

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
    fn new(py: Python<'_>, endpoint: Endpoint, zmq_endpoint: String) -> PyResult<Self> {
        // Don't hold the GIL while blocking on the runtime; its tasks may need it.
        let inner = py
            .allow_threads(move || {
                llm_rs::fpm_publisher::FpmEventRelay::new(endpoint.inner, zmq_endpoint)
            })
            .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Shut down the relay task.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Direct publisher: TRT-LLM adapter -> event plane (no ZMQ hop)
// ---------------------------------------------------------------------------

/// Direct Forward Pass Metrics publisher for in-process producers such as
/// the TRT-LLM adapter. The underlying Rust `FpmDirectPublisher` owns per-DP
/// serialization tasks (each with its own 1s idle heartbeat timer) and a
/// single event-plane publisher task. Python callers do not need to manage
/// heartbeat — the Rust side emits a zeroed snapshot when no data arrives
/// for `IDLE_HEARTBEAT_INTERVAL` (matches vLLM's `HEARTBEAT_INTERVAL = 1.0`).
#[pyclass]
pub(crate) struct FpmDirectPublisher {
    // Owns the CancellationToken that drives Drop-based shutdown of all
    // per-rank serialization tasks and the event-plane publisher task. Held
    // by reference only — keep the field non-underscored so a future cleanup
    // pass does not delete it as "dead", which would leak every spawned task.
    inner: llm_rs::fpm_publisher::FpmDirectPublisher,
    publishers: Vec<FpmPublisher>,
}

#[pymethods]
impl FpmDirectPublisher {
    /// Construct a publisher that owns `dp_size` per-DP-rank channels.
    ///
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    ///     worker_id: Unique worker identifier stamped on every emitted FPM.
    ///     dp_size: Number of DP ranks to allocate handles for; use 1 when
    ///         attention DP is disabled.
    #[new]
    #[pyo3(signature = (endpoint, worker_id, dp_size=1))]
    fn new(py: Python<'_>, endpoint: Endpoint, worker_id: String, dp_size: u32) -> PyResult<Self> {
        let rt = endpoint.inner.component().drt().runtime().secondary();
        // Don't hold the GIL while blocking on the runtime; its tasks may need it.
        let (inner, publishers) = py
            .allow_threads(move || {
                rt.block_on(llm_rs::fpm_publisher::FpmDirectPublisher::new(
                    endpoint.inner,
                    worker_id,
                    dp_size,
                ))
            })
            .map_err(to_pyerr)?;
        Ok(Self { inner, publishers })
    }

    /// Publish one iteration's FPM snapshot for the given DP rank.
    ///
    /// All parameters are keyword-only on the Python side — adjacent ints
    /// with similar units (`scheduled_*` vs `queued_*`, `*_prefill_*` vs
    /// `*_decode_*`) cannot be distinguished by the type system, so a
    /// transposition would silently corrupt every published snapshot.
    /// Forcing kwargs at the boundary is a zero-cost guard.
    ///
    /// Variance fields (`var_prefill_length`, `var_decode_kv_tokens`,
    /// `var_queued_prefill_length`, `var_queued_decode_kv_tokens`) are
    /// defaulted to 0.0 per the MVP scope — the active planner does not
    /// consume them on origin/main. A follow-up PR can add Welford-based
    /// variance computation in TRT-LLM's PyExecutor and a new overload here.
    #[pyo3(signature = (
        *,
        dp_rank,
        scheduled_num_prefill_requests,
        scheduled_sum_prefill_tokens,
        scheduled_sum_prefill_kv_tokens,
        scheduled_num_decode_requests,
        scheduled_sum_decode_kv_tokens,
        queued_num_prefill_requests,
        queued_sum_prefill_tokens,
        queued_num_decode_requests,
        queued_sum_decode_kv_tokens,
        wall_time_secs,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn publish(
        &self,
        dp_rank: u32,
        scheduled_num_prefill_requests: u32,
        scheduled_sum_prefill_tokens: u64,
        scheduled_sum_prefill_kv_tokens: u64,
        scheduled_num_decode_requests: u32,
        scheduled_sum_decode_kv_tokens: u64,
        queued_num_prefill_requests: u32,
        queued_sum_prefill_tokens: u64,
        queued_num_decode_requests: u32,
        queued_sum_decode_kv_tokens: u64,
        wall_time_secs: f64,
    ) -> PyResult<()> {
        let idx = dp_rank as usize;
        if idx >= self.publishers.len() {
            return Err(PyRuntimeError::new_err(format!(
                "dp_rank {dp_rank} out of range; FpmDirectPublisher was constructed with dp_size={}",
                self.publishers.len()
            )));
        }
        let snapshot = ForwardPassSnapshot {
            num_prefill_requests: scheduled_num_prefill_requests,
            sum_prefill_tokens: scheduled_sum_prefill_tokens,
            var_prefill_length: 0.0,
            sum_prefill_kv_tokens: scheduled_sum_prefill_kv_tokens,
            num_decode_requests: scheduled_num_decode_requests,
            sum_decode_kv_tokens: scheduled_sum_decode_kv_tokens,
            var_decode_kv_tokens: 0.0,
            num_queued_prefill: queued_num_prefill_requests,
            sum_queued_prefill_tokens: queued_sum_prefill_tokens,
            var_queued_prefill_length: 0.0,
            num_queued_decode: queued_num_decode_requests,
            sum_queued_decode_kv_tokens: queued_sum_decode_kv_tokens,
            var_queued_decode_kv_tokens: 0.0,
            wall_time_secs,
            ..Default::default()
        };
        self.publishers[idx].publish(snapshot).map_err(to_pyerr)
    }

    /// Shut down the publisher and its per-rank serialization tasks.
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
/// Lifecycle, model cards, and rank samples share one worker entry, so registration,
/// removal, event writes, and snapshots never acquire multiple map locks.
#[pyclass]
pub(crate) struct FpmEventSubscriber {
    endpoint: RuntimeEndpoint,
    cancel: CancellationToken,
    recv_started: Arc<AtomicBool>,
    rx: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>>,
    tracking_started: Arc<AtomicBool>,
    workers: Arc<DashMap<String, WorkerFpm>>,
    max_age: Duration,
}

#[derive(Default)]
struct WorkerModelCards {
    cards: HashMap<ModelCardInstanceId, IndexedModelCard>,
    card_ids: Vec<ModelCardInstanceId>,
    base_card_id: Option<ModelCardInstanceId>,
}

struct IndexedModelCard {
    card: String,
    index: usize,
    dp_ranks: Option<Range<u32>>,
}

fn expected_fpm_ranks(card: &str) -> anyhow::Result<Range<u32>> {
    let mut card: serde_json::Map<String, serde_json::Value> = serde_json::from_str(card)?;
    let config: ModelRuntimeConfig = match card.get_mut("runtime_config") {
        Some(config) => serde_json::from_value(config.take())?,
        None => ModelRuntimeConfig::default(),
    };
    config
        .data_parallel_rank_range()
        .map_err(anyhow::Error::msg)
}

impl WorkerModelCards {
    fn insert(&mut self, card_id: ModelCardInstanceId, card: String) {
        let dp_ranks = match expected_fpm_ranks(&card) {
            Ok(ranks) => Some(ranks),
            Err(error) => {
                tracing::warn!(worker_id = card_id.instance_id, error = %error,
                    "FPM tracker: invalid model-card runtime configuration");
                None
            }
        };
        if let Some(existing) = self.cards.get_mut(&card_id) {
            existing.card = card;
            existing.dp_ranks = dp_ranks;
            return;
        }

        if card_id.model_suffix.is_none() {
            self.base_card_id = Some(card_id.clone());
        }
        let index = self.card_ids.len();
        self.card_ids.push(card_id.clone());
        self.cards.insert(
            card_id,
            IndexedModelCard {
                card,
                index,
                dp_ranks,
            },
        );
    }

    fn remove(&mut self, card_id: &ModelCardInstanceId) {
        let Some(removed) = self.cards.remove(card_id) else {
            return;
        };

        let last_id = self
            .card_ids
            .pop()
            .expect("the card index must contain every stored card");
        if removed.index < self.card_ids.len() {
            self.card_ids[removed.index] = last_id.clone();
            self.cards
                .get_mut(&last_id)
                .expect("the swapped card must remain stored")
                .index = removed.index;
        }

        if self.base_card_id.as_ref() == Some(card_id) {
            self.base_card_id = None;
        }
    }

    fn selected_card(&self) -> Option<&str> {
        self.selected_entry().map(|entry| entry.card.as_str())
    }

    fn selected_ranks(&self) -> Option<&Range<u32>> {
        self.selected_entry()?.dp_ranks.as_ref()
    }

    fn selected_entry(&self) -> Option<&IndexedModelCard> {
        let card_id = self
            .base_card_id
            .as_ref()
            .or_else(|| self.card_ids.first())?;
        self.cards.get(card_id)
    }

    fn is_empty(&self) -> bool {
        self.card_ids.is_empty()
    }
}

fn insert_worker_model_card(
    workers: &DashMap<String, WorkerFpm>,
    worker_id: String,
    card_id: ModelCardInstanceId,
    card: String,
) {
    let mut worker = workers.entry(worker_id).or_default();
    worker.activate(Instant::now());
    let previous = worker.model_cards.selected_ranks().cloned();
    worker.model_cards.insert(card_id, card);
    worker.reconcile_ranks(previous);
}

fn update_worker_model_card_taints(
    workers: &DashMap<String, WorkerFpm>,
    update: &ModelTaintsUpdate,
) -> anyhow::Result<bool> {
    let worker_id = update.id.instance_id.to_string();
    let Some(mut worker) = workers.get_mut(&worker_id) else {
        return Ok(false);
    };
    let Some(indexed) = worker.model_cards.cards.get_mut(&update.id) else {
        return Ok(false);
    };
    let mut card: serde_json::Value = serde_json::from_str(&indexed.card)?;
    let runtime_config = card
        .get_mut("runtime_config")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("model card is missing runtime_config"))?;
    let taints = serde_json::Value::Array(
        update
            .taints
            .iter()
            .cloned()
            .map(serde_json::Value::String)
            .collect(),
    );
    if runtime_config.get("taints") == Some(&taints) {
        return Ok(false);
    }
    runtime_config.insert("taints".to_string(), taints);
    indexed.card = serde_json::to_string(&card)?;
    Ok(true)
}

fn remove_worker_model_card(
    workers: &DashMap<String, WorkerFpm>,
    worker_id: &str,
    card_id: &ModelCardInstanceId,
) -> bool {
    let mut worker = workers.entry(worker_id.to_string()).or_default();
    let previous = worker.model_cards.selected_ranks().cloned();
    worker.model_cards.remove(card_id);
    if !worker.model_cards.is_empty() {
        worker.reconcile_ranks(previous);
        return true;
    }
    worker.remove(Instant::now());
    false
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
        Ok(Self {
            endpoint: endpoint.inner,
            cancel: CancellationToken::new(),
            recv_started: Arc::new(AtomicBool::new(false)),
            rx: Arc::new(std::sync::Mutex::new(None)),
            tracking_started: Arc::new(AtomicBool::new(false)),
            workers: Arc::new(DashMap::new()),
            max_age: DEFAULT_FPM_MAX_AGE,
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
            let endpoint = self.endpoint.clone();
            let cancel = self.cancel.clone();
            let (tx, rx_new) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();

            {
                let mut guard = self.rx.lock().map_err(|e| to_pyerr(format!("{e}")))?;
                *guard = Some(rx_new);
            }

            let rt = endpoint.component().drt().runtime().secondary();
            rt.spawn(async move {
                let mut subscriber = match EventSubscriber::for_endpoint(&endpoint, FPM_TOPIC).await
                {
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

    /// Track samples and discovery state with a fixed freshness and cleanup window.
    /// Legacy snapshots retain samples until worker removal; health snapshots filter by age.
    #[pyo3(signature = (max_age_seconds = DEFAULT_FPM_MAX_AGE.as_secs_f64()))]
    fn start_tracking(&mut self, max_age_seconds: f64) -> PyResult<()> {
        let max_age = Duration::try_from_secs_f64(max_age_seconds)
            .ok()
            .filter(|age| !age.is_zero())
            .ok_or_else(|| PyValueError::new_err("max_age_seconds must be finite and positive"))?;
        if self.recv_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "Cannot call start_tracking() after recv()",
            ));
        }
        if self.tracking_started.swap(true, Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err("Tracking already started"));
        }

        let endpoint = self.endpoint.clone();
        let rt = endpoint.component().drt().runtime().secondary();
        let cancel = self.cancel.clone();
        self.max_age = max_age;
        let workers = self.workers.clone();

        rt.spawn({
            let cancel = cancel.clone();
            let endpoint = endpoint.clone();
            let workers = workers.clone();
            async move {
                let mut subscriber =
                    match EventSubscriber::for_endpoint(&endpoint, FPM_TOPIC).await {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("FPM tracker: failed to create subscriber: {e}");
                            return;
                        }
                    };

                tracing::info!("FPM tracker: listening for forward-pass-metrics events");
                let mut cleanup_tick = tokio::time::interval(FPM_CLEANUP_INTERVAL);
                cleanup_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {
                            tracing::info!("FPM tracker: shutting down event task");
                            break;
                        }
                        _ = cleanup_tick.tick() => {
                            prune_inactive_workers(&workers, Instant::now(), max_age);
                        }
                        event = subscriber.next() => {
                            match event {
                                Some(Ok(envelope)) => {
                                    let payload = envelope.payload.to_vec();
                                    if let Some(key) = extract_fpm_key(&payload) {
                                        record_fpm(
                                            &workers,
                                            key,
                                            payload,
                                            Instant::now(),
                                        );
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

        rt.spawn({
            let cancel = cancel.clone();
            let endpoint = endpoint.clone();
            async move {
                let discovery = endpoint.component().drt().discovery();
                let endpoint_id = endpoint.id();
                let query = DiscoveryQuery::EndpointModels {
                    namespace: endpoint_id.namespace,
                    component: endpoint_id.component,
                    endpoint: endpoint_id.name,
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
                                    // Capture the full card JSON so connectors can build WorkerInfo
                                    // from runtime_config / display_name / kv_cache_block_size / etc.
                                    // without the subscriber having to know which fields matter.
                                    if let DiscoveryInstance::Model { ref card_json, .. } = instance {
                                        match serde_json::to_string(card_json) {
                                            Ok(s) => {
                                                let DiscoveryInstanceId::Model(card_id) = instance.id() else {
                                                    unreachable!("model instance must have a model identity");
                                                };
                                                insert_worker_model_card(
                                                    &workers,
                                                    wid.clone(),
                                                    card_id,
                                                    s,
                                                );
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    "FPM tracker: failed to serialize card_json for {wid}: {e}"
                                                );
                                            }
                                        }
                                    }
                                    tracing::debug!(worker_id = %wid, "FPM tracker: model card added");
                                }
                                Some(Ok(DiscoveryEvent::ModelTaintsUpdated(update))) => {
                                    match update_worker_model_card_taints(&workers, &update) {
                                        Ok(true) => tracing::debug!(
                                            instance_id = update.id.instance_id,
                                            "FPM tracker: model-card taints updated"
                                        ),
                                        Ok(false) => tracing::debug!(
                                            instance_id = update.id.instance_id,
                                            "FPM tracker: ignored unknown or duplicate taint update"
                                        ),
                                        Err(error) => tracing::warn!(
                                            instance_id = update.id.instance_id,
                                            %error,
                                            "FPM tracker: failed to update model-card taints"
                                        ),
                                    }
                                }
                                Some(Ok(DiscoveryEvent::Removed(id))) => {
                                    let DiscoveryInstanceId::Model(card_id) = id else {
                                        tracing::warn!(
                                            "FPM tracker: endpoint model watch returned non-model removal"
                                        );
                                        continue;
                                    };
                                    let removed_id = card_id.instance_id.to_string();
                                    if remove_worker_model_card(&workers, &removed_id, &card_id) {
                                        tracing::debug!(
                                            "FPM tracker: model card removed for worker {removed_id}; \
                                             other cards remain"
                                        );
                                        continue;
                                    }

                                    tracing::debug!(worker_id = %removed_id, "FPM tracker: worker removed");
                                }
                                Some(Ok(DiscoveryEvent::Resync(_))) => {}
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

    /// Latest samples for active workers, without age filtering (legacy API).
    fn get_recent_stats(&self) -> PyResult<FpmSnapshot> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }
        let mut snapshot = HashMap::new();
        for worker in self.workers.iter() {
            if matches!(worker.lifecycle, FpmWorkerLifecycle::Active { .. }) {
                for (&rank, sample) in &worker.ranks {
                    snapshot.insert((worker.key().clone(), rank), sample.payload.clone());
                }
            }
        }
        Ok(snapshot)
    }

    /// Freshness uses local receive time, including idle heartbeats.
    fn get_recent_stats_with_health(&self) -> PyResult<FpmHealthSnapshot> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }
        Ok(fresh_fpm_snapshot(
            &self.workers,
            Instant::now(),
            self.max_age,
        ))
    }

    /// Selected model deployment cards for active workers, as serialized JSON.
    fn get_model_cards(&self) -> PyResult<HashMap<String, String>> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }
        Ok(self
            .workers
            .iter()
            .filter(|worker| matches!(worker.lifecycle, FpmWorkerLifecycle::Active { .. }))
            .filter_map(|worker| {
                worker
                    .model_cards
                    .selected_card()
                    .map(|card| (worker.key().clone(), card.to_string()))
            })
            .collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn active_worker(ranks: Range<u32>, now: Instant) -> WorkerFpm {
        let mut worker = WorkerFpm::default();
        worker.activate(now);
        worker.model_cards.insert(
            card_id(42, None),
            serde_json::json!({
                "runtime_config": {
                    "data_parallel_start_rank": ranks.start,
                    "data_parallel_size": ranks.len(),
                }
            })
            .to_string(),
        );
        worker
    }

    #[test]
    fn re_registration_discards_prior_lifecycle_samples() {
        let workers = DashMap::new();
        let now = Instant::now();
        workers.insert("worker".to_string(), active_worker(0..2, now));
        record_fpm(&workers, ("worker".to_string(), 0), vec![1], now);
        workers.get_mut("worker").unwrap().remove(now);
        let registered_at = now + Duration::from_secs(1);
        workers.get_mut("worker").unwrap().activate(registered_at);
        // A write captured before registration cannot repopulate the new epoch.
        record_fpm(&workers, ("worker".to_string(), 1), vec![2], now);
        let health = fresh_fpm_snapshot(&workers, registered_at, Duration::from_secs(2));
        assert!(health.snapshot.is_empty());
        assert_eq!(health.missing_rank_count, 2);
        record_fpm(&workers, ("worker".to_string(), 0), vec![3], registered_at);
        workers
            .get_mut("worker")
            .unwrap()
            .activate(registered_at + Duration::from_secs(1));
        assert_eq!(workers.get("worker").unwrap().ranks[&0].payload, vec![3]);
    }

    #[test]
    fn stale_samples_are_missing_until_a_fresh_idle_heartbeat() {
        let workers = DashMap::new();
        let now = Instant::now();
        workers.insert("worker".to_string(), active_worker(0..1, now));
        record_fpm(&workers, ("worker".to_string(), 0), vec![0], now);
        let later = now + Duration::from_secs(2);
        let health = fresh_fpm_snapshot(&workers, later, Duration::from_secs(1));
        assert!(health.snapshot.is_empty());
        assert_eq!(health.stale_rank_count, 1);
        assert_eq!(health.missing_rank_count, 1);
        assert_eq!(workers.get("worker").unwrap().ranks.len(), 1);
        record_fpm(&workers, ("worker".to_string(), 0), vec![0], later);
        let health = fresh_fpm_snapshot(&workers, later, Duration::from_secs(1));
        assert_eq!(health.snapshot.len(), 1);
        assert_eq!(health.stale_rank_count, 0);
        assert_eq!(health.missing_rank_count, 0);
    }

    #[test]
    fn partial_dp_pool_reports_missing_ranks_and_unknown_publishers() {
        let workers = DashMap::new();
        let now = Instant::now();
        workers.insert("worker".to_string(), active_worker(4..6, now));
        workers.insert("silent".to_string(), active_worker(0..1, now));
        record_fpm(&workers, ("worker".to_string(), 4), vec![1], now);
        record_fpm(&workers, ("orphan".to_string(), 0), vec![2], now);
        let health = fresh_fpm_snapshot(&workers, now, Duration::from_secs(1));
        assert_eq!(health.snapshot.len(), 1);
        assert_eq!(health.missing_rank_count, 2);
        assert_eq!(health.unjoined_worker_ids, ["orphan"]);
        for (worker, rank) in [("worker", 5), ("silent", 0)] {
            record_fpm(&workers, (worker.to_string(), rank), vec![3], now);
        }
        assert_eq!(
            fresh_fpm_snapshot(&workers, now, Duration::from_secs(1)).missing_rank_count,
            0
        );
    }

    #[test]
    fn cleanup_reclaims_quiet_workers_without_storing_removed_samples() {
        let workers = DashMap::new();
        let now = Instant::now();
        let mut removed = WorkerFpm::default();
        removed.remove(now + Duration::from_millis(500));
        workers.insert("removed".to_string(), removed);
        record_fpm(&workers, ("removed".to_string(), 0), vec![0], now);
        assert_eq!(
            workers.get("removed").unwrap().lifecycle,
            FpmWorkerLifecycle::Removed {
                last_activity: now + Duration::from_millis(500)
            }
        );
        record_fpm(
            &workers,
            ("removed".to_string(), 0),
            vec![1],
            now + Duration::from_millis(750),
        );
        record_fpm(&workers, ("orphan".to_string(), 0), vec![2], now);
        assert!(workers.get("removed").unwrap().ranks.is_empty());
        prune_inactive_workers(
            &workers,
            now + Duration::from_millis(1600),
            Duration::from_secs(1),
        );
        assert!(workers.contains_key("removed"));
        assert!(!workers.contains_key("orphan"));
        prune_inactive_workers(
            &workers,
            now + Duration::from_secs(2),
            Duration::from_secs(1),
        );
        assert!(workers.is_empty());
    }

    #[test]
    fn expected_ranks_use_runtime_config_defaults_and_validation() {
        assert!(expected_fpm_ranks("[]").is_err());
        assert_eq!(expected_fpm_ranks("{}").unwrap(), 0..1);
        assert_eq!(
            expected_fpm_ranks(r#"{"runtime_config":{"unknown_future_field":true}}"#).unwrap(),
            0..1
        );
        assert_eq!(
            expected_fpm_ranks(
                r#"{"runtime_config":{"data_parallel_start_rank":4,"data_parallel_size":2}}"#
            )
            .unwrap(),
            4..6
        );
        for invalid in [0, 4097] {
            assert!(
                expected_fpm_ranks(
                    &serde_json::json!({"runtime_config":{"data_parallel_size":invalid}})
                        .to_string()
                )
                .is_err()
            );
        }
        assert!(expected_fpm_ranks(r#"{"runtime_config":{"data_parallel_size":"bad"}}"#).is_err());
    }

    #[test]
    fn card_update_drops_obsolete_rank_and_preserves_valid_sample() {
        let workers = DashMap::new();
        let now = Instant::now();
        workers.insert("42".to_string(), active_worker(4..6, now));
        for rank in [4, 5] {
            record_fpm(&workers, ("42".to_string(), rank), vec![1], now);
        }
        insert_worker_model_card(
            &workers,
            "42".to_string(),
            card_id(42, None),
            serde_json::json!({
                "runtime_config": {"data_parallel_start_rank": 5, "data_parallel_size": 1}
            })
            .to_string(),
        );

        let health = fresh_fpm_snapshot(&workers, now, Duration::from_secs(1));
        assert_eq!(
            health.snapshot,
            HashMap::from([(("42".to_string(), 5), vec![1])])
        );
        assert!(health.unjoined_worker_ids.is_empty());
        assert_eq!(health.missing_rank_count, 0);
    }

    fn card_id(instance_id: u64, model_suffix: Option<&str>) -> ModelCardInstanceId {
        ModelCardInstanceId {
            namespace: "ns".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id,
            model_suffix: model_suffix.map(str::to_string),
        }
    }

    #[test]
    fn base_model_card_wins_over_lora_cards() {
        let mut cards = WorkerModelCards::default();
        cards.insert(card_id(42, Some("adapter-a")), "lora".to_string());
        cards.insert(card_id(42, None), "base".to_string());
        cards.insert(card_id(42, Some("adapter-b")), "lora-2".to_string());

        assert_eq!(cards.selected_card(), Some("base"));
    }

    #[test]
    fn removing_cards_keeps_worker_live_until_last_card() {
        let cards = DashMap::new();
        let base = card_id(42, None);
        let lora_a = card_id(42, Some("adapter-a"));
        let lora_b = card_id(42, Some("adapter-b"));
        insert_worker_model_card(&cards, "42".to_string(), base.clone(), "base".to_string());
        insert_worker_model_card(
            &cards,
            "42".to_string(),
            lora_a.clone(),
            "lora-a".to_string(),
        );
        insert_worker_model_card(
            &cards,
            "42".to_string(),
            lora_b.clone(),
            "lora-b".to_string(),
        );

        assert!(remove_worker_model_card(&cards, "42", &base));
        assert_eq!(
            cards.get("42").unwrap().model_cards.selected_card(),
            Some("lora-b")
        );

        assert!(remove_worker_model_card(&cards, "42", &lora_b));
        assert_eq!(
            cards.get("42").unwrap().model_cards.selected_card(),
            Some("lora-a")
        );
        assert!(matches!(
            cards.get("42").unwrap().lifecycle,
            FpmWorkerLifecycle::Active { .. }
        ));

        assert!(!remove_worker_model_card(&cards, "42", &lora_a));
        assert!(matches!(
            cards.get("42").unwrap().lifecycle,
            FpmWorkerLifecycle::Removed { .. }
        ));
    }

    #[test]
    fn scoped_taint_update_refreshes_retained_base_card() {
        let cards = DashMap::new();
        let id = card_id(42, None);
        insert_worker_model_card(
            &cards,
            "42".to_string(),
            id.clone(),
            serde_json::json!({"runtime_config": {"taints": ["old"]}}).to_string(),
        );

        assert!(
            update_worker_model_card_taints(
                &cards,
                &ModelTaintsUpdate {
                    id: id.clone(),
                    taints: vec!["blue".to_string(), "gpu".to_string()],
                },
            )
            .unwrap()
        );
        let card: serde_json::Value = serde_json::from_str(
            cards
                .get("42")
                .unwrap()
                .model_cards
                .selected_card()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            card["runtime_config"]["taints"],
            serde_json::json!(["blue", "gpu"])
        );

        assert!(
            !update_worker_model_card_taints(
                &cards,
                &ModelTaintsUpdate {
                    id: card_id(99, None),
                    taints: vec!["unknown".to_string()],
                },
            )
            .unwrap()
        );
        assert_eq!(cards.len(), 1);
    }
}
