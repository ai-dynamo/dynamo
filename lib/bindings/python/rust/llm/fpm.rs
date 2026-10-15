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

use dashmap::{DashMap, DashSet};
use dynamo_mocker::common::protocols::{ForwardPassSnapshot, FpmPublisher};
use futures::StreamExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
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
use llm_rs::local_model::runtime_config::ModelRuntimeConfig as RuntimeConfig;

const FPM_TOPIC: &str = "forward-pass-metrics";

struct TrackedFpm {
    payload: Vec<u8>,
    received_at: Instant,
}

type FpmSnapshot = HashMap<(String, i64), Vec<u8>>;

fn fresh_fpm_snapshot(
    latest: &DashMap<(String, i64), TrackedFpm>,
    expected: &HashMap<String, Option<Range<u32>>>,
    now: Instant,
    max_age: Duration,
) -> (FpmSnapshot, Vec<String>, usize, usize) {
    let mut snapshot = HashMap::new();
    let mut unjoined = HashSet::new();
    let mut stale = 0;
    for entry in latest {
        let expired = now.saturating_duration_since(entry.value().received_at) > max_age;
        let rank_is_expected = expected
            .get(&entry.key().0)
            .and_then(Option::as_ref)
            .is_some_and(|ranks| {
                u32::try_from(entry.key().1).is_ok_and(|rank| ranks.contains(&rank))
            });
        if !rank_is_expected {
            if !expired {
                unjoined.insert(entry.key().0.clone());
            }
        } else if expired {
            stale += 1;
        } else {
            snapshot.insert(entry.key().clone(), entry.value().payload.clone());
        }
    }
    let mut unjoined: Vec<_> = unjoined.into_iter().collect();
    unjoined.sort_unstable();
    let mut fresh_counts = HashMap::new();
    for (worker, _) in snapshot.keys() {
        *fresh_counts.entry(worker.as_str()).or_insert(0usize) += 1;
    }
    let missing = expected
        .iter()
        .map(|(worker, ranks)| {
            ranks.as_ref().map_or(1, |ranks| {
                ranks
                    .len()
                    .saturating_sub(fresh_counts.get(worker.as_str()).copied().unwrap_or(0))
            })
        })
        .sum();
    (snapshot, unjoined, stale, missing)
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
    fn new(endpoint: Endpoint, zmq_endpoint: String) -> PyResult<Self> {
        let inner = llm_rs::fpm_publisher::FpmEventRelay::new(endpoint.inner.clone(), zmq_endpoint)
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
    fn new(endpoint: Endpoint, worker_id: String, dp_size: u32) -> PyResult<Self> {
        let rt = endpoint.inner.component().drt().runtime().secondary();
        let (inner, publishers) = rt
            .block_on(async {
                llm_rs::fpm_publisher::FpmDirectPublisher::new(endpoint.inner, worker_id, dp_size)
                    .await
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
    endpoint: RuntimeEndpoint,
    cancel: CancellationToken,

    // recv mode state (lazily initialised on first recv() call)
    recv_started: Arc<AtomicBool>,
    rx: Arc<std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>>>,

    // tracking mode state
    tracking_started: Arc<AtomicBool>,
    latest_stats: Arc<DashMap<(String, i64), TrackedFpm>>,
    // Worker IDs currently registered in MDC.  Maintained by Task 2
    // (insert on Added, remove on Removed).  Used by get_recent_stats()
    // to filter out ghost entries without contending with Task 1's writes.
    known_workers: Arc<DashSet<String>>,
    // Serialized ModelDeploymentCards grouped by worker. Each worker retains
    // complete card identities for exact removal and one selected base/fallback
    // card for snapshots.
    worker_model_cards: Arc<DashMap<String, WorkerModelCards>>,
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

#[derive(serde::Deserialize)]
struct FpmCardConfig {
    #[serde(default)]
    runtime_config: FpmDpConfig,
}

#[derive(serde::Deserialize)]
#[serde(default)]
struct FpmDpConfig {
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
}

impl Default for FpmDpConfig {
    fn default() -> Self {
        let defaults = RuntimeConfig::default();
        Self {
            data_parallel_start_rank: defaults.data_parallel_start_rank,
            data_parallel_size: defaults.data_parallel_size,
        }
    }
}

fn expected_fpm_ranks(card: &str) -> Option<Range<u32>> {
    serde_json::from_str::<FpmCardConfig>(card)
        .ok()
        .and_then(|card| {
            RuntimeConfig {
                data_parallel_start_rank: card.runtime_config.data_parallel_start_rank,
                data_parallel_size: card.runtime_config.data_parallel_size,
                ..Default::default()
            }
            .data_parallel_rank_range()
            .ok()
        })
}

impl WorkerModelCards {
    fn insert(&mut self, card_id: ModelCardInstanceId, card: String) {
        let dp_ranks = expected_fpm_ranks(&card);
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
    cards: &DashMap<String, WorkerModelCards>,
    worker_id: String,
    card_id: ModelCardInstanceId,
    card: String,
) {
    cards.entry(worker_id).or_default().insert(card_id, card);
}

fn update_worker_model_card_taints(
    cards: &DashMap<String, WorkerModelCards>,
    update: &ModelTaintsUpdate,
) -> anyhow::Result<bool> {
    let worker_id = update.id.instance_id.to_string();
    let Some(mut worker_cards) = cards.get_mut(&worker_id) else {
        return Ok(false);
    };
    let Some(indexed) = worker_cards.cards.get_mut(&update.id) else {
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
    cards: &DashMap<String, WorkerModelCards>,
    worker_id: &str,
    card_id: &ModelCardInstanceId,
) -> bool {
    let Some(mut worker_cards) = cards.get_mut(worker_id) else {
        return false;
    };
    worker_cards.remove(card_id);
    if !worker_cards.is_empty() {
        return true;
    }

    drop(worker_cards);
    cards.remove(worker_id);
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
            latest_stats: Arc::new(DashMap::new()),
            known_workers: Arc::new(DashSet::new()),
            worker_model_cards: Arc::new(DashMap::new()),
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

    /// Start background tracking of the latest FPM per `(worker_id, dp_rank)`.
    ///
    /// Spawns two background tasks:
    ///
    /// 1. **Event consumption** (Task 1): subscribes to FPM events, extracts
    ///    `(worker_id, dp_rank)` from the msgpack payload, stores the latest
    ///    raw bytes in `latest_stats`.  Uses per-shard locking via `DashMap`
    ///    so contention with concurrent readers is minimal.
    ///
    /// 2. **MDC discovery watch** (Task 2): monitors `EndpointModels` for the
    ///    target endpoint.  Maintains `known_workers` (the set of currently
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

        let endpoint = self.endpoint.clone();
        let rt = endpoint.component().drt().runtime().secondary();
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
            let endpoint = endpoint.clone();
            let stats = stats.clone();
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
                                        stats.insert(key, TrackedFpm {
                                            payload,
                                            received_at: Instant::now(),
                                        });
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
        let cards = self.worker_model_cards.clone();
        rt.spawn({
            let cancel = cancel.clone();
            let endpoint = endpoint.clone();
            let stats = stats.clone();
            let known = known.clone();
            let cards = cards.clone();
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
                                                    &cards,
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
                                    known.insert(wid.clone());
                                    tracing::debug!("FPM tracker: worker {wid} added to known set");
                                }
                                Some(Ok(DiscoveryEvent::ModelTaintsUpdated(update))) => {
                                    match update_worker_model_card_taints(&cards, &update) {
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
                                    if remove_worker_model_card(&cards, &removed_id, &card_id) {
                                        tracing::debug!(
                                            "FPM tracker: model card removed for worker {removed_id}; \
                                             other cards remain"
                                        );
                                        continue;
                                    }

                                    known.remove(&removed_id);

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
            .map(|entry| (entry.key().clone(), entry.value().payload.clone()))
            .collect();

        Ok(snapshot)
    }

    /// Return fresh samples, unjoined IDs, stale rank count, and missing rank count.
    /// Freshness uses local receive time, including idle heartbeats, rather than
    /// publisher clocks. The legacy get_recent_stats method retains its semantics.
    fn get_recent_stats_with_health(
        &self,
        max_age_seconds: f64,
    ) -> PyResult<(FpmSnapshot, Vec<String>, usize, usize)> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }
        let max_age = Duration::try_from_secs_f64(max_age_seconds)
            .ok()
            .filter(|age| !age.is_zero())
            .ok_or_else(|| PyValueError::new_err("max_age_seconds must be finite and positive"))?;
        let expected = self
            .known_workers
            .iter()
            .map(|worker| {
                let ranks = self.worker_model_cards.get(worker.key()).and_then(|cards| {
                    cards
                        .selected_entry()
                        .and_then(|entry| entry.dp_ranks.clone())
                });
                (worker.key().clone(), ranks)
            })
            .collect();
        Ok(fresh_fpm_snapshot(
            &self.latest_stats,
            &expected,
            Instant::now(),
            max_age,
        ))
    }

    /// Snapshot of model deployment cards keyed by worker id.
    ///
    /// The snapshot is filtered against `known_workers` so entries for
    /// already-removed workers are not returned.  Values are the raw
    /// `ModelDeploymentCard` serialized as a JSON string; callers parse
    /// whichever fields they need (e.g. `runtime_config`, `display_name`).
    ///
    /// Returns:
    ///     dict mapping `worker_id: str` to `card_json: str`.
    fn get_model_cards(&self) -> PyResult<HashMap<String, String>> {
        if !self.tracking_started.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err(
                "start_tracking() has not been called",
            ));
        }

        let snapshot = self
            .worker_model_cards
            .iter()
            .filter(|entry| self.known_workers.contains(entry.key()))
            .filter_map(|entry| {
                entry
                    .value()
                    .selected_card()
                    .map(|card| (entry.key().clone(), card.to_string()))
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_separates_missing_identity_and_stale_samples() {
        let now = Instant::now();
        let latest = DashMap::new();
        let mut expected = HashMap::new();
        expected.insert("fresh".to_string(), Some(0..1));
        expected.insert("stale".to_string(), Some(0..1));
        for (worker, age) in [("fresh", 0), ("stale", 2), ("orphan", 0), ("removed", 2)] {
            latest.insert(
                (worker.to_string(), 0),
                TrackedFpm {
                    payload: vec![1],
                    received_at: now - Duration::from_secs(age),
                },
            );
        }
        let (snapshot, unjoined, stale, missing) =
            fresh_fpm_snapshot(&latest, &expected, now, Duration::from_secs(1));
        assert_eq!(snapshot.len(), 1);
        assert!(snapshot.contains_key(&("fresh".to_string(), 0)));
        assert_eq!(unjoined, ["orphan"]);
        assert_eq!(stale, 1);
        assert_eq!(missing, 1);

        // An identical idle heartbeat is fresh when it was received again.
        latest.insert(
            ("stale".to_string(), 0),
            TrackedFpm {
                payload: vec![1],
                received_at: now,
            },
        );
        expected.insert("orphan".to_string(), Some(0..1));
        let (snapshot, unjoined, stale, missing) =
            fresh_fpm_snapshot(&latest, &expected, now, Duration::from_secs(1));
        assert_eq!(snapshot.len(), 3);
        assert!(unjoined.is_empty());
        assert_eq!(stale, 0);
        assert_eq!(missing, 0);
    }

    #[test]
    fn snapshot_reports_workers_and_offset_ranks_that_never_published() {
        let now = Instant::now();
        let latest = DashMap::new();
        let expected = HashMap::from([
            ("42".to_string(), Some(4..6)),
            ("99".to_string(), Some(0..1)),
        ]);
        latest.insert(
            ("42".to_string(), 4),
            TrackedFpm {
                payload: vec![1],
                received_at: now,
            },
        );
        let (_, unjoined, stale, missing) =
            fresh_fpm_snapshot(&latest, &expected, now, Duration::from_secs(1));
        assert!(unjoined.is_empty());
        assert_eq!(stale, 0);
        assert_eq!(missing, 2);

        latest.insert(
            ("42".to_string(), 5),
            TrackedFpm {
                payload: vec![1],
                received_at: now,
            },
        );
        assert_eq!(
            fresh_fpm_snapshot(&latest, &expected, now, Duration::from_secs(1)).3,
            1
        );
        latest.insert(
            ("99".to_string(), 0),
            TrackedFpm {
                payload: vec![1],
                received_at: now,
            },
        );
        assert_eq!(
            fresh_fpm_snapshot(&latest, &expected, now, Duration::from_secs(1)).3,
            0
        );
    }

    #[test]
    fn expected_ranks_accept_legacy_defaults_and_validate_explicit_layout() {
        assert_eq!(expected_fpm_ranks("{}"), Some(0..1));
        assert_eq!(
            expected_fpm_ranks(r#"{"runtime_config":{"unknown_future_field":true}}"#),
            Some(0..1)
        );
        assert_eq!(
            expected_fpm_ranks(
                r#"{"runtime_config":{"data_parallel_start_rank":4,"data_parallel_size":2}}"#
            ),
            Some(4..6)
        );
        for invalid in [0, 4097, u32::MAX] {
            assert_eq!(
                expected_fpm_ranks(
                    &serde_json::json!({"runtime_config":{"data_parallel_size":invalid}})
                        .to_string()
                ),
                None
            );
        }
    }

    #[test]
    fn selected_card_rank_range_follows_base_updates_and_removal() {
        let mut cards = WorkerModelCards::default();
        let base = card_id(42, None);
        let card = |start, size| {
            serde_json::json!({
                "runtime_config": {"data_parallel_start_rank": start, "data_parallel_size": size}
            })
            .to_string()
        };
        cards.insert(card_id(42, Some("adapter")), card(99, 2));
        cards.insert(base.clone(), card(4, 2));
        assert_eq!(cards.selected_entry().unwrap().dp_ranks, Some(4..6));
        cards.insert(base.clone(), card(8, 2));
        assert_eq!(cards.selected_entry().unwrap().dp_ranks, Some(8..10));
        cards.remove(&base);
        assert_eq!(cards.selected_entry().unwrap().dp_ranks, Some(99..101));
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
        assert_eq!(cards.get("42").unwrap().selected_card(), Some("lora-b"));

        assert!(remove_worker_model_card(&cards, "42", &lora_b));
        assert_eq!(cards.get("42").unwrap().selected_card(), Some("lora-a"));

        assert!(!remove_worker_model_card(&cards, "42", &lora_a));
        assert!(!cards.contains_key("42"));
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
        let card: serde_json::Value =
            serde_json::from_str(cards.get("42").unwrap().selected_card().unwrap()).unwrap();
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
