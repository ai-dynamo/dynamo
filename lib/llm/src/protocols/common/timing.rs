// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request tracker for capturing request lifecycle metrics.
//!
//! This module provides [`RequestTracker`] for tracking timing and routing information
//! that can be returned to clients via the `nvext` response field.

use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU32, AtomicU64, Ordering},
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use utoipa::ToSchema;

use crate::protocols::openai::nvext::WorkerIdInfo;

/// Sentinel value indicating no worker ID has been set.
/// We use 0 as the sentinel since valid worker IDs are non-zero lease IDs from etcd.
const NO_WORKER_ID: u64 = 0;
const NO_DP_RANK: u32 = u32::MAX;

/// Worker type constants for Prometheus metric labels.
/// These are stored in RequestTracker at routing time to avoid costly MDC lookups
/// when updating per-worker metrics (TTFT, ITL).
pub const WORKER_TYPE_PREFILL: &str = "prefill";
pub const WORKER_TYPE_DECODE: &str = "decode";

/// Phase of the request in disaggregated serving.
///
/// Used to determine which worker ID field to record when routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestPhase {
    /// Prefill-only phase (disaggregated serving)
    Prefill,
    /// Decode phase (disaggregated serving)
    Decode,
    /// Aggregated mode - same worker handles both prefill and decode
    #[default]
    Aggregated,
}

impl std::fmt::Display for RequestPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequestPhase::Prefill => write!(f, "prefill"),
            RequestPhase::Decode => write!(f, "decode"),
            RequestPhase::Aggregated => write!(f, "aggregated"),
        }
    }
}

/// Per-request tracker for timing and routing metrics.
///
/// Captures information throughout the request lifecycle:
/// - `request_received`: When the request was received
/// - `prefill_start_time`: When prefill started (for disaggregated serving)
/// - `first_token_time`: When the first token was generated (set once via OnceLock)
/// - `request_finish_time`: When the request finished (set once via OnceLock)
/// - KV cache hit rate information
/// - Worker IDs and types for per-worker Prometheus metrics
///
/// The `OnceLock` fields ensure that values are set exactly once,
/// which is important for disaggregated serving where the "first token"
/// might appear multiple times.
///
/// Worker IDs use `AtomicU64` instead of `OnceLock<u64>` for lower overhead since
/// the tracker is created for every request. The sentinel value `NO_WORKER_ID` (0)
/// indicates no worker has been recorded yet.
#[derive(Debug)]
pub struct RequestTracker {
    /// When the request was received (monotonic clock for duration calculations)
    request_received: Instant,

    /// When the request was received (wall clock time as epoch milliseconds)
    request_received_epoch_ms: u64,

    /// When prefill started (for disaggregated serving) - set once via OnceLock
    prefill_start_time: OnceLock<Instant>,

    /// When the first token was generated - set once via OnceLock
    first_token_time: OnceLock<Instant>,

    /// When the request finished - set once via OnceLock
    request_finish_time: OnceLock<Instant>,

    /// KV cache overlap blocks (prefix cache hits) - set once via OnceLock
    kv_overlap_blocks: OnceLock<u32>,

    /// Input sequence length in blocks (for hit rate calculation) - set once via OnceLock
    isl_blocks: OnceLock<usize>,

    /// Prefill worker ID (for disaggregated serving).
    /// Uses atomic with compare-exchange for set-once semantics.
    /// Value of 0 (NO_WORKER_ID) means not yet set.
    prefill_worker_id: AtomicU64,

    /// Prefill DP rank. Value of u32::MAX (NO_DP_RANK) means not yet set.
    prefill_dp_rank: AtomicU32,

    /// Decode worker ID. Value of 0 (NO_WORKER_ID) means not yet set.
    decode_worker_id: AtomicU64,

    /// Decode DP rank. Value of u32::MAX (NO_DP_RANK) means not yet set.
    decode_dp_rank: AtomicU32,

    /// Worker type for the prefill worker ("prefill" or "decode").
    /// Stored at routing time to avoid MDC lookup when updating Prometheus metrics.
    /// In aggregated mode, this will be "decode" since the same worker handles both.
    /// This is necessary because TTFT metrics need to know the worker type label,
    /// and looking up MDC by worker_id would require iterating all cards (O(n)).
    prefill_worker_type: OnceLock<&'static str>,

    /// Worker type for the decode worker (always "decode").
    /// Stored for symmetry with prefill_worker_type, though decode is always "decode".
    decode_worker_type: OnceLock<&'static str>,

    /// Request phase (Prefill/Decode/Aggregated)
    phase: Mutex<RequestPhase>,

    /// Semaphore for coordinating phase transitions.
    /// Acquiring a permit blocks subsequent set_phase calls until the permit is dropped.
    /// This prevents race conditions in the bootstrap optimization path where prefill
    /// runs in background and needs to complete record_worker before phase changes.
    phase_semaphore: Arc<Semaphore>,
}

impl RequestTracker {
    /// Create a new request tracker, capturing the current time as request received.
    pub fn new() -> Self {
        let now = Instant::now();
        let epoch_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        RequestTracker {
            request_received: now,
            request_received_epoch_ms: epoch_ms,
            prefill_start_time: OnceLock::new(),
            first_token_time: OnceLock::new(),
            request_finish_time: OnceLock::new(),
            kv_overlap_blocks: OnceLock::new(),
            isl_blocks: OnceLock::new(),
            prefill_worker_id: AtomicU64::new(NO_WORKER_ID),
            prefill_dp_rank: AtomicU32::new(NO_DP_RANK),
            decode_worker_id: AtomicU64::new(NO_WORKER_ID),
            decode_dp_rank: AtomicU32::new(NO_DP_RANK),
            prefill_worker_type: OnceLock::new(),
            decode_worker_type: OnceLock::new(),
            phase: Mutex::new(RequestPhase::Aggregated),
            phase_semaphore: Arc::new(Semaphore::new(1)),
        }
    }

    /// Record when prefill started. Returns true if this was the first call.
    pub fn record_prefill_start(&self) -> bool {
        self.prefill_start_time.set(Instant::now()).is_ok()
    }

    pub fn record_first_token(&self) -> bool {
        self.first_token_time.set(Instant::now()).is_ok()
    }

    pub fn record_finish(&self) -> bool {
        self.request_finish_time.set(Instant::now()).is_ok()
    }

    /// Record KV cache hit information. Returns true if this was the first call.
    pub fn record_kv_hit(&self, overlap_blocks: u32, isl_blocks: usize) -> bool {
        let overlap_set = self.kv_overlap_blocks.set(overlap_blocks).is_ok();
        let isl_set = self.isl_blocks.set(isl_blocks).is_ok();
        overlap_set && isl_set
    }

    /// Time from request received to prefill start (queue/wait time) in milliseconds.
    pub fn prefill_wait_time_ms(&self) -> Option<f64> {
        self.prefill_start_time
            .get()
            .map(|t| t.duration_since(self.request_received).as_secs_f64() * 1000.0)
    }

    /// Time from prefill start to first token (prefill execution time) in milliseconds.
    pub fn prefill_time_ms(&self) -> Option<f64> {
        let prefill_start = self.prefill_start_time.get()?;
        let first_token = self.first_token_time.get()?;
        Some(first_token.duration_since(*prefill_start).as_secs_f64() * 1000.0)
    }

    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time
            .get()
            .map(|t| t.duration_since(self.request_received).as_secs_f64() * 1000.0)
    }

    pub fn total_time_ms(&self) -> Option<f64> {
        self.request_finish_time
            .get()
            .map(|t| t.duration_since(self.request_received).as_secs_f64() * 1000.0)
    }

    pub fn request_received_epoch_ms(&self) -> u64 {
        self.request_received_epoch_ms
    }

    /// KV cache hit rate as a ratio (0.0 to 1.0).
    pub fn kv_hit_rate(&self) -> Option<f64> {
        let overlap = *self.kv_overlap_blocks.get()?;
        let isl = *self.isl_blocks.get()?;
        if isl == 0 {
            return None;
        }
        Some(overlap as f64 / isl as f64)
    }

    /// Record the prefill worker ID. Returns true if this was the first call.
    pub fn record_prefill_worker(&self, id: u64) -> bool {
        self.prefill_worker_id
            .compare_exchange(NO_WORKER_ID, id, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Record the prefill worker ID and DP rank. Returns true if worker_id was recorded for the first time.
    /// Only sets the dp_rank if the worker_id is newly set to avoid mismatched worker_id/dp_rank pairs.
    pub fn record_prefill_worker_with_rank(&self, id: u64, dp_rank: u32) -> bool {
        let is_new = self
            .prefill_worker_id
            .compare_exchange(NO_WORKER_ID, id, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if is_new {
            self.prefill_dp_rank.store(dp_rank, Ordering::SeqCst);
        }
        is_new
    }

    /// Record the prefill worker ID, DP rank, and worker type.
    /// The worker_type is stored to avoid MDC lookup when updating Prometheus metrics.
    /// Returns true if worker_id was recorded for the first time.
    pub fn record_prefill_worker_full(
        &self,
        id: u64,
        dp_rank: u32,
        worker_type: &'static str,
    ) -> bool {
        let is_new = self
            .prefill_worker_id
            .compare_exchange(NO_WORKER_ID, id, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if is_new {
            self.prefill_dp_rank.store(dp_rank, Ordering::SeqCst);
            let _ = self.prefill_worker_type.set(worker_type);
        }
        is_new
    }

    /// Record the decode worker ID. Returns true if this was the first call.
    pub fn record_decode_worker(&self, id: u64) -> bool {
        self.decode_worker_id
            .compare_exchange(NO_WORKER_ID, id, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// Record the decode worker ID and DP rank. Returns true if worker_id was recorded for the first time.
    /// Only sets the dp_rank if the worker_id is newly set to avoid mismatched worker_id/dp_rank pairs.
    pub fn record_decode_worker_with_rank(&self, id: u64, dp_rank: u32) -> bool {
        let is_new = self
            .decode_worker_id
            .compare_exchange(NO_WORKER_ID, id, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if is_new {
            self.decode_dp_rank.store(dp_rank, Ordering::SeqCst);
        }
        is_new
    }

    /// Record the decode worker ID, DP rank, and worker type.
    /// The worker_type is stored to avoid MDC lookup when updating Prometheus metrics.
    /// Returns true if worker_id was recorded for the first time.
    pub fn record_decode_worker_full(
        &self,
        id: u64,
        dp_rank: u32,
        worker_type: &'static str,
    ) -> bool {
        let is_new = self
            .decode_worker_id
            .compare_exchange(NO_WORKER_ID, id, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
        if is_new {
            self.decode_dp_rank.store(dp_rank, Ordering::SeqCst);
            let _ = self.decode_worker_type.set(worker_type);
        }
        is_new
    }

    /// Set the request phase and return a permit that blocks subsequent phase changes.
    ///
    /// The returned permit must be dropped to allow the next `set_phase` call to proceed.
    /// Under normal operation, callers can simply ignore the returned permit (letting it
    /// drop immediately). In the bootstrap optimization path, the permit is held and
    /// passed to the spawned prefill task, which drops it after `record_worker` completes.
    ///
    /// This prevents the race condition where the phase changes to Decode before the
    /// background prefill task has recorded its worker ID.
    pub async fn set_phase(&self, phase: RequestPhase) -> OwnedSemaphorePermit {
        let permit = self
            .phase_semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("phase semaphore should never be closed");
        *self.phase.lock() = phase;
        permit
    }

    /// Get the current request phase.
    pub fn phase(&self) -> RequestPhase {
        *self.phase.lock()
    }

    /// Record worker ID based on the current phase.
    ///
    /// - Prefill phase: records as prefill_worker_id
    /// - Decode phase: records as decode_worker_id
    /// - Aggregated phase: records as both prefill and decode worker
    pub fn record_worker(&self, instance_id: u64) {
        match self.phase() {
            RequestPhase::Prefill => {
                self.record_prefill_worker(instance_id);
            }
            RequestPhase::Decode => {
                self.record_decode_worker(instance_id);
            }
            RequestPhase::Aggregated => {
                self.record_prefill_worker(instance_id);
                self.record_decode_worker(instance_id);
            }
        }
    }

    /// Record worker ID and DP rank based on the current phase.
    ///
    /// - Prefill phase: records as prefill_worker_id/prefill_dp_rank
    /// - Decode phase: records as decode_worker_id/decode_dp_rank
    /// - Aggregated phase: records as both prefill and decode worker/rank
    pub fn record_worker_with_rank(&self, instance_id: u64, dp_rank: u32) {
        match self.phase() {
            RequestPhase::Prefill => {
                self.record_prefill_worker_with_rank(instance_id, dp_rank);
            }
            RequestPhase::Decode => {
                self.record_decode_worker_with_rank(instance_id, dp_rank);
            }
            RequestPhase::Aggregated => {
                self.record_prefill_worker_with_rank(instance_id, dp_rank);
                self.record_decode_worker_with_rank(instance_id, dp_rank);
            }
        }
    }

    /// Record worker ID, DP rank, and worker type based on the current phase.
    ///
    /// This is the preferred method when worker_type is known (from MDC or router config),
    /// as it stores the worker_type for later use in Prometheus metric updates without
    /// requiring an expensive MDC lookup.
    ///
    /// - Prefill phase: records as prefill worker with given worker_type
    /// - Decode phase: records as decode worker with given worker_type
    /// - Aggregated phase: records as both prefill and decode worker with the same worker_type
    pub fn record_worker_full(&self, instance_id: u64, dp_rank: u32, worker_type: &'static str) {
        match self.phase() {
            RequestPhase::Prefill => {
                self.record_prefill_worker_full(instance_id, dp_rank, worker_type);
            }
            RequestPhase::Decode => {
                self.record_decode_worker_full(instance_id, dp_rank, worker_type);
            }
            RequestPhase::Aggregated => {
                // In aggregated mode, both prefill and decode happen on the same worker,
                // so we record the same worker_type for both
                self.record_prefill_worker_full(instance_id, dp_rank, worker_type);
                self.record_decode_worker_full(instance_id, dp_rank, worker_type);
            }
        }
    }

    /// Get worker ID information if any worker IDs have been recorded.
    pub fn get_worker_info(&self) -> Option<WorkerIdInfo> {
        let prefill = self.prefill_worker_id();
        let decode = self.decode_worker_id();

        if prefill.is_none() && decode.is_none() {
            return None;
        }

        Some(WorkerIdInfo {
            prefill_worker_id: prefill,
            prefill_dp_rank: self.prefill_dp_rank(),
            decode_worker_id: decode,
            decode_dp_rank: self.decode_dp_rank(),
        })
    }

    /// Get the decode worker ID if recorded.
    pub fn decode_worker_id(&self) -> Option<u64> {
        let id = self.decode_worker_id.load(Ordering::SeqCst);
        if id == NO_WORKER_ID { None } else { Some(id) }
    }

    /// Get the decode DP rank if recorded.
    pub fn decode_dp_rank(&self) -> Option<u32> {
        let rank = self.decode_dp_rank.load(Ordering::SeqCst);
        if rank == NO_DP_RANK { None } else { Some(rank) }
    }

    /// Get the prefill worker ID if recorded.
    pub fn prefill_worker_id(&self) -> Option<u64> {
        let id = self.prefill_worker_id.load(Ordering::SeqCst);
        if id == NO_WORKER_ID { None } else { Some(id) }
    }

    /// Get the prefill DP rank if recorded.
    pub fn prefill_dp_rank(&self) -> Option<u32> {
        let rank = self.prefill_dp_rank.load(Ordering::SeqCst);
        if rank == NO_DP_RANK { None } else { Some(rank) }
    }

    /// Get the prefill worker type if recorded.
    pub fn prefill_worker_type(&self) -> Option<&'static str> {
        self.prefill_worker_type.get().copied()
    }

    /// Get the decode worker type if recorded.
    pub fn decode_worker_type(&self) -> Option<&'static str> {
        self.decode_worker_type.get().copied()
    }

    pub fn get_timing_info(&self) -> TimingInfo {
        TimingInfo {
            request_received_ms: self.request_received_epoch_ms,
            prefill_wait_time_ms: self.prefill_wait_time_ms(),
            prefill_time_ms: self.prefill_time_ms(),
            ttft_ms: self.ttft_ms(),
            total_time_ms: self.total_time_ms(),
            kv_hit_rate: self.kv_hit_rate(),
        }
    }
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing information for response injection.
///
/// This struct is serialized and included in the response's `nvext` field
/// when the client requests timing information via `extra_fields: ["timing"]`.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TimingInfo {
    /// When the request was received (epoch milliseconds)
    pub request_received_ms: u64,

    /// Time from request received to prefill start (queue/wait time) in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_wait_time_ms: Option<f64>,

    /// Time from prefill start to first token (prefill execution time) in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_time_ms: Option<f64>,

    /// Time to first token in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,

    /// Total request time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<f64>,

    /// KV cache hit rate (0.0 to 1.0) - ratio of cached blocks to total input blocks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_hit_rate: Option<f64>,
}
