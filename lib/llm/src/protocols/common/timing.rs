// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request tracker for capturing request lifecycle metrics.
//!
//! This module provides [`RequestTracker`] for tracking timing and routing information
//! that can be returned to clients via the `nvext` response field.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use utoipa::ToSchema;

use crate::http::service::metrics::{
    WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE, WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE,
    WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE,
};
use crate::protocols::openai::nvext::WorkerIdInfo;

/// Worker type constants for Prometheus metric labels.
/// These are stored in RequestTracker at routing time to avoid costly MDC lookups
/// when updating per-worker metrics (TTFT, ITL).
pub const WORKER_TYPE_PREFILL: &str = "prefill";
pub const WORKER_TYPE_DECODE: &str = "decode";
const UNSET_DP_RANK_LABEL: &str = "none";

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
/// - `first_token_time`: When the first token was generated
/// - `request_finish_time`: When the last token was generated (updated incrementally)
/// - KV cache hit rate information
/// - Worker IDs and types for per-worker Prometheus metrics
///
/// ## Concurrency primitives
///
/// **`OnceLock` (first-write-wins):** Used for values that must capture the earliest
/// observation and ignore later writes. In disaggregated serving, both prefill and decode
/// phases may call `record_first_token`; `OnceLock` ensures the prefill phase's TTFT is
/// preserved. Also used for one-shot metadata: `prefill_start_time`, KV hit info,
/// ISL/cached tokens, worker types, and tokenizer latency.
///
/// **`Mutex` (last-write-wins):** Used for values where later phases should overwrite
/// earlier ones. `request_finish_time` is updated incrementally at each output block
/// boundary so that `avg_itl_ms()` stays current during streaming, and the decode
/// phase's final finish naturally overwrites the prefill phase's earlier finish.
/// `phase` also uses a Mutex since it transitions across phases.
///
/// **`AtomicU64`:** Used for frequently updated counters (`osl_tokens`) and
/// accumulated detokenize timing, where lock-free updates are beneficial.
#[derive(Debug)]
pub struct RequestTracker {
    /// When the request was received (monotonic clock for duration calculations)
    request_received: Instant,

    /// When the request was received (wall clock time as epoch milliseconds)
    request_received_epoch_ms: u64,

    /// When prefill started (for disaggregated serving) - set once via OnceLock
    prefill_start_time: OnceLock<Instant>,

    /// When the first token was generated (set once via OnceLock).
    /// In disaggregated serving, the prefill phase records this first and the
    /// decode phase's attempt is silently ignored, preserving the real TTFT.
    first_token_time: OnceLock<Instant>,

    /// When the decode worker produced its first token (set once via OnceLock).
    /// Separate from `first_token_time` because in disaggregated serving, the prefill
    /// phase locks `first_token_time` first. This field captures the decode phase's
    /// first token for KV transfer latency estimation (`decode_first_token - prefill_complete`).
    decode_first_token_time: OnceLock<Instant>,

    /// When the request finished. Mutex allows the last router phase to
    /// record the final finish time.
    request_finish_time: Mutex<Option<Instant>>,

    /// KV cache overlap blocks (prefix cache hits) - set once via OnceLock
    kv_overlap_blocks: OnceLock<u32>,

    /// Input sequence length in blocks (for hit rate calculation) - set once via OnceLock
    isl_blocks: OnceLock<usize>,

    /// Input sequence length in tokens - set once via OnceLock
    isl_tokens: OnceLock<usize>,

    /// Number of cached tokens (overlap_blocks * block_size) - set once via OnceLock
    cached_tokens: OnceLock<usize>,

    /// Output sequence length in tokens - updated atomically as tokens stream back
    osl_tokens: AtomicU64,

    /// Prefill worker ID (for disaggregated serving) - set once when known.
    prefill_worker_id: OnceLock<u64>,

    /// Prefill DP rank - set once when known.
    prefill_dp_rank: OnceLock<u32>,

    /// Decode worker ID - set once when known.
    decode_worker_id: OnceLock<u64>,

    /// Decode DP rank - set once when known.
    decode_dp_rank: OnceLock<u32>,

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
    /// runs in background and needs to complete worker recording before phase changes.
    phase_semaphore: Arc<Semaphore>,

    /// How long it took to tokenize the input
    tokenize_latency: OnceLock<Duration>,

    /// Accumulated time spent detokenizing output tokens for this request (nanoseconds)
    detokenize_total_ns: AtomicU64,

    /// Number of detokenize samples accumulated for this request
    detokenize_count: AtomicU64,

    /// Router scheduler queue depth at routing time (how many requests were pending)
    router_queue_depth: OnceLock<usize>,

    /// When the prefill result arrived at the router (disaggregated, original path only).
    /// Set in execute_prefill() after the first output is received from the prefill worker.
    prefill_complete_time: OnceLock<Instant>,
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
            decode_first_token_time: OnceLock::new(),
            request_finish_time: Mutex::new(None),
            kv_overlap_blocks: OnceLock::new(),
            isl_blocks: OnceLock::new(),
            isl_tokens: OnceLock::new(),
            cached_tokens: OnceLock::new(),
            osl_tokens: AtomicU64::new(0),
            prefill_worker_id: OnceLock::new(),
            prefill_dp_rank: OnceLock::new(),
            decode_worker_id: OnceLock::new(),
            decode_dp_rank: OnceLock::new(),
            prefill_worker_type: OnceLock::new(),
            decode_worker_type: OnceLock::new(),
            phase: Mutex::new(RequestPhase::Aggregated),
            phase_semaphore: Arc::new(Semaphore::new(1)),
            tokenize_latency: OnceLock::new(),
            detokenize_total_ns: AtomicU64::new(0),
            detokenize_count: AtomicU64::new(0),
            router_queue_depth: OnceLock::new(),
            prefill_complete_time: OnceLock::new(),
        }
    }

    /// Record when prefill started. Returns true if this was the first call.
    pub fn record_prefill_start(&self) -> bool {
        self.prefill_start_time.set(Instant::now()).is_ok()
    }

    pub fn record_first_token(&self) {
        let _ = self.first_token_time.set(Instant::now());
    }

    /// Record when the decode worker produced its first token.
    /// Used for KV transfer latency estimation in disaggregated serving.
    pub fn record_decode_first_token(&self) {
        let _ = self.decode_first_token_time.set(Instant::now());
    }

    pub fn record_finish(&self) {
        *self.request_finish_time.lock() = Some(Instant::now());
    }

    /// Record KV cache hit information. Returns true if this was the first call.
    pub fn record_kv_hit(&self, overlap_blocks: u32, isl_blocks: usize) -> bool {
        let overlap_set = self.kv_overlap_blocks.set(overlap_blocks).is_ok();
        let isl_set = self.isl_blocks.set(isl_blocks).is_ok();
        overlap_set && isl_set
    }

    /// Record input sequence length in tokens and cached token count when known.
    pub fn record_isl(&self, isl_tokens: usize, cached_tokens: Option<usize>) {
        let _ = self.isl_tokens.set(isl_tokens);
        if let Some(cached_tokens) = cached_tokens {
            let _ = self.cached_tokens.set(cached_tokens);
        }
    }

    pub fn isl_tokens(&self) -> Option<usize> {
        self.isl_tokens.get().copied()
    }

    pub fn cached_tokens(&self) -> Option<usize> {
        self.cached_tokens.get().copied()
    }

    /// Record current output sequence length in tokens. Updated at each output block boundary.
    pub fn record_osl(&self, osl: usize) {
        self.osl_tokens.store(osl as u64, Ordering::Relaxed);
    }

    pub fn osl_tokens(&self) -> u64 {
        self.osl_tokens.load(Ordering::Relaxed)
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
        let first_token = self.first_token_time.get()?;
        Some(
            first_token
                .duration_since(self.request_received)
                .as_secs_f64()
                * 1000.0,
        )
    }

    pub fn total_time_ms(&self) -> Option<f64> {
        let finish = (*self.request_finish_time.lock())?;
        Some(finish.duration_since(self.request_received).as_secs_f64() * 1000.0)
    }

    /// Average inter-token latency in milliseconds.
    /// Computed as (finish_time - first_token_time) / (osl - 1).
    /// Returns None if fewer than 2 output tokens or times not recorded.
    pub fn avg_itl_ms(&self) -> Option<f64> {
        let first_token = *self.first_token_time.get()?;
        let finish = (*self.request_finish_time.lock())?;
        let osl = self.osl_tokens.load(Ordering::Relaxed);
        if osl < 2 {
            return None;
        }
        let decode_duration = finish.duration_since(first_token).as_secs_f64() * 1000.0;
        Some(decode_duration / (osl - 1) as f64)
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

    /// Set the request phase and return a permit that blocks subsequent phase changes.
    ///
    /// The returned permit must be dropped to allow the next `set_phase` call to proceed.
    /// In the bootstrap optimization path, the permit is held and passed to the spawned
    /// prefill task, ensuring routing completes before the phase changes.
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

    fn record_once_u64(slot: &OnceLock<u64>, value: u64, field_name: &'static str) {
        if let Some(existing) = slot.get() {
            if *existing != value {
                tracing::error!(
                    field = field_name,
                    existing = *existing,
                    new = value,
                    "Conflicting request tracker write"
                );
            }
            return;
        }
        let _ = slot.set(value);
    }

    fn record_once_u32(slot: &OnceLock<u32>, value: u32, field_name: &'static str) {
        if let Some(existing) = slot.get() {
            if *existing != value {
                tracing::error!(
                    field = field_name,
                    existing = *existing,
                    new = value,
                    "Conflicting request tracker write"
                );
            }
            return;
        }
        let _ = slot.set(value);
    }

    fn record_once_worker_type(
        slot: &OnceLock<&'static str>,
        value: &'static str,
        field_name: &'static str,
    ) {
        if let Some(existing) = slot.get() {
            if *existing != value {
                tracing::error!(
                    field = field_name,
                    existing = *existing,
                    new = value,
                    "Conflicting request tracker write"
                );
            }
            return;
        }
        let _ = slot.set(value);
    }

    fn record_prefill_worker(
        &self,
        instance_id: u64,
        dp_rank: Option<u32>,
        worker_type: &'static str,
    ) {
        Self::record_once_u64(&self.prefill_worker_id, instance_id, "prefill_worker_id");
        if let Some(rank) = dp_rank {
            Self::record_once_u32(&self.prefill_dp_rank, rank, "prefill_dp_rank");
        }
        Self::record_once_worker_type(
            &self.prefill_worker_type,
            worker_type,
            "prefill_worker_type",
        );
    }

    fn record_decode_worker(
        &self,
        instance_id: u64,
        dp_rank: Option<u32>,
        worker_type: &'static str,
    ) {
        Self::record_once_u64(&self.decode_worker_id, instance_id, "decode_worker_id");
        if let Some(rank) = dp_rank {
            Self::record_once_u32(&self.decode_dp_rank, rank, "decode_dp_rank");
        }
        Self::record_once_worker_type(&self.decode_worker_type, worker_type, "decode_worker_type");
    }

    /// Record worker ID, optional DP rank, and worker type based on the current phase.
    ///
    /// Worker ID and type are recorded as soon as they are known. DP rank is recorded only
    /// when it is concrete, allowing the unresolved rank to remain unset until later.
    pub fn record_worker(&self, instance_id: u64, dp_rank: Option<u32>, worker_type: &'static str) {
        match self.phase() {
            RequestPhase::Prefill => self.record_prefill_worker(instance_id, dp_rank, worker_type),
            RequestPhase::Decode => self.record_decode_worker(instance_id, dp_rank, worker_type),
            RequestPhase::Aggregated => {
                self.record_prefill_worker(instance_id, dp_rank, worker_type);
                self.record_decode_worker(instance_id, dp_rank, worker_type);
            }
        }
    }

    pub fn record_tokenize_latency(&self, l: Duration) {
        let _ = self.tokenize_latency.set(l);
    }

    pub fn tokenize_latency(&self) -> Option<Duration> {
        self.tokenize_latency.get().copied()
    }

    pub fn record_detokenize_latency(&self, l: Duration) {
        // u128 -> u64 is safe because max u64 in nanos is over 500 years
        let delta_ns = u64::try_from(l.as_nanos()).unwrap_or(u64::MAX);
        // On an x86 system these atomics are very cheap
        let _ = self.detokenize_total_ns.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            // Saturating add to avoid wrapping to a nonsensical average on overflow.
            |current| Some(current.saturating_add(delta_ns)),
        );
        self.detokenize_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn detokenize_total_latency(&self) -> Option<Duration> {
        let total_ns = self.detokenize_total_ns.load(Ordering::Relaxed);
        let count = self.detokenize_count.load(Ordering::Relaxed);
        if count == 0 {
            // We recorded no observations
            None
        } else {
            Some(Duration::from_nanos(total_ns))
        }
    }

    pub fn detokenize_count(&self) -> u64 {
        self.detokenize_count.load(Ordering::Relaxed)
    }

    /// Record router scheduler queue depth at routing time.
    pub fn record_router_queue_depth(&self, depth: usize) {
        let _ = self.router_queue_depth.set(depth);
    }

    /// Get the router scheduler queue depth recorded at routing time.
    pub fn router_queue_depth(&self) -> Option<usize> {
        self.router_queue_depth.get().copied()
    }

    /// Record when the prefill result was received by the router.
    /// Returns true if this was the first call (OnceLock first-write-wins).
    pub fn record_prefill_complete(&self) -> bool {
        self.prefill_complete_time.set(Instant::now()).is_ok()
    }

    /// Upper-bound estimation of KV cache transfer latency in seconds.
    /// Computed as `decode_first_token_time - prefill_complete_time`, which captures:
    /// router dispatch overhead + network + KV transfer (NIXL) + one decode forward pass.
    /// Works for all disaggregated paths (original and bootstrap).
    /// Returns None if either timestamp was not recorded.
    pub fn kv_transfer_estimated_latency_secs(&self) -> Option<f64> {
        let complete = *self.prefill_complete_time.get()?;
        let first_tok = *self.decode_first_token_time.get()?;
        Some(first_tok.saturating_duration_since(complete).as_secs_f64())
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
        self.decode_worker_id.get().copied()
    }

    /// Get the decode DP rank if recorded.
    pub fn decode_dp_rank(&self) -> Option<u32> {
        self.decode_dp_rank.get().copied()
    }

    /// Get the prefill worker ID if recorded.
    pub fn prefill_worker_id(&self) -> Option<u64> {
        self.prefill_worker_id.get().copied()
    }

    /// Get the prefill DP rank if recorded.
    pub fn prefill_dp_rank(&self) -> Option<u32> {
        self.prefill_dp_rank.get().copied()
    }

    /// Get the prefill worker type if recorded.
    pub fn prefill_worker_type(&self) -> Option<&'static str> {
        self.prefill_worker_type.get().copied()
    }

    /// Get the decode worker type if recorded.
    pub fn decode_worker_type(&self) -> Option<&'static str> {
        self.decode_worker_type.get().copied()
    }

    /// Write TTFT and ISL to per-worker last gauges using prefill worker labels.
    /// Called from the Python binding path on first token.
    pub fn observe_first_token_gauges(&self) {
        let Some(worker_id) = self.prefill_worker_id() else {
            return;
        };
        let worker_id_str = worker_id.to_string();
        let dp_rank_str = self
            .prefill_dp_rank()
            .map_or(UNSET_DP_RANK_LABEL.to_string(), |r| r.to_string());
        let worker_type = self.prefill_worker_type().unwrap_or(WORKER_TYPE_PREFILL);
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];

        if let Some(ttft) = self.ttft_ms() {
            WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE
                .with_label_values(labels)
                .set(ttft / 1000.0);
        }
        if let Some(isl) = self.isl_tokens() {
            WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE
                .with_label_values(labels)
                .set(isl as i64);
        }
    }

    /// Write avg ITL to per-worker last gauge using decode worker labels.
    /// Called at each output block boundary and from the Python binding path.
    pub fn observe_finish_gauges(&self) {
        let Some(worker_id) = self.decode_worker_id() else {
            return;
        };
        let worker_id_str = worker_id.to_string();
        let dp_rank_str = self
            .decode_dp_rank()
            .map_or(UNSET_DP_RANK_LABEL.to_string(), |r| r.to_string());
        let worker_type = self.decode_worker_type().unwrap_or(WORKER_TYPE_DECODE);
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];

        if let Some(avg_itl) = self.avg_itl_ms() {
            WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE
                .with_label_values(labels)
                .set(avg_itl / 1000.0);
        }
    }

    pub fn get_timing_info(&self) -> TimingInfo {
        TimingInfo {
            request_received_ms: self.request_received_epoch_ms,
            prefill_wait_time_ms: self.prefill_wait_time_ms(),
            prefill_time_ms: self.prefill_time_ms(),
            ttft_ms: self.ttft_ms(),
            total_time_ms: self.total_time_ms(),
            kv_hit_rate: self.kv_hit_rate(),
            router_queue_depth: self.router_queue_depth(),
            kv_transfer_estimated_latency_ms: self
                .kv_transfer_estimated_latency_secs()
                .map(|s| s * 1000.0),
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

    /// Number of requests pending in the router scheduler queue at routing time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_queue_depth: Option<usize>,

    /// Upper-bound estimation of KV cache transfer latency in milliseconds (disaggregated only).
    /// Measured as decode_first_token_time - prefill_complete_time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_estimated_latency_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_record_isl_osl() {
        let tracker = RequestTracker::new();

        tracker.record_isl(512, Some(256));
        assert_eq!(tracker.isl_tokens(), Some(512));
        assert_eq!(tracker.cached_tokens(), Some(256));

        tracker.record_osl(100);
        assert_eq!(tracker.osl_tokens(), 100);
    }

    #[test]
    fn test_ttft_ms() {
        let tracker = RequestTracker::new();
        thread::sleep(Duration::from_millis(10));
        tracker.record_first_token();

        let ttft = tracker.ttft_ms().unwrap();
        assert!(ttft >= 5.0, "TTFT should be at least 5ms, got {ttft}");
    }

    #[test]
    fn test_ttft_ms_none_before_first_token() {
        let tracker = RequestTracker::new();
        assert!(tracker.ttft_ms().is_none());
    }

    #[test]
    fn test_avg_itl_ms() {
        let tracker = RequestTracker::new();
        tracker.record_first_token();
        thread::sleep(Duration::from_millis(20));
        tracker.record_osl(11); // 11 tokens => 10 inter-token gaps
        tracker.record_finish();

        let itl = tracker.avg_itl_ms().unwrap();
        assert!(itl > 0.0, "avg ITL should be positive, got {itl}");
    }

    #[test]
    fn test_avg_itl_ms_none_with_single_token() {
        let tracker = RequestTracker::new();
        tracker.record_first_token();
        tracker.record_osl(1);
        tracker.record_finish();

        assert!(
            tracker.avg_itl_ms().is_none(),
            "avg ITL should be None with < 2 output tokens"
        );
    }

    #[test]
    fn test_kv_hit_rate() {
        let tracker = RequestTracker::new();
        tracker.record_kv_hit(3, 10);

        let rate = tracker.kv_hit_rate().unwrap();
        assert!(
            (rate - 0.3).abs() < f64::EPSILON,
            "KV hit rate should be 0.3, got {rate}"
        );
    }

    #[test]
    fn test_kv_hit_rate_zero_isl() {
        let tracker = RequestTracker::new();
        tracker.record_kv_hit(0, 0);
        assert!(
            tracker.kv_hit_rate().is_none(),
            "KV hit rate should be None when isl_blocks is 0"
        );
    }

    #[test]
    fn test_total_time_ms() {
        let tracker = RequestTracker::new();
        thread::sleep(Duration::from_millis(10));
        tracker.record_finish();

        let total = tracker.total_time_ms().unwrap();
        assert!(
            total >= 5.0,
            "total time should be at least 5ms, got {total}"
        );
    }

    #[test]
    fn test_router_queue_depth() {
        let tracker = RequestTracker::new();
        assert!(tracker.router_queue_depth().is_none());

        tracker.record_router_queue_depth(42);
        assert_eq!(tracker.router_queue_depth(), Some(42));

        // OnceLock: second write is ignored
        tracker.record_router_queue_depth(99);
        assert_eq!(tracker.router_queue_depth(), Some(42));

        let timing = tracker.get_timing_info();
        assert_eq!(timing.router_queue_depth, Some(42));
    }

    #[test]
    fn test_observe_first_token_gauges_no_panic_without_worker() {
        let tracker = RequestTracker::new();
        tracker.record_first_token();
        tracker.record_isl(100, Some(50));
        // No worker recorded — should return early without panic
        tracker.observe_first_token_gauges();
    }

    #[test]
    fn test_observe_finish_gauges_no_panic_without_worker() {
        let tracker = RequestTracker::new();
        tracker.record_first_token();
        tracker.record_osl(10);
        tracker.record_finish();
        // No worker recorded — should return early without panic
        tracker.observe_finish_gauges();
    }

    #[test]
    fn test_observe_first_token_gauges_with_worker() {
        let tracker = RequestTracker::new();
        tracker.record_worker(42, Some(0), WORKER_TYPE_PREFILL);
        thread::sleep(Duration::from_millis(5));
        tracker.record_first_token();
        tracker.record_isl(256, Some(128));

        tracker.observe_first_token_gauges();

        let labels = &["42", "0", WORKER_TYPE_PREFILL];
        let ttft_val = WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE
            .with_label_values(labels)
            .get();
        assert!(
            ttft_val > 0.0,
            "TTFT gauge should be positive after observe, got {ttft_val}"
        );

        let isl_val = WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE
            .with_label_values(labels)
            .get();
        assert_eq!(isl_val, 256, "ISL gauge should be 256, got {isl_val}");
    }

    #[test]
    fn test_observe_finish_gauges_with_worker() {
        let tracker = RequestTracker::new();
        tracker.record_worker(99, Some(1), WORKER_TYPE_DECODE);
        tracker.record_first_token();
        thread::sleep(Duration::from_millis(10));
        tracker.record_osl(5);
        tracker.record_finish();

        tracker.observe_finish_gauges();

        let labels = &["99", "1", WORKER_TYPE_DECODE];
        let itl_val = WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE
            .with_label_values(labels)
            .get();
        assert!(
            itl_val > 0.0,
            "ITL gauge should be positive after observe, got {itl_val}"
        );
    }

    #[test]
    fn test_kv_transfer_estimated_latency() {
        let tracker = RequestTracker::new();
        // Before any timestamps: returns None
        assert!(tracker.kv_transfer_estimated_latency_secs().is_none());

        tracker.record_prefill_complete();
        thread::sleep(Duration::from_millis(10));
        tracker.record_decode_first_token();

        let latency = tracker.kv_transfer_estimated_latency_secs().unwrap();
        assert!(
            latency >= 0.005,
            "latency should be at least 5ms, got {latency}"
        );
    }

    #[test]
    fn test_kv_transfer_estimated_latency_none_without_first_token() {
        let tracker = RequestTracker::new();
        tracker.record_prefill_complete();
        assert!(
            tracker.kv_transfer_estimated_latency_secs().is_none(),
            "Should return None when decode_first_token_time is not set"
        );
    }

    #[test]
    fn test_kv_transfer_estimated_latency_none_without_prefill_complete() {
        let tracker = RequestTracker::new();
        tracker.record_decode_first_token();
        assert!(
            tracker.kv_transfer_estimated_latency_secs().is_none(),
            "Should return None when prefill_complete_time is not set"
        );
    }

    #[test]
    fn test_kv_transfer_estimated_latency_oncelock_first_write_wins() {
        let tracker = RequestTracker::new();
        assert!(tracker.record_prefill_complete()); // first call returns true
        assert!(!tracker.record_prefill_complete()); // second call returns false (OnceLock)
    }

    #[test]
    fn test_timing_info_includes_kv_transfer_estimated_latency() {
        let tracker = RequestTracker::new();
        tracker.record_prefill_complete();
        thread::sleep(Duration::from_millis(10));
        tracker.record_decode_first_token();

        let info = tracker.get_timing_info();
        let latency_ms = info
            .kv_transfer_estimated_latency_ms
            .expect("should be Some");
        assert!(
            latency_ms >= 5.0,
            "latency should be at least 5ms, got {latency_ms}"
        );
    }

    #[test]
    fn test_timing_info_kv_transfer_estimated_latency_none_in_aggregated() {
        let tracker = RequestTracker::new();
        // No record_prefill_complete / record_first_token called
        let info = tracker.get_timing_info();
        assert!(
            info.kv_transfer_estimated_latency_ms.is_none(),
            "Should be None in aggregated mode (no timestamps recorded)"
        );
    }

    /// Reproduces the original bug where kv_transfer_estimated_latency was always 0.
    ///
    /// The bug: in disaggregated serving, both `record_first_token()` and
    /// `record_prefill_complete()` were called during the prefill phase with
    /// ~nanoseconds between them, and the latency was computed as
    /// `first_token_time - prefill_complete_time`. Since `first_token_time`
    /// was set *before* `prefill_complete_time`, `saturating_duration_since`
    /// clamped the negative duration to zero.
    ///
    /// The fix: use a separate `decode_first_token_time` field that is only
    /// recorded during the Decode phase, giving a meaningful time gap.
    #[test]
    fn test_kv_transfer_latency_bug_prefill_timestamps_are_zero() {
        let tracker = RequestTracker::new();

        // Simulate the buggy prefill-phase sequence:
        // 1. RequestGuard::on_item() calls record_first_token() during prefill
        tracker.record_first_token();
        // 2. execute_prefill() calls record_prefill_complete() immediately after
        tracker.record_prefill_complete();

        // The OLD computation (first_token_time - prefill_complete_time) would be 0
        // because first_token_time < prefill_complete_time chronologically,
        // and saturating_duration_since clamps to zero.
        let first_tok = *tracker.first_token_time.get().unwrap();
        let complete = *tracker.prefill_complete_time.get().unwrap();
        let old_latency = first_tok.saturating_duration_since(complete).as_secs_f64();
        assert_eq!(
            old_latency, 0.0,
            "Old computation should produce exactly 0.0 (the bug), got {old_latency}"
        );

        // The FIXED computation uses decode_first_token_time which hasn't been set
        // yet, so it correctly returns None (no decode phase has run).
        assert!(
            tracker.kv_transfer_estimated_latency_secs().is_none(),
            "Fixed metric should be None when decode hasn't started"
        );

        // Now simulate the decode phase producing its first token after a delay.
        thread::sleep(Duration::from_millis(10));
        tracker.record_decode_first_token();

        // The FIXED computation (decode_first_token_time - prefill_complete_time)
        // captures the actual KV transfer + decode startup latency.
        let fixed_latency = tracker.kv_transfer_estimated_latency_secs().unwrap();
        assert!(
            fixed_latency >= 0.005,
            "Fixed latency should be >= 5ms (actual KV transfer time), got {fixed_latency}"
        );
    }

    /// Verifies that the decode phase's record_first_token() is rejected by OnceLock
    /// (since prefill already set it), but record_decode_first_token() succeeds.
    #[test]
    fn test_decode_first_token_not_blocked_by_prefill_oncelock() {
        let tracker = RequestTracker::new();

        // Prefill phase sets first_token_time
        tracker.record_first_token();
        let prefill_first_tok = *tracker.first_token_time.get().unwrap();

        thread::sleep(Duration::from_millis(5));

        // Decode phase: record_first_token() is rejected (OnceLock already set)
        tracker.record_first_token();
        let still_prefill_tok = *tracker.first_token_time.get().unwrap();
        assert_eq!(
            prefill_first_tok, still_prefill_tok,
            "first_token_time should be unchanged (OnceLock rejected decode's write)"
        );

        // But record_decode_first_token() succeeds on its own OnceLock
        tracker.record_decode_first_token();
        let decode_tok = *tracker.decode_first_token_time.get().unwrap();
        assert!(
            decode_tok > prefill_first_tok,
            "decode_first_token_time should be later than first_token_time"
        );
    }

    #[test]
    fn test_timing_info_kv_transfer_estimated_latency_serialization() {
        let tracker = RequestTracker::new();
        // When not set, the field should be omitted from JSON (skip_serializing_if)
        let info = tracker.get_timing_info();
        let json = serde_json::to_string(&info).unwrap();
        assert!(
            !json.contains("kv_transfer_estimated_latency_ms"),
            "None field should be omitted from JSON, got: {json}"
        );

        // When set, it should appear
        let tracker2 = RequestTracker::new();
        tracker2.record_prefill_complete();
        tracker2.record_decode_first_token();
        let info2 = tracker2.get_timing_info();
        let json2 = serde_json::to_string(&info2).unwrap();
        assert!(
            json2.contains("kv_transfer_estimated_latency_ms"),
            "Set field should appear in JSON, got: {json2}"
        );
    }
}
