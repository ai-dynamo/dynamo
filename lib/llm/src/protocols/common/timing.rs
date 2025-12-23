// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request tracker for capturing request lifecycle metrics.
//!
//! This module provides [`RequestTracker`] for tracking timing and routing information
//! that can be returned to clients via the `nvext` response field.

use serde::{Deserialize, Serialize};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use utoipa::ToSchema;

use crate::protocols::openai::nvext::WorkerIdInfo;

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
///
/// The `OnceLock` fields ensure that values are set exactly once,
/// which is important for disaggregated serving where the "first token"
/// might appear multiple times.
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

    /// Prefill worker ID (for disaggregated serving) - set once via OnceLock
    prefill_worker_id: OnceLock<u64>,

    /// Decode worker ID - set once via OnceLock
    decode_worker_id: OnceLock<u64>,

    /// Request phase (Prefill/Decode/Aggregated)
    phase: Mutex<RequestPhase>,
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
            prefill_worker_id: OnceLock::new(),
            decode_worker_id: OnceLock::new(),
            phase: Mutex::new(RequestPhase::Aggregated),
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
        self.prefill_worker_id.set(id).is_ok()
    }

    /// Record the decode worker ID. Returns true if this was the first call.
    pub fn record_decode_worker(&self, id: u64) -> bool {
        self.decode_worker_id.set(id).is_ok()
    }

    /// Set the request phase. Can be called multiple times to update the phase.
    pub fn set_phase(&self, phase: RequestPhase) {
        *self.phase.lock().unwrap() = phase;
    }

    /// Get the current request phase.
    pub fn phase(&self) -> RequestPhase {
        *self.phase.lock().unwrap()
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

    /// Get worker ID information if any worker IDs have been recorded.
    pub fn get_worker_info(&self) -> Option<WorkerIdInfo> {
        let prefill = self.prefill_worker_id.get().copied();
        let decode = self.decode_worker_id.get().copied();

        if prefill.is_none() && decode.is_none() {
            return None;
        }

        Some(WorkerIdInfo {
            prefill_worker_id: prefill,
            decode_worker_id: decode,
        })
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
