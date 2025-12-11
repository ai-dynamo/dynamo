// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request timing tracker for capturing request lifecycle metrics.
//!
//! This module provides [`RequestTimingTracker`] for tracking timing information
//! that can be returned to clients via the `nvext` response field.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Per-request timing tracker.
///
/// Captures timing information throughout the request lifecycle:
/// - `request_received`: When the request was received
/// - `first_token_time`: When the first token was generated (set once via OnceLock)
/// - `request_finish_time`: When the request finished (set once via OnceLock)
///
/// The `OnceLock` fields ensure that timing values are set exactly once,
/// which is important for disaggregated serving where the "first token"
/// might appear multiple times.
pub struct RequestTimingTracker {
    /// When the request was received (monotonic clock for duration calculations)
    request_received: Instant,

    /// When the request was received (wall clock time as epoch milliseconds)
    request_received_epoch_ms: u64,

    /// When the first token was generated - set once via OnceLock
    first_token_time: OnceLock<Instant>,

    /// When the request finished - set once via OnceLock
    request_finish_time: OnceLock<Instant>,
}

impl RequestTimingTracker {
    /// Create a new timing tracker, capturing the current time as request received.
    pub fn new() -> Self {
        let now = Instant::now();
        let epoch_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        RequestTimingTracker {
            request_received: now,
            request_received_epoch_ms: epoch_ms,
            first_token_time: OnceLock::new(),
            request_finish_time: OnceLock::new(),
        }
    }

    pub fn record_first_token(&self) -> bool {
        self.first_token_time.set(Instant::now()).is_ok()
    }

    pub fn record_finish(&self) -> bool {
        self.request_finish_time.set(Instant::now()).is_ok()
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

    pub fn get_timing_info(&self) -> TimingInfo {
        TimingInfo {
            request_received_ms: self.request_received_epoch_ms,
            ttft_ms: self.ttft_ms(),
            total_time_ms: self.total_time_ms(),
        }
    }
}

impl Default for RequestTimingTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing information for response injection.
///
/// This struct is serialized and included in the response's `nvext` field
/// when the client requests timing information via `extra_fields: ["timing"]`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TimingInfo {
    /// When the request was received (epoch milliseconds)
    pub request_received_ms: u64,

    /// Time to first token in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,

    /// Total request time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<f64>,
}
