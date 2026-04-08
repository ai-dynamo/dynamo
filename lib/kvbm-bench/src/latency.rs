// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Latency statistics computation.

use serde::{Deserialize, Serialize};

/// Latency statistics computed from a set of timing samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: usize,
    pub min_us: u64,
    pub avg_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub p999_us: u64,
    pub max_us: u64,
    /// Cumulative service time divided by count — valid only for serial runs.
    /// Use wall-clock elapsed from the runner for true throughput.
    pub cumulative_latency_ops_sec: f64,
    /// Sum of all per-operation latencies in seconds.
    pub cumulative_latency_secs: f64,
}

impl LatencyStats {
    /// Compute latency statistics from a slice of microsecond samples.
    ///
    /// Returns `None` if `samples` is empty.
    pub fn from_micros(samples: &[u64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        let count = sorted.len();
        let min_us = sorted[0];
        let max_us = sorted[count - 1];
        let sum: u64 = sorted.iter().sum();
        let avg_us = sum / count as u64;

        let p50_us = percentile(&sorted, 50.0);
        let p95_us = percentile(&sorted, 95.0);
        let p99_us = percentile(&sorted, 99.0);
        let p999_us = percentile(&sorted, 99.9);

        let cumulative_latency_secs = sum as f64 / 1_000_000.0;
        let cumulative_latency_ops_sec = if cumulative_latency_secs > 0.0 {
            count as f64 / cumulative_latency_secs
        } else {
            0.0
        };

        Some(Self {
            count,
            min_us,
            avg_us,
            p50_us,
            p95_us,
            p99_us,
            p999_us,
            max_us,
            cumulative_latency_ops_sec,
            cumulative_latency_secs,
        })
    }

    /// Compute latency statistics from a slice of `Duration` samples.
    ///
    /// Returns `None` if `samples` is empty.
    pub fn from_durations(samples: &[std::time::Duration]) -> Option<Self> {
        let micros: Vec<u64> = samples.iter().map(|d| d.as_micros() as u64).collect();
        Self::from_micros(&micros)
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
