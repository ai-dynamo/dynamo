// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared utilities for benchmark binaries.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Latency statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub min: Duration,
    pub max: Duration,
    pub avg: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub throughput_ops_sec: f64,
}

impl LatencyStats {
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();
        let n = sorted.len();
        let total: Duration = sorted.iter().sum();
        let avg = total / n as u32;

        Some(Self {
            min: sorted[0],
            max: sorted[n - 1],
            avg,
            p50: sorted[n / 2],
            p95: sorted[n * 95 / 100],
            p99: sorted[n * 99 / 100],
            throughput_ops_sec: n as f64 / total.as_secs_f64(),
        })
    }

    /// Print formatted latency statistics to stdout.
    pub fn print(&self, operation: &str, blocks_per_op: usize) {
        println!("\n{} Latency Statistics:", operation);
        println!("  min:  {:>12?}", self.min);
        println!("  avg:  {:>12?}", self.avg);
        println!("  p50:  {:>12?}", self.p50);
        println!("  p95:  {:>12?}", self.p95);
        println!("  p99:  {:>12?}", self.p99);
        println!("  max:  {:>12?}", self.max);
        println!("  throughput: {:.2} ops/sec", self.throughput_ops_sec);
        println!(
            "  throughput: {:.2} blocks/sec",
            self.throughput_ops_sec * blocks_per_op as f64
        );
    }
}

// ---------------------------------------------------------------------------
// Time-bucketed latency statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct TimeBucketStats {
    pub bucket_start_sec: u64,
    pub bucket_end_sec: u64,
    pub count: usize,
    pub latency_min_us: u64,
    pub latency_p50_us: u64,
    pub latency_p95_us: u64,
    pub latency_max_us: u64,
}

/// Compute per-bucket latency statistics.
///
/// Each item is a `(latency, completion_time)` pair where `completion_time`
/// is relative to the measurement start.
pub fn compute_time_bucket_stats(
    items: &[(Duration, Duration)],
    bucket_size_secs: u64,
) -> Vec<TimeBucketStats> {
    if items.is_empty() || bucket_size_secs == 0 {
        return Vec::new();
    }

    let max_completion = items
        .iter()
        .map(|&(_, ct)| ct)
        .max()
        .unwrap_or(Duration::ZERO);

    let num_buckets = (max_completion.as_secs() / bucket_size_secs) + 1;
    let mut bucket_latencies: Vec<Vec<Duration>> = vec![Vec::new(); num_buckets as usize];

    for &(latency, completion_time) in items {
        let bucket_idx = (completion_time.as_secs() / bucket_size_secs) as usize;
        if bucket_idx < bucket_latencies.len() {
            bucket_latencies[bucket_idx].push(latency);
        }
    }

    bucket_latencies
        .iter()
        .enumerate()
        .filter_map(|(idx, latencies)| {
            if latencies.is_empty() {
                return None;
            }

            let stats = LatencyStats::from_durations(latencies)?;
            Some(TimeBucketStats {
                bucket_start_sec: idx as u64 * bucket_size_secs,
                bucket_end_sec: (idx as u64 + 1) * bucket_size_secs,
                count: latencies.len(),
                latency_min_us: stats.min.as_micros() as u64,
                latency_p50_us: stats.p50.as_micros() as u64,
                latency_p95_us: stats.p95.as_micros() as u64,
                latency_max_us: stats.max.as_micros() as u64,
            })
        })
        .collect()
}

pub fn print_time_bucket_report(buckets: &[TimeBucketStats]) {
    if buckets.is_empty() {
        println!("  No time bucket data available");
        return;
    }

    println!(
        "  {:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "Time(s)", "Count", "Min(ms)", "P50(ms)", "P95(ms)", "Max(ms)"
    );
    println!("  {}", "-".repeat(68));

    for bucket in buckets {
        println!(
            "  {:>3}-{:<4} {:>8} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            bucket.bucket_start_sec,
            bucket.bucket_end_sec,
            bucket.count,
            bucket.latency_min_us as f64 / 1000.0,
            bucket.latency_p50_us as f64 / 1000.0,
            bucket.latency_p95_us as f64 / 1000.0,
            bucket.latency_max_us as f64 / 1000.0,
        );
    }
}

// ---------------------------------------------------------------------------
// Latency sample (for raw JSON export)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct LatencySample {
    pub latency_us: u64,
    pub completion_time_ms: u64,
    pub success: bool,
}

// ---------------------------------------------------------------------------
// OpenAI-style chat types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// Model auto-detection
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

pub async fn fetch_model_name(frontend_url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/models", frontend_url);

    println!("  Auto-detecting model from {}...", url);

    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to frontend /v1/models endpoint")?;

    if !response.status().is_success() {
        anyhow::bail!("Models endpoint returned status: {}", response.status());
    }

    let models: ModelsResponse = response
        .json()
        .await
        .context("Failed to parse models response")?;

    match models.data.len() {
        0 => anyhow::bail!("No models found at endpoint. Is a backend running?"),
        1 => {
            let model_id = models.data[0].id.clone();
            println!("  Auto-detected model: {}", model_id);
            Ok(model_id)
        }
        n => {
            println!("  Multiple models available ({}):", n);
            for m in &models.data {
                println!("    - {}", m.id);
            }
            anyhow::bail!("Multiple models available. Please specify --model explicitly.")
        }
    }
}
