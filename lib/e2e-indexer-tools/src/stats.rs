// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sharded latency histograms and rate counters for periodic JSON reports.

use std::sync::atomic::{AtomicU64, Ordering};

use hdrhistogram::Histogram;
use parking_lot::Mutex;
use serde_json::{Value, json};

const SHARDS: usize = 64;
/// One hour in microseconds: the largest recorded latency.
const MAX_US: u64 = 3_600_000_000;

fn histogram() -> Histogram<u64> {
    Histogram::new_with_bounds(1, MAX_US, 3).expect("valid histogram bounds")
}

/// Microsecond latencies with an interval view (reset per report) and a cumulative view.
pub struct Latency {
    shards: Vec<Mutex<(Histogram<u64>, Histogram<u64>)>>,
}

impl Default for Latency {
    fn default() -> Self {
        Self {
            shards: (0..SHARDS)
                .map(|_| Mutex::new((histogram(), histogram())))
                .collect(),
        }
    }
}

impl Latency {
    pub fn record(&self, shard: usize, us: u64) {
        let value = us.clamp(1, MAX_US);
        let mut guard = self.shards[shard % SHARDS].lock();
        guard.0.saturating_record(value);
        guard.1.saturating_record(value);
    }

    /// Drain the interval view.
    pub fn take_interval(&self) -> Histogram<u64> {
        let mut merged = histogram();
        for shard in &self.shards {
            let mut guard = shard.lock();
            merged.add(&guard.0).expect("same bounds");
            guard.0.reset();
        }
        merged
    }

    pub fn cumulative(&self) -> Histogram<u64> {
        let mut merged = histogram();
        for shard in &self.shards {
            merged.add(&shard.lock().1).expect("same bounds");
        }
        merged
    }
}

pub fn summarize_ms(histogram: &Histogram<u64>) -> Value {
    if histogram.is_empty() {
        return json!({ "count": 0 });
    }
    let ms = |us: u64| us as f64 / 1e3;
    json!({
        "count": histogram.len(),
        "p50_ms": ms(histogram.value_at_quantile(0.50)),
        "p90_ms": ms(histogram.value_at_quantile(0.90)),
        "p99_ms": ms(histogram.value_at_quantile(0.99)),
        "p999_ms": ms(histogram.value_at_quantile(0.999)),
        "max_ms": ms(histogram.max()),
        "mean_ms": histogram.mean() / 1e3,
    })
}

/// A counter read as an interval delta and a running total.
#[derive(Default)]
pub struct Counter {
    total: AtomicU64,
    reported: AtomicU64,
}

impl Counter {
    pub fn add(&self, value: u64) {
        self.total.fetch_add(value, Ordering::Relaxed);
    }

    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Delta since the previous call.
    pub fn take_interval(&self) -> u64 {
        let total = self.total();
        total - self.reported.swap(total, Ordering::Relaxed)
    }
}
