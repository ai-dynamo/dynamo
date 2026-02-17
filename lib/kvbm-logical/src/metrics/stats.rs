// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Toggleable periodic sampler for computing rates, gradients, and hit ratios.
//!
//! The raw counters/gauges in [`BlockPoolMetrics`] are always active (zero-cost atomics).
//! This `StatsCollector` is the optional layer that computes derived statistics.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::pool_metrics::{BlockPoolMetrics, MetricsSnapshot};

/// Configuration for the stats collector.
#[derive(Debug, Clone)]
pub struct StatsConfig {
    /// Maximum number of snapshots to retain in the ring buffer.
    pub window_size: usize,
    /// Interval between periodic samples when using `spawn()`.
    pub sample_interval: Duration,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            window_size: 60,
            sample_interval: Duration::from_secs(1),
        }
    }
}

/// Derived statistics computed from counter deltas over time.
#[derive(Debug, Clone, Copy)]
pub struct StatsSnapshot {
    /// Allocations per second.
    pub allocation_rate: f64,
    /// Evictions per second.
    pub eviction_rate: f64,
    /// Ratio of blocks returned to hashes requested in match_blocks.
    pub match_hit_rate: f64,
    /// Ratio of blocks returned to hashes requested in scan_matches.
    pub scan_hit_rate: f64,
    /// Rate of change of allocation_rate (d(alloc_rate)/dt).
    pub allocation_gradient: f64,
    /// Rate of change of eviction_rate (d(eviction_rate)/dt).
    pub eviction_gradient: f64,
    /// Current inflight mutable blocks.
    pub inflight_mutable: i64,
    /// Current inflight immutable blocks.
    pub inflight_immutable: i64,
    /// Current reset pool size.
    pub reset_pool_size: i64,
    /// Current inactive pool size.
    pub inactive_pool_size: i64,
}

struct TimedEntry {
    timestamp: Instant,
    raw: MetricsSnapshot,
    stats: StatsSnapshot,
}

/// Toggleable periodic sampler that computes rates and gradients from raw metrics.
pub struct StatsCollector {
    enabled: AtomicBool,
    metrics: Arc<BlockPoolMetrics>,
    entries: RwLock<VecDeque<TimedEntry>>,
    config: StatsConfig,
}

impl StatsCollector {
    /// Create a new `StatsCollector` for the given metrics source.
    /// Disabled by default — call `set_enabled(true)` to activate.
    pub fn new(metrics: Arc<BlockPoolMetrics>, config: StatsConfig) -> Self {
        Self {
            enabled: AtomicBool::new(false),
            metrics,
            entries: RwLock::new(VecDeque::with_capacity(config.window_size)),
            config,
        }
    }

    /// Enable or disable the stats collector.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Whether the stats collector is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Manually take a sample, computing rates/gradients from the delta since the last snapshot.
    /// No-op if disabled.
    pub fn sample(&self) {
        if !self.is_enabled() {
            return;
        }

        let now = Instant::now();
        let raw = self.metrics.snapshot();

        let mut entries = self.entries.write().expect("entries lock poisoned");

        let stats = if let Some(prev) = entries.back() {
            let dt = now.duration_since(prev.timestamp).as_secs_f64();
            if dt > 0.0 {
                let alloc_rate = (raw.allocations - prev.raw.allocations) as f64 / dt;
                let eviction_rate = (raw.evictions - prev.raw.evictions) as f64 / dt;

                let match_hit_rate = {
                    let delta_req = raw.match_hashes_requested - prev.raw.match_hashes_requested;
                    let delta_ret = raw.match_blocks_returned - prev.raw.match_blocks_returned;
                    if delta_req > 0 {
                        delta_ret as f64 / delta_req as f64
                    } else {
                        0.0
                    }
                };

                let scan_hit_rate = {
                    let delta_req = raw.scan_hashes_requested - prev.raw.scan_hashes_requested;
                    let delta_ret = raw.scan_blocks_returned - prev.raw.scan_blocks_returned;
                    if delta_req > 0 {
                        delta_ret as f64 / delta_req as f64
                    } else {
                        0.0
                    }
                };

                let allocation_gradient = (alloc_rate - prev.stats.allocation_rate) / dt;
                let eviction_gradient = (eviction_rate - prev.stats.eviction_rate) / dt;

                StatsSnapshot {
                    allocation_rate: alloc_rate,
                    eviction_rate,
                    match_hit_rate,
                    scan_hit_rate,
                    allocation_gradient,
                    eviction_gradient,
                    inflight_mutable: raw.inflight_mutable,
                    inflight_immutable: raw.inflight_immutable,
                    reset_pool_size: raw.reset_pool_size,
                    inactive_pool_size: raw.inactive_pool_size,
                }
            } else {
                zero_stats(&raw)
            }
        } else {
            zero_stats(&raw)
        };

        entries.push_back(TimedEntry {
            timestamp: now,
            raw,
            stats,
        });

        while entries.len() > self.config.window_size {
            entries.pop_front();
        }
    }

    /// Get the latest computed stats snapshot.
    pub fn latest(&self) -> Option<StatsSnapshot> {
        let entries = self.entries.read().expect("entries lock poisoned");
        entries.back().map(|e| e.stats)
    }

    /// Get the last `n` stats snapshots (most recent last).
    pub fn window(&self, n: usize) -> Vec<StatsSnapshot> {
        let entries = self.entries.read().expect("entries lock poisoned");
        entries
            .iter()
            .rev()
            .take(n)
            .map(|e| e.stats)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Spawn a tokio task that periodically calls `sample()`.
    /// Returns a `JoinHandle` that can be used to abort the task.
    pub fn spawn(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = Arc::clone(self);
        let interval = this.config.sample_interval;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                this.sample();
            }
        })
    }
}

fn zero_stats(raw: &MetricsSnapshot) -> StatsSnapshot {
    StatsSnapshot {
        allocation_rate: 0.0,
        eviction_rate: 0.0,
        match_hit_rate: 0.0,
        scan_hit_rate: 0.0,
        allocation_gradient: 0.0,
        eviction_gradient: 0.0,
        inflight_mutable: raw.inflight_mutable,
        inflight_immutable: raw.inflight_immutable,
        reset_pool_size: raw.reset_pool_size,
        inactive_pool_size: raw.inactive_pool_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_by_default() {
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        let stats = StatsCollector::new(metrics, StatsConfig::default());

        assert!(!stats.is_enabled());
        stats.sample(); // should be a no-op
        assert!(stats.latest().is_none());
    }

    #[test]
    fn test_enable_and_sample() {
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        let stats = StatsCollector::new(metrics.clone(), StatsConfig::default());
        stats.set_enabled(true);

        // First sample — baseline
        stats.sample();
        let snap = stats.latest().expect("should have a snapshot");
        assert_eq!(snap.allocation_rate, 0.0);

        // Simulate some allocations
        metrics.inc_allocations(100);
        std::thread::sleep(Duration::from_millis(50));

        // Second sample — should compute rate
        stats.sample();
        let snap = stats.latest().expect("should have a snapshot");
        assert!(snap.allocation_rate > 0.0, "allocation rate should be > 0");
    }

    #[test]
    fn test_window() {
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        let config = StatsConfig {
            window_size: 5,
            ..Default::default()
        };
        let stats = StatsCollector::new(metrics.clone(), config);
        stats.set_enabled(true);

        for i in 0..10 {
            metrics.inc_allocations(i + 1);
            stats.sample();
            std::thread::sleep(Duration::from_millis(10));
        }

        let window = stats.window(3);
        assert_eq!(window.len(), 3);

        // Ring buffer should have at most 5 entries
        let all = stats.window(100);
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_hit_rate_computation() {
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        let stats = StatsCollector::new(metrics.clone(), StatsConfig::default());
        stats.set_enabled(true);

        stats.sample(); // baseline
        std::thread::sleep(Duration::from_millis(10));

        metrics.inc_match_hashes_requested(10);
        metrics.inc_match_blocks_returned(7);

        stats.sample();
        let snap = stats.latest().unwrap();
        assert!((snap.match_hit_rate - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_gradient_computation() {
        let metrics = Arc::new(BlockPoolMetrics::new("G1".to_string()));
        let stats = StatsCollector::new(metrics.clone(), StatsConfig::default());
        stats.set_enabled(true);

        // Sample 1: baseline
        stats.sample();
        std::thread::sleep(Duration::from_millis(50));

        // Sample 2: some allocations → positive rate
        metrics.inc_allocations(100);
        stats.sample();
        let snap2 = stats.latest().unwrap();
        assert!(snap2.allocation_rate > 0.0);

        std::thread::sleep(Duration::from_millis(50));

        // Sample 3: more allocations → rate changes → gradient
        metrics.inc_allocations(500);
        stats.sample();
        let snap3 = stats.latest().unwrap();
        // Gradient should be non-zero since rate changed
        // (exact value depends on timing, but should be positive since rate increased)
        assert!(
            snap3.allocation_gradient != 0.0 || snap3.allocation_rate > snap2.allocation_rate,
            "allocation gradient or rate should reflect increasing allocations"
        );
    }
}
