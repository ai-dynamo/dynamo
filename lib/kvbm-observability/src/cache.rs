// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sliding-window cache hit-rate tracking for request-facing KVBM metrics.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
struct CacheStatsEntry {
    host_blocks: u64,
    disk_blocks: u64,
    object_blocks: u64,
    total_blocks: u64,
}

#[derive(Default, Clone, Copy)]
struct AggregatedStats {
    total_blocks_queried: u64,
    host_blocks_hit: u64,
    disk_blocks_hit: u64,
    object_blocks_hit: u64,
}

/// Sliding-window tracker for host, disk, and object hit rates.
pub struct CacheStatsTracker {
    max_recent_requests: usize,
    entries: Mutex<VecDeque<CacheStatsEntry>>,
    aggregated: Mutex<AggregatedStats>,
    last_log_time: Mutex<Instant>,
    log_interval: Duration,
    identifier: Option<String>,
    last_logged_values: Mutex<Option<(u64, u64, u64, u64)>>,
}

impl CacheStatsTracker {
    pub fn new(
        max_recent_requests: usize,
        log_interval: Duration,
        identifier: Option<String>,
    ) -> Self {
        Self {
            max_recent_requests,
            entries: Mutex::new(VecDeque::new()),
            aggregated: Mutex::new(AggregatedStats::default()),
            last_log_time: Mutex::new(Instant::now()),
            log_interval,
            identifier,
            last_logged_values: Mutex::new(None),
        }
    }

    pub fn record(
        &self,
        host_blocks: usize,
        disk_blocks: usize,
        object_blocks: usize,
        total_blocks: usize,
    ) {
        if total_blocks == 0 {
            return;
        }

        let entry = CacheStatsEntry {
            host_blocks: host_blocks as u64,
            disk_blocks: disk_blocks as u64,
            object_blocks: object_blocks as u64,
            total_blocks: total_blocks as u64,
        };

        let mut entries = self.entries.lock().unwrap();
        let mut aggregated = self.aggregated.lock().unwrap();

        entries.push_back(entry);
        aggregated.total_blocks_queried += entry.total_blocks;
        aggregated.host_blocks_hit += entry.host_blocks;
        aggregated.disk_blocks_hit += entry.disk_blocks;
        aggregated.object_blocks_hit += entry.object_blocks;

        while entries.len() > 1 && entries.len() > self.max_recent_requests {
            if let Some(old_entry) = entries.pop_front() {
                aggregated.total_blocks_queried -= old_entry.total_blocks;
                aggregated.host_blocks_hit -= old_entry.host_blocks;
                aggregated.disk_blocks_hit -= old_entry.disk_blocks;
                aggregated.object_blocks_hit -= old_entry.object_blocks;
            }
        }
    }

    pub fn maybe_log(&self) -> bool {
        let now = Instant::now();
        let should_log = {
            let mut last_log = self.last_log_time.lock().unwrap();
            let elapsed = now.duration_since(*last_log);
            if elapsed >= self.log_interval {
                *last_log = now;
                true
            } else {
                false
            }
        };

        if !should_log {
            return false;
        }

        let (total, host, disk, object) = {
            let aggregated = self.aggregated.lock().unwrap();
            (
                aggregated.total_blocks_queried,
                aggregated.host_blocks_hit,
                aggregated.disk_blocks_hit,
                aggregated.object_blocks_hit,
            )
        };

        if total == 0 {
            return false;
        }

        let should_emit = {
            let mut last_logged = self.last_logged_values.lock().unwrap();
            let current = (total, host, disk, object);
            match *last_logged {
                Some(prev) if prev == current => false,
                _ => {
                    *last_logged = Some(current);
                    true
                }
            }
        };

        if !should_emit {
            return false;
        }

        let host_rate = host as f64 / total as f64 * 100.0;
        let disk_rate = disk as f64 / total as f64 * 100.0;
        let object_rate = object as f64 / total as f64 * 100.0;

        let prefix = self
            .identifier
            .as_ref()
            .map(|id| format!("KVBM [{id}] Cache Hit Rates"))
            .unwrap_or_else(|| "KVBM Cache Hit Rates".to_string());

        tracing::info!(
            "{} - Host: {:.1}% ({}/{}), Disk: {:.1}% ({}/{}), Object: {:.1}% ({}/{})",
            prefix,
            host_rate,
            host,
            total,
            disk_rate,
            disk,
            total,
            object_rate,
            object,
            total,
        );
        true
    }

    pub fn rates(&self) -> (f64, f64, f64) {
        let aggregated = self.aggregated.lock().unwrap();
        if aggregated.total_blocks_queried == 0 {
            return (0.0, 0.0, 0.0);
        }

        let total = aggregated.total_blocks_queried as f64;
        (
            aggregated.host_blocks_hit as f64 / total,
            aggregated.disk_blocks_hit as f64 / total,
            aggregated.object_blocks_hit as f64 / total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_rates_include_object_hits() {
        let tracker = CacheStatsTracker::new(100, Duration::from_secs(60), None);
        tracker.record(2, 1, 1, 4);
        let (host, disk, object) = tracker.rates();
        assert_eq!(host, 0.5);
        assert_eq!(disk, 0.25);
        assert_eq!(object, 0.25);
    }
}
