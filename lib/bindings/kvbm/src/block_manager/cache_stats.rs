// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM cache statistics tracking and periodic logging.
//!
//! This module provides cache statistics tracking with a sliding window
//! approach for tracking host and disk cache hit rates.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Default maximum number of recent requests to track in the sliding window
const DEFAULT_MAX_RECENT_REQUESTS: usize = 1000;
const DEFAULT_LOG_INTERVAL_SECS: u64 = 5;

/// Cache statistics entry for a single request
#[derive(Clone, Copy, Debug)]
struct CacheStatsEntry {
    host_blocks: u64,      // Blocks found in host cache
    disk_blocks: u64,      // Blocks found in disk cache
    total_blocks: u64,     // Total blocks queried from host/disk
}

/// Aggregated cache statistics for the current sliding window
#[derive(Default)]
struct AggregatedStats {
    total_blocks_queried: u64,  // Total blocks queried from host/disk (same for both tiers)
    host_blocks_hit: u64,        // Blocks found in host cache
    disk_blocks_hit: u64,        // Blocks found in disk cache
}

/// Cache statistics tracker with sliding window
/// Tracks the most recent N requests (default: 1000) for cache hit rate calculation
pub struct CacheStatsTracker {
    /// Maximum number of recent requests to track
    max_recent_requests: usize,
    /// Queue of recent cache stats entries
    entries: Mutex<VecDeque<CacheStatsEntry>>,
    /// Aggregated values for the current window (single lock for all counters)
    aggregated: Mutex<AggregatedStats>,
    /// Last time we logged statistics
    last_log_time: Mutex<Instant>,
    /// Interval between log messages
    log_interval: Duration,
    /// Optional identifier for this tracker (e.g., worker_id, engine_index)
    /// Used in log messages to distinguish between multiple KVBM instances
    identifier: Option<String>,
    /// Last logged values to avoid duplicate logs when values haven't changed
    /// Format: (total_blocks_queried, host_blocks_hit, disk_blocks_hit)
    last_logged_values: Mutex<Option<(u64, u64, u64)>>,
}

impl CacheStatsTracker {
    /// Create a new cache statistics tracker
    ///
    /// # Arguments
    /// * `identifier` - Optional identifier for this tracker (e.g., worker_id, engine_index).
    ///   Used in log messages to distinguish between multiple KVBM instances.
    ///
    /// The maximum number of recent requests is read from `DYN_KVBM_CACHE_STATS_MAX_REQUESTS` env var
    /// if set, otherwise defaults to 1000.
    ///
    /// The log interval is read from `DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS` env var if set,
    /// otherwise defaults to 5 seconds.
    pub fn new(identifier: Option<String>) -> Self {
        let max_recent_requests = std::env::var("DYN_KVBM_CACHE_STATS_MAX_REQUESTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_RECENT_REQUESTS);

        let log_interval_secs = std::env::var("DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_LOG_INTERVAL_SECS);

        Self {
            max_recent_requests,
            entries: Mutex::new(VecDeque::new()),
            aggregated: Mutex::new(AggregatedStats::default()),
            last_log_time: Mutex::new(Instant::now()),
            log_interval: Duration::from_secs(log_interval_secs),
            identifier,
            last_logged_values: Mutex::new(None),
        }
    }

    /// Record cache statistics for a completed request
    /// Uses sliding window: when max_recent_requests is exceeded, oldest entries are removed
    pub fn record(&self, host_blocks: usize, disk_blocks: usize, total_blocks: usize) {
        if total_blocks == 0 {
            // Skip empty requests
            return;
        }

        let entry = CacheStatsEntry {
            host_blocks: host_blocks as u64,
            disk_blocks: disk_blocks as u64,
            total_blocks: total_blocks as u64,
        };

        // Lock entries and aggregated stats separately to minimize lock contention
        let mut entries = self.entries.lock().unwrap();
        let mut aggregated = self.aggregated.lock().unwrap();

        // Add new entry and update aggregated stats
        entries.push_back(entry);
        aggregated.total_blocks_queried += entry.total_blocks;
        aggregated.host_blocks_hit += entry.host_blocks;
        aggregated.disk_blocks_hit += entry.disk_blocks;

        // Remove oldest entries if we exceed the limit
        // Keep at least one entry (the latest)
        while entries.len() > 1 && entries.len() > self.max_recent_requests {
            if let Some(old_entry) = entries.pop_front() {
                aggregated.total_blocks_queried -= old_entry.total_blocks;
                aggregated.host_blocks_hit -= old_entry.host_blocks;
                aggregated.disk_blocks_hit -= old_entry.disk_blocks;
            }
        }
    }

    /// Check if we should log and do so if enough time has passed
    /// Returns true if logging occurred, false otherwise
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

        if should_log {
            // Read aggregated stats with minimal lock time
            let (total_blocks_queried, host_blocks_hit, disk_blocks_hit) = {
                let aggregated = self.aggregated.lock().unwrap();
                (
                    aggregated.total_blocks_queried,
                    aggregated.host_blocks_hit,
                    aggregated.disk_blocks_hit,
                )
            };

            // Only log if there's activity
            if total_blocks_queried > 0 {
                // Check if values have changed since last log
                let should_log_values = {
                    let mut last_logged = self.last_logged_values.lock().unwrap();
                    let current_values = (total_blocks_queried, host_blocks_hit, disk_blocks_hit);
                    match *last_logged {
                        Some(prev) if prev == current_values => {
                            // Values haven't changed, skip logging
                            false
                        }
                        _ => {
                            // Values changed or first log, update and log
                            *last_logged = Some(current_values);
                            true
                        }
                    }
                };

                if should_log_values {
                    let host_rate = if total_blocks_queried == 0 {
                        0.0
                    } else {
                        (host_blocks_hit as f32 / total_blocks_queried as f32) * 100.0
                    };

                    let disk_rate = if total_blocks_queried == 0 {
                        0.0
                    } else {
                        (disk_blocks_hit as f32 / total_blocks_queried as f32) * 100.0
                    };

                    // Include identifier in log message if available
                    let prefix = if let Some(ref id) = self.identifier {
                        format!("KVBM [{}] Cache Hit Rates", id)
                    } else {
                        "KVBM Cache Hit Rates".to_string()
                    };

                    tracing::info!(
                        "{} - Host: {:.1}% ({}/{}), Disk: {:.1}% ({}/{})",
                        prefix,
                        host_rate,
                        host_blocks_hit,
                        total_blocks_queried,
                        disk_rate,
                        disk_blocks_hit,
                        total_blocks_queried,
                    );
                    return true;
                }
            }
        }
        false
    }

    /// Get current host cache hit rate (0.0-1.0) from the sliding window
    pub fn host_hit_rate(&self) -> f32 {
        let aggregated = self.aggregated.lock().unwrap();
        if aggregated.total_blocks_queried == 0 {
            0.0
        } else {
            aggregated.host_blocks_hit as f32 / aggregated.total_blocks_queried as f32
        }
    }

    /// Get current disk cache hit rate (0.0-1.0) from the sliding window
    pub fn disk_hit_rate(&self) -> f32 {
        let aggregated = self.aggregated.lock().unwrap();
        if aggregated.total_blocks_queried == 0 {
            0.0
        } else {
            aggregated.disk_blocks_hit as f32 / aggregated.total_blocks_queried as f32
        }
    }

    /// Reset the statistics (clears the sliding window)
    /// Useful for test isolation
    #[cfg(test)]
    #[allow(dead_code)] // Keep for test utilities, even if not currently used
    pub fn reset(&self) {
        let mut entries = self.entries.lock().unwrap();
        let mut aggregated = self.aggregated.lock().unwrap();
        let mut last_logged = self.last_logged_values.lock().unwrap();
        entries.clear();
        *aggregated = AggregatedStats::default();
        *last_logged = None;
    }
}

impl Default for CacheStatsTracker {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_stats_tracking() {
        // Use a small window size for testing
        unsafe {
            std::env::set_var("DYN_KVBM_CACHE_STATS_MAX_REQUESTS", "10");
        }
        let tracker = CacheStatsTracker::new(None);
        unsafe {
            std::env::remove_var("DYN_KVBM_CACHE_STATS_MAX_REQUESTS");
        }

        // Record some cache hits
        tracker.record(5, 3, 10); // 50% host, 30% disk
        tracker.record(8, 2, 10); // 80% host, 20% disk

        // Overall: 13/20 = 65% host, 5/20 = 25% disk
        let host_rate = tracker.host_hit_rate();
        let disk_rate = tracker.disk_hit_rate();

        assert!((host_rate - 0.65).abs() < 0.01);
        assert!((disk_rate - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_sliding_window() {
        // Use a small window size for testing
        unsafe {
            std::env::set_var("DYN_KVBM_CACHE_STATS_MAX_REQUESTS", "3");
        }
        let tracker = CacheStatsTracker::new(None);
        unsafe {
            std::env::remove_var("DYN_KVBM_CACHE_STATS_MAX_REQUESTS");
        }

        // Add 5 entries, but max is 3
        tracker.record(10, 5, 10); // Entry 1: 100% host, 50% disk
        tracker.record(0, 0, 10); // Entry 2: 0% host, 0% disk
        tracker.record(5, 5, 10); // Entry 3: 50% host, 50% disk
        tracker.record(10, 10, 10); // Entry 4: 100% host, 100% disk (should remove entry 1)
        tracker.record(0, 0, 10); // Entry 5: 0% host, 0% disk (should remove entry 2)

        // Window should contain entries 3, 4, 5
        // Entry 3: 5/10 host, 5/10 disk
        // Entry 4: 10/10 host, 10/10 disk
        // Entry 5: 0/10 host, 0/10 disk
        // Total: 15/30 host = 50%, 15/30 disk = 50%

        let host_rate = tracker.host_hit_rate();
        let disk_rate = tracker.disk_hit_rate();

        assert!(
            (host_rate - 0.5).abs() < 0.01,
            "host_rate={}, expected=0.5",
            host_rate
        );
        assert!(
            (disk_rate - 0.5).abs() < 0.01,
            "disk_rate={}, expected=0.5",
            disk_rate
        );

        // Verify window size
        let entries_len = tracker.entries.lock().unwrap().len();
        assert_eq!(
            entries_len, 3,
            "Expected 3 entries in window, got {}",
            entries_len
        );
    }
}
