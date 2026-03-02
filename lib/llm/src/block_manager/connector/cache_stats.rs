// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const DEFAULT_MAX_RECENT_REQUESTS: usize = 1000;
const DEFAULT_LOG_INTERVAL_SECS: u64 = 5;

#[derive(Clone, Copy, Debug)]
struct CacheStatsEntry {
    host_blocks: u64,
    disk_blocks: u64,
    object_blocks: u64,
    total_blocks: u64,
}

#[derive(Default)]
struct AggregatedStats {
    total_blocks_queried: u64,
    host_blocks_hit: u64,
    disk_blocks_hit: u64,
    object_blocks_hit: u64,
}

/// Cache statistics tracker with a fixed-size sliding window.
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

        let (total_blocks_queried, host_blocks_hit, disk_blocks_hit, object_blocks_hit) = {
            let aggregated = self.aggregated.lock().unwrap();
            (
                aggregated.total_blocks_queried,
                aggregated.host_blocks_hit,
                aggregated.disk_blocks_hit,
                aggregated.object_blocks_hit,
            )
        };

        if total_blocks_queried == 0 {
            return false;
        }

        let should_log_values = {
            let mut last_logged = self.last_logged_values.lock().unwrap();
            let current_values = (
                total_blocks_queried,
                host_blocks_hit,
                disk_blocks_hit,
                object_blocks_hit,
            );
            match *last_logged {
                Some(prev) if prev == current_values => false,
                _ => {
                    *last_logged = Some(current_values);
                    true
                }
            }
        };

        if !should_log_values {
            return false;
        }

        let host_rate = (host_blocks_hit as f32 / total_blocks_queried as f32) * 100.0;
        let disk_rate = (disk_blocks_hit as f32 / total_blocks_queried as f32) * 100.0;
        let object_rate = (object_blocks_hit as f32 / total_blocks_queried as f32) * 100.0;

        let prefix = if let Some(ref id) = self.identifier {
            format!("KVBM [{}] Cache Hit Rates", id)
        } else {
            "KVBM Cache Hit Rates".to_string()
        };

        tracing::info!(
            "{} - Host: {:.1}% ({}/{}), Disk: {:.1}% ({}/{}), G4: {:.1}% ({}/{})",
            prefix,
            host_rate,
            host_blocks_hit,
            total_blocks_queried,
            disk_rate,
            disk_blocks_hit,
            total_blocks_queried,
            object_rate,
            object_blocks_hit,
            total_blocks_queried,
        );
        true
    }

    pub fn host_hit_rate(&self) -> f32 {
        let aggregated = self.aggregated.lock().unwrap();
        if aggregated.total_blocks_queried == 0 {
            0.0
        } else {
            aggregated.host_blocks_hit as f32 / aggregated.total_blocks_queried as f32
        }
    }

    pub fn disk_hit_rate(&self) -> f32 {
        let aggregated = self.aggregated.lock().unwrap();
        if aggregated.total_blocks_queried == 0 {
            0.0
        } else {
            aggregated.disk_blocks_hit as f32 / aggregated.total_blocks_queried as f32
        }
    }

    pub fn object_hit_rate(&self) -> f32 {
        let aggregated = self.aggregated.lock().unwrap();
        if aggregated.total_blocks_queried == 0 {
            0.0
        } else {
            aggregated.object_blocks_hit as f32 / aggregated.total_blocks_queried as f32
        }
    }
}

impl Default for CacheStatsTracker {
    fn default() -> Self {
        Self::new(None)
    }
}
