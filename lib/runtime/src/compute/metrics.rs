// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for monitoring compute pool operations

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Metrics for the compute pool
#[derive(Debug)]
pub struct ComputeMetrics {
    /// Total number of tasks executed
    tasks_total: AtomicU64,

    /// Number of tasks currently running
    tasks_active: AtomicUsize,

    /// Total time spent in compute tasks (microseconds)
    total_compute_time_us: AtomicU64,

    /// Maximum task duration seen (microseconds)
    max_task_duration_us: AtomicU64,

    /// Number of tasks that took longer than 100ms
    slow_tasks: AtomicU64,
}

impl ComputeMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            tasks_total: AtomicU64::new(0),
            tasks_active: AtomicUsize::new(0),
            total_compute_time_us: AtomicU64::new(0),
            max_task_duration_us: AtomicU64::new(0),
            slow_tasks: AtomicU64::new(0),
        }
    }

    /// Record that a task has started
    pub fn record_task_start(&self) {
        self.tasks_active.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a task has completed
    pub fn record_task_completion(&self, duration: Duration) {
        self.tasks_active.fetch_sub(1, Ordering::Relaxed);
        self.tasks_total.fetch_add(1, Ordering::Relaxed);

        // Use saturating conversion to prevent overflow
        let duration_us = duration.as_micros().min(u64::MAX as u128) as u64;
        self.total_compute_time_us
            .fetch_add(duration_us, Ordering::Relaxed);

        // Update max duration
        let mut current_max = self.max_task_duration_us.load(Ordering::Relaxed);
        while duration_us > current_max {
            match self.max_task_duration_us.compare_exchange_weak(
                current_max,
                duration_us,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }

        // Track slow tasks (> 100ms)
        if duration.as_millis() > 100 {
            self.slow_tasks.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get total number of tasks executed
    pub fn tasks_total(&self) -> u64 {
        self.tasks_total.load(Ordering::Relaxed)
    }

    /// Get number of currently active tasks
    pub fn tasks_active(&self) -> usize {
        self.tasks_active.load(Ordering::Relaxed)
    }

    /// Get average task duration in microseconds
    pub fn avg_task_duration_us(&self) -> f64 {
        let total = self.tasks_total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }

        let total_time = self.total_compute_time_us.load(Ordering::Relaxed);
        total_time as f64 / total as f64
    }

    /// Get maximum task duration in microseconds
    pub fn max_task_duration_us(&self) -> u64 {
        self.max_task_duration_us.load(Ordering::Relaxed)
    }

    /// Get number of slow tasks (> 100ms)
    pub fn slow_tasks(&self) -> u64 {
        self.slow_tasks.load(Ordering::Relaxed)
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.tasks_total.store(0, Ordering::Relaxed);
        self.tasks_active.store(0, Ordering::Relaxed);
        self.total_compute_time_us.store(0, Ordering::Relaxed);
        self.max_task_duration_us.store(0, Ordering::Relaxed);
        self.slow_tasks.store(0, Ordering::Relaxed);
    }
}

impl Default for ComputeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Format metrics as a human-readable string
impl std::fmt::Display for ComputeMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ComputeMetrics {{ tasks_total: {}, tasks_active: {}, avg_duration_ms: {:.2}, max_duration_ms: {:.2}, slow_tasks: {} }}",
            self.tasks_total(),
            self.tasks_active(),
            self.avg_task_duration_us() / 1000.0,
            self.max_task_duration_us() as f64 / 1000.0,
            self.slow_tasks(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        let metrics = ComputeMetrics::new();

        assert_eq!(metrics.tasks_total(), 0);
        assert_eq!(metrics.tasks_active(), 0);

        metrics.record_task_start();
        assert_eq!(metrics.tasks_active(), 1);

        metrics.record_task_completion(Duration::from_millis(50));
        assert_eq!(metrics.tasks_active(), 0);
        assert_eq!(metrics.tasks_total(), 1);
        assert_eq!(metrics.slow_tasks(), 0);

        metrics.record_task_start();
        metrics.record_task_completion(Duration::from_millis(150));
        assert_eq!(metrics.tasks_total(), 2);
        assert_eq!(metrics.slow_tasks(), 1);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = ComputeMetrics::new();

        metrics.record_task_start();
        metrics.record_task_completion(Duration::from_millis(50));
        assert_eq!(metrics.tasks_total(), 1);

        metrics.reset();
        assert_eq!(metrics.tasks_total(), 0);
        assert_eq!(metrics.tasks_active(), 0);
    }
}
