// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation module for compute task timing
//!
//! This module is only compiled when the `compute-validation` feature is enabled.
//! It provides functions to validate that compute tasks are correctly classified
//! as small, medium, or large based on their execution time.

#[cfg(feature = "compute-validation")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "compute-validation")]
use std::time::Duration;
#[cfg(feature = "compute-validation")]
use tracing::warn;

/// Threshold for small tasks in microseconds (<100μs)
#[cfg(feature = "compute-validation")]
pub const SMALL_THRESHOLD_US: u64 = 100;

/// Threshold for medium tasks in microseconds (100μs - 1ms)
#[cfg(feature = "compute-validation")]
pub const MEDIUM_THRESHOLD_US: u64 = 1000;

// Metrics counters for misclassified tasks
#[cfg(feature = "compute-validation")]
static SMALL_MISCLASSIFIED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "compute-validation")]
static MEDIUM_MISCLASSIFIED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "compute-validation")]
static LARGE_MISCLASSIFIED: AtomicU64 = AtomicU64::new(0);

/// Validate that a task classified as small actually completed within the small threshold
#[cfg(feature = "compute-validation")]
pub fn validate_small(elapsed: Duration) {
    let micros = elapsed.as_micros() as u64;
    if micros > SMALL_THRESHOLD_US {
        SMALL_MISCLASSIFIED.fetch_add(1, Ordering::Relaxed);
        warn!(
            task_duration_us = micros,
            threshold_us = SMALL_THRESHOLD_US,
            "compute_small! task exceeded threshold. Consider using compute_medium!"
        );
    }
}

/// Validate that a task classified as medium is within the medium range
#[cfg(feature = "compute-validation")]
pub fn validate_medium(elapsed: Duration) {
    let micros = elapsed.as_micros() as u64;
    if micros < SMALL_THRESHOLD_US {
        MEDIUM_MISCLASSIFIED.fetch_add(1, Ordering::Relaxed);
        warn!(
            task_duration_us = micros,
            threshold_us = SMALL_THRESHOLD_US,
            "compute_medium! task below small threshold. Consider using compute_small!"
        );
    } else if micros > MEDIUM_THRESHOLD_US {
        MEDIUM_MISCLASSIFIED.fetch_add(1, Ordering::Relaxed);
        warn!(
            task_duration_us = micros,
            threshold_us = MEDIUM_THRESHOLD_US,
            "compute_medium! task exceeded threshold. Consider using compute_large!"
        );
    }
}

/// Validate that a task classified as large actually needed offloading
#[cfg(feature = "compute-validation")]
pub fn validate_large(elapsed: Duration) {
    let micros = elapsed.as_micros() as u64;
    if micros < MEDIUM_THRESHOLD_US {
        LARGE_MISCLASSIFIED.fetch_add(1, Ordering::Relaxed);
        warn!(
            task_duration_us = micros,
            threshold_us = MEDIUM_THRESHOLD_US,
            "compute_large! task below medium threshold. Consider using compute_medium! or compute_small!"
        );
    }
}

/// Get metrics about misclassified tasks
#[cfg(feature = "compute-validation")]
pub fn get_misclassification_metrics() -> (u64, u64, u64) {
    (
        SMALL_MISCLASSIFIED.load(Ordering::Relaxed),
        MEDIUM_MISCLASSIFIED.load(Ordering::Relaxed),
        LARGE_MISCLASSIFIED.load(Ordering::Relaxed),
    )
}

/// Reset misclassification metrics
#[cfg(feature = "compute-validation")]
pub fn reset_misclassification_metrics() {
    SMALL_MISCLASSIFIED.store(0, Ordering::Relaxed);
    MEDIUM_MISCLASSIFIED.store(0, Ordering::Relaxed);
    LARGE_MISCLASSIFIED.store(0, Ordering::Relaxed);
}
