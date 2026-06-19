// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for the audit subsystem.
//!
//! Metrics are registered against the default Prometheus registry during
//! `register_audit_metrics()`, called from `audit::init_from_env_with_shutdown`.

use std::sync::LazyLock;

use prometheus::{HistogramOpts, HistogramVec, IntCounterVec, Opts, exponential_buckets};

/// Total segments uploaded to S3, labeled by result ("ok" or "failed").
pub static AUDIT_S3_SEGMENTS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new(
        "dynamo_audit_s3_segments_total",
        "Total audit segments uploaded to S3",
    )
    .namespace("dynamo");
    IntCounterVec::new(opts, &["result"]).expect("audit_s3_segments_total metric")
});

/// Total audit records dropped, labeled by reason:
/// - "bus_lag": broadcast channel lagged, records were overwritten before the
///   sink worker could consume them.
/// - "channel_full": the per-sink mpsc channel was full (try_send failed).
/// - "serialize_error": serde_json serialization of AuditRecord failed.
pub static AUDIT_RECORDS_DROPPED_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    let opts = Opts::new(
        "dynamo_audit_records_dropped_total",
        "Total audit records dropped before reaching the destination",
    )
    .namespace("dynamo");
    IntCounterVec::new(opts, &["reason"]).expect("audit_records_dropped_total metric")
});

/// Histogram of S3 PutObject durations in seconds.
pub static AUDIT_S3_UPLOAD_DURATION_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new(
        "dynamo_audit_s3_upload_duration_seconds",
        "Duration of S3 PutObject calls for audit segments",
    )
    .namespace("dynamo")
    .buckets(exponential_buckets(0.01, 2.0, 12).expect("valid histogram buckets"));
    HistogramVec::new(opts, &[]).expect("audit_s3_upload_duration_seconds metric")
});

/// Histogram of uploaded segment sizes in bytes (compressed).
pub static AUDIT_S3_SEGMENT_SIZE_BYTES: LazyLock<HistogramVec> = LazyLock::new(|| {
    let opts = HistogramOpts::new(
        "dynamo_audit_s3_segment_size_bytes",
        "Size of uploaded audit segments in bytes (gzip-compressed)",
    )
    .namespace("dynamo")
    .buckets(exponential_buckets(1024.0, 4.0, 10).expect("valid histogram buckets"));
    HistogramVec::new(opts, &[]).expect("audit_s3_segment_size_bytes metric")
});

/// Register all audit metrics with the default Prometheus registry.
/// Called once from `audit::init_from_env_with_shutdown`. Safe to call
/// multiple times (duplicate registration is a no-op that logs a warning).
pub fn register_audit_metrics() {
    let registry = prometheus::default_registry();
    let metrics: Vec<Box<dyn prometheus::core::Collector>> = vec![
        Box::new(AUDIT_S3_SEGMENTS_TOTAL.clone()),
        Box::new(AUDIT_RECORDS_DROPPED_TOTAL.clone()),
        Box::new(AUDIT_S3_UPLOAD_DURATION_SECONDS.clone()),
        Box::new(AUDIT_S3_SEGMENT_SIZE_BYTES.clone()),
    ];
    for m in metrics {
        if let Err(e) = registry.register(m) {
            tracing::debug!("audit metric already registered (expected on re-init): {e}");
        }
    }
}
