// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for semantic router (minimal for step 1)

use once_cell::sync::Lazy;
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};

/// Route decisions total
pub static DECISIONS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!(
        "dynamo_semrouter_decisions_total",
        "Total routing decisions",
        &["decision", "transport"]
    )
    .expect("Failed to register dynamo_semrouter_decisions_total")
});

/// Classifier latency
pub static CLASSIFIER_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "dynamo_semrouter_classifier_latency_seconds",
        "Classifier inference latency",
        &["classifier", "transport"]
    )
    .expect("Failed to register dynamo_semrouter_classifier_latency_seconds")
});
