// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA downloading and caching infrastructure
//!
//! This module provides a minimal, extensible interface for downloading LoRA adapters
//! from various sources (local filesystem, S3, etc.) with automatic caching.
//! It also provides routing and lora allocation algorithms for distributing LoRA adapters
//! across workers in a cluster.

pub use dynamo_lora::*;
mod subscription;

pub use subscription::LoadEstimatorSubscription;

#[cfg(test)]
mod metrics_tests;

/// Inject the frontend registry's existing series into the allocation controller.
pub fn controller_metrics() -> metrics::ControllerMetrics {
    use crate::http::service::metrics::*;
    metrics::ControllerMetrics {
        replica_factor: LORA_REPLICA_FACTOR_GAUGE.clone(),
        is_active: LORA_IS_ACTIVE_GAUGE.clone(),
        raw_arrival_count: LORA_RAW_ARRIVAL_COUNT_GAUGE.clone(),
        estimated_load: LORA_ESTIMATED_LOAD_GAUGE.clone(),
        active_requests: LORA_ACTIVE_REQUESTS_GAUGE.clone(),
        churn_loads: LORA_CHURN_LOADS_GAUGE.clone(),
        churn_unloads: LORA_CHURN_UNLOADS_GAUGE.clone(),
        overflow_count: LORA_OVERFLOW_COUNT_GAUGE.clone(),
    }
}
