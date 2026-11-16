// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA integration with frontend discovery, metrics, and event transport.

pub use dynamo_lora::*;
mod subscription;
pub use subscription::LoadEstimatorSubscription;

#[cfg(test)]
mod metrics_tests;

impl dynamo_lora::runtime::RuntimeLoraWorkerFacts
    for crate::local_model::runtime_config::ModelRuntimeConfig
{
    fn runtime_value(&self, key: &str) -> Option<&serde_json::Value> {
        self.runtime_data.get(key)
    }
}

/// Inject the frontend registry's existing series into the allocation controller.
pub fn controller_metrics() -> dynamo_lora::metrics::ControllerMetrics {
    use crate::http::service::metrics::*;
    dynamo_lora::metrics::ControllerMetrics {
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
