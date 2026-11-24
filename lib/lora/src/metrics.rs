// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use prometheus::{IntGaugeVec, Opts};

/// Metric handles supplied by the serving frontend's registry.
///
/// `Default` creates isolated, unregistered handles for simulations and callers
/// without metrics export. Serving frontends use `start_for_endpoint_with_metrics`.
#[derive(Clone)]
pub struct ControllerMetrics {
    pub replica_factor: IntGaugeVec,
    pub is_active: IntGaugeVec,
    pub raw_arrival_count: IntGaugeVec,
    pub estimated_load: IntGaugeVec,
    pub active_requests: IntGaugeVec,
    pub churn_loads: IntGaugeVec,
    pub churn_unloads: IntGaugeVec,
    pub overflow_count: IntGaugeVec,
}

impl Default for ControllerMetrics {
    fn default() -> Self {
        fn gauge(name: &str, labels: &[&str]) -> IntGaugeVec {
            IntGaugeVec::new(Opts::new(name, name), labels).expect("static LoRA metric descriptor")
        }
        Self {
            replica_factor: gauge("lora_replica_factor", &["endpoint", "lora"]),
            is_active: gauge("lora_is_active", &["endpoint", "lora"]),
            raw_arrival_count: gauge("lora_raw_arrival_count", &["endpoint", "lora"]),
            estimated_load: gauge("lora_estimated_load", &["endpoint", "lora"]),
            active_requests: gauge("lora_active_requests", &["endpoint", "lora"]),
            churn_loads: gauge("lora_churn_loads", &["endpoint"]),
            churn_unloads: gauge("lora_churn_unloads", &["endpoint"]),
            overflow_count: gauge("lora_overflow_count", &["endpoint"]),
        }
    }
}
