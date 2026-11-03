// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA downloading and caching infrastructure
//!
//! This module provides a minimal, extensible interface for downloading LoRA adapters
//! from various sources (local filesystem, S3, etc.) with automatic caching.
//! It also provides routing and lora allocation algorithms for distributing LoRA adapters
//! across workers in a cluster.

mod cache;
pub mod config;
pub mod controller;
mod downloader;
pub mod filter;
pub mod load_estimator;
pub mod metrics;
pub mod predictor;
pub mod routing;
mod source;
pub mod state_tracker;
mod subscription;

pub use cache::LoRACache;
pub use config::{LoraAllocationConfig, McfConfig};
pub use controller::LoraController;
pub use downloader::LoRADownloader;
pub use filter::LoraFilter;
pub use load_estimator::{LoadEstimator, LoadEstimatorConfig};
pub use routing::{
    AllocationAlgorithmType, LoraAllocator, LoraReplicaConfig, LoraRoutingTable,
    McfPlacementResult, McfPlacementSolver, McfSolveParams, RendezvousHasher,
    create_lora_allocator,
};
pub use source::{HuggingFaceLoRASource, LoRASource, LocalLoRASource, S3LoRASource};
pub use state_tracker::LoraStateTracker;
pub use subscription::LoadEstimatorSubscription;

#[cfg(test)]
mod metrics_tests;

/// Adapter metadata advertised by a worker model card.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct LoraInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gpu_lora_count: Option<u32>,
}

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

/// Worker runtime flag requiring an adapter registration before request routing.
pub const LORA_REQUIRES_REGISTRATION: &str = "lora_requires_registration";

/// Returns true when LoRA serving is enabled via a truthy `DYN_LORA_ENABLED` env var
/// (`1`/`true`/`on`/`yes`, case-insensitive). This gates the request-time LoRA filter on
/// both the KV and non-KV routing paths, so non-LoRA deployments keep the unmodified
/// routing path with zero added overhead.
pub fn lora_serving_enabled() -> bool {
    dynamo_runtime::config::env_is_truthy(
        dynamo_runtime::config::environment_names::llm::DYN_LORA_ENABLED,
    )
}
