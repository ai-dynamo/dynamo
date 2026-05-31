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
pub mod filtered_router;
pub mod load_estimator;
pub mod predictor;
pub mod routing;
mod source;
pub mod state_tracker;

pub use cache::LoRACache;
pub use config::{LoraAllocationConfig, McfConfig};
pub use controller::LoraController;
pub use downloader::LoRADownloader;
pub use filter::LoraFilter;
pub use filtered_router::LoraFilteredRouter;
pub use load_estimator::{LoadEstimator, LoadEstimatorConfig};
pub use routing::{
    AllocationAlgorithmType, LoraAllocator, LoraReplicaConfig, LoraRoutingTable,
    McfPlacementResult, McfPlacementSolver, McfSolveParams, RendezvousHasher,
    create_lora_allocator,
};
pub use source::{LoRASource, LocalLoRASource, S3LoRASource};
pub use state_tracker::LoraStateTracker;

/// Returns true when LoRA serving is enabled via the `DYN_LORA_ENABLED` env var
/// (`true`/`1`/`yes`, case-insensitive). This gates the request-time LoRA filter on
/// both the KV and non-KV routing paths, so non-LoRA deployments keep the unmodified
/// routing path with zero added overhead.
pub fn lora_serving_enabled() -> bool {
    std::env::var(dynamo_runtime::config::environment_names::llm::DYN_LORA_ENABLED)
        .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(false)
}
