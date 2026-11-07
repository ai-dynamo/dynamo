// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA policy and state, independent of frontend request and transport types.

mod cache;
pub mod config;
pub mod controller;
mod downloader;
pub mod filter;
mod huggingface;
pub mod load_estimator;
pub mod metrics;
pub mod predictor;
pub mod routing;
pub mod runtime;
mod source;
pub mod state_tracker;

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

/// Adapter metadata advertised by a worker model card.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct LoraInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gpu_lora_count: Option<u32>,
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

#[cfg(test)]
mod tests {
    use super::LoraInfo;

    #[test]
    fn lora_info_preserves_model_card_wire_shape() {
        let legacy: LoraInfo = serde_json::from_str(r#"{"name":"adapter"}"#).unwrap();
        assert_eq!(legacy.max_gpu_lora_count, None);
        assert_eq!(
            serde_json::to_string(&legacy).unwrap(),
            r#"{"name":"adapter"}"#
        );
        let bounded = LoraInfo {
            name: "adapter".into(),
            max_gpu_lora_count: Some(4),
        };
        assert_eq!(
            serde_json::to_string(&bounded).unwrap(),
            r#"{"name":"adapter","max_gpu_lora_count":4}"#
        );
    }
}
