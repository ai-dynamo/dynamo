// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-owned replay composition over Dynamo's production routing primitives.
//!
//! The simulator owns the engine, clock and observations. Only routing decisions,
//! queue ordering and affinity bindings are delegated to the native policy crates.

mod affinity;
mod composition;
mod helpers;
mod observation;
mod router;
mod runtime;
#[cfg(test)]
mod tests;

pub use affinity::{ReplayAffinityConfig, ReplayAffinityMode, RoutingEvidence};
pub use composition::NativeDynamoReplayComposition;
pub use dynamo_kv_router::PrefillLoadEstimator;
pub use dynamo_kv_router::config::KvRouterConfig;
use serde::{Deserialize, Serialize};

pub const NATIVE_DYNAMO_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NATIVE_DYNAMO_PROVIDER: &str = "dynamo_kv_router";

pub type ReplayPrefillLoadEstimator = std::sync::Arc<dyn PrefillLoadEstimator>;

/// Configuration serialized in `ReplaySpec.adapters.placement.config`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeDynamoRouterConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity: Option<ReplayAffinityConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub router_config: Option<KvRouterConfig>,
}

impl NativeDynamoRouterConfig {
    pub fn from_value(value: &serde_json::Value) -> anyhow::Result<Self> {
        if value.is_null() {
            Ok(Self::default())
        } else {
            Ok(serde_json::from_value(value.clone())?)
        }
    }
}

#[cfg(feature = "python-replay")]
mod execution;
#[cfg(feature = "python-replay")]
pub use execution::execute_native_router_replay;
