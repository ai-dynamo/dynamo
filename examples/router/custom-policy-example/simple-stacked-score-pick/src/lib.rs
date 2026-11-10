// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `simple-stacked-score-pick` policy.

mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicy;
use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicyFactory;
use dynamo_kv_router::services::selection::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use picker::LowestCostPicker;
use scorer::{ActiveRequestsScorer, UncachedBlocksScorer};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let _: Parameters = parameters.deserialize()?;

    Ok(Arc::new(
        |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                vec![
                    Box::new(ActiveRequestsScorer),
                    Box::new(UncachedBlocksScorer),
                ],
                Box::new(LowestCostPicker),
            )
        },
    ))
}

/// Register the `simple-stacked-score-pick` policy type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection("simple-stacked-score-pick", Arc::new(provider))
}
