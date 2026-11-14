// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Calibrated two-worker sibling routing with the experimental decode-progress input.

mod fallback;
mod picker;

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;
use picker::HotspotProgressPicker;

/// Policy type registered by this example.
pub const POLICY_TYPE: &str = "hotspot-progress";

#[derive(Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Parameters {
    pub(crate) source_active_requests: usize,
    pub(crate) min_prefix_tokens: usize,
    pub(crate) min_remaining_tokens: usize,
    pub(crate) max_progress_age_ms: u64,
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.source_active_requests == 0
            || self.source_active_requests >= 4096
            || self.min_prefix_tokens == 0
            || self.min_remaining_tokens == 0
            || !(1..=60_000).contains(&self.max_progress_age_ms)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "hotspot-progress requires source_active_requests in 1..4096, positive token thresholds, and max_progress_age_ms in 1..=60000",
            ));
        }
        Ok(())
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &SelectionConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(HotspotProgressPicker {
                    parameters,
                    aggregated: matches!(worker_type, WorkerType::Aggregated),
                }),
            )
        },
    ))
}

/// Register the calibrated `hotspot-progress` example policy.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}
