// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `least-busy` policy.
//!
//! The policy optionally filters workers by effective cache overlap, then ranks
//! the remaining workers by active requests.

mod filter;
mod selection;

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{KvRouterConfig, WorkerFilter, WorkerSelectionPolicy};
use filter::MinimumEffectiveOverlapFilter;
use selection::{LeastBusyScorer, LowestCostPicker};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    #[serde(default)]
    min_effective_overlap_blocks: Option<f64>,
}

fn validate_min_effective_overlap_blocks(
    min_effective_overlap_blocks: Option<f64>,
) -> Result<(), WorkerSelectionPolicyProviderError> {
    if let Some(value) = min_effective_overlap_blocks
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(WorkerSelectionPolicyProviderError::new(
            "min_effective_overlap_blocks must be a finite positive number",
        ));
    }
    Ok(())
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    validate_min_effective_overlap_blocks(parameters.min_effective_overlap_blocks)?;
    let min_effective_overlap_blocks = parameters.min_effective_overlap_blocks;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            let filters: Vec<Box<dyn WorkerFilter>> = min_effective_overlap_blocks.map_or_else(
                Vec::new,
                |min_effective_overlap_blocks| {
                    vec![Box::new(MinimumEffectiveOverlapFilter {
                        min_effective_overlap_blocks,
                    }) as Box<dyn WorkerFilter>]
                },
            );
            WorkerSelectionPolicy::new_with_filters(
                config.clone(),
                worker_type,
                filters,
                vec![Box::new(LeastBusyScorer)],
                Box::new(LowestCostPicker),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("least-busy", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_min_effective_overlap_blocks() {
        assert!(validate_min_effective_overlap_blocks(None).is_ok());
        assert!(validate_min_effective_overlap_blocks(Some(8.0)).is_ok());
        assert!(validate_min_effective_overlap_blocks(Some(0.0)).is_err());
        assert!(validate_min_effective_overlap_blocks(Some(-1.0)).is_err());
        assert!(validate_min_effective_overlap_blocks(Some(f64::NAN)).is_err());
    }
}
