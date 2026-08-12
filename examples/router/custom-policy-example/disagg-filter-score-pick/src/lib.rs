// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `disagg-filter-score-pick` policy.
//!
//! The factory selects prefill or decode components for each routing partition.

mod filter;
mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerFilter, WorkerPicker, WorkerScorer, WorkerSelectionPolicy,
};
use filter::MinimumDeviceOverlapFilter;
use picker::{DecodePicker, PrefillPicker};
use scorer::{DecodeLoadScorer, PrefillLoadScorer};

const DECODE_WORKER_TYPE: &str = "decode";

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    min_device_overlap_blocks: f64,
}

fn validate_min_device_overlap_blocks(
    min_device_overlap_blocks: f64,
) -> Result<(), WorkerSelectionPolicyProviderError> {
    if !min_device_overlap_blocks.is_finite() || min_device_overlap_blocks < 0.0 {
        return Err(WorkerSelectionPolicyProviderError::new(
            "min_device_overlap_blocks must be a finite non-negative number",
        ));
    }
    Ok(())
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    validate_min_device_overlap_blocks(parameters.min_device_overlap_blocks)?;
    let min_device_overlap_blocks = parameters.min_device_overlap_blocks;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            let filters: Vec<Box<dyn WorkerFilter>> = vec![Box::new(MinimumDeviceOverlapFilter {
                min_device_overlap_blocks,
            })];
            let (scorers, picker): (Vec<Box<dyn WorkerScorer>>, Box<dyn WorkerPicker>) =
                if worker_type == DECODE_WORKER_TYPE {
                    (vec![Box::new(DecodeLoadScorer)], Box::new(DecodePicker))
                } else {
                    (vec![Box::new(PrefillLoadScorer)], Box::new(PrefillPicker))
                };

            WorkerSelectionPolicy::new_with_filters(
                config.clone(),
                worker_type,
                filters,
                scorers,
                picker,
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("disagg-filter-score-pick", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_min_device_overlap_blocks() {
        assert!(validate_min_device_overlap_blocks(0.0).is_ok());
        assert!(validate_min_device_overlap_blocks(8.0).is_ok());
        assert!(validate_min_device_overlap_blocks(-1.0).is_err());
        assert!(validate_min_device_overlap_blocks(f64::NAN).is_err());
    }
}
