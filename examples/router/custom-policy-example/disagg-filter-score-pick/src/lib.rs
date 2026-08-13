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
use dynamo_kv_router::{KvRouterConfig, WorkerFilter, WorkerSelectionPolicy};
use filter::MinimumDeviceOverlapFilter;
use picker::{DecodePicker, PrefillPicker};
use scorer::{DecodeLoadScorer, PrefillLoadScorer};

const PREFILL_WORKER_TYPE: &str = "prefill";
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

/// Builds the complete policy used by prefill routes.
fn create_prefill_policy(
    config: &KvRouterConfig,
    min_device_overlap_blocks: f64,
) -> WorkerSelectionPolicy {
    let filters: Vec<Box<dyn WorkerFilter>> = vec![Box::new(MinimumDeviceOverlapFilter {
        min_device_overlap_blocks,
    })];

    WorkerSelectionPolicy::new_with_filters(
        config.clone(),
        PREFILL_WORKER_TYPE,
        filters,
        vec![Box::new(PrefillLoadScorer)],
        Box::new(PrefillPicker),
    )
}

/// Builds the complete policy used by decode routes.
fn create_decode_policy(
    config: &KvRouterConfig,
    min_device_overlap_blocks: f64,
) -> WorkerSelectionPolicy {
    let filters: Vec<Box<dyn WorkerFilter>> = vec![Box::new(MinimumDeviceOverlapFilter {
        min_device_overlap_blocks,
    })];

    WorkerSelectionPolicy::new_with_filters(
        config.clone(),
        DECODE_WORKER_TYPE,
        filters,
        vec![Box::new(DecodeLoadScorer)],
        Box::new(DecodePicker),
    )
}

/// Selects a policy from the routing stage supplied by the embedded frontend.
///
/// Discovery has already scoped the worker pool before the factory receives this stage.
fn create_policy(
    config: &KvRouterConfig,
    worker_type: &'static str,
    min_device_overlap_blocks: f64,
) -> WorkerSelectionPolicy {
    match worker_type {
        PREFILL_WORKER_TYPE => create_prefill_policy(config, min_device_overlap_blocks),
        DECODE_WORKER_TYPE => create_decode_policy(config, min_device_overlap_blocks),
        unsupported => panic!(
            "disagg-filter-score-pick does not support worker type {unsupported:?}; expected prefill or decode"
        ),
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    validate_min_device_overlap_blocks(parameters.min_device_overlap_blocks)?;
    let min_device_overlap_blocks = parameters.min_device_overlap_blocks;

    Ok(Arc::new(move |config, worker_type, _partition| {
        create_policy(config, worker_type, min_device_overlap_blocks)
    }))
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

    #[test]
    fn creates_each_supported_worker_policy() {
        let config = KvRouterConfig::default();
        create_policy(&config, PREFILL_WORKER_TYPE, 0.0);
        create_policy(&config, DECODE_WORKER_TYPE, 0.0);
    }

    #[test]
    #[should_panic(expected = "does not support worker type \"select\"")]
    fn rejects_standalone_select_worker_type() {
        create_policy(&KvRouterConfig::default(), "select", 0.0);
    }

    #[test]
    #[should_panic(expected = "does not support worker type \"unknown\"")]
    fn rejects_unknown_worker_type() {
        create_policy(&KvRouterConfig::default(), "unknown", 0.0);
    }
}
