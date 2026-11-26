// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Factory and registration for the `disagg-filter-score-pick` policy.
//!
//! The factory selects prefill or decode components for each routing partition.

mod filter;
mod picker;
mod scorer;

use std::sync::Arc;

use dynamo_kv_router::plugins::worker_selection::*;
use filter::MinimumDeviceOverlapFilter;
use picker::{DecodePicker, PrefillPicker, UnsupportedWorkerTypePicker};
use scorer::{DecodeLoadScorer, PrefillLoadScorer};

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
fn create_prefill_policy(min_device_overlap_blocks: f64) -> WorkerSelectionPolicy {
    let filters: Vec<Box<dyn WorkerFilter>> = vec![Box::new(MinimumDeviceOverlapFilter {
        min_device_overlap_blocks,
    })];

    WorkerSelectionPolicy::new_with_filters(
        WorkerType::Prefill.default_selector_label(),
        filters,
        vec![Box::new(PrefillLoadScorer)],
        Box::new(PrefillPicker),
    )
}

/// Builds the complete policy used by decode routes.
fn create_decode_policy(min_device_overlap_blocks: f64) -> WorkerSelectionPolicy {
    let filters: Vec<Box<dyn WorkerFilter>> = vec![Box::new(MinimumDeviceOverlapFilter {
        min_device_overlap_blocks,
    })];

    WorkerSelectionPolicy::new_with_filters(
        WorkerType::Decode.default_selector_label(),
        filters,
        vec![Box::new(DecodeLoadScorer)],
        Box::new(DecodePicker),
    )
}

/// Selects a policy from the routing stage supplied by the embedded frontend.
///
/// Discovery has already scoped the worker pool before the factory receives this stage.
fn create_policy(worker_type: WorkerType, min_device_overlap_blocks: f64) -> WorkerSelectionPolicy {
    match worker_type {
        WorkerType::Prefill => create_prefill_policy(min_device_overlap_blocks),
        WorkerType::Decode => create_decode_policy(min_device_overlap_blocks),
        unsupported => WorkerSelectionPolicy::new(
            unsupported.as_str(),
            Vec::new(),
            Box::new(UnsupportedWorkerTypePicker::new(unsupported)),
        ),
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    validate_min_device_overlap_blocks(parameters.min_device_overlap_blocks)?;
    let min_device_overlap_blocks = parameters.min_device_overlap_blocks;

    Ok(Arc::new(move |context| {
        create_policy(context.worker_type(), min_device_overlap_blocks)
    }))
}

/// Register the `disagg-filter-score-pick` policy type.
pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection("disagg-filter-score-pick", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::plugins::testing::{CaseWorker, SelectionCase, SelectionCaseError};

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
        create_policy(WorkerType::Prefill, 0.0);
        create_policy(WorkerType::Decode, 0.0);
    }

    #[test]
    fn unsupported_worker_types_return_selection_errors() {
        let case = SelectionCase::new(16, 16).worker(0, CaseWorker::new().total_kv_blocks(1024));

        for worker_type in [WorkerType::Aggregated, WorkerType::Encode] {
            let policy = create_policy(worker_type, 0.0);
            let error = case.select(&policy).unwrap_err();
            assert!(matches!(
                error,
                SelectionCaseError::Policy(WorkerSelectionPolicyError::Failed(message))
                    if message.contains(worker_type.as_str())
            ));
        }
    }
}
