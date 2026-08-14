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
use dynamo_kv_router::{KvRouterConfig, WorkerFilter, WorkerSelectionPolicy, WorkerType};
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
fn create_prefill_policy(
    config: &KvRouterConfig,
    min_device_overlap_blocks: f64,
) -> WorkerSelectionPolicy {
    let filters: Vec<Box<dyn WorkerFilter>> = vec![Box::new(MinimumDeviceOverlapFilter {
        min_device_overlap_blocks,
    })];

    WorkerSelectionPolicy::new_with_filters(
        config.clone(),
        WorkerType::Prefill.default_selector_label(),
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
        WorkerType::Decode.default_selector_label(),
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
    worker_type: WorkerType,
    min_device_overlap_blocks: f64,
) -> WorkerSelectionPolicy {
    match worker_type {
        WorkerType::Prefill => create_prefill_policy(config, min_device_overlap_blocks),
        WorkerType::Decode => create_decode_policy(config, min_device_overlap_blocks),
        unsupported => WorkerSelectionPolicy::new(
            config.clone(),
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

    Ok(Arc::new(move |config, worker_type, _partition| {
        create_policy(config, worker_type, min_device_overlap_blocks)
    }))
}

/// Register the `disagg-filter-score-pick` policy type.
pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register("disagg-filter-score-pick", Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError, WorkerSelector,
    };

    use super::*;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1024)
        }
    }

    fn request() -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

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
        create_policy(&config, WorkerType::Prefill, 0.0);
        create_policy(&config, WorkerType::Decode, 0.0);
    }

    #[test]
    fn unsupported_worker_types_return_selection_errors() {
        let workers = HashMap::from([(0, TestWorker)]);
        let request = request();

        for worker_type in [WorkerType::Aggregated, WorkerType::Encode] {
            let policy = create_policy(&KvRouterConfig::default(), worker_type, 0.0);
            let error = policy
                .select_worker(&workers, &request, request.eligibility(), 16)
                .unwrap_err();
            assert!(matches!(
                error,
                KvSchedulerError::WorkerSelectionPolicy(WorkerSelectionPolicyError::Failed(message))
                    if message.contains(worker_type.as_str())
            ));
        }
    }
}
