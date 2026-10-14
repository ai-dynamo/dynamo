// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-tier worker-selection cost function.
//!
//! Dynamo's built-in selector folds cache overlap and load into one additive cost. This policy
//! instead ranks on two tiers, taking the first that applies. For each eligible worker it reads
//! device and host-pinned KV overlap and active-request count, then:
//!
//! 1. Load tier: if active-request spread exceeds `balance_abs_threshold` and the largest count
//!    exceeds `balance_rel_threshold` times the smallest, select the least-loaded worker.
//! 2. Cache tier: otherwise, if the largest *effective* KV overlap is strictly greater than
//!    `cache_threshold` of the request's block count, select the least-loaded worker holding that
//!    maximum overlap. Effective overlap is `device_blocks + host_cache_weight * host_blocks`.
//! 3. Otherwise, select the least-loaded worker.
//!
//! Both load gates must hold to take step 1, so load displaces cache affinity only when the
//! imbalance is both large in absolute terms and disproportionate.
//!
//! The thresholds and selection order are the exact implementation ported from
//! `experimental/sgl-router`'s `cache_aware_zmq` policy, using Dynamo's authoritative device-KV
//! overlap instead of that router's own approximate cache history. The thresholds are exposed as
//! instance parameters defaulting to that router's values. Set `host_cache_weight` to 0.0 to
//! reproduce its device-only cache ranking.
//!
//! `host_cache_weight` inherits [`KvRouterConfig::host_cache_hit_weight`]
//! (`DYN_ROUTER_HOST_CACHE_HIT_WEIGHT`, default 0.75) unless overridden per instance.
//!
//! Ties between equally ranked workers resolve on candidate row order, which the host leaves
//! unspecified. This matches the ported implementation; note that Dynamo's built-in selector
//! instead samples uniformly among ties.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCacheInput, WorkerCacheInputs, WorkerInputView, WorkerInputs, WorkerLoadInput,
    WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-two-tier-cost-fn";

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults, so an instance
/// uses the same load gates and cache threshold.
const DEFAULT_CACHE_THRESHOLD: f64 = 0.5;
const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 32;
const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.1;

/// Tunables for [`POLICY_TYPE`], named after their `sgl-router` counterparts.
///
/// Every field is optional. Unknown keys are rejected at startup rather than ignored,
/// so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Minimum effective cache overlap on the best worker, as a fraction of request blocks,
    /// before the cache tier applies. Compared strictly.
    cache_threshold: f64,
    /// Minimum active-request spread before the load tier applies.
    balance_abs_threshold: usize,
    /// Minimum ratio of largest to smallest active-request count before the load tier applies.
    balance_rel_threshold: f64,
    /// CPU overlap weight. Omission inherits `KvRouterConfig::host_cache_hit_weight`;
    /// an explicit zero disables CPU cache credit.
    host_cache_weight: Option<f64>,
}

/// Concrete startup values consumed by the picker.
struct PolicyParameters {
    cache_threshold: f64,
    balance_abs_threshold: usize,
    balance_rel_threshold: f64,
    host_cache_weight: f64,
}

impl Default for Parameters {
    /// Use the upstream load gates and cache threshold, inheriting the router's host weight.
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
            host_cache_weight: None,
        }
    }
}

impl Parameters {
    /// Resolve an omitted host weight from the router configuration.
    fn resolve(self, config: &KvRouterConfig) -> PolicyParameters {
        PolicyParameters {
            cache_threshold: self.cache_threshold,
            balance_abs_threshold: self.balance_abs_threshold,
            balance_rel_threshold: self.balance_rel_threshold,
            host_cache_weight: self
                .host_cache_weight
                .unwrap_or(config.host_cache_hit_weight),
        }
    }

    /// Reject invalid thresholds and explicit host weights before constructing the policy.
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.cache_threshold.is_finite() || !(0.0..=1.0).contains(&self.cache_threshold) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "cache_threshold must be a finite number between 0.0 and 1.0",
            ));
        }
        if !self.balance_rel_threshold.is_finite() || self.balance_rel_threshold < 1.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "balance_rel_threshold must be a finite number greater than or equal to 1.0",
            ));
        }
        if let Some(weight) = self.host_cache_weight
            && (!weight.is_finite() || weight < 0.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "host_cache_weight must be a finite number greater than or equal to 0.0",
            ));
        }
        Ok(())
    }
}

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

/// Add disjoint device and host matches with weighted host credit.
fn effective_overlap(cache: WorkerCacheInput<'_>, host_cache_weight: f64) -> f64 {
    cache.device_overlap_blocks() + host_cache_weight * cache.host_overlap_blocks()
}

/// Apply the load gates, then weighted cache affinity, with least load as the fallback.
///
/// Cache and load rows must describe the same candidates in the same order. Empty or
/// mismatched inputs return `None`; a zero-block request falls back to least load.
fn select_row(
    parameters: &PolicyParameters,
    cache: WorkerCacheInputs<'_>,
    load: &[WorkerLoadInput],
    request_blocks: u64,
) -> Option<usize> {
    if cache.is_empty() || cache.len() != load.len() {
        return None;
    }

    let min_load = load.iter().map(|item| item.active_requests()).min()?;
    let max_load = load.iter().map(|item| item.active_requests()).max()?;
    if max_load.saturating_sub(min_load) > parameters.balance_abs_threshold
        && (max_load as f64) > parameters.balance_rel_threshold * (min_load as f64)
    {
        return least_loaded(load, 0..load.len());
    }

    let max_overlap = cache
        .iter()
        .map(|item| effective_overlap(item, parameters.host_cache_weight))
        .max_by(f64::total_cmp)?;
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    if cache_ratio > parameters.cache_threshold {
        return least_loaded(
            load,
            cache.iter().enumerate().filter_map(|(row, item)| {
                // Both values use the same expression, so exact equality is intentional.
                (effective_overlap(item, parameters.host_cache_weight) == max_overlap)
                    .then_some(row)
            }),
        );
    }

    least_loaded(load, 0..load.len())
}

struct TwoTierCostFnPicker {
    parameters: PolicyParameters,
}

impl WorkerPicker for TwoTierCostFnPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    /// Select a candidate row, reporting unavailable cache or load inputs as policy errors.
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        select_row(&self.parameters, cache, load, context.request_blocks())
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

/// Build a policy factory with instance overrides taking precedence over router settings.
fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(TwoTierCostFnPicker {
                    parameters: parameters.resolve(config),
                }),
            )
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register_worker_selection(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    const BLOCK_SIZE: u32 = 16;
    /// Ten blocks, so five overlapping blocks sit exactly on the 0.5 threshold.
    const TEN_BLOCKS: usize = 160;
    const A: u64 = 29;
    const B: u64 = 41;

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

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    /// Select among workers given as `(worker_id, device_overlap_blocks, active_requests)`.
    fn select(workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        select_with(Parameters::default(), workers)
    }

    /// Apply custom parameters to device-only fixtures by supplying zero host overlap.
    fn select_with(parameters: Parameters, workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        select_tiers(
            parameters,
            workers.map(|(id, device_blocks, active)| (id, device_blocks, 0, active)),
            KvRouterConfig::default(),
        )
    }

    /// Select with `(worker_id, device_blocks, host_blocks, active_requests)` fixtures.
    fn select_tiers(
        parameters: Parameters,
        workers: [(u64, usize, usize, usize); 2],
        config: KvRouterConfig,
    ) -> WorkerWithDpRank {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: TEN_BLOCKS,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
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
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (id, device_blocks, host_blocks, active_requests) in workers {
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker(id), device_blocks);
            request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .insert(worker(id), host_blocks);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        let configs = HashMap::from(workers.map(|(id, _, _, _)| (id, TestWorker)));
        let parameters = parameters.resolve(&config);
        WorkerSelectionPolicy::new(
            config,
            "test",
            Vec::new(),
            Box::new(TwoTierCostFnPicker { parameters }),
        )
        .select_worker(WorkerSelectionInput::configured(
            &configs,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap()
        .worker
    }

    #[test]
    fn cache_tier_outranks_a_less_loaded_worker() {
        // Six GPU blocks outrank six CPU blocks and clear the 0.5 threshold despite more load.
        assert_eq!(
            select_tiers(
                Parameters::default(),
                [(A, 0, 6, 0), (B, 6, 0, 4)],
                KvRouterConfig::default(),
            ),
            worker(B)
        );
    }

    #[test]
    fn cache_tier_threshold_is_strict() {
        // Five of ten blocks is exactly 0.5, so the comparison fails and load decides.
        assert_eq!(select([(A, 0, 0), (B, 5, 4)]), worker(A));
    }

    #[test]
    fn parameters_override_the_upstream_defaults() {
        // Three of ten blocks is 0.3: below the 0.5 default, above a tuned 0.2 threshold.
        let workers = [(A, 0, 0), (B, 3, 4)];
        assert_eq!(select(workers), worker(A));

        let tuned = Parameters {
            cache_threshold: 0.2,
            ..Parameters::default()
        };
        assert_eq!(select_with(tuned, workers), worker(B));
    }

    /// CPU-only and mixed residency both provide six effective blocks at the default weight.
    #[test]
    fn host_overlap_can_win_the_cache_tier() {
        for (device_blocks, host_blocks) in [(0, 8), (3, 4)] {
            assert_eq!(
                select_tiers(
                    Parameters::default(),
                    [(A, 0, 0, 0), (B, device_blocks, host_blocks, 4)],
                    KvRouterConfig::default(),
                ),
                worker(B)
            );
        }
    }

    /// Instance overrides take precedence, including zero to opt out of CPU cache affinity.
    #[test]
    fn host_cache_weight_inherits_and_overrides_router_config() {
        for (router_weight, override_weight, expected) in
            [(0.25, None, A), (0.25, Some(0.75), B), (0.75, Some(0.0), A)]
        {
            let parameters = Parameters {
                host_cache_weight: override_weight,
                ..Parameters::default()
            };
            let config = KvRouterConfig {
                host_cache_hit_weight: router_weight,
                ..Default::default()
            };
            assert_eq!(
                select_tiers(parameters, [(A, 0, 0, 0), (B, 0, 8, 4)], config),
                worker(expected)
            );
        }
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let cache = |v| {
            Parameters {
                cache_threshold: v,
                ..Default::default()
            }
            .validate()
        };
        let ratio = |v| {
            Parameters {
                balance_rel_threshold: v,
                ..Default::default()
            }
            .validate()
        };

        assert!(cache(-0.1).is_err() && cache(1.1).is_err() && cache(f64::NAN).is_err());
        assert!(ratio(0.9).is_err() && ratio(f64::NAN).is_err());
        let weight = |v| {
            Parameters {
                host_cache_weight: Some(v),
                ..Default::default()
            }
            .validate()
        };
        for value in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(weight(value).is_err());
        }
        assert!(weight(0.0).is_ok());
        assert!(Parameters::default().validate().is_ok());
    }

    #[test]
    fn load_tier_needs_both_gates() {
        // Spread 40 > 32 and 40 > 1.1 * 0: the load tier fires and ignores B's full overlap.
        assert_eq!(select([(A, 0, 0), (B, 10, 40)]), worker(A));
        // Spread 64 > 32, but 704 is not > 1.1 * 640, so the cache tier still decides. This pair
        // straddles the ratio boundary: 705 would clear it and take the load tier.
        assert_eq!(select([(A, 0, 640), (B, 10, 704)]), worker(B));
        assert_eq!(select([(A, 0, 640), (B, 10, 705)]), worker(A));
    }
}
