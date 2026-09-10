// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-tier worker-selection cost function.
//!
//! Dynamo's built-in selector folds cache overlap and load into one additive cost. This policy
//! instead ranks on two tiers, taking the first that applies. For each eligible worker it reads
//! device-KV overlap and active-request count, then:
//!
//! 1. Load tier: if active-request spread exceeds `balance_abs_threshold` and the largest count
//!    exceeds `balance_rel_threshold` times the smallest, select the least-loaded worker.
//! 2. Cache tier: otherwise, if the largest device-KV overlap is strictly greater than
//!    `cache_threshold` of the request's block count, select the least-loaded worker holding that
//!    maximum overlap.
//! 3. Otherwise, select the least-loaded worker.
//!
//! Both load gates must hold to take step 1, so load displaces cache affinity only when the
//! imbalance is both large in absolute terms and disproportionate.
//!
//! The thresholds and selection order are the exact implementation ported from
//! `experimental/sgl-router`'s `cache_aware_zmq` policy, using Dynamo's authoritative device-KV
//! overlap instead of that router's own approximate cache history. The thresholds are exposed as
//! instance parameters defaulting to that router's values, so an instance with no `parameters`
//! mapping reproduces it exactly.
//!
//! Ties between equally ranked workers resolve on candidate row order, which the host leaves
//! unspecified. This matches the ported implementation; note that Dynamo's built-in selector
//! instead samples uniformly among ties.
//!
//! The optional `adaptive` mode replaces the tier switch with a weighted cache/load cost. A
//! sigmoid increases the load weight with imbalance; pool-wide pressure limits that increase to
//! preserve cache reuse. This is a signal-driven rule, not a latency-reward learner.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerCacheInput, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-two-tier-cost-fn";

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults, so an instance
/// with no `parameters` mapping reproduces that policy exactly.
const DEFAULT_CACHE_THRESHOLD: f64 = 0.5;
const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 32;
const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.1;

/// Two-tier thresholds and optional adaptive cache/load parameters for [`POLICY_TYPE`].
///
/// Every field is optional. Omitting `adaptive` keeps the ported two-tier behavior. Unknown keys
/// are rejected at startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Fraction of the request's blocks that must be device-resident on the best worker before the
    /// cache tier applies. Compared strictly.
    cache_threshold: f64,
    /// Minimum active-request spread before the load tier applies.
    balance_abs_threshold: usize,
    /// Minimum ratio of largest to smallest active-request count before the load tier applies.
    balance_rel_threshold: f64,
    /// Opt into the adaptive weighted cost instead of the two-tier thresholds.
    adaptive: Option<AdaptiveParameters>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
            adaptive: None,
        }
    }
}

impl Parameters {
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
        if let Some(adaptive) = self.adaptive {
            if self.cache_threshold != DEFAULT_CACHE_THRESHOLD
                || self.balance_abs_threshold != DEFAULT_BALANCE_ABS_THRESHOLD
                || self.balance_rel_threshold != DEFAULT_BALANCE_REL_THRESHOLD
            {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "adaptive mode cannot be combined with non-default two-tier thresholds",
                ));
            }
            if adaptive.update_interval_ms == 0 {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "adaptive.update_interval_ms must be positive",
                ));
            }
            if !adaptive.load_scale.is_finite() || adaptive.load_scale < 1.0 {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "adaptive.load_scale must be finite and at least 1.0",
                ));
            }
        }
        Ok(())
    }
}

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

fn select_row(
    parameters: &Parameters,
    cache: &[WorkerCacheInput],
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
        .map(|item| item.device_overlap_blocks())
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
                (item.device_overlap_blocks() == max_overlap).then_some(row)
            }),
        );
    }

    least_loaded(load, 0..load.len())
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct AdaptiveParameters {
    update_interval_ms: u64,
    /// Active requests: damps small load differences and sets the pressure scale.
    load_scale: f64,
}

impl Default for AdaptiveParameters {
    fn default() -> Self {
        Self {
            update_interval_ms: 100,
            load_scale: 32.0,
        }
    }
}

#[derive(Default)]
struct AdaptiveState {
    last_update: Option<Instant>,
    distribution_weight: f64,
}

impl AdaptiveState {
    fn update(&mut self, parameters: AdaptiveParameters, now: Instant, min: usize, max: usize) {
        if self.last_update.is_some_and(|last| {
            now.saturating_duration_since(last)
                < Duration::from_millis(parameters.update_interval_ms)
        }) {
            return;
        }
        // Active counts are Dynamo-owned load proxies, not GPU utilization or queue depth.
        let imbalance = (max - min) as f64 / (max as f64 + parameters.load_scale);
        let pressure = min as f64 / (min as f64 + parameters.load_scale);
        let sigmoid = 1.0 / (1.0 + (-10.0 * (imbalance - 0.5)).exp());
        // llm-d's pressure rule: as every worker fills, cap distribution at 0.5 rather than
        // 0.9. Moving work to cold workers adds prefill work without creating capacity.
        let distribution_max = 0.9 - pressure * (0.9 - 0.5);
        self.distribution_weight = 0.1 + sigmoid * (distribution_max - 0.1);
        self.last_update = Some(now);
    }

    fn select(
        &mut self,
        parameters: AdaptiveParameters,
        now: Instant,
        cache: &[WorkerCacheInput],
        load: &[WorkerLoadInput],
        request_blocks: u64,
    ) -> Option<usize> {
        if cache.is_empty() || cache.len() != load.len() {
            return None;
        }
        let min = load.iter().map(|item| item.active_requests()).min()?;
        let max = load.iter().map(|item| item.active_requests()).max()?;
        let spread = (max - min) as f64;
        self.update(parameters, now, min, max);

        let weight = self.distribution_weight;
        let denominator = parameters.load_scale + spread;
        let mut best = None;
        let mut best_cost = f64::INFINITY;
        for (row, (cache, load)) in cache.iter().zip(load).enumerate() {
            let overlap = cache.device_overlap_blocks();
            let affinity = if request_blocks > 0 && overlap.is_finite() {
                (overlap / request_blocks as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let distribution = (load.active_requests() - min) as f64 / denominator;
            let cost = (1.0 - weight) * (1.0 - affinity) + weight * distribution;
            if cost < best_cost {
                best = Some(row);
                best_cost = cost;
            }
        }
        best
    }
}

struct TwoTierCostFnPicker {
    parameters: Parameters,
    adaptive: AdaptiveState,
}

impl WorkerPicker for TwoTierCostFnPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

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
        let row = match self.parameters.adaptive {
            Some(parameters) => self.adaptive.select(
                parameters,
                Instant::now(),
                cache,
                load,
                context.request_blocks(),
            ),
            None => select_row(&self.parameters, cache, load, context.request_blocks()),
        };
        row.ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
}

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
                    parameters,
                    adaptive: AdaptiveState::default(),
                }),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(POLICY_TYPE, Arc::new(provider))
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

    fn select_with(parameters: Parameters, workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
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
        for (id, overlap_blocks, active_requests) in workers {
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker(id), overlap_blocks);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        let configs = HashMap::from(workers.map(|(id, _, _)| (id, TestWorker)));
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(TwoTierCostFnPicker {
                parameters,
                adaptive: AdaptiveState::default(),
            }),
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
        // Six of ten blocks is 0.6, above the 0.5 threshold, so B wins despite carrying more load.
        assert_eq!(select([(A, 0, 0), (B, 6, 4)]), worker(B));
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

    #[test]
    fn adaptive_moves_from_cache_to_load_and_preserves_cache_under_pressure() {
        let parameters = Parameters {
            adaptive: Some(AdaptiveParameters::default()),
            ..Parameters::default()
        };
        // Modest skew preserves affinity. A hotspot with an idle alternative favors load.
        assert_eq!(select_with(parameters, [(A, 0, 0), (B, 6, 4)]), worker(B));
        assert_eq!(select_with(parameters, [(A, 0, 0), (B, 10, 64)]), worker(A));
        // Same normalized imbalance (2/3), but all workers are busy: preserve the cached prefix.
        let busy = [(A, 0, 320), (B, 10, 1024)];
        assert_eq!(select_with(parameters, busy), worker(B));
        assert_eq!(select(busy), worker(A));
        // Without cache benefit, either mode selects least load even under pressure.
        assert_eq!(
            select_with(parameters, [(A, 0, 320), (B, 0, 1024)]),
            worker(A)
        );
    }

    #[test]
    fn adaptive_updates_on_interval_and_recovers_after_a_hotspot() {
        let parameters = AdaptiveParameters::default();
        let now = Instant::now();
        let tick = Duration::from_millis(parameters.update_interval_ms);
        let mut state = AdaptiveState::default();
        state.update(parameters, now, 0, 0);
        let baseline = state.distribution_weight;
        assert!((0.1..0.2).contains(&baseline));

        state.update(parameters, now + tick / 2, 0, 1024);
        assert_eq!(state.distribution_weight, baseline);
        state.update(parameters, now + tick, 0, 1024);
        assert!(state.distribution_weight > 0.8 && state.distribution_weight < 0.9);
        state.update(parameters, now + tick * 2, 0, 0);
        assert_eq!(state.distribution_weight, baseline);
    }
}
