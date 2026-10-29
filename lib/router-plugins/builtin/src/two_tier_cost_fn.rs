// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-tier worker-selection cost function.
//!
//! Dynamo's built-in selector folds cache overlap and load into one additive cost. This policy
//! instead ranks on two tiers, taking the first that applies. For each eligible worker it reads
//! credited prefix overlap and active-request count, then:
//!
//! 1. Load tier: if active-request spread exceeds `balance_abs_threshold` and the largest count
//!    exceeds `balance_rel_threshold` times the smallest, select the least-loaded worker.
//! 2. Cache tier: otherwise, if the largest credited overlap is strictly greater than
//!    `cache_threshold` of the request's block count, select the least-loaded worker holding that
//!    maximum overlap.
//! 3. Otherwise, select the least-loaded worker.
//!
//! Both load gates must hold to take step 1, so load displaces cache affinity only when the
//! imbalance is both large in absolute terms and disproportionate.
//!
//! Credited overlap is the worker's device-KV overlap plus the lower-tier and remote credits of
//! Dynamo's default policy, under the same parameter names: `host_cache_hit_weight` and
//! `disk_cache_hit_weight` per host-pinned and disk prefix block, and `shared_cache_multiplier`
//! per shared-cache hit beyond the device prefix and per block a KV fetch hint would add beyond
//! the worker's own prefix. The host chooses each worker's fetch source, and booking a worker
//! attaches a fetch hint for that same source. These weights have no `sgl-router` counterpart and
//! default to zero, so credited overlap is device-KV overlap unless an instance sets them.
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

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCacheInput, WorkerCacheInputs, WorkerInputView, WorkerInputs, WorkerLoadInput,
    WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory, WorkerTransferInput, WorkerTransferInputs,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-two-tier-cost-fn";

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults, so an instance
/// with no `parameters` mapping reproduces that policy exactly.
const DEFAULT_CACHE_THRESHOLD: f64 = 0.5;
const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 32;
const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.1;

/// Tunables for [`POLICY_TYPE`]. Thresholds are named after their `sgl-router` counterparts, and
/// cache weights after the default policy's.
///
/// Every field is optional and keeps its default when omitted. Router-level cache weights do not
/// apply. Unknown keys are rejected at startup rather than ignored, so a misremembered name fails
/// loudly.
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
    /// Credit per host-pinned prefix block.
    host_cache_hit_weight: f64,
    /// Credit per disk prefix block.
    disk_cache_hit_weight: f64,
    /// Credit per block a worker could read remotely: shared-cache hits beyond its device prefix,
    /// plus the blocks a KV fetch would add beyond its device, host, and disk prefix.
    shared_cache_multiplier: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
            host_cache_hit_weight: 0.0,
            disk_cache_hit_weight: 0.0,
            shared_cache_multiplier: 0.0,
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
        for (name, value) in [
            ("host_cache_hit_weight", self.host_cache_hit_weight),
            ("disk_cache_hit_weight", self.disk_cache_hit_weight),
            ("shared_cache_multiplier", self.shared_cache_multiplier),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    /// Device overlap plus the weighted lower-tier and remote blocks, defined as the default
    /// policy's credits are. With every weight zero this is exactly the device overlap.
    fn credited_overlap(
        &self,
        cache: WorkerCacheInput<'_>,
        transfer: Option<WorkerTransferInput>,
    ) -> f64 {
        let device = cache.device_overlap_blocks();
        let host = cache.host_overlap_blocks();
        let disk = cache.disk_overlap_blocks();
        let shared = cache
            .shared_hits()
            .map_or(0, |hits| hits.hits_beyond(device.round().max(0.0) as u32));
        // Blocks a KV fetch would add beyond everything this worker holds.
        let fetchable = transfer
            .and_then(|transfer| transfer.source_prefix_blocks())
            .map_or(0.0, |source| {
                (f64::from(source) - (device + host + disk)).max(0.0)
            });
        device
            + self.host_cache_hit_weight * host
            + self.disk_cache_hit_weight * disk
            + self.shared_cache_multiplier * (shared as f64 + fetchable)
    }
}

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

fn select_row(
    parameters: &Parameters,
    cache: WorkerCacheInputs<'_>,
    load: &[WorkerLoadInput],
    transfer: Option<WorkerTransferInputs<'_>>,
    request_blocks: u64,
    overlaps: &mut Vec<f64>,
) -> Option<usize> {
    if cache.is_empty()
        || cache.len() != load.len()
        || transfer.is_some_and(|transfer| transfer.len() != load.len())
    {
        return None;
    }

    let min_load = load.iter().map(|item| item.active_requests()).min()?;
    let max_load = load.iter().map(|item| item.active_requests()).max()?;
    if max_load.saturating_sub(min_load) > parameters.balance_abs_threshold
        && (max_load as f64) > parameters.balance_rel_threshold * (min_load as f64)
    {
        return least_loaded(load, 0..load.len());
    }

    // Computed once, so the maximum and the tie set compare identical values.
    overlaps.clear();
    overlaps.extend(cache.iter().enumerate().map(|(row, item)| {
        parameters.credited_overlap(item, transfer.and_then(|transfer| transfer.get(row)))
    }));
    let max_overlap = overlaps.iter().copied().max_by(f64::total_cmp)?;
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    if cache_ratio > parameters.cache_threshold {
        return least_loaded(
            load,
            overlaps
                .iter()
                .enumerate()
                .filter_map(|(row, &overlap)| (overlap == max_overlap).then_some(row)),
        );
    }

    least_loaded(load, 0..load.len())
}

struct TwoTierCostFnPicker {
    parameters: Parameters,
    /// Credited overlap per candidate row, reused across selections.
    overlaps: Vec<f64>,
}

impl TwoTierCostFnPicker {
    fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            overlaps: Vec::new(),
        }
    }
}

impl WorkerPicker for TwoTierCostFnPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        let inputs = WorkerInputs::CACHE | WorkerInputs::LOAD;
        if self.parameters.shared_cache_multiplier != 0.0 {
            // The host builds fetch sources per selection only for policies that ask.
            inputs | WorkerInputs::KV_TRANSFER
        } else {
            inputs
        }
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
        select_row(
            &self.parameters,
            cache,
            load,
            input.kv_transfer(),
            context.request_blocks(),
            &mut self.overlaps,
        )
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
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
                Box::new(TwoTierCostFnPicker::new(parameters)),
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
    use std::collections::{HashMap, HashSet};

    use dynamo_kv_router::kv_hints::KvTransferCandidates;
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvHintTransferWorkerMetadata, RoutingConstraints,
        WorkerConfigLike, WorkerWithDpRank,
    };
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
        fn kv_hint_transfer_metadata_for_dp_rank(
            &self,
            _dp_rank: u32,
        ) -> Option<KvHintTransferWorkerMetadata<'_>> {
            Some(KvHintTransferWorkerMetadata {
                worker_type: "aggregated",
                source_control_endpoint: Some("tcp://worker:23280"),
            })
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
        pick(
            parameters,
            &request(&workers),
            &workers.map(|(id, _, _)| id),
        )
    }

    /// A ten-block request over workers given as `(worker_id, device_overlap_blocks,
    /// active_requests)`.
    fn request(workers: &[(u64, usize, usize)]) -> SchedulingRequest {
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
        for &(id, overlap_blocks, active_requests) in workers {
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
        request
    }

    /// Fetch candidates for the ten-block prompt, held on device by `owners` as
    /// `(worker_id, prefix_blocks)`.
    fn fetch_candidates(owners: &[(u64, usize)]) -> KvTransferCandidates {
        KvTransferCandidates {
            block_hashes: (0..10).map(ExternalSequenceBlockHash).collect(),
            owner_prefix_blocks: owners
                .iter()
                .map(|&(id, prefix_blocks)| (worker(id).into(), prefix_blocks))
                .collect(),
            routing_snapshot: None,
        }
    }

    fn pick(parameters: Parameters, request: &SchedulingRequest, ids: &[u64]) -> WorkerWithDpRank {
        let configs: HashMap<_, _> = ids.iter().map(|&id| (id, TestWorker)).collect();
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(TwoTierCostFnPicker::new(parameters)),
        )
        .select_worker(WorkerSelectionInput::configured(
            &configs,
            request,
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
        for weights in [
            Parameters {
                host_cache_hit_weight: -0.1,
                ..Default::default()
            },
            Parameters {
                disk_cache_hit_weight: f64::INFINITY,
                ..Default::default()
            },
            Parameters {
                shared_cache_multiplier: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(weights.validate().is_err());
        }
        assert!(Parameters::default().validate().is_ok());
    }

    #[test]
    fn fetch_credit_lets_a_fetching_worker_tie_the_holder() {
        // A holds six of ten blocks on device; idle B can fetch them from A.
        let mut request = request(&[(A, 6, 4), (B, 0, 0)]);
        request.kv_transfer_candidates = Some(fetch_candidates(&[(A, 6)]));
        let upstream = TwoTierCostFnPicker::new(Parameters::default());
        assert!(
            !upstream
                .required_worker_inputs()
                .contains(WorkerInputs::KV_TRANSFER)
        );
        assert_eq!(pick(Parameters::default(), &request, &[A, B]), worker(A));

        // At full credit B's six fetchable blocks tie A's six resident ones; load breaks the tie.
        let credited = Parameters {
            shared_cache_multiplier: 1.0,
            ..Default::default()
        };
        assert_eq!(pick(credited, &request, &[A, B]), worker(B));
    }

    #[test]
    fn host_credit_keeps_an_own_copy_ahead_of_a_peer_fetch() {
        const C: u64 = 53;
        // A holds six blocks host-pinned. Less-loaded B can fetch six from C, which is ineligible.
        let mut request = request(&[(A, 0, 2), (B, 0, 0), (C, 6, 0)]);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(worker(A), 6);
        request.kv_transfer_candidates = Some(fetch_candidates(&[(C, 6)]));
        request.allowed_worker_ids = Some(HashSet::from([A, B]));
        let parameters = Parameters {
            cache_threshold: 0.2,
            host_cache_hit_weight: 0.75,
            shared_cache_multiplier: 0.5,
            ..Default::default()
        };
        // A's credited 4.5 blocks beat B's 3.0.
        assert_eq!(pick(parameters, &request, &[A, B, C]), worker(A));

        // Without host credit, the peer fetch outranks A's own copy.
        let fetch_only = Parameters {
            host_cache_hit_weight: 0.0,
            ..parameters
        };
        assert_eq!(pick(fetch_only, &request, &[A, B, C]), worker(B));
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
