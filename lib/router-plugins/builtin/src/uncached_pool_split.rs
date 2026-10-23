// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Uncached-compute worker-pool split.
//!
//! Isolates large cache-miss prefills onto per-tier compute pools so they stop contending with
//! cache-friendly traffic on the remaining workers. Classification follows the Cold Pool
//! proposal (ai-dynamo/dynamo#12803): it uses each worker's effective prefill — the prompt
//! length minus device-resident cache overlap — rather than raw ISL, so a long agent turn
//! whose prefix is already cached keeps its cache affinity.
//!
//! Tiers are configured as `[min_uncached_tokens, max_uncached_tokens)` bands over effective
//! prefill, each bound to a worker taint (for example `dynamo.topology/worker_pool=cold-1`).
//! A request whose best effective prefill falls in a band is restricted to workers carrying
//! that taint; below every band, or in a gap between bands, it selects normally across all
//! eligible workers.
//!
//! Pool membership is declared by the workers themselves through their published taints, so
//! scaling a pool is a deployment change (label the pods) rather than a router change.
//! Failure is loud and typed: an instance without a configured pool fails at startup, and a
//! tier with no eligible member leaves the picker without a row, which the host classifies as
//! `AllEligibleWorkersOverloaded` — never a silent fallback to warm workers and never an
//! unbounded queue.

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCacheInputs, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerId;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-uncached-pool-split";

/// Default effective-prefill threshold, sized for ~100K-token cold requests.
const DEFAULT_UNCACHED_THRESHOLD_TOKENS: usize = 100_000;

/// One cold tier: requests whose best effective prefill falls in
/// `[min_uncached_tokens, max_uncached_tokens)` are restricted to workers carrying `taint`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PoolTier {
    /// Inclusive lower bound on effective prefill, in tokens. Required and positive.
    min_uncached_tokens: usize,
    /// Exclusive upper bound on effective prefill, in tokens. Omit for unbounded.
    max_uncached_tokens: Option<usize>,
    /// Worker taint identifying this tier's members, for example
    /// `dynamo.topology/worker_pool=cold-1`. Workers publish it through
    /// `topology_domains` / worker labels; see the configuration guide.
    taint: String,
}

/// Tunables for [`POLICY_TYPE`].
///
/// Either `pools` (multi-tier, taint-declared membership) or `compute_workers` (single-tier
/// convenience: the first N workers by candidate order) must be configured. Unknown keys are
/// rejected at startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Effective-prefill token threshold for the single-tier convenience mode. A request
    /// whose fresh-prefill volume on every eligible worker is at or above this value is
    /// restricted to the compute pool. Compared inclusively. Ignored when `pools` is set.
    uncached_threshold_tokens: usize,
    /// Multi-tier configuration. Bands must be disjoint and ordered; each tier's `taint`
    /// must be unique.
    pools: Option<Vec<PoolTier>>,
    /// Single-tier pool size for the no-taint convenience mode: the first N candidates.
    /// Mutually exclusive with `pools`.
    compute_workers: Option<usize>,
    /// When `true`, a warm request with no cache affinity may spill onto idle cold-pool
    /// workers once every warm worker has been removed from eligibility (overload
    /// draining). When `false` (the default), cold workers serve only cold requests;
    /// a fully-drained warm pool rejects with the overload status instead.
    allow_warm_spillover: bool,
    /// Minimum fraction of cold-pool members still eligible before a warm spillover is
    /// allowed: with the default 0.5, at least half the pool's workers must remain
    /// un-drained. 0 escapes whenever any member remains. Only consulted when
    /// `allow_warm_spillover` is `true`.
    spillover_free_threshold: f64,
    /// When `true` (the default), a cold request whose own tier has no eligible member
    /// may upgrade to a higher tier with one (for example a 150K request on a saturated
    /// cold-1 spilling to cold-2). Upgrading keeps the request inside the cold-pool
    /// system; it never crosses down to a lower tier or into the warm pool.
    allow_tier_upgrade: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            uncached_threshold_tokens: DEFAULT_UNCACHED_THRESHOLD_TOKENS,
            pools: None,
            compute_workers: None,
            allow_warm_spillover: false,
            spillover_free_threshold: 0.5,
            allow_tier_upgrade: true,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.uncached_threshold_tokens == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "uncached_threshold_tokens must be positive",
            ));
        }
        if !self.spillover_free_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.spillover_free_threshold)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "spillover_free_threshold must be a fraction in [0.0, 1.0]",
            ));
        }
        match (&self.pools, self.compute_workers) {
            (Some(_), Some(_)) => {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "pools and compute_workers are mutually exclusive",
                ));
            }
            (Some(pools), None) => {
                if pools.is_empty() {
                    return Err(WorkerSelectionPolicyProviderError::new(
                        "pools must contain at least one tier when configured",
                    ));
                }
                let mut last_max: Option<usize> = None;
                let mut seen_taints: HashSet<&str> = HashSet::new();
                for (i, tier) in pools.iter().enumerate() {
                    if tier.min_uncached_tokens == 0 {
                        return Err(WorkerSelectionPolicyProviderError::new(format!(
                            "pools[{i}].min_uncached_tokens must be positive"
                        )));
                    }
                    if let Some(max) = tier.max_uncached_tokens
                        && max <= tier.min_uncached_tokens
                    {
                        return Err(WorkerSelectionPolicyProviderError::new(format!(
                            "pools[{i}].max_uncached_tokens must exceed min_uncached_tokens"
                        )));
                    }
                    if let Some(prev) = last_max
                        && tier.min_uncached_tokens < prev
                    {
                        return Err(WorkerSelectionPolicyProviderError::new(format!(
                            "pools[{i}] overlaps or precedes the previous tier's max"
                        )));
                    }
                    last_max = Some(tier.max_uncached_tokens.unwrap_or(usize::MAX));
                    if !seen_taints.insert(tier.taint.as_str()) {
                        return Err(WorkerSelectionPolicyProviderError::new(format!(
                            "pools[{i}].taint is already used by another tier"
                        )));
                    }
                }
            }
            (None, Some(workers)) => {
                if workers == 0 {
                    return Err(WorkerSelectionPolicyProviderError::new(
                        "compute_workers must be positive",
                    ));
                }
            }
            (None, None) => {
                return Err(WorkerSelectionPolicyProviderError::new(
                    "dynamo-uncached-pool-split requires either pools or compute_workers",
                ));
            }
        }
        Ok(())
    }
}

/// The lane a request's effective prefill falls into.
#[derive(Debug, Clone, PartialEq)]
enum Lane {
    /// Below every band, or in a gap between bands: normal selection over all workers.
    Warm,
    /// Inside a band: restricted to that tier's taint members (or, in the single-tier
    /// convenience mode, to the first `compute_workers` candidates).
    Cold { taint: Option<String> },
}

struct UncachedPoolSplitPicker {
    parameters: Parameters,
    /// Cold members observed in past selections. Overload draining removes workers from
    /// the candidate set before the picker runs, so the live candidates alone
    /// overstate the pool's headroom; the remembered union restores the denominator.
    /// Only grows, and erring large only makes spillover harder, which is the safe
    /// direction. Pool shrinkage (a member permanently removed) is handled by deployment
    /// ordering: removing the taint first, then the worker, keeps this from blocking
    /// spillover indefinitely.
    seen_cold_ids: std::cell::RefCell<HashSet<WorkerId>>,
}

impl UncachedPoolSplitPicker {
    /// Worker ids belonging to any cold tier (taint-declared in pools mode, the first N
    /// by identity in convenience mode).
    fn cold_member_ids(
        &self,
        candidates: &[dynamo_kv_router::plugins::worker_selection::ScoredWorkerCandidate],
    ) -> HashSet<WorkerId> {
        let pools = self.parameters.pools.as_ref();
        if pools.is_some() {
            candidates
                .iter()
                .filter(|candidate| {
                    candidate.taints().is_some_and(|taints| {
                        pools.is_some_and(|pools| {
                            pools.iter().any(|tier| taints.contains(&tier.taint))
                        })
                    })
                })
                .map(|candidate| candidate.worker().worker_id)
                .collect()
        } else {
            let n = self
                .parameters
                .compute_workers
                .unwrap_or(1)
                .min(candidates.len().saturating_sub(1).max(1));
            let mut ids: Vec<WorkerId> = candidates.iter().map(|c| c.worker().worker_id).collect();
            ids.sort_unstable();
            ids.truncate(n);
            ids.into_iter().collect()
        }
    }

    /// Rows a warm request may select from.
    ///
    /// The warm lane is the non-cold workers. Cold workers never join it while any warm
    /// worker remains eligible — including the one holding this request's largest cached
    /// prefix: keeping the split pure costs that follow-up turn a re-prefill, which is
    /// the isolation trade the policy exists to make. Only when the warm lane is empty
    /// (every warm worker drained by overload detection) does `allow_warm_spillover`
    /// decide: with it on, warm traffic may use the cold pool while enough of it remains
    /// eligible (`spillover_free_threshold`, measured against the cold ids in this
    /// candidate set); with it off the picker fails and the host answers with the
    /// overload status code.
    fn warm_candidate_rows(
        &self,
        candidates: &[dynamo_kv_router::plugins::worker_selection::ScoredWorkerCandidate],
        cached: &[usize],
        cold_ids: &HashSet<WorkerId>,
    ) -> Vec<usize> {
        let warm_rows: Vec<usize> = (0..candidates.len())
            .filter(|&row| !cold_ids.contains(&candidates[row].worker().worker_id))
            .collect();
        if !warm_rows.is_empty() || !self.parameters.allow_warm_spillover {
            return warm_rows;
        }
        // Remember every cold member ever seen, then measure headroom against the
        // remembered set: workers drained from eligibility are absent from this
        // selection's candidates, and without the memory the surviving fraction would
        // overstate the pool's health.
        let mut seen = self.seen_cold_ids.borrow_mut();
        seen.extend(cold_ids.iter().copied());
        let cold_eligible = cold_ids.len();
        let cold_present = seen.len();
        drop(seen);
        if cold_present == 0 {
            return warm_rows;
        }
        let free_fraction = cold_eligible as f64 / cold_present as f64;
        if free_fraction + f64::EPSILON < self.parameters.spillover_free_threshold {
            // Below the headroom floor: not enough cold capacity left to borrow.
            return warm_rows;
        }
        let _ = cached;
        (0..candidates.len()).collect()
    }

    fn classify(&self, isl_tokens: usize, best_effective_prefill: usize) -> Lane {
        let effective = best_effective_prefill.min(isl_tokens);
        if let Some(pools) = &self.parameters.pools {
            for tier in pools {
                let above_min = effective >= tier.min_uncached_tokens;
                let below_max = tier.max_uncached_tokens.is_none_or(|max| effective < max);
                if above_min && below_max {
                    return Lane::Cold {
                        taint: Some(tier.taint.clone()),
                    };
                }
            }
            Lane::Warm
        } else if effective >= self.parameters.uncached_threshold_tokens {
            Lane::Cold { taint: None }
        } else {
            Lane::Warm
        }
    }
}

impl WorkerPicker for UncachedPoolSplitPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        let taints = if self.parameters.pools.is_some() {
            WorkerInputs::TAINTS
        } else {
            WorkerInputs::NONE
        };
        WorkerInputs::CACHE | WorkerInputs::LOAD | taints
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        let cached = cached_tokens_per_row(input.cache(), candidates.len(), context.block_size());
        if cached.len() != load.len() {
            return Err(WorkerSelectionPolicyError::failed(
                "candidate and load row mismatch",
            ));
        }

        let isl = context.prompt_tokens();
        // Shared-cache (for example Mooncake HiCache) hits are worker-independent: any
        // candidate can fetch those blocks, so they reduce the fresh-prefill volume for
        // classification the same way device-resident overlap does.
        let shared_cached = shared_cache_tokens(input.cache(), context.block_size());
        let best_cached = cached.iter().copied().max().unwrap_or(0) + shared_cached;
        let best_effective = isl - best_cached.min(isl);

        match self.classify(isl, best_effective) {
            Lane::Warm => {
                let cold_ids = self.cold_member_ids(candidates);
                let rows = self.warm_candidate_rows(candidates, &cached, &cold_ids);
                select_affine(load, &cached, rows.into_iter())
                    .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
            }
            Lane::Cold { taint } => {
                let member_rows: Vec<usize> = match taint.as_deref() {
                    Some(taint) => candidates
                        .iter()
                        .enumerate()
                        .filter(|(_, candidate)| {
                            candidate.taints().is_some_and(|set| set.contains(taint))
                        })
                        .map(|(row, _)| row)
                        .collect(),
                    // Single-tier convenience mode: the first N candidates by worker
                    // identity. Candidate row order is host-defined and unstable, so
                    // membership keys on the worker id instead. Saturating at one fewer
                    // than the candidate count keeps at least one warm worker.
                    None => {
                        let n = self
                            .parameters
                            .compute_workers
                            .unwrap_or(1)
                            .min(candidates.len().saturating_sub(1).max(1));
                        let mut rows: Vec<usize> = (0..candidates.len()).collect();
                        rows.sort_by_key(|&row| candidates[row].worker().worker_id);
                        rows.truncate(n);
                        rows
                    }
                };
                // An empty member set must surface as the host's overload
                // classification, not a policy-specific error string: the request then
                // gets the overload status code (529, or 429 via
                // DYN_HTTP_OVERLOAD_STATUS_CODE) with ServiceOverloaded semantics.
                if let Some(row) = select_affine(load, &cached, member_rows.clone().into_iter()) {
                    return Ok(row);
                }
                // Tier upgrade: this tier has no eligible member and the switch is on,
                // so try each higher tier (a larger uncached band) that has one. The
                // request stays inside the cold-pool system: never down to a lower
                // tier, never into the warm pool.
                if self.parameters.allow_tier_upgrade
                    && let Some(index) = self
                        .parameters
                        .pools
                        .iter()
                        .flat_map(|pools| {
                            pools
                                .iter()
                                .position(|tier| Some(&tier.taint) == taint.as_ref())
                        })
                        .next()
                {
                    let pools = self.parameters.pools.clone().unwrap_or_default();
                    for tier in pools.iter().skip(index + 1) {
                        let upgraded_rows: Vec<usize> = candidates
                            .iter()
                            .enumerate()
                            .filter(|(_, candidate)| {
                                candidate
                                    .taints()
                                    .is_some_and(|set| set.contains(&tier.taint))
                            })
                            .map(|(row, _)| row)
                            .collect();
                        if let Some(row) = select_affine(load, &cached, upgraded_rows.into_iter()) {
                            return Ok(row);
                        }
                    }
                }
                Err(WorkerSelectionPolicyError::failed("no eligible worker"))
            }
        }
    }
}

/// Shared-cache (for example Mooncake HiCache) hit tokens for this request. Worker-
/// independent: the same volume is fetchable by every candidate, so it enters the
/// classification's fresh-prefill estimate but never the per-row cache comparison that
/// ranks workers against each other.
fn shared_cache_tokens(cache: Option<WorkerCacheInputs<'_>>, block_size: u32) -> usize {
    cache
        .and_then(|cache| cache.iter().next().and_then(|row| row.shared_hits()))
        .map(|hits| hits.total_hits as usize * block_size as usize)
        .unwrap_or(0)
}

/// Device-resident cached-token estimates, index-aligned with the candidates.
///
/// Uses device overlap only — the same authoritative signal the two-tier policy scores on —
/// because device-resident blocks are the ones that skip GPU prefill compute; lower-tier
/// hits still pay a transfer. Missing or misaligned cache input degrades to all-zero: with
/// no overlap information the picker treats the request as fully fresh compute, which routes
/// on length alone.
fn cached_tokens_per_row(
    cache: Option<WorkerCacheInputs<'_>>,
    candidates: usize,
    block_size: u32,
) -> Vec<usize> {
    match cache {
        Some(cache) if cache.len() == candidates => cache
            .iter()
            .map(|row| (row.device_overlap_blocks() * block_size as f64) as usize)
            .collect(),
        _ => vec![0; candidates],
    }
}

/// Least-loaded row among `rows`, breaking ties by row order.
fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

/// Normal selection: least-loaded worker among the largest cached-token holders.
fn select_affine(
    load: &[WorkerLoadInput],
    cached: &[usize],
    rows: impl Iterator<Item = usize> + Clone,
) -> Option<usize> {
    let max_cached = rows.clone().map(|row| cached[row]).max()?;
    least_loaded(load, rows.filter(|row| cached[*row] == max_cached))
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
                Box::new(UncachedPoolSplitPicker {
                    parameters: parameters.clone(),
                    seen_cold_ids: std::cell::RefCell::new(HashSet::new()),
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
    /// Ten blocks; with a 100-token threshold this sits above it when fully uncached.
    const ISL_TOKENS: usize = 160;
    const A: u64 = 29;
    const B: u64 = 41;
    const C: u64 = 53;
    const D: u64 = 67;

    struct TestWorker {
        taints: Arc<HashSet<String>>,
    }

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
        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    fn taint_set(values: &[&str]) -> Arc<HashSet<String>> {
        Arc::new(values.iter().map(|v| v.to_string()).collect())
    }

    /// Select among workers given as `(worker_id, device_overlap_blocks, active_requests)`.
    /// `pool_taints` maps worker id to the taints its config publishes.
    type SharedColdIds = std::rc::Rc<std::cell::RefCell<HashSet<WorkerId>>>;

    fn select_result(
        parameters: &Parameters,
        isl_tokens: usize,
        workers: &[(u64, usize, usize)],
        pool_taints: &HashMap<u64, Arc<HashSet<String>>>,
    ) -> Result<WorkerWithDpRank, dynamo_kv_router::scheduling::KvSchedulerError> {
        select_result_stateful(parameters, isl_tokens, workers, pool_taints, None, None)
    }

    /// Like [`select_result`] but with `allowed` restricting the eligible worker ids,
    /// simulating overload draining of everything outside that set.
    fn select_result_allowed(
        parameters: &Parameters,
        isl_tokens: usize,
        workers: &[(u64, usize, usize)],
        pool_taints: &HashMap<u64, Arc<HashSet<String>>>,
        allowed: Option<Vec<u64>>,
    ) -> Result<WorkerWithDpRank, dynamo_kv_router::scheduling::KvSchedulerError> {
        select_result_stateful(parameters, isl_tokens, workers, pool_taints, allowed, None)
    }

    /// Like [`select_result_allowed`] but carrying picker state across calls, so the
    /// remembered cold membership persists between requests like it does in production.
    fn select_result_stateful(
        parameters: &Parameters,
        isl_tokens: usize,
        workers: &[(u64, usize, usize)],
        pool_taints: &HashMap<u64, Arc<HashSet<String>>>,
        allowed: Option<Vec<u64>>,
        shared: Option<&SharedColdIds>,
    ) -> Result<WorkerWithDpRank, dynamo_kv_router::scheduling::KvSchedulerError> {
        select_result_full(
            parameters,
            isl_tokens,
            workers,
            pool_taints,
            allowed,
            shared,
            None,
        )
    }

    /// Full control surface: additionally injects a shared-cache (Mooncake-style) hit
    /// volume in blocks, exercising the classification's shared-cache credit.
    fn select_result_full(
        parameters: &Parameters,
        isl_tokens: usize,
        workers: &[(u64, usize, usize)],
        pool_taints: &HashMap<u64, Arc<HashSet<String>>>,
        allowed: Option<Vec<u64>>,
        shared: Option<&SharedColdIds>,
        shared_cache_blocks: Option<u32>,
    ) -> Result<WorkerWithDpRank, dynamo_kv_router::scheduling::KvSchedulerError> {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: allowed.map(|ids| ids.into_iter().collect()),
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
            shared_cache_hits: shared_cache_blocks.map(|blocks| {
                dynamo_kv_router::protocols::SharedCacheHits::from_ranges(vec![std::ops::Range {
                    start: 0,
                    end: blocks,
                }])
            }),
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (id, overlap_blocks, active_requests) in workers {
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker(*id), *overlap_blocks);
            request.worker_loads.insert(
                worker(*id),
                WorkerLoadProjection {
                    active_requests: *active_requests,
                    ..Default::default()
                },
            );
        }
        let configs: HashMap<u64, TestWorker> = workers
            .iter()
            .map(|(id, _, _)| {
                (
                    *id,
                    TestWorker {
                        taints: pool_taints
                            .get(id)
                            .cloned()
                            .unwrap_or_else(|| taint_set(&[])),
                    },
                )
            })
            .collect();
        let picker = UncachedPoolSplitPicker {
            parameters: parameters.clone(),
            seen_cold_ids: std::cell::RefCell::new(
                shared.map(|rc| rc.borrow().clone()).unwrap_or_default(),
            ),
        };

        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(picker),
        )
        .select_worker(WorkerSelectionInput::configured(
            &configs,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .map(|outcome| outcome.worker)
    }

    /// Two tiers over effective prefill: [100, 160) on taint cold-1, [160, ∞) on cold-2.
    fn two_tier_params() -> Parameters {
        Parameters {
            uncached_threshold_tokens: 100,
            pools: Some(vec![
                PoolTier {
                    min_uncached_tokens: 100,
                    max_uncached_tokens: Some(160),
                    taint: "cold-1".to_string(),
                },
                PoolTier {
                    min_uncached_tokens: 160,
                    max_uncached_tokens: None,
                    taint: "cold-2".to_string(),
                },
            ]),
            compute_workers: None,
            allow_warm_spillover: false,
            spillover_free_threshold: 0.5,
            allow_tier_upgrade: true,
        }
    }

    fn single_tier_params(threshold: usize, compute_workers: usize) -> Parameters {
        Parameters {
            uncached_threshold_tokens: threshold,
            pools: None,
            compute_workers: Some(compute_workers),
            allow_warm_spillover: false,
            spillover_free_threshold: 0.5,
            allow_tier_upgrade: true,
        }
    }

    #[test]
    fn tier_one_request_routes_to_cold1_members() {
        // Best effective prefill 160 - 32 = 128: inside [100, 160), so cold-1. B is idle
        // but carries no taint and must not win.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 2, 5), (B, 0, 0)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(A));
    }

    #[test]
    fn tier_two_request_routes_to_cold2_members() {
        // Fully uncached 160 lands in [160, ∞): cold-2. C carries it and wins over the
        // idle untaunted worker and the busy cold-1 member.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"])), (C, taint_set(&["cold-2"]))]);
        let selected = select_result(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 0, 0), (C, 0, 5), (B, 0, 0)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(C));
    }

    #[test]
    fn tier_members_pick_least_loaded() {
        // Two cold-1 members hold the same overlap; the busier one loses even though the
        // untaunted workers are idle.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"])), (B, taint_set(&["cold-1"]))]);
        let selected = select_result(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 2, 7), (B, 2, 0), (C, 0, 0), (D, 0, 0)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn tier_with_no_member_yields_overload_error() {
        // Effective prefill in the cold-1 band but nobody carries the taint: the picker
        // finds no row and the host must classify the request as overloaded, not route it
        // to warm workers.
        let pool_taints: HashMap<u64, Arc<HashSet<String>>> = HashMap::new();
        let result = select_result(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 2, 0), (B, 0, 0)],
            &pool_taints,
        );
        let Err(error) = result else {
            panic!("cold tier without members must fail");
        };
        assert!(
            matches!(
                error,
                dynamo_kv_router::scheduling::KvSchedulerError::AllEligibleWorkersOverloaded
            ),
            "expected AllEligibleWorkersOverloaded, got: {error}"
        );
    }

    #[test]
    fn long_but_cached_request_keeps_affinity() {
        // 160 tokens with 96 cached on B (six blocks) leaves 64 < 100: below every band,
        // normal selection sends it to the overlap holder even though it is busier and
        // carries no taint.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 0, 0), (B, 6, 4)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn warm_request_prefers_overlap_holder() {
        let pool_taints = HashMap::new();
        let selected = select_result(
            &two_tier_params(),
            80,
            &[(A, 5, 3), (B, 0, 0)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(A));
    }

    #[test]
    fn gap_between_bands_is_warm() {
        // Bands [100, 150) and [200, ∞): best effective prefill 160 falls in the gap and
        // selects normally across all workers.
        let params = Parameters {
            uncached_threshold_tokens: 100,
            pools: Some(vec![
                PoolTier {
                    min_uncached_tokens: 100,
                    max_uncached_tokens: Some(150),
                    taint: "cold-1".to_string(),
                },
                PoolTier {
                    min_uncached_tokens: 200,
                    max_uncached_tokens: None,
                    taint: "cold-2".to_string(),
                },
            ]),
            compute_workers: None,
            allow_warm_spillover: false,
            spillover_free_threshold: 0.5,
            allow_tier_upgrade: true,
        };
        let pool_taints = HashMap::new();
        let selected =
            select_result(&params, ISL_TOKENS, &[(A, 0, 5), (B, 0, 0)], &pool_taints).unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn convenience_mode_restricts_cold_to_first_workers() {
        // No pools, compute_workers=1: a Cold request routes to the first candidate row
        // even when later rows are idle.
        let params = single_tier_params(100, 1);
        let pool_taints = HashMap::new();
        let selected =
            select_result(&params, ISL_TOKENS, &[(A, 0, 5), (B, 0, 0)], &pool_taints).unwrap();
        assert_eq!(selected, worker(A));
    }

    #[test]
    fn convenience_mode_warm_is_unrestricted() {
        let params = single_tier_params(100, 1);
        let pool_taints = HashMap::new();
        let selected = select_result(&params, 80, &[(A, 0, 5), (B, 0, 0)], &pool_taints).unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn warm_without_affinity_avoids_cold_workers_by_default() {
        // A carries the cold-1 taint and is idle; B is warm but busy. A warm request with
        // no overlap must stay on the warm worker even though A is idle.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result(
            &two_tier_params(),
            80,
            &[(A, 0, 0), (B, 0, 6)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn spillover_disabled_rejects_when_warm_pool_drained() {
        // Only the cold worker survives eligibility: without spillover the warm request
        // must fail as overload rather than touch the cold pool.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let result = select_result_allowed(
            &two_tier_params(),
            80,
            &[(A, 0, 0), (B, 0, 6)],
            &pool_taints,
            Some(vec![A]),
        );
        let Err(error) = result else {
            panic!("drained warm pool must fail without spillover");
        };
        assert!(
            matches!(
                error,
                dynamo_kv_router::scheduling::KvSchedulerError::AllEligibleWorkersOverloaded
            ),
            "expected AllEligibleWorkersOverloaded, got: {error}"
        );
    }

    #[test]
    fn affinity_on_cold_worker_does_not_join_cold_pool_when_warm_alive() {
        // The request's largest overlap lives on the cold-1 worker, but the warm worker
        // is still eligible: the warm lane stays warm-only and the request re-prefills
        // rather than contending on the cold pool. Losing the affinity is the cost of
        // keeping the split pure.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result(
            &two_tier_params(),
            80,
            &[(A, 5, 0), (B, 0, 6)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn spillover_respects_free_threshold() {
        // Three cold-1 members and one warm worker. The picker first sees the full pool
        // through a cold request, so the remembered denominator is three; a subsequent
        // warm request with the warm worker drained and one cold member removed
        // (allowed=[B,C]) measures free fraction 2/3 >= 0.5 and may escape.
        let pool_taints = HashMap::from([
            (A, taint_set(&["cold-1"])),
            (B, taint_set(&["cold-1"])),
            (C, taint_set(&["cold-1"])),
        ]);
        let params = Parameters {
            allow_warm_spillover: true,
            ..two_tier_params()
        };
        // Prime the remembered membership with a cold request over the full pool.
        let shared: SharedColdIds = Default::default();
        shared.borrow_mut().extend([A, B, C]);
        let selected = select_result_stateful(
            &params,
            80,
            &[(A, 0, 0), (B, 0, 0), (C, 0, 0), (D, 0, 6)],
            &pool_taints,
            Some(vec![B, C]),
            Some(&shared),
        )
        .unwrap();
        assert!(selected == worker(B) || selected == worker(C));
    }

    #[test]
    fn spillover_blocked_below_free_threshold() {
        // Three cold members (primed via a cold request first), then the warm worker and
        // two cold members drained (allowed=[B]): free fraction 1/3 < 0.5 fails the
        // headroom floor, so the drained warm pool rejects with the overload
        // classification instead of borrowing.
        let pool_taints = HashMap::from([
            (A, taint_set(&["cold-1"])),
            (B, taint_set(&["cold-1"])),
            (C, taint_set(&["cold-1"])),
        ]);
        let params = Parameters {
            allow_warm_spillover: true,
            ..two_tier_params()
        };
        let shared: SharedColdIds = Default::default();
        shared.borrow_mut().extend([A, B, C]);
        let result = select_result_stateful(
            &params,
            80,
            &[(A, 0, 0), (B, 0, 0), (C, 0, 0), (D, 0, 6)],
            &pool_taints,
            Some(vec![B]),
            Some(&shared),
        );
        let Err(error) = result else {
            panic!("below headroom floor must reject");
        };
        assert!(
            matches!(
                error,
                dynamo_kv_router::scheduling::KvSchedulerError::AllEligibleWorkersOverloaded
            ),
            "expected AllEligibleWorkersOverloaded, got: {error}"
        );
    }

    #[test]
    fn spillover_free_threshold_is_validated() {
        assert!(
            Parameters {
                spillover_free_threshold: 1.1,
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                spillover_free_threshold: f64::NAN,
                ..Parameters::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                spillover_free_threshold: 0.0,
                compute_workers: Some(1),
                ..Parameters::default()
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn spillover_enabled_escapes_to_idle_cold_worker() {
        // Same drained-warm-pool scenario with the spillover switch on: the request may
        // now use the idle cold worker.
        let params = Parameters {
            allow_warm_spillover: true,
            ..two_tier_params()
        };
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result_allowed(
            &params,
            80,
            &[(A, 0, 0), (B, 0, 6)],
            &pool_taints,
            Some(vec![A]),
        )
        .unwrap();
        assert_eq!(selected, worker(A));
    }

    #[test]
    fn spillover_enabled_keeps_warm_traffic_on_warm_workers() {
        // Spillover only escapes once the warm pool is drained: with a warm worker still
        // eligible, warm traffic stays on it even though the cold worker is idle.
        let params = Parameters {
            allow_warm_spillover: true,
            ..two_tier_params()
        };
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result(&params, 80, &[(A, 0, 0), (B, 0, 6)], &pool_taints).unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn tier_upgrade_finds_higher_tier_member() {
        // Only a cold-2 member exists; a request in cold-1's band ([100, 160)) with the
        // upgrade switch on (default) routes to the cold-2 worker rather than failing.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-2"]))]);
        let selected = select_result(
            &two_tier_params(),
            150,
            &[(A, 0, 0), (B, 0, 6)],
            &pool_taints,
        )
        .unwrap();
        assert_eq!(selected, worker(A));
    }

    #[test]
    fn tier_upgrade_disabled_fails_without_members() {
        let params = Parameters {
            allow_tier_upgrade: false,
            ..two_tier_params()
        };
        let pool_taints = HashMap::from([(A, taint_set(&["cold-2"]))]);
        let result = select_result(&params, 150, &[(A, 0, 0), (B, 0, 6)], &pool_taints);
        let Err(error) = result else {
            panic!("disabled upgrade with no cold-1 members must fail");
        };
        assert!(
            matches!(
                error,
                dynamo_kv_router::scheduling::KvSchedulerError::AllEligibleWorkersOverloaded
            ),
            "expected AllEligibleWorkersOverloaded, got: {error}"
        );
    }

    #[test]
    fn tier_upgrade_never_downgrades() {
        // A cold-2 request ([160, ∞)) with only cold-1 members available must not fall
        // back down to them: it fails even though cold-1 could physically run it.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let result = select_result(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 0, 0), (B, 0, 6)],
            &pool_taints,
        );
        assert!(result.is_err());
    }

    #[test]
    fn shared_cache_hits_lower_the_cold_classification() {
        // 160-token prompt whose prefix is fully present in the shared cache (Mooncake):
        // every worker can fetch it, so effective prefill is 0 and the request is Warm —
        // it may land on the untaunted worker instead of a cold tier.
        let pool_taints = HashMap::from([(A, taint_set(&["cold-1"]))]);
        let selected = select_result_full(
            &two_tier_params(),
            ISL_TOKENS,
            &[(A, 0, 0), (B, 0, 6)],
            &pool_taints,
            None,
            None,
            Some(10), // ten blocks = 160 tokens, the whole prompt
        )
        .unwrap();
        assert_eq!(selected, worker(B));
    }

    #[test]
    fn shared_cache_hits_do_not_change_per_worker_ranking() {
        // Shared hits are worker-independent: with the same shared volume, the worker
        // holding device overlap still wins the per-row comparison.
        let pool_taints = HashMap::new();
        let selected = select_result_full(
            &two_tier_params(),
            80,
            &[(A, 5, 0), (B, 0, 6)],
            &pool_taints,
            None,
            None,
            Some(3),
        )
        .unwrap();
        assert_eq!(selected, worker(A));
    }

    #[test]
    fn unconfigured_pool_is_rejected() {
        // Neither pools nor compute_workers: startup must fail.
        assert!(Parameters::default().validate().is_err());

        let both = Parameters {
            pools: two_tier_params().pools,
            compute_workers: Some(2),
            ..Parameters::default()
        };
        assert!(both.validate().is_err());

        assert!(two_tier_params().validate().is_ok());
        assert!(single_tier_params(100, 1).validate().is_ok());
    }

    #[test]
    fn overlapping_or_empty_pools_are_rejected() {
        let overlapping = Parameters {
            pools: Some(vec![
                PoolTier {
                    min_uncached_tokens: 100,
                    max_uncached_tokens: Some(200),
                    taint: "cold-1".to_string(),
                },
                PoolTier {
                    min_uncached_tokens: 150,
                    max_uncached_tokens: None,
                    taint: "cold-2".to_string(),
                },
            ]),
            ..Parameters::default()
        };
        assert!(overlapping.validate().is_err());

        let duplicate_taint = Parameters {
            pools: Some(vec![
                PoolTier {
                    min_uncached_tokens: 100,
                    max_uncached_tokens: Some(200),
                    taint: "cold-1".to_string(),
                },
                PoolTier {
                    min_uncached_tokens: 200,
                    max_uncached_tokens: None,
                    taint: "cold-1".to_string(),
                },
            ]),
            ..Parameters::default()
        };
        assert!(duplicate_taint.validate().is_err());

        let empty = Parameters {
            pools: Some(vec![]),
            ..Parameters::default()
        };
        assert!(empty.validate().is_err());

        assert!(single_tier_params(0, 1).validate().is_err());
        assert!(single_tier_params(100, 0).validate().is_err());
    }
}
