// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::collections::HashMap;
#[cfg(any(test, feature = "bench"))]
use std::sync::Arc;

use parking_lot::Mutex;

use super::policy::WorkerSelectionPolicyStateRef;
use super::{
    LogitWeights, MaterializedSelectionInput, ScoredWorkerCandidate, WorkerCandidate,
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionInput,
    WorkerSelector, select_worker_with_policy,
};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};
use crate::scheduling::config::KvRouterConfig;
use crate::scheduling::filter::RoutingEligibility;
use crate::scheduling::types::{KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError};

#[cfg(any(test, feature = "bench"))]
fn softmax_sample_entries<T: Copy>(
    entries: Vec<(T, f64)>,
    temperature: f64,
    sample: f64,
) -> (T, f64) {
    assert!(!entries.is_empty(), "Empty logits for softmax sampling");

    let mut probabilities = Vec::with_capacity(entries.len());
    let row = softmax_sample_index(
        &entries,
        |(_, cost)| *cost,
        temperature,
        sample,
        &mut probabilities,
    );
    entries[row]
}

fn softmax_sample_index<T>(
    entries: &[T],
    cost: impl Fn(&T) -> f64,
    temperature: f64,
    sample: f64,
    probabilities: &mut Vec<f64>,
) -> usize {
    assert!(!entries.is_empty(), "Empty entries for softmax sampling");
    debug_assert_ne!(temperature, 0.0);

    let (min_cost, max_cost) = entries
        .iter()
        .map(&cost)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), cost| {
            (lo.min(cost), hi.max(cost))
        });

    probabilities.clear();
    if min_cost == max_cost {
        probabilities.resize(entries.len(), 1.0 / entries.len() as f64);
    } else {
        let range = max_cost - min_cost;
        let magnitude = if range.is_finite() {
            1.0
        } else {
            min_cost.abs().max(max_cost.abs())
        };
        let min_normalized = min_cost / magnitude;
        let scale = -1.0 / ((max_cost / magnitude - min_normalized) * temperature);
        let max_scaled = min_normalized * scale;
        probabilities.extend(
            entries
                .iter()
                .map(|entry| (cost(entry) / magnitude * scale - max_scaled).exp()),
        );
    }

    let sum: f64 = probabilities.iter().sum();
    for probability in probabilities.iter_mut() {
        *probability /= sum;
    }
    let mut cumulative = 0.0;
    for (row, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if sample <= cumulative {
            return row;
        }
    }
    entries.len() - 1
}

/// Default implementation matching the Python _cost_function.
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
    picker: DefaultWorkerPicker,
}

pub(super) struct DefaultWorkerScorer<C = KvRouterConfig> {
    pub(super) kv_router_config: C,
    pub(super) worker_type: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct DefaultScoringContext {
    min_active_prefill_tokens: usize,
    has_tier_overlap_blocks: bool,
}

pub(super) struct DefaultWorkerPicker {
    default_temperature: f64,
    // Preserve DefaultWorkerSelector's Sync contract. Zero-temperature selection never locks.
    softmax_scratch: Mutex<DefaultSoftmaxScratch>,
    #[cfg(any(test, feature = "bench"))]
    deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

#[derive(Default)]
struct DefaultSoftmaxScratch {
    entries: Vec<(WorkerWithDpRank, f64)>,
    probabilities: Vec<f64>,
}

impl std::fmt::Debug for DefaultWorkerSelector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DefaultWorkerSelector")
            .field("kv_router_config", &self.kv_router_config)
            .field("worker_type", &self.worker_type)
            .finish_non_exhaustive()
    }
}

impl Clone for DefaultWorkerSelector {
    fn clone(&self) -> Self {
        #[cfg(any(test, feature = "bench"))]
        let deterministic_rng = self.picker.deterministic_rng.clone();
        Self::from_parts(
            self.kv_router_config.clone(),
            self.worker_type,
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng,
        )
    }
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self::from_parts(
            kv_router_config.unwrap_or_default(),
            worker_type,
            #[cfg(any(test, feature = "bench"))]
            None,
        )
    }

    #[cfg(any(test, feature = "bench"))]
    pub fn new_seeded(
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
        seed: u64,
    ) -> Self {
        Self::from_parts(
            kv_router_config.unwrap_or_default(),
            worker_type,
            Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
        )
    }

    fn from_parts(
        kv_router_config: KvRouterConfig,
        worker_type: &'static str,
        #[cfg(any(test, feature = "bench"))] deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        let picker = DefaultWorkerPicker::from_parts(
            kv_router_config.router_temperature,
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng,
        );
        Self {
            kv_router_config,
            worker_type,
            picker,
        }
    }
}

#[cfg(test)]
impl DefaultWorkerScorer<KvRouterConfig> {
    pub(super) fn new(kv_router_config: KvRouterConfig, worker_type: &'static str) -> Self {
        Self {
            kv_router_config,
            worker_type,
        }
    }
}

pub(super) fn selection_weights(
    kv_router_config: &KvRouterConfig,
    request: &SchedulingRequest,
) -> LogitWeights {
    LogitWeights {
        overlap_score_credit: request
            .router_config_override
            .as_ref()
            .and_then(|config| config.overlap_score_credit)
            .unwrap_or(kv_router_config.overlap_score_credit),
        overlap_score_credit_decay: kv_router_config.overlap_score_credit_decay,
        prefill_load_scale: request
            .router_config_override
            .as_ref()
            .and_then(|config| config.prefill_load_scale)
            .unwrap_or(kv_router_config.prefill_load_scale),
        shared_cache_multiplier: request
            .router_config_override
            .as_ref()
            .and_then(|config| config.shared_cache_multiplier)
            .unwrap_or(kv_router_config.shared_cache_multiplier),
    }
}

impl DefaultScoringContext {
    fn new<C: WorkerConfigLike>(
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        weights: LogitWeights,
    ) -> Self {
        let min_active_prefill_tokens =
            if request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let mut minimum = usize::MAX;
                eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                    minimum = minimum.min(request.worker_load_for(worker).active_prefill_tokens);
                });
                if minimum == usize::MAX { 0 } else { minimum }
            } else {
                0
            };
        let has_tier_overlap_blocks = !request.overlap.tier_overlap_blocks.device.is_empty()
            || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
            || !request.overlap.tier_overlap_blocks.disk.is_empty();
        Self {
            min_active_prefill_tokens,
            has_tier_overlap_blocks,
        }
    }

    fn device_overlap(self, effective_overlap_blocks: f64, device_overlap_blocks: f64) -> f64 {
        if self.has_tier_overlap_blocks {
            device_overlap_blocks
        } else {
            effective_overlap_blocks
        }
    }
}

fn default_row(
    input: &MaterializedSelectionInput<'_>,
    context: DefaultScoringContext,
    worker: WorkerWithDpRank,
    preferred_taint_multiplier: Option<f64>,
) -> WorkerCandidate {
    input.row_with_device_overlap(
        worker,
        preferred_taint_multiplier,
        WorkerInputs::ALL,
        |effective_overlap_blocks, device_overlap_blocks| {
            context.device_overlap(effective_overlap_blocks, device_overlap_blocks)
        },
    )
}

impl<C: Borrow<KvRouterConfig>> DefaultWorkerScorer<C> {
    fn worker_logit(
        &self,
        context: &WorkerSelectionContext<'_>,
        default_context: DefaultScoringContext,
        row: &WorkerCandidate,
        formula_name: &'static str,
    ) -> f64 {
        let kv_router_config = self.kv_router_config.borrow();
        let weights = context.weights;
        let worker = row.worker;
        let cache = &row.cache;
        let load = &row.load;
        let effective_overlap_blocks = cache.effective_overlap_blocks;
        let device_overlap_blocks = cache.device_overlap_blocks;
        let shared_beyond_device_blocks = cache.shared_beyond_device_blocks;
        let shared_overlap_blocks =
            weights.shared_cache_multiplier * shared_beyond_device_blocks as f64;
        // Normalize backlog above the least-loaded eligible worker by this request's
        // size. The rational decay softly trades cache locality for prefill balance,
        // while leaving workers at the load floor with their full device credit.
        let overlap_credit_decay =
            if context.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let excess_active_prefill_blocks = load
                    .active_prefill_tokens
                    .saturating_sub(default_context.min_active_prefill_tokens)
                    as f64
                    / context.block_size as f64;
                let normalized_prefill_load =
                    excess_active_prefill_blocks / context.request_blocks as f64;
                1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load)
            } else {
                1.0
            };
        let effective_overlap_score_credit = weights.overlap_score_credit * overlap_credit_decay;
        let overlap_credit_blocks = effective_overlap_score_credit * device_overlap_blocks
            + kv_router_config.host_cache_hit_weight * cache.host_overlap_blocks
            + kv_router_config.disk_cache_hit_weight * cache.disk_overlap_blocks
            + shared_overlap_blocks;
        let decode_cost_blocks = load.decode_cost_blocks;
        let active_request_cost_blocks =
            kv_router_config.decode_active_request_weight * load.active_requests as f64;

        // Decode routers normally force `overlap_score_credit=0` through the
        // per-request override, which preserves load-only disagg routing. When
        // conditional disagg leaves a positive overlap credit in place, prefer
        // cache-hot decode workers while still charging decode backlog.
        if self.worker_type == "decode"
            && !context.track_prefill_tokens
            && weights.overlap_score_credit > 0.0
        {
            // Clamp at zero because downstream taint multipliers assume non-negative scores.
            // This loses ordering between workers whose overlap fully offsets decode load, but
            // avoids inverting taint preference among negative-score workers.
            let overlap_adjusted_decode_blocks =
                (decode_cost_blocks - overlap_credit_blocks).max(0.0);
            let logit = overlap_adjusted_decode_blocks + active_request_cost_blocks;
            // Stamped for the same reason as the two rows below: this row is emitted from the
            // `SchedulerQueueActor` task, so the logging layer cannot attach request identity to
            // it, and this branch returns early without reaching them.
            tracing::debug!(
                request_id = context.request_id,
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = max(0, decode_blocks - overlap_credit_blocks) + active_request_cost_blocks \
                 = max(0, {decode_cost_blocks:.3} - {overlap_credit_blocks:.3}) + {active_request_cost_blocks:.3}",
                worker.worker_id,
                worker.dp_rank,
            );
            return logit;
        }

        let adjusted_prefill_blocks = (load.raw_prefill_blocks - overlap_credit_blocks).max(0.0);
        let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
        let logit = prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks;

        // These rows are emitted from the `SchedulerQueueActor` task, which `scheduling::queue`
        // spawns without the caller's request span, so the logging layer cannot attach
        // `x_request_id`/`trace_id` to them. Stamp the identity the row needs to be self-joining:
        // `request_id` is the same value `[ROUTING] Best` logs, and `worker_type` separates the
        // prefill-pool and decode-pool decisions that interleave into one log. Both are evaluated
        // inside the macro so they cost nothing when DEBUG is disabled.
        if shared_beyond_device_blocks > 0 {
            tracing::debug!(
                request_id = context.request_id,
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks, \
                 {} shared blocks beyond device (multiplier={shared_cache_multiplier:.2}): {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} + {active_request_cost_blocks:.3} \
                 (raw_prefill_blocks: {:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3}, \
                 overlap_credit_decay: {overlap_credit_decay:.3})",
                worker.worker_id,
                worker.dp_rank,
                shared_beyond_device_blocks,
                load.raw_prefill_blocks,
                shared_cache_multiplier = weights.shared_cache_multiplier,
                prefill_load_scale = weights.prefill_load_scale
            );
        } else {
            tracing::debug!(
                request_id = context.request_id,
                worker_type = self.worker_type,
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks + active_request_cost_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} + {active_request_cost_blocks:.3} \
                 (raw_prefill_blocks: {:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3}, \
                 overlap_credit_decay: {overlap_credit_decay:.3})",
                worker.worker_id,
                worker.dp_rank,
                load.raw_prefill_blocks,
                prefill_load_scale = weights.prefill_load_scale
            );
        }

        logit
    }

    #[inline]
    fn worker_cost(
        &self,
        context: &WorkerSelectionContext<'_>,
        default_context: DefaultScoringContext,
        row: &WorkerCandidate,
    ) -> f64 {
        let base_score = self.worker_logit(context, default_context, row, "Formula");
        match row.preferred_taint_multiplier {
            // NOTE: This multiplicative bias assumes a non-negative score. Negative
            // overlap scores expose its pre-existing sign sensitivity; keep it for now.
            Some(multiplier) => base_score * multiplier,
            None => base_score,
        }
    }
}

impl DefaultWorkerPicker {
    pub(super) fn new(default_temperature: f64) -> Self {
        Self::from_parts(
            default_temperature,
            #[cfg(any(test, feature = "bench"))]
            None,
        )
    }
}

fn minimum_cost_index(
    candidates: &[ScoredWorkerCandidate],
    mut random_index: impl FnMut(usize) -> usize,
) -> usize {
    let mut best_row = 0;
    let mut best_cost = f64::INFINITY;
    let mut tie_count = 0;
    for (row, candidate) in candidates.iter().enumerate() {
        let cost = candidate.cost;
        if cost < best_cost {
            best_row = row;
            best_cost = cost;
            tie_count = 1;
        } else if cost == best_cost {
            tie_count += 1;
            if random_index(tie_count) == 0 {
                best_row = row;
            }
        }
    }
    best_row
}

#[inline(always)]
pub(super) fn pick_default_worker<C: WorkerConfigLike>(
    scorer: &DefaultWorkerScorer<&KvRouterConfig>,
    picker: &DefaultWorkerPicker,
    input: &MaterializedSelectionInput<'_>,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
) -> Option<(WorkerWithDpRank, f64)> {
    let default_context =
        DefaultScoringContext::new(workers, request, eligibility, input.context.weights);
    if let Some(worker) = eligibility.pinned_worker() {
        let row = default_row(input, default_context, worker, None);
        return Some((
            worker,
            scorer.worker_logit(&input.context, default_context, &row, "Pinned formula"),
        ));
    }

    let temperature = input
        .context
        .router_temperature_override
        .unwrap_or(scorer.kv_router_config.router_temperature);
    let get_score = |worker, config: &C| {
        let preferred_taint_multiplier = request
            .routing_constraints
            .preferred_taint_multiplier(config.taints());
        scorer.worker_cost(
            &input.context,
            default_context,
            &default_row(input, default_context, worker, preferred_taint_multiplier),
        )
    };

    #[cfg(any(test, feature = "bench"))]
    if let Some(rng) = &picker.deterministic_rng {
        let mut candidates = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| candidates.push(worker));
        candidates.sort_unstable_by_key(|worker| (worker.worker_id, worker.dp_rank));
        if candidates.is_empty() {
            return None;
        }

        let mut rng = rng.lock();
        let get_candidate_score = |worker| get_score(worker, &workers[&worker.worker_id]);
        if temperature == 0.0 {
            let mut best_worker = None;
            let mut best_cost = f64::INFINITY;
            let mut tie_count = 0;
            for worker in candidates {
                let cost = get_candidate_score(worker);
                if cost < best_cost {
                    best_worker = Some(worker);
                    best_cost = cost;
                    tie_count = 1;
                } else if cost == best_cost {
                    tie_count += 1;
                    if rng.usize(0..tie_count) == 0 {
                        best_worker = Some(worker);
                    }
                }
            }
            return best_worker.map(|worker| (worker, best_cost));
        }

        let entries = candidates
            .into_iter()
            .map(|worker| (worker, get_candidate_score(worker)))
            .collect();
        return Some(softmax_sample_entries(entries, temperature, rng.f64()));
    }

    if temperature == 0.0 {
        let mut best_worker = None;
        let mut best_cost = f64::INFINITY;
        let mut tie_count = 0;
        eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
            let cost = get_score(worker, config);
            if cost < best_cost {
                best_worker = Some(worker);
                best_cost = cost;
                tie_count = 1;
            } else if cost == best_cost {
                tie_count += 1;
                if fastrand::usize(0..tie_count) == 0 {
                    best_worker = Some(worker);
                }
            }
        });
        return best_worker.map(|worker| (worker, best_cost));
    }

    let mut scratch = picker.softmax_scratch.lock();
    scratch.entries.clear();
    eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
        scratch.entries.push((worker, get_score(worker, config)));
    });
    if scratch.entries.is_empty() {
        None
    } else {
        let DefaultSoftmaxScratch {
            entries,
            probabilities,
        } = &mut *scratch;
        let row = softmax_sample_index(
            entries,
            |(_, cost)| *cost,
            temperature,
            fastrand::f64(),
            probabilities,
        );
        Some(entries[row])
    }
}

impl DefaultWorkerPicker {
    fn from_parts(
        default_temperature: f64,
        #[cfg(any(test, feature = "bench"))] deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        Self {
            default_temperature,
            softmax_scratch: Mutex::default(),
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng,
        }
    }
}

impl WorkerPicker for DefaultWorkerPicker {
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let temperature = context
            .router_temperature_override
            .unwrap_or(self.default_temperature);
        #[cfg(any(test, feature = "bench"))]
        if let Some(rng) = &self.deterministic_rng {
            let mut rng = rng.lock();
            if temperature == 0.0 {
                return Ok(minimum_cost_index(candidates, |count| rng.usize(0..count)));
            }
            let sample = rng.f64();
            drop(rng);
            return Ok(softmax_sample_index(
                candidates,
                |candidate| candidate.cost,
                temperature,
                sample,
                &mut self.softmax_scratch.get_mut().probabilities,
            ));
        }
        if temperature == 0.0 {
            return Ok(minimum_cost_index(candidates, |count| {
                fastrand::usize(0..count)
            }));
        }
        Ok(softmax_sample_index(
            candidates,
            |candidate| candidate.cost,
            temperature,
            fastrand::f64(),
            &mut self.softmax_scratch.get_mut().probabilities,
        ))
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    #[inline(always)]
    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        let (workers, request, eligibility, block_size) = input.into_configured()?;
        select_worker_with_policy(
            &self.kv_router_config,
            self.worker_type,
            WorkerSelectionPolicyStateRef::Default(&self.picker),
            workers,
            request,
            eligibility,
            block_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use rustc_hash::FxHashMap;

    use super::super::test_support::*;
    use super::*;
    use crate::config::RouterConfigOverride;
    use crate::protocols::SharedCacheHits;
    use crate::scheduling::{OverlapSignals, ScheduleMode};

    fn worker_logit(
        selector: &DefaultWorkerSelector,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        weights: LogitWeights,
    ) -> f64 {
        let workers = HashMap::from([(worker.worker_id, TaintedWorkerConfig::default())]);
        let input = MaterializedSelectionInput::new(request, block_size, weights);
        let default_context =
            DefaultScoringContext::new(&workers, request, request.eligibility(), weights);
        DefaultWorkerScorer::new(selector.kv_router_config.clone(), selector.worker_type)
            .worker_logit(
                &input.context,
                default_context,
                &default_row(&input, default_context, worker, None),
                "test",
            )
    }

    #[test]
    fn default_scoring_context_only_computes_minimum_when_decay_uses_it() {
        let workers = HashMap::from([
            (0, TaintedWorkerConfig::default()),
            (1, TaintedWorkerConfig::default()),
        ]);
        let mut request = base_request(64);
        request.worker_loads.insert(
            WorkerWithDpRank::from_worker_id(0),
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 7,
                ..Default::default()
            },
        );
        request.worker_loads.insert(
            WorkerWithDpRank::from_worker_id(1),
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 11,
                ..Default::default()
            },
        );
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 1.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };

        let default_context =
            DefaultScoringContext::new(&workers, &request, request.eligibility(), weights);
        assert_eq!(default_context.min_active_prefill_tokens, 7);

        let weights_without_decay = LogitWeights {
            overlap_score_credit_decay: 0.0,
            ..weights
        };
        assert_eq!(
            DefaultScoringContext::new(
                &workers,
                &request,
                request.eligibility(),
                weights_without_decay,
            )
            .min_active_prefill_tokens,
            0
        );
    }

    #[test]
    fn softmax_sample_orders_extreme_finite_costs() {
        let result = softmax_sample_entries(vec![(0, -f64::MAX), (1, f64::MAX)], 1.0, 0.6);
        assert_eq!(result.0, 0);
    }

    #[test]
    fn test_default_selector_randomizes_zero_temperature_ties() {
        use crate::test_utils::SimpleWorkerConfig;

        let config = KvRouterConfig {
            router_temperature: 0.0,
            ..Default::default()
        };
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let workers = HashMap::from([
            (10, SimpleWorkerConfig::default()),
            (20, SimpleWorkerConfig::default()),
            (30, SimpleWorkerConfig::default()),
        ]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        };
        let mut selected = [false; 3];

        for _ in 0..120 {
            let result = selector
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    16,
                ))
                .unwrap();
            match result.worker.worker_id {
                10 => selected[0] = true,
                20 => selected[1] = true,
                30 => selected[2] = true,
                worker_id => panic!("unexpected worker id: {worker_id}"),
            }
        }

        let selected_count = selected.into_iter().filter(|seen| *seen).count();
        assert!(
            selected_count > 1,
            "zero-temperature tie-breaking should not always select the same worker"
        );
    }

    #[test]
    fn seeded_selector_is_stable_for_ties_and_temperature_sampling() {
        use crate::test_utils::SimpleWorkerConfig;

        for (temperature, expected_prefix) in [
            (
                0.0,
                [
                    10, 30, 10, 10, 30, 20, 10, 10, 10, 20, 10, 20, 30, 10, 20, 20,
                ],
            ),
            (
                0.7,
                [
                    30, 20, 30, 10, 20, 20, 30, 20, 30, 10, 10, 30, 30, 20, 30, 30,
                ],
            ),
        ] {
            let config = KvRouterConfig {
                router_temperature: temperature,
                ..Default::default()
            };
            let mut first = DefaultWorkerSelector::new_seeded(
                Some(KvRouterConfig {
                    router_temperature: 0.0,
                    ..Default::default()
                }),
                "test",
                42,
            );
            first.kv_router_config.router_temperature = temperature;
            let second = DefaultWorkerSelector::new_seeded(Some(config), "test", 42);
            let first_workers = HashMap::from([
                (30, SimpleWorkerConfig::default()),
                (10, SimpleWorkerConfig::default()),
                (20, SimpleWorkerConfig::default()),
            ]);
            let second_workers = HashMap::from([
                (20, SimpleWorkerConfig::default()),
                (30, SimpleWorkerConfig::default()),
                (10, SimpleWorkerConfig::default()),
            ]);
            let request = base_request(16);

            let first_sequence = (0..64)
                .map(|_| {
                    first
                        .select_worker(WorkerSelectionInput::configured(
                            &first_workers,
                            &request,
                            request.eligibility(),
                            16,
                        ))
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();
            let second_sequence = (0..64)
                .map(|_| {
                    second
                        .select_worker(WorkerSelectionInput::configured(
                            &second_workers,
                            &request,
                            request.eligibility(),
                            16,
                        ))
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();

            assert_eq!(first_sequence, second_sequence);
            assert_eq!(
                first_sequence
                    .iter()
                    .take(expected_prefix.len())
                    .map(|worker| worker.worker_id)
                    .collect::<Vec<_>>(),
                expected_prefix,
            );
        }
    }

    #[test]
    fn per_request_overrides_change_selection() {
        use crate::test_utils::SimpleWorkerConfig;

        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);
        let mut request = base_request(4);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 2);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 2);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 1,
                ..Default::default()
            },
        );
        #[allow(clippy::single_range_in_vec_init)]
        let shared_cache_hits = SharedCacheHits::from_ranges(vec![0..4]);
        request.shared_cache_hits = Some(shared_cache_hits);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
            router_temperature: 0.0,
            ..Default::default()
        };
        let select = |request: &SchedulingRequest| {
            DefaultWorkerSelector::new_seeded(Some(config.clone()), "test", 42)
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    request,
                    request.eligibility(),
                    1,
                ))
                .unwrap()
                .worker
        };

        assert_eq!(select(&request), warm_worker);

        for (name, config_override) in [
            (
                "overlap_score_credit",
                RouterConfigOverride {
                    overlap_score_credit: Some(0.0),
                    ..Default::default()
                },
            ),
            (
                "prefill_load_scale",
                RouterConfigOverride {
                    prefill_load_scale: Some(0.0),
                    ..Default::default()
                },
            ),
            (
                "shared_cache_multiplier",
                RouterConfigOverride {
                    shared_cache_multiplier: Some(1.0),
                    ..Default::default()
                },
            ),
            (
                "router_temperature",
                RouterConfigOverride {
                    router_temperature: Some(1.0),
                    ..Default::default()
                },
            ),
        ] {
            request.router_config_override = Some(config_override);
            assert_eq!(select(&request), cold_worker, "{name} override was ignored");
        }
    }

    #[test]
    fn test_overloaded_high_overlap_worker_is_skipped() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request
            .overlap
            .effective_overlap_blocks
            .insert(worker0, 4.0);
        request.overlap.effective_cached_tokens.insert(worker0, 64);

        let overloaded_worker_ids = HashSet::from([0]);
        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
                16,
            ))
            .unwrap();

        assert_eq!(result.worker.worker_id, 1);
    }

    #[test]
    fn test_all_eligible_workers_overloaded_returns_overload_error() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 1.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let request = base_request(16);
        let overloaded_worker_ids = HashSet::from([0, 1]);

        let result = selector.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        ));

        assert!(matches!(
            result,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[test]
    fn test_overloaded_pinned_worker_is_not_rerouted() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let mut request = base_request(16);
        request.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
        let overloaded_worker_ids = HashSet::from([0]);

        let result = selector.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        ));

        assert!(matches!(
            result,
            Err(KvSchedulerError::PinnedWorkerOverloaded { worker_id: 0 })
        ));
    }

    #[test]
    fn test_required_taints_return_no_endpoints_when_no_worker_matches() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([(
            10,
            TaintedWorkerConfig {
                taints: HashSet::from(["mdc-a".to_string()]),
            },
        )]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector.select_worker(WorkerSelectionInput::configured(
            &workers,
            &request,
            request.eligibility(),
            16,
        ));
        assert!(matches!(result, Err(KvSchedulerError::NoEndpoints)));
    }

    #[test]
    fn test_required_taints_filter_out_incompatible_workers() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 20);
    }

    #[test]
    fn test_required_taints_switch_matching_worker_sets_by_label() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let name_a = "mdc-a".to_string();
        let name_b = "mdc-b".to_string();
        let name_c = "mdc-c".to_string();
        let taint_a = TaintedWorkerConfig {
            taints: HashSet::from([name_a.clone()]),
        };
        let taint_b = TaintedWorkerConfig {
            taints: HashSet::from([name_b.clone()]),
        };
        let taint_c = TaintedWorkerConfig {
            taints: HashSet::from([name_c.clone()]),
        };
        let workers = HashMap::from([
            (10, taint_a.clone()),
            (11, taint_a),
            (20, taint_b.clone()),
            (21, taint_b),
            (30, taint_c.clone()),
            (31, taint_c),
        ]);

        for (required_taint, expected_worker_id, noisy_worker_id) in [
            (name_a, 10_u64, 11_u64),
            (name_b, 20_u64, 21_u64),
            (name_c, 30_u64, 31_u64),
        ] {
            let mut decode_blocks = FxHashMap::default();
            decode_blocks.insert(WorkerWithDpRank::from_worker_id(expected_worker_id), 0);
            decode_blocks.insert(WorkerWithDpRank::from_worker_id(noisy_worker_id), 400_000);

            let request = SchedulingRequest {
                mode: ScheduleMode::QueryOnly {
                    request_id: Some("test".into()),
                },
                token_seq: None,
                isl_tokens: 16,
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: HashMap::default(),
                    effective_cached_tokens: HashMap::default(),
                },
                router_hint_candidates: None,
                retain_router_hint_chain: false,
                worker_loads: worker_loads_with_active_decode(decode_blocks),
                track_prefill_tokens: true,
                router_config_override: None,
                lora_name: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_context: None,
                expected_output_tokens: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: crate::protocols::RoutingConstraints {
                    required_taints: HashSet::from([required_taint.clone()]),
                    preferred_taints: HashMap::new(),
                },
                shared_cache_hits: None,
                resp_tx: None,
            };

            let result = selector
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    16,
                ))
                .unwrap();
            assert_eq!(
                result.worker.worker_id, expected_worker_id,
                "required taint {required_taint} should route only within its compatible worker set"
            );
        }
    }

    #[test]
    fn test_preferred_taints_prefer_matching_worker() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(10), 100);
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(20), 90);

        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::new(),
                preferred_taints: HashMap::from([("mdc-a".to_string(), 0.85)]),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 10);
    }

    #[test]
    fn test_negative_preferred_taints_avoid_matching_worker() {
        let selector = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                router_temperature: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let workers = HashMap::from([
            (
                10,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-a".to_string()]),
                },
            ),
            (
                20,
                TaintedWorkerConfig {
                    taints: HashSet::from(["mdc-b".to_string()]),
                },
            ),
        ]);
        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(10), 90);
        decode_blocks.insert(WorkerWithDpRank::from_worker_id(20), 100);

        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: 16,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::new(),
                preferred_taints: HashMap::from([("mdc-a".to_string(), -0.25)]),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert_eq!(result.worker.worker_id, 20);
    }

    /// Test the scoring formula with shared cache hits.
    ///
    /// Request [A, B, C, D], shared_cache_multiplier=0.5, block_size=1
    /// - Worker 0: device=[A,B] (overlap=2), shared has [A,B,C,D] -> shared_beyond=2
    ///   adjusted_prefill = isl - 2 - 0.5*2 = 4-2-1 = 1, logit = 1.0 * 1 + 0 = 1.0
    /// - Worker 1: device=[] (overlap=0), shared has [A,B,C,D] -> shared_beyond=4
    ///   adjusted_prefill = isl - 0.5*4 = 4-2 = 2, logit = 1.0 * 2 + 0 = 2.0
    ///
    /// Worker 0 has lower logit (less work), so it wins.
    #[test]
    fn test_shared_cache_hits_scoring() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 1u32;
        let isl = 4usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 2.0);
        // worker1 has 0 overlap (not in map)

        let mut effective_cached_tokens = HashMap::new();
        effective_cached_tokens.insert(worker0, 2);

        let mut tier_overlap_blocks = crate::scheduling::TierOverlapBlocks::default();
        tier_overlap_blocks.device.insert(worker0, 2);

        #[allow(clippy::single_range_in_vec_init)]
        let shared_hits = SharedCacheHits::from_ranges(vec![0..4]);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            shared_cache_multiplier: 0.5,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks,
                effective_overlap_blocks,
                effective_cached_tokens,
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: Some(shared_hits),
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        // Worker 0 should win: logit 1.0 < 2.0
        assert_eq!(
            result.worker, worker0,
            "Worker 0 should be selected (lower logit due to device and shared cache)"
        );
    }

    #[test]
    fn test_prefill_load_scale_applies_after_overlap_credits() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut effective_cached_tokens = HashMap::new();
        effective_cached_tokens.insert(worker0, 32);

        let mut tier_overlap_blocks = crate::scheduling::TierOverlapBlocks::default();
        tier_overlap_blocks.device.insert(worker0, 2);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            prefill_load_scale: 2.0,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker0, 3);
        decode_blocks.insert(worker1, 0);

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks,
                effective_overlap_blocks: HashMap::new(),
                effective_cached_tokens,
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "prefill load scale should apply before adding decode block load"
        );
    }

    #[test]
    fn test_overlap_credit_above_one_can_prefer_colocated_worker() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);

        let mut request = base_request(128);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 4);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 64);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 5,
                ..Default::default()
            },
        );

        let normal_credit = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.0,
                ..Default::default()
            }),
            "test",
        );
        let amplified_credit = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit: 1.5,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            normal_credit
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            cold_worker
        );
        assert_eq!(
            amplified_credit
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            warm_worker
        );
    }

    #[test]
    fn test_worker_logit_clamps_non_decode_overlap_credit() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.overlap.effective_cached_tokens.insert(worker, 96);
        request.overlap.tier_overlap_blocks.device.insert(worker, 6);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 16,
                active_decode_blocks: 2,
                active_requests: 0,
                additional_active_blocks: 3,
            },
        );
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 2.0,
            shared_cache_multiplier: 0.0,
        };

        assert_eq!(worker_logit(&selector, &request, worker, 16, weights), 7.0);

        request.track_prefill_tokens = false;
        assert_eq!(worker_logit(&selector, &request, worker, 16, weights), 5.0);
    }

    #[test]
    fn test_worker_logit_can_charge_active_requests() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(0);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 100,
                active_requests: 4,
                ..Default::default()
            },
        );
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };
        let default = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let weighted = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                decode_active_request_weight: 32.0,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(worker_logit(&default, &request, worker, 16, weights), 100.0);
        assert_eq!(
            worker_logit(&weighted, &request, worker, 16, weights),
            228.0
        );
    }

    #[test]
    fn test_decode_worker_logit_credits_overlap_without_prefill_tracking() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.track_prefill_tokens = false;
        request.overlap.tier_overlap_blocks.device.insert(worker, 3);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_decode_blocks: 10,
                ..Default::default()
            },
        );
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "decode");
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 0.0,
        };

        assert_eq!(worker_logit(&selector, &request, worker, 16, weights), 7.0);
    }

    #[test]
    fn test_overlap_credit_decay_can_prefer_less_loaded_cold_worker() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let warm_worker = WorkerWithDpRank::from_worker_id(0);
        let cold_worker = WorkerWithDpRank::from_worker_id(1);
        let workers = HashMap::from([
            (warm_worker.worker_id, SimpleWorkerConfig::default()),
            (cold_worker.worker_id, SimpleWorkerConfig::default()),
        ]);

        let mut request = base_request(64);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(warm_worker, 4);
        request
            .overlap
            .effective_cached_tokens
            .insert(warm_worker, 64);
        request.worker_loads.insert(
            warm_worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 48,
                ..Default::default()
            },
        );

        let no_decay = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 0.0,
                ..Default::default()
            }),
            "test",
        );
        let with_decay = DefaultWorkerSelector::new(
            Some(KvRouterConfig {
                overlap_score_credit_decay: 1.0,
                ..Default::default()
            }),
            "test",
        );

        assert_eq!(
            no_decay
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            warm_worker
        );
        assert_eq!(
            with_decay
                .select_worker(WorkerSelectionInput::configured(
                    &workers,
                    &request,
                    request.eligibility(),
                    block_size
                ))
                .unwrap()
                .worker,
            cold_worker
        );
    }

    #[test]
    fn test_effective_overlap_falls_back_when_tier_blocks_are_absent() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 4.0);

        let config = KvRouterConfig {
            overlap_score_credit: 1.0,
            router_temperature: 0.0,
            ..Default::default()
        };

        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());
        workers.insert(1, SimpleWorkerConfig::default());

        let mut decode_blocks = FxHashMap::default();
        decode_blocks.insert(worker0, 1);
        decode_blocks.insert(worker1, 0);

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks,
                effective_cached_tokens: HashMap::new(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "effective overlap should still credit older callers without tier maps"
        );
    }

    #[test]
    fn summary_overlap_fallback_preserves_default_shared_prefix() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let workers = HashMap::from([(worker.worker_id, TaintedWorkerConfig::default())]);
        let mut request = base_request(64);
        request.overlap.effective_overlap_blocks.insert(worker, 2.0);
        #[allow(clippy::single_range_in_vec_init)]
        let shared_hits = SharedCacheHits::from_ranges(vec![0..4]);
        request.shared_cache_hits = Some(shared_hits);
        let weights = LogitWeights {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: 0.0,
            prefill_load_scale: 1.0,
            shared_cache_multiplier: 1.0,
        };
        let input = MaterializedSelectionInput::new(&request, 16, weights);
        let default_context =
            DefaultScoringContext::new(&workers, &request, request.eligibility(), weights);
        let custom_row = input.row(worker, None, WorkerInputs::CACHE);
        let default_row = default_row(&input, default_context, worker, None);

        assert_eq!(custom_row.cache.device_overlap_blocks, 0.0);
        assert_eq!(custom_row.cache.shared_beyond_device_blocks, 4);
        assert_eq!(default_row.cache.device_overlap_blocks, 2.0);
        assert_eq!(default_row.cache.shared_beyond_device_blocks, 2);
    }

    /// Without shared cache hits, the scoring should be unchanged.
    #[test]
    fn test_no_shared_cache_unchanged() {
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut effective_overlap_blocks = HashMap::new();
        effective_overlap_blocks.insert(worker0, 2.0);

        let config = KvRouterConfig::default();
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens: isl,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks,
                effective_cached_tokens: HashMap::new(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                block_size,
            ))
            .unwrap();

        assert_eq!(result.worker, worker0);
    }
}
