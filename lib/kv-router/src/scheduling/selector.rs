// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Borrow;
use std::collections::HashMap;
use std::marker::PhantomData;
#[cfg(any(test, feature = "bench"))]
use std::sync::Arc;

#[cfg(any(test, feature = "bench"))]
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use super::config::KvRouterConfig;
use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// A trait that users can implement to define custom selection logic.
///
/// Generic over `C` so that the scheduling layer does not depend on a concrete config type.
pub trait WorkerSelector<C: WorkerConfigLike> {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

/// Helper function for softmax sampling.
/// Returns the selected worker and its logit.
fn softmax_sample(
    logits: &FxHashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> (WorkerWithDpRank, f64) {
    softmax_sample_with_sample(logits, temperature, fastrand::f64())
}

fn softmax_sample_with_sample(
    logits: &FxHashMap<WorkerWithDpRank, f64>,
    temperature: f64,
    sample: f64,
) -> (WorkerWithDpRank, f64) {
    assert!(!logits.is_empty(), "Empty logits for softmax sampling");

    if temperature == 0.0 {
        let (worker, logit) = logits
            .iter()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .expect("logits non-empty");
        return (*worker, *logit);
    }

    let entries: Vec<(WorkerWithDpRank, f64)> = logits.iter().map(|(w, l)| (*w, *l)).collect();
    softmax_sample_entries(entries, temperature, sample)
}

fn softmax_sample_entries(
    entries: Vec<(WorkerWithDpRank, f64)>,
    temperature: f64,
    sample: f64,
) -> (WorkerWithDpRank, f64) {
    assert!(!entries.is_empty(), "Empty logits for softmax sampling");

    let (min_val, max_val) = entries
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (_, v)| {
            (lo.min(*v), hi.max(*v))
        });

    let mut probs = if min_val == max_val {
        vec![1.0 / entries.len() as f64; entries.len()]
    } else {
        // Negate logits and rescale to [−1/temperature, 0] for numerical stability
        // before softmax. Subtracting the max (which maps to min_val) keeps exp() inputs ≤ 0.
        let scale = -1.0 / ((max_val - min_val) * temperature);
        let max_scaled = min_val * scale;
        entries
            .iter()
            .map(|(_, v)| (v * scale - max_scaled).exp())
            .collect::<Vec<f64>>()
    };

    let sum: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);

    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return entries[i];
        }
    }

    *entries.last().unwrap()
}

/// Default implementation matching the Python _cost_function.
#[derive(Debug, Clone)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
    #[cfg(any(test, feature = "bench"))]
    deterministic_rng: Option<Arc<Mutex<fastrand::Rng>>>,
}

#[derive(Debug, Clone, Copy)]
struct LogitWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

struct WorkerSelectionInput<'a, C> {
    workers: &'a HashMap<WorkerId, C>,
    request: &'a SchedulingRequest,
    eligibility: RoutingEligibility<'a>,
    context: WorkerSelectionContext<'a>,
}

struct WorkerSelectionContext<'a> {
    request_id: &'a str,
    request_blocks: u64,
    block_size: u32,
    track_prefill_tokens: bool,
    weights: LogitWeights,
    min_active_prefill_tokens: usize,
}

struct WorkerSelectionRow {
    worker: WorkerWithDpRank,
    effective_overlap_blocks: f64,
    device_overlap_blocks: f64,
    host_overlap_blocks: f64,
    disk_overlap_blocks: f64,
    shared_beyond: u32,
    raw_prefill_blocks: f64,
    active_prefill_tokens: usize,
    decode_cost_blocks: f64,
    active_requests: usize,
    preferred_taint_multiplier: Option<f64>,
}

trait WorkerScorer {
    fn score(&self, context: &WorkerSelectionContext<'_>, row: &WorkerSelectionRow) -> f64;
}

trait WorkerPicker<C: WorkerConfigLike, S: WorkerScorer> {
    fn pick(&self, scorer: &S, input: &WorkerSelectionInput<'_, C>) -> (WorkerWithDpRank, f64);
}

struct DefaultWorkerScorer<C> {
    kv_router_config: C,
    worker_type: &'static str,
}

struct DefaultWorkerPicker<'a> {
    default_temperature: f64,
    #[cfg(any(test, feature = "bench"))]
    deterministic_rng: Option<&'a Arc<Mutex<fastrand::Rng>>>,
    selector_lifetime: PhantomData<&'a DefaultWorkerSelector>,
}
impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng: None,
        }
    }

    #[cfg(any(test, feature = "bench"))]
    pub fn new_seeded(
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
        seed: u64,
    ) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
            deterministic_rng: Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
        }
    }
}

impl<'a> DefaultWorkerScorer<&'a KvRouterConfig> {
    fn from_selector(selector: &'a DefaultWorkerSelector) -> Self {
        Self {
            kv_router_config: &selector.kv_router_config,
            worker_type: selector.worker_type,
        }
    }
}

impl<C: Borrow<KvRouterConfig>> DefaultWorkerScorer<C> {
    fn selection_weights(&self, request: &SchedulingRequest) -> LogitWeights {
        let kv_router_config = self.kv_router_config.borrow();
        LogitWeights {
            overlap_score_credit: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.overlap_score_credit)
                .unwrap_or(kv_router_config.overlap_score_credit),
            overlap_score_credit_decay: kv_router_config.overlap_score_credit_decay,
            prefill_load_scale: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.prefill_load_scale)
                .unwrap_or(kv_router_config.prefill_load_scale),
            shared_cache_multiplier: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.shared_cache_multiplier)
                .unwrap_or(kv_router_config.shared_cache_multiplier),
        }
    }

    fn worker_logit(
        &self,
        context: &WorkerSelectionContext<'_>,
        row: &WorkerSelectionRow,
        formula_name: &'static str,
    ) -> f64 {
        let kv_router_config = self.kv_router_config.borrow();
        let weights = context.weights;
        let worker = row.worker;
        let effective_overlap_blocks = row.effective_overlap_blocks;
        let shared_overlap_blocks = weights.shared_cache_multiplier * row.shared_beyond as f64;
        // Normalize backlog above the least-loaded eligible worker by this request's
        // size. The rational decay softly trades cache locality for prefill balance,
        // while leaving workers at the load floor with their full device credit.
        let overlap_credit_decay = if context.track_prefill_tokens
            && weights.overlap_score_credit_decay > 0.0
        {
            let excess_active_prefill_blocks =
                row.active_prefill_tokens
                    .saturating_sub(context.min_active_prefill_tokens) as f64
                    / context.block_size as f64;
            let normalized_prefill_load =
                excess_active_prefill_blocks / context.request_blocks as f64;
            1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load)
        } else {
            1.0
        };
        let effective_overlap_score_credit = weights.overlap_score_credit * overlap_credit_decay;
        let overlap_credit_blocks = effective_overlap_score_credit * row.device_overlap_blocks
            + kv_router_config.host_cache_hit_weight * row.host_overlap_blocks
            + kv_router_config.disk_cache_hit_weight * row.disk_overlap_blocks
            + shared_overlap_blocks;
        let decode_cost_blocks = row.decode_cost_blocks;
        let active_request_cost_blocks =
            kv_router_config.decode_active_request_weight * row.active_requests as f64;

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
            tracing::debug!(
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = max(0, decode_blocks - overlap_credit_blocks) + active_request_cost_blocks \
                 = max(0, {decode_cost_blocks:.3} - {overlap_credit_blocks:.3}) + {active_request_cost_blocks:.3}",
                worker.worker_id,
                worker.dp_rank,
            );
            return logit;
        }

        let adjusted_prefill_blocks = (row.raw_prefill_blocks - overlap_credit_blocks).max(0.0);
        let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
        let logit = prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks;

        // These rows are emitted from the `SchedulerQueueActor` task, which `scheduling::queue`
        // spawns without the caller's request span, so the logging layer cannot attach
        // `x_request_id`/`trace_id` to them. Stamp the identity the row needs to be self-joining:
        // `request_id` is the same value `[ROUTING] Best` logs, and `worker_type` separates the
        // prefill-pool and decode-pool decisions that interleave into one log. Both are evaluated
        // inside the macro so they cost nothing when DEBUG is disabled.
        if row.shared_beyond > 0 {
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
                row.shared_beyond,
                row.raw_prefill_blocks,
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
                row.raw_prefill_blocks,
                prefill_load_scale = weights.prefill_load_scale
            );
        }

        logit
    }
}

impl<'a> DefaultWorkerPicker<'a> {
    fn from_selector(selector: &'a DefaultWorkerSelector) -> Self {
        Self {
            default_temperature: selector.kv_router_config.router_temperature,
            #[cfg(any(test, feature = "bench"))]
            deterministic_rng: selector.deterministic_rng.as_ref(),
            selector_lifetime: PhantomData,
        }
    }
}

impl<'a, C: WorkerConfigLike> WorkerSelectionInput<'a, C> {
    fn new(
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
        weights: LogitWeights,
    ) -> Self {
        let min_active_prefill_tokens =
            if request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let mut minimum = usize::MAX;
                eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                    minimum = minimum.min(request.worker_load_for(worker).active_prefill_tokens);
                });
                debug_assert_ne!(minimum, usize::MAX);
                minimum
            } else {
                0
            };
        Self {
            workers,
            request,
            eligibility,
            context: WorkerSelectionContext {
                request_id: request.mode.request_id().unwrap_or("-"),
                request_blocks: request.request_blocks(block_size),
                block_size,
                track_prefill_tokens: request.track_prefill_tokens,
                weights,
                min_active_prefill_tokens,
            },
        }
    }

    fn row(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
    ) -> WorkerSelectionRow {
        let effective_overlap_blocks = self.request.effective_overlap_blocks_for(worker);
        let has_tier_overlap_blocks = !self.request.overlap.tier_overlap_blocks.device.is_empty()
            || !self
                .request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .is_empty()
            || !self.request.overlap.tier_overlap_blocks.disk.is_empty();
        let device_overlap_blocks = self
            .request
            .overlap
            .tier_overlap_blocks
            .device
            .get(&worker)
            .copied()
            .map(|blocks| blocks as f64)
            .unwrap_or_else(|| {
                if has_tier_overlap_blocks {
                    0.0
                } else {
                    effective_overlap_blocks
                }
            });
        let worker_load = self.request.worker_loads.get(&worker).copied();
        let cached_tokens = self.request.effective_cached_tokens_for(worker);
        let raw_prefill_tokens = if self.request.track_prefill_tokens {
            match worker_load {
                Some(load) => {
                    // Preserve the legacy operation order when overlap exceeds the prompt.
                    let uncached_tokens = super::prefill_load::effective_prefill_tokens(
                        self.request.isl_tokens,
                        cached_tokens,
                    );
                    let projected_tokens = load.active_prefill_tokens + uncached_tokens;
                    projected_tokens.saturating_add(cached_tokens)
                }
                None => self.request.isl_tokens,
            }
        } else {
            0
        } as f64;
        let worker_load = worker_load.unwrap_or_default();
        let shared_beyond = self.request.shared_cache_hits.as_ref().map_or(0, |hits| {
            // `hits_beyond` expects the unweighted device prefix depth.
            hits.hits_beyond(device_overlap_blocks.round().max(0.0) as u32)
        });

        WorkerSelectionRow {
            worker,
            effective_overlap_blocks,
            device_overlap_blocks,
            host_overlap_blocks: self
                .request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .get(&worker)
                .copied()
                .unwrap_or(0) as f64,
            disk_overlap_blocks: self
                .request
                .overlap
                .tier_overlap_blocks
                .disk
                .get(&worker)
                .copied()
                .unwrap_or(0) as f64,
            shared_beyond,
            raw_prefill_blocks: raw_prefill_tokens / self.context.block_size as f64,
            active_prefill_tokens: worker_load.active_prefill_tokens,
            decode_cost_blocks: worker_load.potential_decode_blocks() as f64,
            active_requests: worker_load.active_requests,
            preferred_taint_multiplier,
        }
    }
}

impl<C: Borrow<KvRouterConfig>> WorkerScorer for DefaultWorkerScorer<C> {
    fn score(&self, context: &WorkerSelectionContext<'_>, row: &WorkerSelectionRow) -> f64 {
        let base_score = self.worker_logit(context, row, "Formula");
        match row.preferred_taint_multiplier {
            // NOTE: This multiplicative bias assumes a non-negative score. Negative
            // overlap scores expose its pre-existing sign sensitivity; keep it for now.
            Some(multiplier) => base_score * multiplier,
            None => base_score,
        }
    }
}

impl<C, S> WorkerPicker<C, S> for DefaultWorkerPicker<'_>
where
    C: WorkerConfigLike,
    S: WorkerScorer,
{
    fn pick(&self, scorer: &S, input: &WorkerSelectionInput<'_, C>) -> (WorkerWithDpRank, f64) {
        let workers = input.workers;
        let request = input.request;
        let eligibility = input.eligibility;
        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.default_temperature);
        let get_score = |worker, config: &C| {
            let preferred_taint_multiplier = input
                .request
                .routing_constraints
                .preferred_taint_multiplier(config.taints());
            scorer.score(
                &input.context,
                &input.row(worker, preferred_taint_multiplier),
            )
        };

        #[cfg(any(test, feature = "bench"))]
        let deterministic_choice = self.deterministic_rng.as_ref().map(|rng| {
            let mut candidates = Vec::new();
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                candidates.push(worker);
            });
            candidates.sort_unstable_by_key(|worker| (worker.worker_id, worker.dp_rank));

            let mut rng = rng.lock();
            let get_candidate_score = |worker| get_score(worker, &workers[&worker.worker_id]);
            if temperature == 0.0 {
                let mut best_worker = None;
                let mut best_logit = f64::INFINITY;
                let mut tie_count = 0usize;
                for worker in candidates {
                    let score = get_candidate_score(worker);
                    if score < best_logit {
                        best_worker = Some(worker);
                        best_logit = score;
                        tie_count = 1;
                        continue;
                    }

                    if score == best_logit {
                        tie_count += 1;
                        if rng.usize(0..tie_count) == 0 {
                            best_worker = Some(worker);
                        }
                    }
                }
                return (
                    best_worker.expect("eligible worker rank non-empty"),
                    best_logit,
                );
            }

            let entries = candidates
                .into_iter()
                .map(|worker| (worker, get_candidate_score(worker)))
                .collect();
            softmax_sample_entries(entries, temperature, rng.f64())
        });

        let random_choice = || {
            if temperature == 0.0 {
                let mut best_worker = None;
                let mut best_logit = f64::INFINITY;
                let mut tie_count = 0usize;
                eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
                    let score = get_score(worker, config);
                    if score < best_logit {
                        best_worker = Some(worker);
                        best_logit = score;
                        tie_count = 1;
                        return;
                    }

                    if score == best_logit {
                        tie_count += 1;
                        // Reservoir sampling keeps tied minima uniform without collecting workers.
                        if fastrand::usize(0..tie_count) == 0 {
                            best_worker = Some(worker);
                        }
                    }
                });

                return (
                    best_worker.expect("eligible worker rank non-empty"),
                    best_logit,
                );
            }

            let mut worker_logits = FxHashMap::default();
            eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
                worker_logits.insert(worker, get_score(worker, config));
            });
            softmax_sample(&worker_logits, temperature)
        };

        #[cfg(any(test, feature = "bench"))]
        return deterministic_choice.unwrap_or_else(random_choice);
        #[cfg(not(any(test, feature = "bench")))]
        random_choice()
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);
        eligibility.validate_pinned_worker_allowed()?;

        let pinned_worker = eligibility.pinned_worker();

        if pinned_worker.is_none()
            && !eligibility.has_eligible_worker(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            )
        {
            if eligibility.has_eligible_worker_ignoring_overload(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            ) {
                return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
            }

            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.request_blocks(block_size);
        // Borrowed, never allocated; bridges the winner row to `[ROUTING] Best`.
        let request_id = request.mode.request_id().unwrap_or("-");

        let scorer = DefaultWorkerScorer::from_selector(self);
        let weights = scorer.selection_weights(request);

        if let Some(worker) = pinned_worker {
            match eligibility.validate_worker_rank(workers, worker) {
                Ok(_) => {}
                Err(WorkerEligibilityError::WorkerOverloaded { .. }) => {
                    return Err(KvSchedulerError::PinnedWorkerOverloaded {
                        worker_id: worker.worker_id,
                    });
                }
                Err(_) => return Err(KvSchedulerError::NoEndpoints),
            }

            let input =
                WorkerSelectionInput::new(workers, request, eligibility, block_size, weights);
            let row = input.row(worker, None);
            let logit = scorer.worker_logit(&input.context, &row, "Pinned formula");
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
            let cached_tokens = request.effective_cached_tokens_for(worker);

            tracing::info!(
                request_id,
                "Selected pinned worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
                self.worker_type,
                worker.worker_id,
                worker.dp_rank,
                logit,
                effective_overlap_blocks,
            );

            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
                potential_decode_blocks: request
                    .potential_decode_blocks_after_admission(worker, block_size),
            });
        }

        let input = WorkerSelectionInput::new(workers, request, eligibility, block_size, weights);
        let picker = DefaultWorkerPicker::from_selector(self);
        let (best_worker, best_logit) = picker.pick(&scorer, &input);

        let best_host_pinned_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .get(&best_worker)
            .copied()
            .unwrap_or(0);
        let best_disk_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .disk
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        if self.worker_type == "decode" {
            tracing::info!(
                router_mode = "kv",
                request_id,
                worker_id = best_worker.worker_id,
                worker_type = %self.worker_type,
                dp_rank = ?best_worker.dp_rank,
                logit = best_logit,
                host_pinned_blocks = best_host_pinned_overlap_blocks,
                disk_blocks = best_disk_overlap_blocks,
                "Selected worker"
            );
            let effective_overlap_blocks = request.effective_overlap_blocks_for(best_worker);
            let cached_tokens = request.effective_cached_tokens_for(best_worker);

            return Ok(WorkerSelectionResult {
                worker: best_worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
                potential_decode_blocks: request
                    .potential_decode_blocks_after_admission(best_worker, block_size),
            });
        }

        let best_overlap = request.effective_overlap_blocks_for(best_worker);
        let best_cached_tokens = request.effective_cached_tokens_for(best_worker);

        let total_kv_blocks = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks());

        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = best_worker.worker_id,
            worker_type = %self.worker_type,
            dp_rank = ?best_worker.dp_rank,
            logit = best_logit,
            effective_cached_blocks = best_overlap,
            host_pinned_blocks = best_host_pinned_overlap_blocks,
            disk_blocks = best_disk_overlap_blocks,
            total_kv_blocks = ?total_kv_blocks,
            "Selected worker"
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks,
            effective_overlap_blocks: best_overlap,
            cached_tokens: best_cached_tokens,
            potential_decode_blocks: request
                .potential_decode_blocks_after_admission(best_worker, block_size),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::config::RouterConfigOverride;
    use crate::protocols::{SharedCacheHits, WorkerConfigLike};
    use crate::scheduling::{OverlapSignals, ScheduleMode};

    #[derive(Clone, Default)]
    struct TaintedWorkerConfig {
        taints: HashSet<String>,
    }

    impl WorkerConfigLike for TaintedWorkerConfig {
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
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    fn worker_logit(
        selector: &DefaultWorkerSelector,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        weights: LogitWeights,
    ) -> f64 {
        let workers = HashMap::from([(worker.worker_id, TaintedWorkerConfig::default())]);
        let input = WorkerSelectionInput::new(
            &workers,
            request,
            request.eligibility(),
            block_size,
            weights,
        );
        DefaultWorkerScorer::from_selector(selector).worker_logit(
            &input.context,
            &input.row(worker, None),
            "test",
        )
    }

    fn worker_loads_with_active_decode(
        decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    ) -> FxHashMap<WorkerWithDpRank, crate::sequences::WorkerLoadProjection> {
        decode_blocks
            .into_iter()
            .map(|(worker, active_decode_blocks)| {
                (
                    worker,
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks,
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    #[test]
    fn test_softmax_sample_single_key() {
        let mut logits = FxHashMap::default();
        let worker = WorkerWithDpRank::from_worker_id(42);
        for (logit, temperature) in [
            (0.5, 0.1),
            (0.5, 1.0),
            (0.5, 10.0),
            (-100.0, 1.0),
            (100.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ] {
            logits.clear();
            logits.insert(worker, logit);

            let result = softmax_sample(&logits, temperature);
            assert_eq!(result.0, worker, "Should return the only available worker");
            assert_eq!(result.1, logit, "Should return the selected worker's logit");
        }
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        let mut logits = FxHashMap::default();
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);
        let worker4 = WorkerWithDpRank::from_worker_id(4);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker3, 7.0);
        logits.insert(worker4, 3.5);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.0, worker2,
            "Should return worker with smallest logit when temperature is 0"
        );
        assert_eq!(
            result.1, 3.0,
            "Should return the smallest logit when temperature is 0"
        );

        logits.clear();
        let worker5 = WorkerWithDpRank::from_worker_id(5);
        let worker6 = WorkerWithDpRank::from_worker_id(6);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker5, 3.0);
        logits.insert(worker6, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert!(
            result.0 == worker2 || result.0 == worker5,
            "Should return one of the workers tied for the smallest logit"
        );
        assert_eq!(result.1, 3.0, "Should return the tied minimum logit");

        logits.clear();
        let worker10 = WorkerWithDpRank::from_worker_id(10);
        let worker20 = WorkerWithDpRank::from_worker_id(20);
        let worker30 = WorkerWithDpRank::from_worker_id(30);
        logits.insert(worker10, -1.0);
        logits.insert(worker20, -5.0);
        logits.insert(worker30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.0, worker20,
            "Should handle negative logits correctly"
        );
        assert_eq!(result.1, -5.0, "Should return the minimum negative logit");
    }

    #[test]
    fn test_softmax_sample_with_sample_returns_selected_logit() {
        let worker1 = WorkerWithDpRank::from_worker_id(1);
        let worker2 = WorkerWithDpRank::from_worker_id(2);
        let worker3 = WorkerWithDpRank::from_worker_id(3);

        let logits = FxHashMap::from_iter([(worker1, 0.0), (worker2, 3.0), (worker3, 9.0)]);
        let entries: Vec<_> = logits
            .iter()
            .map(|(worker, logit)| (*worker, *logit))
            .collect();
        let values: Vec<_> = entries.iter().map(|(_, logit)| *logit).collect();

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let temperature = 1.0;
        let range = max_val - min_val;
        let scaled: Vec<f64> = values.iter().map(|&v| -(v / range) / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probabilities: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum: f64 = probabilities.iter().sum();
        probabilities.iter_mut().for_each(|p| *p /= sum);

        let target_idx = entries
            .iter()
            .position(|(_, logit)| *logit > min_val)
            .expect("expected at least one non-minimum logit");
        let cumsum_before: f64 = probabilities.iter().take(target_idx).sum();
        let sample = cumsum_before + probabilities[target_idx] / 2.0;

        let result = softmax_sample_with_sample(&logits, temperature, sample);
        assert_eq!(result, entries[target_idx]);
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
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
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
                .select_worker(&workers, &request, request.eligibility(), 16)
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
            let first = DefaultWorkerSelector::new_seeded(Some(config.clone()), "test", 42);
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
                        .select_worker(&first_workers, &request, request.eligibility(), 16)
                        .unwrap()
                        .worker
                })
                .collect::<Vec<_>>();
            let second_sequence = (0..64)
                .map(|_| {
                    second
                        .select_worker(&second_workers, &request, request.eligibility(), 16)
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
                .select_worker(&workers, request, request.eligibility(), 1)
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
            .select_worker(
                &workers,
                &request,
                request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
                16,
            )
            .unwrap();

        assert_eq!(result.worker.worker_id, 1);
    }

    #[test]
    fn test_all_eligible_workers_overloaded_returns_overload_error() {
        use crate::test_utils::SimpleWorkerConfig;

        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let request = base_request(16);
        let overloaded_worker_ids = HashSet::from([0, 1]);

        let result = selector.select_worker(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        );

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

        let result = selector.select_worker(
            &workers,
            &request,
            request.eligibility_with_overloaded(Some(&overloaded_worker_ids)),
            16,
        );

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
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
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

        let result = selector.select_worker(&workers, &request, request.eligibility(), 16);
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
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
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
            .select_worker(&workers, &request, request.eligibility(), 16)
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
                worker_loads: worker_loads_with_active_decode(decode_blocks),
                track_prefill_tokens: true,
                router_config_override: None,
                lora_name: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_id: None,
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
                .select_worker(&workers, &request, request.eligibility(), 16)
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
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
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
            .select_worker(&workers, &request, request.eligibility(), 16)
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
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
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
            .select_worker(&workers, &request, request.eligibility(), 16)
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
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: Some(shared_hits),
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
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
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
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
                .select_worker(&workers, &request, request.eligibility(), block_size)
                .unwrap()
                .worker,
            cold_worker
        );
        assert_eq!(
            amplified_credit
                .select_worker(&workers, &request, request.eligibility(), block_size)
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
                .select_worker(&workers, &request, request.eligibility(), block_size)
                .unwrap()
                .worker,
            warm_worker
        );
        assert_eq!(
            with_decay
                .select_worker(&workers, &request, request.eligibility(), block_size)
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
            worker_loads: worker_loads_with_active_decode(decode_blocks),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "effective overlap should still credit older callers without tier maps"
        );
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
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, request.eligibility(), block_size)
            .unwrap();

        assert_eq!(result.worker, worker0);
    }
}
