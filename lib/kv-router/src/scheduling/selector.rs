// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt;

use rand::Rng;
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
    let mut rng = rand::rng();
    softmax_sample_with_sample(logits, temperature, rng.random())
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
}

#[derive(Debug, Clone, Copy)]
struct LogitWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

/// A candidate worker's score plus the cost breakdown and raw inputs that
/// produced it, so a single log line can explain *why* a worker won without
/// needing a separate debug line per candidate.
#[derive(Debug, Clone, Copy)]
struct CandidateScore {
    logit: f64,
    raw_prefill_blocks: f64,
    overlap_credit_blocks: f64,
    decode_cost_blocks: f64,
    shared_blocks_beyond: u32,
    // Raw inputs (not derived from the other fields above) that fed the
    // formula for this specific candidate -- what varies worker-to-worker
    // for the same request, as opposed to `raw_prefill_blocks` etc., which
    // are already this worker's formula output at each stage.
    device_overlap_blocks: f64,
    active_prefill_tokens: usize,
    active_decode_blocks: usize,
}

/// `worker_id`s in one deployment share a long common prefix (they're
/// generated close together) and differ only in their last few digits --
/// printing the full u64 for every candidate on one line is mostly repeated
/// noise. This keeps the low decimal digits only; full precision is still
/// available via the `chosen_worker_id` field, which is logged in full.
const WORKER_ID_SUFFIX_MOD: u64 = 1_000_000;

fn short_worker_id(id: WorkerId) -> u64 {
    id % WORKER_ID_SUFFIX_MOD
}

/// Formats `worker_id:dp_rank:logit(overlap=.,prefill_tok=.,decode_blk=.),...`
/// directly into the log record's buffer via `Display`, so this only runs
/// (and only allocates whatever the subscriber's own formatting buffer
/// needs) when a subscriber actually renders the event -- unlike building a
/// `String` at the call site with `format!`/`collect`/`join`, which would
/// allocate on every routing decision regardless of whether logging is
/// enabled. This routing-decision log runs unconditionally (not
/// DEBUG-gated), so this difference matters.
///
/// This whole thing is the *value* of a single `candidates` tracing field
/// (see `select_worker`), rendered via the global fmt formatter in
/// lib/runtime/src/logging.rs like every other tracing field in the system
/// -- so its internal `,` separators don't collide with anything outside
/// this value.
///
/// The `(overlap=.,prefill_tok=.,decode_blk=.)` part is this formula's raw
/// per-candidate arguments (see `FORMULA_DESCRIPTION`, logged once at
/// selector construction) -- together the two logs make a decision fully
/// reconstructable: the formula log says what the computation IS, this says
/// what values every candidate (not just the winner) plugged into it. A
/// future `WorkerSelector` with a different cost function would use a
/// different `FORMULA_DESCRIPTION` and argument set; only `worker_id`,
/// `dp_rank`, and `logit` are guaranteed present regardless of formula.
struct CandidatesSummary<'a>(&'a [(WorkerWithDpRank, CandidateScore)]);

impl fmt::Display for CandidatesSummary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (worker, score)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(
                f,
                "{}:{}:{:.3}(overlap={:.1},prefill_tok={},decode_blk={})",
                short_worker_id(worker.worker_id),
                worker.dp_rank,
                score.logit,
                score.device_overlap_blocks,
                score.active_prefill_tokens,
                score.active_decode_blocks,
            )?;
        }
        Ok(())
    }
}

/// The cost function's structure, as a fixed string -- logged once per
/// selector (see `DefaultWorkerSelector::new`) rather than repeated on every
/// decision. Field names here match the `CandidateScore`/`CandidatesSummary`
/// argument names used in every "Routing decision" log, so the two are meant
/// to be read together: this says what the formula IS, those say what
/// arguments were plugged into it for a specific request/candidate.
const FORMULA_DESCRIPTION: &str = "logit = prefill_load_scale * max(raw_prefill_blocks - overlap_credit_blocks, 0) + decode_cost_blocks; \
     overlap_credit_blocks = overlap_score_credit * overlap_credit_decay * device_overlap_blocks \
     + host_cache_hit_weight * host_overlap_blocks + disk_cache_hit_weight * disk_overlap_blocks \
     + shared_cache_multiplier * shared_blocks_beyond";

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        let kv_router_config = kv_router_config.unwrap_or_default();

        // One-time log of the formula's structure and weights (constant for
        // this selector's lifetime, absent a per-request
        // `router_config_override` -- an overridden request's own "Routing
        // decision" line will carry whatever weight it actually used, so it
        // remains self-describing even when it diverges from this default).
        tracing::info!(
            worker_type = %worker_type,
            prefill_load_scale = kv_router_config.prefill_load_scale,
            overlap_score_credit = kv_router_config.overlap_score_credit,
            overlap_score_credit_decay = kv_router_config.overlap_score_credit_decay,
            shared_cache_multiplier = kv_router_config.shared_cache_multiplier,
            host_cache_hit_weight = kv_router_config.host_cache_hit_weight,
            disk_cache_hit_weight = kv_router_config.disk_cache_hit_weight,
            formula = FORMULA_DESCRIPTION,
            "Routing formula"
        );

        Self {
            kv_router_config,
            worker_type,
        }
    }

    fn worker_logit(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        min_active_prefill_tokens: usize,
        weights: LogitWeights,
    ) -> CandidateScore {
        let block_size_f64 = block_size as f64;
        let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
        let has_tier_overlap_blocks = !request.overlap.tier_overlap_blocks.device.is_empty()
            || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
            || !request.overlap.tier_overlap_blocks.disk.is_empty();
        let device_overlap_blocks = request
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
        // `shared_cache_hits::hits_beyond` expects an integer block count, so
        // use the unweighted device prefix depth for this comparison.
        let device_overlap_blocks_u32 = device_overlap_blocks.round().max(0.0) as u32;
        let worker_load = request.worker_loads.get(&worker).copied();
        let raw_prefill_tokens = if request.track_prefill_tokens {
            match worker_load {
                Some(load) => {
                    let cached_tokens = request.effective_cached_tokens_for(worker);
                    // Preserve the legacy operation order when overlap exceeds the prompt.
                    let uncached_tokens = super::prefill_load::effective_prefill_tokens(
                        request.isl_tokens,
                        cached_tokens,
                    );
                    let projected_tokens = load.active_prefill_tokens + uncached_tokens;
                    projected_tokens.saturating_add(cached_tokens)
                }
                None => request.isl_tokens,
            }
        } else {
            0
        } as f64;

        let host_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .get(&worker)
            .copied()
            .unwrap_or(0) as f64;
        let disk_overlap_blocks = request
            .overlap
            .tier_overlap_blocks
            .disk
            .get(&worker)
            .copied()
            .unwrap_or(0) as f64;

        // Credit shared cache hits beyond this worker's device prefix.
        let (shared_overlap_blocks, shared_beyond) =
            if let Some(ref shared_hits) = request.shared_cache_hits {
                let beyond = shared_hits.hits_beyond(device_overlap_blocks_u32);
                (weights.shared_cache_multiplier * (beyond as f64), beyond)
            } else {
                (0.0, 0)
            };

        let raw_prefill_blocks = raw_prefill_tokens / block_size_f64;
        // Normalize backlog above the least-loaded eligible worker by this request's
        // size. The rational decay softly trades cache locality for prefill balance,
        // while leaving workers at the load floor with their full device credit.
        let overlap_credit_decay =
            if request.track_prefill_tokens && weights.overlap_score_credit_decay > 0.0 {
                let active_prefill_tokens = worker_load.unwrap_or_default().active_prefill_tokens;
                let excess_active_prefill_blocks =
                    active_prefill_tokens.saturating_sub(min_active_prefill_tokens) as f64
                        / block_size_f64;
                let normalized_prefill_load =
                    excess_active_prefill_blocks / request.request_blocks(block_size) as f64;
                1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load)
            } else {
                1.0
            };
        let effective_overlap_score_credit = weights.overlap_score_credit * overlap_credit_decay;
        let overlap_credit_blocks = effective_overlap_score_credit * device_overlap_blocks
            + self.kv_router_config.host_cache_hit_weight * host_overlap_blocks
            + self.kv_router_config.disk_cache_hit_weight * disk_overlap_blocks
            + shared_overlap_blocks;
        let adjusted_prefill_blocks = (raw_prefill_blocks - overlap_credit_blocks).max(0.0);
        let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
        let worker_load = worker_load.unwrap_or_default();
        let decode_cost_blocks = worker_load.potential_decode_blocks() as f64;
        let logit = prefill_cost_blocks + decode_cost_blocks;

        // Per-candidate detail is no longer logged here -- select_worker()
        // logs ONE line per request with the winning candidate's breakdown
        // plus a compact summary of every candidate, keyed by request_id so
        // it's joinable. See that log call for the replacement.
        CandidateScore {
            logit,
            raw_prefill_blocks,
            overlap_credit_blocks,
            decode_cost_blocks,
            shared_blocks_beyond: shared_beyond,
            device_overlap_blocks,
            active_prefill_tokens: worker_load.active_prefill_tokens,
            active_decode_blocks: worker_load.active_decode_blocks,
        }
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

        let weights = LogitWeights {
            overlap_score_credit: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.overlap_score_credit)
                .unwrap_or(self.kv_router_config.overlap_score_credit),
            overlap_score_credit_decay: self.kv_router_config.overlap_score_credit_decay,
            prefill_load_scale: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.prefill_load_scale)
                .unwrap_or(self.kv_router_config.prefill_load_scale),
            shared_cache_multiplier: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.shared_cache_multiplier)
                .unwrap_or(self.kv_router_config.shared_cache_multiplier),
        };

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

            let min_active_prefill_tokens = request.worker_load_for(worker).active_prefill_tokens;
            let score = self.worker_logit(
                request,
                worker,
                block_size,
                min_active_prefill_tokens,
                weights,
            );
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
            let cached_tokens = request.effective_cached_tokens_for(worker);

            tracing::info!(
                request_id = request.mode.request_id().unwrap_or("unknown"),
                router_mode = "kv",
                worker_type = %self.worker_type,
                isl_tokens = request.isl_tokens,
                pinned = true,
                num_candidates = 1,
                chosen_worker_id = worker.worker_id,
                chosen_dp_rank = worker.dp_rank,
                chosen_logit = score.logit,
                raw_prefill_blocks = score.raw_prefill_blocks,
                overlap_credit_blocks = score.overlap_credit_blocks,
                decode_cost_blocks = score.decode_cost_blocks,
                shared_blocks_beyond = score.shared_blocks_beyond,
                device_overlap_blocks = score.device_overlap_blocks,
                active_prefill_tokens = score.active_prefill_tokens,
                active_decode_blocks = score.active_decode_blocks,
                effective_cached_blocks = effective_overlap_blocks,
                "Routing decision"
            );

            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
            });
        }

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
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
        let get_score = |worker: WorkerWithDpRank| -> CandidateScore {
            let mut score = self.worker_logit(
                request,
                worker,
                block_size,
                min_active_prefill_tokens,
                weights,
            );
            if let Some(config) = workers.get(&worker.worker_id)
                && let Some(multiplier) = request
                    .routing_constraints
                    .preferred_taint_multiplier(config.taints())
            {
                score.logit *= multiplier;
            }
            score
        };

        // Collected up front (candidate sets here are small -- a handful of
        // eligible workers) so the winner can be picked with the exact same
        // tie-break semantics as before, while still letting the log line
        // below report every candidate's logit and how close the runner-up
        // was, not just the winner.
        let mut candidates: Vec<(WorkerWithDpRank, CandidateScore)> = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
            candidates.push((worker, get_score(worker)));
        });
        let num_candidates = candidates.len();

        let (best_worker, best_score, tie_count) = if temperature == 0.0 {
            let mut best: Option<(WorkerWithDpRank, CandidateScore)> = None;
            let mut tie_count = 0usize;
            let mut rng = rand::rng();
            for &(worker, score) in &candidates {
                match best {
                    Some((_, b)) if score.logit > b.logit => {}
                    Some((_, b)) if score.logit == b.logit => {
                        tie_count += 1;
                        // Reservoir sampling keeps tied minima uniform without collecting workers.
                        if rng.random_range(0..tie_count) == 0 {
                            best = Some((worker, score));
                        }
                    }
                    _ => {
                        best = Some((worker, score));
                        tie_count = 1;
                    }
                }
            }
            let (worker, score) = best.expect("eligible worker rank non-empty");
            (worker, score, tie_count)
        } else {
            // Probabilistic sampling, not a discrete tie-break -- tie_count
            // doesn't apply here, logged as 0 to mean "not applicable".
            let worker_logits: FxHashMap<WorkerWithDpRank, f64> =
                candidates.iter().map(|(w, s)| (*w, s.logit)).collect();
            let (worker, _logit) = softmax_sample(&worker_logits, temperature);
            let score = candidates
                .iter()
                .find(|(w, _)| *w == worker)
                .map(|(_, s)| *s)
                .expect("softmax-sampled worker must be among scored candidates");
            (worker, score, 0)
        };

        // How decisive the pick was: 0.0 means a tie (broken by reservoir
        // sampling above), +inf means there was only one eligible candidate.
        let margin = candidates
            .iter()
            .filter(|(w, _)| *w != best_worker)
            .map(|(_, s)| s.logit)
            .fold(f64::INFINITY, f64::min)
            - best_score.logit;

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
                request_id = request.mode.request_id().unwrap_or("unknown"),
                router_mode = "kv",
                worker_type = %self.worker_type,
                isl_tokens = request.isl_tokens,
                num_candidates,
                chosen_worker_id = best_worker.worker_id,
                chosen_dp_rank = best_worker.dp_rank,
                chosen_logit = best_score.logit,
                margin,
                tie_count,
                raw_prefill_blocks = best_score.raw_prefill_blocks,
                overlap_credit_blocks = best_score.overlap_credit_blocks,
                decode_cost_blocks = best_score.decode_cost_blocks,
                shared_blocks_beyond = best_score.shared_blocks_beyond,
                device_overlap_blocks = best_score.device_overlap_blocks,
                active_prefill_tokens = best_score.active_prefill_tokens,
                active_decode_blocks = best_score.active_decode_blocks,
                host_pinned_blocks = best_host_pinned_overlap_blocks,
                disk_blocks = best_disk_overlap_blocks,
                candidates = %CandidatesSummary(&candidates),
                "Routing decision"
            );
            let effective_overlap_blocks = request.effective_overlap_blocks_for(best_worker);
            let cached_tokens = request.effective_cached_tokens_for(best_worker);

            return Ok(WorkerSelectionResult {
                worker: best_worker,
                required_blocks: request_blocks,
                effective_overlap_blocks,
                cached_tokens,
            });
        }

        let best_overlap = request.effective_overlap_blocks_for(best_worker);
        let best_cached_tokens = request.effective_cached_tokens_for(best_worker);

        let total_kv_blocks = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks());

        tracing::info!(
            request_id = request.mode.request_id().unwrap_or("unknown"),
            router_mode = "kv",
            worker_type = %self.worker_type,
            isl_tokens = request.isl_tokens,
            num_candidates,
            chosen_worker_id = best_worker.worker_id,
            chosen_dp_rank = best_worker.dp_rank,
            chosen_logit = best_score.logit,
            margin,
            tie_count,
            raw_prefill_blocks = best_score.raw_prefill_blocks,
            overlap_credit_blocks = best_score.overlap_credit_blocks,
            decode_cost_blocks = best_score.decode_cost_blocks,
            shared_blocks_beyond = best_score.shared_blocks_beyond,
            device_overlap_blocks = best_score.device_overlap_blocks,
            active_prefill_tokens = best_score.active_prefill_tokens,
            active_decode_blocks = best_score.active_decode_blocks,
            effective_cached_blocks = best_overlap,
            host_pinned_blocks = best_host_pinned_overlap_blocks,
            disk_blocks = best_disk_overlap_blocks,
            total_kv_blocks = ?total_kv_blocks,
            candidates = %CandidatesSummary(&candidates),
            "Routing decision"
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks,
            effective_overlap_blocks: best_overlap,
            cached_tokens: best_cached_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
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
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
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
    fn test_worker_logit_preserves_prefill_accounting_edges() {
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut request = base_request(64);
        request.overlap.effective_cached_tokens.insert(worker, 96);
        request.overlap.tier_overlap_blocks.device.insert(worker, 6);
        request.worker_loads.insert(
            worker,
            crate::sequences::WorkerLoadProjection {
                active_prefill_tokens: 16,
                active_decode_blocks: 2,
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

        assert_eq!(
            selector
                .worker_logit(&request, worker, 16, 0, weights)
                .logit,
            7.0
        );

        request.track_prefill_tokens = false;
        assert_eq!(
            selector
                .worker_logit(&request, worker, 16, 0, weights)
                .logit,
            5.0
        );
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
