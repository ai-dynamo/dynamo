// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use rand::Rng;
use rustc_hash::FxHashMap;

use super::config::KvRouterConfig;
use super::types::{KvSchedulerError, SchedulingRequest, pinned_worker_config};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// A trait that users can implement to define custom selection logic.
///
/// Generic over `C` so that the scheduling layer does not depend on a concrete config type.
pub trait WorkerSelector<C: WorkerConfigLike> {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
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
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: at zero temperature, return a minimum-logit worker directly.
    if temperature == 0.0 {
        let mut logit_iter = logits.iter();
        let (first_key, first_logit) = logit_iter.next().unwrap();

        let mut min_logit = first_logit;
        let mut min_key = first_key;
        for (key, logit) in logit_iter {
            if logit < min_logit {
                min_logit = logit;
                min_key = key;
            }
        }

        return (*min_key, *min_logit);
    }

    let entries: Vec<_> = logits
        .iter()
        .map(|(worker, logit)| (*worker, *logit))
        .collect();
    let values: Vec<_> = entries.iter().map(|(_, logit)| *logit).collect();

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        vec![1.0 / entries.len() as f64; entries.len()]
    } else {
        // Fused normalize -> negate -> scale -> exp, then normalize probabilities
        let range = max_val - min_val;
        let scaled: Vec<f64> = values.iter().map(|&v| -(v / range) / temperature).collect();
        let max_scaled = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
        let sum: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        probs
    };

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return entries[i];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    entries[entries.len() - 1]
}

/// Default implementation matching the Python _cost_function.
#[derive(Debug, Clone)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct WorkerScore {
    overlap_blocks: u32,
    logit: f64,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
        }
    }

    fn worker_score(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        overlap_weight: f64,
        shared_cache_multiplier: f64,
        formula_name: &'static str,
    ) -> WorkerScore {
        let isl = request.isl_tokens;
        let overlap_blocks = request.overlaps.scores.get(&worker).copied().unwrap_or(0);
        let default_prefill_token = if request.track_prefill_tokens { isl } else { 0 };
        let prefill_token = request
            .prefill_tokens
            .get(&worker)
            .copied()
            .unwrap_or(default_prefill_token);

        // Adjust prefill tokens by shared cache hits beyond this worker's device prefix.
        let (adjusted_prefill_token, shared_beyond) =
            if let Some(ref shared_hits) = request.shared_cache_hits {
                let beyond = shared_hits.hits_beyond(overlap_blocks);
                let reduction = shared_cache_multiplier * (beyond as f64) * (block_size as f64);
                let adjusted = (prefill_token as f64 - reduction).max(0.0) as usize;
                (adjusted, beyond)
            } else {
                (prefill_token, 0)
            };

        let potential_prefill_block = (adjusted_prefill_token as f64) / (block_size as f64);
        let decode_block_fallback = (prefill_token as f64) / (block_size as f64);
        let decode_block = request
            .decode_blocks
            .get(&worker)
            .copied()
            .unwrap_or(decode_block_fallback.floor() as usize) as f64;
        let logit = overlap_weight * potential_prefill_block + decode_block;

        if shared_beyond > 0 {
            tracing::debug!(
                "{formula_name} for worker_id={} dp_rank={:?} with {overlap_blocks} device blocks, \
                 {shared_beyond} shared blocks beyond device (multiplier={shared_cache_multiplier:.2}): {logit:.3} \
                 = {overlap_weight:.1} * adjusted_prefill_blocks + decode_blocks \
                 = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3} \
                 (prefill_tokens: {prefill_token} -> {adjusted_prefill_token})",
                worker.worker_id,
                worker.dp_rank
            );
        } else {
            tracing::debug!(
                "{formula_name} for worker_id={} dp_rank={:?} with {overlap_blocks} cached blocks: {logit:.3} \
                 = {overlap_weight:.1} * prefill_blocks + decode_blocks \
                 = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}",
                worker.worker_id,
                worker.dp_rank
            );
        }

        WorkerScore {
            overlap_blocks,
            logit,
        }
    }
}

impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        assert!(request.isl_tokens > 0);
        request.validate_worker_constraints()?;

        let allowed_ids = request.allowed_worker_ids.as_ref();
        let pinned_worker = request.pinned_worker;

        if pinned_worker.is_none()
            && allowed_ids.map_or(workers.is_empty(), |ids| {
                !workers.keys().any(|wid| ids.contains(wid))
            })
        {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let isl = request.isl_tokens;
        let request_blocks = isl.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);

        let shared_cache_multiplier = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.shared_cache_multiplier)
            .unwrap_or(self.kv_router_config.shared_cache_multiplier);

        if let Some(worker) = pinned_worker {
            pinned_worker_config(workers, worker)?;

            let score = self.worker_score(
                request,
                worker,
                block_size,
                overlap_weight,
                shared_cache_multiplier,
                "Pinned formula",
            );

            return Ok(WorkerSelectionResult {
                worker,
                required_blocks: request_blocks as u64,
                overlap_blocks: score.overlap_blocks,
            });
        }

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);

        let get_score = |worker: WorkerWithDpRank| -> f64 {
            self.worker_score(
                request,
                worker,
                block_size,
                overlap_weight,
                shared_cache_multiplier,
                "Formula",
            )
            .logit
        };

        let worker_iter = workers
            .iter()
            .filter(move |(wid, _)| allowed_ids.is_none_or(|ids| ids.contains(wid)))
            .flat_map(|(worker_id, config)| {
                let data_parallel_size = config.data_parallel_size();
                let data_parallel_start_rank = config.data_parallel_start_rank();
                (data_parallel_start_rank..(data_parallel_start_rank + data_parallel_size))
                    .map(move |dp_rank| WorkerWithDpRank::new(*worker_id, dp_rank))
            });

        let (best_worker, best_logit) = if temperature == 0.0 {
            let mut min_workers = Vec::new();
            let mut min_score = f64::INFINITY;
            for worker in worker_iter {
                let score = get_score(worker);
                if score < min_score {
                    min_workers.clear();
                    min_workers.push(worker);
                    min_score = score;
                } else if score == min_score {
                    min_workers.push(worker);
                }
            }

            if min_workers.len() > 1 {
                tracing::debug!(
                    "Multiple workers tied with same logit, using tree size as tie-breaker"
                );
                let tree_sizes: Vec<(usize, &WorkerWithDpRank)> = min_workers
                    .iter()
                    .map(|w| (request.overlaps.tree_sizes.get(w).copied().unwrap_or(0), w))
                    .collect();

                if tree_sizes.iter().all(|(s, _)| *s == tree_sizes[0].0) {
                    let idx = rand::rng().random_range(0..min_workers.len());
                    (min_workers[idx], min_score)
                } else {
                    let (_, worker) = *tree_sizes.iter().min_by_key(|(s, _)| *s).unwrap();
                    (*worker, min_score)
                }
            } else {
                (min_workers[0], min_score)
            }
        } else {
            let mut worker_logits = FxHashMap::default();
            for worker in worker_iter {
                let score = get_score(worker);
                worker_logits.insert(worker, score);
            }

            softmax_sample(&worker_logits, temperature)
        };

        if self.worker_type == "decode" {
            tracing::info!(
                "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}",
                self.worker_type,
                best_worker.worker_id,
                best_worker.dp_rank,
                best_logit,
            );
            return Ok(WorkerSelectionResult {
                worker: best_worker,
                required_blocks: request_blocks as u64,
                overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
            });
        }

        let best_overlap = *overlaps.get(&best_worker).unwrap_or(&0);

        let total_blocks_info = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks())
            .map(|blocks| format!(", total blocks: {}", blocks))
            .unwrap_or_default();

        let tree_size = request
            .overlaps
            .tree_sizes
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        tracing::info!(
            "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, cached blocks: {}, tree size: {}{}",
            self.worker_type,
            best_worker.worker_id,
            best_worker.dp_rank,
            best_logit,
            best_overlap,
            tree_size,
            total_blocks_info
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: overlaps.get(&best_worker).copied().unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::SharedCacheHits;

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

    /// Test the scoring formula with shared cache hits.
    ///
    /// Request [A, B, C, D], shared_cache_multiplier=0.5, block_size=1
    /// - Worker 0: device=[A,B] (overlap=2), shared has [A,B,C,D] -> shared_beyond=2
    ///   adjusted_prefill = isl - 0.5*2*1 = 4-1 = 3, logit = 1.0 * 3 + 0 = 3.0
    /// - Worker 1: device=[] (overlap=0), shared has [A,B,C,D] -> shared_beyond=4
    ///   adjusted_prefill = isl - 0.5*4*1 = 4-2 = 2, logit = 1.0 * 2 + 0 = 2.0
    ///
    /// Worker 1 has lower logit (less work), so it wins.
    #[test]
    fn test_shared_cache_hits_scoring() {
        use crate::protocols::OverlapScores;
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 1u32;
        let isl = 4usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);
        let worker1 = WorkerWithDpRank::from_worker_id(1);

        let mut overlaps = OverlapScores::new();
        overlaps.scores.insert(worker0, 2);
        // worker1 has 0 overlap (not in map)

        #[allow(clippy::single_range_in_vec_init)]
        let shared_hits = SharedCacheHits::from_ranges(vec![0..4]);

        let config = KvRouterConfig {
            overlap_score_weight: 1.0,
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            overlaps,
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            shared_cache_hits: Some(shared_hits),
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, block_size)
            .unwrap();

        // Worker 1 should win: logit 2.0 < 3.0
        assert_eq!(
            result.worker, worker1,
            "Worker 1 should be selected (lower logit due to shared cache)"
        );
    }

    /// Without shared cache hits, the scoring should be unchanged.
    #[test]
    fn test_no_shared_cache_unchanged() {
        use crate::protocols::OverlapScores;
        use crate::test_utils::SimpleWorkerConfig;

        let block_size = 16u32;
        let isl = 64usize;
        let worker0 = WorkerWithDpRank::from_worker_id(0);

        let mut overlaps = OverlapScores::new();
        overlaps.scores.insert(worker0, 2);

        let config = KvRouterConfig::default();
        let selector = DefaultWorkerSelector::new(Some(config), "test");
        let mut workers = HashMap::new();
        workers.insert(0, SimpleWorkerConfig::default());

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            overlaps,
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, block_size)
            .unwrap();

        assert_eq!(result.worker, worker0);
    }
}
