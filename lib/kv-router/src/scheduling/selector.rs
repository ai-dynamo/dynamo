// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use rand::Rng;

use super::config::KvRouterConfig;
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
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

/// Helper function for softmax sampling.
/// Returns a vec of workers: multiple if tied, single if sampled.
fn softmax_sample(
    logits: &HashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> Vec<WorkerWithDpRank> {
    if logits.is_empty() {
        panic!("Empty logits for softmax sampling");
    }

    // Guard: if temperature is 0, return all keys with the smallest logit value (ties)
    if temperature == 0.0 {
        let min_logit = logits.values().fold(f64::INFINITY, |a, &b| a.min(b));

        let min_keys: Vec<_> = logits
            .iter()
            .filter(|&(_, &v)| v == min_logit)
            .map(|(k, _)| *k)
            .collect();

        return min_keys;
    }

    let keys: Vec<_> = logits.keys().copied().collect();
    let values: Vec<_> = logits.values().copied().collect();

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let probabilities = if min_val == max_val {
        vec![1.0 / keys.len() as f64; keys.len()]
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

    let mut rng = rand::rng();
    let sample: f64 = rng.random();

    let mut cumsum = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cumsum += prob;
        if sample <= cumsum {
            return vec![keys[i]];
        }
    }

    // Fallback to last key (shouldn't normally reach here)
    vec![keys[keys.len() - 1]]
}

/// Default implementation matching the Python _cost_function.
#[derive(Debug, Clone)]
pub struct DefaultWorkerSelector {
    pub kv_router_config: KvRouterConfig,
    pub worker_type: &'static str,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
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

        let allowed_ids = request.allowed_worker_ids.as_ref();

        if allowed_ids.map_or(workers.is_empty(), |ids| {
            !workers.keys().any(|wid| ids.contains(wid))
        }) {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let isl = request.isl_tokens;
        let request_blocks = isl.div_ceil(block_size as usize);
        let overlaps = &request.overlaps.scores;

        let decode_blocks = &request.decode_blocks;
        let prefill_tokens = &request.prefill_tokens;

        let mut worker_logits = HashMap::new();

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

        for (worker_id, config) in workers
            .iter()
            .filter(|(wid, _)| allowed_ids.is_none_or(|ids| ids.contains(wid)))
        {
            let data_parallel_size = config.data_parallel_size();
            let data_parallel_start_rank = config.data_parallel_start_rank();

            for dp_rank in data_parallel_start_rank..(data_parallel_start_rank + data_parallel_size)
            {
                let worker = WorkerWithDpRank::new(*worker_id, dp_rank);

                let overlap = *overlaps.get(&worker).unwrap_or(&0);

                let prefill_token = *prefill_tokens.get(&worker).unwrap_or(&isl);

                // Adjust prefill tokens by shared cache hits beyond this worker's device prefix.
                let (adjusted_prefill_token, shared_beyond, shared_cache_reduction) =
                    if let Some(ref shared_hits) = request.shared_cache_hits {
                        let beyond = shared_hits.hits_beyond(overlap);
                        let reduction =
                            shared_cache_multiplier * (beyond as f64) * (block_size as f64);
                        let adjusted = (prefill_token as f64 - reduction).max(0.0) as usize;
                        (adjusted, beyond, reduction)
                    } else {
                        (prefill_token, 0, 0.0)
                    };

                let potential_prefill_block = (adjusted_prefill_token as f64) / (block_size as f64);

                let decode_block = *decode_blocks
                    .get(&worker)
                    .unwrap_or(&(potential_prefill_block.floor() as usize))
                    as f64;

                let logit = overlap_weight * potential_prefill_block + decode_block;

                worker_logits.insert(worker, logit);

                if shared_beyond > 0 {
                    tracing::debug!(
                        "Formula for worker_id={} dp_rank={:?} with {overlap} device blocks, \
                         {shared_beyond} shared blocks beyond device (reduction={shared_cache_reduction:.1}, \
                         multiplier={shared_cache_multiplier:.2}): {logit:.3} \
                         = {overlap_weight:.1} * adjusted_prefill_blocks + decode_blocks \
                         = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3} \
                         (prefill_tokens: {prefill_token} -> {adjusted_prefill_token})",
                        worker.worker_id,
                        worker.dp_rank
                    );
                } else {
                    tracing::debug!(
                        "Formula for worker_id={} dp_rank={:?} with {overlap} cached blocks: {logit:.3} \
                         = {overlap_weight:.1} * prefill_blocks + decode_blocks \
                         = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}",
                        worker.worker_id,
                        worker.dp_rank
                    );
                }
            }
        }

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
        let candidates = softmax_sample(&worker_logits, temperature);

        let best_worker = if candidates.len() > 1 {
            tracing::debug!(
                "Multiple workers tied with same logit, using tree size as tie-breaker"
            );
            let tree_sizes: Vec<(usize, &WorkerWithDpRank)> = candidates
                .iter()
                .map(|w| (request.overlaps.tree_sizes.get(w).copied().unwrap_or(0), w))
                .collect();

            if tree_sizes.iter().all(|(s, _)| *s == tree_sizes[0].0) {
                let idx = rand::rng().random_range(0..candidates.len());
                candidates[idx]
            } else {
                *tree_sizes.iter().min_by_key(|(s, _)| *s).unwrap().1
            }
        } else {
            candidates[0]
        };

        let best_logit = worker_logits[&best_worker];

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
        let mut logits = HashMap::new();
        let worker = WorkerWithDpRank::from_worker_id(42);
        logits.insert(worker, 0.5);

        for temperature in &[0.1, 1.0, 10.0] {
            let result = softmax_sample(&logits, *temperature);
            assert_eq!(result.len(), 1, "Should return exactly one worker");
            assert_eq!(result[0], worker, "Should return the only available worker");
        }

        logits.clear();
        logits.insert(worker, -100.0);
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);

        logits.clear();
        logits.insert(worker, 100.0);
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);

        logits.clear();
        logits.insert(worker, 0.0);
        let result = softmax_sample(&logits, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], worker);
    }

    #[test]
    fn test_softmax_sample_zero_temperature() {
        let mut logits = HashMap::new();
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
            result.len(),
            1,
            "Should return one worker when there's no tie"
        );
        assert_eq!(
            result[0], worker2,
            "Should return worker with smallest logit when temperature is 0"
        );

        logits.clear();
        let worker5 = WorkerWithDpRank::from_worker_id(5);
        let worker6 = WorkerWithDpRank::from_worker_id(6);
        logits.insert(worker1, 5.0);
        logits.insert(worker2, 3.0);
        logits.insert(worker5, 3.0);
        logits.insert(worker6, 7.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(
            result.len(),
            2,
            "Should return all workers with smallest logit when tied"
        );
        assert!(
            result.contains(&worker2) && result.contains(&worker5),
            "Should contain both tied workers"
        );

        logits.clear();
        let worker10 = WorkerWithDpRank::from_worker_id(10);
        let worker20 = WorkerWithDpRank::from_worker_id(20);
        let worker30 = WorkerWithDpRank::from_worker_id(30);
        logits.insert(worker10, -1.0);
        logits.insert(worker20, -5.0);
        logits.insert(worker30, 0.0);

        let result = softmax_sample(&logits, 0.0);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0], worker20,
            "Should handle negative logits correctly"
        );
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
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
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
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
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
