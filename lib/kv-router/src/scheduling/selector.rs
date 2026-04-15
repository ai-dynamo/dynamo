// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use rand::Rng;

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
    logits: &HashMap<WorkerWithDpRank, f64>,
    temperature: f64,
) -> (WorkerWithDpRank, f64) {
    let mut rng = rand::rng();
    softmax_sample_with_sample(logits, temperature, rng.random())
}

fn softmax_sample_with_sample(
    logits: &HashMap<WorkerWithDpRank, f64>,
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
        let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
        let decode_block = request
            .decode_blocks
            .get(&worker)
            .copied()
            .unwrap_or(potential_prefill_block.floor() as usize) as f64;
        let logit = overlap_weight * potential_prefill_block + decode_block;

        tracing::debug!(
            "{formula_name} for worker_id={} dp_rank={:?} with {overlap_blocks} cached blocks: {logit:.3} \
             = {overlap_weight:.1} * prefill_blocks + decode_blocks \
             = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}",
            worker.worker_id,
            worker.dp_rank
        );

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
        let decode_blocks = &request.decode_blocks;
        let prefill_tokens = &request.prefill_tokens;

        let overlap_weight = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.overlap_score_weight)
            .unwrap_or(self.kv_router_config.overlap_score_weight);

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);

        let get_score = |worker: WorkerWithDpRank| -> f64 {
            let effective_overlap_blocks = *request
                .effective_overlap_blocks
                .get(&worker)
                .unwrap_or(&0.0);
            let host_pinned_overlap_blocks = request
                .tier_overlap_blocks
                .host_pinned
                .get(&worker)
                .copied()
                .unwrap_or(0);
            let disk_overlap_blocks = request
                .tier_overlap_blocks
                .disk
                .get(&worker)
                .copied()
                .unwrap_or(0);
            let prefill_token = *prefill_tokens.get(&worker).unwrap_or(&isl);
            let potential_prefill_block = (prefill_token as f64) / (block_size as f64);
            let decode_block = *decode_blocks
                .get(&worker)
                .unwrap_or(&(potential_prefill_block.floor() as usize))
                as f64;

            let logit = overlap_weight * potential_prefill_block + decode_block;

            tracing::debug!(
                "Formula for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks \
                 (host_pinned={} disk={}): {logit:.3} \
                 = {overlap_weight:.1} * prefill_blocks + decode_blocks \
                 = {overlap_weight:.1} * {potential_prefill_block:.3} + {decode_block:.3}",
                worker.worker_id,
                worker.dp_rank,
                host_pinned_overlap_blocks,
                disk_overlap_blocks,
            );

            logit
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
            let mut worker_logits = HashMap::new();
            for worker in worker_iter {
                let score = get_score(worker);
                worker_logits.insert(worker, score);
            }

            softmax_sample(&worker_logits, temperature)
        };

        let best_host_pinned_overlap_blocks = request
            .tier_overlap_blocks
            .host_pinned
            .get(&best_worker)
            .copied()
            .unwrap_or(0);
        let best_disk_overlap_blocks = request
            .tier_overlap_blocks
            .disk
            .get(&best_worker)
            .copied()
            .unwrap_or(0);

        if self.worker_type == "decode" {
            tracing::info!(
                "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, host_pinned blocks: {}, disk blocks: {}",
                self.worker_type,
                best_worker.worker_id,
                best_worker.dp_rank,
                best_logit,
                best_host_pinned_overlap_blocks,
                best_disk_overlap_blocks,
            );
            return Ok(WorkerSelectionResult {
                worker: best_worker,
                required_blocks: request_blocks as u64,
                overlap_blocks: request
                    .effective_overlap_blocks
                    .get(&best_worker)
                    .copied()
                    .unwrap_or(0.0)
                    .round() as u32,
                effective_overlap_blocks: request
                    .effective_overlap_blocks
                    .get(&best_worker)
                    .copied()
                    .unwrap_or(0.0),
                cached_tokens: request
                    .effective_cached_tokens
                    .get(&best_worker)
                    .copied()
                    .unwrap_or(0),
            });
        }

        let best_overlap = request
            .effective_overlap_blocks
            .get(&best_worker)
            .copied()
            .unwrap_or(0.0);

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
            "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}, host_pinned blocks: {}, disk blocks: {}, tree size: {}{}",
            self.worker_type,
            best_worker.worker_id,
            best_worker.dp_rank,
            best_logit,
            best_overlap,
            best_host_pinned_overlap_blocks,
            best_disk_overlap_blocks,
            tree_size,
            total_blocks_info
        );

        Ok(WorkerSelectionResult {
            worker: best_worker,
            required_blocks: request_blocks as u64,
            overlap_blocks: request
                .effective_overlap_blocks
                .get(&best_worker)
                .copied()
                .unwrap_or(0.0)
                .round() as u32,
            effective_overlap_blocks: request
                .effective_overlap_blocks
                .get(&best_worker)
                .copied()
                .unwrap_or(0.0),
            cached_tokens: request
                .effective_cached_tokens
                .get(&best_worker)
                .copied()
                .unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sample_single_key() {
        let mut logits = HashMap::new();
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

        let logits = HashMap::from([(worker1, 0.0), (worker2, 3.0), (worker3, 9.0)]);
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
}
