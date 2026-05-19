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
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

impl DefaultWorkerSelector {
    pub fn new(kv_router_config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self {
            kv_router_config: kv_router_config.unwrap_or_default(),
            worker_type,
        }
    }

    fn worker_logit(
        &self,
        request: &SchedulingRequest,
        worker: WorkerWithDpRank,
        block_size: u32,
        weights: LogitWeights,
        formula_name: &'static str,
    ) -> f64 {
        let block_size_f64 = block_size as f64;
        let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
        let has_tier_overlap_blocks = !request.tier_overlap_blocks.device.is_empty()
            || !request.tier_overlap_blocks.host_pinned.is_empty()
            || !request.tier_overlap_blocks.disk.is_empty();
        let device_overlap_blocks = request
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
        let raw_prefill_tokens = request.raw_prefill_tokens_for(worker) as f64;

        let host_overlap_blocks = request
            .tier_overlap_blocks
            .host_pinned
            .get(&worker)
            .copied()
            .unwrap_or(0) as f64;
        let disk_overlap_blocks = request
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
        let overlap_credit_blocks = weights.overlap_score_credit * device_overlap_blocks
            + self.kv_router_config.host_cache_hit_weight * host_overlap_blocks
            + self.kv_router_config.disk_cache_hit_weight * disk_overlap_blocks
            + shared_overlap_blocks;
        let adjusted_prefill_blocks = (raw_prefill_blocks - overlap_credit_blocks).max(0.0);
        let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
        let decode_cost_blocks = request.decode_blocks.get(&worker).copied().unwrap_or(0) as f64;
        let logit = prefill_cost_blocks + decode_cost_blocks;

        if shared_beyond > 0 {
            tracing::debug!(
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks, \
                 {shared_beyond} shared blocks beyond device (multiplier={shared_cache_multiplier:.2}): {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} \
                 (raw_prefill_blocks: {raw_prefill_blocks:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3})",
                worker.worker_id,
                worker.dp_rank,
                shared_cache_multiplier = weights.shared_cache_multiplier,
                prefill_load_scale = weights.prefill_load_scale
            );
        } else {
            tracing::debug!(
                "{formula_name} for worker_id={} dp_rank={:?} with {effective_overlap_blocks:.2} effective cached blocks: {logit:.3} \
                 = prefill_load_scale * adjusted_prefill_blocks + decode_blocks \
                 = {prefill_load_scale:.3} * {adjusted_prefill_blocks:.3} + {decode_cost_blocks:.3} \
                 (raw_prefill_blocks: {raw_prefill_blocks:.3}, overlap_credit_blocks: {overlap_credit_blocks:.3})",
                worker.worker_id,
                worker.dp_rank,
                prefill_load_scale = weights.prefill_load_scale
            );
        }

        logit
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
        let eligibility = request.eligibility();
        eligibility.validate_pinned_worker()?;

        let pinned_worker = eligibility.pinned_worker();

        if pinned_worker.is_none()
            && !eligibility.has_eligible_worker(
                workers
                    .iter()
                    .map(|(&worker_id, config)| (worker_id, config)),
            )
        {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let request_blocks = request.request_blocks(block_size);

        let weights = LogitWeights {
            overlap_score_credit: request
                .router_config_override
                .as_ref()
                .and_then(|cfg| cfg.overlap_score_credit)
                .unwrap_or(self.kv_router_config.overlap_score_credit),
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
            pinned_worker_config(workers, worker)?;
            if workers
                .get(&worker.worker_id)
                .is_some_and(|config| !eligibility.allows_worker(worker.worker_id, config))
            {
                return Err(KvSchedulerError::NoEndpoints);
            }

            let logit = self.worker_logit(request, worker, block_size, weights, "Pinned formula");
            let effective_overlap_blocks = request.effective_overlap_blocks_for(worker);
            let cached_tokens = request.effective_cached_tokens_for(worker);

            tracing::info!(
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
            });
        }

        let temperature = request
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.router_temperature)
            .unwrap_or(self.kv_router_config.router_temperature);
        let get_score = |worker: WorkerWithDpRank| -> f64 {
            let base_score = self.worker_logit(request, worker, block_size, weights, "Formula");
            let Some(config) = workers.get(&worker.worker_id) else {
                return base_score;
            };
            match request
                .routing_constraints
                .preferred_taint_multiplier(config.taints())
            {
                Some(multiplier) => base_score * multiplier,
                None => base_score,
            }
        };

        let worker_iter = workers
            .iter()
            .filter(move |(worker_id, config)| eligibility.allows_worker(**worker_id, *config))
            .flat_map(|(worker_id, config)| {
                let data_parallel_size = config.data_parallel_size();
                let data_parallel_start_rank = config.data_parallel_start_rank();
                (data_parallel_start_rank..(data_parallel_start_rank + data_parallel_size))
                    .map(move |dp_rank| WorkerWithDpRank::new(*worker_id, dp_rank))
            });

        let (best_worker, best_logit) = if temperature == 0.0 {
            let mut best_worker = None;
            let mut best_logit = f64::INFINITY;
            let mut tie_count = 0usize;
            let mut rng = rand::rng();
            for worker in worker_iter {
                let score = get_score(worker);
                if score < best_logit {
                    best_worker = Some(worker);
                    best_logit = score;
                    tie_count = 1;
                    continue;
                }

                if score == best_logit {
                    tie_count += 1;
                    // Reservoir sampling keeps tied minima uniform without collecting workers.
                    if rng.random_range(0..tie_count) == 0 {
                        best_worker = Some(worker);
                    }
                }
            }

            (best_worker.expect("worker_iter non-empty"), best_logit)
        } else {
            let mut worker_logits = FxHashMap::default();
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

        let total_blocks_info = workers
            .get(&best_worker.worker_id)
            .and_then(|cfg| cfg.total_kv_blocks())
            .map(|blocks| format!(", total blocks: {}", blocks))
            .unwrap_or_default();

        tracing::info!(
            "Selected worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}, host_pinned blocks: {}, disk blocks: {}{}",
            self.worker_type,
            best_worker.worker_id,
            best_worker.dp_rank,
            best_logit,
            best_overlap,
            best_host_pinned_overlap_blocks,
            best_disk_overlap_blocks,
            total_blocks_info
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: 16,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::default(),
            effective_cached_tokens: HashMap::default(),
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
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        };
        let mut selected = [false; 3];

        for _ in 0..120 {
            let result = selector.select_worker(&workers, &request, 16).unwrap();
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
    fn test_required_taints_return_no_endpoints_when_no_worker_matches() {
        let selector = DefaultWorkerSelector::new(Some(KvRouterConfig::default()), "test");
        let workers = HashMap::from([(
            10,
            TaintedWorkerConfig {
                taints: HashSet::from(["mdc-a".to_string()]),
            },
        )]);
        let request = SchedulingRequest {
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: 16,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::default(),
            effective_cached_tokens: HashMap::default(),
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
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector.select_worker(&workers, &request, 16);
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: 16,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::default(),
            effective_cached_tokens: HashMap::default(),
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
            routing_constraints: crate::protocols::RoutingConstraints {
                required_taints: HashSet::from(["mdc-b".to_string()]),
                preferred_taints: HashMap::new(),
            },
            shared_cache_hits: None,
            resp_tx: None,
        };

        let result = selector.select_worker(&workers, &request, 16).unwrap();
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
                maybe_request_id: Some("test".into()),
                token_seq: None,
                isl_tokens: 16,
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
                decode_blocks,
                prefill_tokens: FxHashMap::default(),
                track_prefill_tokens: true,
                router_config_override: None,
                update_states: false,
                lora_name: None,
                priority_jump: 0.0,
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

            let result = selector.select_worker(&workers, &request, 16).unwrap();
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: 16,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::default(),
            effective_cached_tokens: HashMap::default(),
            decode_blocks,
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
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

        let result = selector.select_worker(&workers, &request, 16).unwrap();
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: 16,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::default(),
            effective_cached_tokens: HashMap::default(),
            decode_blocks,
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
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

        let result = selector.select_worker(&workers, &request, 16).unwrap();
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            tier_overlap_blocks,
            effective_overlap_blocks,
            effective_cached_tokens,
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
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: Some(shared_hits),
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, block_size)
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            tier_overlap_blocks,
            effective_overlap_blocks: HashMap::new(),
            effective_cached_tokens,
            decode_blocks,
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, block_size)
            .unwrap();

        assert_eq!(
            result.worker, worker0,
            "prefill load scale should apply before adding decode block load"
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks,
            effective_cached_tokens: HashMap::new(),
            decode_blocks,
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: false,
            lora_name: None,
            priority_jump: 0.0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, block_size)
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
            maybe_request_id: Some("test".into()),
            token_seq: None,
            isl_tokens: isl,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks,
            effective_cached_tokens: HashMap::new(),
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
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };

        let result = selector
            .select_worker(&workers, &request, block_size)
            .unwrap();

        assert_eq!(result.worker, worker0);
    }
}
