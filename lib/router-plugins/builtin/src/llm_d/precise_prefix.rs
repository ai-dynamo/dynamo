// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! llm-d's precise prefix-cache profile: a weighted sum of three scorers.
//!
//! Ported from `deploy/config/epp-precise-prefix-cache-config.yaml` in the llm-d inference
//! scheduler, which weights three scorers 2:1:1 and takes the highest weighted sum. Each scorer
//! returns a value in `[0, 1]`, higher is better:
//!
//! - prefix: `w · min(1, matched_tokens / scale)² + (1 − w) · matched_blocks / request_blocks`,
//!   with llm-d's defaults `w = 0` and `scale = 8192`;
//! - queue: `(max − q) / (max − min)` over the candidates' active requests, or 1 when all equal;
//! - KV-cache utilization: `1 − usage`, with usage the projected active KV footprint over the
//!   worker's advertised capacity. llm-d leaves a worker without metrics unscored; some Dynamo
//!   backends do not advertise capacity, so a worker without it scores a neutral 0.5 instead.
//!
//! Dynamo adds scorer costs, so each scorer contributes `−weight × score` and the picker takes the
//! lowest total, rotating among ties like llm-d's shuffled max-score picker. llm-d's queue scorer
//! reads the engine's waiting queue; Dynamo's active-request count also includes running requests.
//!
//! `class_weights` maps a request's policy class to its own weights, like llm-d's per-request
//! scheduling profiles. Requests without a listed class use `weights`.

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidate, WorkerCandidates, WorkerInputs, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

use super::MaxScorePicker;
use crate::signals::device_overlap_blocks;

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "llm-d-precise-prefix";

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Weights {
    prefix: f64,
    queue: f64,
    kv_cache_utilization: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            prefix: 2.0,
            queue: 1.0,
            kv_cache_utilization: 1.0,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    weights: Weights,
    class_weights: HashMap<String, Weights>,
    prefix_match_length_weight: f64,
    prefix_match_length_scale_tokens: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            weights: Weights::default(),
            class_weights: HashMap::new(),
            prefix_match_length_weight: 0.0,
            prefix_match_length_scale_tokens: 8_192,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        let valid_weight = |weight: f64| weight.is_finite() && weight >= 0.0;
        for (name, weights) in std::iter::once(("weights", &self.weights)).chain(
            self.class_weights
                .iter()
                .map(|(class, weights)| (class.as_str(), weights)),
        ) {
            if ![weights.prefix, weights.queue, weights.kv_cache_utilization]
                .into_iter()
                .all(valid_weight)
            {
                return Err(WorkerSelectionPolicyProviderError::new(format!(
                    "{name}: weights must be finite non-negative numbers"
                )));
            }
        }
        if !(0.0..=1.0).contains(&self.prefix_match_length_weight) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "prefix_match_length_weight must be between 0.0 and 1.0",
            ));
        }
        if self.prefix_match_length_scale_tokens == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "prefix_match_length_scale_tokens must be positive",
            ));
        }
        Ok(())
    }

    fn weights_for(&self, context: &WorkerSelectionContext<'_>) -> &Weights {
        context
            .policy_class()
            .and_then(|class| self.class_weights.get(class))
            .unwrap_or(&self.weights)
    }
}

struct PrefixScorer(Arc<Parameters>);

impl WorkerScorer for PrefixScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let parameters = &self.0;
        let weight = parameters.weights_for(context).prefix;
        let length_weight = parameters.prefix_match_length_weight;
        let request_blocks = context.request_blocks() as f64;
        for (candidate, cost) in candidates.iter().zip(costs) {
            let cache = candidate
                .cache()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
            let matched_blocks = device_overlap_blocks(cache).max(0.0);
            let ratio = if request_blocks == 0.0 {
                0.0
            } else {
                (matched_blocks / request_blocks).min(1.0)
            };
            let length = (matched_blocks * f64::from(context.block_size())
                / parameters.prefix_match_length_scale_tokens as f64)
                .min(1.0)
                .powi(2);
            *cost = -weight * (length_weight * length + (1.0 - length_weight) * ratio);
        }
        Ok(())
    }
}

struct QueueScorer(Arc<Parameters>);

impl WorkerScorer for QueueScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let weight = self.0.weights_for(context).queue;
        let queue = |candidate: WorkerCandidate<'_>| {
            candidate
                .load()
                .map(|load| load.active_requests())
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))
        };
        let (mut min, mut max) = (usize::MAX, 0);
        for candidate in candidates.iter() {
            let queue = queue(candidate)?;
            min = min.min(queue);
            max = max.max(queue);
        }
        for (candidate, cost) in candidates.iter().zip(costs) {
            let score = if max == min {
                1.0
            } else {
                (max - queue(candidate)?) as f64 / (max - min) as f64
            };
            *cost = -weight * score;
        }
        Ok(())
    }
}

struct KvCacheUtilizationScorer(Arc<Parameters>);

impl WorkerScorer for KvCacheUtilizationScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        let weight = self.0.weights_for(context).kv_cache_utilization;
        for (candidate, cost) in candidates.iter().zip(costs) {
            let used = candidate
                .load()
                .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?
                .decode_cost_blocks();
            let total = context
                .worker_capacity(candidate.worker())
                .and_then(|capacity| capacity.total_kv_blocks());
            let score = total.map_or(0.5, |total| 1.0 - (used / total as f64).min(1.0));
            *cost = -weight * score;
        }
        Ok(())
    }
}

fn policy(
    config: &KvRouterConfig,
    label: &'static str,
    parameters: Arc<Parameters>,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config.clone(),
        label,
        vec![
            Box::new(PrefixScorer(Arc::clone(&parameters))),
            Box::new(QueueScorer(Arc::clone(&parameters))),
            Box::new(KvCacheUtilizationScorer(parameters)),
        ],
        Box::new(MaxScorePicker::default()),
    )
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    let parameters = Arc::new(parameters);
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            policy(config, worker_type.as_str(), Arc::clone(&parameters))
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
    use super::*;
    use crate::test_support::{Worker, request, select};

    fn default_policy() -> WorkerSelectionPolicy {
        policy(
            &KvRouterConfig::default(),
            "test",
            Arc::new(Parameters::default()),
        )
    }

    #[test]
    fn prefix_weight_outranks_one_queue_step() {
        // Worker 1 caches the whole prompt (prefix 1 × 2) and runs one more request (queue 0):
        // 2 + 0 + 1 beats worker 0's 0 + 1 + 1.
        let workers = [
            Worker::new(0).requests(1).decode(0, 100),
            Worker::new(1).cached(10).requests(2).decode(0, 100),
        ];
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 1);
    }

    #[test]
    fn kv_cache_utilization_uses_advertised_capacity() {
        // Equal prefix and queue scores; worker 0's KV is 90% used against worker 1's 10%.
        let workers = [
            Worker::new(0).decode(90, 100),
            Worker::new(1).decode(10, 100),
        ];
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 1);
        // Without advertised capacity, worker 1 scores a neutral 0.5 and loses to 0.9.
        let mut workers = [Worker::new(0).decode(10, 100), Worker::new(1)];
        workers[1].total_kv_blocks = None;
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 0);
    }

    #[test]
    fn class_weights_follow_the_requests_policy_class() {
        let parameters = Parameters {
            class_weights: HashMap::from([(
                "batch".to_string(),
                Weights {
                    prefix: 0.0,
                    ..Weights::default()
                },
            )]),
            ..Parameters::default()
        };
        let policy = policy(&KvRouterConfig::default(), "test", Arc::new(parameters));
        let workers = [
            Worker::new(0).requests(1).decode(0, 100),
            Worker::new(1).cached(10).requests(2).decode(0, 100),
        ];
        assert_eq!(select(&policy, request(10, 1), &workers), 1);
        let mut batch = request(10, 1);
        batch.policy_class = Some("batch".to_string());
        assert_eq!(select(&policy, batch, &workers), 0);
    }

    #[test]
    fn rejects_negative_class_weights() {
        let parameters = Parameters {
            class_weights: HashMap::from([(
                "batch".to_string(),
                Weights {
                    queue: -1.0,
                    ..Weights::default()
                },
            )]),
            ..Parameters::default()
        };
        assert!(parameters.validate().is_err());
    }
}
