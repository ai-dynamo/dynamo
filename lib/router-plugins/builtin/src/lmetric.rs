// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LMetric: multiplicative cache-and-load scoring with KV-cache hot-spot mitigation.
//!
//! Ported from "Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling"
//! (<https://arxiv.org/abs/2603.15202>). Each worker scores
//! `new_prefill_tokens × batch_size` and the lowest score wins. The product needs no weight
//! between its cache-aware and load-aware factors: a weight on either factor scales every
//! worker's score equally and cancels out of the comparison.
//!
//! Two factors are floored at one so neither alone can zero a score. A worker holding the whole
//! prompt still prefills one token, and the batch size counts the incoming request. Without the
//! floors, every fully cached worker and every idle worker would tie at zero regardless of the
//! other factor.
//!
//! The hot-spot detector follows the paper's two phases. A request class is the set of requests
//! sharing the first `class_prefix_blocks` prompt blocks. For class `c` with arrival share `x`
//! over the last `window_requests` selections of prompts long enough to have a class, the workers
//! holding its prefix `M`, and all other
//! workers `M̄`, an alarm is raised when `x / (1 − x) > |M| / |M̄|`. While alarmed, `M` is
//! filtered out once `2|M|` consecutive class requests would have been placed on it. The paper
//! uses a time window; this port counts selections so replays stay deterministic.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::protocols::WorkerWithDpRank;

use crate::signals::{device_overlap_blocks, uncached_prompt_tokens};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "lmetric";

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    hotspot_detection: bool,
    /// Prompt blocks that define a request class.
    class_prefix_blocks: usize,
    /// Recent selections over which class shares are measured.
    window_requests: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            hotspot_detection: true,
            class_prefix_blocks: 4,
            window_requests: 1_000,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if self.class_prefix_blocks == 0 || self.window_requests == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "class_prefix_blocks and window_requests must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct ClassState {
    arrivals: usize,
    /// Consecutive alarmed class requests whose best worker held the class prefix.
    hotspot_streak: usize,
}

/// Sliding window of recent request classes.
struct HotspotDetector {
    class_prefix_blocks: usize,
    window_requests: usize,
    window: VecDeque<u64>,
    classes: HashMap<u64, ClassState>,
}

impl HotspotDetector {
    fn new(parameters: &Parameters) -> Self {
        Self {
            class_prefix_blocks: parameters.class_prefix_blocks,
            window_requests: parameters.window_requests,
            window: VecDeque::new(),
            classes: HashMap::new(),
        }
    }

    /// Record one arrival and return its class state and share of the window.
    fn record(&mut self, class: u64) -> (&mut ClassState, f64) {
        if self.window.len() == self.window_requests
            && let Some(expired) = self.window.pop_front()
            && let Some(state) = self.classes.get_mut(&expired)
        {
            state.arrivals -= 1;
            if state.arrivals == 0 {
                self.classes.remove(&expired);
            }
        }
        self.window.push_back(class);
        let window_len = self.window.len() as f64;
        let state = self.classes.entry(class).or_default();
        state.arrivals += 1;
        let share = state.arrivals as f64 / window_len;
        (state, share)
    }
}

struct LMetricPicker {
    detector: Option<HotspotDetector>,
}

impl LMetricPicker {
    fn new(parameters: &Parameters) -> Self {
        Self {
            detector: parameters
                .hotspot_detection
                .then(|| HotspotDetector::new(parameters)),
        }
    }
}

impl WorkerPicker for LMetricPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let candidates = input.candidates();
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let score = |row: usize| -> Option<(f64, WorkerWithDpRank)> {
            let new_prefill = uncached_prompt_tokens(context, cache.get(row)?).max(1);
            let batch_size = load.get(row)?.active_requests() + 1;
            Some((
                new_prefill as f64 * batch_size as f64,
                candidates.get(row)?.worker(),
            ))
        };
        let best_of = |rows: &mut dyn Iterator<Item = usize>| {
            rows.filter_map(|row| score(row).map(|key| (key, row)))
                .min_by(|(left, _), (right, _)| {
                    left.0
                        .total_cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                })
        };
        let best = best_of(&mut (0..candidates.len()))
            .map(|(_, row)| row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;

        let Some(detector) = self.detector.as_mut() else {
            return Ok(best);
        };
        let class_prefix_blocks = detector.class_prefix_blocks;
        let Some(&class) = context
            .prefix_hashes()
            .and_then(|hashes| hashes.get(class_prefix_blocks - 1))
        else {
            return Ok(best);
        };
        let holds_class = |row: usize| {
            cache
                .get(row)
                .is_some_and(|cache| device_overlap_blocks(cache) >= class_prefix_blocks as f64)
        };
        let holders = (0..candidates.len())
            .filter(|&row| holds_class(row))
            .count();
        let others = candidates.len() - holders;
        let (state, share) = detector.record(class);
        if holders == 0 || others == 0 || share * others as f64 <= (1.0 - share) * holders as f64 {
            state.hotspot_streak = 0;
            return Ok(best);
        }
        if !holds_class(best) {
            state.hotspot_streak = 0;
            return Ok(best);
        }
        state.hotspot_streak += 1;
        if state.hotspot_streak < 2 * holders {
            return Ok(best);
        }
        Ok(
            best_of(&mut (0..candidates.len()).filter(|&row| !holds_class(row)))
                .map_or(best, |(_, row)| row),
        )
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
                Box::new(LMetricPicker::new(&parameters)),
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
    use super::*;
    use crate::test_support::{Worker, request, select};

    fn policy(parameters: Parameters) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(LMetricPicker::new(&parameters)),
        )
    }

    fn without_detector() -> WorkerSelectionPolicy {
        policy(Parameters {
            hotspot_detection: false,
            ..Parameters::default()
        })
    }

    #[test]
    fn product_trades_cache_hits_against_batch_size() {
        let policy = without_detector();
        // 10 blocks = 160 tokens. Worker 1 caches 8 blocks: 32 new tokens × 4 = 128 beats
        // worker 0's 160 × 1 = 160 even though worker 1 runs three more requests.
        let workers = [Worker::new(0), Worker::new(1).cached(8).requests(3)];
        assert_eq!(select(&policy, request(10, 1), &workers), 1);
        // A fourth request makes worker 1 cost 32 × 5 = 160 and ties; worker 0 wins the tie.
        let workers = [Worker::new(0), Worker::new(1).cached(8).requests(4)];
        assert_eq!(select(&policy, request(10, 1), &workers), 0);
    }

    #[test]
    fn full_cache_hits_still_compare_batch_sizes() {
        let policy = without_detector();
        let workers = [
            Worker::new(0).cached(10).requests(3),
            Worker::new(1).cached(10).requests(1),
        ];
        assert_eq!(select(&policy, request(10, 1), &workers), 1);
    }

    #[test]
    fn hotspot_alarm_filters_holders_after_a_streak() {
        // One holder among four workers alarms once the class exceeds a quarter of arrivals.
        // The holder stays cheapest, so the streak reaches 2|M| = 2 on the second request.
        let policy = policy(Parameters {
            class_prefix_blocks: 2,
            window_requests: 4,
            ..Parameters::default()
        });
        let workers = [
            Worker::new(0).cached(10).requests(2),
            Worker::new(1).requests(1),
            Worker::new(2).requests(1),
            Worker::new(3).requests(1),
        ];
        assert_eq!(select(&policy, request(10, 7), &workers), 0);
        assert_eq!(select(&policy, request(10, 7), &workers), 1);
    }

    #[test]
    fn a_class_within_its_coverage_is_not_filtered() {
        // Two of four workers hold the class. Alternating with another class settles its share at
        // one half, where x / (1 − x) = 1 does not exceed |M| / |M̄| = 1. The warm-up alarms
        // while the window fills stop short of the 2|M| = 4 streak.
        let policy = policy(Parameters {
            class_prefix_blocks: 2,
            window_requests: 4,
            ..Parameters::default()
        });
        let workers = [
            Worker::new(0).cached(10).requests(2),
            Worker::new(1).cached(10).requests(2),
            Worker::new(2).requests(1),
            Worker::new(3).requests(1),
        ];
        for _ in 0..4 {
            assert_eq!(select(&policy, request(10, 7), &workers), 0);
            select(&policy, request(10, 8), &workers);
        }
    }

    #[test]
    fn rejects_zero_sized_windows() {
        let zero = |class_prefix_blocks, window_requests| {
            Parameters {
                class_prefix_blocks,
                window_requests,
                ..Parameters::default()
            }
            .validate()
        };
        assert!(zero(0, 10).is_err() && zero(4, 0).is_err());
        assert!(Parameters::default().validate().is_ok());
    }
}
