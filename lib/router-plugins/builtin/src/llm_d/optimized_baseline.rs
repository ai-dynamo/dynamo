// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! llm-d's optimized baseline: sticky to cached workers until they saturate.
//!
//! Ported from llm-d's `guides/optimized-baseline` profile, which pairs the
//! `prefix-cache-affinity-filter` with the `token-load-scorer`
//! (<https://llm-d.ai/blog/sticky-until-saturated-token-aware-routing>).
//!
//! - Workers whose cached share of the prompt reaches `affinity_threshold` are sticky. With no
//!   sticky worker, every worker stays in play.
//! - Each worker's TTFT is estimated from its prefill backlog, excluding this request, as
//!   `ttft_source` selects. When the best sticky worker's estimate trails the best non-sticky
//!   worker's by more than `max_ttft_penalty_ms`, stickiness breaks and every worker stays in play;
//!   otherwise only sticky workers do.
//!   - `throughput`: in-flight prefill tokens over `peak_prefill_tokens_per_second`, llm-d's
//!     default. The constant is one calibration point for one model, GPU, and parallelism.
//!   - `modeled`: the host's modeled prefill backlog, available when the router runs a
//!     prefill-load model (`router_prefill_load_model: ais`). Like llm-d's latency-predictor
//!     source, a worker without a modeled value never counts as the fastest.
//!   - `auto` (default): `modeled` when every candidate has a modeled backlog, otherwise
//!     `throughput` for all of them, since the two estimates are not comparable.
//! - Among the remaining workers, the token-load score `1 − min(1, tokens / queue_threshold_tokens)`
//!   picks the highest, with `tokens` the in-flight prefill tokens plus this request's uncached
//!   prompt tokens on that worker. Ties rotate.
//!
//! llm-d counts in-flight tokens as the uncached prompt tokens of requests that have not yet
//! streamed a token, which is Dynamo's `active_prefill_tokens`. The optional exploration
//! probability is omitted. Defaults match llm-d's, including its `peakPrefillThroughput`
//! calibrated for Qwen3-32B TP2 on H100; calibrate it for other deployments.
//!
//! The filter compares workers against each other, so the port runs it in the picker, after the
//! token-load scorer.

use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerCandidates, WorkerInputView, WorkerInputs, WorkerPicker, WorkerScorer,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};

use super::MaxScorePicker;
use crate::signals::{device_overlap_blocks, uncached_prompt_tokens};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "llm-d-optimized-baseline";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum TtftSource {
    Auto,
    Modeled,
    Throughput,
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    affinity_threshold: f64,
    max_ttft_penalty_ms: f64,
    ttft_source: TtftSource,
    peak_prefill_tokens_per_second: f64,
    queue_threshold_tokens: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            affinity_threshold: 0.8,
            max_ttft_penalty_ms: 18_000.0,
            ttft_source: TtftSource::Auto,
            peak_prefill_tokens_per_second: 15_928.0,
            queue_threshold_tokens: 4_194_304,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !(0.0..=1.0).contains(&self.affinity_threshold) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "affinity_threshold must be between 0.0 and 1.0",
            ));
        }
        if !(self.max_ttft_penalty_ms.is_finite() && self.max_ttft_penalty_ms >= 0.0) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "max_ttft_penalty_ms must be a finite non-negative number",
            ));
        }
        if !(self.peak_prefill_tokens_per_second.is_finite()
            && self.peak_prefill_tokens_per_second > 0.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "peak_prefill_tokens_per_second must be a finite positive number",
            ));
        }
        if self.queue_threshold_tokens == 0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "queue_threshold_tokens must be positive",
            ));
        }
        Ok(())
    }
}

struct TokenLoadScorer {
    queue_threshold_tokens: f64,
}

impl WorkerScorer for TokenLoadScorer {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError> {
        for (candidate, cost) in candidates.iter().zip(costs) {
            let (Some(cache), Some(load)) = (candidate.cache(), candidate.load()) else {
                return Err(WorkerSelectionPolicyError::failed(
                    "cache or load input unavailable",
                ));
            };
            let tokens = load.active_prefill_tokens() + uncached_prompt_tokens(context, cache);
            *cost = -(1.0 - (tokens as f64 / self.queue_threshold_tokens).min(1.0));
        }
        Ok(())
    }
}

struct AffinityPicker {
    parameters: Parameters,
    max_score: MaxScorePicker,
}

impl WorkerPicker for AffinityPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        let inputs = WorkerInputs::CACHE | WorkerInputs::LOAD;
        if self.parameters.ttft_source == TtftSource::Throughput {
            inputs
        } else {
            inputs | WorkerInputs::PREFILL_TIME
        }
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
        let request_blocks = context.request_blocks() as f64;
        let sticky = |row: usize| {
            request_blocks > 0.0
                && cache.get(row).is_some_and(|cache| {
                    device_overlap_blocks(cache) / request_blocks
                        >= self.parameters.affinity_threshold
                })
        };
        let modeled_ms = |row: usize| context.modeled_prefill_backlog_ms(candidates[row].worker());
        let use_model = match self.parameters.ttft_source {
            TtftSource::Throughput => false,
            TtftSource::Modeled => true,
            TtftSource::Auto => (0..candidates.len()).all(|row| modeled_ms(row).is_some()),
        };
        let ttft_ms = |row: usize| {
            if use_model {
                // llm-d scores a worker without a prediction as maximally slow.
                modeled_ms(row).map_or(f64::MAX, |ms| ms as f64)
            } else {
                load[row].active_prefill_tokens() as f64
                    / self.parameters.peak_prefill_tokens_per_second
                    * 1_000.0
            }
        };
        let best_ttft = |want_sticky: bool| {
            (0..candidates.len())
                .filter(|&row| sticky(row) == want_sticky)
                .map(ttft_ms)
                .min_by(f64::total_cmp)
        };
        let keep_sticky = match (best_ttft(true), best_ttft(false)) {
            (None, _) => false,
            (Some(_), None) => true,
            (Some(sticky_ttft), Some(other_ttft)) => {
                self.parameters.max_ttft_penalty_ms == 0.0
                    || sticky_ttft - other_ttft <= self.parameters.max_ttft_penalty_ms
            }
        };
        if keep_sticky && self.parameters.affinity_threshold > 0.0 {
            self.max_score
                .pick_lowest(candidates, (0..candidates.len()).filter(|&row| sticky(row)))
        } else {
            self.max_score.pick_lowest(candidates, 0..candidates.len())
        }
    }
}

fn policy(
    config: &KvRouterConfig,
    label: &'static str,
    parameters: Parameters,
) -> WorkerSelectionPolicy {
    WorkerSelectionPolicy::new(
        config.clone(),
        label,
        vec![Box::new(TokenLoadScorer {
            queue_threshold_tokens: parameters.queue_threshold_tokens as f64,
        })],
        Box::new(AffinityPicker {
            parameters,
            max_score: MaxScorePicker::default(),
        }),
    )
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            if parameters.ttft_source == TtftSource::Modeled
                && !config.router_prefill_load_model.is_enabled()
            {
                tracing::warn!(
                    policy = POLICY_TYPE,
                    "ttft_source=modeled without a router prefill-load model never breaks \
                     stickiness; set router_prefill_load_model or use ttft_source=auto"
                );
            }
            policy(config, worker_type.as_str(), parameters)
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
        policy(&KvRouterConfig::default(), "test", Parameters::default())
    }

    #[test]
    fn sticks_to_a_cached_worker_despite_more_prefill_backlog() {
        // Worker 1 caches 9 of 10 blocks but carries 10k in-flight prefill tokens, well within
        // the 18 s TTFT allowance at 15,928 tokens/s.
        let workers = [Worker::new(0), Worker::new(1).cached(9).prefill(10_000)];
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 1);
    }

    #[test]
    fn saturation_breaks_stickiness() {
        // 300k in-flight tokens is about 18.8 s of prefill, past the 18 s allowance, so the
        // idle worker's lower token load wins.
        let workers = [Worker::new(0), Worker::new(1).cached(9).prefill(300_000)];
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 0);
    }

    fn with_source(ttft_source: TtftSource) -> WorkerSelectionPolicy {
        policy(
            &KvRouterConfig::default(),
            "test",
            Parameters {
                ttft_source,
                ..Parameters::default()
            },
        )
    }

    #[test]
    fn modeled_backlog_overrides_the_throughput_constant() {
        // 300k tokens exceed the 18 s allowance at 15,928 tokens/s, but the model predicts 5 s,
        // so the cached worker keeps the request.
        let busy_but_fast = [
            Worker::new(0).modeled(0),
            Worker::new(1).cached(9).prefill(300_000).modeled(5_000),
        ];
        assert_eq!(select(&default_policy(), request(10, 1), &busy_but_fast), 1);
        // 10k tokens look cheap at the constant rate, but the model predicts 25 s.
        let light_but_slow = [
            Worker::new(0).modeled(0),
            Worker::new(1).cached(9).prefill(10_000).modeled(25_000),
        ];
        assert_eq!(
            select(&default_policy(), request(10, 1), &light_but_slow),
            0
        );
    }

    #[test]
    fn auto_falls_back_to_throughput_unless_every_worker_is_modeled() {
        // Worker 0 has no modeled backlog, so both workers use the constant: 300k tokens break
        // stickiness even though worker 1's model predicts 5 s.
        let workers = [
            Worker::new(0),
            Worker::new(1).cached(9).prefill(300_000).modeled(5_000),
        ];
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 0);
        // Forcing the model scores the unmodeled worker as maximally slow, so stickiness holds.
        assert_eq!(
            select(&with_source(TtftSource::Modeled), request(10, 1), &workers),
            1
        );
    }

    #[test]
    fn throughput_source_ignores_the_model() {
        let workers = [
            Worker::new(0).modeled(0),
            Worker::new(1).cached(9).prefill(300_000).modeled(5_000),
        ];
        assert_eq!(
            select(
                &with_source(TtftSource::Throughput),
                request(10, 1),
                &workers
            ),
            0
        );
    }

    #[test]
    fn without_sticky_workers_token_load_decides() {
        // Neither worker reaches 80% of the prompt; worker 0's smaller backlog wins even though
        // worker 1 caches more.
        let workers = [
            Worker::new(0).prefill(100),
            Worker::new(1).cached(5).prefill(1_000),
        ];
        assert_eq!(select(&default_policy(), request(10, 1), &workers), 0);
    }
}
