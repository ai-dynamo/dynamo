// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The default KV policy, implemented solely against the public worker-policy API.

use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerSelectionResult};
use dynamo_kv_router::{
    KvRouterConfig, KvSchedulerError, WorkerInputView, WorkerInputs, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionInput, WorkerSelectionPolicy,
    WorkerSelectionPolicyError, WorkerSelectionPolicyFactory, WorkerSelector,
};
use parking_lot::Mutex;
use std::sync::Arc;

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

/// Construct the builtin default from configured policy parameters.
/// Per-request score overrides are not used. Request load-tracking remains host-owned.
pub fn default_policy(config: KvRouterConfig, worker_label: &'static str) -> WorkerSelectionPolicy {
    policy_with_rng(config, worker_label, None, false)
}

fn policy_with_rng(
    config: KvRouterConfig,
    worker_label: &'static str,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    plain_decode: bool,
) -> WorkerSelectionPolicy {
    let picker = DefaultPicker {
        plain_decode,
        config: config.clone(),
        worker_label,
        rng,
        entries: Vec::new(),
        probabilities: Vec::new(),
    };
    WorkerSelectionPolicy::new(config, worker_label, Vec::new(), Box::new(picker))
        .with_exclusive_affinity(true)
}

/// Factory installed by routing hosts, including hosts without a custom catalog.
pub fn default_factory() -> WorkerSelectionPolicyFactory {
    Arc::new(|config, role, _partition| policy_for_role(config.clone(), role))
}

fn policy_for_role(
    config: KvRouterConfig,
    role: dynamo_kv_router::WorkerType,
) -> WorkerSelectionPolicy {
    let plain_decode =
        role == dynamo_kv_router::WorkerType::Decode && !config.conditional_disagg_enabled;
    policy_with_rng(config, role.default_selector_label(), None, plain_decode)
}

struct DefaultPicker {
    plain_decode: bool,
    config: KvRouterConfig,
    worker_label: &'static str,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
    entries: Vec<(usize, f64)>,
    probabilities: Vec<f64>,
}

impl DefaultPicker {
    fn choose(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<(usize, Option<f64>), WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let candidates = input.candidates();
        if candidates.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }
        let min_prefill =
            if context.tracks_prefill_tokens() && self.config.overlap_score_credit_decay > 0.0 {
                load.iter()
                    .map(|l| l.active_prefill_tokens())
                    .min()
                    .unwrap_or(0)
            } else {
                0
            };
        self.entries.clear();
        let direct_minimum = self.config.router_temperature == 0.0 && self.rng.is_none();
        let mut best_row = 0;
        let mut best_cost = f64::INFINITY;
        let mut ties = 0;
        for (row, candidate) in candidates.iter().enumerate() {
            let cache = &cache[row];
            let load = &load[row];
            let (estimated_overlap, cached_tokens) = cache.accounting_cache_estimate();
            let device = if context.has_tier_matches() {
                cache.device_overlap_blocks()
            } else {
                estimated_overlap
            };
            let shared = if context.has_tier_matches() {
                cache.shared_beyond_device_blocks()
            } else {
                context
                    .shared_cache_hits()
                    .map_or(0, |hits| hits.hits_beyond(device.round().max(0.0) as u32))
            };
            let decay = if context.tracks_prefill_tokens()
                && self.config.overlap_score_credit_decay > 0.0
            {
                let excess = load.active_prefill_tokens().saturating_sub(min_prefill) as f64
                    / context.block_size() as f64;
                1.0 / (1.0
                    + self.config.overlap_score_credit_decay
                        * (excess / context.request_blocks() as f64))
            } else {
                1.0
            };
            // Plain disaggregated decode is load-only. Conditional decode retains cache credit.
            let overlap_credit = if self.plain_decode && !context.tracks_prefill_tokens() {
                0.0
            } else {
                self.config.overlap_score_credit
            };
            let credit = overlap_credit * decay * device
                + self.config.host_cache_hit_weight * cache.host_overlap_blocks()
                + self.config.disk_cache_hit_weight * cache.disk_overlap_blocks()
                + self.config.shared_cache_multiplier * shared as f64;
            let request_cost =
                self.config.decode_active_request_weight * load.active_requests() as f64;
            let logit = if self.worker_label == "decode"
                && !context.tracks_prefill_tokens()
                && overlap_credit > 0.0
            {
                (load.decode_cost_blocks() - credit).max(0.0) + request_cost
            } else {
                let raw_tokens = if !context.tracks_prefill_tokens() {
                    0
                } else if load.is_available() {
                    let uncached = context.prompt_tokens().saturating_sub(cached_tokens);
                    (load.active_prefill_tokens() + uncached).saturating_add(cached_tokens)
                } else {
                    context.prompt_tokens()
                };
                let prefill = (raw_tokens as f64 / context.block_size() as f64 - credit).max(0.0);
                self.config.prefill_load_scale * prefill + load.decode_cost_blocks() + request_cost
            };
            let cost = logit * candidate.preferred_taint_multiplier().unwrap_or(1.0);
            if !cost.is_finite() {
                return Err(WorkerSelectionPolicyError::failed(
                    "default policy produced non-finite cost",
                ));
            }
            if direct_minimum {
                if cost < best_cost {
                    best_row = row;
                    best_cost = cost;
                    ties = 1;
                } else if cost == best_cost {
                    ties += 1;
                    if fastrand::usize(0..ties) == 0 {
                        best_row = row;
                    }
                }
            } else {
                self.entries.push((row, cost));
            }
        }
        if direct_minimum {
            return Ok((best_row, Some(best_cost)));
        }
        if context.pinned_worker().is_some() {
            let (row, cost) = self.entries[0];
            return Ok((row, Some(cost)));
        }
        // Canonical order is required only for deterministic replay, never for production ties.
        if self.rng.is_some() {
            self.entries.sort_unstable_by_key(|(row, _)| {
                let worker = candidates[*row].worker();
                (worker.worker_id, worker.dp_rank)
            });
        }
        let mut rng = self.rng.as_ref().map(|rng| rng.lock());
        let selected = if self.config.router_temperature == 0.0 {
            let mut best = 0;
            let mut best_cost = f64::INFINITY;
            let mut ties = 0;
            for (index, (_, cost)) in self.entries.iter().enumerate() {
                if *cost < best_cost {
                    best = index;
                    best_cost = *cost;
                    ties = 1;
                } else if *cost == best_cost {
                    ties += 1;
                    let sample = match rng.as_mut() {
                        Some(rng) => rng.usize(0..ties),
                        None => fastrand::usize(0..ties),
                    };
                    if sample == 0 {
                        best = index;
                    }
                }
            }
            best
        } else {
            let sample = match rng.as_mut() {
                Some(rng) => rng.f64(),
                None => fastrand::f64(),
            };
            softmax_sample_index(
                &self.entries,
                |(_, cost)| *cost,
                self.config.router_temperature,
                sample,
                &mut self.probabilities,
            )
        };
        let (row, cost) = self.entries[selected];
        Ok((row, Some(cost)))
    }
}

impl WorkerPicker for DefaultPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD | WorkerInputs::PREFERRED_TAINT
    }
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        self.choose(context, input).map(|(row, _)| row)
    }
    fn pick_with_cost(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<(usize, Option<f64>), WorkerSelectionPolicyError> {
        self.choose(context, input)
    }
}

/// Synchronous adapter for direct callers and replay. Scheduler actors use `default_policy`
/// directly and do not acquire this adapter's mutex.
pub struct DefaultWorkerSelector {
    kv_router_config: KvRouterConfig,
    worker_type: &'static str,
    policy: Mutex<WorkerSelectionPolicy>,
    rng: Option<Arc<Mutex<fastrand::Rng>>>,
}
impl DefaultWorkerSelector {
    pub fn new(config: Option<KvRouterConfig>, worker_type: &'static str) -> Self {
        Self::with_rng(config.unwrap_or_default(), worker_type, None)
    }
    /// Construct a reproducible selector. Clones share its random stream, as before.
    pub fn new_seeded(
        config: Option<KvRouterConfig>,
        worker_type: &'static str,
        seed: u64,
    ) -> Self {
        Self::with_rng(
            config.unwrap_or_default(),
            worker_type,
            Some(Arc::new(Mutex::new(fastrand::Rng::with_seed(seed)))),
        )
    }
    fn with_rng(
        config: KvRouterConfig,
        worker_type: &'static str,
        rng: Option<Arc<Mutex<fastrand::Rng>>>,
    ) -> Self {
        Self {
            policy: Mutex::new(policy_with_rng(
                config.clone(),
                worker_type,
                rng.clone(),
                false,
            )),
            kv_router_config: config,
            worker_type,
            rng,
        }
    }
}
impl Clone for DefaultWorkerSelector {
    fn clone(&self) -> Self {
        Self::with_rng(
            self.kv_router_config.clone(),
            self.worker_type,
            self.rng.clone(),
        )
    }
}
impl std::fmt::Debug for DefaultWorkerSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultWorkerSelector")
            .field("kv_router_config", &self.kv_router_config)
            .field("worker_type", &self.worker_type)
            .finish_non_exhaustive()
    }
}
impl<C: WorkerConfigLike> WorkerSelector<C> for DefaultWorkerSelector {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }
    fn uses_exclusive_affinity_target(&self) -> bool {
        true
    }
    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        self.policy.lock().select_worker(input)
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Parameters {
    overlap_score_credit: Option<f64>,
    overlap_score_credit_decay: Option<f64>,
    prefill_load_scale: Option<f64>,
    decode_active_request_weight: Option<f64>,
    host_cache_hit_weight: Option<f64>,
    disk_cache_hit_weight: Option<f64>,
    shared_cache_multiplier: Option<f64>,
    router_temperature: Option<f64>,
}

pub(super) fn register(
    registry: &mut dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry,
) -> Result<(), dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistryError> {
    use dynamo_kv_router::services::selection::WorkerSelectionPolicyProviderError;
    registry.register(
        "dynamo-default-cost-fn",
        Arc::new(|parameters| {
            let parameters: Parameters = parameters.deserialize()?;
            for (name, value) in [
                ("overlap_score_credit", parameters.overlap_score_credit),
                (
                    "overlap_score_credit_decay",
                    parameters.overlap_score_credit_decay,
                ),
                ("prefill_load_scale", parameters.prefill_load_scale),
                (
                    "decode_active_request_weight",
                    parameters.decode_active_request_weight,
                ),
                ("host_cache_hit_weight", parameters.host_cache_hit_weight),
                ("disk_cache_hit_weight", parameters.disk_cache_hit_weight),
                (
                    "shared_cache_multiplier",
                    parameters.shared_cache_multiplier,
                ),
                ("router_temperature", parameters.router_temperature),
            ] {
                if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                    return Err(WorkerSelectionPolicyProviderError::new(format!(
                        "{name} must be finite and non-negative"
                    )));
                }
            }
            Ok(Arc::new(
                move |config: &KvRouterConfig, role, _partition| {
                    let mut config = config.clone();
                    if let Some(value) = parameters.overlap_score_credit {
                        config.overlap_score_credit = value;
                    }
                    if let Some(value) = parameters.overlap_score_credit_decay {
                        config.overlap_score_credit_decay = value;
                    }
                    if let Some(value) = parameters.prefill_load_scale {
                        config.prefill_load_scale = value;
                    }
                    if let Some(value) = parameters.decode_active_request_weight {
                        config.decode_active_request_weight = value;
                    }
                    if let Some(value) = parameters.host_cache_hit_weight {
                        config.host_cache_hit_weight = value;
                    }
                    if let Some(value) = parameters.disk_cache_hit_weight {
                        config.disk_cache_hit_weight = value;
                    }
                    if let Some(value) = parameters.shared_cache_multiplier {
                        config.shared_cache_multiplier = value;
                    }
                    if let Some(value) = parameters.router_temperature {
                        config.router_temperature = value;
                    }
                    policy_for_role(config, role)
                },
            ))
        }),
    )
}
