// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use dashmap::DashMap;
use dynamo_kv_router::protocols::ActiveLoad;
use serde::{Deserialize, Serialize};

use crate::http::service::metrics::{
    WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE, WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE,
    WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE,
};
use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::metrics::WORKER_LOAD_METRICS;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::component::Client;
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{EventSubscriber, TypedEventSubscriber};

use super::{RuntimeConfigWatch, runtime_config_watch};
use crate::worker_type::WorkerType;

// Re-export worker type constants from timing.rs (single source of truth)
pub use crate::protocols::common::timing::{WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL};
const UNSET_DP_RANK_LABEL: &str = "none";
const KV_LOAD_FRESHNESS: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KvWorkerPoolSnapshot {
    pub endpoint: EndpointId,
    pub role: WorkerType,
    pub expected_ranks: u64,
    pub observed_ranks: u64,
    pub capacity_blocks: Option<u64>,
    pub used_blocks: Option<u64>,
    pub free_blocks: Option<u64>,
    pub active_decode_blocks: Option<u64>,
    pub active_prefill_tokens: Option<u64>,
    pub complete: bool,
}

#[derive(Default)]
struct KvMonitorSources {
    decode_workers: HashSet<u64>,
    prefill_workers: HashSet<u64>,
    decode_metrics_healthy: bool,
    prefill_metrics_healthy: bool,
    decode_configs_healthy: bool,
    prefill_configs_healthy: bool,
    decode_recovery_after: Option<Instant>,
    prefill_recovery_after: Option<Instant>,
    prefill_endpoint: Option<EndpointId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadMembership {
    Exact,
    Unknown,
    Foreign,
    Ambiguous,
}

fn classify_load_membership(
    worker_id: u64,
    source_workers: &HashSet<u64>,
    other_workers: &HashSet<u64>,
) -> LoadMembership {
    match (
        source_workers.contains(&worker_id),
        other_workers.contains(&worker_id),
    ) {
        (true, false) => LoadMembership::Exact,
        (false, false) => LoadMembership::Unknown,
        (false, true) => LoadMembership::Foreign,
        (true, true) => LoadMembership::Ambiguous,
    }
}

/// Clean up load and latency Prometheus metrics for a worker across the specified dp_ranks.
///
/// This removes metrics with the given worker_id, dp_rank, and worker_type label combination.
/// Called when workers are removed to prevent stale metrics from accumulating.
fn cleanup_worker_metrics(worker_id: u64, dp_ranks: &[u32], worker_type: &str) {
    let worker_id_str = worker_id.to_string();
    let m = &*WORKER_LOAD_METRICS;
    for dp_rank in dp_ranks {
        let dp_rank_str = dp_rank.to_string();
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];
        let _ = m.active_decode_blocks.remove_label_values(labels);
        let _ = m.active_prefill_tokens.remove_label_values(labels);
        let _ = WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE.remove_label_values(labels);
        let _ = WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE.remove_label_values(labels);
        let _ = WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.remove_label_values(labels);
    }

    let unset_labels = &[worker_id_str.as_str(), UNSET_DP_RANK_LABEL, worker_type];
    let _ = WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE.remove_label_values(unset_labels);
    let _ = WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE.remove_label_values(unset_labels);
    let _ = WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.remove_label_values(unset_labels);
}

fn expected_worker_dp_ranks(states: &DashMap<u64, WorkerLoadState>, worker_id: u64) -> Vec<u32> {
    states.get(&worker_id).map_or_else(
        || vec![0],
        |state| state.expected_dp_ranks.iter().copied().collect(),
    )
}

/// Default value for `max_num_batched_tokens` when the runtime config does not
/// report it. Set high enough that the frac-based overload check (which multiplies
/// this value by the threshold fraction) can never fire with realistic loads.
const DEFAULT_MAX_TOKENS: u64 = 10_000_000;

/// Compute the set of overloaded worker ids across all tracked worker load states
/// under the given thresholds. The returned set mixes decode workers (flagged by
/// `active_decode_blocks`) and prefill workers (flagged by `active_prefill_tokens`).
///
/// A monitor is owned 1-to-1 by its decode/aggregated WorkerSet. In disaggregated
/// serving it additionally subscribes to the explicitly attached prefill endpoint.
/// The mixed set therefore contains only workers from those two serving pools.
fn compute_overloaded_instances(
    worker_load_states: &DashMap<u64, WorkerLoadState>,
    cfg: &LoadThresholdConfig,
) -> Vec<u64> {
    worker_load_states
        .iter()
        .filter_map(|entry| {
            entry
                .value()
                .is_overloaded(
                    cfg.active_decode_blocks_threshold,
                    cfg.active_prefill_tokens_threshold,
                    cfg.active_prefill_tokens_threshold_frac,
                )
                .then_some(*entry.key())
        })
        .collect()
}

/// Publish the overloaded instance set to the decode/main router's Client and, in
/// disaggregated serving, to the registered prefill router's Client.
///
/// Prefill workers are routed by a separate `PrefillRouter` with its own Client.
/// `overloaded_instances` already includes prefill workers flagged via
/// `active_prefill_tokens`, but unless the set is published to the prefill Client
/// the `PrefillRouter`'s scheduler never consults it — making
/// `--active-prefill-tokens-threshold` (and its `_frac` variant) a silent no-op on
/// the prefill path. Ids that are not members of a given pool are
/// ignored when that Client derives its free workers, so publishing the full set
/// to both Clients is safe.
fn publish_overloaded_instances(
    decode_client: &Client,
    prefill_client_holder: &RwLock<Option<Client>>,
    overloaded_instances: &[u64],
) {
    if decode_client.set_overloaded_instances(overloaded_instances) {
        let counts = decode_client.routing_instance_counts();
        tracing::debug!(
            overloaded_instances = ?overloaded_instances,
            free_workers = counts.free,
            total_workers = counts.discovered,
            "overloaded instances changed"
        );
    }

    if let Some(prefill_client) = prefill_client_holder.read().unwrap().clone()
        && prefill_client.set_overloaded_instances(overloaded_instances)
    {
        let counts = prefill_client.routing_instance_counts();
        tracing::debug!(
            overloaded_instances = ?overloaded_instances,
            free_workers = counts.free,
            total_workers = counts.discovered,
            "overloaded instances changed (prefill pool)"
        );
    }
}

fn overload_reconciliation_needed(
    decode_client: &Client,
    prefill_client_holder: &RwLock<Option<Client>>,
) -> bool {
    decode_client.overload_reconciliation_needed()
        || prefill_client_holder
            .read()
            .unwrap()
            .as_ref()
            .is_some_and(Client::overload_reconciliation_needed)
}

fn publish_overloaded_instances_if_needed(
    decode_client: &Client,
    prefill_client_holder: &RwLock<Option<Client>>,
    overloaded_tracker: &OverloadedWorkerTracker,
    overloaded_changed: bool,
) -> bool {
    // NOTE: Recovery still relies on load producers publishing after meaningful capacity or
    // lifecycle changes. This only prevents the next observation from being suppressed when
    // request-path backpressure changed Client state outside this monitor's cached set.
    if !overloaded_changed && !overload_reconciliation_needed(decode_client, prefill_client_holder)
    {
        return false;
    }

    publish_overloaded_instances(
        decode_client,
        prefill_client_holder,
        &overloaded_tracker.ids(),
    );
    true
}

/// Configuration for worker load thresholds used in overload detection.
///
/// All thresholds are opt-in. An unset (`None`) field means the corresponding
/// check is skipped entirely — it never contributes to a worker being marked
/// overloaded. If all three are `None`, overload-based rejection is fully disabled.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LoadThresholdConfig {
    /// KV cache block utilization threshold (0.0-1.0).
    /// Worker is overloaded when `active_decode_blocks / total_blocks > threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_decode_blocks_threshold: Option<f64>,

    /// Absolute prefill token count threshold.
    /// Worker is overloaded when `active_prefill_tokens > threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_prefill_tokens_threshold: Option<u64>,

    /// Fraction of max_num_batched_tokens.
    /// Worker is overloaded when `active_prefill_tokens > frac * max_num_batched_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_prefill_tokens_threshold_frac: Option<f64>,
}

impl LoadThresholdConfig {
    /// Returns true if any threshold is configured.
    pub fn is_configured(&self) -> bool {
        self.active_decode_blocks_threshold.is_some()
            || self.active_prefill_tokens_threshold.is_some()
            || self.active_prefill_tokens_threshold_frac.is_some()
    }

    /// Validate threshold values shared by startup and dynamic configuration.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(threshold) = self.active_decode_blocks_threshold
            && (!threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
        {
            return Err(format!(
                "active_decode_blocks_threshold must be between 0.0 and 1.0, got {threshold}"
            ));
        }

        if let Some(threshold) = self.active_prefill_tokens_threshold_frac
            && (!threshold.is_finite() || threshold < 0.0)
        {
            return Err(format!(
                "active_prefill_tokens_threshold_frac must be a finite value greater than or equal to 0.0, got {threshold}"
            ));
        }

        Ok(())
    }
}

/// Worker load monitoring state per dp_rank
#[derive(Clone, Debug)]
struct DecodeOverloadLatchState {
    latched_overloaded: bool,
    kv_used_blocks_cleared: bool,
    active_decode_blocks_cleared: bool,
}

impl Default for DecodeOverloadLatchState {
    fn default() -> Self {
        Self {
            latched_overloaded: false,
            kv_used_blocks_cleared: true,
            active_decode_blocks_cleared: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WorkerLoadState {
    pub active_decode_blocks: HashMap<u32, u64>,
    active_decode_observed_at: HashMap<u32, Instant>,
    pub kv_used_blocks: HashMap<u32, u64>,
    pub kv_total_blocks: HashMap<u32, u64>,
    pub active_prefill_tokens: HashMap<u32, u64>,
    active_prefill_observed_at: HashMap<u32, Instant>,
    expected_dp_ranks: HashSet<u32>,
    kv_used_observed_at: HashMap<u32, Instant>,
    /// max_num_batched_tokens from runtime config (same for all dp_ranks)
    pub max_num_batched_tokens: HashMap<u32, u64>,
    decode_overload_latches: HashMap<u32, DecodeOverloadLatchState>,
}

impl WorkerLoadState {
    fn clear_observations(&mut self) {
        self.active_decode_blocks.clear();
        self.active_decode_observed_at.clear();
        self.kv_used_blocks.clear();
        self.kv_used_observed_at.clear();
        self.active_prefill_tokens.clear();
        self.active_prefill_observed_at.clear();
        self.decode_overload_latches.clear();
    }

    fn is_decode_signal_overloaded(
        used_blocks: u64,
        total_blocks: u64,
        active_decode_blocks_threshold: f64,
    ) -> bool {
        total_blocks > 0
            && (used_blocks as f64) > (active_decode_blocks_threshold * total_blocks as f64)
    }

    fn current_decode_overloaded(&self, dp_rank: u32, active_decode_blocks_threshold: f64) -> bool {
        let Some(&total_blocks) = self.kv_total_blocks.get(&dp_rank) else {
            return false;
        };

        self.kv_used_blocks
            .get(&dp_rank)
            .is_some_and(|&used_blocks| {
                Self::is_decode_signal_overloaded(
                    used_blocks,
                    total_blocks,
                    active_decode_blocks_threshold,
                )
            })
            || self
                .active_decode_blocks
                .get(&dp_rank)
                .is_some_and(|&active_blocks| {
                    Self::is_decode_signal_overloaded(
                        active_blocks,
                        total_blocks,
                        active_decode_blocks_threshold,
                    )
                })
    }

    fn update_decode_overload_latch(
        &mut self,
        dp_rank: u32,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
        active_decode_blocks_threshold: f64,
    ) {
        let Some(&total_blocks) = self.kv_total_blocks.get(&dp_rank) else {
            return;
        };
        if total_blocks == 0 {
            return;
        }

        let active_decode_overloaded = active_decode_blocks.is_some_and(|value| {
            Self::is_decode_signal_overloaded(value, total_blocks, active_decode_blocks_threshold)
        });
        let kv_used_overloaded = kv_used_blocks.is_some_and(|value| {
            Self::is_decode_signal_overloaded(value, total_blocks, active_decode_blocks_threshold)
        });

        let latch = self.decode_overload_latches.entry(dp_rank).or_default();
        if active_decode_overloaded || kv_used_overloaded {
            latch.latched_overloaded = true;
        }
        if let Some(value) = active_decode_blocks {
            latch.active_decode_blocks_cleared = !Self::is_decode_signal_overloaded(
                value,
                total_blocks,
                active_decode_blocks_threshold,
            );
        }
        if let Some(value) = kv_used_blocks {
            latch.kv_used_blocks_cleared = !Self::is_decode_signal_overloaded(
                value,
                total_blocks,
                active_decode_blocks_threshold,
            );
        }
        if latch.latched_overloaded
            && latch.kv_used_blocks_cleared
            && latch.active_decode_blocks_cleared
        {
            latch.latched_overloaded = false;
        }
    }

    #[cfg(test)]
    fn update_from_active_load(
        &mut self,
        active_load: &ActiveLoad,
        active_decode_blocks_threshold: Option<f64>,
    ) {
        self.update_from_active_load_at(
            active_load,
            active_decode_blocks_threshold,
            Instant::now(),
        );
    }

    fn update_from_active_load_at(
        &mut self,
        active_load: &ActiveLoad,
        active_decode_blocks_threshold: Option<f64>,
        observed_at: Instant,
    ) {
        let dp_rank = active_load.dp_rank;
        if let Some(active_blocks) = active_load.active_decode_blocks {
            self.active_decode_blocks.insert(dp_rank, active_blocks);
            self.active_decode_observed_at.insert(dp_rank, observed_at);
        }
        if let Some(kv_used_blocks) = active_load.kv_used_blocks {
            self.kv_used_blocks.insert(dp_rank, kv_used_blocks);
            self.kv_used_observed_at.insert(dp_rank, observed_at);
        }
        if let Some(active_tokens) = active_load.active_prefill_tokens {
            self.active_prefill_tokens.insert(dp_rank, active_tokens);
            self.active_prefill_observed_at.insert(dp_rank, observed_at);
        }
        if let Some(threshold) = active_decode_blocks_threshold {
            self.update_decode_overload_latch(
                dp_rank,
                active_load.active_decode_blocks,
                active_load.kv_used_blocks,
                threshold,
            );
        }
    }

    /// Returns true if ALL dp_ranks are overloaded based on the threshold logic.
    ///
    /// Each threshold is `Option<T>`. A `None` threshold means that check is
    /// skipped entirely — it cannot contribute to a dp_rank being overloaded. If all
    /// three thresholds are `None`, no dp_rank is ever overloaded.
    ///
    /// For each dp_rank, a dp_rank is overloaded if ANY of these conditions is met (OR logic):
    /// 1. `active_prefill_tokens > active_prefill_tokens_threshold` (absolute, if set)
    /// 2. `active_prefill_tokens > frac * max_num_batched_tokens` (fractional, if set)
    /// 3. decode overload latch set by either `kv_used_blocks` or `active_decode_blocks` (if set)
    ///
    /// The worker is overloaded only if ALL dp_ranks are overloaded.
    pub fn is_overloaded(
        &self,
        active_decode_blocks_threshold: Option<f64>,
        active_prefill_tokens_threshold: Option<u64>,
        active_prefill_tokens_threshold_frac: Option<f64>,
    ) -> bool {
        // Short-circuit if all thresholds are unset (i.e. no overload check can fire)
        if active_decode_blocks_threshold.is_none()
            && active_prefill_tokens_threshold.is_none()
            && active_prefill_tokens_threshold_frac.is_none()
        {
            return false;
        }

        // Get all dp_ranks we know about
        let all_dp_ranks: std::collections::HashSet<_> = self
            .active_decode_blocks
            .keys()
            .chain(self.kv_used_blocks.keys())
            .chain(self.decode_overload_latches.keys())
            .chain(self.active_prefill_tokens.keys())
            .copied()
            .collect();

        // If no dp_ranks known, not overloaded
        if all_dp_ranks.is_empty() {
            return false;
        }

        // Check if ALL dp_ranks are overloaded
        all_dp_ranks.iter().all(|&dp_rank| {
            // Check 1: prefill tokens threshold (absolute token count)
            if let Some(&active_tokens) = self.active_prefill_tokens.get(&dp_rank) {
                if let Some(abs_threshold) = active_prefill_tokens_threshold
                    && active_tokens > abs_threshold
                {
                    return true; // This dp_rank is overloaded due to absolute token threshold
                }

                // Check 2: prefill tokens threshold (fraction of max_num_batched_tokens)
                if let Some(frac) = active_prefill_tokens_threshold_frac {
                    let max_batched = self
                        .max_num_batched_tokens
                        .get(&dp_rank)
                        .copied()
                        .unwrap_or(DEFAULT_MAX_TOKENS);
                    let frac_threshold = (frac * max_batched as f64) as u64;
                    if active_tokens > frac_threshold {
                        return true;
                    }
                }
            }

            // Check 3: decode overload latch (OR-ed from kv_used_blocks and active_decode_blocks)
            if let Some(decode_threshold) = active_decode_blocks_threshold {
                let is_overloaded = self
                    .decode_overload_latches
                    .get(&dp_rank)
                    .map(|latch| latch.latched_overloaded)
                    .unwrap_or_else(|| self.current_decode_overloaded(dp_rank, decode_threshold));
                if is_overloaded {
                    return true;
                }
            }

            // If we can't perform any check or no threshold exceeded, this dp_rank is free
            false
        })
    }

    fn is_overloaded_for_config(&self, config: &LoadThresholdConfig) -> bool {
        self.is_overloaded(
            config.active_decode_blocks_threshold,
            config.active_prefill_tokens_threshold,
            config.active_prefill_tokens_threshold_frac,
        )
    }
}

#[derive(Debug, Default)]
struct OverloadedWorkerTracker {
    overloaded_workers: HashSet<u64>,
}

impl OverloadedWorkerTracker {
    fn update_worker(&mut self, worker_id: u64, overloaded: bool) -> bool {
        if overloaded {
            self.overloaded_workers.insert(worker_id)
        } else {
            self.overloaded_workers.remove(&worker_id)
        }
    }

    fn replace(&mut self, overloaded_workers: HashSet<u64>) -> bool {
        if self.overloaded_workers == overloaded_workers {
            return false;
        }
        self.overloaded_workers = overloaded_workers;
        true
    }

    fn remove_workers(&mut self, removed_workers: &[u64]) -> bool {
        let mut changed = false;
        for worker_id in removed_workers {
            changed |= self.overloaded_workers.remove(worker_id);
        }
        changed
    }

    #[cfg(test)]
    fn contains(&self, worker_id: u64) -> bool {
        self.overloaded_workers.contains(&worker_id)
    }

    fn ids(&self) -> Vec<u64> {
        self.overloaded_workers.iter().copied().collect()
    }
}

fn collect_overloaded_workers(
    worker_load_states: &DashMap<u64, WorkerLoadState>,
    config: &LoadThresholdConfig,
) -> HashSet<u64> {
    worker_load_states
        .iter()
        .filter_map(|entry| {
            entry
                .value()
                .is_overloaded_for_config(config)
                .then_some(*entry.key())
        })
        .collect()
}

fn merge_endpoint_runtime_configs(
    decode_configs: &RuntimeConfigWatch,
    prefill_configs: Option<&RuntimeConfigWatch>,
) -> HashMap<u64, ModelRuntimeConfig> {
    let mut merged = decode_configs.borrow().clone();
    let Some(prefill_configs) = prefill_configs else {
        return merged;
    };

    for (worker_id, config) in prefill_configs.borrow().iter() {
        if merged.contains_key(worker_id) {
            tracing::error!(
                worker_id,
                "worker is registered in both decode and prefill cache-owning endpoints; excluding ambiguous worker"
            );
            merged.remove(worker_id);
            continue;
        }
        merged.insert(*worker_id, config.clone());
    }
    merged
}

fn checked_add(total: &mut Option<u64>, value: u64) -> bool {
    let Some(current) = *total else {
        *total = Some(value);
        return true;
    };
    let Some(next) = current.checked_add(value) else {
        return false;
    };
    *total = Some(next);
    true
}

fn observation_is_fresh(
    observed_at: Instant,
    now: Instant,
    recovery_after: Option<Instant>,
) -> bool {
    now.saturating_duration_since(observed_at) <= KV_LOAD_FRESHNESS
        && recovery_after.is_none_or(|barrier| observed_at > barrier)
}

fn forget_worker_publisher(
    sequences: &mut HashMap<u64, u64>,
    publishers: &mut HashMap<u64, u64>,
    worker_id: u64,
) {
    let Some(publisher_id) = publishers.remove(&worker_id) else {
        return;
    };
    if !publishers.values().any(|current| *current == publisher_id) {
        sequences.remove(&publisher_id);
    }
}

fn set_worker_publisher(
    sequences: &mut HashMap<u64, u64>,
    publishers: &mut HashMap<u64, u64>,
    worker_id: u64,
    publisher_id: u64,
) -> bool {
    let Some(previous) = publishers.insert(worker_id, publisher_id) else {
        return false;
    };
    if previous == publisher_id {
        return false;
    }
    if !publishers.values().any(|current| *current == previous) {
        sequences.remove(&previous);
    }
    true
}

fn pool_snapshot(
    states: &DashMap<u64, WorkerLoadState>,
    endpoint: EndpointId,
    role: WorkerType,
    workers: &HashSet<u64>,
    source_healthy: bool,
    recovery_after: Option<Instant>,
    now: Instant,
) -> KvWorkerPoolSnapshot {
    let mut expected_ranks = 0_u64;
    let mut observed_ranks = 0_u64;
    let mut capacity_blocks = None;
    let mut used_blocks = None;
    let mut active_decode_blocks = None;
    let mut active_prefill_tokens = None;
    let mut active_decode_observed_ranks = 0_u64;
    let mut active_prefill_observed_ranks = 0_u64;
    let mut capacity_overflow = false;
    let mut used_overflow = false;
    let mut active_decode_overflow = false;
    let mut active_prefill_overflow = false;
    let mut complete = source_healthy && !workers.is_empty();

    for worker_id in workers {
        let Some(state) = states.get(worker_id) else {
            complete = false;
            continue;
        };
        if state.expected_dp_ranks.is_empty() {
            complete = false;
        }
        for &dp_rank in &state.expected_dp_ranks {
            expected_ranks = expected_ranks.saturating_add(1);
            let Some(&total) = state.kv_total_blocks.get(&dp_rank) else {
                complete = false;
                continue;
            };
            if !capacity_overflow && !checked_add(&mut capacity_blocks, total) {
                capacity_overflow = true;
                capacity_blocks = None;
                complete = false;
            }
            let Some(&used) = state.kv_used_blocks.get(&dp_rank) else {
                complete = false;
                continue;
            };
            let Some(&observed_at) = state.kv_used_observed_at.get(&dp_rank) else {
                complete = false;
                continue;
            };
            if used > total || !observation_is_fresh(observed_at, now, recovery_after) {
                complete = false;
                continue;
            }
            observed_ranks = observed_ranks.saturating_add(1);
            if !used_overflow && !checked_add(&mut used_blocks, used) {
                used_overflow = true;
                used_blocks = None;
                complete = false;
            }
        }
        if state
            .kv_total_blocks
            .keys()
            .chain(state.kv_used_blocks.keys())
            .any(|dp_rank| !state.expected_dp_ranks.contains(dp_rank))
        {
            complete = false;
        }
        for dp_rank in &state.expected_dp_ranks {
            let Some((&value, &observed_at)) = state
                .active_decode_blocks
                .get(dp_rank)
                .zip(state.active_decode_observed_at.get(dp_rank))
            else {
                continue;
            };
            if !observation_is_fresh(observed_at, now, recovery_after) {
                continue;
            }
            active_decode_observed_ranks = active_decode_observed_ranks.saturating_add(1);
            if !active_decode_overflow && !checked_add(&mut active_decode_blocks, value) {
                active_decode_overflow = true;
                active_decode_blocks = None;
            }
        }

        for dp_rank in &state.expected_dp_ranks {
            let Some((&value, &observed_at)) = state
                .active_prefill_tokens
                .get(dp_rank)
                .zip(state.active_prefill_observed_at.get(dp_rank))
            else {
                continue;
            };
            if !observation_is_fresh(observed_at, now, recovery_after) {
                continue;
            }
            active_prefill_observed_ranks = active_prefill_observed_ranks.saturating_add(1);
            if !active_prefill_overflow && !checked_add(&mut active_prefill_tokens, value) {
                active_prefill_overflow = true;
                active_prefill_tokens = None;
            }
        }
    }

    complete &= expected_ranks > 0 && observed_ranks == expected_ranks;
    if active_decode_observed_ranks != expected_ranks {
        active_decode_blocks = None;
    }
    if active_prefill_observed_ranks != expected_ranks {
        active_prefill_tokens = None;
    }
    let free_blocks = if complete {
        capacity_blocks.and_then(|capacity| used_blocks.and_then(|used| capacity.checked_sub(used)))
    } else {
        None
    };
    complete &= free_blocks.is_some();

    KvWorkerPoolSnapshot {
        endpoint,
        role,
        expected_ranks,
        observed_ranks,
        capacity_blocks,
        used_blocks,
        free_blocks,
        active_decode_blocks,
        active_prefill_tokens,
        complete,
    }
}

/// Worker monitor for tracking KV cache usage and overload states.
///
/// Cloning shares state via internal Arc-wrapped fields. This allows multiple pipelines
/// (e.g., chat and completions) to share the same monitor instance.
///
/// Prometheus metrics are exposed via [`WORKER_LOAD_METRICS`] (defined in `kv_router::sequence`),
/// which should be registered with the HTTP service's Prometheus registry using
/// [`register_worker_load_metrics`](crate::kv_router::metrics::register_worker_load_metrics).
///
/// In disaggregated mode, use `attach_prefill_client` to attach the prefill endpoint so the
/// monitor publishes the overloaded set to the prefill pool and cleans up TTFT metrics when
/// prefill workers are removed.
#[derive(Clone)]
pub struct KvWorkerMonitor {
    /// Decode endpoint client (used for ITL cleanup and overload detection)
    client: Client,
    /// Optional prefill endpoint client (used for TTFT cleanup in disaggregated mode)
    prefill_client: Arc<RwLock<Option<Client>>>,
    /// Notifies the monitoring task when a prefill client is registered
    prefill_client_notify: Arc<Notify>,
    worker_load_states: Arc<DashMap<u64, WorkerLoadState>>,
    sources: Arc<RwLock<KvMonitorSources>>,
    /// Load thresholds for overload detection. Each field is `Option<T>` — unset
    /// means the corresponding check in `is_overloaded` is skipped. If all three are
    /// `None`, rejection is fully disabled.
    thresholds: Arc<RwLock<LoadThresholdConfig>>,
    /// Guard to ensure start_monitoring() only runs once across clones
    started: Arc<AtomicBool>,
    start_lock: Arc<tokio::sync::Mutex<()>>,
    lifecycle: Arc<MonitorLifecycle>,
}

struct MonitorLifecycle {
    cancellation_token: CancellationToken,
    task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
}

impl Drop for MonitorLifecycle {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

impl KvWorkerMonitor {
    /// Create a new worker monitor with the given threshold configuration.
    ///
    /// Unset thresholds (`None`) remain unset and their corresponding checks
    /// in `is_overloaded` are skipped. Thresholds can be updated at runtime via
    /// [`set_load_threshold_config`](Self::set_load_threshold_config) or the
    /// individual setters.
    ///
    /// Prometheus metrics are exposed via [`WORKER_LOAD_METRICS`] and should be registered
    /// using [`register_worker_load_metrics`](crate::kv_router::metrics::register_worker_load_metrics)
    /// during HTTP service setup.
    ///
    /// For disaggregated mode, call `attach_prefill_client` after creation to enable
    /// prefill-pool overload publishing and TTFT metric cleanup when prefill workers
    /// are removed.
    pub fn new(client: Client, config: LoadThresholdConfig) -> Self {
        Self::new_inner(client, config, None)
    }

    pub(crate) fn new_with_task_guard(
        client: Client,
        config: LoadThresholdConfig,
        task_guard: dynamo_runtime::engine::EngineContextGuard,
    ) -> Self {
        Self::new_inner(client, config, Some(task_guard))
    }

    fn new_inner(
        client: Client,
        config: LoadThresholdConfig,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
    ) -> Self {
        let cancellation_token = client.endpoint.drt().child_token();
        Self {
            client,
            prefill_client: Arc::new(RwLock::new(None)),
            prefill_client_notify: Arc::new(Notify::new()),
            worker_load_states: Arc::new(DashMap::new()),
            sources: Arc::new(RwLock::new(KvMonitorSources::default())),
            thresholds: Arc::new(RwLock::new(config)),
            started: Arc::new(AtomicBool::new(false)),
            start_lock: Arc::new(tokio::sync::Mutex::new(())),
            lifecycle: Arc::new(MonitorLifecycle {
                cancellation_token,
                task_guard,
            }),
        }
    }

    /// Returns true iff the user explicitly configured at least one threshold.
    ///
    /// When false, all three per-field checks are skipped in `is_overloaded` and
    /// rejection is fully disabled. Callers that gate 529 responses on overload
    /// detection should check this before enabling the gate.
    pub fn is_configured(&self) -> bool {
        self.thresholds.read().unwrap().is_configured()
    }

    /// Attach the prefill router's `Client` for disaggregated mode.
    ///
    /// This is what wires prefill backpressure end-to-end: once attached, the monitor
    /// publishes the overloaded set to the prefill `Client` (so the PrefillRouter excludes
    /// overloaded workers / sheds when all are over) and watches the prefill
    /// endpoint to clean up TTFT gauges when prefill workers disappear.
    ///
    /// This method can be called after `start_monitoring` - the monitoring loop will
    /// be immediately notified and start watching the prefill endpoint.
    pub fn attach_prefill_client(&self, prefill_client: Client) {
        // Synchronously seed the freshly-attached prefill Client with the current
        // overloaded set BEFORE storing/notifying. Late attachment (prefill router
        // activates after workers are already overloaded) would otherwise leave a
        // window — between attach and the monitor loop's notify-driven seed — where
        // the prefill Client reports an empty overloaded set and admits requests it
        // should shed.
        let cfg = self.thresholds.read().unwrap().clone();
        let overloaded = compute_overloaded_instances(&self.worker_load_states, &cfg);
        prefill_client.set_overloaded_instances(&overloaded);

        let mut guard = self.prefill_client.write().unwrap();
        self.sources.write().unwrap().prefill_endpoint = Some(prefill_client.endpoint.id());
        *guard = Some(prefill_client);
        self.prefill_client_notify.notify_one();
        tracing::debug!(
            "KvWorkerMonitor: prefill client attached (seeded overloaded set; overload publish + TTFT cleanup)"
        );
    }

    pub(crate) fn pool_snapshots(&self, decode_role: WorkerType) -> Vec<KvWorkerPoolSnapshot> {
        let now = Instant::now();
        let sources = self.sources.read().unwrap();
        let mut snapshots = vec![pool_snapshot(
            &self.worker_load_states,
            self.client.endpoint.id(),
            decode_role,
            &sources.decode_workers,
            sources.decode_metrics_healthy && sources.decode_configs_healthy,
            sources.decode_recovery_after,
            now,
        )];
        if let Some(endpoint) = sources.prefill_endpoint.clone() {
            snapshots.push(pool_snapshot(
                &self.worker_load_states,
                endpoint,
                WorkerType::Prefill,
                &sources.prefill_workers,
                sources.prefill_metrics_healthy && sources.prefill_configs_healthy,
                sources.prefill_recovery_after,
                now,
            ));
        }
        snapshots
    }

    /// Get the current active decode blocks threshold, if configured.
    pub fn active_decode_blocks_threshold(&self) -> Option<f64> {
        self.thresholds
            .read()
            .unwrap()
            .active_decode_blocks_threshold
    }

    /// Set the active decode blocks threshold.
    pub fn set_active_decode_blocks_threshold(&self, threshold: f64) {
        self.thresholds
            .write()
            .unwrap()
            .active_decode_blocks_threshold = Some(threshold);
    }

    /// Get the current active prefill tokens threshold, if configured.
    pub fn active_prefill_tokens_threshold(&self) -> Option<u64> {
        self.thresholds
            .read()
            .unwrap()
            .active_prefill_tokens_threshold
    }

    /// Set the active prefill tokens threshold.
    pub fn set_active_prefill_tokens_threshold(&self, threshold: u64) {
        self.thresholds
            .write()
            .unwrap()
            .active_prefill_tokens_threshold = Some(threshold);
    }

    /// Get the current active prefill tokens threshold frac, if configured.
    pub fn active_prefill_tokens_threshold_frac(&self) -> Option<f64> {
        self.thresholds
            .read()
            .unwrap()
            .active_prefill_tokens_threshold_frac
    }

    /// Set the active prefill tokens threshold frac.
    pub fn set_active_prefill_tokens_threshold_frac(&self, frac: f64) {
        self.thresholds
            .write()
            .unwrap()
            .active_prefill_tokens_threshold_frac = Some(frac);
    }

    /// Get the current load threshold configuration. Unset fields are returned
    /// as `None` (no spurious fallback values).
    pub fn load_threshold_config(&self) -> LoadThresholdConfig {
        self.thresholds.read().unwrap().clone()
    }

    /// Update thresholds from a `LoadThresholdConfig`. Only fields that are
    /// `Some` in the input overwrite their counterparts; `None` fields leave
    /// the existing value untouched.
    pub fn set_load_threshold_config(&self, config: &LoadThresholdConfig) {
        let mut guard = self.thresholds.write().unwrap();
        if let Some(v) = config.active_decode_blocks_threshold {
            guard.active_decode_blocks_threshold = Some(v);
        }
        if let Some(v) = config.active_prefill_tokens_threshold {
            guard.active_prefill_tokens_threshold = Some(v);
        }
        if let Some(v) = config.active_prefill_tokens_threshold_frac {
            guard.active_prefill_tokens_threshold_frac = Some(v);
        }
    }
}

#[async_trait]
impl WorkerLoadMonitor for KvWorkerMonitor {
    /// Start background monitoring of worker KV cache usage.
    ///
    /// This is safe to call multiple times (e.g., from cloned monitors shared across
    /// pipelines) - only the first call spawns the background task.
    async fn start_monitoring(&self) -> anyhow::Result<()> {
        let _start_guard = self.start_lock.lock().await;
        if self.started.load(Ordering::Acquire) {
            tracing::debug!("Worker monitoring already started, skipping");
            return Ok(());
        }

        let endpoint = &self.client.endpoint;
        let cancellation_token = self.lifecycle.cancellation_token.child_token();

        let decode_configs_rx =
            match runtime_config_watch(endpoint, cancellation_token.clone()).await {
                Ok(rx) => rx,
                Err(error) => {
                    tracing::error!(
                        endpoint = %endpoint.id(),
                        %error,
                        "KvWorkerMonitor: failed to watch endpoint runtime configs"
                    );
                    return Err(error);
                }
            };

        // Subscribe to KV metrics events using EventSubscriber (Msgpack payloads)
        // This is optional - if NATS isn't available, we skip KV metrics but still do TTFT/ITL cleanup
        let (kv_metrics_rx, decode_metrics_healthy) = match EventSubscriber::for_endpoint(
            endpoint,
            KV_METRICS_SUBJECT,
        )
        .await
        {
            Ok(sub) => (Some(sub.typed::<ActiveLoad>()), true),
            Err(e) => {
                tracing::warn!(
                    "KvWorkerMonitor: KV metrics subscriber not available ({}), skipping load metrics.",
                    e
                );
                (None, false)
            }
        };

        // Watch decode endpoint instances for cleanup (ITL metrics)
        let mut decode_instances_rx = self.client.instance_avail_watcher();

        let worker_load_states = self.worker_load_states.clone();
        let client = self.client.clone();
        let decode_endpoint = endpoint.clone();
        let prefill_client_holder = self.prefill_client.clone();
        let prefill_client_notify = self.prefill_client_notify.clone();
        let thresholds = self.thresholds.clone();
        let started = self.started.clone();
        let task_guard = self.lifecycle.task_guard.clone();
        let sources = self.sources.clone();

        // Spawn background monitoring task
        self.started.store(true, Ordering::Release);
        tokio::spawn(async move {
            let _task_guard = task_guard;
            struct StartedGuard(Arc<AtomicBool>);

            impl Drop for StartedGuard {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }

            let _started_guard = StartedGuard(started);
            let mut kv_metrics_rx = kv_metrics_rx;
            let mut prefill_metrics_rx: Option<TypedEventSubscriber<ActiveLoad>> = None;
            let mut prefill_configs_rx: Option<RuntimeConfigWatch> = None;
            let mut decode_configs_rx = decode_configs_rx;

            let mut prefill_instances_rx: Option<tokio::sync::watch::Receiver<Vec<u64>>> = None;

            let mut overloaded_tracker = OverloadedWorkerTracker::default();
            let mut last_thresholds = thresholds.read().unwrap().clone();
            let mut decode_sequences = HashMap::new();
            let mut prefill_sequences = HashMap::new();
            let mut decode_publishers = HashMap::new();
            let mut prefill_publishers = HashMap::new();
            let mut metrics_reconnect = tokio::time::interval_at(
                tokio::time::Instant::now() + Duration::from_secs(1),
                Duration::from_secs(1),
            );
            metrics_reconnect.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            {
                let mut source_state = sources.write().unwrap();
                source_state.decode_workers =
                    decode_instances_rx.borrow().iter().copied().collect();
                source_state.decode_metrics_healthy = decode_metrics_healthy;
                source_state.decode_configs_healthy = true;
            }
            decode_configs_rx.mark_changed();

            loop {
                // Read from the exact decode endpoint and, when attached, the exact prefill
                // endpoint. The source bit is retained so membership can be validated before
                // accepting worker-owned state.
                let kv_event_future = async {
                    let (prefill_scope, event) = match (&mut kv_metrics_rx, &mut prefill_metrics_rx)
                    {
                        (Some(decode_rx), Some(prefill_rx)) => {
                            tokio::select! {
                                event = decode_rx.next() => (false, event),
                                event = prefill_rx.next() => (true, event),
                            }
                        }
                        (Some(decode_rx), None) => (false, decode_rx.next().await),
                        (None, Some(prefill_rx)) => (true, prefill_rx.next().await),
                        (None, None) => std::future::pending().await,
                    };
                    (prefill_scope, event)
                };

                let config_change_future = async {
                    if let Some(prefill_configs_rx) = &mut prefill_configs_rx {
                        tokio::select! {
                            result = decode_configs_rx.changed() => (false, result),
                            result = prefill_configs_rx.changed() => (true, result),
                        }
                    } else {
                        (false, decode_configs_rx.changed().await)
                    }
                };

                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Worker monitoring cancelled");
                        // `select!` gives no ordering guarantee between this branch and
                        // `config_change_future` below: a worker removal that discovery
                        // already reported may still be sitting unprocessed in
                        // `decode_configs_rx`/`prefill_configs_rx` if this branch wins
                        // the race. Reconcile once more against the latest borrowed
                        // snapshot (not `.changed()`, which only fires once) so that
                        // worker's `cleanup_worker_metrics` still runs before this task
                        // exits. Skipping it would leak that worker's gauges
                        // indefinitely, since they are process-global and nothing else
                        // ever cleans them up. This only clears workers discovery
                        // already dropped, so it does not race a replacement generation
                        // that reuses the same worker id under the next WorkerSet.
                        let runtime_configs = merge_endpoint_runtime_configs(
                            &decode_configs_rx,
                            prefill_configs_rx.as_ref(),
                        );
                        let removed_workers = worker_load_states
                            .iter()
                            .filter(|state| !runtime_configs.contains_key(state.key()))
                            .map(|state| {
                                (
                                    *state.key(),
                                    state.expected_dp_ranks.iter().copied().collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>();
                        for (worker_id, dp_ranks) in removed_workers {
                            cleanup_worker_metrics(worker_id, &dp_ranks, WORKER_TYPE_DECODE);
                            cleanup_worker_metrics(worker_id, &dp_ranks, WORKER_TYPE_PREFILL);
                        }
                        break;
                    }

                    _ = metrics_reconnect.tick() => {
                        if kv_metrics_rx.is_none() {
                            match EventSubscriber::for_endpoint(&decode_endpoint, KV_METRICS_SUBJECT).await {
                                Ok(subscriber) => {
                                    kv_metrics_rx = Some(subscriber.typed::<ActiveLoad>());
                                    sources.write().unwrap().decode_metrics_healthy = true;
                                }
                                Err(error) => tracing::debug!(
                                    endpoint = %decode_endpoint.id(),
                                    %error,
                                    "KvWorkerMonitor: decode KV metrics reconnect failed"
                                ),
                            }
                        }
                        let prefill_client = {
                            prefill_client_holder.read().unwrap().clone()
                        };
                        if prefill_metrics_rx.is_none()
                            && let Some(prefill_client) = prefill_client
                        {
                            let endpoint = prefill_client.endpoint.clone();
                            match EventSubscriber::for_endpoint(&endpoint, KV_METRICS_SUBJECT).await {
                                Ok(subscriber) => {
                                    prefill_metrics_rx = Some(subscriber.typed::<ActiveLoad>());
                                    sources.write().unwrap().prefill_metrics_healthy = true;
                                }
                                Err(error) => tracing::debug!(
                                    endpoint = %endpoint.id(),
                                    %error,
                                    "KvWorkerMonitor: prefill KV metrics reconnect failed"
                                ),
                            }
                        }
                    }

                    // Handle runtime config updates
                    (prefill_scope, result) = config_change_future => {
                        if result.is_err() {
                            if prefill_scope {
                                prefill_configs_rx = None;
                                sources.write().unwrap().prefill_configs_healthy = false;
                                tracing::warn!("prefill runtime-config watch closed");
                                continue;
                            }
                            sources.write().unwrap().decode_configs_healthy = false;
                            tracing::warn!("decode runtime-config watch closed");
                            break;
                        }

                        let recovery_after = Instant::now();
                        {
                            let mut source_state = sources.write().unwrap();
                            if prefill_scope {
                                source_state.prefill_recovery_after = Some(recovery_after);
                            } else {
                                source_state.decode_recovery_after = Some(recovery_after);
                            }
                        }

                        let runtime_configs = merge_endpoint_runtime_configs(
                            &decode_configs_rx,
                            prefill_configs_rx.as_ref(),
                        );

                        // Find workers that are being removed (not in runtime_configs anymore)
                        let removed_workers: Vec<u64> = worker_load_states
                            .iter()
                            .filter(|state| !runtime_configs.contains_key(state.key()))
                            .map(|state| *state.key())
                            .collect();

                        // Clean up Prometheus metrics for removed workers
                        for worker_id in &removed_workers {
                            let dp_ranks =
                                expected_worker_dp_ranks(&worker_load_states, *worker_id);
                            // Clean up metrics for both worker types since we don't know which type this worker was
                            cleanup_worker_metrics(*worker_id, &dp_ranks, WORKER_TYPE_DECODE);
                            cleanup_worker_metrics(*worker_id, &dp_ranks, WORKER_TYPE_PREFILL);
                            tracing::debug!(
                                "Removed Prometheus metrics for worker {}",
                                worker_id
                            );
                            forget_worker_publisher(
                                &mut decode_sequences,
                                &mut decode_publishers,
                                *worker_id,
                            );
                            forget_worker_publisher(
                                &mut prefill_sequences,
                                &mut prefill_publishers,
                                *worker_id,
                            );
                        }

                        worker_load_states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));
                        overloaded_tracker.remove_workers(&removed_workers);
                        client.clear_overloaded_instances_for_removed(&removed_workers);
                        // Mirror the prune to the prefill Client (disagg). Prefill workers are
                        // routed by a separate PrefillRouter with its own Client, so its
                        // overloaded set must be cleared too or removed prefill ids would
                        // linger as phantom-overloaded entries.
                        if let Some(prefill_client) = prefill_client_holder.read().unwrap().clone() {
                            prefill_client.clear_overloaded_instances_for_removed(&removed_workers);
                        }

                        // Update worker load states with runtime config values for all dp_ranks
                        // This ensures we track workers from MDCs even if they don't publish ActiveLoad
                        for (lease_id, runtime_config) in runtime_configs.iter() {
                            let mut state = worker_load_states.entry(*lease_id).or_default();

                            let dp_start = runtime_config.data_parallel_start_rank;
                            let dp_end = dp_start + runtime_config.data_parallel_size;
                            let expected_dp_ranks = (dp_start..dp_end).collect::<HashSet<_>>();

                            state
                                .active_decode_blocks
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state
                                .active_decode_observed_at
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state
                                .kv_used_blocks
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state
                                .kv_used_observed_at
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state
                                .active_prefill_tokens
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state
                                .active_prefill_observed_at
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state
                                .decode_overload_latches
                                .retain(|rank, _| expected_dp_ranks.contains(rank));
                            state.expected_dp_ranks = expected_dp_ranks;

                            // Populate total_blocks for all dp_ranks (they share the same total)
                            state.kv_total_blocks.clear();
                            if let Some(total_blocks) = runtime_config.total_kv_blocks {
                                for dp_rank in dp_start..dp_end {
                                    state.kv_total_blocks.insert(dp_rank, total_blocks);
                                }
                            }

                            // Populate max_num_batched_tokens for all dp_ranks
                            state.max_num_batched_tokens.clear();
                            if let Some(max_batched) = runtime_config.max_num_batched_tokens {
                                for dp_rank in dp_start..dp_end {
                                    state.max_num_batched_tokens.insert(dp_rank, max_batched);
                                }
                            }
                        }

                        let cfg = thresholds.read().unwrap().clone();
                        last_thresholds = cfg.clone();
                        let overloaded_workers = collect_overloaded_workers(&worker_load_states, &cfg);
                        if overloaded_tracker.replace(overloaded_workers) {
                            let overloaded_instances = overloaded_tracker.ids();
                            publish_overloaded_instances(
                                &client,
                                &prefill_client_holder,
                                &overloaded_instances,
                            );
                        }
                    }

                    // Handle KV metrics updates (ActiveLoad) - only if subscriber is available
                    // Note: Prometheus gauges are updated directly by sequence.rs (router's own bookkeeping)
                    // This branch only updates WorkerLoadState for overload detection thresholds.
                    (prefill_scope, kv_event) = kv_event_future => {
                        let Some(event_result) = kv_event else {
                            let recovery_after = Instant::now();
                            if prefill_scope {
                                prefill_metrics_rx = None;
                                let mut source = sources.write().unwrap();
                                source.prefill_metrics_healthy = false;
                                source.prefill_recovery_after = Some(recovery_after);
                                tracing::debug!("prefill KV metrics stream closed");
                            } else {
                                kv_metrics_rx = None;
                                let mut source = sources.write().unwrap();
                                source.decode_metrics_healthy = false;
                                source.decode_recovery_after = Some(recovery_after);
                                tracing::debug!("decode KV metrics stream closed");
                            }
                            continue;
                        };

                        let Ok((envelope, active_load)) = event_result else {
                            let recovery_after = Instant::now();
                            if prefill_scope {
                                let mut source = sources.write().unwrap();
                                source.prefill_metrics_healthy = false;
                                source.prefill_recovery_after = Some(recovery_after);
                            } else {
                                let mut source = sources.write().unwrap();
                                source.decode_metrics_healthy = false;
                                source.decode_recovery_after = Some(recovery_after);
                            }
                            tracing::error!("Error receiving KV metrics event: {event_result:?}");
                            continue;
                        };

                        let worker_id = active_load.worker_id;
                        let dp_rank = active_load.dp_rank;

                        let endpoint_role = if prefill_scope { "prefill" } else { "decode" };
                        let membership = {
                            let source_state = sources.read().unwrap();
                            if prefill_scope {
                                classify_load_membership(
                                    worker_id,
                                    &source_state.prefill_workers,
                                    &source_state.decode_workers,
                                )
                            } else {
                                classify_load_membership(
                                    worker_id,
                                    &source_state.decode_workers,
                                    &source_state.prefill_workers,
                                )
                            }
                        };

                        match membership {
                            LoadMembership::Unknown => {
                                tracing::debug!(
                                    worker_id,
                                    dp_rank,
                                    endpoint_role,
                                    "dropping load event until endpoint membership is discovered"
                                );
                                continue;
                            }
                            LoadMembership::Foreign => {
                                tracing::warn!(
                                    worker_id,
                                    dp_rank,
                                    endpoint_role,
                                    "ignoring load event for worker owned by a different endpoint"
                                );
                                continue;
                            }
                            LoadMembership::Ambiguous => {
                                if let Some(mut state) = worker_load_states.get_mut(&worker_id) {
                                    state.clear_observations();
                                }
                                if overloaded_tracker.update_worker(worker_id, false) {
                                    let overloaded_instances = overloaded_tracker.ids();
                                    publish_overloaded_instances(
                                        &client,
                                        &prefill_client_holder,
                                        &overloaded_instances,
                                    );
                                }
                                tracing::error!(
                                    worker_id,
                                    dp_rank,
                                    "worker is registered in multiple cache-owning endpoints; ignoring ambiguous load event"
                                );
                                continue;
                            }
                            LoadMembership::Exact => {}
                        }
                        if !worker_load_states
                            .get(&worker_id)
                            .is_some_and(|state| state.expected_dp_ranks.contains(&dp_rank))
                        {
                            tracing::debug!(
                                worker_id,
                                dp_rank,
                                endpoint_role,
                                "dropping load event outside the current runtime-config rank set"
                            );
                            continue;
                        }

                        let observed_at = Instant::now();
                        let (sequences, publishers) = if prefill_scope {
                            (&mut prefill_sequences, &mut prefill_publishers)
                        } else {
                            (&mut decode_sequences, &mut decode_publishers)
                        };
                        let gap = match sequences.entry(envelope.publisher_id) {
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                entry.insert(envelope.sequence);
                                false
                            }
                            std::collections::hash_map::Entry::Occupied(mut entry) => {
                                let previous = *entry.get();
                                if envelope.sequence <= previous {
                                    continue;
                                }
                                entry.insert(envelope.sequence);
                                envelope.sequence > previous.saturating_add(1)
                            }
                        };
                        let publisher_changed = set_worker_publisher(
                            sequences,
                            publishers,
                            worker_id,
                            envelope.publisher_id,
                        );
                        {
                            let mut source_state = sources.write().unwrap();
                            if prefill_scope {
                                source_state.prefill_metrics_healthy = true;
                                if gap || publisher_changed {
                                    source_state.prefill_recovery_after = Some(observed_at);
                                }
                            } else {
                                source_state.decode_metrics_healthy = true;
                                if gap || publisher_changed {
                                    source_state.decode_recovery_after = Some(observed_at);
                                }
                            }
                        }
                        if gap || publisher_changed {
                            tracing::warn!(
                                publisher_id = envelope.publisher_id,
                                sequence = envelope.sequence,
                                endpoint_role = if prefill_scope { "prefill" } else { "decode" },
                                "KV metrics source changed or skipped events; waiting for fresh rank snapshots"
                            );
                        }

                        // Snapshot thresholds once per event — rare writes (HTTP endpoint)
                        // mean RwLock contention is effectively zero.
                        let cfg = thresholds.read().unwrap().clone();
                        let thresholds_changed = cfg != last_thresholds;

                        // Update worker load state per dp_rank (for overload detection only).
                        // Note: Prometheus gauges are updated directly by sequence.rs
                        let (total_blocks, worker_overloaded) = {
                            let mut state = worker_load_states.entry(worker_id).or_default();
                            state.update_from_active_load_at(
                                &active_load,
                                cfg.active_decode_blocks_threshold,
                                observed_at,
                            );
                            let total_blocks = state.kv_total_blocks.get(&dp_rank).copied();
                            let worker_overloaded = state.is_overloaded_for_config(&cfg);
                            (total_blocks, worker_overloaded)
                        };

                        if tracing::enabled!(tracing::Level::DEBUG) {
                            tracing::debug!(
                                worker_id,
                                dp_rank,
                                active_decode_blocks = ?active_load.active_decode_blocks,
                                kv_used_blocks = ?active_load.kv_used_blocks,
                                active_prefill_tokens = ?active_load.active_prefill_tokens,
                                total_blocks = ?total_blocks,
                                active_decode_blocks_threshold = ?cfg.active_decode_blocks_threshold,
                                active_prefill_tokens_threshold = ?cfg.active_prefill_tokens_threshold,
                                active_prefill_tokens_threshold_frac = ?cfg.active_prefill_tokens_threshold_frac,
                                worker_overloaded,
                                "processed active load update"
                            );
                        }

                        // Recompute the full overloaded set only when thresholds change;
                        // otherwise incrementally update just this worker. When the set
                        // changes, publish to both the decode Client and (in disaggregated
                        // serving) the prefill Client — see `publish_overloaded_instances`.
                        let overloaded_changed = if thresholds_changed {
                            last_thresholds = cfg.clone();
                            let overloaded_workers =
                                collect_overloaded_workers(&worker_load_states, &cfg);
                            overloaded_tracker.replace(overloaded_workers)
                        } else {
                            overloaded_tracker.update_worker(worker_id, worker_overloaded)
                        };

                        publish_overloaded_instances_if_needed(
                            &client,
                            &prefill_client_holder,
                            &overloaded_tracker,
                            overloaded_changed,
                        );
                    }

                    // Handle decode endpoint instance changes (for ITL and decode metrics cleanup)
                    result = decode_instances_rx.changed() => {
                        if result.is_err() {
                            sources.write().unwrap().decode_workers.clear();
                            tracing::info!("decode endpoint watcher closed");
                            break;
                        }
                        let current_instances: std::collections::HashSet<u64> =
                            decode_instances_rx.borrow().iter().copied().collect();

                        // Find decode workers that disappeared
                        let removed_workers: Vec<u64> = {
                            let source_state = sources.read().unwrap();
                            source_state
                                .decode_workers
                                .difference(&current_instances)
                                .copied()
                                .collect()
                        };

                        if !removed_workers.is_empty() {
                            // Clean up metrics for removed decode workers (with worker_type=decode label)
                            for worker_id in &removed_workers {
                                let dp_ranks =
                                    expected_worker_dp_ranks(&worker_load_states, *worker_id);
                                cleanup_worker_metrics(*worker_id, &dp_ranks, WORKER_TYPE_DECODE);
                                tracing::debug!(
                                    "Cleaned up metrics for removed decode worker {}",
                                    worker_id
                                );
                                forget_worker_publisher(
                                    &mut decode_sequences,
                                    &mut decode_publishers,
                                    *worker_id,
                                );
                                if let Some(mut state) = worker_load_states.get_mut(worker_id) {
                                    state.clear_observations();
                                }
                            }
                            overloaded_tracker.remove_workers(&removed_workers);
                            client.clear_overloaded_instances_for_removed(&removed_workers);
                        }

                        sources.write().unwrap().decode_workers = current_instances;
                    }

                    // Handle prefill endpoint instance changes (for TTFT and prefill metrics cleanup in disaggregated mode)
                    result = async {
                        if let Some(ref mut rx) = prefill_instances_rx {
                            rx.changed().await
                        } else {
                            // No prefill watcher yet, pend forever
                            std::future::pending().await
                        }
                    } => {
                        // Handle channel closure (e.g., all prefill workers went down)
                        let Ok(()) = result else {
                            // Prefill endpoint closed - stop watching to avoid busy loop
                            prefill_instances_rx = None;
                            sources.write().unwrap().prefill_workers.clear();
                            tracing::info!("Prefill endpoint watcher closed, will re-activate when client is set");
                            continue;
                        };

                        let Some(ref rx) = prefill_instances_rx else {
                            continue;
                        };

                        let current_instances: std::collections::HashSet<u64> =
                            rx.borrow().iter().copied().collect();

                        // Find prefill workers that disappeared
                        let removed_workers: Vec<u64> = {
                            let source_state = sources.read().unwrap();
                            source_state
                                .prefill_workers
                                .difference(&current_instances)
                                .copied()
                                .collect()
                        };

                        if !removed_workers.is_empty() {
                            // Clean up metrics for removed prefill workers (with worker_type=prefill label)
                            for worker_id in &removed_workers {
                                let dp_ranks =
                                    expected_worker_dp_ranks(&worker_load_states, *worker_id);
                                cleanup_worker_metrics(*worker_id, &dp_ranks, WORKER_TYPE_PREFILL);
                                tracing::debug!(
                                    "Cleaned up metrics for removed prefill worker {}",
                                    worker_id
                                );
                                forget_worker_publisher(
                                    &mut prefill_sequences,
                                    &mut prefill_publishers,
                                    *worker_id,
                                );
                                if let Some(mut state) = worker_load_states.get_mut(worker_id) {
                                    state.clear_observations();
                                }
                            }
                            overloaded_tracker.remove_workers(&removed_workers);
                            client.clear_overloaded_instances_for_removed(&removed_workers);
                        }

                        sources.write().unwrap().prefill_workers = current_instances;
                    }

                    // Wait for prefill client to be registered (push-based notification)
                    _ = prefill_client_notify.notified() => {
                        let prefill_client = prefill_client_holder.read().unwrap().clone();
                        if let Some(prefill_client) = prefill_client {
                            let prefill_endpoint = prefill_client.endpoint.clone();
                            let rx = prefill_client.instance_avail_watcher();
                            let prefill_workers = rx.borrow().iter().copied().collect::<HashSet<_>>();
                            let prefill_worker_count = prefill_workers.len();
                            prefill_instances_rx = Some(rx);

                            let metrics_result = tokio::select! {
                                _ = cancellation_token.cancelled() => break,
                                result = EventSubscriber::for_endpoint(
                                    &prefill_endpoint,
                                    KV_METRICS_SUBJECT,
                                ) => result,
                            };
                            prefill_metrics_rx = match metrics_result {
                                Ok(subscriber) => {
                                    sources.write().unwrap().prefill_metrics_healthy = true;
                                    Some(subscriber.typed::<ActiveLoad>())
                                }
                                Err(error) => {
                                    sources.write().unwrap().prefill_metrics_healthy = false;
                                    tracing::warn!(
                                        endpoint = %prefill_endpoint.id(),
                                        %error,
                                        "KvWorkerMonitor: prefill KV metrics subscriber not available"
                                    );
                                    None
                                }
                            };
                            let config_result = tokio::select! {
                                _ = cancellation_token.cancelled() => break,
                                result = runtime_config_watch(&prefill_endpoint, cancellation_token.clone()) => result,
                            };
                            prefill_configs_rx = match config_result {
                                Ok(mut rx) => {
                                    rx.mark_changed();
                                    sources.write().unwrap().prefill_configs_healthy = true;
                                    Some(rx)
                                }
                                Err(error) => {
                                    sources.write().unwrap().prefill_configs_healthy = false;
                                    tracing::warn!(
                                        endpoint = %prefill_endpoint.id(),
                                        %error,
                                        "KvWorkerMonitor: prefill runtime-config watch not available"
                                    );
                                    None
                                }
                            };
                            {
                                let mut source_state = sources.write().unwrap();
                                source_state.prefill_endpoint = Some(prefill_endpoint.id());
                                source_state.prefill_workers = prefill_workers;
                                source_state.prefill_recovery_after = Some(Instant::now());
                            }
                            tracing::info!(
                                endpoint = %prefill_endpoint.id(),
                                "KvWorkerMonitor: prefill endpoint watcher activated, tracking {} workers",
                                prefill_worker_count
                            );

                            // Seed the freshly-registered prefill Client with the current
                            // overloaded set. The prefill router can activate after KV events
                            // have already been processed; without this seed the prefill pool
                            // would not learn about already-overloaded workers until the next
                            // KV event arrives.
                            let cfg = thresholds.read().unwrap().clone();
                            let overloaded_instances =
                                compute_overloaded_instances(&worker_load_states, &cfg);
                            prefill_client.set_overloaded_instances(&overloaded_instances);
                        }
                    }
                }
            }

            tracing::info!("Worker monitoring task exiting");
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LoadMembership, LoadThresholdConfig, OverloadedWorkerTracker, WorkerLoadState,
        classify_load_membership, compute_overloaded_instances, forget_worker_publisher,
        overload_reconciliation_needed, pool_snapshot, publish_overloaded_instances,
        publish_overloaded_instances_if_needed, set_worker_publisher,
    };
    use dashmap::DashMap;
    use dynamo_kv_router::protocols::ActiveLoad;
    use dynamo_runtime::protocols::EndpointId;
    use std::collections::{HashMap, HashSet};
    use std::time::{Duration, Instant};

    use crate::worker_type::WorkerType;

    fn endpoint() -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "decode".to_string(),
            name: "generate".to_string(),
        }
    }

    #[test]
    fn kv_pool_snapshot_requires_fresh_capacity_and_occupancy_for_every_rank() {
        let states = DashMap::new();
        let now = Instant::now();
        let mut state = WorkerLoadState::default();
        state.expected_dp_ranks.extend([0, 1]);
        state.kv_total_blocks.extend([(0, 100), (1, 100)]);
        state.kv_used_blocks.extend([(0, 25), (1, 50)]);
        state.kv_used_observed_at.extend([(0, now), (1, now)]);
        states.insert(7, state);

        let snapshot = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            None,
            now,
        );
        assert!(snapshot.complete);
        assert_eq!(snapshot.expected_ranks, 2);
        assert_eq!(snapshot.observed_ranks, 2);
        assert_eq!(snapshot.capacity_blocks, Some(200));
        assert_eq!(snapshot.used_blocks, Some(75));
        assert_eq!(snapshot.free_blocks, Some(125));

        let stale = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            None,
            now + super::KV_LOAD_FRESHNESS + Duration::from_nanos(1),
        );
        assert!(!stale.complete);
        assert_eq!(stale.observed_ranks, 0);

        let unhealthy = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            false,
            None,
            now,
        );
        assert!(!unhealthy.complete);

        let mut second = WorkerLoadState::default();
        second.expected_dp_ranks.insert(2);
        second.kv_total_blocks.insert(2, 50);
        second.kv_used_blocks.insert(2, 10);
        second.kv_used_observed_at.insert(2, now);
        states.insert(8, second);
        let multiple_workers = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7, 8]),
            true,
            None,
            now,
        );
        assert!(multiple_workers.complete);
        assert_eq!(multiple_workers.expected_ranks, 3);
        assert_eq!(multiple_workers.capacity_blocks, Some(250));
        assert_eq!(multiple_workers.used_blocks, Some(85));
        states.remove(&8);

        states.get_mut(&7).unwrap().kv_used_blocks.remove(&1);
        let incomplete = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            None,
            now,
        );
        assert!(!incomplete.complete);
        assert_eq!(incomplete.observed_ranks, 1);
        assert_eq!(incomplete.free_blocks, None);

        let recovery = now + Duration::from_nanos(1);
        {
            let mut state = states.get_mut(&7).unwrap();
            state.kv_used_blocks.insert(1, 50);
        }
        let pre_recovery = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            Some(recovery),
            recovery,
        );
        assert!(!pre_recovery.complete);

        states.get_mut(&7).unwrap().kv_used_observed_at.extend([
            (0, recovery + Duration::from_nanos(1)),
            (1, recovery + Duration::from_nanos(1)),
        ]);
        let recovered_at = recovery + Duration::from_nanos(1);
        let recovered = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            Some(recovery),
            recovered_at,
        );
        assert!(recovered.complete);
    }

    #[test]
    fn kv_pool_snapshot_omits_partial_or_stale_optional_load() {
        let states = DashMap::new();
        let now = Instant::now();
        let mut state = WorkerLoadState::default();
        state.expected_dp_ranks.extend([0, 1]);
        state.kv_total_blocks.extend([(0, 100), (1, 100)]);
        state.kv_used_blocks.extend([(0, 25), (1, 50)]);
        state.kv_used_observed_at.extend([(0, now), (1, now)]);
        state.active_decode_blocks.extend([(0, 10), (1, 20)]);
        state.active_decode_observed_at.extend([(0, now), (1, now)]);
        state.active_prefill_tokens.insert(0, 30);
        state.active_prefill_observed_at.insert(0, now);
        states.insert(7, state);

        let partial = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            None,
            now,
        );
        assert!(partial.complete);
        assert_eq!(partial.active_decode_blocks, Some(30));
        assert_eq!(partial.active_prefill_tokens, None);

        {
            let mut state = states.get_mut(&7).unwrap();
            state.active_prefill_tokens.insert(1, 40);
            state.active_prefill_observed_at.insert(1, now);
            state.active_decode_observed_at.insert(
                1,
                now.checked_sub(super::KV_LOAD_FRESHNESS + Duration::from_nanos(1))
                    .unwrap(),
            );
        }
        let stale = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            None,
            now,
        );
        assert!(stale.complete);
        assert_eq!(stale.active_decode_blocks, None);
        assert_eq!(stale.active_prefill_tokens, Some(70));
    }

    #[test]
    fn clearing_worker_observations_preserves_config_but_requires_fresh_occupancy() {
        let now = Instant::now();
        let mut state = WorkerLoadState::default();
        state.expected_dp_ranks.insert(0);
        state.kv_total_blocks.insert(0, 100);
        state.max_num_batched_tokens.insert(0, 4096);
        state.kv_used_blocks.insert(0, 25);
        state.kv_used_observed_at.insert(0, now);
        state.active_decode_blocks.insert(0, 10);
        state.active_decode_observed_at.insert(0, now);
        state.active_prefill_tokens.insert(0, 20);
        state.active_prefill_observed_at.insert(0, now);

        state.clear_observations();

        assert_eq!(state.expected_dp_ranks, HashSet::from([0]));
        assert_eq!(state.kv_total_blocks.get(&0), Some(&100));
        assert_eq!(state.max_num_batched_tokens.get(&0), Some(&4096));
        assert!(state.kv_used_blocks.is_empty());
        assert!(state.kv_used_observed_at.is_empty());
        assert!(state.active_decode_blocks.is_empty());
        assert!(state.active_decode_observed_at.is_empty());
        assert!(state.active_prefill_tokens.is_empty());
        assert!(state.active_prefill_observed_at.is_empty());

        let states = DashMap::from_iter([(7, state)]);
        let snapshot = pool_snapshot(
            &states,
            endpoint(),
            WorkerType::Decode,
            &HashSet::from([7]),
            true,
            None,
            now,
        );
        assert!(!snapshot.complete);
        assert_eq!(snapshot.expected_ranks, 1);
        assert_eq!(snapshot.observed_ranks, 0);
        assert_eq!(snapshot.capacity_blocks, Some(100));
        assert_eq!(snapshot.used_blocks, None);
        assert_eq!(snapshot.free_blocks, None);
    }

    #[test]
    fn overloaded_worker_tracker_updates_one_worker() {
        let mut tracker = OverloadedWorkerTracker::default();

        assert!(tracker.update_worker(7, true));
        assert!(tracker.contains(7));
        assert!(!tracker.update_worker(7, true));

        assert!(tracker.update_worker(7, false));
        assert!(!tracker.contains(7));
        assert!(!tracker.update_worker(7, false));
    }

    #[test]
    fn publisher_sequence_is_retired_after_its_last_worker_moves() {
        let mut sequences = HashMap::from([(10, 5), (20, 1)]);
        let mut publishers = HashMap::from([(7, 10), (8, 10)]);

        assert!(set_worker_publisher(&mut sequences, &mut publishers, 7, 20,));
        assert!(sequences.contains_key(&10));

        forget_worker_publisher(&mut sequences, &mut publishers, 8);
        assert!(!sequences.contains_key(&10));
        assert_eq!(publishers, HashMap::from([(7, 20)]));
    }

    #[test]
    fn load_membership_requires_the_exact_source_endpoint() {
        assert_eq!(
            classify_load_membership(7, &HashSet::from([7]), &HashSet::new()),
            LoadMembership::Exact
        );
        assert_eq!(
            classify_load_membership(7, &HashSet::new(), &HashSet::new()),
            LoadMembership::Unknown
        );
        assert_eq!(
            classify_load_membership(7, &HashSet::new(), &HashSet::from([7])),
            LoadMembership::Foreign
        );
    }

    #[test]
    fn load_membership_rejects_ambiguous_endpoint_ownership() {
        assert_eq!(
            classify_load_membership(7, &HashSet::from([7]), &HashSet::from([7])),
            LoadMembership::Ambiguous
        );
    }

    #[test]
    fn overloaded_worker_tracker_replaces_and_removes_workers() {
        let mut tracker = OverloadedWorkerTracker::default();

        assert!(tracker.replace(HashSet::from([1, 3, 5])));
        assert!(!tracker.replace(HashSet::from([1, 3, 5])));

        assert!(tracker.remove_workers(&[3, 5]));
        assert!(tracker.contains(1));
        assert!(!tracker.contains(3));
        assert!(!tracker.contains(5));
        assert!(
            tracker.update_worker(3, true),
            "rejoined overloaded workers must be republished after removal"
        );
        assert!(tracker.contains(3));

        assert!(!tracker.remove_workers(&[2, 4]));
    }

    #[test]
    fn load_threshold_config_default_is_not_configured() {
        let config = LoadThresholdConfig::default();
        assert!(!config.is_configured());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn load_threshold_config_validates_decode_fraction() {
        for threshold in [0.0, 0.85, 1.0] {
            let config = LoadThresholdConfig {
                active_decode_blocks_threshold: Some(threshold),
                ..Default::default()
            };
            assert!(config.validate().is_ok(), "threshold={threshold}");
        }

        for threshold in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let config = LoadThresholdConfig {
                active_decode_blocks_threshold: Some(threshold),
                ..Default::default()
            };
            let error = config.validate().unwrap_err();
            assert!(
                error.contains("active_decode_blocks_threshold"),
                "threshold={threshold}, error={error}"
            );
        }
    }

    #[test]
    fn load_threshold_config_validates_prefill_fraction() {
        for threshold in [0.0, 0.9, 64.0] {
            let config = LoadThresholdConfig {
                active_prefill_tokens_threshold_frac: Some(threshold),
                ..Default::default()
            };
            assert!(config.validate().is_ok(), "threshold={threshold}");
        }

        for threshold in [-0.1, f64::NAN, f64::INFINITY] {
            let config = LoadThresholdConfig {
                active_prefill_tokens_threshold_frac: Some(threshold),
                ..Default::default()
            };
            let error = config.validate().unwrap_err();
            assert!(
                error.contains("active_prefill_tokens_threshold_frac"),
                "threshold={threshold}, error={error}"
            );
        }
    }

    #[test]
    fn load_threshold_config_decode_only_is_configured() {
        let config = LoadThresholdConfig {
            active_decode_blocks_threshold: Some(0.85),
            ..Default::default()
        };
        assert!(config.is_configured());
    }

    #[test]
    fn load_threshold_config_prefill_tokens_only_is_configured() {
        let config = LoadThresholdConfig {
            active_prefill_tokens_threshold: Some(10_000),
            ..Default::default()
        };
        assert!(config.is_configured());
    }

    #[test]
    fn load_threshold_config_prefill_frac_only_is_configured() {
        let config = LoadThresholdConfig {
            active_prefill_tokens_threshold_frac: Some(0.9),
            ..Default::default()
        };
        assert!(config.is_configured());
    }

    #[test]
    fn load_threshold_config_all_set_is_configured() {
        let config = LoadThresholdConfig {
            active_decode_blocks_threshold: Some(0.85),
            active_prefill_tokens_threshold: Some(10_000),
            active_prefill_tokens_threshold_frac: Some(0.9),
        };
        assert!(config.is_configured());
    }

    #[test]
    fn is_overloaded_prefers_kv_used_blocks_over_active_decode_blocks() {
        let mut state = WorkerLoadState::default();
        state.active_decode_blocks.insert(0, 10);
        state.kv_used_blocks.insert(0, 90);
        state.kv_total_blocks.insert(0, 100);

        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn is_overloaded_falls_back_to_active_decode_blocks_when_kv_used_missing() {
        let mut state = WorkerLoadState::default();
        state.active_decode_blocks.insert(0, 90);
        state.kv_total_blocks.insert(0, 100);

        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn is_overloaded_recognizes_dp_rank_known_only_from_kv_used_blocks() {
        let mut state = WorkerLoadState::default();
        state.kv_used_blocks.insert(0, 90);
        state.kv_total_blocks.insert(0, 100);

        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_latch_sets_overloaded_if_any_signal_is_overloaded() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
        );

        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_latch_only_clears_after_both_signals_report_not_overloaded() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
        );
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: None,
            },
            Some(0.6),
        );
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(10),
            },
            Some(0.6),
        );
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_latch_clears_with_only_kv_used_blocks_signal() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
        );
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(10),
            },
            Some(0.6),
        );
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_latch_clears_with_only_active_decode_blocks_signal() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(90),
                active_prefill_tokens: None,
                kv_used_blocks: None,
            },
            Some(0.6),
        );
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: None,
            },
            Some(0.6),
        );
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_latch_clears_when_both_signals_are_not_overloaded_in_same_event() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(90),
                active_prefill_tokens: None,
                kv_used_blocks: None,
            },
            Some(0.6),
        );
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: Some(10),
            },
            Some(0.6),
        );
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn is_overloaded_returns_false_when_all_thresholds_are_none() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);
        state.active_decode_blocks.insert(0, 99);
        state.kv_used_blocks.insert(0, 99);
        state.active_prefill_tokens.insert(0, u64::MAX / 2);
        state.max_num_batched_tokens.insert(0, 1_000);

        assert!(!state.is_overloaded(None, None, None));
    }

    #[test]
    fn is_overloaded_with_only_decode_threshold_ignores_prefill_signals() {
        let mut state = WorkerLoadState::default();
        state.max_num_batched_tokens.insert(0, 1_000);
        state.active_prefill_tokens.insert(0, 5_000);

        assert!(!state.is_overloaded(Some(0.6), None, None));
    }

    #[test]
    fn is_overloaded_with_only_prefill_abs_ignores_decode_latch() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(90),
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
        );

        assert!(!state.is_overloaded(None, Some(u64::MAX), None));
    }

    #[test]
    fn is_overloaded_with_only_prefill_frac_ignores_decode_latch() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(90),
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
        );

        assert!(!state.is_overloaded(None, None, Some(2.0)));
    }

    #[test]
    fn is_overloaded_with_only_prefill_abs_fires_when_tokens_exceed_threshold() {
        let mut state = WorkerLoadState::default();
        state.active_prefill_tokens.insert(0, 5_000);

        assert!(state.is_overloaded(None, Some(1_000), None));
    }

    #[test]
    fn is_overloaded_with_only_prefill_frac_fires_when_fraction_exceeded() {
        let mut state = WorkerLoadState::default();
        state.max_num_batched_tokens.insert(0, 1_000);
        state.active_prefill_tokens.insert(0, 2_500);

        assert!(state.is_overloaded(None, None, Some(2.0)));
    }

    #[test]
    fn compute_overloaded_instances_flags_prefill_workers_over_token_threshold() {
        use dashmap::DashMap;
        use std::collections::HashSet;

        let states = DashMap::new();

        // Prefill worker far over the prefill-token threshold.
        let mut prefill = WorkerLoadState::default();
        prefill.active_prefill_tokens.insert(0, 300_000);
        states.insert(1u64, prefill);

        // Prefill worker under the threshold — must not be flagged.
        let mut quiet = WorkerLoadState::default();
        quiet.active_prefill_tokens.insert(0, 100);
        states.insert(2u64, quiet);

        let cfg = LoadThresholdConfig {
            active_prefill_tokens_threshold: Some(5_000),
            ..Default::default()
        };

        let overloaded: HashSet<u64> = compute_overloaded_instances(&states, &cfg)
            .into_iter()
            .collect();
        assert_eq!(overloaded, HashSet::from([1]));
    }

    #[tokio::test]
    async fn unchanged_low_metric_reconciles_request_path_overload() {
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
        use std::sync::RwLock;

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let client = drt
            .namespace("test_request_path_overload_reconciliation".to_string())
            .unwrap()
            .component("test_component".to_string())
            .unwrap()
            .endpoint("decode".to_string())
            .client()
            .await
            .unwrap();
        let prefill_client_holder = RwLock::new(None);
        let mut tracker = OverloadedWorkerTracker::default();

        assert!(!tracker.update_worker(7, false));
        client.mark_overloaded_immediate(7);
        assert_eq!(client.overloaded_instance_ids(), Some(HashSet::from([7])));

        let overloaded_changed = tracker.update_worker(7, false);
        assert!(
            !overloaded_changed,
            "the monitor's cached set remains empty"
        );
        assert!(overload_reconciliation_needed(
            &client,
            &prefill_client_holder
        ));

        assert!(publish_overloaded_instances_if_needed(
            &client,
            &prefill_client_holder,
            &tracker,
            overloaded_changed,
        ));

        assert_eq!(client.overloaded_instance_ids(), None);
        assert!(!client.overload_reconciliation_needed());
        rt.shutdown();
    }

    /// Regression: the overloaded set must reach the prefill
    /// router's Client, not only the decode/main router's Client. Without the
    /// prefill propagation, `--active-prefill-tokens-threshold` is a silent
    /// no-op in disaggregated serving.
    #[tokio::test]
    async fn publish_overloaded_instances_reaches_registered_prefill_client() {
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
        use std::collections::HashSet;
        use std::sync::RwLock;

        let rt = Runtime::from_current().unwrap();
        // process_local avoids needing etcd/nats.
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_prefill_overload_propagation".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();

        let decode_client = component
            .endpoint("decode".to_string())
            .client()
            .await
            .unwrap();
        let prefill_client = component
            .endpoint("prefill".to_string())
            .client()
            .await
            .unwrap();

        let holder: RwLock<Option<_>> = RwLock::new(None);

        // Before the prefill client is registered, only the decode client is updated.
        publish_overloaded_instances(&decode_client, &holder, &[1, 2]);
        assert_eq!(
            decode_client.overloaded_instance_ids(),
            Some(HashSet::from([1, 2]))
        );
        assert_eq!(prefill_client.overloaded_instance_ids(), None);

        // Once registered (as happens via attach_prefill_client on prefill router
        // activation), the prefill client must receive the same set.
        *holder.write().unwrap() = Some(prefill_client.clone());
        publish_overloaded_instances(&decode_client, &holder, &[1, 2]);
        assert_eq!(
            prefill_client.overloaded_instance_ids(),
            Some(HashSet::from([1, 2]))
        );

        rt.shutdown();
    }

    /// Late attachment: if prefill workers are already overloaded when the prefill
    /// router activates, `attach_prefill_client` must seed the new Client with the
    /// current overloaded set synchronously (not wait for the monitor loop), so the
    /// attach->seed window cannot admit requests it should shed.
    #[tokio::test]
    async fn attach_prefill_client_synchronously_seeds_overloaded_set() {
        use super::KvWorkerMonitor;
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
        use std::collections::HashSet;

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let component = drt
            .namespace("test_attach_seed".to_string())
            .unwrap()
            .component("test_component".to_string())
            .unwrap();
        let decode_client = component
            .endpoint("decode".to_string())
            .client()
            .await
            .unwrap();
        let prefill_client = component
            .endpoint("prefill".to_string())
            .client()
            .await
            .unwrap();

        let monitor = KvWorkerMonitor::new(
            decode_client,
            LoadThresholdConfig {
                active_prefill_tokens_threshold: Some(5_000),
                ..Default::default()
            },
        );

        // A prefill worker already over the token threshold, recorded before any
        // prefill client is attached and without the monitor loop running.
        monitor
            .worker_load_states
            .entry(7)
            .or_default()
            .active_prefill_tokens
            .insert(0, 10_000);

        monitor.attach_prefill_client(prefill_client.clone());
        assert_eq!(
            prefill_client.overloaded_instance_ids(),
            Some(HashSet::from([7])),
            "attach must seed the prefill client with the current overloaded set"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn dropping_last_monitor_releases_task_state() {
        use super::KvWorkerMonitor;
        use dynamo_runtime::pipeline::WorkerLoadMonitor;
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
        use std::sync::Arc;

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let client = drt
            .namespace("test_monitor_lifecycle".to_string())
            .unwrap()
            .component("test_component".to_string())
            .unwrap()
            .endpoint("decode".to_string())
            .client()
            .await
            .unwrap();
        let monitor = KvWorkerMonitor::new(client, LoadThresholdConfig::default());
        let monitor_clone = monitor.clone();
        let worker_load_states = Arc::downgrade(&monitor.worker_load_states);

        let (first_start, second_start) =
            tokio::join!(monitor.start_monitoring(), monitor_clone.start_monitoring());
        first_start.unwrap();
        second_start.unwrap();
        drop(monitor);
        assert!(
            !monitor_clone.lifecycle.cancellation_token.is_cancelled()
                && worker_load_states.upgrade().is_some(),
            "dropping one clone must not stop a shared monitor"
        );
        drop(monitor_clone);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while worker_load_states.strong_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("monitor task retained state after its last owner was dropped");

        rt.shutdown();
    }
}
