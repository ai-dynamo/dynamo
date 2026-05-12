// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

use dashmap::DashMap;
use dynamo_kv_router::protocols::ActiveLoad;
use serde::{Deserialize, Serialize};

use crate::http::service::metrics::{
    WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE, WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE,
    WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE,
};
use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::metrics::WORKER_LOAD_METRICS;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::component::Client;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;

// Re-export worker type constants from timing.rs (single source of truth)
pub use crate::protocols::common::timing::{WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL};
const UNSET_DP_RANK_LABEL: &str = "none";

/// Clean up all Prometheus metrics for a worker across the specified dp_ranks.
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

/// Default value for `max_num_batched_tokens` when the runtime config does not
/// report it. Set high enough that the frac-based busy check (which multiplies
/// this value by the threshold fraction) can never fire with realistic loads.
const DEFAULT_MAX_TOKENS: u64 = 10_000_000;

/// Configuration for worker load thresholds used in busy detection.
///
/// All thresholds are opt-in. An unset (`None`) field means the corresponding
/// check is skipped entirely — it never contributes to a worker being marked
/// busy. If all three are `None`, busy-based rejection is fully disabled.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LoadThresholdConfig {
    /// KV cache block utilization threshold (0.0-1.0).
    /// Worker is busy when `active_decode_blocks / total_blocks > threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_decode_blocks_threshold: Option<f64>,

    /// Absolute prefill token count threshold.
    /// Worker is busy when `active_prefill_tokens > threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_prefill_tokens_threshold: Option<u64>,

    /// Fraction of max_num_batched_tokens.
    /// Worker is busy when `active_prefill_tokens > frac * max_num_batched_tokens`.
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
}

/// Worker load monitoring state per dp_rank
#[derive(Clone, Debug)]
struct DecodeBusyLatchState {
    latched_busy: bool,
    kv_used_blocks_cleared: bool,
    active_decode_blocks_cleared: bool,
}

impl Default for DecodeBusyLatchState {
    fn default() -> Self {
        Self {
            latched_busy: false,
            kv_used_blocks_cleared: true,
            active_decode_blocks_cleared: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WorkerLoadState {
    pub active_decode_blocks: HashMap<u32, u64>,
    pub kv_used_blocks: HashMap<u32, u64>,
    pub kv_total_blocks: HashMap<u32, u64>,
    pub active_prefill_tokens: HashMap<u32, u64>,
    /// max_num_batched_tokens from runtime config (same for all dp_ranks)
    pub max_num_batched_tokens: HashMap<u32, u64>,
    decode_busy_latches: HashMap<u32, DecodeBusyLatchState>,
}

impl WorkerLoadState {
    fn is_decode_signal_busy(
        used_blocks: u64,
        total_blocks: u64,
        active_decode_blocks_threshold: f64,
    ) -> bool {
        total_blocks > 0
            && (used_blocks as f64) > (active_decode_blocks_threshold * total_blocks as f64)
    }

    fn current_decode_busy(&self, dp_rank: u32, active_decode_blocks_threshold: f64) -> bool {
        let Some(&total_blocks) = self.kv_total_blocks.get(&dp_rank) else {
            return false;
        };

        self.kv_used_blocks
            .get(&dp_rank)
            .is_some_and(|&used_blocks| {
                Self::is_decode_signal_busy(
                    used_blocks,
                    total_blocks,
                    active_decode_blocks_threshold,
                )
            })
            || self
                .active_decode_blocks
                .get(&dp_rank)
                .is_some_and(|&active_blocks| {
                    Self::is_decode_signal_busy(
                        active_blocks,
                        total_blocks,
                        active_decode_blocks_threshold,
                    )
                })
    }

    fn update_decode_busy_latch(
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

        let active_decode_busy = active_decode_blocks.is_some_and(|value| {
            Self::is_decode_signal_busy(value, total_blocks, active_decode_blocks_threshold)
        });
        let kv_used_busy = kv_used_blocks.is_some_and(|value| {
            Self::is_decode_signal_busy(value, total_blocks, active_decode_blocks_threshold)
        });

        let latch = self.decode_busy_latches.entry(dp_rank).or_default();
        if active_decode_busy || kv_used_busy {
            latch.latched_busy = true;
        }
        if let Some(value) = active_decode_blocks {
            latch.active_decode_blocks_cleared =
                !Self::is_decode_signal_busy(value, total_blocks, active_decode_blocks_threshold);
        }
        if let Some(value) = kv_used_blocks {
            latch.kv_used_blocks_cleared =
                !Self::is_decode_signal_busy(value, total_blocks, active_decode_blocks_threshold);
        }
        if latch.latched_busy && latch.kv_used_blocks_cleared && latch.active_decode_blocks_cleared
        {
            latch.latched_busy = false;
        }
    }

    fn update_from_active_load(
        &mut self,
        active_load: &ActiveLoad,
        active_decode_blocks_threshold: Option<f64>,
    ) {
        let dp_rank = active_load.dp_rank;
        if let Some(active_blocks) = active_load.active_decode_blocks {
            self.active_decode_blocks.insert(dp_rank, active_blocks);
        }
        if let Some(kv_used_blocks) = active_load.kv_used_blocks {
            self.kv_used_blocks.insert(dp_rank, kv_used_blocks);
        }
        if let Some(active_tokens) = active_load.active_prefill_tokens {
            self.active_prefill_tokens.insert(dp_rank, active_tokens);
        }
        if let Some(threshold) = active_decode_blocks_threshold {
            self.update_decode_busy_latch(
                dp_rank,
                active_load.active_decode_blocks,
                active_load.kv_used_blocks,
                threshold,
            );
        }
    }

    /// Returns true if ALL dp_ranks are considered busy based on the threshold logic.
    ///
    /// Each threshold is `Option<T>`. A `None` threshold means that check is
    /// skipped entirely — it cannot contribute to a dp_rank being busy. If all
    /// three thresholds are `None`, no dp_rank is ever busy.
    ///
    /// For each dp_rank, a dp_rank is busy if ANY of these conditions is met (OR logic):
    /// 1. `active_prefill_tokens > active_prefill_tokens_threshold` (absolute, if set)
    /// 2. `active_prefill_tokens > frac * max_num_batched_tokens` (fractional, if set)
    /// 3. decode busy latch set by either `kv_used_blocks` or `active_decode_blocks` (if set)
    ///
    /// The worker is busy only if ALL dp_ranks are busy.
    pub fn is_busy(
        &self,
        active_decode_blocks_threshold: Option<f64>,
        active_prefill_tokens_threshold: Option<u64>,
        active_prefill_tokens_threshold_frac: Option<f64>,
    ) -> bool {
        // Short-circuit if all thresholds are unset (i.e. no busy check can fire)
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
            .chain(self.decode_busy_latches.keys())
            .chain(self.active_prefill_tokens.keys())
            .copied()
            .collect();

        // If no dp_ranks known, not busy
        if all_dp_ranks.is_empty() {
            return false;
        }

        // Check if ALL dp_ranks are busy
        all_dp_ranks.iter().all(|&dp_rank| {
            // Check 1: prefill tokens threshold (absolute token count)
            if let Some(&active_tokens) = self.active_prefill_tokens.get(&dp_rank) {
                if let Some(abs_threshold) = active_prefill_tokens_threshold
                    && active_tokens > abs_threshold
                {
                    return true; // This dp_rank is busy due to absolute token threshold
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

            // Check 3: decode busy latch (OR-ed from kv_used_blocks and active_decode_blocks)
            if let Some(decode_threshold) = active_decode_blocks_threshold {
                let is_busy = self
                    .decode_busy_latches
                    .get(&dp_rank)
                    .map(|latch| latch.latched_busy)
                    .unwrap_or_else(|| self.current_decode_busy(dp_rank, decode_threshold));
                if is_busy {
                    return true;
                }
            }

            // If we can't perform any check or no threshold exceeded, this dp_rank is free
            false
        })
    }
}

/// Worker monitor for tracking KV cache usage and busy states.
///
/// Cloning shares state via internal Arc-wrapped fields. This allows multiple pipelines
/// (e.g., chat and completions) to share the same monitor instance.
///
/// Prometheus metrics are exposed via [`WORKER_LOAD_METRICS`] (defined in `kv_router::sequence`),
/// which should be registered with the HTTP service's Prometheus registry using
/// [`register_worker_load_metrics`](crate::kv_router::metrics::register_worker_load_metrics).
///
/// In disaggregated mode, use `set_prefill_client` to register the prefill endpoint for
/// proper TTFT metric cleanup when prefill workers are removed.
#[derive(Clone)]
pub struct KvWorkerMonitor {
    /// Decode endpoint client (used for ITL cleanup and busy detection)
    client: Client,
    /// Optional prefill endpoint client (used for TTFT cleanup in disaggregated mode)
    prefill_client: Arc<RwLock<Option<Client>>>,
    /// Notifies the monitoring task when a prefill client is registered
    prefill_client_notify: Arc<Notify>,
    worker_load_states: Arc<DashMap<u64, WorkerLoadState>>,
    /// Load thresholds for busy detection. Each field is `Option<T>` — unset
    /// means the corresponding check in `is_busy` is skipped. If all three are
    /// `None`, rejection is fully disabled.
    thresholds: Arc<RwLock<LoadThresholdConfig>>,
    /// Guard to ensure start_monitoring() only runs once across clones
    started: Arc<AtomicBool>,
}

impl KvWorkerMonitor {
    /// Create a new worker monitor with the given threshold configuration.
    ///
    /// Unset thresholds (`None`) remain unset and their corresponding checks
    /// in `is_busy` are skipped. Thresholds can be updated at runtime via
    /// [`set_load_threshold_config`](Self::set_load_threshold_config) or the
    /// individual setters.
    ///
    /// Prometheus metrics are exposed via [`WORKER_LOAD_METRICS`] and should be registered
    /// using [`register_worker_load_metrics`](crate::kv_router::metrics::register_worker_load_metrics)
    /// during HTTP service setup.
    ///
    /// For disaggregated mode, call `set_prefill_client` after creation to enable
    /// proper TTFT metric cleanup when prefill workers are removed.
    pub fn new(client: Client, config: LoadThresholdConfig) -> Self {
        Self {
            client,
            prefill_client: Arc::new(RwLock::new(None)),
            prefill_client_notify: Arc::new(Notify::new()),
            worker_load_states: Arc::new(DashMap::new()),
            thresholds: Arc::new(RwLock::new(config)),
            started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns true iff the user explicitly configured at least one threshold.
    ///
    /// When false, all three per-field checks are skipped in `is_busy` and
    /// rejection is fully disabled. Callers that gate 503 responses on busy
    /// detection should check this before enabling the gate.
    pub fn is_configured(&self) -> bool {
        self.thresholds.read().unwrap().is_configured()
    }

    /// Set the prefill client for disaggregated mode.
    ///
    /// This enables monitoring of prefill endpoint instances for TTFT metric cleanup.
    /// In disaggregated mode, TTFT metrics are attributed to prefill workers, so we need
    /// to watch the prefill endpoint to clean up TTFT gauges when prefill workers disappear.
    ///
    /// This method can be called after `start_monitoring` - the monitoring loop will
    /// be immediately notified and start watching the prefill endpoint.
    pub fn set_prefill_client(&self, prefill_client: Client) {
        let mut guard = self.prefill_client.write().unwrap();
        *guard = Some(prefill_client);
        self.prefill_client_notify.notify_one();
        tracing::debug!("KvWorkerMonitor: prefill client registered for TTFT cleanup");
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
        // Guard: only start once across all clones
        if self.started.swap(true, Ordering::SeqCst) {
            tracing::debug!("Worker monitoring already started, skipping");
            return Ok(());
        }

        let endpoint = &self.client.endpoint;
        let component = endpoint.component();

        let cancellation_token = component.drt().child_token();

        // Watch for runtime config updates from model deployment cards via discovery interface
        let discovery = component.drt().discovery();
        let discovery_stream = match discovery
            .list_and_watch(DiscoveryQuery::AllModels, Some(cancellation_token.clone()))
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                tracing::error!("KvWorkerMonitor: failed to create discovery stream: {}", e);
                // Reset started flag so retry can work
                self.started.store(false, Ordering::SeqCst);
                return Err(e);
            }
        };
        let mut config_events_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

        // Subscribe to KV metrics events using EventSubscriber (Msgpack payloads)
        // This is optional - if NATS isn't available, we skip KV metrics but still do TTFT/ITL cleanup
        let kv_metrics_rx = match EventSubscriber::for_namespace(
            component.namespace(),
            KV_METRICS_SUBJECT,
        )
        .await
        {
            Ok(sub) => Some(sub.typed::<ActiveLoad>()),
            Err(e) => {
                tracing::warn!(
                    "KvWorkerMonitor: KV metrics subscriber not available ({}), skipping load metrics.",
                    e
                );
                None
            }
        };

        // Watch decode endpoint instances for cleanup (ITL metrics)
        let mut decode_instances_rx = self.client.instance_avail_watcher();

        let worker_load_states = self.worker_load_states.clone();
        let client = self.client.clone();
        let prefill_client_holder = self.prefill_client.clone();
        let prefill_client_notify = self.prefill_client_notify.clone();
        let thresholds = self.thresholds.clone();

        // Spawn background monitoring task
        tokio::spawn(async move {
            let mut kv_metrics_rx = kv_metrics_rx; // Move into async block
            let mut previous_busy_instances = Vec::new(); // Track previous state

            // Track decode worker IDs (for ITL cleanup)
            let mut known_decode_workers: std::collections::HashSet<u64> =
                decode_instances_rx.borrow().iter().copied().collect();

            // Track prefill worker IDs (for TTFT cleanup in disaggregated mode)
            let mut known_prefill_workers: std::collections::HashSet<u64> =
                std::collections::HashSet::new();
            let mut prefill_instances_rx: Option<tokio::sync::watch::Receiver<Vec<u64>>> = None;

            let mut known_worker_dp_ranks: HashMap<u64, std::collections::HashSet<u32>> =
                HashMap::new();

            loop {
                // Create a future that either reads from kv_metrics or pends forever if unavailable
                let kv_event_future = async {
                    if let Some(ref mut rx) = kv_metrics_rx {
                        rx.next().await
                    } else {
                        // If no subscriber, pend forever (this branch is effectively disabled)
                        std::future::pending().await
                    }
                };

                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Worker monitoring cancelled");
                        break;
                    }

                    // Handle runtime config updates
                    _ = config_events_rx.changed() => {
                        let runtime_configs = config_events_rx.borrow().clone();

                        // Find workers that are being removed (not in runtime_configs anymore)
                        let removed_workers: Vec<u64> = known_worker_dp_ranks
                            .keys()
                            .filter(|id| !runtime_configs.contains_key(id))
                            .copied()
                            .collect();

                        // Clean up Prometheus metrics for removed workers
                        for worker_id in &removed_workers {
                            if let Some(dp_ranks) = known_worker_dp_ranks.remove(worker_id) {
                                let dp_ranks_vec: Vec<u32> = dp_ranks.into_iter().collect();
                                // Clean up metrics for both worker types since we don't know which type this worker was
                                cleanup_worker_metrics(*worker_id, &dp_ranks_vec, WORKER_TYPE_DECODE);
                                cleanup_worker_metrics(*worker_id, &dp_ranks_vec, WORKER_TYPE_PREFILL);
                                tracing::debug!(
                                    "Removed Prometheus metrics for worker {}",
                                    worker_id
                                );
                            }
                        }

                        worker_load_states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));

                        // Update worker load states with runtime config values for all dp_ranks
                        // This ensures we track workers from MDCs even if they don't publish ActiveLoad
                        for (lease_id, runtime_config) in runtime_configs.iter() {
                            let mut state = worker_load_states.entry(*lease_id).or_default();

                            let dp_start = runtime_config.data_parallel_start_rank;
                            let dp_end = dp_start + runtime_config.data_parallel_size;

                            // Track dp_ranks for this worker (for cleanup when worker disappears)
                            let dp_ranks_set = known_worker_dp_ranks.entry(*lease_id).or_default();
                            for dp_rank in dp_start..dp_end {
                                dp_ranks_set.insert(dp_rank);
                            }

                            // Populate total_blocks for all dp_ranks (they share the same total)
                            if let Some(total_blocks) = runtime_config.total_kv_blocks {
                                for dp_rank in dp_start..dp_end {
                                    state.kv_total_blocks.insert(dp_rank, total_blocks);
                                }
                            }

                            // Populate max_num_batched_tokens for all dp_ranks
                            if let Some(max_batched) = runtime_config.max_num_batched_tokens {
                                for dp_rank in dp_start..dp_end {
                                    state.max_num_batched_tokens.insert(dp_rank, max_batched);
                                }
                            }
                        }
                    }

                    // Handle KV metrics updates (ActiveLoad) - only if subscriber is available
                    // Note: Prometheus gauges are updated directly by sequence.rs (router's own bookkeeping)
                    // This branch only updates WorkerLoadState for busy detection thresholds
                    kv_event = kv_event_future => {
                        let Some(event_result) = kv_event else {
                            tracing::debug!("KV metrics stream closed");
                            break;
                        };

                        let Ok((_envelope, active_load)) = event_result else {
                            tracing::error!("Error receiving KV metrics event: {event_result:?}");
                            continue;
                        };

                        let worker_id = active_load.worker_id;
                        let dp_rank = active_load.dp_rank;

                        // Track known worker/dp_rank combinations for cleanup
                        known_worker_dp_ranks
                            .entry(worker_id)
                            .or_default()
                            .insert(dp_rank);

                        // Snapshot thresholds once per event — rare writes (HTTP endpoint)
                        // mean RwLock contention is effectively zero.
                        let cfg = thresholds.read().unwrap().clone();

                        // Update worker load state per dp_rank (for busy detection only)
                        // Note: Prometheus gauges are updated directly by sequence.rs
                        {
                            let mut state = worker_load_states.entry(worker_id).or_default();
                            state.update_from_active_load(
                                &active_load,
                                cfg.active_decode_blocks_threshold,
                            );
                        }

                        // Recalculate all busy instances and update
                        let busy_instances: Vec<u64> = worker_load_states
                            .iter()
                            .filter_map(|entry| {
                                entry
                                    .value()
                                    .is_busy(
                                        cfg.active_decode_blocks_threshold,
                                        cfg.active_prefill_tokens_threshold,
                                        cfg.active_prefill_tokens_threshold_frac,
                                    )
                                    .then_some(*entry.key())
                            })
                            .collect();

                        // Only update if busy_instances has changed
                        if busy_instances != previous_busy_instances {
                            tracing::debug!("Busy instances changed: {:?}", busy_instances);
                            client.update_free_instances(&busy_instances);
                            previous_busy_instances = busy_instances;
                        }
                    }

                    // Handle decode endpoint instance changes (for ITL and decode metrics cleanup)
                    _ = decode_instances_rx.changed() => {
                        let current_instances: std::collections::HashSet<u64> =
                            decode_instances_rx.borrow().iter().copied().collect();

                        // Find decode workers that disappeared
                        let removed_workers: Vec<u64> = known_decode_workers
                            .difference(&current_instances)
                            .copied()
                            .collect();

                        if !removed_workers.is_empty() {
                            // Clean up metrics for removed decode workers (with worker_type=decode label)
                            for worker_id in &removed_workers {
                                // Get dp_ranks from known_worker_dp_ranks if available, otherwise use [0]
                                let dp_ranks: Vec<u32> = known_worker_dp_ranks
                                    .get(worker_id)
                                    .map(|ranks| ranks.iter().copied().collect())
                                    .unwrap_or_else(|| vec![0]);
                                cleanup_worker_metrics(*worker_id, &dp_ranks, WORKER_TYPE_DECODE);
                                tracing::debug!(
                                    "Cleaned up metrics for removed decode worker {}",
                                    worker_id
                                );
                            }
                        }

                        known_decode_workers = current_instances;
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
                            tracing::info!("Prefill endpoint watcher closed, will re-activate when client is set");
                            continue;
                        };

                        let Some(ref rx) = prefill_instances_rx else {
                            continue;
                        };

                        let current_instances: std::collections::HashSet<u64> =
                            rx.borrow().iter().copied().collect();

                        // Find prefill workers that disappeared
                        let removed_workers: Vec<u64> = known_prefill_workers
                            .difference(&current_instances)
                            .copied()
                            .collect();

                        if !removed_workers.is_empty() {
                            // Clean up metrics for removed prefill workers (with worker_type=prefill label)
                            for worker_id in &removed_workers {
                                // Get dp_ranks from known_worker_dp_ranks if available, otherwise use [0]
                                let dp_ranks: Vec<u32> = known_worker_dp_ranks
                                    .get(worker_id)
                                    .map(|ranks| ranks.iter().copied().collect())
                                    .unwrap_or_else(|| vec![0]);
                                cleanup_worker_metrics(*worker_id, &dp_ranks, WORKER_TYPE_PREFILL);
                                tracing::debug!(
                                    "Cleaned up metrics for removed prefill worker {}",
                                    worker_id
                                );
                            }
                        }

                        known_prefill_workers = current_instances;
                    }

                    // Wait for prefill client to be registered (push-based notification)
                    _ = prefill_client_notify.notified(), if prefill_instances_rx.is_none() => {
                        let guard = prefill_client_holder.read().unwrap();
                        if let Some(ref prefill_client) = *guard {
                            let rx = prefill_client.instance_avail_watcher();
                            known_prefill_workers = rx.borrow().iter().copied().collect();
                            prefill_instances_rx = Some(rx);
                            tracing::info!(
                                "KvWorkerMonitor: prefill endpoint watcher activated, tracking {} workers",
                                known_prefill_workers.len()
                            );
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
    use super::{LoadThresholdConfig, WorkerLoadState};
    use dynamo_kv_router::protocols::ActiveLoad;

    #[test]
    fn load_threshold_config_default_is_not_configured() {
        assert!(!LoadThresholdConfig::default().is_configured());
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
    fn is_busy_prefers_kv_used_blocks_over_active_decode_blocks() {
        let mut state = WorkerLoadState::default();
        state.active_decode_blocks.insert(0, 10);
        state.kv_used_blocks.insert(0, 90);
        state.kv_total_blocks.insert(0, 100);

        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn is_busy_falls_back_to_active_decode_blocks_when_kv_used_missing() {
        let mut state = WorkerLoadState::default();
        state.active_decode_blocks.insert(0, 90);
        state.kv_total_blocks.insert(0, 100);

        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn is_busy_recognizes_dp_rank_known_only_from_kv_used_blocks() {
        let mut state = WorkerLoadState::default();
        state.kv_used_blocks.insert(0, 90);
        state.kv_total_blocks.insert(0, 100);

        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_busy_latch_sets_busy_if_any_signal_is_busy() {
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

        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_busy_latch_only_clears_after_both_signals_report_nonbusy() {
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
        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));

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
        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));

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
        assert!(!state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_busy_latch_clears_with_only_kv_used_blocks_signal() {
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
        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));

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
        assert!(!state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_busy_latch_clears_with_only_active_decode_blocks_signal() {
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
        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));

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
        assert!(!state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_busy_latch_clears_when_both_signals_are_nonbusy_in_same_event() {
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
        assert!(state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));

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
        assert!(!state.is_busy(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn is_busy_returns_false_when_all_thresholds_are_none() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);
        state.active_decode_blocks.insert(0, 99);
        state.kv_used_blocks.insert(0, 99);
        state.active_prefill_tokens.insert(0, u64::MAX / 2);
        state.max_num_batched_tokens.insert(0, 1_000);

        assert!(!state.is_busy(None, None, None));
    }

    #[test]
    fn is_busy_with_only_decode_threshold_ignores_prefill_signals() {
        let mut state = WorkerLoadState::default();
        state.max_num_batched_tokens.insert(0, 1_000);
        state.active_prefill_tokens.insert(0, 5_000);

        assert!(!state.is_busy(Some(0.6), None, None));
    }

    #[test]
    fn is_busy_with_only_prefill_abs_ignores_decode_latch() {
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

        assert!(!state.is_busy(None, Some(u64::MAX), None));
    }

    #[test]
    fn is_busy_with_only_prefill_frac_ignores_decode_latch() {
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

        assert!(!state.is_busy(None, None, Some(2.0)));
    }

    #[test]
    fn is_busy_with_only_prefill_abs_fires_when_tokens_exceed_threshold() {
        let mut state = WorkerLoadState::default();
        state.active_prefill_tokens.insert(0, 5_000);

        assert!(state.is_busy(None, Some(1_000), None));
    }

    #[test]
    fn is_busy_with_only_prefill_frac_fires_when_fraction_exceeded() {
        let mut state = WorkerLoadState::default();
        state.max_num_batched_tokens.insert(0, 1_000);
        state.active_prefill_tokens.insert(0, 2_500);

        assert!(state.is_busy(None, None, Some(2.0)));
    }
}
