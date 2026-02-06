// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{LazyLock, RwLock};

use tokio::sync::Notify;

use dashmap::DashMap;
use prometheus::{IntGaugeVec, Opts, Registry};
use serde::{Deserialize, Serialize};

use crate::http::service::metrics::{
    WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE, WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE,
    WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE,
};
use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::protocols::ActiveLoad;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::component::Client;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::metrics::prometheus_names::frontend_service;
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;

// Re-export worker type constants from timing.rs (single source of truth)
pub use crate::protocols::common::timing::{WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL};

/// Global Prometheus gauge for active decode blocks per worker (labels: worker_id, dp_rank, worker_type)
/// This is shared across all KvWorkerMonitor instances.
pub static WORKER_ACTIVE_DECODE_BLOCKS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::WORKER_ACTIVE_DECODE_BLOCKS
            ),
            "Active KV cache decode blocks per worker",
        ),
        &["worker_id", "dp_rank", "worker_type"],
    )
    .expect("Failed to create worker_active_decode_blocks gauge")
});

/// Global Prometheus gauge for active prefill tokens per worker (labels: worker_id, dp_rank, worker_type)
/// This is shared across all KvWorkerMonitor instances.
pub static WORKER_ACTIVE_PREFILL_TOKENS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "dynamo_frontend_{}",
                frontend_service::WORKER_ACTIVE_PREFILL_TOKENS
            ),
            "Active prefill tokens queued per worker",
        ),
        &["worker_id", "dp_rank", "worker_type"],
    )
    .expect("Failed to create worker_active_prefill_tokens gauge")
});

/// Register the global worker load Prometheus metrics with the given registry.
///
/// This should be called once during HTTP service setup to expose the worker load
/// metrics via the `/metrics` endpoint.
///
/// # Errors
/// Returns an error if the metrics are already registered with the registry.
pub fn register_worker_load_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(WORKER_ACTIVE_DECODE_BLOCKS_GAUGE.clone()))?;
    registry.register(Box::new(WORKER_ACTIVE_PREFILL_TOKENS_GAUGE.clone()))?;
    Ok(())
}

/// Clean up all Prometheus metrics for a worker across the specified dp_ranks.
///
/// This removes metrics with the given worker_id, dp_rank, and worker_type label combination.
/// Called when workers are removed to prevent stale metrics from accumulating.
fn cleanup_worker_metrics(worker_id: u64, dp_ranks: &[u32], worker_type: &str) {
    let worker_id_str = worker_id.to_string();
    for dp_rank in dp_ranks {
        let dp_rank_str = dp_rank.to_string();
        let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];
        let _ = WORKER_ACTIVE_DECODE_BLOCKS_GAUGE.remove_label_values(labels);
        let _ = WORKER_ACTIVE_PREFILL_TOKENS_GAUGE.remove_label_values(labels);
        let _ = WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE.remove_label_values(labels);
        let _ = WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE.remove_label_values(labels);
        let _ = WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.remove_label_values(labels);
    }
}

/// Scale factor for storing f64 thresholds as u32 (10000 = 4 decimal places)
const THRESHOLD_SCALE: u32 = 10000;

/// Default value for max_num_batched_tokens and active_prefill_tokens_threshold
/// when not configured. Set high enough to effectively disable busy detection.
const DEFAULT_MAX_TOKENS: u64 = 10_000_000;

/// Configuration for worker load thresholds used in busy detection.
///
/// All thresholds are optional. When not set, defaults are applied:
/// - `active_decode_blocks_threshold`: 1.0 (effectively disabled)
/// - `active_prefill_tokens_threshold`: 10,000,000 (effectively disabled)
/// - `active_prefill_tokens_threshold_frac`: 1.5 (effectively disabled)
/// - `max_num_batched_tokens` (from runtime config): 10,000,000 if not reported
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

    /// Fraction of max_num_batched_tokens (0.0-1.5+).
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
#[derive(Clone, Debug, Default)]
pub struct WorkerLoadState {
    pub active_decode_blocks: HashMap<u32, u64>,
    pub kv_total_blocks: HashMap<u32, u64>,
    pub active_prefill_tokens: HashMap<u32, u64>,
    /// max_num_batched_tokens from runtime config (same for all dp_ranks)
    pub max_num_batched_tokens: HashMap<u32, u64>,
}

impl WorkerLoadState {
    /// Returns true if ALL dp_ranks are considered busy based on the threshold logic.
    ///
    /// For each dp_rank, a dp_rank is busy if ANY of these conditions is met (OR logic):
    /// 1. `active_prefill_tokens > active_prefill_tokens_threshold` (absolute threshold)
    /// 2. `active_prefill_tokens > frac * max_num_batched_tokens` (fraction-based threshold)
    /// 3. `active_decode_blocks / total_blocks > active_decode_blocks_threshold` (blocks threshold)
    ///
    /// If none of these checks can be performed (missing data), that dp_rank is considered free.
    ///
    /// The worker is busy only if ALL dp_ranks are busy.
    pub fn is_busy(
        &self,
        active_decode_blocks_threshold: f64,
        active_prefill_tokens_threshold: u64,
        active_prefill_tokens_threshold_frac: f64,
    ) -> bool {
        // Get all dp_ranks we know about
        let all_dp_ranks: std::collections::HashSet<_> = self
            .active_decode_blocks
            .keys()
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
                if active_tokens > active_prefill_tokens_threshold {
                    return true; // This dp_rank is busy due to absolute token threshold
                }

                // Check 2: prefill tokens threshold (fraction of max_num_batched_tokens)
                let max_batched = self
                    .max_num_batched_tokens
                    .get(&dp_rank)
                    .copied()
                    .unwrap_or(DEFAULT_MAX_TOKENS);
                let frac_threshold =
                    (active_prefill_tokens_threshold_frac * max_batched as f64) as u64;
                if active_tokens > frac_threshold {
                    return true; // This dp_rank is busy due to frac-based token threshold
                }
            }

            // Check 3: blocks threshold
            // Skip if total_blocks is 0 (no capacity means threshold check is meaningless)
            if let (Some(&active_blocks), Some(&total_blocks)) = (
                self.active_decode_blocks.get(&dp_rank),
                self.kv_total_blocks.get(&dp_rank),
            ) && total_blocks > 0
                && (active_blocks as f64) > (active_decode_blocks_threshold * total_blocks as f64)
            {
                return true; // This dp_rank is busy due to blocks
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
/// Prometheus metrics are exposed via the global gauges [`WORKER_ACTIVE_DECODE_BLOCKS_GAUGE`]
/// and [`WORKER_ACTIVE_PREFILL_TOKENS_GAUGE`], which should be registered with the HTTP
/// service's Prometheus registry using [`register_worker_load_metrics`].
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
    /// Active decode blocks threshold stored as parts-per-10000 (e.g., 8500 = 0.85)
    active_decode_blocks_threshold: Arc<AtomicU32>,
    /// Active prefill tokens threshold stored as literal token count (u64)
    active_prefill_tokens_threshold: Arc<AtomicU64>,
    /// Active prefill tokens threshold as fraction of max_num_batched_tokens, stored scaled
    active_prefill_tokens_threshold_frac: Arc<AtomicU32>,
    /// Guard to ensure start_monitoring() only runs once across clones
    started: Arc<AtomicBool>,
}

impl KvWorkerMonitor {
    /// Create a new worker monitor with the given threshold configuration.
    ///
    /// All thresholds can be dynamically updated via setter methods or
    /// `set_load_threshold_config()`.
    ///
    /// Defaults are applied for any threshold not specified in the config:
    /// - `active_decode_blocks_threshold`: 1.0 (effectively disabled)
    /// - `active_prefill_tokens_threshold`: DEFAULT_MAX_TOKENS (effectively disabled)
    /// - `active_prefill_tokens_threshold_frac`: 1.5 (effectively disabled)
    ///
    /// Prometheus metrics are exposed via the global gauges and should be registered
    /// using [`register_worker_load_metrics`] during HTTP service setup.
    ///
    /// For disaggregated mode, call `set_prefill_client` after creation to enable
    /// proper TTFT metric cleanup when prefill workers are removed.
    pub fn new(client: Client, config: LoadThresholdConfig) -> Self {
        let active_decode_blocks = config.active_decode_blocks_threshold.unwrap_or(1.0);
        let active_prefill_tokens = config
            .active_prefill_tokens_threshold
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let active_prefill_tokens_frac = config.active_prefill_tokens_threshold_frac.unwrap_or(1.5);

        Self {
            client,
            prefill_client: Arc::new(RwLock::new(None)),
            prefill_client_notify: Arc::new(Notify::new()),
            worker_load_states: Arc::new(DashMap::new()),
            active_decode_blocks_threshold: Arc::new(AtomicU32::new(Self::f64_to_scaled(
                active_decode_blocks,
            ))),
            active_prefill_tokens_threshold: Arc::new(AtomicU64::new(active_prefill_tokens)),
            active_prefill_tokens_threshold_frac: Arc::new(AtomicU32::new(Self::f64_to_scaled(
                active_prefill_tokens_frac,
            ))),
            started: Arc::new(AtomicBool::new(false)),
        }
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
        // Notify the monitoring task that prefill client is now available
        self.prefill_client_notify.notify_one();
        tracing::debug!("KvWorkerMonitor: prefill client registered for TTFT cleanup");
    }

    /// Convert a f64 threshold to scaled u32 for atomic storage.
    #[inline]
    fn f64_to_scaled(threshold: f64) -> u32 {
        (threshold * THRESHOLD_SCALE as f64) as u32
    }

    /// Convert a scaled u32 back to f64 threshold.
    #[inline]
    fn scaled_to_f64(scaled: u32) -> f64 {
        scaled as f64 / THRESHOLD_SCALE as f64
    }

    /// Get the current active decode blocks threshold value as f64.
    pub fn active_decode_blocks_threshold(&self) -> f64 {
        Self::scaled_to_f64(self.active_decode_blocks_threshold.load(Ordering::Relaxed))
    }

    /// Set the active decode blocks threshold value from f64.
    pub fn set_active_decode_blocks_threshold(&self, threshold: f64) {
        self.active_decode_blocks_threshold
            .store(Self::f64_to_scaled(threshold), Ordering::Relaxed);
    }

    /// Get the current active prefill tokens threshold value as u64.
    pub fn active_prefill_tokens_threshold(&self) -> u64 {
        self.active_prefill_tokens_threshold.load(Ordering::Relaxed)
    }

    /// Set the active prefill tokens threshold value from u64.
    pub fn set_active_prefill_tokens_threshold(&self, threshold: u64) {
        self.active_prefill_tokens_threshold
            .store(threshold, Ordering::Relaxed);
    }

    /// Get the current active prefill tokens threshold frac value as f64.
    pub fn active_prefill_tokens_threshold_frac(&self) -> f64 {
        Self::scaled_to_f64(
            self.active_prefill_tokens_threshold_frac
                .load(Ordering::Relaxed),
        )
    }

    /// Set the active prefill tokens threshold frac value from f64.
    pub fn set_active_prefill_tokens_threshold_frac(&self, frac: f64) {
        self.active_prefill_tokens_threshold_frac
            .store(Self::f64_to_scaled(frac), Ordering::Relaxed);
    }

    /// Get the current load threshold configuration.
    pub fn load_threshold_config(&self) -> LoadThresholdConfig {
        LoadThresholdConfig {
            active_decode_blocks_threshold: Some(self.active_decode_blocks_threshold()),
            active_prefill_tokens_threshold: Some(self.active_prefill_tokens_threshold()),
            active_prefill_tokens_threshold_frac: Some(self.active_prefill_tokens_threshold_frac()),
        }
    }

    /// Update all thresholds from a LoadThresholdConfig.
    /// Only updates fields that are Some in the config.
    pub fn set_load_threshold_config(&self, config: &LoadThresholdConfig) {
        if let Some(threshold) = config.active_decode_blocks_threshold {
            self.set_active_decode_blocks_threshold(threshold);
        }
        if let Some(threshold) = config.active_prefill_tokens_threshold {
            self.set_active_prefill_tokens_threshold(threshold);
        }
        if let Some(frac) = config.active_prefill_tokens_threshold_frac {
            self.set_active_prefill_tokens_threshold_frac(frac);
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
        let active_decode_blocks_threshold = self.active_decode_blocks_threshold.clone();
        let active_prefill_tokens_threshold = self.active_prefill_tokens_threshold.clone();
        let active_prefill_tokens_threshold_frac =
            self.active_prefill_tokens_threshold_frac.clone();

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

                            // Track dp_ranks for this worker (for cleanup when worker disappears)
                            let dp_ranks_set = known_worker_dp_ranks.entry(*lease_id).or_default();
                            for dp_rank in 0..runtime_config.data_parallel_size {
                                dp_ranks_set.insert(dp_rank);
                            }

                            // Populate total_blocks for all dp_ranks (they share the same total)
                            if let Some(total_blocks) = runtime_config.total_kv_blocks {
                                for dp_rank in 0..runtime_config.data_parallel_size {
                                    state.kv_total_blocks.insert(dp_rank, total_blocks);
                                }
                            }

                            // Populate max_num_batched_tokens for all dp_ranks
                            if let Some(max_batched) = runtime_config.max_num_batched_tokens {
                                for dp_rank in 0..runtime_config.data_parallel_size {
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

                        // Update worker load state per dp_rank (for busy detection only)
                        // Note: Prometheus gauges are updated directly by sequence.rs
                        {
                            let mut state = worker_load_states.entry(worker_id).or_default();
                            if let Some(active_blocks) = active_load.active_decode_blocks {
                                state.active_decode_blocks.insert(dp_rank, active_blocks);
                            }
                            if let Some(active_tokens) = active_load.active_prefill_tokens {
                                state.active_prefill_tokens.insert(dp_rank, active_tokens);
                            }
                        }

                        // Load thresholds dynamically - allows runtime updates
                        let current_active_decode_blocks_threshold =
                            Self::scaled_to_f64(active_decode_blocks_threshold.load(Ordering::Relaxed));
                        let current_active_prefill_tokens_threshold =
                            active_prefill_tokens_threshold.load(Ordering::Relaxed);
                        let current_active_prefill_tokens_threshold_frac =
                            Self::scaled_to_f64(active_prefill_tokens_threshold_frac.load(Ordering::Relaxed));

                        // Recalculate all busy instances and update
                        let busy_instances: Vec<u64> = worker_load_states
                            .iter()
                            .filter_map(|entry| {
                                entry
                                    .value()
                                    .is_busy(
                                        current_active_decode_blocks_threshold,
                                        current_active_prefill_tokens_threshold,
                                        current_active_prefill_tokens_threshold_frac,
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
