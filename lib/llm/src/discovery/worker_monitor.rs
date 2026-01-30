// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::http::service::metrics::{WORKER_LAST_ITL_GAUGE, WORKER_LAST_TTFT_GAUGE};
use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::protocols::ActiveLoad;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::component::Client;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::metrics::prometheus_names::frontend_service;
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use prometheus::{IntGaugeVec, Opts, Registry};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

/// Worker type label values for Prometheus metrics
pub const WORKER_TYPE_PREFILL: &str = "prefill";
pub const WORKER_TYPE_DECODE: &str = "decode";

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

/// Scale factor for storing f64 thresholds as u32 (10000 = 4 decimal places)
const THRESHOLD_SCALE: u32 = 10000;

/// Worker load monitoring state per dp_rank
#[derive(Clone, Debug, Default)]
pub struct WorkerLoadState {
    pub active_decode_blocks: HashMap<u32, u64>,
    pub kv_total_blocks: HashMap<u32, u64>,
    pub active_prefill_tokens: HashMap<u32, u64>,
}

impl WorkerLoadState {
    /// Returns true if ALL dp_ranks are considered busy based on the dual-threshold logic:
    ///
    /// For each dp_rank:
    /// 1. If `active_prefill_tokens` is available, check if tokens exceed the literal threshold.
    ///    If so, that dp_rank is busy.
    /// 2. If not, check if `active_decode_blocks` and `kv_total_blocks` are both available,
    ///    and if blocks exceed threshold. If so, that dp_rank is busy.
    /// 3. If neither check can be performed (missing data), that dp_rank is considered free.
    ///
    /// The worker is busy only if ALL dp_ranks are busy.
    pub fn is_busy(
        &self,
        active_decode_blocks_threshold: f64,
        active_prefill_tokens_threshold: u64,
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
            // First check: prefill tokens threshold (literal token count)
            if let Some(&active_tokens) = self.active_prefill_tokens.get(&dp_rank)
                && active_tokens > active_prefill_tokens_threshold
            {
                return true; // This dp_rank is busy due to tokens
            }

            // Second check: blocks threshold
            // Skip if total_blocks is 0 (no capacity means threshold check is meaningless)
            if let (Some(&active_blocks), Some(&total_blocks)) = (
                self.active_decode_blocks.get(&dp_rank),
                self.kv_total_blocks.get(&dp_rank),
            ) && total_blocks > 0
                && (active_blocks as f64) > (active_decode_blocks_threshold * total_blocks as f64)
            {
                return true; // This dp_rank is busy due to blocks
            }

            // If we can't perform either check, this dp_rank is considered free
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
    worker_load_states: Arc<RwLock<HashMap<u64, WorkerLoadState>>>,
    /// Active decode blocks threshold stored as parts-per-10000 (e.g., 8500 = 0.85)
    active_decode_blocks_threshold: Arc<AtomicU32>,
    /// Active prefill tokens threshold stored as literal token count (u64)
    active_prefill_tokens_threshold: Arc<AtomicU64>,
    /// Guard to ensure start_monitoring() only runs once across clones
    started: Arc<AtomicBool>,
}

impl KvWorkerMonitor {
    /// Create a new worker monitor with the given thresholds.
    ///
    /// - `active_decode_blocks_threshold` (0.0-1.0): Threshold percentage for KV cache block utilization
    /// - `active_prefill_tokens_threshold`: Literal token count threshold for prefill token utilization
    ///
    /// Both thresholds can be dynamically updated via `set_active_decode_blocks_threshold()` and
    /// `set_active_prefill_tokens_threshold()`.
    ///
    /// Prometheus metrics are exposed via the global gauges and should be registered
    /// using [`register_worker_load_metrics`] during HTTP service setup.
    ///
    /// For disaggregated mode, call `set_prefill_client` after creation to enable
    /// proper TTFT metric cleanup when prefill workers are removed.
    pub fn new(
        client: Client,
        active_decode_blocks_threshold: f64,
        active_prefill_tokens_threshold: u64,
    ) -> Self {
        Self {
            client,
            prefill_client: Arc::new(RwLock::new(None)),
            worker_load_states: Arc::new(RwLock::new(HashMap::new())),
            active_decode_blocks_threshold: Arc::new(AtomicU32::new(
                Self::active_decode_blocks_threshold_to_scaled(active_decode_blocks_threshold),
            )),
            active_prefill_tokens_threshold: Arc::new(AtomicU64::new(
                active_prefill_tokens_threshold,
            )),
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
    /// pick up the prefill client on its next iteration.
    pub fn set_prefill_client(&self, prefill_client: Client) {
        let mut guard = self.prefill_client.write().unwrap();
        *guard = Some(prefill_client);
        tracing::debug!("KvWorkerMonitor: prefill client registered for TTFT cleanup");
    }

    /// Convert a f64 active decode blocks threshold (0.0-1.0) to scaled u32 for atomic storage.
    #[inline]
    fn active_decode_blocks_threshold_to_scaled(threshold: f64) -> u32 {
        (threshold * THRESHOLD_SCALE as f64) as u32
    }

    /// Convert a scaled u32 back to f64 active decode blocks threshold (0.0-1.0).
    #[inline]
    fn scaled_to_active_decode_blocks_threshold(scaled: u32) -> f64 {
        scaled as f64 / THRESHOLD_SCALE as f64
    }

    /// Get the current active decode blocks threshold value as f64.
    pub fn active_decode_blocks_threshold(&self) -> f64 {
        Self::scaled_to_active_decode_blocks_threshold(
            self.active_decode_blocks_threshold.load(Ordering::Relaxed),
        )
    }

    /// Set the active decode blocks threshold value from f64.
    pub fn set_active_decode_blocks_threshold(&self, threshold: f64) {
        self.active_decode_blocks_threshold.store(
            Self::active_decode_blocks_threshold_to_scaled(threshold),
            Ordering::Relaxed,
        );
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

    /// Get the worker load states for external access
    pub fn load_states(&self) -> Arc<RwLock<HashMap<u64, WorkerLoadState>>> {
        self.worker_load_states.clone()
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
        let active_decode_blocks_threshold = self.active_decode_blocks_threshold.clone();
        let active_prefill_tokens_threshold = self.active_prefill_tokens_threshold.clone();

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
                                let worker_id_str = worker_id.to_string();
                                for dp_rank in dp_ranks {
                                    let dp_rank_str = dp_rank.to_string();
                                    // Clean up load metrics
                                    let _ = WORKER_ACTIVE_DECODE_BLOCKS_GAUGE
                                        .remove_label_values(&[&worker_id_str, &dp_rank_str]);
                                    let _ = WORKER_ACTIVE_PREFILL_TOKENS_GAUGE
                                        .remove_label_values(&[&worker_id_str, &dp_rank_str]);
                                    // Clean up timing metrics (TTFT/ITL per worker)
                                    let _ = WORKER_LAST_TTFT_GAUGE
                                        .remove_label_values(&[&worker_id_str, &dp_rank_str]);
                                    let _ = WORKER_LAST_ITL_GAUGE
                                        .remove_label_values(&[&worker_id_str, &dp_rank_str]);
                                }
                                tracing::debug!(
                                    "Removed Prometheus metrics for worker {}",
                                    worker_id
                                );
                            }
                        }

                        let mut states = worker_load_states.write().unwrap();
                        states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));

                        // Update worker load states and known_worker_dp_ranks for all workers
                        // This ensures we track workers from MDCs even if they don't publish ActiveLoad
                        for (lease_id, runtime_config) in runtime_configs.iter() {
                            let state = states.entry(*lease_id).or_default();

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
                        let mut states = worker_load_states.write().unwrap();
                        let state = states.entry(worker_id).or_default();

                        if let Some(active_blocks) = active_load.active_decode_blocks {
                            state.active_decode_blocks.insert(dp_rank, active_blocks);
                        }
                        if let Some(active_tokens) = active_load.active_prefill_tokens {
                            state.active_prefill_tokens.insert(dp_rank, active_tokens);
                        }
                        drop(states);

                        // Load thresholds dynamically - allows runtime updates
                        let current_active_decode_blocks_threshold = Self::scaled_to_active_decode_blocks_threshold(
                            active_decode_blocks_threshold.load(Ordering::Relaxed),
                        );
                        let current_active_prefill_tokens_threshold = active_prefill_tokens_threshold.load(Ordering::Relaxed);

                        // Recalculate all busy instances and update
                        let states = worker_load_states.read().unwrap();
                        let busy_instances: Vec<u64> = states
                            .iter()
                            .filter_map(|(&id, state)| {
                                state
                                    .is_busy(current_active_decode_blocks_threshold, current_active_prefill_tokens_threshold)
                                    .then_some(id)
                            })
                            .collect();
                        drop(states);

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
                                let worker_id_str = worker_id.to_string();
                                let _ = WORKER_LAST_ITL_GAUGE
                                    .remove_label_values(&[worker_id_str.as_str(), "0", WORKER_TYPE_DECODE]);
                                let _ = WORKER_ACTIVE_DECODE_BLOCKS_GAUGE
                                    .remove_label_values(&[worker_id_str.as_str(), "0", WORKER_TYPE_DECODE]);
                                let _ = WORKER_ACTIVE_PREFILL_TOKENS_GAUGE
                                    .remove_label_values(&[worker_id_str.as_str(), "0", WORKER_TYPE_DECODE]);
                                tracing::debug!(
                                    "Cleaned up metrics for removed decode worker {}",
                                    worker_id
                                );
                            }
                        }

                        known_decode_workers = current_instances;
                    }

                    // Handle prefill endpoint instance changes (for TTFT and prefill metrics cleanup in disaggregated mode)
                    _ = async {
                        if let Some(ref mut rx) = prefill_instances_rx {
                            rx.changed().await
                        } else {
                            // No prefill watcher yet, pend forever
                            std::future::pending().await
                        }
                    } => {
                        if let Some(ref rx) = prefill_instances_rx {
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
                                    let worker_id_str = worker_id.to_string();
                                    let _ = WORKER_LAST_TTFT_GAUGE
                                        .remove_label_values(&[worker_id_str.as_str(), "0", WORKER_TYPE_PREFILL]);
                                    let _ = WORKER_ACTIVE_DECODE_BLOCKS_GAUGE
                                        .remove_label_values(&[worker_id_str.as_str(), "0", WORKER_TYPE_PREFILL]);
                                    let _ = WORKER_ACTIVE_PREFILL_TOKENS_GAUGE
                                        .remove_label_values(&[worker_id_str.as_str(), "0", WORKER_TYPE_PREFILL]);
                                    tracing::debug!(
                                        "Cleaned up metrics for removed prefill worker {}",
                                        worker_id
                                    );
                                }
                            }

                            known_prefill_workers = current_instances;
                        }
                    }

                    // Periodically check if a prefill client has been registered
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)), if prefill_instances_rx.is_none() => {
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
