// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

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

/// Default value for `max_num_batched_tokens` when the runtime config does not
/// report it. Set high enough that the frac-based overload check (which multiplies
/// this value by the threshold fraction) can never fire with realistic loads.
const DEFAULT_MAX_TOKENS: u64 = 10_000_000;

/// Compute the set of overloaded worker ids across all tracked worker load states
/// under the given thresholds. The returned set mixes decode workers (flagged by
/// `active_decode_blocks`) and prefill workers (flagged by `active_prefill_tokens`).
///
/// Although a monitor is owned 1-to-1 by its (decode/aggregated) WorkerSet — the
/// prefill WorkerSet has none — its load observation is namespace-wide: it subscribes
/// to the namespace-scoped `kv_metrics` subject (`EventSubscriber::for_namespace`), so
/// it receives `ActiveLoad` from every worker in the namespace, prefill and decode
/// alike. Hence the mixed set. `publish_overloaded_instances` then pushes this set to
/// both the decode and prefill `Client`s; each ignores ids outside its own pool.
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
}

/// Default hysteresis window for the decode overload deadline mechanism.
const DEFAULT_DECODE_OVERLOAD_HYSTERESIS_MS: u64 = 5000;

#[derive(Clone, Debug, Default)]
pub struct WorkerLoadState {
    pub active_decode_blocks: HashMap<u32, u64>,
    pub kv_used_blocks: HashMap<u32, u64>,
    pub kv_total_blocks: HashMap<u32, u64>,
    pub active_prefill_tokens: HashMap<u32, u64>,
    /// max_num_batched_tokens from runtime config (same for all dp_ranks)
    pub max_num_batched_tokens: HashMap<u32, u64>,
    /// Per-dp_rank overload deadlines. A dp_rank is considered overloaded while
    /// `Instant::now() < deadline` OR while any current signal still exceeds the
    /// threshold (which refreshes the deadline). Entries are removed once the
    /// deadline expires and all current signals are below the threshold.
    decode_overload_deadlines: HashMap<u32, Instant>,
}

impl WorkerLoadState {
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

    /// Update the decode overload deadline for `dp_rank`.
    ///
    /// If any arriving signal exceeds the threshold the deadline is extended to
    /// `now + hysteresis_duration`. If all arriving signals are below the
    /// threshold **and** the deadline has already expired, the entry is removed
    /// so that `is_overloaded` returns `false` for this dp_rank.
    ///
    /// `hysteresis_duration` is passed explicitly so tests can supply
    /// `Duration::ZERO` to make expiry instantaneous.
    fn update_decode_overload_deadline(
        &mut self,
        dp_rank: u32,
        active_decode_blocks: Option<u64>,
        kv_used_blocks: Option<u64>,
        active_decode_blocks_threshold: f64,
        hysteresis_duration: Duration,
    ) {
        let Some(&total_blocks) = self.kv_total_blocks.get(&dp_rank) else {
            return;
        };
        if total_blocks == 0 {
            return;
        }

        let any_signal_overloaded = active_decode_blocks.is_some_and(|v| {
            Self::is_decode_signal_overloaded(v, total_blocks, active_decode_blocks_threshold)
        }) || kv_used_blocks.is_some_and(|v| {
            Self::is_decode_signal_overloaded(v, total_blocks, active_decode_blocks_threshold)
        });

        if any_signal_overloaded {
            // Extend (or set) the deadline.
            self.decode_overload_deadlines
                .insert(dp_rank, Instant::now() + hysteresis_duration);
        } else {
            // All arriving signals are below threshold. Remove the deadline if it
            // has already expired, so is_overloaded can return false.
            let expired = self
                .decode_overload_deadlines
                .get(&dp_rank)
                .is_none_or(|&deadline| Instant::now() >= deadline);
            if expired {
                self.decode_overload_deadlines.remove(&dp_rank);

                // Drop stored decode-signal values for any signal absent from this
                // event. Without this, a signal that tripped overload and then went
                // silent (e.g. kv_used_blocks=90 followed by events that omit
                // kv_used_blocks) would leave its stale above-threshold value in the
                // map, and current_decode_overloaded would report overload forever —
                // the exact permanent-stuck failure this deadline model exists to
                // prevent, merely relocated from the old latch to the stored signal.
                // A signal present in this event has already been overwritten above
                // with its current (below-threshold) value, so only omitted signals
                // need clearing.
                if active_decode_blocks.is_none() {
                    self.active_decode_blocks.remove(&dp_rank);
                }
                if kv_used_blocks.is_none() {
                    self.kv_used_blocks.remove(&dp_rank);
                }
            }
        }
    }

    fn update_from_active_load(
        &mut self,
        active_load: &ActiveLoad,
        active_decode_blocks_threshold: Option<f64>,
        hysteresis_duration: Duration,
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
            self.update_decode_overload_deadline(
                dp_rank,
                active_load.active_decode_blocks,
                active_load.kv_used_blocks,
                threshold,
                hysteresis_duration,
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
            .chain(self.decode_overload_deadlines.keys())
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

            // Check 3: decode overload deadline (set when kv_used_blocks or
            // active_decode_blocks exceeded the threshold; expires after the
            // hysteresis window if no subsequent signal still exceeds it).
            if let Some(decode_threshold) = active_decode_blocks_threshold {
                // A dp_rank is overloaded if:
                // (a) its deadline is still in the future, OR
                // (b) the current signals still exceed the threshold (no deadline yet,
                //     or the deadline expired but the signals haven't recovered yet —
                //     the next update_decode_overload_deadline call will set a fresh one).
                let deadline_active = self
                    .decode_overload_deadlines
                    .get(&dp_rank)
                    .is_some_and(|&deadline| Instant::now() < deadline);
                let is_overloaded =
                    deadline_active || self.current_decode_overloaded(dp_rank, decode_threshold);
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
    /// Load thresholds for overload detection. Each field is `Option<T>` — unset
    /// means the corresponding check in `is_overloaded` is skipped. If all three are
    /// `None`, rejection is fully disabled.
    thresholds: Arc<RwLock<LoadThresholdConfig>>,
    /// Guard to ensure start_monitoring() only runs once across clones
    started: Arc<AtomicBool>,
    /// How long to keep the decode overload flag set after signals drop below the
    /// threshold. Read once at construction from `DYN_DECODE_OVERLOAD_HYSTERESIS_MS`
    /// (default: 5000 ms).
    decode_overload_hysteresis: Duration,
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
        use dynamo_runtime::config::environment_names::llm::DYN_DECODE_OVERLOAD_HYSTERESIS_MS;
        let hysteresis_ms = std::env::var(DYN_DECODE_OVERLOAD_HYSTERESIS_MS)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_DECODE_OVERLOAD_HYSTERESIS_MS);
        Self {
            client,
            prefill_client: Arc::new(RwLock::new(None)),
            prefill_client_notify: Arc::new(Notify::new()),
            worker_load_states: Arc::new(DashMap::new()),
            thresholds: Arc::new(RwLock::new(config)),
            started: Arc::new(AtomicBool::new(false)),
            decode_overload_hysteresis: Duration::from_millis(hysteresis_ms),
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
        *guard = Some(prefill_client);
        self.prefill_client_notify.notify_one();
        tracing::debug!(
            "KvWorkerMonitor: prefill client attached (seeded overloaded set; overload publish + TTFT cleanup)"
        );
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
        let decode_overload_hysteresis = self.decode_overload_hysteresis;

        // Spawn background monitoring task
        tokio::spawn(async move {
            let mut kv_metrics_rx = kv_metrics_rx; // Move into async block

            // Track decode worker IDs (for ITL cleanup)
            let mut known_decode_workers: std::collections::HashSet<u64> =
                decode_instances_rx.borrow().iter().copied().collect();

            // Track prefill worker IDs (for TTFT cleanup in disaggregated mode)
            let mut known_prefill_workers: std::collections::HashSet<u64> =
                std::collections::HashSet::new();
            let mut prefill_instances_rx: Option<tokio::sync::watch::Receiver<Vec<u64>>> = None;

            let mut known_worker_dp_ranks: HashMap<u64, std::collections::HashSet<u32>> =
                HashMap::new();
            let mut overloaded_tracker = OverloadedWorkerTracker::default();
            let mut last_thresholds = thresholds.read().unwrap().clone();

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
                        let thresholds_changed = cfg != last_thresholds;

                        // Update worker load state per dp_rank (for overload detection only).
                        // Note: Prometheus gauges are updated directly by sequence.rs
                        let (total_blocks, worker_overloaded) = {
                            let mut state = worker_load_states.entry(worker_id).or_default();
                            state.update_from_active_load(
                                &active_load,
                                cfg.active_decode_blocks_threshold,
                                decode_overload_hysteresis,
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

                        if overloaded_changed {
                            let overloaded_instances = overloaded_tracker.ids();
                            publish_overloaded_instances(
                                &client,
                                &prefill_client_holder,
                                &overloaded_instances,
                            );
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
                            overloaded_tracker.remove_workers(&removed_workers);
                            client.clear_overloaded_instances_for_removed(&removed_workers);
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
                            overloaded_tracker.remove_workers(&removed_workers);
                            client.clear_overloaded_instances_for_removed(&removed_workers);
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
        LoadThresholdConfig, OverloadedWorkerTracker, WorkerLoadState,
        compute_overloaded_instances, publish_overloaded_instances,
    };
    use dynamo_kv_router::protocols::ActiveLoad;
    use std::collections::HashSet;
    use std::time::Duration;

    /// Hysteresis used by tests that check the "overloaded stays set" path.
    /// A long enough value that it never expires during a synchronous test.
    const LONG_HYSTERESIS: Duration = Duration::from_secs(3600);

    /// Hysteresis of zero makes the deadline expire immediately — used to test
    /// the "deadline expires, worker is no longer overloaded" path.
    const ZERO_HYSTERESIS: Duration = Duration::ZERO;

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
            LONG_HYSTERESIS,
        );

        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    /// The hysteresis window holds overload across a transient dip: once a
    /// deadline is set, a subsequent below-threshold signal must NOT clear
    /// overload while the deadline is still in the future. This exercises the
    /// `deadline_active` branch of `is_overloaded` directly — distinct from the
    /// stale-stored-value path, because here the current signal is overwritten
    /// with a below-threshold value, so `current_decode_overloaded` is false and
    /// only the future deadline can keep the worker overloaded.
    #[test]
    fn decode_overload_deadline_holds_across_transient_below_threshold_signal() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        // Trip overload with a long hysteresis window: deadline = now + 1h.
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
            LONG_HYSTERESIS,
        );
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        // A below-threshold kv_used_blocks=10 arrives, overwriting the stale 90.
        // current_decode_overloaded is now false, but the deadline is still an
        // hour in the future, so the worker must remain overloaded (hysteresis).
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: Some(10),
            },
            Some(0.6),
            LONG_HYSTERESIS,
        );
        assert!(
            state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)),
            "future deadline must hold overload across a transient below-threshold signal"
        );
    }

    /// Under the deadline model, receiving a single below-threshold signal is
    /// enough to clear overload once the hysteresis window expires.
    /// With ZERO_HYSTERESIS the deadline is already expired on the next check.
    #[test]
    fn decode_overload_clears_with_single_below_threshold_signal_after_deadline() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        // Trip the overload via kv_used_blocks.
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
            ZERO_HYSTERESIS,
        );
        // Deadline is now() + ZERO = now(), so is_overloaded checks current signals.
        // kv_used_blocks is still 90 in state.kv_used_blocks, so current_decode_overloaded
        // returns true.
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        // A single active_decode_blocks signal arrives at a low value; kv_used_blocks
        // is absent from this event. With ZERO_HYSTERESIS the deadline is already
        // expired, and the arriving signal is not overloaded, so the deadline is
        // removed. Because kv_used_blocks was omitted, its stale above-threshold
        // value (90) is dropped on the expired branch — otherwise it would keep the
        // worker overloaded forever even though the signal has gone silent.
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: None,
            },
            Some(0.6),
            ZERO_HYSTERESIS,
        );
        // Deadline expired and the stale kv_used_blocks=90 was cleared, so
        // current_decode_overloaded is false — the worker recovers without needing a
        // fresh below-threshold kv_used_blocks event to arrive.
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_clears_with_only_kv_used_blocks_signal() {
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
            ZERO_HYSTERESIS,
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
            ZERO_HYSTERESIS,
        );
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_clears_with_only_active_decode_blocks_signal() {
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
            ZERO_HYSTERESIS,
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
            ZERO_HYSTERESIS,
        );
        assert!(!state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));
    }

    #[test]
    fn decode_overload_clears_when_both_signals_are_not_overloaded_in_same_event() {
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
            ZERO_HYSTERESIS,
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
            ZERO_HYSTERESIS,
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
            LONG_HYSTERESIS,
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
            LONG_HYSTERESIS,
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

    /// Regression test for the "permanently stuck" bug.
    ///
    /// Scenario: a worker trips the overload latch via `kv_used_blocks=90`, then
    /// *stops emitting `kv_used_blocks` entirely* while `active_decode_blocks`
    /// keeps arriving at a low value. Under the old AND-of-cleared latch the
    /// `kv_used_blocks_cleared` flag was never set, so the latch stayed `true`
    /// forever.
    ///
    /// Under the deadline model with `ZERO_HYSTERESIS` the deadline expires
    /// immediately, and the next event (carrying only a below-threshold
    /// `active_decode_blocks`, with `kv_used_blocks` omitted) finds an expired
    /// deadline; the stale above-threshold `kv_used_blocks` value is dropped, so
    /// `is_overloaded` returns `false` without requiring a fresh below-threshold
    /// `kv_used_blocks` event. This is the crux of the fix: recovery must not
    /// depend on the silent signal ever speaking again.
    #[test]
    fn decode_overload_deadline_expires_when_signal_stops_arriving() {
        let mut state = WorkerLoadState::default();
        state.kv_total_blocks.insert(0, 100);

        // Step 1: kv_used_blocks exceeds threshold — deadline is set (ZERO, so
        // already expired by the time we call is_overloaded, but the current
        // stored kv_used_blocks value is 90, which is above threshold).
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: None,
                active_prefill_tokens: None,
                kv_used_blocks: Some(90),
            },
            Some(0.6),
            ZERO_HYSTERESIS,
        );
        // Still overloaded because state.kv_used_blocks[0] == 90 > threshold.
        assert!(state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)));

        // Step 2: worker stops sending kv_used_blocks entirely. Only
        // active_decode_blocks=10 (below threshold) arrives. The deadline is ZERO so
        // it expired before this call; no arriving signal is above threshold, so the
        // deadline entry is removed AND the stale kv_used_blocks=90 (omitted from this
        // event) is dropped. is_overloaded must now return false — the worker recovers
        // even though kv_used_blocks never sent another value. This is exactly the
        // #10564 permanent-stuck failure, and it must not survive relocation from the
        // old latch to the stored signal map.
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: None,
            },
            Some(0.6),
            ZERO_HYSTERESIS,
        );
        assert!(
            !state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)),
            "worker must recover after the deadline expires even though the \
             overloading kv_used_blocks signal went permanently silent"
        );

        // Step 3: a later below-threshold kv_used_blocks=10 event keeps it clear.
        state.update_from_active_load(
            &ActiveLoad {
                worker_id: 1,
                dp_rank: 0,
                active_decode_blocks: Some(10),
                active_prefill_tokens: None,
                kv_used_blocks: Some(10),
            },
            Some(0.6),
            ZERO_HYSTERESIS,
        );
        assert!(
            !state.is_overloaded(Some(0.6), Some(u64::MAX), Some(2.0)),
            "worker must not be permanently stuck overloaded after signals recover"
        );
    }
}
