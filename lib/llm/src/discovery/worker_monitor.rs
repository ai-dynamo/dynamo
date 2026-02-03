// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::protocols::ActiveLoad;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::component::Client;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;

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
#[derive(Clone)]
pub struct KvWorkerMonitor {
    client: Client,
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
    pub fn new(client: Client, config: LoadThresholdConfig) -> Self {
        let active_decode_blocks = config.active_decode_blocks_threshold.unwrap_or(1.0);
        let active_prefill_tokens = config
            .active_prefill_tokens_threshold
            .unwrap_or(DEFAULT_MAX_TOKENS);
        let active_prefill_tokens_frac = config.active_prefill_tokens_threshold_frac.unwrap_or(1.5);

        Self {
            client,
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
        let discovery_stream = discovery
            .list_and_watch(DiscoveryQuery::AllModels, Some(cancellation_token.clone()))
            .await?;
        let mut config_events_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

        // Subscribe to KV metrics events using EventSubscriber (Msgpack payloads)
        let mut kv_metrics_rx =
            EventSubscriber::for_namespace(component.namespace(), KV_METRICS_SUBJECT)
                .await?
                .typed::<ActiveLoad>();

        let worker_load_states = self.worker_load_states.clone();
        let client = self.client.clone();
        let active_decode_blocks_threshold = self.active_decode_blocks_threshold.clone();
        let active_prefill_tokens_threshold = self.active_prefill_tokens_threshold.clone();
        let active_prefill_tokens_threshold_frac =
            self.active_prefill_tokens_threshold_frac.clone();

        // Spawn background monitoring task
        tokio::spawn(async move {
            let mut previous_busy_instances = Vec::new(); // Track previous state

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Worker monitoring cancelled");
                        break;
                    }

                    // Handle runtime config updates
                    _ = config_events_rx.changed() => {
                        let runtime_configs = config_events_rx.borrow().clone();

                        worker_load_states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));

                        // Update worker load states with runtime config values for all dp_ranks
                        for (lease_id, runtime_config) in runtime_configs.iter() {
                            let mut state = worker_load_states.entry(*lease_id).or_default();

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

                    // Handle KV metrics updates (ActiveLoad)
                    kv_event = kv_metrics_rx.next() => {
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

                        // Update worker load state per dp_rank
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
                }
            }

            tracing::info!("Worker monitoring task exiting");
        });

        Ok(())
    }
}
