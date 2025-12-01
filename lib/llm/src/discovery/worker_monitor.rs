// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::scoring::LoadEvent;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::component::Client;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::traits::events::EventSubscriber;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use tokio_stream::StreamExt;

/// Scale factor for storing f64 thresholds as u32 (10000 = 4 decimal places)
const THRESHOLD_SCALE: u32 = 10000;

/// Worker load monitoring state per dp_rank
#[derive(Clone, Debug, Default)]
pub struct WorkerLoadState {
    pub kv_active_blocks: HashMap<u32, u64>,
    pub kv_total_blocks: HashMap<u32, u64>,
}

impl WorkerLoadState {
    /// Returns true if ALL dp_ranks (that have data in both maps) exceed the threshold
    pub fn is_busy(&self, threshold: f64) -> bool {
        // Get all dp_ranks that exist in both active and total blocks
        let common_dp_ranks: Vec<_> = self
            .kv_active_blocks
            .keys()
            .filter(|dp_rank| self.kv_total_blocks.contains_key(dp_rank))
            .collect();

        // If no common dp_ranks, not busy
        if common_dp_ranks.is_empty() {
            return false;
        }

        // Check if ALL common dp_ranks exceed threshold
        common_dp_ranks.iter().all(|&&dp_rank| {
            if let (Some(&active), Some(&total)) = (
                self.kv_active_blocks.get(&dp_rank),
                self.kv_total_blocks.get(&dp_rank),
            ) {
                total > 0 && (active as f64) > (threshold * total as f64)
            } else {
                false
            }
        })
    }
}

/// Worker monitor for tracking KV cache usage and busy states.
///
/// All fields are `Arc`, so cloning shares state. This allows multiple pipelines
/// (e.g., chat and completions) to share the same monitor instance.
#[derive(Clone)]
pub struct KvWorkerMonitor {
    client: Arc<Client>,
    worker_load_states: Arc<RwLock<HashMap<u64, WorkerLoadState>>>,
    /// Threshold stored as parts-per-10000 (e.g., 8500 = 0.85)
    busy_threshold: Arc<AtomicU32>,
    /// Guard to ensure start_monitoring() only runs once across clones
    started: Arc<AtomicBool>,
}

impl KvWorkerMonitor {
    /// Create a new worker monitor with the given threshold.
    ///
    /// The threshold (0.0-1.0) controls when workers are considered busy based on
    /// KV cache utilization. It can be dynamically updated via `set_threshold()`.
    pub fn new(client: Arc<Client>, threshold: f64) -> Self {
        Self {
            client,
            worker_load_states: Arc::new(RwLock::new(HashMap::new())),
            busy_threshold: Arc::new(AtomicU32::new(Self::threshold_to_scaled(threshold))),
            started: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Convert a f64 threshold (0.0-1.0) to scaled u32 for atomic storage.
    #[inline]
    fn threshold_to_scaled(threshold: f64) -> u32 {
        (threshold * THRESHOLD_SCALE as f64) as u32
    }

    /// Convert a scaled u32 back to f64 threshold (0.0-1.0).
    #[inline]
    fn scaled_to_threshold(scaled: u32) -> f64 {
        scaled as f64 / THRESHOLD_SCALE as f64
    }

    /// Get the current threshold value as f64.
    pub fn threshold(&self) -> f64 {
        Self::scaled_to_threshold(self.busy_threshold.load(Ordering::Relaxed))
    }

    /// Set the threshold value from f64.
    pub fn set_threshold(&self, threshold: f64) {
        self.busy_threshold
            .store(Self::threshold_to_scaled(threshold), Ordering::Relaxed);
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
        let discovery_stream = discovery
            .list_and_watch(DiscoveryQuery::AllModels, Some(cancellation_token.clone()))
            .await?;
        let mut config_events_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

        // Subscribe to KV metrics events
        let mut kv_metrics_rx = component.namespace().subscribe(KV_METRICS_SUBJECT).await?;

        let worker_load_states = self.worker_load_states.clone();
        let client = self.client.clone();
        let busy_threshold = self.busy_threshold.clone();

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

                        let mut states = worker_load_states.write().unwrap();
                        states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));

                        // Update worker load states with total blocks for all dp_ranks
                        for (lease_id, runtime_config) in runtime_configs.iter() {
                            let state = states.entry(*lease_id).or_default();

                            // Populate total_blocks for all dp_ranks (they share the same total)
                            if let Some(total_blocks) = runtime_config.total_kv_blocks {
                                for dp_rank in 0..runtime_config.data_parallel_size {
                                    state.kv_total_blocks.insert(dp_rank, total_blocks);
                                }
                            }
                        }
                    }

                    // Handle KV metrics updates
                    kv_event = kv_metrics_rx.next() => {
                        let Some(event) = kv_event else {
                            tracing::debug!("KV metrics stream closed");
                            break;
                        };

                        if let Ok(load_event) = serde_json::from_slice::<LoadEvent>(&event.payload) {
                            let worker_id = load_event.worker_id;
                            let active_blocks = load_event.data.kv_stats.kv_active_blocks;
                            let dp_rank = load_event.data.worker_stats.data_parallel_rank.unwrap_or(0);

                            // Update worker load state per dp_rank
                            let mut states = worker_load_states.write().unwrap();
                            let state = states.entry(worker_id).or_default();
                            state.kv_active_blocks.insert(dp_rank, active_blocks);
                            drop(states);

                            // Load threshold dynamically - allows runtime updates
                            let scaled_threshold = busy_threshold.load(Ordering::Relaxed);
                            let current_threshold = Self::scaled_to_threshold(scaled_threshold);

                            // Recalculate all busy instances and update
                            let states = worker_load_states.read().unwrap();
                            let busy_instances: Vec<u64> = states
                                .iter()
                                .filter_map(|(&id, state)| {
                                    state.is_busy(current_threshold).then_some(id)
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
                    }
                }
            }

            tracing::info!("Worker monitoring task exiting");
        });

        Ok(())
    }
}
