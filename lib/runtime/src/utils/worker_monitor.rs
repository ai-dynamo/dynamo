// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: Make load comparisons and runtime metrics a generic trait so this monitoring
// system is not tied to KV cache concepts, which are LLM-specific. This would allow
// different types of workers to define their own load metrics and busy thresholds.

use crate::component::{Client, InstanceSource};
use crate::traits::DistributedRuntimeProvider;
use crate::traits::events::EventSubscriber;
use crate::utils::typed_prefix_watcher::{key_extractors, watch_prefix_with_extraction};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::watch;
use tokio_stream::StreamExt;

// Constants for monitoring configuration
const KV_METRICS_SUBJECT: &str = "kv_metrics";

// Internal structs for deserializing metrics events
#[derive(serde::Deserialize)]
struct LoadEvent {
    worker_id: i64,
    data: ForwardPassMetrics,
}

#[derive(serde::Deserialize)]
struct ForwardPassMetrics {
    worker_stats: WorkerStats,
    kv_stats: KvStats,
}

#[derive(serde::Deserialize)]
struct WorkerStats {
    data_parallel_rank: Option<u32>,
}

#[derive(serde::Deserialize)]
struct KvStats {
    kv_active_blocks: u64,
}

#[derive(serde::Deserialize, Clone)]
struct RuntimeConfig {
    total_kv_blocks: Option<u64>,
    data_parallel_size: u32,
}

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

/// Worker monitor for tracking KV cache usage and busy states
pub struct WorkerMonitor {
    client: Arc<Client>,
    worker_load_states: Arc<RwLock<HashMap<i64, WorkerLoadState>>>,
    busy_threshold: f64,
}

impl WorkerMonitor {
    /// Create a new worker monitor with custom threshold
    pub fn new_with_threshold(client: Arc<Client>, busy_threshold: f64) -> Self {
        Self {
            client,
            worker_load_states: Arc::new(RwLock::new(HashMap::new())),
            busy_threshold,
        }
    }

    /// Get the worker load states for external access
    pub fn load_states(&self) -> Arc<RwLock<HashMap<i64, WorkerLoadState>>> {
        self.worker_load_states.clone()
    }

    /// Start background monitoring of worker KV cache usage
    pub async fn start_monitoring(&self) -> anyhow::Result<()> {
        let endpoint = &self.client.endpoint;
        let component = endpoint.component();

        let Some(etcd_client) = component.drt().etcd_client() else {
            // Static mode, no monitoring needed
            return Ok(());
        };

        // WorkerMonitor is in the wrong crate. It deals with LLM things (KV) so it should be in
        // dynamo-llm not dynamo-runtime.
        // That means we cannot use ModelDeploymentCard, so use serde_json::Value for now .
        let runtime_configs_watcher = watch_prefix_with_extraction(
            etcd_client,
            "v1/mdc/", // should be model_card::ROOT_PREFIX but wrong crate
            key_extractors::lease_id,
            |card: serde_json::Value| {
                let runtime_config: Option<RuntimeConfig> = card
                    .get("runtime_config")
                    .and_then(|rc| serde_json::from_value(rc.clone()).ok());
                runtime_config
            },
            component.drt().child_token(),
        )
        .await?;
        let mut config_events_rx = runtime_configs_watcher.receiver();

        // Subscribe to KV metrics events
        let mut kv_metrics_rx = component.namespace().subscribe(KV_METRICS_SUBJECT).await?;

        let worker_load_states = self.worker_load_states.clone();
        let client = self.client.clone();
        let cancellation_token = component.drt().child_token();
        let busy_threshold = self.busy_threshold; // Capture threshold for the closure

        // Spawn background monitoring task
        tokio::spawn(async move {
            let mut previous_busy_instances = Vec::new(); // Track previous state

            loop {
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::debug!("Worker monitoring cancelled");
                        break;
                    }

                    // Handle runtime config updates - now receives full HashMap
                    _ = config_events_rx.changed() => {
                        let runtime_configs = config_events_rx.borrow().clone();

                        let mut states = worker_load_states.write().unwrap();
                        states.retain(|lease_id, _| runtime_configs.contains_key(lease_id));

                        // Update worker load states with total blocks for all dp_ranks
                        for (lease_id, runtime_config) in runtime_configs.iter() {
                            let state = states.entry(*lease_id).or_default();

                            // Populate total_blocks for all dp_ranks (they share the same total)
                            // data_parallel_size defaults to 1 via serde in ModelRuntimeConfig
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

                            // Recalculate all busy instances and update
                            let states = worker_load_states.read().unwrap();
                            let busy_instances: Vec<i64> = states
                                .iter()
                                .filter_map(|(&id, state)| {
                                    state.is_busy(busy_threshold).then_some(id)
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
