// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kv_router::KV_METRICS_SUBJECT;
use crate::kv_router::protocols::ActiveLoad;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::component::Client;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::pipeline::{WorkerLoadMonitor, async_trait};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

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
#[derive(Clone)]
pub struct KvWorkerMonitor {
    client: Client,
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
    pub fn new(
        client: Client,
        active_decode_blocks_threshold: f64,
        active_prefill_tokens_threshold: u64,
    ) -> Self {
        Self {
            client,
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
                }
            }

            tracing::info!("Worker monitoring task exiting");
        });

        Ok(())
    }
}
