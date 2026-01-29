// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::watch;

use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::{DiscoveryQuery, watch_and_extract_field};
use dynamo_runtime::prelude::DistributedRuntimeProvider;

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;

/// Runtime configs for an endpoint with watch-based change notifications.
/// Call `subscribe()` to get a subscriber with its own watch receiver.
pub struct RuntimeConfigs {
    pub configs: Arc<DashMap<WorkerId, Option<ModelRuntimeConfig>>>,
    change_tx: watch::Sender<u64>,
}

impl RuntimeConfigs {
    pub(crate) fn new() -> Self {
        let (change_tx, _) = watch::channel(0u64);
        Self {
            configs: Arc::new(DashMap::new()),
            change_tx,
        }
    }

    /// Create a subscriber that can wait for config changes.
    /// Each subscriber has its own watch receiver, so notifications are not lost.
    pub fn subscribe(&self) -> RuntimeConfigsSubscriber {
        RuntimeConfigsSubscriber {
            configs: self.configs.clone(),
            change_rx: self.change_tx.subscribe(),
        }
    }

    /// Notify all subscribers of a change (internal use only).
    fn notify_change(&self) {
        // Increment counter to notify subscribers
        self.change_tx.send_modify(|v| *v = v.wrapping_add(1));
    }

    /// Returns the number of workers in the configs.
    pub fn num_workers(&self) -> usize {
        self.configs.len()
    }

    /// Update configs with new worker instances and their configs.
    /// Notifies subscribers if a config with Some value is added or a worker is removed.
    pub(crate) fn update(
        &self,
        new_instance_ids: &[WorkerId],
        new_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) {
        // First, remove workers that no longer exist
        let current_workers: HashSet<WorkerId> = self.configs.iter().map(|r| *r.key()).collect();
        let new_workers: HashSet<WorkerId> = new_instance_ids.iter().copied().collect();
        let mut worker_removed = false;
        for removed_worker in current_workers.difference(&new_workers) {
            self.configs.remove(removed_worker);
            worker_removed = true;
        }

        // Then, add/update workers
        // Track if any config became Some (for notify)
        let mut config_added = false;
        for worker_id in new_instance_ids {
            let config = new_configs.get(worker_id).cloned();
            if config.is_some() {
                let prev_config = self.configs.get(worker_id);
                let was_none = prev_config
                    .as_ref()
                    .map(|r| r.value().is_none())
                    .unwrap_or(true);
                if was_none {
                    tracing::info!("RuntimeConfigs: config found for worker_id: {worker_id}");
                    config_added = true;
                }
            }
            self.configs.insert(*worker_id, config);
        }

        // Notify when a config with Some value is added OR a worker is removed
        if config_added || worker_removed {
            self.notify_change();
        }
    }

    /// Spawn background task to watch runtime configs via discovery.
    /// Does not block - consumers should use `subscribe().wait_for_some()` if they need workers.
    pub(crate) async fn start_watcher(self: &Arc<Self>, endpoint: &Endpoint) -> anyhow::Result<()> {
        let component = endpoint.component();
        let cancellation_token = component.drt().primary_token();

        // Set up discovery watch for EndpointModels
        let discovery = component.drt().discovery();
        let endpoint_id = endpoint.id();
        let discovery_key = DiscoveryQuery::EndpointModels {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
        };
        let discovery_stream = discovery
            .list_and_watch(discovery_key.clone(), Some(cancellation_token.clone()))
            .await?;

        // Extract runtime_config from ModelDeploymentCard
        let mut runtime_configs_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

        // Also watch instance IDs
        let client = endpoint.client().await?;
        let mut instance_ids_rx = client.instance_avail_watcher();

        // Spawn background task to watch for config changes
        // Note: We don't block here - consumers should wait on notify for configs they need
        let inner = self.clone();
        let cancel_token = cancellation_token.clone();
        tokio::spawn(async move {
            tracing::trace!("RuntimeConfigs watcher started");
            loop {
                // Wait for either instances or configs to change
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::trace!("RuntimeConfigs watcher shutting down");
                        break;
                    }
                    result = instance_ids_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("instance IDs watch sender shutdown");
                            break;
                        }
                    }
                    result = runtime_configs_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("runtime configs watch sender shutdown");
                            break;
                        }
                    }
                }

                // Get the latest values from both channels
                let new_instance_ids = instance_ids_rx.borrow_and_update().clone();
                let new_configs = runtime_configs_rx.borrow_and_update().clone();

                inner.update(&new_instance_ids, &new_configs);

                tracing::trace!(
                    "RuntimeConfigs: Updated with {} workers",
                    inner.configs.len()
                );
            }
            tracing::trace!("RuntimeConfigs watcher stopped");
        });

        Ok(())
    }
}

/// A subscriber to runtime config changes.
/// Each subscriber has its own watch receiver, ensuring no notifications are lost.
pub struct RuntimeConfigsSubscriber {
    pub configs: Arc<DashMap<WorkerId, Option<ModelRuntimeConfig>>>,
    pub change_rx: watch::Receiver<u64>,
}

impl RuntimeConfigsSubscriber {
    /// Wait until at least one worker has a Some config.
    /// Returns the list of worker IDs that have configs.
    /// This is race-safe: checks the DashMap first, only waits if empty.
    /// Returns empty vec if the sender is dropped (shutdown).
    pub async fn wait_for_some(&mut self) -> Vec<WorkerId> {
        loop {
            let ready: Vec<WorkerId> = self
                .configs
                .iter()
                .filter(|r| r.value().is_some())
                .map(|r| *r.key())
                .collect();

            if !ready.is_empty() {
                return ready;
            }

            // If sender dropped (shutdown), return empty rather than loop forever
            if self.change_rx.changed().await.is_err() {
                tracing::warn!("RuntimeConfigsSubscriber: sender dropped during wait_for_some");
                return vec![];
            }
        }
    }
}
