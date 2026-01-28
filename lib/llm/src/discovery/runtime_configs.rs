// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::watch;

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;

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
                    tracing::info!("ModelManager: Runtime config found for worker_id: {worker_id}");
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

            // Wait for next change; ignore RecvError (sender dropped = shutdown)
            let _ = self.change_rx.changed().await;
        }
    }
}
