// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker set manager for coordinating traffic across multiple worker sets.
//!
//! During rolling updates, workers from old and new versions coexist in separate sets
//! (different dynamo namespaces). The WorkerSetManager:
//! - Tracks all sets matching a namespace prefix
//! - Provides weighted set selection based on worker counts
//! - Auto-removes empty sets when all workers are gone

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::Notify;

use dynamo_runtime::DistributedRuntime;

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;

use super::worker_set::WorkerSet;

/// Manages multiple worker sets for multi-set routing.
///
/// Sets are keyed by their full dynamo namespace and are auto-created when workers
/// are discovered and auto-removed when they become empty.
#[derive(Debug)]
pub struct WorkerSetManager {
    /// Sets keyed by full namespace (e.g., "default-myapp-abc12345")
    sets: DashMap<String, Arc<WorkerSet>>,

    /// Namespace prefix used for discovery (e.g., "default-myapp")
    prefix: String,

    /// Notify on set changes (add/remove set, worker count changes)
    notify: Notify,
}

impl WorkerSetManager {
    /// Create a new worker set manager for the given namespace prefix.
    pub fn new(prefix: String) -> Self {
        Self {
            sets: DashMap::new(),
            prefix,
            notify: Notify::new(),
        }
    }

    /// Get the namespace prefix this manager uses for discovery.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Add or update a worker in the appropriate set.
    ///
    /// Creates the set if it doesn't exist. The set is identified by the full namespace.
    /// Returns true if the set was newly created.
    pub async fn add_worker(
        &self,
        namespace: &str,
        worker_id: WorkerId,
        mdcsum: &str,
        config: Option<ModelRuntimeConfig>,
        drt: Arc<DistributedRuntime>,
    ) -> anyhow::Result<bool> {
        // Check if set already exists
        let set_exists = self.sets.contains_key(namespace);

        if !set_exists {
            // Create new WorkerSet with Client
            let new_set = WorkerSet::new(
                namespace.to_string(),
                mdcsum.to_string(),
                drt.clone(),
            ).await?;

            tracing::info!(
                namespace,
                prefix = %self.prefix,
                mdcsum,
                "Created new worker set with client"
            );

            self.sets.insert(namespace.to_string(), Arc::new(new_set));
            self.notify.notify_waiters();
        }

        // Add worker metadata (config) to the set
        if let Some(set) = self.sets.get(namespace) {
            if set.add_worker(worker_id, config) {
                tracing::debug!(
                    namespace,
                    worker_id,
                    worker_count = set.worker_count(),
                    "Worker added to set"
                );
            }
        }

        Ok(!set_exists)
    }

    /// Remove a worker from its set.
    ///
    /// If the set becomes empty, it is automatically removed.
    pub fn remove_worker(&self, namespace: &str, worker_id: WorkerId) {
        if let Some(set) = self.sets.get(namespace) {
            if set.remove_worker(worker_id).is_some() {
                tracing::debug!(
                    namespace,
                    worker_id,
                    worker_count = set.worker_count(),
                    "Worker removed from set"
                );

                // Auto-remove empty sets
                if set.is_empty() {
                    drop(set); // Release the reference before removing
                    if self.sets.remove(namespace).is_some() {
                        tracing::info!(
                            namespace,
                            prefix = %self.prefix,
                            "Removed empty worker set"
                        );
                    }
                }

                self.notify.notify_waiters();
            }
        }
    }

    /// Update a worker's runtime config.
    pub fn update_worker_config(
        &self,
        namespace: &str,
        worker_id: WorkerId,
        config: Option<ModelRuntimeConfig>,
    ) {
        if let Some(set) = self.sets.get(namespace) {
            set.update_worker_config(worker_id, config);
        }
    }

    /// Get total instance count across all sets.
    pub fn total_instances(&self) -> usize {
        self.sets.iter().map(|entry| entry.worker_count()).sum()
    }

    /// Get the number of sets.
    pub fn set_count(&self) -> usize {
        self.sets.len()
    }

    /// Check if there are any workers across all sets.
    pub fn has_workers(&self) -> bool {
        self.sets.iter().any(|entry| !entry.is_empty())
    }

    /// Get all sets as a vector.
    pub fn sets(&self) -> Vec<Arc<WorkerSet>> {
        self.sets.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Get a specific set by namespace.
    pub fn get_set(&self, namespace: &str) -> Option<Arc<WorkerSet>> {
        self.sets.get(namespace).map(|entry| entry.value().clone())
    }

    /// Find the worker set that contains a given instance ID.
    ///
    /// This is used by routers to get the correct client for an instance.
    pub fn find_set_for_instance(&self, instance_id: WorkerId) -> Option<Arc<WorkerSet>> {
        for entry in self.sets.iter() {
            let set = entry.value();
            if set.instance_ids().contains(&instance_id) {
                return Some(set.clone());
            }
        }
        None
    }

    /// Select a set using weighted random selection based on worker counts.
    ///
    /// A set with 7 workers has 70% chance vs a set with 3 workers having 30% chance.
    /// Returns None if there are no workers in any set.
    pub fn select_weighted(&self) -> Option<Arc<WorkerSet>> {
        let total = self.total_instances();
        if total == 0 {
            return None;
        }

        // Weighted random selection
        use rand::Rng;
        let r = rand::rng().random_range(0..total);
        let mut cumulative = 0usize;

        for entry in self.sets.iter() {
            cumulative += entry.worker_count();
            if r < cumulative {
                return Some(entry.value().clone());
            }
        }

        // Fallback (shouldn't happen unless there's a race)
        self.sets.iter().next().map(|e| e.value().clone())
    }

    /// Wait for set changes (set added/removed, worker count changed).
    pub async fn wait_for_changes(&self) {
        self.notify.notified().await;
    }

    /// Get workers from all sets combined as a single map.
    ///
    /// This is useful when you need all workers regardless of set affiliation.
    pub fn all_workers_with_configs(
        &self,
    ) -> std::collections::HashMap<WorkerId, Option<ModelRuntimeConfig>> {
        let mut result = std::collections::HashMap::new();
        for entry in self.sets.iter() {
            result.extend(entry.workers_with_configs());
        }
        result
    }

    /// Get set weights as a map of namespace to worker count.
    ///
    /// Useful for metrics and debugging.
    pub fn set_weights(&self) -> std::collections::HashMap<String, usize> {
        self.sets
            .iter()
            .map(|entry| (entry.key().clone(), entry.worker_count()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests requiring WorkerSetManager with actual workers are integration tests
    // since WorkerSet now requires a DistributedRuntime and creates Clients.
    // Worker discovery and routing logic are tested via integration tests.

    #[test]
    fn test_worker_set_manager_creation() {
        let manager = WorkerSetManager::new("default-myapp".to_string());

        assert_eq!(manager.prefix(), "default-myapp");
        assert_eq!(manager.set_count(), 0);
        assert_eq!(manager.total_instances(), 0);
        assert!(!manager.has_workers());
    }

    #[test]
    fn test_set_weights_empty() {
        let manager = WorkerSetManager::new("prefix".to_string());
        let weights = manager.set_weights();
        assert!(weights.is_empty());
    }
}
