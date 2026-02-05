// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker set representation for multi-set routing.
//!
//! A WorkerSet groups workers that share the same dynamo namespace and MDC checksum.
//! During rolling updates, multiple sets can exist simultaneously (e.g., old and new versions),
//! with traffic distributed based on worker counts.

use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dynamo_runtime::component::Client;
use dynamo_runtime::DistributedRuntime;

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Information about a single worker in a set.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: WorkerId,
    pub runtime_config: Option<ModelRuntimeConfig>,
}

/// A set of workers sharing the same dynamo namespace and MDC checksum.
///
/// Workers within a set are considered equivalent for routing purposes.
/// Traffic between sets is distributed proportionally to worker counts.
///
/// Each WorkerSet owns a Client that watches its specific namespace for instances.
pub struct WorkerSet {
    /// Full dynamo namespace (e.g., "default-myapp-abc12345")
    namespace: String,

    /// MDC checksum for this set - all workers in set must have same checksum
    mdcsum: String,

    /// Client for discovering and tracking workers in this namespace
    client: Client,

    /// Workers in this set with their runtime configs
    workers: DashMap<WorkerId, WorkerInfo>,

    /// Round-robin counter for load balancing within this set
    round_robin_counter: AtomicUsize,
}

impl WorkerSet {
    /// Create a new worker set for the given namespace and MDC checksum.
    ///
    /// Creates a Client that watches the specific namespace for worker instances.
    pub async fn new(
        namespace: String,
        mdcsum: String,
        drt: Arc<DistributedRuntime>,
    ) -> anyhow::Result<Self> {
        // Create client for this specific namespace
        let component = drt.namespace(&namespace)?.component("backend")?;
        let endpoint = component.endpoint("generate");
        let client = endpoint.client().await?;

        tracing::debug!(
            namespace = %namespace,
            mdcsum = %mdcsum,
            "Created WorkerSet with client"
        );

        Ok(Self {
            namespace,
            mdcsum,
            client,
            workers: DashMap::new(),
            round_robin_counter: AtomicUsize::new(0),
        })
    }

    /// Get the namespace for this set.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Get the MDC checksum for this set.
    pub fn mdcsum(&self) -> &str {
        &self.mdcsum
    }

    /// Get the client for this worker set.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get instance IDs from the client (actual available workers).
    pub fn instance_ids(&self) -> Arc<Vec<u64>> {
        self.client.instance_ids_avail()
    }

    /// Add a worker to this set.
    ///
    /// Returns true if the worker was newly added, false if it already existed.
    pub fn add_worker(&self, worker_id: WorkerId, config: Option<ModelRuntimeConfig>) -> bool {
        let info = WorkerInfo {
            worker_id,
            runtime_config: config,
        };
        self.workers.insert(worker_id, info).is_none()
    }

    /// Remove a worker from this set.
    ///
    /// Returns the worker info if it existed, None otherwise.
    pub fn remove_worker(&self, worker_id: WorkerId) -> Option<WorkerInfo> {
        self.workers.remove(&worker_id).map(|(_, info)| info)
    }

    /// Check if a worker exists in this set.
    pub fn has_worker(&self, worker_id: WorkerId) -> bool {
        self.workers.contains_key(&worker_id)
    }

    /// Get the number of workers in this set (the set's weight).
    ///
    /// Uses the client's instance count as the source of truth.
    pub fn worker_count(&self) -> usize {
        self.instance_ids().len()
    }

    /// Check if this set is empty (has no workers).
    ///
    /// Uses the client's instance list as the source of truth.
    pub fn is_empty(&self) -> bool {
        self.instance_ids().is_empty()
    }

    /// Get all worker IDs in this set.
    ///
    /// Returns instance IDs from the client (live workers).
    pub fn worker_ids(&self) -> Vec<WorkerId> {
        (*self.instance_ids()).clone()
    }

    /// Get workers with their configs as a HashMap (for scheduler compatibility).
    pub fn workers_with_configs(
        &self,
    ) -> std::collections::HashMap<WorkerId, Option<ModelRuntimeConfig>> {
        self.workers
            .iter()
            .map(|entry| (*entry.key(), entry.value().runtime_config.clone()))
            .collect()
    }

    /// Select a worker using round-robin within this set.
    ///
    /// Returns None if the set is empty.
    pub fn select_round_robin(&self) -> Option<WorkerId> {
        let instance_ids = self.instance_ids();
        let count = instance_ids.len();
        if count == 0 {
            return None;
        }

        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % count;
        Some(instance_ids[idx])
    }

    /// Select a random worker from this set.
    ///
    /// Returns None if the set is empty.
    pub fn select_random(&self) -> Option<WorkerId> {
        let instance_ids = self.instance_ids();
        if instance_ids.is_empty() {
            return None;
        }

        use rand::Rng;
        let idx = rand::rng().random_range(0..instance_ids.len());
        Some(instance_ids[idx])
    }

    /// Update a worker's runtime config.
    pub fn update_worker_config(&self, worker_id: WorkerId, config: Option<ModelRuntimeConfig>) {
        if let Some(mut entry) = self.workers.get_mut(&worker_id) {
            entry.runtime_config = config;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Tests requiring WorkerSet creation are integration tests
    // since WorkerSet now requires a DistributedRuntime and creates a Client.
    // Worker discovery and selection are tested via the Client's instance tracking.

    #[test]
    fn test_worker_info() {
        let info = WorkerInfo {
            worker_id: 42,
            runtime_config: None,
        };
        assert_eq!(info.worker_id, 42);
        assert!(info.runtime_config.is_none());
    }
}
