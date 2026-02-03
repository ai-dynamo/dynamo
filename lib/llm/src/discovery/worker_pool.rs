// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker pool representation for multi-pool routing.
//!
//! A WorkerPool groups workers that share the same dynamo namespace and MDC checksum.
//! During rolling updates, multiple pools can exist simultaneously (e.g., old and new versions),
//! with traffic distributed based on worker counts.

use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Information about a single worker in a pool.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: WorkerId,
    pub runtime_config: Option<ModelRuntimeConfig>,
}

/// A pool of workers sharing the same dynamo namespace and MDC checksum.
///
/// Workers within a pool are considered equivalent for routing purposes.
/// Traffic between pools is distributed proportionally to worker counts.
#[derive(Debug)]
pub struct WorkerPool {
    /// Full dynamo namespace (e.g., "default-myapp-abc12345")
    namespace: String,

    /// MDC checksum for this pool - all workers in pool must have same checksum
    mdcsum: String,

    /// Workers in this pool, keyed by WorkerId
    workers: DashMap<WorkerId, WorkerInfo>,

    /// Round-robin counter for load balancing within this pool
    round_robin_counter: AtomicUsize,
}

impl WorkerPool {
    /// Create a new worker pool for the given namespace and MDC checksum.
    pub fn new(namespace: String, mdcsum: String) -> Self {
        Self {
            namespace,
            mdcsum,
            workers: DashMap::new(),
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Get the namespace for this pool.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Get the MDC checksum for this pool.
    pub fn mdcsum(&self) -> &str {
        &self.mdcsum
    }

    /// Add a worker to this pool.
    ///
    /// Returns true if the worker was newly added, false if it already existed.
    pub fn add_worker(&self, worker_id: WorkerId, config: Option<ModelRuntimeConfig>) -> bool {
        let info = WorkerInfo {
            worker_id,
            runtime_config: config,
        };
        self.workers.insert(worker_id, info).is_none()
    }

    /// Remove a worker from this pool.
    ///
    /// Returns the worker info if it existed, None otherwise.
    pub fn remove_worker(&self, worker_id: WorkerId) -> Option<WorkerInfo> {
        self.workers.remove(&worker_id).map(|(_, info)| info)
    }

    /// Check if a worker exists in this pool.
    pub fn has_worker(&self, worker_id: WorkerId) -> bool {
        self.workers.contains_key(&worker_id)
    }

    /// Get the number of workers in this pool (the pool's weight).
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Check if this pool is empty (has no workers).
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all worker IDs in this pool.
    pub fn worker_ids(&self) -> Vec<WorkerId> {
        self.workers.iter().map(|entry| *entry.key()).collect()
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

    /// Select a worker using round-robin within this pool.
    ///
    /// Returns None if the pool is empty.
    pub fn select_round_robin(&self) -> Option<WorkerId> {
        let count = self.workers.len();
        if count == 0 {
            return None;
        }

        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % count;
        self.workers
            .iter()
            .nth(idx)
            .map(|entry| *entry.key())
    }

    /// Select a random worker from this pool.
    ///
    /// Returns None if the pool is empty.
    pub fn select_random(&self) -> Option<WorkerId> {
        use rand::seq::IteratorRandom;
        let mut rng = rand::rng();
        self.workers
            .iter()
            .choose(&mut rng)
            .map(|entry| *entry.key())
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

    #[test]
    fn test_worker_pool_basic() {
        let pool = WorkerPool::new("ns-abc123".to_string(), "checksum1".to_string());

        assert_eq!(pool.namespace(), "ns-abc123");
        assert_eq!(pool.mdcsum(), "checksum1");
        assert!(pool.is_empty());
        assert_eq!(pool.worker_count(), 0);

        // Add workers
        assert!(pool.add_worker(1, None));
        assert!(pool.add_worker(2, None));
        assert!(!pool.add_worker(1, None)); // Duplicate

        assert_eq!(pool.worker_count(), 2);
        assert!(!pool.is_empty());
        assert!(pool.has_worker(1));
        assert!(pool.has_worker(2));
        assert!(!pool.has_worker(3));

        // Remove worker
        let info = pool.remove_worker(1);
        assert!(info.is_some());
        assert_eq!(pool.worker_count(), 1);
        assert!(!pool.has_worker(1));
    }

    #[test]
    fn test_worker_pool_round_robin() {
        let pool = WorkerPool::new("ns".to_string(), "cs".to_string());
        pool.add_worker(10, None);
        pool.add_worker(20, None);
        pool.add_worker(30, None);

        // Round robin should cycle through workers
        let mut seen = std::collections::HashSet::new();
        for _ in 0..6 {
            if let Some(id) = pool.select_round_robin() {
                seen.insert(id);
            }
        }
        // Should have seen all 3 workers
        assert_eq!(seen.len(), 3);
    }
}
