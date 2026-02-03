// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-pool manager for coordinating traffic across multiple worker pools.
//!
//! During rolling updates, workers from old and new versions coexist in separate pools
//! (different dynamo namespaces). The MultiPoolManager:
//! - Tracks all pools matching a namespace prefix
//! - Provides weighted pool selection based on worker counts
//! - Auto-removes empty pools when all workers are gone

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::Notify;

use crate::kv_router::protocols::WorkerId;
use crate::local_model::runtime_config::ModelRuntimeConfig;

use super::worker_pool::WorkerPool;

/// Manages multiple worker pools for multi-pool routing.
///
/// Pools are keyed by their full dynamo namespace and are auto-created when workers
/// are discovered and auto-removed when they become empty.
#[derive(Debug)]
pub struct MultiPoolManager {
    /// Pools keyed by full namespace (e.g., "default-myapp-abc12345")
    pools: DashMap<String, Arc<WorkerPool>>,

    /// Namespace prefix used for discovery (e.g., "default-myapp")
    prefix: String,

    /// Notify on pool changes (add/remove pool, worker count changes)
    notify: Notify,
}

impl MultiPoolManager {
    /// Create a new multi-pool manager for the given namespace prefix.
    pub fn new(prefix: String) -> Self {
        Self {
            pools: DashMap::new(),
            prefix,
            notify: Notify::new(),
        }
    }

    /// Get the namespace prefix this manager uses for discovery.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Add or update a worker in the appropriate pool.
    ///
    /// Creates the pool if it doesn't exist. The pool is identified by the full namespace.
    pub fn add_worker(
        &self,
        namespace: &str,
        worker_id: WorkerId,
        mdcsum: &str,
        config: Option<ModelRuntimeConfig>,
    ) {
        let pool = self
            .pools
            .entry(namespace.to_string())
            .or_insert_with(|| {
                tracing::info!(
                    namespace,
                    prefix = %self.prefix,
                    "Creating new worker pool"
                );
                Arc::new(WorkerPool::new(namespace.to_string(), mdcsum.to_string()))
            })
            .clone();

        if pool.add_worker(worker_id, config) {
            tracing::debug!(
                namespace,
                worker_id,
                worker_count = pool.worker_count(),
                "Worker added to pool"
            );
            self.notify.notify_waiters();
        }
    }

    /// Remove a worker from its pool.
    ///
    /// If the pool becomes empty, it is automatically removed.
    pub fn remove_worker(&self, namespace: &str, worker_id: WorkerId) {
        if let Some(pool) = self.pools.get(namespace) {
            if pool.remove_worker(worker_id).is_some() {
                tracing::debug!(
                    namespace,
                    worker_id,
                    worker_count = pool.worker_count(),
                    "Worker removed from pool"
                );

                // Auto-remove empty pools
                if pool.is_empty() {
                    drop(pool); // Release the reference before removing
                    if self.pools.remove(namespace).is_some() {
                        tracing::info!(
                            namespace,
                            prefix = %self.prefix,
                            "Removed empty worker pool"
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
        if let Some(pool) = self.pools.get(namespace) {
            pool.update_worker_config(worker_id, config);
        }
    }

    /// Get total instance count across all pools.
    pub fn total_instances(&self) -> usize {
        self.pools.iter().map(|entry| entry.worker_count()).sum()
    }

    /// Get the number of pools.
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }

    /// Check if there are any workers across all pools.
    pub fn has_workers(&self) -> bool {
        self.pools.iter().any(|entry| !entry.is_empty())
    }

    /// Get all pools as a vector.
    pub fn pools(&self) -> Vec<Arc<WorkerPool>> {
        self.pools.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Get a specific pool by namespace.
    pub fn get_pool(&self, namespace: &str) -> Option<Arc<WorkerPool>> {
        self.pools.get(namespace).map(|entry| entry.value().clone())
    }

    /// Select a pool using weighted random selection based on worker counts.
    ///
    /// A pool with 7 workers has 70% chance vs a pool with 3 workers having 30% chance.
    /// Returns None if there are no workers in any pool.
    pub fn select_pool_weighted(&self) -> Option<Arc<WorkerPool>> {
        let total = self.total_instances();
        if total == 0 {
            return None;
        }

        // Weighted random selection
        use rand::Rng;
        let r = rand::rng().random_range(0..total);
        let mut cumulative = 0usize;

        for entry in self.pools.iter() {
            cumulative += entry.worker_count();
            if r < cumulative {
                return Some(entry.value().clone());
            }
        }

        // Fallback (shouldn't happen unless there's a race)
        self.pools.iter().next().map(|e| e.value().clone())
    }

    /// Wait for pool changes (pool added/removed, worker count changed).
    pub async fn wait_for_changes(&self) {
        self.notify.notified().await;
    }

    /// Get workers from all pools combined as a single map.
    ///
    /// This is useful when you need all workers regardless of pool affiliation.
    pub fn all_workers_with_configs(
        &self,
    ) -> std::collections::HashMap<WorkerId, Option<ModelRuntimeConfig>> {
        let mut result = std::collections::HashMap::new();
        for entry in self.pools.iter() {
            result.extend(entry.workers_with_configs());
        }
        result
    }

    /// Get pool weights as a map of namespace to worker count.
    ///
    /// Useful for metrics and debugging.
    pub fn pool_weights(&self) -> std::collections::HashMap<String, usize> {
        self.pools
            .iter()
            .map(|entry| (entry.key().clone(), entry.worker_count()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_pool_manager_basic() {
        let manager = MultiPoolManager::new("default-myapp".to_string());

        assert_eq!(manager.prefix(), "default-myapp");
        assert_eq!(manager.pool_count(), 0);
        assert_eq!(manager.total_instances(), 0);
        assert!(!manager.has_workers());

        // Add workers to different pools
        manager.add_worker("default-myapp-abc", 1, "cs1", None);
        manager.add_worker("default-myapp-abc", 2, "cs1", None);
        manager.add_worker("default-myapp-def", 3, "cs2", None);

        assert_eq!(manager.pool_count(), 2);
        assert_eq!(manager.total_instances(), 3);
        assert!(manager.has_workers());

        // Check pool weights
        let weights = manager.pool_weights();
        assert_eq!(weights.get("default-myapp-abc"), Some(&2));
        assert_eq!(weights.get("default-myapp-def"), Some(&1));
    }

    #[test]
    fn test_multi_pool_manager_auto_remove() {
        let manager = MultiPoolManager::new("prefix".to_string());

        manager.add_worker("prefix-a", 1, "cs", None);
        manager.add_worker("prefix-a", 2, "cs", None);
        assert_eq!(manager.pool_count(), 1);

        // Remove workers - pool should auto-remove when empty
        manager.remove_worker("prefix-a", 1);
        assert_eq!(manager.pool_count(), 1); // Still has one worker

        manager.remove_worker("prefix-a", 2);
        assert_eq!(manager.pool_count(), 0); // Pool removed
        assert!(!manager.has_workers());
    }

    #[test]
    fn test_weighted_selection() {
        let manager = MultiPoolManager::new("p".to_string());

        // Create pools with different sizes
        for i in 0..7 {
            manager.add_worker("p-large", i, "cs", None);
        }
        for i in 10..13 {
            manager.add_worker("p-small", i, "cs", None);
        }

        // Run many selections and check distribution roughly matches weights
        let mut large_count = 0;

        for _ in 0..1000 {
            if let Some(pool) = manager.select_pool_weighted() {
                if pool.namespace() == "p-large" {
                    large_count += 1;
                }
            }
        }

        // Large pool (7/10 = 70%) should be selected more often than small (3/10 = 30%)
        // Allow some variance: large should be at least 60% and at most 80%
        let large_ratio = large_count as f64 / 1000.0;
        assert!(
            large_ratio > 0.6 && large_ratio < 0.8,
            "Large pool ratio {} not within expected range",
            large_ratio
        );
    }
}
