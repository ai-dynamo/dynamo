// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pool-aware scheduler for multi-pool routing.
//!
//! This module provides a scheduler that first selects a pool based on weighted random
//! selection (by worker count), then delegates to the pool's scheduler for KV-aware
//! worker selection within that pool.

use std::collections::HashMap;
use std::sync::Arc;

use crate::discovery::{WorkerSetManager, WorkerSet};
use crate::kv_router::{
    KvRouterConfig, WorkerSelector,
    protocols::{WorkerId, WorkerSelectionResult},
    scheduler::{KvSchedulerError, SchedulingRequest},
};
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// A scheduler that supports multi-pool routing.
///
/// In multi-pool mode:
/// 1. Selects a pool using weighted random selection (by worker count)
/// 2. Routes within the selected pool using the standard scheduling algorithm
///
/// In single-pool mode (when pool_manager is None), behaves like the regular scheduler.
pub struct PoolAwareScheduler {
    /// Multi-pool manager (Some when in prefix mode)
    pool_manager: Option<Arc<WorkerSetManager>>,

    /// Block size for KV routing
    block_size: u32,

    /// KV router configuration
    kv_router_config: KvRouterConfig,

    /// Custom worker selector (optional)
    selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
}

impl PoolAwareScheduler {
    /// Create a new pool-aware scheduler.
    ///
    /// If `pool_manager` is Some, enables multi-pool routing.
    /// If None, this scheduler won't be used (caller should use regular scheduler).
    pub fn new(
        pool_manager: Option<Arc<WorkerSetManager>>,
        block_size: u32,
        kv_router_config: KvRouterConfig,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
    ) -> Self {
        Self {
            pool_manager,
            block_size,
            kv_router_config,
            selector,
        }
    }

    /// Check if multi-pool mode is enabled.
    pub fn is_multi_pool(&self) -> bool {
        self.pool_manager.is_some()
    }

    /// Select a pool using weighted random selection.
    ///
    /// Returns None if no pools have workers.
    pub fn select_pool(&self) -> Option<Arc<WorkerSet>> {
        self.pool_manager.as_ref()?.select_weighted()
    }

    /// Get the total number of workers across all pools.
    pub fn total_workers(&self) -> usize {
        self.pool_manager
            .as_ref()
            .map(|pm| pm.total_instances())
            .unwrap_or(0)
    }

    /// Get pool weights for metrics/debugging.
    pub fn set_weights(&self) -> HashMap<String, usize> {
        self.pool_manager
            .as_ref()
            .map(|pm| pm.set_weights())
            .unwrap_or_default()
    }

    /// Get all workers from all pools (for fallback or non-pool-aware routing).
    pub fn all_workers_with_configs(&self) -> HashMap<WorkerId, Option<ModelRuntimeConfig>> {
        self.pool_manager
            .as_ref()
            .map(|pm| pm.all_workers_with_configs())
            .unwrap_or_default()
    }

    /// Select a worker using the multi-pool algorithm:
    /// 1. Select pool weighted by instance count
    /// 2. Use standard worker selection within that pool
    ///
    /// For KV routing, the caller should also use the pool's KvIndexer for cache-aware selection.
    pub fn select_worker_from_pool(
        &self,
        request: &SchedulingRequest,
    ) -> Result<(Arc<WorkerSet>, WorkerSelectionResult), KvSchedulerError> {
        let pool = self
            .select_pool()
            .ok_or(KvSchedulerError::NoEndpoints)?;

        let workers = pool.workers_with_configs();
        if workers.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        // Use custom selector if provided, otherwise use default logic
        let result = if let Some(ref selector) = self.selector {
            selector.select_worker(&workers, request, self.block_size)?
        } else {
            // Default: use the DefaultWorkerSelector logic
            use crate::kv_router::scheduler::DefaultWorkerSelector;
            let default_selector = DefaultWorkerSelector {
                kv_router_config: self.kv_router_config,
            };
            default_selector.select_worker(&workers, request, self.block_size)?
        };

        Ok((pool, result))
    }

    /// Select a random worker from a weighted-selected pool.
    ///
    /// Used for random load balancing with pool awareness.
    pub fn select_random_from_pool(&self) -> Option<(Arc<WorkerSet>, WorkerId)> {
        let pool = self.select_pool()?;
        let worker_id = pool.select_random()?;
        Some((pool, worker_id))
    }

    /// Select a worker using round-robin from a weighted-selected pool.
    ///
    /// Used for round-robin load balancing with pool awareness.
    pub fn select_round_robin_from_pool(&self) -> Option<(Arc<WorkerSet>, WorkerId)> {
        let pool = self.select_pool()?;
        let worker_id = pool.select_round_robin()?;
        Some((pool, worker_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_aware_scheduler_no_pools() {
        let scheduler = PoolAwareScheduler::new(None, 16, KvRouterConfig::default(), None);

        assert!(!scheduler.is_multi_pool());
        assert_eq!(scheduler.total_workers(), 0);
        assert!(scheduler.select_pool().is_none());
    }

    #[test]
    fn test_pool_aware_scheduler_with_pools() {
        let manager = Arc::new(WorkerSetManager::new("prefix".to_string()));

        // Add workers to pools
        manager.add_worker("prefix-a", 1, "cs1", None);
        manager.add_worker("prefix-a", 2, "cs1", None);
        manager.add_worker("prefix-b", 3, "cs2", None);

        let scheduler = PoolAwareScheduler::new(
            Some(manager.clone()),
            16,
            KvRouterConfig::default(),
            None,
        );

        assert!(scheduler.is_multi_pool());
        assert_eq!(scheduler.total_workers(), 3);

        // Pool weights should be correct
        let weights = scheduler.set_weights();
        assert_eq!(weights.get("prefix-a"), Some(&2));
        assert_eq!(weights.get("prefix-b"), Some(&1));

        // Select pool should work
        let pool = scheduler.select_pool();
        assert!(pool.is_some());
    }

    #[test]
    fn test_random_and_round_robin_selection() {
        let manager = Arc::new(WorkerSetManager::new("p".to_string()));
        manager.add_worker("p-a", 10, "cs", None);
        manager.add_worker("p-a", 20, "cs", None);

        let scheduler = PoolAwareScheduler::new(
            Some(manager),
            16,
            KvRouterConfig::default(),
            None,
        );

        // Random selection should return a worker
        let (pool, worker) = scheduler.select_random_from_pool().unwrap();
        assert_eq!(pool.namespace(), "p-a");
        assert!(worker == 10 || worker == 20);

        // Round-robin selection should return workers in sequence
        let (_, w1) = scheduler.select_round_robin_from_pool().unwrap();
        let (_, w2) = scheduler.select_round_robin_from_pool().unwrap();
        assert!(w1 != w2 || pool.worker_count() == 1);
    }
}
