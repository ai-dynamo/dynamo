// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Set-aware scheduler for multi-set routing.
//!
//! This module provides a scheduler that first selects a worker set based on weighted random
//! selection (by worker count), then delegates to the set's scheduler for KV-aware
//! worker selection within that set.

use std::collections::HashMap;
use std::sync::Arc;

use crate::discovery::{WorkerSetManager, WorkerSet};
use crate::kv_router::{
    KvRouterConfig, WorkerSelector,
    protocols::{WorkerId, WorkerSelectionResult},
    scheduler::{KvSchedulerError, SchedulingRequest},
};
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// A scheduler that supports multi-set routing.
///
/// In multi-set mode:
/// 1. Selects a worker set using weighted random selection (by worker count)
/// 2. Routes within the selected set using the standard scheduling algorithm
///
/// In single-set mode (when set_manager is None), behaves like the regular scheduler.
pub struct SetAwareScheduler {
    /// Multi-set manager (Some when in prefix mode)
    set_manager: Option<Arc<WorkerSetManager>>,

    /// Block size for KV routing
    block_size: u32,

    /// KV router configuration
    kv_router_config: KvRouterConfig,

    /// Custom worker selector (optional)
    selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
}

impl SetAwareScheduler {
    /// Create a new set-aware scheduler.
    ///
    /// If `set_manager` is Some, enables multi-set routing.
    /// If None, this scheduler won't be used (caller should use regular scheduler).
    pub fn new(
        set_manager: Option<Arc<WorkerSetManager>>,
        block_size: u32,
        kv_router_config: KvRouterConfig,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
    ) -> Self {
        Self {
            set_manager,
            block_size,
            kv_router_config,
            selector,
        }
    }

    /// Check if multi-set mode is enabled.
    pub fn is_multi_set(&self) -> bool {
        self.set_manager.is_some()
    }

    /// Select a worker set using weighted random selection.
    ///
    /// Returns None if no sets have workers.
    pub fn select_set(&self) -> Option<Arc<WorkerSet>> {
        self.set_manager.as_ref()?.select_weighted()
    }

    /// Get the total number of workers across all sets.
    pub fn total_workers(&self) -> usize {
        self.set_manager
            .as_ref()
            .map(|sm| sm.total_instances())
            .unwrap_or(0)
    }

    /// Get set weights for metrics/debugging.
    pub fn set_weights(&self) -> HashMap<String, usize> {
        self.set_manager
            .as_ref()
            .map(|sm| sm.set_weights())
            .unwrap_or_default()
    }

    /// Get all workers from all sets (for fallback or non-set-aware routing).
    pub fn all_workers_with_configs(&self) -> HashMap<WorkerId, Option<ModelRuntimeConfig>> {
        self.set_manager
            .as_ref()
            .map(|sm| sm.all_workers_with_configs())
            .unwrap_or_default()
    }

    /// Select a worker using the multi-set algorithm:
    /// 1. Select worker set weighted by instance count
    /// 2. Use standard worker selection within that set
    ///
    /// For KV routing, the caller should also use the set's KvIndexer for cache-aware selection.
    pub fn select_worker_from_set(
        &self,
        request: &SchedulingRequest,
    ) -> Result<(Arc<WorkerSet>, WorkerSelectionResult), KvSchedulerError> {
        let set = self
            .select_set()
            .ok_or(KvSchedulerError::NoEndpoints)?;

        let workers = set.workers_with_configs();
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

        Ok((set, result))
    }

    /// Select a random worker from a weighted-selected set.
    ///
    /// Used for random load balancing with set awareness.
    pub fn select_random_from_set(&self) -> Option<(Arc<WorkerSet>, WorkerId)> {
        let set = self.select_set()?;
        let worker_id = set.select_random()?;
        Some((set, worker_id))
    }

    /// Select a worker using round-robin from a weighted-selected set.
    ///
    /// Used for round-robin load balancing with set awareness.
    pub fn select_round_robin_from_set(&self) -> Option<(Arc<WorkerSet>, WorkerId)> {
        let set = self.select_set()?;
        let worker_id = set.select_round_robin()?;
        Some((set, worker_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_aware_scheduler_no_sets() {
        let scheduler = SetAwareScheduler::new(None, 16, KvRouterConfig::default(), None);

        assert!(!scheduler.is_multi_set());
        assert_eq!(scheduler.total_workers(), 0);
        assert!(scheduler.select_set().is_none());
    }

    #[test]
    fn test_set_aware_scheduler_with_sets() {
        let manager = Arc::new(WorkerSetManager::new("prefix".to_string()));

        // Add workers to sets
        manager.add_worker("prefix-a", 1, "cs1", None);
        manager.add_worker("prefix-a", 2, "cs1", None);
        manager.add_worker("prefix-b", 3, "cs2", None);

        let scheduler = SetAwareScheduler::new(
            Some(manager.clone()),
            16,
            KvRouterConfig::default(),
            None,
        );

        assert!(scheduler.is_multi_set());
        assert_eq!(scheduler.total_workers(), 3);

        // Set weights should be correct
        let weights = scheduler.set_weights();
        assert_eq!(weights.get("prefix-a"), Some(&2));
        assert_eq!(weights.get("prefix-b"), Some(&1));

        // Select set should work
        let set = scheduler.select_set();
        assert!(set.is_some());
    }

    #[test]
    fn test_random_and_round_robin_selection() {
        let manager = Arc::new(WorkerSetManager::new("p".to_string()));
        manager.add_worker("p-a", 10, "cs", None);
        manager.add_worker("p-a", 20, "cs", None);

        let scheduler = SetAwareScheduler::new(
            Some(manager),
            16,
            KvRouterConfig::default(),
            None,
        );

        // Random selection should return a worker
        let (set, worker) = scheduler.select_random_from_set().unwrap();
        assert_eq!(set.namespace(), "p-a");
        assert!(worker == 10 || worker == 20);

        // Round-robin selection should return workers in sequence
        let (_, w1) = scheduler.select_round_robin_from_set().unwrap();
        let (_, w2) = scheduler.select_round_robin_from_set().unwrap();
        assert!(w1 != w2 || set.worker_count() == 1);
    }
}
