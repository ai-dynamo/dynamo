// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Filter
//!
//! Pre-filters the set of eligible workers for a LoRA request based on the routing
//! table and loaded state.

use std::collections::{HashMap, HashSet};

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::lora::routing::table::LoraRoutingTable;
use crate::lora::state_tracker::LoraStateTracker;

type WorkerId = u64;

/// Filters workers for LoRA-aware routing.
#[derive(Clone)]
pub struct LoraFilter {
    routing_table: LoraRoutingTable,
    state_tracker: LoraStateTracker,
}

impl LoraFilter {
    pub fn new(routing_table: LoraRoutingTable, state_tracker: LoraStateTracker) -> Self {
        Self {
            routing_table,
            state_tracker,
        }
    }

    /// Filter available worker IDs for a LoRA request.
    ///
    /// Logic:
    /// - `lora_name` is None: return all workers (base model request)
    /// - Active entry: prefer loaded workers in replica set, fall back to full replica set
    /// - Inactive entry: return single HRW-pinned worker (cold-start determinism)
    /// - Not in routing table: return all workers (fallback)
    pub fn filter_worker_ids_for_lora(
        &self,
        lora_name: Option<&str>,
        available: &[u64],
    ) -> Vec<u64> {
        let Some(lora_name) = lora_name else {
            return available.to_vec();
        };

        let Some(config) = self.routing_table.get_config(lora_name) else {
            tracing::debug!(
                lora = lora_name,
                "LoRA not in routing table, returning all workers"
            );
            return available.to_vec();
        };

        let replica_id_set: HashSet<u64> = config.replica_set.iter().map(|w| w.worker_id).collect();

        if config.is_active {
            let loaded = self.state_tracker.get_loaded_workers(lora_name);
            let loaded_ids: HashSet<u64> = loaded.iter().map(|w| w.worker_id).collect();

            // Prefer: replica set ∩ loaded ∩ available
            let loaded_in_set: Vec<u64> = available
                .iter()
                .copied()
                .filter(|id| replica_id_set.contains(id) && loaded_ids.contains(id))
                .collect();
            if !loaded_in_set.is_empty() {
                tracing::debug!(
                    lora = lora_name,
                    count = loaded_in_set.len(),
                    "Filtered to loaded workers in replica set"
                );
                return loaded_in_set;
            }

            // Fall back: replica set ∩ available (lazy load)
            let replica_set: Vec<u64> = available
                .iter()
                .copied()
                .filter(|id| replica_id_set.contains(id))
                .collect();
            if !replica_set.is_empty() {
                tracing::debug!(
                    lora = lora_name,
                    count = replica_set.len(),
                    "LoRA not loaded yet, returning full replica set for lazy load"
                );
                return replica_set;
            }

            tracing::warn!(
                lora = lora_name,
                "Replica set workers not available, falling back to all workers"
            );
            available.to_vec()
        } else {
            // Inactive: cold-start pin
            if let Some(pin_id) = config.replica_set.first().map(|w| w.worker_id)
                && available.contains(&pin_id)
            {
                tracing::debug!(
                    lora = lora_name,
                    worker_id = pin_id,
                    "Cold-start: routing to HRW-pinned worker"
                );
                return vec![pin_id];
            }
            tracing::warn!(
                lora = lora_name,
                "Cold-start pin worker not available, falling back to all workers"
            );
            available.to_vec()
        }
    }

    /// Filter workers for a LoRA request (HashMap variant for KV routing).
    pub fn filter_workers_for_lora(
        &self,
        lora_name: Option<&str>,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> HashMap<WorkerId, ModelRuntimeConfig> {
        let available_ids: Vec<u64> = workers.keys().copied().collect();
        let selected_ids = self.filter_worker_ids_for_lora(lora_name, &available_ids);

        if selected_ids.len() == available_ids.len() {
            return workers.clone();
        }

        let selected_set: HashSet<u64> = selected_ids.into_iter().collect();
        workers
            .iter()
            .filter(|(wid, _)| selected_set.contains(wid))
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::protocols::WorkerWithDpRank;
    use crate::lora::routing::table::{LoraReplicaConfig, LoraRoutingTable};
    use crate::lora::state_tracker::LoraStateTracker;
    use crate::model_card::LoraInfo;
    use std::time::Instant;

    fn make_workers_map(ids: &[u64]) -> HashMap<WorkerId, ModelRuntimeConfig> {
        ids.iter()
            .map(|&id| (id, ModelRuntimeConfig::default()))
            .collect()
    }

    fn make_worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn make_lora_info(name: &str) -> LoraInfo {
        LoraInfo {
            name: name.to_string(),
            max_gpu_lora_count: Some(4),
        }
    }

    #[test]
    fn test_no_lora_returns_all_workers() {
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();
        let filter = LoraFilter::new(rt, st);
        let workers = make_workers_map(&[1, 2, 3]);

        let result = filter.filter_workers_for_lora(None, &workers);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_not_in_routing_table_returns_all() {
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();
        let filter = LoraFilter::new(rt, st);
        let workers = make_workers_map(&[1, 2, 3]);

        let result = filter.filter_workers_for_lora(Some("unknown-lora"), &workers);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_active_lora_filters_to_loaded_workers() {
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();

        rt.update_allocation(
            "lora-a".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-a".to_string(),
                replica_factor: 2,
                replica_set: vec![make_worker(1), make_worker(2)],
                updated_at: Instant::now(),
                is_active: true,
            },
        );
        st.handle_mdc_update(make_worker(1), &make_lora_info("lora-a"));

        let filter = LoraFilter::new(rt, st);
        let workers = make_workers_map(&[1, 2, 3]);

        let result = filter.filter_workers_for_lora(Some("lora-a"), &workers);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key(&1));
    }

    #[test]
    fn test_active_lora_falls_back_to_replica_set() {
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();

        rt.update_allocation(
            "lora-a".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-a".to_string(),
                replica_factor: 2,
                replica_set: vec![make_worker(1), make_worker(2)],
                updated_at: Instant::now(),
                is_active: true,
            },
        );

        let filter = LoraFilter::new(rt, st);
        let workers = make_workers_map(&[1, 2, 3]);

        let result = filter.filter_workers_for_lora(Some("lora-a"), &workers);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key(&1));
        assert!(result.contains_key(&2));
    }

    #[test]
    fn test_inactive_lora_cold_start_pin() {
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();

        rt.update_allocation(
            "lora-b".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-b".to_string(),
                replica_factor: 1,
                replica_set: vec![make_worker(2)],
                updated_at: Instant::now(),
                is_active: false,
            },
        );

        let filter = LoraFilter::new(rt, st);
        let workers = make_workers_map(&[1, 2, 3]);

        let result = filter.filter_workers_for_lora(Some("lora-b"), &workers);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key(&2));
    }

    #[test]
    fn test_inactive_pin_worker_unavailable_falls_back() {
        let rt = LoraRoutingTable::new();
        let st = LoraStateTracker::new();

        rt.update_allocation(
            "lora-b".to_string(),
            LoraReplicaConfig {
                lora_name: "lora-b".to_string(),
                replica_factor: 1,
                replica_set: vec![make_worker(5)],
                updated_at: Instant::now(),
                is_active: false,
            },
        );

        let filter = LoraFilter::new(rt, st);
        let workers = make_workers_map(&[1, 2, 3]);

        let result = filter.filter_workers_for_lora(Some("lora-b"), &workers);
        assert_eq!(result.len(), 3);
    }
}
