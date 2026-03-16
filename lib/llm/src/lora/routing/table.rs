// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Routing Table - Thread-safe data structure for storing LoRA allocation decisions.

use dashmap::DashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::kv_router::protocols::WorkerWithDpRank;

/// Configuration for a single LoRA's allocation
#[derive(Debug, Clone)]
pub struct LoraReplicaConfig {
    /// Name of the LoRA adapter
    pub lora_name: String,

    /// Number of replicas configured
    pub replica_factor: usize,

    /// Workers selected to host this LoRA (in preference order)
    pub replica_set: Vec<WorkerWithDpRank>,

    /// When this allocation was last updated
    pub updated_at: Instant,
}

/// Thread-safe allocation table using DashMap for concurrent access
#[derive(Clone)]
pub struct LoraRoutingTable {
    allocations: Arc<DashMap<String, LoraReplicaConfig>>,
}

impl LoraRoutingTable {
    /// Create a new empty allocation table
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(DashMap::new()),
        }
    }

    /// Get the replica set for a LoRA
    pub fn get_replica_set(&self, lora_name: &str) -> Option<Vec<WorkerWithDpRank>> {
        self.allocations
            .get(lora_name)
            .map(|entry| entry.replica_set.clone())
    }

    /// Get the full configuration for a LoRA
    pub fn get_config(&self, lora_name: &str) -> Option<LoraReplicaConfig> {
        self.allocations.get(lora_name).map(|entry| entry.clone())
    }

    /// Update or insert an allocation configuration
    pub fn update_allocation(&self, lora_name: String, config: LoraReplicaConfig) {
        self.allocations.insert(lora_name, config);
    }

    /// Remove a LoRA from the allocation table
    pub fn remove_lora(&self, lora_name: &str) -> Option<LoraReplicaConfig> {
        self.allocations.remove(lora_name).map(|(_, v)| v)
    }

    /// List all LoRA names in the allocation table
    pub fn list_loras(&self) -> Vec<String> {
        self.allocations
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get the number of LoRAs in the allocation table
    pub fn len(&self) -> usize {
        self.allocations.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Clear all entries from the table
    pub fn clear(&self) {
        self.allocations.clear();
    }
}

impl Default for LoraRoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workers(count: usize) -> Vec<WorkerWithDpRank> {
        (0..count)
            .map(|i| WorkerWithDpRank::new(i as u64, 0))
            .collect()
    }

    #[test]
    fn test_new_table_is_empty() {
        let table = LoraRoutingTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.list_loras().len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let table = LoraRoutingTable::new();
        let workers = make_workers(3);

        let config = LoraReplicaConfig {
            lora_name: "test-lora".to_string(),
            replica_factor: 2,
            replica_set: workers[..2].to_vec(),
            updated_at: Instant::now(),
        };

        table.update_allocation("test-lora".to_string(), config);

        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        let replica_set = table.get_replica_set("test-lora").unwrap();
        assert_eq!(replica_set.len(), 2);
        assert_eq!(replica_set[0].worker_id, 0);
        assert_eq!(replica_set[1].worker_id, 1);
    }

    #[test]
    fn test_get_nonexistent() {
        let table = LoraRoutingTable::new();
        assert!(table.get_replica_set("nonexistent").is_none());
        assert!(table.get_config("nonexistent").is_none());
    }

    #[test]
    fn test_update_existing() {
        let table = LoraRoutingTable::new();
        let workers = make_workers(3);

        // Insert initial config
        let config1 = LoraReplicaConfig {
            lora_name: "test-lora".to_string(),
            replica_factor: 1,
            replica_set: workers[..1].to_vec(),
            updated_at: Instant::now(),
        };
        table.update_allocation("test-lora".to_string(), config1);

        // Update with new config
        let config2 = LoraReplicaConfig {
            lora_name: "test-lora".to_string(),
            replica_factor: 2,
            replica_set: workers[..2].to_vec(),
            updated_at: Instant::now(),
        };
        table.update_allocation("test-lora".to_string(), config2);

        // Should have new config
        assert_eq!(table.len(), 1);
        let replica_set = table.get_replica_set("test-lora").unwrap();
        assert_eq!(replica_set.len(), 2);
    }

    #[test]
    fn test_remove() {
        let table = LoraRoutingTable::new();
        let workers = make_workers(1);

        let config = LoraReplicaConfig {
            lora_name: "test-lora".to_string(),
            replica_factor: 1,
            replica_set: workers.clone(),
            updated_at: Instant::now(),
        };
        table.update_allocation("test-lora".to_string(), config);

        assert_eq!(table.len(), 1);

        let removed = table.remove_lora("test-lora");
        assert!(removed.is_some());
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_list_loras() {
        let table = LoraRoutingTable::new();
        let workers = make_workers(1);

        for i in 0..3 {
            let config = LoraReplicaConfig {
                lora_name: format!("lora-{}", i),
                replica_factor: 1,
                replica_set: workers.clone(),
                updated_at: Instant::now(),
            };
            table.update_allocation(format!("lora-{}", i), config);
        }

        let loras = table.list_loras();
        assert_eq!(loras.len(), 3);
        assert!(loras.contains(&"lora-0".to_string()));
        assert!(loras.contains(&"lora-1".to_string()));
        assert!(loras.contains(&"lora-2".to_string()));
    }

    #[test]
    fn test_clear() {
        let table = LoraRoutingTable::new();
        let workers = make_workers(1);

        for i in 0..3 {
            let config = LoraReplicaConfig {
                lora_name: format!("lora-{}", i),
                replica_factor: 1,
                replica_set: workers.clone(),
                updated_at: Instant::now(),
            };
            table.update_allocation(format!("lora-{}", i), config);
        }

        assert_eq!(table.len(), 3);
        table.clear();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }
}
