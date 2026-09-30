// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Routing Table - Thread-safe data structure for storing LoRA allocation decisions.

use dashmap::DashMap;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use std::sync::Arc;
use std::time::Instant;

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

    /// Whether this LoRA has active load (true) or is a cold-start pin (false).
    pub is_active: bool,
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

    /// Snapshot all (name, config) pairs in a single DashMap pass.
    pub fn snapshot_configs(&self) -> Vec<(String, LoraReplicaConfig)> {
        self.allocations
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
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
