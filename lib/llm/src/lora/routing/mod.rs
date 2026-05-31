// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Allocation Algorithms - HRW and Random
//!
//! Min-Cost Flow placement is implemented separately in [`McfPlacementSolver`]
//! and is not yet wired into the per-LoRA allocator path. It is not exposed
//! as a config value until that integration lands.

use dynamo_kv_router::protocols::WorkerWithDpRank;
use std::collections::HashMap;
use std::str::FromStr;

pub mod hrw;
pub mod mcf_allocator;
pub mod min_cost_flow;
pub mod table;

pub use hrw::RendezvousHasher;
pub use mcf_allocator::{McfPlacementResult, McfPlacementSolver, McfSolveParams};
pub use table::{LoraReplicaConfig, LoraRoutingTable};

/// Trait for LoRA allocation algorithms
pub trait LoraAllocator: Send + Sync {
    /// Returns a list of workers that should host this LoRA, ordered by preference
    fn compute_replica_set(
        &self,
        lora_name: &str,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
    ) -> Vec<WorkerWithDpRank>;

    /// Slot-aware variant: walks the ranked list, skipping workers that are at capacity.
    /// Default implementation falls back to the basic `compute_replica_set`.
    fn compute_replica_set_with_slots(
        &self,
        lora_name: &str,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
        _worker_slot_usage: &HashMap<WorkerWithDpRank, (usize, usize)>,
    ) -> Vec<WorkerWithDpRank> {
        self.compute_replica_set(lora_name, workers, replica_factor)
    }

    /// Name of this algorithm (for logging/metrics)
    fn name(&self) -> &str;
}

/// Per-LoRA allocation algorithm selectable via `DYN_LORA_ALLOCATION_ALGORITHM`.
///
/// `MinCostFlow` selects the global `McfPlacementSolver` churn-aware placement
/// path, which is driven by `LoraController` (see `controller.rs`). The per-LoRA
/// `LoraAllocator` returned by [`create_lora_allocator`] for this variant is only
/// used as a cold-start / fallback allocator; the actual global solve happens in
/// the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationAlgorithmType {
    /// Rendezvous (Highest Random Weight) hashing
    Hrw,
    /// Random selection (for testing)
    Random,
    /// Min-Cost Flow global placement (churn-aware bipartite assignment)
    MinCostFlow,
}

impl FromStr for AllocationAlgorithmType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hrw" => Ok(Self::Hrw),
            "random" => Ok(Self::Random),
            "mcf" | "min_cost_flow" | "mincostflow" => Ok(Self::MinCostFlow),
            _ => Err(format!("Unknown allocation algorithm type: {s}")),
        }
    }
}

/// Create a LoRA allocation algorithm instance.
pub fn create_lora_allocator(algo_type: AllocationAlgorithmType) -> Box<dyn LoraAllocator> {
    match algo_type {
        AllocationAlgorithmType::Hrw => Box::new(RendezvousHasher),
        AllocationAlgorithmType::Random => Box::new(RandomAllocation),
        // MCF uses its own global solver (McfPlacementSolver) in the controller;
        // the per-LoRA allocator here is only a cold-start / fallback path.
        AllocationAlgorithmType::MinCostFlow => Box::new(RendezvousHasher),
    }
}

/// Random allocation algorithm
struct RandomAllocation;

impl LoraAllocator for RandomAllocation {
    fn compute_replica_set(
        &self,
        _lora_name: &str,
        workers: &[WorkerWithDpRank],
        _replica_factor: usize,
    ) -> Vec<WorkerWithDpRank> {
        // Return all workers regardless of replica_factor
        workers.to_vec()
    }

    fn name(&self) -> &str {
        "random"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_lora_allocator() {
        let hrw = create_lora_allocator(AllocationAlgorithmType::Hrw);
        assert_eq!(hrw.name(), "hrw");

        let random = create_lora_allocator(AllocationAlgorithmType::Random);
        assert_eq!(random.name(), "random");
    }

    #[test]
    fn test_mcf_config_string_parses_to_min_cost_flow() {
        // PR4 wires MCF into the controller (McfPlacementSolver), so the config
        // string is now accepted and maps to the MinCostFlow variant.
        for s in &["mcf", "MCF", "min_cost_flow", "mincostflow"] {
            let result = AllocationAlgorithmType::from_str(s);
            assert_eq!(
                result,
                Ok(AllocationAlgorithmType::MinCostFlow),
                "'{s}' should parse to MinCostFlow"
            );
        }
    }

    #[test]
    fn test_random_allocation_basic() {
        let random = RandomAllocation;
        let workers = vec![
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
        ];

        // RandomAllocation returns all workers regardless of replica_factor
        let result = random.compute_replica_set("test-lora", &workers, 2);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].worker_id, 1);
        assert_eq!(result[1].worker_id, 2);
        assert_eq!(result[2].worker_id, 3);
    }
}
