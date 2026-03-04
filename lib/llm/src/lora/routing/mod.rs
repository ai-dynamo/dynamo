// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Allocation Algorithms - HRW and Random

use crate::kv_router::protocols::WorkerWithDpRank;
use std::str::FromStr;

pub mod hrw;
pub mod table;

pub use hrw::RendezvousHasher;
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

    /// Name of this algorithm (for logging/metrics)
    fn name(&self) -> &str;
}

/// Factory for creating allocation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationAlgorithmType {
    /// Rendezvous (Highest Random Weight) hashing
    Hrw,
    /// Random selection (for testing)
    Random,
}

impl FromStr for AllocationAlgorithmType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hrw" => Ok(Self::Hrw),
            "random" => Ok(Self::Random),
            _ => Err(format!("Unknown allocation algorithm type: {}", s)),
        }
    }
}

/// Create a LoRA allocation algorithm instance
pub fn create_lora_allocator(algo_type: AllocationAlgorithmType) -> Box<dyn LoraAllocator> {
    match algo_type {
        AllocationAlgorithmType::Hrw => Box::new(RendezvousHasher),
        AllocationAlgorithmType::Random => Box::new(RandomAllocation),
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
