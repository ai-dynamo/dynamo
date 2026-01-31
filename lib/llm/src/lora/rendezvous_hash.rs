// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Implements Highest Random Weight (HRW) / Rendezvous hashing,
//! for deterministic and stable LORA-to-server assignment.

use blake3;

use crate::kv_router::protocols::WorkerWithDpRank;

pub struct RendezvousHasher;

impl RendezvousHasher {
    /// Compute a deterministic score for a LORA-worker pair
    pub fn compute_score(lora_name: &str, worker: WorkerWithDpRank) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(lora_name.as_bytes());
        hasher.update(&worker.worker_id.to_le_bytes());
        hasher.update(&worker.dp_rank.to_le_bytes());
        let hash = hasher.finalize();

        // Extract first 8 bytes as u64
        let hash_bytes = hash.as_bytes();
        let mut bytes_array = [0u8; 8];
        bytes_array.copy_from_slice(&hash_bytes[..8]);
        u64::from_le_bytes(bytes_array)
    }

    /// Rank all workers by their score for a given LORA
    pub fn rank_workers(
        lora_name: &str,
        workers: &[WorkerWithDpRank],
    ) -> Vec<(WorkerWithDpRank, u64)> {
        let mut ranked: Vec<_> = workers
            .iter()
            .map(|&worker| {
                let score = Self::compute_score(lora_name, worker);
                (worker, score)
            })
            .collect();

        // Sort by score descending (highest first)
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked
    }
}

/// Compute the replica set for a LORA using Rendezvous hashing
pub fn compute_replica_set(
    lora_name: &str,
    workers: &[WorkerWithDpRank],
    replica_factor: usize,
) -> Vec<WorkerWithDpRank> {
    if replica_factor == 0 {
        return vec![];
    }

    if replica_factor > workers.len() {
        tracing::warn!(
            "Replica factor {} exceeds available workers {}, using all workers",
            replica_factor,
            workers.len()
        );
        return workers.to_vec();
    }

    let ranked = RendezvousHasher::rank_workers(lora_name, workers);

    ranked
        .into_iter()
        .take(replica_factor)
        .map(|(worker, _score)| worker)
        .collect()
}

/// Compute replica factor based on demand estimate
/// Uses linear interpolation: replicas = min + (num_workers - min) × demand
///
/// The maximum number of replicas is automatically determined from the number
/// of available workers, automatically scaling with cluster size.
pub fn compute_replica_factor(
    demand_estimate: f64,
    min_replicas: usize,
    num_workers: usize,
) -> usize {
    if min_replicas >= num_workers {
        return min_replicas;
    }

    let demand_clamped = demand_estimate.clamp(0.0, 1.0);
    let range = (num_workers - min_replicas) as f64;
    let factor = min_replicas + (range * demand_clamped).round() as usize;

    factor.clamp(min_replicas, num_workers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_score_deterministic() {
        let worker = WorkerWithDpRank::new(1, 0);
        let score1 = RendezvousHasher::compute_score("lora-a", worker);
        let score2 = RendezvousHasher::compute_score("lora-a", worker);
        assert_eq!(score1, score2, "Same inputs should produce same score");
    }

    #[test]
    fn test_compute_score_different_loras() {
        let worker = WorkerWithDpRank::new(1, 0);
        let score_a = RendezvousHasher::compute_score("lora-a", worker);
        let score_b = RendezvousHasher::compute_score("lora-b", worker);
        assert_ne!(
            score_a, score_b,
            "Different LORAs should produce different scores"
        );
    }

    #[test]
    fn test_compute_score_different_workers() {
        let worker1 = WorkerWithDpRank::new(1, 0);
        let worker2 = WorkerWithDpRank::new(2, 0);
        let score1 = RendezvousHasher::compute_score("lora-a", worker1);
        let score2 = RendezvousHasher::compute_score("lora-a", worker2);
        assert_ne!(
            score1, score2,
            "Different workers should produce different scores"
        );
    }

    #[test]
    fn test_rank_workers_sorted() {
        let workers = vec![
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
            WorkerWithDpRank::new(4, 0),
        ];

        let ranked = RendezvousHasher::rank_workers("lora-a", &workers);

        assert_eq!(ranked.len(), 4);

        // Verify scores are in descending order
        for i in 0..ranked.len() - 1 {
            assert!(
                ranked[i].1 >= ranked[i + 1].1,
                "Scores should be in descending order"
            );
        }
    }

    #[test]
    fn test_compute_replica_set_basic() {
        let workers = vec![
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
            WorkerWithDpRank::new(4, 0),
        ];

        let replica_set = compute_replica_set("lora-a", &workers, 2);

        assert_eq!(replica_set.len(), 2, "Should return 2 replicas");

        // Verify all returned workers are in the original list
        for worker in &replica_set {
            assert!(
                workers.contains(worker),
                "Replica should be from worker list"
            );
        }
    }

    #[test]
    fn test_compute_replica_set_deterministic() {
        let workers = vec![
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
        ];

        let set1 = compute_replica_set("lora-a", &workers, 2);
        let set2 = compute_replica_set("lora-a", &workers, 2);

        assert_eq!(set1, set2, "Same inputs should produce same replica set");
    }

    #[test]
    fn test_compute_replica_set_stability() {
        // Test stability: adding a worker should minimally affect existing placements
        let workers_3 = vec![
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
        ];

        let workers_4 = vec![
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
            WorkerWithDpRank::new(3, 0),
            WorkerWithDpRank::new(4, 0),
        ];

        let set_before = compute_replica_set("lora-a", &workers_3, 2);
        let set_after = compute_replica_set("lora-a", &workers_4, 2);

        // Count how many workers remain in the set
        let overlap = set_before.iter().filter(|w| set_after.contains(w)).count();

        // Expect at least some overlap (ideally 100%, but depends on hash)
        // In practice with Rendezvous hashing, we often see high stability
        assert!(
            overlap >= 1,
            "Adding a worker should preserve at least one existing placement"
        );
    }

    #[test]
    fn test_compute_replica_set_zero_replicas() {
        let workers = vec![WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)];
        let replica_set = compute_replica_set("lora-a", &workers, 0);
        assert_eq!(
            replica_set.len(),
            0,
            "Zero replicas should return empty set"
        );
    }

    #[test]
    fn test_compute_replica_set_exceeds_workers() {
        let workers = vec![WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)];
        let replica_set = compute_replica_set("lora-a", &workers, 5);
        assert_eq!(
            replica_set.len(),
            2,
            "Should cap at number of available workers"
        );
    }

    #[test]
    fn test_compute_replica_factor_low_demand() {
        let factor = compute_replica_factor(0.1, 1, 4);
        assert_eq!(factor, 1, "Low demand should give minimum replicas");
    }

    #[test]
    fn test_compute_replica_factor_medium_demand() {
        let factor = compute_replica_factor(0.5, 1, 4);
        // 1 + round((4-1) * 0.5) = 1 + round(1.5) = 1 + 2 = 3
        assert_eq!(factor, 3, "Medium demand should interpolate");
    }

    #[test]
    fn test_compute_replica_factor_high_demand() {
        let factor = compute_replica_factor(0.9, 1, 4);
        // 1 + (4-1) * 0.9 = 1 + 2.7 = 3.7 -> rounds to 4
        assert_eq!(factor, 4, "High demand should give maximum replicas");
    }

    #[test]
    fn test_compute_replica_factor_bounds() {
        // Below 0 should clamp to 0
        assert_eq!(compute_replica_factor(-0.5, 1, 4), 1);

        // Above 1 should clamp to 1
        assert_eq!(compute_replica_factor(1.5, 1, 4), 4);
    }

    #[test]
    fn test_compute_replica_factor_min_equals_max() {
        let factor = compute_replica_factor(0.5, 3, 3);
        assert_eq!(factor, 3, "When min equals max, should return that value");
    }

    #[test]
    fn test_compute_replica_factor_min_greater_than_max() {
        let factor = compute_replica_factor(0.5, 5, 3);
        assert_eq!(factor, 5, "When min > max, should return min");
    }
}
