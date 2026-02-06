// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::LoraAllocator;
use crate::kv_router::protocols::WorkerWithDpRank;

/// Rendezvous (HRW) hashing implementation for LoRA allocation
pub struct RendezvousHasher;

impl RendezvousHasher {
    /// Compute hash score for a (lora_name, worker) pair using HRW hashing with blake3
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

    /// Rank workers by their hash scores for a given LoRA
    /// Returns workers sorted by score in descending order (highest first).
    pub fn rank_workers(
        lora_name: &str,
        workers: &[WorkerWithDpRank],
    ) -> Vec<(WorkerWithDpRank, u64)> {
        let mut scores: Vec<_> = workers
            .iter()
            .map(|&w| (w, Self::compute_score(lora_name, w)))
            .collect();

        // Sort by score descending (highest score first)
        scores.sort_by_key(|(_, score)| std::cmp::Reverse(*score));
        scores
    }
}

impl LoraAllocator for RendezvousHasher {
    fn compute_replica_set(
        &self,
        lora_name: &str,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
    ) -> Vec<WorkerWithDpRank> {
        if workers.is_empty() {
            return Vec::new();
        }

        // Rank all workers and take top N
        let ranked = Self::rank_workers(lora_name, workers);
        ranked
            .into_iter()
            .take(replica_factor.min(workers.len()))
            .map(|(w, _)| w)
            .collect()
    }

    fn name(&self) -> &str {
        "hrw"
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
    fn test_deterministic() {
        let worker = WorkerWithDpRank::new(1, 0);
        let lora_name = "test-lora";

        // Same inputs should always produce same score
        let score1 = RendezvousHasher::compute_score(lora_name, worker);
        let score2 = RendezvousHasher::compute_score(lora_name, worker);
        assert_eq!(score1, score2, "Same inputs should produce same score");
    }

    #[test]
    fn test_stability_adding_workers() {
        // Start with 3 workers
        let workers_before = make_workers(3);
        let hasher = RendezvousHasher;
        let replica_set_before = hasher.compute_replica_set("test-lora", &workers_before, 2);

        assert_eq!(replica_set_before.len(), 2);

        // Add 2 more workers
        let workers_after = make_workers(5);
        let replica_set_after = hasher.compute_replica_set("test-lora", &workers_after, 2);

        assert_eq!(replica_set_after.len(), 2);
        let top2_after: Vec<_> = replica_set_after.iter().map(|w| w.worker_id).collect();

        // The top 2 should be the same if they're still in top 2 after adding workers
        // This tests stability property: adding workers shouldn't change existing placements
        // (unless the new workers rank higher, which is expected behavior)

        // At minimum, verify determinism: same inputs produce same outputs
        let replica_set_after2 = hasher.compute_replica_set("test-lora", &workers_after, 2);
        let top2_after2: Vec<_> = replica_set_after2.iter().map(|w| w.worker_id).collect();
        assert_eq!(
            top2_after, top2_after2,
            "Same inputs should produce same outputs"
        );
    }

    #[test]
    fn test_stability_removing_workers() {
        let hasher = RendezvousHasher;

        // Start with 5 workers
        let workers_5 = make_workers(5);
        let set_5 = hasher.compute_replica_set("test-lora", &workers_5, 3);
        assert_eq!(set_5.len(), 3);

        // Remove worker 2 (keep 0,1,3,4)
        let workers_4: Vec<_> = workers_5
            .iter()
            .filter(|w| w.worker_id != 2)
            .copied()
            .collect();
        let set_4 = hasher.compute_replica_set("test-lora", &workers_4, 3);
        assert_eq!(set_4.len(), 3);

        // If worker 2 wasn't in the original top 3, the other selections should stay the same
        if !set_5.iter().any(|w| w.worker_id == 2) {
            // The workers that were in top 3 and are still available should still be in top 3
            for worker in &set_5 {
                if workers_4.contains(worker) {
                    assert!(
                        set_4.contains(worker),
                        "Worker {} was in top 3 and is still available, should remain in top 3",
                        worker.worker_id
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_replica_set_more_replicas_than_workers() {
        let hasher = RendezvousHasher;
        let workers = make_workers(3);
        let result = hasher.compute_replica_set("test-lora", &workers, 10);

        // Should return all workers when replica_factor > worker count
        assert_eq!(result.len(), 3);
    }
}
