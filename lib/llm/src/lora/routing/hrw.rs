// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use super::LoraAllocator;
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Rendezvous (HRW) hashing implementation for LoRA allocation
pub struct RendezvousHasher;

impl RendezvousHasher {
    /// Compute hash score for a (lora_name, worker) pair using HRW hashing with blake3
    pub fn compute_score(lora_name: &str, worker: WorkerWithDpRank) -> u64 {
        dynamo_kv_router::rendezvous::legacy_worker_score(lora_name, worker)
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

        // Sort by score descending (highest score first). Break score ties deterministically by
        // (worker_id, dp_rank) so that equal HRW scores can never produce different rankings
        // across router instances — worker input order is not guaranteed (it can come from
        // DashMap iteration), and determinism is a hard requirement for coordination-free
        // placement.
        scores.sort_by(|(wa, sa), (wb, sb)| {
            sb.cmp(sa)
                .then_with(|| wa.worker_id.cmp(&wb.worker_id))
                .then_with(|| wa.dp_rank.cmp(&wb.dp_rank))
        });
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

    fn compute_replica_set_with_slots(
        &self,
        lora_name: &str,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
        worker_slot_usage: &HashMap<WorkerWithDpRank, (usize, usize)>,
    ) -> Vec<WorkerWithDpRank> {
        if workers.is_empty() || replica_factor == 0 {
            return Vec::new();
        }

        let ranked = Self::rank_workers(lora_name, workers);
        let mut result = Vec::with_capacity(replica_factor);

        for (worker, _score) in &ranked {
            if result.len() >= replica_factor {
                break;
            }
            if let Some(&(used, cap)) = worker_slot_usage.get(worker)
                && used >= cap
            {
                continue;
            }
            result.push(*worker);
        }

        // We placed on every non-full worker up to `replica_factor`. If that under-filled
        // the count, accept the smaller set rather than placing on at-capacity workers
        // (REQ 6: skip full workers). Only when EVERY worker is full (result is empty) do we
        // fall back to basic HRW, so the LoRA still has at least one routable home (REQ 7).
        if result.is_empty() {
            return self.compute_replica_set(lora_name, workers, replica_factor);
        }

        result
    }

    fn compute_replica_set_with_slots_sticky(
        &self,
        lora_name: &str,
        workers: &[WorkerWithDpRank],
        replica_factor: usize,
        worker_slot_usage: &HashMap<WorkerWithDpRank, (usize, usize)>,
        prior: &[WorkerWithDpRank],
    ) -> Vec<WorkerWithDpRank> {
        if workers.is_empty() || replica_factor == 0 {
            return Vec::new();
        }

        let ranked = Self::rank_workers(lora_name, workers);
        let prior_set: HashSet<WorkerWithDpRank> = prior.iter().copied().collect();
        let mut result = Vec::with_capacity(replica_factor);
        let mut chosen: HashSet<WorkerWithDpRank> = HashSet::new();

        let is_full = |w: &WorkerWithDpRank| {
            worker_slot_usage
                .get(w)
                .map(|&(used, cap)| used >= cap)
                .unwrap_or(false)
        };

        // Pass 1 (stickiness): retain workers the LoRA already occupies, in HRW-ranked order,
        // as long as they are still present and not full. This anchors a stable LoRA on its
        // existing placement so flickering sibling activity within the tick cannot move it.
        for (worker, _score) in &ranked {
            if result.len() >= replica_factor {
                break;
            }
            if prior_set.contains(worker) && !is_full(worker) {
                result.push(*worker);
                chosen.insert(*worker);
            }
        }

        // Pass 2: fill any remaining slots from the HRW-ranked list, skipping full and
        // already-chosen workers (REQ 6: skip workers at capacity).
        for (worker, _score) in &ranked {
            if result.len() >= replica_factor {
                break;
            }
            if chosen.contains(worker) || is_full(worker) {
                continue;
            }
            result.push(*worker);
            chosen.insert(*worker);
        }

        // Only when EVERY worker is full do we fall back to basic HRW so the LoRA keeps at
        // least one routable home (REQ 7).
        if result.is_empty() {
            return self.compute_replica_set(lora_name, workers, replica_factor);
        }

        result
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

    #[test]
    fn test_slot_aware_skips_full_workers() {
        let hasher = RendezvousHasher;
        let workers = make_workers(5);

        let ranked = RendezvousHasher::rank_workers("test-lora", &workers);
        let mut usage = HashMap::new();
        for (w, _) in ranked.iter().take(2) {
            usage.insert(*w, (4usize, 4usize));
        }
        for (w, _) in ranked.iter().skip(2) {
            usage.insert(*w, (1usize, 4usize));
        }

        let result = hasher.compute_replica_set_with_slots("test-lora", &workers, 2, &usage);

        assert_eq!(result.len(), 2);
        for w in &result {
            let (used, cap) = usage[w];
            assert!(used < cap, "Selected worker {:?} should not be full", w);
        }
    }

    #[test]
    fn test_slot_aware_falls_back_when_all_full() {
        let hasher = RendezvousHasher;
        let workers = make_workers(3);

        let mut usage = HashMap::new();
        for w in &workers {
            usage.insert(*w, (4usize, 4usize));
        }

        let result = hasher.compute_replica_set_with_slots("test-lora", &workers, 2, &usage);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_ranking_equal_scores_tie_break_is_deterministic() {
        // Score ties must break by (worker_id, dp_rank), independent of input order, so all
        // router instances converge on the same ranking even when worker input order varies.
        let workers_fwd = vec![
            WorkerWithDpRank::new(10, 0),
            WorkerWithDpRank::new(10, 1),
            WorkerWithDpRank::new(7, 0),
            WorkerWithDpRank::new(7, 1),
        ];
        let mut workers_rev = workers_fwd.clone();
        workers_rev.reverse();

        // Force an all-ties scenario by ranking with a constant score via identical names is not
        // possible (score depends on worker), so instead assert that ranking a set and its reverse
        // yields identical order — the tie-break (and full ordering) must be input-order-invariant.
        let r1: Vec<_> = RendezvousHasher::rank_workers("lora-x", &workers_fwd)
            .into_iter()
            .map(|(w, _)| (w.worker_id, w.dp_rank))
            .collect();
        let r2: Vec<_> = RendezvousHasher::rank_workers("lora-x", &workers_rev)
            .into_iter()
            .map(|(w, _)| (w.worker_id, w.dp_rank))
            .collect();
        assert_eq!(r1, r2, "ranking must be invariant to worker input order");
    }

    #[test]
    fn test_sticky_retains_prior_placement() {
        // Stickiness: when the LoRA's prior workers are still present and not full, the sticky
        // variant keeps them even though a clean HRW pass would pick a different (top-ranked)
        // set. This is what prevents transient sibling activity from moving a stable LoRA.
        let hasher = RendezvousHasher;
        let workers = make_workers(5);
        let usage: HashMap<_, _> = workers.iter().map(|w| (*w, (0usize, 4usize))).collect();

        // Natural slot-aware top-2 (no prior hint).
        let natural = hasher.compute_replica_set_with_slots("lora-x", &workers, 2, &usage);
        assert_eq!(natural.len(), 2);

        // Choose a prior set of two workers that are NOT the natural top-2.
        let prior: Vec<_> = workers
            .iter()
            .filter(|w| !natural.contains(w))
            .copied()
            .take(2)
            .collect();
        assert_eq!(prior.len(), 2);

        let sticky =
            hasher.compute_replica_set_with_slots_sticky("lora-x", &workers, 2, &usage, &prior);
        let sset: HashSet<_> = sticky.iter().copied().collect();
        let pset: HashSet<_> = prior.iter().copied().collect();
        assert_eq!(
            sset, pset,
            "sticky must retain the prior (non-full) placement"
        );
    }

    #[test]
    fn test_sticky_replaces_full_prior_worker() {
        // A prior worker that is now at capacity must be dropped and replaced; a non-full prior
        // worker is retained. Capacity safety (REQ 6) is preserved alongside stickiness.
        let hasher = RendezvousHasher;
        let workers = make_workers(5);
        let prior = vec![workers[0], workers[1]];

        let mut usage: HashMap<_, _> = workers.iter().map(|w| (*w, (0usize, 4usize))).collect();
        usage.insert(workers[0], (4, 4)); // prior[0] is full

        let sticky =
            hasher.compute_replica_set_with_slots_sticky("lora-x", &workers, 2, &usage, &prior);
        assert_eq!(sticky.len(), 2);
        assert!(
            !sticky.contains(&workers[0]),
            "a full prior worker must be dropped"
        );
        assert!(
            sticky.contains(&workers[1]),
            "a non-full prior worker must be retained"
        );
    }
}
