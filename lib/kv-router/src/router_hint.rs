// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Candidate KV chains retained for router-planned migration.

use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterHintRootCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(WorkerWithDpRank, usize)>,
}

impl RouterHintRootCandidates {
    pub fn best_source<F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut is_eligible_source: F,
    ) -> Option<(WorkerWithDpRank, Vec<ExternalSequenceBlockHash>)>
    where
        F: FnMut(WorkerWithDpRank) -> bool,
    {
        let (source, prefix_blocks) = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(worker, blocks)| {
                *blocks > prefix_blocks_to_beat && is_eligible_source(*worker)
            })
            .max_by(|(left_worker, left_blocks), (right_worker, right_blocks)| {
                left_blocks
                    .cmp(right_blocks)
                    .then_with(|| right_worker.cmp(left_worker))
            })?;

        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_source_selects_longest_eligible_prefix() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let excluded = WorkerWithDpRank::new(9, 0);
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ],
            owner_prefix_blocks: vec![(worker_b, 2), (excluded, 3), (worker_a, 3)],
        };

        let selected = candidates.best_source(0, |worker| worker != excluded);

        assert_eq!(
            selected,
            Some((
                worker_a,
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                ],
            ))
        );
    }

    #[test]
    fn best_source_fails_closed_on_invalid_prefix_length() {
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![ExternalSequenceBlockHash(101)],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 0), 2)],
        };

        assert!(candidates.best_source(0, |_| true).is_none());
    }

    #[test]
    fn best_source_requires_prefix_longer_than_threshold() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
                ExternalSequenceBlockHash(104),
            ],
            owner_prefix_blocks: vec![(worker_a, 3), (worker_b, 4)],
        };

        assert!(
            candidates
                .best_source(3, |worker| worker == worker_a)
                .is_none()
        );
        assert_eq!(
            candidates.best_source(3, |_| true),
            Some((
                worker_b,
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                    ExternalSequenceBlockHash(104),
                ],
            ))
        );
    }
}
