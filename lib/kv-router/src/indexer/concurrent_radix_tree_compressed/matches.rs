// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConcurrentRadixTreeCompressed {
    /// Traverse the radix tree to find the best match for a given sequence of
    /// [`LocalBlockHash`]es, returning both overlap scores and the last matched
    /// `ExternalSequenceBlockHash` per worker (used for lower-tier continuation).
    ///
    /// Workers in `full_edge_workers` are tracked in the `active` set and continue
    /// into children. Workers in `worker_cutoffs` are scored at the node where their
    /// cutoff falls short and are never propagated into children.
    pub fn find_match_details_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
    ) -> MatchDetails {
        let next_child = sequence
            .first()
            .and_then(|&local_hash| self.root.child_snapshot(local_hash));
        self.find_details_from_seq(next_child, SliceHashSequence(sequence), early_exit)
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn find_details_from_seq<S: HashSequence>(
        &self,
        next_child: Option<SharedNode>,
        sequence: S,
        early_exit: bool,
    ) -> MatchDetails {
        let mut details = MatchDetails::new();
        if sequence.len() == 0 {
            return details;
        }

        let walk_result = {
            let MatchDetails {
                overlap_scores: ref mut scores,
                ref mut last_matched_hashes,
            } = details;
            Self::walk_match_path(
                next_child,
                &sequence,
                early_exit,
                scores,
                Some(last_matched_hashes),
            )
        };

        Self::record_surviving_details(&mut details, &walk_result);
        details
    }

    #[cfg_attr(feature = "profile", inline(never))]
    fn walk_match_path<S: HashSequence>(
        mut next_child: Option<SharedNode>,
        sequence: &S,
        early_exit: bool,
        scores: &mut OverlapScores,
        mut last_matched_hashes: Option<
            &mut FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
        >,
    ) -> MatchWalkResult {
        let mut active: FxHashSet<WorkerWithDpRank> = FxHashSet::default();
        let mut active_count: usize = 0;
        let mut matched_depth: u32 = 0;
        let mut seq_pos: usize = 0;
        let mut first_node = true;
        // Last ExternalSequenceBlockHash from the previous fully-matched edge.
        // Workers that drop at a node boundary (not present in the new node)
        // were last matched at the end of the previous edge.
        let mut prev_edge_last_hash: Option<ExternalSequenceBlockHash> = None;

        loop {
            if seq_pos >= sequence.len() {
                break;
            }
            let child = match next_child.take() {
                Some(c) => c,
                None => break,
            };

            let outcome = child.find_match_step(FindStepInput {
                sequence,
                seq_pos,
                first_node,
                prev_depth: matched_depth,
                prev_edge_last_hash,
                active: &mut active,
                active_count,
                scores,
                last_matched_hashes: last_matched_hashes.as_deref_mut(),
            });
            let edge_len = outcome.edge_len;
            let edge_match_len = outcome.edge_match_len;
            active_count = outcome.active_count;
            next_child = outcome.next_child;
            prev_edge_last_hash = outcome.prev_edge_last_hash;
            if first_node {
                first_node = false;
            }

            if active_count == 0 {
                break;
            }
            matched_depth += edge_match_len as u32;
            if edge_match_len < edge_len {
                break;
            }
            seq_pos += edge_match_len;
            if early_exit && active_count == 1 {
                break;
            }
        }

        MatchWalkResult {
            active,
            matched_depth,
            prev_edge_last_hash,
        }
    }

    fn record_surviving_details(details: &mut MatchDetails, walk_result: &MatchWalkResult) {
        for worker in &walk_result.active {
            details
                .overlap_scores
                .scores
                .insert(*worker, walk_result.matched_depth);
            if let Some(hash) = walk_result.prev_edge_last_hash {
                details.last_matched_hashes.insert(*worker, hash);
            }
        }
    }

    fn record_surviving_scores(scores: &mut OverlapScores, walk_result: &MatchWalkResult) {
        for worker in &walk_result.active {
            scores.scores.insert(*worker, walk_result.matched_depth);
        }
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub fn find_matches_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        let next_child = sequence
            .first()
            .and_then(|&local_hash| self.root.child_snapshot(local_hash));
        let sequence = SliceHashSequence(sequence);
        let walk_result =
            Self::walk_match_path(next_child, &sequence, early_exit, &mut scores, None);
        Self::record_surviving_scores(&mut scores, &walk_result);
        scores
    }
}
