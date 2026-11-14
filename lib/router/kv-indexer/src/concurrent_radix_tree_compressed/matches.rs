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
        self.find_match_details_impl_with_options(sequence, early_exit, false)
    }

    pub fn find_match_details_impl_with_options(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
        retain_kv_transfer_chain: bool,
    ) -> MatchDetails {
        let next_child = sequence
            .first()
            .and_then(|&local_hash| self.root.child_snapshot(local_hash));
        self.find_details_from_seq(
            next_child,
            SliceHashSequence(sequence),
            early_exit,
            retain_kv_transfer_chain,
        )
    }

    pub fn find_match_details_impl_with_observer(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
        retain_kv_transfer_chain: bool,
        observer: &mut dyn MatchObserver,
    ) -> MatchDetails {
        let next_child = sequence
            .first()
            .and_then(|&local_hash| self.root.child_snapshot(local_hash));
        self.find_details_from_seq_observing(
            next_child,
            SliceHashSequence(sequence),
            early_exit,
            retain_kv_transfer_chain,
            Some(observer),
        )
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn find_details_from_seq<S: HashSequence>(
        &self,
        next_child: Option<SharedNode>,
        sequence: S,
        early_exit: bool,
        retain_kv_transfer_chain: bool,
    ) -> MatchDetails {
        self.find_details_from_seq_observing(
            next_child,
            sequence,
            early_exit,
            retain_kv_transfer_chain,
            None,
        )
    }

    fn find_details_from_seq_observing<S: HashSequence>(
        &self,
        next_child: Option<SharedNode>,
        sequence: S,
        early_exit: bool,
        retain_kv_transfer_chain: bool,
        mut observer: Option<&mut dyn MatchObserver>,
    ) -> MatchDetails {
        let mut details = MatchDetails::new();
        if sequence.len() == 0 {
            return details;
        }
        let mut kv_transfer_chain =
            retain_kv_transfer_chain.then(|| Vec::with_capacity(sequence.len()));

        let walk_result = {
            let MatchDetails {
                overlap_scores: ref mut scores,
                ref mut last_matched_hashes,
                kv_transfer_candidates: _,
                hybrid_resume_caps: _,
            } = details;
            Self::walk_match_path(
                next_child,
                &sequence,
                early_exit,
                scores,
                Some(last_matched_hashes),
                kv_transfer_chain.as_mut(),
                observer
                    .as_mut()
                    .map(|observer| &mut **observer as &mut dyn MatchObserver),
            )
        };

        Self::record_surviving_details(&mut details, walk_result, observer);
        if let Some(block_hashes) = kv_transfer_chain {
            details.retain_kv_transfer_candidates(block_hashes);
        }
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
        mut kv_transfer_chain: Option<&mut Vec<ExternalSequenceBlockHash>>,
        mut observer: Option<&mut dyn MatchObserver>,
    ) -> MatchWalkResult {
        let mut active = WorkerSet::default();
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
                scores,
                last_matched_hashes: last_matched_hashes.as_deref_mut(),
                kv_transfer_chain: kv_transfer_chain.as_deref_mut(),
                observer: observer
                    .as_mut()
                    .map(|observer| &mut **observer as &mut dyn MatchObserver),
            });
            let edge_len = outcome.edge_len;
            let edge_match_len = outcome.edge_match_len;
            let active_count = outcome.active_count;
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

    fn record_surviving_details(
        details: &mut MatchDetails,
        walk_result: MatchWalkResult,
        mut observer: Option<&mut dyn MatchObserver>,
    ) {
        let mut active = walk_result.active.into_workers();
        if let Some(observer) = observer.as_deref_mut() {
            match observer.emit_match_details(
                active,
                walk_result.matched_depth,
                walk_result.prev_edge_last_hash,
                details,
            ) {
                Ok(()) => return,
                Err(workers) => active = workers,
            }
        }
        // A uniform result has no earlier cutoffs or workers dropped at a node
        // boundary. Its already-owned active set can carry the immutable caps.
        let eligible = if details.overlap_scores.scores.is_empty() && !active.is_empty() {
            walk_result.prev_edge_last_hash.and_then(|hash| {
                observer.as_mut().and_then(|observer| {
                    observer.uniform_resume_workers(walk_result.matched_depth, hash)
                })
            })
        } else {
            None
        };
        if let Some(eligible) = eligible {
            let mut confirmed = true;
            for worker in &active {
                confirmed &= eligible.contains(worker);
                details
                    .overlap_scores
                    .scores
                    .insert(*worker, walk_result.matched_depth);
                if let Some(hash) = walk_result.prev_edge_last_hash {
                    details.last_matched_hashes.insert(*worker, hash);
                }
            }
            if confirmed {
                details.hybrid_resume_caps = super::super::HybridResumeCaps::uniform_confirmed(
                    active,
                    walk_result.matched_depth,
                );
            }
        } else {
            for worker in &active {
                details
                    .overlap_scores
                    .scores
                    .insert(*worker, walk_result.matched_depth);
                if let Some(hash) = walk_result.prev_edge_last_hash {
                    details.last_matched_hashes.insert(*worker, hash);
                }
            }
        }
    }

    fn record_surviving_scores(scores: &mut OverlapScores, walk_result: &MatchWalkResult) {
        for worker in &walk_result.active {
            scores.scores.insert(*worker, walk_result.matched_depth);
        }
    }

    // NOTE(perf): A reusable compact result sink reduced result-map work in
    // profiles but did not improve end-to-end throughput under contention.
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
        let walk_result = Self::walk_match_path(
            next_child,
            &sequence,
            early_exit,
            &mut scores,
            None,
            None,
            None,
        );
        Self::record_surviving_scores(&mut scores, &walk_result);
        scores
    }
}
