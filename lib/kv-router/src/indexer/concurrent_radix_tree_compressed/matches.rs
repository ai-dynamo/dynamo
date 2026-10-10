// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crossbeam_epoch::Guard;

use super::children::LookupPin;
use super::*;

impl ConcurrentRadixTreeCompressed {
    /// Traverse the radix tree to find the best match for a given sequence of
    /// [`LocalBlockHash`]es, returning both overlap scores and the last matched
    /// `ExternalSequenceBlockHash` per worker (used for lower-tier continuation).
    ///
    /// Ranks with full-edge coverage are tracked in the `active` slot set and continue
    /// into children. Ranks with a cutoff are scored at the node where their cutoff
    /// falls short and are never propagated into children.
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
        let guard = LookupPin::new();
        let next_child = sequence
            .first()
            .and_then(|&local_hash| self.root.child_ref(local_hash, &guard));
        self.find_details_from_seq(
            next_child,
            SliceHashSequence(sequence),
            early_exit,
            retain_kv_transfer_chain,
            &guard,
        )
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn find_details_from_seq<'g, S: HashSequence>(
        &self,
        next_child: Option<&'g Node>,
        sequence: S,
        early_exit: bool,
        retain_kv_transfer_chain: bool,
        guard: &'g Guard,
    ) -> MatchDetails {
        let mut details = MatchDetails::new();
        if sequence.len() == 0 {
            return details;
        }
        let mut kv_transfer_chain =
            retain_kv_transfer_chain.then(|| Vec::with_capacity(sequence.len()));

        {
            let MatchDetails {
                overlap_scores: ref mut scores,
                ref mut last_matched_hashes,
                kv_transfer_candidates: _,
            } = details;
            self.walk_match_path(
                next_child,
                &sequence,
                early_exit,
                scores,
                Some(last_matched_hashes),
                kv_transfer_chain.as_mut(),
                guard,
            );
        }

        if let Some(block_hashes) = kv_transfer_chain {
            details.retain_kv_transfer_candidates(block_hashes);
        }
        details
    }

    /// Walks from `next_child` borrowing each node under `guard` rather than cloning its
    /// `Arc`, so the walk touches no shared reference counts, and scores every rank it
    /// matched, including the ranks still active at the end of the walk. The caller pins
    /// once for the whole walk and keeps the first node alive for `'g`.
    #[cfg_attr(feature = "profile", inline(never))]
    #[allow(clippy::too_many_arguments)]
    fn walk_match_path<'g, S: HashSequence>(
        &self,
        mut next_child: Option<&'g Node>,
        sequence: &S,
        early_exit: bool,
        scores: &mut OverlapScores,
        mut last_matched_hashes: Option<
            &mut FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
        >,
        mut kv_transfer_chain: Option<&mut Vec<ExternalSequenceBlockHash>>,
        guard: &'g Guard,
    ) {
        // The slot table loaded under the walk's pin maps every bit the walk sees.
        let table = self.slots.table(guard);
        let mut active = SlotSet::default();
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

            let outcome = child.find_match_step(
                FindStepInput {
                    sequence,
                    seq_pos,
                    first_node,
                    prev_depth: matched_depth,
                    prev_edge_last_hash,
                    table,
                    active: &mut active,
                    scores,
                    last_matched_hashes: last_matched_hashes.as_deref_mut(),
                    kv_transfer_chain: kv_transfer_chain.as_deref_mut(),
                },
                guard,
            );
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

        for worker in active.iter().filter_map(|slot| table.owner(slot)) {
            scores.scores.insert(worker, matched_depth);
            if let Some(last_matched_hashes) = last_matched_hashes.as_deref_mut()
                && let Some(hash) = prev_edge_last_hash
            {
                last_matched_hashes.insert(worker, hash);
            }
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

        let guard = LookupPin::new();
        let next_child = sequence
            .first()
            .and_then(|&local_hash| self.root.child_ref(local_hash, &guard));
        let sequence = SliceHashSequence(sequence);
        self.walk_match_path(
            next_child,
            &sequence,
            early_exit,
            &mut scores,
            None,
            None,
            &guard,
        );
        scores
    }
}
