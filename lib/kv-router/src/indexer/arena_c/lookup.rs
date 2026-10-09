// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lookups: one epoch pin per walk, ROOT's table probed without locks, and every other
//! run read under its state read lock, one run at a time.
//!
//! Per hop, under the run's state lock: match the query against the run's window for `m`
//! positions, probe the child at offset `m` for the next query hash, and split the
//! candidate slots. A slot continues into that child only if it holds the run's first `m`
//! positions (whole, or a cutoff of at least `m`); every other slot is scored where its
//! holding ends. The intersection with the run's coverage is taken on every hop. There is
//! no retry loop: a dead run or a mismatched head ends the walk, which can only
//! undercount.

use rustc_hash::FxHashMap;

use super::slots::SlotTable;
use super::table::child_key;
use super::*;
use crate::indexer::MatchDetails;

/// Slot words kept on the stack before a walk spills to the heap.
const STACK_WORDS: usize = 4;

/// Where a walk writes its results.
struct Sink<'a, 't> {
    table: &'t SlotTable,
    scores: &'a mut OverlapScores,
    last_matched: Option<&'a mut FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>>,
}

impl Sink<'_, '_> {
    #[inline]
    fn score(&mut self, slot: Slot, depth: usize, last: u64) {
        let Some(rank) = self.table.owner(slot) else {
            return;
        };
        self.scores.scores.insert(rank, depth as u32);
        if let Some(last_matched) = self.last_matched.as_deref_mut() {
            last_matched.insert(rank, ExternalSequenceBlockHash(last));
        }
    }

    #[inline]
    fn score_bits(&mut self, word: usize, mut bits: u64, depth: usize, last: u64) {
        while bits != 0 {
            self.score(
                Slot::from_index(word * 64 + bits.trailing_zeros() as usize),
                depth,
                last,
            );
            bits &= bits - 1;
        }
    }

    fn score_all(&mut self, words: &[u64], depth: usize, last: u64) {
        for (word, &bits) in words.iter().enumerate() {
            self.score_bits(word, bits, depth, last);
        }
    }
}

impl ArenaIndexC {
    pub fn find_matches_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();
        self.walk(sequence, early_exit, &mut scores, None, None);
        scores
    }

    /// Scores plus each scored rank's last matched sequence hash and, optionally, the
    /// matched chain of external hashes.
    pub fn find_match_details_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
        retain_kv_transfer_chain: bool,
    ) -> MatchDetails {
        let mut details = MatchDetails::new();
        let mut chain = retain_kv_transfer_chain.then(|| Vec::with_capacity(sequence.len()));
        {
            let MatchDetails {
                overlap_scores,
                last_matched_hashes,
                ..
            } = &mut details;
            self.walk(
                sequence,
                early_exit,
                overlap_scores,
                Some(last_matched_hashes),
                chain.as_mut(),
            );
        }
        if let Some(chain) = chain {
            details.retain_kv_transfer_candidates(chain);
        }
        details
    }

    #[cfg_attr(feature = "profile", inline(never))]
    fn walk(
        &self,
        query: &[LocalBlockHash],
        early_exit: bool,
        scores: &mut OverlapScores,
        last_matched: Option<&mut FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>>,
        mut chain: Option<&mut Vec<ExternalSequenceBlockHash>>,
    ) {
        let Some(&head) = query.first() else {
            return;
        };
        let guard = crossbeam_epoch::pin();
        let table = self.slots.table(&guard);
        let store = &self.store;
        // ROOT never splits or appends; its table is epoch-published, so no lock.
        let Some(mut run_id) = store.find_child(store.run(ROOT), child_key(0, head.0)) else {
            return;
        };
        let words = self.live_words.load(Ordering::Acquire);
        let mut stack = [[0u64; STACK_WORDS]; 3];
        let mut heap: Vec<u64>;
        let (active, next, whole) = if words <= STACK_WORDS {
            let [a, b, c] = &mut stack;
            (&mut a[..words], &mut b[..words], &mut c[..words])
        } else {
            heap = vec![0; 3 * words];
            let (a, rest) = heap.split_at_mut(words);
            let (b, c) = rest.split_at_mut(words);
            (a, b, c)
        };
        let (mut active, mut next) = (active, next);

        let mut sink = Sink {
            table,
            scores,
            last_matched,
        };
        let mut pos = 0usize;
        let mut first = true;
        // External hash at depth `pos - 1`, for slots scored at the start of a run.
        let mut prev_last = 0u64;
        loop {
            let run = store.run(run_id);
            let state = run.state.read();
            if run.is_dead() {
                drop(state);
                sink.score_all(active, pos, prev_last);
                return;
            }
            let window = store.window(run);
            let avail = window.len().min(query.len() - pos);
            let mut m = 0;
            while m < avail && window.local(m) == query[pos + m].0 {
                m += 1;
            }
            if m == 0 {
                drop(state);
                sink.score_all(active, pos, prev_last);
                return;
            }
            if let Some(chain) = chain.as_deref_mut() {
                chain.extend((0..m).map(|i| ExternalSequenceBlockHash(window.ext(i))));
            }
            let cont = if pos + m < query.len() {
                store.find_child(run, child_key(m as u32, query[pos + m].0))
            } else {
                None
            };
            let end_hash = window.ext(m - 1);

            whole.fill(0);
            store.for_each_whole_word(run, |index, bits| {
                if let Some(word) = whole.get_mut(index) {
                    *word = bits;
                }
            });
            let mut any_next = false;
            for w in 0..words {
                let continuing = if first {
                    whole[w]
                } else {
                    active[w] & whole[w]
                };
                // Whole holders hold `m <= len` positions; what is left is partial or none.
                active[w] = if first { 0 } else { active[w] & !whole[w] };
                if cont.is_some() {
                    next[w] = continuing;
                    any_next |= continuing != 0;
                } else {
                    next[w] = 0;
                    sink.score_bits(w, continuing, pos + m, end_hash);
                }
            }
            for (slot, cutoff) in store.cutoff_entries(run) {
                let (w, bit) = (slot.index() / 64, 1u64 << (slot.index() % 64));
                // A set whole bit is authoritative during a promotion.
                if w >= words || whole[w] & bit != 0 {
                    continue;
                }
                if !first {
                    if active[w] & bit == 0 {
                        continue;
                    }
                    active[w] &= !bit;
                }
                let cutoff = cutoff as usize;
                if cutoff >= m && cont.is_some() {
                    next[w] |= bit;
                    any_next = true;
                } else {
                    let depth = cutoff.min(m);
                    sink.score(slot, pos + depth, window.ext(depth - 1));
                }
            }
            // Slots that hold nothing here stop where the previous run left them.
            if !first {
                sink.score_all(active, pos, prev_last);
            }
            drop(state);

            pos += m;
            prev_last = end_hash;
            // Slots continue only when there is a child to continue into.
            let Some(child) = cont.filter(|_| any_next) else {
                return;
            };
            std::mem::swap(&mut active, &mut next);
            if early_exit && active.iter().map(|w| w.count_ones()).sum::<u32>() == 1 {
                sink.score_all(active, pos, prev_last);
                return;
            }
            run_id = child;
            first = false;
        }
    }
}
