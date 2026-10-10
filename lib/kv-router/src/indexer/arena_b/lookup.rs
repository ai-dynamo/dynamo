// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lookups (spec B 6). A walk writes no shared memory: it pins the epoch once, only so
//! the slot table it maps slots through stays valid, and reads every run through a
//! validated attempt (`Runs::read`).
//!
//! The rule: a slot continues into the child at offset `m` only if it holds the run's
//! first `m` positions. At an end child (`m == len`) only whole holders continue; at a
//! divergence inside the run, partial holders with a cutoff of at least `m` continue too.
//! The coverage intersection is taken on every hop.

use super::ArenaIndex;
use super::runs::{Cutoff, Probe, ROOT, ROOT_GEN, Read, RunId, child_key};
use super::slots::SlotTable;
use crate::indexer::MatchDetails;
use crate::protocols::{
    ExternalSequenceBlockHash, LocalBlockHash, OverlapScores, WorkerWithDpRank,
};

/// Where a walk's results go.
pub(crate) trait Sink {
    /// Whether the walk must load external hashes.
    const EXTS: bool;

    fn score(
        &mut self,
        rank: WorkerWithDpRank,
        depth: u32,
        last: Option<ExternalSequenceBlockHash>,
    );

    fn chain(&mut self, _exts: &[ExternalSequenceBlockHash]) {}

    fn reserve(&mut self, _ranks: usize) {}
}

impl Sink for OverlapScores {
    const EXTS: bool = false;

    #[inline]
    fn score(
        &mut self,
        rank: WorkerWithDpRank,
        depth: u32,
        _last: Option<ExternalSequenceBlockHash>,
    ) {
        self.scores
            .entry(rank)
            .and_modify(|score| *score = (*score).max(depth))
            .or_insert(depth);
    }

    fn reserve(&mut self, ranks: usize) {
        self.scores.reserve(ranks);
    }
}

pub(crate) struct DetailsSink {
    pub(crate) details: MatchDetails,
    pub(crate) chain: Option<Vec<ExternalSequenceBlockHash>>,
}

impl Sink for DetailsSink {
    const EXTS: bool = true;

    fn score(
        &mut self,
        rank: WorkerWithDpRank,
        depth: u32,
        last: Option<ExternalSequenceBlockHash>,
    ) {
        let scores = &mut self.details.overlap_scores.scores;
        if scores.get(&rank).is_some_and(|&existing| existing >= depth) {
            return;
        }
        scores.insert(rank, depth);
        if let Some(last) = last {
            self.details.last_matched_hashes.insert(rank, last);
        }
    }

    fn chain(&mut self, exts: &[ExternalSequenceBlockHash]) {
        if let Some(chain) = self.chain.as_mut() {
            chain.extend_from_slice(exts);
        }
    }

    fn reserve(&mut self, ranks: usize) {
        self.details.overlap_scores.scores.reserve(ranks);
        self.details.last_matched_hashes.reserve(ranks);
    }
}

/// What one validated attempt on a run produced.
struct Hop {
    /// Positions of the run that match the query from the walk's position.
    m: usize,
    len: usize,
    cont: Option<(RunId, u32)>,
}

/// Scratch reused across the hops of one walk, and across walks on one thread.
#[derive(Default)]
struct Scratch {
    whole: Vec<u64>,
    cuts: Vec<Cutoff>,
    exts: Vec<ExternalSequenceBlockHash>,
    active: Vec<u64>,
    next: Vec<u64>,
}

impl Scratch {
    fn reset(&mut self, live_words: usize) {
        for words in [&mut self.whole, &mut self.active, &mut self.next] {
            words.clear();
            words.resize(live_words, 0);
        }
        self.cuts.clear();
        self.exts.clear();
    }
}

thread_local! {
    static SCRATCH: std::cell::Cell<Option<Box<Scratch>>> = const { std::cell::Cell::new(None) };
}

#[inline]
fn for_each_bit(word: usize, mut bits: u64, mut f: impl FnMut(usize)) {
    while bits != 0 {
        f(word * 64 + bits.trailing_zeros() as usize);
        bits &= bits - 1;
    }
}

/// The largest cutoff `slot` has among `cuts`, or 0.
#[inline]
fn cutoff_of(cuts: &[Cutoff], slot: usize) -> u32 {
    cuts.iter()
        .filter(|c| c.slot == slot)
        .map(|c| c.cutoff)
        .max()
        .unwrap_or(0)
}

impl ArenaIndex {
    /// Scores, as CRTC's `find_matches_impl`.
    pub fn find_matches_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::new();
        self.walk(sequence, early_exit, &mut scores);
        scores
    }

    /// Scores, last matched hashes and optionally the transfer chain, with the outputs of
    /// CRTC's `find_match_details_impl_with_options`.
    pub fn find_match_details(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
        retain_kv_transfer_chain: bool,
    ) -> MatchDetails {
        let mut sink = DetailsSink {
            details: MatchDetails::new(),
            chain: retain_kv_transfer_chain.then(|| Vec::with_capacity(sequence.len())),
        };
        self.walk(sequence, early_exit, &mut sink);
        let DetailsSink { mut details, chain } = sink;
        if let Some(chain) = chain {
            details.retain_kv_transfer_candidates(chain);
        }
        details
    }

    fn walk<S: Sink>(&self, seq: &[LocalBlockHash], early_exit: bool, sink: &mut S) {
        if seq.is_empty() {
            return;
        }
        let guard = crossbeam_epoch::pin();
        let table = self.slots.table(&guard);
        let live_words = self.slots.live_words();
        if live_words == 0 {
            return;
        }
        let runs = &self.runs;
        let first = runs.read(ROOT, ROOT_GEN, |_, snap| {
            match runs.probe_child(snap.children, child_key(0, seq[0])) {
                Probe::Torn => None,
                probe => Some(probe),
            }
        });
        let Read::Ok(Probe::Found(mut run, mut generation)) = first else {
            return;
        };

        let mut scratch = SCRATCH.with(|cell| cell.take()).unwrap_or_default();
        scratch.reset(live_words);
        let mut active = std::mem::take(&mut scratch.active);
        let mut next = std::mem::take(&mut scratch.next);
        let mut pos = 0usize;
        let mut first_hop = true;
        let mut prev: Option<ExternalSequenceBlockHash> = None;
        let mut exited_early = false;
        let mut scored_all = false;

        loop {
            let hop = runs.read(run, generation, |h, snap| {
                let columns = runs.columns(snap)?;
                let len = columns.len();
                let avail = seq.len() - pos;
                let mut m = 0;
                while m < len && m < avail && columns.local(m) == seq[pos + m] {
                    m += 1;
                }
                if m == 0 {
                    return Some(Hop { m, len, cont: None });
                }
                let cont = if pos + m < seq.len() {
                    match runs.probe_child(snap.children, child_key(m as u32, seq[pos + m])) {
                        Probe::Torn => return None,
                        Probe::Found(id, generation) => Some((id, generation)),
                        Probe::Absent => None,
                    }
                } else {
                    None
                };
                // Whole words, then cutoff entries, then whole words again: a bit seen in
                // either read counts, so a promotion between them never undercounts.
                scratch.whole.fill(0);
                let whole = &mut scratch.whole;
                runs.for_each_whole_word(h, snap, live_words, |w, bits| whole[w] |= bits)?;
                scratch.cuts.clear();
                let cuts = &mut scratch.cuts;
                runs.for_each_cutoff(snap.cutoffs, |c| cuts.push(c))?;
                runs.for_each_whole_word(h, snap, live_words, |w, bits| whole[w] |= bits)?;
                if S::EXTS {
                    scratch.exts.clear();
                    scratch.exts.extend((0..m).map(|i| columns.ext(i)));
                }
                Some(Hop { m, len, cont })
            });
            let Read::Ok(hop) = hop else {
                break;
            };
            if hop.m == 0 {
                break;
            }
            let Hop { m, len, cont } = hop;
            let exts = &scratch.exts;
            let last_at = |k: usize| {
                if !S::EXTS {
                    None
                } else if k == 0 {
                    prev
                } else {
                    exts.get(k - 1).copied()
                }
            };
            let score = |sink: &mut S, slot: usize, k: usize| {
                let depth = pos + k;
                if depth == 0 {
                    return;
                }
                if let Some(rank) = table.owner(slot) {
                    sink.score(rank, depth as u32, last_at(k));
                }
            };

            next.fill(0);
            let continues = cont.is_some();
            if first_hop {
                // Candidates: whole holders and slots with entries.
                let mut ranks = 0;
                for (w, &bits) in scratch.whole.iter().enumerate() {
                    ranks += bits.count_ones() as usize;
                    if continues {
                        next[w] |= bits;
                    } else {
                        for_each_bit(w, bits, |slot| score(sink, slot, m));
                    }
                }
                sink.reserve(ranks + scratch.cuts.len());
                for (i, c) in scratch.cuts.iter().enumerate() {
                    let (w, bit) = (c.slot / 64, 1u64 << (c.slot % 64));
                    if scratch.whole.get(w).is_some_and(|bits| bits & bit != 0) {
                        continue;
                    }
                    // Handle each slot once, at its largest entry.
                    if scratch.cuts[..i].iter().any(|e| e.slot == c.slot) {
                        continue;
                    }
                    let held = cutoff_of(&scratch.cuts, c.slot) as usize;
                    if held >= m && continues && w < next.len() {
                        next[w] |= bit;
                    } else {
                        score(sink, c.slot, held.min(m));
                    }
                }
            } else {
                for w in 0..active.len() {
                    let a = active[w];
                    if a == 0 {
                        continue;
                    }
                    let whole = a & scratch.whole[w];
                    if continues {
                        next[w] |= whole;
                    } else {
                        for_each_bit(w, whole, |slot| score(sink, slot, m));
                    }
                    for_each_bit(w, a & !scratch.whole[w], |slot| {
                        let held = cutoff_of(&scratch.cuts, slot) as usize;
                        if held >= m && continues {
                            next[w] |= 1 << (slot % 64);
                        } else {
                            score(sink, slot, held.min(m));
                        }
                    });
                }
            }
            debug_assert!(m <= len);
            if S::EXTS {
                sink.chain(&scratch.exts);
            }
            // Without a continuation every candidate was scored in this hop.
            let Some((child, child_gen)) = cont else {
                scored_all = true;
                break;
            };
            let remaining: u32 = next.iter().map(|w| w.count_ones()).sum();
            if remaining == 0 {
                scored_all = true;
                break;
            }
            pos += m;
            if S::EXTS {
                prev = scratch.exts.last().copied();
            }
            if early_exit && remaining == 1 {
                exited_early = true;
                break;
            }
            std::mem::swap(&mut active, &mut next);
            (run, generation) = (child, child_gen);
            first_hop = false;
        }

        // The walk stopped at `pos`: an early exit, or a child that was gone, poisoned, or
        // matched nothing. Every slot still continuing holds the query's first `pos` blocks.
        if !scored_all {
            let survivors = if exited_early { &next } else { &active };
            self.score_survivors(table, survivors, pos, prev, sink);
        }
        scratch.active = active;
        scratch.next = next;
        SCRATCH.with(|cell| cell.set(Some(scratch)));
    }

    fn score_survivors<S: Sink>(
        &self,
        table: &SlotTable,
        survivors: &[u64],
        pos: usize,
        prev: Option<ExternalSequenceBlockHash>,
        sink: &mut S,
    ) {
        if pos == 0 {
            return;
        }
        for (w, &bits) in survivors.iter().enumerate() {
            for_each_bit(w, bits, |slot| {
                if let Some(rank) = table.owner(slot) {
                    sink.score(rank, pos as u32, prev);
                }
            });
        }
    }
}
