// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dump: every rank's holdings as `Stored` events.
//!
//! The physical layout depends on arrival order (who appended, where children hang), so
//! the dump is canonical per rank instead: each event is a maximal chain of the rank's
//! own held positions, broken only where the rank's held paths branch. A rank is
//! credited in a child at offset `o` only if it holds the parent's first `o` positions,
//! like a reader. Events of one rank come parent-first. Not a consistent cut: each run is
//! read under its own locks.

use rustc_hash::FxHashMap;

use super::*;
use crate::indexer::compressed_radix::append_dump_events;

/// One run as the dump read it.
struct RunSnapshot {
    edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    holdings: FxHashMap<WorkerWithDpRank, u32>,
    /// `(offset, child)` sorted by offset.
    children: Vec<(u32, u32)>,
}

/// A chain being built for one rank.
struct Chain {
    parent: Option<ExternalSequenceBlockHash>,
    blocks: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
}

impl ArenaIndexC {
    /// Reads every reachable run once.
    fn dump_snapshot(&self) -> (Vec<u32>, FxHashMap<u32, RunSnapshot>) {
        let store = &self.store;
        let mut runs = FxHashMap::default();
        let roots: Vec<u32> = store
            .children_of(store.run(ROOT))
            .into_iter()
            .map(|(_, child, _)| child)
            .collect();
        let mut queue: Vec<(u32, u32)> = store
            .children_of(store.run(ROOT))
            .into_iter()
            .map(|(_, child, generation)| (child, generation))
            .collect();
        while let Some((id, generation)) = queue.pop() {
            // One pin per run: the slot table maps this run's bits.
            let guard = crossbeam_epoch::pin();
            let table = self.slots.table(&guard);
            let run = store.run(id);
            let _gate = run.gate.read();
            let _state = run.state.read();
            if run.is_dead() || run.generation.load(Ordering::Relaxed) != generation {
                continue;
            }
            let window = store.window(run);
            let edge = (0..window.len())
                .map(|i| {
                    (
                        LocalBlockHash(window.local(i)),
                        ExternalSequenceBlockHash(window.ext(i)),
                    )
                })
                .collect();
            let len = run.len();
            let mut holdings = FxHashMap::default();
            store.for_each_whole(run, |slot| {
                if let Some(rank) = table.owner(slot) {
                    holdings.insert(rank, len);
                }
            });
            for (slot, cutoff) in store.cutoff_entries(run) {
                if let Some(rank) = table.owner(slot) {
                    holdings.entry(rank).or_insert(cutoff);
                }
            }
            let start = run.start.load(Ordering::Relaxed);
            let mut children = Vec::new();
            for (_, child, child_generation) in store.children_of(run) {
                let offset = store.run(child).start.load(Ordering::Relaxed) - start;
                children.push((offset, child));
                queue.push((child, child_generation));
            }
            children.sort_unstable();
            runs.insert(
                id,
                RunSnapshot {
                    edge,
                    holdings,
                    children,
                },
            );
        }
        (roots, runs)
    }

    pub(super) fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        let (roots, runs) = self.dump_snapshot();
        let mut ranks: Vec<WorkerWithDpRank> = runs
            .values()
            .flat_map(|run| run.holdings.keys().copied())
            .collect();
        ranks.sort_unstable();
        ranks.dedup();

        let mut events = Vec::new();
        let mut event_id = 0u64;
        for rank in ranks {
            // (run, chain the run continues) still to visit.
            let mut stack: Vec<(u32, Chain)> = roots
                .iter()
                .map(|&run| {
                    (
                        run,
                        Chain {
                            parent: None,
                            blocks: Vec::new(),
                        },
                    )
                })
                .collect();
            while let Some((id, mut chain)) = stack.pop() {
                let Some(run) = runs.get(&id) else {
                    continue;
                };
                let held = run.holdings.get(&rank).copied().unwrap_or(0) as usize;
                if held == 0 {
                    if !chain.blocks.is_empty() {
                        Self::emit(&mut events, &mut event_id, rank, chain);
                    }
                    continue;
                }
                let held_children = |offset: usize| -> Vec<u32> {
                    run.children
                        .iter()
                        .filter(|&&(o, child)| {
                            o as usize == offset
                                && runs
                                    .get(&child)
                                    .is_some_and(|c| c.holdings.contains_key(&rank))
                        })
                        .map(|&(_, child)| child)
                        .collect()
                };
                for o in 0..held {
                    chain.blocks.push(run.edge[o]);
                    if o + 1 == held {
                        break;
                    }
                    let next = held_children(o + 1);
                    if next.is_empty() {
                        continue;
                    }
                    // The rank holds its own next position and these children: a branch.
                    let parent = Some(run.edge[o].1);
                    let done = std::mem::replace(
                        &mut chain,
                        Chain {
                            parent,
                            blocks: Vec::new(),
                        },
                    );
                    Self::emit(&mut events, &mut event_id, rank, done);
                    for child in next {
                        stack.push((
                            child,
                            Chain {
                                parent,
                                blocks: Vec::new(),
                            },
                        ));
                    }
                }
                // The chain continues into a lone held end child, or ends here.
                let ends = held_children(held);
                if let [child] = ends[..] {
                    stack.push((child, chain));
                    continue;
                }
                let parent = Some(run.edge[held - 1].1);
                Self::emit(&mut events, &mut event_id, rank, chain);
                for child in ends {
                    stack.push((
                        child,
                        Chain {
                            parent,
                            blocks: Vec::new(),
                        },
                    ));
                }
            }
        }
        events
    }

    fn emit(
        events: &mut Vec<RouterEvent>,
        event_id: &mut u64,
        rank: WorkerWithDpRank,
        chain: Chain,
    ) {
        append_dump_events(events, event_id, chain.parent, &chain.blocks, &[rank], &[]);
    }

    /// Every live run reachable from ROOT, ROOT excluded.
    pub(super) fn reachable_runs(&self) -> Vec<u32> {
        let store = &self.store;
        let _guard = crossbeam_epoch::pin();
        let mut out = Vec::new();
        let mut queue: Vec<u32> = store
            .children_of(store.run(ROOT))
            .into_iter()
            .map(|(_, child, _)| child)
            .collect();
        while let Some(id) = queue.pop() {
            let run = store.run(id);
            let _state = run.state.read();
            if run.is_dead() {
                continue;
            }
            out.push(id);
            queue.extend(store.children_of(run).into_iter().map(|(_, c, _)| c));
        }
        out
    }

    /// Lengths of every live run.
    pub(super) fn run_lengths(&self) -> Vec<usize> {
        self.reachable_runs()
            .into_iter()
            .map(|id| self.store.run(id).len() as usize)
            .collect()
    }
}
