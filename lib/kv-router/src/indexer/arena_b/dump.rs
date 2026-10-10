// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dump_events` (spec B 7.7): `Stored` events from the shared runs, walked from the root
//! with each run read under its lock. Not a consistent cut, as CRTC's is not.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use rustc_hash::FxHashMap;

use super::ArenaIndex;
use super::runs::{ROOT, ROOT_GEN, RunId};
use crate::indexer::compressed_radix::append_dump_events;
use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash, RouterEvent, WorkerWithDpRank};

/// A run to dump, with the parent's credited holdings and this run's offset in it.
struct DumpItem {
    run: RunId,
    generation: u32,
    parent_hash: Option<ExternalSequenceBlockHash>,
    /// `None` under the root: every holder is credited.
    parent: Option<(Arc<FxHashMap<WorkerWithDpRank, u32>>, u32)>,
}

impl ArenaIndex {
    pub(crate) fn dump_tree_as_events(&self) -> Vec<RouterEvent> {
        let mut events = Vec::new();
        let mut event_id = 0u64;
        let mut queue = VecDeque::new();
        if let Some(root) = self.runs.lock(ROOT, ROOT_GEN) {
            for (_, id, generation) in self.runs.children_of(root.snap().children) {
                queue.push_back(DumpItem {
                    run: id,
                    generation,
                    parent_hash: None,
                    parent: None,
                });
            }
        }
        while let Some(item) = queue.pop_front() {
            let guard = crossbeam_epoch::pin();
            let table = self.slots.table(&guard);
            let Some(locked) = self.runs.lock(item.run, item.generation) else {
                continue;
            };
            let snap = locked.snap();
            let Some(columns) = self.runs.columns(&snap) else {
                continue;
            };
            let edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = (0..columns.len())
                .map(|i| (columns.local(i), columns.ext(i)))
                .collect();
            // Credit a rank here only if it holds the parent's positions before this run.
            let credited = |rank: &WorkerWithDpRank| match &item.parent {
                None => true,
                Some((holdings, offset)) => holdings.get(rank).is_some_and(|h| h >= offset),
            };
            let mut holdings: FxHashMap<WorkerWithDpRank, u32> = FxHashMap::default();
            let mut whole = Vec::new();
            for slot in self.runs.whole_slots(&locked) {
                if let Some(rank) = table.owner(slot).filter(|rank| credited(rank)) {
                    whole.push(rank);
                    holdings.insert(rank, snap.len);
                }
            }
            let mut partial = Vec::new();
            for cut in self.runs.cutoffs(&snap) {
                if let Some(rank) = table.owner(cut.slot).filter(|rank| credited(rank))
                    && !holdings.contains_key(&rank)
                {
                    partial.push((rank, cut.cutoff as usize));
                    holdings.insert(rank, cut.cutoff);
                }
            }
            whole.sort_unstable();
            partial.sort_unstable();
            append_dump_events(
                &mut events,
                &mut event_id,
                item.parent_hash,
                &edge,
                &whole,
                &partial,
            );
            if holdings.is_empty() {
                continue;
            }
            let holdings = Arc::new(holdings);
            for (_, id, generation) in self.runs.children_of(snap.children) {
                let Some(child) = self.runs.header(id) else {
                    continue;
                };
                // A linked child cannot die while its parent is locked.
                let offset = child.start.load(Ordering::Relaxed).wrapping_sub(snap.start);
                if offset == 0 || offset as usize > edge.len() {
                    continue;
                }
                if !holdings.values().any(|&h| h >= offset) {
                    continue;
                }
                queue.push_back(DumpItem {
                    run: id,
                    generation,
                    parent_hash: Some(edge[offset as usize - 1].1),
                    parent: Some((holdings.clone(), offset)),
                });
            }
        }
        events
    }
}
