// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Snapshots: a frozen copy of a rank's ledger at one sequence, ordered parents first for
//! export on whichever thread serves it.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;
use std::sync::OnceLock;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::rank::Health;
use crate::{BlockHash, BlockKey, BlockView, Cursor, Ledger, Parent, StreamIndex, TierCounts};

/// The longest chain of blocks, each the only child of the one before, all held with the same
/// counts. A host cuts it further wherever its own stored message would differ: per tier, where
/// a per-block attribute changes, or at its block or byte cap.
///
/// A run belongs to the image whose [`SnapshotImage::runs`] returned it.
#[derive(Debug)]
pub struct SnapshotRun {
    /// What precedes the first block.
    pub parent: Parent,
    /// The parent is not in this snapshot (unknown, or evicted mid-chain while this block
    /// stayed), or the run sits on a parent cycle or is reachable only through one, in which
    /// case its parent can come later in the export. A consumer that needs parent-first replay
    /// sets these aside.
    pub orphan: bool,
    pub counts: TierCounts,
    blocks: Range<usize>,
}

impl SnapshotRun {
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

pub(crate) struct Plan {
    order: Vec<BlockHash>,
    runs: Vec<SnapshotRun>,
}

/// Orders a ledger for replay: every run after the run that holds its parent, roots first,
/// orphans last, and the same order for the same ledger.
pub(crate) fn plan<P>(ledger: &Ledger<P>) -> Plan {
    let mut children: FxHashMap<u64, Vec<BlockHash>> =
        FxHashMap::with_capacity_and_hasher(ledger.len(), Default::default());
    let mut roots = Vec::new();
    let mut orphans = Vec::new();
    for block in ledger.iter() {
        match block.parent {
            Parent::Root => roots.push(block.key.hash),
            Parent::Block(parent) if ledger.contains(parent) => {
                children.entry(parent.0).or_default().push(block.key.hash)
            }
            Parent::Block(_) | Parent::Unknown => orphans.push(block.key.hash),
        }
    }
    roots.sort_unstable();
    orphans.sort_unstable();
    for siblings in children.values_mut() {
        siblings.sort_unstable();
    }

    let mut planner = Planner {
        ledger,
        children,
        seen: FxHashSet::with_capacity_and_hasher(ledger.len(), Default::default()),
        plan: Plan {
            order: Vec::with_capacity(ledger.len()),
            runs: Vec::new(),
        },
    };
    planner.emit(roots, false);
    planner.emit(orphans, true);
    if planner.plan.order.len() < ledger.len() {
        // Whatever is left sits on a parent cycle or hangs off one (a hash collision or a
        // broken publisher): reachable from no root and no orphan. Export it anyway so the
        // consumer holds what the ledger holds. Each pass cuts in at the lowest hash not yet
        // exported and flags that run an orphan; what is reachable from it follows as
        // ordinary runs.
        let mut cyclic = ledger
            .iter()
            .map(|block| block.key.hash)
            .filter(|hash| !planner.seen.contains(&hash.0))
            .collect::<Vec<_>>();
        cyclic.sort_unstable();
        for top in cyclic {
            planner.emit(vec![top], true);
        }
    }
    planner.plan
}

struct Planner<'a, P> {
    ledger: &'a Ledger<P>,
    children: FxHashMap<u64, Vec<BlockHash>>,
    seen: FxHashSet<u64>,
    plan: Plan,
}

impl<P> Planner<'_, P> {
    fn emit(&mut self, tops: Vec<BlockHash>, orphan: bool) {
        let mut queue = tops
            .into_iter()
            .map(|top| (top, orphan))
            .collect::<VecDeque<_>>();
        while let Some((start, orphan)) = queue.pop_front() {
            if !self.seen.insert(start.0) {
                continue;
            }
            let Some(first) = self.ledger.get(start) else {
                continue;
            };
            let begin = self.plan.order.len();
            self.plan.order.push(start);
            let mut tail = start;
            let branches = loop {
                let kids = self.children.remove(&tail.0).unwrap_or_default();
                if let [only] = kids[..]
                    && !self.seen.contains(&only.0)
                    && self
                        .ledger
                        .get(only)
                        .is_some_and(|kid| kid.counts == first.counts)
                {
                    self.seen.insert(only.0);
                    self.plan.order.push(only);
                    tail = only;
                    continue;
                }
                break kids;
            };
            self.plan.runs.push(SnapshotRun {
                parent: first.parent,
                orphan,
                counts: first.counts,
                blocks: begin..self.plan.order.len(),
            });
            queue.extend(branches.into_iter().map(|kid| (kid, false)));
        }
    }
}

/// Replays a whole ledger into an index, parents first.
pub(crate) fn project<P, I: StreamIndex>(ledger: &Ledger<P>, index: &mut I) {
    let plan = plan(ledger);
    let mut keys = Vec::new();
    for run in &plan.runs {
        keys.clear();
        keys.extend(
            plan.order[run.blocks.clone()]
                .iter()
                .filter_map(|hash| ledger.get(*hash))
                .map(|block| block.key),
        );
        for tier in run.counts.tiers().iter() {
            index.stored(tier, run.parent, &keys);
        }
    }
}

/// One rank's state after every batch below `cursor().next_seq`, and nothing later.
///
/// The stream's owner makes the copy between two batches; ordering and encoding happen on the
/// caller's thread, once, and every consumer that joins while the image is cached shares the
/// result. It is `Send + Sync` when the payload is.
pub struct SnapshotImage<P> {
    cursor: Cursor,
    health: Health,
    ledger: Ledger<P>,
    plan: OnceLock<Plan>,
    memo: OnceLock<Box<dyn Any + Send + Sync>>,
}

impl<P> fmt::Debug for SnapshotImage<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SnapshotImage")
            .field("cursor", &self.cursor)
            .field("health", &self.health)
            .field("blocks", &self.ledger.len())
            .finish_non_exhaustive()
    }
}

impl<P> SnapshotImage<P> {
    pub(crate) fn new(cursor: Cursor, health: Health, ledger: Ledger<P>) -> Self {
        Self {
            cursor,
            health,
            ledger,
            plan: OnceLock::new(),
            memo: OnceLock::new(),
        }
    }

    /// Where a consumer continues after loading this image.
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    pub fn health(&self) -> Health {
        self.health
    }

    pub fn len(&self) -> usize {
        self.ledger.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ledger.is_empty()
    }

    /// The frozen ledger. An in-process consumer can clone it instead of loading runs.
    pub fn ledger(&self) -> &Ledger<P> {
        &self.ledger
    }

    /// The image as parent-first runs. The first call orders the image: a sort and a few hash
    /// probes per block, with an allocation per parent, so make it off the stream's owner.
    /// Later calls, from any thread, reuse that order; callers that race the first one block
    /// until it is done.
    pub fn runs(&self) -> &[SnapshotRun] {
        &self.plan().runs
    }

    /// The blocks of one run of this image, in chain order. `run` must come from this image's
    /// [`runs`](Self::runs): a run of another image names other blocks, or panics.
    pub fn run_blocks<'a>(
        &'a self,
        run: &SnapshotRun,
    ) -> impl Iterator<Item = BlockView<'a, P>> + use<'a, P> {
        self.plan().order[run.blocks.clone()]
            .iter()
            .filter_map(|hash| self.ledger.get(*hash))
    }

    /// The chain of keys of one run, which is what an index takes.
    pub fn run_keys(&self, run: &SnapshotRun) -> Vec<BlockKey> {
        self.run_blocks(run).map(|block| block.key).collect()
    }

    /// Host data derived from this image, built by the first caller and shared by the rest:
    /// the encoded stored messages, for example. Keep it independent of any one request (cut
    /// chunks per request with [`chunk_by`]). `None` if the image already holds a memo of
    /// another type.
    pub fn memo<T: Any + Send + Sync>(&self, build: impl FnOnce(&Self) -> T) -> Option<&T> {
        self.memo
            .get_or_init(|| Box::new(build(self)))
            .downcast_ref()
    }

    fn plan(&self) -> &Plan {
        self.plan.get_or_init(|| plan(&self.ledger))
    }
}

/// Cuts `items` into consecutive chunks whose measured size stays within `max_bytes`. An item
/// larger than the cap gets a chunk of its own.
pub fn chunk_by<T, F: FnMut(&T) -> usize>(
    items: &[T],
    max_bytes: usize,
    size: F,
) -> ChunkBy<'_, T, F> {
    ChunkBy {
        rest: items,
        max_bytes,
        size,
    }
}

/// See [`chunk_by`].
pub struct ChunkBy<'a, T, F> {
    rest: &'a [T],
    max_bytes: usize,
    size: F,
}

impl<T, F> fmt::Debug for ChunkBy<'_, T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkBy")
            .field("items_left", &self.rest.len())
            .field("max_bytes", &self.max_bytes)
            .finish_non_exhaustive()
    }
}

impl<'a, T, F: FnMut(&T) -> usize> Iterator for ChunkBy<'a, T, F> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        let mut bytes = 0usize;
        let mut take = 0;
        for item in self.rest {
            bytes = bytes.saturating_add((self.size)(item));
            if take > 0 && bytes > self.max_bytes {
                break;
            }
            take += 1;
        }
        let (chunk, rest) = self.rest.split_at(take);
        self.rest = rest;
        Some(chunk)
    }
}
