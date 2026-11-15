// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run-compressed edge storage for the lower tier.
//!
//! Consecutive edges of one chain are stored together as a run: a vector of
//! `(local hash, sequence hash)` blocks plus the owners of each position
//! ([`Coverage`]). A query walks a run with one lock and sequential compares
//! instead of one map lookup per block.
//!
//! Two maps find runs:
//! - `heads` maps the key of a run's first edge to the run;
//! - `blocks` maps every stored block's sequence hash to its run and position,
//!   so a query can enter mid-run and a remove can find its block.
//!
//! An edge `(parent, local)` therefore lives in exactly one place: right after
//! `parent`'s block inside a run, or at the head of a run. A branch never
//! splits a run; the diverging edge heads a new one.
//!
//! A position nobody holds is a hole. It reads as a missing edge, but stays in
//! the run so that storing the block again only changes coverage. A run is
//! freed when nothing in it is held, and its trailing holes are trimmed once
//! they outnumber its held blocks.
//!
//! # Locking
//!
//! - `structure` serializes every change to which blocks exist: appending
//!   blocks, creating a run, trimming, freeing. Queries and coverage changes
//!   never take it.
//! - A store or remove first works without `structure`. When it finds that it
//!   has to change the structure, it takes `structure` and looks again.
//! - Order: `structure`, then one run lock, then a `heads` shard, then a
//!   `blocks` shard. No run lock is taken while a map guard is held, and a
//!   thread holds one run lock at a time.
//! - A run reference read from a map is checked after locking: the run must
//!   not be dead and must hold the expected block at that position.
//! - A new run enters `heads` before its lock is first released, so a scan of
//!   `heads` sees every run a write has touched.

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry as MapEntry;
use parking_lot::{ArcRwLockReadGuard, Mutex, RawRwLock, RwLock, RwLockUpgradableReadGuard};
use rustc_hash::FxBuildHasher;

use super::coverage::Coverage;
use super::{
    EdgeCursor, EdgeLookup, EdgeRef, IndexedResidencyOwner, OwnerId, OwnerSet, OwnerTable,
    ROOT_PARENT, TransitionKey,
};
use crate::prelude::{ExternalSequenceBlockHash, KvCacheStoredBlockData, LocalBlockHash};

/// Longest run. A walk pays one map lookup and one lock per run, so longer
/// runs gain little, while a run's worst case grows with its length: when
/// owners differ block by block, every position is its own coverage segment
/// and one store or remove moves up to all of them. Tests use a short limit
/// so that small chains span several runs.
const MAX_RUN_BLOCKS: usize = if cfg!(test) { 16 } else { 512 };

/// How often tests reached the paths that are rare or need a race.
#[cfg(test)]
#[derive(Default)]
pub(super) struct Reached {
    /// A run was freed.
    pub(super) freed: std::sync::atomic::AtomicUsize,
    /// A run's trailing holes were trimmed after a remove.
    pub(super) trimmed: std::sync::atomic::AtomicUsize,
    /// A store dropped trailing holes to extend the run with another block.
    pub(super) dropped_holes: std::sync::atomic::AtomicUsize,
    /// A store found its edge created while it waited for `structure`.
    pub(super) looked_again: std::sync::atomic::AtomicUsize,
    /// A write found a run reference that was no longer valid.
    pub(super) stale: std::sync::atomic::AtomicUsize,
    /// A walk moved from one run to the next.
    pub(super) cursor_hops: std::sync::atomic::AtomicUsize,
}

macro_rules! reached {
    ($store:expr, $what:ident) => {
        #[cfg(test)]
        $store
            .reached
            .$what
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
}

type Run = RwLock<RunState>;

struct RunState {
    /// Sequence hash before `blocks[0]`; fixed for the run's life.
    parent: ExternalSequenceBlockHash,
    /// Edge `i` is `(blocks[i - 1].1, blocks[i].0) -> blocks[i].1`, with
    /// `parent` standing in before the first block.
    blocks: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    coverage: Coverage,
    /// Set when the run is unlinked from the maps.
    dead: bool,
}

/// How a caller reached a run.
#[derive(Clone, Copy)]
enum Entry {
    /// Through `blocks`: the block at this position should have this hash.
    After(u32, ExternalSequenceBlockHash),
    /// Through `heads`, under this key.
    Head(TransitionKey),
}

impl RunState {
    /// Index of the edge `entry` leads to, or `None` when the reference is
    /// stale: the run died, or the block is no longer where the map said.
    fn entry_index(&self, entry: Entry) -> Option<usize> {
        if self.dead {
            return None;
        }
        match entry {
            Entry::After(pos, hash) => {
                (self.blocks.get(pos as usize)?.1 == hash).then_some(pos as usize + 1)
            }
            Entry::Head(key) => (self.parent == key.parent_hash
                && self.blocks.first()?.0 == key.local_hash)
                .then_some(0),
        }
    }

    /// Whether the run should be freed or trimmed.
    fn needs_reclaim(&self) -> bool {
        let live = self.coverage.live();
        live == 0 || self.coverage.len() - self.coverage.live_end() > live
    }

    /// Drops the blocks at and past `len` and returns their hashes.
    fn truncate(&mut self, len: usize) -> Vec<ExternalSequenceBlockHash> {
        let dropped = self.blocks.drain(len..).map(|block| block.1).collect();
        self.coverage.truncate(len as u32);
        if self.blocks.capacity() > 4 * self.blocks.len().max(4) {
            self.blocks.shrink_to_fit();
        }
        dropped
    }
}

struct BlockRef {
    run: Arc<Run>,
    pos: u32,
}

// One per stored block.
const _: () = assert!(size_of::<BlockRef>() == 16);

/// The owner a write is for.
#[derive(Clone, Copy)]
pub(super) struct Owner<'a> {
    pub(super) id: OwnerId,
    pub(super) owner: IndexedResidencyOwner,
    pub(super) table: &'a OwnerTable,
}

/// What one step of a store did.
#[derive(Default)]
struct Progress {
    /// Leading blocks of the event now in place.
    matched: usize,
    /// Positions the owner did not hold before.
    added: u32,
    /// The next block contradicts what is stored; the event stops.
    conflict: bool,
}

/// Hashes whose ownership changed, collected only when someone listens.
type Changed<'a> = Option<&'a mut Vec<ExternalSequenceBlockHash>>;

pub(super) struct RunStore {
    heads: DashMap<TransitionKey, Arc<Run>, FxBuildHasher>,
    blocks: DashMap<ExternalSequenceBlockHash, BlockRef, FxBuildHasher>,
    structure: Mutex<()>,
    #[cfg(test)]
    pub(super) reached: Reached,
}

impl RunStore {
    pub(super) fn new() -> Self {
        Self {
            heads: DashMap::with_hasher(FxBuildHasher),
            blocks: DashMap::with_hasher(FxBuildHasher),
            structure: Mutex::new(()),
            #[cfg(test)]
            reached: Reached::default(),
        }
    }

    /// The run and position `blocks` gives for `hash`. The caller validates it
    /// after locking the run.
    fn locate(&self, hash: ExternalSequenceBlockHash) -> Option<(Arc<Run>, u32)> {
        self.blocks
            .get(&hash)
            .map(|block| (block.run.clone(), block.pos))
    }

    fn head(&self, key: &TransitionKey) -> Option<Arc<Run>> {
        self.heads.get(key).map(|run| run.value().clone())
    }

    fn runs(&self) -> Vec<Arc<Run>> {
        self.heads.iter().map(|run| run.value().clone()).collect()
    }

    /// Stores `blocks` after `parent` for `who`, up to the first conflict.
    /// Returns how many of them `who` did not already hold.
    pub(super) fn store(
        &self,
        who: Owner<'_>,
        parent: Option<ExternalSequenceBlockHash>,
        blocks: &[KvCacheStoredBlockData],
        mut changed: Changed<'_>,
    ) -> u32 {
        let mut parent = parent;
        let mut next = 0;
        let mut added = 0;
        // Whether the edge after `parent` may sit inside the run holding
        // `parent`. After a step that made progress it cannot: the step ended
        // where that run stopped matching.
        let mut look_inside = true;
        while next < blocks.len() {
            let rest = &blocks[next..];
            let key = TransitionKey::new(parent, rest[0].tokens_hash);

            let mut progress = None;
            if look_inside
                && let Some(hash) = parent
                && let Some((run, pos)) = self.locate(hash)
            {
                progress = self.cover(&run, Entry::After(pos, hash), who, rest, &mut changed);
            }
            if progress.is_none()
                && let Some(run) = self.head(&key)
            {
                progress = self.cover(&run, Entry::Head(key), who, rest, &mut changed);
            }
            let progress = progress.unwrap_or_else(|| self.place(who, parent, rest, &mut changed));

            added += progress.added;
            if progress.conflict {
                break;
            }
            // Only `place` reports no progress without a conflict: the edge
            // appeared while it waited, so look again.
            look_inside = progress.matched == 0;
            if progress.matched > 0 {
                next += progress.matched;
                parent = Some(rest[progress.matched - 1].block_hash);
            }
        }
        added
    }

    /// Adds `who` to the leading `blocks` already in `run` at `entry`. `None`
    /// when the reference is stale or the run does not hold the first block.
    fn cover(
        &self,
        run: &Arc<Run>,
        entry: Entry,
        who: Owner<'_>,
        blocks: &[KvCacheStoredBlockData],
        changed: &mut Changed<'_>,
    ) -> Option<Progress> {
        let state = run.upgradable_read();
        let start = state.entry_index(entry);
        #[cfg(test)]
        {
            if start.is_none() {
                reached!(self, stale);
            }
        }
        let start = start?;
        let mut progress = Progress::default();
        for (stored, block) in state.blocks[start..].iter().zip(blocks) {
            if stored.0 != block.tokens_hash {
                break;
            }
            if stored.1 != block.block_hash {
                progress.conflict = true;
                break;
            }
            progress.matched += 1;
        }
        if progress.matched == 0 {
            return progress.conflict.then_some(progress);
        }
        let range = start as u32..(start + progress.matched) as u32;
        if !state.coverage.covers(range.clone(), who.id, who.owner) {
            let mut state = RwLockUpgradableReadGuard::upgrade(state);
            let RunState {
                blocks: stored,
                coverage,
                ..
            } = &mut *state;
            progress.added = coverage.add(range, who.id, who.owner, who.table, |span| {
                if let Some(changed) = changed {
                    changed.extend(
                        stored[span.start as usize..span.end as usize]
                            .iter()
                            .map(|block| block.1),
                    );
                }
            });
        }
        Some(progress)
    }

    /// Creates the edge for `blocks[0]` after `parent`, and as many of the
    /// following blocks as belong in the same run. Reports no progress when
    /// the edge exists by now.
    fn place(
        &self,
        who: Owner<'_>,
        parent: Option<ExternalSequenceBlockHash>,
        blocks: &[KvCacheStoredBlockData],
        changed: &mut Changed<'_>,
    ) -> Progress {
        let _structure = self.structure.lock();
        let key = TransitionKey::new(parent, blocks[0].tokens_hash);

        // The run to extend, when `parent` is its last block.
        let mut tail = None;
        if let Some(hash) = parent
            && let Some((run, pos)) = self.locate(hash)
        {
            let mut state = run.write();
            if let Some(after) = state.entry_index(Entry::After(pos, hash)) {
                if after == state.blocks.len() {
                    drop(state);
                    tail = Some(run);
                } else if state.blocks[after].0 == key.local_hash {
                    reached!(self, looked_again);
                    return Progress::default();
                } else if state.coverage.live_end() as usize <= after {
                    // Only holes follow the parent: drop them and extend here.
                    reached!(self, dropped_holes);
                    let dropped = state.truncate(after);
                    drop(state);
                    self.forget(&run, dropped);
                    tail = Some(run);
                }
            }
        }
        if self.heads.contains_key(&key) {
            reached!(self, looked_again);
            return Progress::default();
        }

        let (run, len) = match tail {
            Some(run) => {
                let len = run.read().blocks.len();
                if len < MAX_RUN_BLOCKS {
                    (run, len)
                } else {
                    (self.new_run(key), 0)
                }
            }
            None => (self.new_run(key), 0),
        };

        // Claim the hashes first. Until the run holds the blocks, a reader
        // that finds one of these entries sees a position past the run's end
        // and treats the block as absent.
        let mut progress = Progress::default();
        for (offset, block) in blocks.iter().enumerate().take(MAX_RUN_BLOCKS - len) {
            if offset > 0
                && self.heads.contains_key(&TransitionKey::new(
                    Some(blocks[offset - 1].block_hash),
                    block.tokens_hash,
                ))
            {
                // A run already starts at this edge; the caller continues there.
                break;
            }
            match self.blocks.entry(block.block_hash) {
                MapEntry::Occupied(_) => {
                    // This hash is already stored under another edge.
                    progress.conflict = true;
                    break;
                }
                MapEntry::Vacant(slot) => {
                    slot.insert(BlockRef {
                        run: run.clone(),
                        pos: (len + offset) as u32,
                    });
                }
            }
            progress.matched += 1;
        }
        if progress.matched == 0 {
            return progress;
        }

        let placed = &blocks[..progress.matched];
        {
            let mut state = run.write();
            state.blocks.extend(
                placed
                    .iter()
                    .map(|block| (block.tokens_hash, block.block_hash)),
            );
            state
                .coverage
                .push(placed.len() as u32, Some(OwnerSet::Single(who.id)));
            if len == 0 {
                self.heads.insert(key, run.clone());
            }
        }
        progress.added = placed.len() as u32;
        if let Some(changed) = changed {
            changed.extend(placed.iter().map(|block| block.block_hash));
        }
        progress
    }

    fn new_run(&self, key: TransitionKey) -> Arc<Run> {
        Arc::new(RwLock::new(RunState {
            parent: key.parent_hash,
            blocks: Vec::new(),
            coverage: Coverage::default(),
            dead: false,
        }))
    }

    /// Removes `who` from the blocks with these hashes. Returns how many of
    /// them `who` held.
    pub(super) fn remove(
        &self,
        who: Owner<'_>,
        hashes: &[ExternalSequenceBlockHash],
        mut changed: Changed<'_>,
    ) -> u32 {
        let mut removed = 0;
        let mut next = 0;
        while next < hashes.len() {
            let hash = hashes[next];
            next += 1;
            let Some((run, pos)) = self.locate(hash) else {
                continue;
            };
            let state = run.upgradable_read();
            if state.entry_index(Entry::After(pos, hash)).is_none() {
                continue;
            }
            // Take the event's following hashes while they are this block's
            // neighbours, in either direction.
            let pos = pos as usize;
            let (mut low, mut high) = (pos, pos + 1);
            while next < hashes.len()
                && state
                    .blocks
                    .get(high)
                    .is_some_and(|block| block.1 == hashes[next])
            {
                high += 1;
                next += 1;
            }
            if high == pos + 1 {
                while next < hashes.len() && low > 0 && state.blocks[low - 1].1 == hashes[next] {
                    low -= 1;
                    next += 1;
                }
            }
            let range = low as u32..high as u32;
            if !state.coverage.intersects(range.clone(), who.id, who.owner) {
                continue;
            }
            let state = RwLockUpgradableReadGuard::upgrade(state);
            removed += self.uncover(&run, state, range, who, &mut changed);
        }
        removed
    }

    /// Removes `who` from every block. Returns how many it held.
    pub(super) fn remove_owner(&self, who: Owner<'_>, mut changed: Changed<'_>) -> u32 {
        let mut removed = 0;
        for run in self.runs() {
            let state = run.upgradable_read();
            let all = 0..state.coverage.len();
            if state.dead || !state.coverage.intersects(all.clone(), who.id, who.owner) {
                continue;
            }
            let state = RwLockUpgradableReadGuard::upgrade(state);
            removed += self.uncover(&run, state, all, who, &mut changed);
        }
        removed
    }

    fn uncover(
        &self,
        run: &Arc<Run>,
        mut state: parking_lot::RwLockWriteGuard<'_, RunState>,
        range: std::ops::Range<u32>,
        who: Owner<'_>,
        changed: &mut Changed<'_>,
    ) -> u32 {
        let RunState {
            blocks: stored,
            coverage,
            ..
        } = &mut *state;
        let removed = coverage.remove(range, who.id, who.owner, who.table, |span| {
            if let Some(changed) = changed {
                changed.extend(
                    stored[span.start as usize..span.end as usize]
                        .iter()
                        .map(|block| block.1),
                );
            }
        });
        let reclaim = state.needs_reclaim();
        drop(state);
        if reclaim {
            self.reclaim(run);
        }
        removed
    }

    /// Frees `run` when nothing in it is held, or trims its trailing holes
    /// when they outnumber the held blocks.
    fn reclaim(&self, run: &Arc<Run>) {
        let _structure = self.structure.lock();
        let mut state = run.write();
        if state.dead || !state.needs_reclaim() {
            return;
        }
        if state.coverage.live() == 0 {
            reached!(self, freed);
            let blocks = std::mem::take(&mut state.blocks);
            state.coverage = Coverage::default();
            state.dead = true;
            let key = TransitionKey {
                parent_hash: state.parent,
                local_hash: blocks[0].0,
            };
            drop(state);
            self.heads.remove_if(&key, |_, head| Arc::ptr_eq(head, run));
            self.forget(run, blocks.into_iter().map(|block| block.1));
        } else {
            reached!(self, trimmed);
            let live_end = state.coverage.live_end() as usize;
            let dropped = state.truncate(live_end);
            drop(state);
            self.forget(run, dropped);
        }
    }

    fn forget(&self, run: &Arc<Run>, hashes: impl IntoIterator<Item = ExternalSequenceBlockHash>) {
        for hash in hashes {
            self.blocks
                .remove_if(&hash, |_, block| Arc::ptr_eq(&block.run, run));
        }
    }

    /// Calls `visit` for each stretch of held blocks sharing one owner set,
    /// with the hash before the stretch and its `(local, sequence)` blocks.
    /// Stretches of one run come in chain order.
    pub(super) fn for_each_held(
        &self,
        mut visit: impl FnMut(
            &OwnerSet,
            Option<ExternalSequenceBlockHash>,
            &[(LocalBlockHash, ExternalSequenceBlockHash)],
        ),
    ) {
        for run in self.runs() {
            let state = run.read();
            for (span, owners) in state.coverage.segments() {
                let Some(owners) = owners else {
                    continue;
                };
                let before = match span.start {
                    0 => (state.parent != ROOT_PARENT).then_some(state.parent),
                    start => Some(state.blocks[start as usize - 1].1),
                };
                visit(
                    owners,
                    before,
                    &state.blocks[span.start as usize..span.end as usize],
                );
            }
        }
    }

    /// Sizes of the structure, for benchmarks and tests.
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn census(&self) -> RunCensus {
        let mut census = RunCensus::default();
        for run in self.runs() {
            let state = run.read();
            let len = state.blocks.len();
            census.runs += 1;
            census.stored_blocks += len;
            census.held_blocks += state.coverage.live() as usize;
            census.segments += state.coverage.segments().count();
            census.longest_run = census.longest_run.max(len);
            census.runs_by_length[match len {
                0..=1 => 0,
                2..=15 => 1,
                16..=255 => 2,
                _ => 3,
            }] += 1;
        }
        census
    }
}

/// Sizes of a lower-tier index's run storage.
#[cfg(any(test, feature = "bench"))]
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct RunCensus {
    pub runs: usize,
    /// Blocks physically in runs, held or not.
    pub stored_blocks: usize,
    /// Blocks held by at least one owner.
    pub held_blocks: usize,
    /// Stretches of one owner set, including holes.
    pub segments: usize,
    pub longest_run: usize,
    /// Runs of 1, 2-15, 16-255 and 256 or more blocks.
    pub runs_by_length: [usize; 4],
}

impl EdgeLookup for RunStore {
    type Cursor<'a> = RunCursor<'a>;

    fn cursor(&self) -> RunCursor<'_> {
        RunCursor {
            store: self,
            run: None,
            index: 0,
            segment: 0,
        }
    }

    fn root_owners<R>(
        &self,
        local_hash: LocalBlockHash,
        read: impl FnOnce(&OwnerSet) -> R,
    ) -> Option<R> {
        let key = TransitionKey::new(None, local_hash);
        let run = self.head(&key)?;
        let state = run.read();
        state.entry_index(Entry::Head(key))?;
        state.coverage.owners_at(0).map(read)
    }
}

/// A walk's place in the run storage: a read lock on one run and the block
/// the walk last matched there.
pub(super) struct RunCursor<'a> {
    store: &'a RunStore,
    run: Option<ArcRwLockReadGuard<RawRwLock, RunState>>,
    index: usize,
    segment: usize,
}

impl RunCursor<'_> {
    /// Locks `run` and stands on the edge `entry` leads to when that edge is
    /// for `local_hash`. Leaves the cursor outside any run otherwise.
    fn enter(&mut self, run: Arc<Run>, entry: Entry, local_hash: LocalBlockHash) -> bool {
        let state = run.read_arc();
        let Some(index) = state.entry_index(entry) else {
            return false;
        };
        if state
            .blocks
            .get(index)
            .is_none_or(|block| block.0 != local_hash)
        {
            return false;
        }
        self.index = index;
        self.segment = state.coverage.segment_index(index as u32);
        self.run = Some(state);
        true
    }
}

impl EdgeCursor for RunCursor<'_> {
    #[inline]
    fn edge(
        &mut self,
        parent: Option<ExternalSequenceBlockHash>,
        local_hash: LocalBlockHash,
    ) -> Option<EdgeRef<'_>> {
        // Step to the next block of the run the walk is already in.
        let mut inside = false;
        let mut looked_inside = false;
        if let Some(state) = &self.run
            && parent == Some(state.blocks[self.index].1)
        {
            looked_inside = true;
            let next = self.index + 1;
            if state
                .blocks
                .get(next)
                .is_some_and(|block| block.0 == local_hash)
            {
                self.index = next;
                if next as u32 >= state.coverage.segment_end(self.segment) {
                    self.segment += 1;
                }
                inside = true;
            }
        }
        if !inside {
            // Release the run before locking another.
            self.run = None;
            // The edge follows its parent's block in that block's run, or
            // heads a run.
            let entered = !looked_inside
                && parent.is_some_and(|hash| {
                    self.store.locate(hash).is_some_and(|(run, pos)| {
                        self.enter(run, Entry::After(pos, hash), local_hash)
                    })
                });
            if !entered {
                let key = TransitionKey::new(parent, local_hash);
                let run = self.store.head(&key)?;
                if !self.enter(run, Entry::Head(key), local_hash) {
                    return None;
                }
                if looked_inside {
                    reached!(self.store, cursor_hops);
                }
            }
        }
        let state = self.run.as_ref()?;
        // A hole reads as a missing edge.
        let owners = state.coverage.segment_owners(self.segment)?;
        Some(EdgeRef {
            child_hash: state.blocks[self.index].1,
            owners,
        })
    }
}

#[cfg(test)]
impl RunStore {
    /// Checks every structural rule. Call when no write is in flight.
    pub(super) fn assert_invariants(&self) {
        let mut stored = 0;
        let heads: Vec<(TransitionKey, Arc<Run>)> = self
            .heads
            .iter()
            .map(|head| (*head.key(), head.value().clone()))
            .collect();
        for (key, run) in &heads {
            let (key, run) = (*key, run);
            let state = run.read();
            assert!(!state.dead, "dead run in heads");
            assert_eq!(state.parent, key.parent_hash);
            assert_eq!(state.blocks[0].0, key.local_hash);
            assert!(state.blocks.len() <= MAX_RUN_BLOCKS);
            assert_eq!(state.coverage.len() as usize, state.blocks.len());
            state.coverage.assert_invariants();
            assert!(!state.needs_reclaim(), "run left unreclaimed");
            for (pos, block) in state.blocks.iter().enumerate() {
                let (indexed, indexed_pos) =
                    self.locate(block.1).expect("block missing from index");
                assert!(
                    Arc::ptr_eq(&indexed, run),
                    "block indexed under another run"
                );
                assert_eq!(indexed_pos as usize, pos);
                if pos > 0 {
                    let inner = TransitionKey::new(Some(state.blocks[pos - 1].1), block.0);
                    assert!(
                        !self.heads.contains_key(&inner),
                        "edge stored both inside a run and as a head"
                    );
                }
            }
            stored += state.blocks.len();
        }
        assert_eq!(self.blocks.len(), stored, "stale block index entries");
    }

    /// How many runs start at the root, after a block that is not stored,
    /// after a block inside another run, and after another run's last block.
    pub(super) fn head_kinds(&self) -> [usize; 4] {
        let mut kinds = [0; 4];
        let parents: Vec<ExternalSequenceBlockHash> = self
            .heads
            .iter()
            .map(|head| head.key().parent_hash)
            .collect();
        for parent in parents {
            let kind = if parent == ROOT_PARENT {
                0
            } else {
                match self.locate(parent) {
                    None => 1,
                    Some((run, pos)) => {
                        if pos as usize + 1 < run.read().blocks.len() {
                            2
                        } else {
                            3
                        }
                    }
                }
            };
            kinds[kind] += 1;
        }
        kinds
    }

    /// Every held edge: its key, child hash and owners.
    pub(super) fn held_edges(
        &self,
    ) -> rustc_hash::FxHashMap<TransitionKey, (ExternalSequenceBlockHash, OwnerSet)> {
        let mut edges = rustc_hash::FxHashMap::default();
        self.for_each_held(|owners, before, blocks| {
            let mut parent = before;
            for (local_hash, hash) in blocks {
                let previous = edges.insert(
                    TransitionKey::new(parent, *local_hash),
                    (*hash, owners.clone()),
                );
                assert!(previous.is_none(), "edge stored twice");
                parent = Some(*hash);
            }
        });
        edges
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::super::test_owners::indexed_owners;
    use super::*;
    use crate::testing::stored_blocks_with_sequence_hashes;

    /// The branches that only a race between lanes reaches, driven directly.
    #[test]
    fn writes_check_again_what_they_resolved_without_a_lock() {
        let store = RunStore::new();
        let table = OwnerTable::default();
        let owners = indexed_owners(2);
        let who = |n: usize| Owner {
            id: table.intern(owners[n]),
            owner: owners[n],
            table: &table,
        };
        let locals = [1, 2, 3].map(LocalBlockHash);
        let blocks = stored_blocks_with_sequence_hashes(&locals, &[101, 102, 103]);
        let first = blocks[0].block_hash;
        assert_eq!(store.store(who(0), None, &blocks, None), 3);

        // Another lane created the edge while this one waited for `structure`,
        // as a run's head or inside the parent's run.
        for (parent, rest) in [(None, &blocks[..]), (Some(first), &blocks[1..])] {
            let progress = store.place(who(1), parent, rest, &mut None);
            assert_eq!((progress.matched, progress.conflict), (0, false));
        }
        assert_eq!(store.reached.looked_again.load(Ordering::Relaxed), 2);

        // The run was freed after this lane read its reference.
        let (run, pos) = store.locate(first).unwrap();
        let hashes: Vec<_> = blocks.iter().map(|block| block.block_hash).collect();
        assert_eq!(store.remove(who(0), &hashes, None), 3);
        let entry = Entry::After(pos, first);
        assert!(
            store
                .cover(&run, entry, who(1), &blocks[1..], &mut None)
                .is_none()
        );
        assert_eq!(store.reached.stale.load(Ordering::Relaxed), 1);
        assert_eq!(store.store(who(1), Some(first), &blocks[1..], None), 2);
        store.assert_invariants();
    }
}
