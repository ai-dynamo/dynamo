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
//! A run grows at both ends. Blocks stored after their parent extend the run
//! that ends with it. Blocks stored before their child, as a write-back cache
//! sends them, go in front of the run that starts after them, and that run's
//! head edge moves inside it. An edge only ever moves that way, from a head
//! to inside the same run. Two runs are never joined: a block that arrives
//! between a run that ends at its parent and a run that starts after it
//! extends the first.
//!
//! A position nobody holds is a hole. It reads as a missing edge, but stays in
//! the run so that storing the block again only changes coverage. A run is
//! freed when nothing in it is held, and its trailing holes are trimmed once
//! they outnumber its held blocks.
//!
//! # Locking
//!
//! - `structure` serializes every change to which blocks exist: adding blocks
//!   at either end of a run, creating a run, trimming, freeing. Queries and
//!   coverage changes never take it. It owns the map that finds the run to
//!   grow at the front.
//! - A store or remove first works without `structure`. When it finds that it
//!   has to change the structure, it takes `structure` and looks again.
//! - Order: `structure`, then one run lock, then a `heads` shard, then a
//!   `blocks` shard. No run lock is taken while a map guard is held, and a
//!   thread holds one run lock at a time.
//! - A run reference read from a map is checked after locking: the run must
//!   not be dead and must hold the expected block at that place.
//! - A reader looks an edge up in `heads` first and in `blocks` second. A
//!   run's head moves inside it only after `blocks` names the new blocks, so
//!   that order finds an edge that existed throughout.
//! - A new run enters `heads` before its lock is first released, and a run's
//!   head is moved under its lock. A scan copies `heads` under `structure`, so
//!   it sees every run a write has touched exactly once.

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::entry::Entry as MapEntry;
use parking_lot::{ArcRwLockReadGuard, Mutex, RawRwLock, RwLock, RwLockUpgradableReadGuard};
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::blocks::Blocks;
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

/// Offset of a new run's first block. Offsets fall as a run grows at the front
/// and wrap; tests start low so that they do.
const FIRST_OFFSET: u32 = if cfg!(test) { 2 } else { 1 << 31 };

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
    /// A store put blocks in front of the run that follows them.
    pub(super) prepended: std::sync::atomic::AtomicUsize,
    /// The run that follows a store's blocks had no room for them.
    pub(super) follower_full: std::sync::atomic::AtomicUsize,
    /// A store extended a run up to the start of another.
    pub(super) bridged: std::sync::atomic::AtomicUsize,
    /// A walk found an edge inside the run whose head it had been.
    pub(super) head_moved: std::sync::atomic::AtomicUsize,
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
    /// Sequence hash before `blocks[0]`. It changes when the run grows at the
    /// front.
    parent: ExternalSequenceBlockHash,
    /// Offset of `blocks[0]`. A block keeps its offset while it is in the
    /// run, so growing at the front lowers this instead.
    front: u32,
    /// Edge `i` is `(blocks[i - 1].1, blocks[i].0) -> blocks[i].1`, with
    /// `parent` standing in before the first block.
    blocks: Blocks,
    coverage: Coverage,
    /// Set when the run is unlinked from the maps.
    dead: bool,
}

/// How a caller reached a run.
#[derive(Clone, Copy)]
enum Entry {
    /// Through `blocks`: the block at this offset should have this hash.
    After(u32, ExternalSequenceBlockHash),
    /// Through `heads`, under this key.
    Head(TransitionKey),
}

impl RunState {
    /// Index of the block at `off`. Past the end for an offset the run does
    /// not hold, including one before its front.
    fn index_of(&self, off: u32) -> usize {
        off.wrapping_sub(self.front) as usize
    }

    /// Index of the edge `entry` leads to, or `None` when the reference is
    /// stale: the run died, the block is no longer where the map said, or the
    /// run's head moved.
    fn entry_index(&self, entry: Entry) -> Option<usize> {
        if self.dead {
            return None;
        }
        match entry {
            Entry::After(off, hash) => {
                let index = self.index_of(off);
                (self.blocks.get(index)?.1 == hash).then_some(index + 1)
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
        self.coverage.truncate(len as u32);
        self.blocks.truncate(len)
    }
}

struct BlockRef {
    run: Arc<Run>,
    off: u32,
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

/// One head run that follows each sequence hash: its `parent`. Blocks stored
/// before their child go in front of that run. Runs that share a parent keep
/// the first of them here, and runs at the root none.
type Followers = FxHashMap<ExternalSequenceBlockHash, Arc<Run>>;

pub(super) struct RunStore {
    heads: DashMap<TransitionKey, Arc<Run>, FxBuildHasher>,
    blocks: DashMap<ExternalSequenceBlockHash, BlockRef, FxBuildHasher>,
    structure: Mutex<Followers>,
    #[cfg(test)]
    pub(super) reached: Reached,
}

impl RunStore {
    pub(super) fn new() -> Self {
        Self {
            heads: DashMap::with_hasher(FxBuildHasher),
            blocks: DashMap::with_hasher(FxBuildHasher),
            structure: Mutex::new(Followers::default()),
            #[cfg(test)]
            reached: Reached::default(),
        }
    }

    /// The run and offset `blocks` gives for `hash`. The caller validates it
    /// after locking the run.
    fn locate(&self, hash: ExternalSequenceBlockHash) -> Option<(Arc<Run>, u32)> {
        self.blocks
            .get(&hash)
            .map(|block| (block.run.clone(), block.off))
    }

    fn head(&self, key: &TransitionKey) -> Option<Arc<Run>> {
        self.heads.get(key).map(|run| run.value().clone())
    }

    /// Every run. `heads` is copied under `structure`: a head that moves
    /// while the copy is made could otherwise be missed at both its keys.
    fn runs(&self) -> Vec<Arc<Run>> {
        let mut runs = Vec::with_capacity(self.heads.len());
        let _structure = self.structure.lock();
        runs.extend(self.heads.iter().map(|run| run.value().clone()));
        runs
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
        // `parent`. After a `cover` step that made progress it cannot: the
        // step ended where that run stopped matching.
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
            let covered = progress.is_some();
            let progress = progress.unwrap_or_else(|| self.place(who, parent, rest, &mut changed));

            added += progress.added;
            if progress.conflict {
                break;
            }
            // `place` can leave the next edge inside the run it grew: it puts
            // blocks in front of the run that starts with that edge. It also
            // reports no progress, without a conflict, when the edge appeared
            // while it waited.
            look_inside = !covered;
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
    /// following blocks as belong in the same run: at the end of the run that
    /// ends with `parent`, else in front of the run that starts after them,
    /// else in a new run. Reports no progress when the edge exists by now.
    fn place(
        &self,
        who: Owner<'_>,
        parent: Option<ExternalSequenceBlockHash>,
        blocks: &[KvCacheStoredBlockData],
        changed: &mut Changed<'_>,
    ) -> Progress {
        let mut followers = self.structure.lock();
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

        let tail = tail.and_then(|run| {
            let (len, end) = {
                let state = run.read();
                let len = state.blocks.len();
                (len, state.front.wrapping_add(len as u32))
            };
            (len < MAX_RUN_BLOCKS).then_some((run, len, end))
        });
        // A new run, unless the blocks go in front of the run after them.
        let (run, len, end) = tail.unwrap_or_else(|| (self.new_run(key), 0, FIRST_OFFSET));

        // Claim the hashes first. Until a run holds the blocks, a reader that
        // finds one of these entries sees an offset outside the run and
        // treats the block as absent.
        let mut progress = Progress::default();
        // The run that starts at the edge the claim stopped at.
        let mut next_run = None;
        for (offset, block) in blocks.iter().enumerate().take(MAX_RUN_BLOCKS - len) {
            if offset > 0 {
                let next =
                    TransitionKey::new(Some(blocks[offset - 1].block_hash), block.tokens_hash);
                next_run = self.head(&next);
                if next_run.is_some() {
                    // A run already starts at this edge; the caller continues there.
                    break;
                }
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
                        off: end.wrapping_add(offset as u32),
                    });
                }
            }
            progress.matched += 1;
        }
        if progress.matched == 0 {
            return progress;
        }

        let placed = &blocks[..progress.matched];
        let last = placed[placed.len() - 1].block_hash;
        #[cfg(test)]
        {
            if len > 0 && (next_run.is_some() || followers.contains_key(&last)) {
                reached!(self, bridged);
            }
        }
        // Blocks that start a run go in front of the run after them instead.
        let prepended = len == 0
            && next_run
                .or_else(|| followers.get(&last).cloned())
                .is_some_and(|follower| self.prepend(&mut followers, &follower, key, placed, who));
        if !prepended {
            let mut state = run.write();
            state.blocks.extend(
                placed
                    .iter()
                    .map(|block| (block.tokens_hash, block.block_hash)),
                MAX_RUN_BLOCKS,
            );
            state
                .coverage
                .push(placed.len() as u32, Some(OwnerSet::Single(who.id)));
            if len == 0 {
                self.heads.insert(key, run.clone());
                if key.parent_hash != ROOT_PARENT {
                    followers
                        .entry(key.parent_hash)
                        .or_insert_with(|| run.clone());
                }
            }
        }
        progress.added = placed.len() as u32;
        if let Some(changed) = changed {
            changed.extend(placed.iter().map(|block| block.block_hash));
        }
        progress
    }

    /// Puts `placed`, already claimed in `blocks`, in front of `follower`,
    /// the run that starts after them, and moves its head to `key`. False
    /// when the run is not that one any more or has no room.
    fn prepend(
        &self,
        followers: &mut Followers,
        follower: &Arc<Run>,
        key: TransitionKey,
        placed: &[KvCacheStoredBlockData],
        who: Owner<'_>,
    ) -> bool {
        let last = placed[placed.len() - 1].block_hash;
        // Only a holder of `structure` changes what is read here, so it holds
        // until the write below.
        let front = {
            let state = follower.read();
            if state.dead || state.parent != last {
                return false;
            }
            if state.blocks.len() + placed.len() > MAX_RUN_BLOCKS {
                reached!(self, follower_full);
                return false;
            }
            state.front.wrapping_sub(placed.len() as u32)
        };
        // Name the blocks' places before the run holds them: their offsets
        // fall before its front until then. A reader that finds the head gone
        // looks here next, and waits on the run's lock for the rest.
        for (index, block) in placed.iter().enumerate() {
            self.blocks.insert(
                block.block_hash,
                BlockRef {
                    run: follower.clone(),
                    off: front.wrapping_add(index as u32),
                },
            );
        }
        let mut state = follower.write();
        let old = TransitionKey {
            parent_hash: state.parent,
            local_hash: state.blocks[0].0,
        };
        state.blocks.push_front(
            placed
                .iter()
                .map(|block| (block.tokens_hash, block.block_hash)),
            MAX_RUN_BLOCKS,
        );
        state
            .coverage
            .push_front(placed.len() as u32, Some(OwnerSet::Single(who.id)));
        state.front = front;
        state.parent = key.parent_hash;
        self.heads.insert(key, follower.clone());
        self.heads
            .remove_if(&old, |_, head| Arc::ptr_eq(head, follower));
        drop(state);
        if followers
            .get(&last)
            .is_some_and(|run| Arc::ptr_eq(run, follower))
        {
            followers.remove(&last);
        }
        if key.parent_hash != ROOT_PARENT {
            followers
                .entry(key.parent_hash)
                .or_insert_with(|| follower.clone());
        }
        reached!(self, prepended);
        true
    }

    fn new_run(&self, key: TransitionKey) -> Arc<Run> {
        Arc::new(RwLock::new(RunState {
            parent: key.parent_hash,
            front: FIRST_OFFSET,
            blocks: Blocks::default(),
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
            let Some((run, off)) = self.locate(hash) else {
                continue;
            };
            let state = run.upgradable_read();
            let Some(after) = state.entry_index(Entry::After(off, hash)) else {
                continue;
            };
            // Take the event's following hashes while they are this block's
            // neighbours, in either direction.
            let pos = after - 1;
            let (mut low, mut high) = (pos, after);
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
        let mut followers = self.structure.lock();
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
            if followers
                .get(&key.parent_hash)
                .is_some_and(|follower| Arc::ptr_eq(follower, run))
            {
                followers.remove(&key.parent_hash);
            }
            self.forget(run, blocks.iter().map(|block| block.1));
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

    /// Sizes of the structure, for benchmarks and tests. Call when no write is in flight.
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn census(&self) -> RunCensus {
        let mut census = RunCensus::default();
        let mut starts = Vec::new();
        for run in self.runs() {
            let state = run.read();
            let len = state.blocks.len();
            if len == 0 {
                continue;
            }
            census.runs += 1;
            census.stored_blocks += len;
            census.held_blocks += state.coverage.live() as usize;
            census.unheld_trailing_blocks += len - state.coverage.live_end() as usize;
            census.spare_slots += state.blocks.spare_slots();
            census.segments += state.coverage.segments().count();
            census.longest_run = census.longest_run.max(len);
            census.runs_by_length[match len {
                0..=1 => 0,
                2..=15 => 1,
                16..=255 => 2,
                _ => 3,
            }] += 1;
            starts.push((state.parent, len));
        }
        // What each run starts after, one run lock at a time.
        for (parent, len) in starts {
            let kind = if parent == ROOT_PARENT {
                0
            } else {
                match self.locate(parent) {
                    None => 1,
                    Some((before, off)) => {
                        let before = before.read();
                        if before.index_of(off) + 1 < before.blocks.len() {
                            2
                        } else {
                            if before.blocks.len() + len <= MAX_RUN_BLOCKS {
                                census.joinable_runs += 1;
                            }
                            3
                        }
                    }
                }
            };
            census.runs_by_start[kind] += 1;
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
    /// Runs that start at the root, after a block that is not stored, after a block inside
    /// another run (a branch), and after another run's last block (a chain cut in two).
    pub runs_by_start: [usize; 4],
    /// Runs that start after another run's last block and would fit in that run: the cuts
    /// that joining runs would remove.
    pub joinable_runs: usize,
    /// Blocks nobody holds at the end of their run; `stored_blocks - held_blocks` counts all.
    pub unheld_trailing_blocks: usize,
    /// Block slots allocated beyond the stored blocks.
    pub spare_slots: usize,
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
    /// Stands on the edge `(parent, local_hash)`, given what `heads` held for
    /// it. An edge moves from a head to inside that run and never back, and
    /// `blocks` names its new place first, so looking there second finds an
    /// edge that existed throughout.
    fn find(
        &mut self,
        head: Option<Arc<Run>>,
        parent: Option<ExternalSequenceBlockHash>,
        local_hash: LocalBlockHash,
    ) -> bool {
        let key = TransitionKey::new(parent, local_hash);
        #[cfg(test)]
        let was_head = head.is_some();
        if head.is_some_and(|run| self.enter(run, Entry::Head(key), local_hash)) {
            return true;
        }
        let entered = parent.is_some_and(|hash| {
            self.store
                .locate(hash)
                .is_some_and(|(run, off)| self.enter(run, Entry::After(off, hash), local_hash))
        });
        #[cfg(test)]
        {
            if was_head && entered {
                reached!(self.store, head_moved);
            }
        }
        entered
    }

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
            // The edge heads a run, or follows its parent's block in that
            // block's run. Read `heads` before letting go of the current run:
            // while it is locked its blocks stay, so an edge that neither
            // follows the parent there nor heads a run does not exist.
            let head = self.store.head(&TransitionKey::new(parent, local_hash));
            // Release the run before locking another.
            self.run = None;
            if looked_inside && head.is_none() {
                return None;
            }
            if !self.find(head, parent, local_hash) {
                return None;
            }
            if looked_inside {
                reached!(self.store, cursor_hops);
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
                let (indexed, indexed_off) =
                    self.locate(block.1).expect("block missing from index");
                assert!(
                    Arc::ptr_eq(&indexed, run),
                    "block indexed under another run"
                );
                assert_eq!(state.index_of(indexed_off), pos);
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

        let mut distinct: Vec<*const Run> = heads.iter().map(|(_, run)| Arc::as_ptr(run)).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), heads.len(), "run under two head keys");

        for (parent, follower) in self.structure.lock().iter() {
            assert_ne!(*parent, ROOT_PARENT);
            let state = follower.read();
            assert!(!state.dead, "dead run among followers");
            assert_eq!(state.parent, *parent, "follower does not follow its key");
            let key = TransitionKey {
                parent_hash: *parent,
                local_hash: state.blocks[0].0,
            };
            let head = self.head(&key).expect("follower is not a head");
            assert!(Arc::ptr_eq(&head, follower));
        }
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

    /// A chain of `len` blocks; block `i` has local and sequence hash `base + i`.
    fn chain(base: u64, len: usize) -> Vec<KvCacheStoredBlockData> {
        let hashes: Vec<u64> = (0..len as u64).map(|i| base + i).collect();
        let locals: Vec<_> = hashes.iter().copied().map(LocalBlockHash).collect();
        stored_blocks_with_sequence_hashes(&locals, &hashes)
    }

    /// Stores `chain[range]` after the block before it in the chain.
    fn store_piece(
        store: &RunStore,
        who: Owner<'_>,
        chain: &[KvCacheStoredBlockData],
        range: std::ops::Range<usize>,
    ) -> u32 {
        let parent = range.start.checked_sub(1).map(|i| chain[i].block_hash);
        store.store(who, parent, &chain[range], None)
    }

    /// Run lengths, longest first.
    fn run_lengths(store: &RunStore) -> Vec<usize> {
        let mut lengths: Vec<_> = store
            .runs()
            .iter()
            .map(|run| run.read().blocks.len())
            .collect();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        lengths
    }

    fn reached(counter: &std::sync::atomic::AtomicUsize) -> usize {
        counter.load(Ordering::Relaxed)
    }

    /// A chain stored last block first, as a write-back cache sends it, builds
    /// the runs a parent-first store builds, whatever the size of each store.
    #[test]
    fn child_first_stores_build_whole_runs() {
        let table = OwnerTable::default();
        let owner = indexed_owners(1)[0];
        let who = Owner {
            id: table.intern(owner),
            owner,
            table: &table,
        };
        let blocks = chain(1_000, 40);
        let reference = RunStore::new();
        assert_eq!(reference.store(who, None, &blocks, None), 40);
        let edges = |store: &RunStore| -> FxHashMap<TransitionKey, ExternalSequenceBlockHash> {
            store
                .held_edges()
                .into_iter()
                .map(|(key, (hash, _))| (key, hash))
                .collect()
        };

        // Tests cap a run at 16 blocks. A run that is full when a store
        // arrives in front of it stays as it is, and the store starts another.
        for (piece, lengths, prepended) in [(1, [16, 16, 8], 37), (3, [15, 15, 10], 11)] {
            let store = RunStore::new();
            let mut end = blocks.len();
            while end > 0 {
                let start = end.saturating_sub(piece);
                assert_eq!(
                    store_piece(&store, who, &blocks, start..end),
                    (end - start) as u32
                );
                end = start;
            }
            store.assert_invariants();
            assert_eq!(run_lengths(&store), lengths);
            assert_eq!(reached(&store.reached.prepended), prepended);
            assert_eq!(reached(&store.reached.follower_full), 2);
            let census = store.census();
            assert_eq!(census.runs_by_start, [1, 0, 0, 2]);
            assert_eq!(edges(&store), edges(&reference));

            // A walk from the root crosses all three runs.
            let mut cursor = store.cursor();
            let mut parent = None;
            for block in &blocks {
                let edge = cursor
                    .edge(parent, block.tokens_hash)
                    .expect("a stored edge");
                assert_eq!(edge.child_hash, block.block_hash);
                parent = Some(block.block_hash);
            }
        }
    }

    /// What growing at the front leaves as separate runs, and what it joins.
    #[test]
    fn a_store_between_two_runs_extends_the_first() {
        let table = OwnerTable::default();
        let owner = indexed_owners(1)[0];
        let who = Owner {
            id: table.intern(owner),
            owner,
            table: &table,
        };

        // A store that lands between a run ending at its parent and a run
        // starting after it extends the first; the two stay apart.
        let store = RunStore::new();
        let blocks = chain(1_000, 3);
        store_piece(&store, who, &blocks, 0..1);
        store_piece(&store, who, &blocks, 2..3);
        store_piece(&store, who, &blocks, 1..2);
        store.assert_invariants();
        assert_eq!(run_lengths(&store), [2, 1]);
        assert_eq!(reached(&store.reached.bridged), 1);
        let census = store.census();
        assert_eq!(
            (census.runs_by_start, census.joinable_runs),
            ([1, 0, 0, 1], 1)
        );

        // Of two runs that start after the same block, the first one stored
        // takes the blocks in front, and the other becomes a branch off it.
        let store = RunStore::new();
        let blocks = chain(2_000, 3);
        let sibling = chain(2_100, 1);
        store_piece(&store, who, &blocks, 2..3);
        store.store(who, Some(blocks[1].block_hash), &sibling, None);
        store_piece(&store, who, &blocks, 1..2);
        store.assert_invariants();
        assert_eq!(run_lengths(&store), [2, 1]);
        assert_eq!(store.census().runs_by_start, [0, 1, 1, 0]);
        store_piece(&store, who, &blocks, 0..1);
        store.assert_invariants();
        assert_eq!(run_lengths(&store), [3, 1]);
        assert_eq!(store.census().runs_by_start, [1, 0, 1, 0]);

        // A run that started after a full run keeps its place as the run to
        // grow when that block is trimmed away and stored again.
        let store = RunStore::new();
        let blocks = chain(3_000, 17);
        store_piece(&store, who, &blocks, 0..17);
        assert_eq!(run_lengths(&store), [16, 1]);
        let hashes: Vec<_> = blocks.iter().map(|block| block.block_hash).collect();
        assert_eq!(store.remove(who, &hashes[7..16], None), 9);
        assert_eq!(reached(&store.reached.trimmed), 1);
        assert_eq!(run_lengths(&store), [7, 1]);
        store_piece(&store, who, &blocks, 15..16);
        store.assert_invariants();
        assert_eq!(run_lengths(&store), [7, 2]);
        assert_eq!(reached(&store.reached.prepended), 1);
    }

    /// A run's head edge moves inside it when blocks go in front. Whoever
    /// read the head before that finds the edge after its parent's block.
    #[test]
    fn an_edge_that_moved_inside_its_run_is_still_found() {
        let table = OwnerTable::default();
        let owners = indexed_owners(2);
        let who = |n: usize| Owner {
            id: table.intern(owners[n]),
            owner: owners[n],
            table: &table,
        };
        let blocks = chain(1_000, 3);
        let (middle, last) = (&blocks[1], &blocks[2]);
        let key = TransitionKey::new(Some(middle.block_hash), last.tokens_hash);

        let store = RunStore::new();
        store_piece(&store, who(0), &blocks, 2..3);
        let head = store.head(&key).expect("the last block heads a run");
        let (_, off) = store.locate(last.block_hash).unwrap();
        store_piece(&store, who(0), &blocks, 1..2);
        assert_eq!(reached(&store.reached.prepended), 1);

        // The head reference is stale; the block keeps its offset.
        assert!(store.head(&key).is_none());
        assert_eq!(head.read().entry_index(Entry::Head(key)), None);
        let after_last = head.read().entry_index(Entry::After(off, last.block_hash));
        assert_eq!(after_last, Some(2));
        let covered = store.cover(&head, Entry::Head(key), who(1), &blocks[2..], &mut None);
        assert!(covered.is_none());
        assert_eq!(reached(&store.reached.stale), 1);

        let mut cursor = store.cursor();
        assert!(cursor.find(Some(head), Some(middle.block_hash), last.tokens_hash));
        assert_eq!(reached(&store.reached.head_moved), 1);
        let state = cursor.run.as_ref().unwrap();
        assert_eq!(state.blocks[cursor.index].1, last.block_hash);
        drop(cursor);

        // A store that ends at an existing head goes in front of that run and
        // carries on inside it, without taking `structure` again.
        let store = RunStore::new();
        store_piece(&store, who(0), &blocks, 2..3);
        assert_eq!(store.store(who(1), None, &blocks, None), 3);
        store.assert_invariants();
        assert_eq!(run_lengths(&store), [3]);
        assert_eq!(reached(&store.reached.looked_again), 0);
    }

    /// One lane stores chains last block first while walks ask for the edge
    /// at the front of the growing run: the edge whose place is changing.
    #[test]
    fn a_walk_finds_the_edge_at_the_front_of_a_growing_run() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        const LEN: usize = 24;
        let table = OwnerTable::default();
        let owner = indexed_owners(1)[0];
        let who = Owner {
            id: table.intern(owner),
            owner,
            table: &table,
        };
        let chains: Vec<_> = (1..=2_000).map(|n| chain(n * 1_000, LEN)).collect();
        let store = RunStore::new();
        // The chain and block last stored, or `usize::MAX` before any.
        let front = AtomicUsize::new(usize::MAX);
        let done = AtomicBool::new(false);

        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| {
                    while !done.load(Ordering::Relaxed) {
                        let at = front.load(Ordering::Acquire);
                        if at == usize::MAX {
                            continue;
                        }
                        let (blocks, at) = (&chains[at / LEN], at % LEN);
                        let parent = at.checked_sub(1).map(|i| blocks[i].block_hash);
                        let mut cursor = store.cursor();
                        let edge = cursor
                            .edge(parent, blocks[at].tokens_hash)
                            .expect("a stored edge");
                        assert_eq!(edge.child_hash, blocks[at].block_hash);
                    }
                });
            }
            for (n, blocks) in chains.iter().enumerate() {
                for at in (0..LEN).rev() {
                    assert_eq!(store_piece(&store, who, blocks, at..at + 1), 1);
                    front.store(n * LEN + at, Ordering::Release);
                }
            }
            done.store(true, Ordering::Relaxed);
        });
        store.assert_invariants();
        assert_eq!(reached(&store.reached.prepended), chains.len() * (LEN - 2));
    }

    /// An owner clears, and a scan counts another owner's blocks, while a
    /// second lane stores blocks in front of both owners' runs, moving their
    /// heads. Neither may miss a run or see one twice.
    #[test]
    fn a_clear_and_a_scan_see_every_run_while_heads_move() {
        use std::sync::Barrier;
        use std::sync::atomic::AtomicBool;

        const CHAINS: usize = 256;
        const LEN: usize = 4;
        let table = OwnerTable::default();
        let owners = indexed_owners(3);
        let who = |n: usize| Owner {
            id: table.intern(owners[n]),
            owner: owners[n],
            table: &table,
        };
        let (cleared, pinned, other) = (who(0), who(1), who(2));
        let store = RunStore::new();
        let mut base = 0;
        let mut new_chains = || -> Vec<_> {
            (0..CHAINS)
                .map(|_| {
                    base += 1_000;
                    chain(base, LEN)
                })
                .collect()
        };

        for round in 1..=8 {
            // Each owner holds the last block of its chains.
            let (cleared_chains, pinned_chains) = (new_chains(), new_chains());
            for (owner, chains) in [(cleared, &cleared_chains), (pinned, &pinned_chains)] {
                for blocks in chains {
                    assert_eq!(store_piece(&store, owner, blocks, LEN - 1..LEN), 1);
                }
            }
            let barrier = Barrier::new(3);
            let done = AtomicBool::new(false);
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    barrier.wait();
                    for blocks in cleared_chains.iter().chain(&pinned_chains) {
                        for at in (0..LEN - 1).rev() {
                            store_piece(&store, other, blocks, at..at + 1);
                        }
                    }
                    done.store(true, Ordering::Relaxed);
                });
                scope.spawn(|| {
                    barrier.wait();
                    while !done.load(Ordering::Relaxed) {
                        let mut held = 0;
                        store.for_each_held(|owners, _, blocks| {
                            if owners.contains(pinned.id, pinned.owner) {
                                held += blocks.len();
                            }
                        });
                        assert_eq!(held, round * CHAINS);
                    }
                });
                barrier.wait();
                assert_eq!(store.remove_owner(cleared, None) as usize, CHAINS);
            });
        }
        store.assert_invariants();
        assert!(reached(&store.reached.prepended) > 0);
    }
}
