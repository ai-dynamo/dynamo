// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine-side half of KV state, and the bridge from it to this crate's prefix indexes.
//!
//! Everything about a rank's stream (the ledger, the replay window, snapshots, cursors) is the
//! [`dynamo_kv_stream`] crate, which engines embed without any other Dynamo crate; it is
//! re-exported here whole. What this module adds is what needs Dynamo's types: two
//! [`StreamIndex`] adapters and the conversion of a snapshot to `RouterEvent`s. They reach the
//! existing indexes through the entry points those already have, so the tries and the event
//! lanes are used as they are:
//!
//! | Adapter | Index | Entry point |
//! |---------|-------|-------------|
//! | [`RadixIndex`] | [`RadixTree`], one per rank, on the stream's own thread | `apply_event` |
//! | [`LaneIndex`] | a shared [`ThreadPoolIndexer`], which is the way to the concurrent tree | `enqueue_event`, on the rank's sticky lane |
//!
//! Both index one tier (the engine's device tier); blocks on other tiers stay in the ledger.

use std::sync::Arc;

use rustc_hash::FxHashSet;

pub use dynamo_kv_stream::*;

use crate::prelude::*;
use crate::{RadixTree, SyncIndexer, ThreadPoolIndexer};

/// Turns stream calls into the `RouterEvent`s the indexes take.
#[derive(Debug)]
struct EventBuilder {
    worker: WorkerWithDpRank,
    primary: Tier,
    next_event_id: u64,
    skipped: u64,
    seen: FxHashSet<u64>,
}

impl EventBuilder {
    fn new(worker: WorkerWithDpRank, primary: Tier) -> Self {
        Self {
            worker,
            primary,
            next_event_id: 0,
            skipped: 0,
            seen: FxHashSet::default(),
        }
    }

    fn event(&mut self, data: KvCacheEventData) -> RouterEvent {
        let event_id = self.next_event_id;
        self.next_event_id += 1;
        RouterEvent::new(
            self.worker.worker_id,
            KvCacheEvent {
                event_id,
                data,
                dp_rank: self.worker.dp_rank,
            },
        )
    }

    /// `None` when nothing of the store can be indexed. A trie needs a known parent and a
    /// content hash for every block, so an unknown parent drops the store and a block without
    /// a content hash drops itself and everything after it in the chain. A store that names
    /// its own parent or one hash twice is dropped whole: the concurrent tree trusts its
    /// publishers not to send one, and would index the hash at two depths.
    fn stored(&mut self, tier: Tier, parent: Parent, blocks: &[BlockKey]) -> Option<RouterEvent> {
        if tier != self.primary {
            return None;
        }
        let indexable = match parent {
            Parent::Unknown => 0,
            Parent::Root | Parent::Block(_) => blocks
                .iter()
                .position(|key| key.content.is_none())
                .unwrap_or(blocks.len()),
        };
        let indexable = if self.repeats_a_hash(parent, &blocks[..indexable]) {
            0
        } else {
            indexable
        };
        self.skipped += (blocks.len() - indexable) as u64;
        if indexable == 0 {
            return None;
        }
        Some(self.event(stored_data(parent, &blocks[..indexable])))
    }

    fn repeats_a_hash(&mut self, parent: Parent, blocks: &[BlockKey]) -> bool {
        let parent = match parent {
            Parent::Block(hash) => Some(hash),
            Parent::Root | Parent::Unknown => None,
        };
        if let [only] = blocks {
            return parent == Some(only.hash);
        }
        self.seen.clear();
        self.seen.extend(parent.map(|hash| hash.0));
        blocks.iter().any(|key| !self.seen.insert(key.hash.0))
    }

    fn removed(&mut self, tier: Tier, hashes: &[BlockHash]) -> Option<RouterEvent> {
        (tier == self.primary).then(|| {
            self.event(KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: hashes
                    .iter()
                    .map(|hash| ExternalSequenceBlockHash(hash.0))
                    .collect(),
            }))
        })
    }

    fn cleared(&mut self, tiers: TierSet) -> Option<RouterEvent> {
        tiers
            .contains(self.primary)
            .then(|| self.event(KvCacheEventData::Cleared))
    }
}

fn stored_data(parent: Parent, blocks: &[BlockKey]) -> KvCacheEventData {
    KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: match parent {
            Parent::Block(hash) => Some(ExternalSequenceBlockHash(hash.0)),
            Parent::Root | Parent::Unknown => None,
        },
        start_position: None,
        blocks: blocks
            .iter()
            .map(|key| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(key.hash.0),
                tokens_hash: LocalBlockHash(key.content.0),
                mm_extra_info: None,
            })
            .collect(),
    })
}

/// A rank's own radix tree, updated on the thread that owns the stream.
///
/// The tree is not `Send`, so the stream that holds it stays on one thread. Use it where the
/// rank answers prefix matches itself and for parity tests; a host that only serves streams
/// and snapshots uses [`NoIndex`].
pub struct RadixIndex {
    tree: RadixTree,
    events: EventBuilder,
    rejected: u64,
}

impl RadixIndex {
    /// `primary` is the stream's tier slot for the engine's device tier. Events on any other
    /// slot are ignored here and not counted in [`skipped`](Self::skipped).
    pub fn new(worker: WorkerWithDpRank, primary: Tier) -> Self {
        Self {
            tree: RadixTree::new(),
            events: EventBuilder::new(worker, primary),
            rejected: 0,
        }
    }

    pub fn tree(&self) -> &RadixTree {
        &self.tree
    }

    /// Events the tree refused, such as a store whose parent it no longer covers.
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// Blocks that could not be indexed: no content hash, no known parent, or a store that
    /// repeats a hash.
    pub fn skipped(&self) -> u64 {
        self.events.skipped
    }

    fn apply(&mut self, event: Option<RouterEvent>) {
        if let Some(event) = event
            && self.tree.apply_event(event).is_err()
        {
            self.rejected += 1;
        }
    }
}

impl StreamIndex for RadixIndex {
    fn stored(&mut self, tier: Tier, parent: Parent, blocks: &[BlockKey]) {
        let event = self.events.stored(tier, parent, blocks);
        self.apply(event);
    }

    fn removed(&mut self, tier: Tier, hashes: &[BlockHash]) {
        let event = self.events.removed(tier, hashes);
        self.apply(event);
    }

    fn cleared(&mut self, tiers: TierSet) {
        let event = self.events.cleared(tiers);
        self.apply(event);
    }
}

/// A rank's events sent to a shared [`ThreadPoolIndexer`] on that rank's sticky lane.
///
/// This is how a stream reaches the concurrent tree: the pool's lanes are its only writers.
/// The index trails the ledger by whatever the lane has queued, and the lane only logs what
/// the tree refuses. After a clear, a host that must not publish before the index has caught
/// up waits on the pool's `flush_and_wait`.
pub struct LaneIndex<T: SyncIndexer> {
    pool: Arc<ThreadPoolIndexer<T>>,
    events: EventBuilder,
    offline: u64,
}

impl<T: SyncIndexer> LaneIndex<T> {
    /// `primary` is the stream's tier slot for the engine's device tier; events on any other
    /// slot are ignored.
    ///
    /// The pool must hold nothing for `worker` (for [`RankStream::seeded`], exactly what the
    /// seeded ledger holds). Dropping a `LaneIndex` leaves the rank's entries in the pool: a
    /// host that replaces a rank's stream, rather than calling `start_epoch` with
    /// [`Carry::Drop`] on it, first removes the rank from the pool.
    pub fn new(pool: Arc<ThreadPoolIndexer<T>>, worker: WorkerWithDpRank, primary: Tier) -> Self {
        Self {
            pool,
            events: EventBuilder::new(worker, primary),
            offline: 0,
        }
    }

    /// Events dropped because the pool had shut down.
    pub fn offline(&self) -> u64 {
        self.offline
    }

    /// Blocks that could not be indexed: no content hash, no known parent, or a store that
    /// repeats a hash.
    pub fn skipped(&self) -> u64 {
        self.events.skipped
    }

    fn enqueue(&mut self, event: Option<RouterEvent>) {
        if let Some(event) = event
            && self.pool.enqueue_event(event).is_err()
        {
            self.offline += 1;
        }
    }
}

impl<T: SyncIndexer> StreamIndex for LaneIndex<T> {
    fn stored(&mut self, tier: Tier, parent: Parent, blocks: &[BlockKey]) {
        let event = self.events.stored(tier, parent, blocks);
        self.enqueue(event);
    }

    fn removed(&mut self, tier: Tier, hashes: &[BlockHash]) {
        let event = self.events.removed(tier, hashes);
        self.enqueue(event);
    }

    fn cleared(&mut self, tiers: TierSet) {
        let event = self.events.cleared(tiers);
        self.enqueue(event);
    }
}

/// A snapshot as the replay-ordered `RouterEvent`s of a `TreeDump`: what a Dynamo router
/// already knows how to install. Only blocks on `primary` that a trie can hold are included:
/// a run is left out when its parent is, and cut at its first block without a content hash.
///
/// This is the ledger's view, which can hold more than a live index fed from the same stream
/// (see [`StreamIndex`]): a `dump_tree_as_events` of that index is not the reference for it.
///
/// Device tier only. A router installs a `TreeDump` by resetting every tier of the rank, so a
/// host whose rank holds blocks on other tiers adds those events itself
/// (`RouterEvent::with_storage_tier`) before it sends this as a whole-rank dump.
///
/// Returns the events and the number of blocks left out.
pub fn image_to_router_events<P>(
    image: &SnapshotImage<P>,
    worker: WorkerWithDpRank,
    primary: Tier,
) -> (Vec<RouterEvent>, usize) {
    let mut builder = EventBuilder::new(worker, primary);
    let mut emitted = FxHashSet::default();
    let mut events = Vec::new();
    let mut left_out = 0;
    for run in image.runs() {
        let keys = image.run_keys(run);
        let reachable = run.counts.get(primary) != 0
            && match run.parent {
                Parent::Root => true,
                Parent::Block(parent) => emitted.contains(&parent.0),
                Parent::Unknown => false,
            };
        let event = reachable
            .then(|| builder.stored(primary, run.parent, &keys))
            .flatten();
        let Some(event) = event else {
            left_out += keys.len();
            continue;
        };
        if let KvCacheEventData::Stored(store) = &event.event.data {
            left_out += keys.len() - store.blocks.len();
            emitted.extend(store.blocks.iter().map(|block| block.block_hash.0));
        }
        events.push(event);
    }
    (events, left_out)
}
