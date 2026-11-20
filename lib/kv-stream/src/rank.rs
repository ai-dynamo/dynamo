// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One rank's stream: sequence checking, the fold into the ledger and the index, the replay
//! window, and admission of consumers by cursor or snapshot.

use std::fmt;
use std::sync::Arc;

use crate::snapshot::project;
use crate::{
    BlockHash, BlockKey, Cursor, Epoch, History, Ledger, NoIndex, Parent, Removed, SnapshotImage,
    Stored, StreamIndex, Tier, TierSet,
};

/// Limits and semantics for one rank.
#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    /// Tiers where each store is a reference that one remove takes back. See [`Ledger::new`]
    /// for what that needs from the engine. A scope that first appears mid-stream is added
    /// with [`BatchWriter::set_counted`].
    pub counted: TierSet,
    /// Replay window, in batches and in bytes (as measured by the host).
    pub history_max_batches: usize,
    pub history_max_bytes: usize,
    /// A cached snapshot is reused while the batches after its cut fit in this many bytes.
    /// Past that, loading the old image plus its tail costs more than a fresh image.
    pub snapshot_tail_max_bytes: usize,
    /// Hard cap on ledger blocks. At the cap, new blocks are not recorded, the stream reports
    /// itself incomplete, and an attached index is given a store only up to the first block
    /// the ledger refused.
    pub max_blocks: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            counted: TierSet::EMPTY,
            history_max_batches: 10_000,
            history_max_bytes: 256 << 20,
            snapshot_tail_max_bytes: 64 << 20,
            max_blocks: usize::MAX,
        }
    }
}

/// How far the ledger can be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Health {
    /// False when stores may have been missed: the ledger can under-report what the rank holds.
    pub complete: bool,
    /// False when removes or clears may have been missed: the ledger can report blocks that
    /// are gone.
    pub exact: bool,
}

impl Health {
    pub const FULL: Self = Self {
        complete: true,
        exact: true,
    };
    pub const UNKNOWN: Self = Self {
        complete: false,
        exact: false,
    };
}

/// What [`RankStream::start_epoch`] does with the state it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carry {
    /// Start empty. Right after an engine restart, and the safe answer to a gap.
    Drop,
    /// Keep the ledger and mark it neither complete nor exact. For a gap with no engine view
    /// to [`reseed`](RankStream::reseed) from, a stale view routes better than an empty one.
    Keep,
}

/// What [`RankStream::admit`] does for a consumer whose cursor cannot be continued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnMiss {
    Fail,
    Snapshot,
    LiveOnly,
}

/// How a consumer starts.
///
/// Replay does not carry health: a batch in the backlog may have degraded the stream, and the
/// batch object cannot say so. Whichever way a consumer starts, the host sends it the
/// stream's health once it has caught up, in the owner step that registers it for live
/// batches.
pub enum Admission<P> {
    /// Its cursor is still covered. In this owner step, send [`RankStream::backlog`] from
    /// `from_seq`, then `health` (the stream's health now, which is the consumer's after the
    /// backlog), and register the consumer for live batches.
    Events { from_seq: u64, health: Health },
    /// Send the image. A snapshot can take longer to send than the replay window lasts, so
    /// register the consumer only after the last chunk: call [`RankStream::resume`] with
    /// `image.cursor()`, then read the backlog, send [`RankStream::info`]'s health and register,
    /// all in that one owner step. A refusal there is an error for this consumer, not a reason
    /// to send another snapshot.
    Snapshot { image: Arc<SnapshotImage<P>> },
    /// Nothing before `next_seq`: register the consumer for live batches in this owner step.
    LiveOnly { next_seq: u64 },
}

impl<P> fmt::Debug for Admission<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Events { from_seq, health } => f
                .debug_struct("Events")
                .field("from_seq", from_seq)
                .field("health", health)
                .finish(),
            Self::Snapshot { image } => f.debug_struct("Snapshot").field("image", image).finish(),
            Self::LiveOnly { next_seq } => f
                .debug_struct("LiveOnly")
                .field("next_seq", next_seq)
                .finish(),
        }
    }
}

/// Why a cursor cannot be continued, or the stream cannot serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Refusal {
    #[error("no cursor given")]
    NoCursor,
    #[error("cursor is from another epoch")]
    WrongEpoch { current: Epoch },
    #[error("cursor is older than the replay window, which starts at {oldest}")]
    Evicted { oldest: u64 },
    #[error("cursor is ahead of the stream, which is at {next}")]
    Ahead { next: u64 },
    #[error("a batch failed halfway; the stream needs a new epoch")]
    Poisoned,
}

/// Why a batch was not applied. Nothing changed unless the variant says otherwise. Every
/// variant gives the batch back.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ApplyError<B, E> {
    #[error("batch {got} was already applied; expected {expected}")]
    Stale { expected: u64, got: u64, batch: B },
    #[error("batch {got} arrived but {expected} is missing")]
    Gap { expected: u64, got: u64, batch: B },
    #[error("an earlier batch failed halfway; the stream needs a new epoch")]
    Poisoned { batch: B },
    /// The host's fold failed. The batch may be half applied, so the stream is now poisoned.
    #[error("fold failed: {error}")]
    Fold { batch: B, error: E },
}

/// A new epoch must differ from the current one, or cursors from before it would continue
/// across the change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a new epoch must differ from the current one")]
pub struct SameEpoch;

/// Why [`RankStream::reseed`] did not finish. `SameEpoch` and `Evicted` change nothing; both
/// can be ruled out beforehand from [`RankStream::info`], which keeps the staged ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ReseedError<E> {
    #[error("a reseed needs a new epoch")]
    SameEpoch,
    #[error(
        "the engine view is as of batch {at_seq}, but the replay window starts at {oldest}: \
         the batches between them are gone"
    )]
    Evicted { oldest: u64, at_seq: u64 },
    /// The host's refold failed. The staged state is in place under the new epoch with part of
    /// the tail on top, so the stream is poisoned.
    #[error("refold failed: {0}")]
    Refold(E),
}

/// The result of one applied batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    pub next_seq: u64,
    pub health: Health,
    /// Health got worse in this batch and is still short of [`Health::FULL`] at its end.
    /// Consumers already following cannot see that from the batch itself, so the host sends
    /// them the new health in band. An improvement (a total clear) never sets it.
    pub degraded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamInfo {
    pub epoch: Epoch,
    pub oldest_seq: u64,
    pub next_seq: u64,
    pub blocks: usize,
    pub health: Health,
    pub poisoned: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamStats {
    pub batches: u64,
    pub epochs: u64,
    pub snapshots_built: u64,
    pub snapshots_reused: u64,
}

/// One rank's KV state. See the [crate docs](crate) for the contract.
///
/// - `B`: the host's batch object, kept for replay (its encoded stream message).
/// - `P`: the host's per-block payload, given back by snapshots. Make it cheap to clone
///   (`Arc`-backed): a snapshot clones the ledger.
/// - `I`: an optional prefix index fed from the same fold.
///
/// What costs O(blocks) on the owner: building a snapshot image, dropping a retired image when
/// the owner holds its last reference, a clear of some tiers, a total clear or
/// [`Carry::Drop`] (payload drops), a reseed, and the occasional batch that makes the ledger's
/// hash table grow or rehash. Everything else is per event.
pub struct RankStream<B, P = (), I = NoIndex> {
    config: StreamConfig,
    epoch: Epoch,
    ledger: Ledger<P>,
    history: History<B>,
    index: I,
    health: Health,
    poisoned: bool,
    cached: Option<Arc<SnapshotImage<P>>>,
    stats: StreamStats,
    keys: Vec<BlockKey>,
    hashes: Vec<BlockHash>,
}

impl<B, P, I: StreamIndex> RankStream<B, P, I> {
    /// A stream that expects `first_seq` next. Both engines number a publisher's batches from
    /// 0, so a stream that starts at 0 has seen everything and is complete; one that joins
    /// later is not, until a reseed or a total clear.
    pub fn new(config: StreamConfig, epoch: Epoch, first_seq: u64, index: I) -> Self {
        let health = Health {
            complete: first_seq == 0,
            exact: true,
        };
        let cursor = Cursor {
            epoch,
            next_seq: first_seq,
        };
        Self::seeded(config, cursor, Ledger::new(config.counted), health, index)
    }

    /// A stream that starts from state loaded elsewhere: a consumer that just loaded a
    /// snapshot. `index` must already hold what `ledger` holds.
    pub fn seeded(
        config: StreamConfig,
        cursor: Cursor,
        ledger: Ledger<P>,
        health: Health,
        index: I,
    ) -> Self {
        Self {
            config,
            epoch: cursor.epoch,
            ledger,
            history: History::new(
                cursor.next_seq,
                config.history_max_batches,
                config.history_max_bytes,
            ),
            index,
            health,
            poisoned: false,
            cached: None,
            stats: StreamStats::default(),
            keys: Vec::new(),
            hashes: Vec::new(),
        }
    }

    pub fn cursor(&self) -> Cursor {
        Cursor {
            epoch: self.epoch,
            next_seq: self.history.next(),
        }
    }

    pub fn info(&self) -> StreamInfo {
        StreamInfo {
            epoch: self.epoch,
            oldest_seq: self.history.oldest(),
            next_seq: self.history.next(),
            blocks: self.ledger.len(),
            health: self.health,
            poisoned: self.poisoned,
        }
    }

    pub fn stats(&self) -> StreamStats {
        self.stats
    }

    pub fn ledger(&self) -> &Ledger<P> {
        &self.ledger
    }

    pub fn index(&self) -> &I {
        &self.index
    }

    /// An empty ledger with this stream's tier semantics, to stage a reseed into.
    pub fn empty_ledger(&self) -> Ledger<P> {
        Ledger::new(self.ledger.counted())
    }

    /// Applies the engine's batch `seq`, then keeps `batch` for replay.
    ///
    /// `fold` reads the batch and replays its events into the [`BatchWriter`], in order. The
    /// batch is applied whole or the stream is poisoned: an `Err` or a panic from `fold` may
    /// leave it half applied, and the stream then refuses everything until
    /// [`start_epoch`](Self::start_epoch) or [`reseed`](Self::reseed).
    ///
    /// `bytes` is the host's measure of the batch, used for the replay window's byte cap.
    pub fn apply_batch<E>(
        &mut self,
        seq: u64,
        batch: B,
        bytes: usize,
        fold: impl FnOnce(&B, &mut BatchWriter<'_, P, I>) -> Result<(), E>,
    ) -> Result<Applied, ApplyError<B, E>> {
        if self.poisoned {
            return Err(ApplyError::Poisoned { batch });
        }
        let expected = self.history.next();
        if seq < expected {
            return Err(ApplyError::Stale {
                expected,
                got: seq,
                batch,
            });
        }
        if seq > expected {
            return Err(ApplyError::Gap {
                expected,
                got: seq,
                batch,
            });
        }

        let before = self.health;
        // Set first, so a panic that unwinds out of `fold` leaves the stream poisoned.
        self.poisoned = true;
        let mut writer = BatchWriter {
            ledger: &mut self.ledger,
            index: &mut self.index,
            health: &mut self.health,
            keys: &mut self.keys,
            hashes: &mut self.hashes,
            max_blocks: self.config.max_blocks,
            worsened: false,
            cleared_all: false,
        };
        let folded = fold(&batch, &mut writer);
        let (worsened, cleared_all) = (writer.worsened, writer.cleared_all);
        if let Err(error) = folded {
            return Err(ApplyError::Fold { batch, error });
        }
        self.poisoned = false;

        self.history.push(batch, bytes);
        self.stats.batches += 1;
        if self.health != before || cleared_all || !self.cached_is_fresh() {
            // The image no longer describes the stream, or it is all stale after a total
            // clear, or its tail costs more than a new image. Free it now rather than at the
            // next admission.
            self.cached = None;
        }
        Ok(Applied {
            next_seq: self.history.next(),
            health: self.health,
            degraded: worsened && self.health != Health::FULL,
        })
    }

    /// Starts a new epoch whose first batch is `next_seq`: the host's answer to an engine
    /// restart or a gap it could not fill. Cursors from before are refused from here on, the
    /// replay window is emptied, and a poisoned stream is usable again.
    pub fn start_epoch(
        &mut self,
        epoch: Epoch,
        next_seq: u64,
        carry: Carry,
    ) -> Result<(), SameEpoch> {
        if epoch == self.epoch {
            return Err(SameEpoch);
        }
        match carry {
            Carry::Drop => {
                self.ledger.clear(TierSet::ALL);
                self.index.cleared(TierSet::ALL);
                self.health = Health {
                    complete: next_seq == 0,
                    exact: true,
                };
            }
            Carry::Keep => self.health = Health::UNKNOWN,
        }
        self.epoch = epoch;
        self.history.reset(next_seq);
        self.cached = None;
        self.poisoned = false;
        self.stats.epochs += 1;
        Ok(())
    }

    /// Replaces the state with `staged`, which the host built from the engine's own view as of
    /// just before batch `at_seq` (a tree walk, an engine snapshot). This is how an incomplete
    /// or poisoned stream, or one stalled at a gap, becomes complete again without an engine
    /// restart.
    ///
    /// - `at_seq` inside the replay window: the retained batches from `at_seq` are folded
    ///   again on top through `refold`, and the stream continues where it was.
    /// - `at_seq` past the stream (the view is newer than anything applied, as it is after a
    ///   gap or a failed fold): the stream jumps to `at_seq`; batches below it arrive as
    ///   [`ApplyError::Stale`].
    /// - `at_seq` before the window: refused, the batches in between are gone.
    ///
    /// Always a new epoch: consumers cannot tell which of their blocks the reseed dropped.
    /// After [`ReseedError::Refold`] the stream is already on `epoch`, so recovering from that
    /// takes another one. With an index attached this replays the whole ledger into it,
    /// O(blocks log blocks).
    pub fn reseed<E>(
        &mut self,
        epoch: Epoch,
        at_seq: u64,
        staged: Ledger<P>,
        health: Health,
        mut refold: impl FnMut(&B, &mut BatchWriter<'_, P, I>) -> Result<(), E>,
    ) -> Result<(), ReseedError<E>> {
        if epoch == self.epoch {
            return Err(ReseedError::SameEpoch);
        }
        if at_seq < self.history.oldest() {
            return Err(ReseedError::Evicted {
                oldest: self.history.oldest(),
                at_seq,
            });
        }
        self.epoch = epoch;
        self.ledger = staged;
        self.health = health;
        self.cached = None;
        self.stats.epochs += 1;
        // As in `apply_batch`: poisoned until the tail is folded whole.
        self.poisoned = true;
        if at_seq > self.history.next() {
            self.history.reset(at_seq);
        } else {
            self.history.drop_before(at_seq);
        }
        if I::PROJECTS {
            self.index.cleared(TierSet::ALL);
            project(&self.ledger, &mut self.index);
        }
        for batch in self.history.range(at_seq) {
            let mut writer = BatchWriter {
                ledger: &mut self.ledger,
                index: &mut self.index,
                health: &mut self.health,
                keys: &mut self.keys,
                hashes: &mut self.hashes,
                max_blocks: self.config.max_blocks,
                worsened: false,
                cleared_all: false,
            };
            refold(batch, &mut writer).map_err(ReseedError::Refold)?;
        }
        self.poisoned = false;
        Ok(())
    }

    /// Where a consumer holding `cursor` continues, if it can. After its backlog the consumer
    /// is at this stream's state, so the host also sends it [`info`](Self::info)'s health: see
    /// [`Admission`].
    pub fn resume(&self, cursor: Option<Cursor>) -> Result<u64, Refusal> {
        if self.poisoned {
            return Err(Refusal::Poisoned);
        }
        let cursor = cursor.ok_or(Refusal::NoCursor)?;
        if cursor.epoch != self.epoch {
            return Err(Refusal::WrongEpoch {
                current: self.epoch,
            });
        }
        if cursor.next_seq < self.history.oldest() {
            return Err(Refusal::Evicted {
                oldest: self.history.oldest(),
            });
        }
        if cursor.next_seq > self.history.next() {
            return Err(Refusal::Ahead {
                next: self.history.next(),
            });
        }
        Ok(cursor.next_seq)
    }

    /// The retained batches from `from_seq` on. Read it in the same owner step as the
    /// [`resume`](Self::resume) that produced `from_seq` (or the [`admit`](Self::admit) that
    /// returned [`Admission::Events`]), and register the consumer for live batches in that
    /// step too, so no batch falls between the backlog and the live feed.
    pub fn backlog(&self, from_seq: u64) -> impl Iterator<Item = &B> {
        self.history.range(from_seq)
    }

    /// Gives up the cached snapshot image, if there is one. The stream otherwise keeps the
    /// last image until its tail outgrows [`StreamConfig::snapshot_tail_max_bytes`], the
    /// replay window passes its cut, health changes or a total clear arrives. Dropping the
    /// last reference walks every block, so drop the returned image off the owner.
    pub fn release_snapshot(&mut self) -> Option<Arc<SnapshotImage<P>>> {
        self.cached.take()
    }

    /// The host-measured size of [`backlog`](Self::backlog) from `from_seq`: what a host
    /// compares against the cost of a snapshot before it replays a long tail.
    pub fn backlog_bytes(&self, from_seq: u64) -> u64 {
        self.history.bytes_from(from_seq)
    }

    fn cached_is_fresh(&self) -> bool {
        self.cached.as_ref().is_some_and(|image| {
            let cut = image.cursor().next_seq;
            self.history.covers(cut)
                && self.history.bytes_from(cut) <= self.config.snapshot_tail_max_bytes as u64
        })
    }
}

impl<B, P: Clone, I: StreamIndex> RankStream<B, P, I> {
    /// The state as of now, or a recent image whose tail is still cheap to replay.
    ///
    /// A new image clones the ledger on the owner, between two batches: O(blocks). Consumers
    /// that arrive together share one image, so a burst of joiners costs one copy. Ordering
    /// the image ([`SnapshotImage::runs`]) is left to the thread that serves it.
    pub fn snapshot(&mut self) -> Result<Arc<SnapshotImage<P>>, Refusal> {
        if self.poisoned {
            return Err(Refusal::Poisoned);
        }
        if self.cached_is_fresh()
            && let Some(image) = &self.cached
        {
            self.stats.snapshots_reused += 1;
            return Ok(image.clone());
        }
        let image = Arc::new(SnapshotImage::new(
            self.cursor(),
            self.health,
            self.ledger.clone(),
        ));
        self.stats.snapshots_built += 1;
        self.cached = Some(image.clone());
        Ok(image)
    }

    /// Decides how a consumer starts: from its cursor when the replay window still covers it,
    /// otherwise as `on_miss` says. Each [`Admission`] variant says what the host does next.
    pub fn admit(
        &mut self,
        cursor: Option<Cursor>,
        on_miss: OnMiss,
    ) -> Result<Admission<P>, Refusal> {
        let miss = match self.resume(cursor) {
            Ok(from_seq) => {
                return Ok(Admission::Events {
                    from_seq,
                    health: self.health,
                });
            }
            Err(Refusal::Poisoned) => return Err(Refusal::Poisoned),
            Err(miss) => miss,
        };
        match on_miss {
            OnMiss::Fail => Err(miss),
            OnMiss::Snapshot => Ok(Admission::Snapshot {
                image: self.snapshot()?,
            }),
            OnMiss::LiveOnly => Ok(Admission::LiveOnly {
                next_seq: self.history.next(),
            }),
        }
    }
}

impl<B, P, I> fmt::Debug for RankStream<B, P, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankStream")
            .field("epoch", &self.epoch)
            .field("history", &self.history)
            .field("blocks", &self.ledger.len())
            .field("health", &self.health)
            .field("poisoned", &self.poisoned)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl<P, I> fmt::Debug for BatchWriter<'_, P, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchWriter")
            .field("blocks", &self.ledger.len())
            .field("health", &self.health)
            .finish_non_exhaustive()
    }
}

/// The write side of one batch: what a host's fold calls for each event, in event order.
pub struct BatchWriter<'a, P, I> {
    ledger: &'a mut Ledger<P>,
    index: &'a mut I,
    health: &'a mut Health,
    keys: &'a mut Vec<BlockKey>,
    hashes: &'a mut Vec<BlockHash>,
    max_blocks: usize,
    /// A health bit went from true to false in this batch.
    worsened: bool,
    cleared_all: bool,
}

impl<P, I: StreamIndex> BatchWriter<'_, P, I> {
    /// A chain of blocks stored on `tier`: `parent` precedes the first, each block precedes
    /// the next.
    ///
    /// `payload(i, key)` runs only for blocks the rank did not hold, and the first payload
    /// stays. A host whose engine can announce a block before it knows its payload (vLLM's
    /// token-less offload placeholders) notes which indexes `payload` ran for and upgrades the
    /// others with [`payload_mut`](Self::payload_mut).
    ///
    /// When the engine's event does not give a usable chain (a token-less placeholder, hashes
    /// that do not line up with tokens), store each block on its own with [`Parent::Unknown`].
    pub fn store(
        &mut self,
        tier: Tier,
        parent: Parent,
        blocks: impl IntoIterator<Item = BlockKey>,
        mut payload: impl FnMut(usize, BlockKey) -> P,
    ) {
        self.keys.clear();
        let mut link = parent;
        // False from the first block the ledger refused at its cap: an index cannot hold what
        // follows a block it was not given, and must not grow past the cap either.
        let mut recorded = true;
        for (i, key) in blocks.into_iter().enumerate() {
            match self
                .ledger
                .store(tier, key, link, self.max_blocks, || payload(i, key))
            {
                Stored::Dropped => {
                    recorded = false;
                    self.mark_incomplete();
                }
                // The count stops short of the engine's, so the block can go early.
                Stored::Saturated => self.mark_incomplete(),
                Stored::Fresh | Stored::Added | Stored::Duplicate => {}
            }
            link = Parent::Block(key.hash);
            if I::PROJECTS && recorded {
                self.keys.push(key);
            }
        }
        if !self.keys.is_empty() {
            self.index.stored(tier, parent, self.keys.as_slice());
        }
    }

    /// Blocks removed from `tier`, one reference each.
    pub fn remove(&mut self, tier: Tier, hashes: impl IntoIterator<Item = BlockHash>) {
        self.hashes.clear();
        for hash in hashes {
            match self.ledger.remove(tier, hash) {
                Removed::Last { .. } | Removed::Unknown if I::PROJECTS => self.hashes.push(hash),
                Removed::Last { .. } | Removed::Unknown | Removed::Still => {}
            }
        }
        if !self.hashes.is_empty() {
            self.index.removed(tier, self.hashes.as_slice());
        }
    }

    /// The engine emptied `tiers` and may still hold blocks elsewhere (vLLM clears one block
    /// pool). Walks every block of the ledger. Health is unchanged, whatever `tiers` names.
    pub fn clear(&mut self, tiers: TierSet) {
        self.ledger.clear(tiers);
        self.index.cleared(tiers);
    }

    /// The engine emptied everything the rank can hold (SGLang's clear does). The state is
    /// then known to be empty, which restores full health.
    pub fn clear_all(&mut self) {
        self.ledger.clear(TierSet::ALL);
        self.index.cleared(TierSet::ALL);
        *self.health = Health::FULL;
        self.cleared_all = true;
    }

    /// Makes `tier` counted from here on, for a scope the host first meets mid-stream (a new
    /// vLLM block pool). It only goes one way: a set tier holds each block once, which is a
    /// valid count.
    pub fn set_counted(&mut self, tier: Tier) {
        self.ledger.set_counted(tier);
    }

    /// The fold met a store it could not record (a scope with no tier slot left).
    pub fn mark_incomplete(&mut self) {
        self.worsened |= self.health.complete;
        self.health.complete = false;
    }

    /// The fold met a remove, a clear or an unknown event it could not record, or a store that
    /// no remove will pair with on a counted tier.
    pub fn mark_inexact(&mut self) {
        self.worsened |= self.health.exact;
        self.health.exact = false;
    }

    /// The payload of a held block, for a field the engine attaches after the first store.
    pub fn payload_mut(&mut self, hash: BlockHash) -> Option<&mut P> {
        self.ledger.payload_mut(hash)
    }

    pub fn ledger(&self) -> &Ledger<P> {
        self.ledger
    }
}
