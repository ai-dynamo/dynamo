// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One stream: a state folded from ordered batches, the recent batches, and a shared snapshot.

use std::sync::Arc;

use crate::history::History;

/// What a batch did to the state, as the host's fold reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Stores and removes only.
    Delta,
    /// The batch emptied state (a clear). A snapshot cut before it is mostly stale, so the
    /// shared one is dropped and the next request cuts a new one.
    Reset,
}

/// Whether a batch continued the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    InOrder,
    /// The batch was applied, but `expected` was the next sequence. The replay window was
    /// restarted at this batch: consumers behind it need a snapshot.
    OutOfOrder {
        expected: u64,
    },
}

/// The state after every batch up to and including `last_seq`, and nothing later.
#[derive(Debug)]
pub struct Snapshot<S> {
    /// `None` when no batch had been applied.
    pub last_seq: Option<u64>,
    pub state: S,
}

struct Shared<S> {
    snapshot: Arc<Snapshot<S>>,
    /// Batches applied since the cut. All of them are still in the history.
    tail: usize,
}

/// An ordered stream of batches `B` folded into a state `S`.
///
/// The host calls [`apply`](Self::apply) for each batch, in order, from one place. Consumers
/// that fall behind continue from [`since`](Self::since) while the window still covers them,
/// and otherwise from [`snapshot`](Self::snapshot).
pub struct Replay<B, S> {
    state: S,
    history: History<B>,
    last_seq: Option<u64>,
    shared: Option<Shared<S>>,
    max_tail: usize,
    snapshots_built: u64,
}

impl<B, S> Replay<B, S> {
    /// A stream that keeps the newest `max_batches` batches. The first batch applied sets
    /// where the sequence starts.
    pub fn new(state: S, max_batches: usize) -> Self {
        Self {
            state,
            history: History::new(max_batches),
            last_seq: None,
            shared: None,
            // Half the window, so the tail of a snapshot that is still shared is always retained.
            max_tail: max_batches / 2,
            snapshots_built: 0,
        }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    /// The sequence of the newest batch applied.
    pub fn last_seq(&self) -> Option<u64> {
        self.last_seq
    }

    /// The sequence of the oldest batch [`since`](Self::since) can still return.
    pub fn oldest_seq(&self) -> Option<u64> {
        self.history.oldest()
    }

    /// Folds `batch` into the state, then keeps it for replay.
    ///
    /// A `seq` that does not follow the last one is still applied: the stream is advisory, and
    /// the host decides what a gap means. The window restarts at that batch, because batches
    /// on either side of a gap cannot be replayed as one run.
    pub fn apply(
        &mut self,
        seq: u64,
        batch: B,
        fold: impl FnOnce(&B, &mut S) -> Effect,
    ) -> Applied {
        let expected = self.last_seq.map(|last| last.wrapping_add(1));
        let in_order = expected.is_none_or(|expected| expected == seq);
        let effect = fold(&batch, &mut self.state);

        if expected != Some(seq) {
            self.history.restart(seq);
        }
        self.history.push(batch);
        self.last_seq = Some(seq);

        // Drop a snapshot that can no longer be served now, not at the next request, which
        // may never come: it holds a copy of the whole state.
        let max_tail = self.max_tail;
        let still_shared = in_order
            && effect == Effect::Delta
            && self.shared.as_mut().is_some_and(|shared| {
                shared.tail += 1;
                shared.tail <= max_tail
            });
        if !still_shared {
            self.shared = None;
        }

        match expected {
            Some(expected) if !in_order => Applied::OutOfOrder { expected },
            _ => Applied::InOrder,
        }
    }

    /// The retained batches from `from_seq` on, or `None` when the window no longer reaches
    /// back that far.
    pub fn since(&self, from_seq: u64) -> Option<impl Iterator<Item = &B>> {
        self.history.since(from_seq)
    }

    /// Whether a snapshot is being kept for the next [`snapshot`](Self::snapshot) call.
    pub fn has_shared_snapshot(&self) -> bool {
        self.shared.is_some()
    }

    /// How many times [`snapshot`](Self::snapshot) copied the state.
    pub fn snapshots_built(&self) -> u64 {
        self.snapshots_built
    }
}

impl<B, S: Clone> Replay<B, S> {
    /// A snapshot and the batches applied after it. Loading the snapshot and then applying
    /// those batches gives the state as of now.
    ///
    /// The first call copies the state, which is O(state). Later calls share that copy until
    /// more than half the window has been applied on top of it, a batch resets state, or the
    /// sequence breaks.
    pub fn snapshot(&mut self) -> (Arc<Snapshot<S>>, impl Iterator<Item = &B>) {
        let shared = self.shared.get_or_insert_with(|| {
            self.snapshots_built += 1;
            Shared {
                snapshot: Arc::new(Snapshot {
                    last_seq: self.last_seq,
                    state: self.state.clone(),
                }),
                tail: 0,
            }
        });
        (shared.snapshot.clone(), self.history.newest(shared.tail))
    }
}
