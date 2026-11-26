// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One stream: a state folded from ordered batches, and the recent batches for replay.

use crate::history::History;

/// What a batch did to the state, as the host's fold reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// Stores and removes only.
    Delta,
    /// The batch emptied state (a clear). A copy cut before it is mostly stale, so the open
    /// cut is closed and the next consumer gets a new one.
    Reset,
}

/// Whether a batch continued the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applied {
    InOrder,
    /// The batch was applied, but `expected` was the next sequence. The replay window was
    /// restarted at this batch: consumers behind it need a copy of the state.
    OutOfOrder {
        expected: u64,
    },
}

/// An ordered stream of batches `B` folded into a state `S`.
///
/// The host calls [`apply`](Self::apply) for each batch, in order, from one place. A consumer
/// that fell behind continues from [`since`](Self::since) while the window still covers it.
/// Otherwise the host gives it a copy of the state, taken at a [`cut`](Self::cut), and then
/// the [`tail`](Self::tail) of batches applied after that cut.
pub struct Replay<B, S> {
    state: S,
    history: History<B>,
    last_seq: Option<u64>,
    /// Batches applied since the open cut. All of them are still in the history.
    cut_tail: Option<usize>,
    max_tail: usize,
}

impl<B, S> Replay<B, S> {
    /// A stream that keeps the newest `max_batches` batches. The first batch applied sets
    /// where the sequence starts.
    pub fn new(state: S, max_batches: usize) -> Self {
        Self {
            state,
            history: History::new(max_batches),
            last_seq: None,
            cut_tail: None,
            // Half the window, so the tail of an open cut is always retained.
            max_tail: max_batches / 2,
        }
    }

    #[cfg(test)]
    pub(crate) fn state(&self) -> &S {
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
        let expected = self.last_seq.map(|last| last + 1);
        let in_order = expected.is_none_or(|expected| expected == seq);
        let effect = fold(&batch, &mut self.state);

        if expected != Some(seq) {
            self.history.restart(seq);
        }
        self.history.push(batch);
        self.last_seq = Some(seq);

        // Close a cut that can no longer be served now, not at the next request, which may
        // never come: the host holds a copy of the whole state for it.
        self.cut_tail = self
            .cut_tail
            .map(|tail| tail + 1)
            .filter(|&tail| in_order && effect == Effect::Delta && tail <= self.max_tail);

        match expected {
            Some(expected) if !in_order => Applied::OutOfOrder { expected },
            _ => Applied::InOrder,
        }
    }

    /// The retained batches from `from_seq` on, which is an empty iterator when `from_seq` is
    /// past the newest batch. `None` when the window no longer reaches back that far.
    pub fn since(&self, from_seq: u64) -> Option<impl Iterator<Item = &B>> {
        self.history.since(from_seq)
    }

    /// Opens a cut at the state as of now. The host copies what it needs from the state
    /// before the next [`apply`](Self::apply), and can share that copy with consumers for as
    /// long as [`tail`](Self::tail) returns the batches to follow it.
    pub fn cut(&mut self) -> &S {
        self.cut_tail = Some(0);
        &self.state
    }

    /// The batches applied since the open cut: a copy taken at the cut, then these, is the
    /// state as of now.
    ///
    /// `None` when no cut is open. A cut closes once more than half the window has been
    /// applied on top of it, a batch resets state, or the sequence breaks.
    pub fn tail(&self) -> Option<impl Iterator<Item = &B>> {
        self.cut_tail.map(|tail| self.history.newest(tail))
    }
}
