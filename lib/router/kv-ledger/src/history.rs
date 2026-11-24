// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The replay window: the most recent batches, in the host's own type, addressed by sequence.

use std::collections::VecDeque;

/// A dense run of batches `[oldest, next)`, bounded by count.
pub(crate) struct History<B> {
    batches: VecDeque<B>,
    oldest: u64,
    max_batches: usize,
}

impl<B> History<B> {
    pub(crate) fn new(max_batches: usize) -> Self {
        Self {
            batches: VecDeque::new(),
            oldest: 0,
            max_batches,
        }
    }

    /// The sequence of the oldest retained batch, or `None` when nothing is retained.
    pub(crate) fn oldest(&self) -> Option<u64> {
        (!self.batches.is_empty()).then_some(self.oldest)
    }

    /// Appends the next batch, then drops the oldest while over the cap.
    pub(crate) fn push(&mut self, batch: B) {
        self.batches.push_back(batch);
        while self.batches.len() > self.max_batches {
            self.batches.pop_front();
            self.oldest += 1;
        }
    }

    /// The retained batches from `seq` on, oldest first, or `None` when `seq` is older than
    /// the window or nothing is retained.
    pub(crate) fn since(&self, seq: u64) -> Option<impl Iterator<Item = &B>> {
        let skip = usize::try_from(seq.checked_sub(self.oldest()?)?).ok()?;
        Some(self.batches.iter().skip(skip))
    }

    /// The newest `count` batches, oldest first.
    pub(crate) fn newest(&self, count: usize) -> impl Iterator<Item = &B> {
        self.batches
            .iter()
            .skip(self.batches.len().saturating_sub(count))
    }

    /// Empties the window; the next batch pushed is `next_seq`.
    pub(crate) fn restart(&mut self, next_seq: u64) {
        self.batches.clear();
        self.oldest = next_seq;
    }
}
