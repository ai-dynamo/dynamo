// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The replay window: the most recent batches, in the host's own type, addressed by the
//! engine's batch sequence and bounded by count and by bytes.

use std::collections::VecDeque;
use std::fmt;

struct Retained<B> {
    batch: B,
    /// Bytes pushed up to and including this batch, since the history was created.
    end_total: u64,
}

/// A dense run of batches `[oldest, next)`.
///
/// `B` is whatever the host fans out (an encoded protobuf message, an `Arc` of one), so replay
/// hands back the same object the live path sent and nothing is re-encoded.
pub struct History<B> {
    batches: VecDeque<Retained<B>>,
    oldest: u64,
    pushed_total: u64,
    evicted_total: u64,
    max_batches: usize,
    max_bytes: u64,
}

impl<B> fmt::Debug for History<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("History")
            .field("oldest", &self.oldest)
            .field("next", &self.next())
            .field("bytes", &(self.pushed_total - self.evicted_total))
            .finish_non_exhaustive()
    }
}

impl<B> History<B> {
    /// An empty window whose first batch will be `next_seq`.
    pub fn new(next_seq: u64, max_batches: usize, max_bytes: usize) -> Self {
        Self {
            batches: VecDeque::new(),
            oldest: next_seq,
            pushed_total: 0,
            evicted_total: 0,
            max_batches: max_batches.max(1),
            max_bytes: max_bytes as u64,
        }
    }

    /// The sequence of the oldest retained batch (equal to [`next`](Self::next) when empty).
    pub fn oldest(&self) -> u64 {
        self.oldest
    }

    /// The sequence the next pushed batch gets.
    pub fn next(&self) -> u64 {
        self.oldest + self.batches.len() as u64
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// Whether a consumer at `seq` can continue from here with no batch missing.
    pub fn covers(&self, seq: u64) -> bool {
        self.oldest <= seq && seq <= self.next()
    }

    /// Bytes retained from `seq` to the end. O(1).
    pub fn bytes_from(&self, seq: u64) -> u64 {
        if seq >= self.next() {
            return 0;
        }
        if seq <= self.oldest {
            return self.pushed_total - self.evicted_total;
        }
        let before = &self.batches[(seq - self.oldest - 1) as usize];
        self.pushed_total - before.end_total
    }

    /// Appends the next batch, then evicts from the front while over either cap. The newest
    /// batch is always kept, even when it alone exceeds the byte cap: live readers that follow
    /// by position must be able to read the batch that was just published.
    pub fn push(&mut self, batch: B, bytes: usize) {
        self.pushed_total += bytes as u64;
        self.batches.push_back(Retained {
            batch,
            end_total: self.pushed_total,
        });
        while self.batches.len() > 1
            && (self.batches.len() > self.max_batches
                || self.pushed_total - self.evicted_total > self.max_bytes)
        {
            self.evict_front();
        }
    }

    /// The retained batches from `seq` on, oldest first. Check [`covers`](Self::covers) first:
    /// a `seq` below the window starts at the oldest retained batch.
    pub fn range(&self, seq: u64) -> impl Iterator<Item = &B> {
        let skip = seq.saturating_sub(self.oldest) as usize;
        self.batches.iter().skip(skip).map(|kept| &kept.batch)
    }

    /// Drops every batch below `seq`.
    pub fn drop_before(&mut self, seq: u64) {
        while self.oldest < seq && !self.batches.is_empty() {
            self.evict_front();
        }
    }

    /// Empties the window and restarts it at `next_seq`.
    pub fn reset(&mut self, next_seq: u64) {
        self.batches.clear();
        self.evicted_total = self.pushed_total;
        self.oldest = next_seq;
    }

    fn evict_front(&mut self) {
        if let Some(front) = self.batches.pop_front() {
            self.evicted_total = front.end_total;
            self.oldest += 1;
        }
    }
}
