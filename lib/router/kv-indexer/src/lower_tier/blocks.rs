// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The blocks of one run: a vector that can also grow at the front.

use std::ops::Deref;

use crate::prelude::{ExternalSequenceBlockHash, LocalBlockHash};

/// One block's local and sequence hash.
pub(super) type Block = (LocalBlockHash, ExternalSequenceBlockHash);

const UNUSED: Block = (LocalBlockHash(0), ExternalSequenceBlockHash(0));

/// At most `limit` blocks. It reads as their slice, so a walk indexes it like a vector.
///
/// A run that grows at the front keeps unused slots there, a quarter of its length when they run
/// out, so inserting at the front moves the blocks only now and then.
#[derive(Debug, Default)]
pub(super) struct Blocks {
    /// `slots[unused..]` are the blocks.
    slots: Vec<Block>,
    unused: usize,
}

impl Deref for Blocks {
    type Target = [Block];

    #[inline]
    fn deref(&self) -> &[Block] {
        &self.slots[self.unused..]
    }
}

impl Blocks {
    /// Appends `blocks`. Slots, used or not, stay within `limit`.
    pub(super) fn extend(&mut self, blocks: impl IntoIterator<Item = Block>, limit: usize) {
        self.slots.extend(blocks);
        if self.slots.len() > limit {
            self.slots.drain(..self.unused);
            self.unused = 0;
        }
    }

    /// Inserts `blocks` before the first block. The result must fit `limit`.
    pub(super) fn push_front(
        &mut self,
        blocks: impl ExactSizeIterator<Item = Block>,
        limit: usize,
    ) {
        let count = blocks.len();
        debug_assert!(self.len() + count <= limit);
        if self.unused < count {
            let unused = count.max(self.len() / 4).min(limit - self.len());
            let more = unused - self.unused;
            self.slots.reserve_exact(more);
            self.slots.splice(..0, std::iter::repeat_n(UNUSED, more));
            self.unused = unused;
        }
        self.unused -= count;
        for (slot, block) in self.slots[self.unused..].iter_mut().zip(blocks) {
            *slot = block;
        }
    }

    /// Drops the blocks at and past `len` and returns their sequence hashes.
    pub(super) fn truncate(&mut self, len: usize) -> Vec<ExternalSequenceBlockHash> {
        let dropped = self
            .slots
            .drain(self.unused + len..)
            .map(|block| block.1)
            .collect();
        if self.slots.capacity() > 4 * len.max(4) {
            self.slots.drain(..self.unused);
            self.unused = 0;
            self.slots.shrink_to_fit();
        }
        dropped
    }

    /// Slots allocated beyond the blocks, at either end.
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn spare_slots(&self) -> usize {
        self.slots.capacity() - self.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    /// Random appends, front inserts and truncations against a deque.
    #[test]
    fn matches_a_deque() {
        const LIMIT: usize = 16;
        for seed in 0..64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut blocks = Blocks::default();
            let mut model: VecDeque<Block> = VecDeque::new();
            let mut next = 0u64;
            let mut grew_front = 0;
            for _ in 0..400 {
                let room = LIMIT - model.len();
                let count = rng.usize(1..=4).min(room);
                let new: Vec<Block> = (0..count)
                    .map(|_| {
                        next += 1;
                        (LocalBlockHash(next), ExternalSequenceBlockHash(next))
                    })
                    .collect();
                match rng.u32(..5) {
                    0 | 1 if count > 0 => {
                        blocks.extend(new.iter().copied(), LIMIT);
                        model.extend(new);
                    }
                    2 | 3 if count > 0 => {
                        blocks.push_front(new.iter().copied(), LIMIT);
                        for block in new.into_iter().rev() {
                            model.push_front(block);
                        }
                        grew_front += 1;
                    }
                    _ => {
                        let len = rng.usize(..=model.len());
                        let dropped = blocks.truncate(len);
                        let expected: Vec<_> = model.drain(len..).map(|block| block.1).collect();
                        assert_eq!(dropped, expected, "seed {seed}");
                    }
                }
                assert!(blocks.iter().eq(model.iter()), "seed {seed}");
                assert!(blocks.slots.len() <= LIMIT, "seed {seed}");
            }
            assert!(grew_front > 0);
        }
    }
}
