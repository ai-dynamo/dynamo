// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The block ledger: which blocks are held, and how they chain.

use std::collections::VecDeque;
use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

/// What the ledger keeps for one held block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block<T> {
    /// The block that precedes it, or `None` for the first block of a sequence.
    pub parent: Option<u64>,
    /// The host's value for the block (Dynamo keeps the hash of the block's tokens there).
    pub content: T,
}

// A slot is per block, millions per rank: keep its size a decision, not an accident.
const _: () = assert!(std::mem::size_of::<(u64, Block<u64>)>() == 32);

/// Blocks in sequence order: `parent` precedes the first, and each block precedes the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain<T> {
    pub parent: Option<u64>,
    /// `(hash, content)` of each block.
    pub blocks: Vec<(u64, T)>,
}

/// Not a slot: the end of a list of children.
const NONE: u32 = u32::MAX;

/// The blocks held in one place, keyed by the engine's block hash.
///
/// A host with several owners or tiers keeps one ledger for each. A block is held or it is
/// not: a second store changes nothing, and one remove drops it. An engine that holds
/// duplicate blocks under one hash (vLLM) needs its removes counted before they reach the
/// ledger.
///
/// `T` is the host's value for a block, fixed by the block's first store. It is `Copy`, so
/// copying the ledger copies memory and removing a block frees nothing. A host with
/// variable-size data per block keeps that data itself and puts its id here.
///
/// Blocks sit in the order they were stored, except those a remove has moved. The blocks of
/// a sequence are therefore usually next to each other, which is what makes
/// [`chains`](Self::chains) a mostly sequential read.
#[derive(Debug, Clone)]
pub struct Ledger<T> {
    /// Every held block with its hash. A remove fills its slot with the last block.
    slots: Vec<(u64, Block<T>)>,
    /// Block hash to slot.
    index: FxHashMap<u64, u32>,
}

impl<T> Default for Ledger<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            index: FxHashMap::default(),
        }
    }
}

impl<T: Copy> Ledger<T> {
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Records `hash` as held. The first store of a block fixes its `content` and `parent`.
    pub fn store(&mut self, hash: u64, content: T, parent: Option<u64>) {
        if let Entry::Vacant(slot) = self.index.entry(hash) {
            let at = u32::try_from(self.slots.len())
                .ok()
                .filter(|&at| at != NONE)
                .expect("a ledger holds fewer than 2^32 - 1 blocks");
            slot.insert(at);
            self.slots.push((hash, Block { parent, content }));
        }
    }

    /// Drops `hash`. Returns whether it was held.
    pub fn remove(&mut self, hash: u64) -> bool {
        let Some(at) = self.index.remove(&hash) else {
            return false;
        };
        self.slots.swap_remove(at as usize);
        // The last block now sits in the freed slot, unless the freed slot was the last.
        if let Some(&(moved, _)) = self.slots.get(at as usize) {
            self.index.insert(moved, at);
        }
        true
    }

    /// Every held block with its hash, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, Block<T>)> + '_ {
        self.slots.iter().copied()
    }

    /// The blocks that chain back to the start of a sequence, as chains ordered so that each
    /// comes after the chain that holds its parent.
    ///
    /// A block whose parent is not held is left out, with everything after it: a prefix index
    /// cannot hold a block it cannot reach. The same blocks always give the same chains in
    /// the same order, wherever they sit. Walks every block, so call it on a copy, off the
    /// thread that applies events.
    pub fn chains(&self) -> Vec<Chain<T>> {
        // Each block's children as a list of slots: `first_child` starts the list of the
        // block in that slot, and `next_sibling` continues it.
        let mut first_child = vec![NONE; self.slots.len()];
        let mut next_sibling = vec![NONE; self.slots.len()];
        let mut roots = Vec::new();
        for (at, (_, block)) in self.slots.iter().enumerate() {
            let Some(parent) = block.parent else {
                roots.push(at as u32);
                continue;
            };
            // Blocks stored together sit together, so the parent is usually in the next slot
            // and needs no lookup. A hash is held once, so a neighbour with the parent's hash
            // is the parent.
            let holds_parent =
                |slot: usize| matches!(self.slots.get(slot), Some(&(hash, _)) if hash == parent);
            let parent_at = if at > 0 && holds_parent(at - 1) {
                Some(at - 1)
            } else if holds_parent(at + 1) {
                Some(at + 1)
            } else {
                self.index.get(&parent).map(|&slot| slot as usize)
            };
            if let Some(parent_at) = parent_at {
                next_sibling[at] = first_child[parent_at];
                first_child[parent_at] = at as u32;
            }
        }
        // Order by hash, never by slot: the result must not depend on where blocks sit.
        let hash = |slot: u32| self.slots[slot as usize].0;
        roots.sort_unstable_by_key(|&slot| hash(slot));

        // A block is queued only through its parent's list, and only from a parent that was
        // itself reached, so each block is visited once and a parent cycle is never entered.
        let mut chains = Vec::new();
        let mut queue = VecDeque::from(roots);
        while let Some(start) = queue.pop_front() {
            let (first, block) = self.slots[start as usize];
            let mut blocks = vec![(first, block.content)];
            let mut tail = start;
            let branches = loop {
                let child = first_child[tail as usize];
                if child != NONE && next_sibling[child as usize] == NONE {
                    let (hash, block) = self.slots[child as usize];
                    blocks.push((hash, block.content));
                    tail = child;
                    continue;
                }
                let mut children = Vec::new();
                let mut next = child;
                while next != NONE {
                    children.push(next);
                    next = next_sibling[next as usize];
                }
                children.sort_unstable_by_key(|&slot| hash(slot));
                break children;
            };
            chains.push(Chain {
                parent: block.parent,
                blocks,
            });
            queue.extend(branches);
        }
        chains
    }
}
