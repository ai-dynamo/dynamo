// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The block ledger: which blocks one owner holds, on which tiers, and how they chain.

use std::collections::VecDeque;
use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

use crate::{Tier, TierSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    parent: u64,
    content: u64,
    tiers: TierSet,
    has_parent: bool,
}

// An entry is per block, millions per rank: keep its size a decision, not an accident.
const _: () = assert!(std::mem::size_of::<(u64, Block)>() == 32);

impl Block {
    fn parent(&self) -> Option<u64> {
        self.has_parent.then_some(self.parent)
    }
}

/// A block as the ledger holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockView {
    pub hash: u64,
    pub content: u64,
    /// The block that precedes it, or `None` for the first block of a sequence.
    pub parent: Option<u64>,
    pub tiers: TierSet,
}

/// Blocks in sequence order: `parent` precedes the first, and each block precedes the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub parent: Option<u64>,
    /// `(hash, content)` of each block.
    pub blocks: Vec<(u64, u64)>,
}

/// One owner's blocks, keyed by the engine's block hash.
///
/// A block is on a tier or it is not: a second store on the same tier changes nothing, and one
/// remove takes it off. An engine that holds duplicate blocks under one hash (vLLM) needs its
/// removes counted before they reach the ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ledger {
    blocks: FxHashMap<u64, Block>,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Records `hash` as held on `tier`.
    ///
    /// `content` is a `u64` of the host's that [`chains`](Self::chains) and
    /// [`iter`](Self::iter) give back (Dynamo keeps the hash of the block's tokens there).
    /// The first store of a block fixes its `content` and `parent`.
    pub fn store(&mut self, tier: Tier, hash: u64, content: u64, parent: Option<u64>) {
        match self.blocks.entry(hash) {
            Entry::Vacant(slot) => {
                slot.insert(Block {
                    parent: parent.unwrap_or(0),
                    content,
                    tiers: TierSet::of(tier),
                    has_parent: parent.is_some(),
                });
            }
            Entry::Occupied(mut slot) => {
                let block = slot.get_mut();
                block.tiers = block.tiers.with(tier);
            }
        }
    }

    /// Takes `hash` off `tier`. Returns whether the tier held it.
    pub fn remove(&mut self, tier: Tier, hash: u64) -> bool {
        let Entry::Occupied(mut slot) = self.blocks.entry(hash) else {
            return false;
        };
        let block = slot.get_mut();
        if !block.tiers.contains(tier) {
            return false;
        }
        block.tiers = block.tiers.without(tier);
        if block.tiers.is_empty() {
            slot.remove();
        }
        true
    }

    /// Every held block, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = BlockView> + '_ {
        self.blocks.iter().map(|(&hash, block)| BlockView {
            hash,
            content: block.content,
            parent: block.parent(),
            tiers: block.tiers,
        })
    }

    /// The blocks on `tier` that chain back to the start of a sequence through blocks on
    /// `tier`, as chains ordered so that each comes after the chain that holds its parent.
    ///
    /// A block whose parent is not on `tier` is left out, with everything after it: a prefix
    /// index cannot hold a block it cannot reach. The same ledger always gives the same order.
    /// Walks every block, so call it on a copy, off the thread that applies events.
    pub fn chains(&self, tier: Tier) -> Vec<Chain> {
        let mut children: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        let mut roots = Vec::new();
        for (&hash, block) in &self.blocks {
            if !block.tiers.contains(tier) {
                continue;
            }
            match block.parent() {
                None => roots.push(hash),
                Some(parent) => children.entry(parent).or_default().push(hash),
            }
        }
        roots.sort_unstable();

        // A block is queued only through its parent's list, which is taken when the parent is
        // reached, so each block is visited once and a parent cycle is never entered.
        let mut chains = Vec::new();
        let mut queue = VecDeque::from(roots);
        while let Some(start) = queue.pop_front() {
            let first = &self.blocks[&start];
            let mut blocks = vec![(start, first.content)];
            let mut tail = start;
            let branches = loop {
                let mut kids = children.remove(&tail).unwrap_or_default();
                if let [only] = kids[..] {
                    blocks.push((only, self.blocks[&only].content));
                    tail = only;
                    continue;
                }
                kids.sort_unstable();
                break kids;
            };
            chains.push(Chain {
                parent: first.parent(),
                blocks,
            });
            queue.extend(branches);
        }
        chains
    }
}
