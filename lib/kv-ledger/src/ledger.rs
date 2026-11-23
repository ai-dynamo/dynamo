// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The block ledger: which blocks are held, and how they chain.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

/// What the ledger keeps for one held block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// The block that precedes it, or `None` for the first block of a sequence.
    pub parent: Option<u64>,
    /// A `u64` of the host's (Dynamo keeps the hash of the block's tokens there).
    pub content: u64,
}

// An entry is per block, millions per rank: keep its size a decision, not an accident.
const _: () = assert!(std::mem::size_of::<(u64, Block)>() == 32);

/// Blocks in sequence order: `parent` precedes the first, and each block precedes the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub parent: Option<u64>,
    /// `(hash, content)` of each block.
    pub blocks: Vec<(u64, u64)>,
}

/// The blocks held in one place, keyed by the engine's block hash.
///
/// A host with several owners or tiers keeps one ledger for each. A block is held or it is
/// not: a second store changes nothing, and one remove drops it. An engine that holds
/// duplicate blocks under one hash (vLLM) needs its removes counted before they reach the
/// ledger.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ledger {
    blocks: FxHashMap<u64, Block>,
}

impl Ledger {
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Records `hash` as held. The first store of a block fixes its `content` and `parent`.
    pub fn store(&mut self, hash: u64, content: u64, parent: Option<u64>) {
        self.blocks.entry(hash).or_insert(Block { parent, content });
    }

    /// Drops `hash`. Returns whether it was held.
    pub fn remove(&mut self, hash: u64) -> bool {
        self.blocks.remove(&hash).is_some()
    }

    /// Every held block with its hash, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, Block)> + '_ {
        self.blocks.iter().map(|(&hash, &block)| (hash, block))
    }

    /// The blocks that chain back to the start of a sequence, as chains ordered so that each
    /// comes after the chain that holds its parent.
    ///
    /// A block whose parent is not held is left out, with everything after it: a prefix index
    /// cannot hold a block it cannot reach. The same ledger always gives the same order.
    /// Walks every block, so call it on a copy, off the thread that applies events.
    pub fn chains(&self) -> Vec<Chain> {
        let mut children: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        let mut roots = Vec::new();
        for (&hash, block) in &self.blocks {
            match block.parent {
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
                parent: first.parent,
                blocks,
            });
            queue.extend(branches);
        }
        chains
    }
}
