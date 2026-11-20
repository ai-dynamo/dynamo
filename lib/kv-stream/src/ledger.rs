// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The residency ledger: which blocks a rank holds, on which tiers, how many times, and how
//! they chain. It follows both engines' semantics (a set per tier, or a count per tier) and is
//! independent of any prefix index.

use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

use crate::{BlockHash, BlockKey, ContentHash, Parent, Tier, TierCounts, TierSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ParentKind {
    Root,
    Block,
    Unknown,
}

/// One ledger entry. `P` is the host's per-block payload (tokens, salt, LoRA name, multimodal
/// spans): whatever its snapshot must give back. It sits here and in no index, so the tries'
/// per-block layout does not change. `P = ()` costs nothing.
#[derive(Debug, Clone, PartialEq)]
struct Block<P> {
    parent: u64,
    content: u64,
    counts: TierCounts,
    parent_kind: ParentKind,
    payload: P,
}

// An entry is per block, millions per rank: keep its size a decision, not an accident. 33 bytes
// of fields round up to 40; with the key a table slot is 48 bytes, and 64 with a 16-byte payload
// such as an `Arc<[T]>`. The table is at most 7/8 full, so a block costs more than its slot.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<Block<()>>() == 40);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<(u64, Block<()>)>() == 48);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<(u64, Block<std::sync::Arc<[i64]>>)>() == 64);

impl<P> Block<P> {
    fn parent(&self) -> Parent {
        match self.parent_kind {
            ParentKind::Root => Parent::Root,
            ParentKind::Block => Parent::Block(BlockHash(self.parent)),
            ParentKind::Unknown => Parent::Unknown,
        }
    }

    fn set_parent(&mut self, parent: Parent) {
        (self.parent, self.parent_kind) = match parent {
            Parent::Root => (0, ParentKind::Root),
            Parent::Block(hash) => (hash.0, ParentKind::Block),
            Parent::Unknown => (0, ParentKind::Unknown),
        };
    }
}

/// What a store did to the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stored {
    /// The rank did not hold this block on any tier.
    Fresh,
    /// A known block gained a tier or, on a counted tier, another reference.
    Added,
    /// A set-like tier already held it: nothing changed.
    Duplicate,
    /// A counted tier hit the count limit. The count stays there, below the engine's, so later
    /// removes can drop the block while the engine still holds it: the ledger under-reports.
    Saturated,
    /// The ledger is at its block cap and did not record the block.
    Dropped,
}

/// What a remove did to the ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Removed {
    /// The tier's last reference went. `gone` when no tier holds the block any more.
    Last { gone: bool },
    /// A counted tier still holds other references.
    Still,
    /// The tier did not hold the block.
    Unknown,
}

/// Counters for events that did not change the ledger the way the engine meant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LedgerStats {
    pub duplicate_stores: u64,
    pub unknown_removes: u64,
    pub saturated_counts: u64,
    pub self_parents: u64,
    pub dropped_over_cap: u64,
}

/// A block as the ledger holds it.
#[derive(Debug)]
pub struct BlockView<'a, P> {
    pub key: BlockKey,
    pub parent: Parent,
    pub counts: TierCounts,
    pub payload: &'a P,
}

/// One rank's residency.
#[derive(Debug, Clone)]
pub struct Ledger<P = ()> {
    blocks: FxHashMap<u64, Block<P>>,
    counted: TierSet,
    stats: LedgerStats,
}

/// Two ledgers are equal when they hold the same blocks with the same parents, counts and
/// payloads; counters are not state.
impl<P: PartialEq> PartialEq for Ledger<P> {
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks
    }
}

impl<P> Ledger<P> {
    /// `counted` names the tiers where every store is a reference that one remove takes back
    /// (vLLM's block pool keeps duplicate blocks under one hash and removes them one by one).
    /// Every other tier is a set (SGLang: a block is on a tier or it is not).
    ///
    /// A counted tier is right only while the engine pairs every store with a remove. A store
    /// that reports a block again without holding it again (vLLM with
    /// `kv_cache_report_mode=full`) leaves a reference nothing takes back: the host keeps such
    /// stores out of the fold, or marks the batch inexact.
    pub fn new(counted: TierSet) -> Self {
        Self {
            blocks: FxHashMap::default(),
            counted,
            stats: LedgerStats::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn stats(&self) -> LedgerStats {
        self.stats
    }

    /// The tiers that count references.
    pub fn counted(&self) -> TierSet {
        self.counted
    }

    /// Makes `tier` counted from here on. A set tier holds each block once, which is a valid
    /// count, so nothing is rewritten.
    pub fn set_counted(&mut self, tier: Tier) {
        self.counted = self.counted.with(tier);
    }

    pub fn get(&self, hash: BlockHash) -> Option<BlockView<'_, P>> {
        self.blocks.get(&hash.0).map(|block| view(hash, block))
    }

    pub fn contains(&self, hash: BlockHash) -> bool {
        self.blocks.contains_key(&hash.0)
    }

    /// The payload of a held block, for a field the engine attaches after the first store.
    pub fn payload_mut(&mut self, hash: BlockHash) -> Option<&mut P> {
        self.blocks.get_mut(&hash.0).map(|block| &mut block.payload)
    }

    /// Every held block, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = BlockView<'_, P>> {
        self.blocks
            .iter()
            .map(|(hash, block)| view(BlockHash(*hash), block))
    }

    /// Records one store of `key` on `tier`.
    ///
    /// `payload` runs only when the rank did not hold the block. A store of a held block keeps
    /// the first payload, and fills in the parent or content hash if the first store did not
    /// know them.
    pub fn store(
        &mut self,
        tier: Tier,
        key: BlockKey,
        parent: Parent,
        max_blocks: usize,
        payload: impl FnOnce() -> P,
    ) -> Stored {
        let parent = match parent {
            Parent::Block(hash) if hash == key.hash => {
                self.stats.self_parents += 1;
                Parent::Unknown
            }
            other => other,
        };
        let counted = self.counted.contains(tier);
        let held = self.blocks.len();
        match self.blocks.entry(key.hash.0) {
            Entry::Vacant(slot) => {
                if held >= max_blocks {
                    self.stats.dropped_over_cap += 1;
                    return Stored::Dropped;
                }
                let mut block = Block {
                    parent: 0,
                    content: key.content.0,
                    counts: TierCounts::default(),
                    parent_kind: ParentKind::Unknown,
                    payload: payload(),
                };
                block.set_parent(parent);
                block.counts.0[tier.slot()] = 1;
                slot.insert(block);
                Stored::Fresh
            }
            Entry::Occupied(mut slot) => {
                let block = slot.get_mut();
                if block.parent_kind == ParentKind::Unknown && parent != Parent::Unknown {
                    block.set_parent(parent);
                }
                if block.content == 0 {
                    block.content = key.content.0;
                }
                let count = &mut block.counts.0[tier.slot()];
                if *count == 0 {
                    *count = 1;
                    Stored::Added
                } else if !counted {
                    self.stats.duplicate_stores += 1;
                    Stored::Duplicate
                } else if *count == u16::MAX {
                    self.stats.saturated_counts += 1;
                    Stored::Saturated
                } else {
                    *count += 1;
                    Stored::Added
                }
            }
        }
    }

    /// Records one remove of `hash` from `tier`.
    pub fn remove(&mut self, tier: Tier, hash: BlockHash) -> Removed {
        let Entry::Occupied(mut slot) = self.blocks.entry(hash.0) else {
            self.stats.unknown_removes += 1;
            return Removed::Unknown;
        };
        let count = &mut slot.get_mut().counts.0[tier.slot()];
        if *count == 0 {
            self.stats.unknown_removes += 1;
            return Removed::Unknown;
        }
        *count -= 1;
        if *count != 0 {
            return Removed::Still;
        }
        let gone = slot.get().counts.is_empty();
        if gone {
            slot.remove();
        }
        Removed::Last { gone }
    }

    /// Empties `tiers`. Walks every block unless `tiers` is [`TierSet::ALL`].
    pub fn clear(&mut self, tiers: TierSet) {
        if tiers.is_all() {
            self.blocks.clear();
            return;
        }
        self.blocks.retain(|_, block| {
            for tier in tiers.iter() {
                block.counts.0[tier.slot()] = 0;
            }
            !block.counts.is_empty()
        });
    }

    /// Puts one block in with explicit counts, replacing any entry for the same hash. This is
    /// how a consumer loads a snapshot, or a host stages a reseed from the engine's own tree.
    /// A count above 1 on a set tier is taken as 1.
    pub fn load_block(
        &mut self,
        key: BlockKey,
        parent: Parent,
        mut counts: TierCounts,
        payload: P,
    ) {
        for (slot, count) in counts.0.iter_mut().enumerate() {
            if *count > 1 && self.counted.0 & (1 << slot) == 0 {
                *count = 1;
            }
        }
        if counts.is_empty() {
            self.blocks.remove(&key.hash.0);
            return;
        }
        let mut block = Block {
            parent: 0,
            content: key.content.0,
            counts,
            parent_kind: ParentKind::Unknown,
            payload,
        };
        block.set_parent(parent);
        self.blocks.insert(key.hash.0, block);
    }
}

fn view<P>(hash: BlockHash, block: &Block<P>) -> BlockView<'_, P> {
    BlockView {
        key: BlockKey::new(hash, ContentHash(block.content)),
        parent: block.parent(),
        counts: block.counts,
        payload: &block.payload,
    }
}
