// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-owned multimodal identity for SGLang sources with wide pad values.
//!
//! Since sgl-project/sglang#43043, SGLang fills each multimodal item with pad
//! values derived from its own identity, so event tokens no longer equal
//! Dynamo's routing pads. SGLang echoes the caller's per-item routing label as
//! block-relative spans. This module rewrites those positions to
//! `pad_value_for_mm_hash(label)` before hashing and gives every block in a
//! multimodal chain a Dynamo-owned id, `compute_next_seq_hash(parent, local)`.
//! Because those ids derive only from Dynamo's own hashes, the index sees one
//! id chain per local hash chain even when several engine blocks share a label,
//! for example two different images behind one URL. Engine ids are translated
//! through a per-worker map for parent links and removals, and a reference
//! count publishes a removal only when the last engine block leaves.
//!
//! A store is cut at the first block whose hashed tokens hold a multimodal
//! value no span maps. That block and every block chained after it are
//! excluded, including later stores under an excluded parent, and their
//! removals never reach the index. An engine id is never both mapped and
//! excluded.
//!
//! Every source excludes values that did not fit in `u32`: they all narrow to
//! one sentinel, so indexing them would give different images one local hash.
//! Only a source flagged as engine-owned multimodal identity maps labeled spans
//! and excludes every other value at or above `MM_PAD_SHIFT_VALUE`. Legacy
//! sources keep those values, because their pads are Dynamo's own routing
//! values.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{
    BlockExtraInfo, ExternalSequenceBlockHash, KvCacheEventData, MM_PAD_SHIFT_VALUE,
    PlacementEvent, StorageTier, WorkerWithDpRank, compute_next_seq_hash, pad_value_for_mm_hash,
};

use super::types::{BlockHashValue, RawKvEvent, WIDE_TOKEN_SENTINEL};

/// Parent seed for a multimodal chain that starts at the root.
const MM_CHAIN_ROOT: u64 = 0x6d6d_5f72_6f6f_7421;

/// Bound on the exclusion set. Losing entries never indexes an excluded block:
/// its descendants then fail the indexer's parent lookup.
const DEFAULT_EXCLUDED_CAPACITY: usize = 1 << 20;

/// Bound on engine blocks with a Dynamo-owned id. A live mapping is never
/// evicted: once full, new multimodal chains are excluded like unmapped ones.
const DEFAULT_ID_CAPACITY: usize = 1 << 22;

type Key = (WorkerWithDpRank, StorageTier, u64);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineMmStats {
    /// Blocks published under a Dynamo-owned id.
    pub translated_blocks: u64,
    /// Engine blocks whose Dynamo-owned id was already indexed (aliases).
    pub aliased_blocks: u64,
    /// Blocks dropped because their hashed tokens hold multimodal values no
    /// span maps.
    pub excluded_unmapped: u64,
    /// Blocks dropped because their parent was dropped.
    pub excluded_parent: u64,
    /// Blocks dropped because their spans were out of bounds, overlapping, or
    /// covered non-multimodal tokens.
    pub invalid_spans: u64,
    /// Engine blocks re-stored with a different label than first seen.
    pub relabel_conflicts: u64,
    /// Blocks excluded because the id map was full.
    pub map_full: u64,
    /// Exclusion entries not recorded because the set was full.
    pub excluded_overflow: u64,
}

#[derive(Debug, Clone)]
pub(super) struct EngineMmState {
    engine_owned: bool,
    ids: FxHashMap<Key, u64>,
    refs: FxHashMap<Key, u32>,
    excluded: FxHashSet<Key>,
    excluded_capacity: usize,
    id_capacity: usize,
    pub(super) stats: EngineMmStats,
}

/// Which blocks of a store carried labeled multimodal tokens.
pub(super) struct PreparedStore {
    has_mm: Vec<bool>,
}

enum BlockIssue {
    Unmapped,
    InvalidSpan,
}

impl EngineMmState {
    pub(super) fn new(engine_owned: bool) -> Self {
        Self {
            engine_owned,
            ids: FxHashMap::default(),
            refs: FxHashMap::default(),
            excluded: FxHashSet::default(),
            excluded_capacity: DEFAULT_EXCLUDED_CAPACITY,
            id_capacity: DEFAULT_ID_CAPACITY,
            stats: EngineMmStats::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn with_capacities(
        engine_owned: bool,
        excluded_capacity: usize,
        id_capacity: usize,
    ) -> Self {
        Self {
            excluded_capacity,
            id_capacity,
            ..Self::new(engine_owned)
        }
    }

    pub(super) fn engine_owned(&self) -> bool {
        self.engine_owned
    }

    fn is_multimodal(&self, token: u32) -> bool {
        if self.engine_owned {
            u64::from(token) >= MM_PAD_SHIFT_VALUE
        } else {
            token == WIDE_TOKEN_SENTINEL
        }
    }

    /// Rewrite labeled spans to Dynamo routing pads and cut the store at the
    /// first block whose hashed tokens hold a multimodal value that no span
    /// maps. Returns `None` when nothing in the store may be indexed.
    pub(super) fn prepare_store(
        &mut self,
        raw: &mut RawKvEvent,
        worker: WorkerWithDpRank,
        tier: StorageTier,
    ) -> Option<PreparedStore> {
        let RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            block_mm_infos,
            is_eagle,
            ..
        } = raw
        else {
            return Some(PreparedStore { has_mm: Vec::new() });
        };

        if !self.excluded.is_empty()
            && let Some(parent) = parent_block_hash.map(BlockHashValue::into_u64)
            && self.excluded.contains(&(worker, tier, parent))
        {
            self.stats.excluded_parent += block_hashes.len() as u64;
            for hash in block_hashes.iter() {
                self.exclude((worker, tier, hash.into_u64()));
            }
            return None;
        }

        // Spans are consumed here; leaving them would make conversion mix the
        // labels into the block hash, which the request side never does. Legacy
        // sources keep their multimodal infos.
        let spans = if self.engine_owned {
            block_mm_infos.take()
        } else {
            None
        };
        // Conversion hashes each block's tokens and, for Eagle, also the next
        // block's first token. Classify exactly those windows.
        let block_len = *block_size;
        let eagle = is_eagle.unwrap_or(false);
        let window = block_len + usize::from(eagle);
        let mut has_mm = Vec::with_capacity(block_hashes.len());
        let mut cut: Option<(usize, BlockIssue)> = None;

        for block_idx in 0..block_hashes.len() {
            let start = block_idx * block_len;
            let end = start + window;
            if end > token_ids.len() {
                // Conversion drops short blocks; nothing to classify here.
                break;
            }
            let result = if self.engine_owned && !eagle {
                let block_spans = spans
                    .as_ref()
                    .and_then(|spans| spans.get(block_idx))
                    .and_then(Option::as_ref);
                rewrite_block(&mut token_ids[start..end], block_spans)
            } else if token_ids[start..end]
                .iter()
                .any(|&token| self.is_multimodal(token))
            {
                // Legacy sources and Eagle windows carry no usable labels.
                Err(BlockIssue::Unmapped)
            } else {
                Ok(false)
            };
            match result {
                Ok(block_has_mm) => has_mm.push(block_has_mm),
                Err(issue) => {
                    cut = Some((block_idx, issue));
                    break;
                }
            }
        }

        if let Some((block_idx, issue)) = cut {
            let dropped = (block_hashes.len() - block_idx) as u64;
            match issue {
                BlockIssue::Unmapped => self.stats.excluded_unmapped += dropped,
                BlockIssue::InvalidSpan => self.stats.invalid_spans += dropped,
            }
            for hash in block_hashes.drain(block_idx..) {
                self.exclude((worker, tier, hash.into_u64()));
            }
            // Keep the overlap token the last surviving Eagle block hashes.
            token_ids.truncate(block_idx * block_len + usize::from(eagle && block_idx > 0));
        }

        if block_hashes.is_empty() {
            return None;
        }
        Some(PreparedStore { has_mm })
    }

    /// Replace engine ids with Dynamo-owned ids for blocks in a multimodal
    /// chain. Leading blocks already indexed under the same Dynamo-owned id are
    /// dropped from the store. Returns `false` when nothing new remains.
    pub(super) fn translate_store(
        &mut self,
        event: &mut PlacementEvent,
        prepared: &PreparedStore,
        worker: WorkerWithDpRank,
        tier: StorageTier,
    ) -> bool {
        // Without a label or a mapped parent, no block joins a chain.
        if self.ids.is_empty() && !prepared.has_mm.contains(&true) {
            return true;
        }
        let KvCacheEventData::Stored(store) = &mut event.event.data else {
            return true;
        };

        let parent_engine = store.parent_hash.map(|hash| hash.0);
        let mut in_chain =
            parent_engine.is_some_and(|parent| self.ids.contains_key(&(worker, tier, parent)));
        let mut parent_effective = parent_engine.map(|parent| {
            self.ids
                .get(&(worker, tier, parent))
                .copied()
                .unwrap_or(parent)
        });
        let mut run_parent = parent_effective;
        let mut publishing = false;

        let blocks = std::mem::take(&mut store.blocks);
        let mut published = Vec::with_capacity(blocks.len());
        // Engine ids dropped with the rest of the store (relabel or full map).
        let mut dropped: Vec<u64> = Vec::new();
        let mut remaining = blocks.into_iter().enumerate();

        while let Some((block_idx, mut block)) = remaining.next() {
            let engine_id = block.block_hash.0;
            let labeled = prepared.has_mm.get(block_idx).copied().unwrap_or(false);
            let mut already_indexed = false;

            if in_chain || labeled {
                in_chain = true;
                let dynamo_id = compute_next_seq_hash(
                    parent_effective.unwrap_or(MM_CHAIN_ROOT),
                    block.tokens_hash,
                );
                let key = (worker, tier, engine_id);
                match self.ids.get(&key).copied() {
                    Some(existing) if existing == dynamo_id => already_indexed = true,
                    Some(_) => {
                        // The engine re-stored a block with a different label.
                        // Keep the first mapping and drop the rest of the store:
                        // publishing it would give one external id two local
                        // hashes.
                        self.stats.relabel_conflicts += 1;
                        dropped.push(engine_id);
                        dropped.extend(remaining.by_ref().map(|(_, rest)| rest.block_hash.0));
                        break;
                    }
                    None if self.ids.len() >= self.id_capacity => {
                        // Never evict a live mapping; exclude this block and
                        // the rest of the store as if they were unmapped.
                        dropped.push(engine_id);
                        dropped.extend(remaining.by_ref().map(|(_, rest)| rest.block_hash.0));
                        self.stats.map_full += dropped.len() as u64;
                        break;
                    }
                    None => {
                        // A block excluded earlier, for example while its
                        // label was missing, is mapped from now on.
                        self.excluded.remove(&key);
                        self.ids.insert(key, dynamo_id);
                        let refs = self.refs.entry((worker, tier, dynamo_id)).or_insert(0);
                        *refs += 1;
                        if *refs > 1 {
                            already_indexed = true;
                            self.stats.aliased_blocks += 1;
                        }
                    }
                }
                block.block_hash = ExternalSequenceBlockHash(dynamo_id);
                parent_effective = Some(dynamo_id);
            } else {
                parent_effective = Some(engine_id);
            }

            if !publishing && already_indexed {
                run_parent = parent_effective;
                continue;
            }
            publishing = true;
            if in_chain {
                self.stats.translated_blocks += 1;
            }
            published.push(block);
        }

        for engine_id in dropped {
            self.exclude((worker, tier, engine_id));
        }

        if published.is_empty() {
            return false;
        }
        store.parent_hash = run_parent.map(ExternalSequenceBlockHash);
        store.blocks = published;
        true
    }

    /// Translate removed engine ids. Excluded ids never reached the index. A
    /// Dynamo-owned id is removed from the index only when the last engine
    /// block mapped to it is removed.
    pub(super) fn translate_remove(
        &mut self,
        event: &mut PlacementEvent,
        worker: WorkerWithDpRank,
        tier: StorageTier,
    ) {
        if self.ids.is_empty() && self.excluded.is_empty() {
            return;
        }
        let KvCacheEventData::Removed(remove) = &mut event.event.data else {
            return;
        };
        let mut kept = Vec::with_capacity(remove.block_hashes.len());
        for hash in remove.block_hashes.drain(..) {
            let key = (worker, tier, hash.0);
            if self.excluded.remove(&key) {
                continue;
            }
            match self.ids.remove(&key) {
                Some(dynamo_id) => {
                    let refs_key = (worker, tier, dynamo_id);
                    if let Some(refs) = self.refs.get_mut(&refs_key) {
                        *refs -= 1;
                        if *refs == 0 {
                            self.refs.remove(&refs_key);
                            kept.push(ExternalSequenceBlockHash(dynamo_id));
                        }
                    }
                }
                None => kept.push(hash),
            }
        }
        remove.block_hashes = kept;
    }

    /// Forget all state for a worker after the engine cleared its cache.
    pub(super) fn clear(&mut self, worker: WorkerWithDpRank) {
        self.ids.retain(|(known, _, _), _| *known != worker);
        self.refs.retain(|(known, _, _), _| *known != worker);
        self.excluded.retain(|(known, _, _)| *known != worker);
    }

    /// Exclude an engine id from the index. A mapped id stays mapped: its
    /// removal must still reach the index through its Dynamo-owned id.
    fn exclude(&mut self, key: Key) {
        if self.ids.contains_key(&key) {
            return;
        }
        if self.excluded.len() < self.excluded_capacity {
            self.excluded.insert(key);
        } else if !self.excluded.contains(&key) {
            self.stats.excluded_overflow += 1;
        }
    }

    #[cfg(test)]
    pub(super) fn tracked_ids(&self) -> usize {
        self.ids.len()
    }
}

/// Rewrite labeled spans in one block. Returns whether the block carried
/// labeled multimodal tokens.
fn rewrite_block(tokens: &mut [u32], info: Option<&BlockExtraInfo>) -> Result<bool, BlockIssue> {
    let mut covered = vec![false; tokens.len()];
    let mut has_mm = false;
    if let Some(info) = info {
        for object in &info.mm_objects {
            let pad = pad_value_for_mm_hash(object.mm_hash);
            for &(start, end) in &object.offsets {
                if start >= end || end > tokens.len() {
                    return Err(BlockIssue::InvalidSpan);
                }
                for position in start..end {
                    if covered[position] || u64::from(tokens[position]) < MM_PAD_SHIFT_VALUE {
                        return Err(BlockIssue::InvalidSpan);
                    }
                    tokens[position] = pad;
                    covered[position] = true;
                    has_mm = true;
                }
            }
        }
    }
    let unmapped = tokens
        .iter()
        .zip(&covered)
        .any(|(&token, &covered)| !covered && u64::from(token) >= MM_PAD_SHIFT_VALUE);
    if unmapped {
        return Err(BlockIssue::Unmapped);
    }
    Ok(has_mm)
}
