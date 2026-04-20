// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KV Manager (kvbm-logical G1 backend)
//!
//! Synchronous vLLM-flavour G1 block manager built on `kvbm-logical::BlockManager<G1>`.
//! Translates the mocker's `MoveBlock` protocol into the RAII lifecycle
//! (allocate → stage → register → drop) exposed by kvbm-logical.
//!
//! ## MoveBlock semantics
//!
//! - **Use**: check active pool → clone `ImmutableBlock` to bump refcount; check
//!   active+inactive via `match_blocks(plh)` → reactivate; otherwise allocate a
//!   new `MutableBlock`, stage with PLH, and register. On capacity exhaustion
//!   returns partial count so the scheduler can preempt the oldest running
//!   request.
//! - **Destroy**: drop all RAII handles for the block. Emits a `Removed` KV
//!   event to match the mocker's existing router protocol.
//! - **Deref**: pop one `ImmutableBlock` clone; when the vec empties, the block
//!   transitions to kvbm-logical's inactive pool (RAII return).
//! - **Promote**: PartialBlock (`MutableBlock`) → FullBlock (`ImmutableBlock`).
//!   Collapses onto an existing registered handle if the PLH / SequenceHash is
//!   already present; otherwise stages + registers a new block.
//!
//! ## Eviction backends
//!
//! Three backends are exposed via [`MockerEvictionBackend`]:
//! - `Lineage` (default) — parent-chain aware, evicts leaves first. Subsumes
//!   the `push_front` preemption-priority behaviour of the old `LRUEvictor`.
//! - `Lru` — simple recency-based LRU.
//! - `MultiLru` — 4-tier frequency-aware LRU (requires TinyLFU tracker).

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, PositionalLineageHash, SequenceHash};
use kvbm_logical::registry::BlockRegistry;
use kvbm_logical::tinylfu::TinyLFUTracker;
use kvbm_logical::{BlockManager, ImmutableBlock, MutableBlock};
use uuid::Uuid;

use crate::common::kv_cache_trace;
use crate::common::protocols::{
    G1, KvEventPublishers, MockerEvictionBackend, MoveBlock, PrefillCost,
};
use crate::common::sequence::ActiveSequence;

/// Synchronous G1 KV block manager backed by `kvbm-logical::BlockManager<G1>`.
pub struct KvManager {
    block_manager: BlockManager<G1>,
    max_capacity: usize,
    block_size: usize,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,

    /// PartialBlocks (still filling tokens) held as `MutableBlock`.
    /// Dropped blocks return to kvbm-logical's reset pool.
    active_partial: HashMap<Uuid, MutableBlock<G1>>,

    /// FullBlocks held as `ImmutableBlock`, keyed by `SequenceHash`. The vec
    /// length is the mocker's reference count — each `Use` pushes a clone,
    /// each `Deref` pops one. When the vec empties, the block transitions to
    /// kvbm-logical's inactive pool (RAII return on drop of the last clone).
    active_full: HashMap<SequenceHash, Vec<ImmutableBlock<G1>>>,

    /// Shadow registry of (PLH → mocker u64 seq_hash) for every block that has
    /// been registered in kvbm-logical. kvbm-logical's registry is keyed by
    /// `PositionalLineageHash`, but the router's radix tree is keyed by the
    /// mocker's u64 `SequenceHash` on `UniqueBlock::FullBlock`. We keep this
    /// map so we can emit router-compatible `Removed` events when kvbm-logical
    /// silently evicts inactive blocks inside `allocate_blocks`.
    registered_plhs: HashMap<PositionalLineageHash, SequenceHash>,

    /// Last observed value of kvbm-logical's `evictions` metric. Used to skip
    /// the O(N) presence scan in `emit_evicted_events` when no eviction has
    /// happened since the previous check.
    last_evictions_seen: u64,
}

impl KvManager {
    pub fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self::new_with_eviction_backend(
            max_capacity,
            block_size,
            kv_event_publishers,
            dp_rank,
            MockerEvictionBackend::default(),
        )
    }

    pub fn new_with_eviction_backend(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        eviction_backend: MockerEvictionBackend,
    ) -> Self {
        debug_assert!(max_capacity > 0, "max_capacity must be > 0");

        let mut registry_builder = BlockRegistry::builder();
        if matches!(eviction_backend, MockerEvictionBackend::MultiLru) {
            let tracker = Arc::new(TinyLFUTracker::new(max_capacity));
            registry_builder = registry_builder.frequency_tracker(tracker);
        }
        let registry = registry_builder.build();

        let mut mgr_builder = BlockManager::builder()
            .block_count(max_capacity)
            .block_size(block_size)
            .registry(registry);
        mgr_builder = match eviction_backend {
            MockerEvictionBackend::Lineage => mgr_builder.with_lineage_backend(),
            MockerEvictionBackend::Lru => mgr_builder.with_lru_backend(),
            MockerEvictionBackend::MultiLru => mgr_builder.with_multi_lru_backend(),
        };
        let block_manager = mgr_builder.build().expect("BlockManager build failed");

        if !kv_event_publishers.is_empty() {
            tracing::info!(
                "KvManager initialized with event sink for DP rank {dp_rank} with block_size {block_size}, eviction={eviction_backend:?}"
            );
        }

        Self {
            block_manager,
            max_capacity,
            block_size,
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
            active_partial: HashMap::new(),
            active_full: HashMap::new(),
            registered_plhs: HashMap::new(),
            last_evictions_seen: 0,
        }
    }

    /// Detect any registered blocks that kvbm-logical silently evicted from
    /// its inactive pool (typically during `allocate_blocks`). For each, emit
    /// a `Removed` KV event so the router's radix tree stays in sync.
    ///
    /// Fast path: kvbm-logical exposes a monotonic `evictions` counter in its
    /// metrics snapshot. If the counter hasn't advanced since the previous
    /// call, no eviction happened and we skip the O(N) presence scan — this
    /// is the common case and makes the per-call overhead O(1).
    fn emit_evicted_events(&mut self) {
        if self.registered_plhs.is_empty() {
            return;
        }
        let current_evictions = self.block_manager.metrics().snapshot().evictions;
        if current_evictions == self.last_evictions_seen {
            return;
        }
        self.last_evictions_seen = current_evictions;

        let plhs: Vec<PositionalLineageHash> = self.registered_plhs.keys().copied().collect();
        let presence = self
            .block_manager
            .block_registry()
            .check_presence::<G1>(&plhs);
        let mut evicted = Vec::new();
        for (plh, present) in presence {
            if !present && let Some(seq_hash) = self.registered_plhs.remove(&plh) {
                evicted.push(seq_hash);
            }
        }
        if !evicted.is_empty() {
            self.publish_kv_event(evicted, &[], None, false, None);
        }
    }

    /// Emit a `Stored` or `Removed` KV event to the router.
    /// Ported verbatim from the old `vllm_backend::publish_kv_event` to
    /// preserve KV-aware routing semantics (parent_hash chaining, token_ids).
    fn publish_kv_event(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<u64>,
        is_store: bool,
        token_ids: Option<Vec<Vec<u32>>>,
    ) {
        if full_blocks.is_empty() {
            return;
        }

        kv_cache_trace::log_vllm_trace(
            if is_store { "allocation" } else { "eviction" },
            self.dp_rank,
            self.block_size,
            self.num_active_blocks(),
            self.num_inactive_blocks(),
            self.max_capacity,
        );

        if self.kv_event_publishers.is_empty() {
            return;
        }

        let event_data = if is_store {
            debug_assert_eq!(
                local_hashes.len(),
                full_blocks.len(),
                "publish_kv_event: stored blocks must be 1:1 with local_hashes"
            );

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: full_blocks
                    .into_iter()
                    .zip(local_hashes.iter())
                    .map(|(global_hash, local_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(*local_hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            })
        } else {
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: full_blocks
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            })
        };

        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let event = KvCacheEvent {
            event_id,
            data: event_data,
            dp_rank: self.dp_rank,
        };

        if let Err(e) = self
            .kv_event_publishers
            .publish(event, token_ids.as_deref())
        {
            tracing::warn!("Failed to publish KV event: {e}");
        }
    }

    /// Process a `MoveBlock` instruction synchronously.
    ///
    /// For `MoveBlock::Use`, returns the number of blocks successfully allocated.
    /// On partial failure, blocks `0..N` are committed but block `N+1` could not
    /// be allocated (capacity exhausted); the scheduler uses this to trigger
    /// preemption.
    ///
    /// For `Destroy` / `Deref` / `Promote`, returns 1 on success and panics on
    /// invalid state (consistent with the old `vllm_backend` semantics).
    pub fn process(&mut self, event: &MoveBlock) -> usize {
        match event {
            MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent) => self.process_use(
                blocks,
                local_hashes,
                plhs,
                token_ids.as_deref(),
                parent.as_ref(),
            ),
            MoveBlock::Destroy(hashes) => {
                self.process_destroy(hashes);
                1
            }
            MoveBlock::Deref(hashes) => {
                self.process_deref(hashes);
                1
            }
            MoveBlock::Promote(uuid, seq_hash, parent_hash, local_hash, plh, token_ids) => {
                self.process_promote(
                    *uuid,
                    *seq_hash,
                    *parent_hash,
                    *local_hash,
                    *plh,
                    token_ids.clone(),
                );
                1
            }
        }
    }

    fn process_use(
        &mut self,
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        plhs: &[PositionalLineageHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
    ) -> usize {
        // Upstream invariant: caller must supply exactly one PLH per FullBlock in
        // `blocks`.
        let expected_full_blocks = blocks
            .iter()
            .filter(|b| matches!(b, UniqueBlock::FullBlock(_)))
            .count();
        assert_eq!(
            plhs.len(),
            expected_full_blocks,
            "Use: plhs.len() must match FullBlock count in blocks"
        );
        assert!(
            local_hashes.is_empty() || local_hashes.len() == expected_full_blocks,
            "Use: local_hashes must be empty or match FullBlock count ({} vs {})",
            local_hashes.len(),
            expected_full_blocks,
        );

        let mut blocks_stored = Vec::<SequenceHash>::new();
        // Track the local_hash for each block we actually store, in push order.
        let mut stored_local_hashes = Vec::<BlockHash>::new();
        let mut stored_token_ids: Option<Vec<Vec<u32>>> = token_ids.map(|_| Vec::new());

        let mut parent_block: Option<&UniqueBlock> = parent;
        let mut plh_idx = 0usize;
        let mut allocated = 0usize;

        for (i, block) in blocks.iter().enumerate() {
            match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    // Already active — bump refcount by cloning the first handle.
                    if let Some(vec) = self.active_full.get_mut(seq_hash) {
                        let cloned = vec[0].clone();
                        vec.push(cloned);
                        parent_block = Some(block);
                        plh_idx += 1;
                        allocated += 1;
                        continue;
                    }

                    // Try active+inactive pools via PLH lookup.
                    let plh = plhs.get(plh_idx).copied();
                    plh_idx += 1;
                    let matched = plh
                        .map(|p| self.block_manager.match_blocks(&[p]))
                        .unwrap_or_default();
                    if let Some(immutable) = matched.into_iter().next() {
                        self.active_full
                            .entry(*seq_hash)
                            .or_default()
                            .push(immutable);
                        // Re-announce to router
                        blocks_stored.push(*seq_hash);
                        if let Some(lh) = local_hashes.get(i) {
                            stored_local_hashes.push(*lh);
                        }
                        if let (Some(ref mut stids), Some(ids)) =
                            (stored_token_ids.as_mut(), token_ids)
                        {
                            stids.push(ids[i].clone());
                        }
                        allocated += 1;
                        continue;
                    }

                    // Allocate new block → stage → register.
                    // NOTE: we do NOT update `parent_block` here — it must point
                    // to the block *before* the first newly-stored one so the
                    // Stored event's `parent_hash` correctly anchors the radix
                    // chain (subsequent stored blocks are chained by position
                    // within the event).
                    let Some(mut alloc) = self.block_manager.allocate_blocks(1) else {
                        break; // capacity exhausted; scheduler will preempt
                    };
                    let mutable = alloc.pop().unwrap();
                    // `plh` is guaranteed `Some` by the front assert on plhs.len().
                    let plh =
                        plh.expect("Use: PLH missing for FullBlock (caller invariant broken)");
                    let complete = mutable.stage(plh, self.block_size).expect("stage failed");
                    let immutable = self.block_manager.register_block(complete);
                    self.active_full
                        .entry(*seq_hash)
                        .or_default()
                        .push(immutable);
                    self.registered_plhs.insert(plh, *seq_hash);

                    blocks_stored.push(*seq_hash);
                    if let Some(lh) = local_hashes.get(i) {
                        stored_local_hashes.push(*lh);
                    }
                    if let (Some(ref mut stids), Some(ids)) = (stored_token_ids.as_mut(), token_ids)
                    {
                        stids.push(ids[i].clone());
                    }
                    allocated += 1;
                }
                UniqueBlock::PartialBlock(uuid) => {
                    if self.active_partial.contains_key(uuid) {
                        // Partial block already held; treat as a no-op success.
                        allocated += 1;
                        continue;
                    }
                    let Some(mut alloc) = self.block_manager.allocate_blocks(1) else {
                        break;
                    };
                    let mutable = alloc.pop().unwrap();
                    self.active_partial.insert(*uuid, mutable);
                    allocated += 1;
                }
            }
        }

        let parent_hash = match parent_block {
            None => None,
            Some(UniqueBlock::FullBlock(block)) => Some(*block),
            Some(UniqueBlock::PartialBlock(_)) => panic!("parent block cannot be partial"),
        };
        self.publish_kv_event(
            blocks_stored,
            &stored_local_hashes,
            parent_hash,
            true,
            stored_token_ids,
        );

        // Detect any blocks kvbm-logical evicted from its inactive pool
        // during the `allocate_blocks` calls above and emit `Removed` events.
        self.emit_evicted_events();

        allocated
    }

    fn process_destroy(&mut self, blocks: &[UniqueBlock]) {
        let mut destroyed = Vec::<SequenceHash>::new();
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => {
                    self.active_partial
                        .remove(uuid)
                        .expect("Destroy: partial block not in active pool");
                }
                UniqueBlock::FullBlock(seq_hash) => {
                    self.active_full
                        .remove(seq_hash)
                        .expect("Destroy: full block not in active pool");
                    destroyed.push(*seq_hash);
                }
            }
        }
        self.publish_kv_event(destroyed, &[], None, false, None);
    }

    fn process_deref(&mut self, blocks: &[UniqueBlock]) {
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(_) => {
                    panic!("Deref on PartialBlock is not valid");
                }
                UniqueBlock::FullBlock(seq_hash) => {
                    let vec = self
                        .active_full
                        .get_mut(seq_hash)
                        .expect("Deref: full block not in active pool");
                    vec.pop();
                    if vec.is_empty() {
                        self.active_full.remove(seq_hash);
                    }
                }
            }
        }
    }

    fn process_promote(
        &mut self,
        uuid: Uuid,
        seq_hash: SequenceHash,
        parent_hash: Option<u64>,
        local_hash: BlockHash,
        plh: PositionalLineageHash,
        token_ids: Option<Vec<u32>>,
    ) {
        let mutable = self
            .active_partial
            .remove(&uuid)
            .expect("Promote: partial block not found");

        // Detect collision: seq_hash already has registered handles (active or inactive).
        let is_new = if let Some(vec) = self.active_full.get_mut(&seq_hash) {
            // Collision on active pool — drop MutableBlock, clone existing handle.
            drop(mutable);
            let existing = vec[0].clone();
            vec.push(existing);
            false
        } else if let Some(immutable) = self.block_manager.match_blocks(&[plh]).into_iter().next() {
            // Collision on inactive pool — reactivate existing handle.
            drop(mutable);
            self.active_full.insert(seq_hash, vec![immutable]);
            false
        } else {
            // Fresh registration.
            let complete = mutable
                .stage(plh, self.block_size)
                .expect("stage failed during promote");
            let immutable = self.block_manager.register_block(complete);
            self.active_full.insert(seq_hash, vec![immutable]);
            self.registered_plhs.insert(plh, seq_hash);
            true
        };

        if is_new {
            self.publish_kv_event(
                vec![seq_hash],
                &[local_hash],
                parent_hash,
                true,
                token_ids.map(|t| vec![t]),
            );
        }
    }

    pub fn current_capacity(&self) -> usize {
        self.block_manager.total_blocks() - self.block_manager.available_blocks()
    }

    pub fn current_capacity_perc(&self) -> f64 {
        self.current_capacity() as f64 / self.max_capacity as f64
    }

    pub fn num_active_blocks(&self) -> usize {
        self.active_partial.len() + self.active_full.values().map(|v| v.len()).sum::<usize>()
    }

    pub fn get_active_perc(&self) -> f64 {
        self.num_active_blocks() as f64 / self.max_capacity as f64
    }

    pub fn num_inactive_blocks(&self) -> usize {
        self.block_manager.metrics().snapshot().inactive_pool_size as usize
    }

    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn dp_rank(&self) -> u32 {
        self.dp_rank
    }

    /// Calculate the prefill cost for a sequence by scanning `unique_blocks` in
    /// order and counting the longest prefix that is cached (active or
    /// inactive). Stops at first cache miss — KV states are computed
    /// sequentially, so anything after a miss must be recomputed.
    pub fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        let seq_blocks = sequence.unique_blocks();

        // Without prefix caching, each `UniqueBlock::FullBlock` carries a
        // randomised hash that can't possibly be in the cache across requests
        // — skip the PLH lookup (PLH is deterministic from tokens) to stay
        // consistent with that no-reuse contract.
        let overlap_blocks = if sequence.enable_prefix_caching() {
            let plhs = sequence.positional_lineage_hashes();
            let mut overlap = 0;
            for (i, block) in seq_blocks.iter().enumerate() {
                match block {
                    UniqueBlock::FullBlock(seq_hash) => {
                        if self.active_full.contains_key(seq_hash) {
                            overlap += 1;
                            continue;
                        }
                        let Some(plh) = plhs.get(i).copied() else {
                            break;
                        };
                        let presence = self
                            .block_manager
                            .block_registry()
                            .check_presence::<G1>(&[plh]);
                        if presence.first().is_some_and(|(_, present)| *present) {
                            overlap += 1;
                        } else {
                            break;
                        }
                    }
                    UniqueBlock::PartialBlock(_) => break,
                }
            }
            overlap
        } else {
            0
        };

        let new_blocks = seq_blocks.len() - overlap_blocks;
        let cached_tokens = (overlap_blocks * self.block_size).min(sequence.num_input_tokens());
        let new_tokens = sequence.num_input_tokens() - cached_tokens;

        PrefillCost {
            new_blocks,
            new_tokens,
            cached_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mgr(capacity: usize, block_size: usize) -> KvManager {
        KvManager::new_with_event_sink(capacity, block_size, KvEventPublishers::default(), 0)
    }

    fn plh(v: u64) -> PositionalLineageHash {
        PositionalLineageHash::new(v, None, 0)
    }

    fn use_full(mgr: &mut KvManager, seq_hash: u64, p: PositionalLineageHash) -> usize {
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(seq_hash)],
            vec![],
            vec![p],
            None,
            None,
        ))
    }

    fn use_partial(mgr: &mut KvManager, uuid: Uuid) -> usize {
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::PartialBlock(uuid)],
            vec![],
            vec![],
            None,
            None,
        ))
    }

    fn deref_full(mgr: &mut KvManager, seq_hash: u64) {
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::FullBlock(seq_hash)]));
    }

    fn destroy_full(mgr: &mut KvManager, seq_hash: u64) {
        mgr.process(&MoveBlock::Destroy(vec![UniqueBlock::FullBlock(seq_hash)]));
    }

    #[test]
    fn test_use_single_full_block() {
        let mut mgr = make_mgr(10, 16);
        assert_eq!(use_full(&mut mgr, 1, plh(100)), 1);
        assert_eq!(mgr.num_active_blocks(), 1);
    }

    #[test]
    fn test_duplicate_use_bumps_refcount() {
        let mut mgr = make_mgr(10, 16);
        use_full(&mut mgr, 1, plh(100));
        use_full(&mut mgr, 1, plh(100));
        assert_eq!(mgr.num_active_blocks(), 2);
    }

    #[test]
    fn test_capacity_exhaustion_returns_partial() {
        let mut mgr = make_mgr(4, 16);
        for i in 0..4 {
            assert_eq!(use_full(&mut mgr, i, plh(i + 100)), 1);
        }
        // Fifth allocation fails - returns 0 (no blocks allocated)
        assert_eq!(use_full(&mut mgr, 4, plh(500)), 0);
    }

    #[test]
    fn test_deref_returns_to_inactive() {
        let mut mgr = make_mgr(4, 16);
        use_full(&mut mgr, 1, plh(100));
        deref_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 0);
    }

    #[test]
    fn test_inactive_reuse_via_match_blocks() {
        let mut mgr = make_mgr(10, 16);
        let p = plh(100);
        use_full(&mut mgr, 1, p);
        deref_full(&mut mgr, 1);
        // Use with same PLH reuses the inactive block.
        assert_eq!(use_full(&mut mgr, 2, p), 1);
    }

    #[test]
    fn test_eviction_frees_inactive_for_new_allocation() {
        let mut mgr = make_mgr(4, 16);
        for i in 0..4 {
            use_full(&mut mgr, i, plh(i + 100));
        }
        for i in 0..4 {
            deref_full(&mut mgr, i);
        }
        for i in 10..14 {
            assert_eq!(use_full(&mut mgr, i, plh(i + 1000)), 1);
        }
        assert_eq!(mgr.num_active_blocks(), 4);
    }

    #[test]
    fn test_promote_basic() {
        let mut mgr = make_mgr(10, 16);
        let uuid = Uuid::new_v4();
        use_partial(&mut mgr, uuid);
        mgr.process(&MoveBlock::Promote(uuid, 42, None, 0, plh(500), None));
        assert_eq!(mgr.num_active_blocks(), 1);
        assert!(mgr.active_partial.is_empty());
        assert!(mgr.active_full.contains_key(&42));
    }

    #[test]
    #[should_panic(expected = "Promote: partial block not found")]
    fn test_promote_nonexistent_panics() {
        let mut mgr = make_mgr(10, 16);
        mgr.process(&MoveBlock::Promote(
            Uuid::new_v4(),
            42,
            None,
            0,
            plh(500),
            None,
        ));
    }

    #[test]
    #[should_panic(expected = "Deref on PartialBlock is not valid")]
    fn test_deref_on_partial_panics() {
        let mut mgr = make_mgr(10, 16);
        let uuid = Uuid::new_v4();
        use_partial(&mut mgr, uuid);
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::PartialBlock(uuid)]));
    }

    #[test]
    fn test_destroy_full_block() {
        let mut mgr = make_mgr(10, 16);
        use_full(&mut mgr, 1, plh(100));
        destroy_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 0);
    }

    #[test]
    fn test_prefill_cost_no_overlap() {
        let mgr = make_mgr(10, 16);
        let tokens: Vec<u32> = (0..35).collect();
        let seq = ActiveSequence::new(tokens, 10, Some(16), true, false);
        let cost = mgr.get_prefill_cost(&seq);
        assert_eq!(cost.new_blocks, seq.unique_blocks().len());
        assert_eq!(cost.new_tokens, 35);
    }

    #[test]
    fn test_eviction_backend_lru_and_multi_lru() {
        for backend in [MockerEvictionBackend::Lru, MockerEvictionBackend::MultiLru] {
            let mut mgr = KvManager::new_with_eviction_backend(
                4,
                16,
                KvEventPublishers::default(),
                0,
                backend,
            );
            for i in 0..4u64 {
                assert_eq!(use_full(&mut mgr, i, plh(i + 100)), 1);
            }
            for i in 0..4u64 {
                deref_full(&mut mgr, i);
            }
            for i in 10..14u64 {
                assert_eq!(
                    use_full(&mut mgr, i, plh(i + 1000)),
                    1,
                    "backend={backend:?}"
                );
            }
            assert_eq!(mgr.num_active_blocks(), 4);
        }
    }

    #[test]
    fn test_failure_on_max_capacity() {
        fn use_batch(mgr: &mut KvManager, ids: &[u64]) -> usize {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let plhs: Vec<_> = ids.iter().map(|&id| plh(id)).collect();
            mgr.process(&MoveBlock::Use(blocks, vec![], plhs, None, None))
        }

        let mut mgr = make_mgr(10, 16);

        // Fill capacity in a single Use batch.
        let ids: Vec<u64> = (0..10).collect();
        assert_eq!(use_batch(&mut mgr, &ids), 10, "all 10 should allocate");
        assert_eq!(mgr.current_capacity(), 10);

        // One more block must return 0 (no partial allocation possible, not panic).
        assert_eq!(
            use_batch(&mut mgr, &[10]),
            0,
            "over-capacity Use must return 0"
        );
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        // Batch helpers local to this test. Each FullBlock gets a unique PLH
        // derived from its id so the PLH->SequenceHash mapping stays 1:1.
        fn use_blocks(mgr: &mut KvManager, ids: &[u64]) {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let plhs: Vec<_> = ids.iter().map(|&id| plh(id)).collect();
            mgr.process(&MoveBlock::Use(blocks, vec![], plhs, None, None));
        }
        fn destroy_blocks(mgr: &mut KvManager, ids: &[u64]) {
            let blocks = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            mgr.process(&MoveBlock::Destroy(blocks));
        }
        fn deref_blocks(mgr: &mut KvManager, ids: &[u64]) {
            let blocks = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            mgr.process(&MoveBlock::Deref(blocks));
        }
        fn refcount(mgr: &KvManager, id: u64) -> usize {
            mgr.active_full.get(&id).map(|v| v.len()).unwrap_or(0)
        }
        fn assert_active(mgr: &KvManager, expected: &[(u64, usize)]) {
            let total: usize = expected.iter().map(|&(_, r)| r).sum();
            assert_eq!(
                mgr.num_active_blocks(),
                total,
                "active count mismatch; expected={expected:?}"
            );
            for &(id, r) in expected {
                assert_eq!(refcount(mgr, id), r, "block {id} refcount mismatch");
            }
        }
        // Inactive membership helper. Uses `check_presence::<G1>` (non-mutating)
        // against a snapshot of PLHs to confirm each expected id is present in
        // kvbm-logical AND absent from `active_full`. Also checks total count
        // matches so we catch stray inactive entries too.
        //
        // NOTE: under kvbm-logical, `Destroy` removes the block from
        // `active_full` but the `ImmutableBlock` drop returns it to the
        // inactive pool — so a destroyed block appears as inactive here until
        // evicted. This differs from the old `HashCache` where `Destroy`
        // removed the block entirely.
        fn assert_inactive_blocks(mgr: &KvManager, expected_ids: &[u64]) {
            assert_eq!(
                mgr.num_inactive_blocks(),
                expected_ids.len(),
                "inactive count mismatch; expected={expected_ids:?}"
            );
            let plhs: Vec<_> = expected_ids.iter().map(|&id| plh(id)).collect();
            let presence = mgr
                .block_manager
                .block_registry()
                .check_presence::<G1>(&plhs);
            for ((_, present), &id) in presence.iter().zip(expected_ids.iter()) {
                assert!(
                    *present,
                    "block {id} expected in inactive pool, not found in registry"
                );
                assert!(
                    !mgr.active_full.contains_key(&id),
                    "block {id} expected inactive but is in active pool"
                );
            }
        }

        let mut mgr = make_mgr(10, 16);

        // Use blocks 0..=4, then 0, 1, 5, 6 — 0 and 1 bump refcount to 2.
        use_blocks(&mut mgr, &[0, 1, 2, 3, 4]);
        use_blocks(&mut mgr, &[0, 1, 5, 6]);
        assert_active(
            &mgr,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        // Destroy block 4; deref 0, 1, 2, 3. 2 and 3 drop to inactive via
        // RAII; 4 is destroyed (removed from active_full + Removed emitted)
        // but kvbm-logical still has it in the inactive pool.
        destroy_blocks(&mut mgr, &[4]);
        deref_blocks(&mut mgr, &[0, 1, 2, 3]);
        assert_active(&mgr, &[(0, 1), (1, 1), (5, 1), (6, 1)]);
        assert_inactive_blocks(&mgr, &[2, 3, 4]);

        // Destroy block 6; deref 0, 1, 5. Active drains; inactive = {0..=6}.
        destroy_blocks(&mut mgr, &[6]);
        deref_blocks(&mut mgr, &[0, 1, 5]);
        assert_active(&mgr, &[]);
        assert_inactive_blocks(&mgr, &[0, 1, 2, 3, 4, 5, 6]);

        // Re-use 0, 1, 2 (reactivates from inactive) + 7, 8, 9 (new, 3 free
        // slots). No eviction needed — inactive shrinks to {3, 4, 5, 6}.
        use_blocks(&mut mgr, &[0, 1, 2, 7, 8, 9]);
        assert_active(&mgr, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);
        assert_inactive_blocks(&mgr, &[3, 4, 5, 6]);

        // Allocate through capacity: 10, 11, 12 force eviction of 3 inactive
        // entries. Exact survivor depends on eviction order (Lineage/LRU), so
        // only assert count.
        use_blocks(&mut mgr, &[10, 11, 12]);
        assert_eq!(mgr.num_active_blocks(), 9);
        assert_eq!(mgr.num_inactive_blocks(), 1);

        // One more block keeps us at full capacity without panicking.
        use_blocks(&mut mgr, &[13]);
        assert_eq!(mgr.num_active_blocks(), 10);
        assert_eq!(mgr.num_inactive_blocks(), 0);
    }

    #[test]
    fn test_chunked_prefill_parent_hash() {
        use std::sync::Mutex;

        use crate::common::protocols::KvCacheEventSink;

        #[derive(Default)]
        struct CapturingSink {
            events: Mutex<Vec<KvCacheEvent>>,
        }
        impl KvCacheEventSink for CapturingSink {
            fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
                self.events.lock().unwrap().push(event);
                Ok(())
            }
        }

        let block_size = 64;
        let tokens: Vec<u32> = (0..512).collect(); // 8 full blocks
        let mut seq = ActiveSequence::new(tokens, 100, Some(block_size), true, false);

        let sink = Arc::new(CapturingSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone() as _), None);
        let mut mgr = KvManager::new_with_event_sink(256, block_size, publishers, 0);

        // Chunk 1: blocks 0..=3 (cumulative 256 tokens).
        let signal = seq.prepare_allocation(256).unwrap();
        mgr.process(&signal);
        seq.commit_allocation(256);

        // Chunk 2: blocks 4..=7 (cumulative 512 tokens).
        let signal = seq.prepare_allocation(512).unwrap();
        mgr.process(&signal);
        seq.commit_allocation(512);

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 2, "expected two Stored events");

        let KvCacheEventData::Stored(ref store1) = events[0].data else {
            panic!("expected Stored event");
        };
        assert!(
            store1.parent_hash.is_none(),
            "first chunk should have no parent_hash"
        );

        let KvCacheEventData::Stored(ref store2) = events[1].data else {
            panic!("expected Stored event");
        };
        let UniqueBlock::FullBlock(expected_hash) = seq.unique_blocks()[3].clone() else {
            panic!("expected FullBlock at index 3");
        };
        assert_eq!(
            store2.parent_hash,
            Some(ExternalSequenceBlockHash(expected_hash)),
            "second chunk's parent_hash should be block 3's seq_hash"
        );
    }

    #[test]
    fn test_repreempt_after_partial_recompute_only_frees_reallocated_blocks() {
        let mut seq = ActiveSequence::new((0..6).collect(), 16, Some(4), true, false);
        let mut mgr = make_mgr(16, 4);

        let signal = seq.take_creation_signal().unwrap();
        assert_eq!(mgr.process(&signal), 2);

        for _ in 0..3 {
            let signals = seq.generate();
            for signal in &signals {
                mgr.process(signal);
            }
            if seq.generated_tokens() < seq.max_output_tokens() {
                seq.commit_allocation(seq.len());
            }
        }
        assert_eq!(mgr.num_active_blocks(), 3);

        let first_reset = seq.reset_with_signal();
        for signal in &first_reset {
            mgr.process(signal);
        }
        assert_eq!(mgr.num_active_blocks(), 0);

        let prompt_only = seq.prepare_allocation(seq.num_input_tokens()).unwrap();
        assert_eq!(mgr.process(&prompt_only), 2);
        seq.commit_allocation(seq.num_input_tokens());
        assert_eq!(mgr.num_active_blocks(), 2);

        let second_reset = seq.reset_with_signal();
        for signal in &second_reset {
            mgr.process(signal);
        }
        assert_eq!(mgr.num_active_blocks(), 0);
    }
}
