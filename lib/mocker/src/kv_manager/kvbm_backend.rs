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
//! - **Deref**: release one request-owned handle. For `PartialBlock` this drops
//!   the unique `MutableBlock` and returns it to the reset pool. For
//!   `FullBlock` this pops one `ImmutableBlock` clone; when the vec empties,
//!   the block transitions to kvbm-logical's inactive pool (RAII return).
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

/// Classification for each block processed inside `Use`.
///
/// - `ActiveHit`: block is already pinned in `active_full` / `active_partial`;
///   we just bump our local refcount (handle clone).
/// - `InactiveHit`: block was in kvbm-logical's inactive pool and was
///   reactivated via `match_blocks(plh)`.
/// - `NewStore`: block was freshly allocated, staged, and registered.
///
/// The router radix tree already knows about `ActiveHit` and `InactiveHit`
/// (it only forgets on explicit `Removed`), so only `NewStore` should emit a
/// `Stored` KV event. Both hit outcomes still advance the parent cursor so
/// subsequent `NewStore` batches anchor to the last reused full block.
enum UseOutcome {
    ActiveHit,
    InactiveHit,
    NewStore,
}

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
    /// evicts inactive blocks as a side effect of `allocate_blocks_with_evictions`.
    registered_plhs: HashMap<PositionalLineageHash, SequenceHash>,
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
            // `local_hashes` is either empty (caller has no token-derived
            // hashes to publish) or 1:1 with `full_blocks`. Match the
            // front-door contract in `process_use`.
            debug_assert!(
                local_hashes.is_empty() || local_hashes.len() == full_blocks.len(),
                "publish_kv_event: local_hashes must be empty or 1:1 with full_blocks ({} vs {})",
                local_hashes.len(),
                full_blocks.len(),
            );

            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: full_blocks
                    .into_iter()
                    .enumerate()
                    .map(|(i, global_hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(global_hash),
                        tokens_hash: LocalBlockHash(
                            local_hashes.get(i).copied().unwrap_or_default(),
                        ),
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
    /// For `Deref` / `Promote`, returns 1 on success and panics on
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
        let mut stored_local_hashes = Vec::<BlockHash>::new();
        let mut stored_token_ids: Option<Vec<Vec<u32>>> = token_ids.map(|_| Vec::new());
        let mut evicted_plhs = Vec::<PositionalLineageHash>::new();

        let mut parent_block: Option<&UniqueBlock> = parent;
        let mut plh_idx = 0usize;
        let mut allocated = 0usize;

        for (i, block) in blocks.iter().enumerate() {
            let outcome = match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    // Active hit — bump refcount by cloning the first handle.
                    if let Some(vec) = self.active_full.get_mut(seq_hash) {
                        let cloned = vec[0].clone();
                        vec.push(cloned);
                        plh_idx += 1;
                        UseOutcome::ActiveHit
                    } else {
                        // Not active: try inactive via PLH lookup, else allocate fresh.
                        let plh = plhs[plh_idx];
                        plh_idx += 1;
                        let matched = self.block_manager.match_blocks(&[plh]);
                        if let Some(immutable) = matched.into_iter().next() {
                            self.active_full
                                .entry(*seq_hash)
                                .or_default()
                                .push(immutable);
                            UseOutcome::InactiveHit
                        } else {
                            let Some((mut alloc, evicted)) =
                                self.block_manager.allocate_blocks_with_evictions(1)
                            else {
                                break; // capacity exhausted; scheduler will preempt
                            };
                            evicted_plhs.extend(evicted);
                            let mutable = alloc.pop().unwrap();
                            let complete =
                                mutable.stage(plh, self.block_size).expect("stage failed");
                            let immutable = self.block_manager.register_block(complete);
                            self.active_full
                                .entry(*seq_hash)
                                .or_default()
                                .push(immutable);
                            self.registered_plhs.insert(plh, *seq_hash);
                            UseOutcome::NewStore
                        }
                    }
                }
                UniqueBlock::PartialBlock(uuid) => {
                    if self.active_partial.contains_key(uuid) {
                        UseOutcome::ActiveHit
                    } else {
                        let Some((mut alloc, evicted)) =
                            self.block_manager.allocate_blocks_with_evictions(1)
                        else {
                            break;
                        };
                        evicted_plhs.extend(evicted);
                        let mutable = alloc.pop().unwrap();
                        self.active_partial.insert(*uuid, mutable);
                        UseOutcome::ActiveHit
                    }
                }
            };

            match outcome {
                UseOutcome::ActiveHit | UseOutcome::InactiveHit => {
                    // Router already has this block; no `Stored` event.
                    // Advance the parent cursor across the reused prefix so any
                    // subsequent `NewStore` batches anchor at the last reused
                    // full block.
                    if matches!(block, UniqueBlock::FullBlock(_)) {
                        parent_block = Some(block);
                    }
                }
                UseOutcome::NewStore => {
                    // Freshly registered: announce to router.
                    // NOTE: we do NOT advance `parent_block` here — within a
                    // single `Stored` event, consecutive blocks chain via their
                    // position in `blocks[]`, so `parent_hash` must remain the
                    // block *before* the first newly-stored one.
                    if let UniqueBlock::FullBlock(seq_hash) = block {
                        blocks_stored.push(*seq_hash);
                        if let Some(lh) = local_hashes.get(i) {
                            stored_local_hashes.push(*lh);
                        }
                        if let (Some(ref mut stids), Some(ids)) =
                            (stored_token_ids.as_mut(), token_ids)
                        {
                            stids.push(ids[i].clone());
                        }
                    }
                }
            }
            allocated += 1;
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

        // Translate any blocks kvbm-logical evicted from its inactive pool
        // during the allocations above into router `Removed` events.
        if !evicted_plhs.is_empty() {
            let evicted: Vec<SequenceHash> = evicted_plhs
                .into_iter()
                .filter_map(|plh| self.registered_plhs.remove(&plh))
                .collect();
            if !evicted.is_empty() {
                self.publish_kv_event(evicted, &[], None, false, None);
            }
        }

        allocated
    }

    fn process_deref(&mut self, blocks: &[UniqueBlock]) {
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => {
                    self.active_partial
                        .remove(uuid)
                        .expect("Deref: partial block not in active pool");
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

    /// Number of **distinct** physically-resident KV blocks currently pinned
    /// by mocker (not available for eviction).
    pub fn num_active_blocks(&self) -> usize {
        // kvbm-logical partitions physical blocks into three pools:
        //   total = reset + inactive + active
        // where `available = reset + inactive`. So `total - available` is
        // exactly the number of registered (ImmutableBlock) full blocks.
        self.block_manager.total_blocks() - self.block_manager.available_blocks()
    }

    /// Total number of held RAII handles (refcount-style): one per held
    /// `MutableBlock` plus one per cloned `ImmutableBlock` in `active_full`.
    /// Shared-prefix reuse inflates this above the distinct-block count.
    pub fn num_active_block_refs(&self) -> usize {
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
    use std::sync::Mutex;

    use super::*;
    use crate::common::protocols::KvCacheEventSink;

    /// Capturing event sink for router-publication assertions.
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

    fn make_mgr(capacity: usize, block_size: usize) -> KvManager {
        KvManager::new_with_event_sink(capacity, block_size, KvEventPublishers::default(), 0)
    }

    fn make_mgr_capturing(capacity: usize, block_size: usize) -> (KvManager, Arc<CapturingSink>) {
        let sink = Arc::new(CapturingSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone() as _), None);
        (
            KvManager::new_with_event_sink(capacity, block_size, publishers, 0),
            sink,
        )
    }

    fn make_mgr_capturing_with_backend(
        capacity: usize,
        block_size: usize,
        backend: MockerEvictionBackend,
    ) -> (KvManager, Arc<CapturingSink>) {
        let sink = Arc::new(CapturingSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone() as _), None);
        (
            KvManager::new_with_eviction_backend(capacity, block_size, publishers, 0, backend),
            sink,
        )
    }

    fn plh(v: u64) -> PositionalLineageHash {
        PositionalLineageHash::new(v, None, 0)
    }

    fn lineage_plh(id: u64) -> PositionalLineageHash {
        match id {
            0 => PositionalLineageHash::new(0, None, 0),
            1 => PositionalLineageHash::new(1, Some(0), 1),
            2 => PositionalLineageHash::new(2, Some(1), 2),
            3 => PositionalLineageHash::new(3, Some(2), 3),
            4 => PositionalLineageHash::new(4, Some(3), 4),
            5 => PositionalLineageHash::new(5, Some(1), 2),
            6 => PositionalLineageHash::new(6, Some(5), 3),
            7 => PositionalLineageHash::new(7, Some(2), 3),
            8 => PositionalLineageHash::new(8, Some(7), 4),
            9 => PositionalLineageHash::new(9, Some(8), 5),
            10 => PositionalLineageHash::new(10, None, 0),
            11 => PositionalLineageHash::new(11, Some(10), 1),
            12 => PositionalLineageHash::new(12, Some(11), 2),
            13 => PositionalLineageHash::new(13, None, 0),
            _ => plh(id),
        }
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

    fn deref_partial(mgr: &mut KvManager, uuid: Uuid) {
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::PartialBlock(uuid)]));
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
        // Same seq_hash used twice: only one distinct physical block is
        // resident, but the mocker holds two RAII handles.
        assert_eq!(mgr.num_active_blocks(), 1);
        assert_eq!(mgr.num_active_block_refs(), 2);
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
    fn test_deref_partial_returns_to_reset() {
        let mut mgr = make_mgr(10, 16);
        let uuid = Uuid::new_v4();
        use_partial(&mut mgr, uuid);
        assert_eq!(mgr.active_partial.len(), 1);
        deref_partial(&mut mgr, uuid);
        assert!(mgr.active_partial.is_empty());
        assert_eq!(mgr.num_active_block_refs(), 0);
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
        assert_eq!(mgr.num_active_blocks(), 10);

        // One more block must return 0 (no partial allocation possible, not panic).
        assert_eq!(
            use_batch(&mut mgr, &[10]),
            0,
            "over-capacity Use must return 0"
        );
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        fn use_blocks(mgr: &mut KvManager, ids: &[u64]) -> usize {
            let blocks: Vec<_> = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            let plhs: Vec<_> = ids.iter().map(|&id| lineage_plh(id)).collect();
            mgr.process(&MoveBlock::Use(blocks, vec![], plhs, None, None))
        }
        fn deref_blocks(mgr: &mut KvManager, ids: &[u64]) {
            let blocks = ids.iter().map(|&id| UniqueBlock::FullBlock(id)).collect();
            mgr.process(&MoveBlock::Deref(blocks));
        }
        fn refcount(mgr: &KvManager, id: u64) -> usize {
            mgr.active_full.get(&id).map(|v| v.len()).unwrap_or(0)
        }
        fn assert_active(mgr: &KvManager, expected: &[(u64, usize)]) {
            let distinct = expected.len();
            let total_refs: usize = expected.iter().map(|&(_, r)| r).sum();
            assert_eq!(
                mgr.num_active_blocks(),
                distinct,
                "distinct active-block count mismatch; expected={expected:?}"
            );
            assert_eq!(
                mgr.num_active_block_refs(),
                total_refs,
                "active handle-refcount mismatch; expected={expected:?}"
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
        // NOTE: under kvbm-logical, once the last `ImmutableBlock` handle is
        // dropped, the block returns to the inactive pool and remains matchable
        // until eviction.
        fn assert_inactive_blocks(mgr: &KvManager, expected_ids: &[u64]) {
            assert_eq!(
                mgr.num_inactive_blocks(),
                expected_ids.len(),
                "inactive count mismatch; expected={expected_ids:?}"
            );
            let plhs: Vec<_> = expected_ids.iter().map(|&id| lineage_plh(id)).collect();
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
        fn drain_events(sink: &Arc<CapturingSink>) -> Vec<KvCacheEvent> {
            std::mem::take(&mut *sink.events.lock().unwrap())
        }
        fn assert_stored_event(
            event: &KvCacheEvent,
            expected_blocks: &[u64],
            expected_parent: Option<u64>,
        ) {
            let KvCacheEventData::Stored(data) = &event.data else {
                panic!("expected Stored event, got {:?}", event.data);
            };
            let actual_blocks: Vec<u64> =
                data.blocks.iter().map(|block| block.block_hash.0).collect();
            assert_eq!(actual_blocks, expected_blocks, "stored blocks mismatch");
            assert_eq!(
                data.parent_hash.map(|hash| hash.0),
                expected_parent,
                "stored parent_hash mismatch"
            );
        }
        fn assert_removed_event(event: &KvCacheEvent, expected_blocks: &[u64]) {
            let KvCacheEventData::Removed(data) = &event.data else {
                panic!("expected Removed event, got {:?}", event.data);
            };
            let actual_blocks: Vec<u64> = data.block_hashes.iter().map(|hash| hash.0).collect();
            assert_eq!(actual_blocks, expected_blocks, "removed blocks mismatch");
        }

        let (mut mgr, sink) =
            make_mgr_capturing_with_backend(10, 16, MockerEvictionBackend::Lineage);

        // Use blocks 0..=4, then 0, 1, 5, 6 — 0 and 1 bump refcount to 2.
        assert_eq!(use_blocks(&mut mgr, &[0, 1, 2, 3, 4]), 5);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 1, "expected one Stored event for [0..=4]");
        assert_stored_event(&events[0], &[0, 1, 2, 3, 4], None);

        assert_eq!(use_blocks(&mut mgr, &[0, 1, 5, 6]), 4);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 1, "expected one Stored event for [5, 6]");
        assert_stored_event(&events[0], &[5, 6], Some(1));
        assert_active(
            &mgr,
            &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
        );

        // Leaf-to-root release order is what makes the resulting inactive set
        // deterministic under the Lineage backend.
        deref_blocks(&mut mgr, &[4, 3, 2, 1, 0]);
        let events = drain_events(&sink);
        assert!(events.is_empty(), "Deref should not emit KV events");
        assert_active(&mgr, &[(0, 1), (1, 1), (5, 1), (6, 1)]);
        assert_inactive_blocks(&mgr, &[2, 3, 4]);

        // Release the second branch leaf-to-root too. Active drains; inactive = {0..=6}.
        deref_blocks(&mut mgr, &[6, 5, 1, 0]);
        let events = drain_events(&sink);
        assert!(events.is_empty(), "Deref should not emit KV events");
        assert_active(&mgr, &[]);
        assert_inactive_blocks(&mgr, &[0, 1, 2, 3, 4, 5, 6]);

        // Re-use 0, 1, 2 (reactivates from inactive) + 7, 8, 9 (new, 3 free
        // slots). No eviction needed — inactive shrinks to {3, 4, 5, 6}.
        assert_eq!(use_blocks(&mut mgr, &[0, 1, 2, 7, 8, 9]), 6);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 1, "expected one Stored event for [7, 8, 9]");
        assert_stored_event(&events[0], &[7, 8, 9], Some(2));
        assert_active(&mgr, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]);
        assert_inactive_blocks(&mgr, &[3, 4, 5, 6]);

        // Capacity pressure now forces exact leaf-first evictions: 4, then 3,
        // then 6. The sole inactive survivor is 5.
        assert_eq!(use_blocks(&mut mgr, &[10, 11, 12]), 3);
        let events = drain_events(&sink);
        assert_eq!(
            events.len(),
            2,
            "expected Stored + Removed for [10, 11, 12]"
        );
        assert_stored_event(&events[0], &[10, 11, 12], None);
        assert_removed_event(&events[1], &[4, 3, 6]);
        assert_active(
            &mgr,
            &[
                (0, 1),
                (1, 1),
                (2, 1),
                (7, 1),
                (8, 1),
                (9, 1),
                (10, 1),
                (11, 1),
                (12, 1),
            ],
        );
        assert_inactive_blocks(&mgr, &[5]);

        assert_eq!(use_blocks(&mut mgr, &[13]), 1);
        let events = drain_events(&sink);
        assert_eq!(events.len(), 2, "expected Stored + Removed for [13]");
        assert_stored_event(&events[0], &[13], None);
        assert_removed_event(&events[1], &[5]);
        assert_active(
            &mgr,
            &[
                (0, 1),
                (1, 1),
                (2, 1),
                (7, 1),
                (8, 1),
                (9, 1),
                (10, 1),
                (11, 1),
                (12, 1),
                (13, 1),
            ],
        );
        assert_eq!(mgr.num_inactive_blocks(), 0);
    }

    #[test]
    fn test_chunked_prefill_parent_hash() {
        let block_size = 64;
        let tokens: Vec<u32> = (0..512).collect(); // 8 full blocks
        let mut seq = ActiveSequence::new(tokens, 100, Some(block_size), true, false);

        let (mut mgr, sink) = make_mgr_capturing(256, block_size);

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

    /// When a FullBlock is used, deref'd (becomes inactive in kvbm-logical),
    /// then used again, the router already knows about it — reactivation must
    /// NOT emit a second `Stored` event.
    #[test]
    fn test_inactive_hit_does_not_republish_stored() {
        let (mut mgr, sink) = make_mgr_capturing(4, 16);

        // First Use: fresh registration → 1 Stored.
        use_full(&mut mgr, 1, plh(100));
        // Deref → block transitions to inactive pool. No Removed (we don't
        // emit one on Deref).
        deref_full(&mut mgr, 1);
        // Second Use: match_blocks reactivates from inactive → InactiveHit.
        // No new Stored should fire.
        use_full(&mut mgr, 1, plh(100));

        let events = sink.events.lock().unwrap();
        let stored_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
            .count();
        let removed_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Removed(_)))
            .count();
        assert_eq!(stored_count, 1, "reactivation must not re-emit Stored");
        assert_eq!(removed_count, 0, "Deref must not emit Removed");
    }

    /// After reusing a prefix [A, B] and storing a new suffix [C], the
    /// `Stored` event for C must anchor `parent_hash` to B (the last reused
    /// full block), not to whatever parent the caller originally passed.
    #[test]
    fn test_stored_suffix_anchors_to_last_reused_block() {
        let (mut mgr, sink) = make_mgr_capturing(8, 16);

        // Prime the cache with [A=10, B=11].
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(10), UniqueBlock::FullBlock(11)],
            vec![],
            vec![plh(10), plh(11)],
            None,
            None,
        ));
        // Drop both to inactive.
        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 11);

        // Clear captured events from priming.
        sink.events.lock().unwrap().clear();

        // New request reuses [A, B] and stores a new block C=12.
        mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(10),
                UniqueBlock::FullBlock(11),
                UniqueBlock::FullBlock(12),
            ],
            vec![],
            vec![plh(10), plh(11), plh(12)],
            None,
            None, // no explicit parent → scheduler would pass None for a head-chunk
        ));

        let events = sink.events.lock().unwrap();
        // Only one Stored (for C); no Stored for reused A or B.
        assert_eq!(events.len(), 1, "only new suffix should fire a Stored");
        let KvCacheEventData::Stored(ref data) = events[0].data else {
            panic!("expected Stored");
        };
        assert_eq!(data.blocks.len(), 1, "Stored must only include C");
        assert_eq!(data.blocks[0].block_hash, ExternalSequenceBlockHash(12));
        assert_eq!(
            data.parent_hash,
            Some(ExternalSequenceBlockHash(11)),
            "parent_hash must anchor to last reused full block (B=11)"
        );
    }

    /// Two requests sharing a prefix must not inflate scheduler-visible
    /// occupancy. The distinct count reflects physically-resident blocks; the
    /// refcount metric reflects held handles.
    #[test]
    fn test_shared_prefix_distinct_vs_refcount() {
        let mut mgr = make_mgr(8, 16);

        // Request A uses [10, 11, 12].
        mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(10),
                UniqueBlock::FullBlock(11),
                UniqueBlock::FullBlock(12),
            ],
            vec![],
            vec![plh(10), plh(11), plh(12)],
            None,
            None,
        ));
        assert_eq!(mgr.num_active_blocks(), 3);
        assert_eq!(mgr.num_active_block_refs(), 3);

        // Request B reuses prefix [10, 11] and adds its own block [13].
        mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(10),
                UniqueBlock::FullBlock(11),
                UniqueBlock::FullBlock(13),
            ],
            vec![],
            vec![plh(10), plh(11), plh(13)],
            None,
            None,
        ));

        // Distinct resident blocks: {10, 11, 12, 13} = 4 (scheduler view).
        assert_eq!(
            mgr.num_active_blocks(),
            4,
            "shared prefix must not inflate distinct count"
        );
        // Handle count: 10 and 11 each held twice, 12 once, 13 once → 6.
        assert_eq!(
            mgr.num_active_block_refs(),
            6,
            "handle count should reflect per-request refcount"
        );
    }

    /// With `enable_prefix_caching=false`, each sequence should still be able
    /// to reactivate its OWN inactive blocks after preemption and re-admit.
    #[test]
    fn test_random_plh_stable_across_preempt_retry() {
        // 4 blocks of size 16 → 64 tokens of prompt.
        let block_size = 16;
        let tokens: Vec<u32> = (0..64).collect();
        let mut seq = ActiveSequence::new(tokens, 100, Some(block_size), false, false);

        let (mut mgr, sink) = make_mgr_capturing(8, block_size);

        // Admit: allocate prompt blocks.
        let signal = seq.take_creation_signal().unwrap();
        assert_eq!(mgr.process(&signal), 4);
        assert_eq!(mgr.num_active_blocks(), 4);

        // Preempt: reset_with_signal frees all active blocks (Deref) →
        // kvbm-logical keeps them in the inactive pool (no Removed events).
        let reset_signals = seq.reset_with_signal();
        for signal in &reset_signals {
            mgr.process(signal);
        }
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_inactive_blocks(), 4);

        // Re-admit: prompt blocks must reactivate via InactiveHit, NOT allocate
        // fresh. The cached per-sequence PLHs are what make this work.
        let signal = seq.take_creation_signal().unwrap();
        assert_eq!(mgr.process(&signal), 4);
        assert_eq!(mgr.num_active_blocks(), 4);
        assert_eq!(mgr.num_inactive_blocks(), 0);

        // Router-event witness: only ONE `Stored` (from the original admit).
        let events = sink.events.lock().unwrap();
        let stored_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
            .count();
        assert_eq!(
            stored_count, 1,
            "preempted request should self-match on re-admit (no duplicate Stored)"
        );
    }

    #[test]
    fn test_eviction_emits_exact_removed_event() {
        // Capacity = 2. Use three blocks (10, 11, 12); deref 10, 11 to push
        // them into the inactive pool; then use a third distinct block (12)
        // that isn't already in the active or inactive pool — this forces
        // allocation → inactive-pool eviction.
        let (mut mgr, sink) = make_mgr_capturing(2, 16);

        // Seed 10 and 11 in the inactive pool.
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(10), UniqueBlock::FullBlock(11)],
            vec![],
            vec![plh(10), plh(11)],
            None,
            None,
        ));
        deref_full(&mut mgr, 10);
        deref_full(&mut mgr, 11);
        assert_eq!(mgr.num_active_blocks(), 0);
        assert_eq!(mgr.num_inactive_blocks(), 2);

        sink.events.lock().unwrap().clear();

        // Introduce block 12 → must evict exactly one of {10, 11}.
        use_full(&mut mgr, 12, plh(12));

        let events = sink.events.lock().unwrap();
        let removed: Vec<u64> = events
            .iter()
            .filter_map(|e| match &e.data {
                KvCacheEventData::Removed(data) => Some(
                    data.block_hashes
                        .iter()
                        .map(|ExternalSequenceBlockHash(h)| *h)
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .flatten()
            .collect();
        let stored_count = events
            .iter()
            .filter(|e| matches!(e.data, KvCacheEventData::Stored(_)))
            .count();

        assert_eq!(
            removed.len(),
            1,
            "exactly one block should be reported as evicted"
        );
        assert!(
            removed[0] == 10 || removed[0] == 11,
            "evicted hash must be one we seeded ({}), got {}",
            "10 or 11",
            removed[0]
        );
        assert_eq!(stored_count, 1, "one Stored event for the fresh block 12");
    }
}
