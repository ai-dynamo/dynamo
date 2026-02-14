// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KvbmLogical backend for the mocker's KV manager.
//!
//! Wraps kvbm-logical's `BlockManager<MockMeta>` and translates `MoveBlock` signals
//! into the RAII block lifecycle (allocate → stage → register → drop).

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_tokens::PositionalLineageHash;
use dynamo_tokens::blocks::UniqueBlock;
use kvbm_logical::registry::BlockRegistry;
use kvbm_logical::tinylfu::TinyLFUTracker;
use kvbm_logical::{BlockManager, ImmutableBlock, MutableBlock};
use uuid::Uuid;

use crate::kv_manager::KvBackend;
use crate::protocols::{KvCacheEventSink, MockMeta, MockerEvictionBackend, MoveBlock};

/// KV manager backend powered by kvbm-logical's production `BlockManager`.
///
/// Translates the mocker's `MoveBlock` signal protocol into kvbm-logical's
/// RAII block lifecycle: allocate → stage → register → drop.
pub struct KvbmLogicalKvManager {
    block_manager: BlockManager<MockMeta>,
    max_capacity: usize,
    block_size: usize,
    _dp_rank: u32,
    // Kept for future EventsManager integration
    _kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,

    /// Partial (generation) blocks held as MutableBlock — not yet registered.
    active_partial: HashMap<Uuid, MutableBlock<MockMeta>>,

    /// Full blocks held as ImmutableBlock, keyed by SequenceHash.
    /// Vec tracks reference count — each Use adds one ImmutableBlock clone.
    active_full: HashMap<u64, Vec<ImmutableBlock<MockMeta>>>,
}

impl KvbmLogicalKvManager {
    pub fn new(
        max_capacity: usize,
        block_size: usize,
        dp_rank: u32,
        kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
        eviction_backend: MockerEvictionBackend,
    ) -> Self {
        let mut registry_builder = BlockRegistry::builder();
        if matches!(eviction_backend, MockerEvictionBackend::MultiLru) {
            let tracker = Arc::new(TinyLFUTracker::new(max_capacity));
            registry_builder = registry_builder.frequency_tracker(tracker);
        }
        let registry = registry_builder.build();
        let mut builder = BlockManager::builder()
            .block_count(max_capacity)
            .block_size(block_size)
            .registry(registry);

        builder = match eviction_backend {
            MockerEvictionBackend::Lineage => builder.with_lineage_backend(),
            MockerEvictionBackend::Lru => builder.with_lru_backend(),
            MockerEvictionBackend::MultiLru => builder.with_multi_lru_backend(),
        };

        let block_manager = builder.build().expect("BlockManager build failed");

        tracing::info!(
            "KvbmLogicalKvManager initialized for DP rank {dp_rank} with block_size {block_size}, \
             eviction={eviction_backend:?}"
        );

        Self {
            block_manager,
            max_capacity,
            block_size,
            _dp_rank: dp_rank,
            _kv_event_sink: kv_event_sink,
            active_partial: HashMap::new(),
            active_full: HashMap::new(),
        }
    }

    fn process_use(&mut self, blocks: &[UniqueBlock], plhs: &[PositionalLineageHash]) -> bool {
        // PLH vec may be shorter than blocks (partial blocks have no PLH)
        for (i, block) in blocks.iter().enumerate() {
            match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    // First check if we already hold this block actively
                    if let Some(vec) = self.active_full.get(seq_hash) {
                        // Clone an existing ImmutableBlock to bump the ref count
                        let cloned = vec[0].clone();
                        self.active_full.get_mut(seq_hash).unwrap().push(cloned);
                        continue;
                    }

                    // Try match_blocks from the BlockManager (active + inactive pools)
                    let plh = plhs.get(i).copied().unwrap_or_default();
                    let matched = self.block_manager.match_blocks(&[plh]);
                    if let Some(immutable) = matched.into_iter().next() {
                        self.active_full
                            .entry(*seq_hash)
                            .or_default()
                            .push(immutable);
                        continue;
                    }

                    // Allocate a new block, stage it, register it
                    let Some(mut allocated) = self.block_manager.allocate_blocks(1) else {
                        return false;
                    };
                    let mutable = allocated.pop().unwrap();
                    let complete = mutable.stage(plh, self.block_size).expect("stage failed");
                    let immutable = self.block_manager.register_block(complete);
                    self.active_full
                        .entry(*seq_hash)
                        .or_default()
                        .push(immutable);
                }
                UniqueBlock::PartialBlock(uuid) => {
                    // Allocate a MutableBlock — held until promoted or destroyed
                    let Some(mut allocated) = self.block_manager.allocate_blocks(1) else {
                        return false;
                    };
                    let mutable = allocated.pop().unwrap();
                    self.active_partial.insert(*uuid, mutable);
                }
            }
        }
        true
    }

    fn process_destroy(&mut self, blocks: &[UniqueBlock]) {
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(uuid) => {
                    // Drop MutableBlock → RAII returns to reset pool
                    self.active_partial.remove(uuid);
                }
                UniqueBlock::FullBlock(seq_hash) => {
                    // Drop all ImmutableBlocks → RAII returns to inactive pool
                    self.active_full.remove(seq_hash);
                }
            }
        }
    }

    fn process_deref(&mut self, blocks: &[UniqueBlock]) {
        for block in blocks {
            match block {
                UniqueBlock::PartialBlock(_) => {
                    panic!("Deref on PartialBlock is not valid");
                }
                UniqueBlock::FullBlock(seq_hash) => {
                    if let Some(vec) = self.active_full.get_mut(seq_hash) {
                        // Pop one ImmutableBlock — its drop moves it to inactive
                        vec.pop();
                        if vec.is_empty() {
                            self.active_full.remove(seq_hash);
                        }
                    }
                }
            }
        }
    }

    fn process_promote(&mut self, uuid: Uuid, seq_hash: u64, plh: PositionalLineageHash) {
        let mutable = self
            .active_partial
            .remove(&uuid)
            .expect("Promote: partial block not found");

        let complete = mutable
            .stage(plh, self.block_size)
            .expect("stage failed during promote");
        let immutable = self.block_manager.register_block(complete);

        // If this seq_hash already exists (hash collision with another block),
        // add to the Vec; otherwise create new entry
        self.active_full
            .entry(seq_hash)
            .or_default()
            .push(immutable);
    }
}

impl KvBackend for KvbmLogicalKvManager {
    fn process(&mut self, event: &MoveBlock) -> bool {
        match event {
            MoveBlock::Use(blocks, _local_hashes, plhs) => self.process_use(blocks, plhs),
            MoveBlock::Destroy(blocks) => {
                self.process_destroy(blocks);
                true
            }
            MoveBlock::Deref(blocks) => {
                self.process_deref(blocks);
                true
            }
            MoveBlock::Promote(uuid, seq_hash, _parent_hash, _local_hash, plh) => {
                self.process_promote(*uuid, *seq_hash, *plh);
                true
            }
        }
    }

    fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn num_active_blocks(&self) -> usize {
        self.active_partial.len() + self.active_full.values().map(|v| v.len()).sum::<usize>()
    }

    fn num_inactive_blocks(&self) -> usize {
        let total = self.block_manager.total_blocks();
        let available = self.block_manager.available_blocks();
        let in_use = total - available;
        in_use.saturating_sub(self.num_active_blocks())
    }

    fn current_capacity(&self) -> usize {
        self.block_manager.total_blocks() - self.block_manager.available_blocks()
    }

    fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize {
        blocks
            .iter()
            .filter(|block| match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    // Not in our active set
                    !self.active_full.contains_key(seq_hash)
                    // For a complete probe, we'd also check the block_manager's pools,
                    // but without the PLH we can't do match_blocks. For the mocker's
                    // scheduling purposes, checking active_full is sufficient.
                }
                UniqueBlock::PartialBlock(uuid) => !self.active_partial.contains_key(uuid),
            })
            .count()
    }

    fn is_block_cached(&self, seq_hash: u64, plh: Option<PositionalLineageHash>) -> bool {
        // Check active set first
        if self.active_full.contains_key(&seq_hash) {
            return true;
        }
        // Check block_manager registry for cached blocks (read-only)
        if let Some(plh) = plh {
            let presence = self
                .block_manager
                .block_registry()
                .check_presence::<MockMeta>(&[plh]);
            if presence.first().is_some_and(|(_, present)| *present) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::ActiveSequence;
    use kvbm_logical::metrics::MetricsSnapshot;

    fn make_manager(capacity: usize, block_size: usize) -> KvbmLogicalKvManager {
        KvbmLogicalKvManager::new(
            capacity,
            block_size,
            0,
            None,
            MockerEvictionBackend::Lineage,
        )
    }

    fn plh(val: u64) -> PositionalLineageHash {
        PositionalLineageHash::new(val, None, 0)
    }

    fn snap(mgr: &KvbmLogicalKvManager) -> MetricsSnapshot {
        mgr.block_manager.metrics().snapshot()
    }

    fn use_full(
        mgr: &mut KvbmLogicalKvManager,
        seq_hash: u64,
        plh_val: PositionalLineageHash,
    ) -> bool {
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(seq_hash)],
            vec![],
            vec![plh_val],
        ))
    }

    fn use_partial(mgr: &mut KvbmLogicalKvManager, uuid: Uuid) -> bool {
        mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::PartialBlock(uuid)],
            vec![],
            vec![],
        ))
    }

    fn deref_full(mgr: &mut KvbmLogicalKvManager, seq_hash: u64) {
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::FullBlock(seq_hash)]));
    }

    fn destroy_full(mgr: &mut KvbmLogicalKvManager, seq_hash: u64) {
        mgr.process(&MoveBlock::Destroy(vec![UniqueBlock::FullBlock(seq_hash)]));
    }

    fn destroy_partial(mgr: &mut KvbmLogicalKvManager, uuid: Uuid) {
        mgr.process(&MoveBlock::Destroy(vec![UniqueBlock::PartialBlock(uuid)]));
    }

    // ========================================================================
    // 1. Basic allocation & capacity tracking
    // ========================================================================

    #[test]
    fn test_use_single_full_block() {
        let mut mgr = make_manager(10, 16);
        let p = plh(100);

        assert!(use_full(&mut mgr, 1, p));

        assert_eq!(mgr.num_active_blocks(), 1);
        assert_eq!(mgr.current_capacity(), 1);

        let s = snap(&mgr);
        assert_eq!(s.allocations, 1);
        assert_eq!(s.allocations_from_reset, 1);
        assert_eq!(s.registrations, 1);
        assert_eq!(s.stagings, 1);
        assert_eq!(s.inflight_immutable, 1);
        assert_eq!(s.inflight_mutable, 0);
    }

    #[test]
    fn test_use_single_partial_block() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();

        assert!(use_partial(&mut mgr, uuid));

        assert_eq!(mgr.num_active_blocks(), 1);

        let s = snap(&mgr);
        assert_eq!(s.allocations, 1);
        assert_eq!(s.inflight_mutable, 1);
        assert_eq!(s.registrations, 0);
        assert_eq!(s.inflight_immutable, 0);
    }

    #[test]
    fn test_use_batch_full_and_partial() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();
        let p1 = plh(100);
        let p2 = plh(200);

        let result = mgr.process(&MoveBlock::Use(
            vec![
                UniqueBlock::FullBlock(1),
                UniqueBlock::FullBlock(2),
                UniqueBlock::PartialBlock(uuid),
            ],
            vec![],
            vec![p1, p2],
        ));
        assert!(result);

        assert_eq!(mgr.num_active_blocks(), 3);

        let s = snap(&mgr);
        assert_eq!(s.allocations, 3);
        assert_eq!(s.registrations, 2);
        assert_eq!(s.inflight_mutable, 1);
        assert_eq!(s.inflight_immutable, 2);
    }

    #[test]
    fn test_accessors() {
        let mgr = make_manager(42, 64);
        assert_eq!(mgr.max_capacity(), 42);
        assert_eq!(mgr.block_size(), 64);
    }

    // ========================================================================
    // 2. Destroy & Deref with metrics validation
    // ========================================================================

    #[test]
    fn test_destroy_partial_block() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();

        use_partial(&mut mgr, uuid);
        assert_eq!(mgr.num_active_blocks(), 1);
        let available_before = mgr.block_manager.available_blocks();

        destroy_partial(&mut mgr, uuid);

        assert_eq!(mgr.num_active_blocks(), 0);

        let s = snap(&mgr);
        assert_eq!(s.inflight_mutable, 0);
        // MutableBlock drop returns to reset pool, increasing available blocks
        assert!(
            mgr.block_manager.available_blocks() > available_before,
            "MutableBlock drop should return to reset pool"
        );
    }

    #[test]
    fn test_destroy_full_block() {
        let mut mgr = make_manager(10, 16);
        let p = plh(100);

        use_full(&mut mgr, 1, p);
        assert_eq!(mgr.num_active_blocks(), 1);
        let available_before = mgr.block_manager.available_blocks();

        destroy_full(&mut mgr, 1);

        assert_eq!(mgr.num_active_blocks(), 0);

        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 0);
        // ImmutableBlock RAII drop returns the block to a pool
        assert!(
            mgr.block_manager.available_blocks() > available_before,
            "destroyed block should return to a pool"
        );
    }

    #[test]
    fn test_deref_full_block() {
        let mut mgr = make_manager(10, 16);
        let p = plh(100);

        use_full(&mut mgr, 1, p);
        assert_eq!(mgr.num_active_blocks(), 1);
        let available_before = mgr.block_manager.available_blocks();

        deref_full(&mut mgr, 1);

        assert_eq!(mgr.num_active_blocks(), 0);
        // Block is no longer active — it returned to a pool via RAII
        assert!(
            mgr.block_manager.available_blocks() > available_before,
            "deref'd block should return to a pool"
        );

        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 0);
    }

    #[test]
    fn test_deref_with_multiple_refs() {
        let mut mgr = make_manager(10, 16);
        let p = plh(100);

        // First Use: allocate + register
        use_full(&mut mgr, 1, p);
        // Second Use: duplicate → clone, no new allocation
        use_full(&mut mgr, 1, p);

        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 2);
        assert_eq!(s.allocations, 1, "second Use should clone, not allocate");

        // First Deref — drops one clone but block stays active (still 1 ref)
        deref_full(&mut mgr, 1);
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 1);
        assert_eq!(mgr.num_active_blocks(), 1);

        // Second Deref — last ref, block returns to pool via RAII
        let available_before = mgr.block_manager.available_blocks();
        deref_full(&mut mgr, 1);
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 0);
        assert_eq!(mgr.num_active_blocks(), 0);
        assert!(
            mgr.block_manager.available_blocks() > available_before,
            "last deref should return block to pool"
        );
    }

    #[test]
    #[should_panic(expected = "Deref on PartialBlock is not valid")]
    fn test_deref_on_partial_panics() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();
        use_partial(&mut mgr, uuid);
        mgr.process(&MoveBlock::Deref(vec![UniqueBlock::PartialBlock(uuid)]));
    }

    // ========================================================================
    // 3. Reference counting (duplicate Use)
    // ========================================================================

    #[test]
    fn test_duplicate_use_bumps_refcount() {
        let mut mgr = make_manager(10, 16);
        let p = plh(100);

        use_full(&mut mgr, 1, p);
        use_full(&mut mgr, 1, p);

        assert_eq!(mgr.num_active_blocks(), 2);

        let s = snap(&mgr);
        assert_eq!(s.allocations, 1, "only one real allocation");
        assert_eq!(s.registrations, 1, "only one registration");
    }

    // ========================================================================
    // 4. Promote (partial → full lifecycle)
    // ========================================================================

    #[test]
    fn test_promote_basic() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();
        let p = plh(500);

        use_partial(&mut mgr, uuid);
        let s = snap(&mgr);
        assert_eq!(s.inflight_mutable, 1);
        assert_eq!(s.inflight_immutable, 0);

        mgr.process(&MoveBlock::Promote(uuid, 42, None, 0, p));

        assert_eq!(mgr.num_active_blocks(), 1);
        assert!(mgr.active_partial.is_empty());
        assert!(mgr.active_full.contains_key(&42));

        let s = snap(&mgr);
        assert_eq!(s.inflight_mutable, 0);
        assert_eq!(s.inflight_immutable, 1);
        assert_eq!(s.registrations, 1);
        assert_eq!(s.stagings, 1);
    }

    #[test]
    fn test_promote_then_deref() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();
        let p = plh(500);

        use_partial(&mut mgr, uuid);
        let available_before = mgr.block_manager.available_blocks();

        mgr.process(&MoveBlock::Promote(uuid, 42, None, 0, p));
        deref_full(&mut mgr, 42);

        assert_eq!(mgr.num_active_blocks(), 0);
        // Block returned to pool via RAII
        assert!(
            mgr.block_manager.available_blocks() > available_before,
            "deref'd promoted block should return to pool"
        );

        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 0);
    }

    #[test]
    #[should_panic(expected = "Promote: partial block not found")]
    fn test_promote_nonexistent_panics() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();
        let p = plh(500);
        mgr.process(&MoveBlock::Promote(uuid, 42, None, 0, p));
    }

    // ========================================================================
    // 5. Inactive pool reuse (match_blocks / prefix caching)
    // ========================================================================

    #[test]
    fn test_inactive_reuse_via_match_blocks() {
        let mut mgr = make_manager(10, 16);
        let p = plh(100);

        // Use and deref block 1 → returns to pool with PLH=p
        use_full(&mut mgr, 1, p);
        deref_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 0);

        let s_before = snap(&mgr);
        let alloc_before = s_before.allocations;

        // Use block 2 with the same PLH → should match from pool (active PRT or inactive)
        use_full(&mut mgr, 2, p);

        let s = snap(&mgr);
        // Block should be reused: either via match_blocks or from PRT upgrade.
        // Either way, no NEW allocation is needed.
        assert_eq!(
            s.allocations, alloc_before,
            "no new allocation needed — block reused"
        );
    }

    #[test]
    fn test_no_match_allocates_new() {
        let mut mgr = make_manager(10, 16);
        let p1 = plh(100);
        let p2 = plh(200);

        // Use and deref block 1 with PLH=p1
        use_full(&mut mgr, 1, p1);
        deref_full(&mut mgr, 1);

        // Use block 2 with different PLH=p2 → no match, new allocation
        use_full(&mut mgr, 2, p2);

        let s = snap(&mgr);
        assert_eq!(s.allocations, 2, "second Use should require new allocation");
    }

    // ========================================================================
    // 6. Capacity exhaustion & eviction
    // ========================================================================

    #[test]
    fn test_capacity_exhaustion_returns_false() {
        let mut mgr = make_manager(4, 16);

        for i in 0..4 {
            assert!(use_full(&mut mgr, i, plh(i + 100)));
        }

        // 5th block should fail
        let result = use_full(&mut mgr, 4, plh(500));
        assert!(!result, "should fail at capacity");

        let s = snap(&mgr);
        assert_eq!(s.allocations, 4);
    }

    #[test]
    fn test_eviction_frees_inactive_for_new_allocation() {
        let mut mgr = make_manager(4, 16);

        // Fill capacity with 4 blocks
        for i in 0..4 {
            use_full(&mut mgr, i, plh(i + 100));
        }

        // Deref all → blocks return to pools via RAII
        for i in 0..4 {
            deref_full(&mut mgr, i);
        }
        assert_eq!(mgr.num_active_blocks(), 0);

        // Use 4 new blocks with different PLHs → must reclaim pool blocks
        for i in 10..14 {
            assert!(use_full(&mut mgr, i, plh(i + 1000)));
        }

        let s = snap(&mgr);
        assert_eq!(s.allocations, 8);
        assert_eq!(mgr.num_active_blocks(), 4);
    }

    #[test]
    fn test_partial_allocation_failure() {
        let mut mgr = make_manager(4, 16);

        // Fill all 4 slots with full blocks
        for i in 0..4 {
            use_full(&mut mgr, i, plh(i + 100));
        }

        // Partial block allocation should fail — no room
        let uuid = Uuid::new_v4();
        let result = use_partial(&mut mgr, uuid);
        assert!(!result, "should fail when capacity is exhausted");
    }

    // ========================================================================
    // 7. probe_new_blocks
    // ========================================================================

    #[test]
    fn test_probe_all_new() {
        let mgr = make_manager(10, 16);
        let blocks = vec![UniqueBlock::FullBlock(1), UniqueBlock::FullBlock(2)];
        assert_eq!(mgr.probe_new_blocks(&blocks), 2);
    }

    #[test]
    fn test_probe_some_existing() {
        let mut mgr = make_manager(10, 16);
        use_full(&mut mgr, 1, plh(100));

        let blocks = vec![UniqueBlock::FullBlock(1), UniqueBlock::FullBlock(2)];
        assert_eq!(mgr.probe_new_blocks(&blocks), 1);
    }

    #[test]
    fn test_probe_partial_existing() {
        let mut mgr = make_manager(10, 16);
        let uuid_existing = Uuid::new_v4();
        let uuid_new = Uuid::new_v4();
        use_partial(&mut mgr, uuid_existing);

        let blocks = vec![
            UniqueBlock::PartialBlock(uuid_existing),
            UniqueBlock::PartialBlock(uuid_new),
        ];
        assert_eq!(mgr.probe_new_blocks(&blocks), 1);
    }

    // ========================================================================
    // 8. get_prefill_cost
    // ========================================================================

    #[test]
    fn test_prefill_cost_no_overlap() {
        let mgr = make_manager(10, 16);
        // 35 tokens with block_size=16 → 2 full blocks + 1 partial (3 tokens left)
        let tokens: Vec<u32> = (0..35).collect();
        let seq = ActiveSequence::new(tokens, 10, Some(16), true);

        let cost = mgr.get_prefill_cost(&seq);
        assert_eq!(cost.new_blocks, seq.unique_blocks().len());
        assert_eq!(cost.new_tokens, 35);
    }

    #[test]
    fn test_prefill_cost_full_overlap() {
        let mut mgr = make_manager(10, 16);
        // 35 tokens → 2 full blocks + 1 partial
        let tokens: Vec<u32> = (0..35).collect();
        let seq = ActiveSequence::new(tokens, 10, Some(16), true);

        // Use all full blocks (the first N that are FullBlock)
        for block in seq.unique_blocks() {
            if let UniqueBlock::FullBlock(h) = block {
                let plh_val = seq.positional_lineage_hashes();
                let idx = seq.unique_blocks().iter().position(|b| b == block).unwrap();
                if let Some(p) = plh_val.get(idx) {
                    use_full(&mut mgr, *h, *p);
                }
            }
        }

        let cost = mgr.get_prefill_cost(&seq);
        // Only the partial block is new
        assert_eq!(cost.new_blocks, 1);
        // 2 full blocks cached = 32 tokens cached, 3 new tokens
        assert_eq!(cost.new_tokens, 35 - 32);
    }

    #[test]
    fn test_prefill_cost_partial_overlap() {
        let mut mgr = make_manager(10, 16);
        // 35 tokens → 2 full blocks + 1 partial
        let tokens: Vec<u32> = (0..35).collect();
        let seq = ActiveSequence::new(tokens, 10, Some(16), true);

        // Use only the first full block
        if let UniqueBlock::FullBlock(h) = &seq.unique_blocks()[0] {
            let p = seq.positional_lineage_hashes()[0];
            use_full(&mut mgr, *h, p);
        }

        let cost = mgr.get_prefill_cost(&seq);
        // First block cached → new_blocks = total - 1
        assert_eq!(cost.new_blocks, seq.unique_blocks().len() - 1);
        // 1 full block cached = 16 tokens cached
        assert_eq!(cost.new_tokens, 35 - 16);
    }

    // ========================================================================
    // 9. Capacity percentage accessors
    // ========================================================================

    #[test]
    fn test_current_capacity_perc() {
        let mut mgr = make_manager(10, 16);
        for i in 0..5 {
            use_full(&mut mgr, i, plh(i + 100));
        }
        let perc = mgr.current_capacity_perc();
        assert!((perc - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_active_perc() {
        let mut mgr = make_manager(10, 16);
        for i in 0..3 {
            use_full(&mut mgr, i, plh(i + 100));
        }
        let perc = mgr.get_active_perc();
        assert!((perc - 0.3).abs() < f64::EPSILON);
    }

    // ========================================================================
    // 10. Full lifecycle (mirrors Manual's test_block_lifecycle_stringent)
    // ========================================================================

    #[test]
    fn test_full_lifecycle() {
        let mut mgr = make_manager(10, 16);

        // Step 1: Use blocks 0..5
        for i in 0u64..5 {
            assert!(use_full(&mut mgr, i, plh(i + 100)));
        }
        assert_eq!(mgr.num_active_blocks(), 5);
        let s = snap(&mgr);
        assert_eq!(s.allocations, 5);
        assert_eq!(s.registrations, 5);
        assert_eq!(s.inflight_immutable, 5);

        // Step 2: Use blocks 0, 1, 5, 6 (0 and 1 are duplicates → refcount bump)
        for &i in &[0u64, 1, 5, 6] {
            assert!(use_full(&mut mgr, i, plh(i + 100)));
        }
        assert_eq!(mgr.num_active_blocks(), 9); // 5 + 2 dup + 2 new
        let s = snap(&mgr);
        assert_eq!(s.allocations, 7); // 5 original + 2 new (5, 6)
        assert_eq!(s.inflight_immutable, 9);

        // Step 3: Destroy block 4
        destroy_full(&mut mgr, 4);
        assert_eq!(mgr.num_active_blocks(), 8);
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 8);

        // Step 4: Deref blocks 0, 1, 2, 3 (each once — drops refcount)
        for &i in &[0u64, 1, 2, 3] {
            deref_full(&mut mgr, i);
        }
        // 0 and 1 still have 1 ref each; 2 and 3 drop to zero
        assert_eq!(mgr.num_active_blocks(), 4); // 0(1), 1(1), 5(1), 6(1)
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 4);

        // Step 5: Destroy block 6
        destroy_full(&mut mgr, 6);
        // Step 6: Deref blocks 0, 1, 5
        for &i in &[0u64, 1, 5] {
            deref_full(&mut mgr, i);
        }
        assert_eq!(mgr.num_active_blocks(), 0);
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 0);
        // All blocks returned to pools — available should equal total
        assert_eq!(
            mgr.block_manager.available_blocks(),
            mgr.block_manager.total_blocks(),
            "all blocks should be available after full release"
        );

        // Step 7: Use 7, 8, 9 and 0, 1, 2 (some may reuse from pools)
        for &i in &[7u64, 8, 9] {
            assert!(use_full(&mut mgr, i, plh(i + 1000)));
        }
        for &i in &[0u64, 1, 2] {
            assert!(use_full(&mut mgr, i, plh(i + 100)));
        }
        assert_eq!(mgr.num_active_blocks(), 6);
        let s = snap(&mgr);
        assert!(
            s.allocations >= 7,
            "should have at least original allocations"
        );
    }

    // ========================================================================
    // 11. Eviction backend variants
    // ========================================================================

    #[test]
    fn test_eviction_backend_lru() {
        let mut mgr = KvbmLogicalKvManager::new(4, 16, 0, None, MockerEvictionBackend::Lru);

        // Fill capacity
        for i in 0..4 {
            assert!(use_full(&mut mgr, i, plh(i + 100)));
        }
        // Deref all → inactive
        for i in 0..4 {
            deref_full(&mut mgr, i);
        }

        // Use 4 new blocks → must evict inactive
        for i in 10..14 {
            assert!(use_full(&mut mgr, i, plh(i + 1000)));
        }

        let s = snap(&mgr);
        assert_eq!(s.evictions, 4);
        assert_eq!(s.allocations, 8);
    }

    #[test]
    fn test_eviction_backend_multi_lru() {
        let cap = 4;
        let mut mgr = KvbmLogicalKvManager::new(cap, 16, 0, None, MockerEvictionBackend::MultiLru);

        // Fill capacity
        for i in 0..4u64 {
            assert!(use_full(&mut mgr, i, plh(i + 100)));
        }
        // Deref all → inactive pool
        for i in 0..4u64 {
            deref_full(&mut mgr, i);
        }

        // Use 4 new blocks with different PLHs → must evict inactive
        for i in 10..14u64 {
            assert!(use_full(&mut mgr, i, plh(i + 1000)));
        }

        let s = snap(&mgr);
        assert_eq!(s.allocations, 8);
        assert_eq!(mgr.num_active_blocks(), 4);
    }

    // ========================================================================
    // 12. Mixed partial + full lifecycle
    // ========================================================================

    #[test]
    fn test_mixed_partial_full_lifecycle() {
        let mut mgr = make_manager(10, 16);
        let uuid = Uuid::new_v4();
        let p1 = plh(100);
        let p2 = plh(200);

        // Use Full(1) + Partial(uuid) in a batch
        let result = mgr.process(&MoveBlock::Use(
            vec![UniqueBlock::FullBlock(1), UniqueBlock::PartialBlock(uuid)],
            vec![],
            vec![p1],
        ));
        assert!(result);
        assert_eq!(mgr.num_active_blocks(), 2);

        let s = snap(&mgr);
        assert_eq!(s.inflight_mutable, 1);
        assert_eq!(s.inflight_immutable, 1);
        assert_eq!(s.allocations, 2);

        // Promote partial → Full(2)
        mgr.process(&MoveBlock::Promote(uuid, 2, None, 0, p2));

        assert_eq!(mgr.num_active_blocks(), 2);
        let s = snap(&mgr);
        assert_eq!(s.inflight_mutable, 0);
        assert_eq!(s.inflight_immutable, 2);
        assert_eq!(s.registrations, 2);
        assert_eq!(s.stagings, 2);

        // Deref Full(1) → returns to pool
        deref_full(&mut mgr, 1);
        assert_eq!(mgr.num_active_blocks(), 1);
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 1);

        // Deref Full(2) → returns to pool
        deref_full(&mut mgr, 2);
        assert_eq!(mgr.num_active_blocks(), 0);
        let s = snap(&mgr);
        assert_eq!(s.inflight_immutable, 0);
        // All blocks returned — available should equal total
        assert_eq!(
            mgr.block_manager.available_blocks(),
            mgr.block_manager.total_blocks(),
        );
    }
}
