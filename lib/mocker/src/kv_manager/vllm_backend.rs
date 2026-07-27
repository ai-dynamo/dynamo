// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM G1 manager over a minimal physical block-pool model.
//!
//! [`VllmKvManager`] is the coordinator. It owns one manager per KV cache group,
//! sizes every allocation as a single reservation against the shared pool, and
//! reconciles per-group prefix hits the way vLLM's `HybridKVCacheCoordinator`
//! does. Per-group request block tables live in the group managers; the pool
//! owns physical occupancy, duplicate copies, prefix pins, and LRU eviction.

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, SequenceHash};
use uuid::Uuid;

use crate::cache::vllm_block_pool::{
    BlockReservation, GroupedHash, KvCacheGroupId, ReserveOutcome, VllmBlockPool,
};
use crate::common::kv_cache_trace;
use crate::common::protocols::{KvEventPublishers, PrefillCost};
use crate::common::sequence::ActiveSequence;

use super::vllm_groups::{
    CacheHit, FullAttentionGroup, GroupAllocation, GroupManager, PendingStore, StoredBlock,
};

struct StoreGroup {
    parent_hash: Option<SequenceHash>,
    blocks: Vec<SequenceHash>,
    local_hashes: Option<Vec<BlockHash>>,
    token_ids: Option<Vec<Vec<u32>>>,
}

impl StoreGroup {
    fn from_block(block: StoredBlock) -> Self {
        let PendingStore {
            parent_hash,
            local_hash,
            token_ids,
        } = block.metadata;
        Self {
            parent_hash,
            blocks: vec![block.hash],
            local_hashes: local_hash.map(|hash| vec![hash]),
            token_ids: token_ids.map(|ids| vec![ids]),
        }
    }

    fn can_append(&self, block: &StoredBlock) -> bool {
        self.local_hashes.is_some() == block.metadata.local_hash.is_some()
            && self.token_ids.is_some() == block.metadata.token_ids.is_some()
    }

    fn push(&mut self, block: StoredBlock) {
        self.blocks.push(block.hash);
        if let (Some(hashes), Some(hash)) = (&mut self.local_hashes, block.metadata.local_hash) {
            hashes.push(hash);
        }
        if let (Some(token_ids), Some(ids)) = (&mut self.token_ids, block.metadata.token_ids) {
            token_ids.push(ids);
        }
    }
}
pub(crate) struct NativeDecodeBlockReservation {
    pool: BlockReservation,
}

impl NativeDecodeBlockReservation {
    pub(crate) fn len(&self) -> usize {
        self.pool.fresh_len()
    }
}

pub(crate) struct NativeDestinationReservation {
    request_id: Uuid,
    pool: BlockReservation,
    layout: Option<VllmBlockLayout>,
    /// Attention blocks the reservation covers, excluding other groups' share.
    attention_fresh: usize,
    /// Leading attention prefix pins, which precede other groups' pins in the
    /// reservation.
    attention_prefix: usize,
}

impl NativeDestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self, block_size: usize) -> usize {
        self.attention_fresh.saturating_mul(block_size)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pool.len()
    }
}

pub(super) enum VllmAcquire<T> {
    Ready(T),
    CapacityExhausted,
}

pub(super) struct VllmBlockLayout {
    blocks: Vec<UniqueBlock>,
    local_hashes: Vec<BlockHash>,
    token_ids: Option<Vec<Vec<u32>>>,
    parent: Option<UniqueBlock>,
}

impl VllmBlockLayout {
    pub(super) fn new(
        blocks: Vec<UniqueBlock>,
        local_hashes: Vec<BlockHash>,
        token_ids: Option<Vec<Vec<u32>>>,
        parent: Option<UniqueBlock>,
    ) -> Self {
        Self {
            blocks,
            local_hashes,
            token_ids,
            parent,
        }
    }
}

pub(crate) struct VllmKvManager {
    pool: VllmBlockPool,
    /// The model's KV cache groups, in the order this coordinator visits them:
    /// prefix pins are reserved and later activated in that order. Group ids are
    /// assigned when the groups are built, and every lookup goes by group kind,
    /// so nothing below `new` depends on the order.
    groups: Vec<GroupManager>,
    block_size: usize,
    enable_prefix_caching: bool,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,
}

impl VllmKvManager {
    pub(crate) fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        enable_prefix_caching: bool,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        if !kv_event_publishers.is_empty() {
            tracing::info!(dp_rank, block_size, "VllmKvManager initialized");
        }
        Self {
            pool: VllmBlockPool::new(max_capacity),
            groups: vec![GroupManager::FullAttention(FullAttentionGroup::new(
                KvCacheGroupId(0),
                block_size,
                enable_prefix_caching,
            ))],
            block_size,
            enable_prefix_caching,
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
        }
    }

    /// The main attention group, which every request has a dense block table in.
    ///
    /// vLLM's hybrid coordinator singles this group out the same way: it is the
    /// dense reference for cross-group cache-hit reconciliation, and the only
    /// group whose blocks the router's single namespace can describe.
    fn attention(&self) -> &FullAttentionGroup {
        self.groups
            .iter()
            .find_map(|group| group.as_full_attention())
            .expect("a model has a main attention group")
    }

    /// Disjoint borrows of the pool and the attention group, which the group
    /// needs together to move blocks in or out of the pool.
    fn attention_mut(&mut self) -> (&mut VllmBlockPool, &mut FullAttentionGroup) {
        let Self { pool, groups, .. } = self;
        let attention = groups
            .iter_mut()
            .find_map(|group| group.as_full_attention_mut())
            .expect("a model has a main attention group");
        (pool, attention)
    }

    fn attention_id(&self) -> KvCacheGroupId {
        self.attention().id()
    }

    /// Groups that only constrain the attention group's decisions, never seed
    /// them: they can reduce a cache hit and consume capacity, but they own no
    /// dense table and no router-visible block.
    fn other_groups(&self) -> impl Iterator<Item = &GroupManager> {
        self.groups
            .iter()
            .filter(|group| !group.is_full_attention())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn use_for_request(
        &mut self,
        request_id: Uuid,
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
        reusable_prefix_blocks: usize,
    ) -> VllmAcquire<usize> {
        let alloc = GroupAllocation {
            request_id,
            blocks,
            local_hashes,
            token_ids,
            parent,
            reusable_prefix_blocks,
            cache_fresh: false,
        };
        self.process_use(&alloc, None)
    }

    pub(super) fn deref_for_request(&mut self, request_id: Uuid, blocks: &[UniqueBlock]) {
        let (pool, attention) = self.attention_mut();
        attention.deref(pool, request_id, blocks);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn promote_for_request(
        &mut self,
        request_id: Uuid,
        uuid: Uuid,
        hash: SequenceHash,
        parent_hash: Option<SequenceHash>,
        local_hash: Option<BlockHash>,
        token_ids: Option<Vec<u32>>,
    ) {
        let materialize_store_events = self.materialize_store_events();
        let (pool, attention) = self.attention_mut();
        let stored = attention.promote(
            pool,
            request_id,
            uuid,
            hash,
            PendingStore {
                parent_hash,
                local_hash,
                token_ids,
            },
            materialize_store_events,
        );
        if let Some(stored) = stored {
            self.publish_store_sequence(vec![Some(stored)]);
        }
    }

    /// Publish full blocks completed by this scheduling decision.
    ///
    /// `computed_before` is a finalization watermark: every earlier complete
    /// block was either finalized by a prior decision or arrived as an already
    /// cache-visible prefix/destination block. Restricting the scan to this
    /// delta avoids revisiting the full request block table on every decode
    /// step.
    pub(crate) fn finalize_computed_prefix(
        &mut self,
        request_id: Uuid,
        computed_before: usize,
        computed_after: usize,
    ) {
        if !self.enable_prefix_caching {
            return;
        }
        assert!(
            computed_before <= computed_after,
            "computed token count cannot move backwards during one scheduling decision"
        );
        let first_new_block = computed_before / self.block_size;
        let completed_blocks = computed_after / self.block_size;
        if first_new_block == completed_blocks {
            return;
        }

        let materialize_store_events = self.materialize_store_events();
        let stores = {
            let (pool, attention) = self.attention_mut();
            attention.finalize_computed_prefix(
                pool,
                request_id,
                first_new_block,
                completed_blocks,
                materialize_store_events,
            )
        };
        if let Some(stores) = stores {
            self.publish_store_sequence(stores);
        }
    }

    pub(crate) fn reserve_decode_blocks(
        &mut self,
        count: usize,
    ) -> VllmAcquire<NativeDecodeBlockReservation> {
        let Some(outcome) = self.pool.reserve(&[], count) else {
            return VllmAcquire::CapacityExhausted;
        };
        self.publish_removed(outcome.removed);
        VllmAcquire::Ready(NativeDecodeBlockReservation {
            pool: outcome.reservation,
        })
    }

    pub(crate) fn release_decode_reservation(&mut self, reservation: NativeDecodeBlockReservation) {
        self.pool.cancel(reservation.pool);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn use_decode_reservation_for_request(
        &mut self,
        request_id: Uuid,
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
        reservation: &mut NativeDecodeBlockReservation,
    ) {
        let alloc = GroupAllocation {
            request_id,
            blocks,
            local_hashes,
            token_ids,
            parent,
            reusable_prefix_blocks: 0,
            cache_fresh: false,
        };
        let outcome = self.process_use(&alloc, Some(&mut reservation.pool));
        assert!(
            matches!(outcome, VllmAcquire::Ready(allocated) if allocated == blocks.len()),
            "reserved decode allocation must be infallible"
        );
    }

    pub(crate) fn reserve_destination_at(
        &mut self,
        request_id: Uuid,
        layout: Option<VllmBlockLayout>,
        _eviction_now_ms: Option<f64>,
    ) -> VllmAcquire<NativeDestinationReservation> {
        assert!(
            !self.attention().holds_request(request_id),
            "destination request already owns a block table"
        );
        let (prefix, fresh, attention_fresh, attention_prefix) = match layout.as_ref() {
            Some(layout) => {
                let alloc = Self::destination_alloc(request_id, layout, 0);
                FullAttentionGroup::validate_use_metadata(&alloc);
                self.attention().validate_fresh_partials(alloc.blocks);
                // No prior lookup authorized this transfer's reusable prefix, so
                // the attention group discovers it here; every group then sizes
                // itself against the prefix the request will keep.
                let attention_prefix = self
                    .attention()
                    .resident_prefix(&self.pool, &layout.blocks)
                    .len();
                let alloc = Self::destination_alloc(request_id, layout, attention_prefix);
                let (prefix, fresh) = self.group_demand(&alloc);
                let attention_fresh = self.attention().fresh_blocks(&alloc);
                (prefix, fresh, attention_fresh, attention_prefix)
            }
            None => (Vec::new(), 0, 0, 0),
        };
        let Some(outcome) = self.pool.reserve(&prefix, fresh) else {
            return VllmAcquire::CapacityExhausted;
        };
        self.publish_removed(outcome.removed);
        VllmAcquire::Ready(NativeDestinationReservation {
            request_id,
            pool: outcome.reservation,
            layout,
            attention_fresh,
            attention_prefix,
        })
    }

    /// Destination transfers arrive as one already-computed prefix, so their
    /// allocation covers the whole layout at once.
    fn destination_alloc(
        request_id: Uuid,
        layout: &VllmBlockLayout,
        reusable_prefix_blocks: usize,
    ) -> GroupAllocation<'_> {
        GroupAllocation {
            request_id,
            blocks: &layout.blocks,
            local_hashes: &layout.local_hashes,
            token_ids: layout.token_ids.as_deref(),
            parent: layout.parent.as_ref(),
            reusable_prefix_blocks,
            cache_fresh: true,
        }
    }

    pub(crate) fn activate_destination(&mut self, reservation: NativeDestinationReservation) {
        let NativeDestinationReservation {
            request_id,
            mut pool,
            layout,
            attention_prefix,
            ..
        } = reservation;
        assert!(
            !self.attention().holds_request(request_id),
            "destination request already owns a block table"
        );
        let Some(layout) = layout else {
            self.pool.cancel(pool);
            return;
        };
        let mut alloc = Self::destination_alloc(request_id, &layout, attention_prefix);
        alloc.cache_fresh = self.enable_prefix_caching;
        FullAttentionGroup::validate_use_metadata(&alloc);
        self.attention().validate_fresh_partials(alloc.blocks);
        self.commit_groups(&alloc, &mut pool);
        assert_eq!(pool.len(), 0, "destination reservation was not consumed");
        self.pool.cancel(pool);
    }

    pub(crate) fn cancel_destination(&mut self, reservation: NativeDestinationReservation) {
        self.pool.cancel(reservation.pool);
    }

    fn process_use(
        &mut self,
        alloc: &GroupAllocation,
        reservation: Option<&mut BlockReservation>,
    ) -> VllmAcquire<usize> {
        FullAttentionGroup::validate_use_metadata(alloc);
        self.attention().validate_fresh_partials(alloc.blocks);
        assert!(alloc.reusable_prefix_blocks <= alloc.blocks.len());
        assert!(self.enable_prefix_caching || alloc.reusable_prefix_blocks == 0);
        assert!(
            alloc.reusable_prefix_blocks == 0
                || self.attention().block_count(alloc.request_id) == 0,
            "only a request's first allocation may reuse a prefix"
        );

        let (prefix, fresh) = self.group_demand(alloc);

        match reservation {
            Some(reservation) => {
                assert!(prefix.is_empty(), "decode cannot reuse a new prefix");
                if reservation.fresh_len() < fresh {
                    return VllmAcquire::CapacityExhausted;
                }
                self.commit_groups(alloc, reservation);
            }
            None => {
                let Some(ReserveOutcome {
                    mut reservation,
                    removed,
                }) = self.pool.reserve(&prefix, fresh)
                else {
                    return VllmAcquire::CapacityExhausted;
                };
                self.publish_removed(removed);
                self.commit_groups(alloc, &mut reservation);
                assert_eq!(reservation.len(), 0, "Use reservation was not consumed");
                self.pool.cancel(reservation);
            }
        }
        VllmAcquire::Ready(alloc.blocks.len())
    }

    /// Prefix pins and fresh pool blocks every group needs for `alloc`.
    ///
    /// Groups are visited in coordinator order, which fixes the order the pins
    /// are reserved in and so the order [`Self::commit_groups`] activates them.
    fn group_demand(&self, alloc: &GroupAllocation) -> (Vec<GroupedHash>, usize) {
        let mut prefix = Vec::new();
        let mut fresh = 0;
        for group in &self.groups {
            prefix.extend(group.prefix_keys(alloc));
            fresh += group.fresh_blocks(alloc);
        }
        (prefix, fresh)
    }

    /// Commit every group against the shared reservation, publishing only the
    /// main attention group's store events.
    fn commit_groups(&mut self, alloc: &GroupAllocation, reservation: &mut BlockReservation) {
        let materialize_store_events = self.materialize_store_events();
        let stores = {
            let Self { pool, groups, .. } = self;
            // Pins come back in the order [`Self::group_demand`] reserved them,
            // which is the order the groups are visited here.
            let prefix_copies = pool.activate_prefix(reservation);
            let mut stores = None;
            for group in groups.iter_mut() {
                match group {
                    GroupManager::FullAttention(attention) => {
                        stores = attention.commit(
                            pool,
                            reservation,
                            alloc,
                            &prefix_copies,
                            materialize_store_events,
                        );
                    }
                }
            }
            stores
        };
        if let Some(stores) = stores {
            self.publish_store_sequence(stores);
        }
    }

    fn materialize_store_events(&self) -> bool {
        !self.kv_event_publishers.is_empty() || *kv_cache_trace::KV_CACHE_TRACE_ENABLED
    }

    fn publish_store_sequence(&mut self, stores: Vec<Option<StoredBlock>>) {
        let mut group: Option<StoreGroup> = None;
        for store in stores {
            let Some(store) = store else {
                self.flush_store_group(&mut group);
                continue;
            };
            if group
                .as_ref()
                .is_some_and(|current| !current.can_append(&store))
            {
                self.flush_store_group(&mut group);
            }
            match &mut group {
                Some(current) => current.push(store),
                None => group = Some(StoreGroup::from_block(store)),
            }
        }
        self.flush_store_group(&mut group);
    }

    fn flush_store_group(&mut self, group: &mut Option<StoreGroup>) {
        let Some(group) = group.take() else {
            return;
        };
        self.publish_kv_event(
            group.blocks,
            group.local_hashes.as_deref().unwrap_or(&[]),
            group.parent_hash,
            true,
            group.token_ids,
        );
    }

    /// Publish evictions for the main attention group only.
    ///
    /// `KvCacheEvent` has no group field and the router indexes a single block
    /// namespace, so a non-attention group's eviction must not be reported as
    /// an attention block disappearing.
    fn publish_removed(&mut self, keys: Vec<GroupedHash>) {
        let attention = self.attention_id();
        let hashes = keys
            .into_iter()
            .filter(|key| key.group == attention)
            .map(|key| key.hash)
            .collect::<Vec<_>>();
        if !hashes.is_empty() {
            self.publish_kv_event(hashes, &[], None, false, None);
        }
    }

    fn publish_kv_event(
        &mut self,
        full_blocks: Vec<SequenceHash>,
        local_hashes: &[BlockHash],
        parent_hash: Option<SequenceHash>,
        is_store: bool,
        token_ids: Option<Vec<Vec<u32>>>,
    ) {
        if !self.enable_prefix_caching || full_blocks.is_empty() {
            return;
        }
        if *kv_cache_trace::KV_CACHE_TRACE_ENABLED {
            kv_cache_trace::log_vllm_trace(
                if is_store { "allocation" } else { "eviction" },
                self.dp_rank,
                self.block_size,
                self.num_active_blocks(),
                self.num_inactive_blocks(),
                self.max_capacity(),
            );
        }
        if self.kv_event_publishers.is_empty() {
            return;
        }
        assert!(local_hashes.is_empty() || local_hashes.len() == full_blocks.len());
        assert!(
            token_ids
                .as_ref()
                .is_none_or(|ids| ids.len() == full_blocks.len())
        );

        let data = if is_store {
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: full_blocks
                    .into_iter()
                    .enumerate()
                    .map(|(index, hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(
                            local_hashes.get(index).copied().unwrap_or_default(),
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
        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data,
            dp_rank: self.dp_rank,
        };
        self.next_event_id = self
            .next_event_id
            .checked_add(1)
            .unwrap_or_else(|| panic!("KV event ID overflow"));
        if let Err(error) = self
            .kv_event_publishers
            .publish(event, token_ids.as_deref())
        {
            tracing::warn!(error = %error, "failed to publish native G1 KV event");
        }
    }

    pub(crate) fn num_active_blocks(&self) -> usize {
        self.pool.num_active()
    }

    pub(crate) fn num_active_block_refs(&self) -> usize {
        self.pool.num_active_refs()
    }

    pub(crate) fn num_inactive_blocks(&self) -> usize {
        self.pool.num_inactive()
    }

    pub(crate) fn get_active_perc(&self) -> f64 {
        self.num_active_blocks() as f64 / self.max_capacity() as f64
    }

    pub(crate) fn max_capacity(&self) -> usize {
        self.pool.capacity()
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn dp_rank(&self) -> u32 {
        self.dp_rank
    }

    #[cfg(test)]
    pub(crate) fn request_block_count(&self, request_id: Uuid) -> usize {
        self.attention().block_count(request_id)
    }

    pub(crate) fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        let blocks = sequence.unique_blocks();
        let attention = if self.enable_prefix_caching && sequence.enable_prefix_caching() {
            self.attention().longest_cache_hit(&self.pool, blocks)
        } else {
            CacheHit::default()
        };

        // vLLM's hybrid coordinator seeds the candidate with the dense
        // full-attention hit and reduces it until every group agrees on one
        // boundary. Groups share a block size here, so a single pass converges
        // (its `is_simple_hybrid` shortcut).
        let mut reconciled = attention.tokens;
        for group in self.other_groups() {
            let Some(hit) = group.longest_cache_hit(&self.pool, blocks) else {
                continue;
            };
            reconciled = reconciled.min(hit.tokens);
        }

        let overlap_blocks = reconciled / self.block_size;
        let cached_tokens = reconciled.min(sequence.len());
        let active_cached_tokens = attention.active_tokens.min(cached_tokens);
        PrefillCost {
            new_blocks: blocks.len() - overlap_blocks,
            new_tokens: sequence.len() - cached_tokens,
            cached_tokens,
            active_cached_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::common::protocols::{RawKvEvent, RawKvEventSink};

    #[derive(Default)]
    struct CapturingRawSink {
        events: Mutex<Vec<RawKvEvent>>,
    }

    impl CapturingRawSink {
        fn take(&self) -> Vec<RawKvEvent> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }
    }

    impl RawKvEventSink for CapturingRawSink {
        fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    const BLOCK_SIZE: usize = 4;

    fn attention_manager(capacity: usize) -> VllmKvManager {
        VllmKvManager::new_with_event_sink(
            capacity,
            BLOCK_SIZE,
            true,
            KvEventPublishers::default(),
            0,
        )
    }

    fn use_full(
        manager: &mut VllmKvManager,
        owner: Uuid,
        hashes: &[u64],
        reusable_prefix_blocks: usize,
    ) -> VllmAcquire<usize> {
        let blocks = hashes
            .iter()
            .copied()
            .map(UniqueBlock::FullBlock)
            .collect::<Vec<_>>();
        let local_hashes = hashes.iter().map(|hash| hash + 100).collect::<Vec<_>>();
        manager.use_for_request(
            owner,
            &blocks,
            &local_hashes,
            None,
            None,
            reusable_prefix_blocks,
        )
    }

    fn ready<T>(outcome: VllmAcquire<T>) -> T {
        match outcome {
            VllmAcquire::Ready(value) => value,
            _ => panic!("unexpected allocation failure"),
        }
    }

    fn attention_hit(manager: &VllmKvManager, hash: SequenceHash) -> bool {
        manager
            .pool
            .prefix_hit(GroupedHash::new(manager.attention_id(), hash))
            .is_some()
    }

    #[test]
    fn duplicate_full_hashes_consume_physical_capacity() {
        let mut manager = attention_manager(2);
        for owner in [Uuid::from_u128(1), Uuid::from_u128(2)] {
            ready(use_full(&mut manager, owner, &[7], 0));
            manager.finalize_computed_prefix(owner, 0, 4);
        }
        assert_eq!(manager.num_active_blocks(), 2);
        assert!(matches!(
            use_full(&mut manager, Uuid::from_u128(3), &[8], 0),
            VllmAcquire::CapacityExhausted
        ));
    }

    #[test]
    fn authorized_prefix_reuses_one_physical_copy() {
        let mut manager = attention_manager(2);
        let first = Uuid::from_u128(1);
        ready(use_full(&mut manager, first, &[7], 0));
        manager.finalize_computed_prefix(first, 0, 4);
        manager.deref_for_request(first, &[UniqueBlock::FullBlock(7)]);

        ready(use_full(&mut manager, Uuid::from_u128(2), &[7], 1));
        assert_eq!(manager.num_active_blocks(), 1);
        assert_eq!(manager.num_inactive_blocks(), 0);
    }

    #[test]
    fn full_block_is_hidden_until_computed() {
        let mut manager = attention_manager(2);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7], 0));
        assert!(!attention_hit(&manager, 7));
        manager.finalize_computed_prefix(owner, 0, 4);
        assert!(attention_hit(&manager, 7));
    }

    #[test]
    fn finalization_only_visits_blocks_completed_by_this_decision() {
        let mut manager = attention_manager(2);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7, 8], 0));

        manager.finalize_computed_prefix(owner, 0, 4);
        assert!(attention_hit(&manager, 7));
        assert!(!attention_hit(&manager, 8));

        manager.finalize_computed_prefix(owner, 4, 8);
        assert!(attention_hit(&manager, 8));
    }

    #[test]
    fn finalization_handles_unaligned_decision_boundaries() {
        let mut manager = attention_manager(3);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7, 8, 9], 0));

        manager.finalize_computed_prefix(owner, 3, 9);
        assert!(attention_hit(&manager, 7));
        assert!(attention_hit(&manager, 8));
        assert!(!attention_hit(&manager, 9));
    }

    #[test]
    fn cached_prefix_watermark_finalizes_only_the_fresh_suffix() {
        let mut manager = attention_manager(2);
        let seed = Uuid::from_u128(1);
        ready(use_full(&mut manager, seed, &[7], 0));
        manager.finalize_computed_prefix(seed, 0, 4);
        manager.deref_for_request(seed, &[UniqueBlock::FullBlock(7)]);

        let owner = Uuid::from_u128(2);
        ready(use_full(&mut manager, owner, &[7, 8], 1));
        manager.finalize_computed_prefix(owner, 4, 8);
        assert!(attention_hit(&manager, 7));
        assert!(attention_hit(&manager, 8));
    }

    #[test]
    fn request_release_evicts_leaf_before_parent() {
        let mut manager = attention_manager(2);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7, 8], 0));
        manager.finalize_computed_prefix(owner, 0, 8);

        // Deref signals describe the request tail first.
        manager.deref_for_request(
            owner,
            &[UniqueBlock::FullBlock(8), UniqueBlock::FullBlock(7)],
        );
        ready(use_full(&mut manager, Uuid::from_u128(2), &[9], 0));

        assert!(attention_hit(&manager, 7), "parent should remain resident");
        assert!(!attention_hit(&manager, 8), "leaf should be evicted first");
    }

    #[test]
    fn event_enabled_finalization_preserves_store_payload() {
        let sink = Arc::new(CapturingRawSink::default());
        let publishers = KvEventPublishers::new(None, Some(sink.clone()));
        let mut manager = VllmKvManager::new_with_event_sink(2, BLOCK_SIZE, true, publishers, 3);
        let owner = Uuid::from_u128(1);
        let token_ids = vec![vec![1, 2, 3, 4]];
        let parent = UniqueBlock::FullBlock(6);

        ready(manager.use_for_request(
            owner,
            &[UniqueBlock::FullBlock(7)],
            &[107],
            Some(&token_ids),
            Some(&parent),
            0,
        ));
        manager.finalize_computed_prefix(owner, 0, 4);

        let mut events = sink.take();
        assert_eq!(events.len(), 1);
        let event = events.pop().unwrap();
        assert_eq!(event.event.event_id, 0);
        assert_eq!(event.event.dp_rank, 3);
        assert_eq!(event.block_token_ids, Some(token_ids));
        let KvCacheEventData::Stored(stored) = event.event.data else {
            panic!("expected Stored event")
        };
        assert_eq!(stored.parent_hash, Some(ExternalSequenceBlockHash(6)));
        assert_eq!(stored.blocks.len(), 1);
        assert_eq!(stored.blocks[0].block_hash, ExternalSequenceBlockHash(7));
        assert_eq!(stored.blocks[0].tokens_hash, LocalBlockHash(107));
    }

    /// Removals are published per group, so an attention eviction must still
    /// reach the router.
    #[test]
    fn attention_eviction_is_published_as_a_removal() {
        let sink = Arc::new(CapturingRawSink::default());
        let publishers = KvEventPublishers::new(None, Some(sink.clone()));
        let mut manager = VllmKvManager::new_with_event_sink(1, BLOCK_SIZE, true, publishers, 0);

        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7], 0));
        manager.finalize_computed_prefix(owner, 0, 4);
        manager.deref_for_request(owner, &[UniqueBlock::FullBlock(7)]);
        sink.take();

        // The next request needs the pool's only block back.
        ready(use_full(&mut manager, Uuid::from_u128(2), &[8], 0));
        let removed = sink
            .take()
            .into_iter()
            .filter_map(|event| match event.event.data {
                KvCacheEventData::Removed(removed) => Some(removed.block_hashes),
                _ => None,
            })
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(removed, vec![ExternalSequenceBlockHash(7)]);
    }
}
