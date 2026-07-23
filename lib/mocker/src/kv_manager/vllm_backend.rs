// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM G1 manager over a minimal physical block-pool model.
//!
//! The manager owns request block tables and KV-event metadata. The pool owns
//! physical occupancy, duplicate copies, prefix pins, and LRU eviction.

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{BlockHash, SequenceHash};
use rustc_hash::{FxHashMap, FxHashSet};
use uuid::Uuid;

use crate::cache::vllm_block_pool::{BlockCopyId, BlockReservation, ReserveOutcome, VllmBlockPool};
use crate::common::kv_cache_trace;
use crate::common::protocols::{KvEventPublishers, PrefillCost};
use crate::common::sequence::ActiveSequence;

struct PendingStore {
    parent_hash: Option<SequenceHash>,
    local_hash: Option<BlockHash>,
    token_ids: Option<Vec<u32>>,
}

struct FullBlock {
    copy: BlockCopyId,
    hash: SequenceHash,
    /// Whether a freshly allocated full block still needs to become cache-visible.
    pending_cache: bool,
    /// Event metadata retained until `pending_cache` is finalized.
    pending_store: Option<PendingStore>,
}

enum OwnedBlock {
    Partial { copy: BlockCopyId, uuid: Uuid },
    Full(FullBlock),
}

struct StoredBlock {
    hash: SequenceHash,
    metadata: PendingStore,
}

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
}

impl NativeDestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self, block_size: usize) -> usize {
        self.pool.fresh_len().saturating_mul(block_size)
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
    request_blocks: FxHashMap<Uuid, Vec<OwnedBlock>>,
    partial_uuids: FxHashSet<Uuid>,
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
            request_blocks: FxHashMap::default(),
            partial_uuids: FxHashSet::default(),
            block_size,
            enable_prefix_caching,
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
        }
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
        self.process_use(
            request_id,
            blocks,
            local_hashes,
            token_ids,
            parent,
            reusable_prefix_blocks,
            None,
        )
    }

    pub(super) fn deref_for_request(&mut self, request_id: Uuid, blocks: &[UniqueBlock]) {
        self.process_deref(request_id, blocks);
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
        self.process_promote(request_id, uuid, hash, parent_hash, local_hash, token_ids);
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
        let Some(blocks) = self.request_blocks.get_mut(&request_id) else {
            panic!("request {request_id} owns no block table")
        };
        let completed_blocks = completed_blocks.min(blocks.len());
        if first_new_block >= completed_blocks {
            return;
        }
        let mut stores = materialize_store_events
            .then(|| Vec::with_capacity(completed_blocks - first_new_block));
        for block in &mut blocks[first_new_block..completed_blocks] {
            match block {
                OwnedBlock::Full(full) => {
                    if !full.pending_cache {
                        if let Some(stores) = &mut stores {
                            stores.push(None);
                        }
                        continue;
                    }
                    full.pending_cache = false;
                    let metadata = full.pending_store.take();
                    let became_visible = self.pool.cache_private(full.copy, full.hash);
                    if let Some(stores) = &mut stores {
                        stores.push(became_visible.then(|| {
                            StoredBlock {
                                hash: full.hash,
                                metadata: metadata.expect(
                                    "materialized pending store must retain event metadata",
                                ),
                            }
                        }));
                    } else {
                        debug_assert!(metadata.is_none());
                    }
                }
                OwnedBlock::Partial { .. } => break,
            }
        }
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
        let outcome = self.process_use(
            request_id,
            blocks,
            local_hashes,
            token_ids,
            parent,
            0,
            Some(&mut reservation.pool),
        );
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
            !self.request_blocks.contains_key(&request_id),
            "destination request already owns a block table"
        );
        let (prefix, fresh) = match layout.as_ref() {
            Some(VllmBlockLayout {
                blocks,
                local_hashes,
                token_ids,
                parent,
            }) => {
                Self::validate_use_metadata(
                    blocks,
                    local_hashes,
                    token_ids.as_deref(),
                    parent.as_ref(),
                );
                self.validate_fresh_partials(blocks);
                let prefix = self.resident_prefix(blocks);
                let fresh = blocks.len() - prefix.len();
                (prefix, fresh)
            }
            None => (Vec::new(), 0),
        };
        let Some(outcome) = self.pool.reserve(&prefix, fresh) else {
            return VllmAcquire::CapacityExhausted;
        };
        self.publish_removed(outcome.removed);
        VllmAcquire::Ready(NativeDestinationReservation {
            request_id,
            pool: outcome.reservation,
            layout,
        })
    }

    pub(crate) fn activate_destination(&mut self, reservation: NativeDestinationReservation) {
        let NativeDestinationReservation {
            request_id,
            mut pool,
            layout,
        } = reservation;
        assert!(
            !self.request_blocks.contains_key(&request_id),
            "destination request already owns a block table"
        );
        let Some(VllmBlockLayout {
            blocks,
            local_hashes,
            token_ids,
            parent,
        }) = layout
        else {
            self.pool.cancel(pool);
            return;
        };
        Self::validate_use_metadata(
            &blocks,
            &local_hashes,
            token_ids.as_deref(),
            parent.as_ref(),
        );
        self.validate_fresh_partials(&blocks);
        self.commit_layout(
            request_id,
            &blocks,
            &local_hashes,
            token_ids.as_deref(),
            parent.as_ref(),
            &mut pool,
            self.enable_prefix_caching,
        );
        assert_eq!(pool.len(), 0, "destination reservation was not consumed");
        self.pool.cancel(pool);
    }

    pub(crate) fn cancel_destination(&mut self, reservation: NativeDestinationReservation) {
        self.pool.cancel(reservation.pool);
    }

    #[allow(clippy::too_many_arguments)]
    fn process_use(
        &mut self,
        request_id: Uuid,
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
        reusable_prefix_blocks: usize,
        reservation: Option<&mut BlockReservation>,
    ) -> VllmAcquire<usize> {
        Self::validate_use_metadata(blocks, local_hashes, token_ids, parent);
        self.validate_fresh_partials(blocks);
        assert!(reusable_prefix_blocks <= blocks.len());
        assert!(self.enable_prefix_caching || reusable_prefix_blocks == 0);
        assert!(
            reusable_prefix_blocks == 0
                || self
                    .request_blocks
                    .get(&request_id)
                    .is_none_or(Vec::is_empty),
            "only a request's first allocation may reuse a prefix"
        );

        let prefix = if reusable_prefix_blocks == 0 {
            Vec::new()
        } else {
            blocks[..reusable_prefix_blocks]
                .iter()
                .map(|block| match block {
                    UniqueBlock::FullBlock(hash) => *hash,
                    UniqueBlock::PartialBlock(_) => {
                        panic!("a reusable prefix can contain only full blocks")
                    }
                })
                .collect::<Vec<_>>()
        };
        let fresh = blocks.len() - reusable_prefix_blocks;

        match reservation {
            Some(reservation) => {
                assert!(prefix.is_empty(), "decode cannot reuse a new prefix");
                if reservation.fresh_len() < fresh {
                    return VllmAcquire::CapacityExhausted;
                }
                self.commit_layout(
                    request_id,
                    blocks,
                    local_hashes,
                    token_ids,
                    parent,
                    reservation,
                    false,
                );
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
                self.commit_layout(
                    request_id,
                    blocks,
                    local_hashes,
                    token_ids,
                    parent,
                    &mut reservation,
                    false,
                );
                assert_eq!(reservation.len(), 0, "Use reservation was not consumed");
                self.pool.cancel(reservation);
            }
        }
        VllmAcquire::Ready(blocks.len())
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_layout(
        &mut self,
        request_id: Uuid,
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
        reservation: &mut BlockReservation,
        cache_fresh: bool,
    ) {
        let prefix_len = reservation.len() - reservation.fresh_len();
        let prefix_copies = self.pool.activate_prefix(reservation);
        assert_eq!(prefix_copies.len(), prefix_len);
        let mut prefix_copies = prefix_copies.into_iter();
        let mut cursor = match parent {
            None => None,
            Some(UniqueBlock::FullBlock(hash)) => Some(*hash),
            Some(UniqueBlock::PartialBlock(_)) => unreachable!("validated above"),
        };
        let mut full_idx = 0;
        let materialize_store_events = self.materialize_store_events();
        let owned = self.request_blocks.entry(request_id).or_default();
        owned.reserve(blocks.len());
        let mut stores =
            (cache_fresh && materialize_store_events).then(|| Vec::with_capacity(blocks.len()));

        for (block_idx, block) in blocks.iter().enumerate() {
            match block {
                UniqueBlock::FullBlock(hash) => {
                    let local_hash = local_hashes.get(full_idx).copied();
                    full_idx += 1;

                    if block_idx < prefix_len {
                        let Some(copy) = prefix_copies.next() else {
                            panic!("prefix reservation returned too few copies")
                        };
                        owned.push(OwnedBlock::Full(FullBlock {
                            copy,
                            hash: *hash,
                            pending_cache: false,
                            pending_store: None,
                        }));
                        cursor = Some(*hash);
                        continue;
                    }

                    if cache_fresh {
                        let (copy, became_visible) = self.pool.allocate_cached(reservation, *hash);
                        owned.push(OwnedBlock::Full(FullBlock {
                            copy,
                            hash: *hash,
                            pending_cache: false,
                            pending_store: None,
                        }));
                        if let Some(stores) = &mut stores {
                            let metadata = PendingStore {
                                parent_hash: cursor,
                                local_hash,
                                token_ids: token_ids.and_then(|ids| ids.get(full_idx - 1).cloned()),
                            };
                            stores.push(became_visible.then_some(StoredBlock {
                                hash: *hash,
                                metadata,
                            }));
                        }
                    } else {
                        let copy = self.pool.allocate_private(reservation);
                        let pending_cache = self.enable_prefix_caching;
                        let pending_store =
                            (pending_cache && materialize_store_events).then(|| PendingStore {
                                parent_hash: cursor,
                                local_hash,
                                token_ids: token_ids.and_then(|ids| ids.get(full_idx - 1).cloned()),
                            });
                        owned.push(OwnedBlock::Full(FullBlock {
                            copy,
                            hash: *hash,
                            pending_cache,
                            pending_store,
                        }));
                    }
                    cursor = Some(*hash);
                }
                UniqueBlock::PartialBlock(uuid) => {
                    let copy = self.pool.allocate_private(reservation);
                    assert!(
                        self.partial_uuids.insert(*uuid),
                        "partial block {uuid} is already allocated"
                    );
                    owned.push(OwnedBlock::Partial { copy, uuid: *uuid });
                    if let Some(stores) = &mut stores {
                        stores.push(None);
                    }
                }
            }
        }
        assert!(prefix_copies.next().is_none());
        if let Some(stores) = stores {
            self.publish_store_sequence(stores);
        }
    }

    fn validate_use_metadata(
        blocks: &[UniqueBlock],
        local_hashes: &[BlockHash],
        token_ids: Option<&[Vec<u32>]>,
        parent: Option<&UniqueBlock>,
    ) {
        let full_blocks = blocks
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert!(
            local_hashes.is_empty() || local_hashes.len() == full_blocks,
            "local hashes must be empty or align with full blocks"
        );
        assert!(
            token_ids.is_none_or(|ids| ids.len() == full_blocks),
            "token IDs must align with full blocks"
        );
        assert!(!matches!(parent, Some(UniqueBlock::PartialBlock(_))));
    }

    fn validate_fresh_partials(&self, blocks: &[UniqueBlock]) {
        let mut first_partial = None;
        for (index, uuid) in blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| match block {
                UniqueBlock::PartialBlock(uuid) => Some((index, uuid)),
                UniqueBlock::FullBlock(_) => None,
            })
        {
            let repeated_in_layout = first_partial.is_some_and(|first| {
                first == *uuid
                    || blocks[..index].iter().any(
                        |block| matches!(block, UniqueBlock::PartialBlock(seen) if seen == uuid),
                    )
            });
            assert!(
                !self.partial_uuids.contains(uuid) && !repeated_in_layout,
                "partial block {uuid} is already allocated"
            );
            first_partial.get_or_insert(*uuid);
        }
    }

    fn resident_prefix(&self, blocks: &[UniqueBlock]) -> Vec<SequenceHash> {
        if !self.enable_prefix_caching {
            return Vec::new();
        }
        blocks
            .iter()
            .map_while(|block| match block {
                UniqueBlock::FullBlock(hash) if self.pool.prefix_hit(*hash).is_some() => {
                    Some(*hash)
                }
                _ => None,
            })
            .collect()
    }

    /// Release request blocks in caller-provided eviction-priority order.
    ///
    /// Like vLLM's `BlockPool::free_blocks`, the physical pool is lineage-agnostic:
    /// reversing the request-owned table here makes suffix/leaf blocks older LRU
    /// candidates than their parents, so capacity pressure evicts the leaf first.
    fn process_deref(&mut self, request_id: Uuid, blocks: &[UniqueBlock]) {
        let released = {
            let Some(owned) = self.request_blocks.get_mut(&request_id) else {
                panic!("request {request_id} owns no block table")
            };
            assert!(
                blocks.len() <= owned.len(),
                "request releases too many blocks"
            );
            let start = owned.len() - blocks.len();
            for (expected, actual) in blocks.iter().zip(owned[start..].iter().rev()) {
                match (expected, actual) {
                    (UniqueBlock::FullBlock(expected), OwnedBlock::Full(full)) => {
                        assert_eq!(*expected, full.hash, "full-block Deref mismatch");
                    }
                    (UniqueBlock::PartialBlock(expected), OwnedBlock::Partial { uuid, .. }) => {
                        assert_eq!(expected, uuid, "partial Deref mismatch")
                    }
                    _ => panic!("Deref block kind disagrees with request table"),
                }
            }
            owned.split_off(start)
        };
        if self.request_blocks[&request_id].is_empty() {
            self.request_blocks.remove(&request_id);
        }

        for block in released.into_iter().rev() {
            match block {
                OwnedBlock::Partial { copy, uuid } => {
                    assert!(self.partial_uuids.remove(&uuid));
                    self.pool.release(copy);
                }
                OwnedBlock::Full(full) => self.pool.release(full.copy),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_promote(
        &mut self,
        request_id: Uuid,
        uuid: Uuid,
        hash: SequenceHash,
        parent_hash: Option<SequenceHash>,
        local_hash: Option<BlockHash>,
        token_ids: Option<Vec<u32>>,
    ) {
        let materialize_store_events = self.materialize_store_events();
        let Some(blocks) = self.request_blocks.get_mut(&request_id) else {
            panic!("request {request_id} owns no block table")
        };
        let Some(last) = blocks.last_mut() else {
            panic!("Promote requires a request-owned partial tail")
        };
        let copy = match last {
            OwnedBlock::Partial { copy, uuid: actual } => {
                assert_eq!(*actual, uuid, "Promote partial UUID mismatch");
                *copy
            }
            OwnedBlock::Full(_) => panic!("Promote requires a partial tail"),
        };
        assert!(self.partial_uuids.remove(&uuid));

        let became_visible = self.enable_prefix_caching && self.pool.cache_private(copy, hash);
        *last = OwnedBlock::Full(FullBlock {
            copy,
            hash,
            pending_cache: false,
            pending_store: None,
        });
        if became_visible && materialize_store_events {
            self.publish_store_sequence(vec![Some(StoredBlock {
                hash,
                metadata: PendingStore {
                    parent_hash,
                    local_hash,
                    token_ids,
                },
            })]);
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

    fn publish_removed(&mut self, hashes: Vec<SequenceHash>) {
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
        self.request_blocks.get(&request_id).map_or(0, Vec::len)
    }

    pub(crate) fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        let (overlap_blocks, active_overlap_blocks) =
            if self.enable_prefix_caching && sequence.enable_prefix_caching() {
                let mut overlap = 0;
                let mut active = 0;
                for block in sequence.unique_blocks() {
                    let UniqueBlock::FullBlock(hash) = block else {
                        break;
                    };
                    let Some(hit) = self.pool.prefix_hit(*hash) else {
                        break;
                    };
                    overlap += 1;
                    active += usize::from(hit.is_active);
                }
                (overlap, active)
            } else {
                (0, 0)
            };
        let new_blocks = sequence.unique_blocks().len() - overlap_blocks;
        let cached_tokens = (overlap_blocks * self.block_size).min(sequence.len());
        let active_cached_tokens = (active_overlap_blocks * self.block_size).min(sequence.len());
        PrefillCost {
            new_blocks,
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

    #[test]
    fn duplicate_full_hashes_consume_physical_capacity() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
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
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
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
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7], 0));
        assert!(manager.pool.prefix_hit(7).is_none());
        manager.finalize_computed_prefix(owner, 0, 4);
        assert!(manager.pool.prefix_hit(7).is_some());
    }

    #[test]
    fn finalization_only_visits_blocks_completed_by_this_decision() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7, 8], 0));

        manager.finalize_computed_prefix(owner, 0, 4);
        assert!(manager.pool.prefix_hit(7).is_some());
        assert!(manager.pool.prefix_hit(8).is_none());

        manager.finalize_computed_prefix(owner, 4, 8);
        assert!(manager.pool.prefix_hit(8).is_some());
    }

    #[test]
    fn finalization_handles_unaligned_decision_boundaries() {
        let mut manager =
            VllmKvManager::new_with_event_sink(3, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7, 8, 9], 0));

        manager.finalize_computed_prefix(owner, 3, 9);
        assert!(manager.pool.prefix_hit(7).is_some());
        assert!(manager.pool.prefix_hit(8).is_some());
        assert!(manager.pool.prefix_hit(9).is_none());
    }

    #[test]
    fn cached_prefix_watermark_finalizes_only_the_fresh_suffix() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let seed = Uuid::from_u128(1);
        ready(use_full(&mut manager, seed, &[7], 0));
        manager.finalize_computed_prefix(seed, 0, 4);
        manager.deref_for_request(seed, &[UniqueBlock::FullBlock(7)]);

        let owner = Uuid::from_u128(2);
        ready(use_full(&mut manager, owner, &[7, 8], 1));
        manager.finalize_computed_prefix(owner, 4, 8);
        assert!(manager.pool.prefix_hit(7).is_some());
        assert!(manager.pool.prefix_hit(8).is_some());
    }

    #[test]
    fn request_release_evicts_leaf_before_parent() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        ready(use_full(&mut manager, owner, &[7, 8], 0));
        manager.finalize_computed_prefix(owner, 0, 8);

        // Deref signals describe the request tail first.
        manager.deref_for_request(
            owner,
            &[UniqueBlock::FullBlock(8), UniqueBlock::FullBlock(7)],
        );
        ready(use_full(&mut manager, Uuid::from_u128(2), &[9], 0));

        assert!(
            manager.pool.prefix_hit(7).is_some(),
            "parent should remain resident"
        );
        assert!(
            manager.pool.prefix_hit(8).is_none(),
            "leaf should be evicted first"
        );
    }

    #[test]
    fn event_enabled_finalization_preserves_store_payload() {
        let sink = Arc::new(CapturingRawSink::default());
        let publishers = KvEventPublishers::new(None, Some(sink.clone()));
        let mut manager = VllmKvManager::new_with_event_sink(2, 4, true, publishers, 3);
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
}
