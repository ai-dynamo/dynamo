// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM G1 manager over a minimal physical block-pool model.
//!
//! Each request lease owns its physical-copy IDs and visibility state. The
//! manager owns KV-event metadata, while the pool owns occupancy, duplicate
//! copies, prefix pins, and LRU eviction.

use uuid::Uuid;

use crate::engine::cache::vllm_block_pool::{
    BlockCopyId, BlockReservation, ReserveOutcome, VllmBlockPool,
};
use crate::engine::common::hashing::{BlockHash, SequenceHash};
use crate::engine::common::kv_cache_trace;
use crate::engine::common::protocols::{KvEventPublishers, PrefillCost};
use crate::engine::common::sequence::{BlockIdentity, RequestSequence};
use crate::engine::{KvBlock, KvEvent, KvEventData, StoredBlocks};

struct PendingStore {
    parent_hash: Option<SequenceHash>,
    local_hash: Option<BlockHash>,
    token_ids: Option<Vec<u32>>,
}

#[derive(Debug)]
struct BlockLeaseEntry {
    identity: BlockIdentity,
    copy: Option<BlockCopyId>,
    /// Whether a freshly allocated full block still needs to become cache-visible.
    pending_cache: bool,
}

/// Move-only native-G1 ownership token attached to one scheduler request.
#[derive(Debug)]
#[must_use = "a native block lease must be finished, aborted, retracted, or moved into a hold"]
pub(crate) struct BlockRequestLease {
    owner: Uuid,
    entries: Vec<BlockLeaseEntry>,
    allocated_tokens: usize,
}

impl BlockRequestLease {
    pub(crate) fn new(owner: Uuid, identities: Vec<BlockIdentity>) -> Self {
        let mut entries = Vec::with_capacity(identities.capacity());
        entries.extend(identities.into_iter().map(|identity| BlockLeaseEntry {
            identity,
            copy: None,
            pending_cache: false,
        }));
        Self {
            owner,
            entries,
            allocated_tokens: 0,
        }
    }

    pub(crate) fn owner(&self) -> Uuid {
        self.owner
    }

    pub(crate) fn allocated_tokens(&self) -> usize {
        self.allocated_tokens
    }

    pub(crate) fn resident_block_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.copy.is_some())
            .count()
    }

    #[cfg(test)]
    pub(crate) fn entry_capacity(&self) -> usize {
        self.entries.capacity()
    }

    pub(crate) fn append_partial(&mut self) {
        // One scheduler decision can materialize more than one token (for
        // example speculative decoding). In that case the previous partial
        // block may already be logically complete but still await
        // `finalize_lease_computed_prefix`, so its identity is intentionally
        // unresolved while the next partial entry is opened.
        self.entries.push(BlockLeaseEntry {
            identity: BlockIdentity::partial(),
            copy: None,
            pending_cache: false,
        });
    }

    fn debug_assert_owner(&self, owner: Uuid) {
        debug_assert_eq!(self.owner, owner, "native lease owner mismatch");
    }
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

pub(crate) struct DecodeBlockReservation {
    pool: BlockReservation,
}

pub(crate) struct DestinationReservation {
    request_id: Uuid,
    pool: BlockReservation,
}

impl DestinationReservation {
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

pub(crate) struct VllmKvManager {
    pool: VllmBlockPool,
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
            block_size,
            enable_prefix_caching,
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
        }
    }

    /// Atomically allocate the native lease through `cumulative_tokens`.
    ///
    /// Capacity is reserved before either physical residency or the allocation
    /// watermark changes, so exhaustion leaves the lease unchanged.
    pub(crate) fn allocate_lease(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        cumulative_tokens: usize,
        reusable_prefix_blocks: usize,
    ) -> VllmAcquire<usize> {
        lease.debug_assert_owner(owner);
        let previous_blocks = lease
            .allocated_tokens
            .div_ceil(self.block_size)
            .min(lease.entries.len());
        let target_blocks = cumulative_tokens
            .div_ceil(self.block_size)
            .min(lease.entries.len());
        if target_blocks <= previous_blocks {
            lease.allocated_tokens = cumulative_tokens;
            return VllmAcquire::Ready(0);
        }
        assert!(
            reusable_prefix_blocks == 0 || previous_blocks == 0,
            "only a request's first allocation may reuse a prefix"
        );
        assert!(
            reusable_prefix_blocks <= target_blocks - previous_blocks,
            "reusable prefix exceeds the newly allocated block range"
        );
        assert!(self.enable_prefix_caching || reusable_prefix_blocks == 0);

        let count = target_blocks - previous_blocks;
        let prefix = lease.entries[previous_blocks..previous_blocks + reusable_prefix_blocks]
            .iter()
            .map(|entry| {
                entry
                    .identity
                    .sequence_hash
                    .expect("reusable prefix must contain only complete blocks")
            });
        let Some(ReserveOutcome {
            mut reservation,
            removed,
        }) = self.pool.reserve_exact_prefix(prefix, count)
        else {
            return VllmAcquire::CapacityExhausted;
        };
        assert_eq!(
            reservation.len() - reservation.fresh_len(),
            reusable_prefix_blocks,
            "exact native prefix reservation returned the wrong hit count"
        );
        self.publish_removed(removed);
        self.commit_lease_range(
            lease,
            previous_blocks,
            target_blocks,
            &mut reservation,
            false,
            None,
        );
        assert_eq!(reservation.len(), 0, "native reservation was not consumed");
        self.pool.cancel(reservation);
        lease.allocated_tokens = cumulative_tokens;
        VllmAcquire::Ready(count)
    }

    pub(crate) fn allocate_lease_from_decode_reservation(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        cumulative_tokens: usize,
        reservation: &mut DecodeBlockReservation,
    ) {
        lease.debug_assert_owner(owner);
        let previous_blocks = lease
            .allocated_tokens
            .div_ceil(self.block_size)
            .min(lease.entries.len());
        let target_blocks = cumulative_tokens
            .div_ceil(self.block_size)
            .min(lease.entries.len());
        if target_blocks <= previous_blocks {
            lease.allocated_tokens = cumulative_tokens;
            return;
        }
        let count = target_blocks - previous_blocks;
        assert!(
            reservation.pool.fresh_len() >= count,
            "decode reservation does not cover the native lease growth"
        );
        self.commit_lease_range(
            lease,
            previous_blocks,
            target_blocks,
            &mut reservation.pool,
            false,
            None,
        );
        lease.allocated_tokens = cumulative_tokens;
    }

    pub(crate) fn finalize_lease_computed_prefix(
        &mut self,
        owner: Uuid,
        sequence: &mut RequestSequence,
        lease: &mut BlockRequestLease,
        computed_before: usize,
        computed_after: usize,
    ) {
        lease.debug_assert_owner(owner);
        assert!(
            computed_before <= computed_after,
            "computed token count cannot move backwards during one scheduling decision"
        );
        let first_new_block = computed_before / self.block_size;
        let completed_blocks = (computed_after / self.block_size).min(lease.entries.len());
        if first_new_block >= completed_blocks {
            return;
        }

        let materialize_store_events =
            self.enable_prefix_caching && self.materialize_store_events();
        let mut stores = materialize_store_events
            .then(|| Vec::with_capacity(completed_blocks - first_new_block));
        for position in first_new_block..completed_blocks {
            let parent_hash = position
                .checked_sub(1)
                .and_then(|parent| lease.entries[parent].identity.sequence_hash);
            if lease.entries[position].identity.sequence_hash.is_none() {
                lease.entries[position].identity =
                    sequence.complete_block_identity(position, parent_hash);
                lease.entries[position].pending_cache = self.enable_prefix_caching;
            }

            let entry = &mut lease.entries[position];
            if !entry.pending_cache {
                if let Some(stores) = &mut stores {
                    stores.push(None);
                }
                sequence.discard_completed_block(position);
                continue;
            }
            entry.pending_cache = false;
            let copy = entry
                .copy
                .expect("computed native block must retain physical residency");
            let hash = entry
                .identity
                .sequence_hash
                .expect("computed native block must have a sequence hash");
            let became_visible = self.pool.cache_private(copy, hash);
            if let Some(stores) = &mut stores {
                stores.push(became_visible.then(|| StoredBlock {
                    hash,
                    metadata: PendingStore {
                        parent_hash,
                        local_hash: entry.identity.local_hash,
                        token_ids: sequence.block_token_ids(position),
                    },
                }));
            }
            sequence.discard_completed_block(position);
        }
        if let Some(stores) = stores {
            self.publish_store_sequence(stores);
        }
        #[cfg(debug_assertions)]
        sequence.debug_assert_finalized_range(
            lease.entries.len(),
            lease.entries[first_new_block..completed_blocks]
                .iter()
                .map(|entry| entry.identity),
            lease.entries.last().map(|entry| entry.identity),
        );
    }

    pub(crate) fn reserve_destination_lease(
        &mut self,
        owner: Uuid,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
        _eviction_now_ms: Option<f64>,
    ) -> VllmAcquire<DestinationReservation> {
        lease.debug_assert_owner(owner);
        assert_eq!(
            lease.resident_block_count(),
            0,
            "destination request already owns physical blocks"
        );
        let prompt_blocks = sequence
            .num_input_tokens()
            .div_ceil(self.block_size)
            .min(lease.entries.len());
        let prefix_candidates = lease.entries[..prompt_blocks]
            .iter()
            .map_while(|entry| entry.identity.sequence_hash);
        let Some(outcome) = self
            .pool
            .reserve_resident_prefix(prefix_candidates, prompt_blocks)
        else {
            return VllmAcquire::CapacityExhausted;
        };
        self.publish_removed(outcome.removed);
        VllmAcquire::Ready(DestinationReservation {
            request_id: owner,
            pool: outcome.reservation,
        })
    }

    pub(crate) fn activate_destination_lease(
        &mut self,
        owner: Uuid,
        sequence: &RequestSequence,
        lease: &mut BlockRequestLease,
        mut reservation: DestinationReservation,
    ) {
        lease.debug_assert_owner(owner);
        debug_assert_eq!(
            lease.resident_block_count(),
            0,
            "destination request already owns physical blocks"
        );
        assert_eq!(reservation.request_id, owner, "destination owner mismatch");
        let prompt_blocks = sequence
            .num_input_tokens()
            .div_ceil(self.block_size)
            .min(lease.entries.len());
        self.commit_lease_range(
            lease,
            0,
            prompt_blocks,
            &mut reservation.pool,
            self.enable_prefix_caching,
            Some(sequence),
        );
        lease.allocated_tokens = sequence.num_input_tokens();
        assert_eq!(
            reservation.pool.len(),
            0,
            "destination reservation was not consumed"
        );
        self.pool.cancel(reservation.pool);
    }

    pub(crate) fn preempt_lease(&mut self, owner: Uuid, lease: &mut BlockRequestLease) {
        lease.debug_assert_owner(owner);
        self.release_lease_entries(lease);
        lease.allocated_tokens = 0;
    }

    pub(crate) fn finish_lease(&mut self, owner: Uuid, mut lease: BlockRequestLease) {
        lease.debug_assert_owner(owner);
        self.release_lease_entries(&mut lease);
        lease.allocated_tokens = 0;
    }

    fn release_lease_entries(&mut self, lease: &mut BlockRequestLease) {
        for entry in lease.entries.iter_mut().rev() {
            if let Some(copy) = entry.copy.take() {
                self.pool.release(copy);
            }
            entry.pending_cache = false;
        }
    }

    fn commit_lease_range(
        &mut self,
        lease: &mut BlockRequestLease,
        start: usize,
        end: usize,
        reservation: &mut BlockReservation,
        cache_fresh: bool,
        sequence: Option<&RequestSequence>,
    ) {
        assert!(start <= end && end <= lease.entries.len());
        let prefix_len = reservation.len() - reservation.fresh_len();
        let mut prefix_copies = self.pool.activate_prefix(reservation);
        assert_eq!(prefix_copies.len(), prefix_len);
        let materialize_store_events = self.materialize_store_events();
        let mut stores =
            (cache_fresh && materialize_store_events).then(|| Vec::with_capacity(end - start));

        for (offset, position) in (start..end).enumerate() {
            let parent_hash = position
                .checked_sub(1)
                .and_then(|parent| lease.entries[parent].identity.sequence_hash);
            let entry = &mut lease.entries[position];
            assert!(
                entry.copy.is_none(),
                "native lease entry is already resident"
            );
            if offset < prefix_len {
                let (hash, copy) = prefix_copies
                    .next()
                    .expect("prefix reservation returned too few copies");
                assert_eq!(
                    entry.identity.sequence_hash,
                    Some(hash),
                    "reserved prefix hash changed before activation"
                );
                entry.copy = Some(copy);
                entry.pending_cache = false;
                if let Some(stores) = &mut stores {
                    stores.push(None);
                }
                continue;
            }

            let Some(hash) = entry.identity.sequence_hash else {
                entry.copy = Some(self.pool.allocate_private(reservation));
                entry.pending_cache = false;
                if let Some(stores) = &mut stores {
                    stores.push(None);
                }
                continue;
            };
            if cache_fresh && self.enable_prefix_caching {
                let (copy, became_visible) = self.pool.allocate_cached(reservation, hash);
                entry.copy = Some(copy);
                entry.pending_cache = false;
                if let Some(stores) = &mut stores {
                    stores.push(became_visible.then(|| StoredBlock {
                        hash,
                        metadata: PendingStore {
                            parent_hash,
                            local_hash: entry.identity.local_hash,
                            token_ids: sequence.and_then(|seq| seq.block_token_ids(position)),
                        },
                    }));
                }
            } else {
                entry.copy = Some(self.pool.allocate_private(reservation));
                entry.pending_cache = self.enable_prefix_caching;
                if let Some(stores) = &mut stores {
                    stores.push(None);
                }
            }
        }
        assert!(prefix_copies.next().is_none());
        if let Some(stores) = stores {
            self.publish_store_sequence(stores);
        }
    }

    pub(crate) fn get_lease_prefill_cost(
        &self,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
    ) -> PrefillCost {
        let (overlap_blocks, active_overlap_blocks) =
            if self.enable_prefix_caching && sequence.enable_prefix_caching() {
                let mut overlap = 0;
                let mut active = 0;
                for entry in &lease.entries {
                    let Some(hash) = entry.identity.sequence_hash else {
                        break;
                    };
                    let Some(hit) = self.pool.prefix_hit(hash) else {
                        break;
                    };
                    overlap += 1;
                    active += usize::from(hit.is_active);
                }
                (overlap, active)
            } else {
                (0, 0)
            };
        let new_blocks = lease.entries.len() - overlap_blocks;
        let cached_tokens = (overlap_blocks * self.block_size).min(sequence.len());
        let active_cached_tokens = (active_overlap_blocks * self.block_size).min(sequence.len());
        PrefillCost {
            new_blocks,
            new_tokens: sequence.len() - cached_tokens,
            cached_tokens,
            active_cached_tokens,
        }
    }

    pub(crate) fn reserve_decode_blocks(
        &mut self,
        count: usize,
    ) -> VllmAcquire<DecodeBlockReservation> {
        let Some(outcome) = self.pool.reserve(&[], count) else {
            return VllmAcquire::CapacityExhausted;
        };
        self.publish_removed(outcome.removed);
        VllmAcquire::Ready(DecodeBlockReservation {
            pool: outcome.reservation,
        })
    }

    pub(crate) fn release_decode_reservation(&mut self, reservation: DecodeBlockReservation) {
        self.pool.cancel(reservation.pool);
    }

    pub(crate) fn cancel_destination(&mut self, reservation: DestinationReservation) {
        self.pool.cancel(reservation.pool);
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
            KvEventData::Stored(StoredBlocks {
                parent_hash,
                start_position: None,
                blocks: full_blocks
                    .into_iter()
                    .enumerate()
                    .map(|(index, hash)| KvBlock {
                        block_hash: hash,
                        tokens_hash: local_hashes.get(index).copied().unwrap_or_default(),
                        token_ids: token_ids.as_ref().and_then(|ids| ids.get(index).cloned()),
                    })
                    .collect(),
            })
        } else {
            KvEventData::Removed {
                block_hashes: full_blocks,
            }
        };
        let event = KvEvent {
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

    pub(crate) fn num_inactive_blocks(&self) -> usize {
        self.pool.num_inactive()
    }

    pub(crate) fn max_capacity(&self) -> usize {
        self.pool.capacity()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::engine::common::protocols::KvCacheEventSink;

    #[derive(Default)]
    struct CapturingNativeSink {
        events: Mutex<Vec<KvEvent>>,
    }

    impl CapturingNativeSink {
        fn take(&self) -> Vec<KvEvent> {
            std::mem::take(&mut *self.events.lock().unwrap())
        }
    }

    impl KvCacheEventSink for CapturingNativeSink {
        fn publish(&self, event: KvEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    fn request(
        owner: Uuid,
        hashes: &[u64],
        emit_token_ids: bool,
    ) -> (RequestSequence, BlockRequestLease) {
        let tokens = (0..hashes.len() * 4).map(|token| token as u32).collect();
        let (sequence, _) = RequestSequence::new(tokens, 0, 0, 4, true, true, emit_token_ids, None);
        let identities = hashes
            .iter()
            .copied()
            .map(|hash| BlockIdentity {
                sequence_hash: Some(hash),
                local_hash: Some(hash + 100),
            })
            .collect();
        (sequence, BlockRequestLease::new(owner, identities))
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
        let mut leases = Vec::new();
        for owner in [Uuid::from_u128(1), Uuid::from_u128(2)] {
            let (mut sequence, mut lease) = request(owner, &[7], false);
            ready(manager.allocate_lease(owner, &mut lease, 4, 0));
            manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 4);
            leases.push(lease);
        }
        assert_eq!(manager.num_active_blocks(), 2);
        let third = Uuid::from_u128(3);
        let (_, mut third_lease) = request(third, &[8], false);
        assert!(matches!(
            manager.allocate_lease(third, &mut third_lease, 4, 0),
            VllmAcquire::CapacityExhausted
        ));
        assert_eq!(third_lease.allocated_tokens(), 0);
        assert_eq!(third_lease.resident_block_count(), 0);
    }

    #[test]
    fn authorized_prefix_reuses_one_physical_copy() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let first = Uuid::from_u128(1);
        let (mut first_sequence, mut first_lease) = request(first, &[7], false);
        ready(manager.allocate_lease(first, &mut first_lease, 4, 0));
        manager.finalize_lease_computed_prefix(first, &mut first_sequence, &mut first_lease, 0, 4);
        manager.finish_lease(first, first_lease);

        let second = Uuid::from_u128(2);
        let (_, mut second_lease) = request(second, &[7], false);
        ready(manager.allocate_lease(second, &mut second_lease, 4, 1));
        assert_eq!(manager.num_active_blocks(), 1);
        assert_eq!(manager.num_inactive_blocks(), 0);
    }

    #[test]
    #[should_panic(expected = "only a request's first allocation may reuse a prefix")]
    fn later_native_allocation_rejects_prefix_reuse() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(3);
        let (_, mut lease) = request(owner, &[7, 8], false);
        ready(manager.allocate_lease(owner, &mut lease, 4, 0));

        let _ = manager.allocate_lease(owner, &mut lease, 8, 1);
    }

    #[test]
    #[should_panic(expected = "reusable prefix exceeds the newly allocated block range")]
    fn native_allocation_rejects_excessive_reusable_prefix() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(4);
        let (_, mut lease) = request(owner, &[7, 8], false);

        let _ = manager.allocate_lease(owner, &mut lease, 4, 2);
    }

    #[test]
    fn full_block_is_hidden_until_computed() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        let (mut sequence, mut lease) = request(owner, &[7], false);
        ready(manager.allocate_lease(owner, &mut lease, 4, 0));
        assert!(manager.pool.prefix_hit(7).is_none());
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 4);
        assert!(manager.pool.prefix_hit(7).is_some());
    }

    #[test]
    fn finalization_only_visits_blocks_completed_by_this_decision() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        let (mut sequence, mut lease) = request(owner, &[7, 8], false);
        ready(manager.allocate_lease(owner, &mut lease, 8, 0));

        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 4);
        assert!(manager.pool.prefix_hit(7).is_some());
        assert!(manager.pool.prefix_hit(8).is_none());

        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 4, 8);
        assert!(manager.pool.prefix_hit(8).is_some());
    }

    #[test]
    fn finalization_handles_unaligned_decision_boundaries() {
        let mut manager =
            VllmKvManager::new_with_event_sink(3, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        let (mut sequence, mut lease) = request(owner, &[7, 8, 9], false);
        ready(manager.allocate_lease(owner, &mut lease, 12, 0));

        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 3, 9);
        assert!(manager.pool.prefix_hit(7).is_some());
        assert!(manager.pool.prefix_hit(8).is_some());
        assert!(manager.pool.prefix_hit(9).is_none());
    }

    #[test]
    fn cached_prefix_watermark_finalizes_only_the_fresh_suffix() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let seed = Uuid::from_u128(1);
        let (mut seed_sequence, mut seed_lease) = request(seed, &[7], false);
        ready(manager.allocate_lease(seed, &mut seed_lease, 4, 0));
        manager.finalize_lease_computed_prefix(seed, &mut seed_sequence, &mut seed_lease, 0, 4);
        manager.finish_lease(seed, seed_lease);

        let owner = Uuid::from_u128(2);
        let (mut sequence, mut lease) = request(owner, &[7, 8], false);
        ready(manager.allocate_lease(owner, &mut lease, 8, 1));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 4, 8);
        assert!(manager.pool.prefix_hit(7).is_some());
        assert!(manager.pool.prefix_hit(8).is_some());
    }

    #[test]
    fn request_release_evicts_leaf_before_parent() {
        let mut manager =
            VllmKvManager::new_with_event_sink(2, 4, true, KvEventPublishers::default(), 0);
        let owner = Uuid::from_u128(1);
        let (mut sequence, mut lease) = request(owner, &[7, 8], false);
        ready(manager.allocate_lease(owner, &mut lease, 8, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 0, 8);
        manager.finish_lease(owner, lease);

        let next = Uuid::from_u128(2);
        let (_, mut next_lease) = request(next, &[9], false);
        ready(manager.allocate_lease(next, &mut next_lease, 4, 0));

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
        let sink = Arc::new(CapturingNativeSink::default());
        let publishers = KvEventPublishers::new(Some(sink.clone()));
        let mut manager = VllmKvManager::new_with_event_sink(2, 4, true, publishers, 3);
        let owner = Uuid::from_u128(1);
        let token_ids = [vec![4, 5, 6, 7]];
        let (mut sequence, mut lease) = request(owner, &[6, 7], true);
        ready(manager.allocate_lease(owner, &mut lease, 8, 0));
        manager.finalize_lease_computed_prefix(owner, &mut sequence, &mut lease, 4, 8);

        let mut events = sink.take();
        assert_eq!(events.len(), 1);
        let event = events.pop().unwrap();
        assert_eq!(event.event_id, 0);
        assert_eq!(event.dp_rank, 3);
        let KvEventData::Stored(stored) = event.data else {
            panic!("expected Stored event")
        };
        assert_eq!(stored.parent_hash, Some(6));
        assert_eq!(stored.blocks.len(), 1);
        assert_eq!(stored.blocks[0].block_hash, 7);
        assert_eq!(stored.blocks[0].tokens_hash, 107);
        assert_eq!(stored.blocks[0].token_ids, Some(token_ids[0].clone()));
    }
}
