// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-facing facade over the native vLLM-style G1 manager.
//!
//! AISimulate owns a single native G1 implementation. External tiers and
//! provider selection are deliberately absent here.

use uuid::Uuid;

use crate::engine::common::protocols::{KvEventPublishers, PrefillCost};
use crate::engine::common::sequence::RequestSequence;

use super::G1Acquire;
use super::vllm_backend::{
    BlockRequestLease, DecodeBlockReservation as VllmDecodeBlockReservation,
    DestinationReservation as VllmDestinationReservation, VllmAcquire, VllmKvManager,
};

fn into_g1_acquire<T>(outcome: VllmAcquire<T>) -> G1Acquire<T> {
    match outcome {
        VllmAcquire::Ready(value) => G1Acquire::Ready(value),
        VllmAcquire::CapacityExhausted => G1Acquire::CapacityExhausted,
    }
}

pub(crate) struct DecodeBlockReservation {
    inner: VllmDecodeBlockReservation,
}

pub(crate) struct DestinationReservation {
    inner: VllmDestinationReservation,
}

impl DestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self, block_size: usize) -> usize {
        self.inner.transferable_prompt_tokens(block_size)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Native GPU-block accounting shared by vLLM and TensorRT-LLM schedulers.
pub(crate) struct G1Manager {
    inner: VllmKvManager,
}

impl G1Manager {
    pub(crate) fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self::new_with_caching(max_capacity, block_size, kv_event_publishers, dp_rank, true)
    }

    pub(crate) fn new_with_caching(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        enable_prefix_caching: bool,
    ) -> Self {
        Self {
            inner: VllmKvManager::new_with_event_sink(
                max_capacity,
                block_size,
                enable_prefix_caching,
                kv_event_publishers,
                dp_rank,
            ),
        }
    }

    pub(crate) fn allocate_native(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        cumulative_tokens: usize,
        reusable_prefix_blocks: usize,
    ) -> G1Acquire<usize> {
        into_g1_acquire(self.inner.allocate_lease(
            owner,
            lease,
            cumulative_tokens,
            reusable_prefix_blocks,
        ))
    }

    pub(crate) fn finalize_native_computed_prefix(
        &mut self,
        owner: Uuid,
        computed_before: usize,
        computed_after: usize,
        sequence: &mut RequestSequence,
        lease: &mut BlockRequestLease,
    ) {
        self.inner.finalize_lease_computed_prefix(
            owner,
            sequence,
            lease,
            computed_before,
            computed_after,
        );
    }

    pub(crate) fn preempt_native(&mut self, owner: Uuid, lease: &mut BlockRequestLease) {
        self.inner.preempt_lease(owner, lease);
    }

    pub(crate) fn finish_native(&mut self, owner: Uuid, lease: BlockRequestLease) {
        self.inner.finish_lease(owner, lease);
    }

    pub(crate) fn get_native_prefill_cost(
        &self,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
    ) -> PrefillCost {
        self.inner.get_lease_prefill_cost(sequence, lease)
    }

    pub(crate) fn reserve_native_destination_at(
        &mut self,
        owner: Uuid,
        sequence: &RequestSequence,
        lease: &BlockRequestLease,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<DestinationReservation> {
        into_g1_acquire(self.inner.reserve_destination_lease(
            owner,
            sequence,
            lease,
            eviction_now_ms,
        ))
        .map(|inner| DestinationReservation { inner })
    }

    pub(crate) fn activate_native_destination(
        &mut self,
        owner: Uuid,
        sequence: &RequestSequence,
        lease: &mut BlockRequestLease,
        reservation: DestinationReservation,
    ) {
        self.inner
            .activate_destination_lease(owner, sequence, lease, reservation.inner);
    }

    pub(crate) fn cancel_destination(&mut self, reservation: DestinationReservation) {
        self.inner.cancel_destination(reservation.inner);
    }

    pub(crate) fn reserve_decode_blocks(
        &mut self,
        count: usize,
    ) -> G1Acquire<DecodeBlockReservation> {
        into_g1_acquire(self.inner.reserve_decode_blocks(count))
            .map(|inner| DecodeBlockReservation { inner })
    }

    pub(crate) fn use_native_decode_reservation(
        &mut self,
        owner: Uuid,
        lease: &mut BlockRequestLease,
        cumulative_tokens: usize,
        reservation: &mut DecodeBlockReservation,
    ) {
        self.inner.allocate_lease_from_decode_reservation(
            owner,
            lease,
            cumulative_tokens,
            &mut reservation.inner,
        );
    }

    pub(crate) fn release_decode_reservation(&mut self, reservation: DecodeBlockReservation) {
        self.inner.release_decode_reservation(reservation.inner);
    }

    pub(crate) fn num_active_blocks(&self) -> usize {
        self.inner.num_active_blocks()
    }

    #[cfg(test)]
    pub(crate) fn num_inactive_blocks(&self) -> usize {
        self.inner.num_inactive_blocks()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::KvEventData;
    use crate::engine::common::protocols::KvEventPublishers;
    use crate::engine::scheduler::capture_kv_event_sink;

    #[test]
    fn native_finalization_publishes_parent_before_promoted_tail() {
        let owner = Uuid::from_u128(81_001);
        let (buffer, sink) = capture_kv_event_sink();
        let mut manager =
            G1Manager::new_with_event_sink(8, 4, KvEventPublishers::new(Some(sink)), 0);
        let (mut sequence, identities) = RequestSequence::new(
            (0..8).collect(),
            4,
            4,
            4,
            true,
            true,
            false,
            Some(vec![8, 9, 10, 11]),
        );
        let mut lease = BlockRequestLease::new(owner, identities);
        assert!(matches!(
            manager.allocate_native(owner, &mut lease, 8, 0),
            G1Acquire::Ready(_)
        ));
        manager.finalize_native_computed_prefix(owner, 0, 8, &mut sequence, &mut lease);

        for _ in 0..4 {
            let (_, opened_partial) = sequence.generate_token();
            if opened_partial {
                lease.append_partial();
            }
        }
        assert!(matches!(
            manager.allocate_native(owner, &mut lease, 12, 0),
            G1Acquire::Ready(_)
        ));
        manager.finalize_native_computed_prefix(owner, 8, 12, &mut sequence, &mut lease);

        let stored = buffer
            .drain()
            .into_iter()
            .filter_map(|event| match event.data {
                KvEventData::Stored(stored) => Some(stored),
                KvEventData::Removed { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].blocks.len(), 2);
        assert_eq!(stored[0].parent_hash, None);
        assert_eq!(stored[1].blocks.len(), 1);
        assert_eq!(stored[1].parent_hash, Some(stored[0].blocks[1].block_hash));
    }
}
