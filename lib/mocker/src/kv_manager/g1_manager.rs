// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary scheduler-facing facade for selecting a G1 implementation.
//!
//! This module is not a third KV manager. It forwards each operation to either
//! KVBM G1 or vLLM G1, selected when the manager is constructed. Remove this
//! facade when the KVBM G1 implementation is removed.

#[cfg(feature = "kvbm-offload")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "kvbm-offload")]
use dynamo_tokens::PositionalLineageHash;
use dynamo_tokens::blocks::UniqueBlock;
#[cfg(feature = "kvbm-offload")]
use kvbm_logical::ImmutableBlock;
use uuid::Uuid;

#[cfg(feature = "kvbm-offload")]
use crate::common::protocols::G1;
use crate::common::protocols::{
    G1Backend, KvEventPublishers, MockerEvictionBackend, MoveBlock, PrefillCost,
};
use crate::common::sequence::ActiveSequence;
#[cfg(feature = "kvbm-offload")]
use crate::kvbm_offload::{MockOffloadEngine, OffloadId};

use super::G1Acquire;
use super::kvbm_backend;
#[cfg(feature = "kvbm-offload")]
use super::kvbm_backend::{BatchSwapInOutcome, SwapInRegistrationBlock, SwapInRegistrationOutcome};
use super::vllm_backend::{
    NativeDecodeBlockReservation, NativeDestinationReservation, VllmAcquire, VllmBlockLayout,
    VllmKvManager,
};

enum G1ManagerBackend {
    Kvbm(kvbm_backend::KvManager),
    Native(VllmKvManager),
}

fn validate_plh_alignment(blocks: &[UniqueBlock], plh_count: usize) {
    let full_blocks = blocks
        .iter()
        .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
        .count();
    assert_eq!(plh_count, full_blocks, "PLHs must align with full blocks");
}

fn into_g1_acquire<T>(outcome: VllmAcquire<T>) -> G1Acquire<T> {
    match outcome {
        VllmAcquire::Ready(value) => G1Acquire::Ready(value),
        VllmAcquire::CapacityExhausted => G1Acquire::CapacityExhausted,
    }
}

fn process_vllm_event(
    manager: &mut VllmKvManager,
    owner: Uuid,
    event: &MoveBlock,
    reusable_prefix_blocks: usize,
) -> VllmAcquire<usize> {
    match event {
        MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent) => {
            validate_plh_alignment(blocks, plhs.len());
            manager.use_for_request(
                owner,
                blocks,
                local_hashes,
                token_ids.as_deref(),
                parent.as_ref(),
                reusable_prefix_blocks,
            )
        }
        MoveBlock::Deref(blocks) => {
            assert_eq!(reusable_prefix_blocks, 0);
            manager.deref_for_request(owner, blocks);
            VllmAcquire::Ready(1)
        }
        MoveBlock::Promote(uuid, hash, parent_hash, local_hash, _plh, token_ids) => {
            assert_eq!(reusable_prefix_blocks, 0);
            manager.promote_for_request(
                owner,
                *uuid,
                *hash,
                *parent_hash,
                *local_hash,
                token_ids.clone(),
            );
            VllmAcquire::Ready(1)
        }
    }
}

pub(crate) struct DecodeBlockReservation {
    inner: DecodeBlockReservationBackend,
}

enum DecodeBlockReservationBackend {
    Kvbm(kvbm_backend::DecodeBlockReservation),
    Native(NativeDecodeBlockReservation),
}

impl DecodeBlockReservation {
    pub(crate) fn len(&self) -> usize {
        match &self.inner {
            DecodeBlockReservationBackend::Kvbm(reservation) => reservation.len(),
            DecodeBlockReservationBackend::Native(reservation) => reservation.len(),
        }
    }
}

pub(crate) struct DestinationReservation {
    inner: DestinationReservationBackend,
}

enum DestinationReservationBackend {
    Kvbm(kvbm_backend::VllmDestinationReservation),
    Native(NativeDestinationReservation),
}

impl DestinationReservation {
    pub(crate) fn transferable_prompt_tokens(&self, block_size: usize) -> usize {
        match &self.inner {
            DestinationReservationBackend::Kvbm(reservation) => {
                reservation.transferable_prompt_tokens(block_size)
            }
            DestinationReservationBackend::Native(reservation) => {
                reservation.transferable_prompt_tokens(block_size)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        match &self.inner {
            DestinationReservationBackend::Kvbm(reservation) => reservation.len(),
            DestinationReservationBackend::Native(reservation) => reservation.len(),
        }
    }
}

/// Temporary facade over the G1 implementation selected by [`G1Backend`].
///
/// This is not a separate manager implementation and will be removed with the
/// KVBM G1 backend.
pub struct G1Manager {
    backend: G1ManagerBackend,
}

impl G1Manager {
    /// Constructs the default KVBM G1 backend.
    pub fn new_with_event_sink(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self::new_with_backend(
            max_capacity,
            block_size,
            kv_event_publishers,
            dp_rank,
            G1Backend::Kvbm,
        )
    }

    pub fn new_with_backend(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        backend: G1Backend,
    ) -> Self {
        Self::new_with_backend_and_caching(
            max_capacity,
            block_size,
            kv_event_publishers,
            dp_rank,
            backend,
            true,
        )
    }

    pub fn new_with_backend_and_caching(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        backend: G1Backend,
        enable_prefix_caching: bool,
    ) -> Self {
        let backend = match backend {
            G1Backend::Kvbm => {
                G1ManagerBackend::Kvbm(kvbm_backend::KvManager::new_with_event_sink(
                    max_capacity,
                    block_size,
                    kv_event_publishers,
                    dp_rank,
                ))
            }
            G1Backend::Native => G1ManagerBackend::Native(VllmKvManager::new_with_event_sink(
                max_capacity,
                block_size,
                enable_prefix_caching,
                kv_event_publishers,
                dp_rank,
            )),
        };
        Self { backend }
    }

    /// Make newly allocated full blocks prefix-cache-visible once the current
    /// scheduling decision reaches their complete token range. KVBM retains
    /// its historical eager lifecycle; the native backend mirrors vLLM's
    /// per-request `allocate_slots()` / `cache_blocks()` boundary.
    pub(crate) fn finalize_computed_prefix(
        &mut self,
        owner: Uuid,
        computed_before: usize,
        computed_after: usize,
        sequence: &mut ActiveSequence,
    ) {
        if let G1ManagerBackend::Native(manager) = &mut self.backend {
            if let Some(promote) = sequence.promote_computed_tail(computed_after) {
                assert!(
                    matches!(
                        process_vllm_event(manager, owner, &promote, 0),
                        VllmAcquire::Ready(_)
                    ),
                    "computed-tail promotion must be infallible"
                );
            }
            manager.finalize_computed_prefix(owner, computed_before, computed_after);
        }
    }

    pub fn new_with_eviction_backend(
        max_capacity: usize,
        block_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
        eviction_backend: MockerEvictionBackend,
    ) -> Self {
        Self {
            backend: G1ManagerBackend::Kvbm(kvbm_backend::KvManager::new_with_eviction_backend(
                max_capacity,
                block_size,
                kv_event_publishers,
                dp_rank,
                eviction_backend,
            )),
        }
    }

    /// Owner-aware production entrypoint. KVBM deliberately ignores `owner`;
    /// native G1 uses it to address the exact physical request block table.
    pub(crate) fn process_for_request(
        &mut self,
        owner: Uuid,
        event: &MoveBlock,
        reusable_prefix_blocks: usize,
    ) -> G1Acquire<usize> {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.process(event),
            G1ManagerBackend::Native(manager) => into_g1_acquire(process_vllm_event(
                manager,
                owner,
                event,
                reusable_prefix_blocks,
            )),
        }
    }

    /// Compatibility entrypoint for KVBM-focused unit tests. Native callers
    /// must provide request ownership through [`Self::process_for_request`].
    #[cfg(test)]
    pub(crate) fn process(&mut self, event: &MoveBlock) -> G1Acquire<usize> {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.process(event),
            G1ManagerBackend::Native(_) => {
                panic!("native G1 operations require a request owner")
            }
        }
    }

    pub(crate) fn reserve_decode_blocks(
        &mut self,
        count: usize,
    ) -> G1Acquire<DecodeBlockReservation> {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => {
                manager
                    .reserve_decode_blocks(count)
                    .map(|inner| DecodeBlockReservation {
                        inner: DecodeBlockReservationBackend::Kvbm(inner),
                    })
            }
            G1ManagerBackend::Native(manager) => {
                into_g1_acquire(manager.reserve_decode_blocks(count)).map(|inner| {
                    DecodeBlockReservation {
                        inner: DecodeBlockReservationBackend::Native(inner),
                    }
                })
            }
        }
    }

    pub(crate) fn process_decode_signal_for_request(
        &mut self,
        owner: Uuid,
        event: &MoveBlock,
        reservation: &mut DecodeBlockReservation,
    ) {
        match (&mut self.backend, &mut reservation.inner) {
            (G1ManagerBackend::Kvbm(manager), DecodeBlockReservationBackend::Kvbm(inner)) => {
                manager.process_decode_signal(event, inner);
            }
            (G1ManagerBackend::Native(manager), DecodeBlockReservationBackend::Native(inner)) => {
                match event {
                    MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent) => {
                        validate_plh_alignment(blocks, plhs.len());
                        manager.use_decode_reservation_for_request(
                            owner,
                            blocks,
                            local_hashes,
                            token_ids.as_deref(),
                            parent.as_ref(),
                            inner,
                        );
                    }
                    _ => assert!(
                        matches!(
                            process_vllm_event(manager, owner, event, 0),
                            VllmAcquire::Ready(_)
                        ),
                        "non-Use decode signal must be infallible"
                    ),
                }
            }
            _ => panic!("decode reservation belongs to a different G1 backend"),
        }
    }

    pub(crate) fn release_decode_reservation(&mut self, reservation: DecodeBlockReservation) {
        match (&mut self.backend, reservation.inner) {
            (G1ManagerBackend::Kvbm(_), DecodeBlockReservationBackend::Kvbm(inner)) => drop(inner),
            (G1ManagerBackend::Native(manager), DecodeBlockReservationBackend::Native(inner)) => {
                manager.release_decode_reservation(inner);
            }
            _ => panic!("decode reservation belongs to a different G1 backend"),
        }
    }

    pub(crate) fn reserve_destination_at(
        &mut self,
        owner: Uuid,
        sequence: &ActiveSequence,
        eviction_now_ms: Option<f64>,
    ) -> G1Acquire<DestinationReservation> {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => manager
                .reserve_destination_at(sequence, eviction_now_ms)
                .map(|inner| DestinationReservation {
                    inner: DestinationReservationBackend::Kvbm(inner),
                }),
            G1ManagerBackend::Native(manager) => {
                let layout = match sequence.prepare_allocation(sequence.num_input_tokens()) {
                    None => None,
                    Some(MoveBlock::Use(blocks, local_hashes, plhs, token_ids, parent)) => {
                        validate_plh_alignment(&blocks, plhs.len());
                        Some(VllmBlockLayout::new(
                            blocks,
                            local_hashes,
                            token_ids,
                            parent,
                        ))
                    }
                    Some(_) => panic!("destination allocation must be a Use signal"),
                };
                into_g1_acquire(manager.reserve_destination_at(owner, layout, eviction_now_ms)).map(
                    |inner| DestinationReservation {
                        inner: DestinationReservationBackend::Native(inner),
                    },
                )
            }
        }
    }

    pub(crate) fn activate_destination(&mut self, reservation: DestinationReservation) {
        match (&mut self.backend, reservation.inner) {
            (G1ManagerBackend::Kvbm(manager), DestinationReservationBackend::Kvbm(inner)) => {
                manager.activate_destination(inner);
            }
            (G1ManagerBackend::Native(manager), DestinationReservationBackend::Native(inner)) => {
                manager.activate_destination(inner);
            }
            _ => panic!("destination reservation belongs to a different G1 backend"),
        }
    }

    pub(crate) fn cancel_destination(&mut self, reservation: DestinationReservation) {
        match (&mut self.backend, reservation.inner) {
            (G1ManagerBackend::Kvbm(_), DestinationReservationBackend::Kvbm(inner)) => drop(inner),
            (G1ManagerBackend::Native(manager), DestinationReservationBackend::Native(inner)) => {
                manager.cancel_destination(inner);
            }
            _ => panic!("destination reservation belongs to a different G1 backend"),
        }
    }

    pub fn num_active_blocks(&self) -> usize {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.num_active_blocks(),
            G1ManagerBackend::Native(manager) => manager.num_active_blocks(),
        }
    }

    pub fn num_active_block_refs(&self) -> usize {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.num_active_block_refs(),
            G1ManagerBackend::Native(manager) => manager.num_active_block_refs(),
        }
    }

    pub fn num_inactive_blocks(&self) -> usize {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.num_inactive_blocks(),
            G1ManagerBackend::Native(manager) => manager.num_inactive_blocks(),
        }
    }

    pub fn get_active_perc(&self) -> f64 {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.get_active_perc(),
            G1ManagerBackend::Native(manager) => manager.get_active_perc(),
        }
    }

    pub fn max_capacity(&self) -> usize {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.max_capacity(),
            G1ManagerBackend::Native(manager) => manager.max_capacity(),
        }
    }

    pub fn block_size(&self) -> usize {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.block_size(),
            G1ManagerBackend::Native(manager) => manager.block_size(),
        }
    }

    pub fn dp_rank(&self) -> u32 {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.dp_rank(),
            G1ManagerBackend::Native(manager) => manager.dp_rank(),
        }
    }

    #[cfg(test)]
    pub(crate) fn request_block_count(&self, owner: Uuid, sequence: &ActiveSequence) -> usize {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.active_block_ids(sequence).len(),
            G1ManagerBackend::Native(manager) => manager.request_block_count(owner),
        }
    }

    pub fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.get_prefill_cost(sequence),
            G1ManagerBackend::Native(manager) => manager.get_prefill_cost(sequence),
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub fn attach_new_offload_engine(
        &mut self,
        engine: MockOffloadEngine,
    ) -> Arc<Mutex<MockOffloadEngine>> {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.attach_new_offload_engine(engine),
            G1ManagerBackend::Native(_) => {
                panic!("legacy kvbm-offload cannot be attached to native G1")
            }
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub fn has_offload_engine(&self) -> bool {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.has_offload_engine(),
            G1ManagerBackend::Native(_) => false,
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub fn tick_offload_engine(&mut self, now_ms: f64) {
        if let G1ManagerBackend::Kvbm(manager) = &mut self.backend {
            manager.tick_offload_engine(now_ms);
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub fn earliest_offload_deadline(&self) -> Option<f64> {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.earliest_offload_deadline(),
            G1ManagerBackend::Native(_) => None,
        }
    }

    pub(crate) fn refresh_offload_dependency(
        &self,
        dependency: super::OffloadDependency,
    ) -> Option<super::OffloadDependency> {
        match &self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.refresh_offload_dependency(dependency),
            G1ManagerBackend::Native(_) => None,
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub fn try_batch_swap_in(
        &mut self,
        remaining_plhs: &[PositionalLineageHash],
        prefix_pins: Vec<ImmutableBlock<G1>>,
        now_ms: Option<f64>,
    ) -> BatchSwapInOutcome {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => {
                manager.try_batch_swap_in(remaining_plhs, prefix_pins, now_ms)
            }
            G1ManagerBackend::Native(_) => {
                drop(prefix_pins);
                BatchSwapInOutcome::NoHits
            }
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn cancel_swap_in(&mut self, id: OffloadId) -> bool {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.cancel_swap_in(id),
            G1ManagerBackend::Native(_) => false,
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn register_completed_swap_in(
        &mut self,
        id: OffloadId,
        entries: Vec<SwapInRegistrationBlock>,
        parent_hash: Option<u64>,
    ) -> SwapInRegistrationOutcome {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => {
                manager.register_completed_swap_in(id, entries, parent_hash)
            }
            G1ManagerBackend::Native(_) => {
                panic!("native G1 cannot complete a legacy KVBM swap-in")
            }
        }
    }

    #[cfg(feature = "kvbm-offload")]
    pub(crate) fn try_pin_g1_prefix(
        &mut self,
        prefix_plhs: &[PositionalLineageHash],
    ) -> Option<Vec<ImmutableBlock<G1>>> {
        match &mut self.backend {
            G1ManagerBackend::Kvbm(manager) => manager.try_pin_g1_prefix(prefix_plhs),
            G1ManagerBackend::Native(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "PLHs must align with full blocks")]
    fn vllm_boundary_rejects_misaligned_lineage_metadata() {
        let mut manager =
            G1Manager::new_with_backend(2, 4, KvEventPublishers::default(), 0, G1Backend::Native);
        let event = MoveBlock::Use(
            vec![UniqueBlock::FullBlock(7)],
            vec![107],
            Vec::new(),
            None,
            None,
        );

        let _ = manager.process_for_request(Uuid::from_u128(1), &event, 0);
    }
}
