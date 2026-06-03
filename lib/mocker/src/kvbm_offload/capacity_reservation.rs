// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Destination-capacity reservation helpers for mock offload pipelines.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use kvbm_engine::offload::{
    EvalContext, OffloadPolicy, PolicyBatchFuture, PolicyFuture, sync_batch_result, sync_result,
};
use kvbm_logical::blocks::BlockMetadata;
use kvbm_logical::manager::BlockManager;

/// Shared destination-capacity reservations for lower-tier pipelines.
///
/// Policy evaluation can run concurrently across workers. `BlockManager`
/// exposes only currently-free blocks, so accepted-but-not-yet-registered
/// transfers need an atomic side reservation to avoid over-admitting into a
/// shared tier.
#[derive(Debug, Default)]
pub(crate) struct CapacityReservations {
    reserved_blocks: AtomicUsize,
}

impl CapacityReservations {
    pub(crate) fn reserved_blocks(&self) -> usize {
        self.reserved_blocks.load(Ordering::Acquire)
    }

    pub(crate) fn try_reserve(self: &Arc<Self>, available_blocks: usize, blocks: usize) -> bool {
        if blocks == 0 {
            return true;
        }

        self.reserved_blocks
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                let next = reserved.checked_add(blocks)?;
                (next <= available_blocks).then_some(next)
            })
            .is_ok()
    }

    pub(crate) fn try_reserve_up_to(&self, available_blocks: usize, blocks: usize) -> usize {
        if blocks == 0 {
            return 0;
        }

        let mut reserved = self.reserved_blocks.load(Ordering::Acquire);
        loop {
            let free = available_blocks.saturating_sub(reserved);
            let to_reserve = blocks.min(free);
            if to_reserve == 0 {
                return 0;
            }
            let next = reserved + to_reserve;
            match self.reserved_blocks.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return to_reserve,
                Err(actual) => reserved = actual,
            }
        }
    }

    pub(crate) fn release(&self, blocks: usize) {
        if blocks == 0 {
            return;
        }
        let _ =
            self.reserved_blocks
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                    Some(reserved.saturating_sub(blocks))
                });
    }
}

#[derive(Debug)]
pub(crate) struct CapacityReservationGuard {
    reservations: Arc<CapacityReservations>,
    blocks: usize,
}

impl CapacityReservationGuard {
    pub(crate) fn new(reservations: Arc<CapacityReservations>, blocks: usize) -> Self {
        Self {
            reservations,
            blocks,
        }
    }

    fn release(&mut self) {
        let blocks = std::mem::take(&mut self.blocks);
        self.reservations.release(blocks);
    }
}

impl Drop for CapacityReservationGuard {
    fn drop(&mut self) {
        self.release();
    }
}

/// Destination-capacity guard for an offload pipeline.
///
/// kvbm-engine's executor currently reports allocation failure after policy
/// evaluation if the destination tier is full. For mocker, a full lower tier
/// should behave like a cache miss/drop rather than producing an executor
/// error or pinning simulated source capacity, so we filter before enqueueing
/// work that cannot allocate a destination block.
pub(crate) struct CapacityReservationPolicy<Src: BlockMetadata, Dst: BlockMetadata> {
    dst_manager: Arc<BlockManager<Dst>>,
    capacity_reservations: Arc<CapacityReservations>,
    _marker: PhantomData<Src>,
}

impl<Src: BlockMetadata, Dst: BlockMetadata> CapacityReservationPolicy<Src, Dst> {
    pub(crate) fn new(
        dst_manager: Arc<BlockManager<Dst>>,
        capacity_reservations: Arc<CapacityReservations>,
    ) -> Self {
        Self {
            dst_manager,
            capacity_reservations,
            _marker: PhantomData,
        }
    }

    fn try_reserve_capacity(&self, blocks: usize) -> bool {
        let available = self.dst_manager.available_blocks();
        self.capacity_reservations.try_reserve(available, blocks)
    }
}

impl<Src: BlockMetadata, Dst: BlockMetadata> OffloadPolicy<Src>
    for CapacityReservationPolicy<Src, Dst>
{
    fn name(&self) -> &str {
        "CapacityReservationPolicy"
    }

    fn evaluate<'a>(&'a self, _ctx: &'a EvalContext<Src>) -> PolicyFuture<'a> {
        sync_result(Ok(self.try_reserve_capacity(1)))
    }

    fn evaluate_batch<'a>(&'a self, contexts: &'a [EvalContext<Src>]) -> PolicyBatchFuture<'a> {
        if contexts.is_empty() {
            return sync_batch_result(Ok(Vec::new()));
        }

        let available = self.dst_manager.available_blocks();
        let reserved = self
            .capacity_reservations
            .try_reserve_up_to(available, contexts.len());
        let mut results = vec![true; reserved];
        results.resize(contexts.len(), false);

        sync_batch_result(Ok(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use dynamo_tokens::PositionalLineageHash;
    use futures::future::Either;
    use kvbm_engine::offload::{EvalContext, OffloadPolicy};
    use kvbm_engine::{G1 as EngineG1, G2};
    use kvbm_logical::pools::BlockDuplicationPolicy;
    use kvbm_logical::registry::BlockRegistry;

    #[test]
    fn capacity_reservation_policy_rejects_when_reserved_capacity_is_full() {
        let g2_manager = Arc::new(
            BlockManager::<G2>::builder()
                .block_count(1)
                .block_size(64)
                .registry(BlockRegistry::new())
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .with_lineage_backend()
                .build()
                .expect("BlockManager<G2> should build with valid config"),
        );
        let reservations = Arc::new(CapacityReservations::default());
        let filter =
            CapacityReservationPolicy::<EngineG1, G2>::new(g2_manager, reservations.clone());

        let ctx = EvalContext::from_external(0, PositionalLineageHash::new(10, None, 0));
        let Either::Left(result) = filter.evaluate(&ctx) else {
            panic!("CapacityReservationPolicy should evaluate synchronously");
        };
        assert!(
            futures::executor::block_on(result).expect("policy should succeed"),
            "first block should reserve the only free destination slot"
        );

        let ctx = EvalContext::from_external(1, PositionalLineageHash::new(11, None, 1));
        let Either::Left(result) = filter.evaluate(&ctx) else {
            panic!("CapacityReservationPolicy should evaluate synchronously");
        };
        assert!(
            !futures::executor::block_on(result).expect("policy should succeed"),
            "reserved destination capacity should reject another block"
        );

        reservations.release(1);
        let ctx = EvalContext::from_external(2, PositionalLineageHash::new(12, None, 2));
        let Either::Left(result) = filter.evaluate(&ctx) else {
            panic!("CapacityReservationPolicy should evaluate synchronously");
        };
        assert!(futures::executor::block_on(result).expect("policy should succeed"));
    }
}
