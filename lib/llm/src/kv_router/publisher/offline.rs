// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY (e2e indexer-contention campaign): the KV event publisher's outbound
//! pipeline without a runtime.
//!
//! [`OfflinePublisherPipeline`] runs engine event lists through the same stages as
//! `run_event_processor_loop` with the default batching (no timeout): store/remove dedup,
//! coalescing within one engine list, outbound event-ID assignment, and the event-plane
//! envelope split. Phantom publishers use it to replay exactly what a live worker's
//! direct-ZMQ publisher would have sent for a captured engine event stream.

use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, Placement, PlacementEvent, RouterEvent, StorageTier, WorkerId,
};

use super::DEFAULT_MAX_BATCH_BLOCKS;
use super::batching::BatchingState;
use super::dedup::{EventDedupFilter, EventDedupPolicy};
use super::sinks::{
    MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS, MAX_EVENT_PLANE_KV_EVENTS_PER_BATCH, emit,
    event_plane_event_batches,
};

pub struct OfflinePublisherPipeline {
    worker_id: WorkerId,
    batching: BatchingState,
    dedup: EventDedupFilter,
}

impl OfflinePublisherPipeline {
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            batching: BatchingState::new(DEFAULT_MAX_BATCH_BLOCKS),
            dedup: EventDedupFilter::new(),
        }
    }

    /// The outbound event ID the next emitted event receives.
    pub fn next_event_id(&self) -> u64 {
        self.batching.next_publish_id
    }

    /// Process one native engine list (one `publish_batch_with_storage_tiers` call) and return
    /// the event-plane envelopes it produces, each the `Vec<RouterEvent>` payload of one publish.
    pub async fn process_list(
        &mut self,
        events: Vec<(KvCacheEvent, StorageTier)>,
    ) -> Vec<Vec<RouterEvent>> {
        let worker_id = self.worker_id;
        let mut output = Vec::new();
        for (event, storage_tier) in events {
            let placement_event = PlacementEvent::new(
                Placement::local_worker(worker_id, event.dp_rank, storage_tier),
                event,
            );
            if !matches!(placement_event.event.data, KvCacheEventData::Cleared) {
                self.batching
                    .push(
                        placement_event,
                        &None,
                        worker_id,
                        &mut self.dedup,
                        &mut output,
                    )
                    .await;
                continue;
            }
            self.batching
                .flush(&None, worker_id, &mut self.dedup, &mut output)
                .await;
            let domain = placement_event.placement.residency_domain;
            let dp_rank = placement_event.event.dp_rank;
            self.dedup
                .clear_rank_domain(dp_rank, domain, EventDedupPolicy::RefCounted);
            emit(
                &None,
                worker_id,
                storage_tier,
                domain,
                KvCacheEvent {
                    event_id: self.batching.next_publish_id,
                    data: KvCacheEventData::Cleared,
                    dp_rank,
                },
                None,
                &mut output,
            )
            .await;
            self.batching.next_publish_id = self
                .batching
                .next_publish_id
                .checked_add(1)
                .expect("KV event publisher outbound cursor exhausted");
        }
        if self.batching.has_pending() {
            self.batching
                .flush(&None, worker_id, &mut self.dedup, &mut output)
                .await;
        }
        event_plane_event_batches(
            &output,
            MAX_EVENT_PLANE_KV_EVENTS_PER_BATCH,
            MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS,
        )
        .map(<[RouterEvent]>::to_vec)
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash,
    };

    use super::*;

    fn stored(parent: Option<u64>, block: u64) -> (KvCacheEvent, StorageTier) {
        let data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent.map(ExternalSequenceBlockHash),
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(block),
                tokens_hash: LocalBlockHash(block),
                mm_extra_info: None,
            }],
        });
        (
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: 0,
            },
            StorageTier::Device,
        )
    }

    fn removed(block: u64) -> (KvCacheEvent, StorageTier) {
        let data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(block)],
        });
        (
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: 0,
            },
            StorageTier::Device,
        )
    }

    #[tokio::test]
    async fn coalesces_within_a_list_and_numbers_events_contiguously() {
        let mut pipeline = OfflinePublisherPipeline::new(9);
        let first = pipeline
            .process_list(vec![stored(None, 1), stored(Some(1), 2), removed(1)])
            .await;
        assert_eq!(first.len(), 1);
        let ids: Vec<_> = first[0].iter().map(|event| event.event.event_id).collect();
        assert_eq!(ids, vec![1, 2]);
        assert!(first[0].iter().all(|event| event.worker_id == 9));
        assert!(matches!(
            &first[0][0].event.data,
            KvCacheEventData::Stored(store) if store.blocks.len() == 2
        ));
        // A new list never coalesces with the previous one; IDs continue.
        let second = pipeline.process_list(vec![stored(Some(2), 3)]).await;
        assert_eq!(second[0][0].event.event_id, 3);
        assert_eq!(pipeline.next_event_id(), 4);
    }
}
