// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for AISimulate: native Router integration with AISim-owned types.
// Derived from https://github.com/ai-dynamo/dynamo/blob/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/lib/mocker/src/replay/offline/extensions/kv_events/mod.rs
// Derived from https://github.com/ai-dynamo/dynamo/blob/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/lib/mocker/src/engine_observations.rs

use aisimulate_core::engine::{KvEvent, KvEventData, KvEventTier};
use aisimulate_core::replay::{
    EngineEventBatch, KvIngestEventEncoder, ReplayEngineObservation, WorkerStage,
};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
};

#[derive(Debug, Default)]
pub struct RouterEventBatch(pub Vec<RouterEvent>);

impl EngineEventBatch for RouterEventBatch {
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
    }
}

#[derive(Debug, Default)]
pub struct RouterEventObservation;

impl ReplayEngineObservation for RouterEventObservation {
    type Batch = RouterEventBatch;

    const CAPTURE_ENGINE_KV_EVENTS: bool = true;

    fn capture_engine_kv_events(stage: WorkerStage) -> bool {
        !matches!(stage, WorkerStage::Decode)
    }

    fn observe_engine_events(
        stage: WorkerStage,
        worker_id: usize,
        _dp_rank: u32,
        events: Vec<KvEvent>,
    ) -> Self::Batch {
        // Disaggregated decode placement is load-only. Aggregated and prefill
        // placement consume engine mutations at the shared completion boundary.
        if matches!(stage, WorkerStage::Decode) {
            return RouterEventBatch::default();
        }
        let worker_id = u64::try_from(worker_id)
            .expect("logical replay worker id must fit the Dynamo Router wire type");
        RouterEventBatch(
            events
                .into_iter()
                .map(|event| {
                    let tier = match event.tier {
                        KvEventTier::Device => StorageTier::Device,
                        KvEventTier::HostPinned => StorageTier::HostPinned,
                    };
                    let (event, _) = dynamo_kv_event(event);
                    RouterEvent::with_storage_tier(worker_id, event, tier)
                })
                .collect(),
        )
    }

    fn stored_hashes(batch: &Self::Batch) -> Vec<u64> {
        batch
            .0
            .iter()
            .flat_map(|event| match &event.event.data {
                dynamo_kv_router::protocols::KvCacheEventData::Stored(store) => {
                    store.blocks.as_slice()
                }
                dynamo_kv_router::protocols::KvCacheEventData::Removed(_)
                | dynamo_kv_router::protocols::KvCacheEventData::Cleared => &[],
            })
            .map(|block| block.tokens_hash.0)
            .collect()
    }

    fn kv_ingest_event_count(batch: &Self::Batch) -> Option<usize> {
        Some(batch.0.len())
    }

    fn encode_kv_ingest(
        batch: &Self::Batch,
        encoder: &mut KvIngestEventEncoder<'_>,
    ) -> anyhow::Result<()> {
        encode_events(encoder, &batch.0)
    }
}

fn encode_events(
    encoder: &mut KvIngestEventEncoder<'_>,
    events: &[RouterEvent],
) -> anyhow::Result<()> {
    use dynamo_kv_router::protocols::KvCacheEventData;

    for event in events {
        let (tier_tag, tier_name) = storage_tier_identity(event.storage_tier);
        encoder.begin_event(
            event.worker_id,
            event.event.dp_rank,
            tier_tag,
            tier_name,
            event.event.event_id,
        );
        match &event.event.data {
            KvCacheEventData::Stored(stored) => {
                encoder.begin_kind(0, "stored");
                encoder.put_optional_u64(stored.parent_hash.map(|hash| hash.0));
                encoder.put_optional_u32(stored.start_position);
                encoder.put_len(stored.blocks.len(), "stored KV block count")?;
                encoder.add_blocks(stored.blocks.len(), "stored KV block count")?;
                for block in &stored.blocks {
                    encoder.put_u64(block.block_hash.0);
                    encoder.put_u64(block.tokens_hash.0);
                    match &block.mm_extra_info {
                        Some(extra) => {
                            encoder.put_u8(1);
                            encoder.put_len(extra.mm_objects.len(), "multimodal object count")?;
                            for object in &extra.mm_objects {
                                encoder.put_u64(object.mm_hash);
                                encoder.put_len(object.offsets.len(), "multimodal offset count")?;
                                for &(start, end) in &object.offsets {
                                    encoder.put_len(start, "multimodal start offset")?;
                                    encoder.put_len(end, "multimodal end offset")?;
                                }
                            }
                        }
                        None => encoder.put_u8(0),
                    }
                }
            }
            KvCacheEventData::Removed(removed) => {
                encoder.begin_kind(1, "removed");
                encoder.put_len(removed.block_hashes.len(), "removed KV block count")?;
                encoder.add_blocks(removed.block_hashes.len(), "removed KV block count")?;
                for hash in &removed.block_hashes {
                    encoder.put_u64(hash.0);
                }
            }
            KvCacheEventData::Cleared => encoder.begin_kind(2, "cleared"),
        }
    }
    Ok(())
}

fn storage_tier_identity(tier: StorageTier) -> (u8, &'static str) {
    match tier {
        StorageTier::Device => (0, "device"),
        StorageTier::HostPinned => (1, "host_pinned"),
        StorageTier::Disk => (2, "disk"),
        StorageTier::External => (3, "external"),
    }
}

pub(crate) fn dynamo_kv_event(event: KvEvent) -> (KvCacheEvent, Option<Vec<Vec<u32>>>) {
    let (data, block_token_ids) = match event.data {
        KvEventData::Stored(stored) => {
            let block_token_ids = stored
                .blocks
                .iter()
                .map(|block| block.token_ids.clone())
                .collect::<Option<Vec<_>>>();
            let data = KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: stored.parent_hash.map(ExternalSequenceBlockHash),
                start_position: stored.start_position.map(|position| {
                    u32::try_from(position)
                        .expect("native KV start position exceeds the Dynamo router protocol")
                }),
                blocks: stored
                    .blocks
                    .into_iter()
                    .map(|block| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block.block_hash),
                        tokens_hash: LocalBlockHash(block.tokens_hash),
                        mm_extra_info: None,
                    })
                    .collect(),
            });
            (data, block_token_ids)
        }
        KvEventData::Removed { block_hashes } => (
            KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: block_hashes
                    .into_iter()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            }),
            None,
        ),
    };
    (
        KvCacheEvent {
            event_id: event.event_id,
            data,
            dp_rank: event.dp_rank,
        },
        block_token_ids,
    )
}

#[cfg(test)]
mod tests {
    use aisimulate_core::engine::{KvBlock, StoredBlocks};
    use dynamo_kv_router::RadixTree;
    use dynamo_kv_router::protocols::{
        KvCacheEventData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
    };

    use super::*;

    fn stored_event(
        event_id: u64,
        dp_rank: u32,
        parent_hash: Option<u64>,
        start_position: usize,
        blocks: &[(u64, u64, &[u32])],
    ) -> KvEvent {
        KvEvent {
            tier: aisimulate_core::engine::KvEventTier::Device,
            event_id,
            dp_rank,
            data: KvEventData::Stored(StoredBlocks {
                parent_hash,
                session_id: None,
                start_position: Some(start_position),
                blocks: blocks
                    .iter()
                    .map(|(block_hash, tokens_hash, token_ids)| KvBlock {
                        block_hash: *block_hash,
                        tokens_hash: *tokens_hash,
                        token_ids: Some(token_ids.to_vec()),
                    })
                    .collect(),
            }),
        }
    }

    #[test]
    fn observation_preserves_worker_rank_cache_tier_and_parent_link() {
        let mut host = stored_event(1, 3, Some(99), 7, &[(101, 11, &[1, 2, 3, 4])]);
        host.tier = KvEventTier::HostPinned;
        let mut device = stored_event(2, 3, Some(101), 8, &[(102, 22, &[5, 6, 7, 8])]);
        if let KvEventData::Stored(stored) = &mut device.data {
            stored.start_position = None;
        }
        let batch = RouterEventObservation::observe_engine_events(
            WorkerStage::Prefill,
            7,
            3,
            vec![host.clone(), device],
        );
        assert_eq!(batch.0.len(), 2);
        assert_eq!(batch.0[0].worker_id, 7);
        assert_eq!(batch.0[0].event.dp_rank, 3);
        assert_eq!(batch.0[0].storage_tier, StorageTier::HostPinned);
        assert_eq!(batch.0[1].storage_tier, StorageTier::Device);
        let KvCacheEventData::Stored(stored) = &batch.0[1].event.data else {
            panic!("expected stored device event");
        };
        assert_eq!(stored.parent_hash, Some(ExternalSequenceBlockHash(101)));
        assert_eq!(stored.start_position, None);
        assert!(!RouterEventObservation::capture_engine_kv_events(
            WorkerStage::Decode
        ));
        assert!(
            RouterEventObservation::observe_engine_events(WorkerStage::Decode, 7, 3, vec![host])
                .is_empty()
        );
    }

    #[test]
    fn stored_and_removed_events_preserve_router_identity_and_token_metadata() {
        let (event, token_ids) = dynamo_kv_event(stored_event(
            17,
            3,
            Some(99),
            7,
            &[(101, 11, &[1, 2, 3, 4]), (102, 22, &[5, 6, 7, 8])],
        ));
        assert_eq!(event.event_id, 17);
        assert_eq!(event.dp_rank, 3);
        let KvCacheEventData::Stored(stored) = event.data else {
            panic!("expected converted Stored event");
        };
        assert_eq!(stored.parent_hash, Some(ExternalSequenceBlockHash(99)));
        assert_eq!(stored.start_position, Some(7));
        assert_eq!(stored.blocks.len(), 2);
        assert_eq!(stored.blocks[0].block_hash, ExternalSequenceBlockHash(101));
        assert_eq!(stored.blocks[0].tokens_hash, LocalBlockHash(11));
        assert_eq!(stored.blocks[1].block_hash, ExternalSequenceBlockHash(102));
        assert_eq!(stored.blocks[1].tokens_hash, LocalBlockHash(22));
        assert_eq!(token_ids, Some(vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]));

        let (event, token_ids) = dynamo_kv_event(KvEvent {
            tier: aisimulate_core::engine::KvEventTier::Device,
            event_id: 18,
            dp_rank: 3,
            data: KvEventData::Removed {
                block_hashes: vec![101, 102],
            },
        });
        let KvCacheEventData::Removed(removed) = event.data else {
            panic!("expected converted Removed event");
        };
        assert_eq!(
            removed.block_hashes,
            vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102)
            ]
        );
        assert_eq!(token_ids, None);
    }

    #[test]
    fn converted_store_and_remove_events_apply_cleanly_to_the_dynamo_indexer() {
        let mut indexer = RadixTree::new();
        let worker = WorkerWithDpRank::new(7, 3);

        let first = dynamo_kv_event(stored_event(1, 3, None, 0, &[(101, 11, &[1, 2, 3, 4])])).0;
        let second = dynamo_kv_event(stored_event(
            2,
            3,
            Some(101),
            1,
            &[(102, 22, &[5, 6, 7, 8])],
        ))
        .0;
        indexer.apply_event(RouterEvent::new(7, first)).unwrap();
        indexer.apply_event(RouterEvent::new(7, second)).unwrap();

        let matches = indexer.find_matches(vec![LocalBlockHash(11), LocalBlockHash(22)], false);
        assert_eq!(matches.scores.get(&worker), Some(&2));

        let removed = dynamo_kv_event(KvEvent {
            tier: aisimulate_core::engine::KvEventTier::Device,
            event_id: 3,
            dp_rank: 3,
            data: KvEventData::Removed {
                block_hashes: vec![101, 102],
            },
        })
        .0;
        indexer.apply_event(RouterEvent::new(7, removed)).unwrap();
        let matches = indexer.find_matches(vec![LocalBlockHash(11), LocalBlockHash(22)], false);
        assert!(matches.scores.is_empty());
    }
}
