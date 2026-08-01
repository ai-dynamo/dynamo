// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo observation conversion shared by offline replay and Live Mocker.

use aisimulate_engine::{ForwardPassMetrics, KvEvent, KvEventData};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash,
};

use crate::common::protocols::ForwardPassSnapshot;

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

pub(crate) fn dynamo_forward_pass_snapshot(
    dp_rank: u32,
    metrics: ForwardPassMetrics,
) -> ForwardPassSnapshot {
    ForwardPassSnapshot {
        dp_rank,
        num_prefill_requests: metrics.num_prefill_requests,
        sum_prefill_tokens: metrics.sum_prefill_tokens,
        var_prefill_length: metrics.var_prefill_length,
        sum_prefill_kv_tokens: metrics.sum_prefill_kv_tokens,
        num_decode_requests: metrics.num_decode_requests,
        sum_decode_kv_tokens: metrics.sum_decode_kv_tokens,
        var_decode_kv_tokens: metrics.var_decode_kv_tokens,
        num_queued_prefill: metrics.num_queued_prefill,
        sum_queued_prefill_tokens: metrics.sum_queued_prefill_tokens,
        var_queued_prefill_length: metrics.var_queued_prefill_length,
        num_queued_decode: metrics.num_queued_decode,
        sum_queued_decode_kv_tokens: metrics.sum_queued_decode_kv_tokens,
        var_queued_decode_kv_tokens: metrics.var_queued_decode_kv_tokens,
        wall_time_secs: metrics.duration_ms / 1_000.0,
        ..ForwardPassSnapshot::default()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aisimulate_engine::{KvBlock, StoredBlocks};
    use dynamo_kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
    use dynamo_kv_router::protocols::{
        KvCacheEventData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
    };
    use tokio_util::sync::CancellationToken;

    use super::*;

    fn stored_event(
        event_id: u64,
        dp_rank: u32,
        parent_hash: Option<u64>,
        start_position: usize,
        blocks: &[(u64, u64, &[u32])],
    ) -> KvEvent {
        KvEvent {
            event_id,
            dp_rank,
            data: KvEventData::Stored(StoredBlocks {
                parent_hash,
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

    #[tokio::test]
    async fn converted_store_and_remove_events_apply_cleanly_to_the_dynamo_indexer() {
        let cancel = CancellationToken::new();
        let indexer = KvIndexer::new_with_pruning(
            cancel.clone(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            None,
        );
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
        indexer.apply_event(RouterEvent::new(7, first)).await;
        indexer.apply_event(RouterEvent::new(7, second)).await;
        indexer.flush().await;

        let matches = indexer
            .find_matches(vec![LocalBlockHash(11), LocalBlockHash(22)])
            .await
            .unwrap();
        assert_eq!(matches.scores.get(&worker), Some(&2));

        let removed = dynamo_kv_event(KvEvent {
            event_id: 3,
            dp_rank: 3,
            data: KvEventData::Removed {
                block_hashes: vec![101, 102],
            },
        })
        .0;
        indexer.apply_event(RouterEvent::new(7, removed)).await;
        indexer.flush().await;
        let matches = indexer
            .find_matches(vec![LocalBlockHash(11), LocalBlockHash(22)])
            .await
            .unwrap();
        assert!(matches.scores.is_empty());

        cancel.cancel();
    }
}
