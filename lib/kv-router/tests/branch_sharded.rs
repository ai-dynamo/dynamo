// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::indexer::{BranchShardedIndexer, KvIndexerInterface, KvIndexerMetrics};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
    compute_seq_hash_for_block,
};
use tokio_util::sync::CancellationToken;

fn make_branch_sharded_indexer(num_shards: usize, prefix_depth: u32) -> BranchShardedIndexer {
    let token = CancellationToken::new();
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    BranchShardedIndexer::new(token, num_shards, prefix_depth, 32, metrics)
}

fn make_store_event(worker_id: u64, local_hashes: &[u64]) -> RouterEvent {
    let local_block_hashes: Vec<LocalBlockHash> =
        local_hashes.iter().copied().map(LocalBlockHash).collect();
    let seq_hashes = compute_seq_hash_for_block(&local_block_hashes);

    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: local_block_hashes
                    .iter()
                    .zip(seq_hashes.iter())
                    .map(|(&local, &seq)| KvCacheStoredBlockData {
                        tokens_hash: local,
                        block_hash: ExternalSequenceBlockHash(seq),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
            dp_rank: 0,
        },
    }
}

fn make_store_event_with_parent(
    worker_id: u64,
    prefix_hashes: &[u64],
    local_hashes: &[u64],
) -> RouterEvent {
    let prefix_block_hashes: Vec<LocalBlockHash> =
        prefix_hashes.iter().copied().map(LocalBlockHash).collect();
    let prefix_seq_hashes = compute_seq_hash_for_block(&prefix_block_hashes);
    let parent_hash = prefix_seq_hashes
        .last()
        .map(|&hash| ExternalSequenceBlockHash(hash));

    let full_hashes: Vec<u64> = prefix_hashes
        .iter()
        .chain(local_hashes.iter())
        .copied()
        .collect();
    let full_block_hashes: Vec<LocalBlockHash> =
        full_hashes.iter().copied().map(LocalBlockHash).collect();
    let full_seq_hashes = compute_seq_hash_for_block(&full_block_hashes);

    let new_block_hashes: Vec<LocalBlockHash> =
        local_hashes.iter().copied().map(LocalBlockHash).collect();
    let new_seq_hashes = &full_seq_hashes[prefix_hashes.len()..];

    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
                blocks: new_block_hashes
                    .iter()
                    .zip(new_seq_hashes.iter())
                    .map(|(&local, &seq)| KvCacheStoredBlockData {
                        tokens_hash: local,
                        block_hash: ExternalSequenceBlockHash(seq),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
            dp_rank: 0,
        },
    }
}

fn branch_hash(blocks: &[u64]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    blocks.iter().fold(FNV_OFFSET_BASIS, |mut hash, block| {
        for byte in block.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

fn make_remove_event(worker_id: u64, local_hashes: &[u64]) -> RouterEvent {
    let local_block_hashes: Vec<LocalBlockHash> =
        local_hashes.iter().copied().map(LocalBlockHash).collect();
    let seq_hashes = compute_seq_hash_for_block(&local_block_hashes);

    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: seq_hashes
                    .iter()
                    .map(|&hash| ExternalSequenceBlockHash(hash))
                    .collect(),
            }),
            dp_rank: 0,
        },
    }
}

fn find_boundary_shifting_sequence(num_shards: usize, prefix_depth: usize) -> Vec<u64> {
    assert!(prefix_depth >= 2);

    for start in 1..128u64 {
        let seq: Vec<u64> = (start..start + prefix_depth as u64).collect();
        let partial_shard = (branch_hash(&seq[..prefix_depth - 1]) as usize) % num_shards;
        let final_shard = (branch_hash(&seq[..prefix_depth]) as usize) % num_shards;
        if partial_shard != final_shard {
            return seq;
        }
    }

    panic!("failed to find a prefix that changes shard assignment");
}

async fn flush_and_settle(index: &dyn KvIndexerInterface) {
    index.flush().await;
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
}

#[tokio::test]
async fn branch_sharded_indexer_handles_short_queries_after_root_store() {
    let index = make_branch_sharded_indexer(4, 3);

    index
        .apply_event(make_store_event(7, &[11, 12, 13, 14]))
        .await;
    flush_and_settle(&index).await;

    let short_scores = index.find_matches(vec![LocalBlockHash(11)]).await.unwrap();
    assert_eq!(short_scores.scores[&WorkerWithDpRank::from_worker_id(7)], 1);

    let medium_scores = index
        .find_matches(vec![LocalBlockHash(11), LocalBlockHash(12)])
        .await
        .unwrap();
    assert_eq!(
        medium_scores.scores[&WorkerWithDpRank::from_worker_id(7)],
        2
    );

    let full_scores = index
        .find_matches(vec![
            LocalBlockHash(11),
            LocalBlockHash(12),
            LocalBlockHash(13),
        ])
        .await
        .unwrap();
    assert_eq!(full_scores.scores[&WorkerWithDpRank::from_worker_id(7)], 3);
}

#[tokio::test]
async fn branch_sharded_indexer_replays_prefix_chain_when_shard_changes() {
    let index = make_branch_sharded_indexer(8, 3);
    let sequence = find_boundary_shifting_sequence(8, 3);
    let prefix = &sequence[..2];
    let continuation = &sequence[2..3];

    index.apply_event(make_store_event(23, prefix)).await;
    flush_and_settle(&index).await;

    index
        .apply_event(make_store_event_with_parent(23, prefix, continuation))
        .await;
    flush_and_settle(&index).await;

    let full_query: Vec<LocalBlockHash> = sequence.iter().copied().map(LocalBlockHash).collect();
    let full_scores = index.find_matches(full_query).await.unwrap();
    assert_eq!(full_scores.scores[&WorkerWithDpRank::from_worker_id(23)], 3);

    let partial_query: Vec<LocalBlockHash> = prefix.iter().copied().map(LocalBlockHash).collect();
    let partial_scores = index.find_matches(partial_query).await.unwrap();
    assert_eq!(
        partial_scores.scores[&WorkerWithDpRank::from_worker_id(23)],
        2
    );
}

#[tokio::test]
async fn branch_sharded_indexer_removes_replayed_prefix_from_all_resident_shards() {
    let index = make_branch_sharded_indexer(8, 3);
    let sequence = find_boundary_shifting_sequence(8, 3);
    let prefix = &sequence[..2];
    let continuation = &sequence[2..3];

    index.apply_event(make_store_event(31, prefix)).await;
    flush_and_settle(&index).await;

    index
        .apply_event(make_store_event_with_parent(31, prefix, continuation))
        .await;
    flush_and_settle(&index).await;

    index.apply_event(make_remove_event(31, prefix)).await;
    flush_and_settle(&index).await;

    let partial_query: Vec<LocalBlockHash> = prefix.iter().copied().map(LocalBlockHash).collect();
    let partial_scores = index.find_matches(partial_query).await.unwrap();
    assert!(
        partial_scores
            .scores
            .get(&WorkerWithDpRank::from_worker_id(31))
            .is_none()
    );
}
