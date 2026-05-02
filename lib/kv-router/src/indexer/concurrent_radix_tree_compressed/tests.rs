// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
use crate::test_utils::{
    assert_score, flush_and_settle, make_remove_event_with_parent, make_store_event,
    make_store_event_with_parent, snapshot_events, snapshot_tree,
};

#[tokio::test]
async fn test_extends_decode_tail_in_place() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);
    let worker = WorkerWithDpRank::new(0, 0);

    index.apply_event(make_store_event(0, &[1, 2, 3])).await;
    index
        .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[4]))
        .await;
    index
        .apply_event(make_store_event_with_parent(0, &[1, 2, 3, 4], &[5]))
        .await;
    index
        .apply_event(make_store_event_with_parent(0, &[1, 2, 3, 4, 5], &[6]))
        .await;
    flush_and_settle(&index).await;

    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker, 6).await;
    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_eq!(
        snapshot_tree(&index).await,
        vec![make_store_event(0, &[1, 2, 3, 4, 5, 6])]
    );
}

#[tokio::test]
async fn test_extension_downgrade_can_split_later() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let worker2 = WorkerWithDpRank::new(2, 0);

    index.apply_event(make_store_event(1, &[1, 2, 3])).await;
    index.apply_event(make_store_event(2, &[1, 2, 3])).await;
    flush_and_settle(&index).await;

    index
        .apply_event(make_store_event_with_parent(1, &[1, 2, 3], &[4, 5, 6]))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6).await;
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 3).await;

    index
        .apply_event(make_store_event_with_parent(2, &[1, 2, 3], &[7, 8]))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 3);
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6).await;
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 3).await;
    assert_score(&index, &[1, 2, 3, 7, 8], worker1, 3).await;
    assert_score(&index, &[1, 2, 3, 7, 8], worker2, 5).await;

    let expected = snapshot_events(vec![
        make_store_event(1, &[1, 2, 3]),
        make_store_event_with_parent(1, &[1, 2, 3], &[4, 5, 6]),
        make_store_event(2, &[1, 2, 3]),
        make_store_event_with_parent(2, &[1, 2, 3], &[7, 8]),
    ]);
    assert_eq!(snapshot_tree(&index).await, expected);
}

#[tokio::test]
async fn test_reuses_prefix_suffix_and_extends_to_nine() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let worker2 = WorkerWithDpRank::new(2, 0);

    index
        .apply_event(make_store_event(1, &[1, 2, 3, 4, 5, 6]))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6).await;

    index.apply_event(make_store_event(2, &[1, 2, 3])).await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6).await;
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 3).await;

    index
        .apply_event(make_store_event_with_parent(2, &[1, 2, 3], &[4, 5, 6]))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6).await;
    assert_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 6).await;

    index
        .apply_event(make_store_event_with_parent(
            2,
            &[1, 2, 3, 4, 5, 6],
            &[7, 8, 9],
        ))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &[1, 2, 3, 4, 5, 6, 7, 8, 9], worker1, 6).await;
    assert_score(&index, &[1, 2, 3, 4, 5, 6, 7, 8, 9], worker2, 9).await;

    let expected = snapshot_events(vec![
        make_store_event(1, &[1, 2, 3, 4, 5, 6]),
        make_store_event(2, &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ]);
    assert_eq!(snapshot_tree(&index).await, expected);
}

#[tokio::test]
async fn test_reuses_internal_suffix_and_extends_leaf_without_split() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);
    let worker1 = WorkerWithDpRank::new(1, 0);
    let worker2 = WorkerWithDpRank::new(2, 0);
    let one_to_10: Vec<u64> = (1..=10).collect();
    let one_to_35: Vec<u64> = (1..=35).collect();
    let one_to_40: Vec<u64> = (1..=40).collect();
    let eleven_to_40: Vec<u64> = (11..=40).collect();

    index.apply_event(make_store_event(1, &one_to_35)).await;
    index.apply_event(make_store_event(2, &one_to_10)).await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &one_to_40, worker1, 35).await;
    assert_score(&index, &one_to_40, worker2, 10).await;

    index
        .apply_event(make_store_event_with_parent(2, &one_to_10, &eleven_to_40))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_score(&index, &one_to_40, worker1, 35).await;
    assert_score(&index, &one_to_40, worker2, 40).await;

    let expected = snapshot_events(vec![
        make_store_event(1, &one_to_35),
        make_store_event(2, &one_to_40),
    ]);
    assert_eq!(snapshot_tree(&index).await, expected);
}

#[tokio::test]
async fn test_restore_after_mid_chain_remove_updates_score() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);
    let worker = WorkerWithDpRank::new(0, 0);

    index.apply_event(make_store_event(0, &[1, 2, 3])).await;
    flush_and_settle(&index).await;

    assert_score(&index, &[1, 2, 3], worker, 3).await;
    assert_eq!(index.backend().tree_size_for_worker(worker), Some(3));

    index
        .apply_event(make_remove_event_with_parent(0, &[1], &[2]))
        .await;
    flush_and_settle(&index).await;

    assert_score(&index, &[1, 2, 3], worker, 1).await;
    assert_eq!(index.backend().tree_size_for_worker(worker), Some(1));

    index
        .apply_event(make_store_event_with_parent(0, &[1], &[2, 3]))
        .await;
    flush_and_settle(&index).await;

    assert_score(&index, &[1, 2, 3], worker, 3).await;
    assert_eq!(index.backend().tree_size_for_worker(worker), Some(3));
}

#[tokio::test]
async fn test_partial_node_drops_unreachable_descendants() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);

    index.apply_event(make_store_event(0, &[1, 2, 3])).await;
    index
        .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]))
        .await;
    flush_and_settle(&index).await;

    index
        .apply_event(make_remove_event_with_parent(0, &[1], &[2]))
        .await;
    flush_and_settle(&index).await;

    assert_eq!(snapshot_tree(&index).await, vec![make_store_event(0, &[1])]);
}

#[tokio::test]
async fn test_cleanup_prunes_dead_children_under_live_prefix() {
    let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);

    index.apply_event(make_store_event(0, &[1, 2, 3])).await;
    index
        .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]))
        .await;
    index
        .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[6, 7]))
        .await;
    flush_and_settle(&index).await;

    index
        .apply_event(make_remove_event_with_parent(0, &[1, 2, 3], &[4, 5]))
        .await;
    index
        .apply_event(make_remove_event_with_parent(0, &[1, 2, 3], &[6, 7]))
        .await;
    flush_and_settle(&index).await;

    let expected_snapshot = vec![make_store_event(0, &[1, 2, 3])];
    assert_eq!(snapshot_tree(&index).await, expected_snapshot);
    assert_eq!(index.backend().raw_child_edge_count(), 3);

    index.backend().run_cleanup_for_test();

    assert_eq!(index.backend().raw_child_edge_count(), 1);
    assert_eq!(
        snapshot_tree(&index).await,
        vec![make_store_event(0, &[1, 2, 3])]
    );
    assert_score(&index, &[1, 2, 3], WorkerWithDpRank::new(0, 0), 3).await;
}
