// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
use crate::test_utils::{
    assert_score, flush_and_settle, make_clear_event_with_dp_rank, make_remove_event_with_parent,
    make_store_event, make_store_event_with_dp_rank, make_store_event_with_parent, remove_event,
    snapshot_events, snapshot_tree,
};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

type DirectLookup = LaneLookup;

fn worker(worker_id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(worker_id, 0)
}

/// `worker`'s coverage slot, allocated as its first store would.
fn slot(index: &ConcurrentRadixTreeCompressed, worker: WorkerWithDpRank) -> coverage::Slot {
    let guard = crossbeam_epoch::pin();
    index.event_worker_for_test(worker, &guard).slot
}

fn direct_lookup() -> DirectLookup {
    LaneLookup::default()
}

fn stored_data(event: RouterEvent) -> KvCacheStoreData {
    match event.event.data {
        KvCacheEventData::Stored(op) => op,
        _ => unreachable!("expected a store event"),
    }
}

fn worker_lookup_len(lookup: &DirectLookup, worker: WorkerWithDpRank) -> Option<usize> {
    lookup.block_count(worker)
}

async fn index_block_count(index: &ThreadPoolIndexer<ConcurrentRadixTreeCompressed>) -> usize {
    index
        .shard_sizes()
        .await
        .into_iter()
        .map(|snapshot| snapshot.block_count)
        .sum()
}

fn local_hashes(query: &[u64]) -> Vec<LocalBlockHash> {
    query.iter().copied().map(LocalBlockHash).collect()
}

fn remove_hashes_with_parent(
    prefix_hashes: &[u64],
    local_hashes: &[u64],
) -> Vec<ExternalSequenceBlockHash> {
    match make_remove_event_with_parent(0, prefix_hashes, local_hashes)
        .event
        .data
    {
        KvCacheEventData::Removed(op) => op.block_hashes,
        _ => unreachable!("make_remove_event_with_parent must create a remove event"),
    }
}

fn apply_direct(
    index: &ConcurrentRadixTreeCompressed,
    lookup: &mut DirectLookup,
    event: RouterEvent,
) {
    index.apply_event(lookup, event, None).unwrap();
    lookup.assert_invariants();
}

fn assert_direct_score(
    index: &ConcurrentRadixTreeCompressed,
    query: &[u64],
    worker: WorkerWithDpRank,
    expected: u32,
) {
    let scores = index.find_matches_impl(&local_hashes(query), false);
    assert_eq!(
        scores.scores.get(&worker).copied(),
        Some(expected),
        "query={query:?} worker={worker:?} scores={:?}",
        scores.scores
    );
}

fn assert_edge_lengths(index: &ConcurrentRadixTreeCompressed, expected: &[usize]) {
    assert_eq!(index.edge_lengths_for_test(), expected.to_vec());
}

fn edge_topology(edge: &[u64], children: Vec<EdgeTopologyForTest>) -> EdgeTopologyForTest {
    EdgeTopologyForTest {
        edge: edge.to_vec(),
        children,
    }
}

fn race_two_events(
    index: Arc<ConcurrentRadixTreeCompressed>,
    mut left_lookup: DirectLookup,
    left_event: RouterEvent,
    mut right_lookup: DirectLookup,
    right_event: RouterEvent,
) -> (DirectLookup, DirectLookup) {
    let barrier = Arc::new(Barrier::new(3));

    let left_index = index.clone();
    let left_barrier = barrier.clone();
    let left = thread::spawn(move || {
        left_barrier.wait();
        apply_direct(&left_index, &mut left_lookup, left_event);
        left_lookup
    });

    let right_barrier = barrier.clone();
    let right = thread::spawn(move || {
        right_barrier.wait();
        apply_direct(&index, &mut right_lookup, right_event);
        right_lookup
    });

    barrier.wait();
    (left.join().unwrap(), right.join().unwrap())
}

mod race_tests {
    mod store {
        use super::super::*;

        #[test]
        fn race_divergent_tail_extensions_split_compressed_leaf() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let worker1 = worker(1);
            let worker2 = worker(2);
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();

            apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
            apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3, 4]));

            let (lookup1, lookup2) = race_two_events(
                index.clone(),
                lookup1,
                make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6]),
                lookup2,
                make_store_event_with_parent(2, &[1, 2, 3, 4], &[7, 8]),
            );

            assert_eq!(index.raw_child_edge_count(), 3);
            assert_edge_lengths(&index, &[2, 2, 4]);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 4);
            assert_direct_score(&index, &[1, 2, 3, 4, 7, 8], worker1, 4);
            assert_direct_score(&index, &[1, 2, 3, 4, 7, 8], worker2, 6);
            assert_eq!(worker_lookup_len(&lookup1, worker1), Some(6));
            assert_eq!(worker_lookup_len(&lookup2, worker2), Some(6));
        }

        #[test]
        fn race_identical_tail_extensions_stay_compressed() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let worker1 = worker(1);
            let worker2 = worker(2);
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();

            apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
            apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3, 4]));

            let (lookup1, lookup2) = race_two_events(
                index.clone(),
                lookup1,
                make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6]),
                lookup2,
                make_store_event_with_parent(2, &[1, 2, 3, 4], &[5, 6]),
            );

            assert_eq!(index.raw_child_edge_count(), 1);
            assert_edge_lengths(&index, &[6]);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 6);
            assert_eq!(worker_lookup_len(&lookup1, worker1), Some(6));
            assert_eq!(worker_lookup_len(&lookup2, worker2), Some(6));
        }

        #[test]
        fn race_suffix_reuse_with_divergent_split_preserves_workers() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let worker0 = worker(0);
            let worker1 = worker(1);
            let worker2 = worker(2);
            let mut lookup0 = direct_lookup();
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();

            apply_direct(
                &index,
                &mut lookup0,
                make_store_event(0, &[1, 2, 3, 4, 5, 6]),
            );
            apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3]));
            apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3]));

            let (_lookup1, _lookup2) = race_two_events(
                index.clone(),
                lookup1,
                make_store_event_with_parent(1, &[1, 2, 3], &[4, 5, 6]),
                lookup2,
                make_store_event_with_parent(2, &[1, 2, 3], &[7, 8]),
            );

            assert_eq!(index.raw_child_edge_count(), 3);
            assert_edge_lengths(&index, &[2, 3, 3]);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 3);
            assert_direct_score(&index, &[1, 2, 3, 7, 8], worker0, 3);
            assert_direct_score(&index, &[1, 2, 3, 7, 8], worker1, 3);
            assert_direct_score(&index, &[1, 2, 3, 7, 8], worker2, 5);
        }

        #[test]
        fn stale_scan_cannot_commit_after_split() {
            let index = ConcurrentRadixTreeCompressed::new();
            let worker1 = worker(1);
            let worker2 = worker(2);
            let worker3 = worker(3);
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();
            let mut lookup3 = direct_lookup();

            apply_direct(
                &index,
                &mut lookup1,
                make_store_event(1, &[1, 2, 3, 4, 5, 6]),
            );
            apply_direct(
                &index,
                &mut lookup2,
                make_store_event(2, &[1, 2, 3, 4, 5, 6]),
            );

            let node = index
                .root
                .child_snapshot(LocalBlockHash(1))
                .expect("root child should exist");
            let blocks = stored_data(make_store_event(3, &[1, 2, 3, 4, 5, 6])).blocks;
            let stale_scan = node.scan_store_prefix(&blocks);

            apply_direct(
                &index,
                &mut lookup2,
                make_store_event_with_parent(2, &[1, 2, 3], &[7]),
            );

            assert!(
                node.promote_to_full_with_version(slot(&index, worker3), stale_scan.shape_version)
                    .is_none()
            );

            apply_direct(
                &index,
                &mut lookup3,
                make_store_event(3, &[1, 2, 3, 4, 5, 6]),
            );

            assert_eq!(
                index.edge_topology_for_test(),
                vec![edge_topology(
                    &[1, 2, 3],
                    vec![
                        edge_topology(&[4, 5, 6], vec![]),
                        edge_topology(&[7], vec![])
                    ],
                )],
            );
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker2, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker3, 6);
        }

        #[test]
        fn tail_parent_split_before_child_lookup_repairs_to_suffix() {
            let index = ConcurrentRadixTreeCompressed::new();
            let worker1 = worker(1);
            let worker2 = worker(2);
            let worker3 = worker(3);
            let worker4 = worker(4);
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();
            let mut lookup3 = direct_lookup();
            let mut lookup4 = direct_lookup();

            for (worker_id, lookup) in [
                (1, &mut lookup1),
                (2, &mut lookup2),
                (3, &mut lookup3),
                (4, &mut lookup4),
            ] {
                apply_direct(&index, lookup, make_store_event(worker_id, &[1, 2, 3, 4]));
            }
            apply_direct(
                &index,
                &mut lookup1,
                make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6]),
            );
            apply_direct(
                &index,
                &mut lookup2,
                make_store_event_with_parent(2, &[1, 2, 3, 4], &[7, 8]),
            );

            let stale_parent = index
                .root
                .child_snapshot(LocalBlockHash(1))
                .expect("root child should exist");
            let continuation = stored_data(make_store_event_with_parent(3, &[1, 2, 3, 4], &[9]));
            let parent_hash = continuation.parent_hash.expect("continuation has a parent");
            let plan = stale_parent
                .plan_store_parent_edge(parent_hash, &continuation.blocks)
                .expect("tail parent should be present before the split");
            assert!(matches!(
                plan.action,
                ParentEdgePlanAction::InsertFromParent
            ));

            apply_direct(
                &index,
                &mut lookup4,
                make_store_event_with_parent(4, &[1, 2], &[10]),
            );

            index
                .insert_blocks_from_for_test(
                    &mut lookup3,
                    worker3,
                    &stale_parent,
                    parent_hash,
                    &continuation.blocks,
                )
                .unwrap();

            assert_eq!(
                index.edge_topology_for_test(),
                vec![edge_topology(
                    &[1, 2],
                    vec![
                        edge_topology(
                            &[3, 4],
                            vec![
                                edge_topology(&[5, 6], vec![]),
                                edge_topology(&[7, 8], vec![]),
                                edge_topology(&[9], vec![]),
                            ],
                        ),
                        edge_topology(&[10], vec![]),
                    ],
                )],
            );
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 7, 8], worker2, 6);
            assert_direct_score(&index, &[1, 2, 3, 4, 9], worker3, 5);
            assert_direct_score(&index, &[1, 2, 9], worker3, 2);
            assert_direct_score(&index, &[1, 2, 10], worker4, 3);
        }

        #[test]
        fn stale_tail_cursor_replans_before_descending_same_local_child() {
            let index = ConcurrentRadixTreeCompressed::new();
            let worker_a = worker(1);
            let worker_b = worker(2);
            let mut lookup_a = direct_lookup();
            let mut lookup_b = direct_lookup();

            apply_direct(&index, &mut lookup_a, make_store_event(1, &[1]));
            apply_direct(&index, &mut lookup_b, make_store_event(2, &[1]));

            let stale_parent = index
                .root
                .child_snapshot(LocalBlockHash(1))
                .expect("root child should exist");
            let continuation = stored_data(make_store_event_with_parent(1, &[1], &[7]));
            let parent_hash = continuation.parent_hash.expect("continuation has a parent");
            let plan = stale_parent
                .plan_store_parent_edge(parent_hash, &continuation.blocks)
                .expect("tail parent should be present before extension");
            let action = stale_parent.apply_store_parent_edge_plan(
                slot(&index, worker_a),
                plan,
                &continuation.blocks,
                &index.reclaim.capacity,
            );
            assert!(matches!(action, ParentEdgeAction::InsertFromParent(None)));

            let b_extension = make_store_event_with_parent(2, &[1], &[5, 6, 7]);
            let b_extension_data = stored_data(b_extension.clone());
            assert_ne!(
                continuation.blocks[0].block_hash, b_extension_data.blocks[2].block_hash,
                "same local child must have distinct sequence hashes at different depths",
            );
            apply_direct(&index, &mut lookup_b, b_extension);
            apply_direct(
                &index,
                &mut lookup_b,
                make_store_event_with_parent(2, &[1, 5, 6], &[8]),
            );

            index
                .insert_blocks_from_for_test(
                    &mut lookup_a,
                    worker_a,
                    &stale_parent,
                    parent_hash,
                    &continuation.blocks,
                )
                .unwrap();

            assert_direct_score(&index, &[1, 7], worker_a, 2);
            assert_direct_score(&index, &[1, 5, 6, 7], worker_b, 4);
            assert_direct_score(&index, &[1, 5, 6, 8], worker_b, 4);
            assert_eq!(
                index.edge_topology_for_test(),
                vec![edge_topology(
                    &[1],
                    vec![
                        edge_topology(
                            &[5, 6],
                            vec![edge_topology(&[7], vec![]), edge_topology(&[8], vec![]),],
                        ),
                        edge_topology(&[7], vec![]),
                    ],
                )],
            );
        }
    }

    mod remove {
        use super::super::*;

        #[test]
        fn race_split_with_remove_repairs_stale_lookup_on_restore() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let worker1 = worker(1);
            let worker2 = worker(2);
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();

            apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
            apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3, 4]));

            let (_lookup1, mut lookup2) = race_two_events(
                index.clone(),
                lookup1,
                make_store_event_with_parent(1, &[1, 2], &[7]),
                lookup2,
                make_remove_event_with_parent(2, &[1, 2], &[3]),
            );

            assert_direct_score(&index, &[1, 2, 7], worker1, 3);
            assert_direct_score(&index, &[1, 2, 3, 4], worker2, 2);

            apply_direct(
                &index,
                &mut lookup2,
                make_store_event_with_parent(2, &[1, 2], &[3, 4]),
            );

            assert_direct_score(&index, &[1, 2, 3, 4], worker2, 4);
        }

        #[test]
        fn race_remove_keeps_children_needed_by_another_full_worker() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let worker1 = worker(1);
            let worker2 = worker(2);
            let worker3 = worker(3);
            let mut lookup1 = direct_lookup();
            let mut lookup2 = direct_lookup();
            let mut lookup3 = direct_lookup();

            apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
            apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3, 4]));
            apply_direct(
                &index,
                &mut lookup1,
                make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6]),
            );
            apply_direct(&index, &mut lookup3, make_store_event(3, &[1, 2, 3, 4]));
            apply_direct(
                &index,
                &mut lookup3,
                make_store_event_with_parent(3, &[1, 2, 3, 4], &[7, 8]),
            );

            let barrier = Arc::new(Barrier::new(3));
            let reader_index = index.clone();
            let reader_barrier = barrier.clone();
            let reader = thread::spawn(move || {
                reader_barrier.wait();
                for _ in 0..256 {
                    assert_direct_score(&reader_index, &[1, 2, 3, 4, 5, 6], worker1, 6);
                }
            });

            let remover_index = index.clone();
            let remover_barrier = barrier.clone();
            let remover = thread::spawn(move || {
                remover_barrier.wait();
                apply_direct(
                    &remover_index,
                    &mut lookup2,
                    make_remove_event_with_parent(2, &[1], &[2]),
                );
                lookup2
            });

            barrier.wait();
            reader.join().unwrap();
            let _lookup2 = remover.join().unwrap();

            assert_eq!(index.raw_child_edge_count(), 3);
            assert_edge_lengths(&index, &[2, 2, 4]);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4], worker2, 1);
            assert_direct_score(&index, &[1, 2, 3, 4, 7, 8], worker3, 6);
        }
    }

    mod clear {
        use super::super::*;

        #[tokio::test]
        async fn clear_sweeps_only_its_rank_across_split_suffixes() {
            let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32);
            let worker_a0 = WorkerWithDpRank::new(1, 0);
            let worker_a1 = WorkerWithDpRank::new(1, 1);
            let worker_b = WorkerWithDpRank::new(2, 0);

            index
                .apply_event(make_store_event_with_dp_rank(1, &[1, 2], 0))
                .await;
            index
                .apply_event(make_store_event_with_dp_rank(1, &[1, 2], 1))
                .await;
            index.apply_event(make_store_event(2, &[1, 2])).await;
            flush_and_settle(&index).await;

            index
                .apply_event(make_store_event_with_parent(2, &[1], &[3]))
                .await;
            flush_and_settle(&index).await;

            index.apply_event(make_clear_event_with_dp_rank(1, 0)).await;
            flush_and_settle(&index).await;

            index
                .apply_event(make_store_event_with_dp_rank(1, &[1], 0))
                .await;
            index
                .apply_event(make_store_event_with_dp_rank(1, &[1], 1))
                .await;
            flush_and_settle(&index).await;

            assert_score(&index, &[1, 2], worker_a0, 1).await;
            assert_score(&index, &[1, 2], worker_a1, 2).await;
            assert_score(&index, &[1, 2], worker_b, 2).await;
            assert_score(&index, &[1, 3], worker_b, 2).await;
        }

        #[test]
        fn clear_before_split_prevents_copying_removed_coverage() {
            let index = ConcurrentRadixTreeCompressed::new();
            let worker_a = worker(1);
            let worker_b = worker(2);
            let mut lookup_a = direct_lookup();
            let mut lookup_b = direct_lookup();

            apply_direct(&index, &mut lookup_a, make_store_event(1, &[1, 2]));
            apply_direct(&index, &mut lookup_b, make_store_event(2, &[1, 2]));
            apply_direct(&index, &mut lookup_a, make_clear_event_with_dp_rank(1, 0));
            apply_direct(
                &index,
                &mut lookup_b,
                make_store_event_with_parent(2, &[1], &[3]),
            );
            apply_direct(&index, &mut lookup_a, make_store_event(1, &[1]));

            assert_direct_score(&index, &[1, 2], worker_a, 1);
            assert_direct_score(&index, &[1, 2], worker_b, 2);
            assert_direct_score(&index, &[1, 3], worker_b, 2);
        }

        #[tokio::test]
        async fn worker_removal_respects_dp_rank_and_worker_targets() {
            let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 4, 32);
            let worker_a0 = WorkerWithDpRank::new(1, 0);
            let worker_a1 = WorkerWithDpRank::new(1, 1);
            let worker_b = WorkerWithDpRank::new(2, 0);

            index
                .apply_event(make_store_event_with_dp_rank(1, &[1, 2], 0))
                .await;
            index
                .apply_event(make_store_event_with_dp_rank(1, &[1, 2], 1))
                .await;
            index.apply_event(make_store_event(2, &[1, 2])).await;
            flush_and_settle(&index).await;
            index
                .apply_event(make_store_event_with_parent(2, &[1], &[3]))
                .await;
            flush_and_settle(&index).await;

            index.remove_worker_dp_rank(1, 0).await;
            flush_and_settle(&index).await;
            let scores = index.find_matches(local_hashes(&[1, 2])).await.unwrap();
            assert!(!scores.scores.contains_key(&worker_a0));
            assert_eq!(scores.scores.get(&worker_a1), Some(&2));
            assert_eq!(scores.scores.get(&worker_b), Some(&2));

            index.remove_worker(1).await;
            flush_and_settle(&index).await;
            let scores = index.find_matches(local_hashes(&[1, 2])).await.unwrap();
            assert!(!scores.scores.contains_key(&worker_a0));
            assert!(!scores.scores.contains_key(&worker_a1));
            assert_eq!(scores.scores.get(&worker_b), Some(&2));
        }
    }

    mod cleanup {
        use super::super::*;

        #[test]
        fn race_cleanup_with_dead_child_reuse_keeps_restored_child() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let worker = worker(0);
            let mut lookup = direct_lookup();

            apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2, 3]));
            apply_direct(
                &index,
                &mut lookup,
                make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]),
            );
            apply_direct(
                &index,
                &mut lookup,
                make_store_event_with_parent(0, &[1, 2, 3], &[6, 7]),
            );
            apply_direct(
                &index,
                &mut lookup,
                make_remove_event_with_parent(0, &[1, 2, 3], &[4, 5]),
            );
            apply_direct(
                &index,
                &mut lookup,
                make_remove_event_with_parent(0, &[1, 2, 3], &[6, 7]),
            );

            assert_eq!(index.raw_child_edge_count(), 3);
            assert_direct_score(&index, &[1, 2, 3], worker, 3);

            let barrier = Arc::new(Barrier::new(3));
            let cleanup_index = index.clone();
            let cleanup_barrier = barrier.clone();
            let cleanup = thread::spawn(move || {
                cleanup_barrier.wait();
                cleanup_index.run_cleanup_for_test();
            });

            let store_index = index.clone();
            let store_barrier = barrier.clone();
            let store = thread::spawn(move || {
                store_barrier.wait();
                apply_direct(
                    &store_index,
                    &mut lookup,
                    make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]),
                );
                lookup
            });

            barrier.wait();
            cleanup.join().unwrap();
            let _lookup = store.join().unwrap();

            let edge_lengths = index.edge_lengths_for_test();
            assert!(
                edge_lengths == vec![5] || edge_lengths == vec![2, 3],
                "unexpected edge lengths: {edge_lengths:?}"
            );
            assert_eq!(index.raw_child_edge_count(), edge_lengths.len());
            assert_direct_score(&index, &[1, 2, 3, 4, 5], worker, 5);
        }
    }

    mod read {
        use super::super::*;

        #[test]
        fn race_find_during_split_never_overcounts() {
            let index = Arc::new(ConcurrentRadixTreeCompressed::new());
            let branches: Vec<(WorkerWithDpRank, Vec<u64>)> = vec![
                (worker(1), vec![5, 6]),
                (worker(2), vec![7, 8]),
                (worker(3), vec![9, 10]),
                (worker(4), vec![11, 12]),
            ];
            let mut seeded = Vec::new();

            for (worker, _) in &branches {
                let mut lookup = direct_lookup();
                apply_direct(
                    &index,
                    &mut lookup,
                    make_store_event(worker.worker_id, &[1, 2, 3, 4]),
                );
                seeded.push((*worker, lookup));
            }

            let barrier = Arc::new(Barrier::new(branches.len() + 2));
            let reader_index = index.clone();
            let reader_branches = branches.clone();
            let reader_barrier = barrier.clone();
            let reader = thread::spawn(move || {
                reader_barrier.wait();
                for _ in 0..512 {
                    for (branch_worker, suffix) in &reader_branches {
                        let mut query = vec![1, 2, 3, 4];
                        query.extend(suffix);
                        let scores = reader_index.find_matches_impl(&local_hashes(&query), false);
                        for (score_worker, score) in scores.scores {
                            let max_expected = if score_worker == *branch_worker { 6 } else { 4 };
                            assert!(
                                score <= query.len() as u32,
                                "score {score} exceeds query length for worker {score_worker:?} query={query:?}",
                            );
                            assert!(
                                score <= max_expected,
                                "score {score} exceeds reachable depth {max_expected} for worker {score_worker:?} query={query:?}",
                            );
                        }
                    }
                }
            });

            let mut writers = Vec::new();
            for ((branch_worker, mut lookup), (_, suffix)) in
                seeded.into_iter().zip(branches.iter())
            {
                let writer_index = index.clone();
                let writer_barrier = barrier.clone();
                let suffix = suffix.clone();
                writers.push(thread::spawn(move || {
                    writer_barrier.wait();
                    apply_direct(
                        &writer_index,
                        &mut lookup,
                        make_store_event_with_parent(
                            branch_worker.worker_id,
                            &[1, 2, 3, 4],
                            &suffix,
                        ),
                    );
                }));
            }

            barrier.wait();
            for writer in writers {
                writer.join().unwrap();
            }
            reader.join().unwrap();

            assert_eq!(index.raw_child_edge_count(), 5);
            assert_edge_lengths(&index, &[2, 2, 2, 2, 4]);
            for (branch_worker, suffix) in &branches {
                let mut query = vec![1, 2, 3, 4];
                query.extend(suffix);
                for (other_worker, _) in &branches {
                    let expected = if other_worker == branch_worker { 6 } else { 4 };
                    assert_direct_score(&index, &query, *other_worker, expected);
                }
            }
        }
    }
}

mod remove_tests {
    use super::*;
    use crate::test_utils::{make_remove_event, make_store_event_full, router_event};

    #[test]
    fn remove_multiple_hashes_from_same_compressed_edge() {
        let index = Arc::new(ConcurrentRadixTreeCompressed::new());
        let worker0 = worker(0);
        let worker1 = worker(1);
        let mut lookup0 = direct_lookup();
        let mut lookup1 = direct_lookup();

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event(0, &[1, 2, 3, 4, 5, 6]),
        );
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event(1, &[1, 2, 3, 4, 5, 6]),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 6);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);

        apply_direct(
            &index,
            &mut lookup0,
            make_remove_event_with_parent(0, &[1, 2], &[3, 4, 5]),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 2);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event_with_parent(0, &[1, 2], &[3, 4, 5, 6]),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 6);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
    }

    #[test]
    fn batched_remove_uses_minimum_edge_position_not_event_order() {
        let index = ConcurrentRadixTreeCompressed::new();
        let worker0 = worker(0);
        let worker1 = worker(1);
        let mut lookup0 = direct_lookup();
        let mut lookup1 = direct_lookup();

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event(0, &[1, 2, 3, 4, 5, 6]),
        );
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event(1, &[1, 2, 3, 4, 5, 6]),
        );

        let remove_hashes = remove_hashes_with_parent(&[1, 2], &[3, 4, 5]);
        let out_of_order_hashes = vec![remove_hashes[2], remove_hashes[0], remove_hashes[1]];
        apply_direct(
            &index,
            &mut lookup0,
            remove_event(0, 0, 0, out_of_order_hashes),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 2);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(2));

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event_with_parent(0, &[1, 2], &[3, 4, 5, 6]),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 6);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(6));
    }

    #[test]
    fn batched_remove_processes_hashes_moved_by_split_before_restore() {
        let index = ConcurrentRadixTreeCompressed::new();
        let worker0 = worker(0);
        let worker1 = worker(1);
        let mut lookup0 = direct_lookup();
        let mut lookup1 = direct_lookup();

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event(0, &[1, 2, 3, 4, 5, 6]),
        );
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event(1, &[1, 2, 3, 4, 5, 6]),
        );
        let remove_hashes = remove_hashes_with_parent(&[1, 2], &[3, 4, 5]);
        let group_node = lookup0
            .node(worker0, remove_hashes[0])
            .expect("remove hash should point to the pre-split group node");

        apply_direct(
            &index,
            &mut lookup1,
            make_store_event_with_parent(1, &[1, 2, 3], &[7]),
        );
        assert_eq!(
            index.edge_topology_for_test(),
            vec![edge_topology(
                &[1, 2, 3],
                vec![
                    edge_topology(&[4, 5, 6], vec![]),
                    edge_topology(&[7], vec![])
                ],
            )],
        );

        // Only the head of the run is still on the pre-split node; the rest resolve
        // through lookup repair onto the suffix.
        assert!(group_node.contains_edge_hash(remove_hashes[0]));
        assert!(!group_node.contains_edge_hash(remove_hashes[1]));
        apply_direct(
            &index,
            &mut lookup0,
            make_remove_event_with_parent(0, &[1, 2], &[3, 4, 5]),
        );
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 2);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event_with_parent(0, &[1, 2], &[3]),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 3);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(3));
    }

    #[test]
    fn batched_remove_falls_back_when_all_grouped_hashes_move_before_restore() {
        let index = ConcurrentRadixTreeCompressed::new();
        let worker0 = worker(0);
        let worker1 = worker(1);
        let mut lookup0 = direct_lookup();
        let mut lookup1 = direct_lookup();

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event(0, &[1, 2, 3, 4, 5, 6]),
        );
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event(1, &[1, 2, 3, 4, 5, 6]),
        );
        let remove_hashes = remove_hashes_with_parent(&[1, 2, 3], &[4, 5]);
        let group_node = lookup0
            .node(worker0, remove_hashes[0])
            .expect("remove hash should point to the pre-split group node");

        apply_direct(
            &index,
            &mut lookup1,
            make_store_event_with_parent(1, &[1, 2, 3], &[7]),
        );
        assert_eq!(
            index.edge_topology_for_test(),
            vec![edge_topology(
                &[1, 2, 3],
                vec![
                    edge_topology(&[4, 5, 6], vec![]),
                    edge_topology(&[7], vec![])
                ],
            )],
        );

        // The whole run moved off the node the lookup still names, so the grouped
        // removal declines without touching it and the remove path repairs instead.
        assert!(
            group_node
                .remove_worker_for_leading_hashes(slot(&index, worker0), &remove_hashes)
                .is_none()
        );
        apply_direct(
            &index,
            &mut lookup0,
            make_remove_event_with_parent(0, &[1, 2, 3], &[4, 5]),
        );
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 3);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(3));

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event_with_parent(0, &[1, 2, 3], &[4]),
        );

        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 4);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(4));
    }

    /// The six-event grouped-removal repro. Evicting `[1]` leaves its node without a full
    /// holder, so its children are unlinked while the lane still names `B = [2, 20, 21]`
    /// for those hashes. Restoring `[1]` and then `[2, 20]` puts 20 on a new live node
    /// `B'`. Removing 20 and 21 must take 20 off `B'` in either order, also when 21
    /// resolves the run to the unlinked `B`.
    fn grouped_removal_repro_events(tail_first: bool) -> Vec<RouterEvent> {
        let mut evicted = remove_hashes_with_parent(&[1, 2], &[20, 21]);
        if tail_first {
            evicted.reverse();
        }
        vec![
            make_store_event(0, &[1, 30]),
            make_store_event_with_parent(0, &[1], &[2, 20, 21]),
            make_remove_event(0, &[1]),
            make_store_event(0, &[1]),
            make_store_event_with_parent(0, &[1], &[2, 20]),
            remove_event(0, 0, 0, evicted),
        ]
    }

    #[test]
    fn grouped_removal_leaves_hashes_named_on_another_node() {
        for tail_first in [true, false] {
            let index = ConcurrentRadixTreeCompressed::new();
            let mut lookup = direct_lookup();
            for event in grouped_removal_repro_events(tail_first) {
                apply_direct(&index, &mut lookup, event);
            }
            let scores = index.find_matches_impl(&local_hashes(&[1, 2, 20]), false);
            assert_eq!(
                scores.scores.get(&worker(0)),
                Some(&2),
                "tail_first={tail_first}"
            );
            // 20 and 21 are gone from the lane; 1, 2 and the unreachable 30 stay.
            assert_eq!(worker_lookup_len(&lookup, worker(0)), Some(3));
        }
    }

    /// Randomized differential test against each rank's set of cached blocks. Removes come
    /// tail-first, head-first or shuffled and may leave holes, stores restore parts of
    /// evicted chains, and ranks are cleared one at a time, so unlinked subtrees keep stale
    /// coverage and lookup entries. The tree may undercount there, but it must never score
    /// a rank past the leading blocks the rank holds, and a lane must never keep an entry
    /// for a block its rank no longer holds. `CRTC_REMOVE_FUZZ_SEEDS` sets the number of
    /// seeds.
    #[test]
    fn random_hole_streams_never_overcount_or_leak_entries() {
        let seeds = std::env::var("CRTC_REMOVE_FUZZ_SEEDS")
            .ok()
            .and_then(|seeds| seeds.parse().ok())
            .unwrap_or(40u64);
        // Lanes split ranks by worker id, as `ThreadPoolIndexer` does with two threads.
        let ranks = [
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(1, 0),
            WorkerWithDpRank::new(2, 0),
        ];
        let mut failures = Vec::new();
        let mut repair_scans = 0;

        for seed in 0..seeds {
            let mut rng = fastrand::Rng::with_seed(seed);
            // A base chain plus branches off prefixes of earlier chains.
            let mut docs: Vec<Vec<u64>> = vec![(1..=24).collect()];
            for branch in 1..10 {
                let base = &docs[rng.usize(..docs.len())];
                let mut doc = base[..rng.usize(1..base.len())].to_vec();
                doc.extend((0..rng.u64(1..=12)).map(|i| 1_000 * branch + i));
                docs.push(doc);
            }
            let seqs: Vec<Vec<u64>> = docs
                .iter()
                .map(|doc| compute_seq_hash_for_block(&local_hashes(doc)))
                .collect();

            let index = ConcurrentRadixTreeCompressed::new();
            let mut lanes = [direct_lookup(), direct_lookup()];
            let mut held = vec![FxHashSet::<u64>::default(); ranks.len()];
            for step in 0..1_000 {
                let r = rng.usize(..ranks.len());
                let rank = ranks[r];
                let d = rng.usize(..docs.len());
                let (doc, seq) = (&docs[d], &seqs[d]);
                let data = match rng.u32(..100) {
                    0..=46 => {
                        // Any cached parent can take a store, so evicted chains come back
                        // in parts.
                        let starts: Vec<usize> = (0..doc.len())
                            .filter(|&start| start == 0 || held[r].contains(&seq[start - 1]))
                            .collect();
                        let start = starts[rng.usize(..starts.len())];
                        let end = rng.usize(start + 1..=doc.len());
                        let mut op = stored_data(make_store_event_full(
                            rank.worker_id,
                            &doc[..end],
                            rank.dp_rank,
                            None,
                            None,
                        ));
                        op.parent_hash = start
                            .checked_sub(1)
                            .map(|parent| ExternalSequenceBlockHash(seq[parent]));
                        op.blocks.drain(..start);
                        KvCacheEventData::Stored(op)
                    }
                    47..=96 => {
                        let cached: Vec<u64> = seq
                            .iter()
                            .copied()
                            .filter(|hash| held[r].contains(hash))
                            .collect();
                        if cached.is_empty() {
                            continue;
                        }
                        // Cached blocks from some position on, skipping any holes.
                        let from = rng.usize(..cached.len());
                        let len = rng.usize(1..=cached.len() - from);
                        let mut evicted = cached[from..from + len].to_vec();
                        match rng.u32(..3) {
                            0 => evicted.reverse(),
                            1 => rng.shuffle(&mut evicted),
                            _ => {}
                        }
                        for hash in &evicted {
                            held[r].remove(hash);
                        }
                        KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: evicted
                                .into_iter()
                                .map(ExternalSequenceBlockHash)
                                .collect(),
                        })
                    }
                    _ => {
                        held[r].clear();
                        KvCacheEventData::Cleared
                    }
                };
                let stored = match &data {
                    KvCacheEventData::Stored(op) => {
                        op.blocks.iter().map(|block| block.block_hash.0).collect()
                    }
                    _ => Vec::new(),
                };

                let lookup = &mut lanes[(rank.worker_id % 2) as usize];
                // A store under a parent the tree no longer holds is rejected, which
                // only undercounts; the rank then does not hold its blocks either.
                if index
                    .apply_event(
                        lookup,
                        router_event(rank.worker_id, step, rank.dp_rank, data),
                        None,
                    )
                    .is_ok()
                {
                    held[r].extend(stored);
                }
                lookup.assert_invariants();
                for (rank, held) in ranks.iter().zip(&held) {
                    let lane = &lanes[(rank.worker_id % 2) as usize];
                    for hash in lane.hashes_for_test(*rank) {
                        if !held.contains(&hash.0) {
                            failures.push((
                                seed,
                                format!(
                                    "seed={seed} step={step} rank={rank:?} keeps an entry for \
                                     evicted {hash:?}"
                                ),
                            ));
                        }
                    }
                }
                for (doc, seq) in docs.iter().zip(&seqs) {
                    let scores = index.find_matches_impl(&local_hashes(doc), false).scores;
                    for (rank, held) in ranks.iter().zip(&held) {
                        let prefix = seq.iter().take_while(|hash| held.contains(hash)).count();
                        let score = scores.get(rank).map_or(0, |&score| score as usize);
                        if score > prefix {
                            failures.push((
                                seed,
                                format!(
                                    "seed={seed} step={step} rank={rank:?} doc={doc:?} \
                                     score={score} held_prefix={prefix}"
                                ),
                            ));
                        }
                    }
                }
            }
            repair_scans += index
                .bench_metrics
                .lookup_repair_scans
                .load(Ordering::Relaxed);
        }

        // Cross-lane splits must leave stale entries for lookup repair to resolve.
        assert!(repair_scans > 0, "the streams never repaired a lookup");
        let failed_seeds: FxHashSet<_> = failures.iter().map(|&(seed, _)| seed).collect();
        assert!(
            failures.is_empty(),
            "{} failures in {} of {seeds} seeds, first: {}",
            failures.len(),
            failed_seeds.len(),
            failures[0].1
        );
    }

    /// The same repro through `ThreadPoolIndexer` with one and four lanes. Other workers
    /// keep the remaining lanes busy in a disjoint subtree.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn grouped_removal_repro_scores_two_through_the_thread_pool() {
        for lanes in [1, 4] {
            for tail_first in [true, false] {
                let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), lanes, 32);
                for other in 1..4 {
                    index
                        .apply_event(make_store_event(other, &[7, 8, other + 100]))
                        .await;
                }
                for event in grouped_removal_repro_events(tail_first) {
                    index.apply_event(event).await;
                }
                flush_and_settle(&index).await;
                assert_score(&index, &[1, 2, 20], worker(0), 2).await;
                for other in 1..4 {
                    assert_score(&index, &[7, 8, other + 100], worker(other), 3).await;
                }
            }
        }
    }
}

mod structural_tests {
    use super::*;

    #[test]
    fn split_preserves_high_fanout_children_under_original_suffix() {
        let index = ConcurrentRadixTreeCompressed::new();
        let worker0 = worker(0);
        let worker1 = worker(1);
        let mut lookup0 = direct_lookup();
        let mut lookup1 = direct_lookup();

        apply_direct(&index, &mut lookup0, make_store_event(0, &[1, 2, 3, 4]));
        for branch in 10..15 {
            apply_direct(
                &index,
                &mut lookup0,
                make_store_event_with_parent(0, &[1, 2, 3, 4], &[branch]),
            );
        }
        let parent = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        let original_children = parent.child_edges_snapshot();
        assert_eq!(original_children.len(), 5);

        apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 99]));

        let suffix = parent.child_snapshot(LocalBlockHash(3)).unwrap();
        assert_eq!(parent.edge_local_hashes_for_test(), vec![1, 2]);
        assert_eq!(parent.child_edges_snapshot().len(), 2);
        assert_eq!(suffix.edge_local_hashes_for_test(), vec![3, 4]);
        assert_eq!(suffix.child_edges_snapshot().len(), original_children.len());
        for (hash, original_child) in original_children {
            assert!(Arc::ptr_eq(
                &suffix.child_snapshot(hash).unwrap(),
                &original_child,
            ));
            assert_direct_score(&index, &[1, 2, 3, 4, hash.0], worker0, 5);
        }
        assert_direct_score(&index, &[1, 2, 99], worker1, 3);
    }

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

    #[test]
    fn test_internal_split_reparents_existing_children_to_suffix() {
        let index = ConcurrentRadixTreeCompressed::new();
        let worker1 = worker(1);
        let worker2 = worker(2);
        let worker3 = worker(3);
        let mut lookup1 = direct_lookup();
        let mut lookup2 = direct_lookup();
        let mut lookup3 = direct_lookup();

        apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
        apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3, 4]));
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6]),
        );
        apply_direct(
            &index,
            &mut lookup2,
            make_store_event_with_parent(2, &[1, 2, 3, 4], &[7, 8]),
        );

        assert_eq!(
            index.edge_topology_for_test(),
            vec![edge_topology(
                &[1, 2, 3, 4],
                vec![
                    edge_topology(&[5, 6], vec![]),
                    edge_topology(&[7, 8], vec![]),
                ],
            )],
        );

        apply_direct(&index, &mut lookup3, make_store_event(3, &[1, 2, 3, 4]));
        apply_direct(
            &index,
            &mut lookup3,
            make_store_event_with_parent(3, &[1, 2], &[9]),
        );

        assert_eq!(
            index.edge_topology_for_test(),
            vec![edge_topology(
                &[1, 2],
                vec![
                    edge_topology(
                        &[3, 4],
                        vec![
                            edge_topology(&[5, 6], vec![]),
                            edge_topology(&[7, 8], vec![]),
                        ],
                    ),
                    edge_topology(&[9], vec![]),
                ],
            )],
        );
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_direct_score(&index, &[1, 2, 3, 4, 7, 8], worker2, 6);
        assert_direct_score(&index, &[1, 2, 9], worker3, 3);
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
}

mod remove_cleanup_tests {
    use super::*;

    #[tokio::test]
    async fn test_restore_after_mid_chain_remove_updates_score() {
        let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, 32);
        let worker = WorkerWithDpRank::new(0, 0);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        flush_and_settle(&index).await;

        assert_score(&index, &[1, 2, 3], worker, 3).await;
        assert_eq!(index_block_count(&index).await, 3);

        index
            .apply_event(make_remove_event_with_parent(0, &[1], &[2]))
            .await;
        flush_and_settle(&index).await;

        assert_score(&index, &[1, 2, 3], worker, 1).await;
        assert_eq!(index_block_count(&index).await, 1);

        index
            .apply_event(make_store_event_with_parent(0, &[1], &[2, 3]))
            .await;
        flush_and_settle(&index).await;

        assert_score(&index, &[1, 2, 3], worker, 3).await;
        assert_eq!(index_block_count(&index).await, 3);
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

    #[tokio::test]
    async fn test_cleanup_does_not_reopen_internal_node_for_extension() {
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
        index.backend().run_cleanup_for_test();

        assert_edge_lengths(index.backend(), &[3]);

        index
            .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[8, 9]))
            .await;
        flush_and_settle(&index).await;

        assert_edge_lengths(index.backend(), &[2, 3]);
        assert_score(&index, &[1, 2, 3, 8, 9], WorkerWithDpRank::new(0, 0), 5).await;
    }
}

/// Deterministic reproduction of per-worker lookup-entry leakage under the
/// split + children-clear + lazy-repair interplay (ai-dynamo/dynamo#10774,
/// production signature: prefill tracked_blocks climbing ~2x the physical
/// pool while RSS stays modest).
///
/// Two lookup maps emulate two event threads (sticky worker routing means a
/// worker's events are serialized, but its lookup is never repaired by other
/// threads' splits except lazily on access):
///
/// 1. w1 stores [1,2,3,4] -> node N; w1's lookup has 4 entries -> N.
/// 2. w2 stores [1,2,9] -> splits N into N=[1,2] with children S=[3,4], X=[9].
///    w1's entries for the 3,4 externals now point at N but live in S (stale
///    by design; lazy repair).
/// 3. w1 removes the head external -> w1 dropped from N, entries for N's edge
///    scrubbed. w1's stale entries for S's hashes remain.
/// 4. w2 removes the head external -> N's full_edge_workers empties ->
///    clear_children_if_unreachable DROPS S and X from N.
/// 5. w1's removes for the 3,4 externals arrive (the engine evicted them and
///    the producer is correct): lookup_node returns N, the grouped removal
///    misses because N no longer contains the hash, and repair_stale's subtree
///    scan fails because S was detached. Before the fix the remove was skipped
///    WITHOUT scrubbing the lookup entries -> they leaked forever and
///    block_count never returned to zero.
#[test]
fn remove_after_split_and_children_clear_scrubs_lookup() {
    let index = ConcurrentRadixTreeCompressed::new();
    let w1 = worker(1);
    let w2 = worker(2);
    let mut l1 = direct_lookup();
    let mut l2 = direct_lookup();

    // 1. w1 stores the 4-block chain.
    apply_direct(&index, &mut l1, make_store_event(1, &[1, 2, 3, 4]));
    assert_eq!(worker_lookup_len(&l1, w1), Some(4));

    // 2. w2 stores a chain sharing the [1,2] prefix, splitting N at pos 2.
    apply_direct(&index, &mut l2, make_store_event(2, &[1, 2, 9]));

    // 3+4. Remove the shared head block for both workers. This empties N's
    // full-edge coverage and drops its children (S=[3,4], X=[9]) from the
    // live tree. (Each remove of the head hash cuts that worker's coverage
    // of N to zero and eagerly scrubs N's own edge hashes.)
    let head = remove_hashes_with_parent(&[], &[1, 2]);
    apply_direct(
        &index,
        &mut l1,
        remove_event(1, 100, 0, vec![head[0], head[1]]),
    );
    apply_direct(
        &index,
        &mut l2,
        remove_event(2, 101, 0, vec![head[0], head[1]]),
    );

    // w1 still holds stale entries for the [3,4] suffix externals.
    assert_eq!(worker_lookup_len(&l1, w1), Some(2));

    // 5. The engine evicts the suffix blocks; correct producer emits removes.
    let suffix = remove_hashes_with_parent(&[1, 2], &[3, 4]);
    apply_direct(&index, &mut l1, remove_event(1, 102, 0, suffix));

    // Every stored block has now been removed for w1: its lookup must be
    // empty, or tracked_blocks counts phantom blocks forever.
    assert_eq!(worker_lookup_len(&l1, w1), Some(0));

    // And w2's cleanup path must also drain fully.
    let w2_leaf = remove_hashes_with_parent(&[1, 2], &[9]);
    apply_direct(&index, &mut l2, remove_event(2, 103, 0, w2_leaf));
    assert_eq!(worker_lookup_len(&l2, w2), Some(0));
}

#[test]
fn successful_repair_does_not_restore_scrubbed_other_worker_entries() {
    let index = ConcurrentRadixTreeCompressed::new();
    let scrubbed_worker = worker(1);
    let repairing_worker = worker(2);
    let mut shared_lookup = direct_lookup();
    let mut splitter_lookup = direct_lookup();

    apply_direct(
        &index,
        &mut shared_lookup,
        make_store_event(1, &[1, 2, 3, 4]),
    );
    apply_direct(
        &index,
        &mut shared_lookup,
        make_store_event(2, &[1, 2, 3, 4]),
    );

    // A split from another event thread leaves both local lookups pointing at
    // the old prefix node for the [3, 4] suffix.
    apply_direct(
        &index,
        &mut splitter_lookup,
        make_store_event(3, &[1, 2, 9]),
    );
    let suffix_hashes = remove_hashes_with_parent(&[1, 2], &[3, 4]);

    // A resolve-miss remove scrubs lookup state but cannot update coverage on
    // the node it failed to find. Model that boundary directly while keeping
    // the resolved live node's coverage intact.
    for &hash in &suffix_hashes {
        shared_lookup.remove(scrubbed_worker, hash);
    }
    assert_eq!(worker_lookup_len(&shared_lookup, scrubbed_worker), Some(2));
    assert_direct_score(&index, &[1, 2, 3, 4], scrubbed_worker, 4);

    // A different worker's stale lookup successfully resolves the suffix and
    // repairs the event thread's shared lookup map.
    let guard = crossbeam_epoch::pin();
    let resolved = index
        .resolve_lookup(
            &mut shared_lookup,
            index.event_worker_for_test(repairing_worker, &guard),
            suffix_hashes[0],
            LookupRepairDirection::TowardTail,
        )
        .expect("repairing worker should resolve the split suffix");
    drop(guard);

    shared_lookup.assert_invariants();
    for &hash in &suffix_hashes {
        assert!(Arc::ptr_eq(
            &shared_lookup.node(repairing_worker, hash).unwrap(),
            &resolved
        ));
    }

    for hash in suffix_hashes {
        assert!(
            !shared_lookup.contains(scrubbed_worker, hash),
            "repair for another worker restored a scrubbed lookup entry"
        );
    }
}

#[test]
fn drop_frees_the_tree_without_waiting_for_the_epoch() {
    let index = ConcurrentRadixTreeCompressed::new();
    let mut lookup = direct_lookup();
    apply_direct(&index, &mut lookup, make_store_event(1, &[1, 2, 3]));
    apply_direct(&index, &mut lookup, make_store_event(2, &[1, 2, 9]));
    drop(lookup);

    let leaf = index
        .root
        .child_snapshot(LocalBlockHash(1))
        .and_then(|prefix| prefix.child_snapshot(LocalBlockHash(3)))
        .expect("the split moves [3] under [1, 2]");
    // Wait out snapshots retired while building, so only the parent map and `leaf` hold it.
    let deadline = Instant::now() + Duration::from_secs(30);
    while Arc::strong_count(&leaf) != 2 {
        assert!(Instant::now() < deadline, "retired snapshots never expired");
        children::NodeChildren::flush_retired();
        children::NodeChildren::drain_graveyard(usize::MAX);
        thread::yield_now();
    }
    let weak = Arc::downgrade(&leaf);
    drop(leaf);

    drop(index);
    assert!(
        weak.upgrade().is_none(),
        "the drop left a node to the epoch"
    );
}

/// Writers validate lookup entries inside their own locked operation instead of
/// probing the node first, and repair only when that operation reports a miss.
mod fused_lookup_validation_tests {
    use super::*;

    fn repair_scans(index: &ConcurrentRadixTreeCompressed) -> u64 {
        index
            .bench_metrics
            .lookup_repair_scans
            .load(Ordering::Relaxed)
    }

    fn split_six_block_chain_at_three() -> (ConcurrentRadixTreeCompressed, DirectLookup) {
        let index = ConcurrentRadixTreeCompressed::new();
        let mut lookup0 = direct_lookup();
        let mut lookup1 = direct_lookup();
        apply_direct(
            &index,
            &mut lookup0,
            make_store_event(0, &[1, 2, 3, 4, 5, 6]),
        );
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event(1, &[1, 2, 3, 4, 5, 6]),
        );
        // A store on another event thread splits the edge; lookup0 still names the
        // prefix node for the moved suffix.
        apply_direct(
            &index,
            &mut lookup1,
            make_store_event_with_parent(1, &[1, 2, 3], &[7]),
        );
        (index, lookup0)
    }

    #[test]
    fn remove_with_stale_head_repairs_once_and_consumes_run_on_suffix() {
        let (index, mut lookup0) = split_six_block_chain_at_three();
        let worker0 = worker(0);
        let worker1 = worker(1);
        let prefix = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        let suffix = prefix.child_snapshot(LocalBlockHash(4)).unwrap();
        let moved = remove_hashes_with_parent(&[1, 2, 3], &[4, 5, 6]);
        assert!(Arc::ptr_eq(
            &lookup0.node(worker0, moved[1]).unwrap(),
            &prefix
        ));
        assert_eq!(repair_scans(&index), 0);

        apply_direct(
            &index,
            &mut lookup0,
            remove_event(0, 0, 0, moved[1..].to_vec()),
        );

        // One scan resolves the run head; the rest of the run is consumed on the
        // suffix under the same lock instead of being resolved hash by hash.
        assert_eq!(repair_scans(&index), 1);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 4);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(4));
        // Head-ward repair pointed the surviving suffix entry at the suffix, so a
        // later remove of it needs no second scan.
        assert!(Arc::ptr_eq(
            &lookup0.node(worker0, moved[0]).unwrap(),
            &suffix
        ));
        apply_direct(&index, &mut lookup0, remove_event(0, 1, 0, vec![moved[0]]));
        assert_eq!(repair_scans(&index), 1);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker0, 3);
        assert_eq!(worker_lookup_len(&lookup0, worker0), Some(3));
    }

    #[test]
    fn store_with_stale_parent_reports_missing_and_repairs_once() {
        let (index, mut lookup0) = split_six_block_chain_at_three();
        let worker0 = worker(0);
        let prefix = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        let continuation = stored_data(make_store_event_with_parent(0, &[1, 2, 3, 4, 5, 6], &[8]));
        let parent_hash = continuation.parent_hash.unwrap();
        assert!(Arc::ptr_eq(
            &lookup0.node(worker0, parent_hash).unwrap(),
            &prefix
        ));
        assert!(matches!(
            prefix.parent_coverage(slot(&index, worker0), parent_hash),
            ParentCoverage::Missing
        ));

        apply_direct(
            &index,
            &mut lookup0,
            make_store_event_with_parent(0, &[1, 2, 3, 4, 5, 6], &[8]),
        );

        assert_eq!(repair_scans(&index), 1);
        assert_eq!(
            index.edge_topology_for_test(),
            vec![edge_topology(
                &[1, 2, 3],
                vec![
                    edge_topology(&[4, 5, 6, 8], vec![]),
                    edge_topology(&[7], vec![])
                ],
            )],
        );
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6, 8], worker0, 7);
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker(1), 6);
    }

    #[test]
    fn store_with_resolved_but_uncovered_stale_parent_fails_without_spinning() {
        let (index, mut lookup0) = split_six_block_chain_at_three();
        let worker0 = worker(0);
        let prefix = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        // A worker-removal sweep on another lane drops worker 0 from the tree while
        // lookup0, which that sweep did not see, keeps the stale entry.
        index.remove_worker_coverage(
            &mut direct_lookup(),
            WorkerRemovalTarget::WorkerId(worker0.worker_id),
            true,
        );
        let continuation = make_store_event_with_parent(0, &[1, 2, 3, 4, 5, 6], &[8]);
        let parent_hash = stored_data(continuation.clone()).parent_hash.unwrap();
        assert!(Arc::ptr_eq(
            &lookup0.node(worker0, parent_hash).unwrap(),
            &prefix
        ));

        // Repair resolves the suffix but cannot rewrite the entry, because worker 0
        // no longer covers the parent there. Re-reading the entry would spin.
        let result = index.apply_event(&mut lookup0, continuation, None);

        assert!(matches!(
            result,
            Err(KvCacheEventError::ParentBlockNotFound)
        ));
        assert_eq!(repair_scans(&index), 1);
        assert!(lookup0.contains_worker(worker0));
        assert!(lookup0.node(worker0, parent_hash).is_none());
        lookup0.assert_invariants();
    }

    #[test]
    fn store_with_detached_stale_parent_fails_without_spinning() {
        let index = ConcurrentRadixTreeCompressed::new();
        let mut lookup1 = direct_lookup();
        let mut lookup2 = direct_lookup();
        apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
        apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 9]));
        // Evicting the shared head for both workers detaches the split suffix, so
        // lookup1's entries for [3, 4] name a node whose subtree no longer has them.
        let head = remove_hashes_with_parent(&[], &[1, 2]);
        apply_direct(&index, &mut lookup1, remove_event(1, 0, 0, head.clone()));
        apply_direct(&index, &mut lookup2, remove_event(2, 1, 0, head));
        assert_eq!(worker_lookup_len(&lookup1, worker(1)), Some(2));

        let result = index.apply_event(
            &mut lookup1,
            make_store_event_with_parent(1, &[1, 2, 3, 4], &[5]),
            None,
        );

        assert!(matches!(
            result,
            Err(KvCacheEventError::ParentBlockNotFound)
        ));
        assert_eq!(repair_scans(&index), 0);
    }

    #[test]
    fn anchor_flag_agrees_with_anchor_map() {
        let index = ConcurrentRadixTreeCompressed::new();
        let mut lookup = direct_lookup();
        apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2]));
        let regular = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        let regular_hashes = remove_hashes_with_parent(&[], &[1, 2]);

        // Anchor ids borrow router-owned prefix hashes, so one can equal a block
        // hash that a regular node also holds.
        let shared_id = regular_hashes[1];
        let private_id = ExternalSequenceBlockHash(0xA11C_0000);
        for anchor_id in [shared_id, private_id] {
            index
                .apply_anchor(
                    worker(1),
                    AnchorTask {
                        anchor_id,
                        anchor_local_hash: LocalBlockHash(2),
                        anchor_depth: 2,
                    },
                )
                .unwrap();
        }

        let map_says = |hash: ExternalSequenceBlockHash, node: &SharedNode| {
            index
                .anchor_nodes
                .get(&hash)
                .is_some_and(|anchor| Arc::ptr_eq(anchor.value(), node))
        };
        let shared_anchor = index.anchor_nodes.get(&shared_id).unwrap().clone();
        let private_anchor = index.anchor_nodes.get(&private_id).unwrap().clone();
        let cases = [
            (shared_id, &shared_anchor, true),
            (private_id, &private_anchor, true),
            (shared_id, &regular, false),
            (regular_hashes[0], &regular, false),
        ];
        for (hash, node, expected) in cases {
            assert_eq!(map_says(hash, node), expected);
            assert_eq!(index.is_anchor_node(hash, node), expected);
        }

        // A store under the anchor takes the anchor path and never extends its edge.
        let continuation = make_store_event_with_parent(3, &[1, 2], &[5]);
        apply_direct(&index, &mut lookup, continuation);
        assert_eq!(shared_anchor.edge_len_for_test(), 1);
        assert_eq!(shared_anchor.children_snapshot().len(), 1);
    }
}

mod slot_coverage_tests {
    use super::*;
    use crate::indexer::concurrent_radix_tree_compressed::coverage::{
        Slot, SlotSet, wait_for_pinned_threads,
    };
    use crate::test_utils::{make_store_event_full, router_event};

    fn blocks(locals: &[u64]) -> Vec<KvCacheStoredBlockData> {
        stored_data(make_store_event(0, locals)).blocks
    }

    fn slots(indices: &[u16]) -> SlotSet {
        indices.iter().copied().map(Slot::new).collect()
    }

    fn wait_for_slot_release(index: &ConcurrentRadixTreeCompressed, slot: Slot) {
        while !index.slots.is_released(slot) {
            wait_for_pinned_threads();
        }
    }

    /// Splitting at `pos` keeps full ranks on both halves, promotes cutoffs at or past
    /// the boundary on the prefix, carries the remainder of cutoffs past it to the
    /// suffix, and moves the original children to the suffix.
    #[test]
    fn split_moves_cutoffs_at_and_past_the_boundary() {
        let node = Arc::new(Node::from_blocks_for_slot(
            &blocks(&[1, 2, 3, 4, 5, 6]),
            Slot::new(0),
        ));
        let child_a = Arc::new(Node::from_blocks_for_slot(&blocks(&[7]), Slot::new(0)));
        let child_b = Arc::new(Node::from_blocks_for_slot(&blocks(&[8]), Slot::new(0)));
        node.attach_child_for_test(child_a.clone());
        node.attach_child_for_test(child_b.clone());
        // Slot 300 exercises an overflow chunk on both halves.
        node.set_coverage_for_test(
            &[Slot::new(0), Slot::new(300)],
            &[
                (Slot::new(1), 1),
                (Slot::new(2), 3),
                (Slot::new(3), 5),
                (Slot::new(301), 4),
            ],
        );

        let suffix = node.split_for_test(3);

        assert_eq!(node.edge_len_for_test(), 3);
        assert_eq!(suffix.edge_len_for_test(), 3);
        let (prefix_full, prefix_cutoffs) = node.coverage_for_test();
        assert_eq!(prefix_full, slots(&[0, 2, 3, 300, 301]));
        assert_eq!(prefix_cutoffs, vec![(Slot::new(1), 1)]);
        let (suffix_full, suffix_cutoffs) = suffix.coverage_for_test();
        assert_eq!(suffix_full, slots(&[0, 300]));
        assert_eq!(suffix_cutoffs, vec![(Slot::new(3), 2), (Slot::new(301), 1)]);

        // The suffix keeps the original children; the prefix holds only the suffix.
        let prefix_children = node.child_edges_snapshot();
        assert_eq!(prefix_children.len(), 1);
        assert!(Arc::ptr_eq(&prefix_children[0].1, &suffix));
        assert!(Arc::ptr_eq(
            &suffix.child_snapshot(LocalBlockHash(7)).unwrap(),
            &child_a
        ));
        assert!(Arc::ptr_eq(
            &suffix.child_snapshot(LocalBlockHash(8)).unwrap(),
            &child_b
        ));
    }

    #[test]
    fn extension_demotes_other_full_ranks_to_the_old_tail() {
        let index = ConcurrentRadixTreeCompressed::new();
        let worker1 = worker(1);
        let worker2 = worker(2);
        let mut lookup = direct_lookup();

        apply_direct(&index, &mut lookup, make_store_event(1, &[1, 2, 3]));
        apply_direct(&index, &mut lookup, make_store_event(2, &[1, 2, 3]));
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(1, &[1, 2, 3], &[4, 5]),
        );

        assert_edge_lengths(&index, &[5]);
        let node = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        let (full, cutoffs) = node.coverage_for_test();
        assert_eq!(full, [slot(&index, worker1)].into_iter().collect());
        assert_eq!(cutoffs, vec![(slot(&index, worker2), 3)]);
        assert_direct_score(&index, &[1, 2, 3, 4, 5], worker1, 5);
        assert_direct_score(&index, &[1, 2, 3, 4, 5], worker2, 3);
    }

    struct ModelNode {
        edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
        full: FxHashSet<WorkerWithDpRank>,
        cutoffs: FxHashMap<WorkerWithDpRank, usize>,
    }

    type WalkScores = (
        FxHashMap<WorkerWithDpRank, u32>,
        FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
    );

    /// The hash-set walk this tree used before slots, without its equal-size skip, over
    /// a single chain of nodes.
    fn hash_set_walk(
        chain: &[ModelNode],
        query: &[LocalBlockHash],
        early_exit: bool,
    ) -> WalkScores {
        let mut scores = FxHashMap::default();
        let mut last = FxHashMap::default();
        let mut active = FxHashSet::default();
        let mut matched_depth = 0u32;
        let mut seq_pos = 0;
        let mut first_node = true;
        let mut prev_edge_last_hash = None;
        let mut next = (chain[0].edge[0].0 == query[0]).then_some(0);

        while seq_pos < query.len() {
            let Some(index) = next.take() else {
                break;
            };
            let node = &chain[index];
            let edge_len = node.edge.len();
            let walk_len = edge_len.min(query.len() - seq_pos);
            let mut edge_match_len = 1;
            while edge_match_len < walk_len
                && node.edge[edge_match_len].0 == query[seq_pos + edge_match_len]
            {
                edge_match_len += 1;
            }
            let edge_hash_at = |depth: usize| node.edge[depth - 1].1;

            if first_node {
                active = node.full.clone();
                for (&worker, &cutoff) in &node.cutoffs {
                    let contribution = cutoff.min(edge_match_len);
                    if contribution > 0 {
                        scores.insert(worker, contribution as u32);
                        last.insert(worker, edge_hash_at(contribution));
                    }
                }
            } else {
                active.retain(|worker| {
                    if node.full.contains(worker) {
                        return true;
                    }
                    let effective = node
                        .cutoffs
                        .get(worker)
                        .map_or(0, |&cutoff| cutoff.min(edge_match_len));
                    scores.insert(*worker, matched_depth + effective as u32);
                    if effective > 0 {
                        last.insert(*worker, edge_hash_at(effective));
                    } else if let Some(hash) = prev_edge_last_hash {
                        last.insert(*worker, hash);
                    }
                    false
                });
            }

            let active_count = active.len();
            next = (edge_match_len == edge_len
                && active_count > 0
                && seq_pos + edge_match_len < query.len()
                && chain
                    .get(index + 1)
                    .is_some_and(|child| child.edge[0].0 == query[seq_pos + edge_match_len]))
            .then_some(index + 1);
            prev_edge_last_hash = Some(node.edge[edge_match_len - 1].1);
            first_node = false;

            if active_count == 0 {
                break;
            }
            matched_depth += edge_match_len as u32;
            if edge_match_len < edge_len {
                break;
            }
            seq_pos += edge_match_len;
            if early_exit && active_count == 1 {
                break;
            }
        }

        for worker in active {
            scores.insert(worker, matched_depth);
            if let Some(hash) = prev_edge_last_hash {
                last.insert(worker, hash);
            }
        }
        (scores, last)
    }

    /// Random coverage, including sets that are not prefix-closed and sets of equal size,
    /// must score exactly like the always-intersecting hash-set walk.
    #[test]
    fn slot_walk_matches_hash_set_walk_on_random_coverage() {
        const WORKERS: u64 = 300;
        let mut rng = fastrand::Rng::with_seed(0x2545_F491_4F6C_DD1D);

        for _ in 0..200 {
            let index = ConcurrentRadixTreeCompressed::new();
            let ranks: Vec<_> = (0..WORKERS)
                .map(|id| (worker(id), slot(&index, worker(id))))
                .collect();
            // Workers come from a small pool so coverage sets often coincide in size.
            let pool: Vec<_> = (0..rng.usize(2..14))
                .map(|_| ranks[rng.usize(..WORKERS as usize)])
                .collect();

            let chain_len = rng.u64(1..=5);
            let mut chain = Vec::new();
            let mut locals = Vec::new();
            let mut parent = index.root.clone();
            for depth in 0..chain_len {
                let edge_len = rng.u64(1..=4);
                let edge_locals: Vec<u64> = (0..edge_len).map(|i| depth * 10 + i + 1).collect();
                let mut all = locals.clone();
                all.extend(&edge_locals);
                let edge_blocks = blocks(&all)[locals.len()..].to_vec();
                locals = all;

                let mut full = Vec::new();
                let mut cutoffs = Vec::new();
                let mut model = ModelNode {
                    edge: edge_blocks
                        .iter()
                        .map(|block| (block.tokens_hash, block.block_hash))
                        .collect(),
                    full: FxHashSet::default(),
                    cutoffs: FxHashMap::default(),
                };
                for &(rank, rank_slot) in &pool {
                    match rng.u32(..4) {
                        0 | 1 => {
                            full.push(rank_slot);
                            model.full.insert(rank);
                        }
                        2 if edge_len > 1 => {
                            let cutoff = rng.usize(1..edge_len as usize);
                            cutoffs.push((rank_slot, cutoff));
                            model.cutoffs.insert(rank, cutoff);
                        }
                        _ => {}
                    }
                }
                let node = Arc::new(Node::from_blocks_for_slot(&edge_blocks, Slot::new(0)));
                node.set_coverage_for_test(&full, &cutoffs);
                parent.attach_child_for_test(node.clone());
                parent = node;
                chain.push(model);
            }

            for _ in 0..8 {
                let mut query: Vec<_> = locals[..rng.usize(1..=locals.len())]
                    .iter()
                    .copied()
                    .map(LocalBlockHash)
                    .collect();
                if rng.u32(..3) == 0 {
                    let at = rng.usize(..query.len());
                    query[at] = LocalBlockHash(9999);
                }
                let early_exit = rng.u32(..4) == 0;

                let expected = hash_set_walk(&chain, &query, early_exit);
                let details = index.find_match_details_impl(&query, early_exit);
                assert_eq!(details.overlap_scores.scores, expected.0, "query={query:?}");
                assert_eq!(details.last_matched_hashes, expected.1, "query={query:?}");
            }
        }
    }

    /// After a head-first eviction the child still lists the evicted rank, so the child's
    /// full set has the same size as the walk's active set but different members. The
    /// walk must intersect anyway and stop the rank that never stored the child.
    #[test]
    fn equal_sized_coverage_after_head_first_eviction_scores_exactly() {
        let index = ConcurrentRadixTreeCompressed::new();
        let evicted = worker(1);
        let sharer = worker(2);
        let brancher = worker(3);
        let mut lookup = direct_lookup();

        apply_direct(&index, &mut lookup, make_store_event(1, &[1, 2, 5, 6]));
        apply_direct(&index, &mut lookup, make_store_event(2, &[1, 2, 5, 6]));
        apply_direct(&index, &mut lookup, make_store_event(3, &[1, 2, 7]));
        assert_edge_lengths(&index, &[1, 2, 2]);
        // The evicted rank drops the shared head but still covers [5, 6].
        let head = remove_hashes_with_parent(&[], &[1]);
        apply_direct(&index, &mut lookup, remove_event(1, 10, 0, head));

        let scores = index.find_matches_impl(&local_hashes(&[1, 2, 5, 6]), false);
        assert_eq!(scores.scores.get(&sharer), Some(&4));
        assert_eq!(scores.scores.get(&brancher), Some(&2));
        assert!(!scores.scores.contains_key(&evicted));
    }

    /// Per-rank cache contents, keyed by sequence hash, used as the reference model.
    #[derive(Default)]
    struct RankModel {
        /// Sequence hash -> (parent sequence hash, cached children).
        cached: FxHashMap<u64, (Option<u64>, u32)>,
        live: Vec<Vec<u64>>,
    }

    impl RankModel {
        fn prefix_len(&self, seqs: &[u64]) -> usize {
            seqs.iter()
                .take_while(|seq| self.cached.contains_key(seq))
                .count()
        }
    }

    fn seq_hashes(locals: &[u64]) -> Vec<u64> {
        compute_seq_hash_for_block(&local_hashes(locals))
    }

    /// Leaf-first evictions keep every rank's coverage prefix-closed, so with one event
    /// lane the tree must score exactly like the model, also after whole-worker removals
    /// hand recycled slots to fresh worker ids.
    #[test]
    fn random_strict_streams_with_slot_recycling_match_the_model() {
        let mut rng = fastrand::Rng::with_seed(0x9E37_79B9_7F4A_7C15);
        let pool_seq = |s: u64, d: u64, u: u64| -> Vec<u64> {
            let mut seq: Vec<u64> = (0..1 + s % 4).map(|i| 1_000 + s * 10 + i).collect();
            seq.extend((0..d % 5).map(|i| 2_000 + s * 100 + d * 10 + i));
            seq.extend((0..1 + u % 3).map(|i| 3_000 + s * 1_000 + d * 100 + u * 10 + i));
            seq
        };

        let index = ConcurrentRadixTreeCompressed::new();
        let mut lookup = direct_lookup();
        let mut models: FxHashMap<WorkerWithDpRank, RankModel> = (0..8)
            .map(|id| (worker(id), RankModel::default()))
            .collect();
        let mut next_worker_id = 8;
        let mut recycled = 0;

        for step in 0..4_000u64 {
            let mut workers: Vec<_> = models.keys().copied().collect();
            workers.sort_by_key(|worker| worker.worker_id);
            let worker = workers[rng.usize(..workers.len())];
            let model = models.get_mut(&worker).unwrap();
            match rng.u32(..100) {
                0..=44 => {
                    let seq = pool_seq(rng.u64(..3), rng.u64(..4), rng.u64(..3));
                    let seqs = seq_hashes(&seq);
                    let target = rng.usize(1..=seq.len());
                    let cached = model.prefix_len(&seqs[..target]);
                    if cached < target {
                        let parent =
                            (cached > 0).then(|| ExternalSequenceBlockHash(seqs[cached - 1]));
                        let event =
                            make_store_event_full(worker.worker_id, &seq[..target], 0, None, None);
                        let mut op = stored_data(event);
                        op.parent_hash = parent;
                        op.blocks.drain(..cached);
                        apply_direct(
                            &index,
                            &mut lookup,
                            router_event(worker.worker_id, step, 0, KvCacheEventData::Stored(op)),
                        );
                        let mut parent = parent.map(|hash| hash.0);
                        for &seq_hash in &seqs[cached..target] {
                            if let Some(entry) = parent.and_then(|p| model.cached.get_mut(&p)) {
                                entry.1 += 1;
                            }
                            model.cached.insert(seq_hash, (parent, 0));
                            parent = Some(seq_hash);
                        }
                    }
                    model.live.push(seq[..target].to_vec());
                    if model.live.len() > 16 {
                        model.live.remove(0);
                    }
                }
                45..=89 => {
                    if model.live.is_empty() {
                        continue;
                    }
                    let seq = model.live[rng.usize(..model.live.len())].clone();
                    let seqs = seq_hashes(&seq);
                    let mut pos = model.prefix_len(&seqs);
                    let mut removed = Vec::new();
                    let want = rng.usize(1..=3);
                    while pos > 0 && removed.len() < want {
                        let seq_hash = seqs[pos - 1];
                        let children = model.cached[&seq_hash].1;
                        // Leaf first: a block may go once its only cached child went.
                        let only_child_removed = children == 1 && !removed.is_empty();
                        if children != 0 && !only_child_removed {
                            break;
                        }
                        removed.push(seq_hash);
                        pos -= 1;
                    }
                    if removed.is_empty() {
                        continue;
                    }
                    if rng.bool() {
                        removed.reverse();
                    }
                    apply_direct(
                        &index,
                        &mut lookup,
                        remove_event(
                            worker.worker_id,
                            step,
                            0,
                            removed
                                .iter()
                                .copied()
                                .map(ExternalSequenceBlockHash)
                                .collect(),
                        ),
                    );
                    for seq_hash in removed {
                        let (parent, _) = model.cached.remove(&seq_hash).unwrap();
                        if let Some(entry) = parent.and_then(|p| model.cached.get_mut(&p)) {
                            entry.1 -= 1;
                        }
                    }
                }
                90..=94 => {
                    apply_direct(
                        &index,
                        &mut lookup,
                        make_clear_event_with_dp_rank(worker.worker_id, 0),
                    );
                    *model = RankModel::default();
                }
                _ => {
                    let slot = index.slot_for_test(worker);
                    index.remove_worker_coverage(
                        &mut lookup,
                        WorkerRemovalTarget::WorkerId(worker.worker_id),
                        true,
                    );
                    lookup.assert_invariants();
                    models.remove(&worker);
                    if let Some(slot) = slot {
                        wait_for_slot_release(&index, slot);
                        recycled += 1;
                    }
                    models.insert(self::worker(next_worker_id), RankModel::default());
                    next_worker_id += 1;
                }
            }

            if step % 50 != 0 {
                continue;
            }
            let mut queries: Vec<Vec<u64>> = models
                .values()
                .flat_map(|m| m.live.iter().cloned())
                .collect();
            queries.extend((0..16).map(|_| pool_seq(rng.u64(..3), rng.u64(..4), rng.u64(..3))));
            for query in queries {
                let seqs = seq_hashes(&query);
                let expected: FxHashMap<_, _> = models
                    .iter()
                    .filter_map(|(&worker, model)| {
                        let len = model.prefix_len(&seqs);
                        (len > 0).then_some((worker, len as u32))
                    })
                    .collect();
                let got = index.find_matches_impl(&local_hashes(&query), false).scores;
                assert_eq!(got, expected, "step={step} query={query:?}");
            }
        }
        assert!(recycled > 0, "the stream never recycled a slot");
    }

    /// Builds `[1, 2] -> [3, 4, 5, 6]`, has every rank evict `[1]` so the child is unlinked
    /// while the old and surviving ranks still cover it, removes the old rank, and hands
    /// its slot to a fresh worker that stores the same blocks on the reachable path.
    /// Returns the index, the lane holding the survivor and the fresh worker, and both.
    fn recycled_slot_over_unlinked_subtree() -> (
        ConcurrentRadixTreeCompressed,
        DirectLookup,
        WorkerWithDpRank,
        WorkerWithDpRank,
    ) {
        let index = ConcurrentRadixTreeCompressed::new();
        let old = worker(1);
        let survivor = worker(2);
        let fresh = worker(4);
        let mut lane = direct_lookup();
        let mut other_lane = direct_lookup();

        // [1, 2] is internal before the long chain arrives, so the chain is its child.
        apply_direct(&index, &mut other_lane, make_store_event(3, &[1, 2]));
        apply_direct(
            &index,
            &mut other_lane,
            make_store_event_with_parent(3, &[1, 2], &[9]),
        );
        apply_direct(&index, &mut lane, make_store_event(1, &[1, 2, 3, 4, 5, 6]));
        apply_direct(&index, &mut lane, make_store_event(2, &[1, 2, 3, 4, 5, 6]));
        assert_edge_lengths(&index, &[1, 2, 4]);
        let old_slot = index.slot_for_test(old).unwrap();

        let head = remove_hashes_with_parent(&[], &[1]);
        apply_direct(&index, &mut lane, remove_event(1, 10, 0, head.clone()));
        apply_direct(&index, &mut lane, remove_event(2, 11, 0, head.clone()));
        apply_direct(&index, &mut other_lane, remove_event(3, 12, 0, head));
        assert_eq!(index.raw_child_edge_count(), 1);

        index.remove_worker_coverage(&mut lane, WorkerRemovalTarget::WorkerId(1), true);
        wait_for_slot_release(&index, old_slot);
        apply_direct(&index, &mut lane, make_store_event(4, &[1, 2, 3, 4, 5, 6]));
        assert_eq!(index.slot_for_test(fresh), Some(old_slot));
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], fresh, 6);
        assert!(
            !index
                .find_matches_impl(&local_hashes(&[1, 2, 3, 4, 5, 6]), false)
                .scores
                .contains_key(&survivor)
        );
        (index, lane, survivor, fresh)
    }

    /// A split on the unlinked subtree copies the recycled slot's stale bit to the
    /// suffix; it must not move the fresh worker's entries there.
    #[test]
    fn split_on_unlinked_subtree_keeps_recycled_slot_entries_in_place() {
        let (index, mut lane, survivor, fresh) = recycled_slot_over_unlinked_subtree();

        // The survivor's lookup still names the unlinked [3, 4, 5, 6] and splits it.
        apply_direct(
            &index,
            &mut lane,
            make_store_event_with_parent(survivor.worker_id, &[1, 2, 3, 4], &[7]),
        );
        assert_eq!(index.raw_child_edge_count(), 2);

        // Evicting the tail must cut the fresh worker on the reachable path.
        let tail = remove_hashes_with_parent(&[1, 2, 3, 4, 5], &[6]);
        apply_direct(
            &index,
            &mut lane,
            remove_event(fresh.worker_id, 20, 0, tail),
        );
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], fresh, 5);
    }

    /// Lazy repair into the unlinked subtree resolves a node carrying the recycled slot's
    /// stale bit; it must not move the fresh worker's entries there either.
    #[test]
    fn repair_into_unlinked_subtree_keeps_recycled_slot_entries_in_place() {
        let (index, mut lane, survivor, fresh) = recycled_slot_over_unlinked_subtree();
        let h5 = remove_hashes_with_parent(&[1, 2, 3, 4], &[5]);
        let h6 = remove_hashes_with_parent(&[1, 2, 3, 4, 5], &[6]);
        let unlinked = lane
            .node(survivor, h6[0])
            .expect("the survivor still names the unlinked node");

        // A rank on another lane that also names the unlinked node splits it, leaving
        // this lane's survivor entries for [5, 6] on the prefix.
        let splitter = worker(5);
        let mut splitter_lane = direct_lookup();
        let continuation = stored_data(make_store_event_with_parent(5, &[1, 2, 3, 4], &[8]));
        unlinked.set_coverage_for_test(
            &[
                index.slot_for_test(survivor).unwrap(),
                index.slot_for_test(fresh).unwrap(),
                slot(&index, splitter),
            ],
            &[],
        );
        splitter_lane.insert(splitter, continuation.parent_hash.unwrap(), &unlinked);
        apply_direct(
            &index,
            &mut splitter_lane,
            make_store_event_with_parent(5, &[1, 2, 3, 4], &[8]),
        );
        assert_eq!(unlinked.edge_len_for_test(), 2);

        // The survivor's stale entry repairs this lane toward the head of the suffix.
        apply_direct(
            &index,
            &mut lane,
            remove_event(survivor.worker_id, 21, 0, h6.clone()),
        );

        // The fresh worker evicts tail first; both blocks must leave the reachable path.
        apply_direct(&index, &mut lane, remove_event(fresh.worker_id, 22, 0, h6));
        apply_direct(&index, &mut lane, remove_event(fresh.worker_id, 23, 0, h5));
        assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], fresh, 4);
    }

    /// A promote under the shared gate either lands before a split, which then copies
    /// the bit to the suffix, or sees the split's version bump and does nothing.
    #[test]
    fn promote_racing_split_keeps_the_slot_on_both_halves_or_neither() {
        let promoted_slot = Slot::new(5);
        for _ in 0..2_000 {
            let node = Arc::new(Node::from_blocks_for_slot(
                &blocks(&[1, 2, 3, 4, 5, 6]),
                Slot::new(0),
            ));
            let version = node.shape_version_for_test();
            let barrier = Arc::new(Barrier::new(2));

            let promoter_node = node.clone();
            let promoter_barrier = barrier.clone();
            let promoter = thread::spawn(move || {
                promoter_barrier.wait();
                promoter_node.promote_to_full_with_version(promoted_slot, version)
            });
            barrier.wait();
            let suffix = node.split_for_test(3);
            let promoted = promoter.join().unwrap();

            let on_prefix = node.coverage_for_test().0.contains(promoted_slot);
            let on_suffix = suffix.coverage_for_test().0.contains(promoted_slot);
            match promoted {
                Some(true) => assert!(on_prefix && on_suffix),
                None => assert!(!on_prefix && !on_suffix),
                Some(false) => panic!("the slot was never full"),
            }
        }
    }

    /// A split promotes a cutoff that reaches its split point to full coverage on the
    /// prefix but leaves only the remainder on the suffix, so a plan made before the
    /// split must be rejected even though the slot's bit on the prefix is now set.
    #[test]
    fn stale_plans_fail_after_a_split_promoted_their_cutoff() {
        let partial = Slot::new(7);
        let node = Node::from_blocks_for_slot(&blocks(&[1, 2, 3, 4, 5, 6]), Slot::new(0));
        node.set_coverage_for_test(&[Slot::new(0)], &[(partial, 4)]);
        let version = node.shape_version_for_test();

        let suffix = node.split_for_test(3);
        assert!(node.coverage_for_test().0.contains(partial));
        assert_eq!(suffix.coverage_for_test().1, vec![(partial, 1)]);

        assert_eq!(node.promote_to_full_with_version(partial, version), None);
        assert_eq!(node.cover_prefix_with_version(partial, 2, version), None);
        let current = node.shape_version_for_test();
        assert_eq!(
            node.promote_to_full_with_version(partial, current),
            Some(false)
        );
        assert_eq!(suffix.coverage_for_test().1, vec![(partial, 1)]);
    }

    /// Promotes under the shared gate and whole-slot drops under the exclusive gate
    /// flip distinct bits of one word without losing an update.
    #[test]
    fn concurrent_promotes_and_drops_on_one_word_lose_no_update() {
        const THREADS: u16 = 32;
        const ROUNDS: usize = 2_000;
        let edge = blocks(&[1, 2, 3]);
        let head = edge[0].block_hash;
        let node = Arc::new(Node::from_blocks_for_slot(&edge, Slot::new(63)));
        let version = node.shape_version_for_test();
        let barrier = Arc::new(Barrier::new(THREADS as usize));

        let handles: Vec<_> = (0..THREADS)
            .map(|index| {
                let node = node.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let slot = Slot::new(index);
                    barrier.wait();
                    for _ in 0..ROUNDS {
                        assert_eq!(node.promote_to_full_with_version(slot, version), Some(true));
                        assert!(node.coverage_for_test().0.contains(slot));
                        let removal = node
                            .remove_worker_for_leading_hashes(slot, &[head])
                            .unwrap();
                        assert_eq!((removal.consumed, removal.stale_hashes.len()), (1, 3));
                        assert!(!node.coverage_for_test().0.contains(slot));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(node.coverage_for_test().0, slots(&[63]));
    }

    #[test]
    fn promote_deletes_only_its_own_stale_cutoff() {
        let node = Node::from_blocks_for_slot(&blocks(&[1, 2, 3, 4]), Slot::new(0));
        node.set_coverage_for_test(&[Slot::new(0)], &[(Slot::new(1), 2), (Slot::new(2), 3)]);
        let version = node.shape_version_for_test();

        assert_eq!(
            node.promote_to_full_with_version(Slot::new(3), version),
            Some(true)
        );
        assert_eq!(
            node.promote_to_full_with_version(Slot::new(1), version),
            Some(true)
        );
        assert_eq!(
            node.promote_to_full_with_version(Slot::new(1), version),
            Some(false)
        );
        // A stale version fails whether or not the bit is already set.
        assert_eq!(
            node.promote_to_full_with_version(Slot::new(1), version + 1),
            None
        );
        assert_eq!(
            node.promote_to_full_with_version(Slot::new(4), version + 1),
            None
        );
        let (full, cutoffs) = node.coverage_for_test();
        assert_eq!(full, slots(&[0, 1, 3]));
        assert_eq!(cutoffs, vec![(Slot::new(2), 3)]);

        assert_eq!(
            node.promote_to_full_with_version(Slot::new(2), version),
            Some(true)
        );
        assert!(node.coverage_for_test().1.is_empty());
        assert_eq!(
            node.promote_to_full_with_version(Slot::new(5), version),
            Some(true)
        );
        assert_eq!(node.coverage_for_test().0, slots(&[0, 1, 2, 3, 5]));
    }

    /// Readers racing a rank that drops off and returns to the head node only ever see
    /// it fully matched or absent.
    #[test]
    fn whole_slot_drops_racing_readers_never_overcount() {
        let index = Arc::new(ConcurrentRadixTreeCompressed::new());
        let stable = worker(1);
        let toggled = worker(2);
        let mut lookup = direct_lookup();
        apply_direct(&index, &mut lookup, make_store_event(1, &[1, 2, 3, 4]));
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6, 7, 8]),
        );
        apply_direct(&index, &mut lookup, make_store_event(2, &[1, 2, 3, 4]));
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(2, &[1, 2, 3, 4], &[9]),
        );
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(2, &[1, 2, 3, 4], &[5, 6, 7, 8]),
        );
        assert_edge_lengths(&index, &[1, 4, 4]);
        let head = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        let head_hash = remove_hashes_with_parent(&[], &[1])[0];
        let toggled_slot = slot(&index, toggled);
        let version = head.shape_version_for_test();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reads = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let index = index.clone();
                let stop = stop.clone();
                let reads = reads.clone();
                thread::spawn(move || {
                    let query = local_hashes(&[1, 2, 3, 4, 5, 6, 7, 8]);
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let scores = index.find_matches_impl(&query, false).scores;
                        assert_eq!(scores.get(&stable), Some(&8));
                        assert!(
                            matches!(scores.get(&toggled), None | Some(&8)),
                            "{scores:?}"
                        );
                        reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                })
            })
            .collect();

        // Keep toggling until the readers have overlapped plenty of drops and promotes.
        while reads.load(std::sync::atomic::Ordering::Relaxed) < 20_000 {
            assert!(
                head.remove_worker_for_leading_hashes(toggled_slot, &[head_hash])
                    .is_some()
            );
            assert_eq!(
                head.promote_to_full_with_version(toggled_slot, version),
                Some(true)
            );
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for reader in readers {
            reader.join().unwrap();
        }
    }

    #[tokio::test]
    async fn removed_worker_slot_is_recycled_without_its_coverage() {
        let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 4, 32);
        let old = WorkerWithDpRank::new(1, 0);
        let other = WorkerWithDpRank::new(2, 0);
        let fresh = WorkerWithDpRank::new(3, 0);

        index.apply_event(make_store_event(1, &[1, 2, 3, 4])).await;
        index.apply_event(make_store_event(2, &[1, 2])).await;
        index.apply_event(make_store_event(1, &[7, 8])).await;
        flush_and_settle(&index).await;
        let old_slot = index.backend().slot_for_test(old).unwrap();

        index.remove_worker(1).await;
        flush_and_settle(&index).await;
        assert_eq!(index.backend().slot_for_test(old), None);
        wait_for_slot_release(index.backend(), old_slot);

        index.apply_event(make_store_event(3, &[1, 2, 3])).await;
        flush_and_settle(&index).await;
        assert_eq!(index.backend().slot_for_test(fresh), Some(old_slot));

        let scores = index
            .find_matches(local_hashes(&[1, 2, 3, 4]))
            .await
            .unwrap();
        assert_eq!(scores.scores.get(&fresh), Some(&3));
        assert_eq!(scores.scores.get(&other), Some(&2));
        assert!(!scores.scores.contains_key(&old));
        let scores = index.find_matches(local_hashes(&[7, 8])).await.unwrap();
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    async fn clear_keeps_the_slot() {
        let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32);
        let rank = WorkerWithDpRank::new(1, 0);

        index.apply_event(make_store_event(1, &[1, 2, 3])).await;
        flush_and_settle(&index).await;
        let slot = index.backend().slot_for_test(rank).unwrap();

        index.apply_event(make_clear_event_with_dp_rank(1, 0)).await;
        flush_and_settle(&index).await;
        assert_eq!(index.backend().slot_for_test(rank), Some(slot));
        assert!(
            index
                .find_matches(local_hashes(&[1, 2, 3]))
                .await
                .unwrap()
                .scores
                .is_empty()
        );

        index.apply_event(make_store_event(1, &[1, 2])).await;
        flush_and_settle(&index).await;
        assert_eq!(index.backend().slot_for_test(rank), Some(slot));
        assert_score(&index, &[1, 2, 3], rank, 2).await;
    }

    #[tokio::test]
    async fn rank_removal_and_restore_takes_a_new_mapping_without_old_coverage() {
        let index = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32);
        let rank0 = WorkerWithDpRank::new(1, 0);
        let rank1 = WorkerWithDpRank::new(1, 1);

        index
            .apply_event(make_store_event_with_dp_rank(1, &[1, 2, 3], 0))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(1, &[1, 2, 3], 1))
            .await;
        flush_and_settle(&index).await;
        assert!(index.backend().slot_for_test(rank0).is_some());
        let rank1_slot = index.backend().slot_for_test(rank1).unwrap();

        index
            .reset_worker_dp_rank_and_wait(1, 0)
            .await
            .expect("rank reset should complete");
        assert_eq!(index.backend().slot_for_test(rank0), None);
        assert_eq!(index.backend().slot_for_test(rank1), Some(rank1_slot));

        index
            .apply_event(make_store_event_with_dp_rank(1, &[1], 0))
            .await;
        flush_and_settle(&index).await;
        assert!(index.backend().slot_for_test(rank0).is_some());
        assert_score(&index, &[1, 2, 3], rank0, 1).await;
        assert_score(&index, &[1, 2, 3], rank1, 3).await;
    }

    /// An event that resolved a slot before its rank was unmapped may still be setting
    /// bits. The release waits for it, so its late bit is swept before the slot is freed.
    #[test]
    fn release_waits_for_events_that_resolved_the_old_slot() {
        let index = Arc::new(ConcurrentRadixTreeCompressed::new());
        let removed = worker(1);
        let mut lookup = direct_lookup();
        apply_direct(&index, &mut lookup, make_store_event(1, &[1, 2, 3]));
        apply_direct(&index, &mut lookup, make_store_event(2, &[7, 8]));
        let node = index.root.child_snapshot(LocalBlockHash(7)).unwrap();
        let removed_slot = slot(&index, removed);

        let (resolved_tx, resolved_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let late_index = index.clone();
        let late_node = node.clone();
        // Stands in for an event on another lane that resolved the slot under its guard.
        let late_event = thread::spawn(move || {
            let guard = crossbeam_epoch::pin();
            let slot = late_index.slots.table(&guard).slot_of(removed).unwrap();
            resolved_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            late_node.promote_slot_to_full_edge(slot);
            drop(guard);
        });
        resolved_rx.recv().unwrap();

        let remover_index = index.clone();
        let remover = thread::spawn(move || {
            remover_index.remove_worker_coverage(
                &mut direct_lookup(),
                WorkerRemovalTarget::DpRank(removed),
                true,
            );
        });
        // Leave a removal that does not wait for the pinned event time to finish first.
        thread::sleep(std::time::Duration::from_millis(50));
        go_tx.send(()).unwrap();
        late_event.join().unwrap();
        remover.join().unwrap();

        assert_eq!(index.slot_for_test(removed), None);
        assert!(!node.coverage_for_test().0.contains(removed_slot));
    }

    /// A sweep drops each node it has cleared, so a split suffix allocated later can reuse
    /// a freed node's address. The sweep must still visit that suffix, which carries the
    /// slot, or the slot's next owner is credited past what it stored.
    #[test]
    fn sweep_visits_a_split_suffix_that_reuses_a_freed_node_address() {
        let index = ConcurrentRadixTreeCompressed::new();
        let removed = worker(1);
        let fresh = worker(4);
        let mut removed_lane = direct_lookup();
        let mut helper_lane = direct_lookup();

        // [1] has five children only the removed worker covers, so clearing [1] frees them.
        for child in 101..=105 {
            apply_direct(&index, &mut removed_lane, make_store_event(1, &[1, child]));
        }
        // [20] -> [21] -> [22, 23, 24, 25], shared with the helper.
        for lane_worker in [1, 3] {
            let lane = if lane_worker == 1 {
                &mut removed_lane
            } else {
                &mut helper_lane
            };
            apply_direct(
                &index,
                lane,
                make_store_event(lane_worker, &[20, 21, 22, 23, 24, 25]),
            );
        }
        apply_direct(&index, &mut helper_lane, make_store_event(3, &[20, 30]));
        apply_direct(&index, &mut helper_lane, make_store_event(3, &[20, 21, 31]));
        let target = index
            .root
            .child_snapshot(LocalBlockHash(20))
            .and_then(|node| node.child_snapshot(LocalBlockHash(21)))
            .and_then(|node| node.child_snapshot(LocalBlockHash(22)))
            .unwrap();
        assert_eq!(target.edge_len_for_test(), 4);
        let target = Arc::as_ptr(&target) as usize;
        let removed_slot = index.slot_for_test(removed).unwrap();

        // The removed worker's lookup would keep its nodes alive.
        drop(removed_lane);
        let slots = index.slots.unmap(WorkerRemovalTarget::WorkerId(1));
        wait_for_pinned_threads();
        let mut split = false;
        index.sweep_slots(&slots.iter().copied().collect(), &mut 0, |node, _| {
            if !split && Arc::as_ptr(node) as usize == target {
                split = true;
                // Another lane splits [22, 23 | 24, 25] just before the sweep reaches it.
                apply_direct(
                    &index,
                    &mut helper_lane,
                    make_store_event(3, &[20, 21, 22, 23, 40]),
                );
            }
            true
        });
        assert!(split);
        index.slots.release(slots);
        wait_for_slot_release(&index, removed_slot);

        let mut fresh_lane = direct_lookup();
        apply_direct(
            &index,
            &mut fresh_lane,
            make_store_event(4, &[20, 21, 22, 23]),
        );
        assert_eq!(index.slot_for_test(fresh), Some(removed_slot));
        assert_direct_score(&index, &[20, 21, 22, 23, 24, 25], fresh, 4);
    }

    /// A removal whose ranks another lane already unmapped returns, and so is
    /// acknowledged, only after that lane's sweep released them. So does a `Cleared`.
    #[test]
    fn removal_that_loses_the_unmap_race_waits_for_the_winning_sweep() {
        let index = Arc::new(ConcurrentRadixTreeCompressed::new());
        let removed = worker(1);
        let mut lane = direct_lookup();
        apply_direct(&index, &mut lane, make_store_event(1, &[1, 2, 3]));
        let removed_slot = index.slot_for_test(removed).unwrap();

        // The winning lane has unmapped the rank and not yet swept it.
        let slots = index.slots.unmap(WorkerRemovalTarget::WorkerId(1));
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let losers: Vec<_> = (0..2)
            .map(|kind| {
                let index = index.clone();
                let done_tx = done_tx.clone();
                thread::spawn(move || {
                    let mut lookup = direct_lookup();
                    if kind == 0 {
                        index.remove_worker_coverage(
                            &mut lookup,
                            WorkerRemovalTarget::DpRank(removed),
                            true,
                        );
                    } else {
                        index.clear_worker_coverage(&mut lookup, removed);
                    }
                    let score = index
                        .find_matches_impl(&local_hashes(&[1, 2, 3]), false)
                        .scores
                        .get(&removed)
                        .copied();
                    done_tx.send(score).unwrap();
                })
            })
            .collect();
        thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            done_rx.try_recv().is_err(),
            "a loser returned before the sweep"
        );

        wait_for_pinned_threads();
        index.sweep_slots(&slots.iter().copied().collect(), &mut 0, |_, _| true);
        index.slots.release(slots);
        for loser in losers {
            loser.join().unwrap();
        }
        assert_eq!(done_rx.try_iter().collect::<Vec<_>>(), vec![None, None]);
        wait_for_slot_release(&index, removed_slot);
    }

    /// Events admitted after a whole-worker removal's lane barrier can give the rank a new
    /// slot while lane 0 still sweeps the old one. A `Cleared` then clears the new slot and
    /// still returns only after the old one is released.
    #[test]
    fn clear_after_a_new_slot_waits_for_the_old_slot_release() {
        let index = Arc::new(ConcurrentRadixTreeCompressed::new());
        let rank = worker(1);
        let mut lane = direct_lookup();
        apply_direct(&index, &mut lane, make_store_event(1, &[1, 2, 3]));
        let old_slot = index.slot_for_test(rank).unwrap();
        let slots = index.slots.unmap(WorkerRemovalTarget::WorkerId(1));
        apply_direct(&index, &mut lane, make_store_event(1, &[7, 8]));
        assert_ne!(index.slot_for_test(rank), Some(old_slot));

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let clearer_index = index.clone();
        let clearer = thread::spawn(move || {
            clearer_index.clear_worker_coverage(&mut lane, rank);
            let scores = [&[1, 2, 3][..], &[7, 8]].map(|hashes| {
                clearer_index
                    .find_matches_impl(&local_hashes(hashes), false)
                    .scores
                    .get(&rank)
                    .copied()
            });
            done_tx.send(scores).unwrap();
        });
        thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            done_rx.try_recv().is_err(),
            "the clear returned before the old slot was released"
        );

        wait_for_pinned_threads();
        index.sweep_slots(&slots.iter().copied().collect(), &mut 0, |_, _| true);
        index.slots.release(slots);
        clearer.join().unwrap();
        assert_eq!(done_rx.recv().unwrap(), [None, None]);
        wait_for_slot_release(&index, old_slot);
    }

    /// A `Cleared` that resolved its slot before the rank was removed must not clear the
    /// slot once it has been recycled to another rank.
    #[test]
    fn stale_clear_leaves_a_recycled_slot_alone() {
        let index = ConcurrentRadixTreeCompressed::new();
        let removed = worker(1);
        let fresh = worker(2);
        let mut lane = direct_lookup();
        apply_direct(&index, &mut lane, make_store_event(1, &[1, 2, 3]));
        let stale_slot = index.slot_for_test(removed).unwrap();

        index.remove_worker_coverage(&mut lane, WorkerRemovalTarget::DpRank(removed), true);
        wait_for_slot_release(&index, stale_slot);
        apply_direct(&index, &mut lane, make_store_event(2, &[1, 2, 3]));
        assert_eq!(index.slot_for_test(fresh), Some(stale_slot));

        let finished = index.clear_rank_slot(removed, stale_slot, &mut 0);
        assert_direct_score(&index, &[1, 2, 3], fresh, 3);
        assert!(!finished);
    }

    /// Like a reader, a dump credits a rank below a parent only if the rank covers the
    /// parent's whole edge.
    #[test]
    fn dump_skips_ranks_that_do_not_cover_the_parent() {
        let index = ConcurrentRadixTreeCompressed::new();
        let mut lane = direct_lookup();
        apply_direct(&index, &mut lane, make_store_event(1, &[1, 2, 3, 4]));
        apply_direct(&index, &mut lane, make_store_event(2, &[1, 2, 3, 4]));
        apply_direct(&index, &mut lane, make_store_event(3, &[1, 2, 9]));
        assert_edge_lengths(&index, &[1, 2, 2]);
        let parent = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        parent.set_coverage_for_test(&[slot(&index, worker(1)), slot(&index, worker(3))], &[]);

        let dumped = index.dump_tree_as_events();
        assert!(dumped.iter().all(|event| event.worker_id != 2));
        let workers: Vec<_> = snapshot_events(dumped)
            .iter()
            .map(|event| event.worker_id)
            .collect();
        assert_eq!(workers, vec![1, 1, 3, 3]);
    }
}

/// Volume-triggered stale-leaf reclamation (`reclaim.rs`) and the sweep's anchor seeds.
mod reclaim_tests {
    use super::*;
    use crate::test_utils::{make_remove_event, stored_blocks_with_sequence_hashes};

    /// Sweeps as soon as any block is dead, with no gap between sweeps.
    fn eager() -> ReclaimConfig {
        ReclaimConfig {
            volume_sweep: true,
            dead_floor: 1,
            min_gap: Duration::ZERO,
            ..ReclaimConfig::default()
        }
    }

    fn store_with_seq(
        worker_id: u64,
        parent: Option<u64>,
        locals: &[u64],
        seqs: &[u64],
    ) -> RouterEvent {
        crate::test_utils::router_event(
            worker_id,
            0,
            0,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(&local_hashes(locals), seqs),
            }),
        )
    }

    /// Worker 0 holds `[1, 2, 3]` with two children and worker 1 stores twelve leaves of
    /// four blocks under it, then evicts them all. With the volume trigger the next enqueue schedules a
    /// sweep and the tree returns to its baseline shape; without it, the dead leaves stay
    /// until the five-minute timer.
    async fn evict_leaves_and_count(config: ReclaimConfig) -> (CrtcShapeReport, u64) {
        let index = ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::with_reclaim_config(config),
            1,
            32,
        );
        // `[1, 2, 3]` is internal before the baseline, so the leaves below cannot split it.
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        for tail in [4, 5] {
            index
                .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[tail]))
                .await;
        }
        index.apply_event(make_store_event(1, &[1, 2, 3])).await;
        flush_and_settle(&index).await;
        let baseline = index.backend().probe_shape().nodes;

        for leaf in 0..12u64 {
            let tail: Vec<u64> = (0..4).map(|i| 100 + leaf * 10 + i).collect();
            index
                .apply_event(make_store_event_with_parent(1, &[1, 2, 3], &tail))
                .await;
        }
        flush_and_settle(&index).await;
        assert!(index.backend().probe_shape().nodes > baseline);
        for leaf in 0..12u64 {
            let tail: Vec<u64> = (0..4).map(|i| 100 + leaf * 10 + i).collect();
            index
                .apply_event(make_remove_event_with_parent(1, &[1, 2, 3], &tail))
                .await;
        }
        flush_and_settle(&index).await;
        // The lane flushed its tally when it went idle; the next enqueue sees the volume.
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        flush_and_settle(&index).await;

        assert_score(&index, &[1, 2, 3, 4], worker(0), 4).await;
        (index.backend().probe_shape(), baseline)
    }

    #[tokio::test]
    async fn volume_trigger_returns_node_count_to_baseline() {
        let (shape, baseline) = evict_leaves_and_count(eager()).await;
        assert_eq!(shape.nodes, baseline, "{shape:?}");
        assert_eq!(shape.dead_blocks, 0, "{shape:?}");
        assert!(shape.sweeps_volume >= 1, "{shape:?}");
        assert_eq!(shape.reclaimed_blocks, 48, "{shape:?}");

        // The control: the timer alone leaves the dead leaves linked.
        let (shape, baseline) = evict_leaves_and_count(ReclaimConfig::legacy()).await;
        assert!(shape.nodes > baseline, "{shape:?}");
        assert_eq!(shape.sweeps_volume + shape.sweeps_other, 0, "{shape:?}");
    }

    /// The default floor keeps small trees from sweeping on volume.
    #[tokio::test]
    async fn default_floor_does_not_sweep_small_trees() {
        let (shape, baseline) = evict_leaves_and_count(ReclaimConfig::default()).await;
        assert!(shape.nodes > baseline, "{shape:?}");
        assert_eq!(shape.sweeps_volume, 0, "{shape:?}");
    }

    /// Stores and evictions move the lane tally, and a sweep overwrites the shared
    /// estimates with its exact recount.
    #[test]
    fn estimates_follow_stores_evictions_and_sweeps() {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(eager());
        let mut lookup = direct_lookup();
        // A root child, a leaf extension, a split with a new tail, and a new child.
        apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2, 3]));
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]),
        );
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(0, &[1, 2, 3], &[6]),
        );
        apply_direct(&index, &mut lookup, make_store_event(1, &[9, 8]));
        index.reclaim.flush(&mut lookup.tally, true);
        assert_eq!(index.reclaim.estimates(), (0, 8));
        assert_eq!(index.probe_shape().linked_blocks, 8);

        // Worker 1 leaves `[9, 8]` holder-less; worker 0 leaves `[4, 5]` holder-less.
        apply_direct(&index, &mut lookup, make_remove_event(1, &[9]));
        apply_direct(
            &index,
            &mut lookup,
            make_remove_event_with_parent(0, &[1, 2, 3], &[4]),
        );
        index.reclaim.flush(&mut lookup.tally, true);
        assert_eq!(index.reclaim.estimates(), (4, 8));
        let shape = index.probe_shape();
        assert_eq!((shape.dead_blocks, shape.linked_blocks), (4, 8));

        let outcome = index.sweep_stale_children();
        assert_eq!(outcome.reclaimed_blocks, 4);
        assert_eq!(index.reclaim.estimates(), (0, 4));
        let shape = index.probe_shape();
        assert_eq!((shape.dead_blocks, shape.linked_blocks), (0, 4));
        assert_eq!(index.probe_check(&[&lookup]), Ok(0));
        assert_direct_score(&index, &[1, 2, 3, 6], worker(0), 4);
    }

    /// A dead child under a branch anchor is reclaimed: the sweep seeds from every anchor
    /// as well as the root, and never unlinks the anchor itself.
    #[test]
    fn sweep_reclaims_dead_child_under_anchor() {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(eager());
        let mut lookup = direct_lookup();
        let anchor_id = 0xA11C_0000;
        index
            .apply_anchor(
                worker(1),
                AnchorTask {
                    anchor_id: ExternalSequenceBlockHash(anchor_id),
                    anchor_local_hash: LocalBlockHash(2),
                    anchor_depth: 2,
                },
            )
            .unwrap();
        apply_direct(
            &index,
            &mut lookup,
            store_with_seq(1, Some(anchor_id), &[5, 6], &[0xA5, 0xA6]),
        );
        let anchor = index
            .anchor_nodes
            .get(&ExternalSequenceBlockHash(anchor_id))
            .unwrap()
            .clone();
        assert_eq!(anchor.children_snapshot().len(), 1);

        apply_direct(
            &index,
            &mut lookup,
            remove_event(
                1,
                1,
                0,
                vec![
                    ExternalSequenceBlockHash(0xA5),
                    ExternalSequenceBlockHash(0xA6),
                ],
            ),
        );
        assert_eq!(index.probe_shape().dead_leaves, 1);

        let outcome = index.sweep_stale_children();
        assert_eq!(outcome.reclaimed_nodes, 1);
        assert!(anchor.children_snapshot().is_empty());
        assert!(
            index
                .anchor_nodes
                .contains_key(&ExternalSequenceBlockHash(anchor_id))
        );
        assert_eq!(index.probe_check(&[&lookup]), Ok(0));

        // The anchor still takes stores.
        apply_direct(
            &index,
            &mut lookup,
            store_with_seq(1, Some(anchor_id), &[5], &[0xA5]),
        );
        assert_eq!(anchor.children_snapshot().len(), 1);
    }

    /// A holder-less internal node whose children all die is unlinked in the same pass,
    /// after them.
    #[test]
    fn sweep_cascades_through_dead_internal_nodes() {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(eager());
        let mut lookup = direct_lookup();
        apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2, 3, 4]));
        // Splits `[1, 2, 3, 4]` into `[1, 2] -> {[3, 4], [7]}`.
        apply_direct(
            &index,
            &mut lookup,
            make_store_event_with_parent(0, &[1, 2], &[7]),
        );
        let prefix = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        // Drop every holder from the whole chain without clearing children, as a racing
        // removal can leave it.
        let guard = crossbeam_epoch::pin();
        let mut nodes = vec![prefix.clone()];
        prefix.for_each_child(&guard, |_, child| nodes.push(child.clone()));
        drop(guard);
        for node in &nodes {
            node.set_coverage_for_test(&[], &[]);
        }
        drop(nodes);
        drop(prefix);
        drop(lookup);

        let outcome = index.sweep_stale_children();
        assert_eq!(outcome.reclaimed_nodes, 3, "{outcome:?}");
        assert_eq!(index.raw_child_edge_count(), 0);
    }

    /// A dead leaf a lane still names stays linked, and the probe says why.
    #[test]
    fn held_dead_leaf_is_skipped_and_reported() {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(eager());
        let mut lookup = direct_lookup();
        apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2]));
        let leaf = index.root.child_snapshot(LocalBlockHash(1)).unwrap();
        leaf.set_coverage_for_test(&[], &[]);

        let outcome = index.sweep_stale_children();
        assert_eq!((outcome.reclaimed_nodes, outcome.skipped_held), (0, 1));
        drop(leaf);
        assert_eq!(index.probe_check(&[&lookup]), Ok(1));
        assert_eq!(index.probe_shape().dead_leaves_held, 1);
    }

    /// Two scheduling paths share one in-flight flag: while a volume sweep is scheduled,
    /// neither the volume check nor the timer schedules another.
    #[test]
    fn volume_and_timer_share_one_in_flight_sweep() {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(eager());
        let mut lookup = direct_lookup();
        apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2]));
        apply_direct(&index, &mut lookup, make_remove_event(0, &[1]));
        index.reclaim.flush(&mut lookup.tally, true);

        assert!(index.try_schedule_cleanup());
        assert!(!index.try_schedule_cleanup());
        index.run_cleanup_task();
        // The sweep reclaimed everything, so nothing is due any more.
        assert!(!index.try_schedule_cleanup());
        assert_eq!(index.probe_shape().sweeps_volume, 1);
    }

    /// The gap rule: after a sweep, a volume sweep waits `min_gap`.
    #[test]
    fn min_gap_delays_the_next_volume_sweep() {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(ReclaimConfig {
            min_gap: Duration::from_secs(3600),
            ..eager()
        });
        let mut lookup = direct_lookup();
        index.sweep_stale_children();
        apply_direct(&index, &mut lookup, make_store_event(0, &[1, 2]));
        apply_direct(&index, &mut lookup, make_remove_event(0, &[1]));
        index.reclaim.flush(&mut lookup.tally, true);
        assert!(!index.try_schedule_cleanup());

        index.probe_set_reclaim(eager());
        // The gap is still ten times the last sweep's duration.
        std::thread::sleep(Duration::from_millis(20));
        assert!(index.try_schedule_cleanup());
        index.cancel_scheduled_cleanup();
    }

    /// The two race tests, with sweeps running back to back on two threads throughout.
    #[test]
    fn races_hold_under_back_to_back_sweeps() {
        for round in 0..50 {
            let index = Arc::new(ConcurrentRadixTreeCompressed::with_reclaim_config(eager()));
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let sweepers: Vec<_> = (0..2)
                .map(|_| {
                    let index = index.clone();
                    let stop = stop.clone();
                    thread::spawn(move || {
                        while !stop.load(Ordering::Relaxed) {
                            index.run_cleanup_task();
                        }
                    })
                })
                .collect();

            // race_remove_keeps_children_needed_by_another_full_worker
            let (worker1, worker2, worker3) = (worker(1), worker(2), worker(3));
            let (mut lookup1, mut lookup2, mut lookup3) =
                (direct_lookup(), direct_lookup(), direct_lookup());
            apply_direct(&index, &mut lookup1, make_store_event(1, &[1, 2, 3, 4]));
            apply_direct(&index, &mut lookup2, make_store_event(2, &[1, 2, 3, 4]));
            apply_direct(
                &index,
                &mut lookup1,
                make_store_event_with_parent(1, &[1, 2, 3, 4], &[5, 6]),
            );
            apply_direct(&index, &mut lookup3, make_store_event(3, &[1, 2, 3, 4]));
            apply_direct(
                &index,
                &mut lookup3,
                make_store_event_with_parent(3, &[1, 2, 3, 4], &[7, 8]),
            );
            let reader_index = index.clone();
            let reader = thread::spawn(move || {
                for _ in 0..256 {
                    assert_direct_score(&reader_index, &[1, 2, 3, 4, 5, 6], worker1, 6);
                }
            });
            apply_direct(
                &index,
                &mut lookup2,
                make_remove_event_with_parent(2, &[1], &[2]),
            );
            reader.join().unwrap();
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            assert_direct_score(&index, &[1, 2, 3, 4], worker2, 1);
            assert_direct_score(&index, &[1, 2, 3, 4, 7, 8], worker3, 6);

            // race_cleanup_with_dead_child_reuse_keeps_restored_child, on worker 4
            let worker4 = worker(4);
            let mut lookup4 = direct_lookup();
            apply_direct(&index, &mut lookup4, make_store_event(4, &[11, 12, 13]));
            for tail in [[14, 15], [16, 17]] {
                apply_direct(
                    &index,
                    &mut lookup4,
                    make_store_event_with_parent(4, &[11, 12, 13], &tail),
                );
            }
            for tail in [[14, 15], [16, 17]] {
                apply_direct(
                    &index,
                    &mut lookup4,
                    make_remove_event_with_parent(4, &[11, 12, 13], &tail),
                );
            }
            if round % 2 == 0 {
                thread::yield_now();
            }
            apply_direct(
                &index,
                &mut lookup4,
                make_store_event_with_parent(4, &[11, 12, 13], &[14, 15]),
            );
            assert_direct_score(&index, &[11, 12, 13, 14, 15], worker4, 5);

            stop.store(true, Ordering::Relaxed);
            for sweeper in sweepers {
                sweeper.join().unwrap();
            }
            index.probe_quiesce();
            assert_direct_score(&index, &[11, 12, 13, 14, 15], worker4, 5);
            assert_direct_score(&index, &[1, 2, 3, 4, 5, 6], worker1, 6);
            let held = index
                .probe_check(&[&lookup1, &lookup2, &lookup3, &lookup4])
                .unwrap();
            assert_eq!(held, 0, "round {round}");
        }
    }
}

/// Exact-size split prefixes and capped leaf slack (`reclaim.rs`, `EdgeCapacity`).
mod edge_capacity_tests {
    use super::*;

    /// Sixty-four workers share eight 32-block prompts, then decode 256 blocks each, one
    /// to four at a time, while siblings split each other's prompts. Returns edge capacity
    /// over edge length.
    fn decode_workload_slack(config: ReclaimConfig) -> f64 {
        let index = ConcurrentRadixTreeCompressed::with_reclaim_config(config);
        let mut lookup = direct_lookup();
        let mut rng = fastrand::Rng::with_seed(7);
        for worker_id in 0..64u64 {
            let prompt: Vec<u64> = (0..32).map(|i| 1_000 * (worker_id % 8) + i).collect();
            // Workers diverge from their prompt group at a random point.
            let shared = rng.usize(8..=32);
            let mut seq: Vec<u64> = prompt[..shared].to_vec();
            seq.extend((shared..32).map(|i| 1_000_000 * (worker_id + 1) + i as u64));
            apply_direct(&index, &mut lookup, make_store_event(worker_id, &seq));
            let mut next = 0;
            while next < 256 {
                let take = rng.usize(1..=4).min(256 - next);
                let tail: Vec<u64> = (next..next + take)
                    .map(|i| 2_000_000 * (worker_id + 1) + i as u64)
                    .collect();
                apply_direct(
                    &index,
                    &mut lookup,
                    make_store_event_with_parent(worker_id, &seq, &tail),
                );
                seq.extend(tail);
                next += take;
            }
            assert_direct_score(&index, &seq, worker(worker_id), seq.len() as u32);
        }
        let memory = index.probe_memory();
        assert_eq!(
            memory.edge_len_bytes,
            index.probe_shape().linked_blocks * 16
        );
        memory.edge_slack()
    }

    #[test]
    fn decode_edges_stay_within_slack() {
        let slack = decode_workload_slack(ReclaimConfig::default());
        assert!(slack <= 1.15, "edge slack {slack:.3}");
        let slack_quarter = decode_workload_slack(ReclaimConfig {
            leaf_slack_divisor: 4,
            ..ReclaimConfig::default()
        });
        assert!(
            slack_quarter <= 1.3,
            "edge slack {slack_quarter:.3} with divisor 4"
        );
        let legacy = decode_workload_slack(ReclaimConfig::legacy());
        assert!(legacy > slack + 0.2, "legacy {legacy:.3} vs {slack:.3}");
        eprintln!(
            "edge slack: default {slack:.3}, divisor 4 {slack_quarter:.3}, legacy {legacy:.3}"
        );
    }
}
