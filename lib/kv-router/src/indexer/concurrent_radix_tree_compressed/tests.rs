// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
use crate::test_utils::{
    assert_score, flush_and_settle, make_remove_event_with_parent, make_store_event,
    make_store_event_with_parent, snapshot_events, snapshot_tree,
};
use std::sync::{Arc, Barrier};
use std::thread;

type DirectLookup = FxHashMap<WorkerWithDpRank, WorkerLookup>;

fn worker(worker_id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(worker_id, 0)
}

fn direct_lookup() -> DirectLookup {
    FxHashMap::default()
}

fn worker_lookup_len(lookup: &DirectLookup, worker: WorkerWithDpRank) -> Option<usize> {
    lookup.get(&worker).map(|worker_lookup| worker_lookup.len())
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

fn apply_direct(
    index: &ConcurrentRadixTreeCompressed,
    lookup: &mut DirectLookup,
    event: RouterEvent,
) {
    index.apply_event(lookup, event, None).unwrap();
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

mod structural_tests {
    use super::*;

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
