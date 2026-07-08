// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicBool, Ordering},
};

use crate::indexer::pruning::PruneConfig;
use crate::indexer::{KvIndexerInterface, SyncIndexer, ThreadPoolIndexer, WorkerTask};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier, WorkerWithDpRank,
    compute_seq_hash_for_block,
};

use super::addressing::{CkfAddressing, CkfProbe};
use super::bucket::{CuckooBucketStore, PackedBucket, TransposedCkfTable};
use super::event_indexer::bucket_count;
use super::mutator::{CuckooMutator, DcWriterState};
use super::search::{
    find_prefix_depths, find_prefix_depths_with_test_stats, fixed_window_prefix_depths,
    linear_prefix_depths,
};
use super::{CkfBuildError, CkfConfig, DC_COUNT, EventTransposedCkfIndexer};

const TEST_SEED: u64 = 0x1234_5678_9ABC_DEF0;

#[test]
fn validates_config_and_worker_mapping() {
    let workers = workers();
    assert_eq!(bucket_count(1).unwrap(), 2);
    assert_eq!(bucket_count(16).unwrap(), 8);

    let zero_capacity = CkfConfig {
        expected_blocks_per_dc: 0,
        ..CkfConfig::default()
    };
    assert_eq!(
        EventTransposedCkfIndexer::new(workers, zero_capacity).unwrap_err(),
        CkfBuildError::ExpectedCapacityZero
    );

    let mut duplicate_workers = workers;
    duplicate_workers[1] = duplicate_workers[0];
    assert_eq!(
        EventTransposedCkfIndexer::new(duplicate_workers, CkfConfig::default()).unwrap_err(),
        CkfBuildError::DuplicateWorker {
            worker: duplicate_workers[0]
        }
    );
}

#[test]
fn packed_bucket_swar_matches_scalar_membership() {
    let fingerprints = [1, 0x1234, 0x8000, u16::MAX];
    let mut bucket = PackedBucket::default();
    for (slot, fingerprint) in fingerprints.into_iter().enumerate() {
        bucket = bucket.with_slot(slot, fingerprint);
    }

    for fingerprint in [1, 2, 0x1234, 0x1235, 0x8000, 0xFFFE, u16::MAX] {
        assert_eq!(
            bucket.contains(fingerprint),
            bucket.scalar_contains(fingerprint),
            "fingerprint={fingerprint:#06x}"
        );
    }

    assert_eq!(bucket.first_empty(), None);
    assert_eq!(bucket.first(1), Some(0));
    assert_eq!(bucket.first(0x8000), Some(2));
    assert_eq!(bucket.with_slot(1, 0).first_empty(), Some(1));
}

#[test]
fn addressing_is_nonzero_and_alternate_bucket_is_an_involution() {
    let addressing = CkfAddressing::new(1024, TEST_SEED);
    assert_eq!(
        compute_seq_hash_for_block(&[
            LocalBlockHash(0x0123_4567_89AB_CDEF),
            LocalBlockHash(0x1111_2222_3333_4444),
            LocalBlockHash(0xDEAD_BEEF_CAFE_BABE),
        ]),
        [
            0x0123_4567_89AB_CDEF,
            0x98E7_E55F_299C_F592,
            0x65D5_3B99_C525_2D19,
        ]
    );
    assert_eq!(
        [0u64, 1, 0x0123_4567_89AB_CDEF, u64::MAX].map(|hash| addressing.prepare(hash)),
        [
            CkfProbe {
                fingerprint: 7_750,
                bucket_a: 587,
                bucket_b: 372,
            },
            CkfProbe {
                fingerprint: 64_153,
                bucket_a: 348,
                bucket_b: 36,
            },
            CkfProbe {
                fingerprint: 45_802,
                bucket_a: 865,
                bucket_b: 909,
            },
            CkfProbe {
                fingerprint: 31_767,
                bucket_a: 112,
                bucket_b: 217,
            },
        ]
    );
    for hash in 0..10_000u64 {
        let probe = addressing.prepare(hash);
        assert_ne!(probe.fingerprint, 0);
        assert_ne!(probe.bucket_a, probe.bucket_b);
        assert_eq!(
            addressing.alternate_bucket(probe.bucket_b, probe.fingerprint),
            probe.bucket_a
        );
    }
}

#[test]
fn bounded_relocation_rolls_back_every_write_and_emits_no_dirty_bucket() {
    let table = TransposedCkfTable::<1>::new(2).unwrap();
    let view = table.lane(0);
    let full_a = PackedBucket::default()
        .with_slot(0, 1)
        .with_slot(1, 2)
        .with_slot(2, 3)
        .with_slot(3, 4);
    let full_b = PackedBucket::default()
        .with_slot(0, 5)
        .with_slot(1, 6)
        .with_slot(2, 7)
        .with_slot(3, 8);
    view.store_bucket(0, full_a);
    view.store_bucket(1, full_b);

    let addressing = CkfAddressing::new(2, TEST_SEED);
    let mut state = DcWriterState::new(8, 1, TEST_SEED).unwrap();
    let mut dirty = Vec::new();
    let result = CuckooMutator::new(&view, &addressing, 1).insert(
        ExternalSequenceBlockHash(999),
        &mut state.rng,
        &mut state.scratch,
        |bucket| dirty.push(bucket),
    );

    assert!(matches!(
        result,
        Err(crate::protocols::KvCacheEventError::CapacityExhausted)
    ));
    assert_eq!(view.load_bucket(0), full_a);
    assert_eq!(view.load_bucket(1), full_b);
    assert!(dirty.is_empty());
}

#[test]
fn distinct_hashes_with_one_ckf_representation_keep_separate_copies() {
    let addressing = CkfAddressing::new(2, TEST_SEED);
    let (first, second, representation) = colliding_hashes(addressing);
    let table = TransposedCkfTable::<1>::new(2).unwrap();
    let view = table.lane(0);
    let mut state = DcWriterState::new(8, 20, TEST_SEED).unwrap();
    let mutator = CuckooMutator::new(&view, &addressing, 20);

    mutator
        .insert(first, &mut state.rng, &mut state.scratch, |_| {})
        .unwrap();
    mutator
        .insert(second, &mut state.rng, &mut state.scratch, |_| {})
        .unwrap();
    assert_eq!(fingerprint_copies(&view, representation), 2);

    mutator.remove(first, |_| {}).unwrap();
    assert_eq!(fingerprint_copies(&view, representation), 1);
    assert_eq!(table.probe(representation), 1);
}

#[test]
fn full_binary_search_and_terminal_window_follow_the_defined_contract() {
    let masks = [1u16, 1, 1, 0, 1, 1, 1, 1];
    let window_one = find_prefix_depths::<1>(masks.len(), 1, 1, |_| {}, |position| masks[position]);
    let window_eight =
        find_prefix_depths::<1>(masks.len(), 1, 8, |_| {}, |position| masks[position]);
    let linear = linear_prefix_depths::<1>(masks.len(), 1, |position| masks[position]);

    assert_eq!(window_one, [8]);
    assert_eq!(window_eight, [3]);
    assert_eq!(linear, [3]);
}

#[test]
fn grouped_search_probes_each_binary_midpoint_once() {
    let masks = [u16::MAX, u16::MAX, u16::MAX, 0, 0, 0, 0, 0];
    let mut counts = [0usize; 8];
    let depths = find_prefix_depths::<16>(
        masks.len(),
        u16::MAX,
        1,
        |_| {},
        |position| {
            counts[position] += 1;
            masks[position]
        },
    );

    assert!(depths.into_iter().all(|depth| depth == 3));
    assert!(counts.into_iter().all(|count| count <= 2));
}

#[tokio::test]
async fn thread_pool_uses_external_hashes_returns_full_map_and_clears_all_ranks() {
    let workers = workers_with_shared_worker();
    let index = ThreadPoolIndexer::new(
        EventTransposedCkfIndexer::new(workers, CkfConfig::new(32)).unwrap(),
        2,
        32,
    );
    let local_hashes = vec![LocalBlockHash(11), LocalBlockHash(22)];
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
    assert!(index.backend().find_matches(&[], false).scores.is_empty());

    KvIndexerInterface::apply_event(&index, store_event(workers[0], &sequence_hashes, 10_000))
        .await;
    KvIndexerInterface::apply_event(&index, store_event(workers[1], &sequence_hashes, 20_000))
        .await;
    KvIndexerInterface::flush(&index).await;

    let scores = KvIndexerInterface::find_matches(&index, local_hashes.clone())
        .await
        .unwrap();
    assert_eq!(scores.scores.get(&workers[0]), Some(&2));
    assert_eq!(scores.scores.get(&workers[1]), Some(&2));
    assert!(scores.frequencies.is_empty());

    KvIndexerInterface::remove_worker_dp_rank(&index, workers[0].worker_id, workers[0].dp_rank)
        .await;
    KvIndexerInterface::flush(&index).await;
    let scores = KvIndexerInterface::find_matches(&index, local_hashes.clone())
        .await
        .unwrap();
    assert!(!scores.scores.contains_key(&workers[0]));
    assert_eq!(scores.scores.get(&workers[1]), Some(&2));

    KvIndexerInterface::apply_event(&index, store_event(workers[0], &sequence_hashes, 30_000))
        .await;
    KvIndexerInterface::flush(&index).await;

    KvIndexerInterface::apply_event(&index, clear_event(workers[0].worker_id)).await;
    KvIndexerInterface::flush(&index).await;
    let scores = KvIndexerInterface::find_matches(&index, local_hashes)
        .await
        .unwrap();
    assert!(!scores.scores.contains_key(&workers[0]));
    assert!(!scores.scores.contains_key(&workers[1]));
    let stats = index.worker_lookup_stats().await;
    assert_eq!(stats.block_count_for_worker(workers[0]), None);
    assert_eq!(stats.block_count_for_worker(workers[1]), None);
    index.shutdown();
}

#[tokio::test]
async fn thread_pool_reports_unsupported_dump_and_exact_stats() {
    let workers = workers();
    let index = ThreadPoolIndexer::new(
        EventTransposedCkfIndexer::new(workers, CkfConfig::new(32)).unwrap(),
        2,
        32,
    );
    let hashes = [91u64, 92, 93];
    KvIndexerInterface::apply_event(&index, store_event(workers[0], &hashes, 40_000)).await;
    KvIndexerInterface::flush(&index).await;

    let stats = index.worker_lookup_stats().await;
    assert_eq!(stats.block_count_for_worker(workers[0]), Some(3));
    assert!(matches!(
        KvIndexerInterface::dump_events(&index).await,
        Err(crate::indexer::KvRouterError::Unsupported(_))
    ));
    index.shutdown();
}

#[test]
fn worker_acknowledges_partial_event_failure_and_continues_serving_tasks() {
    let workers = workers();
    let index = EventTransposedCkfIndexer::new(workers, CkfConfig::new(32)).unwrap();
    let (sender, receiver) = flume::unbounded();

    std::thread::scope(|scope| {
        let worker = scope.spawn(|| index.worker(receiver, None).unwrap());
        sender
            .send(WorkerTask::Event(store_event(workers[0], &[10, 20], 1000)))
            .unwrap();

        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        sender
            .send(WorkerTask::EventWithAck {
                event: remove_event(workers[0], &[999, 20]),
                resp: ack_tx,
            })
            .unwrap();
        assert!(!ack_rx.blocking_recv().unwrap());

        let (stats_tx, stats_rx) = tokio::sync::oneshot::channel();
        sender.send(WorkerTask::Stats(stats_tx)).unwrap();
        let stats = stats_rx.blocking_recv().unwrap();
        assert_eq!(stats.block_count_for_worker(workers[0]), Some(1));

        let (flush_tx, flush_rx) = tokio::sync::oneshot::channel();
        sender.send(WorkerTask::Flush(flush_tx)).unwrap();
        flush_rx.blocking_recv().unwrap();
        sender.send(WorkerTask::Terminate).unwrap();
        worker.join().unwrap();
    });
}

#[test]
fn pruning_construction_is_rejected_before_threads_start() {
    let backend = EventTransposedCkfIndexer::new(workers(), CkfConfig::new(32)).unwrap();
    let result = std::panic::catch_unwind(|| {
        ThreadPoolIndexer::new_with_pruning(backend, 1, 32, PruneConfig::default())
    });
    assert!(result.is_err());
}

#[test]
fn removal_continues_after_a_missing_hash_and_retains_first_error() {
    let workers = workers();
    let index = EventTransposedCkfIndexer::new(workers, CkfConfig::new(32)).unwrap();
    let mut states = std::array::from_fn(|_| None);
    index
        .apply_event(&mut states, store_event(workers[0], &[10, 20], 1000), None)
        .unwrap();

    let result = index.apply_event(&mut states, remove_event(workers[0], &[999, 20]), None);
    assert!(matches!(
        result,
        Err(crate::protocols::KvCacheEventError::BlockNotFound)
    ));
    assert!(
        states[0]
            .as_ref()
            .expect("writer state")
            .resident
            .contains(&ExternalSequenceBlockHash(10))
    );
    assert!(
        !states[0]
            .as_ref()
            .expect("writer state")
            .resident
            .contains(&ExternalSequenceBlockHash(20))
    );
}

#[test]
fn digest_updates_roll_back_when_ckf_mutation_fails() {
    let workers = workers();
    let mut config = CkfConfig::new(1);
    config.max_kicks = 1;
    let index = EventTransposedCkfIndexer::new(workers, config).unwrap();
    let mut states = std::array::from_fn(|_| None);
    index
        .apply_event(
            &mut states,
            store_event(workers[0], &(0..8).collect::<Vec<_>>(), 1000),
            None,
        )
        .unwrap();

    assert!(matches!(
        index.apply_event(&mut states, store_event(workers[0], &[8], 2000), None),
        Err(crate::protocols::KvCacheEventError::CapacityExhausted)
    ));
    let state = states[0].as_ref().expect("writer state");
    assert_eq!(state.resident.len(), 8);
    assert!(!state.resident.contains(&ExternalSequenceBlockHash(8)));

    let resident = ExternalSequenceBlockHash(0);
    let probe = index.addressing.prepare(resident.0);
    let view = index.table.lane(0);
    view.store_bucket(probe.bucket_a, PackedBucket::default());
    view.store_bucket(probe.bucket_b, PackedBucket::default());
    assert!(matches!(
        index.apply_event(&mut states, remove_event(workers[0], &[resident.0]), None),
        Err(crate::protocols::KvCacheEventError::IndexerInvariantViolation)
    ));
    assert!(
        states[0]
            .as_ref()
            .expect("writer state")
            .resident
            .contains(&resident)
    );
}

#[test]
fn invalid_representation_collision_removal_does_not_delete_the_resident_hash() {
    let workers = workers();
    let index = EventTransposedCkfIndexer::new(workers, CkfConfig::new(1)).unwrap();
    let (resident, absent, probe) = colliding_hashes(index.addressing);
    let mut states = std::array::from_fn(|_| None);
    index
        .apply_event(
            &mut states,
            store_event(workers[0], &[resident.0], 5000),
            None,
        )
        .unwrap();

    let result = index.apply_event(&mut states, remove_event(workers[0], &[absent.0]), None);
    assert!(matches!(
        result,
        Err(crate::protocols::KvCacheEventError::BlockNotFound)
    ));
    assert_eq!(index.table.probe(probe) & 1, 1);
    assert!(
        states[0]
            .as_ref()
            .expect("writer state")
            .resident
            .contains(&resident)
    );
}

#[test]
fn successful_mutation_reports_each_final_dirty_image_once() {
    let workers = workers();
    let index = EventTransposedCkfIndexer::new(workers, CkfConfig::new(32)).unwrap();
    let mut states = std::array::from_fn(|_| None);
    let mut dirty = Vec::new();
    index
        .apply_event_with_dirty(
            &mut states,
            store_event(workers[0], &[10], 6000),
            None,
            |lane, bucket| dirty.push((lane, bucket)),
        )
        .unwrap();

    assert!(!dirty.is_empty());
    let mut unique = HashSet::new();
    for (lane, bucket) in dirty {
        assert!(unique.insert((lane, bucket.bucket)));
        assert_eq!(
            index.table.lane(lane).load_bucket(bucket.bucket),
            bucket.value
        );
    }
}

#[test]
fn window_eight_repairs_holes_at_offsets_two_through_eight() {
    const QUERY_LEN: usize = 33;

    for offset in 2..=8 {
        let hole = QUERY_LEN - offset;
        let mut masks = [1u16; QUERY_LEN];
        masks[hole] = 0;
        let window_one =
            fixed_window_prefix_depths::<1>(QUERY_LEN, 1, 1, |_| {}, |position| masks[position]);
        let window_eight =
            fixed_window_prefix_depths::<1>(QUERY_LEN, 1, 8, |_| {}, |position| masks[position]);
        let linear = linear_prefix_depths::<1>(QUERY_LEN, 1, |position| masks[position]);

        assert_eq!(window_one, [QUERY_LEN as u32], "offset={offset}");
        assert_eq!(window_eight, [hole as u32], "offset={offset}");
        assert_eq!(linear, [hole as u32], "offset={offset}");
    }
}

#[test]
fn holes_beyond_the_terminal_window_remain_advisory() {
    const QUERY_LEN: usize = 33;
    const HOLE: usize = QUERY_LEN - 9;
    let mut masks = [1u16; QUERY_LEN];
    masks[HOLE] = 0;
    let window_eight =
        fixed_window_prefix_depths::<1>(QUERY_LEN, 1, 8, |_| {}, |position| masks[position]);
    let linear = linear_prefix_depths::<1>(QUERY_LEN, 1, |position| masks[position]);

    assert_eq!(window_eight, [QUERY_LEN as u32]);
    assert_eq!(linear, [HOLE as u32]);
}

#[test]
fn terminal_branch_fallback_groups_lanes_and_repairs_a_clipped_false_pivot() {
    const QUERY_LEN: usize = 65;
    const LINEAR_MISS: usize = 40;
    const FALSE_PIVOT: usize = 48;

    let membership = |position: usize| {
        if position < LINEAR_MISS || position == FALSE_PIVOT {
            0b11
        } else {
            0
        }
    };
    let mut probed = Vec::new();
    let result = find_prefix_depths_with_test_stats::<2>(
        QUERY_LEN,
        0b11,
        2,
        |_| {},
        |position| {
            probed.push(position);
            membership(position)
        },
    );
    let linear = linear_prefix_depths::<2>(QUERY_LEN, 0b11, membership);

    assert_eq!(result.depths, linear);
    assert_eq!(result.depths, [LINEAR_MISS as u32; 2]);
    assert!(!probed.contains(&17), "fallback must begin after B=32");
    for fallback_position in 33..=LINEAR_MISS {
        assert_eq!(
            probed
                .iter()
                .filter(|&&position| position == fallback_position)
                .count(),
            1,
            "fallback_position={fallback_position}"
        );
    }
    assert_eq!(result.fallback.left_edge_lanes, 2);
    assert_eq!(result.fallback.activated_lanes, 2);
    assert_eq!(result.fallback.probe_calls, 8);
    assert_eq!(result.fallback.lane_probes, 16);
    assert_eq!(result.fallback.provenance_skips, 0);
}

#[test]
fn terminal_branch_fallback_skips_second_order_provenance_without_an_inverted_range() {
    const QUERY_LEN: usize = 65;
    const LINEAR_MISS: usize = 41;

    let membership =
        |position: usize| u16::from(position < LINEAR_MISS || (48..=51).contains(&position));
    let result = find_prefix_depths_with_test_stats::<1>(QUERY_LEN, 1, 8, |_| {}, membership);

    assert_eq!(linear_prefix_depths::<1>(QUERY_LEN, 1, membership), [41]);
    assert_eq!(result.depths, [44]);
    assert_eq!(result.fallback.left_edge_lanes, 1);
    assert_eq!(result.fallback.activated_lanes, 0);
    assert_eq!(result.fallback.probe_calls, 0);
    assert_eq!(result.fallback.provenance_skips, 1);
}

#[test]
fn terminal_branch_fallback_documents_a_false_predecessor_residual() {
    const QUERY_LEN: usize = 65;
    const LINEAR_MISS: usize = 20;

    let membership =
        |position: usize| u16::from(position < LINEAR_MISS || position == 32 || position == 48);
    let result = find_prefix_depths_with_test_stats::<1>(QUERY_LEN, 1, 8, |_| {}, membership);

    assert_eq!(linear_prefix_depths::<1>(QUERY_LEN, 1, membership), [20]);
    assert_eq!(result.depths, [33]);
    assert_eq!(result.fallback.left_edge_lanes, 1);
    assert_eq!(result.fallback.activated_lanes, 1);
    assert_eq!(result.fallback.probe_calls, 1);
    assert_eq!(result.fallback.provenance_skips, 0);
}

#[cfg(feature = "metrics")]
#[test]
fn provenance_fallback_metrics_record_prebound_lane_and_probe_counts() {
    use crate::indexer::{
        KvIndexerMetrics, METRIC_CKF_FALLBACK_ACTIVATED_LANES, METRIC_CKF_FALLBACK_LANE_PROBES,
        METRIC_CKF_FALLBACK_LEFT_EDGE_LANES, METRIC_CKF_FALLBACK_PROBE_CALLS,
        METRIC_CKF_FALLBACK_PROVENANCE_SKIPS,
    };

    let membership = |position: usize| u16::from(position < 40 || position == 48);
    let result = find_prefix_depths_with_test_stats::<1>(65, 1, 2, |_| {}, membership);
    let metrics = KvIndexerMetrics::new_unregistered();
    let counters = metrics.prebind_ckf_search();
    counters.record(
        result.fallback.left_edge_lanes,
        result.fallback.activated_lanes,
        result.fallback.probe_calls,
        result.fallback.lane_probes,
        result.fallback.provenance_skips,
    );

    let value = |kind| metrics.ckf_search_fallback.with_label_values(&[kind]).get();
    assert_eq!(value(METRIC_CKF_FALLBACK_LEFT_EDGE_LANES), 1);
    assert_eq!(value(METRIC_CKF_FALLBACK_ACTIVATED_LANES), 1);
    assert_eq!(value(METRIC_CKF_FALLBACK_PROBE_CALLS), 8);
    assert_eq!(value(METRIC_CKF_FALLBACK_LANE_PROBES), 8);
    assert_eq!(value(METRIC_CKF_FALLBACK_PROVENANCE_SKIPS), 0);
}

#[test]
fn store_remove_churn_drains_every_physical_slot() {
    let table = TransposedCkfTable::<1>::new(64).unwrap();
    let view = table.lane(0);
    let addressing = CkfAddressing::new(64, TEST_SEED);
    let mut state = DcWriterState::new(128, 500, TEST_SEED).unwrap();
    let hashes: Vec<_> = (0..128)
        .map(|hash| ExternalSequenceBlockHash(10_000 + hash))
        .collect();
    let mutator = CuckooMutator::new(&view, &addressing, 500);
    for &hash in &hashes {
        mutator
            .insert(hash, &mut state.rng, &mut state.scratch, |_| {})
            .unwrap();
        state.resident.insert(hash);
    }
    for &hash in hashes.iter().rev() {
        mutator.remove(hash, |_| {}).unwrap();
        state.resident.remove(&hash);
    }

    assert!(state.resident.is_empty());
    assert!(
        (0..view.bucket_count())
            .all(|bucket| { view.load_bucket(bucket) == PackedBucket::default() })
    );
}

#[tokio::test]
async fn concurrent_queries_and_mutations_stay_bounded_and_drain_consistently() {
    let workers = workers();
    let index = Arc::new(ThreadPoolIndexer::new(
        EventTransposedCkfIndexer::new(workers, CkfConfig::new(128)).unwrap(),
        2,
        32,
    ));
    let query: Vec<_> = (1..=64).map(LocalBlockHash).collect();
    let sequence_hashes = compute_seq_hash_for_block(&query);
    KvIndexerInterface::apply_event(
        index.as_ref(),
        store_event(workers[0], &sequence_hashes, 10_000),
    )
    .await;
    KvIndexerInterface::flush(index.as_ref()).await;

    let barrier = Arc::new(Barrier::new(2));
    let stop = Arc::new(AtomicBool::new(false));
    let query_index = Arc::clone(&index);
    let query_barrier = Arc::clone(&barrier);
    let query_stop = Arc::clone(&stop);
    let query_copy = query.clone();
    let reader = std::thread::spawn(move || {
        query_barrier.wait();
        while !query_stop.load(Ordering::Acquire) {
            let scores = query_index.backend().find_matches(&query_copy, false);
            assert!(scores.scores.keys().all(|worker| workers.contains(worker)));
            assert!(
                scores
                    .scores
                    .values()
                    .all(|&depth| depth <= query_copy.len() as u32)
            );
        }
    });

    barrier.wait();
    KvIndexerInterface::apply_event(
        index.as_ref(),
        remove_event(workers[0], &sequence_hashes[..16]),
    )
    .await;
    KvIndexerInterface::apply_event(index.as_ref(), clear_event(workers[0].worker_id)).await;
    KvIndexerInterface::apply_event(
        index.as_ref(),
        store_event(workers[0], &sequence_hashes, 20_000),
    )
    .await;
    KvIndexerInterface::flush(index.as_ref()).await;
    stop.store(true, Ordering::Release);
    reader.join().unwrap();

    let scores = index.backend().find_matches(&query, false);
    assert_eq!(scores.scores.get(&workers[0]), Some(&(query.len() as u32)));
    let probes = index.backend().prepared_probes(&query);
    let linear = index.backend().linear_depths(&probes);
    assert_eq!(linear[0], query.len() as u32);
    let stats = index.worker_lookup_stats().await;
    assert_eq!(stats.block_count_for_worker(workers[0]), Some(query.len()));
    index.shutdown();
}

#[cfg(feature = "metrics")]
#[test]
fn new_event_errors_have_finite_metric_labels() {
    use crate::indexer::{
        KvIndexerMetrics, METRIC_EVENT_STORED, METRIC_STATUS_CAPACITY_EXHAUSTED,
        METRIC_STATUS_INDEXER_INVARIANT_VIOLATION,
    };

    let metrics = KvIndexerMetrics::new_unregistered();
    metrics.increment_event_applied(
        METRIC_EVENT_STORED,
        Err(crate::protocols::KvCacheEventError::CapacityExhausted),
    );
    metrics.increment_event_applied(
        METRIC_EVENT_STORED,
        Err(crate::protocols::KvCacheEventError::IndexerInvariantViolation),
    );
    assert_eq!(
        metrics
            .kv_cache_events_applied
            .with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_CAPACITY_EXHAUSTED])
            .get(),
        1
    );
    assert_eq!(
        metrics
            .kv_cache_events_applied
            .with_label_values(&[
                METRIC_EVENT_STORED,
                METRIC_STATUS_INDEXER_INVARIANT_VIOLATION,
            ])
            .get(),
        1
    );
}

#[cfg(feature = "metrics")]
#[tokio::test]
async fn duplicate_warning_and_partial_error_metrics_are_event_scoped() {
    use crate::indexer::{
        KvIndexerMetrics, METRIC_EVENT_REMOVED, METRIC_STATUS_BLOCK_NOT_FOUND,
        METRIC_WARNING_DUPLICATE_STORE,
    };

    let workers = workers();
    let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
    let index = ThreadPoolIndexer::new_with_metrics(
        EventTransposedCkfIndexer::new(workers, CkfConfig::new(32)).unwrap(),
        1,
        32,
        Some(Arc::clone(&metrics)),
    );
    KvIndexerInterface::apply_event(&index, store_event(workers[0], &[10, 20], 1000)).await;
    KvIndexerInterface::apply_event(&index, store_event(workers[0], &[10, 20], 2000)).await;
    KvIndexerInterface::apply_event(&index, store_event(workers[0], &[10, 30], 3000)).await;
    KvIndexerInterface::apply_event(&index, remove_event(workers[0], &[999, 20])).await;
    KvIndexerInterface::flush(&index).await;

    assert_eq!(
        metrics
            .kv_cache_event_warnings
            .with_label_values(&[METRIC_WARNING_DUPLICATE_STORE])
            .get(),
        1
    );
    assert_eq!(
        metrics
            .kv_cache_events_applied
            .with_label_values(&[METRIC_EVENT_REMOVED, METRIC_STATUS_BLOCK_NOT_FOUND])
            .get(),
        1
    );
    index.shutdown();
}

#[derive(Default)]
struct StudyStats {
    queries: u64,
    distinct_queries: u64,
    observations: u64,
    paired_depth_differences: u64,
    paired_array_differences: u64,
    window_one_full_map_mismatches: u64,
    window_eight_full_map_mismatches: u64,
    window_one_mismatches: u64,
    window_eight_mismatches: u64,
    window_one_inflation: u64,
    window_eight_inflation: u64,
    window_one_inflation_magnitude: u64,
    window_eight_inflation_magnitude: u64,
    window_one_under_reports: u64,
    window_eight_under_reports: u64,
    window_one_wrong_best: u64,
    window_eight_wrong_best: u64,
    window_one_probes: u64,
    window_eight_probes: u64,
}

impl StudyStats {
    fn record(
        &mut self,
        linear: [u32; DC_COUNT],
        window_one: [u32; DC_COUNT],
        window_eight: [u32; DC_COUNT],
        window_one_probes: u64,
        window_eight_probes: u64,
    ) {
        self.queries += 1;
        self.observations += DC_COUNT as u64;
        self.window_one_probes += window_one_probes;
        self.window_eight_probes += window_eight_probes;
        if window_one != window_eight {
            self.paired_array_differences += 1;
        }
        if linear != window_one {
            self.window_one_full_map_mismatches += 1;
        }
        if linear != window_eight {
            self.window_eight_full_map_mismatches += 1;
        }
        if best_lane(linear) != best_lane(window_one) {
            self.window_one_wrong_best += 1;
        }
        if best_lane(linear) != best_lane(window_eight) {
            self.window_eight_wrong_best += 1;
        }

        for lane in 0..DC_COUNT {
            if window_one[lane] != window_eight[lane] {
                self.paired_depth_differences += 1;
            }
            record_depth_error(
                linear[lane],
                window_one[lane],
                &mut self.window_one_mismatches,
                &mut self.window_one_inflation,
                &mut self.window_one_inflation_magnitude,
                &mut self.window_one_under_reports,
            );
            record_depth_error(
                linear[lane],
                window_eight[lane],
                &mut self.window_eight_mismatches,
                &mut self.window_eight_inflation,
                &mut self.window_eight_inflation_magnitude,
                &mut self.window_eight_under_reports,
            );
        }
    }
}

fn selected_verification_window(stats: &StudyStats) -> usize {
    if stats.paired_depth_differences == 0
        && stats.window_one_mismatches <= stats.window_eight_mismatches
    {
        1
    } else {
        8
    }
}

#[test]
fn default_selection_decision_covers_both_outcomes() {
    let no_window_difference = StudyStats {
        window_one_mismatches: 4,
        window_eight_mismatches: 4,
        ..StudyStats::default()
    };
    assert_eq!(selected_verification_window(&no_window_difference), 1);

    let paired_difference = StudyStats {
        paired_depth_differences: 1,
        ..StudyStats::default()
    };
    assert_eq!(selected_verification_window(&paired_difference), 8);

    let additional_linear_disagreement = StudyStats {
        window_one_mismatches: 1,
        ..StudyStats::default()
    };
    assert_eq!(
        selected_verification_window(&additional_linear_disagreement),
        8
    );
}

#[test]
fn study_corpus_contains_1024_distinct_queries_at_every_length() {
    const QUERIES: usize = 1_024;
    const QUERY_LENGTHS: [usize; 5] = [1, 8, 32, 128, 512];
    let addressing = CkfAddressing::new(16_384, study_seed(0));
    let chains = study_chains(study_seed(0), addressing, QUERIES, 512);

    for query_len in QUERY_LENGTHS {
        assert_eq!(
            distinct_query_count(&chains, query_len),
            QUERIES,
            "query_len={query_len}"
        );
    }
}

#[test]
#[ignore = "deterministic fixed-window CKF accuracy control"]
fn fixed_verification_window_study() {
    const BUCKETS: usize = 16_384;
    const QUERIES_PER_CELL: usize = 1_024;
    const MAX_QUERY_LEN: usize = 512;
    const MAX_RESIDENT_PREFIX: usize = 32;
    const MAX_KICKS: usize = 4_096;
    const OCCUPANCIES: [usize; 3] = [50, 80, 90];
    const QUERY_LENGTHS: [usize; 5] = [1, 8, 32, 128, 512];
    const SEEDS: usize = 8;

    let mut totals = StudyStats::default();
    for seed_index in 0..SEEDS {
        let seed = study_seed(seed_index);
        let addressing = CkfAddressing::new(BUCKETS, seed);
        let chains = study_chains(seed, addressing, QUERIES_PER_CELL, MAX_QUERY_LEN);
        let query_hashes: HashSet<u64> = chains
            .iter()
            .flat_map(|chain| chain.sequence_hashes.iter().copied())
            .collect();
        let prefix_depths = study_prefix_depths(seed_index, QUERIES_PER_CELL, MAX_RESIDENT_PREFIX);
        let distinct_queries: HashMap<usize, usize> = QUERY_LENGTHS
            .into_iter()
            .map(|query_len| (query_len, distinct_query_count(&chains, query_len)))
            .collect();
        assert!(
            distinct_queries
                .values()
                .all(|&count| count == QUERIES_PER_CELL),
            "seed_index={seed_index} distinct_queries={distinct_queries:?}"
        );

        for occupancy in OCCUPANCIES {
            let table = TransposedCkfTable::<DC_COUNT>::new(BUCKETS).unwrap();
            fill_study_table(
                &table,
                addressing,
                &chains,
                &prefix_depths,
                &query_hashes,
                occupancy,
                MAX_KICKS,
                seed,
            );

            for query_len in QUERY_LENGTHS {
                totals.distinct_queries += distinct_queries[&query_len] as u64;
                for chain in chains.iter().take(QUERIES_PER_CELL) {
                    let probes = &chain.probes[..query_len];
                    let linear =
                        linear_prefix_depths::<DC_COUNT>(probes.len(), u16::MAX, |position| {
                            table.probe(probes[position])
                        });
                    let mut window_one_probes = 0u64;
                    let window_one = fixed_window_prefix_depths::<DC_COUNT>(
                        probes.len(),
                        u16::MAX,
                        1,
                        |position| table.prefetch_probe(probes[position]),
                        |position| {
                            window_one_probes += 1;
                            table.probe(probes[position])
                        },
                    );
                    let mut window_eight_probes = 0u64;
                    let window_eight = fixed_window_prefix_depths::<DC_COUNT>(
                        probes.len(),
                        u16::MAX,
                        8,
                        |position| table.prefetch_probe(probes[position]),
                        |position| {
                            window_eight_probes += 1;
                            table.probe(probes[position])
                        },
                    );
                    totals.record(
                        linear,
                        window_one,
                        window_eight,
                        window_one_probes,
                        window_eight_probes,
                    );
                }
            }
        }
    }

    assert_eq!(totals.queries, 122_880);
    assert_eq!(totals.distinct_queries, totals.queries);
    assert_eq!(totals.observations, 1_966_080);
    let expected_default = selected_verification_window(&totals);
    println!(
        "FIXED_WINDOW_STUDY queries={} distinct_queries={} observations={} expected_fixed_window={} paired_depth_differences={} paired_array_differences={} w1_full_map_mismatches={} w8_full_map_mismatches={} w1_lane_mismatches={} w8_lane_mismatches={} w1_inflation={} w8_inflation={} w1_inflation_magnitude={} w8_inflation_magnitude={} w1_under_reports={} w8_under_reports={} w1_wrong_best={} w8_wrong_best={} w1_probes={} w8_probes={}",
        totals.queries,
        totals.distinct_queries,
        totals.observations,
        expected_default,
        totals.paired_depth_differences,
        totals.paired_array_differences,
        totals.window_one_full_map_mismatches,
        totals.window_eight_full_map_mismatches,
        totals.window_one_mismatches,
        totals.window_eight_mismatches,
        totals.window_one_inflation,
        totals.window_eight_inflation,
        totals.window_one_inflation_magnitude,
        totals.window_eight_inflation_magnitude,
        totals.window_one_under_reports,
        totals.window_eight_under_reports,
        totals.window_one_wrong_best,
        totals.window_eight_wrong_best,
        totals.window_one_probes,
        totals.window_eight_probes,
    );
    assert_eq!(expected_default, 8);
}

#[derive(Debug, Default)]
struct FallbackCandidateStats {
    full_map_mismatches: u64,
    lane_mismatches: u64,
    inflation: u64,
    inflation_magnitude: u64,
    under_reports: u64,
    wrong_best: u64,
    probes: u64,
    fallback_probes: u64,
}

impl FallbackCandidateStats {
    fn record(
        &mut self,
        linear: [u32; DC_COUNT],
        actual: [u32; DC_COUNT],
        probes: u64,
        fallback_probes: u64,
    ) {
        self.full_map_mismatches += u64::from(linear != actual);
        self.wrong_best += u64::from(best_lane(linear) != best_lane(actual));
        self.probes += probes;
        self.fallback_probes += fallback_probes;
        for (linear, actual) in linear.into_iter().zip(actual) {
            match actual.cmp(&linear) {
                std::cmp::Ordering::Greater => {
                    self.lane_mismatches += 1;
                    self.inflation += 1;
                    self.inflation_magnitude += u64::from(actual - linear);
                }
                std::cmp::Ordering::Less => {
                    self.lane_mismatches += 1;
                    self.under_reports += 1;
                }
                std::cmp::Ordering::Equal => {}
            }
        }
    }

    fn matches_linear(&self) -> bool {
        self.lane_mismatches == 0
    }
}

#[derive(Debug, Default)]
struct FallbackWindowStudyStats {
    queries: u64,
    distinct_queries: u64,
    observations: u64,
    window_one: FallbackCandidateStats,
    window_two: FallbackCandidateStats,
    window_eight: FallbackCandidateStats,
}

fn selected_fallback_window(stats: &FallbackWindowStudyStats) -> usize {
    if stats.window_one.matches_linear() {
        1
    } else if stats.window_two.matches_linear() {
        2
    } else {
        8
    }
}

#[test]
fn fallback_selection_prefers_the_smallest_linear_equivalent_window() {
    let window_two = FallbackWindowStudyStats {
        window_one: FallbackCandidateStats {
            lane_mismatches: 1,
            ..FallbackCandidateStats::default()
        },
        ..FallbackWindowStudyStats::default()
    };
    assert_eq!(selected_fallback_window(&window_two), 2);

    let window_eight = FallbackWindowStudyStats {
        window_one: FallbackCandidateStats {
            lane_mismatches: 1,
            ..FallbackCandidateStats::default()
        },
        window_two: FallbackCandidateStats {
            lane_mismatches: 1,
            ..FallbackCandidateStats::default()
        },
        ..FallbackWindowStudyStats::default()
    };
    assert_eq!(selected_fallback_window(&window_eight), 8);
}

#[test]
#[ignore = "deterministic provenance-fallback window accuracy study"]
fn provenance_fallback_window_study() {
    const BUCKETS: usize = 16_384;
    const QUERIES_PER_CELL: usize = 1_024;
    const MAX_QUERY_LEN: usize = 512;
    const MAX_RESIDENT_PREFIX: usize = 32;
    const MAX_KICKS: usize = 4_096;
    const OCCUPANCIES: [usize; 3] = [50, 80, 90];
    const QUERY_LENGTHS: [usize; 5] = [1, 8, 32, 128, 512];
    const SEEDS: usize = 8;

    let mut totals = FallbackWindowStudyStats::default();
    for seed_index in 0..SEEDS {
        let seed = study_seed(seed_index);
        let addressing = CkfAddressing::new(BUCKETS, seed);
        let chains = study_chains(seed, addressing, QUERIES_PER_CELL, MAX_QUERY_LEN);
        let query_hashes: HashSet<u64> = chains
            .iter()
            .flat_map(|chain| chain.sequence_hashes.iter().copied())
            .collect();
        let prefix_depths = study_prefix_depths(seed_index, QUERIES_PER_CELL, MAX_RESIDENT_PREFIX);
        let distinct_queries: HashMap<usize, usize> = QUERY_LENGTHS
            .into_iter()
            .map(|query_len| (query_len, distinct_query_count(&chains, query_len)))
            .collect();
        assert!(
            distinct_queries
                .values()
                .all(|&count| count == QUERIES_PER_CELL),
            "seed_index={seed_index} distinct_queries={distinct_queries:?}"
        );

        for occupancy in OCCUPANCIES {
            let table = TransposedCkfTable::<DC_COUNT>::new(BUCKETS).unwrap();
            fill_study_table(
                &table,
                addressing,
                &chains,
                &prefix_depths,
                &query_hashes,
                occupancy,
                MAX_KICKS,
                seed,
            );

            for query_len in QUERY_LENGTHS {
                totals.distinct_queries += distinct_queries[&query_len] as u64;
                for chain in chains.iter().take(QUERIES_PER_CELL) {
                    let probes = &chain.probes[..query_len];
                    let linear =
                        linear_prefix_depths::<DC_COUNT>(probes.len(), u16::MAX, |position| {
                            table.probe(probes[position])
                        });
                    let (window_one, window_one_probes) = run_fallback_candidate(&table, probes, 1);
                    let (window_two, window_two_probes) = run_fallback_candidate(&table, probes, 2);
                    let (window_eight, window_eight_probes) =
                        run_fallback_candidate(&table, probes, 8);

                    totals.queries += 1;
                    totals.observations += DC_COUNT as u64;
                    totals.window_one.record(
                        linear,
                        window_one.depths,
                        window_one_probes,
                        window_one.fallback.probe_calls,
                    );
                    totals.window_two.record(
                        linear,
                        window_two.depths,
                        window_two_probes,
                        window_two.fallback.probe_calls,
                    );
                    totals.window_eight.record(
                        linear,
                        window_eight.depths,
                        window_eight_probes,
                        window_eight.fallback.probe_calls,
                    );
                }
            }
        }
    }

    assert_eq!(totals.queries, 122_880);
    assert_eq!(totals.distinct_queries, totals.queries);
    assert_eq!(totals.observations, 1_966_080);
    let expected_default = selected_fallback_window(&totals);
    println!(
        "FALLBACK_WINDOW_STUDY queries={} distinct_queries={} observations={} expected_default={} w1={:?} w2={:?} w8={:?}",
        totals.queries,
        totals.distinct_queries,
        totals.observations,
        expected_default,
        totals.window_one,
        totals.window_two,
        totals.window_eight,
    );
    assert!(!totals.window_one.matches_linear());
    assert!(totals.window_two.matches_linear());
    assert!(totals.window_eight.matches_linear());
    assert_eq!(expected_default, 2);
    assert_eq!(
        CkfConfig::default().search.verification_window,
        expected_default
    );
}

fn run_fallback_candidate(
    table: &TransposedCkfTable<DC_COUNT>,
    probes: &[CkfProbe],
    window: usize,
) -> (super::search::PrefixSearchResult<DC_COUNT>, u64) {
    let mut probe_calls = 0u64;
    let result = find_prefix_depths_with_test_stats::<DC_COUNT>(
        probes.len(),
        u16::MAX,
        window,
        |position| table.prefetch_probe(probes[position]),
        |position| {
            probe_calls += 1;
            table.probe(probes[position])
        },
    );
    (result, probe_calls)
}

struct StudyChain {
    local_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<u64>,
    probes: Vec<CkfProbe>,
}

fn study_chains(
    seed: u64,
    addressing: CkfAddressing,
    chain_count: usize,
    chain_len: usize,
) -> Vec<StudyChain> {
    let mut rng = StudyRng(seed);
    (0..chain_count)
        .map(|_| {
            let local_hashes: Vec<_> = (0..chain_len).map(|_| LocalBlockHash(rng.next())).collect();
            let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
            let probes = sequence_hashes
                .iter()
                .copied()
                .map(|hash| addressing.prepare(hash))
                .collect();
            StudyChain {
                local_hashes,
                sequence_hashes,
                probes,
            }
        })
        .collect()
}

fn distinct_query_count(chains: &[StudyChain], query_len: usize) -> usize {
    chains
        .iter()
        .map(|chain| &chain.local_hashes[..query_len])
        .collect::<HashSet<_>>()
        .len()
}

fn study_prefix_depths(
    seed_index: usize,
    chain_count: usize,
    max_query_len: usize,
) -> Vec<[usize; DC_COUNT]> {
    (0..chain_count)
        .map(|chain| {
            std::array::from_fn(|lane| {
                (chain * 131 + lane * 37 + seed_index * 17) % (max_query_len + 1)
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn fill_study_table(
    table: &TransposedCkfTable<DC_COUNT>,
    addressing: CkfAddressing,
    chains: &[StudyChain],
    prefix_depths: &[[usize; DC_COUNT]],
    query_hashes: &HashSet<u64>,
    occupancy: usize,
    max_kicks: usize,
    seed: u64,
) {
    let target = BUCKET_SLOTS_PER_LANE * occupancy / 100;
    let mut noise = StudyRng(seed ^ (occupancy as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    let prefix_depths_by_lane: [Vec<_>; DC_COUNT] = std::array::from_fn(|lane| {
        prefix_depths
            .iter()
            .map(|chain_depths| chain_depths[lane])
            .collect()
    });
    for (lane, lane_prefix_depths) in prefix_depths_by_lane.iter().enumerate() {
        let view = table.lane(lane);
        let mut state = DcWriterState::new(target, max_kicks, seed ^ lane as u64).unwrap();
        for (chain, &prefix_depth) in chains.iter().zip(lane_prefix_depths) {
            for &hash in &chain.sequence_hashes[..prefix_depth] {
                insert_study_hash(&view, &addressing, &mut state, hash, max_kicks);
            }
        }
        while state.resident.len() < target {
            let hash = noise.next();
            if query_hashes.contains(&hash)
                || state.resident.contains(&ExternalSequenceBlockHash(hash))
            {
                continue;
            }
            insert_study_hash(&view, &addressing, &mut state, hash, max_kicks);
        }
        assert_eq!(
            state.resident.len(),
            target,
            "lane={lane} occupancy={occupancy}"
        );
    }
}

const BUCKET_SLOTS_PER_LANE: usize = 16_384 * 4;

fn insert_study_hash<S: CuckooBucketStore>(
    store: &S,
    addressing: &CkfAddressing,
    state: &mut DcWriterState,
    hash: u64,
    max_kicks: usize,
) {
    let hash = ExternalSequenceBlockHash(hash);
    if state.resident.contains(&hash) {
        return;
    }
    CuckooMutator::new(store, addressing, max_kicks)
        .insert(hash, &mut state.rng, &mut state.scratch, |_| {})
        .unwrap_or_else(|error| panic!("failed to reach study occupancy: {error}"));
    state.resident.insert(hash);
}

fn record_depth_error(
    expected: u32,
    actual: u32,
    mismatches: &mut u64,
    inflation: &mut u64,
    inflation_magnitude: &mut u64,
    under_reports: &mut u64,
) {
    match actual.cmp(&expected) {
        std::cmp::Ordering::Greater => {
            *mismatches += 1;
            *inflation += 1;
            *inflation_magnitude += u64::from(actual - expected);
        }
        std::cmp::Ordering::Less => {
            *mismatches += 1;
            *under_reports += 1;
        }
        std::cmp::Ordering::Equal => {}
    }
}

fn best_lane(depths: [u32; DC_COUNT]) -> Option<usize> {
    depths
        .into_iter()
        .enumerate()
        .filter(|(_, depth)| *depth > 0)
        .max_by(|(left_lane, left), (right_lane, right)| {
            left.cmp(right).then_with(|| right_lane.cmp(left_lane))
        })
        .map(|(lane, _)| lane)
}

fn study_seed(index: usize) -> u64 {
    let mut rng = StudyRng(0x5DEE_CE66_D1B5_4A33 ^ index as u64);
    rng.next()
}

struct StudyRng(u64);

impl StudyRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

fn workers() -> [WorkerWithDpRank; DC_COUNT] {
    std::array::from_fn(|lane| WorkerWithDpRank::new(100 + lane as u64, 0))
}

fn workers_with_shared_worker() -> [WorkerWithDpRank; DC_COUNT] {
    std::array::from_fn(|lane| match lane {
        0 => WorkerWithDpRank::new(7, 0),
        1 => WorkerWithDpRank::new(7, 1),
        _ => WorkerWithDpRank::new(100 + lane as u64, 0),
    })
}

fn store_event(
    worker: WorkerWithDpRank,
    sequence_hashes: &[u64],
    token_hash_base: u64,
) -> RouterEvent {
    RouterEvent {
        worker_id: worker.worker_id,
        storage_tier: StorageTier::Device,
        event: KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: Some(0),
                blocks: sequence_hashes
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, hash)| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(hash),
                        tokens_hash: LocalBlockHash(token_hash_base + index as u64),
                        mm_extra_info: None,
                    })
                    .collect(),
            }),
            dp_rank: worker.dp_rank,
        },
    }
}

fn remove_event(worker: WorkerWithDpRank, hashes: &[u64]) -> RouterEvent {
    RouterEvent {
        worker_id: worker.worker_id,
        storage_tier: StorageTier::Device,
        event: KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: hashes
                    .iter()
                    .copied()
                    .map(ExternalSequenceBlockHash)
                    .collect(),
            }),
            dp_rank: worker.dp_rank,
        },
    }
}

fn clear_event(worker_id: u64) -> RouterEvent {
    RouterEvent {
        worker_id,
        storage_tier: StorageTier::Device,
        event: KvCacheEvent {
            event_id: 3,
            data: KvCacheEventData::Cleared,
            dp_rank: 0,
        },
    }
}

fn colliding_hashes(
    addressing: CkfAddressing,
) -> (
    ExternalSequenceBlockHash,
    ExternalSequenceBlockHash,
    CkfProbe,
) {
    let mut seen = HashMap::new();
    for hash in 0..1_000_000u64 {
        let probe = addressing.prepare(hash);
        let key = (probe.fingerprint, probe.bucket_a, probe.bucket_b);
        if let Some(first) = seen.insert(key, hash) {
            return (
                ExternalSequenceBlockHash(first),
                ExternalSequenceBlockHash(hash),
                probe,
            );
        }
    }
    panic!("failed to find deterministic CKF representation collision")
}

fn fingerprint_copies<S: CuckooBucketStore>(store: &S, probe: CkfProbe) -> usize {
    [probe.bucket_a, probe.bucket_b]
        .into_iter()
        .map(|bucket| {
            let bucket = store.load_bucket(bucket);
            (0..4)
                .filter(|&slot| bucket.slot(slot) == probe.fingerprint)
                .count()
        })
        .sum()
}
