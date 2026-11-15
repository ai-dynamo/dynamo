// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rustc_hash::{FxHashMap, FxHashSet};

use dynamo_kv_hashing::LocalBlockHash;
use dynamo_kv_hashing::routing::compute_seq_hash_for_block;

use super::addressing::{CkfAddressing, CkfProbe};
use super::bucket::{CuckooBucketStore, PackedBucket, TransposedCkfTable};
use super::mutator::{CuckooInsertionScratch, CuckooMutator};
use super::search::{
    SearchPhase, SearchTraceEvent, SearchTraceKind, find_prefix_depths,
    find_prefix_depths_with_test_stats, find_prefix_depths_with_test_trace, linear_prefix_depths,
};
use super::{CanonicalSequenceBlockHash, CkfConfig};

const TEST_SEED: u64 = 0x1234_5678_9ABC_DEF0;

#[test]
fn config_reports_bucket_count_without_allocating_storage() {
    assert_eq!(CkfConfig::new(1).bucket_count().unwrap(), 2);
    assert_eq!(CkfConfig::new(1_048_576).bucket_count().unwrap(), 524_288);
    assert!(CkfConfig::new(0).bucket_count().is_err());
}

fn canonical(hash: u64) -> CanonicalSequenceBlockHash {
    CanonicalSequenceBlockHash::root(LocalBlockHash(hash))
}

const fn search_trace_event(
    kind: SearchTraceKind,
    phase: SearchPhase,
    position: usize,
    lanes: u16,
) -> SearchTraceEvent {
    SearchTraceEvent {
        kind,
        phase,
        position,
        lanes,
    }
}

fn grouped_level_events(phase: SearchPhase, groups: &[(usize, u16)]) -> Vec<SearchTraceEvent> {
    groups
        .iter()
        .map(|&(position, lanes)| {
            search_trace_event(SearchTraceKind::Prefetch, phase, position, lanes)
        })
        .chain(groups.iter().map(|&(position, lanes)| {
            search_trace_event(SearchTraceKind::Probe, phase, position, lanes)
        }))
        .collect()
}

fn sequential_probe_events(phase: SearchPhase, groups: &[(usize, u16)]) -> Vec<SearchTraceEvent> {
    groups
        .iter()
        .flat_map(|&(position, lanes)| {
            [
                search_trace_event(SearchTraceKind::Prefetch, phase, position, lanes),
                search_trace_event(SearchTraceKind::Probe, phase, position, lanes),
            ]
        })
        .collect()
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
fn bounded_relocation_rolls_back_every_write_and_reports_no_touched_bucket() {
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
    let mut rng = TEST_SEED;
    let mut scratch = CuckooInsertionScratch::new(1).unwrap();
    let mut touched = Vec::new();
    let result = CuckooMutator::new(&view, &addressing, 1).insert_with_originals(
        canonical(999),
        &mut rng,
        &mut scratch,
        |bucket, before| touched.push((bucket, before)),
    );

    assert!(matches!(
        result,
        Err(dynamo_kv_events::event::KvCacheEventError::CapacityExhausted)
    ));
    assert_eq!(view.load_bucket(0), full_a);
    assert_eq!(view.load_bucket(1), full_b);
    assert!(touched.is_empty());
}

#[test]
fn distinct_hashes_with_one_ckf_representation_keep_separate_copies() {
    let addressing = CkfAddressing::new(2, TEST_SEED);
    let (first, second, representation) = colliding_hashes(addressing);
    let table = TransposedCkfTable::<1>::new(2).unwrap();
    let view = table.lane(0);
    let mut rng = TEST_SEED;
    let mut scratch = CuckooInsertionScratch::new(20).unwrap();
    let mutator = CuckooMutator::new(&view, &addressing, 20);

    mutator
        .insert_with_originals(first, &mut rng, &mut scratch, |_, _| {})
        .unwrap();
    mutator
        .insert_with_originals(second, &mut rng, &mut scratch, |_, _| {})
        .unwrap();
    assert_eq!(fingerprint_copies(&view, representation), 2);

    mutator.remove_with_original(first, |_, _| {}).unwrap();
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
fn exponential_search_prefetches_exactly_one_sample_ahead() {
    let (result, trace) = find_prefix_depths_with_test_trace::<1>(9, 1, 1, |_| {}, |_| 1);
    let schedule: Vec<_> = trace
        .into_iter()
        .filter(|event| matches!(event.phase, SearchPhase::Initial | SearchPhase::Exponential))
        .collect();

    assert_eq!(result.depths, [9]);
    assert_eq!(
        schedule,
        vec![
            search_trace_event(SearchTraceKind::Probe, SearchPhase::Initial, 0, 1),
            search_trace_event(SearchTraceKind::Prefetch, SearchPhase::Exponential, 1, 1),
            search_trace_event(SearchTraceKind::Prefetch, SearchPhase::Exponential, 2, 1),
            search_trace_event(SearchTraceKind::Probe, SearchPhase::Exponential, 1, 1),
            search_trace_event(SearchTraceKind::Prefetch, SearchPhase::Exponential, 4, 1),
            search_trace_event(SearchTraceKind::Probe, SearchPhase::Exponential, 2, 1),
            search_trace_event(SearchTraceKind::Prefetch, SearchPhase::Exponential, 8, 1),
            search_trace_event(SearchTraceKind::Probe, SearchPhase::Exponential, 4, 1),
            search_trace_event(SearchTraceKind::Probe, SearchPhase::Exponential, 8, 1),
        ]
    );
}

#[test]
fn divergent_lanes_group_binary_midpoints_once_per_level() {
    const PREFIX_DEPTHS: [usize; 16] = [3, 3, 6, 7, 11, 11, 15, 17, 3, 6, 11, 15, 17, 7, 3, 11];
    let membership = |position: usize| {
        PREFIX_DEPTHS
            .iter()
            .enumerate()
            .fold(0u16, |mask, (lane, &depth)| {
                mask | (u16::from(position < depth) << lane)
            })
    };
    let (result, trace) =
        find_prefix_depths_with_test_trace::<16>(17, u16::MAX, 1, |_| {}, membership);
    let binary_schedule: Vec<_> = trace
        .into_iter()
        .filter(|event| event.phase == SearchPhase::Binary)
        .collect();

    assert_eq!(result.depths, PREFIX_DEPTHS.map(|depth| depth as u32));
    assert_eq!(
        binary_schedule,
        [
            grouped_level_events(
                SearchPhase::Binary,
                &[(3, 0x4103), (6, 0x220c), (12, 0x8c70)]
            ),
            grouped_level_events(
                SearchPhase::Binary,
                &[(5, 0x0204), (7, 0x2008), (10, 0x8430), (14, 0x0840)],
            ),
            grouped_level_events(SearchPhase::Binary, &[(11, 0x8430), (15, 0x0840)]),
        ]
        .concat()
    );
}

#[test]
fn overlapping_verification_windows_isolate_lanes_and_retire_first_misses() {
    const PREFIX_DEPTHS: [usize; 4] = [9, 10, 11, 12];
    let membership = |position: usize| {
        PREFIX_DEPTHS
            .iter()
            .enumerate()
            .fold(0u16, |mask, (lane, &depth)| {
                let hole = matches!((lane, position), (0, 5 | 7) | (1, 7) | (2, 6 | 9) | (3, 5));
                mask | (u16::from(position < depth && !hole) << lane)
            })
    };
    let (result, trace) =
        find_prefix_depths_with_test_trace::<4>(17, 0b1111, 4, |_| {}, membership);
    let verification_schedule: Vec<_> = trace
        .into_iter()
        .filter(|event| event.phase == SearchPhase::Verification)
        .collect();

    assert_eq!(result.depths, [5, 7, 9, 12]);
    assert_eq!(
        verification_schedule,
        sequential_probe_events(
            SearchPhase::Verification,
            &[
                (5, 0b0001),
                (6, 0b0010),
                (7, 0b0110),
                (8, 0b1100),
                (9, 0b1100),
                (10, 0b1000),
                (11, 0b1000),
            ],
        )
    );
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
fn staggered_fallback_cursors_group_only_lanes_at_the_same_position() {
    let membership = |position: usize| {
        u16::from(position < 44 || position == 48)
            | (u16::from(position < 42 || position == 44) << 1)
    };
    let (result, trace) = find_prefix_depths_with_test_trace::<2>(65, 0b11, 2, |_| {}, membership);
    let fallback_schedule: Vec<_> = trace
        .into_iter()
        .filter(|event| event.phase == SearchPhase::Fallback)
        .collect();
    let expected_groups: Vec<_> = (33..=40)
        .map(|position| (position, 0b01))
        .chain((41..=42).map(|position| (position, 0b11)))
        .chain((43..=44).map(|position| (position, 0b01)))
        .collect();

    assert_eq!(result.depths, [44, 42]);
    assert_eq!(result.fallback.left_edge_lanes, 2);
    assert_eq!(result.fallback.activated_lanes, 2);
    assert_eq!(result.fallback.probe_calls, 12);
    assert_eq!(result.fallback.lane_probes, 14);
    assert_eq!(result.fallback.provenance_skips, 0);
    assert_eq!(
        fallback_schedule,
        sequential_probe_events(SearchPhase::Fallback, &expected_groups)
    );
}

#[test]
fn valid_fallback_lane_progresses_while_second_order_lane_is_skipped() {
    let membership = |position: usize| {
        u16::from(position < 40 || position == 48)
            | (u16::from(position < 41 || (48..=51).contains(&position)) << 1)
    };
    let (result, trace) = find_prefix_depths_with_test_trace::<2>(65, 0b11, 8, |_| {}, membership);
    let fallback_probes: Vec<_> = trace
        .into_iter()
        .filter(|event| {
            event.phase == SearchPhase::Fallback && event.kind == SearchTraceKind::Probe
        })
        .collect();

    assert_eq!(result.depths, [40, 44]);
    assert_eq!(result.fallback.left_edge_lanes, 2);
    assert_eq!(result.fallback.activated_lanes, 1);
    assert_eq!(result.fallback.probe_calls, 8);
    assert_eq!(result.fallback.lane_probes, 8);
    assert_eq!(result.fallback.provenance_skips, 1);
    assert_eq!(
        fallback_probes,
        (33..=40)
            .map(|position| {
                search_trace_event(
                    SearchTraceKind::Probe,
                    SearchPhase::Fallback,
                    position,
                    0b01,
                )
            })
            .collect::<Vec<_>>()
    );
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

#[test]
fn store_remove_churn_drains_every_physical_slot() {
    let table = TransposedCkfTable::<1>::new(64).unwrap();
    let view = table.lane(0);
    let addressing = CkfAddressing::new(64, TEST_SEED);
    let mut rng = TEST_SEED;
    let mut scratch = CuckooInsertionScratch::new(500).unwrap();
    let mut resident = FxHashSet::default();
    let hashes: Vec<_> = (0..128).map(|hash| canonical(10_000 + hash)).collect();
    let mutator = CuckooMutator::new(&view, &addressing, 500);
    for &hash in &hashes {
        mutator
            .insert_with_originals(hash, &mut rng, &mut scratch, |_, _| {})
            .unwrap();
        resident.insert(hash);
    }
    for &hash in hashes.iter().rev() {
        mutator.remove_with_original(hash, |_, _| {}).unwrap();
        resident.remove(&hash);
    }

    assert!(resident.is_empty());
    assert!(
        (0..view.bucket_count())
            .all(|bucket| { view.load_bucket(bucket) == PackedBucket::default() })
    );
}

#[cfg(feature = "export")]
#[test]
fn new_event_errors_have_finite_metric_labels() {
    use crate::metrics::{
        KvIndexerMetrics, METRIC_EVENT_STORED, METRIC_STATUS_CAPACITY_EXHAUSTED,
        METRIC_STATUS_INDEXER_INVARIANT_VIOLATION,
    };

    let metrics = KvIndexerMetrics::new_unregistered();
    metrics.increment_event_applied(
        METRIC_EVENT_STORED,
        Err(dynamo_kv_events::event::KvCacheEventError::CapacityExhausted),
    );
    metrics.increment_event_applied(
        METRIC_EVENT_STORED,
        Err(dynamo_kv_events::event::KvCacheEventError::IndexerInvariantViolation),
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

fn colliding_hashes(
    addressing: CkfAddressing,
) -> (
    CanonicalSequenceBlockHash,
    CanonicalSequenceBlockHash,
    CkfProbe,
) {
    let mut seen = FxHashMap::default();
    for hash in 0..1_000_000u64 {
        let probe = addressing.prepare(hash);
        let key = (probe.fingerprint, probe.bucket_a, probe.bucket_b);
        if let Some(first) = seen.insert(key, hash) {
            return (canonical(first), canonical(hash), probe);
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
