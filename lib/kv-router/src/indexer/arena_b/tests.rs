// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Branch-B tests: every invariant of spec B 11 through `probe_check`, the B-only hooks
//! (generation and version wrap, poisoning, forced retries, migration), the CRTC race
//! tests ported, the child-table and long-chain shape tests, and an interval-bound soak.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use tokio::sync::oneshot;

use super::runs::{Frees, MAX_RUN_LEN, Probe, ROOT, ROOT_GEN, Read, child_key};
use super::*;
use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer, WorkerTask};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheRemoveData,
    KvCacheStoreData, LocalBlockHash, WorkerWithDpRank, compute_seq_hash_for_block,
};
use crate::test_utils::{
    make_clear_event, make_remove_event, make_remove_event_with_parent, make_store_event,
    make_store_event_with_parent, remove_event, router_event, stored_blocks_with_sequence_hashes,
};

fn rank(id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(id, 0)
}

fn locals(hashes: &[u64]) -> Vec<LocalBlockHash> {
    hashes.iter().copied().map(LocalBlockHash).collect()
}

fn seqs(chain: &[u64]) -> Vec<u64> {
    compute_seq_hash_for_block(&locals(chain))
}

/// Stores `chain[from..to]` for `worker`, with the parent at `from - 1`.
fn store(worker: WorkerWithDpRank, chain: &[u64], from: usize, to: usize) -> RouterEvent {
    let hashes = seqs(&chain[..to]);
    router_event(
        worker.worker_id,
        0,
        worker.dp_rank,
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: from
                .checked_sub(1)
                .map(|i| ExternalSequenceBlockHash(hashes[i])),
            start_position: None,
            blocks: stored_blocks_with_sequence_hashes(
                &locals(&chain[from..to]),
                &hashes[from..to],
            ),
        }),
    )
}

/// Removes the blocks of `chain` at `positions`.
fn remove(worker: WorkerWithDpRank, chain: &[u64], positions: &[usize]) -> RouterEvent {
    let hashes = seqs(chain);
    remove_event(
        worker.worker_id,
        0,
        worker.dp_rank,
        positions
            .iter()
            .map(|&p| ExternalSequenceBlockHash(hashes[p]))
            .collect(),
    )
}

fn score_of(index: &ArenaIndex, query: &[u64], worker: WorkerWithDpRank) -> u32 {
    index
        .find_matches(&locals(query), false)
        .scores
        .get(&worker)
        .copied()
        .unwrap_or(0)
}

fn score(index: &ArenaIndex, query: &[u64], worker: u64) -> u32 {
    score_of(index, query, rank(worker))
}

fn check(index: &ArenaIndex) {
    if let Err(errors) = index.probe_check() {
        panic!("structural check failed:\n{errors}");
    }
}

fn apply(index: &ArenaIndex, event: RouterEvent) {
    index.apply_event_inline(event).expect("event applies");
}

// ----------------------------------------------------------------------------
// Semantics
// ----------------------------------------------------------------------------

#[test]
fn store_lookup_and_divergent_children() {
    let index = ArenaIndex::new();
    apply(&index, make_store_event(0, &[1, 2, 3]));
    assert_eq!(score(&index, &[1, 2, 3], 0), 3);
    assert_eq!(score(&index, &[1, 2], 0), 2);
    assert_eq!(score(&index, &[1, 5], 0), 1);
    assert_eq!(score(&index, &[9], 0), 0);
    check(&index);

    apply(&index, make_store_event(1, &[1, 2]));
    assert_eq!(score(&index, &[1, 2, 3], 1), 2);
    assert_eq!(score(&index, &[1, 2, 3], 0), 3);
    check(&index);

    // Worker 1 continues past its partial cutoff with a divergent block: a child at
    // offset 2 inside the run, which only holders of the first two positions enter.
    apply(&index, make_store_event_with_parent(1, &[1, 2], &[7, 8]));
    assert_eq!(score(&index, &[1, 2, 7, 8], 1), 4);
    assert_eq!(score(&index, &[1, 2, 7, 8], 0), 2);
    assert_eq!(score(&index, &[1, 2, 3], 1), 2);
    check(&index);

    apply(&index, make_remove_event_with_parent(0, &[1], &[2]));
    assert_eq!(score(&index, &[1, 2, 3], 0), 1);
    check(&index);

    apply(&index, make_clear_event(1));
    assert_eq!(score(&index, &[1, 2, 7, 8], 1), 0);
    check(&index);
    apply(&index, make_clear_event(0));
    assert!(
        index
            .find_matches(&locals(&[1, 2, 3]), false)
            .scores
            .is_empty()
    );
    check(&index);
    assert_eq!(index.live_run_count(), 0, "eager reclamation leaves no run");
}

#[test]
fn a_partial_holder_does_not_continue_past_its_cutoff() {
    let index = ArenaIndex::new();
    apply(&index, store(rank(0), &[1, 2, 3, 4], 0, 4));
    apply(&index, store(rank(1), &[1, 2, 3, 4], 0, 2));
    // Rank 0 hangs an end child; rank 1 holds only the first two positions.
    apply(&index, store(rank(0), &[1, 2, 3, 4, 5], 4, 5));
    assert_eq!(score(&index, &[1, 2, 3, 4, 5], 1), 2);
    assert_eq!(score(&index, &[1, 2, 3, 4, 5], 0), 5);
    // A child at offset 3, under a position rank 1 does not hold.
    apply(&index, store(rank(0), &[1, 2, 3, 9], 3, 4));
    assert_eq!(score(&index, &[1, 2, 3, 9], 1), 2);
    assert_eq!(score(&index, &[1, 2, 3, 9], 0), 4);
    check(&index);
}

#[test]
fn decode_extension_appends_in_place_for_a_sole_holder() {
    let index = ArenaIndex::new();
    let mut prefix = vec![1u64, 2, 3];
    apply(&index, make_store_event(0, &prefix));
    for next in 4..40u64 {
        apply(&index, make_store_event_with_parent(0, &prefix, &[next]));
        prefix.push(next);
    }
    assert_eq!(score(&index, &prefix, 0), 39);
    assert_eq!(index.live_run_count(), 1);
    assert!(index.stats().appends_in_place > 0);
    assert!(index.stats().appends_realloc > 0);
    check(&index);
}

#[test]
fn shared_decode_tails_hang_end_children_instead_of_demoting() {
    let index = ArenaIndex::new();
    let base = [1u64, 2, 3];
    for worker in 0..3 {
        apply(&index, make_store_event(worker, &base));
    }
    // Every rank extends with the same block: the first creates the end child, the
    // others find it.
    for worker in 0..3 {
        apply(&index, make_store_event_with_parent(worker, &base, &[4]));
    }
    for worker in 0..3 {
        assert_eq!(score(&index, &[1, 2, 3, 4], worker), 4);
    }
    assert_eq!(index.live_run_count(), 2);
    assert!(index.stats().appends_in_place == 0);
    check(&index);
}

#[test]
fn mid_chain_removal_truncates_and_restore_heals() {
    let index = ArenaIndex::new();
    apply(&index, make_store_event(0, &[1, 2, 3, 4]));
    apply(&index, make_remove_event_with_parent(0, &[1], &[2]));
    assert_eq!(score(&index, &[1, 2, 3, 4], 0), 1);
    check(&index);
    apply(&index, make_store_event_with_parent(0, &[1], &[2]));
    assert_eq!(score(&index, &[1, 2, 3, 4], 0), 2);
    // A store under an orphan is rejected: its entry went with the truncation.
    assert!(
        index
            .apply_event_inline(make_store_event_with_parent(0, &[1, 2, 3, 4], &[5]))
            .is_err()
    );
    apply(&index, make_store_event_with_parent(0, &[1, 2], &[3, 4]));
    assert_eq!(score(&index, &[1, 2, 3, 4], 0), 4);
    check(&index);
}

#[test]
fn restoring_an_evicted_block_reconnects_a_held_descendant() {
    // Spec B 12: a still-held descendant becomes reachable again once the hole heals.
    let index = ArenaIndex::new();
    let chain = [1u64, 2, 3, 4, 5, 6];
    apply(&index, store(rank(0), &chain, 0, 3));
    apply(&index, store(rank(1), &chain, 0, 3));
    apply(&index, store(rank(0), &chain, 3, 6));
    // Rank 1 shares the first run, so rank 0's tail is a separate end child.
    apply(&index, remove(rank(0), &chain, &[1]));
    assert_eq!(score(&index, &chain, 0), 1);
    apply(&index, store(rank(0), &chain, 1, 3));
    assert_eq!(score(&index, &chain, 0), 6);
    check(&index);
}

#[test]
fn removal_of_unknown_rank_is_block_not_found() {
    let index = ArenaIndex::new();
    assert!(matches!(
        index.apply_event_inline(make_remove_event(5, &[1])),
        Err(KvCacheEventError::BlockNotFound)
    ));
    apply(&index, make_clear_event(5));
    apply(&index, make_store_event(5, &[1]));
    apply(&index, make_clear_event(5));
    assert!(matches!(
        index.apply_event_inline(make_remove_event(5, &[1])),
        Err(KvCacheEventError::BlockNotFound)
    ));
}

#[test]
fn stale_parent_is_rejected_and_its_entry_dropped() {
    let index = ArenaIndex::new();
    apply(&index, make_store_event(0, &[1, 2, 3]));
    apply(&index, make_remove_event_with_parent(0, &[1, 2], &[3]));
    assert!(matches!(
        index.apply_event_inline(make_store_event_with_parent(0, &[1, 2, 3], &[4])),
        Err(KvCacheEventError::ParentBlockNotFound)
    ));
    assert!(matches!(
        index.apply_event_inline(make_store_event_with_parent(0, &[9], &[4])),
        Err(KvCacheEventError::ParentBlockNotFound)
    ));
    check(&index);
}

#[test]
fn details_report_the_scored_tail_and_the_transfer_chain() {
    let index = ArenaIndex::new();
    apply(&index, make_store_event(0, &[1, 2, 3]));
    apply(&index, make_store_event(1, &[1, 2]));
    apply(&index, make_store_event_with_parent(1, &[1, 2], &[7]));
    let seq = seqs(&[1, 2, 3]);
    let details = index.find_match_details(&locals(&[1, 2, 3, 4]), false, true);
    assert_eq!(details.overlap_scores.scores.get(&rank(0)), Some(&3));
    assert_eq!(details.overlap_scores.scores.get(&rank(1)), Some(&2));
    assert_eq!(
        details.last_matched_hashes.get(&rank(0)),
        Some(&ExternalSequenceBlockHash(seq[2]))
    );
    assert_eq!(
        details.last_matched_hashes.get(&rank(1)),
        Some(&ExternalSequenceBlockHash(seq[1]))
    );
    let candidates = details.kv_transfer_candidates.expect("chain retained");
    assert_eq!(
        candidates.block_hashes,
        seq.iter()
            .map(|&h| ExternalSequenceBlockHash(h))
            .collect::<Vec<_>>()
    );
    // Through a child: the tail hash comes from the child run.
    let seq = seqs(&[1, 2, 7]);
    let details = index.find_match_details(&locals(&[1, 2, 7]), false, false);
    assert_eq!(details.overlap_scores.scores.get(&rank(1)), Some(&3));
    assert_eq!(
        details.last_matched_hashes.get(&rank(1)),
        Some(&ExternalSequenceBlockHash(seq[2]))
    );
}

#[test]
fn early_exit_stops_at_a_single_continuing_rank() {
    let index = ArenaIndex::new();
    apply(&index, store(rank(0), &[1, 2, 3, 4], 0, 4));
    apply(&index, store(rank(1), &[1, 2], 0, 2));
    apply(&index, store(rank(0), &[1, 2, 3, 4, 5], 4, 5));
    let scores = index.find_matches(&locals(&[1, 2, 3, 4, 5]), true);
    // Rank 1 drops at the run; rank 0 continues alone and is scored where it continued.
    assert_eq!(scores.scores.get(&rank(1)), Some(&2));
    assert!(
        scores
            .scores
            .get(&rank(0))
            .is_some_and(|&s| (4..=5).contains(&s))
    );
}

#[test]
fn dump_round_trips_without_overcounting() {
    let index = ArenaIndex::new();
    let mut rng = fastrand::Rng::with_seed(7);
    let chains: Vec<Vec<u64>> = (0..12)
        .map(|c| {
            let mut chain = vec![1, 2];
            chain.extend((0..rng.u64(1..8)).map(|i| 100 * (c % 4) + i));
            chain
        })
        .collect();
    for worker in 0..6u64 {
        for chain in chains.iter().skip(worker as usize % 3).step_by(2) {
            let to = rng.usize(1..=chain.len());
            let _ = index.apply_event_inline(store(rank(worker), chain, 0, to));
        }
    }
    let events = index.dump_tree_as_events();
    let replay = ArenaIndex::new();
    for event in events {
        replay
            .apply_event_inline(event)
            .expect("dumped events chain from held parents");
    }
    for chain in &chains {
        let original = index.find_matches(&locals(chain), false).scores;
        let replayed = replay.find_matches(&locals(chain), false).scores;
        assert_eq!(
            original, replayed,
            "hole-free history must round-trip exactly"
        );
    }
    check(&replay);
}

#[test]
fn lifecycle_delegate_sees_first_and_last_owners() {
    #[derive(Default)]
    struct Counting {
        created: AtomicU64,
        removed: AtomicU64,
    }
    impl crate::indexer::KvIndexerDelegate for Counting {
        fn on_create(&self, _hash: ExternalSequenceBlockHash) {
            self.created.fetch_add(1, Ordering::Relaxed);
        }
        fn on_remove(&self, _hash: ExternalSequenceBlockHash) {
            self.removed.fetch_add(1, Ordering::Relaxed);
        }
    }
    let delegate = Arc::new(Counting::default());
    let index = ArenaIndex::new_with_delegate(delegate.clone());
    apply(&index, make_store_event(0, &[1, 2, 3]));
    apply(&index, make_store_event(1, &[1, 2]));
    assert_eq!(delegate.created.load(Ordering::Relaxed), 3);
    apply(&index, make_remove_event_with_parent(0, &[1, 2], &[3]));
    assert_eq!(delegate.removed.load(Ordering::Relaxed), 1);
    apply(&index, make_clear_event(1));
    assert_eq!(delegate.removed.load(Ordering::Relaxed), 1);
    apply(&index, make_clear_event(0));
    assert_eq!(delegate.removed.load(Ordering::Relaxed), 3);
}

// ----------------------------------------------------------------------------
// Structure: splits, chunks, child tables, long chains
// ----------------------------------------------------------------------------

#[test]
fn prefix_cap_splits_keep_scores_and_promote_inside_the_step() {
    let index = ArenaIndex::new();
    let chain: Vec<u64> = (1..=64).collect();
    let owner = rank(1000);
    apply(&index, store(owner, &chain, 0, chain.len()));
    // 40 ranks hold distinct prefixes: more than PARTIAL_CAP partial holders.
    let ranks: Vec<_> = (0..40u64).map(rank).collect();
    for (i, &r) in ranks.iter().enumerate() {
        apply(&index, store(r, &chain, 0, 1 + (i * 3 + 1) % 63));
        check(&index);
    }
    assert!(index.stats().splits_prefix_cap > 0, "the cap forced splits");
    for (i, &r) in ranks.iter().enumerate() {
        assert_eq!(score_of(&index, &chain, r) as usize, 1 + (i * 3 + 1) % 63);
    }
    assert_eq!(score_of(&index, &chain, owner), 64);
    // Map entries named the pre-split run; they resolve through forwarding records.
    for (i, &r) in ranks.iter().enumerate().step_by(3) {
        let held = 1 + (i * 3 + 1) % 63;
        apply(&index, remove(r, &chain, &[held - 1]));
        assert_eq!(score_of(&index, &chain, r) as usize, held - 1);
        if held > 2 {
            apply(&index, store(r, &chain, held - 1, held));
            assert_eq!(score_of(&index, &chain, r) as usize, held);
        }
    }
    check(&index);
    // Removals of whole holders demote them to partial holders, which can split again.
    for &r in ranks.iter().take(10) {
        apply(
            &index,
            store(r, &chain, score_of(&index, &chain, r) as usize, 64),
        );
    }
    for (i, &r) in ranks.iter().take(10).enumerate() {
        apply(&index, remove(r, &chain, &[60 - i]));
        assert_eq!(score_of(&index, &chain, r) as usize, 60 - i);
    }
    check(&index);
    for &r in &ranks {
        apply(&index, make_clear_event(r.worker_id));
    }
    apply(&index, make_clear_event(owner.worker_id));
    check(&index);
    assert_eq!(index.live_run_count(), 0);
}

#[test]
fn high_slots_use_overflow_chunks() {
    let index = ArenaIndex::new();
    let chain = [5u64, 6, 7, 8];
    for worker in 0..600u64 {
        let to = 1 + (worker as usize % 4);
        apply(&index, store(rank(worker), &chain, 0, to));
    }
    for worker in (0..600u64).step_by(37) {
        assert_eq!(
            score(&index, &chain, worker) as usize,
            1 + worker as usize % 4
        );
    }
    let memory = index.memory_report();
    assert!(memory.overflow_bytes > 0);
    for worker in (128..600u64).step_by(2) {
        apply(&index, make_clear_event(worker));
    }
    for worker in (129..600u64).step_by(74) {
        assert_eq!(
            score(&index, &chain, worker) as usize,
            1 + worker as usize % 4
        );
    }
    assert_eq!(score(&index, &chain, 130), 0);
    check(&index);
}

#[test]
fn child_table_load_stays_bounded_under_churn() {
    let index = ArenaIndex::new();
    let mut rng = fastrand::Rng::with_seed(11);
    let mut live: Vec<u64> = Vec::new();
    // 3,000 live children under the root, with churn.
    for head in 1..=3000u64 {
        apply(&index, make_store_event(head % 8, &[head]));
        live.push(head);
    }
    let root_table = |index: &ArenaIndex| index.runs.lock(ROOT, ROOT_GEN).unwrap().snap().children;
    let mut samples = Vec::new();
    for round in 0..6000u64 {
        let i = rng.usize(..live.len());
        let head = live.swap_remove(i);
        apply(&index, make_remove_event(head % 8, &[head]));
        let fresh = 10_000 + round;
        apply(&index, make_store_event(fresh % 8, &[fresh]));
        live.push(fresh);
        if round % 50 == 0 {
            let table = root_table(&index);
            let (used, slots) = index.runs.child_load(table);
            assert!(used * 4 <= slots * 3, "used {used} of {slots}");
            assert_eq!(index.runs.child_live(table), 3000);
            let trials = 512u64;
            let total: u64 = (0..trials)
                .map(|t| {
                    u64::from(probe_length(
                        &index,
                        table,
                        child_key(0, LocalBlockHash(1 << 40 | t)),
                    ))
                })
                .sum();
            samples.push(total as f64 / trials as f64);
        }
    }
    let shape = index.shape_report();
    assert_eq!(shape.runs_live, 3000);
    // Linear probing at the 3/4 claim limit costs up to (1 + 1/(1/4)^2)/2 = 8.5 slots per
    // miss; a rebuild resets the table to 3/8 live. Over the churn the mean stays below 4.
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let worst = samples.iter().copied().fold(0.0, f64::max);
    eprintln!(
        "CHILD_TABLE mean_miss_probe={mean:.2} worst_sample={worst:.2} rebuilds={}",
        index.stats().table_rebuilds
    );
    assert!(mean < 4.0, "time-averaged miss probe {mean}");
    assert!(worst < 8.5 + 1.0, "worst miss probe {worst}");
    check(&index);
}

/// Slots a miss for `key` visits, as a reader would.
fn probe_length(index: &ArenaIndex, table: super::arena::Addr, key: u64) -> u32 {
    let (_, slots) = index.runs.child_load(table);
    let words = index
        .runs
        .arena
        .slice(table, 2 + 2 * slots as usize)
        .unwrap();
    let mask = slots as usize - 1;
    let mut i = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - slots.trailing_zeros())) as usize;
    let mut probes = 1;
    while words[2 + 2 * i].load(Ordering::Relaxed) != 0 {
        i = (i + 1) & mask;
        probes += 1;
    }
    probes
}

#[test]
fn a_million_block_chain_at_page_size_one() {
    let index = ArenaIndex::new();
    let chain: Vec<u64> = (1..=1_000_000u64).collect();
    apply(&index, store(rank(0), &chain, 0, chain.len()));
    assert_eq!(score(&index, &chain, 0), 1_000_000);
    // Longer stores continue in end children of MAX_RUN_LEN-block runs.
    assert_eq!(
        index.live_run_count(),
        chain.len().div_ceil(MAX_RUN_LEN as usize)
    );
    apply(
        &index,
        store(
            rank(0),
            &[chain.clone(), vec![7, 8, 9]].concat(),
            chain.len(),
            chain.len() + 3,
        ),
    );
    assert_eq!(
        score(&index, &[chain.clone(), vec![7, 8]].concat(), 0),
        1_000_002
    );
    apply(&index, remove(rank(0), &chain, &[999_999]));
    assert_eq!(score(&index, &chain, 0), 999_999);
    apply(&index, remove(rank(0), &chain, &[500_000]));
    assert_eq!(score(&index, &chain, 0), 500_000);
    check(&index);
    apply(&index, make_clear_event(0));
    assert_eq!(index.live_run_count(), 0);
    check(&index);
}

// ----------------------------------------------------------------------------
// B-only hooks
// ----------------------------------------------------------------------------

#[test]
fn generation_wrap_skips_zero_and_stale_reads_fail() {
    let index = ArenaIndex::new();
    let runs = &index.runs;
    let array = runs
        .new_array(&[(LocalBlockHash(1), ExternalSequenceBlockHash(1))])
        .unwrap();
    let (id, _) = runs
        .create(ROOT, 0, LocalBlockHash(1), array, 0, 1)
        .unwrap();
    runs.set_generation(id, u32::MAX);
    let mut frees = Frees::default();
    {
        let locked = runs.lock(id, u32::MAX).unwrap();
        assert!(runs.kill(&locked, &mut frees));
    }
    runs.flush(&mut frees);
    let array = runs
        .new_array(&[(LocalBlockHash(2), ExternalSequenceBlockHash(2))])
        .unwrap();
    let (again, generation) = runs
        .create(ROOT, 0, LocalBlockHash(2), array, 0, 1)
        .unwrap();
    assert_eq!(again, id, "the freed id is reused");
    assert_eq!(generation, 1, "generation 0 is skipped on wrap");
    assert!(matches!(
        runs.read(id, u32::MAX, |_, s| Some(s.len)),
        Read::Gone
    ));
    assert!(matches!(runs.read(id, 1, |_, s| Some(s.len)), Read::Ok(1)));
}

#[test]
fn version_wrap_keeps_validating() {
    let index = ArenaIndex::new();
    apply(&index, make_store_event(0, &[1, 2, 3]));
    let (child, generation) = root_child(&index, 1);
    index.runs.set_version(child, u64::MAX - 1);
    // An append opens and closes a step, wrapping the version through zero.
    apply(&index, make_store_event(1, &[1, 2, 3]));
    apply(&index, make_store_event_with_parent(0, &[1, 2, 3], &[4]));
    apply(&index, make_remove_event_with_parent(0, &[1, 2], &[3]));
    let version = index
        .runs
        .header(child)
        .unwrap()
        .version
        .load(Ordering::Acquire);
    assert!(version < 16, "version wrapped to {version}");
    assert_eq!(
        index
            .runs
            .header(child)
            .unwrap()
            .generation
            .load(Ordering::Relaxed),
        generation
    );
    assert_eq!(score(&index, &[1, 2, 3], 1), 3);
    assert_eq!(score(&index, &[1, 2, 3], 0), 2);
    check(&index);
}

fn root_child(index: &ArenaIndex, head: u64) -> (u32, u32) {
    let Read::Ok(Probe::Found(id, generation)) = index.runs.read(ROOT, ROOT_GEN, |_, snap| {
        Some(
            index
                .runs
                .probe_child(snap.children, child_key(0, LocalBlockHash(head))),
        )
    }) else {
        panic!("root child {head} is missing");
    };
    (id, generation)
}

#[test]
fn a_poisoned_run_reads_as_absent_and_its_writers_fail_visibly() {
    let index = ArenaIndex::new();
    apply(&index, store(rank(0), &[1, 2], 0, 2));
    apply(&index, store(rank(1), &[1, 2], 0, 2));
    apply(&index, store(rank(0), &[1, 2, 3, 4], 2, 4));
    let (first, _) = root_child(&index, 1);
    let child = {
        let Read::Ok(Probe::Found(id, _)) =
            index.runs.read(first, root_child(&index, 1).1, |_, snap| {
                Some(
                    index
                        .runs
                        .probe_child(snap.children, child_key(2, LocalBlockHash(3))),
                )
            })
        else {
            panic!("child missing");
        };
        id
    };
    index.runs.poison(child);
    // The walk stops at the poisoned run: rank 0 keeps credit only for the first run.
    assert_eq!(score(&index, &[1, 2, 3, 4], 0), 2);
    assert!(index.stats().poisoned_seen > 0);
    // Writers that need the run fail with an error instead of hanging or panicking.
    let result = index.apply_event_inline(store(rank(0), &[1, 2, 3, 4, 5], 4, 5));
    assert!(result.is_err());
    // Everything else keeps working.
    apply(&index, store(rank(2), &[1, 9], 0, 2));
    assert_eq!(score(&index, &[1, 9], 2), 2);
}

#[test]
fn forced_retries_fall_back_to_the_lock() {
    let index = ArenaIndex::new();
    apply(&index, make_store_event(0, &[1, 2, 3]));
    let fallbacks = index.stats().reader_fallbacks;
    index.runs.forced_failures.store(200, Ordering::Relaxed);
    assert_eq!(score(&index, &[1, 2, 3], 0), 3);
    assert!(index.stats().reader_fallbacks > fallbacks);
    index.runs.forced_failures.store(0, Ordering::Relaxed);
}

#[test]
fn locked_reader_mode_scores_like_optimistic_reads() {
    let optimistic = ArenaIndex::new();
    let locked = ArenaIndex::with_config(ArenaConfig {
        reader: ReaderMode::Locked,
        ..ArenaConfig::default()
    });
    let mut rng = fastrand::Rng::with_seed(3);
    let chains: Vec<Vec<u64>> = (0..8)
        .map(|c| (0..rng.u64(2..12)).map(|i| (c % 3) * 100 + i).collect())
        .collect();
    for step in 0..400u64 {
        let worker = rank(step % 5);
        let chain = &chains[rng.usize(..chains.len())];
        let event = if rng.bool() {
            store(worker, chain, 0, rng.usize(1..=chain.len()))
        } else {
            remove(worker, chain, &[rng.usize(..chain.len())])
        };
        let a = optimistic.apply_event_inline(event.clone()).is_ok();
        let b = locked.apply_event_inline(event).is_ok();
        assert_eq!(a, b);
    }
    for chain in &chains {
        assert_eq!(
            optimistic.find_matches(&locals(chain), false).scores,
            locked.find_matches(&locals(chain), false).scores
        );
    }
    assert_eq!(optimistic.stats().reader_fallbacks, 0);
    check(&locked);
}

// ----------------------------------------------------------------------------
// The lane pool
// ----------------------------------------------------------------------------

/// Lanes driven directly through `SyncIndexer::worker`, like `ThreadPoolIndexer` but with
/// the test choosing the lane of every task.
struct Lanes {
    index: Arc<ArenaIndex>,
    senders: Vec<flume::Sender<WorkerTask>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl Lanes {
    fn new(index: ArenaIndex, lanes: usize) -> Self {
        let index = Arc::new(index);
        let mut senders = Vec::new();
        let mut threads = Vec::new();
        for _ in 0..lanes {
            let (tx, rx) = flume::unbounded();
            let backend = index.clone();
            threads.push(thread::spawn(move || backend.worker(rx, None).unwrap()));
            senders.push(tx);
        }
        Self {
            index,
            senders,
            threads,
        }
    }

    fn apply(&self, lane: usize, event: RouterEvent) -> bool {
        let (resp, rx) = oneshot::channel();
        self.senders[lane]
            .send(WorkerTask::EventWithAck { event, resp })
            .unwrap();
        rx.blocking_recv().unwrap()
    }

    fn send_ack(&self, lane: usize, event: RouterEvent) -> oneshot::Receiver<bool> {
        let (resp, rx) = oneshot::channel();
        self.senders[lane]
            .send(WorkerTask::EventWithAck { event, resp })
            .unwrap();
        rx
    }

    fn send(&self, lane: usize, event: RouterEvent) {
        self.senders[lane].send(WorkerTask::Event(event)).unwrap();
    }

    fn flush(&self) {
        let receivers: Vec<_> = self
            .senders
            .iter()
            .map(|tx| {
                let (resp, rx) = oneshot::channel();
                tx.send(WorkerTask::Flush(resp)).unwrap();
                rx
            })
            .collect();
        for rx in receivers {
            rx.blocking_recv().unwrap();
        }
    }
}

impl Drop for Lanes {
    fn drop(&mut self) {
        for tx in &self.senders {
            let _ = tx.send(WorkerTask::Terminate);
        }
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }
}

#[test]
fn a_rank_that_moves_lanes_keeps_its_map() {
    // The cross-lane probe: CRTC keeps a rank's lookup per lane and scores 4 here.
    let lanes = Lanes::new(ArenaIndex::new(), 2);
    assert!(lanes.apply(0, make_store_event(1, &[1, 2, 3, 4])));
    lanes.flush();
    assert!(lanes.apply(1, make_remove_event_with_parent(1, &[1, 2], &[3, 4])));
    lanes.flush();
    assert_eq!(score(&lanes.index, &[1, 2, 3, 4], 1), 2);
    check(&lanes.index);
}

#[test]
fn idle_lanes_steal_ranks_from_a_busy_lane() {
    let lanes = Lanes::new(ArenaIndex::new(), 4);
    let chain: Vec<u64> = (1..=24).collect();
    // Every rank is fed through lane 0; lanes 1 to 3 have only stealing to do.
    for round in 0..200u64 {
        for worker in 0..32u64 {
            let mut c = chain.clone();
            c.push(1000 + worker * 1000 + round);
            lanes.send(
                0,
                store(rank(worker), &c, if round == 0 { 0 } else { 24 }, 25),
            );
            if round > 0 {
                let mut previous = chain.clone();
                previous.push(1000 + worker * 1000 + round - 1);
                lanes.send(0, remove(rank(worker), &previous, &[24]));
            }
        }
    }
    lanes.flush();
    let stats = lanes.index.stats();
    assert!(stats.steals > 0, "no lane stole: {stats:?}");
    assert_eq!(stats.claim_check_failures, 0);
    for worker in 0..32u64 {
        let mut c = chain.clone();
        c.push(1000 + worker * 1000 + 199);
        assert_eq!(score(&lanes.index, &c, worker), 25);
    }
    check(&lanes.index);
}

#[test]
fn no_steal_and_no_fast_path_ablations_still_apply_in_order() {
    for config in [
        ArenaConfig {
            steal: false,
            ..ArenaConfig::default()
        },
        ArenaConfig {
            inline_fast_path: false,
            ..ArenaConfig::default()
        },
    ] {
        let lanes = Lanes::new(ArenaIndex::with_config(config), 3);
        let chain: Vec<u64> = (1..=10).collect();
        for worker in 0..9u64 {
            let lane = worker as usize % 3;
            lanes.send(lane, store(rank(worker), &chain, 0, 10));
            lanes.send(lane, remove(rank(worker), &chain, &[9, 8]));
            lanes.send(lane, store(rank(worker), &chain, 8, 9));
        }
        lanes.flush();
        for worker in 0..9u64 {
            assert_eq!(score(&lanes.index, &chain, worker), 9);
        }
        if !config.inline_fast_path {
            assert_eq!(lanes.index.stats().inline_fast_path, 0);
        }
        if !config.steal {
            assert_eq!(lanes.index.stats().steals, 0);
        }
        check(&lanes.index);
    }
}

#[test]
fn second_producer_clears_and_barriers_wait_for_stolen_work() {
    // A `ResetScope::All` clear reaches a rank through a second lane too.
    let lanes = Lanes::new(ArenaIndex::new(), 2);
    for worker in 0..8u64 {
        lanes.send(0, make_store_event(worker, &[1, 2, worker + 10]));
    }
    for worker in 0..8u64 {
        lanes.send(1, make_clear_event(worker));
    }
    lanes.flush();
    for worker in 0..8u64 {
        lanes.send(0, make_store_event(worker, &[1, 2, worker + 10]));
    }
    lanes.flush();
    for worker in 0..8u64 {
        assert_eq!(score(&lanes.index, &[1, 2, worker + 10], worker), 3);
    }
    // Each lane reports the ranks whose cells it created; together they cover every rank.
    let mut blocks = FxHashMap::default();
    for tx in &lanes.senders {
        let (resp, rx) = oneshot::channel();
        tx.send(WorkerTask::Stats(resp)).unwrap();
        for (r, count) in rx.blocking_recv().unwrap().worker_blocks {
            assert!(blocks.insert(r, count).is_none(), "{r:?} reported twice");
        }
    }
    assert_eq!(blocks.len(), 8);
    assert_eq!(blocks.values().sum::<usize>(), 24);
    check(&lanes.index);
}

#[tokio::test]
async fn thread_pool_contracts_hold() {
    // Sibling ranks, whole-worker removal, and cold rank removal through the pool.
    let pool = ThreadPoolIndexer::new(ArenaIndex::new(), 2, 16);
    pool.apply_event(crate::test_utils::make_store_event_with_dp_rank(
        7,
        &[10],
        0,
    ))
    .await;
    pool.apply_event(crate::test_utils::make_store_event_with_dp_rank(
        7,
        &[20],
        1,
    ))
    .await;
    pool.flush().await;
    crate::test_utils::assert_score(&pool, &[10], WorkerWithDpRank::new(7, 0), 1).await;
    crate::test_utils::assert_score(&pool, &[20], WorkerWithDpRank::new(7, 1), 1).await;
    pool.remove_worker(7).await;
    crate::test_utils::assert_no_scores(&pool, &[10]).await;
    crate::test_utils::assert_no_scores(&pool, &[20]).await;
    pool.apply_event(crate::test_utils::make_store_event_with_dp_rank(
        7,
        &[30],
        1,
    ))
    .await;
    pool.flush().await;
    crate::test_utils::assert_score(&pool, &[30], WorkerWithDpRank::new(7, 1), 1).await;

    pool.remove_worker_dp_rank(9, 0).await;
    pool.apply_event(make_store_event(9, &[2])).await;
    pool.flush().await;
    crate::test_utils::assert_score(&pool, &[2], rank(9), 1).await;
    pool.remove_worker_dp_rank(9, 0).await;
    pool.flush().await;
    crate::test_utils::assert_no_scores(&pool, &[2]).await;
    let dump = pool.dump_events().await.unwrap();
    assert_eq!(dump.len(), 1, "only rank (7, 1) holds anything: {dump:?}");
    check(pool.backend());
}

// ----------------------------------------------------------------------------
// Ported CRTC race tests
// ----------------------------------------------------------------------------

/// CRTC's `race_find_during_split_never_overcounts`, with repeated local hashes: writers
/// raise partial holdings past the cap, so the shared run splits under the readers.
#[test]
fn race_find_during_split_never_overcounts_with_repeated_hashes() {
    for seed in 0..8u64 {
        let index = Arc::new(ArenaIndex::new());
        // A chain that repeats one local hash, then diverges per rank.
        let mut base = vec![7u64; 24];
        base.extend([8, 7, 8, 7]);
        let holder = rank(500);
        apply(&index, store(holder, &base, 0, base.len()));
        let writers = 24u64;
        let barrier = Arc::new(Barrier::new(writers as usize + 2));
        let done = Arc::new(AtomicBool::new(false));
        let reader = {
            let (index, barrier, done, base) =
                (index.clone(), barrier.clone(), done.clone(), base.clone());
            thread::spawn(move || {
                barrier.wait();
                while !done.load(Ordering::Acquire) {
                    for w in 0..writers {
                        let mut query = base[..(w as usize % 20) + 4].to_vec();
                        query.extend([1000 + w, 1000 + w]);
                        let scores = index.find_matches(&locals(&query), false);
                        for (r, s) in scores.scores {
                            let bound = if r == holder {
                                (w as usize % 20) + 4
                            } else if r == rank(w) {
                                query.len()
                            } else {
                                (r.worker_id as usize % 20) + 4
                            };
                            assert!(
                                s as usize <= bound,
                                "{r:?} scored {s} > {bound} on {query:?}"
                            );
                        }
                    }
                }
            })
        };
        let threads: Vec<_> = (0..writers)
            .map(|w| {
                let (index, barrier, base) = (index.clone(), barrier.clone(), base.clone());
                thread::spawn(move || {
                    let cut = (w as usize % 20) + 4;
                    barrier.wait();
                    if (w + seed) % 2 == 0 {
                        index
                            .apply_event_inline(store(rank(w), &base, 0, cut))
                            .unwrap();
                    } else {
                        index
                            .apply_event_inline(store(rank(w), &base, 0, 1))
                            .unwrap();
                        index
                            .apply_event_inline(store(rank(w), &base, 1, cut))
                            .unwrap();
                    }
                    let mut chain = base[..cut].to_vec();
                    chain.extend([1000 + w, 1000 + w]);
                    index
                        .apply_event_inline(store(rank(w), &chain, cut, chain.len()))
                        .unwrap();
                })
            })
            .collect();
        barrier.wait();
        for t in threads {
            t.join().unwrap();
        }
        done.store(true, Ordering::Release);
        reader.join().unwrap();
        assert!(index.stats().splits_prefix_cap > 0);
        for w in 0..writers {
            let cut = (w as usize % 20) + 4;
            let mut chain = base[..cut].to_vec();
            chain.extend([1000 + w, 1000 + w]);
            assert_eq!(score_of(&index, &chain, rank(w)) as usize, chain.len());
        }
        check(&index);
    }
}

/// CRTC's `whole_slot_drops_racing_readers_never_overcount`: whole holders drop their
/// bits, are demoted, and whole runs die while readers walk through them. Checked against
/// the interval-bound oracle: a score may only credit blocks the rank held at some instant
/// of the lookup.
#[test]
fn whole_slot_drops_racing_readers_never_overcount() {
    let index = Arc::new(ArenaIndex::new());
    let chain: Vec<u64> = (1..=12).collect();
    let hashes = Arc::new(seqs(&chain));
    let ranks = 16usize;
    let oracle = Arc::new(Oracle::new(ranks));
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let (index, done, oracle, chain, hashes) = (
                index.clone(),
                done.clone(),
                oracle.clone(),
                chain.clone(),
                hashes.clone(),
            );
            thread::spawn(move || {
                let all: Vec<usize> = (0..ranks).collect();
                while !done.load(Ordering::Acquire) {
                    let start = oracle.begin(&all, &hashes);
                    let scores = index.find_matches(&locals(&chain), false).scores;
                    oracle.finish(&all, start, &hashes, &scores, &chain);
                }
            })
        })
        .collect();
    let writers: Vec<_> = (0..ranks)
        .map(|w| {
            let (index, oracle, chain, hashes) =
                (index.clone(), oracle.clone(), chain.clone(), hashes.clone());
            thread::spawn(move || {
                let mut rng = fastrand::Rng::with_seed(w as u64);
                let r = rank(w as u64);
                let mut held = 0usize;
                for _ in 0..400 {
                    let to = rng.usize(1..=chain.len());
                    if to > held {
                        let _ = oracle.add(w, &hashes[held..to]);
                        index
                            .apply_event_inline(store(r, &chain, held, to))
                            .unwrap();
                        held = to;
                    }
                    // Dynamo semantics: the removal drops the named block only; the rank
                    // keeps holding the blocks after it, just not reachably.
                    let cut = rng.usize(..held);
                    index.apply_event_inline(remove(r, &chain, &[cut])).unwrap();
                    oracle.remove(w, &hashes[cut..=cut]);
                    held = cut;
                    if rng.u32(..8) == 0 {
                        index
                            .apply_event_inline(make_clear_event(w as u64))
                            .unwrap();
                        oracle.remove(w, &hashes);
                        held = 0;
                    }
                }
                index
                    .apply_event_inline(make_clear_event(w as u64))
                    .unwrap();
                oracle.remove(w, &hashes);
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    done.store(true, Ordering::Release);
    for r in readers {
        r.join().unwrap();
    }
    assert_eq!(oracle.violations.load(Ordering::Relaxed), 0);
    assert!(oracle.checked.load(Ordering::Relaxed) > 0);
    assert!(index.find_matches(&locals(&chain), false).scores.is_empty());
    check(&index);
    assert_eq!(index.live_run_count(), 0);
}

/// CRTC's `random_strict_streams_with_slot_recycling_match_the_model`: engine-valid
/// streams with leaf-first eviction, clears, and rank removals that recycle slots must
/// score exactly as a set model, after every step.
#[test]
fn random_strict_streams_with_slot_recycling_match_the_model() {
    for seed in 0..12u64 {
        let index = ArenaIndex::new();
        let mut rng = fastrand::Rng::with_seed(seed);
        let pool: Vec<Vec<u64>> = (0..10)
            .map(|c| {
                let mut chain = vec![1, 2 + c % 2];
                chain.extend((0..rng.u64(1..10)).map(|i| 10 * c + i + 100));
                chain
            })
            .collect();
        let mut model: FxHashMap<WorkerWithDpRank, FxHashSet<u64>> = FxHashMap::default();
        let mut next_worker = 6u64;
        let mut workers: Vec<u64> = (0..6).collect();
        for _ in 0..1500 {
            let w = workers[rng.usize(..workers.len())];
            let r = rank(w);
            let held = model.entry(r).or_default();
            let chain = &pool[rng.usize(..pool.len())];
            let hashes = seqs(chain);
            let prefix = hashes.iter().take_while(|h| held.contains(h)).count();
            match rng.u32(..100) {
                0..55 if prefix < chain.len() => {
                    let to = rng.usize(prefix + 1..=chain.len());
                    apply(&index, store(r, chain, prefix, to));
                    held.extend(&hashes[prefix..to]);
                }
                55..90 if prefix > 0 => {
                    // Leaf-first: only a block whose descendants this rank does not hold.
                    let last = prefix - 1;
                    let has_child = pool.iter().any(|other| {
                        let other_hashes = seqs(other);
                        other_hashes.len() > prefix
                            && other_hashes[last] == hashes[last]
                            && held.contains(&other_hashes[prefix])
                    });
                    if !has_child {
                        apply(&index, remove(r, chain, &[last]));
                        held.remove(&hashes[last]);
                    }
                }
                90..95 => {
                    apply(&index, make_clear_event(w));
                    held.clear();
                }
                95..100 => {
                    index.remove_rank_inline(r);
                    model.remove(&r);
                    workers.retain(|&x| x != w);
                    workers.push(next_worker);
                    next_worker += 1;
                }
                _ => {}
            }
            for chain in &pool {
                let hashes = seqs(chain);
                let scores = index.find_matches(&locals(chain), false).scores;
                for (r, held) in &model {
                    let expected = hashes.iter().take_while(|h| held.contains(h)).count() as u32;
                    assert_eq!(
                        scores.get(r).copied().unwrap_or(0),
                        expected,
                        "seed {seed}: {r:?} on {chain:?}"
                    );
                }
                assert!(scores.keys().all(|r| model.contains_key(r)));
            }
        }
        check(&index);
    }
}

// ----------------------------------------------------------------------------
// Interval-bound soak
// ----------------------------------------------------------------------------

/// A rank's held blocks as its writer publishes them: adds are logged before the event
/// is applied, removals after it is acknowledged, so the union of what was held at a
/// lookup's start and what was added during it bounds every score the lookup may give
/// (the interval-bound oracle of spec B 11).
#[derive(Default)]
struct RankLog {
    held: FxHashSet<u64>,
    epoch: u64,
    added: Vec<(u64, u64)>,
    /// Epochs below this were trimmed from `added`.
    trimmed: u64,
}

struct Oracle {
    logs: Vec<Mutex<RankLog>>,
    violations: AtomicU64,
    checked: AtomicU64,
}

impl Oracle {
    fn new(ranks: usize) -> Self {
        Self {
            logs: (0..ranks).map(|_| Mutex::new(RankLog::default())).collect(),
            violations: AtomicU64::new(0),
            checked: AtomicU64::new(0),
        }
    }

    fn holds_prefix(&self, rank: usize, hashes: &[u64]) -> usize {
        let log = self.logs[rank].lock();
        hashes.iter().take_while(|h| log.held.contains(h)).count()
    }

    /// Logs `hashes` as added before the store is sent. Returns the ones that were not
    /// held yet, which are all a failed store may take back.
    fn add(&self, rank: usize, hashes: &[u64]) -> Vec<u64> {
        let mut log = self.logs[rank].lock();
        log.epoch += 1;
        let epoch = log.epoch;
        log.added.extend(hashes.iter().map(|&h| (epoch, h)));
        let fresh: Vec<u64> = hashes
            .iter()
            .copied()
            .filter(|h| !log.held.contains(h))
            .collect();
        log.held.extend(hashes);
        if log.added.len() > 200_000 {
            let cut = log.added.len() / 2;
            log.trimmed = log.added[cut].0;
            log.added.drain(..cut);
        }
        fresh
    }

    /// After a failed store: the blocks it would have added were never held.
    fn unadd(&self, rank: usize, hashes: &[u64]) {
        let mut log = self.logs[rank].lock();
        for h in hashes {
            log.held.remove(h);
        }
    }

    fn remove(&self, rank: usize, hashes: &[u64]) {
        let mut log = self.logs[rank].lock();
        for h in hashes {
            log.held.remove(h);
        }
    }

    fn begin(&self, ranks: &[usize], hashes: &[u64]) -> Vec<(Vec<bool>, u64)> {
        ranks
            .iter()
            .map(|&i| {
                let log = self.logs[i].lock();
                (
                    hashes.iter().map(|h| log.held.contains(h)).collect(),
                    log.epoch,
                )
            })
            .collect()
    }

    fn finish(
        &self,
        ranks: &[usize],
        start: Vec<(Vec<bool>, u64)>,
        hashes: &[u64],
        scores: &FxHashMap<WorkerWithDpRank, u32>,
        query: &[u64],
    ) {
        for (&i, (mut held, epoch)) in ranks.iter().zip(start) {
            let log = self.logs[i].lock();
            if epoch < log.trimmed {
                continue;
            }
            for &(_, hash) in log.added.iter().rev().take_while(|(e, _)| *e > epoch) {
                for (p, h) in hashes.iter().enumerate() {
                    if *h == hash {
                        held[p] = true;
                    }
                }
            }
            let bound = held.iter().take_while(|&&h| h).count() as u32;
            let got = scores
                .get(&WorkerWithDpRank::new(i as u64, 0))
                .copied()
                .unwrap_or(0);
            self.checked.fetch_add(1, Ordering::Relaxed);
            if got > bound {
                self.violations.fetch_add(1, Ordering::Relaxed);
                eprintln!("interval bound: rank {i} scored {got} > {bound} on {query:?}");
            }
        }
    }
}

struct SoakConfig {
    lanes: usize,
    ranks_per_lane: usize,
    steps: usize,
    readers: usize,
}

/// A chain for the soak: a system prompt, then either a prefix of one long shared document
/// (many distinct partial holders, so runs split) or a short variant, then a decode tail.
fn make_chain(rng: &mut fastrand::Rng) -> Vec<u64> {
    let system = rng.u64(..3);
    let mut chain: Vec<u64> = (0..4).map(|i| system * 10 + i).collect();
    if rng.bool() {
        chain.extend((0..rng.u64(1..48)).map(|i| 9000 + i));
    } else {
        let variant = rng.u64(..3);
        chain.extend((0..rng.u64(1..6)).map(|i| 500 + system * 50 + variant * 10 + i));
    }
    let tail = rng.u64(..4);
    chain.extend((0..rng.u64(0..3)).map(|i| 7000 + tail * 10 + i));
    chain
}

fn interval_bound_soak(config: SoakConfig) {
    let lanes = Arc::new(Lanes::new(ArenaIndex::new(), config.lanes));
    let ranks = config.lanes * config.ranks_per_lane;
    let oracle = Arc::new(Oracle::new(ranks));
    let done = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..config.readers)
        .map(|seed| {
            let (lanes, oracle, done) = (lanes.clone(), oracle.clone(), done.clone());
            thread::spawn(move || {
                let mut rng = fastrand::Rng::with_seed(1000 + seed as u64);
                while !done.load(Ordering::Acquire) {
                    let chain = make_chain(&mut rng);
                    let hashes = seqs(&chain);
                    let probe: Vec<usize> = (0..4).map(|_| rng.usize(..ranks)).collect();
                    let start = oracle.begin(&probe, &hashes);
                    let scores = lanes.index.find_matches(&locals(&chain), false).scores;
                    oracle.finish(&probe, start, &hashes, &scores, &chain);
                }
            })
        })
        .collect();
    let writers: Vec<_> = (0..config.lanes)
        .map(|lane| {
            let (lanes, oracle) = (lanes.clone(), oracle.clone());
            let (steps, per) = (config.steps, config.ranks_per_lane);
            thread::spawn(move || {
                let mut rng = fastrand::Rng::with_seed(lane as u64);
                let mut live: Vec<Vec<Vec<u64>>> = vec![Vec::new(); per];
                enum Post {
                    Store {
                        local: usize,
                        chain: Vec<u64>,
                        fresh: Vec<u64>,
                    },
                    Remove(Vec<u64>),
                    Clear,
                }
                // Each round sends one event per rank of this lane without waiting, so
                // mailboxes back up and idle lanes steal, then collects the acks.
                for _ in 0..steps / per {
                    let mut pending = Vec::with_capacity(per);
                    for local in 0..per {
                        let id = lane * per + local;
                        let r = WorkerWithDpRank::new(id as u64, 0);
                        let roll = rng.u32(..100);
                        if roll < 45 || live[local].is_empty() {
                            let chain = make_chain(&mut rng);
                            let hashes = seqs(&chain);
                            let prefix = oracle.holds_prefix(id, &hashes);
                            if prefix == chain.len() {
                                continue;
                            }
                            let to = rng.usize(prefix + 1..=chain.len());
                            let fresh = oracle.add(id, &hashes[prefix..to]);
                            let rx = lanes.send_ack(lane, store(r, &chain, prefix, to));
                            pending.push((
                                id,
                                rx,
                                Post::Store {
                                    local,
                                    chain: chain[..to].to_vec(),
                                    fresh,
                                },
                            ));
                        } else if roll < 95 {
                            let chain = live[local][rng.usize(..live[local].len())].clone();
                            let hashes = seqs(&chain);
                            let held: Vec<usize> = {
                                let log = oracle.logs[id].lock();
                                (0..hashes.len())
                                    .filter(|&p| log.held.contains(&hashes[p]))
                                    .collect()
                            };
                            if held.is_empty() {
                                continue;
                            }
                            // Tail runs mostly, single mid-chain holes sometimes.
                            let positions: Vec<usize> = if rng.u32(..4) == 0 {
                                vec![held[rng.usize(..held.len())]]
                            } else {
                                held.iter().rev().take(rng.usize(1..=3)).copied().collect()
                            };
                            let rx = lanes.send_ack(lane, remove(r, &chain, &positions));
                            let removed = positions.iter().map(|&p| hashes[p]).collect();
                            pending.push((id, rx, Post::Remove(removed)));
                        } else {
                            let rx = lanes.send_ack(lane, make_clear_event(id as u64));
                            pending.push((id, rx, Post::Clear));
                        }
                    }
                    for (id, rx, post) in pending {
                        let ok = rx.blocking_recv().unwrap();
                        match post {
                            Post::Store {
                                local,
                                chain,
                                fresh,
                            } => {
                                if ok {
                                    live[local].push(chain);
                                    if live[local].len() > 12 {
                                        live[local].remove(0);
                                    }
                                } else {
                                    oracle.unadd(id, &fresh);
                                }
                            }
                            Post::Remove(removed) => oracle.remove(id, &removed),
                            Post::Clear => {
                                let all: Vec<u64> =
                                    oracle.logs[id].lock().held.iter().copied().collect();
                                oracle.remove(id, &all);
                            }
                        }
                    }
                }
            })
        })
        .collect();
    for w in writers {
        w.join().unwrap();
    }
    done.store(true, Ordering::Release);
    for r in readers {
        r.join().unwrap();
    }
    lanes.flush();
    let stats = lanes.index.stats();
    eprintln!(
        "INTERVAL lanes={} ranks={} steps_per_lane={} checked={} violations={} splits={} steals={} \
         inline={} unlinks={} restarts={} store_failures={} reader_retries={} fallbacks={}",
        config.lanes,
        ranks,
        config.steps,
        oracle.checked.load(Ordering::Relaxed),
        oracle.violations.load(Ordering::Relaxed),
        stats.splits_prefix_cap,
        stats.steals,
        stats.inline_fast_path,
        stats.unlinks,
        stats.store_restarts,
        stats.store_failures,
        stats.reader_retries,
        stats.reader_fallbacks,
    );
    assert_eq!(oracle.violations.load(Ordering::Relaxed), 0);
    assert!(stats.splits_prefix_cap > 0, "the soak must split runs");
    check(&lanes.index);
}

#[test]
fn interval_bound_soak_short() {
    interval_bound_soak(SoakConfig {
        lanes: 4,
        ranks_per_lane: 8,
        steps: 3_000,
        readers: 2,
    });
}

/// Spec B 14: 16 lanes, 100k steps per lane. `ARENA_SOAK_STEPS` overrides the steps.
#[test]
#[ignore = "long-running soak; run explicitly"]
fn interval_bound_soak_16_lanes() {
    interval_bound_soak(SoakConfig {
        lanes: 16,
        ranks_per_lane: 4,
        steps: crate::indexer::harness::env_u64("ARENA_SOAK_STEPS", 100_000) as usize,
        readers: 4,
    });
}

#[test]
fn removal_hash_types_line_up() {
    // Guard for the helpers above: a removal names the stored sequence hashes.
    let event = remove(rank(0), &[1, 2, 3], &[2]);
    let KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }) = event.event.data else {
        panic!("not a removal");
    };
    assert_eq!(
        block_hashes,
        vec![ExternalSequenceBlockHash(seqs(&[1, 2, 3])[2])]
    );
}

/// The bench's observation protocol: every observed event is recorded in the buffer of
/// the queue it was sent to, even when another lane applied it, and seals are barriers.
#[cfg(feature = "bench")]
#[tokio::test]
async fn observed_events_land_in_their_queue_buffers() {
    use crate::indexer::ThreadPoolObservationPlan;

    let indexer = ThreadPoolIndexer::new(ArenaIndex::new(), 2, 16);
    let epoch = std::time::Instant::now();
    let mut observation = indexer
        .begin_observation(ThreadPoolObservationPlan {
            epoch,
            expected_events_by_worker: vec![(1, 40), (2, 40)],
        })
        .await
        .unwrap();
    let mut queue_of = FxHashMap::default();
    for i in 0..40u64 {
        for worker in [1u64, 2] {
            let id = (worker * 100 + i) as u32;
            let receipt = observation
                .enqueue_observed_owned(make_store_event(worker, &[worker, 10 + i]), id)
                .unwrap();
            queue_of.insert(id, receipt.event_worker);
        }
    }
    let sealed = observation.close_observed_producers().seal().await.unwrap();
    let snapshot = sealed.harvest().await.unwrap();
    assert_eq!(snapshot.buffers.len(), 2);
    let mut seen = 0;
    for (queue, buffer) in snapshot.buffers.iter().enumerate() {
        assert!(!buffer.overflowed());
        for record in buffer.records() {
            assert_eq!(queue_of[&record.correlation_id], queue);
            assert!(record.success);
            seen += 1;
        }
    }
    assert_eq!(seen, 80);
    check(indexer.backend());
}
