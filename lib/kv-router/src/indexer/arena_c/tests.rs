// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! White-box tests: one per invariant of spec C (S1 to S10, C1 to C3 where a unit test
//! can see them), the C-only hooks, the ROOT table retirement, and CRTC's race tests
//! ported to this backend. The shared harness (`harness::arena_c`) covers I1, I2, I4
//! and the soak; the loom models live in `lib/kv-router/loom-arena-c`.

use std::sync::Barrier;
use std::sync::atomic::AtomicBool;
use std::thread;

use rustc_hash::FxHashSet;
use tokio::sync::oneshot;

use super::run::{PARTIAL_CAP, SEALED};
use super::store::Resolved;
use super::table::{child_key, claim_budget};
use super::types::{BlockPos, WorkerRemovalTarget};
use super::*;
use crate::indexer::{SyncIndexer, WorkerTask};
use crate::test_utils::{remove_event, router_event, stored_blocks_with_sequence_hashes};

fn rank(id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(id, 0)
}

fn locals(path: &[u64]) -> Vec<LocalBlockHash> {
    path.iter().copied().map(LocalBlockHash).collect()
}

fn seqs(path: &[u64]) -> Vec<u64> {
    compute_seq_hash_for_block(&locals(path))
}

fn ext(path: &[u64], position: usize) -> ExternalSequenceBlockHash {
    ExternalSequenceBlockHash(seqs(path)[position])
}

/// An index with inline lanes: events apply on the calling thread, rank `r` on lane
/// `r.worker_id % lanes`.
struct Rig {
    index: ArenaIndexC,
    lanes: Vec<CLane>,
    next_id: u64,
}

impl Rig {
    fn new(lanes: usize) -> Self {
        let index = ArenaIndexC::new();
        let lanes = (0..lanes).map(|_| index.new_lane()).collect();
        Self {
            index,
            lanes,
            next_id: 0,
        }
    }

    /// The index and `r`'s lane.
    fn split(&mut self, r: WorkerWithDpRank) -> (&ArenaIndexC, &mut CLane) {
        let n = self.lanes.len();
        (&self.index, &mut self.lanes[r.worker_id as usize % n])
    }

    fn event(
        &mut self,
        r: WorkerWithDpRank,
        data: KvCacheEventData,
    ) -> Result<(), KvCacheEventError> {
        self.next_id += 1;
        let event = router_event(r.worker_id, self.next_id, r.dp_rank, data);
        let (index, lane) = self.split(r);
        index.apply_event(lane, event, None)
    }

    fn flush_lanes(&mut self) {
        for lane in &mut self.lanes {
            self.index.flush_frees(lane);
        }
    }

    /// Stores `path[from..to]` under `path[from - 1]`.
    fn store(
        &mut self,
        r: WorkerWithDpRank,
        path: &[u64],
        from: usize,
        to: usize,
    ) -> Result<(), KvCacheEventError> {
        let hashes = seqs(&path[..to]);
        let data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: from
                .checked_sub(1)
                .map(|i| ExternalSequenceBlockHash(hashes[i])),
            start_position: None,
            blocks: stored_blocks_with_sequence_hashes(&locals(&path[from..to]), &hashes[from..to]),
        });
        self.event(r, data)
    }

    fn store_all(&mut self, r: WorkerWithDpRank, path: &[u64]) {
        self.store(r, path, 0, path.len()).unwrap();
    }

    fn remove(&mut self, r: WorkerWithDpRank, path: &[u64], positions: &[usize]) {
        let hashes = seqs(path);
        let data = KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: positions
                .iter()
                .map(|&p| ExternalSequenceBlockHash(hashes[p]))
                .collect(),
        });
        self.event(r, data).unwrap();
    }

    fn clear(&mut self, r: WorkerWithDpRank) {
        self.event(r, KvCacheEventData::Cleared).unwrap();
    }

    fn remove_rank(&mut self, r: WorkerWithDpRank) {
        let (index, lane) = self.split(r);
        index.remove_ranks(lane, WorkerRemovalTarget::DpRank(r), true);
    }

    fn score(&self, r: WorkerWithDpRank, path: &[u64]) -> u32 {
        self.index
            .find_matches_impl(&locals(path), false)
            .scores
            .get(&r)
            .copied()
            .unwrap_or(0)
    }

    fn slot(&self, r: WorkerWithDpRank) -> Slot {
        self.index
            .slots
            .table(&crossbeam_epoch::pin())
            .slot_of(r)
            .expect("rank has a slot")
    }

    /// The run holding position `position` of `path` for `r`, via `r`'s entry.
    fn pos(&mut self, r: WorkerWithDpRank, path: &[u64], position: usize) -> BlockPos {
        let key = ext(path, position);
        let (index, lane) = self.split(r);
        let lookup = lane.ranks.get_mut(&r).expect("rank has a lookup");
        index
            .resolve_entry(lookup, key, None)
            .expect("entry resolves")
            .0
    }

    fn check(&mut self) {
        {
            let mut lanes: Vec<&mut CLane> = self.lanes.iter_mut().collect();
            self.index.probe_quiesce(&mut lanes);
        }
        let lanes: Vec<&CLane> = self.lanes.iter().collect();
        if let Err(violation) = self.index.probe_check(&lanes) {
            panic!("{violation}");
        }
    }

    fn shape(&self) -> probe::ShapeReport {
        self.index.shape_report()
    }

    fn quiesce(&mut self) {
        let mut lanes: Vec<&mut CLane> = self.lanes.iter_mut().collect();
        self.index.probe_quiesce(&mut lanes);
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.flush_lanes();
    }
}

// ----------------------------------------------------------------------------
// Storage layout: S1, S2, S7, children at any offset, appends
// ----------------------------------------------------------------------------

#[test]
fn divergence_hangs_a_child_at_its_offset_without_splitting() {
    let mut rig = Rig::new(1);
    let a = [1, 2, 3, 4, 5, 6];
    let b = [1, 2, 3, 9];
    rig.store_all(rank(1), &a);
    rig.store_all(rank(2), &b);

    let shape = rig.shape();
    assert_eq!(shape.runs_live, 2);
    assert_eq!(shape.splits_prefix_cap, 0);
    // The child hangs at offset 3 of the first run, keyed by (3, 9).
    let parent = rig.pos(rank(1), &a, 0).run();
    let child = rig.pos(rank(2), &b, 3);
    assert_eq!(child.offset, 0);
    assert_eq!(
        rig.index
            .store
            .find_child(rig.index.store.run(parent), child_key(3, 9)),
        Some(child.run())
    );

    assert_eq!(rig.score(rank(1), &a), 6);
    assert_eq!(rig.score(rank(2), &a), 3);
    assert_eq!(rig.score(rank(1), &b), 3);
    assert_eq!(rig.score(rank(2), &b), 4);
    let details = rig.index.find_match_details_impl(&locals(&b), false, true);
    assert_eq!(details.last_matched_hashes[&rank(1)], ext(&b, 2));
    assert_eq!(details.last_matched_hashes[&rank(2)], ext(&b, 3));
    rig.check();
}

#[test]
fn partial_holder_continues_into_the_child_at_its_divergence() {
    let mut rig = Rig::new(1);
    let a = [1, 2, 3, 4, 5, 6];
    let b = [1, 2, 3, 4, 9];
    rig.store_all(rank(1), &a);
    // Rank 1 evicts block 5: it keeps a cutoff of 4 on the run.
    rig.remove(rank(1), &a, &[4]);
    rig.store_all(rank(2), &b);
    // Rank 1 stores block 9 under block 4: it descends into rank 2's child.
    rig.store(rank(1), &b, 4, 5).unwrap();
    assert_eq!(rig.shape().runs_live, 2);
    assert_eq!(rig.score(rank(1), &b), 5);
    assert_eq!(rig.score(rank(2), &b), 5);
    assert_eq!(rig.score(rank(1), &a), 4);
    rig.check();
}

#[test]
fn decode_appends_in_place_then_by_reallocation() {
    let mut rig = Rig::new(1);
    let mut path = vec![1, 2, 3];
    rig.store_all(rank(1), &path);
    let run_id = rig.pos(rank(1), &path, 0).run();
    let first_array = rig.index.store.run(run_id).array.load(Ordering::Relaxed);
    let (_, capacity) = rig.index.store.array_extent(first_array);
    let mut grew = false;
    for next in 4..40u64 {
        path.push(next);
        rig.store(rank(1), &path, path.len() - 1, path.len())
            .unwrap();
        let run = rig.index.store.run(run_id);
        assert_eq!(run.len() as usize, path.len(), "decode stays in one run");
        let array = run.array.load(Ordering::Relaxed);
        if path.len() <= capacity as usize {
            assert_eq!(
                array, first_array,
                "appends in place while the array has room"
            );
        } else if array != first_array {
            grew = true;
        }
    }
    assert!(grew);
    assert_eq!(rig.shape().runs_live, 1);
    assert_eq!(rig.score(rank(1), &path), path.len() as u32);
    rig.check();
}

#[test]
fn shared_tail_extension_hangs_an_end_child_instead_of_appending() {
    let mut rig = Rig::new(2);
    let base = [1, 2, 3];
    rig.store_all(rank(1), &base);
    rig.store_all(rank(2), &base);
    let extended = [1, 2, 3, 4];
    rig.store(rank(1), &extended, 3, 4).unwrap();
    // Two whole holders: no append, an end child at offset 3.
    let root_run = rig.pos(rank(1), &base, 0).run();
    assert_eq!(rig.index.store.run(root_run).len(), 3);
    rig.store(rank(2), &extended, 3, 4).unwrap();
    assert_eq!(rig.shape().runs_live, 2);
    assert_eq!(rig.score(rank(1), &extended), 4);
    assert_eq!(rig.score(rank(2), &extended), 4);
    rig.check();
}

#[test]
fn append_reprobes_an_end_child_a_claim_linked_after_the_plan() {
    let mut rig = Rig::new(3);
    let a = [1, 2, 3];
    rig.store_all(rank(1), &a);
    // Give the run a child table, so a later claim does not bump its version.
    rig.store_all(rank(3), &[1, 7]);
    let run_id = rig.pos(rank(1), &a, 0).run();
    let version = rig.index.store.run(run_id).version.load(Ordering::Relaxed);

    // Rank 2 becomes whole (a promotion) and claims the end child [4] under the shared
    // gate, then drops back to a partial holder: none of this bumps the version.
    let b = [1, 2, 3, 4];
    rig.store_all(rank(2), &b);
    rig.remove(rank(2), &b, &[2]);
    let run = rig.index.store.run(run_id);
    assert_eq!(run.version.load(Ordering::Relaxed), version);
    assert!(rig.index.store.whole_sole(run, rig.slot(rank(1))));

    // Rank 1's append, planned before the claim, must descend into the end child.
    let blocks = stored_blocks_with_sequence_hashes(&locals(&[4]), &seqs(&b)[3..4]);
    let slot = rig.slot(rank(1));
    let (index, lane) = rig.split(rank(1));
    let descended = index
        .probe_append(run_id, version, slot, &blocks, lane)
        .unwrap();
    let end_child = rig.pos(rank(2), &b, 3).run();
    assert_eq!(descended, Some(end_child));
    assert_eq!(
        rig.index.store.run(run_id).len(),
        3,
        "the append was abandoned"
    );
    rig.check();
}

#[test]
fn appends_continue_past_children_unlike_crtc_sticky_internal() {
    let mut rig = Rig::new(2);
    let a = [1, 2, 3];
    rig.store_all(rank(1), &a);
    rig.store_all(rank(2), &[1, 2, 8]);
    let run_id = rig.pos(rank(1), &a, 0).run();
    rig.store(rank(1), &[1, 2, 3, 4], 3, 4).unwrap();
    assert_eq!(rig.index.store.run(run_id).len(), 4);
    assert_eq!(rig.score(rank(1), &[1, 2, 3, 4]), 4);
    assert_eq!(rig.score(rank(2), &[1, 2, 8]), 3);
    rig.check();
}

#[test]
fn a_million_block_chain_continues_in_end_children() {
    let mut rig = Rig::new(1);
    let path: Vec<u64> = (1..=1_000_000).collect();
    rig.store_all(rank(1), &path);
    let shape = rig.shape();
    assert_eq!(
        shape.runs_live,
        1_000_000u64.div_ceil(run::MAX_RUN_LEN as u64)
    );
    assert_eq!(rig.score(rank(1), &path), 1_000_000);
    rig.remove(rank(1), &path, &[999_999]);
    assert_eq!(rig.score(rank(1), &path), 999_999);
    rig.remove(rank(1), &path, &[500_000]);
    assert_eq!(rig.score(rank(1), &path), 500_000);
    rig.check();
}

// ----------------------------------------------------------------------------
// S3, S8: prefix-cap splits and forwarding
// ----------------------------------------------------------------------------

/// One more partial holder than a run may carry.
const CROWD: u64 = PARTIAL_CAP as u64 + 1;

/// Rank 0 holds `path` whole; ranks 1..=n hold prefixes of distinct lengths 2..n+2.
fn crowd(rig: &mut Rig, path: &[u64], n: u64) {
    rig.store_all(rank(0), path);
    for r in 1..=n {
        rig.store(rank(r), path, 0, r as usize + 1).unwrap();
    }
}

#[test]
fn partial_holders_past_the_cap_split_at_the_median() {
    let mut rig = Rig::new(4);
    let path: Vec<u64> = (1..=40).collect();
    // Children past and before the future split point.
    rig.store_all(rank(100), &[&path[..30], &[88][..]].concat());
    rig.store_all(rank(101), &[&path[..4], &[77][..]].concat());
    let run_id = rig.pos(rank(100), &path, 0).run();
    let version = rig.index.store.run(run_id).version.load(Ordering::Relaxed);
    crowd(&mut rig, &path, CROWD);

    let shape = rig.shape();
    assert!(shape.splits_prefix_cap >= 1, "{shape}");
    let run = rig.index.store.run(run_id);
    assert!(run.flag(SEALED));
    assert!(
        run.version.load(Ordering::Relaxed) > version,
        "S10: versions only grow"
    );
    let forwards = rig.index.store.forwards(run);
    assert!(!forwards.is_empty());
    assert_eq!(forwards.last().unwrap().at, run.len());
    assert!(rig.index.store.cutoff_count(run) <= PARTIAL_CAP);

    for r in 1..=CROWD {
        assert_eq!(rig.score(rank(r), &path), r as u32 + 1);
    }
    assert_eq!(rig.score(rank(0), &path), 40);
    assert_eq!(rig.score(rank(100), &[&path[..30], &[88][..]].concat()), 31);
    assert_eq!(rig.score(rank(101), &[&path[..4], &[77][..]].concat()), 5);
    rig.check();
}

#[test]
fn moved_entries_resolve_through_forwards_and_compress_on_use() {
    let mut rig = Rig::new(1);
    let path: Vec<u64> = (1..=40).collect();
    rig.store_all(rank(0), &path);
    let run_id = rig.pos(rank(0), &path, 0).run();
    rig.index.probe_skip_path_compression(true);
    crowd(&mut rig, &path, CROWD);
    let at = rig.index.store.run(run_id).len();
    assert!(at < 40);

    // Rank 0's entry for position 39 still names the split prefix.
    let key = ext(&path, 39);
    let stored = rig.lanes[0].ranks[&rank(0)].map.get(key).unwrap();
    assert_eq!(stored, BlockPos::new(run_id, 39));
    let Resolved::At { pos, held } = rig.index.resolve(key, stored, Some(rig.slot(rank(0)))) else {
        panic!("a moved entry resolves through the forwarding record");
    };
    assert_ne!(pos.run(), run_id);
    assert_eq!(pos.offset, 39 - at);
    assert!(held > pos.offset);
    assert!(rig.shape().max_forward_hops >= 1);

    // Without compression the entry stays; with it, the first use rewrites it.
    rig.pos(rank(0), &path, 39);
    assert_eq!(rig.lanes[0].ranks[&rank(0)].map.get(key), Some(stored));
    rig.index.probe_skip_path_compression(false);
    rig.pos(rank(0), &path, 39);
    assert_eq!(rig.lanes[0].ranks[&rank(0)].map.get(key), Some(pos));
    rig.check();
}

/// A store that has placed blocks up to `(run, 3)` continues after `run` is split at 3,
/// the suffix dies, and `run` is split again at 2: position 3 now has no live run, so the
/// store follows its last held position (2) into the second suffix and continues there.
#[test]
fn a_store_mid_placement_follows_its_last_held_position_through_two_splits() {
    let mut rig = Rig::new(1);
    let long = [1, 2, 3, 4, 5, 6];
    rig.store_all(rank(1), &long);
    rig.store(rank(2), &long, 0, 3).unwrap();
    let run_id = rig.pos(rank(1), &long, 0).run();
    {
        let (index, lane) = rig.split(rank(1));
        index.probe_split(run_id, 3, lane);
    }
    // The first suffix [4, 5, 6] loses its only holder and unlinks.
    rig.remove(rank(1), &long, &[3]);
    assert_eq!(rig.shape().runs_live, 1);
    {
        let (index, lane) = rig.split(rank(1));
        index.probe_split(run_id, 2, lane);
    }
    // Rank 2 continues with block 9 at position 3 of the original run.
    let branch = [1, 2, 3, 9];
    let blocks = stored_blocks_with_sequence_hashes(&locals(&[9]), &seqs(&branch)[3..4]);
    let (index, lane) = rig.split(rank(2));
    assert!(
        index
            .probe_place_at(lane, rank(2), run_id, 3, &blocks)
            .unwrap()
    );
    assert_eq!(rig.score(rank(2), &branch), 4);
    assert_eq!(rig.score(rank(1), &branch), 3);
    rig.check();
}

#[test]
fn removal_past_a_split_follows_the_forward() {
    let mut rig = Rig::new(2);
    let path: Vec<u64> = (1..=40).collect();
    crowd(&mut rig, &path, CROWD);
    // Tail-first, head-first and middle removals by the whole holder.
    rig.remove(rank(0), &path, &[39, 38]);
    assert_eq!(rig.score(rank(0), &path), 38);
    rig.remove(rank(0), &path, &[30]);
    assert_eq!(rig.score(rank(0), &path), 30);
    rig.remove(rank(3), &path, &[0]);
    assert_eq!(rig.score(rank(3), &path), 0);
    rig.check();
}

// ----------------------------------------------------------------------------
// S4: entries and coverage agree
// ----------------------------------------------------------------------------

#[test]
fn removals_truncate_and_scrub_only_uncovered_entries() {
    let mut rig = Rig::new(2);
    let path: Vec<u64> = (1..=10).collect();
    let branch = [1, 2, 3, 4, 20, 21];
    rig.store_all(rank(1), &path);
    rig.store(rank(1), &branch, 4, 6).unwrap();
    rig.store_all(rank(2), &path);
    rig.check();

    // Tail-first.
    rig.remove(rank(1), &path, &[9, 8]);
    assert_eq!(rig.score(rank(1), &path), 8);
    rig.check();
    // Mid-chain: the rank keeps the prefix, and the branch below keeps its bits (a hole).
    rig.remove(rank(1), &path, &[6]);
    assert_eq!(rig.score(rank(1), &path), 6);
    assert_eq!(rig.score(rank(1), &branch), 6);
    rig.check();
    // Head-first: everything of the run goes, the branch child keeps its own coverage
    // and entries but is unreachable for the rank.
    rig.remove(rank(1), &path, &[0]);
    assert_eq!(rig.score(rank(1), &path), 0);
    assert_eq!(rig.score(rank(1), &branch), 0);
    assert_eq!(rig.score(rank(2), &path), 10);
    rig.check();
    let lookup = &rig.lanes[1].ranks[&rank(1)];
    assert!(lookup.map.contains_key(ext(&branch, 5)));
    assert!(!lookup.map.contains_key(ext(&path, 3)));
}

#[test]
fn a_store_under_an_uncovered_parent_is_rejected_and_drops_the_entry() {
    let mut rig = Rig::new(1);
    let path = [1, 2, 3, 4];
    rig.store_all(rank(1), &path);
    rig.remove(rank(1), &path, &[1]);
    // The entry for block 3 was scrubbed with the truncation.
    assert_eq!(
        rig.store(rank(1), &[1, 2, 3, 4, 5], 4, 5),
        Err(KvCacheEventError::ParentBlockNotFound)
    );
    assert_eq!(
        rig.store(rank(9), &path, 1, 2),
        Err(KvCacheEventError::ParentBlockNotFound)
    );
    rig.check();
}

#[test]
fn clear_through_the_map_reaches_split_suffixes() {
    let mut rig = Rig::new(3);
    let path: Vec<u64> = (1..=40).collect();
    rig.store_all(rank(0), &[&path[..], &[99][..]].concat());
    crowd(&mut rig, &path, CROWD);
    assert!(rig.shape().splits_prefix_cap >= 1);
    rig.clear(rank(0));
    assert_eq!(rig.score(rank(0), &path), 0);
    assert_eq!(rig.score(rank(0), &[&path[..], &[99][..]].concat()), 0);
    assert!(!rig.lanes[0].ranks.contains_key(&rank(0)));
    for r in 1..=CROWD {
        assert_eq!(rig.score(rank(r), &path), r as u32 + 1);
    }
    // The rank keeps its slot and stores again.
    let slot = rig.slot(rank(0));
    rig.store_all(rank(0), &[5, 6]);
    assert_eq!(rig.slot(rank(0)), slot);
    rig.check();
}

#[test]
fn removed_rank_releases_its_slot_after_the_sweep() {
    let mut rig = Rig::new(2);
    let path = [1, 2, 3];
    rig.store_all(rank(1), &path);
    rig.store_all(rank(2), &path);
    let slot = rig.slot(rank(1));
    rig.remove_rank(rank(1));
    rig.quiesce();
    assert_eq!(rig.score(rank(1), &path), 0);
    // A new rank takes the released slot and is credited only with its own blocks.
    rig.store_all(rank(3), &[7, 8]);
    assert_eq!(rig.slot(rank(3)), slot);
    assert_eq!(rig.score(rank(3), &path), 0);
    assert_eq!(rig.score(rank(3), &[7, 8]), 2);
    assert_eq!(rig.score(rank(2), &path), 3);
    rig.check();
}

// ----------------------------------------------------------------------------
// S9 and reclamation: eager unlink, pending retries, volume sweep
// ----------------------------------------------------------------------------

#[test]
fn emptied_runs_unlink_eagerly_and_cascade() {
    let mut rig = Rig::new(1);
    let a = [1, 2, 3];
    let b = [1, 2, 3, 4];
    rig.store_all(rank(1), &a);
    rig.store_all(rank(2), &a);
    rig.store(rank(2), &b, 3, 4).unwrap();
    assert_eq!(rig.shape().runs_live, 2);
    rig.remove(rank(2), &b, &[3]);
    assert_eq!(
        rig.shape().runs_live,
        1,
        "the end child went with its last holder"
    );
    rig.remove(rank(1), &a, &[0]);
    rig.remove(rank(2), &a, &[0]);
    let shape = rig.shape();
    assert_eq!(shape.runs_live, 0);
    assert_eq!(shape.unlinks, 2);
    assert_eq!(shape.pending_unlinks, 0);
    // ROOT's slot is a tombstone; a re-store links a fresh run.
    rig.store_all(rank(1), &a);
    assert_eq!(rig.score(rank(1), &a), 3);
    rig.check();
}

#[test]
fn a_failed_child_try_lock_defers_to_pending_and_idle_retries_unlink() {
    let mut rig = Rig::new(1);
    let a = [1, 2, 3];
    let b = [1, 2, 3, 4, 5];
    rig.store_all(rank(1), &a);
    rig.store_all(rank(2), &a);
    rig.store(rank(2), &b, 3, 5).unwrap();
    rig.index.probe_fail_try_locks(1);
    rig.remove(rank(2), &b, &[3]);
    let shape = rig.shape();
    assert_eq!(shape.pending_unlinks, 1);
    assert_eq!(shape.runs_live, 2);
    assert_eq!(shape.dead_linked_blocks, 2);
    // Idle retry unlinks it.
    assert!(!rig.index.retry_pending(&mut rig.lanes[0]));
    let shape = rig.shape();
    assert_eq!(shape.pending_unlinks, 0);
    assert_eq!(shape.runs_live, 1);
    rig.check();
}

#[test]
fn volume_sweep_unlinks_what_pending_lost() {
    let mut rig = Rig::new(1);
    let a = [1, 2];
    rig.store_all(rank(1), &a);
    // A second whole holder, so the leaves hang as end children instead of appending.
    rig.store_all(rank(2), &a);
    for leaf in 10..20u64 {
        rig.store(rank(1), &[1, 2, leaf], 2, 3).unwrap();
    }
    rig.index.probe_fail_try_locks(10);
    for leaf in 10..20u64 {
        rig.remove(rank(1), &[1, 2, leaf], &[2]);
    }
    assert_eq!(rig.shape().pending_unlinks, 10);
    // Drop the pending list, as a lost lane would; the sweep still finds the leaves.
    rig.index.forget_pending(&mut rig.lanes[0]);
    rig.index.volume_sweep(&mut rig.lanes[0]);
    let shape = rig.shape();
    assert_eq!(shape.runs_live, 1);
    assert_eq!(shape.volume_sweeps, 1);
    assert_eq!(shape.dead_linked_blocks, 0);
    rig.check();
}

#[test]
fn freed_ids_are_not_reissued_while_a_reader_is_pinned() {
    let mut rig = Rig::new(1);
    rig.store_all(rank(1), &[1, 2]);
    let doomed = rig.pos(rank(1), &[1, 2], 0).run();

    let (pinned_tx, pinned_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let reader = thread::spawn(move || {
        let _pin = crossbeam_epoch::pin();
        pinned_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    pinned_rx.recv().unwrap();
    let pin = rig.index.probe_hold_pin();
    drop(pin);

    rig.remove(rank(1), &[1, 2], &[0]);
    rig.flush_lanes();
    for _ in 0..256 {
        crossbeam_epoch::pin().flush();
    }
    for leaf in 100..300u64 {
        rig.store_all(rank(2), &[leaf]);
        assert_ne!(rig.pos(rank(2), &[leaf], 0).run(), doomed);
    }
    assert!(!rig.index.store.slab.free_ids().contains(&doomed));

    release_tx.send(()).unwrap();
    reader.join().unwrap();
    rig.quiesce();
    assert!(rig.index.store.slab.free_ids().contains(&doomed));
    rig.check();
}

#[test]
fn root_table_retirement_keeps_old_tables_readable_under_a_pin() {
    let index = ArenaIndexC::new();
    let mut lane = index.new_lane();
    let mut id = 0;
    let mut store_root = |index: &ArenaIndexC, lane: &mut CLane, head: u64| {
        id += 1;
        index
            .apply_event(lane, stored(rank(1), id, &[head], 0, 1), None)
            .unwrap();
    };
    let first: Vec<u64> = (1..=10).collect();
    for &head in &first {
        store_root(&index, &mut lane, head);
    }
    let root_table = |index: &ArenaIndexC| index.store.run(ROOT).children.load(Ordering::Acquire);
    let old = root_table(&index);

    let (looked_tx, looked_rx) = std::sync::mpsc::channel::<()>();
    let (grown_tx, grown_rx) = std::sync::mpsc::channel::<()>();
    thread::scope(|scope| {
        let (index, first) = (&index, &first);
        let reader = scope.spawn(move || {
            let _pin = crossbeam_epoch::pin();
            let table = index.store.table(old);
            let lookups = || -> Vec<Option<u64>> {
                first
                    .iter()
                    .map(|&head| table.find(child_key(0, head)))
                    .collect()
            };
            let before = lookups();
            looked_tx.send(()).unwrap();
            grown_rx.recv().unwrap();
            // The table was replaced twice meanwhile, yet reads the same under the pin.
            assert_eq!(before, lookups());
            assert!(before.iter().all(Option::is_some));
        });
        looked_rx.recv().unwrap();
        let mut tables = FxHashSet::default();
        tables.insert(old);
        for head in 1000..1300u64 {
            store_root(index, &mut lane, head);
            tables.insert(root_table(index));
        }
        assert!(tables.len() >= 3, "ROOT's table grew twice");
        index.flush_frees(&mut lane);
        for _ in 0..256 {
            crossbeam_epoch::pin().flush();
        }
        assert!(
            !index
                .store
                .arena
                .free_blocks()
                .iter()
                .any(|block| block.addr == old),
            "a retired table is reused only after the pinned reader leaves"
        );
        grown_tx.send(()).unwrap();
        reader.join().unwrap();
    });
    index.probe_quiesce(&mut [&mut lane]);
    assert!(
        index
            .store
            .arena
            .free_blocks()
            .iter()
            .any(|block| block.addr == old)
    );
    index.probe_check(&[&lane]).unwrap();
    index.flush_frees(&mut lane);
}

// ----------------------------------------------------------------------------
// S6: child tables
// ----------------------------------------------------------------------------

/// Mean slots examined (the terminating empty slot included) by lookups of absent keys.
fn mean_miss_probe(table: table::Table<'_>, offset: u32) -> f64 {
    let mask = table.slots() as usize - 1;
    let samples = 2000u64;
    let total: usize = (0..samples)
        .map(|i| {
            let key = child_key(offset, 1 << 40 | i);
            let mut slot = ((key >> 1) as usize) & mask;
            let mut probes = 1;
            while table.occupied(slot) {
                slot = (slot + 1) & mask;
                probes += 1;
            }
            probes
        })
        .sum();
    total as f64 / samples as f64
}

#[test]
fn three_thousand_children_stay_under_three_quarters_with_short_probes() {
    let mut rig = Rig::new(1);
    let mut rng = fastrand::Rng::with_seed(7);
    let mut live: Vec<u64> = Vec::new();
    let mut next = 1u64;
    // A shared run [1_000_000] with 3000 children at offset 1, then churn.
    let parent = 1_000_000u64;
    rig.store_all(rank(1), &[parent]);
    // A second whole holder, so leaves hang as children instead of appending.
    rig.store_all(rank(2), &[parent]);
    for _ in 0..3000 {
        rig.store(rank(1), &[parent, next], 1, 2).unwrap();
        live.push(next);
        next += 1;
    }
    let parent_run = rig.pos(rank(1), &[parent], 0).run();
    let mut probes = Vec::new();
    for step in 0..20_000 {
        if rng.bool() && !live.is_empty() {
            let leaf = live.swap_remove(rng.usize(..live.len()));
            rig.remove(rank(1), &[parent, leaf], &[1]);
        } else {
            rig.store(rank(1), &[parent, next], 1, 2).unwrap();
            live.push(next);
            next += 1;
        }
        if step % 200 == 0 {
            let store = &rig.index.store;
            let table = store.table(store.run(parent_run).children.load(Ordering::Acquire));
            assert!(table.used() <= claim_budget(table.slots()));
            assert_eq!(table.live() as usize, live.len());
            probes.push(mean_miss_probe(table, 1));
        }
    }
    let mean = probes.iter().sum::<f64>() / probes.len() as f64;
    let worst = probes.iter().copied().fold(0.0, f64::max);
    assert!(
        mean < 4.0,
        "time-averaged miss probe {mean:.2} (worst {worst:.2})"
    );
    // Re-storing an unlinked leaf reuses its own tombstone instead of spending budget.
    let leaf = live.pop().unwrap();
    rig.remove(rank(1), &[parent, leaf], &[1]);
    let store = &rig.index.store;
    let used = store
        .table(store.run(parent_run).children.load(Ordering::Acquire))
        .used();
    rig.store(rank(1), &[parent, leaf], 1, 2).unwrap();
    live.push(leaf);
    let store = &rig.index.store;
    assert_eq!(
        store
            .table(store.run(parent_run).children.load(Ordering::Acquire))
            .used(),
        used
    );
    for &leaf in live.iter().take(100) {
        assert_eq!(rig.score(rank(1), &[parent, leaf]), 2);
    }
    rig.check();
}

// ----------------------------------------------------------------------------
// Memory accounting
// ----------------------------------------------------------------------------

#[test]
fn memory_report_counts_memberships_and_sixteen_byte_map_slots() {
    let mut rig = Rig::new(2);
    let shared: Vec<u64> = (1..=64).collect();
    for r in 0..8u64 {
        let mut path = shared.clone();
        path.extend((0..16).map(|i| 1000 * (r + 1) + i));
        rig.store_all(rank(r), &path);
    }
    rig.quiesce();
    let lanes: Vec<&CLane> = rig.lanes.iter().collect();
    let report = rig.index.memory_report_for(&lanes);
    assert_eq!(report.memberships, 8 * 80);
    assert_eq!(report.distinct_blocks, 64 + 8 * 16);
    let map_slots: usize = rig
        .lanes
        .iter()
        .flat_map(|lane| lane.ranks.values())
        .map(|lookup| lookup.map.capacity())
        .sum();
    assert_eq!(report.map_bytes, 16 * map_slots as u64);
    assert!(report.header_bytes <= 96 * 10);
    assert!(
        report.arena_reserved_bytes
            >= report.arena_used_bytes + report.arena_free_bytes + report.arena_stranded_bytes
    );
    assert!(report.slab_reserved_bytes >= report.header_bytes);
    assert!(report.array_bytes_live > 0 && report.child_table_bytes > 0);
    assert_eq!(report.forward_bytes + report.overflow_bytes, 0);
    assert!(report.cutoff_table_bytes > 0 && report.array_bytes_slack > 0);
    assert_eq!(report.epoch_pending_bytes, 0);
    assert!(std::mem::size_of::<run::RunHeader>() <= 96);
    rig.check();
}

// ----------------------------------------------------------------------------
// I5: dump round trip
// ----------------------------------------------------------------------------

#[test]
fn dump_round_trip_never_scores_higher_and_is_exact_without_holes() {
    for seed in 0..20u64 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let holes = seed % 2 == 1;
        let mut rig = Rig::new(3);
        let mut paths: Vec<(WorkerWithDpRank, Vec<u64>)> = Vec::new();
        for _ in 0..200 {
            let r = rank(rng.u64(..6));
            let path: Vec<u64> = (0..rng.usize(1..12)).map(|_| rng.u64(1..5)).collect();
            if holes && rng.u32(..4) == 0 && !path.is_empty() {
                let at = rng.usize(..path.len());
                let _ = rig.store(r, &path, 0, path.len());
                rig.remove(r, &path, &[at]);
            } else {
                rig.store_all(r, &path);
            }
            paths.push((r, path));
        }
        rig.quiesce();
        let events = rig.index.dump_tree_as_events();
        let mut replay = Rig::new(3);
        for event in events {
            let r = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
            replay.event(r, event.event.data).unwrap();
        }
        for (r, path) in &paths {
            let original = rig.score(*r, path);
            let replayed = replay.score(*r, path);
            assert!(replayed <= original, "seed {seed}: {replayed} > {original}");
            if !holes {
                assert_eq!(replayed, original, "seed {seed}");
            }
        }
        replay.check();
    }
}

// ----------------------------------------------------------------------------
// CRTC's race tests, ported: lanes on threads, readers checking ever-held bounds
// ----------------------------------------------------------------------------

/// Lanes running `SyncIndexer::worker`, with acknowledged events.
struct Lanes {
    senders: Vec<flume::Sender<WorkerTask>>,
    threads: Vec<thread::JoinHandle<()>>,
}

impl Lanes {
    fn new(index: Arc<ArenaIndexC>, lanes: usize) -> Self {
        let mut senders = Vec::new();
        let mut threads = Vec::new();
        for _ in 0..lanes {
            let (tx, rx) = flume::unbounded();
            let index = Arc::clone(&index);
            threads.push(thread::spawn(move || index.worker(rx, None).unwrap()));
            senders.push(tx);
        }
        Self { senders, threads }
    }

    fn apply(&self, lane: usize, event: RouterEvent) -> bool {
        let (resp, rx) = oneshot::channel();
        self.senders[lane]
            .send(WorkerTask::EventWithAck { event, resp })
            .unwrap();
        rx.blocking_recv().unwrap()
    }

    fn task(&self, lane: usize, task: WorkerTask) {
        self.senders[lane].send(task).unwrap();
    }

    fn flush(&self) {
        for sender in &self.senders {
            let (resp, rx) = oneshot::channel();
            sender.send(WorkerTask::Flush(resp)).unwrap();
            rx.blocking_recv().unwrap();
        }
    }
}

impl Drop for Lanes {
    fn drop(&mut self) {
        for sender in &self.senders {
            let _ = sender.send(WorkerTask::Terminate);
        }
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }
}

fn stored(r: WorkerWithDpRank, id: u64, path: &[u64], from: usize, to: usize) -> RouterEvent {
    let hashes = seqs(&path[..to]);
    router_event(
        r.worker_id,
        id,
        r.dp_rank,
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: from
                .checked_sub(1)
                .map(|i| ExternalSequenceBlockHash(hashes[i])),
            start_position: None,
            blocks: stored_blocks_with_sequence_hashes(&locals(&path[from..to]), &hashes[from..to]),
        }),
    )
}

fn removed(r: WorkerWithDpRank, id: u64, path: &[u64], positions: &[usize]) -> RouterEvent {
    let hashes = seqs(path);
    remove_event(
        r.worker_id,
        id,
        r.dp_rank,
        positions
            .iter()
            .map(|&p| ExternalSequenceBlockHash(hashes[p]))
            .collect(),
    )
}

#[test]
fn race_find_during_split_never_overcounts() {
    // Repeated local hashes, so every prefix of the hot run looks alike and splits land
    // on repeated edges.
    let path: Vec<u64> = vec![7; 64];
    let index = Arc::new(ArenaIndexC::new());
    let lanes = Lanes::new(Arc::clone(&index), 4);
    let ranks = 24u64;
    // Highest prefix each rank has ever stored.
    let ever: Arc<Vec<AtomicUsize>> = Arc::new((0..ranks).map(|_| AtomicUsize::new(0)).collect());
    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let index = Arc::clone(&index);
            let ever = Arc::clone(&ever);
            let stop = Arc::clone(&stop);
            let path = path.clone();
            thread::spawn(move || {
                let query = locals(&path);
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let scores = index.find_matches_impl(&query, false);
                    for (r, &score) in &scores.scores {
                        let bound = ever[r.worker_id as usize].load(Ordering::Acquire);
                        assert!(
                            score as usize <= bound,
                            "rank {r:?} scored {score} > {bound}"
                        );
                    }
                    reads += 1;
                }
                reads
            })
        })
        .collect();
    let mut rng = fastrand::Rng::with_seed(3);
    let mut id = 0;
    for round in 0..3000 {
        let r = rank(rng.u64(..ranks));
        let lane = r.worker_id as usize % 4;
        let len = rng.usize(1..=path.len());
        ever[r.worker_id as usize].fetch_max(len, Ordering::AcqRel);
        id += 1;
        lanes.apply(lane, stored(r, id, &path, 0, len));
        if round % 3 == 0 {
            id += 1;
            let cut = rng.usize(..len);
            lanes.apply(lane, removed(r, id, &path, &[cut]));
        }
    }
    stop.store(true, Ordering::Relaxed);
    let reads: u64 = readers.into_iter().map(|r| r.join().unwrap()).sum();
    assert!(reads > 0);
    lanes.flush();
    assert!(
        index.shape_report().splits_prefix_cap > 0,
        "the test must split"
    );
}

#[test]
fn whole_slot_drops_racing_readers_never_overcount() {
    let index = Arc::new(ArenaIndexC::new());
    let lanes = Lanes::new(Arc::clone(&index), 4);
    let stop = Arc::new(AtomicBool::new(false));
    // Generation g of rank slot k stores path [k, g]; a reader seeing rank (k, g) credited
    // on [k, g'] with g' != g is an overcount through a recycled slot.
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let index = Arc::clone(&index);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut rng = fastrand::Rng::with_seed(9);
                while !stop.load(Ordering::Relaxed) {
                    let k = rng.u64(..8);
                    let g = rng.u64(..64);
                    let scores = index.find_matches_impl(&locals(&[1000 + k, g]), false);
                    for (r, &score) in &scores.scores {
                        let owner_k = r.worker_id / 1000;
                        let owner_g = r.worker_id % 1000;
                        assert_eq!(owner_k, k, "credited a rank of another prefix");
                        if score > 1 {
                            assert_eq!(owner_g, g, "credited a block of another generation");
                        }
                    }
                }
            })
        })
        .collect();
    let mut id = 0;
    for g in 0..64u64 {
        for k in 0..8u64 {
            let r = WorkerWithDpRank::new(k * 1000 + g, 0);
            let lane = (k as usize) % 4;
            id += 1;
            assert!(lanes.apply(lane, stored(r, id, &[1000 + k, g], 0, 2)));
        }
        for k in 0..8u64 {
            let r = WorkerWithDpRank::new(k * 1000 + g, 0);
            lanes.task(
                (k as usize) % 4,
                WorkerTask::RemoveWorkerDpRank {
                    worker_id: r.worker_id,
                    dp_rank: 0,
                    sweep_tree: true,
                },
            );
        }
    }
    lanes.flush();
    stop.store(true, Ordering::Relaxed);
    for reader in readers {
        reader.join().unwrap();
    }
    assert_eq!(index.shape_report().runs_live, 0);
}

#[test]
fn promotions_racing_splits_settle_exactly() {
    let path: Vec<u64> = (1..=48).collect();
    let index = Arc::new(ArenaIndexC::new());
    let lanes = Lanes::new(Arc::clone(&index), 4);
    let barrier = Arc::new(Barrier::new(4));
    let ranks_per_lane = 8u64;
    thread::scope(|scope| {
        for lane in 0..4usize {
            let lanes = &lanes;
            let barrier = Arc::clone(&barrier);
            let path = path.clone();
            scope.spawn(move || {
                let mut rng = fastrand::Rng::with_seed(lane as u64);
                barrier.wait();
                for step in 0..400u64 {
                    let r = rank(lane as u64 + 4 * rng.u64(..ranks_per_lane));
                    let id = (lane as u64) << 32 | step;
                    // Partial stores create cutoffs (and splits); whole ones promote.
                    let len = if rng.bool() {
                        path.len()
                    } else {
                        rng.usize(1..path.len())
                    };
                    lanes.apply(lane, stored(r, id, &path, 0, len));
                }
            });
        }
    });
    lanes.flush();
    // Every rank's final score is its longest prefix: stores never shrink a holding.
    let scores = index.find_matches_impl(&locals(&path), false);
    for (r, &score) in &scores.scores {
        assert!(score >= 1, "{r:?}");
    }
    assert!(scores.scores.len() <= 32);
    let full: Vec<_> = scores
        .scores
        .iter()
        .filter(|&(_, &score)| score as usize == path.len())
        .collect();
    assert!(!full.is_empty());
    drop(lanes);
}

#[test]
fn restore_after_eager_unlink_races_readers() {
    let path: Vec<u64> = (1..=12).collect();
    let index = Arc::new(ArenaIndexC::new());
    let lanes = Lanes::new(Arc::clone(&index), 2);
    let stop = Arc::new(AtomicBool::new(false));
    let reader = {
        let index = Arc::clone(&index);
        let stop = Arc::clone(&stop);
        let path = path.clone();
        thread::spawn(move || {
            let query = locals(&path);
            while !stop.load(Ordering::Relaxed) {
                let scores = index.find_matches_impl(&query, false);
                for &score in scores.scores.values() {
                    assert!(score as usize <= path.len());
                }
            }
        })
    };
    let mut id = 0;
    for round in 0..2000u64 {
        let r = rank(round % 2);
        let lane = (round % 2) as usize;
        id += 1;
        assert!(lanes.apply(lane, stored(r, id, &path, 0, path.len())));
        id += 1;
        // Remove from the head: the run empties and unlinks eagerly once both are gone.
        lanes.apply(lane, removed(r, id, &path, &[0]));
    }
    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();
    lanes.flush();
    let shape = index.shape_report();
    assert_eq!(shape.runs_live, 0, "{shape}");
    assert!(shape.unlinks > 0);
}

// ----------------------------------------------------------------------------
// ThreadPoolIndexer contracts with this backend
// ----------------------------------------------------------------------------

mod thread_pool {
    use super::*;
    use crate::indexer::{KvIndexerInterface, ThreadPoolIndexer};
    use crate::test_utils::{
        assert_no_scores, assert_score, make_store_event, make_store_event_with_dp_rank,
    };

    #[tokio::test]
    async fn cold_removals_then_store_score_on_one_and_four_lanes() {
        for lanes in [1, 4] {
            for rank_removal in [false, true] {
                let indexer = ThreadPoolIndexer::new(ArenaIndexC::new(), lanes, 16);
                indexer.apply_event(make_store_event(1, &[1])).await;
                if rank_removal {
                    indexer.remove_worker_dp_rank(2, 0).await;
                } else {
                    indexer.remove_worker(2).await;
                }
                indexer.apply_event(make_store_event(2, &[2])).await;
                indexer.flush().await;
                assert_score(&indexer, &[2], rank(2), 1).await;
                assert_score(&indexer, &[1], rank(1), 1).await;
            }
        }
    }

    #[tokio::test]
    async fn sibling_ranks_use_independent_sticky_queues() {
        let indexer = ThreadPoolIndexer::new(ArenaIndexC::new(), 2, 16);
        indexer
            .apply_event(make_store_event_with_dp_rank(7, &[10], 0))
            .await;
        indexer
            .apply_event(make_store_event_with_dp_rank(7, &[20], 1))
            .await;
        indexer.flush().await;
        assert_score(&indexer, &[10], WorkerWithDpRank::new(7, 0), 1).await;
        assert_score(&indexer, &[20], WorkerWithDpRank::new(7, 1), 1).await;
    }

    #[tokio::test]
    async fn whole_worker_remove_barriers_every_rank_lane_before_returning() {
        let indexer = ThreadPoolIndexer::new(ArenaIndexC::new(), 2, 16);
        indexer
            .apply_event(make_store_event_with_dp_rank(7, &[10], 0))
            .await;
        indexer
            .apply_event(make_store_event_with_dp_rank(7, &[20], 1))
            .await;
        indexer.remove_worker(7).await;
        assert_no_scores(&indexer, &[10]).await;
        assert_no_scores(&indexer, &[20]).await;
        indexer
            .apply_event(make_store_event_with_dp_rank(7, &[30], 1))
            .await;
        indexer.flush().await;
        assert_score(&indexer, &[30], WorkerWithDpRank::new(7, 1), 1).await;
    }

    #[tokio::test]
    async fn approximate_lru_and_anchors_are_unsupported() {
        let backend = ArenaIndexC::new();
        assert!(!backend.supports_routing_decision_pruning());
        assert!(
            backend
                .find_matches_from_anchor(
                    crate::indexer::AnchorRef {
                        anchor_id: ExternalSequenceBlockHash(1),
                        anchor_local_hash: LocalBlockHash(1),
                        anchor_depth: 1,
                    },
                    &[],
                )
                .is_err()
        );
        let indexer = ThreadPoolIndexer::new(backend, 2, 16);
        indexer.apply_event(make_store_event(1, &[1, 2])).await;
        indexer.flush().await;
        let dump = indexer.dump_events().await.unwrap();
        assert_eq!(dump.len(), 1);
    }
}
