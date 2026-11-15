// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The run storage checked against the flat storage it replaced.

use std::sync::atomic::{AtomicBool, Ordering};

use rustc_hash::{FxHashMap, FxHashSet};

use super::flat::{FlatIndex, FlatLane};
use super::test_owners::{cache_owner, residency_owners};
use super::{
    EdgeCursor, EdgeLookup, IndexedResidencyOwner, LowerTierContinuation, LowerTierIndexer,
    LowerTierMatchDetails, OwnerSet, OwnerTable, TransitionKey, WorkerBlockIndex,
};
use crate::prelude::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, ResidencyDomain, ResidencyOwner, ResidencyProjection,
    ResidencyRoutingSnapshot, RouterEvent, StorageTier, WireResidencyDomain, WorkerWithDpRank,
};
use crate::residency::RouterHintSourceMetadata;
use crate::testing::stored_blocks_with_sequence_hashes;

struct Node {
    parent: Option<usize>,
    local: LocalBlockHash,
    hash: ExternalSequenceBlockHash,
    children: Vec<usize>,
}

/// Chains that share prefixes. Each node is one block. Its sequence hash is
/// unique, as the engine hash invariant requires; local hashes repeat under
/// different parents.
struct Forest {
    nodes: Vec<Node>,
    roots: Vec<usize>,
}

impl Forest {
    fn random(rng: &mut fastrand::Rng, size: usize) -> Self {
        let mut forest = Self {
            nodes: Vec::new(),
            roots: Vec::new(),
        };
        for idx in 0..size {
            // Mostly extend the latest chain, so that runs get long.
            let parent = match rng.u32(..20) {
                _ if idx == 0 => None,
                0 => None,
                1..=3 => Some(rng.usize(..idx)),
                _ => Some(idx - 1),
            };
            let siblings = match parent {
                Some(parent) => &forest.nodes[parent].children,
                None => &forest.roots,
            };
            let mut local = rng.u64(1..=8);
            while siblings
                .iter()
                .any(|&node| forest.nodes[node].local.0 == local)
            {
                local += 1;
            }
            forest.nodes.push(Node {
                parent,
                local: LocalBlockHash(local),
                hash: ExternalSequenceBlockHash(1_000 + idx as u64),
                children: Vec::new(),
            });
            match parent {
                Some(parent) => forest.nodes[parent].children.push(idx),
                None => forest.roots.push(idx),
            }
        }
        forest
    }

    fn hash(&self, node: usize) -> ExternalSequenceBlockHash {
        self.nodes[node].hash
    }

    fn parent_hash(&self, node: usize) -> Option<ExternalSequenceBlockHash> {
        self.nodes[node].parent.map(|parent| self.hash(parent))
    }

    /// `start` and up to `max_len - 1` descendants along one chain.
    fn chain_down(&self, rng: &mut fastrand::Rng, start: usize, max_len: usize) -> Vec<usize> {
        let mut path = vec![start];
        while path.len() < max_len {
            let children = &self.nodes[*path.last().unwrap()].children;
            if children.is_empty() {
                break;
            }
            path.push(children[rng.usize(..children.len())]);
        }
        path
    }

    fn root_path(&self, node: usize) -> Vec<usize> {
        let mut path = vec![node];
        while let Some(parent) = self.nodes[*path.last().unwrap()].parent {
            path.push(parent);
        }
        path.reverse();
        path
    }

    fn child(&self, parent: Option<usize>, local: LocalBlockHash) -> Option<usize> {
        let children = match parent {
            Some(parent) => &self.nodes[parent].children,
            None => &self.roots,
        };
        children
            .iter()
            .copied()
            .find(|&node| self.nodes[node].local == local)
    }

    fn blocks(&self, path: &[usize]) -> Vec<KvCacheStoredBlockData> {
        path.iter()
            .map(|&node| KvCacheStoredBlockData {
                tokens_hash: self.nodes[node].local,
                block_hash: self.nodes[node].hash,
                mm_extra_info: None,
            })
            .collect()
    }

    fn locals(&self, path: &[usize]) -> Vec<LocalBlockHash> {
        path.iter().map(|&node| self.nodes[node].local).collect()
    }
}

fn event(owner: ResidencyOwner, data: KvCacheEventData) -> RouterEvent {
    match owner {
        ResidencyOwner::Worker(worker) => RouterEvent::with_residency_domain(
            worker.worker_id,
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: worker.dp_rank,
            },
            StorageTier::HostPinned,
            ResidencyDomain::Worker,
        ),
        ResidencyOwner::CacheOwner(source) => RouterEvent::with_cache_owner(
            0,
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: 0,
            },
            StorageTier::HostPinned,
            source,
        ),
    }
}

fn store_event(forest: &Forest, owner: ResidencyOwner, path: &[usize]) -> RouterEvent {
    event(
        owner,
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: forest.parent_hash(path[0]),
            start_position: None,
            blocks: forest.blocks(path),
        }),
    )
}

/// A store of a chain piece, a remove (a chain piece in either order, or
/// scattered blocks, held or not), or a clear.
fn random_event(rng: &mut fastrand::Rng, forest: &Forest, owner: ResidencyOwner) -> RouterEvent {
    let nodes = forest.nodes.len();
    match rng.u32(..20) {
        0 => event(owner, KvCacheEventData::Cleared),
        1..=9 => {
            let (start, len) = (rng.usize(..nodes), rng.usize(1..=24));
            let path = forest.chain_down(rng, start, len);
            store_event(forest, owner, &path)
        }
        kind => {
            let mut removed = if kind % 3 == 0 {
                (0..rng.usize(1..=12)).map(|_| rng.usize(..nodes)).collect()
            } else {
                let (start, len) = (rng.usize(..nodes), rng.usize(1..=24));
                forest.chain_down(rng, start, len)
            };
            if kind % 3 == 2 {
                removed.reverse();
            }
            event(
                owner,
                KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: removed.iter().map(|&node| forest.hash(node)).collect(),
                }),
            )
        }
    }
}

fn hint_source() -> RouterHintSourceMetadata {
    RouterHintSourceMetadata {
        source_control_endpoint: "tcp://owner:23280".to_string(),
        worker_type: "prefill".to_string(),
        topology_domains: Default::default(),
    }
}

fn members(owners: &OwnerSet, table: &OwnerTable) -> FxHashSet<IndexedResidencyOwner> {
    let mut members = FxHashSet::default();
    owners.for_each_owner(table, |owner| {
        members.insert(owner);
    });
    members
}

type Edges =
    FxHashMap<TransitionKey, (ExternalSequenceBlockHash, FxHashSet<IndexedResidencyOwner>)>;

fn run_edges(index: &LowerTierIndexer) -> Edges {
    index
        .store
        .held_edges()
        .into_iter()
        .map(|(key, (hash, owners))| (key, (hash, members(&owners, &index.owners))))
        .collect()
}

fn flat_edges(flat: &FlatIndex) -> Edges {
    flat.held_edges()
        .into_iter()
        .map(|(key, (hash, owners))| (key, (hash, owners.into_iter().collect())))
        .collect()
}

/// Checks that every lane's block count for an owner is what the edges hold.
fn assert_lane_counts(edges: &Edges, lanes: &[&WorkerBlockIndex]) {
    let mut held: FxHashMap<IndexedResidencyOwner, usize> = FxHashMap::default();
    for (_, owners) in edges.values() {
        for owner in owners {
            *held.entry(*owner).or_default() += 1;
        }
    }
    let mut counted = 0;
    for lane in lanes {
        for (owner, state) in lane.iter() {
            assert_eq!(
                state.blocks,
                held.get(owner).copied().unwrap_or(0),
                "{owner:?}"
            );
            counted += usize::from(state.blocks > 0);
        }
    }
    assert_eq!(counted, held.len(), "an owner holds blocks no lane counts");
}

fn assert_same_details(runs: LowerTierMatchDetails, flat: LowerTierMatchDetails, context: &str) {
    assert_eq!(runs.hits, flat.hits, "{context}");
    assert_eq!(
        runs.next_continuations, flat.next_continuations,
        "{context}"
    );
    match (runs.kv_transfer_extensions, flat.kv_transfer_extensions) {
        (None, None) => {}
        (Some(mut runs), Some(mut flat)) => {
            assert_eq!(runs.block_hashes, flat.block_hashes, "{context}");
            assert_eq!(
                runs.owner_prefix_blocks, flat.owner_prefix_blocks,
                "{context}"
            );
            // Interval order follows hash-set iteration, which the two
            // storages do not share.
            for intervals in [&mut runs.source_intervals, &mut flat.source_intervals] {
                intervals.sort_by_key(|interval| (interval.source, interval.start, interval.end));
            }
            assert_eq!(runs.source_intervals, flat.source_intervals, "{context}");
        }
        _ => panic!("only one storage returned extensions: {context}"),
    }
}

/// The run-backed indexer and the flat reference, fed the same events.
struct Both {
    runs: LowerTierIndexer,
    lane: WorkerBlockIndex,
    flat: FlatIndex,
    flat_lane: FlatLane,
    snapshots: [ResidencyRoutingSnapshot; 2],
}

impl Both {
    fn new() -> Self {
        let worker = WorkerWithDpRank::new(100, 0);
        // Two cache owners count for one worker, and both are hint sources.
        let projection =
            ResidencyProjection::new([(cache_owner(1), worker), (cache_owner(3), worker)]).unwrap();
        Self {
            runs: LowerTierIndexer::new(),
            lane: WorkerBlockIndex::default(),
            flat: FlatIndex::new(),
            flat_lane: FlatLane::default(),
            snapshots: [
                ResidencyRoutingSnapshot::from_projection(ResidencyProjection::default()),
                ResidencyRoutingSnapshot::new(
                    projection,
                    [
                        (cache_owner(1), hint_source(), None),
                        (cache_owner(3), hint_source(), None),
                    ],
                ),
            ],
        }
    }

    fn apply(&mut self, event: RouterEvent) {
        self.runs
            .apply_event(&mut self.lane, event.clone())
            .unwrap();
        self.flat.apply(&mut self.flat_lane, event);
    }

    /// Every test worker has one rank, so removing the worker and removing
    /// its rank both clear the one worker owner.
    fn remove_worker(&mut self, worker_id: u64, by_rank: bool) {
        if by_rank {
            self.runs
                .remove_worker_dp_rank(&mut self.lane, worker_id, 0);
        } else {
            self.runs.remove_worker(&mut self.lane, worker_id);
        }
        let worker = ResidencyOwner::worker(WorkerWithDpRank::new(worker_id, 0));
        self.flat.clear(&mut self.flat_lane, worker);
    }

    fn check_state(&self, context: &str) {
        self.runs.store.assert_invariants();
        let edges = run_edges(&self.runs);
        assert_eq!(edges, flat_edges(&self.flat), "{context}");
        assert_lane_counts(&edges, &[&self.lane]);
    }

    /// Queries along random chains, from random continuation points.
    fn check_queries(&self, rng: &mut fastrand::Rng, forest: &Forest, context: &str) {
        for _ in 0..6 {
            let node = rng.usize(..forest.nodes.len());
            let mut path = forest.root_path(node);
            path.extend(forest.chain_down(rng, node, 12).into_iter().skip(1));
            let mut locals = forest.locals(&path);
            if rng.bool() {
                // A block nobody stored.
                locals.push(LocalBlockHash(999));
            }

            let mut continuations = FxHashMap::default();
            for worker_id in [100, 102, 999] {
                let start = rng.usize(..=path.len());
                let continuation = if start == 0 {
                    LowerTierContinuation::from_root(0)
                } else if rng.u32(..10) == 0 {
                    // A hash the index never saw.
                    LowerTierContinuation::new(start, ExternalSequenceBlockHash(7))
                } else {
                    LowerTierContinuation::new(start, forest.hash(path[start - 1]))
                };
                continuations.insert(WorkerWithDpRank::new(worker_id, 0), continuation);
            }

            for snapshot in &self.snapshots {
                for retain in [false, true] {
                    assert_same_details(
                        self.runs.query_match_details_with_options_and_snapshot(
                            &locals,
                            &continuations,
                            retain,
                            snapshot,
                        ),
                        self.flat.query(&locals, &continuations, retain, snapshot),
                        context,
                    );
                }
                let mut runs = self.runs.root_workers(locals[0], snapshot.projection());
                let mut flat = self.flat.root_workers(locals[0], snapshot.projection());
                runs.sort();
                flat.sort();
                assert_eq!(runs, flat, "{context}");
            }
        }
    }

    /// One cursor asked for edges in a walk-like order with jumps: every
    /// answer must be the flat lookup's, however the cursor got there.
    fn check_cursor(&self, rng: &mut fastrand::Rng, forest: &Forest, context: &str) {
        let mut runs = self.runs.store.cursor();
        let mut flat = self.flat.cursor();
        // The node whose hash is the next call's parent.
        let mut at: Option<usize> = None;
        for _ in 0..200 {
            if rng.u32(..6) == 0 {
                at = rng.bool().then(|| rng.usize(..forest.nodes.len()));
            }
            let children = match at {
                Some(node) => &forest.nodes[node].children,
                None => &forest.roots,
            };
            let local = if children.is_empty() || rng.u32(..8) == 0 {
                LocalBlockHash(rng.u64(1..=12))
            } else {
                forest.nodes[children[rng.usize(..children.len())]].local
            };
            let parent = at.map(|node| forest.hash(node));

            let got = runs
                .edge(parent, local)
                .map(|edge| (edge.child_hash, members(edge.owners, &self.runs.owners)));
            let expected = flat
                .edge(parent, local)
                .map(|edge| (edge.child_hash, members(edge.owners, self.flat.owners())));
            assert_eq!(
                got, expected,
                "{context}: edge after {parent:?} for {local:?}"
            );

            // Follow the chain when the block exists, held or not.
            at = forest.child(at, local).or(at);
        }
    }
}

#[test]
fn matches_the_flat_reference_on_random_streams() {
    let owners = residency_owners(4);
    let mut reached = [0usize; 6];
    let mut head_kinds = [0usize; 4];
    let mut shared_edges = 0;
    for seed in 0..24 {
        let mut rng = fastrand::Rng::with_seed(seed);
        let forest = Forest::random(&mut rng, 160);
        let mut both = Both::new();
        for step in 0..600 {
            match rng.u32(..150) {
                0 => both.remove_worker(100, false),
                1 => both.remove_worker(102, true),
                _ => {
                    let owner = owners[rng.usize(..owners.len())];
                    both.apply(random_event(&mut rng, &forest, owner));
                }
            }
            if step % 6 == 0 {
                let context = format!("seed {seed} step {step}");
                both.check_state(&context);
                both.check_queries(&mut rng, &forest, &context);
                both.check_cursor(&mut rng, &forest, &context);
                for (total, kind) in head_kinds.iter_mut().zip(both.runs.store.head_kinds()) {
                    *total += kind;
                }
                shared_edges += run_edges(&both.runs)
                    .values()
                    .filter(|(_, owners)| owners.len() > 2)
                    .count();
            }
        }
        let store = &both.runs.store.reached;
        for (total, counter) in reached.iter_mut().zip([
            &store.freed,
            &store.trimmed,
            &store.dropped_holes,
            &store.cursor_hops,
            &store.looked_again,
            &store.stale,
        ]) {
            *total += counter.load(Ordering::Relaxed);
        }
    }

    // The streams must have exercised what the run storage adds.
    let [
        freed,
        trimmed,
        dropped_holes,
        cursor_hops,
        looked_again,
        stale,
    ] = reached;
    assert!(freed > 0, "no run was freed");
    assert!(trimmed > 0, "no trailing holes were trimmed");
    assert!(
        dropped_holes > 0,
        "no store extended a run over dropped holes"
    );
    assert!(cursor_hops > 0, "no walk crossed from one run to the next");
    assert_eq!(
        (looked_again, stale),
        (0, 0),
        "single-threaded writes cannot race"
    );
    let [roots, absent_parent, branches, continued] = head_kinds;
    assert!(roots > 0 && absent_parent > 0 && branches > 0 && continued > 0);
    assert!(shared_edges > 0, "no edge had three or more owners");
}

/// Event lanes with disjoint owners write to shared chains while queries
/// run. A worker whose blocks nobody else touches must read exactly, a worker
/// with no blocks must read zero, and the final state must be what the same
/// streams give when applied one after another.
#[test]
fn concurrent_lanes_match_the_sequential_result() {
    let mut reached = [0usize; 2];
    for seed in 0..6 {
        let mut rng = fastrand::Rng::with_seed(1_000 + seed);
        let forest = Forest::random(&mut rng, 120);
        let index = LowerTierIndexer::new();

        let pinned = WorkerWithDpRank::new(900, 0);
        let absent = WorkerWithDpRank::new(999, 0);
        let mut pinned_lane = WorkerBlockIndex::default();
        let pinned_paths: Vec<Vec<usize>> = (0..3)
            .map(|_| forest.root_path(rng.usize(..forest.nodes.len())))
            .collect();
        let pinned_events: Vec<RouterEvent> = pinned_paths
            .iter()
            .map(|path| store_event(&forest, ResidencyOwner::worker(pinned), path))
            .collect();
        for event in &pinned_events {
            index.apply_event(&mut pinned_lane, event.clone()).unwrap();
        }

        let streams: Vec<Vec<RouterEvent>> = (0..4)
            .map(|lane| {
                let owners = [
                    ResidencyOwner::worker(WorkerWithDpRank::new(200 + lane, 0)),
                    ResidencyOwner::cache_owner(cache_owner(10 + lane as u8)),
                ];
                (0..3_000)
                    .map(|_| {
                        let owner = owners[rng.usize(..2)];
                        random_event(&mut rng, &forest, owner)
                    })
                    .collect()
            })
            .collect();
        let snapshot = ResidencyRoutingSnapshot::new(
            ResidencyProjection::default(),
            (10..14).map(|slot| (cache_owner(slot), hint_source(), None)),
        );

        let stop = AtomicBool::new(false);
        let lanes: Vec<WorkerBlockIndex> = std::thread::scope(|scope| {
            let readers: Vec<_> = (0..2)
                .map(|reader| {
                    let (index, forest, paths, stop, snapshot) =
                        (&index, &forest, &pinned_paths, &stop, &snapshot);
                    scope.spawn(move || {
                        let mut rng = fastrand::Rng::with_seed(reader);
                        loop {
                            let path = &paths[rng.usize(..paths.len())];
                            let start = rng.usize(..path.len());
                            let mut continuations = FxHashMap::default();
                            continuations.insert(
                                pinned,
                                match start {
                                    0 => LowerTierContinuation::from_root(0),
                                    start => LowerTierContinuation::new(
                                        start,
                                        forest.hash(path[start - 1]),
                                    ),
                                },
                            );
                            continuations.insert(absent, LowerTierContinuation::from_root(0));
                            let details = index.query_match_details_with_options_and_snapshot(
                                &forest.locals(path),
                                &continuations,
                                rng.bool(),
                                snapshot,
                            );
                            assert_eq!(details.hits[&pinned], path.len() - start);
                            assert_eq!(details.hits[&absent], 0);
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                    })
                })
                .collect();
            let writers: Vec<_> = streams
                .iter()
                .map(|stream| {
                    let index = &index;
                    scope.spawn(move || {
                        let mut lane = WorkerBlockIndex::default();
                        for event in stream {
                            index.apply_event(&mut lane, event.clone()).unwrap();
                        }
                        lane
                    })
                })
                .collect();
            let lanes = writers
                .into_iter()
                .map(|writer| writer.join().unwrap())
                .collect();
            stop.store(true, Ordering::Relaxed);
            for reader in readers {
                reader.join().unwrap();
            }
            lanes
        });

        let flat = FlatIndex::new();
        let mut flat_lane = FlatLane::default();
        for event in pinned_events
            .into_iter()
            .chain(streams.into_iter().flatten())
        {
            flat.apply(&mut flat_lane, event);
        }
        index.store.assert_invariants();
        let edges = run_edges(&index);
        assert_eq!(edges, flat_edges(&flat), "seed {seed}");
        let mut all_lanes: Vec<&WorkerBlockIndex> = lanes.iter().collect();
        all_lanes.push(&pinned_lane);
        assert_lane_counts(&edges, &all_lanes);

        let store = &index.store.reached;
        for (total, counter) in reached.iter_mut().zip([&store.freed, &store.trimmed]) {
            *total += counter.load(Ordering::Relaxed);
        }
    }
    let [freed, trimmed] = reached;
    assert!(freed > 0 && trimmed > 0);
}

/// A legacy all-domain clear that names a cache owner is applied on the
/// worker's lane and on the cache owner's lane. Only the lane that owns each
/// owner may act on it, or the slower lane would erase what the cache owner
/// stores after the clear.
#[test]
fn a_clear_leaves_the_owners_of_other_lanes() {
    let chain = |locals: &[u64], hashes: &[u64]| {
        let locals: Vec<_> = locals.iter().copied().map(LocalBlockHash).collect();
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: stored_blocks_with_sequence_hashes(&locals, hashes),
        })
    };
    let index = LowerTierIndexer::new();
    let mut worker_lane = WorkerBlockIndex::default();
    let mut owner_lane = WorkerBlockIndex::default();
    let worker = ResidencyOwner::worker(WorkerWithDpRank::new(29, 0));
    let source = cache_owner(1);
    let owner = ResidencyOwner::cache_owner(source);
    let clear = RouterEvent {
        worker_id: 29,
        state_source: Some(source),
        session_id: None,
        storage_tier: StorageTier::Device,
        residency_domain: WireResidencyDomain::default(),
        event: KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Cleared,
            dp_rank: 0,
        },
    };

    let apply = |lane: &mut WorkerBlockIndex, event: RouterEvent| {
        index.apply_event(lane, event).unwrap();
    };
    apply(&mut worker_lane, event(worker, chain(&[1, 2], &[101, 102])));
    apply(&mut owner_lane, event(owner, chain(&[1, 2], &[101, 102])));
    // The cache owner's lane is past the clear before the worker's lane
    // reaches it.
    apply(&mut owner_lane, clear.clone());
    apply(&mut owner_lane, event(owner, chain(&[1], &[101])));
    apply(&mut worker_lane, clear);

    let key = TransitionKey::new(None, LocalBlockHash(1));
    let held = (
        ExternalSequenceBlockHash(101),
        FxHashSet::from_iter([IndexedResidencyOwner::from_exact(owner)]),
    );
    assert_eq!(run_edges(&index), Edges::from_iter([(key, held)]));
}
