// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustc_hash::FxHashSet;

use crate::rank::Health;
use crate::snapshot::SnapshotImage;
use crate::*;

const GPU: Tier = Tier::new(0).unwrap();
const CPU: Tier = Tier::new(1).unwrap();
const BATCH_BYTES: usize = 100;

/// A stand-in for an engine's wire event. Hashes are plain numbers.
#[derive(Debug, Clone)]
enum Ev {
    Store {
        tier: Tier,
        parent: Parent,
        blocks: Vec<u64>,
    },
    Remove {
        tier: Tier,
        hashes: Vec<u64>,
    },
    Clear(TierSet),
    ClearAll,
}

type Batch = Arc<Vec<Ev>>;
/// SGLang-shaped payload: wide token ids, cheap to clone.
type Tokens = Arc<[i64]>;
type Stream<I = NoIndex> = RankStream<Batch, Tokens, I>;

fn content(hash: u64) -> ContentHash {
    ContentHash(hash + 1_000)
}

fn key(hash: u64) -> BlockKey {
    BlockKey::new(BlockHash(hash), content(hash))
}

fn root(tier: Tier, blocks: &[u64]) -> Ev {
    Ev::Store {
        tier,
        parent: Parent::Root,
        blocks: blocks.to_vec(),
    }
}

fn under(tier: Tier, parent: u64, blocks: &[u64]) -> Ev {
    Ev::Store {
        tier,
        parent: Parent::Block(BlockHash(parent)),
        blocks: blocks.to_vec(),
    }
}

fn remove(tier: Tier, hashes: &[u64]) -> Ev {
    Ev::Remove {
        tier,
        hashes: hashes.to_vec(),
    }
}

/// The host's fold: one writer call per engine event.
fn fold<I: StreamIndex>(
    batch: &Batch,
    writer: &mut BatchWriter<'_, Tokens, I>,
) -> Result<(), Infallible> {
    for event in batch.iter() {
        match event {
            Ev::Store {
                tier,
                parent,
                blocks,
            } => writer.store(
                *tier,
                *parent,
                blocks.iter().map(|hash| key(*hash)),
                |_, key| Arc::from([key.hash.0 as i64, -(key.hash.0 as i64)]),
            ),
            Ev::Remove { tier, hashes } => {
                writer.remove(*tier, hashes.iter().map(|hash| BlockHash(*hash)))
            }
            Ev::Clear(tiers) => writer.clear(*tiers),
            Ev::ClearAll => writer.clear_all(),
        }
    }
    Ok(())
}

fn config(counted: TierSet) -> StreamConfig {
    StreamConfig {
        counted,
        ..StreamConfig::default()
    }
}

fn stream(counted: TierSet) -> Stream {
    RankStream::new(config(counted), Epoch(1), 0, NoIndex)
}

fn apply<I: StreamIndex>(stream: &mut Stream<I>, events: Vec<Ev>) -> Applied {
    let seq = stream.cursor().next_seq;
    stream
        .apply_batch(seq, Arc::new(events), BATCH_BYTES, fold)
        .unwrap()
}

fn count(stream: &Stream<impl StreamIndex>, hash: u64, tier: Tier) -> u16 {
    stream
        .ledger()
        .get(BlockHash(hash))
        .map_or(0, |block| block.counts.get(tier))
}

#[test]
fn set_residency_tracks_each_tier() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1, 2]), root(CPU, &[1])]);
    assert_eq!((count(&stream, 1, GPU), count(&stream, 1, CPU)), (1, 1));

    apply(&mut stream, vec![root(GPU, &[1])]);
    assert_eq!(count(&stream, 1, GPU), 1);
    assert_eq!(stream.ledger().stats().duplicate_stores, 1);

    apply(&mut stream, vec![remove(GPU, &[1])]);
    assert_eq!((count(&stream, 1, GPU), count(&stream, 1, CPU)), (0, 1));
    apply(&mut stream, vec![remove(CPU, &[1])]);
    assert!(!stream.ledger().contains(BlockHash(1)));
    assert_eq!(stream.ledger().len(), 1);
}

#[test]
fn counted_block_stays_until_its_last_remove() {
    let mut stream = stream(TierSet::of(GPU));
    apply(
        &mut stream,
        vec![root(GPU, &[1]), root(GPU, &[1]), root(GPU, &[1])],
    );
    assert_eq!(count(&stream, 1, GPU), 3);

    apply(&mut stream, vec![remove(GPU, &[1]), remove(GPU, &[1])]);
    assert_eq!(count(&stream, 1, GPU), 1);
    apply(&mut stream, vec![remove(GPU, &[1])]);
    assert!(stream.ledger().is_empty());

    apply(&mut stream, vec![remove(GPU, &[1])]);
    assert_eq!(stream.ledger().stats().unknown_removes, 1);
}

#[test]
fn a_store_that_knows_more_fills_in_the_parent() {
    let mut stream = stream(TierSet::EMPTY);
    let orphan = Ev::Store {
        tier: GPU,
        parent: Parent::Unknown,
        blocks: vec![5],
    };
    apply(&mut stream, vec![orphan]);
    let first = stream.ledger().get(BlockHash(5)).unwrap().payload.clone();
    assert_eq!(
        stream.ledger().get(BlockHash(5)).unwrap().parent,
        Parent::Unknown
    );

    apply(&mut stream, vec![under(GPU, 4, &[5])]);
    let block = stream.ledger().get(BlockHash(5)).unwrap();
    assert_eq!(block.parent, Parent::Block(BlockHash(4)));
    assert!(
        Arc::ptr_eq(block.payload, &first),
        "the first payload is kept"
    );

    apply(&mut stream, vec![under(GPU, 7, &[7])]);
    assert_eq!(
        stream.ledger().get(BlockHash(7)).unwrap().parent,
        Parent::Unknown
    );
    assert_eq!(stream.ledger().stats().self_parents, 1);
}

#[test]
fn a_pool_clear_keeps_other_tiers_and_a_total_clear_restores_health() {
    // Joined at batch 5, so stores before it were missed.
    let mut stream: Stream = RankStream::new(config(TierSet::EMPTY), Epoch(1), 5, NoIndex);
    assert!(!stream.info().health.complete);
    apply(&mut stream, vec![root(GPU, &[1, 2]), root(CPU, &[2, 3])]);

    apply(&mut stream, vec![Ev::Clear(TierSet::of(GPU))]);
    assert!(!stream.ledger().contains(BlockHash(1)));
    assert_eq!((count(&stream, 2, GPU), count(&stream, 2, CPU)), (0, 1));
    assert_eq!(count(&stream, 3, CPU), 1);
    assert!(!stream.info().health.complete);

    let applied = apply(&mut stream, vec![Ev::ClearAll]);
    assert!(stream.ledger().is_empty());
    assert_eq!(applied.health, Health::FULL);
    assert!(!applied.degraded, "an improvement is not a degrade");
}

#[test]
fn a_degrade_after_a_total_clear_in_one_batch_is_reported() {
    let config = StreamConfig {
        max_blocks: 2,
        ..StreamConfig::default()
    };
    // Incomplete before the batch and incomplete after it, for a different reason.
    let mut stream: Stream = RankStream::new(config, Epoch(1), 5, NoIndex);
    let applied = apply(&mut stream, vec![Ev::ClearAll, root(GPU, &[1, 2, 3])]);
    assert!(!applied.health.complete);
    assert!(applied.degraded);
}

#[test]
fn naming_every_slot_is_not_a_total_clear() {
    let mut stream: Stream = RankStream::new(config(TierSet::EMPTY), Epoch(1), 5, NoIndex);
    apply(&mut stream, vec![root(GPU, &[1])]);
    apply(&mut stream, vec![Ev::Clear(TierSet::ALL)]);
    assert!(stream.ledger().is_empty());
    assert!(!stream.info().health.complete);
}

#[test]
fn stale_and_gap_apply_nothing() {
    let mut stream: Stream = RankStream::new(config(TierSet::EMPTY), Epoch(1), 3, NoIndex);
    let batch = || Arc::new(vec![root(GPU, &[1])]);

    let stale = stream.apply_batch(2, batch(), BATCH_BYTES, fold);
    assert!(matches!(
        stale,
        Err(ApplyError::Stale {
            expected: 3,
            got: 2,
            ..
        })
    ));
    let gap = stream.apply_batch(5, batch(), BATCH_BYTES, fold);
    assert!(matches!(
        gap,
        Err(ApplyError::Gap {
            expected: 3,
            got: 5,
            ..
        })
    ));
    assert!(stream.ledger().is_empty());
    assert_eq!(stream.cursor().next_seq, 3);

    // A stream told to expect batch 0 treats a late first batch as a gap too.
    let mut fresh = self::stream(TierSet::EMPTY);
    let late = fresh.apply_batch(1, batch(), BATCH_BYTES, fold);
    assert!(matches!(
        late,
        Err(ApplyError::Gap {
            expected: 0,
            got: 1,
            ..
        })
    ));
}

#[test]
fn a_new_epoch_drops_or_keeps_state() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1, 2])]);
    let old = stream.cursor();

    assert_eq!(
        stream.start_epoch(Epoch(1), 10, Carry::Keep),
        Err(SameEpoch)
    );
    assert_eq!(stream.resume(Some(old)), Ok(1));
    stream.start_epoch(Epoch(2), 10, Carry::Keep).unwrap();
    assert_eq!(stream.ledger().len(), 2);
    assert_eq!(stream.info().health, Health::UNKNOWN);
    assert_eq!(
        stream.cursor(),
        Cursor {
            epoch: Epoch(2),
            next_seq: 10
        }
    );
    assert_eq!(
        stream.resume(Some(old)),
        Err(Refusal::WrongEpoch { current: Epoch(2) })
    );

    stream.start_epoch(Epoch(3), 0, Carry::Drop).unwrap();
    assert!(stream.ledger().is_empty());
    assert_eq!(stream.info().health, Health::FULL);
}

#[test]
fn a_failed_fold_poisons_the_stream() {
    let mut stream = stream(TierSet::EMPTY);
    let failed = stream.apply_batch(0, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
        writer.store(GPU, Parent::Root, [key(1)], |_, _| Arc::from([1]));
        Err("cannot decode the second event")
    });
    assert!(matches!(failed, Err(ApplyError::Fold { .. })));

    let refused = stream.apply_batch(0, Arc::new(vec![]), BATCH_BYTES, fold);
    assert!(matches!(refused, Err(ApplyError::Poisoned { .. })));
    assert_eq!(stream.resume(Some(stream.cursor())), Err(Refusal::Poisoned));
    assert!(matches!(stream.snapshot(), Err(Refusal::Poisoned)));
    assert!(matches!(
        stream.admit(None, OnMiss::LiveOnly),
        Err(Refusal::Poisoned)
    ));

    stream.start_epoch(Epoch(2), 0, Carry::Drop).unwrap();
    apply(&mut stream, vec![root(GPU, &[1])]);
    assert_eq!(stream.ledger().len(), 1);
}

#[test]
fn a_panicking_fold_poisons_the_stream() {
    let mut stream = stream(TierSet::EMPTY);
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let _ = stream.apply_batch(
            0,
            Arc::new(vec![]),
            BATCH_BYTES,
            |_, _| -> Result<(), Infallible> { panic!("fold bug") },
        );
    }));
    assert!(unwound.is_err());
    assert!(stream.info().poisoned);
}

#[test]
fn history_caps_by_count_and_bytes_and_keeps_the_newest() {
    let mut history = History::new(0, 3, 100);
    for batch in 0..5u32 {
        history.push(batch, 10);
    }
    assert_eq!((history.oldest(), history.next()), (2, 5));
    assert_eq!(history.bytes_from(0), 30);
    assert_eq!(history.bytes_from(4), 10);
    assert_eq!(history.bytes_from(5), 0);
    assert_eq!(history.range(3).copied().collect::<Vec<_>>(), [3, 4]);

    history.push(5, 500);
    assert_eq!((history.oldest(), history.next()), (5, 6));
    assert_eq!(history.bytes_from(5), 500);
    assert!(history.covers(5) && history.covers(6) && !history.covers(4));
    assert_eq!(history.len(), 1);
}

fn short_history() -> Stream {
    let config = StreamConfig {
        history_max_batches: 2,
        ..StreamConfig::default()
    };
    let mut stream = RankStream::new(config, Epoch(1), 0, NoIndex);
    for hash in 1..=4 {
        apply(&mut stream, vec![root(GPU, &[hash])]);
    }
    stream
}

#[test]
fn resume_classifies_cursors() {
    let stream = short_history();
    let at = |next_seq| {
        Some(Cursor {
            epoch: Epoch(1),
            next_seq,
        })
    };

    assert_eq!(stream.resume(None), Err(Refusal::NoCursor));
    assert_eq!(
        stream.resume(Some(Cursor {
            epoch: Epoch(7),
            next_seq: 3
        })),
        Err(Refusal::WrongEpoch { current: Epoch(1) })
    );
    assert_eq!(stream.resume(at(1)), Err(Refusal::Evicted { oldest: 2 }));
    assert_eq!(stream.resume(at(2)), Ok(2));
    assert_eq!(stream.resume(at(4)), Ok(4));
    assert_eq!(stream.resume(at(5)), Err(Refusal::Ahead { next: 4 }));
}

#[test]
fn admit_follows_on_miss() {
    let mut stream = short_history();
    let covered = Some(Cursor {
        epoch: Epoch(1),
        next_seq: 3,
    });
    let evicted = Some(Cursor {
        epoch: Epoch(1),
        next_seq: 0,
    });

    let Ok(Admission::Events { from_seq, health }) = stream.admit(covered, OnMiss::Fail) else {
        panic!("a covered cursor continues");
    };
    assert_eq!(health, Health::FULL);
    assert_eq!(stream.backlog(from_seq).count(), 1);

    assert!(matches!(
        stream.admit(evicted, OnMiss::Fail),
        Err(Refusal::Evicted { oldest: 2 })
    ));
    let Ok(Admission::Snapshot { image }) = stream.admit(evicted, OnMiss::Snapshot) else {
        panic!("an evicted cursor gets a snapshot");
    };
    assert_eq!(image.cursor(), stream.cursor());
    assert_eq!(image.len(), 4);
    assert!(matches!(
        stream.admit(None, OnMiss::LiveOnly),
        Ok(Admission::LiveOnly { next_seq: 4 })
    ));

    // Three more batches move the two-batch window past the image's cut.
    for hash in 5..=7 {
        apply(&mut stream, vec![root(GPU, &[hash])]);
    }
    let later = stream.snapshot().unwrap();
    assert_eq!(stream.resume(Some(later.cursor())), Ok(7));
}

/// A small deterministic generator, so the property test below needs no extra dependency.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }

    fn event(&mut self) -> Ev {
        let tier = if self.below(4) == 0 { CPU } else { GPU };
        let hash = 1 + self.below(30);
        match self.below(20) {
            0 => Ev::Clear(TierSet::of(tier)),
            1 => Ev::ClearAll,
            2..=8 => remove(tier, &[hash, 1 + self.below(30)]),
            9..=11 => root(tier, &[hash, hash + 1]),
            _ => under(tier, hash, &[hash + 1, hash + 2, hash + 3]),
        }
    }
}

/// Loads a snapshot the way a remote consumer does: from its runs, not from the frozen ledger.
/// Also checks the export order: a run that is not an orphan comes after its parent.
fn load_runs(image: &SnapshotImage<Tokens>, counted: TierSet) -> Ledger<Tokens> {
    let mut loaded = Ledger::new(counted);
    let mut seen = FxHashSet::default();
    for run in image.runs() {
        match run.parent {
            Parent::Block(parent) if !run.orphan => {
                assert!(seen.contains(&parent), "parent {parent:?} exported first")
            }
            _ => {}
        }
        let mut parent = run.parent;
        for block in image.run_blocks(run) {
            loaded.load_block(block.key, parent, run.counts, block.payload.clone());
            seen.insert(block.key.hash);
            parent = Parent::Block(block.key.hash);
        }
    }
    assert_eq!(seen.len(), image.len(), "every block is exported once");
    loaded
}

#[test]
fn snapshot_then_tail_equals_following_from_the_start() {
    for seed in 0..20 {
        for counted in [TierSet::EMPTY, TierSet::of(GPU)] {
            let mut random = Lcg(seed);
            let mut leader = stream(counted);
            let cut = random.below(40);
            let mut image = None;
            for batch in 0..40 {
                if batch == cut {
                    image = Some(leader.snapshot().unwrap());
                }
                let events = (0..1 + random.below(4)).map(|_| random.event()).collect();
                apply(&mut leader, events);
            }
            let image = image.unwrap();
            assert_eq!(image.cursor().next_seq, cut);

            // The consumer loads the image, then follows the batches from the cut.
            let loaded = load_runs(&image, counted);
            assert!(loaded == *image.ledger(), "runs carry the whole image");
            let mut consumer: Stream = RankStream::seeded(
                config(counted),
                image.cursor(),
                loaded,
                image.health(),
                NoIndex,
            );
            let tail = leader
                .backlog(image.cursor().next_seq)
                .cloned()
                .collect::<Vec<_>>();
            for (offset, batch) in tail.into_iter().enumerate() {
                consumer
                    .apply_batch(cut + offset as u64, batch, BATCH_BYTES, fold)
                    .unwrap();
            }
            assert!(
                consumer.ledger() == leader.ledger(),
                "seed {seed}, counted {counted:?}"
            );
            assert_eq!(consumer.cursor(), leader.cursor());
        }
    }
}

#[test]
fn runs_split_where_chains_branch_or_counts_change() {
    let mut stream = stream(TierSet::EMPTY);
    apply(
        &mut stream,
        vec![
            root(GPU, &[1, 2, 3]),
            // Block 3 is the only child of 2 but is also on the CPU tier: the counts change.
            root(CPU, &[3]),
            under(GPU, 3, &[4]),
            under(GPU, 3, &[5]),
            root(GPU, &[10]),
        ],
    );
    let image = stream.snapshot().unwrap();
    let runs = image
        .runs()
        .iter()
        .map(|run| {
            let hashes = image.run_keys(run).iter().map(|key| key.hash.0).collect();
            (run.parent, hashes)
        })
        .collect::<Vec<(Parent, Vec<u64>)>>();
    let after = |hash| Parent::Block(BlockHash(hash));
    assert_eq!(
        runs,
        [
            (Parent::Root, vec![1, 2]),
            (Parent::Root, vec![10]),
            (after(2), vec![3]),
            (after(3), vec![4]),
            (after(3), vec![5]),
        ]
    );
    load_runs(&image, TierSet::EMPTY);
}

#[test]
fn a_mid_chain_remove_exports_the_rest_as_an_orphan() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1, 2, 3]), remove(GPU, &[2])]);
    let image = stream.snapshot().unwrap();
    let [head, rest] = image.runs() else {
        panic!("two runs");
    };
    assert!(!head.orphan);
    assert!(rest.orphan);
    assert_eq!(rest.parent, Parent::Block(BlockHash(2)));
    assert_eq!(image.run_keys(rest), [key(3)]);
}

#[test]
fn blocks_on_a_parent_cycle_are_still_exported() {
    let mut ledger: Ledger = Ledger::new(TierSet::EMPTY);
    let held = TierCounts::default().with(GPU, 1);
    ledger.load_block(key(1), Parent::Block(BlockHash(2)), held, ());
    ledger.load_block(key(2), Parent::Block(BlockHash(1)), held, ());
    // Two blocks that hang off the cycle.
    ledger.load_block(key(3), Parent::Block(BlockHash(1)), held, ());
    ledger.load_block(key(4), Parent::Block(BlockHash(3)), held, ());
    let cursor = Cursor {
        epoch: Epoch(1),
        next_seq: 0,
    };
    let image = SnapshotImage::new(cursor, Health::FULL, ledger);
    assert_eq!(image.runs().iter().map(SnapshotRun::len).sum::<usize>(), 4);
    // The export cuts in at the lowest hash, which here is on the cycle: that block is the
    // only orphan and the rest follow their parents.
    let orphans = image
        .runs()
        .iter()
        .filter(|run| run.orphan)
        .map(|run| image.run_keys(run))
        .collect::<Vec<_>>();
    assert_eq!(orphans, [vec![key(1)]]);
}

#[test]
fn chunk_by_respects_the_cap() {
    let sizes = [3usize, 3, 3, 10, 1];
    let chunks = chunk_by(&sizes, 6, |size| *size).collect::<Vec<_>>();
    assert_eq!(chunks, [&[3, 3][..], &[3], &[10], &[1]]);
}

#[test]
fn a_snapshot_is_shared_while_its_tail_is_cheap() {
    let config = StreamConfig {
        snapshot_tail_max_bytes: 250,
        ..StreamConfig::default()
    };
    let mut stream: Stream = RankStream::new(config, Epoch(1), 0, NoIndex);
    apply(&mut stream, vec![root(GPU, &[1])]);
    let first = stream.snapshot().unwrap();

    apply(&mut stream, vec![root(GPU, &[2])]);
    apply(&mut stream, vec![root(GPU, &[3])]);
    let shared = stream.snapshot().unwrap();
    assert!(Arc::ptr_eq(&first, &shared));
    assert_eq!(stream.backlog(shared.cursor().next_seq).count(), 2);

    // A third batch puts 300 bytes after the cut: a fresh image is cheaper to load.
    apply(&mut stream, vec![root(GPU, &[4])]);
    let fresh = stream.snapshot().unwrap();
    assert!(!Arc::ptr_eq(&first, &fresh));
    assert_eq!(fresh.len(), 4);

    // Losing health also retires the image: its tail cannot say so.
    let degraded = stream
        .apply_batch(4, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
            writer.mark_inexact();
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert!(degraded.degraded);
    assert!(!stream.snapshot().unwrap().health().exact);
    assert_eq!(stream.stats().snapshots_built, 3);
    assert_eq!(stream.stats().snapshots_reused, 1);
}

#[test]
fn a_total_clear_retires_the_cached_image() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1, 2, 3])]);
    let full = stream.snapshot().unwrap();
    // Health is unchanged and the tail is tiny, but the image is now all stale.
    apply(&mut stream, vec![Ev::ClearAll]);
    let empty = stream.snapshot().unwrap();
    assert!(!Arc::ptr_eq(&full, &empty));
    assert!(empty.is_empty());
}

#[test]
fn memo_builds_once() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1, 2])]);
    let image = stream.snapshot().unwrap();
    let builds = AtomicUsize::new(0);
    let encode = |image: &SnapshotImage<Tokens>| {
        builds.fetch_add(1, Ordering::Relaxed);
        vec![0u8; image.len()]
    };
    assert_eq!(image.memo(encode).map(Vec::len), Some(2));
    assert_eq!(image.memo(encode).map(Vec::len), Some(2));
    assert_eq!(builds.load(Ordering::Relaxed), 1);
    assert!(image.memo(|_| String::new()).is_none());
}

#[test]
fn the_block_cap_stops_growth_and_reports_it() {
    let config = StreamConfig {
        max_blocks: 2,
        ..StreamConfig::default()
    };
    let mut stream: Stream = RankStream::new(config, Epoch(1), 0, NoIndex);
    let applied = apply(&mut stream, vec![root(GPU, &[1, 2, 3])]);
    assert_eq!(stream.ledger().len(), 2);
    assert!(applied.degraded && !applied.health.complete);
}

#[test]
fn a_reseed_replaces_state_and_reapplies_the_retained_batches() {
    let batches = [
        vec![root(GPU, &[1, 2])],
        vec![under(GPU, 2, &[3])],
        vec![remove(GPU, &[3]), root(GPU, &[7])],
        vec![under(GPU, 2, &[4])],
    ];
    // The engine's own view: everything from batch 0.
    let mut truth = stream(TierSet::EMPTY);
    for batch in &batches[..2] {
        apply(&mut truth, batch.clone());
    }
    let engine_view_before_2 = truth.ledger().clone();
    for batch in &batches[2..] {
        apply(&mut truth, batch.clone());
    }

    // The frontend restarted and joined at batch 2.
    let mut late: Stream = RankStream::new(config(TierSet::EMPTY), Epoch(1), 2, NoIndex);
    for batch in &batches[2..] {
        apply(&mut late, batch.clone());
    }
    let old = late.cursor();

    assert_eq!(
        late.reseed(Epoch(1), 2, late.empty_ledger(), Health::FULL, fold),
        Err(ReseedError::SameEpoch)
    );
    assert_eq!(
        late.reseed(Epoch(2), 1, late.empty_ledger(), Health::FULL, fold),
        Err(ReseedError::Evicted {
            oldest: 2,
            at_seq: 1
        })
    );
    late.reseed(Epoch(2), 2, engine_view_before_2, Health::FULL, fold)
        .unwrap();
    assert!(late.ledger() == truth.ledger());
    assert_eq!(late.info().health, Health::FULL);
    assert_eq!(late.cursor().next_seq, 4);
    assert_eq!(
        late.resume(Some(old)),
        Err(Refusal::WrongEpoch { current: Epoch(2) })
    );
}

#[test]
fn tier_map_assigns_slots_until_they_run_out() {
    let mut tiers = TierMap::new();
    let gpu = tiers.intern(&("GPU", "LOCAL")).unwrap();
    assert_eq!(tiers.intern(&("GPU", "LOCAL")), Some(gpu));
    for scope in ["a", "b", "c", "d", "e", "f", "g"] {
        assert!(tiers.intern(&("CPU", scope)).is_some());
    }
    assert_eq!(tiers.len(), MAX_TIERS);
    assert_eq!(tiers.intern(&("DISK", "LOCAL")), None);
    assert_eq!(tiers.key(gpu), Some(&("GPU", "LOCAL")));

    // A host that holds the scope as borrowed strings builds an owned key only on first sight.
    let mut owned: TierMap<(String, u32)> = TierMap::new();
    let lookup = |tiers: &mut TierMap<(String, u32)>, medium: &str, group: u32| {
        tiers.intern_with(
            |known| known.0 == medium && known.1 == group,
            || (medium.to_owned(), group),
        )
    };
    let pool = lookup(&mut owned, "GPU", 0).unwrap();
    assert_ne!(lookup(&mut owned, "GPU", 1), Some(pool));
    assert_eq!(lookup(&mut owned, "GPU", 0), Some(pool));
    assert_eq!(owned.len(), 2);
}

#[test]
fn engine_hashes_narrow_to_64_bits() {
    let digest = (0u8..32).collect::<Vec<_>>();
    assert_eq!(
        BlockHash::from_be_bytes(&digest),
        Some(BlockHash(u64::from_be_bytes([
            24, 25, 26, 27, 28, 29, 30, 31
        ])))
    );
    assert_eq!(BlockHash::from_be_bytes(&[1, 2]), Some(BlockHash(0x0102)));
    assert_eq!(BlockHash::from_be_bytes(&[]), None);
    assert_eq!(BlockHash::from_i64(-1), BlockHash(u64::MAX));
}

#[test]
fn a_reseed_from_a_newer_view_recovers_a_poisoned_stream() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1])]);
    let failed = stream.apply_batch(1, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
        writer.store(GPU, Parent::Root, [key(2)], |_, _| Arc::from([2]));
        Err("cannot decode the second event")
    });
    assert!(matches!(failed, Err(ApplyError::Fold { .. })));
    assert!(stream.info().poisoned);

    // The engine's view, fetched afterwards, already includes batches 1 and 2.
    let mut view = stream.empty_ledger();
    let held = TierCounts::default().with(GPU, 1);
    view.load_block(key(9), Parent::Root, held, Arc::from([9]));
    stream
        .reseed(Epoch(2), 3, view, Health::FULL, fold)
        .unwrap();
    assert!(!stream.info().poisoned);
    assert_eq!(stream.cursor().next_seq, 3);
    assert_eq!(stream.ledger().len(), 1);

    // Batches the view already covers are stale; the stream continues from the view.
    let late = stream.apply_batch(2, Arc::new(vec![]), BATCH_BYTES, fold);
    assert!(matches!(late, Err(ApplyError::Stale { expected: 3, .. })));
    apply(&mut stream, vec![under(GPU, 9, &[10])]);
    assert_eq!(stream.info().health, Health::FULL);
}

#[test]
fn a_failed_refold_leaves_the_stream_poisoned() {
    let mut late: Stream = RankStream::new(config(TierSet::EMPTY), Epoch(1), 2, NoIndex);
    apply(&mut late, vec![root(GPU, &[1])]);
    apply(&mut late, vec![root(GPU, &[2])]);

    let failed = late.reseed(Epoch(2), 2, late.empty_ledger(), Health::FULL, |_, _| {
        Err("the staged view lacks what this batch needs")
    });
    assert!(matches!(failed, Err(ReseedError::Refold(_))));
    assert!(late.info().poisoned);
    assert!(matches!(late.snapshot(), Err(Refusal::Poisoned)));

    // The failed reseed already moved the stream to its epoch, so recovery needs another.
    assert_eq!(late.start_epoch(Epoch(2), 0, Carry::Drop), Err(SameEpoch));
    late.start_epoch(Epoch(3), 0, Carry::Drop).unwrap();
    assert!(!late.info().poisoned);
}

#[test]
fn a_saturated_count_makes_the_ledger_incomplete() {
    let mut stream: RankStream<(), (), NoIndex> =
        RankStream::new(config(TierSet::of(GPU)), Epoch(1), 0, NoIndex);
    let applied = stream
        .apply_batch(0, (), BATCH_BYTES, |_, writer| {
            for _ in 0..=u16::MAX {
                writer.store(GPU, Parent::Root, [key(1)], |_, _| ());
            }
            Ok::<_, Infallible>(())
        })
        .unwrap();
    // One store was not counted, so the block can go one remove early: under-reporting.
    assert_eq!(stream.ledger().stats().saturated_counts, 1);
    assert_eq!(
        applied.health,
        Health {
            complete: false,
            exact: true
        }
    );
    assert!(applied.degraded);
}

#[test]
fn a_scope_met_mid_stream_can_be_counted() {
    let mut stream = stream(TierSet::of(GPU));
    stream
        .apply_batch(0, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
            // A second block pool shows up after the stream was configured.
            writer.set_counted(CPU);
            for _ in 0..2 {
                writer.store(CPU, Parent::Root, [key(1)], |_, _| Arc::from([1]));
            }
            writer.remove(CPU, [BlockHash(1)]);
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert_eq!(count(&stream, 1, CPU), 1);
    assert!(stream.empty_ledger().counted().contains(CPU));
}

#[test]
fn a_placeholder_payload_is_upgraded_by_the_store_that_knows_it() {
    let mut stream = stream(TierSet::EMPTY);
    // vLLM-shaped: an offload tier announces the block with no tokens, then the pool stores
    // it with them.
    stream
        .apply_batch(0, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
            writer.store(
                CPU,
                Parent::Unknown,
                [BlockKey::opaque(BlockHash(1))],
                |_, _| Arc::from([]),
            );
            let tokens: Tokens = Arc::from([11, 12]);
            let mut fresh = false;
            writer.store(GPU, Parent::Root, [key(1)], |_, _| {
                fresh = true;
                tokens.clone()
            });
            if !fresh
                && let Some(held) = writer.payload_mut(BlockHash(1))
                && held.is_empty()
            {
                *held = tokens;
            }
            Ok::<_, Infallible>(())
        })
        .unwrap();
    let image = stream.snapshot().unwrap();
    let block = image.ledger().get(BlockHash(1)).unwrap();
    assert_eq!(block.key, key(1));
    assert_eq!(block.payload[..], [11, 12]);
    assert_eq!(block.parent, Parent::Root);
}

#[test]
fn load_block_takes_a_set_tier_count_as_one() {
    let mut ledger: Ledger = Ledger::new(TierSet::of(CPU));
    let counts = TierCounts::default().with(GPU, 3).with(CPU, 3);
    ledger.load_block(key(1), Parent::Root, counts, ());
    let held = ledger.get(BlockHash(1)).unwrap().counts;
    assert_eq!((held.get(GPU), held.get(CPU)), (1, 3));
    assert_eq!(
        ledger.remove(GPU, BlockHash(1)),
        Removed::Last { gone: false }
    );
}

// A snapshot is ordered and encoded on another thread, and a host may move a stream between
// tasks.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    const fn movable<T: Send>() {}
    shared::<SnapshotImage<Tokens>>();
    movable::<Stream>();
};

#[test]
fn a_consumer_that_resumes_through_a_degrade_is_given_the_health() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1])]);
    let before = stream.cursor();
    let image = stream.snapshot().unwrap();
    // A degrade the batch object cannot carry: the host could not record one of its events.
    stream
        .apply_batch(1, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
            writer.mark_inexact();
            Ok::<_, Infallible>(())
        })
        .unwrap();
    let now = Health {
        complete: true,
        exact: false,
    };

    // A cursor from before the degrade is still covered, and replay alone would leave the
    // consumer believing the ledger exact.
    let admitted = stream.admit(Some(before), OnMiss::Fail).unwrap();
    assert!(matches!(admitted, Admission::Events { from_seq: 1, health } if health == now));
    // The same holds for a consumer still loading an image cut before the degrade.
    assert_eq!(image.health(), Health::FULL);
    assert_eq!(stream.resume(Some(image.cursor())), Ok(1));
    assert_eq!(stream.info().health, now);
}

#[test]
fn a_host_can_release_the_cached_image() {
    let mut stream = stream(TierSet::EMPTY);
    apply(&mut stream, vec![root(GPU, &[1])]);
    let image = stream.snapshot().unwrap();
    let released = stream.release_snapshot().unwrap();
    assert!(Arc::ptr_eq(&image, &released));
    assert!(stream.release_snapshot().is_none());
    assert!(!Arc::ptr_eq(&image, &stream.snapshot().unwrap()));
}
