// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A rank's stream against the crate's own indexes.

use std::convert::Infallible;
use std::sync::Arc;

use crate::prelude::*;
use crate::stream::{
    Applied, BatchWriter, BlockHash, BlockKey, Carry, ContentHash, Epoch, Health, LaneIndex,
    NoIndex, Parent, RadixIndex, RankStream, StreamConfig, StreamIndex, Tier, TierCounts, TierSet,
    image_to_router_events,
};
use crate::{ConcurrentRadixTreeCompressed, RadixTree, SyncIndexer, ThreadPoolIndexer, testing};

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
}

type Batch = Arc<Vec<Ev>>;
type Tokens = Arc<[i64]>;
type Stream<I> = RankStream<Batch, Tokens, I>;

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
                |_, key| Arc::from([key.hash.0 as i64]),
            ),
            Ev::Remove { tier, hashes } => {
                writer.remove(*tier, hashes.iter().map(|hash| BlockHash(*hash)))
            }
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

fn worker() -> WorkerWithDpRank {
    WorkerWithDpRank::new(7, 0)
}

fn local(hashes: &[u64]) -> Vec<LocalBlockHash> {
    hashes
        .iter()
        .map(|hash| LocalBlockHash(content(*hash).0))
        .collect()
}

fn store_event(event_id: u64, parent: Option<u64>, blocks: &[u64]) -> RouterEvent {
    let data = KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: parent.map(ExternalSequenceBlockHash),
        start_position: None,
        blocks: testing::stored_blocks_with_sequence_hashes(&local(blocks), blocks),
    });
    testing::router_event(7, event_id, 0, data)
}

fn remove_event(event_id: u64, hashes: &[u64]) -> RouterEvent {
    let block_hashes = hashes
        .iter()
        .copied()
        .map(ExternalSequenceBlockHash)
        .collect();
    testing::remove_event(7, event_id, 0, block_hashes)
}

/// vLLM-shaped stream (duplicate blocks, one remove per block, a mid-chain eviction, then a
/// re-store) against a tree fed what `EventDedupFilter` passes under its ref-counted policy:
/// every store, and a remove only when the last duplicate goes.
#[test]
fn radix_index_matches_a_tree_fed_through_the_repeat_store_filter() {
    let mut stream: Stream<RadixIndex> = RankStream::new(
        config(TierSet::of(GPU)),
        Epoch(1),
        0,
        RadixIndex::new(worker(), GPU),
    );
    apply(
        &mut stream,
        vec![root(GPU, &[1, 2, 3]), root(GPU, &[1, 2, 3])],
    );
    // Two removes of the duplicated block 2: only the second reaches the tree, which then
    // forgets block 3 although the engine still holds it.
    apply(&mut stream, vec![remove(GPU, &[2]), remove(GPU, &[2])]);
    assert_eq!(
        stream
            .index()
            .tree()
            .find_matches(local(&[1, 2, 3]), false)
            .scores[&worker()],
        1
    );
    // The engine caches the suffix again. The repeated store of 3 must reach the tree.
    apply(&mut stream, vec![under(GPU, 1, &[2, 3])]);

    let mut today = RadixTree::new();
    for event in [
        store_event(0, None, &[1, 2, 3]),
        store_event(1, None, &[1, 2, 3]),
        remove_event(2, &[2]),
        store_event(3, Some(1), &[2, 3]),
    ] {
        today.apply_event(event).unwrap();
    }

    let query = local(&[1, 2, 3]);
    let expected = today.find_matches(query.clone(), false);
    assert_eq!(expected.scores[&worker()], 3);
    assert_eq!(
        stream
            .index()
            .tree()
            .find_matches(query.clone(), false)
            .scores,
        expected.scores
    );
    assert_eq!(stream.index().rejected(), 0);
    assert_eq!(
        (
            count(&stream, 1, GPU),
            count(&stream, 2, GPU),
            count(&stream, 3, GPU)
        ),
        (2, 1, 3)
    );

    // The snapshot, as a router's TreeDump, rebuilds the same tree.
    let image = stream.snapshot().unwrap();
    let (events, left_out) = image_to_router_events(&image, worker(), GPU);
    assert_eq!(left_out, 0);
    let mut rebuilt = RadixTree::new();
    for event in events {
        rebuilt.apply_event(event).unwrap();
    }
    assert_eq!(rebuilt.find_matches(query, false).scores, expected.scores);
}

#[test]
fn radix_index_leaves_out_what_a_trie_cannot_hold() {
    let mut stream: RankStream<Batch, (), RadixIndex> = RankStream::new(
        config(TierSet::EMPTY),
        Epoch(1),
        0,
        RadixIndex::new(worker(), GPU),
    );
    stream
        .apply_batch(0, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
            // No content hash on the second block; another tier; an unknown parent.
            writer.store(
                GPU,
                Parent::Root,
                [key(1), BlockKey::opaque(BlockHash(2)), key(3)],
                |_, _| (),
            );
            writer.store(CPU, Parent::Root, [key(20)], |_, _| ());
            writer.store(GPU, Parent::Unknown, [key(30)], |_, _| ());
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert_eq!(stream.ledger().len(), 5);
    assert_eq!(stream.index().tree().current_size(), 1);
    assert_eq!(stream.index().skipped(), 3);
    assert_eq!(stream.index().rejected(), 0);
}

#[tokio::test]
async fn lane_index_reaches_the_concurrent_tree_through_its_lane() {
    let pool = Arc::new(ThreadPoolIndexer::new(
        ConcurrentRadixTreeCompressed::new(),
        2,
        16,
    ));
    let mut stream: Stream<LaneIndex<ConcurrentRadixTreeCompressed>> = RankStream::new(
        config(TierSet::of(GPU)),
        Epoch(1),
        0,
        LaneIndex::new(pool.clone(), worker(), GPU),
    );
    apply(
        &mut stream,
        vec![root(GPU, &[1, 2, 3]), root(GPU, &[1, 2, 3])],
    );
    apply(&mut stream, vec![remove(GPU, &[3]), under(GPU, 3, &[4])]);
    pool.flush_and_wait().await.unwrap();

    let scores = pool.backend().find_matches(&local(&[1, 2, 3, 4]), false);
    assert_eq!(scores.scores[&worker()], 4);
    assert_eq!(stream.index().offline(), 0);
}

fn radix_stream(config: StreamConfig, first_seq: u64) -> Stream<RadixIndex> {
    RankStream::new(config, Epoch(1), first_seq, RadixIndex::new(worker(), GPU))
}

fn score(stream: &Stream<RadixIndex>, hashes: &[u64]) -> u32 {
    let scores = stream.index().tree().find_matches(local(hashes), false);
    scores.scores.get(&worker()).copied().unwrap_or(0)
}

#[test]
fn a_reseed_rebuilds_the_index_from_the_staged_ledger() {
    let batches = [
        vec![root(GPU, &[1, 2])],
        vec![under(GPU, 2, &[3])],
        vec![remove(GPU, &[3]), root(GPU, &[7])],
        vec![under(GPU, 2, &[4])],
    ];
    let mut truth = radix_stream(config(TierSet::EMPTY), 0);
    for batch in &batches[..2] {
        apply(&mut truth, batch.clone());
    }
    let engine_view_before_2 = truth.ledger().clone();
    for batch in &batches[2..] {
        apply(&mut truth, batch.clone());
    }

    // Joined at batch 2: the tree never saw blocks 1 and 2, so it refuses the store under 2.
    let mut late = radix_stream(config(TierSet::EMPTY), 2);
    for batch in &batches[2..] {
        apply(&mut late, batch.clone());
    }
    assert_eq!(score(&late, &[1, 2, 4]), 0);

    late.reseed(Epoch(2), 2, engine_view_before_2, Health::FULL, fold)
        .unwrap();
    assert!(late.ledger() == truth.ledger());
    for query in [&[1, 2, 4][..], &[1, 2, 3], &[7]] {
        assert_eq!(score(&late, query), score(&truth, query), "{query:?}");
    }
    assert_eq!(score(&late, &[1, 2, 4]), 3);
}

#[test]
fn a_snapshot_dump_leaves_out_what_a_trie_cannot_hold() {
    let mut stream: RankStream<Batch, (), NoIndex> =
        RankStream::new(config(TierSet::EMPTY), Epoch(1), 0, NoIndex);
    stream
        .apply_batch(0, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
            // Cut at block 2, which has no content hash; block 4 hangs off the part cut away.
            writer.store(
                GPU,
                Parent::Root,
                [key(1), BlockKey::opaque(BlockHash(2)), key(3)],
                |_, _| (),
            );
            writer.store(GPU, Parent::Block(BlockHash(3)), [key(4)], |_, _| ());
            writer.store(CPU, Parent::Root, [key(20)], |_, _| ());
            writer.store(GPU, Parent::Unknown, [key(30)], |_, _| ());
            Ok::<_, Infallible>(())
        })
        .unwrap();
    let image = stream.snapshot().unwrap();
    let (events, left_out) = image_to_router_events(&image, worker(), GPU);
    assert_eq!((events.len(), left_out), (1, 5));
    let mut rebuilt = RadixTree::new();
    for event in events {
        rebuilt.apply_event(event).unwrap();
    }
    assert_eq!(rebuilt.current_size(), 1);
}

/// The ledger keeps chaining block 3 after block 2 is evicted and stored again alone; the live
/// tree re-covers only what the store named. A dump of the snapshot is the larger, true view.
#[test]
fn a_snapshot_dump_can_hold_more_than_the_live_index() {
    let mut stream = radix_stream(config(TierSet::EMPTY), 0);
    apply(&mut stream, vec![root(GPU, &[1, 2, 3])]);
    apply(&mut stream, vec![remove(GPU, &[2])]);
    apply(&mut stream, vec![under(GPU, 1, &[2])]);
    assert_eq!(score(&stream, &[1, 2, 3]), 2);

    let image = stream.snapshot().unwrap();
    let (events, left_out) = image_to_router_events(&image, worker(), GPU);
    assert_eq!(left_out, 0);
    let mut rebuilt = RadixTree::new();
    for event in events {
        rebuilt.apply_event(event).unwrap();
    }
    assert_eq!(
        rebuilt.find_matches(local(&[1, 2, 3]), false).scores[&worker()],
        3
    );
}

#[test]
fn the_block_cap_bounds_the_index_too() {
    let config = StreamConfig {
        max_blocks: 2,
        ..StreamConfig::default()
    };
    let mut stream = radix_stream(config, 0);
    apply(&mut stream, vec![root(GPU, &[1, 2, 3, 4])]);
    assert_eq!(stream.ledger().len(), 2);
    assert_eq!(stream.index().tree().current_size(), 2);
}

#[tokio::test]
async fn a_store_that_repeats_a_hash_never_reaches_the_concurrent_tree() {
    let pool = Arc::new(ThreadPoolIndexer::new(
        ConcurrentRadixTreeCompressed::new(),
        2,
        16,
    ));
    let mut stream: Stream<LaneIndex<ConcurrentRadixTreeCompressed>> = RankStream::new(
        config(TierSet::of(GPU)),
        Epoch(1),
        0,
        LaneIndex::new(pool.clone(), worker(), GPU),
    );
    apply(&mut stream, vec![root(GPU, &[1, 2])]);
    // A store that names its own parent, and one that names a block twice.
    apply(
        &mut stream,
        vec![
            under(GPU, 2, &[2, 3]),
            under(GPU, 2, &[5, 5]),
            under(GPU, 2, &[2]),
        ],
    );
    pool.flush_and_wait().await.unwrap();

    assert_eq!(stream.index().skipped(), 5);
    let scores = pool.backend().find_matches(&local(&[1, 2, 2]), false);
    assert_eq!(scores.scores[&worker()], 2);
}

/// The gap case: the stream missed the batch that removed block 8, so only the reseed's clear
/// of the index can make the tree forget it.
#[test]
fn a_reseed_past_a_gap_forgets_what_the_index_held() {
    let mut late = radix_stream(config(TierSet::EMPTY), 2);
    apply(&mut late, vec![root(GPU, &[8])]);
    assert_eq!(score(&late, &[8]), 1);

    // The engine's view as of just before batch 4: batch 3, which the stream never saw,
    // removed block 8.
    let mut view = late.empty_ledger();
    let mut parent = Parent::Root;
    for hash in [1, 2, 3] {
        let held = TierCounts::default().with(GPU, 1);
        view.load_block(key(hash), parent, held, Arc::from([hash as i64]));
        parent = Parent::Block(BlockHash(hash));
    }
    late.reseed(Epoch(2), 4, view, Health::FULL, fold).unwrap();
    assert_eq!(late.cursor().next_seq, 4);
    assert_eq!(score(&late, &[8]), 0);
    assert_eq!(score(&late, &[1, 2, 3]), 3);

    apply(&mut late, vec![under(GPU, 3, &[4])]);
    assert_eq!(score(&late, &[1, 2, 3, 4]), 4);
}

/// Only the device tier reaches the tree: removes and clears of another tier leave it alone,
/// and a clear of the device tier or a dropped epoch empties it.
#[test]
fn the_index_follows_clears_of_its_own_tier_only() {
    let mut stream = radix_stream(config(TierSet::EMPTY), 0);
    apply(
        &mut stream,
        vec![root(GPU, &[1, 2, 3]), root(CPU, &[1, 2, 3])],
    );
    let step = |stream: &mut Stream<RadixIndex>,
                change: &dyn Fn(&mut BatchWriter<'_, Tokens, RadixIndex>)| {
        let seq = stream.cursor().next_seq;
        stream
            .apply_batch(seq, Arc::new(vec![]), BATCH_BYTES, |_, writer| {
                change(writer);
                Ok::<_, Infallible>(())
            })
            .unwrap();
    };

    step(&mut stream, &|writer| {
        writer.remove(CPU, [BlockHash(2)]);
        writer.clear(TierSet::of(CPU));
    });
    assert_eq!(score(&stream, &[1, 2, 3]), 3);

    step(&mut stream, &|writer| writer.clear(TierSet::of(GPU)));
    assert_eq!(score(&stream, &[1, 2, 3]), 0);
    assert!(stream.ledger().is_empty());

    apply(&mut stream, vec![root(GPU, &[1, 2])]);
    assert_eq!(score(&stream, &[1, 2, 3]), 2);
    step(&mut stream, &|writer| writer.clear_all());
    assert_eq!(score(&stream, &[1, 2, 3]), 0);

    apply(&mut stream, vec![root(GPU, &[1, 2])]);
    stream.start_epoch(Epoch(2), 0, Carry::Drop).unwrap();
    assert_eq!(score(&stream, &[1, 2, 3]), 0);
}
