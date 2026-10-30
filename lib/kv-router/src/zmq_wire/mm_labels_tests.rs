// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang routing-label spans with Dynamo-owned multimodal block ids.
//!
//! Events are encoded the way SGLang publishes them (tagged msgpack maps) and
//! decoded with the production decoder, so these tests cover decoding,
//! normalization and indexing together.

use serde::Serialize;

use super::*;
use crate::PositionalIndexer;
use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
use crate::indexer::{
    BranchShardedIndexer, KvIndexerInterface, LowerTierIndexer, RadixTree, ThreadPoolIndexer,
};
use crate::protocols::{
    BlockHashOptions, KvCacheStoreData, LocalBlockHash, RouterEvent, compute_block_hash_for_seq,
    compute_next_seq_hash, pad_value_for_mm_hash,
};

const BLOCK: usize = 16;
const A: u64 = 1;
const B: u64 = 2;

/// Routing labels: Dynamo's per-item hash, e.g. of an image URL.
const LABEL_U: u64 = 0x9e37_79b9_7f4a_7c15;
const LABEL_V: u64 = 0x0123_4567_89ab_cdef;

/// SGLang pad values for two different images (four 62-bit values each).
const IMAGE_X: [u64; 4] = [
    1_102_202_002_673_214_419,
    2_808_702_701_737_579_037,
    2_713_072_869_372_394_503,
    2_713_360_236_592_209_549,
];
const IMAGE_Y: [u64; 4] = [
    3_100_000_000_000_000_011,
    3_100_000_000_000_000_013,
    3_100_000_000_000_000_017,
    3_100_000_000_000_000_019,
];

// Engine (SGLang) block ids. Text blocks are shared; image blocks differ per image.
const E_T: u64 = 0x7e57_0000_0000_0001;
const E_IX: u64 = 0x1a6e_0000_0000_00a1;
const E_KX: u64 = 0x1a6e_0000_0000_00a2;
const E_IY: u64 = 0x1a6e_0000_0000_00b1;
const E_KY: u64 = 0x1a6e_0000_0000_00b2;

fn text(start: u64) -> Vec<u64> {
    (start..start + BLOCK as u64).collect()
}

fn image(values: [u64; 4], len: usize) -> Vec<u64> {
    (0..len).map(|index| values[index % 4]).collect()
}

fn routing_pad(label: u64, len: usize) -> Vec<u64> {
    vec![u64::from(pad_value_for_mm_hash(label)); len]
}

#[derive(Serialize)]
struct WireStored {
    #[serde(rename = "type")]
    kind: &'static str,
    block_hashes: Vec<i64>,
    parent_block_hash: Option<i64>,
    token_ids: Vec<u64>,
    block_size: usize,
    lora_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    medium: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mm_routing_spans: Option<Vec<Vec<(u32, u32, String)>>>,
}

#[derive(Serialize)]
struct WireRemoved {
    #[serde(rename = "type")]
    kind: &'static str,
    block_hashes: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    medium: Option<&'static str>,
}

#[derive(Serialize)]
struct WireCleared {
    #[serde(rename = "type")]
    kind: &'static str,
}

/// Eagle/NextN stores carry overlapping token pairs.
#[derive(Serialize)]
struct WireStoredBigram {
    #[serde(rename = "type")]
    kind: &'static str,
    block_hashes: Vec<i64>,
    parent_block_hash: Option<i64>,
    token_ids: Vec<(u64, u64)>,
    block_size: usize,
    lora_id: Option<i64>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Wire {
    Stored(WireStored),
    StoredBigram(WireStoredBigram),
    Removed(WireRemoved),
    Cleared(WireCleared),
}

struct Block {
    id: u64,
    tokens: Vec<u64>,
    /// `(start, length, label)` relative to the block.
    spans: Vec<(u32, u32, u64)>,
}

fn block(id: u64, tokens: Vec<u64>, spans: Vec<(u32, u32, u64)>) -> Block {
    assert_eq!(tokens.len(), BLOCK);
    Block { id, tokens, spans }
}

fn stored(parent: Option<u64>, blocks: &[Block]) -> Wire {
    stored_on(parent, blocks, None)
}

fn stored_on(parent: Option<u64>, blocks: &[Block], medium: Option<&'static str>) -> Wire {
    let any_spans = blocks.iter().any(|block| !block.spans.is_empty());
    Wire::Stored(WireStored {
        kind: "BlockStored",
        block_hashes: blocks.iter().map(|block| block.id as i64).collect(),
        parent_block_hash: parent.map(|parent| parent as i64),
        token_ids: blocks
            .iter()
            .flat_map(|block| block.tokens.iter().copied())
            .collect(),
        block_size: BLOCK,
        lora_id: None,
        medium,
        mm_routing_spans: any_spans.then(|| {
            blocks
                .iter()
                .map(|block| {
                    block
                        .spans
                        .iter()
                        .map(|&(start, len, label)| (start, len, format!("{label:016x}")))
                        .collect()
                })
                .collect()
        }),
    })
}

fn removed(ids: &[u64]) -> Wire {
    removed_on(ids, None)
}

fn removed_on(ids: &[u64], medium: Option<&'static str>) -> Wire {
    Wire::Removed(WireRemoved {
        kind: "BlockRemoved",
        block_hashes: ids.iter().map(|id| *id as i64).collect(),
        medium,
    })
}

/// An Eagle store over `tokens` (`ids.len() * BLOCK + 1` tokens), sent as the
/// overlapping pairs SGLang publishes.
fn stored_eagle(ids: &[u64], tokens: &[u64]) -> Wire {
    assert_eq!(tokens.len(), ids.len() * BLOCK + 1);
    Wire::StoredBigram(WireStoredBigram {
        kind: "BlockStored",
        block_hashes: ids.iter().map(|id| *id as i64).collect(),
        parent_block_hash: None,
        token_ids: tokens.windows(2).map(|pair| (pair[0], pair[1])).collect(),
        block_size: BLOCK,
        lora_id: None,
    })
}

fn cleared() -> Wire {
    Wire::Cleared(WireCleared {
        kind: "AllBlocksCleared",
    })
}

fn decode_one(wire: Wire) -> RawKvEvent {
    let batch = (0.0f64, vec![wire], None::<i32>);
    let bytes = rmp_serde::to_vec_named(&batch).expect("encode");
    decode_event_batch(&bytes)
        .expect("SGLang-shaped batch decodes")
        .events
        .into_iter()
        .next()
        .expect("one event")
}

fn labeled_normalizer() -> ZmqEventNormalizer {
    ZmqEventNormalizer::new(BLOCK as u32).with_engine_owned_mm_identity(true)
}

fn worker(id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::new(id, 0)
}

fn run(
    normalizer: &mut ZmqEventNormalizer,
    worker_id: u64,
    event_id: u64,
    wire: Wire,
) -> PlacementEvent {
    run_on(normalizer, worker(worker_id), event_id, wire)
}

fn run_on(
    normalizer: &mut ZmqEventNormalizer,
    worker: WorkerWithDpRank,
    event_id: u64,
    wire: Wire,
) -> PlacementEvent {
    normalizer
        .normalize(decode_one(wire), event_id, worker)
        .expect("accepted events always yield a placement event")
}

fn store_data(event: &PlacementEvent) -> Option<&KvCacheStoreData> {
    match &event.event.data {
        KvCacheEventData::Stored(store) => Some(store),
        _ => None,
    }
}

fn removed_ids(event: &PlacementEvent) -> Vec<u64> {
    match &event.event.data {
        KvCacheEventData::Removed(remove) => {
            remove.block_hashes.iter().map(|hash| hash.0).collect()
        }
        other => panic!("expected a removal, got {other:?}"),
    }
}

fn is_noop(event: &PlacementEvent) -> bool {
    matches!(&event.event.data, KvCacheEventData::Removed(remove) if remove.block_hashes.is_empty())
}

fn request_hashes(blocks: &[Vec<u64>]) -> Vec<LocalBlockHash> {
    let tokens: Vec<u32> = blocks
        .concat()
        .into_iter()
        .map(|token| u32::try_from(token).expect("routing tokens fit in u32"))
        .collect();
    compute_block_hash_for_seq(
        &tokens,
        BLOCK as u32,
        BlockHashOptions {
            block_mm_infos: None,
            lora_name: None,
            is_eagle: None,
            cache_namespace: None,
        },
    )
}

fn hashes(tokens: &[u64], is_eagle: bool) -> Vec<LocalBlockHash> {
    let tokens: Vec<u32> = tokens
        .iter()
        .map(|&token| u32::try_from(token).expect("routing tokens fit in u32"))
        .collect();
    compute_block_hash_for_seq(
        &tokens,
        BLOCK as u32,
        BlockHashOptions {
            block_mm_infos: None,
            lora_name: None,
            is_eagle: Some(is_eagle),
            cache_namespace: None,
        },
    )
}

/// Request: text, one image (labeled), text. Routing tokens use Dynamo's pad.
fn request(label: u64) -> Vec<LocalBlockHash> {
    request_hashes(&[text(100), routing_pad(label, BLOCK), text(300)])
}

/// SGLang store for that request: text block, image block, text block.
fn image_request_store(image_values: [u64; 4], label: u64, ids: [u64; 3]) -> Wire {
    stored(
        None,
        &[
            block(ids[0], text(100), vec![]),
            block(
                ids[1],
                image(image_values, BLOCK),
                vec![(0, BLOCK as u32, label)],
            ),
            block(ids[2], text(300), vec![]),
        ],
    )
}

fn router_event(worker_id: u64, event: &PlacementEvent) -> RouterEvent {
    RouterEvent::new(worker_id, event.event.clone())
}

fn score(scores: &crate::protocols::OverlapScores, worker_id: u64) -> u32 {
    scores.scores.get(&worker(worker_id)).copied().unwrap_or(0)
}

/// Both the router default (CRTC behind a thread pool) and the single-threaded
/// radix tree, driven with the same events.
struct Indexes {
    radix: RadixTree,
    crtc: ThreadPoolIndexer<ConcurrentRadixTreeCompressed>,
}

impl Indexes {
    fn new() -> Self {
        Self {
            radix: RadixTree::new(),
            crtc: ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 4, BLOCK as u32),
        }
    }

    async fn apply(&mut self, worker_id: u64, event: &PlacementEvent) {
        let event = router_event(worker_id, event);
        self.radix
            .apply_event(event.clone())
            .expect("radix tree accepts event");
        self.crtc
            .apply_event_and_wait(event)
            .await
            .expect("CRTC accepts event");
    }

    /// Scores per worker as (radix tree, CRTC); the two must agree.
    async fn scores(&self, query: &[LocalBlockHash], workers: &[u64]) -> Vec<u32> {
        let radix = self.radix.find_matches(query.to_vec(), false);
        let crtc = KvIndexerInterface::find_matches(&self.crtc, query.to_vec())
            .await
            .expect("CRTC query");
        workers
            .iter()
            .map(|&worker_id| {
                let (radix_score, crtc_score) = (score(&radix, worker_id), score(&crtc, worker_id));
                assert_eq!(
                    radix_score, crtc_score,
                    "indexers disagree for worker {worker_id}"
                );
                radix_score
            })
            .collect()
    }
}

#[test]
fn sglang_map_with_wide_tokens_and_spans_decodes() {
    let raw = decode_one(image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]));
    let RawKvEvent::BlockStored {
        token_ids,
        block_mm_infos,
        ..
    } = raw
    else {
        panic!("expected BlockStored");
    };
    // Wide values no longer fail the batch; they narrow to the sentinel.
    assert!(
        token_ids[BLOCK..2 * BLOCK]
            .iter()
            .all(|&token| token == WIDE_TOKEN_SENTINEL)
    );
    let infos = block_mm_infos.expect("spans become per-block multimodal info");
    assert!(infos[0].is_none() && infos[2].is_none());
    let image_info = infos[1].as_ref().expect("image block info");
    assert_eq!(image_info.mm_objects.len(), 1);
    assert_eq!(image_info.mm_objects[0].mm_hash, LABEL_U);
    assert_eq!(image_info.mm_objects[0].offsets, vec![(0, BLOCK)]);
}

#[test]
fn labeled_image_block_hashes_match_the_request_side() {
    let mut normalizer = labeled_normalizer();
    let event = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    let store = store_data(&event).expect("store published");
    let local: Vec<LocalBlockHash> = store.blocks.iter().map(|block| block.tokens_hash).collect();
    assert_eq!(local, request(LABEL_U));

    // The text block keeps SGLang's id; the image chain gets Dynamo-owned ids.
    let d_image = compute_next_seq_hash(E_T, local[1]);
    let d_after = compute_next_seq_hash(d_image, local[2]);
    let ids: Vec<u64> = store
        .blocks
        .iter()
        .map(|block| block.block_hash.0)
        .collect();
    assert_eq!(ids, vec![E_T, d_image, d_after]);
    assert_eq!(store.parent_hash, None);
}

#[test]
fn same_label_on_different_images_gets_the_same_ids() {
    let mut normalizer = labeled_normalizer();
    let on_a = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    let on_b = run(
        &mut normalizer,
        B,
        1,
        image_request_store(IMAGE_Y, LABEL_U, [E_T, E_IY, E_KY]),
    );
    let ids = |event: &PlacementEvent| -> Vec<u64> {
        store_data(event)
            .unwrap()
            .blocks
            .iter()
            .map(|block| block.block_hash.0)
            .collect()
    };
    assert_eq!(ids(&on_a), ids(&on_b));
}

/// Cross-worker alias scenario on any interface-driven index. Returns the
/// [A, B] scores after both stores, after B evicts, and after A evicts.
async fn alias_across_workers_scores<I: KvIndexerInterface>(index: &I) -> Vec<Vec<u32>> {
    let mut normalizer = labeled_normalizer();
    let query = request(LABEL_U);
    let steps = [
        vec![
            (A, image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX])),
            (B, image_request_store(IMAGE_Y, LABEL_U, [E_T, E_IY, E_KY])),
        ],
        vec![(B, removed(&[E_IY, E_KY]))],
        vec![(A, removed(&[E_IX, E_KX]))],
    ];
    let mut event_id = 0;
    let mut observed = Vec::new();
    for step in steps {
        for (worker_id, wire) in step {
            event_id += 1;
            let event = run(&mut normalizer, worker_id, event_id, wire);
            KvIndexerInterface::apply_event(index, router_event(worker_id, &event)).await;
        }
        let _ = KvIndexerInterface::flush(index).await;
        let scores = KvIndexerInterface::find_matches(index, query.clone())
            .await
            .expect("query");
        observed.push(vec![score(&scores, A), score(&scores, B)]);
    }
    observed
}

#[tokio::test]
async fn alias_across_workers_is_clean_in_branch_sharded_and_lower_tier_indexes() {
    let expected = vec![vec![3, 3], vec![3, 1], vec![1, 1]];
    for depth in [1, 2] {
        let shards = (0..2)
            .map(|_| ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, BLOCK as u32))
            .collect();
        let index = BranchShardedIndexer::new(shards, depth, BLOCK as u32);
        assert_eq!(
            alias_across_workers_scores(&index).await,
            expected,
            "branch-sharded depth {depth}"
        );
    }
    let lower = ThreadPoolIndexer::new(LowerTierIndexer::new(), 4, BLOCK as u32);
    assert_eq!(
        alias_across_workers_scores(&lower).await,
        expected,
        "lower tier"
    );
}

#[tokio::test]
async fn alias_across_workers_leaves_no_stale_entries() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let query = request(LABEL_U);

    // Same URL label, different image content on two workers.
    let a = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    let b = run(
        &mut normalizer,
        B,
        1,
        image_request_store(IMAGE_Y, LABEL_U, [E_T, E_IY, E_KY]),
    );
    indexes.apply(A, &a).await;
    indexes.apply(B, &b).await;
    assert_eq!(indexes.scores(&query, &[A, B]).await, vec![3, 3]);

    // B evicts its image and the text after it.
    let b_evict = run(&mut normalizer, B, 2, removed(&[E_IY, E_KY]));
    indexes.apply(B, &b_evict).await;
    assert_eq!(indexes.scores(&query, &[A, B]).await, vec![3, 1]);

    // A evicts too; only the shared text prefix remains on both.
    let a_evict = run(&mut normalizer, A, 2, removed(&[E_IX, E_KX]));
    indexes.apply(A, &a_evict).await;
    assert_eq!(indexes.scores(&query, &[A, B]).await, vec![1, 1]);
}

#[tokio::test]
async fn alias_on_one_worker_is_reference_counted() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let query = request(LABEL_U);

    let first = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    indexes.apply(A, &first).await;

    // The URL now serves image Y. SGLang reuses the cached text prefix and
    // stores only the new suffix under it.
    let second = run(
        &mut normalizer,
        A,
        2,
        stored(
            Some(E_T),
            &[
                block(
                    E_IY,
                    image(IMAGE_Y, BLOCK),
                    vec![(0, BLOCK as u32, LABEL_U)],
                ),
                block(E_KY, text(300), vec![]),
            ],
        ),
    );
    assert!(is_noop(&second), "already indexed under the same ids");
    assert_eq!(normalizer.engine_mm_stats().aliased_blocks, 2);

    // Evicting image X keeps the entry: A still holds image Y under it.
    let evict_x = run(&mut normalizer, A, 3, removed(&[E_IX, E_KX]));
    assert!(removed_ids(&evict_x).is_empty());
    indexes.apply(A, &evict_x).await;
    assert_eq!(indexes.scores(&query, &[A]).await, vec![3]);

    // Evicting image Y removes it.
    let evict_y = run(&mut normalizer, A, 4, removed(&[E_IY, E_KY]));
    assert_eq!(removed_ids(&evict_y).len(), 2);
    indexes.apply(A, &evict_y).await;
    assert_eq!(indexes.scores(&query, &[A]).await, vec![1]);
    assert_eq!(normalizer.engine_mm_tracked_ids(), 0);
}

#[tokio::test]
async fn unlabeled_image_is_excluded_with_its_descendants() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();

    let event = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[
                block(E_T, text(100), vec![]),
                block(E_IX, image(IMAGE_X, BLOCK), vec![]),
                block(E_KX, text(300), vec![]),
            ],
        ),
    );
    let store = store_data(&event).expect("text prefix still published");
    assert_eq!(store.blocks.len(), 1);
    assert_eq!(store.blocks[0].block_hash.0, E_T);
    indexes.apply(A, &event).await;

    // A later store chained under the excluded block is dropped too.
    let child = run(
        &mut normalizer,
        A,
        2,
        stored(Some(E_KX), &[block(0xc0de, text(500), vec![])]),
    );
    assert!(is_noop(&child));

    // Removals of excluded blocks never reach the index; only E_T passes through.
    let evict = run(&mut normalizer, A, 3, removed(&[E_IX, E_KX, 0xc0de, E_T]));
    assert_eq!(removed_ids(&evict), vec![E_T]);
    indexes.apply(A, &evict).await;

    let stats = normalizer.engine_mm_stats();
    assert_eq!(stats.excluded_unmapped, 2);
    assert_eq!(stats.excluded_parent, 1);
    // Both indexes must reach zero after the text-prefix block is evicted.
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![0]);
}

/// The id map never evicts a live mapping. When it is full, a new multimodal
/// chain is excluded like an unmapped one, its text prefix still routes, and
/// it is mapped once capacity frees up.
#[tokio::test]
async fn full_id_map_falls_back_to_exclusion_without_evicting() {
    let mut normalizer = labeled_normalizer().with_engine_mm_capacities(1 << 10, 2);
    let mut indexes = Indexes::new();

    // X under U fills the map: its image block and the text block after it.
    let a = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    indexes.apply(A, &a).await;
    assert_eq!(normalizer.engine_mm_tracked_ids(), 2);

    // Y under V needs two more mappings: excluded, text prefix kept.
    let b = run(
        &mut normalizer,
        B,
        1,
        image_request_store(IMAGE_Y, LABEL_V, [E_T, E_IY, E_KY]),
    );
    let store = store_data(&b).expect("text prefix still published");
    assert_eq!(store.blocks.len(), 1);
    assert_eq!(store.blocks[0].block_hash.0, E_T);
    indexes.apply(B, &b).await;
    assert_eq!(normalizer.engine_mm_stats().map_full, 2);
    assert_eq!(indexes.scores(&request(LABEL_U), &[A, B]).await, vec![3, 1]);
    assert_eq!(indexes.scores(&request(LABEL_V), &[A, B]).await, vec![1, 1]);

    // A evicts its image, which frees capacity. B evicts the excluded blocks
    // (nothing reaches the index) and stores Y again: now it is mapped.
    let a_evict = run(&mut normalizer, A, 2, removed(&[E_IX, E_KX]));
    indexes.apply(A, &a_evict).await;
    assert_eq!(normalizer.engine_mm_tracked_ids(), 0);
    let b_evict = run(&mut normalizer, B, 2, removed(&[E_IY, E_KY]));
    assert!(is_noop(&b_evict));
    indexes.apply(B, &b_evict).await;
    let b_again = run(
        &mut normalizer,
        B,
        3,
        image_request_store(IMAGE_Y, LABEL_V, [E_T, E_IY, E_KY]),
    );
    indexes.apply(B, &b_again).await;
    assert_eq!(indexes.scores(&request(LABEL_V), &[A, B]).await, vec![1, 3]);
}

/// A store dropped entirely becomes an empty removal so the event id sequence
/// has no gap. Every index must take it as a no-op, including for a worker it
/// has not seen yet (an unlabeled image-first request right after startup).
#[tokio::test]
async fn fully_excluded_store_is_a_noop_for_an_unseen_worker() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let event = run(
        &mut normalizer,
        A,
        1,
        stored(None, &[block(E_IX, image(IMAGE_X, BLOCK), vec![])]),
    );
    assert!(is_noop(&event));
    indexes.apply(A, &event).await;

    let positional = ThreadPoolIndexer::new(PositionalIndexer::new(8), 2, BLOCK as u32);
    positional
        .apply_event_and_wait(router_event(A, &event))
        .await
        .expect("positional index accepts an empty removal");

    let lower = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, BLOCK as u32);
    let host = run(
        &mut normalizer,
        A,
        2,
        stored_on(
            None,
            &[block(E_IX, image(IMAGE_X, BLOCK), vec![])],
            Some("CPU_PINNED"),
        ),
    );
    assert!(is_noop(&host));
    lower
        .apply_event_and_wait(RouterEvent::with_storage_tier(
            A,
            host.event.clone(),
            StorageTier::HostPinned,
        ))
        .await
        .expect("lower tier accepts an empty removal");
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![0]);
}

/// Exclusion state is advisory: with none remembered, a descendant of an
/// excluded block reaches the index, which rejects it for its unknown parent.
#[tokio::test]
async fn losing_exclusion_state_never_indexes_unmapped_descendants() {
    let mut normalizer = labeled_normalizer().with_engine_mm_capacities(0, usize::MAX);
    let mut indexes = Indexes::new();
    let first = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[
                block(E_T, text(100), vec![]),
                block(E_IX, image(IMAGE_X, BLOCK), vec![]),
            ],
        ),
    );
    indexes.apply(A, &first).await;
    assert_eq!(normalizer.engine_mm_stats().excluded_overflow, 1);

    let child = run(
        &mut normalizer,
        A,
        2,
        stored(Some(E_IX), &[block(E_KX, text(300), vec![])]),
    );
    assert!(
        store_data(&child).is_some(),
        "not dropped without exclusion state"
    );
    let event = router_event(A, &child);
    assert!(indexes.radix.apply_event(event.clone()).is_err());
    let _ = indexes.crtc.apply_event_and_wait(event).await;

    let query = request_hashes(&[text(100), routing_pad(LABEL_U, BLOCK), text(300)]);
    assert_eq!(indexes.scores(&query, &[A]).await, vec![1]);
}

/// The DC relay's producer is stricter than the router's indexes: a store under
/// an unknown parent is a source protocol fault there (the relay host then
/// rejects the source). Exclusion state therefore matters to the relay, so it
/// must not overflow while the engine still holds the excluded blocks.
#[test]
fn losing_exclusion_state_is_a_source_fault_for_the_dc_relay() {
    use crate::indexer::cuckoo::{CkfConfig, DcCkfState};
    use crate::protocols::KvCacheEventError;

    let events = |excluded_capacity: usize| {
        let mut normalizer =
            labeled_normalizer().with_engine_mm_capacities(excluded_capacity, usize::MAX);
        let first = run(
            &mut normalizer,
            A,
            1,
            stored(
                None,
                &[
                    block(E_T, text(100), vec![]),
                    block(E_IX, image(IMAGE_X, BLOCK), vec![]),
                ],
            ),
        );
        let child = run(
            &mut normalizer,
            A,
            2,
            stored(Some(E_IX), &[block(E_KX, text(300), vec![])]),
        );
        [first, child]
    };
    let relay_errors = |events: [PlacementEvent; 2]| -> Vec<Option<KvCacheEventError>> {
        let mut relay = DcCkfState::new(CkfConfig::new(64)).expect("relay producer");
        events
            .iter()
            .map(|event| {
                relay
                    .apply_event(router_event(A, event))
                    .first_error()
                    .copied()
            })
            .collect()
    };

    assert_eq!(relay_errors(events(1 << 10)), vec![None, None]);
    assert_eq!(
        relay_errors(events(0)),
        vec![None, Some(KvCacheEventError::ParentBlockNotFound)]
    );
}

#[test]
fn multimodal_values_that_fit_in_u32_are_still_classified() {
    let mut normalizer = labeled_normalizer();
    // An image starts in the last two tokens of the text block; its values
    // happen to fit in u32, so only the >= 1_000_000 rule catches them.
    let mut first = text(100);
    first[BLOCK - 2] = 1_500_000;
    first[BLOCK - 1] = 1_600_000;
    let event = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[block(E_T, text(50), vec![]), block(E_IX, first, vec![])],
        ),
    );
    let store = store_data(&event).unwrap();
    assert_eq!(
        store.blocks.len(),
        1,
        "the block with unlabeled values is cut"
    );
    assert_eq!(normalizer.engine_mm_stats().excluded_unmapped, 1);
}

#[test]
fn partial_and_multiple_items_per_block_are_rewritten() {
    let mut normalizer = labeled_normalizer();
    // Block 0: 10 text tokens, then the first 6 tokens of image U.
    // Block 1: the last 10 tokens of image U, then 6 tokens of image V.
    let mut block0 = text(100)[..10].to_vec();
    block0.extend(image(IMAGE_X, 6));
    let mut block1 = image(IMAGE_X, 16)[6..].to_vec();
    block1.extend(image(IMAGE_Y, 6));
    let event = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[
                block(E_IX, block0, vec![(10, 6, LABEL_U)]),
                block(E_KX, block1, vec![(0, 10, LABEL_U), (10, 6, LABEL_V)]),
            ],
        ),
    );
    let mut routing0 = text(100)[..10].to_vec();
    routing0.extend(routing_pad(LABEL_U, 6));
    let mut routing1 = routing_pad(LABEL_U, 10);
    routing1.extend(routing_pad(LABEL_V, 6));
    let local: Vec<LocalBlockHash> = store_data(&event)
        .unwrap()
        .blocks
        .iter()
        .map(|block| block.tokens_hash)
        .collect();
    assert_eq!(local, request_hashes(&[routing0, routing1]));
}

/// A named list of `(start, length, label)` spans for one block.
type SpanCase = (&'static str, Vec<(u32, u32, u64)>);

#[test]
fn invalid_spans_fail_closed() {
    let cases: Vec<SpanCase> = vec![
        ("out of bounds", vec![(10, 10, LABEL_U)]),
        ("overlapping", vec![(0, 10, LABEL_U), (5, 11, LABEL_V)]),
        ("covers text", vec![(0, BLOCK as u32, LABEL_U)]),
    ];
    for (name, spans) in cases {
        let mut normalizer = labeled_normalizer();
        let tokens = if name == "covers text" {
            let mut tokens = image(IMAGE_X, BLOCK);
            tokens[3] = 42;
            tokens
        } else {
            image(IMAGE_X, BLOCK)
        };
        let event = run(
            &mut normalizer,
            A,
            1,
            stored(
                None,
                &[block(E_T, text(100), vec![]), block(E_IX, tokens, spans)],
            ),
        );
        assert_eq!(store_data(&event).unwrap().blocks.len(), 1, "{name}");
        assert_eq!(normalizer.engine_mm_stats().invalid_spans, 1, "{name}");
    }
}

#[tokio::test]
async fn relabeled_engine_block_keeps_the_first_mapping() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let first = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    indexes.apply(A, &first).await;

    // The same engine block arrives again with a different label.
    let relabeled = run(
        &mut normalizer,
        A,
        2,
        stored(
            Some(E_T),
            &[block(
                E_IX,
                image(IMAGE_X, BLOCK),
                vec![(0, BLOCK as u32, LABEL_V)],
            )],
        ),
    );
    assert!(is_noop(&relabeled));
    assert_eq!(normalizer.engine_mm_stats().relabel_conflicts, 1);
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![3]);
    assert_eq!(indexes.scores(&request(LABEL_V), &[A]).await, vec![1]);

    // The first mapping still owns the block, so its eviction reaches the index.
    let evict = run(&mut normalizer, A, 3, removed(&[E_KX, E_IX]));
    indexes.apply(A, &evict).await;
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
}

#[test]
fn cleared_resets_tracked_ids() {
    let mut normalizer = labeled_normalizer();
    run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    assert_eq!(normalizer.engine_mm_tracked_ids(), 2);
    let cleared = run(
        &mut normalizer,
        A,
        2,
        Wire::Cleared(WireCleared {
            kind: "AllBlocksCleared",
        }),
    );
    assert!(matches!(cleared.event.data, KvCacheEventData::Cleared));
    assert_eq!(normalizer.engine_mm_tracked_ids(), 0);
}

#[test]
fn storage_tiers_are_counted_separately() {
    let mut normalizer = labeled_normalizer();
    let blocks = |ids: [u64; 3]| {
        vec![
            block(ids[0], text(100), vec![]),
            block(
                ids[1],
                image(IMAGE_X, BLOCK),
                vec![(0, BLOCK as u32, LABEL_U)],
            ),
            block(ids[2], text(300), vec![]),
        ]
    };
    let gpu = run(
        &mut normalizer,
        A,
        1,
        stored_on(None, &blocks([E_T, E_IX, E_KX]), Some("GPU")),
    );
    let cpu = run(
        &mut normalizer,
        A,
        2,
        stored_on(None, &blocks([E_T, E_IX, E_KX]), Some("CPU_PINNED")),
    );
    assert_eq!(store_data(&gpu).unwrap().blocks.len(), 3);
    assert_eq!(
        store_data(&cpu).unwrap().blocks.len(),
        3,
        "host tier indexed separately"
    );

    // Evicting from the GPU removes only the GPU entries.
    let gpu_evict = run(
        &mut normalizer,
        A,
        3,
        removed_on(&[E_IX, E_KX], Some("GPU")),
    );
    assert_eq!(removed_ids(&gpu_evict).len(), 2);
    let cpu_evict = run(
        &mut normalizer,
        A,
        4,
        removed_on(&[E_IX, E_KX], Some("CPU_PINNED")),
    );
    assert_eq!(removed_ids(&cpu_evict).len(), 2);
    assert_eq!(normalizer.engine_mm_tracked_ids(), 0);
}

/// Without the engine-owned flag, every value that did not fit in `u32`
/// narrows to one sentinel; indexing those blocks would make different images
/// share a local hash. They are excluded like engine-owned values, descendants
/// and removals included, so a later chained store never reaches the index
/// under an unknown parent. Spans are ignored without the flag.
#[test]
fn unflagged_source_excludes_wide_blocks_and_their_descendants() {
    let mut normalizer = ZmqEventNormalizer::new(BLOCK as u32);
    let event = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    let store = store_data(&event).unwrap();
    assert_eq!(store.blocks.len(), 1);
    assert_eq!(store.blocks[0].block_hash.0, E_T);

    let child = run(
        &mut normalizer,
        A,
        2,
        stored(Some(E_KX), &[block(0xc0de, text(500), vec![])]),
    );
    assert!(is_noop(&child));
    let evict = run(&mut normalizer, A, 3, removed(&[E_IX, E_KX, 0xc0de, E_T]));
    assert_eq!(removed_ids(&evict), vec![E_T]);
}

/// Legacy sources use Dynamo's own pads, which sit above `MM_PAD_SHIFT_VALUE`
/// but fit in `u32`; only the engine-owned flag excludes those.
#[test]
fn unflagged_source_keeps_legacy_pads() {
    let mut legacy_pad = text(100);
    legacy_pad[BLOCK - 1] = u64::from(pad_value_for_mm_hash(LABEL_U));
    let wire = || {
        stored(
            None,
            &[
                block(E_T, text(50), vec![]),
                block(E_IX, legacy_pad.clone(), vec![]),
            ],
        )
    };

    let mut unflagged = ZmqEventNormalizer::new(BLOCK as u32);
    let kept = run(&mut unflagged, A, 1, wire());
    assert_eq!(store_data(&kept).unwrap().blocks.len(), 2);

    let mut flagged = labeled_normalizer();
    let cut = run(&mut flagged, A, 1, wire());
    assert_eq!(store_data(&cut).unwrap().blocks.len(), 1);
}

#[test]
fn text_only_events_are_unchanged() {
    let mut labeled = labeled_normalizer();
    let mut plain = ZmqEventNormalizer::new(BLOCK as u32);
    let wire = || {
        stored(
            None,
            &[
                block(E_T, text(100), vec![]),
                block(E_KX, text(300), vec![]),
            ],
        )
    };
    let with_flag = run(&mut labeled, A, 1, wire());
    let without_flag = run(&mut plain, A, 1, wire());
    assert_eq!(with_flag.event, without_flag.event);
}

/// Eagle hashes each block together with the next block's first token, so a
/// multimodal value there belongs to the preceding block. Both flag states
/// classify exactly the hashed windows and keep the overlap token the last
/// surviving block needs.
#[test]
fn eagle_overlap_token_is_classified_with_the_preceding_block() {
    for engine_owned in [false, true] {
        let normalizer =
            || ZmqEventNormalizer::new(BLOCK as u32).with_engine_owned_mm_identity(engine_owned);

        // The image starts at the overlap token of block 0.
        let mut at_overlap: Vec<u64> = (100..100 + BLOCK as u64).collect();
        at_overlap.extend(image(IMAGE_X, BLOCK + 1));
        let event = run(
            &mut normalizer(),
            A,
            1,
            stored_eagle(&[E_T, E_IX], &at_overlap),
        );
        assert!(is_noop(&event), "engine_owned={engine_owned}");

        // The image starts one token later: block 0 is text, overlap included.
        let mut after_overlap: Vec<u64> = (100..101 + BLOCK as u64).collect();
        after_overlap.extend(image(IMAGE_X, BLOCK));
        let event = run(
            &mut normalizer(),
            A,
            1,
            stored_eagle(&[E_T, E_IX], &after_overlap),
        );
        let store = store_data(&event).expect("text block published");
        assert_eq!(store.blocks.len(), 1, "engine_owned={engine_owned}");
        assert_eq!(store.blocks[0].block_hash.0, E_T);
        assert_eq!(
            store.blocks[0].tokens_hash,
            hashes(&after_overlap[..=BLOCK], true)[0],
            "block 0 hashes its tokens plus the overlap token"
        );
    }
}

/// Exclusions belong to one worker, DP rank and storage tier. A clear forgets
/// only that rank's exclusions; another tier never held the excluded blocks.
#[test]
fn exclusions_are_scoped_to_worker_rank_and_tier() {
    let mut normalizer = labeled_normalizer();
    let rank0 = WorkerWithDpRank::new(A, 0);
    let rank1 = WorkerWithDpRank::new(A, 1);
    let unlabeled_store = || {
        stored(
            None,
            &[
                block(E_T, text(100), vec![]),
                block(E_IX, image(IMAGE_X, BLOCK), vec![]),
                block(E_KX, text(300), vec![]),
            ],
        )
    };
    run_on(&mut normalizer, rank0, 1, unlabeled_store());
    run_on(&mut normalizer, rank1, 1, unlabeled_store());

    let host = run_on(
        &mut normalizer,
        rank0,
        2,
        removed_on(&[E_IX], Some("CPU_PINNED")),
    );
    assert_eq!(removed_ids(&host), vec![E_IX]);

    let clear = run_on(&mut normalizer, rank0, 3, cleared());
    assert!(matches!(clear.event.data, KvCacheEventData::Cleared));
    let rank0_evict = run_on(&mut normalizer, rank0, 4, removed(&[E_IX, E_KX]));
    assert_eq!(removed_ids(&rank0_evict), vec![E_IX, E_KX]);
    let rank1_evict = run_on(&mut normalizer, rank1, 2, removed(&[E_IX, E_KX]));
    assert!(removed_ids(&rank1_evict).is_empty());
}

/// SGLang's label registry is bounded, so a later store of a cached block can
/// arrive without its span. The block keeps its mapping: its eviction still
/// reaches the index.
#[tokio::test]
async fn restore_without_label_keeps_the_mapping_for_eviction() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let first = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    indexes.apply(A, &first).await;
    let unlabeled = run(
        &mut normalizer,
        A,
        2,
        stored(
            Some(E_T),
            &[
                block(E_IX, image(IMAGE_X, BLOCK), vec![]),
                block(E_KX, text(300), vec![]),
            ],
        ),
    );
    indexes.apply(A, &unlabeled).await;

    let evict = run(&mut normalizer, A, 3, removed(&[E_KX, E_IX]));
    indexes.apply(A, &evict).await;
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
}

/// A block excluded while unlabeled and later stored with its label is mapped
/// from then on: descendants attach to it and its eviction reaches the index.
#[tokio::test]
async fn readmitted_block_is_mapped() {
    for evict_after in [false, true] {
        let mut normalizer = labeled_normalizer();
        let mut indexes = Indexes::new();
        let unlabeled = run(
            &mut normalizer,
            A,
            1,
            stored(
                None,
                &[
                    block(E_T, text(100), vec![]),
                    block(E_IX, image(IMAGE_X, BLOCK), vec![]),
                ],
            ),
        );
        indexes.apply(A, &unlabeled).await;
        let labeled = run(
            &mut normalizer,
            A,
            2,
            stored(
                Some(E_T),
                &[block(
                    E_IX,
                    image(IMAGE_X, BLOCK),
                    vec![(0, BLOCK as u32, LABEL_U)],
                )],
            ),
        );
        indexes.apply(A, &labeled).await;

        if evict_after {
            let evict = run(&mut normalizer, A, 3, removed(&[E_IX]));
            indexes.apply(A, &evict).await;
            assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
        } else {
            let child = run(
                &mut normalizer,
                A,
                3,
                stored(Some(E_IX), &[block(E_KX, text(300), vec![])]),
            );
            indexes.apply(A, &child).await;
            assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![3]);
        }
    }
}
