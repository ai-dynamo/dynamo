// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang routing-label spans with Dynamo-owned multimodal block ids.
//!
//! Events are encoded the way SGLang publishes them (tagged msgpack maps) and
//! decoded with the production decoder, so these tests cover decoding,
//! normalization and indexing together.

use serde::Serialize;

use dynamo_kv_events::event::{KvCacheEventData, KvCacheEventError, KvCacheStoreData, RouterEvent};
use dynamo_kv_events::placement::PlacementEvent;
use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_events::zmq_wire::{RawKvEvent, ZmqEventNormalizer, decode_event_batch};
use dynamo_kv_hashing::LocalBlockHash;
use dynamo_kv_hashing::multimodal::pad_value_for_mm_hash;
use dynamo_kv_hashing::routing::{
    BlockHashOptions, compute_block_hash_for_seq, compute_next_seq_hash,
};
use dynamo_router_protocol::worker::WorkerWithDpRank;

use crate::{
    BranchShardedIndexer, ConcurrentRadixTreeCompressed, KvIndexerInterface, LowerTierContinuation,
    LowerTierIndexer, OverlapScores, RadixTree, ThreadPoolIndexer,
};

const BLOCK: usize = 16;
const A: u64 = 1;
const B: u64 = 2;

/// Routing labels: Dynamo's per-item hash, e.g. of an image URL.
const LABEL_U: u64 = 0x9e37_79b9_7f4a_7c15;
const LABEL_V: u64 = 0x0123_4567_89ab_cdef;
const ENGINE_SALT_X: &str =
    "sglang-mm-v1:0000000000000000000000000000000000000000000000000000000000000001";
const ENGINE_SALT_Y: &str =
    "sglang-mm-v1:0000000000000000000000000000000000000000000000000000000000000002";

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
    cache_salt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_cache_salt: Option<Option<String>>,
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

#[derive(Serialize)]
#[serde(untagged)]
enum Wire {
    Stored(WireStored),
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
        cache_salt: None,
        user_cache_salt: None,
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
    request_hashes_in_namespace(blocks, None)
}

fn request_hashes_in_namespace(
    blocks: &[Vec<u64>],
    cache_namespace: Option<&str>,
) -> Vec<LocalBlockHash> {
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
            cache_namespace,
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

fn score(scores: &OverlapScores, worker_id: u64) -> u32 {
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
fn invalid_or_overflowing_labels_leave_media_unmapped() {
    for label in ["not-a-hash", "10000000000000000"] {
        let mut normalizer = labeled_normalizer();
        let Wire::Stored(mut wire) = image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX])
        else {
            unreachable!();
        };
        wire.mm_routing_spans.as_mut().unwrap()[1][0].2 = label.to_string();
        let event = run(&mut normalizer, A, 1, Wire::Stored(wire));
        let store = store_data(&event).expect("text prefix survives an unusable label");
        assert_eq!(store.blocks.len(), 1, "{label}");
        assert_eq!(store.blocks[0].tokens_hash, request(LABEL_U)[0]);
        let evict = run(&mut normalizer, A, 2, removed(&[E_IX, E_KX]));
        assert!(is_noop(&evict), "{label}");
    }
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

    // Text prefixes also need canonical ids: SGLang salts every native id.
    let d_text = store.blocks[0].block_hash.0;
    assert_ne!(d_text, E_T);
    let d_image = compute_next_seq_hash(d_text, local[1]);
    let d_after = compute_next_seq_hash(d_image, local[2]);
    let ids: Vec<u64> = store
        .blocks
        .iter()
        .map(|block| block.block_hash.0)
        .collect();
    assert_eq!(ids, vec![d_text, d_image, d_after]);
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
}

/// The id map never evicts a live mapping. When it is full, a new multimodal
/// chain is excluded like an unmapped one, an existing text prefix still routes, and
/// it is mapped once capacity frees up.
#[tokio::test]
async fn full_id_map_falls_back_to_exclusion_without_evicting() {
    let mut normalizer = labeled_normalizer()
        .with_engine_mm_excluded_capacity(1 << 10)
        .with_engine_mm_id_capacity(4);
    let mut indexes = Indexes::new();

    let prefix = run(
        &mut normalizer,
        B,
        1,
        stored(None, &[block(E_T, text(100), vec![])]),
    );
    indexes.apply(B, &prefix).await;

    // A's three blocks plus B's existing text prefix fill the map.
    let a = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    indexes.apply(A, &a).await;

    // Y under V needs two more mappings; its existing prefix stays indexed.
    let b = run(
        &mut normalizer,
        B,
        2,
        image_request_store(IMAGE_Y, LABEL_V, [E_T, E_IY, E_KY]),
    );
    assert!(is_noop(&b));
    indexes.apply(B, &b).await;
    assert_eq!(indexes.scores(&request(LABEL_U), &[A, B]).await, vec![3, 1]);
    assert_eq!(indexes.scores(&request(LABEL_V), &[A, B]).await, vec![1, 1]);

    // A evicts its image, which frees capacity. B evicts the excluded blocks
    // (nothing reaches the index) and stores Y again: now it is mapped.
    let a_evict = run(&mut normalizer, A, 2, removed(&[E_IX, E_KX]));
    indexes.apply(A, &a_evict).await;
    let b_evict = run(&mut normalizer, B, 3, removed(&[E_IY, E_KY]));
    assert!(is_noop(&b_evict));
    indexes.apply(B, &b_evict).await;
    let b_again = run(
        &mut normalizer,
        B,
        4,
        image_request_store(IMAGE_Y, LABEL_V, [E_T, E_IY, E_KY]),
    );
    indexes.apply(B, &b_again).await;
    assert_eq!(indexes.scores(&request(LABEL_V), &[A, B]).await, vec![1, 3]);
}

/// Exhausted exclusion state must not send unknown-parent stores to an index.
#[tokio::test]
async fn exclusion_overflow_drops_descendants_and_preserves_text_eviction() {
    let mut normalizer = labeled_normalizer().with_engine_mm_excluded_capacity(0);
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

    let child = run(
        &mut normalizer,
        A,
        2,
        stored(Some(E_IX), &[block(E_KX, text(300), vec![])]),
    );
    assert!(is_noop(&child));
    indexes.apply(A, &child).await;

    let query = request_hashes(&[text(100), routing_pad(LABEL_U, BLOCK), text(300)]);
    assert_eq!(indexes.scores(&query, &[A]).await, vec![1]);
    let evict = run(&mut normalizer, A, 3, removed(&[E_IX, E_KX, E_T]));
    let event = router_event(A, &evict);
    // Unknown removals are reported, but must not prevent the text eviction.
    assert_eq!(
        indexes.radix.apply_event(event.clone()),
        Err(KvCacheEventError::BlockNotFound)
    );
    indexes.crtc.apply_event_and_wait(event).await.unwrap();
    assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);
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
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![3]);
    assert_eq!(indexes.scores(&request(LABEL_V), &[A]).await, vec![1]);

    // The first mapping still owns the block, so its eviction reaches the index.
    let evict = run(&mut normalizer, A, 3, removed(&[E_KX, E_IX]));
    indexes.apply(A, &evict).await;
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
}

#[test]
fn cleared_releases_mapping_capacity() {
    let mut normalizer = labeled_normalizer().with_engine_mm_id_capacity(3);
    let initial = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    assert_eq!(store_data(&initial).unwrap().blocks.len(), 3);
    let cleared = run(
        &mut normalizer,
        A,
        2,
        Wire::Cleared(WireCleared {
            kind: "AllBlocksCleared",
        }),
    );
    assert!(matches!(cleared.event.data, KvCacheEventData::Cleared));
    let restored = run(
        &mut normalizer,
        A,
        3,
        image_request_store(IMAGE_X, LABEL_V, [E_T, E_IX, E_KX]),
    );
    assert_eq!(store_data(&restored).unwrap().blocks.len(), 3);
    let other = run(
        &mut normalizer,
        B,
        1,
        image_request_store(IMAGE_Y, LABEL_U, [E_T, E_IY, E_KY]),
    );
    assert!(
        is_noop(&other),
        "all mappings, including text, consume capacity"
    );
}

#[test]
fn storage_tier_replays_and_evictions_preserve_mapping_capacity() {
    for (first, second) in [("GPU", "CPU_PINNED"), ("CPU_PINNED", "GPU")] {
        let mut normalizer = labeled_normalizer()
            .with_engine_mm_excluded_capacity(16)
            .with_engine_mm_id_capacity(2);
        let image_store = |medium, id, label| {
            stored_on(
                None,
                &[block(
                    id,
                    image(IMAGE_X, BLOCK),
                    vec![(0, BLOCK as u32, label)],
                )],
                Some(medium),
            )
        };
        let gpu = run(&mut normalizer, A, 1, image_store("GPU", E_IX, LABEL_U));
        let routing_id = store_data(&gpu).unwrap().blocks[0].block_hash.0;
        let cpu = run(
            &mut normalizer,
            A,
            2,
            image_store("CPU_PINNED", E_IX, LABEL_U),
        );
        assert_eq!(store_data(&cpu).unwrap().blocks[0].block_hash.0, routing_id);

        let replay = run(
            &mut normalizer,
            A,
            3,
            image_store("CPU_PINNED", E_IX, LABEL_U),
        );
        assert!(is_noop(&replay));
        let unknown = run(&mut normalizer, A, 4, removed_on(&[E_IX], Some("DISK")));
        assert_eq!(removed_ids(&unknown), vec![E_IX]);
        let full = run(&mut normalizer, B, 1, image_store("GPU", E_IY, LABEL_V));
        assert!(is_noop(&full));

        let evict = run(&mut normalizer, A, 5, removed_on(&[E_IX], Some(first)));
        assert_eq!(removed_ids(&evict), vec![routing_id]);
        let admitted = run(&mut normalizer, B, 2, image_store("GPU", E_IY, LABEL_V));
        assert!(store_data(&admitted).is_some());
        let evict = run(&mut normalizer, A, 6, removed_on(&[E_IX], Some(second)));
        assert_eq!(removed_ids(&evict), vec![routing_id]);
        run(&mut normalizer, B, 3, cleared());
        let fresh_gpu = run(&mut normalizer, A, 7, image_store("GPU", E_IX, LABEL_V));
        let fresh_cpu = run(
            &mut normalizer,
            A,
            8,
            image_store("CPU_PINNED", E_IX, LABEL_V),
        );
        assert!(store_data(&fresh_gpu).is_some());
        assert!(store_data(&fresh_cpu).is_some());
        let full = run(&mut normalizer, B, 4, image_store("GPU", E_IY, LABEL_U));
        assert!(is_noop(&full));
    }
}

#[test]
fn unusable_second_tier_store_preserves_the_original_mapping() {
    for spans in [vec![], vec![(0, BLOCK as u32, LABEL_V)]] {
        let mut normalizer = labeled_normalizer();
        let gpu = run(
            &mut normalizer,
            A,
            1,
            stored(
                None,
                &[block(
                    E_IX,
                    image(IMAGE_X, BLOCK),
                    vec![(0, BLOCK as u32, LABEL_U)],
                )],
            ),
        );
        let routing_id = store_data(&gpu).unwrap().blocks[0].block_hash.0;
        let cpu = run(
            &mut normalizer,
            A,
            2,
            stored_on(
                None,
                &[block(E_IX, image(IMAGE_X, BLOCK), spans)],
                Some("CPU_PINNED"),
            ),
        );
        assert!(is_noop(&cpu));
        let cpu_evict = run(
            &mut normalizer,
            A,
            3,
            removed_on(&[E_IX], Some("CPU_PINNED")),
        );
        assert!(is_noop(&cpu_evict));
        let gpu_evict = run(&mut normalizer, A, 4, removed(&[E_IX]));
        assert_eq!(removed_ids(&gpu_evict), vec![routing_id]);
    }
}

#[test]
fn mapped_suffix_with_a_forgotten_parent_keeps_its_eviction_identity() {
    let mut normalizer = labeled_normalizer();
    let original = run(
        &mut normalizer,
        A,
        1,
        image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
    );
    let suffix_id = store_data(&original).unwrap().blocks[2].block_hash.0;
    run(&mut normalizer, A, 2, removed(&[E_IX]));
    let replay = run(
        &mut normalizer,
        A,
        3,
        stored(Some(E_IX), &[block(E_KX, text(300), vec![])]),
    );
    assert!(is_noop(&replay));
    let evict = run(&mut normalizer, A, 4, removed(&[E_KX]));
    assert_eq!(removed_ids(&evict), vec![suffix_id]);
}

#[tokio::test]
async fn unknown_parent_chain_is_excluded_until_replayed_from_root() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let child = || stored(Some(E_T), &[block(E_IX, text(300), vec![])]);
    let query = request_hashes(&[text(100), text(300)]);
    for (i, wire) in [
        child(),
        stored(Some(E_IX), &[block(E_KX, text(500), vec![])]),
        removed(&[E_IX, E_KX]),
    ]
    .into_iter()
    .enumerate()
    {
        let event = run(&mut normalizer, A, i as u64 + 1, wire);
        assert!(is_noop(&event));
        indexes.apply(A, &event).await;
    }
    assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);

    let root = run(
        &mut normalizer,
        A,
        4,
        stored(None, &[block(E_T, text(100), vec![])]),
    );
    indexes.apply(A, &root).await;
    let replay = run(&mut normalizer, A, 5, child());
    indexes.apply(A, &replay).await;
    assert_eq!(indexes.scores(&query, &[A]).await, vec![2]);
    let evict = run(&mut normalizer, A, 6, removed(&[E_IX, E_T]));
    indexes.apply(A, &evict).await;
    assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);
}

#[tokio::test]
async fn host_continuation_uses_the_gpu_parents_routing_identity() {
    let mut normalizer = labeled_normalizer();
    let mut indexes = Indexes::new();
    let lower = ThreadPoolIndexer::new(LowerTierIndexer::new(), 2, BLOCK as u32);
    let gpu = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[
                block(E_T, text(100), vec![]),
                block(
                    E_IX,
                    image(IMAGE_X, BLOCK),
                    vec![(0, BLOCK as u32, LABEL_U)],
                ),
            ],
        ),
    );
    let image_id = store_data(&gpu).unwrap().blocks[1].block_hash;
    indexes.apply(A, &gpu).await;
    let query = request(LABEL_U);
    assert_eq!(indexes.scores(&query, &[A]).await, vec![2]);

    // Write-back can put a text suffix on CPU without its image parent.
    let host = run(
        &mut normalizer,
        A,
        2,
        stored_on(
            Some(E_IX),
            &[block(E_KX, text(300), vec![])],
            Some("CPU_PINNED"),
        ),
    );
    lower
        .apply_event_and_wait(RouterEvent::with_storage_tier(
            A,
            host.event,
            StorageTier::HostPinned,
        ))
        .await
        .unwrap();
    let continuations =
        std::collections::HashMap::from([(worker(A), LowerTierContinuation::new(2, image_id))]);
    assert_eq!(
        lower
            .backend()
            .query_contiguous_hits(&query, &continuations)
            .get(&worker(A)),
        Some(&1)
    );

    let evict = run(
        &mut normalizer,
        A,
        3,
        removed_on(&[E_KX], Some("CPU_PINNED")),
    );
    lower
        .apply_event_and_wait(RouterEvent::with_storage_tier(
            A,
            evict.event,
            StorageTier::HostPinned,
        ))
        .await
        .unwrap();
    assert_eq!(
        lower
            .backend()
            .query_contiguous_hits(&query, &continuations)
            .get(&worker(A))
            .copied()
            .unwrap_or(0),
        0
    );
    assert_eq!(indexes.scores(&query, &[A]).await, vec![2]);
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

#[test]
fn text_only_events_preserve_hashes_and_translate_evictions() {
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
    let mut without_flag = run(&mut plain, A, 1, wire());
    let mapped = store_data(&with_flag).unwrap();
    let KvCacheEventData::Stored(native) = &mut without_flag.event.data else {
        panic!("text store");
    };
    assert_eq!(
        native
            .blocks
            .iter()
            .map(|block| block.block_hash.0)
            .collect::<Vec<_>>(),
        vec![E_T, E_KX]
    );
    for (native, mapped) in native.blocks.iter_mut().zip(&mapped.blocks) {
        assert_ne!(native.block_hash, mapped.block_hash);
        native.block_hash = mapped.block_hash;
    }
    assert_eq!(
        with_flag.event, without_flag.event,
        "only external ids change"
    );
    let evict = run(&mut labeled, A, 2, removed(&[E_T, E_KX]));
    assert_eq!(
        removed_ids(&evict),
        mapped
            .blocks
            .iter()
            .map(|block| block.block_hash.0)
            .collect::<Vec<_>>()
    );
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

fn with_sglang_salts(wire: Wire, engine: Option<&str>, caller: Option<&str>) -> Wire {
    let Wire::Stored(mut store) = wire else {
        unreachable!("salts belong to stores");
    };
    store.cache_salt = engine.map(str::to_owned);
    store.user_cache_salt = Some(caller.map(str::to_owned));
    Wire::Stored(store)
}

#[tokio::test]
async fn caller_namespace_text_prefix_aliases_survive_eviction_and_clear() {
    for caller in [None, Some("tenant-a")] {
        // A text node can be stored before its request's image node. Its native
        // id still includes the request's full internal multimodal salt.
        for first_is_plain_text in [false, true] {
            let mut normalizer = labeled_normalizer();
            let mut indexes = Indexes::new();
            let query = request_hashes_in_namespace(&[text(100)], caller);
            let first_salt = if first_is_plain_text {
                caller
            } else {
                Some(ENGINE_SALT_X)
            };
            let first = run(
                &mut normalizer,
                A,
                1,
                with_sglang_salts(
                    stored(None, &[block(E_T, text(100), vec![])]),
                    first_salt,
                    caller,
                ),
            );
            let second = run(
                &mut normalizer,
                A,
                2,
                with_sglang_salts(
                    stored(None, &[block(E_IY, text(100), vec![])]),
                    Some(ENGINE_SALT_Y),
                    caller,
                ),
            );
            indexes.apply(A, &first).await;
            indexes.apply(A, &second).await;
            assert_eq!(
                indexes.scores(&query, &[A]).await,
                vec![1],
                "caller={caller:?}, plain={first_is_plain_text}"
            );
            let first_evict = run(&mut normalizer, A, 3, removed(&[E_T]));
            indexes.apply(A, &first_evict).await;
            assert_eq!(
                indexes.scores(&query, &[A]).await,
                vec![1],
                "remaining text alias must stay resident"
            );
            let second_evict = run(&mut normalizer, A, 4, removed(&[E_IY]));
            indexes.apply(A, &second_evict).await;
            assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);

            let again = || {
                with_sglang_salts(
                    stored(None, &[block(E_T, text(100), vec![])]),
                    first_salt,
                    caller,
                )
            };
            let store = run(&mut normalizer, A, 5, again());
            indexes.apply(A, &store).await;
            let clear = run(&mut normalizer, A, 6, cleared());
            indexes.apply(A, &clear).await;
            assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);
            let store = run(&mut normalizer, A, 7, again());
            indexes.apply(A, &store).await;
            assert_eq!(indexes.scores(&query, &[A]).await, vec![1]);
            let evict = run(&mut normalizer, A, 8, removed(&[E_T]));
            indexes.apply(A, &evict).await;
            assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);
        }
    }
}

#[tokio::test]
async fn caller_namespace_image_aliases_route_and_evict_across_workers() {
    for caller in [None, Some("tenant-a")] {
        let mut normalizer = labeled_normalizer();
        let mut indexes = Indexes::new();
        let tokens = [text(100), routing_pad(LABEL_U, BLOCK), text(300)];
        let query = request_hashes_in_namespace(&tokens, caller);
        let other_namespace = request_hashes_in_namespace(&tokens, Some("tenant-b"));
        let first = run(
            &mut normalizer,
            A,
            1,
            with_sglang_salts(
                image_request_store(IMAGE_X, LABEL_U, [E_T, E_IX, E_KX]),
                Some(ENGINE_SALT_X),
                caller,
            ),
        );
        // Even the leading text block gets a different native id when the
        // same URL resolves to another image and SGLang changes its full salt.
        let second_text_id = E_T + 1;
        let second = run(
            &mut normalizer,
            B,
            1,
            with_sglang_salts(
                image_request_store(IMAGE_Y, LABEL_U, [second_text_id, E_IY, E_KY]),
                Some(ENGINE_SALT_Y),
                caller,
            ),
        );
        indexes.apply(A, &first).await;
        indexes.apply(B, &second).await;
        assert_eq!(indexes.scores(&query, &[A, B]).await, vec![3, 3]);
        assert_eq!(indexes.scores(&other_namespace, &[A, B]).await, vec![0, 0]);
        let evict = run(
            &mut normalizer,
            B,
            2,
            removed(&[E_KY, E_IY, second_text_id]),
        );
        indexes.apply(B, &evict).await;
        assert_eq!(indexes.scores(&query, &[A, B]).await, vec![3, 0]);
        let evict = run(&mut normalizer, A, 2, removed(&[E_KX, E_IX, E_T]));
        indexes.apply(A, &evict).await;
        assert_eq!(indexes.scores(&query, &[A, B]).await, vec![0, 0]);
    }
}
