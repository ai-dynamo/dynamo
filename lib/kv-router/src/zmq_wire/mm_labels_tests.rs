// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang sources with engine-owned multimodal identity: wide-token decoding
//! and exclusion of multimodal blocks.
//!
//! Events are encoded the way SGLang publishes them (tagged msgpack maps) and
//! decoded with the production decoder, so these tests cover decoding,
//! normalization and indexing together.

use serde::Serialize;

use super::*;
use crate::PositionalIndexer;
use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
use crate::indexer::{KvIndexerInterface, LowerTierIndexer, RadixTree, ThreadPoolIndexer};
use crate::protocols::{
    BlockHashOptions, KvCacheStoreData, LocalBlockHash, RouterEvent, compute_block_hash_for_seq,
    pad_value_for_mm_hash,
};

const BLOCK: usize = 16;
const A: u64 = 1;

/// Dynamo's per-item hash for an image, e.g. of its URL.
const LABEL_U: u64 = 0x9e37_79b9_7f4a_7c15;

/// SGLang pad values for one image (four 62-bit values).
const IMAGE_X: [u64; 4] = [
    1_102_202_002_673_214_419,
    2_808_702_701_737_579_037,
    2_713_072_869_372_394_503,
    2_713_360_236_592_209_549,
];

// Engine (SGLang) block ids.
const E_T: u64 = 0x7e57_0000_0000_0001;
const E_IX: u64 = 0x1a6e_0000_0000_00a1;
const E_KX: u64 = 0x1a6e_0000_0000_00a2;

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
    StoredBigram(WireStoredBigram),
    Removed(WireRemoved),
    Cleared(WireCleared),
}

struct Block {
    id: u64,
    tokens: Vec<u64>,
}

fn block(id: u64, tokens: Vec<u64>) -> Block {
    assert_eq!(tokens.len(), BLOCK);
    Block { id, tokens }
}

fn stored(parent: Option<u64>, blocks: &[Block]) -> Wire {
    stored_on(parent, blocks, None)
}

fn stored_on(parent: Option<u64>, blocks: &[Block], medium: Option<&'static str>) -> Wire {
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

fn engine_owned_normalizer() -> ZmqEventNormalizer {
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

fn request_hashes(blocks: &[Vec<u64>]) -> Vec<LocalBlockHash> {
    hashes(&blocks.concat(), false)
}

/// Request: text, one image, text. Routing tokens use Dynamo's pad.
fn request(label: u64) -> Vec<LocalBlockHash> {
    request_hashes(&[text(100), routing_pad(label, BLOCK), text(300)])
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

#[tokio::test]
async fn image_is_excluded_with_its_descendants() {
    let mut normalizer = engine_owned_normalizer();
    let mut indexes = Indexes::new();

    let event = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[
                block(E_T, text(100)),
                block(E_IX, image(IMAGE_X, BLOCK)),
                block(E_KX, text(300)),
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
        stored(Some(E_KX), &[block(0xc0de, text(500))]),
    );
    assert!(is_noop(&child));

    // Removals of excluded blocks never reach the index.
    let evict = run(&mut normalizer, A, 3, removed(&[E_IX, E_KX, 0xc0de, E_T]));
    assert_eq!(removed_ids(&evict), vec![E_T]);

    let stats = normalizer.engine_mm_stats();
    assert_eq!(stats.excluded_unmapped, 2);
    assert_eq!(stats.excluded_parent, 1);
    assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
}

/// A store dropped entirely becomes an empty removal so the event id sequence
/// has no gap. Every index must take it as a no-op, including for a worker it
/// has not seen yet (an image-first request right after startup).
#[tokio::test]
async fn fully_excluded_store_is_a_noop_for_an_unseen_worker() {
    let mut normalizer = engine_owned_normalizer();
    let mut indexes = Indexes::new();
    let event = run(
        &mut normalizer,
        A,
        1,
        stored(None, &[block(E_IX, image(IMAGE_X, BLOCK))]),
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
            &[block(E_IX, image(IMAGE_X, BLOCK))],
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
    let mut normalizer = engine_owned_normalizer().with_engine_mm_excluded_capacity(0);
    let mut indexes = Indexes::new();
    let first = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[block(E_T, text(100)), block(E_IX, image(IMAGE_X, BLOCK))],
        ),
    );
    indexes.apply(A, &first).await;
    assert_eq!(normalizer.engine_mm_stats().excluded_overflow, 1);

    let child = run(
        &mut normalizer,
        A,
        2,
        stored(Some(E_IX), &[block(E_KX, text(300))]),
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
            engine_owned_normalizer().with_engine_mm_excluded_capacity(excluded_capacity);
        let first = run(
            &mut normalizer,
            A,
            1,
            stored(
                None,
                &[block(E_T, text(100)), block(E_IX, image(IMAGE_X, BLOCK))],
            ),
        );
        let child = run(
            &mut normalizer,
            A,
            2,
            stored(Some(E_IX), &[block(E_KX, text(300))]),
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
    let mut normalizer = engine_owned_normalizer();
    // An image starts in the last two tokens of the text block; its values
    // happen to fit in u32, so only the >= 1_000_000 rule catches them.
    let mut first = text(100);
    first[BLOCK - 2] = 1_500_000;
    first[BLOCK - 1] = 1_600_000;
    let event = run(
        &mut normalizer,
        A,
        1,
        stored(None, &[block(E_T, text(50)), block(E_IX, first)]),
    );
    let store = store_data(&event).unwrap();
    assert_eq!(
        store.blocks.len(),
        1,
        "the block with multimodal values is cut"
    );
    assert_eq!(normalizer.engine_mm_stats().excluded_unmapped, 1);
}

/// Without the engine-owned flag, every value that did not fit in `u32`
/// narrows to one sentinel; indexing those blocks would make different images
/// share a local hash. They are excluded like engine-owned values, descendants
/// and removals included, so a later chained store never reaches the index
/// under an unknown parent.
#[test]
fn unflagged_source_excludes_wide_blocks_and_their_descendants() {
    let mut normalizer = ZmqEventNormalizer::new(BLOCK as u32);
    let event = run(
        &mut normalizer,
        A,
        1,
        stored(
            None,
            &[
                block(E_T, text(100)),
                block(E_IX, image(IMAGE_X, BLOCK)),
                block(E_KX, text(300)),
            ],
        ),
    );
    let store = store_data(&event).unwrap();
    assert_eq!(store.blocks.len(), 1);
    assert_eq!(store.blocks[0].block_hash.0, E_T);

    let child = run(
        &mut normalizer,
        A,
        2,
        stored(Some(E_KX), &[block(0xc0de, text(500))]),
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
            &[block(E_T, text(50)), block(E_IX, legacy_pad.clone())],
        )
    };

    let mut unflagged = ZmqEventNormalizer::new(BLOCK as u32);
    let kept = run(&mut unflagged, A, 1, wire());
    assert_eq!(store_data(&kept).unwrap().blocks.len(), 2);

    let mut flagged = engine_owned_normalizer();
    let cut = run(&mut flagged, A, 1, wire());
    assert_eq!(store_data(&cut).unwrap().blocks.len(), 1);
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
    let mut normalizer = engine_owned_normalizer();
    let rank0 = WorkerWithDpRank::new(A, 0);
    let rank1 = WorkerWithDpRank::new(A, 1);
    let image_store = || {
        stored(
            None,
            &[
                block(E_T, text(100)),
                block(E_IX, image(IMAGE_X, BLOCK)),
                block(E_KX, text(300)),
            ],
        )
    };
    run_on(&mut normalizer, rank0, 1, image_store());
    run_on(&mut normalizer, rank1, 1, image_store());

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

#[test]
fn text_only_events_are_unchanged() {
    let mut flagged = engine_owned_normalizer();
    let mut plain = ZmqEventNormalizer::new(BLOCK as u32);
    let wire = || stored(None, &[block(E_T, text(100)), block(E_KX, text(300))]);
    let with_flag = run(&mut flagged, A, 1, wire());
    let without_flag = run(&mut plain, A, 1, wire());
    assert_eq!(with_flag.event, without_flag.event);
}
