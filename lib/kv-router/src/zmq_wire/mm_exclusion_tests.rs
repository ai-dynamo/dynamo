// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang sources with engine-owned multimodal identity: wide-token decoding
//! and exclusion of multimodal blocks.
//!
//! Events are encoded the way SGLang publishes them (tagged msgpack maps) and
//! decoded with the production decoder, so these tests cover decoding,
//! normalization and indexing together.

use serde_json::{Value, json};

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
const LABEL_U: u64 = 0x9e37_79b9_7f4a_7c15;
const IMAGE_X: [u64; 4] = [1 << 40, 2 << 40, 3 << 40, 4 << 40];
const E_T: u64 = 1;
const E_IX: u64 = 2;
const E_KX: u64 = 3;

fn text(start: u64) -> Vec<u64> {
    (start..start + BLOCK as u64).collect()
}

fn image(values: [u64; 4], len: usize) -> Vec<u64> {
    (0..len).map(|index| values[index % 4]).collect()
}

fn routing_pad(label: u64, len: usize) -> Vec<u64> {
    vec![u64::from(pad_value_for_mm_hash(label)); len]
}

fn block(id: u64, tokens: Vec<u64>) -> (u64, Vec<u64>) {
    (id, tokens)
}

fn stored(parent: Option<u64>, blocks: &[(u64, Vec<u64>)]) -> Value {
    json!({
        "type": "BlockStored",
        "block_hashes": blocks.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        "parent_block_hash": parent,
        "token_ids": blocks.iter().flat_map(|(_, tokens)| tokens).collect::<Vec<_>>(),
        "block_size": BLOCK,
    })
}

fn stored_eagle(ids: &[u64], tokens: &[u64]) -> Value {
    json!({
        "type": "BlockStored",
        "block_hashes": ids,
        "parent_block_hash": null,
        "token_ids": tokens.windows(2).collect::<Vec<_>>(),
        "block_size": BLOCK,
    })
}

fn removed(ids: &[u64]) -> Value {
    json!({"type": "BlockRemoved", "block_hashes": ids})
}

fn cleared() -> Value {
    json!({"type": "AllBlocksCleared"})
}

fn decode_one(wire: Value) -> RawKvEvent {
    let bytes = rmp_serde::to_vec_named(&(0.0f64, vec![wire], None::<i32>)).unwrap();
    decode_event_batch(&bytes).unwrap().events.remove(0)
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
    wire: Value,
) -> PlacementEvent {
    run_on(normalizer, worker(worker_id), event_id, wire)
}

fn run_on(
    normalizer: &mut ZmqEventNormalizer,
    worker: WorkerWithDpRank,
    event_id: u64,
    wire: Value,
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

#[test]
fn native_clear_preserves_gpu_mm_exclusion_until_its_rank_removes_it() {
    let mut normalizer = engine_owned_normalizer();
    let other_rank = WorkerWithDpRank::new(A, 1);
    let gpu = |mut wire: Value| {
        wire["medium"] = json!("gpu_direct");
        wire
    };
    let image_store = || stored(None, &[block(E_IX, image(IMAGE_X, BLOCK))]);
    let child_store = || stored(Some(E_IX), &[block(E_KX, text(300))]);

    for rank in [worker(A), other_rank] {
        assert!(is_noop(&run_on(
            &mut normalizer,
            rank,
            1,
            gpu(image_store())
        )));
    }
    assert!(is_noop(&run(&mut normalizer, A, 2, image_store())));
    run(&mut normalizer, A, 3, cleared());

    assert!(store_data(&run(&mut normalizer, A, 4, child_store())).is_some());
    assert!(is_noop(&run(&mut normalizer, A, 5, gpu(child_store()))));
    assert!(is_noop(&run_on(
        &mut normalizer,
        other_rank,
        2,
        gpu(removed(&[E_IX]))
    )));
    assert!(store_data(&run_on(&mut normalizer, other_rank, 3, gpu(child_store()))).is_some());
    assert!(is_noop(&run(&mut normalizer, A, 6, gpu(child_store()))));
    assert!(is_noop(&run(&mut normalizer, A, 7, gpu(removed(&[E_IX])))));
    assert!(store_data(&run(&mut normalizer, A, 8, gpu(child_store()))).is_some());
}

#[test]
fn native_clear_preserves_gpu_overflow_without_retaining_native_overflow() {
    let mut normalizer = engine_owned_normalizer().with_engine_mm_excluded_capacity(0);
    let gpu = |mut wire: Value| {
        wire["medium"] = json!("gpu_direct");
        wire
    };
    let image_store = stored(None, &[block(E_IX, image(IMAGE_X, BLOCK))]);
    assert!(is_noop(&run(&mut normalizer, A, 1, gpu(image_store))));
    run(&mut normalizer, A, 2, cleared());

    let child = || stored(Some(E_IX), &[block(E_KX, text(300))]);
    assert!(is_noop(&run(&mut normalizer, A, 3, gpu(child()))));
    assert!(store_data(&run(&mut normalizer, A, 4, child())).is_some());
    assert!(
        store_data(&run_on(
            &mut normalizer,
            WorkerWithDpRank::new(A, 1),
            1,
            gpu(child()),
        ))
        .is_some()
    );
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
    for (engine_owned, pads) in [
        (true, IMAGE_X),
        (false, IMAGE_X),
        (true, [1_000_000, 1_000_001, 1_000_002, 1_000_003]),
    ] {
        let mut normalizer =
            ZmqEventNormalizer::new(BLOCK as u32).with_engine_owned_mm_identity(engine_owned);
        let mut indexes = Indexes::new();

        let event = run(
            &mut normalizer,
            A,
            1,
            stored(
                None,
                &[
                    block(E_T, text(100)),
                    block(E_IX, image(pads, BLOCK)),
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
        assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
        indexes.apply(A, &evict).await;

        assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![0]);
    }
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
    let mut host_store = stored(None, &[block(E_IX, image(IMAGE_X, BLOCK))]);
    host_store["medium"] = json!("CPU_PINNED");
    let host = run(&mut normalizer, A, 2, host_store);
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

/// Overflow must preserve the relay source, existing text evictions and root
/// stores. A cache clear reopens continuations only for the cleared rank.
#[tokio::test]
async fn exclusion_overflow_preserves_the_relay_source_and_text_evictions() {
    use crate::indexer::cuckoo::{CkfConfig, DcCkfState};

    let mut normalizer = engine_owned_normalizer().with_engine_mm_excluded_capacity(1);
    let mut indexes = Indexes::new();
    let mut relay = DcCkfState::new(CkfConfig::new(64)).unwrap();
    let continuation = || stored(Some(E_T), &[block(E_KX, text(300))]);
    let wires = [
        stored(None, &[block(E_T, text(100))]),
        continuation(),
        stored(
            None,
            &[block(E_IX, image(IMAGE_X, BLOCK)), block(4, text(500))],
        ),
        stored(Some(4), &[block(5, text(600))]),
        continuation(),
        stored(None, &[block(6, text(700))]),
    ];
    for (i, wire) in wires.into_iter().enumerate() {
        let event = run(&mut normalizer, A, i as u64 + 1, wire);
        assert_eq!(is_noop(&event), (2..=4).contains(&i));
        indexes.apply(A, &event).await;
        assert_eq!(
            relay.apply_event(router_event(A, &event)).first_error(),
            None
        );
    }
    let query = request_hashes(&[text(100), text(300)]);
    assert_eq!(indexes.scores(&query, &[A]).await, vec![2]);
    assert_eq!(
        indexes.scores(&request_hashes(&[text(700)]), &[A]).await,
        vec![1]
    );

    // Unknown image removals must not hide evictions of text indexed before
    // overflow, including the continuation whose replay was suppressed.
    let evict = run(&mut normalizer, A, 7, removed(&[E_IX, 4, 5, E_KX, E_T, 6]));
    assert_eq!(removed_ids(&evict), vec![4, 5, E_KX, E_T, 6]);
    let event = router_event(A, &evict);
    assert_eq!(relay.apply_event(event.clone()).first_error(), None);
    assert_eq!(
        indexes.radix.apply_event(event.clone()),
        Err(crate::protocols::KvCacheEventError::BlockNotFound)
    );
    indexes.crtc.apply_event_and_wait(event).await.unwrap();
    assert_eq!(indexes.scores(&query, &[A]).await, vec![0]);
    assert_eq!(
        indexes.scores(&request_hashes(&[text(700)]), &[A]).await,
        vec![0]
    );

    let scopes = [worker(A), WorkerWithDpRank::new(A, 1), worker(A + 1)];
    for &scope in &scopes[1..] {
        run_on(
            &mut normalizer,
            scope,
            1,
            stored(
                None,
                &[block(E_IX, image(IMAGE_X, BLOCK)), block(4, text(500))],
            ),
        );
    }
    let clear = run(&mut normalizer, A, 8, cleared());
    indexes.apply(A, &clear).await;
    assert_eq!(
        relay.apply_event(router_event(A, &clear)).first_error(),
        None
    );
    for scope in scopes {
        run_on(
            &mut normalizer,
            scope,
            9,
            stored(None, &[block(E_T, text(100))]),
        );
        let event = run_on(&mut normalizer, scope, 10, continuation());
        assert_eq!(store_data(&event).is_some(), scope == worker(A));
    }
}

#[tokio::test]
async fn malformed_multimodal_stores_do_not_suppress_text_eviction() {
    for excluded_parent in [false, true] {
        let mut normalizer = engine_owned_normalizer();
        let mut indexes = Indexes::new();
        let text_store = run(
            &mut normalizer,
            A,
            1,
            stored(None, &[block(E_T, text(100))]),
        );
        indexes.apply(A, &text_store).await;
        let malformed = if excluded_parent {
            run(
                &mut normalizer,
                A,
                2,
                stored(None, &[block(E_IX, image(IMAGE_X, BLOCK))]),
            );
            stored(Some(E_IX), &[block(E_T, text(100)), block(E_T, text(300))])
        } else {
            stored(
                None,
                &[
                    block(E_T, text(100)),
                    block(E_IX, image(IMAGE_X, BLOCK)),
                    block(E_T, text(300)),
                ],
            )
        };
        let mut raw = decode_one(malformed);
        let before = rmp_serde::to_vec_named(&raw).unwrap();
        assert!(
            normalizer
                .engine_mm
                .prepare_store(&mut raw, worker(A), StorageTier::Device),
            "leave malformed stores for convert_event to reject and log"
        );
        assert_eq!(rmp_serde::to_vec_named(&raw).unwrap(), before);
        let rejected = normalizer
            .normalize_preprocessed(raw, 3, worker(A))
            .unwrap();
        assert!(is_noop(&rejected));
        indexes.apply(A, &rejected).await;
        assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![1]);
        let evict = run(&mut normalizer, A, 4, removed(&[E_T]));
        assert_eq!(removed_ids(&evict), vec![E_T]);
        indexes.apply(A, &evict).await;
        assert_eq!(indexes.scores(&request(LABEL_U), &[A]).await, vec![0]);
    }
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

    let mut host_remove = removed(&[E_IX]);
    host_remove["medium"] = json!("CPU_PINNED");
    let host = run_on(&mut normalizer, rank0, 2, host_remove);
    assert_eq!(removed_ids(&host), vec![E_IX]);

    let clear = run_on(&mut normalizer, rank0, 3, cleared());
    assert!(matches!(clear.event.data, KvCacheEventData::Cleared));
    let rank0_evict = run_on(&mut normalizer, rank0, 4, removed(&[E_IX, E_KX]));
    assert_eq!(removed_ids(&rank0_evict), vec![E_IX, E_KX]);
    let rank1_evict = run_on(&mut normalizer, rank1, 2, removed(&[E_IX, E_KX]));
    assert!(removed_ids(&rank1_evict).is_empty());
}
