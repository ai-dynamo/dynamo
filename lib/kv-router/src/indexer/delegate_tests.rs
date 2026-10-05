// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use rstest::rstest;
use tokio_util::sync::CancellationToken;

use super::*;
use crate::protocols::*;
use crate::test_utils::{
    make_clear_event_with_dp_rank, make_remove_event, make_store_event_with_dp_rank,
};

#[derive(Default)]
struct Recorder(Mutex<Vec<(bool, ExternalSequenceBlockHash)>>);

impl KvIndexerDelegate for Recorder {
    fn on_create(&self, hash: ExternalSequenceBlockHash) {
        self.0.lock().unwrap().push((true, hash));
    }

    fn on_remove(&self, hash: ExternalSequenceBlockHash) {
        self.0.lock().unwrap().push((false, hash));
    }
}

impl Recorder {
    fn take(&self) -> Vec<(bool, ExternalSequenceBlockHash)> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

fn indexer(variant: &str, delegate: Arc<Recorder>) -> Box<dyn KvIndexerInterface + Sync> {
    match variant {
        "single" => Box::new(
            KvIndexer::builder(
                CancellationToken::new(),
                32,
                Arc::new(KvIndexerMetrics::new_unregistered()),
            )
            .delegate(delegate)
            .build(),
        ),
        "compressed" => Box::new(ThreadPoolIndexer::new(
            concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed::new_with_delegate(
                delegate,
            ),
            4,
            32,
        )),
        "positional" => Box::new(ThreadPoolIndexer::new(
            positional::PositionalIndexer::new_with_delegate(
                32,
                positional::SearchMode::Strided,
                delegate,
            ),
            4,
            32,
        )),
        "lower" => Box::new(ThreadPoolIndexer::new(
            LowerTierIndexer::new_with_delegate(delegate),
            4,
            32,
        )),
        "local" => Box::new(LocalKvIndexer::new_with_delegate(
            CancellationToken::new(),
            32,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            128,
            delegate,
        )),
        "sharded" => Box::new(BranchShardedIndexer::new_with_delegate(
            2, 4, 1, 32, delegate,
        )),
        _ => unreachable!(),
    }
}

#[rstest]
#[tokio::test]
async fn delegate_tracks_first_and_last_owner(
    #[values("single", "compressed", "positional", "lower", "local", "sharded")] variant: &str,
) {
    let recorder = Arc::new(Recorder::default());
    let indexer = indexer(variant, recorder.clone());
    let hash = ExternalSequenceBlockHash(17);
    for (worker, rank) in [(1, 0), (1, 0), (1, 1), (2, 0)] {
        indexer
            .apply_event(make_store_event_with_dp_rank(worker, &[17], rank))
            .await;
        indexer.flush().await;
    }
    assert_eq!(recorder.take(), vec![(true, hash)]);

    indexer.remove_worker_dp_rank(1, 0).await;
    indexer.flush().await;
    indexer
        .apply_event(make_clear_event_with_dp_rank(1, 1))
        .await;
    indexer.flush().await;
    assert!(recorder.take().is_empty());

    indexer.remove_worker(2).await;
    indexer.flush().await;
    assert_eq!(recorder.take(), vec![(false, hash)]);
    indexer.apply_event(make_remove_event(2, &[17])).await;
    indexer.flush().await;
    assert!(recorder.take().is_empty());

    indexer
        .apply_event(make_store_event_with_dp_rank(2, &[17], 0))
        .await;
    indexer.flush().await;
    indexer.apply_event(make_remove_event(2, &[17])).await;
    indexer.flush().await;
    assert_eq!(recorder.take(), vec![(true, hash), (false, hash)]);
    indexer.shutdown();
}

#[rstest]
#[tokio::test]
async fn delegate_shared_prefix_split_and_clear(
    #[values("single", "compressed", "positional", "lower", "local", "sharded")] variant: &str,
) {
    let recorder = Arc::new(Recorder::default());
    let indexer = indexer(variant, recorder.clone());
    for (worker, hashes) in [(1, vec![10, 20, 30]), (2, vec![10, 20, 40])] {
        indexer
            .apply_event(make_store_event_with_dp_rank(worker, &hashes, 0))
            .await;
        indexer.flush().await;
    }
    let created = recorder.take();
    assert_eq!(created.len(), 4, "{created:?}");
    assert!(created.iter().all(|(create, _)| *create));
    indexer.remove_worker(1).await;
    indexer.flush().await;
    assert_eq!(recorder.take().len(), 1);
    indexer.remove_worker(2).await;
    indexer.flush().await;
    let removed = recorder.take();
    assert_eq!(removed.len(), 3, "{removed:?}");
    assert!(removed.iter().all(|(create, _)| !*create));
    indexer.shutdown();
}

#[rstest]
#[tokio::test]
async fn delegate_parallel_owners_and_reset(
    #[values("single", "compressed", "positional")] variant: &str,
) {
    let recorder = Arc::new(Recorder::default());
    let indexer = indexer(variant, recorder.clone());
    for worker in 0..64 {
        indexer
            .apply_event(make_store_event_with_dp_rank(worker, &[55], 0))
            .await;
    }
    indexer.flush().await;
    assert_eq!(recorder.take(), vec![(true, ExternalSequenceBlockHash(55))]);
    for worker in 0..64 {
        indexer
            .reset_worker_dp_rank_and_wait(worker, 0)
            .await
            .unwrap();
    }
    assert_eq!(
        recorder.take(),
        vec![(false, ExternalSequenceBlockHash(55))]
    );
    indexer.shutdown();
}

#[rstest]
#[tokio::test]
async fn delegate_rejected_store_is_silent(
    #[values("single", "compressed", "positional")] variant: &str,
) {
    let recorder = Arc::new(Recorder::default());
    let indexer = indexer(variant, recorder.clone());
    indexer
        .apply_event(crate::test_utils::make_store_event_with_parent(
            1,
            &[99],
            &[100],
        ))
        .await;
    indexer.flush().await;
    assert!(recorder.take().is_empty());
    indexer.shutdown();
}

#[derive(Default)]
pub(super) struct CanonicalRecorder(Mutex<Vec<(bool, cuckoo::CanonicalSequenceBlockHash)>>);

impl CanonicalRecorder {
    pub(super) fn take(&self) -> Vec<(bool, cuckoo::CanonicalSequenceBlockHash)> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl KvIndexerDelegate<cuckoo::CanonicalSequenceBlockHash> for CanonicalRecorder {
    fn on_create(&self, hash: cuckoo::CanonicalSequenceBlockHash) {
        self.0.lock().unwrap().push((true, hash));
    }
    fn on_remove(&self, hash: cuckoo::CanonicalSequenceBlockHash) {
        self.0.lock().unwrap().push((false, hash));
    }
}

#[test]
fn delegate_cuckoo_replacement_preserves_shared_ownership() {
    use cuckoo::*;
    let recorder = Arc::new(CanonicalRecorder::default());
    let mut indexer = DcCkfState::new_with_delegate(CkfConfig::new(128), recorder.clone()).unwrap();
    for worker in [1, 1, 2] {
        let result = indexer.apply_event(make_store_event_with_dp_rank(worker, &[17], 0));
        assert!(result.first_error().is_none());
    }
    let hash = CanonicalSequenceBlockHash::root(LocalBlockHash(17));
    assert_eq!(*recorder.0.lock().unwrap(), vec![(true, hash)]);
    indexer
        .replace_rank(WorkerWithDpRank::new(1, 0), DcCkfRankReplacement::default())
        .unwrap();
    assert_eq!(recorder.0.lock().unwrap().len(), 1);
    indexer
        .replace_rank(WorkerWithDpRank::new(2, 0), DcCkfRankReplacement::default())
        .unwrap();
    assert_eq!(
        *recorder.0.lock().unwrap(),
        vec![(true, hash), (false, hash)]
    );
    indexer.apply_event(make_store_event_with_dp_rank(2, &[17], 0));
    indexer.apply_event(make_remove_event(2, &[17]));
    assert_eq!(
        *recorder.0.lock().unwrap(),
        vec![(true, hash), (false, hash), (true, hash), (false, hash)]
    );
}

#[test]
fn delegate_cuckoo_nonempty_replacement_reports_only_ownership_changes() {
    use cuckoo::*;
    let recorder = Arc::new(CanonicalRecorder::default());
    let mut indexer = DcCkfState::new_with_delegate(CkfConfig::new(128), recorder.clone()).unwrap();
    let worker = WorkerWithDpRank::new(1, 0);
    for (owner, hash) in [(1, 17), (1, 19), (2, 19)] {
        assert!(
            indexer
                .apply_event(make_store_event_with_dp_rank(owner, &[hash], 0))
                .first_error()
                .is_none()
        );
    }
    recorder.take();
    let mut replacement = DcCkfRankReplacement::default();
    for hash in [17, 23] {
        replacement
            .push_event(make_store_event_with_dp_rank(1, &[hash], 0))
            .unwrap();
    }
    indexer.replace_rank(worker, replacement).unwrap();
    let canonical = |hash| CanonicalSequenceBlockHash::root(LocalBlockHash(hash));
    assert_eq!(recorder.take(), vec![(true, canonical(23))]);
    indexer
        .replace_rank(WorkerWithDpRank::new(2, 0), DcCkfRankReplacement::default())
        .unwrap();
    assert_eq!(recorder.take(), vec![(false, canonical(19))]);
    let mut replacement = DcCkfRankReplacement::default();
    replacement
        .push_event(make_store_event_with_dp_rank(1, &[23], 0))
        .unwrap();
    indexer.replace_rank(worker, replacement).unwrap();
    assert_eq!(recorder.take(), vec![(false, canonical(17))]);
}

#[rstest]
#[tokio::test]
async fn delegate_matches_dump_after_mixed_owner_changes(
    #[values("single", "compressed", "positional", "lower", "local", "sharded")] variant: &str,
) {
    let recorder = Arc::new(Recorder::default());
    let indexer = indexer(variant, recorder.clone());
    let mut notified = std::collections::BTreeSet::new();
    for step in 0..48 {
        let worker = step % 4;
        let rank = (step / 4) % 2;
        match step % 7 {
            0 => indexer.remove_worker(worker).await,
            1 => indexer.remove_worker_dp_rank(worker, rank as u32).await,
            2 => {
                indexer
                    .apply_event(make_clear_event_with_dp_rank(worker, rank as u32))
                    .await
            }
            _ => {
                indexer
                    .apply_event(make_store_event_with_dp_rank(
                        worker,
                        &[7, 8, 10 + worker % 2],
                        rank as u32,
                    ))
                    .await
            }
        }
        indexer.flush().await;
        for (created, hash) in recorder.take() {
            if created {
                assert!(
                    notified.insert(hash),
                    "duplicate create at step {step}: {hash:?}"
                );
            } else {
                assert!(
                    notified.remove(&hash),
                    "unpaired remove at step {step}: {hash:?}"
                );
            }
        }
        let mut indexed = std::collections::BTreeSet::new();
        for event in indexer.dump_events().await.unwrap() {
            if let KvCacheEventData::Stored(store) = event.event.data {
                indexed.extend(store.blocks.into_iter().map(|block| block.block_hash));
            }
        }
        assert_eq!(notified, indexed, "{variant}, step {step}");
    }
    indexer.shutdown();
    assert!(
        recorder.take().is_empty(),
        "shutdown must not synthesize evictions"
    );
}

// A delegate consumer that tracks blocks fleet-wide needs one key per content across workers,
// and, to join a request to its blocks, a key equal to the hash the frontend computes for the
// request. The tests below take vLLM-format BlockStored events through the ZMQ decoder and the
// normalizer, as the event listener does.

/// Block size and prompt shared with the vLLM block hasher test,
/// `components/src/dynamo/vllm/tests/test_vllm_block_hash_identity.py`.
const VLLM_BLOCK_SIZE: u32 = 16;

fn vllm_prompt() -> Vec<u32> {
    (1..=4 * VLLM_BLOCK_SIZE).collect()
}

/// vLLM 0.30 KV event block hashes of `vllm_prompt()` (sha256, integer event hashes) on a worker
/// with `PYTHONHASHSEED=0`, and on one with `PYTHONHASHSEED=1`. The Python test asserts the same
/// values against vLLM's own block hasher.
const VLLM_HASHES_SEED_0: [u64; 4] = [
    0x6896_072a_e8b3_1325,
    0x7d92_dcbb_5b55_9e31,
    0xbbc7_ecb5_486a_3322,
    0xc0c5_12ea_dd3f_7a91,
];
const VLLM_HASHES_SEED_1: [u64; 4] = [
    0xb3c6_2a8b_8d1a_4706,
    0x91d3_abee_af6a_2ace,
    0x9bd6_dacd_8202_afd2,
    0x9ed1_2b85_db86_845e,
];

#[derive(serde::Serialize)]
struct SglangNamespace {
    cache_salt: &'static str,
}

/// Where a BlockStored event carries its LoRA adapter or cache salt.
#[derive(Clone, Copy, Debug)]
enum Scope {
    Plain,
    Lora(&'static str),
    CacheSalt(&'static str),
}

/// One BlockStored event in the msgpack sequence form, decoded as the ZMQ listener does.
/// Position 7 is the LoRA name (vLLM) or a map with the cache salt (SGLang).
fn vllm_stored(hashes: &[u64], tokens: &[u32], scope: Scope) -> crate::zmq_wire::RawKvEvent {
    use crate::zmq_wire::BlockHashValue;
    let hashes: Vec<BlockHashValue> = hashes
        .iter()
        .copied()
        .map(BlockHashValue::Unsigned)
        .collect();
    let (tag, parent, size, lora_id, medium) = (
        "BlockStored",
        Option::<BlockHashValue>::None,
        VLLM_BLOCK_SIZE as usize,
        Option::<u64>::None,
        Option::<String>::None,
    );
    let tokens = tokens.to_vec();
    let bytes = match scope {
        Scope::Plain => rmp_serde::to_vec(&(tag, hashes, parent, tokens, size, lora_id, medium)),
        Scope::Lora(name) => {
            rmp_serde::to_vec(&(tag, hashes, parent, tokens, size, lora_id, medium, name))
        }
        Scope::CacheSalt(cache_salt) => rmp_serde::to_vec_named(&(
            tag,
            hashes,
            parent,
            tokens,
            size,
            lora_id,
            medium,
            SglangNamespace { cache_salt },
        )),
    }
    .unwrap();
    rmp_serde::from_slice(&bytes).unwrap()
}

fn normalized(raw: crate::zmq_wire::RawKvEvent, worker_id: u64) -> RouterEvent {
    crate::zmq_wire::ZmqEventNormalizer::new(VLLM_BLOCK_SIZE)
        .normalize(raw, 1, WorkerWithDpRank::new(worker_id, 0))
        .expect("a text BlockStored event normalizes")
        .into_router_event()
        .expect("a device-tier event targets the primary index")
}

/// The sequence hashes the frontend computes for a request's blocks.
fn frontend_hashes(tokens: &[u32], scope: Scope) -> Vec<u64> {
    let (lora_name, cache_namespace) = match scope {
        Scope::Plain => (None, None),
        Scope::Lora(name) => (Some(name), None),
        Scope::CacheSalt(salt) => (None, Some(salt)),
    };
    let local = compute_block_hash_for_seq(
        tokens,
        VLLM_BLOCK_SIZE,
        BlockHashOptions {
            lora_name,
            cache_namespace,
            ..BlockHashOptions::default()
        },
    );
    compute_seq_hash_for_block(&local)
}

/// The keys a radix-tree delegate reports as gaining a first owner.
async fn engine_keys(events: Vec<RouterEvent>) -> Vec<u64> {
    let recorder = Arc::new(Recorder::default());
    let indexer = KvIndexer::builder(
        CancellationToken::new(),
        VLLM_BLOCK_SIZE,
        Arc::new(KvIndexerMetrics::new_unregistered()),
    )
    .delegate(recorder.clone())
    .build();
    for event in events {
        indexer.apply_event(event).await;
    }
    indexer.flush().await;
    indexer.shutdown();
    let notes = recorder.take();
    assert!(notes.iter().all(|(created, _)| *created), "no removal here");
    notes.into_iter().map(|(_, hash)| hash.0).collect()
}

/// The keys the cuckoo delegate reports as gaining a first owner, as a set.
fn canonical_keys(events: Vec<RouterEvent>) -> std::collections::BTreeSet<u64> {
    let recorder = Arc::new(CanonicalRecorder::default());
    let mut state =
        cuckoo::DcCkfState::new_with_delegate(cuckoo::CkfConfig::new(1024), recorder.clone())
            .unwrap();
    for event in events {
        let outcome = state.apply_event(event);
        assert!(
            outcome.first_error().is_none(),
            "{:?}",
            outcome.first_error()
        );
    }
    let notes = recorder.take();
    let keys: std::collections::BTreeSet<u64> =
        notes.iter().map(|(_, hash)| hash.as_u64()).collect();
    assert_eq!(keys.len(), notes.len(), "one notification per key");
    keys
}

/// The radix-tree delegate keys a block by the engine's hash. Two vLLM workers with one
/// `PYTHONHASHSEED` give one content one key per block; workers with different seeds give it
/// two, so a consumer would see a block's last copy go while another worker still holds it.
#[tokio::test]
async fn engine_keys_agree_across_vllm_workers_only_with_one_seed() {
    let prompt = vllm_prompt();
    let alike = engine_keys(vec![
        normalized(vllm_stored(&VLLM_HASHES_SEED_0, &prompt, Scope::Plain), 1),
        normalized(vllm_stored(&VLLM_HASHES_SEED_0, &prompt, Scope::Plain), 2),
    ])
    .await;
    assert_eq!(alike, VLLM_HASHES_SEED_0);

    let apart = engine_keys(vec![
        normalized(vllm_stored(&VLLM_HASHES_SEED_0, &prompt, Scope::Plain), 1),
        normalized(vllm_stored(&VLLM_HASHES_SEED_1, &prompt, Scope::Plain), 2),
    ])
    .await;
    let mut expected = VLLM_HASHES_SEED_0.to_vec();
    expected.extend(VLLM_HASHES_SEED_1);
    assert_eq!(apart, expected, "one key per block per seed");
}

/// The radix-tree delegate's key (vLLM's hash) never equals the frontend's sequence hash. The
/// cuckoo delegate's canonical key does, for plain, LoRA, and salted blocks, and it depends on
/// the tokens only: workers with different seeds still give one key per block.
#[tokio::test]
async fn the_canonical_key_equals_the_frontend_hash_and_the_engine_key_does_not() {
    let prompt = vllm_prompt();
    for scope in [
        Scope::Plain,
        Scope::Lora("adapter-a"),
        Scope::CacheSalt("tenant-a"),
    ] {
        let frontend = frontend_hashes(&prompt, scope);
        let engine = engine_keys(vec![normalized(
            vllm_stored(&VLLM_HASHES_SEED_0, &prompt, scope),
            1,
        )])
        .await;
        assert!(
            engine.iter().all(|hash| !frontend.contains(hash)),
            "{scope:?}: an engine key matched a frontend hash"
        );
        let canonical = canonical_keys(vec![
            normalized(vllm_stored(&VLLM_HASHES_SEED_0, &prompt, scope), 1),
            normalized(vllm_stored(&VLLM_HASHES_SEED_1, &prompt, scope), 2),
        ]);
        assert_eq!(canonical, frontend.into_iter().collect(), "{scope:?}");
    }
}
