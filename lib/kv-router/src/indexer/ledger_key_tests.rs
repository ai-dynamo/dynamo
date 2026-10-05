// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Which key an ownership delegate reports for a block, when vLLM-format KV events enter
//! Dynamo through the ZMQ event decoder.
//!
//! A history ledger built on the delegate (first copy appears, last copy goes) needs two facts:
//!
//! - One content on two workers is one key. The radix-tree delegate reports the engine's own
//!   block hash, so this holds only when the workers hash alike. vLLM derives the root of its
//!   hash chain from `PYTHONHASHSEED` when it is set; a test in
//!   `components/src/dynamo/vllm/tests/test_vllm_block_hash_identity.py` covers that side. Here:
//!   equal engine hashes from two workers give one key, different ones give two.
//! - The key equals the hash the frontend computes for a request. The radix-tree delegate's key
//!   (vLLM's hash) does not; the cuckoo delegate's canonical key does, for plain and LoRA blocks.

use std::sync::{Arc, Mutex};

use rmp_serde::{from_slice, to_vec};
use tokio_util::sync::CancellationToken;

use super::cuckoo::{CanonicalSequenceBlockHash, CkfConfig, DcCkfState};
use super::*;
use crate::protocols::*;
use crate::zmq_wire::{BlockHashValue, RawKvEvent, ZmqEventNormalizer};

const BLOCK_SIZE: u32 = 4;

/// Records every first-owner (`true`) and last-owner (`false`) notification.
struct Recorder<H>(Mutex<Vec<(bool, H)>>);

impl<H> Default for Recorder<H> {
    fn default() -> Self {
        Self(Mutex::new(Vec::new()))
    }
}

impl<H: Send + 'static> KvIndexerDelegate<H> for Recorder<H> {
    fn on_create(&self, hash: H) {
        self.0.lock().unwrap().push((true, hash));
    }

    fn on_remove(&self, hash: H) {
        self.0.lock().unwrap().push((false, hash));
    }
}

impl<H: Copy> Recorder<H> {
    fn created(&self) -> Vec<H> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(created, _)| *created)
            .map(|(_, hash)| *hash)
            .collect()
    }
}

/// One vLLM `BlockStored` event in its msgpack sequence form, decoded as the ZMQ listener does.
fn vllm_stored(hashes: &[u64], tokens: &[u32], lora_name: Option<&str>) -> RawKvEvent {
    let hashes: Vec<BlockHashValue> = hashes
        .iter()
        .copied()
        .map(BlockHashValue::Unsigned)
        .collect();
    let bytes = to_vec(&(
        "BlockStored",
        hashes,
        Option::<BlockHashValue>::None,
        tokens.to_vec(),
        BLOCK_SIZE as usize,
        Option::<u64>::None,
        Option::<String>::None,
        lora_name.map(str::to_owned),
    ))
    .unwrap();
    from_slice(&bytes).unwrap()
}

fn router_event(raw: RawKvEvent, worker_id: u64) -> RouterEvent {
    ZmqEventNormalizer::new(BLOCK_SIZE)
        .normalize(raw, 1, WorkerWithDpRank::new(worker_id, 0))
        .expect("a text BlockStored event normalizes")
        .into_router_event()
        .expect("a device-tier event targets the primary index")
}

fn tokens() -> Vec<u32> {
    (1..=4 * BLOCK_SIZE).collect()
}

/// The sequence hashes the frontend computes for a request's blocks.
fn frontend_hashes(tokens: &[u32], lora_name: Option<&str>) -> Vec<u64> {
    let local = compute_block_hash_for_seq(
        tokens,
        BLOCK_SIZE,
        BlockHashOptions {
            lora_name,
            ..BlockHashOptions::default()
        },
    );
    compute_seq_hash_for_block(&local)
}

async fn radix_created(events: Vec<RouterEvent>) -> Vec<u64> {
    let recorder = Arc::new(Recorder::<ExternalSequenceBlockHash>::default());
    let indexer = KvIndexer::builder(
        CancellationToken::new(),
        BLOCK_SIZE,
        Arc::new(KvIndexerMetrics::new_unregistered()),
    )
    .delegate(recorder.clone())
    .build();
    for event in events {
        indexer.apply_event(event).await;
    }
    indexer.flush().await;
    indexer.shutdown();
    recorder.created().into_iter().map(|hash| hash.0).collect()
}

fn canonical_created(events: Vec<RouterEvent>) -> Vec<u64> {
    let recorder = Arc::new(Recorder::<CanonicalSequenceBlockHash>::default());
    let mut state = DcCkfState::new_with_delegate(CkfConfig::new(1024), recorder.clone()).unwrap();
    for event in events {
        let outcome = state.apply_event(event);
        assert!(
            outcome.first_error().is_none(),
            "{:?}",
            outcome.first_error()
        );
    }
    recorder
        .created()
        .into_iter()
        .map(CanonicalSequenceBlockHash::as_u64)
        .collect()
}

/// Two workers that hash alike store one content: the delegate reports one key per block, the
/// engine hash. Two workers that hash differently (another `PYTHONHASHSEED`) store the same
/// content under two keys per block: a ledger would see the block go when one copy goes while
/// the other stays.
#[tokio::test]
async fn one_content_on_two_workers_is_one_key_only_when_they_hash_alike() {
    let tokens = tokens();
    let seed_0 = [0xA1, 0xA2, 0xA3, 0xA4];
    let seed_1 = [0xB1, 0xB2, 0xB3, 0xB4];

    let alike = radix_created(vec![
        router_event(vllm_stored(&seed_0, &tokens, None), 1),
        router_event(vllm_stored(&seed_0, &tokens, None), 2),
    ])
    .await;
    assert_eq!(alike, seed_0, "one key per block, the engine hash");

    let apart = radix_created(vec![
        router_event(vllm_stored(&seed_0, &tokens, None), 1),
        router_event(vllm_stored(&seed_1, &tokens, None), 2),
    ])
    .await;
    assert_eq!(
        apart.len(),
        2 * seed_0.len(),
        "two keys per block: {apart:x?}"
    );
}

/// The radix-tree delegate keys a block by vLLM's hash, which is not the frontend's hash. The
/// cuckoo delegate keys it by the canonical hash, which is: the same rolling hash the frontend
/// computes from the request's tokens, with or without a LoRA adapter. It depends on the tokens
/// only, so two workers that hash differently still give one canonical key per block.
#[tokio::test]
async fn the_canonical_key_equals_the_frontend_hash_and_the_engine_key_does_not() {
    let tokens = tokens();
    let seed_0 = [0xA1, 0xA2, 0xA3, 0xA4];
    let seed_1 = [0xB1, 0xB2, 0xB3, 0xB4];
    for lora_name in [None, Some("adapter-a")] {
        let frontend = frontend_hashes(&tokens, lora_name);

        let engine = radix_created(vec![router_event(
            vllm_stored(&seed_0, &tokens, lora_name),
            1,
        )])
        .await;
        assert_eq!(engine, seed_0);
        assert!(
            engine.iter().all(|hash| !frontend.contains(hash)),
            "an engine key matched a frontend hash"
        );

        let canonical = canonical_created(vec![
            router_event(vllm_stored(&seed_0, &tokens, lora_name), 1),
            router_event(vllm_stored(&seed_1, &tokens, lora_name), 2),
        ]);
        // The cuckoo state notifies in map order: compare the keys as sets, and count them.
        assert_eq!(canonical.len(), frontend.len(), "one key per block");
        assert_eq!(
            canonical.iter().collect::<std::collections::BTreeSet<_>>(),
            frontend.iter().collect(),
            "lora {lora_name:?}"
        );
    }
}
