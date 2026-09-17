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
        "concurrent" => Box::new(ThreadPoolIndexer::new(
            concurrent_radix_tree::ConcurrentRadixTree::new_with_delegate(delegate),
            4,
            32,
        )),
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
    #[values(
        "single",
        "concurrent",
        "compressed",
        "positional",
        "lower",
        "local",
        "sharded"
    )]
    variant: &str,
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
    #[values(
        "single",
        "concurrent",
        "compressed",
        "positional",
        "lower",
        "local",
        "sharded"
    )]
    variant: &str,
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
    #[values("single", "concurrent", "compressed", "positional")] variant: &str,
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
    #[values("single", "concurrent", "compressed", "positional")] variant: &str,
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
    #[values(
        "single",
        "concurrent",
        "compressed",
        "positional",
        "lower",
        "local",
        "sharded"
    )]
    variant: &str,
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
