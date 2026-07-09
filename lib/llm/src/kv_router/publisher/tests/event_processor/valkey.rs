// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test that extending a batch does NOT change parent_hash
/// First event with parent_hash=None should keep it None even if subsequent events have Some(X)
#[tokio::test]
async fn test_batch_parent_hash_preserved_when_extending() {
    let timeout_ms = Some(100); // 100ms timeout

    let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
    let publisher = MockPublisher::new();
    let publisher_clone = publisher.clone();
    let cancellation_token = CancellationToken::new();

    let handle = tokio::spawn(async move {
        run_event_processor_loop(
            publisher_clone,
            1,
            cancellation_token,
            rx,
            None,
            timeout_ms,
            DEFAULT_MAX_BATCH_BLOCKS,
        )
        .await
    });

    // First event: parent_hash=None, block_hash=1
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 0,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None, // Root block with no parent
            start_position: Some(10),
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(1),
                tokens_hash: LocalBlockHash(100),
                mm_extra_info: None,
            }],
        }),
        dp_rank: 0,
    }))
    .unwrap();
    tokio::task::yield_now().await;

    // Second event: parent_hash=Some(1), block_hash=2 (sequential)
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 1,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(1)), // Points to previous block
            start_position: Some(999),
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(2),
                tokens_hash: LocalBlockHash(200),
                mm_extra_info: None,
            }],
        }),
        dp_rank: 0,
    }))
    .unwrap();
    tokio::task::yield_now().await;

    // Third event: parent_hash=Some(2), block_hash=3 (sequential)
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 2,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(2)),
            start_position: Some(1_234),
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(3),
                tokens_hash: LocalBlockHash(300),
                mm_extra_info: None,
            }],
        }),
        dp_rank: 0,
    }))
    .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert_eq!(
        events.len(),
        1,
        "All 3 sequential events should batch into 1"
    );

    // The batch should have parent_hash=None (preserved from first event)
    if let KvCacheEventData::Stored(data) = &events[0].event.data {
        assert_eq!(data.blocks.len(), 3, "Batch should have 3 blocks");
        assert_eq!(
            data.parent_hash, None,
            "Batch parent_hash should remain None (from first event), NOT overwritten by subsequent events"
        );
        assert_eq!(
            data.start_position,
            Some(10),
            "Batch start_position should remain anchored to the first event"
        );
    } else {
        panic!("Expected Stored event");
    }
}

struct GapRecoveryOps {
    operations: Arc<Mutex<Vec<&'static str>>>,
}

impl sinks::ValkeyWorkerEventOps for GapRecoveryOps {
    fn apply_owned_batch<'a>(&'a self, _events: &'a [RouterEvent]) -> sinks::ValkeyOperation<'a> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("apply");
            Ok(())
        })
    }

    fn unregister(&self) -> sinks::ValkeyOperation<'_> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("unregister");
            Ok(())
        })
    }

    fn renew(&self) -> sinks::ValkeyOperation<'_> {
        Box::pin(async { Ok(()) })
    }

    fn gc_step(&self, inspection_budget: u32) -> sinks::ValkeyGcOperation<'_> {
        Box::pin(async move {
            self.operations.lock().unwrap().push("gc");
            Ok([u64::from(inspection_budget), 0, 0, 0, 0, 0, 0, 0])
        })
    }
}

#[tokio::test]
async fn direct_valkey_raw_gap_permanently_fences_before_exit() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let integrity = ValkeyPublisherIntegrity::with_backend(
        1,
        9,
        Arc::new(GapRecoveryOps {
            operations: operations.clone(),
        }),
    );
    let publisher = ValkeyEventPublisher::new(integrity.clone());
    let (tx, rx) = mpsc::channel::<PublisherInput>(4);
    let ingress = PlacementEventSender::direct_valkey(tx, integrity.clone());

    for event_id in [0, 2] {
        ingress
            .send(local_gpu_event(KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(event_id)],
                }),
                dp_rank: 0,
            }))
            .unwrap();
    }
    drop(ingress);

    tokio::time::timeout(
        Duration::from_secs(1),
        crate::kv_router::publisher::event_processor::start_direct_valkey_event_processor(
            publisher.clone(),
            publisher,
            1,
            CancellationToken::new(),
            PlacementEventReceiver::DirectValkey(rx),
            None,
            None,
        ),
    )
    .await
    .expect("gap recovery processor should finish");

    assert_eq!(
        operations.lock().unwrap().as_slice(),
        &["apply", "unregister"]
    );
    assert_eq!(integrity.state(), sinks::IntegrityState::Fenced);
}

#[tokio::test(start_paused = true)]
async fn worker_lifecycle_gc_respects_schedule_and_cancels_cleanly() {
    let operations = Arc::new(Mutex::new(Vec::new()));
    let integrity = ValkeyPublisherIntegrity::with_backend(
        1,
        9,
        Arc::new(GapRecoveryOps {
            operations: operations.clone(),
        }),
    );
    let cancellation = CancellationToken::new();
    let interval_ms = 1_000;
    let initial_delay_ms = valkey_gc_initial_delay_ms(interval_ms, 9);
    let task = spawn_worker_lifecycle_gc(integrity, 1, 9, interval_ms, 256, cancellation.clone());
    tokio::task::yield_now().await;

    tokio::time::advance(Duration::from_millis(initial_delay_ms - 1)).await;
    tokio::task::yield_now().await;
    assert!(operations.lock().unwrap().is_empty());

    tokio::time::advance(Duration::from_millis(1)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(operations.lock().unwrap().as_slice(), &["gc"]);

    tokio::time::advance(Duration::from_millis(interval_ms)).await;
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert_eq!(operations.lock().unwrap().as_slice(), &["gc", "gc"]);

    cancellation.cancel();
    task.await
        .expect("GC task should stop cleanly on cancellation");
}
