// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test non-sequential stored events trigger flush
#[tokio::test]
async fn test_run_event_processor_loop_non_sequential_flush() {
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
        // SLEEP HERE?! so that events are not batched!
    });

    for i in 0..3 {
        let event = KvCacheEvent {
            event_id: i as u64,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash((i + 1) as u64 * 100)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(i as u64),
                    tokens_hash: LocalBlockHash(i as u64 * 100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        };
        tx.send(local_gpu_event(event)).unwrap();
    }

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert!(!events.is_empty(), "Should have received events");

    // With non-sequential parent hashes, each event should trigger a flush
    // So we expect 3 separate events
    assert_eq!(
        events.len(),
        3,
        "Non-sequential events should trigger flush, resulting in 3 separate events"
    );

    let total_blocks: usize = events
        .iter()
        .map(|e| {
            if let KvCacheEventData::Stored(data) = &e.event.data {
                data.blocks.len()
            } else {
                0
            }
        })
        .sum();
    assert_eq!(total_blocks, 3, "All 3 blocks should be accounted for");
}

/// Test that reusing an older parent hash breaks the current sequential batch.
#[tokio::test]
async fn test_run_event_processor_loop_reused_parent_hash_breaks_chain() {
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

    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 0,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
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

    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 1,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(1)),
            start_position: Some(11_111),
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

    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 2,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(1)),
            start_position: Some(20),
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
        2,
        "Reused parent hash should flush the current batch before starting a new one"
    );

    if let KvCacheEventData::Stored(data) = &events[0].event.data {
        assert_eq!(
            data.blocks.len(),
            2,
            "First batch should keep the valid chain"
        );
        assert_eq!(
            data.parent_hash, None,
            "First batch should preserve the original root parent"
        );
        assert_eq!(
            data.start_position,
            Some(10),
            "First batch should preserve the original start position"
        );
    } else {
        panic!("Expected first event to be Stored");
    }

    if let KvCacheEventData::Stored(data) = &events[1].event.data {
        assert_eq!(
            data.blocks.len(),
            1,
            "Second batch should contain only the inconsistent event"
        );
        assert_eq!(
            data.parent_hash,
            Some(ExternalSequenceBlockHash(1)),
            "Second batch should preserve the reused parent hash"
        );
        assert_eq!(
            data.start_position,
            Some(20),
            "Second batch should keep the new root's start position"
        );
    } else {
        panic!("Expected second event to be Stored");
    }
}
