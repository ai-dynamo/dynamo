// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test that flushed events have correct metadata (event_id, dp_rank)
/// This verifies that metadata is NOT overwritten before flush
#[tokio::test]
async fn test_flushed_events_have_correct_metadata() {
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

    // Send first batch: 3 events with dp_rank=0, event_ids 10-12
    for i in 0..3 {
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 10 + i as u64,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;
    }

    // Send second batch: 2 events with dp_rank=1, event_ids 20-21
    // This should flush the first batch with dp_rank=0
    for i in 0..2 {
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 20 + i as u64,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash((i + 3) as u64)],
            }),
            dp_rank: 1,
        }))
        .unwrap();
        tokio::task::yield_now().await;
    }

    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert_eq!(
        events.len(),
        2,
        "Should have 2 events (one per dp_rank batch)"
    );

    // First event should have dp_rank=0 and monotonic batch event_id=1
    assert_eq!(
        events[0].event.dp_rank, 0,
        "First batch should have dp_rank=0"
    );
    assert_eq!(
        events[0].event.event_id, 1,
        "First batch should have monotonic event_id=1"
    );

    // Second event should have dp_rank=1 and monotonic batch event_id=2
    assert_eq!(
        events[1].event.dp_rank, 1,
        "Second batch should have dp_rank=1"
    );
    assert_eq!(
        events[1].event.event_id, 2,
        "Second batch should have monotonic event_id=2"
    );
}

/// Test that events after a long idle period flush immediately (stale timer).
/// This gives low latency for sparse important events after idle periods.
/// After the initial stale flush, subsequent rapid events batch normally.
#[tokio::test]
async fn test_first_event_after_idle_flushes_immediately_then_batches() {
    let timeout_ms = Some(50); // 50ms timeout

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

    // Wait longer than timeout to simulate idle period (timer becomes stale)
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Send 3 events rapidly - first should flush immediately (stale timer),
    // remaining 2 should batch together
    for i in 0..3 {
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: i as u64,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;
    }

    // Wait for timeout to elapse so remaining batch flushes
    tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    // First event flushes immediately (stale timer), remaining 2 batch together
    assert_eq!(
        events.len(),
        2,
        "First event should flush immediately (stale), remaining 2 should batch"
    );

    // First event has 1 hash, second event (batch) has 2 hashes
    let first_len = if let KvCacheEventData::Removed(data) = &events[0].event.data {
        data.block_hashes.len()
    } else {
        0
    };
    let second_len = if let KvCacheEventData::Removed(data) = &events[1].event.data {
        data.block_hashes.len()
    } else {
        0
    };
    assert_eq!(first_len, 1, "First event should have 1 hash");
    assert_eq!(second_len, 2, "Second event (batched) should have 2 hashes");
}

/// Test that stored events with dp_rank change have correct metadata
#[tokio::test]
async fn test_stored_events_with_dp_rank_change_correct_metadata() {
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

    // Send first batch: 2 sequential stored events with dp_rank=0, event_ids 100-101
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 100,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(0)),
            start_position: None,
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
        event_id: 101,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(1)),
            start_position: None,
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

    // Send second batch: 1 event with dp_rank=1, event_id=200
    // This should flush the first batch with dp_rank=0, event_id=101
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 200,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(0)),
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(100),
                tokens_hash: LocalBlockHash(1000),
                mm_extra_info: None,
            }],
        }),
        dp_rank: 1,
    }))
    .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert_eq!(
        events.len(),
        2,
        "Should have 2 events (one per dp_rank batch)"
    );

    // First batch: dp_rank=0, monotonic event_id=1
    assert_eq!(
        events[0].event.dp_rank, 0,
        "First batch should have dp_rank=0"
    );
    assert_eq!(
        events[0].event.event_id, 1,
        "First batch should have monotonic event_id=1"
    );

    // Second batch: dp_rank=1, monotonic event_id=2
    assert_eq!(
        events[1].event.dp_rank, 1,
        "Second batch should have dp_rank=1"
    );
    assert_eq!(
        events[1].event.event_id, 2,
        "Second batch should have monotonic event_id=2"
    );

    // Verify block counts
    if let KvCacheEventData::Stored(data) = &events[0].event.data {
        assert_eq!(data.blocks.len(), 2, "First batch should have 2 blocks");
    } else {
        panic!("Expected Stored event");
    }
    if let KvCacheEventData::Stored(data) = &events[1].event.data {
        assert_eq!(data.blocks.len(), 1, "Second batch should have 1 block");
    } else {
        panic!("Expected Stored event");
    }
}
