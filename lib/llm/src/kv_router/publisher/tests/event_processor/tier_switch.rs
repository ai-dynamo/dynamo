// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test that switching between Removed and Stored events causes immediate flush
#[tokio::test]
async fn test_event_type_switching_causes_flush() {
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

    // Send a Removed event
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 0,
        data: KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(0)],
        }),
        dp_rank: 0,
    }))
    .unwrap();

    // Small sleep
    tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;

    // Send a Stored event (should cause flush of the Removed event)
    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 1,
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

    // Give time for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    // Should have 2 events: one Removed, one Stored (not batched together)
    assert_eq!(
        events.len(),
        2,
        "Switching from Removed to Stored should cause immediate flush, resulting in 2 separate events"
    );
}

#[tokio::test]
async fn test_host_tier_events_are_published_and_preserved() {
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
            Some(100),
            DEFAULT_MAX_BATCH_BLOCKS,
        )
        .await
    });

    tx.send(local_host_event(KvCacheEvent {
        event_id: 0,
        data: KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(42)],
        }),
        dp_rank: 0,
    }))
    .unwrap();

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();
    assert_eq!(
        events.len(),
        1,
        "Expected a single published host-tier event"
    );
    assert_eq!(events[0].storage_tier, StorageTier::HostPinned);

    let KvCacheEventData::Removed(data) = &events[0].event.data else {
        panic!("Expected Removed event");
    };
    assert_eq!(data.block_hashes, vec![ExternalSequenceBlockHash(42)]);
}

#[tokio::test]
async fn test_storage_tier_change_causes_flush() {
    let timeout_ms = Some(100);

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

    tx.send(local_host_event(KvCacheEvent {
        event_id: 0,
        data: KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1)],
        }),
        dp_rank: 0,
    }))
    .unwrap();
    tokio::task::yield_now().await;

    tx.send(local_gpu_event(KvCacheEvent {
        event_id: 1,
        data: KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(2)],
        }),
        dp_rank: 0,
    }))
    .unwrap();

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();
    assert_eq!(
        events.len(),
        2,
        "Changing storage tier should flush the current batch"
    );
    assert_eq!(events[0].storage_tier, StorageTier::HostPinned);
    assert_eq!(events[1].storage_tier, StorageTier::Device);
}

/// Test that dp_rank change causes immediate flush
#[tokio::test]
async fn test_dp_rank_change_causes_flush() {
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

    // Send events with dp_rank=0
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

    // Send events with dp_rank=1 (should cause flush of previous batch)
    for i in 3..6 {
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: i as u64,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
            }),
            dp_rank: 1,
        }))
        .unwrap();
        tokio::task::yield_now().await;
    }

    // Give time for processing
    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    // Should have 2 events: one for dp_rank=0 batch, one for dp_rank=1 batch
    assert_eq!(
        events.len(),
        2,
        "dp_rank change should cause immediate flush, resulting in 2 separate events"
    );

    // Verify all 6 block hashes are accounted for
    let total_hashes: usize = events
        .iter()
        .map(|e| {
            if let KvCacheEventData::Removed(data) = &e.event.data {
                data.block_hashes.len()
            } else {
                0
            }
        })
        .sum();
    assert_eq!(
        total_hashes, 6,
        "All 6 block hashes should be accounted for"
    );

    // Verify dp_rank is correct for each batch
    assert_eq!(
        events[0].event.dp_rank, 0,
        "First batch should have dp_rank=0"
    );
    assert_eq!(
        events[1].event.dp_rank, 1,
        "Second batch should have dp_rank=1"
    );
}
