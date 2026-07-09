// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test sequential stored events accumulate with different counts
/// Uses a longer timeout (100ms) to ensure events have time to batch
#[tokio::test]
async fn test_run_event_processor_loop_batches_stored_events_20() {
    test_stored_events_batching(20, Some(100)).await; // 20 events, 100ms timeout
}

#[tokio::test]
async fn test_run_event_processor_loop_batches_stored_events_10() {
    test_stored_events_batching(10, Some(100)).await; // 10 events, 100ms timeout
}

#[tokio::test]
async fn test_run_event_processor_loop_batches_stored_events_5() {
    test_stored_events_batching(5, Some(100)).await; // 5 events, 100ms timeout
}

#[tokio::test]
async fn test_run_event_processor_loop_batches_stored_events_3() {
    test_stored_events_batching(3, Some(100)).await; // 3 events, 100ms timeout
}

/// Helper function to test stored events batching with configurable count and timeout
async fn test_stored_events_batching(event_count: usize, timeout_ms: Option<u64>) {
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

    for i in 0..event_count {
        // For sequential batching, each event's parent_hash should be the previous event's block_hash
        let parent_hash = if i == 0 {
            Some(ExternalSequenceBlockHash(0)) // First event has parent_hash = 0
        } else {
            Some(ExternalSequenceBlockHash((i - 1) as u64)) // Subsequent events reference previous block
        };

        let event = KvCacheEvent {
            event_id: i as u64,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
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
        // Small sleep to allow event processor to batch events
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
    }

    // Give the processor time to process all events before closing the channel
    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert!(
        !events.is_empty(),
        "Should have received at least one event"
    );

    // With a long timeout, events should be batched. Either 1 or can be at most 2, if the first event flushes separately due to initial timestamp.
    assert!(
        events.len() <= 2,
        "With long timeout ({timeout_ms:?}) and sequential parent hashes, all {event_count} events should batch into at most 2 output events (got {})",
        events.len()
    );
    if events.len() == 2 {
        // If we got 2 events, the first one should contain only the first block, and the second should contain the rest
        if let KvCacheEventData::Stored(data) = &events[0].event.data {
            assert_eq!(
                data.blocks.len(),
                1,
                "If 2 events, first event should have 1 block (got {})",
                data.blocks.len()
            );
        } else {
            panic!("Expected Stored event");
        }
    }

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
    assert_eq!(
        total_blocks, event_count,
        "All {} blocks should be accounted for",
        event_count
    );
}
