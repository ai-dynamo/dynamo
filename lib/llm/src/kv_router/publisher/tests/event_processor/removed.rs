// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test that pushing N removed events results in batched output
/// Uses a 10ms timeout to ensure events are batched (events sent rapidly)
#[tokio::test]
async fn test_run_event_processor_loop_batches_removed_events_20() {
    test_removed_events_batching(20, Some(10)).await; // 20 events, 10ms timeout
}

#[tokio::test]
async fn test_run_event_processor_loop_batches_removed_events_10() {
    test_removed_events_batching(10, Some(10)).await; // 10 events, 10ms timeout
}

#[tokio::test]
async fn test_run_event_processor_loop_batches_removed_events_5() {
    test_removed_events_batching(5, Some(10)).await; // 5 events, 10ms timeout
}

#[tokio::test]
async fn test_run_event_processor_loop_batches_removed_events_3() {
    test_removed_events_batching(3, Some(10)).await; // 3 events, 10ms timeout
}

/// Helper function to test removed events batching with configurable count and timeout
async fn test_removed_events_batching(event_count: usize, timeout_ms: Option<u64>) {
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
        let event = KvCacheEvent {
            event_id: i as u64,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
            }),
            dp_rank: 0,
        };
        tx.send(local_gpu_event(event)).unwrap();
        // Yield to allow event processor to process the event
        tokio::task::yield_now().await;
    }

    // Wait for timeout to elapse so all events flush together as one batch
    // Add small buffer to ensure flush happens before channel close
    tokio::time::sleep(tokio::time::Duration::from_millis(
        timeout_ms.unwrap_or(0) + 1,
    ))
    .await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert!(
        !events.is_empty(),
        "Should have received at least one event"
    );

    // With a long timeout (100ms) and rapid event sending, all events should batch into few output events
    // (first event may flush separately, rest should batch together)
    assert!(
        events.len() <= 2,
        "With long timeout ({timeout_ms:?}), all {event_count} events should batch into at most 2 output events (got {})",
        events.len()
    );

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
        total_hashes, event_count,
        "All {} block hashes should be accounted for",
        event_count
    );
}
