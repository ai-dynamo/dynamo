// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Test that with short timeout and slow input, events are NOT batched
/// Parametrized over different timeout values: 0ms, 0.1ms, 0.2ms
/// All use 2ms delay between events, so each event times out before the next arrives
#[tokio::test]
async fn test_run_event_processor_loop_no_batching_with_slow_input_0ms() {
    test_no_batching_with_slow_input(None).await; // disabled (no timeout)
}

#[tokio::test]
async fn test_run_event_processor_loop_no_batching_with_slow_input_0_1ms() {
    test_no_batching_with_slow_input(Some(1)).await; // 1ms timeout (was 0.1ms in us)
}

#[tokio::test]
async fn test_run_event_processor_loop_no_batching_with_slow_input_0_2ms() {
    test_no_batching_with_slow_input(Some(2)).await; // 2ms timeout (was 0.2ms in us)
}

/// Helper function to test no batching with slow input
async fn test_no_batching_with_slow_input(timeout_ms: Option<u64>) {
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

    // Send 5 removed events with 2ms delay between each
    // Since timeout is <= 0.2ms, each event should timeout and be sent individually
    for i in 0..5 {
        let event = KvCacheEvent {
            event_id: i as u64,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
            }),
            dp_rank: 0,
        };
        tx.send(local_gpu_event(event)).unwrap();
        // Wait 2ms between events (much longer than the timeout)
        // This ensures each event times out before the next one arrives
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
    }

    // Give the processor time to process the last event
    tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

    drop(tx);
    handle.await.unwrap();

    let events = publisher.get_events();

    assert!(!events.is_empty(), "Should have received events");

    // With slow input (2ms delay) and short timeout, most events should be sent individually
    // We expect at least 3 separate events (showing reduced batching)
    assert!(
        events.len() >= 3,
        "With slow input (2ms delay) and timeout={timeout_ms:?}, should have at least 3 separate events (got {})",
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
        total_hashes, 5,
        "All 5 block hashes should be accounted for"
    );
}
