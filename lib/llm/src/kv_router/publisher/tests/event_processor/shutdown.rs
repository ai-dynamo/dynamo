// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn publisher_shutdown_waits_for_tracked_writes() {
    let cancellation_token = CancellationToken::new();
    let task_tracker = TaskTracker::new();
    let write_started = Arc::new(Notify::new());
    let release_write = Arc::new(Notify::new());
    task_tracker.spawn({
        let cancellation_token = cancellation_token.clone();
        let write_started = write_started.clone();
        let release_write = release_write.clone();
        async move {
            cancellation_token.cancelled().await;
            write_started.notify_one();
            release_write.notified().await;
        }
    });

    let shutdown_handle = KvEventPublisherShutdown {
        cancellation_token,
        task_tracker,
        operation_cancel: None,
    };
    let shutdown = tokio::spawn(async move { shutdown_handle.shutdown_and_wait().await });
    write_started.notified().await;
    assert!(!shutdown.is_finished());
    release_write.notify_one();
    assert_eq!(
        shutdown.await.unwrap(),
        KvEventPublisherShutdownOutcome::Drained
    );
}

#[tokio::test]
async fn publisher_shutdown_cancels_stuck_valkey_operation_after_drain_timeout() {
    let cancellation_token = CancellationToken::new();
    let operation_cancel = CancellationToken::new();
    let task_tracker = TaskTracker::new();
    task_tracker.spawn({
        let cancellation_token = cancellation_token.clone();
        let operation_cancel = operation_cancel.clone();
        async move {
            cancellation_token.cancelled().await;
            operation_cancel.cancelled().await;
        }
    });
    let shutdown_handle = KvEventPublisherShutdown {
        cancellation_token,
        task_tracker,
        operation_cancel: Some(operation_cancel.clone()),
    };

    let outcome = shutdown_publishers_and_wait_with_timeouts(
        &[shutdown_handle],
        Duration::from_millis(10),
        Duration::from_millis(100),
    )
    .await;

    assert_eq!(outcome, KvEventPublisherShutdownOutcome::Forced);
    assert!(operation_cancel.is_cancelled());
}

#[tokio::test]
async fn publisher_shutdown_is_bounded_when_tracked_task_ignores_cancellation() {
    let cancellation_token = CancellationToken::new();
    let operation_cancel = CancellationToken::new();
    let task_tracker = TaskTracker::new();
    let task = task_tracker.spawn(std::future::pending::<()>());
    let shutdown_handle = KvEventPublisherShutdown {
        cancellation_token,
        task_tracker,
        operation_cancel: Some(operation_cancel.clone()),
    };

    let outcome = shutdown_publishers_and_wait_with_timeouts(
        &[shutdown_handle],
        Duration::from_millis(10),
        Duration::from_millis(10),
    )
    .await;

    assert_eq!(outcome, KvEventPublisherShutdownOutcome::TimedOut);
    assert!(operation_cancel.is_cancelled());
    task.abort();
    let _ = task.await;
}

#[tokio::test]
async fn cancellation_drains_events_accepted_before_shutdown() {
    let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
    for event_id in 0..2 {
        tx.send(local_gpu_event(KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(event_id)],
            }),
            dp_rank: 0,
        }))
        .unwrap();
    }

    let publisher = MockPublisher::new();
    let cancellation_token = CancellationToken::new();
    cancellation_token.cancel();
    run_event_processor_loop(
        publisher.clone(),
        1,
        cancellation_token,
        rx,
        None,
        Some(100),
        DEFAULT_MAX_BATCH_BLOCKS,
    )
    .await;

    assert!(
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(2)],
            }),
            dp_rank: 0,
        }))
        .is_err()
    );
    let events = publisher.get_events();
    assert_eq!(events.len(), 1);
    let KvCacheEventData::Removed(removed) = &events[0].event.data else {
        panic!("expected one drained remove batch");
    };
    assert_eq!(
        removed.block_hashes,
        vec![ExternalSequenceBlockHash(0), ExternalSequenceBlockHash(1)]
    );
}
