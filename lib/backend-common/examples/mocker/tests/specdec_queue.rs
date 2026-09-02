// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_mocker_backend::specdec::queue::{
    CancellationPhase, FakeScheduler, JobSpec, QueueError, QueueEvent, SchedulerConfig, TokenMode,
};
use uuid::Uuid;

fn job(id: u128, prefill_ms: u64, tokens: u32, mode: TokenMode) -> JobSpec {
    JobSpec {
        request_id: Uuid::from_u128(id),
        prompt_token_ids: vec![7, 8],
        max_output_tokens: tokens,
        prefill_duration: Duration::from_millis(prefill_ms),
        token_interval: Duration::from_millis(10),
        token_mode: mode,
    }
}

async fn next(handle: &mut dynamo_mocker_backend::specdec::queue::JobHandle) -> QueueEvent {
    handle.recv().await.expect("scheduler closed unexpectedly")
}

#[tokio::test(start_paused = true)]
async fn queue_is_fifo_bounded_and_honors_configured_concurrency() {
    let scheduler = FakeScheduler::start(SchedulerConfig {
        queue_capacity: 1,
        concurrency: 1,
        output_capacity: 8,
    })
    .unwrap();

    let mut first = scheduler
        .submit(job(1, 100, 1, TokenMode::Echo))
        .await
        .unwrap();
    assert_eq!(next(&mut first).await, QueueEvent::Queued);
    assert_eq!(next(&mut first).await, QueueEvent::PrefillStarted);

    let mut second = scheduler
        .submit(job(2, 0, 1, TokenMode::Echo))
        .await
        .unwrap();
    assert_eq!(next(&mut second).await, QueueEvent::Queued);
    assert!(matches!(
        scheduler.submit(job(3, 0, 1, TokenMode::Echo)).await,
        Err(QueueError::Full)
    ));

    tokio::time::advance(Duration::from_millis(120)).await;
    while next(&mut first).await != (QueueEvent::Complete { emitted_tokens: 1 }) {}
    assert_eq!(next(&mut second).await, QueueEvent::PrefillStarted);
    scheduler.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn token_modes_are_deterministic_under_paused_time() {
    let scheduler = FakeScheduler::start(SchedulerConfig::default()).unwrap();
    let mut echo = scheduler
        .submit(job(11, 0, 3, TokenMode::Echo))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_secs(1)).await;
    let mut echo_tokens = Vec::new();
    while let Some(event) = echo.recv().await {
        match event {
            QueueEvent::Token { token_id, .. } => echo_tokens.push(token_id),
            QueueEvent::Complete { .. } => break,
            _ => {}
        }
    }
    assert_eq!(echo_tokens, vec![7, 8, 7]);

    let mut first = scheduler
        .submit(job(12, 0, 3, TokenMode::Counter))
        .await
        .unwrap();
    let mut second = scheduler
        .submit(job(12, 0, 3, TokenMode::Counter))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_secs(1)).await;
    let mut first_tokens = Vec::new();
    let mut second_tokens = Vec::new();
    while let Some(event) = first.recv().await {
        match event {
            QueueEvent::Token { token_id, .. } => first_tokens.push(token_id),
            QueueEvent::Complete { .. } => break,
            _ => {}
        }
    }
    while let Some(event) = second.recv().await {
        match event {
            QueueEvent::Token { token_id, .. } => second_tokens.push(token_id),
            QueueEvent::Complete { .. } => break,
            _ => {}
        }
    }
    assert_eq!(first_tokens, second_tokens);
    scheduler.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn cancellation_and_shutdown_reap_queued_and_running_jobs() {
    let scheduler = FakeScheduler::start(SchedulerConfig {
        queue_capacity: 2,
        concurrency: 1,
        output_capacity: 8,
    })
    .unwrap();
    let mut prefilling = scheduler
        .submit(job(21, 10_000, 1, TokenMode::Echo))
        .await
        .unwrap();
    assert_eq!(next(&mut prefilling).await, QueueEvent::Queued);
    assert_eq!(next(&mut prefilling).await, QueueEvent::PrefillStarted);
    prefilling.cancel();
    assert_eq!(
        next(&mut prefilling).await,
        QueueEvent::Cancelled {
            phase: CancellationPhase::Prefilling
        }
    );

    let mut emitting = scheduler
        .submit(job(22, 0, 10, TokenMode::Echo))
        .await
        .unwrap();
    assert_eq!(next(&mut emitting).await, QueueEvent::Queued);
    assert_eq!(next(&mut emitting).await, QueueEvent::PrefillStarted);
    assert_eq!(next(&mut emitting).await, QueueEvent::PrefillComplete);
    emitting.cancel();
    assert_eq!(
        next(&mut emitting).await,
        QueueEvent::Cancelled {
            phase: CancellationPhase::Emitting
        }
    );

    let mut running = scheduler
        .submit(job(23, 10_000, 1, TokenMode::Echo))
        .await
        .unwrap();
    assert_eq!(next(&mut running).await, QueueEvent::Queued);
    assert_eq!(next(&mut running).await, QueueEvent::PrefillStarted);
    scheduler.shutdown().await.unwrap();
    assert_eq!(
        next(&mut running).await,
        QueueEvent::Cancelled {
            phase: CancellationPhase::Prefilling
        }
    );
    assert!(running.recv().await.is_none());
}
