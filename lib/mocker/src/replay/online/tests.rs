// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use dashmap::DashMap;
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::Instant;
use uuid::Uuid;

use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs, SglangArgs};
use crate::loadgen::{SessionTrace, Trace, TurnTrace};
use crate::replay::ReplayRouterMode;
use crate::replay::router::ReplayRouter;

use super::entrypoints::{
    simulate_concurrency_requests_with_stats, simulate_concurrency_workload_with_stats,
    simulate_trace_requests, simulate_trace_requests_with_stats,
    simulate_trace_workload_with_stats,
};
use super::state::{LiveReplayMode, SharedLiveRuntimeStats, WorkloadDispatchState, record_arrival};
use super::task::{RequestTaskContext, run_request_task, wait_for_workload_progress};

fn replay_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .speedup_ratio(1000.0)
        .block_size(64)
        .build()
        .unwrap()
}

fn sglang_replay_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(512)
        .speedup_ratio(1000.0)
        .sglang(Some(SglangArgs {
            page_size: Some(2),
            ..Default::default()
        }))
        .build()
        .unwrap()
}

fn request(uuid: u128, token: u32, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
    DirectRequest {
        tokens: vec![token; 64],
        max_output_tokens: 2,
        uuid: Some(Uuid::from_u128(uuid)),
        dp_rank: 0,
        arrival_timestamp_ms,
    }
}

fn multiturn_trace() -> Trace {
    Trace {
        block_size: 1,
        sessions: vec![
            SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![11, 12, 13, 14],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 6,
                        max_output_tokens: 2,
                        hash_ids: vec![21, 22, 23, 24, 25, 26],
                        delay_after_previous_ms: 5.0,
                    },
                ],
            },
            SessionTrace {
                session_id: "session-b".to_string(),
                first_arrival_timestamp_ms: Some(1.0),
                turns: vec![TurnTrace {
                    input_length: 5,
                    max_output_tokens: 2,
                    hash_ids: vec![31, 32, 33, 34, 35],
                    delay_after_previous_ms: 0.0,
                }],
            },
        ],
    }
}

#[test]
fn test_online_trace_replay_single_worker_completes() {
    let args = replay_args();
    let requests = vec![request(1, 11, Some(0.0)), request(2, 22, Some(1.0))];

    let report =
        simulate_trace_requests(args, None, requests, 1, 1.0, ReplayRouterMode::RoundRobin)
            .unwrap();

    assert_eq!(report.request_counts.num_requests, 2);
    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.request_counts.total_output_tokens, 4);
    assert!(report.throughput.wall_time_ms >= 0.0);
}

#[test]
fn test_online_trace_workload_completes_multiturn_sessions() {
    let args = replay_args();
    let (report, _) =
        simulate_trace_workload_with_stats(args, multiturn_trace(), 2, ReplayRouterMode::KvRouter)
            .unwrap();

    assert_eq!(report.request_counts.num_requests, 3);
    assert_eq!(report.request_counts.completed_requests, 3);
    assert_eq!(report.request_counts.total_input_tokens, 15);
    assert_eq!(report.request_counts.total_output_tokens, 6);
}

#[test]
fn test_online_concurrency_workload_respects_global_cap() {
    let args = replay_args();
    let (report, stats) = simulate_concurrency_workload_with_stats(
        args,
        multiturn_trace(),
        1,
        2,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();

    assert_eq!(report.request_counts.num_requests, 3);
    assert_eq!(report.request_counts.completed_requests, 3);
    assert_eq!(stats.max_in_flight_seen, 1);
}

#[tokio::test]
async fn test_record_arrival_uses_caller_arrival_timestamp() {
    let (arrival_tx, mut arrival_rx) = mpsc::unbounded_channel();
    let uuid = Uuid::from_u128(999);
    let arrival_at_ms = 123.0;
    let request = request(999, 42, Some(arrival_at_ms));

    let recorded_uuid = record_arrival(&arrival_tx, &request, arrival_at_ms).unwrap();

    let arrival = arrival_rx.recv().await.unwrap();
    assert_eq!(recorded_uuid, uuid);
    assert_eq!(arrival.uuid, uuid);
    assert_eq!(arrival.at_ms, arrival_at_ms);
}

#[tokio::test]
async fn test_trace_arrivals_are_not_blocked_by_queued_router_selection() {
    let args = MockEngineArgs::builder()
        .speedup_ratio(1000.0)
        .block_size(64)
        .max_num_seqs(Some(1))
        .max_num_batched_tokens(Some(8))
        .build()
        .unwrap();
    let start = Instant::now();
    let router = Arc::new(ReplayRouter::new(
        ReplayRouterMode::KvRouter,
        &args,
        None,
        1,
    ));
    let senders: Arc<[mpsc::UnboundedSender<DirectRequest>]> =
        Arc::from(vec![mpsc::unbounded_channel::<DirectRequest>().0]);
    let requests = Arc::new(DashMap::new());
    let stats = Arc::new(SharedLiveRuntimeStats::default());
    let (arrival_tx, mut arrival_rx) = mpsc::unbounded_channel();
    let task_ctx = RequestTaskContext {
        senders,
        router: Arc::clone(&router),
        requests,
        stats,
        workload: None,
    };
    let mut tasks = JoinSet::new();
    let mut pending = VecDeque::from(vec![
        request(1, 11, Some(0.0)),
        request(2, 22, Some(1.0)),
        request(3, 33, Some(2.0)),
    ]);

    while let Some(request) = pending.pop_front() {
        let arrival_ms = request.arrival_timestamp_ms.unwrap_or(0.0);
        let deadline = start + tokio::time::Duration::from_secs_f64(arrival_ms / 1000.0);
        tokio::time::sleep_until(deadline).await;
        record_arrival(&arrival_tx, &request, arrival_ms).unwrap();
        tasks.spawn(run_request_task(task_ctx.clone(), request, None));
    }

    let first = tokio::time::timeout(tokio::time::Duration::from_millis(50), arrival_rx.recv())
        .await
        .unwrap()
        .unwrap();
    let second = tokio::time::timeout(tokio::time::Duration::from_millis(50), arrival_rx.recv())
        .await
        .unwrap()
        .unwrap();
    let third = tokio::time::timeout(tokio::time::Duration::from_millis(50), arrival_rx.recv())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(first.uuid, Uuid::from_u128(1));
    assert_eq!(second.uuid, Uuid::from_u128(2));
    assert_eq!(third.uuid, Uuid::from_u128(3));
    assert_eq!(first.at_ms, 0.0);
    assert_eq!(second.at_ms, 1.0);
    assert_eq!(third.at_ms, 2.0);

    tasks.abort_all();
    router.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_workload_wakeup_is_not_lost_when_completion_happens_before_await() {
    let mut driver = Trace {
        block_size: 1,
        sessions: vec![SessionTrace {
            session_id: "session-a".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![1, 2, 3, 4],
                    delay_after_previous_ms: 0.0,
                },
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![5, 6, 7, 8],
                    delay_after_previous_ms: 5.0,
                },
            ],
        }],
    }
    .into_trace_driver()
    .unwrap();
    let first = driver.pop_ready(0.0, 1);
    assert_eq!(first.len(), 1);

    let workload = WorkloadDispatchState {
        driver: Mutex::new(driver),
        wakeup: Notify::new(),
        start: Instant::now(),
    };

    let wake = workload.wakeup.notified();
    tokio::pin!(wake);

    let (is_drained, next_ready_ms) = {
        let mut driver = workload.driver.lock().unwrap();
        (driver.is_drained(), driver.next_ready_time_ms())
    };
    assert!(!is_drained);
    assert_eq!(next_ready_ms, None);

    {
        let mut driver = workload.driver.lock().unwrap();
        driver.on_complete(first[0].request_uuid, 5.0).unwrap();
    }
    workload.wakeup.notify_waiters();

    tokio::time::timeout(tokio::time::Duration::from_millis(50), &mut wake)
        .await
        .unwrap();
    assert_eq!(
        workload.driver.lock().unwrap().next_ready_time_ms(),
        Some(10.0)
    );
}

#[tokio::test]
async fn test_concurrency_workload_waits_for_wakeup_when_next_turn_is_completion_gated() {
    let semaphore = Arc::new(Semaphore::new(1));
    let notify = Arc::new(Notify::new());
    let wake = notify.notified();
    tokio::pin!(wake);

    assert!(
        tokio::time::timeout(
            tokio::time::Duration::from_millis(20),
            wait_for_workload_progress(
                LiveReplayMode::Concurrency { max_in_flight: 1 },
                Some(semaphore.as_ref()),
                None,
                Instant::now(),
                wake.as_mut(),
            ),
        )
        .await
        .is_err(),
        "concurrency workload should wait for wakeup when no turn is time-ready"
    );

    let wake = notify.notified();
    tokio::pin!(wake);
    let wait = wait_for_workload_progress(
        LiveReplayMode::Concurrency { max_in_flight: 1 },
        Some(semaphore.as_ref()),
        None,
        Instant::now(),
        wake.as_mut(),
    );
    let notify_task = {
        let notify = Arc::clone(&notify);
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
            notify.notify_waiters();
        })
    };

    tokio::time::timeout(tokio::time::Duration::from_millis(50), wait)
        .await
        .unwrap();
    notify_task.await.unwrap();
}

#[test]
fn test_online_trace_replay_uses_round_robin_dispatch() {
    let args = replay_args();
    let requests = vec![
        request(1, 1, Some(0.0)),
        request(2, 2, Some(100.0)),
        request(3, 3, Some(200.0)),
        request(4, 4, Some(300.0)),
        request(5, 5, Some(400.0)),
    ];

    let (_, stats) =
        simulate_trace_requests_with_stats(args, requests, 3, 1.0, ReplayRouterMode::RoundRobin)
            .unwrap();

    assert_eq!(stats.dispatch_history, vec![0, 1, 2, 0, 1]);
}

#[test]
fn test_online_concurrency_replay_respects_max_in_flight() {
    let args = replay_args();
    let requests = vec![
        request(1, 10, None),
        request(2, 20, None),
        request(3, 30, None),
        request(4, 40, None),
    ];

    let (report, stats) = simulate_concurrency_requests_with_stats(
        args,
        requests,
        2,
        2,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap();

    assert_eq!(report.request_counts.completed_requests, 4);
    assert!(stats.max_in_flight_seen <= 2);
}

#[test]
fn test_online_trace_replay_populates_admit_reuse_stats() {
    let args = replay_args();
    let requests = vec![request(1, 77, Some(0.0)), request(2, 77, Some(5.0))];

    let report =
        simulate_trace_requests(args, None, requests, 1, 1.0, ReplayRouterMode::RoundRobin)
            .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    assert!(report.prefix_cache_reused_ratio > 0.0);
}

#[test]
fn test_online_trace_replay_kv_router_prefers_cached_worker() {
    let args = replay_args();
    let requests = vec![request(1, 88, Some(0.0)), request(2, 88, Some(500.0))];

    let (_, stats) =
        simulate_trace_requests_with_stats(args, requests, 2, 1.0, ReplayRouterMode::KvRouter)
            .unwrap();

    assert_eq!(stats.dispatch_history.len(), 2);
    assert_eq!(stats.dispatch_history[0], stats.dispatch_history[1]);
}

#[test]
fn test_online_trace_replay_sglang_single_worker_completes() {
    let args = sglang_replay_args();
    let requests = vec![request(101, 7, Some(0.0)), request(102, 8, Some(1.0))];

    let report =
        simulate_trace_requests(args, None, requests, 1, 1.0, ReplayRouterMode::RoundRobin)
            .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.request_counts.total_output_tokens, 4);
}

#[test]
fn test_online_trace_replay_sglang_kv_router_smoke() {
    let args = sglang_replay_args();
    let requests = vec![request(111, 9, Some(0.0)), request(112, 9, Some(500.0))];

    let (report, stats) =
        simulate_trace_requests_with_stats(args, requests, 2, 1.0, ReplayRouterMode::KvRouter)
            .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(stats.dispatch_history.len(), 2);
}

#[test]
fn test_online_concurrency_replay_kv_router_respects_max_in_flight() {
    let args = replay_args();
    let requests = vec![
        request(1, 10, None),
        request(2, 20, None),
        request(3, 10, None),
        request(4, 20, None),
    ];

    let (report, stats) =
        simulate_concurrency_requests_with_stats(args, requests, 2, 2, ReplayRouterMode::KvRouter)
            .unwrap();

    assert_eq!(report.request_counts.completed_requests, 4);
    assert!(stats.max_in_flight_seen <= 2);
}

#[test]
fn test_online_trace_replay_kv_router_marks_prefill_and_free_once() {
    let args = replay_args();
    let requests = vec![DirectRequest {
        tokens: vec![9; 64],
        max_output_tokens: 1,
        uuid: Some(Uuid::from_u128(9)),
        dp_rank: 0,
        arrival_timestamp_ms: Some(0.0),
    }];

    let (_, stats) =
        simulate_trace_requests_with_stats(args, requests, 1, 1.0, ReplayRouterMode::KvRouter)
            .unwrap();

    assert_eq!(stats.prefill_marked_count, 1);
    assert_eq!(stats.freed_count, 1);
}
