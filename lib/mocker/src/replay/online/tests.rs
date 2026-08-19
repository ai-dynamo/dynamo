// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use dynamo_kv_router::PrefillLoadEstimator;
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};
use tokio::sync::{Notify, mpsc, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::common::protocols::{
    DirectRequest, EngineType, MockEngineArgs, PreemptionMode, SglangArgs,
};
use crate::live::ObservedAdmission;
use crate::loadgen::{AgenticTrace, AgenticTurnTrace, SessionTrace, Trace, TurnTrace};
use crate::replay::{ReplayRouterMode, ReplayTerminalStatus, SlaThresholds};

use super::entrypoints::{
    OnlineReplayConfig, OnlineReplayOptions, simulate_agentic_trace_workload,
    simulate_concurrency_requests_with_stats, simulate_concurrency_workload_with_stats,
    simulate_trace_requests, simulate_trace_requests_with_stats, simulate_trace_workload,
    simulate_trace_workload_with_stats,
};
use super::live_runtime::LiveRuntime;
use super::recorder::{
    OnlineRecorderOptions, OnlineTraceRecorder, TerminalObservation, forward_admissions,
};
use super::state::{LiveReplayMode, WorkloadDispatchState, arrival_event};
use super::task::wait_for_workload_progress;
use super::{ReplayPlacement, ReplayRouter};

fn replay_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .speedup_ratio(1000.0)
        .block_size(64)
        .build()
        .unwrap()
}

fn replay_config(
    args: MockEngineArgs,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    options: OnlineReplayOptions,
) -> OnlineReplayConfig {
    OnlineReplayConfig::new(args, None, None, num_workers, router_mode, options)
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
        output_token_ids: None,
        uuid: Some(Uuid::from_u128(uuid)),
        dp_rank: 0,
        arrival_timestamp_ms,
        ..Default::default()
    }
}

#[tokio::test(start_paused = true)]
async fn admission_timestamp_is_preserved_when_forwarding_is_delayed() {
    let start = Instant::now();
    let uuid = Uuid::from_u128(99);
    let request = request(99, 9, None);
    let recorder = OnlineTraceRecorder::start(OnlineRecorderOptions {
        capture_per_request: true,
        ..Default::default()
    });
    let recorder_tx = recorder.sender();
    recorder_tx
        .record_arrival(arrival_event(&request, 0.0).unwrap())
        .unwrap();

    let (admission_tx, admission_rx) = mpsc::unbounded_channel();
    admission_tx
        .send(ObservedAdmission {
            event: crate::scheduler::AdmissionEvent {
                uuid,
                reused_input_tokens: 0,
            },
            observed_at: Instant::now(),
        })
        .unwrap();
    tokio::time::advance(Duration::from_secs(1)).await;
    drop(admission_tx);
    forward_admissions(start, admission_rx, recorder.sender())
        .await
        .unwrap();

    recorder_tx
        .record_terminal(TerminalObservation {
            uuid,
            token_times_ms: vec![500.0],
            terminal_time_ms: 500.0,
            status: ReplayTerminalStatus::Completed,
        })
        .unwrap();
    drop(recorder_tx);
    let report = recorder.finish(500.0).await.unwrap();
    let record = &report.per_request[0];
    assert_eq!(record.first_admit_ms, Some(0.0));
    assert!(record.first_admit_ms <= record.first_token_ms);
    assert!(record.first_admit_ms.unwrap() <= record.terminal_time_ms);
}

fn trtllm_reject_args() -> MockEngineArgs {
    // 4 GPU blocks * block_size 4 = 16-token to-completion budget per request.
    MockEngineArgs::builder()
        .engine_type(EngineType::Trtllm)
        .block_size(4)
        .num_gpu_blocks(4)
        .max_num_batched_tokens(Some(64))
        .max_num_seqs(Some(4))
        .enable_prefix_caching(false)
        .enable_chunked_prefill(true)
        .speedup_ratio(1000.0)
        .build()
        .unwrap()
}

fn reject_request(uuid: u128, prompt_tokens: u32, max_output: usize) -> DirectRequest {
    let base = uuid as u32 * 100_000;
    DirectRequest {
        tokens: (base..base + prompt_tokens).collect(),
        max_output_tokens: max_output,
        output_token_ids: None,
        uuid: Some(Uuid::from_u128(uuid)),
        dp_rank: 0,
        arrival_timestamp_ms: None,
        ..Default::default()
    }
}

struct FixedPrefillLoadEstimator {
    duration: Duration,
}

impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
    fn predict_prefill_duration(
        &self,
        _batch_size: usize,
        _effective_isl: usize,
        _prefix: usize,
    ) -> anyhow::Result<Duration> {
        Ok(self.duration)
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
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 6,
                        max_output_tokens: 2,
                        hash_ids: vec![21, 22, 23, 24, 25, 26],
                        delay_after_previous_ms: 5.0,
                        ..Default::default()
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
                    ..Default::default()
                }],
            },
        ],
    }
}

#[test]
fn test_online_trace_replay_single_worker_completes() {
    let args = replay_args();
    let requests = vec![request(1, 11, Some(0.0)), request(2, 22, Some(1.0))];

    let report = simulate_trace_requests(
        replay_config(
            args,
            1,
            ReplayRouterMode::RoundRobin,
            OnlineReplayOptions::default(),
        ),
        requests,
        1.0,
    )
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
fn online_report_options_populate_request_goodput_and_capacity_metrics() {
    let args = MockEngineArgs::builder()
        .speedup_ratio(1000.0)
        .block_size(64)
        .aic_tp_size(Some(2))
        .build()
        .unwrap();
    let report = simulate_trace_workload(
        replay_config(
            args,
            2,
            ReplayRouterMode::KvRouter,
            OnlineReplayOptions {
                record_per_request: true,
                sla: SlaThresholds {
                    e2e_ms: Some(1_000_000.0),
                    ..Default::default()
                },
            },
        ),
        multiturn_trace(),
        true,
    )
    .unwrap();

    assert_eq!(report.per_request.len(), 3);
    assert!(
        report
            .per_request
            .iter()
            .all(|record| record.session_id.is_some() && record.decode_worker_idx.is_some())
    );
    assert_eq!(report.goodput.unwrap().completed_requests, 3);
    assert_eq!(report.throughput.prefill_worker_seconds, 0.0);
    assert_eq!(report.throughput.decode_gpus_per_worker, 2);
    let expected_worker_seconds = 2.0 * report.throughput.duration_ms / 1000.0;
    assert!(
        (report.throughput.decode_worker_seconds - expected_worker_seconds).abs() < f64::EPSILON
    );
    let expected_gpu_hours = expected_worker_seconds * 2.0 / 3600.0;
    assert!((report.throughput.gpu_hours - expected_gpu_hours).abs() < f64::EPSILON);
}

#[test]
fn online_agentic_trace_releases_dependency_after_parent_completion() {
    let trace = AgenticTrace {
        block_size: 64,
        turns: vec![
            AgenticTurnTrace {
                request_id: "root".to_string(),
                session_id: "root".to_string(),
                input_length: 64,
                max_output_tokens: 2,
                hash_ids: vec![1],
                first_ready_timestamp_ms: Some(0.0),
                prefix_reset: true,
                ..Default::default()
            },
            AgenticTurnTrace {
                request_id: "dependent".to_string(),
                session_id: "dependent".to_string(),
                input_length: 64,
                max_output_tokens: 2,
                hash_ids: vec![2],
                first_ready_timestamp_ms: Some(0.0),
                delay_after_dependencies_ms: 5.0,
                wait_for: vec!["root".to_string()],
                prefix_reset: true,
                ..Default::default()
            },
        ],
    };
    let report = simulate_agentic_trace_workload(
        replay_config(
            replay_args(),
            2,
            ReplayRouterMode::KvRouter,
            OnlineReplayOptions {
                record_per_request: true,
                ..Default::default()
            },
        ),
        trace,
    )
    .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    let root = report
        .per_request
        .iter()
        .find(|record| record.session_id.as_deref() == Some("root"))
        .unwrap();
    let dependent = report
        .per_request
        .iter()
        .find(|record| record.session_id.as_deref() == Some("dependent"))
        .unwrap();
    assert!(dependent.arrival_time_ms >= root.terminal_time_ms + 5.0);
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

#[test]
fn test_record_arrival_uses_caller_arrival_timestamp() {
    let uuid = Uuid::from_u128(999);
    let arrival_at_ms = 123.0;
    let request = request(999, 42, Some(arrival_at_ms));

    let arrival = arrival_event(&request, arrival_at_ms).unwrap();
    assert_eq!(arrival.uuid, uuid);
    assert_eq!(arrival.at_ms, arrival_at_ms);
}

#[tokio::test(start_paused = true)]
async fn test_online_kv_router_prefill_load_estimator_decays_active_tokens() {
    let args = replay_args();
    let router = ReplayRouter::new(
        ReplayRouterMode::KvRouter,
        &args,
        Some(KvRouterConfig {
            router_track_prefill_tokens: true,
            router_prefill_load_model: RouterPrefillLoadModel::Aic,
            ..KvRouterConfig::default()
        }),
        Some(Arc::new(FixedPrefillLoadEstimator {
            duration: Duration::from_secs(10),
        })),
        1,
    )
    .unwrap();

    assert_eq!(
        router
            .select_worker(&request(1, 11, Some(0.0)), 1, 1)
            .await
            .unwrap(),
        ReplayPlacement {
            worker_idx: 0,
            dp_rank: 0,
        }
    );
    assert_eq!(
        router.debug_potential_loads(0, true)[0].potential_prefill_tokens,
        64
    );

    tokio::time::advance(Duration::from_secs(5)).await;
    assert_eq!(
        router.debug_potential_loads(0, true)[0].potential_prefill_tokens,
        32
    );

    tokio::time::advance(Duration::from_secs(5)).await;
    assert_eq!(
        router.debug_potential_loads(0, true)[0].potential_prefill_tokens,
        0
    );

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
                    ..Default::default()
                },
                TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![5, 6, 7, 8],
                    delay_after_previous_ms: 5.0,
                    ..Default::default()
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
    let notify = Arc::new(Notify::new());
    let wake = notify.notified();
    tokio::pin!(wake);

    assert!(
        tokio::time::timeout(
            tokio::time::Duration::from_millis(20),
            wait_for_workload_progress(None, Instant::now(), wake.as_mut()),
        )
        .await
        .is_err(),
        "concurrency workload should wait for wakeup when no turn is time-ready"
    );

    let wake = notify.notified();
    tokio::pin!(wake);
    let wait = wait_for_workload_progress(None, Instant::now(), wake.as_mut());
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

#[tokio::test]
async fn test_online_trace_replay_uses_grouped_attention_dp_engine() {
    let mut args = replay_args();
    args.dp_size = 2;
    let pending = VecDeque::from(vec![
        request(1, 1, Some(0.0)),
        request(2, 2, Some(1.0)),
        request(3, 3, Some(2.0)),
        request(4, 4, Some(3.0)),
    ]);
    let runtime = LiveRuntime::new(
        replay_config(
            args,
            1,
            ReplayRouterMode::RoundRobin,
            OnlineReplayOptions::default(),
        ),
        pending,
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )
    .unwrap();

    assert_eq!(runtime.engines().len(), 2);
    let (report, stats) = runtime.run().await.unwrap();
    assert_eq!(report.request_counts.completed_requests, 4);
    assert_eq!(stats.dispatch_history, vec![0, 0, 0, 0]);
}

#[tokio::test]
async fn terminal_delivery_waits_for_grouped_completion_boundary_before_shutdown() {
    let mut terminal_request = request(10, 10, Some(0.0));
    terminal_request.max_output_tokens = 1;
    terminal_request.output_token_ids = Some(vec![10_010]);
    let runtime = LiveRuntime::new(
        replay_config(
            replay_args(),
            1,
            ReplayRouterMode::RoundRobin,
            OnlineReplayOptions::default(),
        ),
        VecDeque::from(vec![terminal_request]),
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )
    .unwrap();
    let engines = runtime.engines();
    let boundary = engines[0].pause_completion_boundary_before_finish();
    let run = tokio::spawn(runtime.run());

    tokio::time::timeout(Duration::from_secs(1), boundary.wait_until_reached())
        .await
        .expect("completion dispatcher should reach the final boundary");
    tokio::time::timeout(Duration::from_secs(1), async {
        while engines[0].active_request_count() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("terminal output should be delivered before boundary finish");
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert!(
        !engines[0].group_is_cancelled(),
        "LiveRunSession must drain the completion boundary before shutdown cancellation"
    );
    assert!(!run.is_finished());

    boundary.release();
    let (report, _) = tokio::time::timeout(Duration::from_secs(1), run)
        .await
        .expect("orderly boundary release should finish replay")
        .unwrap()
        .unwrap();
    assert_eq!(report.request_counts.completed_requests, 1);
}

#[tokio::test]
async fn test_online_concurrency_replay_reaches_but_does_not_exceed_cap() {
    let args = replay_args();
    let requests = VecDeque::from(vec![
        request(1, 10, None),
        request(2, 20, None),
        request(3, 30, None),
        request(4, 40, None),
    ]);
    let (gate_tx, gate_rx) = watch::channel(false);
    let runtime = LiveRuntime::new_with_output_gate(
        replay_config(
            args,
            2,
            ReplayRouterMode::KvRouter,
            OnlineReplayOptions::default(),
        ),
        requests,
        LiveReplayMode::Concurrency { max_in_flight: 2 },
        gate_rx,
        CancellationToken::new(),
    )
    .unwrap();
    let engines = runtime.engines();
    let run = tokio::spawn(runtime.run());

    tokio::time::timeout(Duration::from_secs(1), async {
        while engines
            .iter()
            .map(|engine| engine.active_request_count())
            .sum::<usize>()
            != 2
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("two requests should reach the gated engines");
    assert!(!run.is_finished());
    gate_tx.send(true).unwrap();
    let (report, stats) = run.await.unwrap().unwrap();
    assert_eq!(report.request_counts.completed_requests, 4);
    assert_eq!(stats.max_in_flight_seen, 2);
}

/// Live-runtime regression for terminal-rejection propagation. An oversized
/// request (footprint exceeds the whole KV pool) must reach a terminal state so
/// its waiter is notified — otherwise the request task blocks forever on
/// `wait_for_completion` and the live run never drains. The valid follower runs
/// to completion; the rejected request is excluded from the report.
#[test]
fn trtllm_oversized_request_rejected_unblocks_follower_live() {
    let oversized = reject_request(1, 20, 8); // 20-token prompt = 5 blocks > 4-block pool
    let valid = reject_request(2, 4, 4); // 2 blocks, fits
    let (report, stats) = simulate_concurrency_requests_with_stats(
        trtllm_reject_args(),
        vec![oversized, valid],
        1, // max_in_flight = 1: rejection must notify the waiter or the run hangs
        1,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();
    assert_eq!(
        report.request_counts.num_requests, 2,
        "both requests arrived"
    );
    assert_eq!(
        report.request_counts.completed_requests, 1,
        "only the valid request completes; the rejected one is excluded"
    );
    assert_eq!(stats.prefill_marked_count, 1);
    assert_eq!(stats.freed_count, 2);
}

#[test]
fn test_online_trace_replay_populates_admit_reuse_stats() {
    let args = replay_args();
    let mut requests = vec![request(1, 77, Some(0.0)), request(2, 77, Some(5.0))];
    // A fully cached one-block prompt must recompute that block to produce
    // logits. Use two blocks so this remains a positive-reuse metrics test.
    for request in &mut requests {
        request.tokens.resize(128, 77);
    }

    let report = simulate_trace_requests(
        replay_config(
            args,
            1,
            ReplayRouterMode::RoundRobin,
            OnlineReplayOptions::default(),
        ),
        requests,
        1.0,
    )
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

    let report = simulate_trace_requests(
        replay_config(
            args,
            1,
            ReplayRouterMode::RoundRobin,
            OnlineReplayOptions::default(),
        ),
        requests,
        1.0,
    )
    .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.request_counts.total_output_tokens, 4);
}

#[test]
fn test_online_trace_replay_sglang_zero_output_drains_without_phantom_token() {
    let mut zero_output = request(103, 7, Some(0.0));
    zero_output.max_output_tokens = 0;
    let (report, stats) = simulate_trace_requests_with_stats(
        sglang_replay_args(),
        vec![zero_output, request(104, 8, Some(1.0))],
        1,
        1.0,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();

    // Zero-output trace rows count as completed prefill work but do not
    // manufacture a token or latency sample. The normal follower also completes.
    assert_eq!(report.request_counts.num_requests, 2);
    assert_eq!(report.request_counts.completed_requests, 2);
    assert_eq!(report.request_counts.total_input_tokens, 128);
    assert_eq!(report.request_counts.total_output_tokens, 2);
    assert_eq!(stats.prefill_marked_count, 1);
    assert_eq!(stats.freed_count, 2);
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
fn test_online_trace_replay_kv_router_marks_prefill_and_free_once() {
    let args = replay_args();
    let requests = vec![DirectRequest {
        tokens: vec![9; 64],
        max_output_tokens: 1,
        output_token_ids: None,
        uuid: Some(Uuid::from_u128(9)),
        dp_rank: 0,
        arrival_timestamp_ms: Some(0.0),
        ..Default::default()
    }];

    let (_, stats) =
        simulate_trace_requests_with_stats(args, requests, 1, 1.0, ReplayRouterMode::KvRouter)
            .unwrap();

    assert_eq!(stats.prefill_marked_count, 1);
    assert_eq!(stats.freed_count, 1);
}

#[test]
fn test_online_replay_crosses_a_bounded_preemption_edge_and_drains() {
    let args = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(6)
        .max_num_batched_tokens(Some(16))
        .max_num_seqs(Some(2))
        .enable_chunked_prefill(true)
        .enable_prefix_caching(false)
        .preemption_mode(PreemptionMode::Lifo)
        .speedup_ratio(1000.0)
        .build()
        .unwrap();
    let requests = (0..2)
        .map(|request_idx| DirectRequest {
            tokens: (0..8).map(|token| request_idx * 100 + token).collect(),
            max_output_tokens: 8,
            uuid: Some(Uuid::from_u128(request_idx as u128 + 1)),
            ..Default::default()
        })
        .collect();

    let (report, stats) = simulate_concurrency_requests_with_stats(
        args,
        requests,
        2,
        1,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap();

    assert_eq!(report.request_counts.completed_requests, 2);
    assert!(
        (1..=3).contains(&stats.vllm_preemptions_total),
        "fixture should cross the capacity edge without preemption cycling: {}",
        stats.vllm_preemptions_total
    );
}

#[test]
fn test_online_replay_four_workers_clean_every_lifecycle() {
    const REQUEST_COUNT: usize = 16;
    const WORKER_COUNT: usize = 4;

    let requests = (0..REQUEST_COUNT)
        .map(|request_idx| DirectRequest {
            tokens: vec![request_idx as u32; 64],
            max_output_tokens: 2,
            uuid: Some(Uuid::from_u128(request_idx as u128 + 1)),
            ..Default::default()
        })
        .collect();
    let (report, stats) = simulate_concurrency_requests_with_stats(
        replay_args(),
        requests,
        REQUEST_COUNT,
        WORKER_COUNT,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();

    assert_eq!(report.request_counts.completed_requests, REQUEST_COUNT);
    assert_eq!(stats.dispatch_history.len(), REQUEST_COUNT);
    assert_eq!(stats.prefill_marked_count, REQUEST_COUNT);
    assert_eq!(stats.freed_count, REQUEST_COUNT);
    for worker_idx in 0..WORKER_COUNT {
        assert!(stats.dispatch_history.contains(&worker_idx));
    }
}

#[tokio::test]
async fn injected_cancellation_terminates_an_outer_arrival_wait() {
    let cancel = CancellationToken::new();
    let runtime = LiveRuntime::new(
        replay_config(
            replay_args(),
            1,
            ReplayRouterMode::KvRouter,
            OnlineReplayOptions::default(),
        ),
        VecDeque::from(vec![request(700, 7, Some(60_000.0))]),
        LiveReplayMode::Trace,
        cancel.clone(),
    )
    .unwrap();
    let run = tokio::spawn(runtime.run());

    tokio::task::yield_now().await;
    assert!(!run.is_finished());
    cancel.cancel();
    let error = tokio::time::timeout(Duration::from_secs(1), run)
        .await
        .expect("cancellation should terminate replay promptly")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_string(), "online replay cancelled");
}

#[tokio::test]
async fn router_bookkeeping_failures_fail_replay_closed() {
    let mark_runtime = LiveRuntime::new(
        replay_config(
            replay_args(),
            1,
            ReplayRouterMode::KvRouter,
            OnlineReplayOptions::default(),
        ),
        VecDeque::from(vec![request(900, 9, Some(0.0))]),
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )
    .unwrap();
    mark_runtime.router().fail_mark_prefill();

    let mark_error = mark_runtime.run().await.unwrap_err();
    assert!(
        mark_error
            .to_string()
            .contains("injected mark-prefill failure")
    );

    let mut zero_output = request(901, 9, Some(0.0));
    zero_output.max_output_tokens = 0;
    let free_runtime = LiveRuntime::new(
        replay_config(
            replay_args(),
            1,
            ReplayRouterMode::KvRouter,
            OnlineReplayOptions::default(),
        ),
        VecDeque::from(vec![zero_output]),
        LiveReplayMode::Trace,
        CancellationToken::new(),
    )
    .unwrap();
    free_runtime.router().fail_free();

    let free_error = free_runtime.run().await.unwrap_err();
    assert!(free_error.to_string().contains("injected free failure"));
}
