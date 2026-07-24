// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use super::super::entrypoints::{
    run_concurrency_collect, run_concurrency_workload_collect, run_trace_collect,
    run_trace_workload_collect,
};
use super::super::extensions::kv_events::HandoffDisaggRuntime;
use super::super::planner_hook::PlannerTickDecision;
use super::*;
use crate::common::protocols::{
    EngineType, KvTransferTimingMode, MockEngineArgs, SglangArgs, WorkerType,
};
use crate::loadgen::{SessionTrace, Trace, TurnTrace};
use crate::replay::TraceSimulationReport;

struct CaptureOnceHook {
    at_ms: f64,
    captured: Rc<RefCell<Option<PlannerTickMetrics>>>,
}

impl PlannerHook for CaptureOnceHook {
    fn initial_tick_ms(&mut self) -> anyhow::Result<f64> {
        Ok(self.at_ms)
    }

    fn on_tick(&mut self, metrics: PlannerTickMetrics) -> anyhow::Result<PlannerTickDecision> {
        *self.captured.borrow_mut() = Some(metrics);
        Ok(PlannerTickDecision::default())
    }
}

fn staged_args(worker_type: WorkerType, speedup_ratio: f64) -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(64)
        .num_gpu_blocks(256)
        .max_num_batched_tokens(Some(8192))
        .max_num_seqs(Some(8))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .speedup_ratio(speedup_ratio)
        .decode_speedup_ratio(speedup_ratio)
        .worker_type(worker_type)
        .build()
        .unwrap()
}

fn sglang_staged_args(worker_type: WorkerType, speedup_ratio: f64) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .block_size(64)
        .num_gpu_blocks(512)
        .max_num_batched_tokens(Some(8192))
        .max_num_seqs(Some(8))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .speedup_ratio(speedup_ratio)
        .decode_speedup_ratio(speedup_ratio)
        .worker_type(worker_type)
        .sglang(Some(SglangArgs {
            page_size: Some(64),
            ..Default::default()
        }))
        .build()
        .unwrap()
}

fn disagg_config() -> OfflineDisaggReplayConfig {
    OfflineDisaggReplayConfig {
        prefill_args: staged_args(WorkerType::Prefill, 1000.0),
        decode_args: staged_args(WorkerType::Decode, 1000.0),
        num_prefill_workers: 2,
        num_decode_workers: 2,
    }
}

fn sglang_disagg_config() -> OfflineDisaggReplayConfig {
    OfflineDisaggReplayConfig {
        prefill_args: sglang_staged_args(WorkerType::Prefill, 1000.0),
        decode_args: sglang_staged_args(WorkerType::Decode, 1000.0),
        num_prefill_workers: 2,
        num_decode_workers: 2,
    }
}

fn forced_chunked_handoff_config(engine_type: EngineType) -> OfflineDisaggReplayConfig {
    let mut config = match engine_type {
        EngineType::Vllm => disagg_config(),
        EngineType::Sglang => sglang_disagg_config(),
        EngineType::Trtllm => unreachable!(),
    };
    config.num_prefill_workers = 1;
    config.num_decode_workers = 1;
    config.prefill_args.speedup_ratio = 1.0;
    config.prefill_args.max_num_batched_tokens = Some(64);
    if let Some(sglang) = config.prefill_args.sglang.as_mut() {
        sglang.chunked_prefill_size = Some(64);
        sglang.max_prefill_tokens = Some(64);
    }
    config
}

fn disagg_config_with_handoff_delay() -> OfflineDisaggReplayConfig {
    let mut config = disagg_config();
    config.prefill_args.kv_transfer_bandwidth = Some(1.0);
    config.prefill_args.kv_bytes_per_token = Some(1_000_000);
    config
}

fn transfer_timing_config(
    engine_type: EngineType,
    mode: KvTransferTimingMode,
    decode_workers: usize,
) -> OfflineDisaggReplayConfig {
    let mut config = match engine_type {
        EngineType::Vllm => disagg_config(),
        EngineType::Sglang => sglang_disagg_config(),
        EngineType::Trtllm => unreachable!(),
    };
    config.num_prefill_workers = 1;
    config.num_decode_workers = decode_workers;
    config.prefill_args.kv_transfer_bandwidth = Some(1.0);
    config.prefill_args.kv_bytes_per_token = Some(1_000_000);
    config.prefill_args.kv_transfer_timing_mode = mode;
    config.decode_args.kv_transfer_timing_mode = mode;
    config
}

fn cleanup_overtake_args(engine_type: EngineType, worker_type: WorkerType) -> MockEngineArgs {
    let mut builder = MockEngineArgs::builder()
        .engine_type(engine_type)
        .block_size(512)
        .num_gpu_blocks(20_000)
        .max_num_batched_tokens(Some(32_768))
        .max_num_seqs(Some(64))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .speedup_ratio(1.0)
        .decode_speedup_ratio(1.0)
        .worker_type(worker_type);
    if worker_type == WorkerType::Prefill {
        builder = builder
            .kv_transfer_bandwidth(Some(100.0))
            .kv_bytes_per_token(Some(131_072));
    }
    if engine_type == EngineType::Sglang {
        builder = builder.sglang(Some(SglangArgs {
            page_size: Some(512),
            ..Default::default()
        }));
    }
    builder.build().unwrap()
}

fn cleanup_overtake_config(engine_type: EngineType) -> OfflineDisaggReplayConfig {
    OfflineDisaggReplayConfig {
        prefill_args: cleanup_overtake_args(engine_type, WorkerType::Prefill),
        decode_args: cleanup_overtake_args(engine_type, WorkerType::Decode),
        num_prefill_workers: 2,
        num_decode_workers: 2,
    }
}

fn trtllm_reject_staged_args(worker_type: WorkerType) -> MockEngineArgs {
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
        .worker_type(worker_type)
        .build()
        .unwrap()
}

fn trtllm_reject_disagg_config() -> OfflineDisaggReplayConfig {
    OfflineDisaggReplayConfig {
        prefill_args: trtllm_reject_staged_args(WorkerType::Prefill),
        decode_args: trtllm_reject_staged_args(WorkerType::Decode),
        num_prefill_workers: 1,
        num_decode_workers: 1,
    }
}

#[test]
fn trtllm_disaggregation_is_rejected_before_runtime_state() {
    let result = DisaggRuntime::new(
        &trtllm_reject_disagg_config(),
        None,
        None,
        VecDeque::from([request(1, 4, 4, 0.0)]),
        ReplayMode::Concurrency { max_in_flight: 1 },
        ReplayRouterMode::RoundRobin,
    );
    assert!(matches!(
        result,
        Err(error) if error.to_string().contains("does not support TRT-LLM")
    ));
}

fn scaling_test_args(worker_type: WorkerType) -> MockEngineArgs {
    MockEngineArgs::builder()
        .block_size(64)
        .num_gpu_blocks(512)
        .max_num_batched_tokens(Some(64))
        .max_num_seqs(Some(8))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .speedup_ratio(1.0)
        .decode_speedup_ratio(1.0)
        .worker_type(worker_type)
        .build()
        .unwrap()
}

fn scaling_test_disagg_config() -> OfflineDisaggReplayConfig {
    OfflineDisaggReplayConfig {
        prefill_args: scaling_test_args(WorkerType::Prefill),
        decode_args: scaling_test_args(WorkerType::Decode),
        num_prefill_workers: 1,
        num_decode_workers: 1,
    }
}

fn pending_destination_scale_down_config() -> OfflineDisaggReplayConfig {
    let prefill = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(32)
        .max_num_batched_tokens(Some(64))
        .max_num_seqs(Some(4))
        .worker_type(WorkerType::Prefill)
        .speedup_ratio(1000.0)
        .build()
        .unwrap();
    let decode = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(2)
        .max_num_batched_tokens(Some(64))
        .max_num_seqs(Some(4))
        .worker_type(WorkerType::Decode)
        .speedup_ratio(1.0)
        .decode_speedup_ratio(1.0)
        .build()
        .unwrap();
    OfflineDisaggReplayConfig {
        prefill_args: prefill,
        decode_args: decode,
        num_prefill_workers: 1,
        num_decode_workers: 1,
    }
}

fn router_config() -> ReplayKvRouterConfig {
    ReplayKvRouterConfig {
        router_queue_threshold: Some(1.25),
        ..ReplayKvRouterConfig::default()
    }
}

fn planner_router_config() -> ReplayKvRouterConfig {
    ReplayKvRouterConfig {
        router_queue_threshold: Some(0.5),
        ..ReplayKvRouterConfig::default()
    }
}

fn request(
    uuid: u128,
    prompt_tokens: usize,
    output_tokens: usize,
    arrival_ms: f64,
) -> DirectRequest {
    DirectRequest {
        tokens: vec![1; prompt_tokens],
        max_output_tokens: output_tokens,
        uuid: Some(Uuid::from_u128(uuid)),
        dp_rank: 0,
        arrival_timestamp_ms: Some(arrival_ms),
        ..Default::default()
    }
}

#[test]
fn planner_tick_emits_idle_fpm_for_both_disagg_pools() {
    let mut config = disagg_config();
    config.num_prefill_workers = 1;
    config.num_decode_workers = 1;
    let pending = crate::replay::normalize_trace_requests(
        vec![request(9_301, 64, 1, 0.0), request(9_302, 64, 1, 3_000.0)],
        1.0,
    )
    .unwrap();
    let captured = Rc::new(RefCell::new(None));
    let hook = CaptureOnceHook {
        at_ms: 2_000.0,
        captured: Rc::clone(&captured),
    };

    DisaggRuntime::new(
        &config,
        None,
        None,
        pending,
        ReplayMode::Trace,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap()
    .with_planner_hook(Box::new(hook))
    .run()
    .unwrap();

    let metrics = captured
        .borrow_mut()
        .take()
        .expect("planner tick must fire");
    assert_eq!(metrics.now_ms, 2_000.0);
    for snapshots in [&metrics.prefill_fpm, &metrics.decode_fpm] {
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].0, 0);
        assert_eq!(snapshots[0].1.dp_rank, 0);
        assert_eq!(snapshots[0].1.wall_time_secs, 0.0);
        assert_eq!(snapshots[0].1.num_prefill_requests, 0);
        assert_eq!(snapshots[0].1.num_decode_requests, 0);
        assert_eq!(snapshots[0].1.num_queued_prefill, 0);
        assert_eq!(snapshots[0].1.num_queued_decode, 0);
    }
}

fn run_trace_with_details(
    config: &OfflineDisaggReplayConfig,
    requests: Vec<DirectRequest>,
    router_config: Option<ReplayKvRouterConfig>,
    router_mode: ReplayRouterMode,
) -> TraceSimulationReport {
    let pending = crate::replay::normalize_trace_requests(requests, 1.0).unwrap();
    let (collector, _) = DisaggRuntime::new(
        config,
        router_config,
        None,
        pending,
        ReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .with_per_request_records(true)
    .run()
    .unwrap();
    collector.finish()
}

#[rstest::rstest]
#[case(EngineType::Vllm)]
#[case(EngineType::Sglang)]
fn zero_output_disagg_does_not_count_a_source_token(#[case] engine_type: EngineType) {
    let mut config = match engine_type {
        EngineType::Vllm => disagg_config(),
        EngineType::Sglang => sglang_disagg_config(),
        EngineType::Trtllm => unreachable!(),
    };
    config.num_prefill_workers = 1;
    config.num_decode_workers = 1;
    let request = DirectRequest {
        tokens: vec![1; 64],
        max_output_tokens: 0,
        uuid: Some(Uuid::from_u128(90_010)),
        arrival_timestamp_ms: Some(0.0),
        ..Default::default()
    };
    let mut runtime =
        HandoffDisaggRuntime::new_handoff_conformance(&config, VecDeque::from([request])).unwrap();

    runtime.run_to_completion().unwrap();

    assert_eq!(
        runtime
            .flow
            .conformance_capture
            .as_ref()
            .unwrap()
            .source_output_tokens,
        0
    );
    let report = std::mem::take(&mut runtime.collector).finish();
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.request_counts.total_output_tokens, 0);
}

fn multiturn_trace() -> Trace {
    Trace {
        block_size: 64,
        sessions: vec![
            SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 64,
                        max_output_tokens: 2,
                        hash_ids: vec![11],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 192,
                        max_output_tokens: 2,
                        hash_ids: vec![21, 22, 23],
                        delay_after_previous_ms: 10.0,
                        ..Default::default()
                    },
                ],
            },
            SessionTrace {
                session_id: "session-b".to_string(),
                first_arrival_timestamp_ms: Some(5.0),
                turns: vec![TurnTrace {
                    input_length: 128,
                    max_output_tokens: 2,
                    hash_ids: vec![31, 32],
                    delay_after_previous_ms: 0.0,
                    ..Default::default()
                }],
            },
        ],
    }
}

fn transition_index(transitions: &[DisaggTransition], needle: DisaggTransition) -> usize {
    transitions
        .iter()
        .position(|transition| *transition == needle)
        .unwrap()
}

#[test]
fn test_derive_stage_router_configs_force_required_overrides() {
    let config = ReplayKvRouterConfig {
        overlap_score_credit: 1.0,
        router_track_active_blocks: true,
        router_assume_kv_reuse: true,
        router_track_prefill_tokens: true,
        ..ReplayKvRouterConfig::default()
    };
    let args = staged_args(WorkerType::Prefill, 1.0);
    let prefill = derive_prefill_router_config(&args, Some(config.clone()));
    let decode = derive_decode_router_config(&args, Some(config));

    assert!(!prefill.router_track_active_blocks);
    assert_eq!(decode.overlap_score_credit, 0.0);
    assert!(!decode.router_assume_kv_reuse);
    assert!(!decode.router_track_prefill_tokens);
}

#[rstest::rstest]
#[case(ReplayRouterMode::RoundRobin)]
#[case(ReplayRouterMode::KvRouter)]
fn test_trace_smoke_reports_decode_only_tokens(#[case] router_mode: ReplayRouterMode) {
    let config = disagg_config();
    let requests = vec![request(1, 128, 3, 5.0)];

    let router_config = (router_mode == ReplayRouterMode::KvRouter).then(router_config);
    let (collector, stats) = run_trace_collect(&config, requests, router_config, 1.0, router_mode);
    let snapshot = collector.snapshot(Uuid::from_u128(1)).unwrap();
    let report = collector.finish();

    assert_eq!(snapshot.arrival_time_ms, 0.0);
    assert!(snapshot.first_admit_ms.is_some());
    assert!(snapshot.first_token_ms.is_some());
    assert_eq!(snapshot.output_length, 3);
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.request_counts.total_output_tokens, 3);
    assert_eq!(
        stats.request_snapshots[&Uuid::from_u128(1)].phase,
        DisaggPhase::Done
    );
}

#[test]
fn decode_terminal_retains_handoff_until_deferred_cleanup_drains() {
    let request_shapes = [
        (6_755, 500),
        (7_319, 490),
        (7_234, 794),
        (2_287, 316),
        (9_013, 3),
        (6_506, 3),
        (4_824, 173),
        (3_119, 20),
        (23_090, 453),
    ];

    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let requests = request_shapes
            .into_iter()
            .enumerate()
            .map(|(index, (input, output))| {
                let id = index as u128 + 1;
                let mut request = request(id, input, output, 0.0);
                request.tokens = (0..input)
                    .map(|position| {
                        if position < 512 {
                            position as u32
                        } else {
                            index as u32 * 100_000 + position as u32
                        }
                    })
                    .collect();
                request
            })
            .collect::<Vec<_>>();
        let expected_output_tokens: usize = request_shapes.iter().map(|(_, output)| *output).sum();
        let (collector, stats) = run_trace_collect(
            &cleanup_overtake_config(engine_type),
            requests,
            None,
            1.0,
            ReplayRouterMode::RoundRobin,
        );
        let report = collector.finish();

        assert_eq!(
            report.request_counts.completed_requests,
            request_shapes.len()
        );
        assert_eq!(
            report.request_counts.total_output_tokens,
            expected_output_tokens
        );
        assert!(
            (1..=request_shapes.len()).any(|id| {
                let uuid = Uuid::from_u128(id as u128);
                let decode_done = stats
                    .transition_log
                    .iter()
                    .position(|event| *event == DisaggTransition::RequestMarkedDone { uuid });
                let source_released = stats
                    .transition_log
                    .iter()
                    .position(|event| *event == DisaggTransition::SourceReleased { uuid });
                decode_done
                    .zip(source_released)
                    .is_some_and(|(done, released)| done < released)
            }),
            "{engine_type:?} never exercised decode completion before source cleanup"
        );
        assert!(
            stats
                .request_snapshots
                .values()
                .all(|snapshot| snapshot.phase == DisaggPhase::Done)
        );
    }
}

#[rstest::rstest]
#[case(ReplayRouterMode::RoundRobin)]
#[case(ReplayRouterMode::KvRouter)]
fn test_prefill_and_decode_use_separate_worker_pools(#[case] router_mode: ReplayRouterMode) {
    let config = disagg_config();
    let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 10.0)];

    let router_config = (router_mode == ReplayRouterMode::KvRouter).then(router_config);
    let (_, stats) = run_trace_collect(&config, requests, router_config, 1.0, router_mode);

    for uuid in [Uuid::from_u128(1), Uuid::from_u128(2)] {
        assert!(stats.prefill_assignments.contains_key(&uuid));
        assert!(stats.decode_assignments.contains_key(&uuid));
        assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
        assert_eq!(
            stats.request_snapshots[&uuid].prefill_worker_idx,
            Some(stats.prefill_assignments[&uuid])
        );
        assert_eq!(
            stats.request_snapshots[&uuid].decode_worker_idx,
            Some(stats.decode_assignments[&uuid])
        );
    }
}

#[test]
fn source_cleanup_events_restore_prefill_prefix_routing() {
    let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 100.0)];

    for config in [disagg_config(), sglang_disagg_config()] {
        let (collector, stats) = run_trace_collect(
            &config,
            requests.clone(),
            Some(router_config()),
            1.0,
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(
            stats.prefill_assignments[&Uuid::from_u128(1)],
            stats.prefill_assignments[&Uuid::from_u128(2)],
        );
        assert!(
            collector
                .snapshot(Uuid::from_u128(2))
                .unwrap()
                .reused_input_tokens
                > 0
        );
    }
}

#[test]
fn test_hidden_prefill_reports_reused_tokens_even_when_decode_prefix_caching_is_disabled() {
    let mut config = disagg_config();
    config.num_prefill_workers = 1;
    config.num_decode_workers = 1;
    config.decode_args.enable_prefix_caching = false;

    let requests = vec![request(1, 128, 2, 0.0), request(2, 128, 2, 100.0)];
    let (collector, _) = run_trace_collect(
        &config,
        requests,
        Some(router_config()),
        1.0,
        ReplayRouterMode::KvRouter,
    );

    let request_2 = collector.snapshot(Uuid::from_u128(2)).unwrap();
    let report = collector.finish();

    assert!(request_2.reused_input_tokens > 0);
    assert!(report.prefix_cache_reused_ratio > 0.0);
}

#[rstest::rstest]
#[case(ReplayRouterMode::RoundRobin)]
#[case(ReplayRouterMode::KvRouter)]
fn test_concurrency_backfill_waits_for_decode_completion(#[case] router_mode: ReplayRouterMode) {
    let config = disagg_config();
    let requests = vec![
        DirectRequest {
            tokens: vec![1; 128],
            max_output_tokens: 3,
            uuid: Some(Uuid::from_u128(1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        },
        DirectRequest {
            tokens: vec![2; 128],
            max_output_tokens: 3,
            uuid: Some(Uuid::from_u128(2)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            ..Default::default()
        },
    ];

    let router_config = (router_mode == ReplayRouterMode::KvRouter).then(router_config);
    let (collector, stats) =
        run_concurrency_collect(&config, requests, router_config, 1, router_mode);
    let first = collector.snapshot(Uuid::from_u128(1)).unwrap();
    let second = collector.snapshot(Uuid::from_u128(2)).unwrap();

    assert_eq!(first.arrival_time_ms, 0.0);
    assert_eq!(second.arrival_time_ms, first.last_token_ms.unwrap());
    assert_eq!(
        stats.request_snapshots[&Uuid::from_u128(1)].phase,
        DisaggPhase::Done
    );
    assert_eq!(
        stats.request_snapshots[&Uuid::from_u128(2)].phase,
        DisaggPhase::Done
    );
}

#[test]
fn test_source_release_waits_for_destination_activation() {
    for config in [disagg_config(), sglang_disagg_config()] {
        let (_, stats) = run_trace_collect(
            &config,
            vec![request(1, 128, 2, 0.0)],
            Some(router_config()),
            1.0,
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(stats.prefill_marked_count, 1);
        assert_eq!(stats.prefill_router_freed_count, 1);
        assert_eq!(stats.decode_router_freed_count, 1);
        let transitions = &stats.transition_log;
        let uuid = Uuid::from_u128(1);
        let mark_idx =
            transition_index(transitions, DisaggTransition::PrefillMarkCompleted { uuid });
        let free_idx = transition_index(transitions, DisaggTransition::PrefillFree { uuid });
        let held_idx = transition_index(transitions, DisaggTransition::SourceHeld { uuid });
        let activated_idx =
            transition_index(transitions, DisaggTransition::DestinationActivated { uuid });
        let released_idx = transition_index(transitions, DisaggTransition::SourceReleased { uuid });
        assert!(mark_idx < held_idx);
        assert!(held_idx < activated_idx);
        assert!(activated_idx < released_idx);
        assert!(released_idx < free_idx);
        assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
    }
}

#[rstest::rstest]
#[case(EngineType::Vllm)]
#[case(EngineType::Sglang)]
fn chunked_prefill_handoff_waits_for_full_materialization(#[case] engine_type: EngineType) {
    let config = forced_chunked_handoff_config(engine_type);
    let uuid = Uuid::from_u128(1);
    let pending =
        crate::replay::normalize_trace_requests(vec![request(uuid.as_u128(), 192, 2, 0.0)], 1.0)
            .unwrap();
    let mut runtime = DisaggRuntime::new(
        &config,
        None,
        None,
        pending,
        ReplayMode::Trace,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap()
    .with_per_request_records(true)
    .with_fpm_capture();

    let mut prefill_fpm = Vec::new();
    for _ in 0..32 {
        if runtime
            .stats
            .transition_log
            .contains(&DisaggTransition::SourceHeld { uuid })
        {
            break;
        }
        let next = runtime
            .next_timestamp()
            .expect("chunked source must retain scheduled work");
        runtime.advance_to(next).unwrap();
        prefill_fpm.extend(runtime.drain_prefill_fpm());
    }

    assert!(
        runtime
            .stats
            .transition_log
            .contains(&DisaggTransition::SourceHeld { uuid }),
        "source must reach terminal hold"
    );
    let prefill_chunks = prefill_fpm
        .iter()
        .filter(|(_, snapshot)| snapshot.sum_prefill_tokens > 0)
        .collect::<Vec<_>>();
    assert!(
        prefill_chunks.len() >= 3,
        "192 prompt tokens with a 64-token budget must span at least three passes"
    );
    assert_eq!(
        prefill_chunks
            .iter()
            .map(|(_, snapshot)| snapshot.sum_prefill_tokens)
            .sum::<u64>(),
        192
    );

    assert!(runtime.advance_to(f64::MAX).unwrap());
    let decode_fpm = runtime.drain_decode_fpm();
    runtime.finish_test_stats();

    let transitions = &runtime.stats.transition_log;
    let held = transition_index(transitions, DisaggTransition::SourceHeld { uuid });
    let activated = transition_index(transitions, DisaggTransition::DestinationActivated { uuid });
    let admitted = transition_index(transitions, DisaggTransition::DecodeAdmitted { uuid });
    let released = transition_index(transitions, DisaggTransition::SourceReleased { uuid });
    assert!(held < activated);
    assert!(activated < admitted);
    assert!(activated < released);
    assert!(
        decode_fpm
            .iter()
            .all(|(_, snapshot)| snapshot.num_prefill_requests == 0
                && snapshot.sum_prefill_tokens == 0),
        "activated destination must not recompute prompt chunks"
    );
    assert!(
        decode_fpm
            .iter()
            .any(|(_, snapshot)| snapshot.num_decode_requests > 0)
    );
    assert!(runtime.prefill_engine.is_drained());
    assert!(runtime.decode_engine.is_drained());
    assert!(runtime.flow.action_queues.is_empty());
    assert_eq!(
        runtime.stats.request_snapshots[&uuid].phase,
        DisaggPhase::Done
    );

    let report = runtime.collector.finish();
    assert_eq!(report.request_counts.completed_requests, 1);
    assert_eq!(report.request_counts.total_input_tokens, 192);
    assert_eq!(report.request_counts.total_output_tokens, 2);
    assert_eq!(
        report.per_request[0].terminal_status,
        ReplayTerminalStatus::Completed
    );
}

#[test]
fn per_request_handoff_detail_preserves_backend_causality_and_stage_reuse() {
    for (engine_type, config) in [
        (EngineType::Vllm, disagg_config()),
        (EngineType::Sglang, sglang_disagg_config()),
    ] {
        let report = run_trace_with_details(
            &config,
            vec![request(1, 128, 2, 0.0), request(2, 128, 2, 100.0)],
            Some(router_config()),
            ReplayRouterMode::KvRouter,
        );

        assert_eq!(report.per_request.len(), 2);
        let first = &report.per_request[0];
        let second = &report.per_request[1];
        for record in &report.per_request {
            assert_eq!(record.terminal_status, ReplayTerminalStatus::Completed);
            let prefill_admit = record.prefill_admit_ms.unwrap();
            let source_held = record.source_held_ms.unwrap();
            let destination_reserved = record.destination_reserved_ms.unwrap();
            let destination_activated = record.destination_activated_ms.unwrap();
            let source_released = record.source_released_ms.unwrap();
            let decode_admit = record.decode_admit_ms.unwrap();
            assert!(prefill_admit <= source_held);
            assert!(source_held <= source_released);
            assert!(destination_reserved <= destination_activated);
            assert!(destination_activated <= source_released);
            assert!(destination_activated <= decode_admit);
            match engine_type {
                EngineType::Vllm => assert!(source_held <= destination_reserved),
                EngineType::Sglang => assert!(destination_reserved <= prefill_admit),
                EngineType::Trtllm => unreachable!(),
            }
        }

        assert_eq!(first.prefill_route_overlap_tokens, Some(0));
        assert_eq!(first.prefill_admit_ms, first.first_admit_ms);
        assert_eq!(first.decode_reused_input_tokens, Some(0));
        assert_eq!(second.prefill_route_overlap_tokens, Some(128));
        assert_eq!(second.decode_route_overlap_tokens, Some(0));
        assert!(second.reused_input_tokens > 0);
    }
}

#[test]
fn rejected_prefill_remains_rejected_during_failed_handoff_cleanup() {
    let mut config = disagg_config();
    config.num_prefill_workers = 1;
    config.num_decode_workers = 1;
    config.prefill_args.num_gpu_blocks = 1;

    let report = run_trace_with_details(
        &config,
        vec![request(1, 128, 2, 0.0)],
        None,
        ReplayRouterMode::RoundRobin,
    );

    assert_eq!(report.per_request.len(), 1);
    let record = &report.per_request[0];
    assert_eq!(record.terminal_status, ReplayTerminalStatus::Rejected);
    assert!(record.first_token_ms.is_none());
    assert!(record.last_token_ms.is_none());
    assert!(record.ttft_ms.is_none());
    assert!(record.e2e_latency_ms.is_none());
}

#[test]
fn test_permanently_unavailable_destination_unwinds_without_stalling() {
    for mut config in [disagg_config(), sglang_disagg_config()] {
        config.num_prefill_workers = 1;
        config.num_decode_workers = 1;
        config.decode_args.num_gpu_blocks = 1;

        let pending =
            crate::replay::normalize_trace_requests(vec![request(1, 128, 2, 0.0)], 1.0).unwrap();
        let (collector, stats) = DisaggRuntime::new(
            &config,
            None,
            None,
            pending,
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_per_request_records(true)
        .run()
        .unwrap();
        let report = collector.finish();
        let uuid = Uuid::from_u128(1);

        assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
        assert_eq!(report.request_counts.completed_requests, 0);
        assert_eq!(report.per_request.len(), 1);
        assert_eq!(
            report.per_request[0].terminal_status,
            ReplayTerminalStatus::Failed
        );
        assert!(report.per_request[0].first_token_ms.is_none());
        assert!(
            stats
                .transition_log
                .contains(&DisaggTransition::RequestMarkedDone { uuid })
        );
        assert!(!stats.transition_log.iter().any(|transition| matches!(
            transition,
            DisaggTransition::DestinationAccepted { uuid: observed }
                | DisaggTransition::DestinationReserved { uuid: observed }
                if *observed == uuid
        )));
    }
}

#[test]
fn handoff_delay_is_applied_once_to_decode_visible_ttft() {
    let requests = vec![request(1, 128, 2, 0.0)];

    let (baseline_collector, baseline_stats) = run_trace_collect(
        &disagg_config(),
        requests.clone(),
        None,
        1.0,
        ReplayRouterMode::RoundRobin,
    );
    let (delayed_collector, delayed_stats) = run_trace_collect(
        &disagg_config_with_handoff_delay(),
        requests,
        None,
        1.0,
        ReplayRouterMode::RoundRobin,
    );

    let baseline = baseline_collector.snapshot(Uuid::from_u128(1)).unwrap();
    let delayed = delayed_collector.snapshot(Uuid::from_u128(1)).unwrap();
    let baseline_ttft = baseline.first_token_ms.unwrap() - baseline.arrival_time_ms;
    let delayed_ttft = delayed.first_token_ms.unwrap() - delayed.arrival_time_ms;
    let uuid = Uuid::from_u128(1);
    let handoff_delta = delayed_stats.handoff_ms[&uuid] - baseline_stats.handoff_ms[&uuid];

    assert!(
        delayed_ttft >= baseline_ttft + 120.0,
        "expected delayed TTFT to include roughly 128ms of handoff delay, baseline={baseline_ttft}, delayed={delayed_ttft}"
    );
    assert!(
        (handoff_delta - 128.0).abs() < 1e-6,
        "handoff delay must be applied once, observed delta={handoff_delta}ms"
    );
    let queued_idx = transition_index(
        &delayed_stats.transition_log,
        DisaggTransition::TransferQueued { uuid },
    );
    let activated_idx = transition_index(
        &delayed_stats.transition_log,
        DisaggTransition::DestinationActivated { uuid },
    );
    assert!(queued_idx < activated_idx);
}

#[test]
fn destination_missing_timing_uses_isolated_destination_cache_state() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        for (seed_tokens, measured_tokens, expected_missing_ms) in [
            (None, 128, 128.0),
            (Some(64), 128, 64.0),
            (Some(64), 64, 0.0),
        ] {
            let requests = seed_tokens
                .map(|tokens| request(1, tokens, 1, 0.0))
                .into_iter()
                .chain(std::iter::once(request(2, measured_tokens, 2, 1_000.0)))
                .collect::<Vec<_>>();
            let full = run_trace_with_details(
                &transfer_timing_config(engine_type, KvTransferTimingMode::FullPrompt, 1),
                requests.clone(),
                None,
                ReplayRouterMode::RoundRobin,
            );
            let missing = run_trace_with_details(
                &transfer_timing_config(engine_type, KvTransferTimingMode::DestinationMissing, 1),
                requests,
                None,
                ReplayRouterMode::RoundRobin,
            );
            let full_record = full
                .per_request
                .iter()
                .find(|record| {
                    record.input_length == measured_tokens && record.arrival_time_ms > 0.0
                })
                .or_else(|| {
                    full.per_request
                        .iter()
                        .find(|record| record.input_length == measured_tokens)
                })
                .unwrap();
            let missing_record = missing
                .per_request
                .iter()
                .find(|record| {
                    record.input_length == measured_tokens && record.arrival_time_ms > 0.0
                })
                .or_else(|| {
                    missing
                        .per_request
                        .iter()
                        .find(|record| record.input_length == measured_tokens)
                })
                .unwrap();
            let full_transfer_span = full_record.destination_activated_ms.unwrap()
                - full_record.destination_reserved_ms.unwrap();
            let missing_transfer_span = missing_record.destination_activated_ms.unwrap()
                - missing_record.destination_reserved_ms.unwrap();

            let expected_reduction = measured_tokens as f64 - expected_missing_ms;
            assert!(
                ((full_transfer_span - missing_transfer_span) - expected_reduction).abs() < 1e-6,
                "{engine_type:?} seed={seed_tokens:?} measured={measured_tokens}: full span {full_transfer_span}, missing span {missing_transfer_span}"
            );
            assert!(
                missing_transfer_span >= expected_missing_ms
                    && missing_transfer_span - expected_missing_ms < 1.0,
                "{engine_type:?} seed={seed_tokens:?} measured={measured_tokens}: missing span {missing_transfer_span}"
            );
            assert_eq!(full_record.terminal_status, ReplayTerminalStatus::Completed);
            assert_eq!(
                missing_record.terminal_status,
                ReplayTerminalStatus::Completed
            );
            assert!(
                missing_record.source_held_ms.unwrap()
                    <= missing_record.destination_activated_ms.unwrap()
            );
            assert!(
                missing_record.destination_reserved_ms.unwrap()
                    <= missing_record.destination_activated_ms.unwrap()
            );
            assert!(
                missing_record.destination_activated_ms.unwrap()
                    <= missing_record.source_released_ms.unwrap()
            );
        }
    }
}

#[test]
fn source_only_reuse_does_not_reduce_destination_missing_transfer() {
    for engine_type in [EngineType::Vllm, EngineType::Sglang] {
        let report = run_trace_with_details(
            &transfer_timing_config(engine_type, KvTransferTimingMode::DestinationMissing, 2),
            // Two full blocks leave one reusable block after vLLM's required
            // final-block recompute. A one-block prompt correctly reports
            // zero reuse and would not exercise this test's source-hit path.
            vec![request(1, 128, 1, 0.0), request(2, 128, 2, 1_000.0)],
            None,
            ReplayRouterMode::RoundRobin,
        );
        let measured = report
            .per_request
            .iter()
            .find(|record| record.arrival_time_ms > 0.0)
            .unwrap();
        let transfer_span =
            measured.destination_activated_ms.unwrap() - measured.destination_reserved_ms.unwrap();

        assert!(measured.reused_input_tokens > 0);
        assert_eq!(measured.decode_reused_input_tokens, Some(0));
        assert!(transfer_span >= 128.0 && transfer_span - 128.0 < 1.0);
    }
}

#[test]
fn test_cancellation_during_transfer_ignores_retired_completion_event() {
    for mode in [
        KvTransferTimingMode::FullPrompt,
        KvTransferTimingMode::DestinationMissing,
    ] {
        let mut config = disagg_config_with_handoff_delay();
        config.prefill_args.kv_transfer_timing_mode = mode;
        config.decode_args.kv_transfer_timing_mode = mode;
        let uuid = Uuid::from_u128(1);
        let mut runtime = DisaggRuntime::new(
            &config,
            None,
            None,
            VecDeque::from([request(1, 128, 2, 0.0)]),
            ReplayMode::Trace,
            ReplayRouterMode::RoundRobin,
        )
        .unwrap()
        .with_per_request_records(true);

        runtime.drain_current_timestamp().unwrap();
        for _ in 0..16 {
            if runtime.state(uuid).unwrap().phase == DisaggPhase::TransferPending {
                break;
            }
            let next = runtime.next_timestamp().unwrap();
            runtime.advance_now_ms(next);
            runtime.drain_current_timestamp().unwrap();
        }
        assert_eq!(
            runtime.state(uuid).unwrap().phase,
            DisaggPhase::TransferPending
        );
        let handoff_id = runtime.state(uuid).unwrap().handoff_id;
        runtime.apply_scaling(0, 0).unwrap();
        assert_eq!(runtime.total_prefill_count(), 1);
        assert_eq!(runtime.total_decode_count(), 1);
        runtime
            .apply_handoff_fact(uuid, HandoffFact::Canceled { handoff_id })
            .unwrap();

        runtime.drain_current_timestamp().unwrap();
        assert!(runtime.events.iter().all(|event| !matches!(
            &event.kind,
            crate::replay::offline::events::SimulationEventKind::TransferComplete { .. }
        )));
        while !runtime.is_done() {
            let next = runtime.next_timestamp().unwrap();
            runtime.advance_now_ms(next);
            runtime.drain_current_timestamp().unwrap();
        }
        assert_eq!(runtime.state(uuid).unwrap().phase, DisaggPhase::Done);
        assert_eq!(runtime.total_prefill_count(), 0);
        assert_eq!(runtime.total_decode_count(), 0);
        assert!(
            !runtime
                .stats
                .transition_log
                .contains(&DisaggTransition::DestinationActivated { uuid })
        );
        let records = runtime.collector.per_request_records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].terminal_status, ReplayTerminalStatus::Canceled);
    }
}

#[test]
fn test_source_first_handoff_waits_for_decode_scale_up() {
    let config = disagg_config();
    let uuid = Uuid::from_u128(1);
    for router_mode in [ReplayRouterMode::RoundRobin, ReplayRouterMode::KvRouter] {
        let mut runtime = DisaggRuntime::new(
            &config,
            (router_mode == ReplayRouterMode::KvRouter).then(router_config),
            None,
            VecDeque::from([request(1, 128, 2, 0.0)]),
            ReplayMode::Trace,
            router_mode,
        )
        .unwrap();

        runtime.drain_current_timestamp().unwrap();
        runtime.apply_scaling(1, 0).unwrap();
        assert_eq!(runtime.total_decode_count(), 0);

        for _ in 0..16 {
            if !runtime.flow.action_queues.waiting_decode.is_empty() {
                break;
            }
            let next = runtime.next_timestamp().unwrap();
            runtime.advance_now_ms(next);
            runtime.drain_current_timestamp().unwrap();
        }
        assert_eq!(runtime.flow.action_queues.waiting_decode.len(), 1);
        assert!(!runtime.state(uuid).unwrap().coordinator.is_complete());

        let wait_until = runtime.now_ms() + 100.0;
        assert!(!runtime.advance_to(wait_until).unwrap());
        assert_eq!(runtime.now_ms(), wait_until);

        runtime.apply_scaling(1, 1).unwrap();
        let (_, stats) = runtime.run().unwrap();
        assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
    }
}

#[test]
fn source_first_workload_stays_compact_until_prefill_worker_submission() {
    let mut config = disagg_config();
    config.num_prefill_workers = 1;
    let trace = Trace {
        block_size: 64,
        sessions: vec![SessionTrace {
            session_id: "compact-prefill".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 128,
                max_output_tokens: 4,
                hash_ids: vec![31, 32],
                ..Default::default()
            }],
        }],
    };
    let driver = WorkloadDriver::new_trace(trace, 64).unwrap();
    let mut runtime = DisaggRuntime::new_workload(
        &config,
        Some(router_config()),
        None,
        driver,
        ReplayMode::Trace,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();
    runtime.apply_scaling(0, config.num_decode_workers).unwrap();

    assert!(runtime.release_ready_arrivals().unwrap());
    assert!(!runtime.drive_pending_actions().unwrap());
    let uuid = *runtime.flow.requests.keys().next().unwrap();
    assert!(
        runtime
            .state(uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_none()
    );

    runtime.apply_scaling(1, config.num_decode_workers).unwrap();
    assert!(runtime.drive_pending_actions().unwrap());

    assert_eq!(runtime.state(uuid).unwrap().input_length().unwrap(), 128);
    assert!(
        runtime
            .state(uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_some()
    );
}

#[test]
fn destination_first_workload_materializes_for_decode_reservation_then_completes() {
    let mut config = sglang_disagg_config();
    config.num_prefill_workers = 1;
    config.num_decode_workers = 1;
    let trace = Trace {
        block_size: 64,
        sessions: vec![SessionTrace {
            session_id: "destination-first-compact".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 128,
                max_output_tokens: 4,
                hash_ids: vec![61, 62],
                ..Default::default()
            }],
        }],
    };
    let driver = WorkloadDriver::new_trace(trace, 64).unwrap();
    let mut runtime = DisaggRuntime::new_workload(
        &config,
        Some(router_config()),
        None,
        driver,
        ReplayMode::Trace,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();
    runtime.apply_scaling(0, 0).unwrap();

    assert!(runtime.release_ready_arrivals().unwrap());
    assert!(!runtime.drive_pending_actions().unwrap());
    let uuid = *runtime.flow.requests.keys().next().unwrap();
    assert_eq!(runtime.state(uuid).unwrap().input_length().unwrap(), 128);
    assert!(
        runtime
            .state(uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_none()
    );

    runtime.apply_scaling(0, 1).unwrap();
    assert!(runtime.drive_pending_actions().unwrap());
    assert!(
        runtime
            .state(uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_some(),
        "SGLang destination-first routing currently materializes before prefill"
    );

    runtime.apply_scaling(1, 1).unwrap();
    let (_, stats) = runtime.run().unwrap();

    assert_eq!(stats.request_snapshots[&uuid].phase, DisaggPhase::Done);
}

#[test]
fn canceling_worker_waiting_compact_prefill_drops_deferred_prompt() {
    let mut config = disagg_config();
    config.num_prefill_workers = 1;
    let trace = Trace {
        block_size: 64,
        sessions: vec![SessionTrace {
            session_id: "cancel-compact-prefill".to_string(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 128,
                max_output_tokens: 4,
                hash_ids: vec![71, 72],
                ..Default::default()
            }],
        }],
    };
    let driver = WorkloadDriver::new_trace(trace, 64).unwrap();
    let mut runtime = DisaggRuntime::new_workload(
        &config,
        Some(router_config()),
        None,
        driver,
        ReplayMode::Trace,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();
    runtime.apply_scaling(0, config.num_decode_workers).unwrap();

    assert!(runtime.release_ready_arrivals().unwrap());
    assert!(!runtime.drive_pending_actions().unwrap());
    let uuid = *runtime.flow.requests.keys().next().unwrap();
    assert!(
        runtime
            .state(uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_none()
    );

    let handoff_id = runtime.state(uuid).unwrap().handoff_id;
    runtime
        .apply_handoff_fact(uuid, HandoffFact::Canceled { handoff_id })
        .unwrap();
    runtime.drain_current_timestamp().unwrap();

    assert_eq!(runtime.state(uuid).unwrap().phase, DisaggPhase::Done);
    assert!(
        runtime.state(uuid).unwrap().materialized_tokens().is_err(),
        "cancellation should release the deferred request payload"
    );
}

#[test]
fn pending_destination_booking_survives_scale_down_until_cleanup() {
    let first = Uuid::from_u128(1);
    let pending = Uuid::from_u128(2);
    let mut runtime = DisaggRuntime::new(
        &pending_destination_scale_down_config(),
        Some(router_config()),
        None,
        VecDeque::from([request(1, 4, 2, 0.0), request(2, 8, 2, 0.0)]),
        ReplayMode::Concurrency { max_in_flight: 2 },
        ReplayRouterMode::KvRouter,
    )
    .unwrap();

    runtime.drain_current_timestamp().unwrap();
    for _ in 0..32 {
        let accepted = runtime
            .stats
            .transition_log
            .contains(&DisaggTransition::DestinationAccepted { uuid: pending });
        let reserved = runtime
            .stats
            .transition_log
            .contains(&DisaggTransition::DestinationReserved { uuid: pending });
        if accepted && !reserved {
            break;
        }
        let next = runtime.next_timestamp().unwrap();
        runtime.advance_now_ms(next);
        runtime.drain_current_timestamp().unwrap();
    }
    assert!(
        runtime
            .stats
            .transition_log
            .contains(&DisaggTransition::DestinationAccepted { uuid: pending })
    );
    assert!(
        !runtime
            .stats
            .transition_log
            .contains(&DisaggTransition::DestinationReserved { uuid: pending })
    );
    assert_eq!(runtime.state(pending).unwrap().decode_worker_idx(), Some(0));
    let before = runtime.decode_placement.debug_snapshot(runtime.now_ms());
    assert!(
        before
            .active_tokens_by_worker
            .iter()
            .any(|(worker, _)| *worker == 0)
    );

    runtime.apply_scaling(1, 0).unwrap();
    assert_eq!(runtime.active_decode_count(), 0);
    assert_eq!(runtime.total_decode_count(), 1);
    assert_eq!(runtime.state(pending).unwrap().decode_worker_idx(), Some(0));

    let handoff_id = runtime.state(pending).unwrap().handoff_id;
    runtime
        .apply_handoff_fact(pending, HandoffFact::Canceled { handoff_id })
        .unwrap();
    runtime.drain_current_timestamp().unwrap();
    while !runtime.is_done() {
        let next = runtime.next_timestamp().unwrap();
        runtime.advance_now_ms(next);
        runtime.drain_current_timestamp().unwrap();
    }

    assert_eq!(runtime.state(first).unwrap().phase, DisaggPhase::Done);
    assert_eq!(runtime.state(pending).unwrap().phase, DisaggPhase::Done);
    assert_eq!(runtime.total_decode_count(), 0);
    assert_eq!(runtime.stats.decode_router_freed_count, 1);
    let after = runtime.decode_placement.debug_snapshot(runtime.now_ms());
    assert!(after.pending.is_empty());
    assert!(after.active_tokens_by_worker.is_empty());
    assert!(after.active_blocks_by_worker.is_empty());
}

#[test]
fn test_apply_scaling_drains_prefill_router_pending_immediately() {
    let config = scaling_test_disagg_config();
    let mut runtime = DisaggRuntime::new(
        &config,
        Some(planner_router_config()),
        None,
        VecDeque::from([request(1, 64, 8, 0.0), request(2, 64, 8, 0.0)]),
        ReplayMode::Trace,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();

    runtime.advance_to(0.0).unwrap();
    assert_eq!(
        runtime.state(Uuid::from_u128(2)).unwrap().phase,
        DisaggPhase::QueuedPrefill
    );

    runtime.apply_scaling(2, 1).unwrap();

    assert_eq!(
        runtime.state(Uuid::from_u128(2)).unwrap().phase,
        DisaggPhase::RunningPrefill
    );
    assert_eq!(runtime.stats.prefill_assignments[&Uuid::from_u128(2)], 1);
}

#[test]
fn prefill_router_pending_workload_keeps_prompt_compact() {
    let config = scaling_test_disagg_config();
    let trace = Trace {
        block_size: 64,
        sessions: (0..2)
            .map(|session| SessionTrace {
                session_id: format!("compact-router-{session}"),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 128,
                    max_output_tokens: 8,
                    hash_ids: vec![41 + session, 51 + session],
                    ..Default::default()
                }],
            })
            .collect(),
    };
    let driver = WorkloadDriver::new_trace(trace, 64).unwrap();
    let mut runtime = DisaggRuntime::new_workload(
        &config,
        Some(planner_router_config()),
        None,
        driver,
        ReplayMode::Trace,
        ReplayRouterMode::KvRouter,
    )
    .unwrap();

    runtime.advance_to(0.0).unwrap();

    let queued_uuid = runtime
        .flow
        .requests
        .iter()
        .find_map(|(uuid, state)| (state.phase == DisaggPhase::QueuedPrefill).then_some(*uuid))
        .expect("one request should wait in the prefill router");
    assert!(
        runtime
            .state(queued_uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_none()
    );
    let queued_router_request = runtime
        .prefill_placement
        .debug_snapshot(runtime.now_ms())
        .pending
        .into_iter()
        .find(|pending| pending.uuid == queued_uuid)
        .expect("queued request should remain in the prefill router");
    assert_eq!(queued_router_request.expected_output_tokens, Some(1));

    runtime.apply_scaling(2, 1).unwrap();

    assert_eq!(
        runtime.state(queued_uuid).unwrap().phase,
        DisaggPhase::RunningPrefill
    );
    assert!(
        runtime
            .state(queued_uuid)
            .unwrap()
            .materialized_tokens()
            .unwrap()
            .is_some()
    );
}

#[test]
fn test_advance_to_moves_clock_across_idle_gap() {
    let config = disagg_config();
    let mut runtime = DisaggRuntime::new(
        &config,
        None,
        None,
        VecDeque::from([request(1, 64, 2, 1000.0)]),
        ReplayMode::Trace,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap();

    runtime.advance_to(500.0).unwrap();

    assert_eq!(runtime.now_ms(), 500.0);
    let stats = runtime.drain_traffic();
    assert!((stats.duration_s - 0.5).abs() < 1e-9);
}

#[test]
fn test_disagg_traffic_uses_context_capped_output_length() {
    let mut config = disagg_config();
    config.prefill_args.max_model_len = Some(8);
    config.decode_args.max_model_len = Some(8);
    let mut runtime = DisaggRuntime::new(
        &config,
        None,
        None,
        VecDeque::from([request(1, 7, 4, 0.0)]),
        ReplayMode::Trace,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap();

    assert!(runtime.advance_to(1000.0).unwrap());
    let stats = runtime.drain_traffic();
    assert_eq!(stats.num_req, 1);
    assert_eq!(stats.avg_osl, 1.0);
}

/// Setting `max_sim_time_ms` causes `run()` to break before scheduled
/// arrivals past the cap. This test verifies the cap operates on
/// **simulated** time (`now_ms`), not real wall-clock time: with
/// staggered arrivals at 0/1/2/3/4 seconds of sim time and a 2.5s cap,
/// the simulated duration must stay ≤ cap, while the cap-less variant
/// (next test) reaches ≥ 4s of sim duration. Real wall-clock runtime
/// is microseconds in both cases (speedup_ratio=1000).
#[test]
fn test_disagg_max_sim_time_truncates_run() {
    let config = disagg_config();
    let submitted = 5;
    let cap_ms = 2500.0;
    let requests = VecDeque::from([
        request(1, 64, 2, 0.0),
        request(2, 64, 2, 1000.0),
        request(3, 64, 2, 2000.0),
        request(4, 64, 2, 3000.0),
        request(5, 64, 2, 4000.0),
    ]);
    let (collector, _) = DisaggRuntime::new(
        &config,
        None,
        None,
        requests,
        ReplayMode::Trace,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap()
    .with_max_sim_time_ms(Some(cap_ms))
    .run()
    .unwrap();
    let report = collector.finish();
    assert!(
        report.request_counts.num_requests < submitted,
        "cap should admit fewer than {} requests; got num_requests={}",
        submitted,
        report.request_counts.num_requests
    );
    assert!(
        report.throughput.duration_ms <= cap_ms,
        "simulated duration must respect cap; got duration_ms={} cap_ms={}",
        report.throughput.duration_ms,
        cap_ms
    );
}

/// Sanity: without a cap, the same setup admits all submitted requests
/// and the simulated duration extends past the last arrival timestamp.
#[test]
fn test_disagg_no_cap_completes_everything() {
    let config = disagg_config();
    let requests = VecDeque::from([
        request(1, 64, 2, 0.0),
        request(2, 64, 2, 1000.0),
        request(3, 64, 2, 2000.0),
        request(4, 64, 2, 3000.0),
        request(5, 64, 2, 4000.0),
    ]);
    let (collector, _) = DisaggRuntime::new(
        &config,
        None,
        None,
        requests,
        ReplayMode::Trace,
        ReplayRouterMode::RoundRobin,
    )
    .unwrap()
    .run()
    .unwrap();
    let report = collector.finish();
    assert_eq!(report.request_counts.completed_requests, 5);
    assert_eq!(report.request_counts.num_requests, 5);
    assert!(
        report.throughput.duration_ms >= 4000.0,
        "uncapped sim duration should extend past last arrival; got {}",
        report.throughput.duration_ms
    );
}

#[test]
fn test_trace_workload_follow_up_turn_arrives_after_completion_plus_delay() {
    let (collector, _) = run_trace_workload_collect(
        &disagg_config(),
        multiturn_trace(),
        None,
        ReplayRouterMode::RoundRobin,
    );
    let snapshots = collector.snapshots();
    let first_turn = snapshots
        .iter()
        .find(|snapshot| snapshot.input_length == 64)
        .unwrap();
    let second_turn = snapshots
        .iter()
        .find(|snapshot| snapshot.input_length == 192)
        .unwrap();
    let session_b = snapshots
        .iter()
        .find(|snapshot| snapshot.input_length == 128)
        .unwrap();

    assert_eq!(first_turn.arrival_time_ms, 0.0);
    assert_eq!(session_b.arrival_time_ms, 5.0);
    assert!(
        second_turn.arrival_time_ms >= first_turn.last_token_ms.unwrap() + 10.0,
        "follow-up turn should unlock after completion plus delay"
    );
}

#[test]
fn test_concurrency_workload_holds_session_slot_depth_first() {
    let (collector, _) = run_concurrency_workload_collect(
        &disagg_config(),
        multiturn_trace(),
        None,
        1,
        ReplayRouterMode::RoundRobin,
    );
    let mut input_lengths = collector
        .snapshots()
        .into_iter()
        .map(|snapshot| (snapshot.arrival_time_ms, snapshot.input_length))
        .collect::<Vec<_>>();
    input_lengths.sort_by(|left, right| left.0.total_cmp(&right.0));

    assert_eq!(
        input_lengths
            .into_iter()
            .map(|(_, input_length)| input_length)
            .collect::<Vec<_>>(),
        vec![64, 192, 128]
    );
}
