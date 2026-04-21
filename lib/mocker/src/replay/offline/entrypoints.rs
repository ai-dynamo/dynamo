// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::WorkerId;

#[cfg(test)]
use super::agg::AggRuntimeStats;
use super::agg::{AggRuntime, ReplayMode as AggReplayMode};
use super::core::ReplayWorkerCore;
#[cfg(test)]
use super::disagg::DisaggRuntimeStats;
use super::disagg::{DisaggRuntime, ReplayMode as DisaggReplayMode};
use super::normalize_trace_requests;
use super::single::{SingleReplayMode, SingleRuntime};
use crate::common::protocols::{DirectRequest, EngineType, MockEngineArgs};
use crate::loadgen::{Trace, WorkloadDriver};
use crate::replay::OfflineDisaggReplayConfig;
use crate::replay::{
    ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayTimedKvEvent, ReplayTimedOutputSignal,
    ReplayTimedRequest, ReplayWorkerArtifacts, TraceCollector, TraceSimulationReport,
};
use crate::scheduler::RouterEventVisibility;

fn timestamp_us_from_ms(timestamp_ms: f64) -> u64 {
    if !timestamp_ms.is_finite() || timestamp_ms <= 0.0 {
        return 0;
    }

    (timestamp_ms * 1000.0) as u64
}

pub(crate) fn generate_trace_worker_artifacts(
    args: MockEngineArgs,
    trace: Trace,
) -> Result<ReplayWorkerArtifacts> {
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let mut worker = ReplayWorkerCore::new_with_kv_capture(args, WorkerId::default());
    let mut driver = trace.into_trace_driver_with_block_size(engine_block_size)?;
    let mut collector = TraceCollector::default();
    let mut artifacts = ReplayWorkerArtifacts::default();
    let mut current_time_ms = 0.0;

    while !driver.is_drained() || !worker.is_empty() {
        for ready_turn in driver.pop_ready(current_time_ms, usize::MAX) {
            let replay_hashes = ready_turn
                .replay_hashes
                .ok_or_else(|| anyhow::anyhow!("offline artifacts require synthesized hashes"))?;
            collector.on_arrival(
                ready_turn.request_uuid,
                ready_turn.scheduled_ready_at_ms,
                ready_turn.request.tokens.len(),
                ready_turn.request.max_output_tokens,
            );
            artifacts.requests.push(ReplayTimedRequest {
                uuid: ready_turn.request_uuid,
                timestamp_us: timestamp_us_from_ms(current_time_ms),
                scheduled_ready_at_ms: ready_turn.scheduled_ready_at_ms,
                input_length: ready_turn.request.tokens.len(),
                output_length: ready_turn.request.max_output_tokens,
                replay_hashes,
            });
            worker.receive(ready_turn.request);
        }

        if worker.is_empty() {
            let Some(next_ready_ms) = driver.next_ready_time_ms() else {
                break;
            };
            current_time_ms = next_ready_ms;
            continue;
        }

        let pass_start_ms = current_time_ms;
        let pass = worker.execute_pass(&mut collector, current_time_ms);
        current_time_ms = pass.end_ms;

        let kv_event_timestamp_us = match pass.router_event_visibility {
            RouterEventVisibility::PassStart => timestamp_us_from_ms(pass_start_ms),
            RouterEventVisibility::PassEnd => timestamp_us_from_ms(current_time_ms),
        };
        artifacts
            .kv_events
            .extend(pass.kv_events.into_iter().map(|event| ReplayTimedKvEvent {
                event: event.event,
                timestamp_us: kv_event_timestamp_us,
            }));

        let output_timestamp_us = timestamp_us_from_ms(current_time_ms);
        for signal in pass.output_signals {
            if signal.completed {
                driver.on_complete(signal.uuid, current_time_ms)?;
            }
            artifacts.output_signals.push(ReplayTimedOutputSignal {
                signal,
                timestamp_us: output_timestamp_us,
            });
        }
    }

    Ok(artifacts)
}

pub(crate) fn simulate_trace(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_trace_single(args, requests, arrival_speedup_ratio)
    } else {
        simulate_trace_multi(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            num_workers,
            arrival_speedup_ratio,
            router_mode,
        )
    }
}

pub(crate) fn simulate_concurrency(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_concurrency_single(args, requests, max_in_flight)
    } else {
        simulate_concurrency_multi(
            args,
            router_config,
            prefill_load_estimator,
            requests,
            max_in_flight,
            num_workers,
            router_mode,
        )
    }
}

pub(crate) fn simulate_trace_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_trace_workload_single(args, trace)
    } else {
        simulate_trace_workload_multi(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
        )
    }
}

pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == EngineType::Vllm {
        simulate_concurrency_workload_single(args, trace, max_in_flight)
    } else {
        simulate_concurrency_workload_multi(
            args,
            router_config,
            prefill_load_estimator,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
        )
    }
}

pub(crate) fn simulate_trace_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = DisaggRuntime::new(
        &config,
        router_config,
        prefill_load_estimator,
        pending,
        DisaggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let pending = VecDeque::from(requests);
    let (collector, _) = DisaggRuntime::new(
        &config,
        router_config,
        prefill_load_estimator,
        pending,
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let driver = WorkloadDriver::new_trace(trace, config.prefill_args.block_size)?;
    let (collector, _) = DisaggRuntime::new_workload(
        &config,
        router_config,
        prefill_load_estimator,
        driver,
        DisaggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let driver = WorkloadDriver::new_concurrency(trace, config.prefill_args.block_size)?;
    let (collector, _) = DisaggRuntime::new_workload(
        &config,
        router_config,
        prefill_load_estimator,
        driver,
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let collector = SingleRuntime::new(args, pending, SingleReplayMode::Trace).run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let collector = SingleRuntime::new(
        args,
        pending,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_workload_single(
    args: MockEngineArgs,
    trace: Trace,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let collector = SingleRuntime::new_workload(
        args,
        trace.into_trace_driver_with_block_size(engine_block_size)?,
        SingleReplayMode::Trace,
    )
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_workload_single(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let collector = SingleRuntime::new_workload(
        args,
        trace.into_concurrency_driver_with_block_size(engine_block_size)?,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio)?;
    let (collector, _) = AggRuntime::new(
        &args,
        router_config,
        prefill_load_estimator,
        pending,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let pending = VecDeque::from(requests);
    let (collector, _) = AggRuntime::new(
        &args,
        router_config,
        prefill_load_estimator,
        pending,
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_trace_workload_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let (collector, _) = AggRuntime::new_workload(
        &args,
        router_config,
        prefill_load_estimator,
        trace.into_trace_driver_with_block_size(args.block_size)?,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_workload_multi(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    let (collector, _) = AggRuntime::new_workload(
        &args,
        router_config,
        prefill_load_estimator,
        trace.into_concurrency_driver_with_block_size(args.block_size)?,
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .run()?;
    Ok(collector.finish())
}

#[cfg(test)]
pub(super) fn run_trace_single_collect(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> TraceCollector {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio).unwrap();
    SingleRuntime::new(args, pending, SingleReplayMode::Trace)
        .run()
        .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_single_collect(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
) -> TraceCollector {
    SingleRuntime::new(
        args,
        VecDeque::from(requests),
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_workload_single_collect(
    args: MockEngineArgs,
    trace: Trace,
) -> TraceCollector {
    let engine_block_size = args.block_size;
    SingleRuntime::new_workload(
        args,
        trace
            .into_trace_driver_with_block_size(engine_block_size)
            .unwrap(),
        SingleReplayMode::Trace,
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_workload_single_collect(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
) -> TraceCollector {
    let engine_block_size = args.block_size;
    SingleRuntime::new_workload(
        args,
        trace
            .into_concurrency_driver_with_block_size(engine_block_size)
            .unwrap(),
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_multi_collect_with_stats(
    args: &MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    let pending = normalize_trace_requests(requests, 1.0).unwrap();
    AggRuntime::new(
        args,
        None,
        None,
        pending,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_multi_collect_with_stats(
    args: &MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new(
        args,
        None,
        None,
        VecDeque::from(requests),
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_workload_multi_collect_with_stats(
    args: &MockEngineArgs,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new_workload(
        args,
        None,
        None,
        trace
            .into_trace_driver_with_block_size(args.block_size)
            .unwrap(),
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_workload_multi_collect_with_stats(
    args: &MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, AggRuntimeStats) {
    AggRuntime::new_workload(
        args,
        None,
        None,
        trace
            .into_concurrency_driver_with_block_size(args.block_size)
            .unwrap(),
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_collect(
    config: &OfflineDisaggReplayConfig,
    requests: Vec<DirectRequest>,
    router_config: Option<KvRouterConfig>,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    let pending = normalize_trace_requests(requests, arrival_speedup_ratio).unwrap();
    DisaggRuntime::new(
        config,
        router_config,
        None,
        pending,
        DisaggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_collect(
    config: &OfflineDisaggReplayConfig,
    requests: Vec<DirectRequest>,
    router_config: Option<KvRouterConfig>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new(
        config,
        router_config,
        None,
        VecDeque::from(requests),
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_trace_workload_collect(
    config: &OfflineDisaggReplayConfig,
    trace: Trace,
    router_config: Option<KvRouterConfig>,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new_workload(
        config,
        router_config,
        None,
        trace
            .into_trace_driver_with_block_size(config.prefill_args.block_size)
            .unwrap(),
        DisaggReplayMode::Trace,
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
pub(super) fn run_concurrency_workload_collect(
    config: &OfflineDisaggReplayConfig,
    trace: Trace,
    router_config: Option<KvRouterConfig>,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> (TraceCollector, DisaggRuntimeStats) {
    DisaggRuntime::new_workload(
        config,
        router_config,
        None,
        trace
            .into_concurrency_driver_with_block_size(config.prefill_args.block_size)
            .unwrap(),
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )
    .unwrap()
    .run()
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::generate_trace_worker_artifacts;
    use crate::common::protocols::MockEngineArgs;
    use crate::loadgen::{SessionTrace, Trace, TurnTrace};

    #[test]
    fn test_generate_trace_worker_artifacts_emits_monotonic_event_timestamps() {
        let args = MockEngineArgs::builder()
            .block_size(2)
            .num_gpu_blocks(1024)
            .max_num_batched_tokens(None)
            .max_num_seqs(None)
            .enable_prefix_caching(true)
            .speedup_ratio(1000.0)
            .build()
            .unwrap();
        let trace = Trace {
            block_size: 2,
            sessions: vec![SessionTrace {
                session_id: "session-a".to_string(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![1, 2],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 4,
                        max_output_tokens: 2,
                        hash_ids: vec![3, 4],
                        delay_after_previous_ms: 5.0,
                    },
                ],
            }],
        };

        let artifacts = generate_trace_worker_artifacts(args, trace).unwrap();

        assert_eq!(artifacts.requests.len(), 2);
        assert!(!artifacts.kv_events.is_empty());
        assert!(
            artifacts
                .kv_events
                .windows(2)
                .all(|events| events[0].timestamp_us <= events[1].timestamp_us)
        );

        let first_uuid = artifacts.requests[0].uuid;
        let first_completion_ms = artifacts
            .output_signals
            .iter()
            .find(|signal| signal.signal.uuid == first_uuid && signal.signal.completed)
            .expect("first request must complete")
            .timestamp_us as f64
            / 1000.0;
        assert!(
            artifacts.requests[1].scheduled_ready_at_ms + 0.1 >= first_completion_ms + 5.0,
            "expected second request to wait for completion plus delay"
        );
    }
}
