// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aisimulate_core::engine::{Backend, EngineConfig, KvEvent, TimingModelConfig};
use aisimulate_core::replay::loadgen::{SessionTrace, Trace, TurnTrace, WorkloadDriver};
use aisimulate_core::replay::{
    CURRENT_REPLAY_SPEC_VERSION, ProviderSpec, ReplayAdapters, ReplayArtifactKvEventVisibility,
    ReplayArtifacts, ReplayCaptureOptions, ReplayDeterminism, ReplayEngineConfig,
    ReplayEngineFactory, ReplayReport, ReplayRuntimeInput, ReplaySpec, ReplayTopology, Replayer,
    RoundRobinComposition, WorkerPoolSpec,
};

fn spec(backend: Backend, workers: usize, dp_size: u32) -> ReplaySpec {
    let engine = ReplayEngineConfig {
        dp_size,
        rank: EngineConfig {
            num_gpu_blocks: 64,
            block_size: 4,
            max_num_seqs: 4,
            max_num_batched_tokens: 64,
            timing_model: TimingModelConfig::Fixed {
                prefill_ms: 10.0,
                decode_ms: 2.0,
            },
            ..EngineConfig::for_backend(backend)
        },
        ..ReplayEngineConfig::default()
    };
    ReplaySpec {
        version: CURRENT_REPLAY_SPEC_VERSION,
        topology: ReplayTopology::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                startup_delay_ms: 0.0,
            },
        },
        engine: serde_json::to_value(engine).unwrap(),
        adapters: ReplayAdapters {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        },
        max_sim_time_ms: None,
        max_in_flight: None,
        record_per_request: true,
        sla: Default::default(),
        requests: Vec::new(),
    }
}

fn turn(index: usize) -> TurnTrace {
    let first_hash = 11 + u32::try_from(index).unwrap() * 10;
    TurnTrace {
        input_length: 8,
        max_output_tokens: 2,
        hash_ids: vec![first_hash, first_hash + 1],
        ..TurnTrace::default()
    }
}

fn workload(arrivals_ms: &[f64]) -> WorkloadDriver {
    Trace {
        block_size: 4,
        sessions: arrivals_ms
            .iter()
            .enumerate()
            .map(|(index, arrival_ms)| SessionTrace {
                session_id: format!("session-{index}"),
                first_arrival_timestamp_ms: Some(*arrival_ms),
                turns: vec![turn(index)],
            })
            .collect(),
    }
    .into_trace_driver_with_block_size(4)
    .unwrap()
}

fn replayer(spec: ReplaySpec, arrivals_ms: &[f64]) -> Replayer<RoundRobinComposition> {
    Replayer::new(spec, ReplayEngineFactory::new())
        .unwrap()
        .with_runtime_input(ReplayRuntimeInput::Workload(Box::new(workload(
            arrivals_ms,
        ))))
        .with_capture_options(ReplayCaptureOptions {
            determinism: ReplayDeterminism::CanonicalV1,
            ..ReplayCaptureOptions::default()
        })
}

fn capture(
    backend: Backend,
    visibility: ReplayArtifactKvEventVisibility,
) -> (ReplayReport, ReplayArtifacts) {
    replayer(spec(backend, 1, 1), &[0.0])
        .run_with_artifacts(visibility)
        .unwrap()
}

fn kv_parts(artifacts: &ReplayArtifacts) -> (Vec<KvEvent>, Vec<f64>) {
    artifacts
        .kv_events
        .iter()
        .map(|event| (event.event.clone(), event.observed_at_ms))
        .unzip()
}

#[test]
fn common_agg_runtime_captures_requests_outputs_and_the_same_report() {
    let replay_spec = spec(Backend::Vllm, 1, 1);
    let report = replayer(replay_spec.clone(), &[0.0]).run().unwrap();
    let (artifact_report, artifacts) = replayer(replay_spec, &[0.0])
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
    assert_eq!(
        serde_json::to_value(report.clone().with_wall_time_ms(0.0)).unwrap(),
        serde_json::to_value(artifact_report.clone().with_wall_time_ms(0.0)).unwrap()
    );
    assert_eq!(
        serde_json::to_value(report.per_request).unwrap(),
        serde_json::to_value(artifact_report.per_request).unwrap()
    );

    let request = &artifacts.requests[0];
    assert_eq!(
        (
            request.observed_at_ms,
            request.scheduled_ready_at_ms,
            request.input_length,
            request.output_length,
        ),
        (0.0, 0.0, 8, 2)
    );
    assert_eq!(
        request.replay_hashes,
        Some(turn(0).to_replay_hashes(4, 4).unwrap())
    );
    assert_eq!(
        artifacts
            .outputs
            .iter()
            .map(|output| {
                (
                    output.observed_at_ms,
                    output.request_id == request.request_id,
                    output.token_id.is_some(),
                    output.completed,
                    output.rejected,
                    output.cached_tokens,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (12.0, true, true, false, false, Some(0)),
            (14.0, true, true, true, false, None),
        ]
    );
}

#[test]
fn request_arrivals_and_hashes_use_the_agg_event_loop_clock() {
    let (_, artifacts) = replayer(spec(Backend::Vllm, 1, 1), &[0.0, 5.0])
        .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
        .unwrap();
    assert_eq!(
        artifacts
            .requests
            .iter()
            .map(|request| (request.observed_at_ms, request.scheduled_ready_at_ms))
            .collect::<Vec<_>>(),
        vec![(0.0, 0.0), (5.0, 5.0)]
    );
}

#[test]
fn native_and_normalized_kv_visibility_preserve_raw_order() {
    for backend in [Backend::Vllm, Backend::Trtllm, Backend::Sglang] {
        let (_, native) = capture(backend, ReplayArtifactKvEventVisibility::Native);
        let (_, start) = capture(backend, ReplayArtifactKvEventVisibility::PassStart);
        let (_, end) = capture(backend, ReplayArtifactKvEventVisibility::PassEnd);
        let (events, native_times) = kv_parts(&native);
        let (start_events, start_times) = kv_parts(&start);
        let (end_events, end_times) = kv_parts(&end);
        assert!(!events.is_empty());
        assert_eq!(events, start_events);
        assert_eq!(events, end_events);
        assert!(start_times.iter().zip(&end_times).all(|(a, b)| a <= b));
        assert!(start_times.iter().zip(&end_times).any(|(a, b)| a < b));
        assert_eq!(native_times, end_times);
    }
}

#[test]
fn capped_passes_respect_visibility_boundaries() {
    let capped = |visibility| {
        let mut replay_spec = spec(Backend::Vllm, 1, 1);
        replay_spec.max_sim_time_ms = Some(1.0);
        replayer(replay_spec, &[0.0])
            .run_with_artifacts(visibility)
            .unwrap()
            .1
    };
    let native = capped(ReplayArtifactKvEventVisibility::Native);
    let start = capped(ReplayArtifactKvEventVisibility::PassStart);
    let end = capped(ReplayArtifactKvEventVisibility::PassEnd);
    assert!(native.kv_events.is_empty());
    assert!(start.kv_events.is_empty());
    assert!(end.kv_events.is_empty());
}

#[test]
fn artifact_capture_rejects_unsupported_topologies() {
    let error = |replay_spec| {
        replayer(replay_spec, &[0.0])
            .run_with_artifacts(ReplayArtifactKvEventVisibility::Native)
            .unwrap_err()
            .to_string()
    };
    for replay_spec in [spec(Backend::Vllm, 2, 1), spec(Backend::Vllm, 1, 2)] {
        assert!(error(replay_spec).contains("one logical DP1 worker"));
    }
    let mut disagg = spec(Backend::Vllm, 1, 1);
    disagg.topology = ReplayTopology::Disaggregated {
        prefill: WorkerPoolSpec::default(),
        decode: WorkerPoolSpec::default(),
        handoff_latency_ms: 0.0,
    };
    assert!(error(disagg).contains("require aggregated topology"));
}
