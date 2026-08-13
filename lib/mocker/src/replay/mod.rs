// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod artifacts;
mod collector;
mod entrypoints;
pub(crate) mod offline;
mod online;
mod router_shared;
mod validate;

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use dynamo_kv_router::PrefillLoadEstimator;

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use artifacts::native_g1_parent_chain_artifact;
pub use artifacts::{
    ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest, ReplayWorkerArtifacts,
};
pub(crate) use collector::TraceCollector;
#[cfg(test)]
pub(crate) use collector::TraceRequestStatsSnapshot;
pub use collector::{
    PerRequestAdmissionRecord, PerRequestRecord, PerRequestRoutingRecord, ReplayRequestPool,
    ReplayRoutingOutcome, ReplayTerminalStatus, ReplayWorkerLifecycleStatus, SlaThresholds,
    TraceAccountingReconciliation, TraceDistributionStats, TraceGlobalTopologyAccounting,
    TraceGoodputStats, TraceInterTokenLatencyStats, TraceLatencyStats, TracePoolAccounting,
    TraceRequestCounts, TraceSimulationReport, TraceTerminalCounts, TraceThroughputStats,
    TraceTokenStatusCounts, TraceTopologyAccounting, TraceWorkerAccounting,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayRouterMode {
    RoundRobin,
    KvRouter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayArgsMode {
    Aggregated,
    Disagg,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplayDeterminism {
    #[default]
    Random,
    CanonicalV1,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayCaptureOptions {
    pub capture_per_request: bool,
    pub capture_planner_details: bool,
    pub capture_canonical_evidence: bool,
    pub determinism: ReplayDeterminism,
}

impl ReplayCaptureOptions {
    pub fn effective_per_request(self) -> bool {
        self.capture_per_request || self.capture_canonical_evidence
    }
}

thread_local! {
    static REPLAY_DETERMINISM: Cell<ReplayDeterminism> =
        const { Cell::new(ReplayDeterminism::Random) };
}

pub fn with_replay_determinism<T>(determinism: ReplayDeterminism, run: impl FnOnce() -> T) -> T {
    struct Reset(ReplayDeterminism);
    impl Drop for Reset {
        fn drop(&mut self) {
            REPLAY_DETERMINISM.with(|current| current.set(self.0));
        }
    }

    let previous = REPLAY_DETERMINISM.with(|current| current.replace(determinism));
    let _reset = Reset(previous);
    run()
}

#[cfg(feature = "replay-bench")]
pub(crate) fn canonical_replay_active() -> bool {
    REPLAY_DETERMINISM.with(|current| current.get() == ReplayDeterminism::CanonicalV1)
}

pub type ReplayPrefillLoadEstimator = Arc<dyn PrefillLoadEstimator>;

#[derive(Clone, Debug)]
pub struct OfflineDisaggReplayConfig {
    pub prefill_args: MockEngineArgs,
    pub decode_args: MockEngineArgs,
    pub num_prefill_workers: usize,
    pub num_decode_workers: usize,
}

impl OfflineDisaggReplayConfig {
    pub fn normalized(self) -> anyhow::Result<Self> {
        Ok(Self {
            prefill_args: self.prefill_args.normalized()?,
            decode_args: self.decode_args.normalized()?,
            num_prefill_workers: self.num_prefill_workers,
            num_decode_workers: self.num_decode_workers,
        })
    }
}

pub use entrypoints::{
    ReplayKvEventVisibility, generate_trace_worker_artifacts_offline,
    generate_trace_worker_artifacts_offline_with_kv_event_visibility,
    simulate_agentic_trace_live_workload_with_router_mode_and_options,
    simulate_agentic_trace_workload_with_router_mode, simulate_concurrency_file,
    simulate_concurrency_file_disagg_with_router_mode,
    simulate_concurrency_file_disagg_with_router_mode_and_format,
    simulate_concurrency_file_disagg_with_router_mode_and_format_and_scaling_policy,
    simulate_concurrency_file_with_router_mode,
    simulate_concurrency_file_with_router_mode_and_format,
    simulate_concurrency_file_with_router_mode_and_format_and_scaling_policy,
    simulate_concurrency_live_file, simulate_concurrency_live_file_with_router_mode,
    simulate_concurrency_live_file_with_router_mode_and_format,
    simulate_concurrency_live_file_with_router_mode_and_format_and_options,
    simulate_concurrency_live_requests, simulate_concurrency_live_requests_with_router_mode,
    simulate_concurrency_live_requests_with_router_mode_and_options,
    simulate_concurrency_live_workload, simulate_concurrency_live_workload_with_router_mode,
    simulate_concurrency_live_workload_with_router_mode_and_options, simulate_concurrency_requests,
    simulate_concurrency_requests_disagg_with_router_mode,
    simulate_concurrency_requests_disagg_with_router_mode_and_scaling_policy,
    simulate_concurrency_requests_with_router_mode,
    simulate_concurrency_requests_with_router_mode_and_scaling_policy,
    simulate_concurrency_workload, simulate_concurrency_workload_disagg_with_router_mode,
    simulate_concurrency_workload_disagg_with_router_mode_and_options,
    simulate_concurrency_workload_disagg_with_router_mode_and_options_and_scaling_policy,
    simulate_concurrency_workload_with_router_mode,
    simulate_concurrency_workload_with_router_mode_and_options,
    simulate_concurrency_workload_with_router_mode_and_options_and_scaling_policy,
    simulate_loaded_trace_disagg_with_router_mode_and_options,
    simulate_loaded_trace_disagg_with_router_mode_and_options_and_scaling_policy,
    simulate_loaded_trace_live_with_router_mode,
    simulate_loaded_trace_live_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_options_and_scaling_policy, simulate_trace_file,
    simulate_trace_file_disagg_with_router_mode,
    simulate_trace_file_disagg_with_router_mode_and_format,
    simulate_trace_file_disagg_with_router_mode_and_format_and_scaling_policy,
    simulate_trace_file_with_router_mode, simulate_trace_file_with_router_mode_and_format,
    simulate_trace_file_with_router_mode_and_format_and_scaling_policy, simulate_trace_live_file,
    simulate_trace_live_file_with_router_mode,
    simulate_trace_live_file_with_router_mode_and_format,
    simulate_trace_live_file_with_router_mode_and_format_and_options, simulate_trace_live_requests,
    simulate_trace_live_requests_with_router_mode,
    simulate_trace_live_requests_with_router_mode_and_options, simulate_trace_live_workload,
    simulate_trace_live_workload_with_router_mode,
    simulate_trace_live_workload_with_router_mode_and_options, simulate_trace_requests,
    simulate_trace_requests_disagg_with_router_mode,
    simulate_trace_requests_disagg_with_router_mode_and_scaling_policy,
    simulate_trace_requests_with_router_mode,
    simulate_trace_requests_with_router_mode_and_scaling_policy, simulate_trace_workload,
    simulate_trace_workload_disagg_with_router_mode,
    simulate_trace_workload_disagg_with_router_mode_and_options_and_scaling_policy,
    simulate_trace_workload_with_router_mode,
    simulate_trace_workload_with_router_mode_and_options_and_scaling_policy,
};
pub use offline::components::TrafficStats;
#[doc(hidden)]
pub use offline::run_offline_handoff_conformance;
pub use offline::scaling::{ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot};
pub use offline::{
    CANONICAL_RESULT_EXCLUSIONS, CANONICAL_SCHEMA_VERSION, CanonicalAicIdentity,
    CanonicalAicImplementation, CanonicalDeterminismMetadata, CanonicalEngineConfig,
    CanonicalExecutionMetadata, CanonicalReplayCoverage, CanonicalReplayMetadata,
    CanonicalReplayRecord, CanonicalReplayRouterMode, CanonicalReplayTopology,
    CanonicalRouterMetadata, CanonicalSemanticFeatures, CanonicalSlaMetadata,
    CanonicalSyntheticSpec, CanonicalWorkloadMetadata, ControlledReplayOptions,
    EnginePressureState, KvIngestBoundary, KvIngestBoundaryStats, KvIngestEvidence,
    LifecycleOperation, OfflineReplaySession, OfflineRuntimeEvidence, PoolRouter, PoolSpec,
    PressureEvidence, PressureKind, PressureRecord, ReplayAgenticRequest, ReplayAgenticWorkflow,
    ReplayEvent, ReplayEventData, ReplayPendingPlacement, ReplayPlacementCandidate,
    ReplayRequestSpec, ReplayRoutingConstraints, ReplaySessionOptions, ReplaySessionRouter,
    ReplaySnapshot, ReplayStepStatus, ReplayWorkSource, ReplayWorkSourceContext,
    ReplayWorkSubmission, ReplayWorkerSnapshot, WorkerLifecycleTransition,
    WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolState, WorkerSpec, WorkerTarget,
    canonical_engine_pool_metadata, canonical_router_metadata, canonical_topology,
    simulate_controlled_aggregated, simulate_controlled_aggregated_kv_router_with_options,
    simulate_controlled_aggregated_with_options, with_runtime_evidence,
};
#[doc(hidden)]
pub use offline::{CapturedReplayEvent, CapturedReplayEventData, CapturedReplayEventDataView};
pub use validate::validate_replay_args_mode;

pub(crate) fn normalize_trace_requests(
    mut requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> anyhow::Result<VecDeque<DirectRequest>> {
    if !arrival_speedup_ratio.is_finite() || arrival_speedup_ratio <= 0.0 {
        anyhow::bail!(
            "arrival_speedup_ratio must be a finite positive number, got {arrival_speedup_ratio}"
        );
    }

    requests.sort_by(|left, right| {
        let left_ts = left
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        let right_ts = right
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        left_ts.total_cmp(&right_ts)
    });

    let first_arrival_ms = requests
        .first()
        .and_then(|request| request.arrival_timestamp_ms)
        .ok_or_else(|| anyhow::anyhow!("trace replay requires at least one timestamped request"))?;

    Ok(VecDeque::from(
        requests
            .into_iter()
            .map(|mut request| {
                let arrival_timestamp_ms = request
                    .arrival_timestamp_ms
                    .expect("trace replay requests must have an arrival timestamp")
                    - first_arrival_ms;
                let arrival_timestamp_ms = arrival_timestamp_ms / arrival_speedup_ratio;
                request.arrival_timestamp_ms = Some(arrival_timestamp_ms);
                request
            })
            .collect::<Vec<_>>(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_replay_itl_uses_per_token_gaps() {
        fn assert_ddsketch_relative_error(actual: f64, expected: f64) {
            assert!((actual - expected).abs() <= expected.abs() * 0.001 + f64::EPSILON);
        }

        let mut collector = TraceCollector::default();
        let uuid = Uuid::from_u128(11);

        collector.on_arrival(uuid, 0.0, 4, 4);
        collector.on_admit(uuid, 0.0, 0);
        collector.on_token(uuid, 10.0);
        collector.on_token(uuid, 11.0);
        collector.on_token(uuid, 12.0);
        collector.on_token(uuid, 110.0);
        collector.on_terminal(uuid, 110.0, ReplayTerminalStatus::Completed);

        let report = collector.finish();

        assert!((report.latency.tpot.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert!((report.latency.itl.distribution.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert_ddsketch_relative_error(report.latency.itl.distribution.median_ms, 1.0);
        assert_ddsketch_relative_error(report.latency.itl.distribution.p75_ms, 98.0);
        assert_ddsketch_relative_error(report.latency.itl.distribution.p90_ms, 98.0);
        assert_ddsketch_relative_error(report.latency.itl.distribution.p95_ms, 98.0);
        assert_eq!(report.latency.itl.max_ms, 98.0);
        assert_eq!(report.latency.ttst.min_ms, 1.0);
        assert_eq!(report.latency.ttst.max_ms, 1.0);
        assert_eq!(
            report.latency.output_token_throughput_per_user.min_ms,
            1000.0 / 98.0
        );
        assert_eq!(
            report.latency.output_token_throughput_per_user.max_ms,
            1000.0
        );
    }

    #[test]
    fn test_normalize_trace_requests_applies_arrival_speedup_ratio() {
        let requests = vec![
            DirectRequest {
                tokens: vec![1; 4],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![2; 4],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(200.0),
                ..Default::default()
            },
        ];

        let normalized = normalize_trace_requests(requests, 10.0).unwrap();
        let arrivals = normalized
            .into_iter()
            .map(|request| request.arrival_timestamp_ms.unwrap())
            .collect::<Vec<_>>();

        assert_eq!(arrivals, vec![0.0, 10.0]);
    }
}
