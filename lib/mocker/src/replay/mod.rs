// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod artifacts;
mod entrypoints;
pub(crate) mod offline;
mod online;
mod router_shared;
mod validate;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use dynamo_kv_router::PrefillLoadEstimator;

/// Backward-compatible Dynamo Mocker name for [`aisimulate_core::ReplayReport`].
pub use aisimulate_core::ReplayReport as TraceSimulationReport;
pub(crate) use aisimulate_core::replay::TraceCollector;
pub use aisimulate_core::replay::{
    CanonicalReplayCoverage, CanonicalReplayRecord, LifecycleOperation, OfflineRuntimeEvidence,
    PerRequestRecord, ReplayCaptureOptions, ReplayDeterminism, ReplayTerminalStatus, SlaThresholds,
    TraceDistributionStats, TraceGoodputStats, TraceInterTokenLatencyStats, TraceLatencyStats,
    TraceRequestCounts, TraceThroughputStats,
};
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use artifacts::native_g1_parent_chain_artifact;
pub use artifacts::{
    ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest, ReplayWorkerArtifacts,
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

pub use aisimulate_core::replay::TrafficStats;
pub use aisimulate_core::replay::{
    ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot,
};
pub use entrypoints::{
    ReplayKvEventVisibility, generate_trace_worker_artifacts_offline,
    generate_trace_worker_artifacts_offline_with_kv_event_visibility,
    simulate_agentic_trace_live_workload_with_router_mode_and_options,
    simulate_agentic_trace_workload_disagg_with_router_mode,
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
    simulate_loaded_trace_disagg_with_router_mode_and_capture_options,
    simulate_loaded_trace_disagg_with_router_mode_and_options,
    simulate_loaded_trace_disagg_with_router_mode_and_options_and_scaling_policy,
    simulate_loaded_trace_live_with_router_mode,
    simulate_loaded_trace_live_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_capture_options,
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
#[doc(hidden)]
pub use offline::run_offline_handoff_conformance;
pub use validate::validate_replay_args_mode;

pub(crate) fn normalize_trace_requests(
    requests: Vec<DirectRequest>,
    arrival_speedup_ratio: f64,
) -> anyhow::Result<VecDeque<DirectRequest>> {
    if !arrival_speedup_ratio.is_finite() || arrival_speedup_ratio <= 0.0 {
        anyhow::bail!(
            "arrival_speedup_ratio must be a finite positive number, got {arrival_speedup_ratio}"
        );
    }

    // Lift every arrival timestamp out before sorting. `arrival_timestamp_ms` is an
    // `Option<f64>` on a `Deserialize` DTO, so a caller-supplied trace can carry a missing or
    // non-finite value; neither the comparator nor the rebasing pass below can report an error,
    // and a NaN would additionally sort first under `total_cmp` and poison `first_arrival_ms`
    // for every request.
    let mut timestamped = requests
        .into_iter()
        .enumerate()
        .map(|(index, request)| match request.arrival_timestamp_ms {
            Some(arrival_timestamp_ms) if arrival_timestamp_ms.is_finite() => {
                Ok((arrival_timestamp_ms, request))
            }
            Some(arrival_timestamp_ms) => anyhow::bail!(
                "trace replay request {index} has a non-finite arrival timestamp, got \
                 {arrival_timestamp_ms}"
            ),
            None => anyhow::bail!("trace replay request {index} is missing an arrival timestamp"),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Stable sort: coincident arrivals keep their authored order.
    timestamped.sort_by(|(left_ts, _), (right_ts, _)| left_ts.total_cmp(right_ts));

    let first_arrival_ms = timestamped
        .first()
        .map(|(arrival_timestamp_ms, _)| *arrival_timestamp_ms)
        .ok_or_else(|| anyhow::anyhow!("trace replay requires at least one timestamped request"))?;

    Ok(timestamped
        .into_iter()
        .map(|(arrival_timestamp_ms, mut request)| {
            request.arrival_timestamp_ms =
                Some((arrival_timestamp_ms - first_arrival_ms) / arrival_speedup_ratio);
            request
        })
        .collect::<VecDeque<_>>())
}

pub(crate) fn effective_agentic_lanes(
    requested_lanes: Option<usize>,
    play_count: usize,
) -> Option<usize> {
    requested_lanes.map(|lane_count| lane_count.min(play_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

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

    fn request_at(arrival_timestamp_ms: Option<f64>, id: u128) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; 4],
            max_output_tokens: 1,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(id)),
            dp_rank: 0,
            arrival_timestamp_ms,
            ..Default::default()
        }
    }

    #[test]
    fn test_normalize_trace_requests_rejects_missing_arrival_timestamp() {
        let error =
            normalize_trace_requests(vec![request_at(Some(0.0), 1), request_at(None, 2)], 1.0)
                .expect_err("a request without an arrival timestamp must be rejected, not panic");
        assert_eq!(
            error.to_string(),
            "trace replay request 1 is missing an arrival timestamp"
        );
    }

    #[test]
    fn test_normalize_trace_requests_rejects_non_finite_arrival_timestamp() {
        // A NaN arrival sorts first under `total_cmp`, so without validation it would silently
        // become `first_arrival_ms` and poison every rebased arrival in the trace.
        let error = normalize_trace_requests(
            vec![request_at(Some(10.0), 1), request_at(Some(f64::NAN), 2)],
            1.0,
        )
        .expect_err("a non-finite arrival timestamp must be rejected");
        assert_eq!(
            error.to_string(),
            "trace replay request 1 has a non-finite arrival timestamp, got NaN"
        );

        let error = normalize_trace_requests(vec![request_at(Some(f64::INFINITY), 1)], 1.0)
            .expect_err("an infinite arrival timestamp must be rejected");
        assert_eq!(
            error.to_string(),
            "trace replay request 0 has a non-finite arrival timestamp, got inf"
        );
    }

    #[test]
    fn test_effective_agentic_lanes_are_bounded_by_play_count() {
        assert_eq!(effective_agentic_lanes(None, 3), None);
        assert_eq!(effective_agentic_lanes(Some(2), 3), Some(2));
        assert_eq!(effective_agentic_lanes(Some(usize::MAX), 3), Some(3));
    }
}
