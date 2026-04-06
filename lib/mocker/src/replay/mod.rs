// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod artifacts;
mod collector;
mod entrypoints;
pub(crate) mod offline;
mod online;
mod router;
mod validate;

use std::collections::VecDeque;

use crate::common::protocols::{DirectRequest, MockEngineArgs};

pub use artifacts::{
    ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest, ReplayWorkerArtifacts,
};
pub(crate) use collector::TraceCollector;
#[cfg(test)]
pub(crate) use collector::TraceRequestStatsSnapshot;
pub use collector::{
    TraceDistributionStats, TraceInterTokenLatencyStats, TraceLatencyStats, TraceRequestCounts,
    TraceSimulationReport, TraceThroughputStats,
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
    generate_trace_worker_artifacts_offline, simulate_concurrency_file,
    simulate_concurrency_file_disagg_with_router_mode, simulate_concurrency_file_with_router_mode,
    simulate_concurrency_live_file, simulate_concurrency_live_file_with_router_mode,
    simulate_concurrency_live_requests, simulate_concurrency_live_requests_with_router_mode,
    simulate_concurrency_live_workload, simulate_concurrency_live_workload_with_router_mode,
    simulate_concurrency_requests, simulate_concurrency_requests_disagg_with_router_mode,
    simulate_concurrency_requests_with_router_mode, simulate_concurrency_workload,
    simulate_concurrency_workload_disagg_with_router_mode,
    simulate_concurrency_workload_with_router_mode, simulate_trace_file,
    simulate_trace_file_disagg_with_router_mode, simulate_trace_file_with_router_mode,
    simulate_trace_live_file, simulate_trace_live_file_with_router_mode,
    simulate_trace_live_requests, simulate_trace_live_requests_with_router_mode,
    simulate_trace_live_workload, simulate_trace_live_workload_with_router_mode,
    simulate_trace_requests, simulate_trace_requests_disagg_with_router_mode,
    simulate_trace_requests_with_router_mode, simulate_trace_workload,
    simulate_trace_workload_disagg_with_router_mode, simulate_trace_workload_with_router_mode,
};
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
        let mut collector = TraceCollector::default();
        let uuid = Uuid::from_u128(11);

        collector.on_arrival(uuid, 0.0, 4, 4);
        collector.on_admit(uuid, 0.0, 0);
        collector.on_token(uuid, 10.0);
        collector.on_token(uuid, 11.0);
        collector.on_token(uuid, 12.0);
        collector.on_token(uuid, 110.0);

        let report = collector.finish();

        assert!((report.latency.tpot.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert!((report.latency.itl.distribution.mean_ms - (100.0 / 3.0)).abs() < 1e-9);
        assert_eq!(report.latency.itl.distribution.median_ms, 1.0);
        assert_eq!(report.latency.itl.distribution.p75_ms, 98.0);
        assert_eq!(report.latency.itl.distribution.p90_ms, 98.0);
        assert_eq!(report.latency.itl.distribution.p95_ms, 98.0);
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
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
            },
            DirectRequest {
                tokens: vec![2; 4],
                max_output_tokens: 1,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(200.0),
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
