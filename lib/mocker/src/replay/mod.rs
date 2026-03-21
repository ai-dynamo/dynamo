// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod collector;
mod entrypoints;
mod live;
mod loader;
pub(crate) mod runtime;
mod validate;

use std::collections::VecDeque;

use crate::common::protocols::DirectRequest;

pub(crate) use collector::TraceCollector;
#[cfg(test)]
pub(crate) use collector::TraceRequestStatsSnapshot;
pub use collector::{
    TraceDistributionStats, TraceInterTokenLatencyStats, TraceLatencyStats, TraceRequestCounts,
    TraceSimulationReport, TraceThroughputStats,
};
pub use entrypoints::{
    simulate_concurrency_file, simulate_concurrency_live_file, simulate_concurrency_requests,
    simulate_trace_file, simulate_trace_live_file,
};

pub(crate) fn normalize_trace_requests(
    mut requests: Vec<DirectRequest>,
) -> anyhow::Result<VecDeque<DirectRequest>> {
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
}
