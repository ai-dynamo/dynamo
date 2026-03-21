// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::replay::TraceSimulationReport;
use std::collections::VecDeque;

pub(crate) mod core;
pub(crate) mod events;
pub(crate) mod multi;
pub(crate) mod single;
pub(crate) mod state;

fn normalize_trace_requests(
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

pub(crate) fn simulate_trace(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    num_workers: usize,
) -> anyhow::Result<TraceSimulationReport> {
    if num_workers == 1 {
        single::simulate_trace_single(args, requests)
    } else {
        multi::simulate_trace_multi(args, requests, num_workers)
    }
}

pub(crate) fn simulate_concurrency(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
) -> anyhow::Result<TraceSimulationReport> {
    if num_workers == 1 {
        single::simulate_concurrency_single(args, requests, max_in_flight)
    } else {
        multi::simulate_concurrency_multi(args, requests, max_in_flight, num_workers)
    }
}
