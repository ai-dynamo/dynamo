// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::replay::TraceSimulationReport;

pub(crate) mod core;
pub(crate) mod events;
pub(crate) mod multi;
pub(crate) mod single;

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
