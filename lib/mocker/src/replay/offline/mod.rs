// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::loadgen::Trace;
pub(crate) use crate::replay::normalize_trace_requests;
use crate::replay::{ReplayRouterMode, TraceSimulationReport};
use dynamo_kv_router::config::KvRouterConfig;

pub(crate) mod core;
pub(crate) mod events;
pub(crate) mod multi;
pub(crate) mod single;
pub(crate) mod state;

pub(crate) fn simulate_trace(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    requests: Vec<DirectRequest>,
    num_workers: usize,
    arrival_speedup_ratio: f64,
    router_mode: ReplayRouterMode,
) -> anyhow::Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == crate::common::protocols::EngineType::Vllm {
        single::simulate_trace_single(args, requests, arrival_speedup_ratio)
    } else {
        multi::simulate_trace_multi(
            args,
            router_config,
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
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> anyhow::Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == crate::common::protocols::EngineType::Vllm {
        single::simulate_concurrency_single(args, requests, max_in_flight)
    } else {
        multi::simulate_concurrency_multi(
            args,
            router_config,
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
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> anyhow::Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == crate::common::protocols::EngineType::Vllm {
        single::simulate_trace_workload_single(args, trace)
    } else {
        multi::simulate_trace_workload_multi(args, router_config, trace, num_workers, router_mode)
    }
}

pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> anyhow::Result<TraceSimulationReport> {
    if num_workers == 1 && args.engine_type == crate::common::protocols::EngineType::Vllm {
        single::simulate_concurrency_workload_single(args, trace, max_in_flight)
    } else {
        multi::simulate_concurrency_workload_multi(
            args,
            router_config,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
        )
    }
}
