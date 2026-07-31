// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use crate::replay::normalize_trace_requests;

pub(crate) mod agg;
pub(crate) mod components;
pub(crate) mod core;
pub(crate) mod disagg;
mod entrypoints;
pub(crate) mod events;
mod executor;
pub(crate) mod extensions;
mod progress;
pub(crate) mod runtime_utils;
pub(crate) mod scaling;
pub(crate) mod single;
pub(crate) mod state;

pub use entrypoints::run_offline_handoff_conformance;
pub(crate) use entrypoints::{
    generate_trace_worker_artifacts, generate_trace_worker_artifacts_with_visibility,
    simulate_agentic_trace_workload, simulate_concurrency_disagg_with_scaling_policy,
    simulate_concurrency_with_scaling_policy, simulate_concurrency_workload_accumulating_deltas,
    simulate_concurrency_workload_disagg_with_scaling_policy,
    simulate_concurrency_workload_with_scaling_policy, simulate_trace_disagg_with_scaling_policy,
    simulate_trace_with_scaling_policy, simulate_trace_workload_accumulating_deltas,
    simulate_trace_workload_disagg_with_scaling_policy,
    simulate_trace_workload_with_scaling_policy,
};

#[cfg(test)]
mod firewall_tests;
