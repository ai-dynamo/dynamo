// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use crate::replay::normalize_trace_requests;

mod entrypoints;
pub(crate) mod extensions;

pub use entrypoints::run_offline_handoff_conformance;
pub(crate) use entrypoints::{
    generate_trace_worker_artifacts, generate_trace_worker_artifacts_with_visibility,
    simulate_agentic_trace_workload, simulate_concurrency_disagg_with_scaling_policy,
    simulate_concurrency_with_scaling_policy, simulate_concurrency_workload_accumulating_deltas,
    simulate_concurrency_workload_disagg_with_scaling_policy,
    simulate_concurrency_workload_with_scaling_policy, simulate_trace_disagg_with_scaling_policy,
    simulate_trace_with_scaling_policy, simulate_trace_workload_accumulating_deltas,
    simulate_trace_workload_disagg_with_capture_options,
    simulate_trace_workload_disagg_with_scaling_policy,
    simulate_trace_workload_with_capture_options, simulate_trace_workload_with_scaling_policy,
};

#[cfg(test)]
mod firewall_tests;
