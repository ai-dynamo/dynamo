// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use crate::replay::normalize_trace_requests;

pub(crate) mod agg;
pub(crate) mod components;
pub(crate) mod core;
pub(crate) mod disagg;
mod entrypoints;
pub(crate) mod events;
pub(crate) mod planner_hook;
mod progress;
pub(crate) mod runtime_utils;
mod session;
pub(crate) mod single;
pub(crate) mod state;

pub use entrypoints::run_offline_handoff_conformance;
pub(crate) use entrypoints::{
    generate_trace_worker_artifacts, generate_trace_worker_artifacts_with_visibility,
    simulate_agentic_trace_workload, simulate_concurrency, simulate_concurrency_disagg,
    simulate_concurrency_workload, simulate_concurrency_workload_accumulating_deltas,
    simulate_concurrency_workload_disagg, simulate_trace, simulate_trace_disagg,
    simulate_trace_workload, simulate_trace_workload_accumulating_deltas,
    simulate_trace_workload_disagg,
};
pub use session::{
    MAX_REPLAY_SESSION_OBSERVATIONS, MAX_REPLAY_SESSION_REPLICAS, MAX_REPLAY_SESSION_REQUESTS,
    MAX_REPLAY_SESSION_SCALE_ACTIONS, ReplayAdvanceSnapshot, ReplayCheckpointKind,
    ReplayReplicaLifecycle, ReplayReplicaTelemetry, ReplaySampledAdvance, ReplayScaleAction,
    ReplaySession, ReplaySessionCheckpoint, ReplaySessionConfig, ReplaySessionState,
    ReplaySessionTelemetry, ReplayTelemetryWindow, ReplayTimelineObservation,
};
