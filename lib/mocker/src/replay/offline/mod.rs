// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use crate::replay::normalize_trace_requests;

pub(crate) mod agg;
mod canonical;
pub(crate) mod components;
mod controlled;
pub(crate) mod core;
pub(crate) mod disagg;
mod entrypoints;
pub(crate) mod events;
pub(crate) mod evidence;
mod executor;
pub(crate) mod extensions;
mod interactive;
mod progress;
pub(crate) mod runtime_utils;
pub(crate) mod scaling;
pub(crate) mod single;
pub(crate) mod state;
mod topology;

pub use canonical::{
    CANONICAL_RESULT_EXCLUSIONS, CANONICAL_SCHEMA_VERSION, CanonicalAicIdentity,
    CanonicalAicImplementation, CanonicalDeterminismMetadata, CanonicalEngineConfig,
    CanonicalExecutionMetadata, CanonicalReplayCoverage, CanonicalReplayMetadata,
    CanonicalReplayRecord, CanonicalReplayTopology, CanonicalSemanticFeatures,
    CanonicalSlaMetadata, CanonicalSyntheticSpec, CanonicalWorkloadMetadata,
    canonical_engine_pool_metadata, canonical_topology,
};
pub use controlled::{
    ControlledReplayOptions, ReplayWorkSource, ReplayWorkSourceContext, ReplayWorkSubmission,
    simulate_controlled_aggregated, simulate_controlled_aggregated_kv_router_with_options,
    simulate_controlled_aggregated_with_options,
};
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
pub use evidence::{
    EnginePressureState, KvIngestBoundary, KvIngestBoundaryStats, KvIngestEvidence,
    LifecycleOperation, OfflineRuntimeEvidence, PressureEvidence, PressureKind, PressureRecord,
    WorkerLifecycleTransition, WorkerLifecycleTransitionKind, WorkerPool, WorkerPoolState,
    with_runtime_evidence,
};
pub use extensions::kv_router::{
    CanonicalReplayRouterMode, CanonicalRouterMetadata, canonical_router_metadata,
};
pub use interactive::{
    OfflineReplaySession, ReplayAgenticRequest, ReplayAgenticWorkflow, ReplayEvent,
    ReplayEventData, ReplayPendingPlacement, ReplayPlacementCandidate, ReplayRequestSpec,
    ReplayRoutingConstraints, ReplaySessionOptions, ReplaySessionRouter, ReplaySnapshot,
    ReplayStepStatus, ReplayWorkerSnapshot,
};
pub use topology::{PoolRouter, PoolSpec, WorkerSpec, WorkerTarget};

#[cfg(test)]
mod firewall_tests;
