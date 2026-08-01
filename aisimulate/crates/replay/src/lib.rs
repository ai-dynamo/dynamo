// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo-neutral, deterministic offline replay.
//!
//! [`Replayer`] owns virtual time, the event queue, logical-worker lifecycle,
//! placement/scaling policy composition, and report collection. Engine
//! scheduling and timing are supplied by `aisimulate-engine` through its
//! generalized-engine contract. Dynamo integrations depend on this crate and
//! provide placement/scaling policies; replay never depends back on Dynamo.

// The mechanically moved load-generator sources still use the historical
// `crate::common::protocols::DirectRequest` path. Keep that path as a private
// alias to the Replay-owned request DTO; no Engine internals are re-exported.
pub(crate) mod common {
    pub(crate) mod protocols {
        pub(crate) use crate::protocol::DirectRequest;
    }
}

mod agg;
mod artifact;
mod canonical;
mod capture;
pub(crate) mod components;
pub(crate) mod core;
mod disagg;
mod error;
pub(crate) mod event;
mod evidence;
mod handoff;
pub(crate) mod events {
    pub(crate) use crate::event::*;
}
mod engine;
pub mod loadgen;
mod progress;
mod protocol;
mod replayer;
mod report;
mod runtime_utils;
pub(crate) mod scaling;
mod spec;
pub(crate) mod state;

/// Compatibility namespace used by the moved aggregated/disaggregated
/// runtimes. Public callers use the crate-root contracts below.
pub(crate) mod replay {
    #[cfg(test)]
    use std::collections::VecDeque;

    #[cfg(test)]
    use crate::protocol::DirectRequest;

    #[cfg(test)]
    pub(crate) use crate::report::TraceSimulationReport;
    pub(crate) use crate::report::{ReplayTerminalStatus, TraceCollector};

    #[derive(Clone)]
    pub(crate) struct OfflineDisaggReplayConfig {
        pub(crate) prefill_factory: crate::engine::ReplayRoleFactory,
        pub(crate) decode_factory: crate::engine::ReplayRoleFactory,
        pub(crate) prefill_startup_time_ms: Option<f64>,
        pub(crate) decode_startup_time_ms: Option<f64>,
        pub(crate) num_prefill_workers: usize,
        pub(crate) num_decode_workers: usize,
        pub(crate) handoff_latency_ms: f64,
    }

    impl OfflineDisaggReplayConfig {
        pub(crate) fn prefill_factory(
            &self,
            _emit_kv_events: bool,
        ) -> anyhow::Result<crate::ReplayRoleFactory> {
            Ok(self.prefill_factory.clone())
        }

        pub(crate) fn decode_factory(
            &self,
            _emit_kv_events: bool,
        ) -> anyhow::Result<crate::ReplayRoleFactory> {
            Ok(self.decode_factory.clone())
        }

        pub(crate) fn prefill_startup_time_ms(&self) -> Option<f64> {
            self.prefill_startup_time_ms
        }

        pub(crate) fn decode_startup_time_ms(&self) -> Option<f64> {
            self.decode_startup_time_ms
        }

        pub(crate) fn handoff_latency_ms(&self) -> f64 {
            self.handoff_latency_ms
        }
    }

    #[cfg(test)]
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
            left.arrival_timestamp_ms
                .expect("trace request must have an arrival timestamp")
                .total_cmp(
                    &right
                        .arrival_timestamp_ms
                        .expect("trace request must have an arrival timestamp"),
                )
        });
        let first = requests
            .first()
            .and_then(|request| request.arrival_timestamp_ms)
            .ok_or_else(|| anyhow::anyhow!("trace replay requires at least one request"))?;
        for request in &mut requests {
            request.arrival_timestamp_ms = Some(
                (request.arrival_timestamp_ms.expect("validated timestamp") - first)
                    / arrival_speedup_ratio,
            );
        }
        Ok(requests.into())
    }
}

pub use aisimulate_engine::{HandoffId, HandoffTransferTiming};
pub use artifact::{
    ReplayArtifactKvEvent, ReplayArtifactKvEventVisibility, ReplayArtifactOutput,
    ReplayArtifactRequest, ReplayArtifacts,
};
pub use canonical::{
    CANONICAL_RESULT_EXCLUSIONS, CANONICAL_SCHEMA_VERSION, CanonicalReplayCoverage,
    CanonicalReplayRecord, canonicalize_json,
};
pub use capture::{CANONICAL_SELECTOR_SEED, ReplayCaptureOptions, ReplayDeterminism};
pub use components::TrafficStats;
#[doc(hidden)]
pub use components::{NoReplayMetadata, ReplayAdmissionMetadata, ReplayEngineObservation};
pub use core::round_robin::{AggregatedRoundRobinPlacement, PoolRoundRobinPlacement};
pub use core::{EngineEventBatch, NoEngineEvents};
pub use core::{
    Placement, PlacementCacheSample, PlacementDecision, PlacementEffects, PlacementPolicy,
    RequestIdentity, WorkerTopology,
};
#[doc(hidden)]
pub use engine::ReplayRoleFactory;
#[doc(hidden)]
pub use engine::run_engine_handoff_conformance;
pub use engine::{
    ReplayEngineConfig, ReplayEngineFactory, ReplayRoleConfig, run_engine_replay,
    run_engine_replay_with_optional_role_timing, run_engine_replay_with_timing,
};
pub use error::{ReplayError, ReplayResult};
pub use evidence::{
    EnginePressureState, KvIngestBoundary, KvIngestBoundaryStats, KvIngestEventEncoder,
    KvIngestEvidence, LifecycleOperation, OfflineRuntimeEvidence, PressureEvidence, PressureKind,
    PressureRecord, WorkerLifecycleTransition, WorkerLifecycleTransitionKind, WorkerPool,
    WorkerPoolState,
};
pub use handoff::{
    HandoffAction, HandoffActionId, HandoffActionOutcome, HandoffCompletion, HandoffFact,
    HandoffOrder, IssuedHandoffAction, NormalizedHandoffConformance, NormalizedHandoffEvent,
    NormalizedStoredTiming,
};
#[doc(hidden)]
pub use handoff::{
    HandoffCoordinatorCore, expected_normalized_handoff, validate_transfer_delay_ms,
    validate_transfer_timing,
};
pub use protocol::ForwardPassSnapshot;
#[doc(hidden)]
pub use protocol::{DirectRequest, ReplayPromptTokenSource, ReplayRequestContext};
#[doc(hidden)]
pub use replayer::ReplayRuntimeInput;
pub use replayer::{ReplayComposition, Replayer, RoundRobinComposition};
#[doc(hidden)]
pub use report::TraceCollector;
pub use report::{
    PerRequestAdmissionRecord, PerRequestRecord, PerRequestRoutingRecord, ReplayRequestPool,
    ReplayRoutingOutcome, ReplayTerminalStatus, ReplayTerminalStatus as RequestTerminalStatus,
    SlaThresholds, TraceDistributionStats, TraceGoodputStats, TraceInterTokenLatencyStats,
    TraceLatencyStats, TraceRequestCounts, TraceSimulationReport, TraceThroughputStats,
};
pub use scaling::{NoScaling, ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot};
pub use spec::{
    CURRENT_REPLAY_SPEC_VERSION, ProviderSpec, ReplayAdapters, ReplayRequest,
    ReplayRoutingMetadata, ReplaySpec, ReplayTopology, WorkerPoolSpec, WorkerStage,
};
