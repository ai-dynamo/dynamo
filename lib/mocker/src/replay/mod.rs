// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod agentic;
mod artifacts;
mod entrypoints;
pub(crate) mod offline;
mod online;
mod router_controls;
mod router_shared;
mod validate;

use std::collections::VecDeque;
use std::sync::Arc;

use crate::common::protocols::{DirectRequest, MockerConfig};
use dynamo_kv_router::PrefillLoadEstimator;

pub use agentic::AgenticReplayOptions;
/// Backward-compatible Dynamo Mocker name for [`aisimulate_core::ReplayReport`].
pub use aisimulate_core::ReplayReport as TraceSimulationReport;
pub(crate) use aisimulate_core::replay::TraceCollector;
pub use aisimulate_core::replay::{
    CanonicalReplayCoverage, CanonicalReplayRecord, LifecycleOperation, OfflineRuntimeEvidence,
    PerRequestRecord, ReplayCaptureOptions, ReplayDeterminism, ReplayTerminalStatus, SlaThresholds,
    TraceDistributionStats, TraceGoodputStats, TraceInterTokenLatencyStats, TraceLatencyStats,
    TraceRequestCounts, TraceThroughputStats,
};
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use artifacts::native_g1_parent_chain_artifact;
pub use artifacts::{
    ReplayTimedKvEvent, ReplayTimedOutputSignal, ReplayTimedRequest, ReplayWorkerArtifacts,
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

pub type ReplayPrefillLoadEstimator = Arc<dyn PrefillLoadEstimator>;

#[derive(Clone, Debug)]
pub struct OfflineDisaggReplayConfig {
    pub prefill_args: MockerConfig,
    pub decode_args: MockerConfig,
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

pub use aisimulate_core::replay::TrafficStats;
pub use aisimulate_core::replay::{
    ReplayScalingDecision, ReplayScalingPolicy, ReplayScalingSnapshot,
    ReplaySchedulerIntervalMetrics, ReplaySchedulerMetricsSnapshot, ReplayTelemetryObserver,
    ReplayTelemetrySampleKind, ReplayTelemetrySnapshot, ReplayTrafficMetricsSnapshot,
};

/// Optional policy-neutral telemetry sink for one offline replay.
///
/// Telemetry is deliberately configured independently from
/// [`ReplayScalingPolicy`]: observing a replay must not change the planner
/// callback contract or its decision cadence.
pub struct ReplayTelemetryOptions {
    pub sample_interval_ms: f64,
    pub observer: Box<dyn ReplayTelemetryObserver>,
}

/// Optional replay observers installed for one invocation.
///
/// Scaling and telemetry share an invocation without coupling their callback
/// contracts or sampling cadence.
#[derive(Default)]
pub struct ReplayRuntimeObservers {
    pub scaling_policy: Option<Box<dyn ReplayScalingPolicy>>,
    pub telemetry: Option<ReplayTelemetryOptions>,
}
pub use entrypoints::{
    ReplayKvEventVisibility, generate_trace_worker_artifacts_offline,
    generate_trace_worker_artifacts_offline_with_kv_event_visibility,
    simulate_agentic_trace_live_workload_with_router_mode_and_options,
    simulate_agentic_trace_workload_disagg_with_options,
    simulate_agentic_trace_workload_disagg_with_router_mode,
    simulate_agentic_trace_workload_disagg_with_router_mode_and_runtime_observers,
    simulate_agentic_trace_workload_disagg_with_router_mode_and_telemetry,
    simulate_agentic_trace_workload_with_options, simulate_agentic_trace_workload_with_router_mode,
    simulate_agentic_trace_workload_with_router_mode_and_runtime_observers,
    simulate_agentic_trace_workload_with_router_mode_and_telemetry, simulate_concurrency_file,
    simulate_concurrency_file_disagg_with_router_mode,
    simulate_concurrency_file_disagg_with_router_mode_and_format,
    simulate_concurrency_file_disagg_with_router_mode_and_format_and_runtime_observers,
    simulate_concurrency_file_with_router_mode,
    simulate_concurrency_file_with_router_mode_and_format,
    simulate_concurrency_file_with_router_mode_and_format_and_runtime_observers,
    simulate_concurrency_live_file, simulate_concurrency_live_file_with_router_mode,
    simulate_concurrency_live_file_with_router_mode_and_format,
    simulate_concurrency_live_file_with_router_mode_and_format_and_options,
    simulate_concurrency_live_requests, simulate_concurrency_live_requests_with_router_mode,
    simulate_concurrency_live_requests_with_router_mode_and_options,
    simulate_concurrency_live_workload, simulate_concurrency_live_workload_with_router_mode,
    simulate_concurrency_live_workload_with_router_mode_and_options, simulate_concurrency_requests,
    simulate_concurrency_requests_disagg_with_router_mode,
    simulate_concurrency_requests_disagg_with_router_mode_and_runtime_observers,
    simulate_concurrency_requests_with_router_mode,
    simulate_concurrency_requests_with_router_mode_and_runtime_observers,
    simulate_concurrency_workload, simulate_concurrency_workload_disagg_with_router_mode,
    simulate_concurrency_workload_disagg_with_router_mode_and_options,
    simulate_concurrency_workload_disagg_with_router_mode_and_options_and_runtime_observers,
    simulate_concurrency_workload_with_router_mode,
    simulate_concurrency_workload_with_router_mode_and_options,
    simulate_concurrency_workload_with_router_mode_and_options_and_runtime_observers,
    simulate_loaded_trace_disagg_with_router_mode_and_capture_options,
    simulate_loaded_trace_disagg_with_router_mode_and_options,
    simulate_loaded_trace_disagg_with_router_mode_and_options_and_runtime_observers,
    simulate_loaded_trace_live_with_router_mode,
    simulate_loaded_trace_live_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_capture_options,
    simulate_loaded_trace_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_options_and_runtime_observers, simulate_trace_file,
    simulate_trace_file_disagg_with_router_mode,
    simulate_trace_file_disagg_with_router_mode_and_format,
    simulate_trace_file_disagg_with_router_mode_and_format_and_runtime_observers,
    simulate_trace_file_with_router_mode, simulate_trace_file_with_router_mode_and_format,
    simulate_trace_file_with_router_mode_and_format_and_runtime_observers,
    simulate_trace_live_file, simulate_trace_live_file_with_router_mode,
    simulate_trace_live_file_with_router_mode_and_format,
    simulate_trace_live_file_with_router_mode_and_format_and_options, simulate_trace_live_requests,
    simulate_trace_live_requests_with_router_mode,
    simulate_trace_live_requests_with_router_mode_and_options, simulate_trace_live_workload,
    simulate_trace_live_workload_with_router_mode,
    simulate_trace_live_workload_with_router_mode_and_options, simulate_trace_requests,
    simulate_trace_requests_disagg_with_router_mode,
    simulate_trace_requests_disagg_with_router_mode_and_runtime_observers,
    simulate_trace_requests_with_router_mode,
    simulate_trace_requests_with_router_mode_and_runtime_observers, simulate_trace_workload,
    simulate_trace_workload_disagg_with_router_mode,
    simulate_trace_workload_disagg_with_router_mode_and_options_and_runtime_observers,
    simulate_trace_workload_with_router_mode,
    simulate_trace_workload_with_router_mode_and_options_and_runtime_observers,
};
#[doc(hidden)]
pub use offline::run_offline_handoff_conformance;
pub use router_controls::{
    ReplayRouterControls, ReplayRouterParameters, apply_router_parameters, parse_router_worker_type,
};
pub use validate::validate_replay_args_mode;

/// Settings for the offline replays a host runs on one thread, installed by [`with_replay_scope`].
///
/// They reach replay internals that the entrypoint signatures do not carry. The default scope is
/// the behavior outside any [`with_replay_scope`].
#[derive(Clone, Debug, Default)]
pub struct ReplayScope {
    /// Delays the KV cache events (blocks stored and removed) the router's indexer observes by this
    /// much simulated time. Prefill and request completions stay immediate, as a live router
    /// observes them in-band on the response path. `0.0` keeps synchronous updates. Offline
    /// KV-router replay only.
    pub kv_event_lag_ms: f64,
    /// Registry in which each offline KV-router placement registers its selector, so the host can
    /// change router policy parameters during the replay. `None` registers nothing.
    pub router_controls: Option<ReplayRouterControls>,
    /// Request identity and KV-router selection semantics of offline replays that do not pass
    /// their own [`ReplayCaptureOptions`]. [`ReplayDeterminism::CanonicalV1`] assigns ordinal
    /// request IDs and seeds every KV-router selector, so two replays of the same input and
    /// configuration produce the same report.
    pub determinism: ReplayDeterminism,
}

thread_local! {
    static REPLAY_SCOPE: std::cell::RefCell<ReplayScope> = const {
        std::cell::RefCell::new(ReplayScope {
            kv_event_lag_ms: 0.0,
            router_controls: None,
            determinism: ReplayDeterminism::Random,
        })
    };
}

/// Run `replay` with `scope` installed for the offline replays constructed on this thread.
///
/// The scope replaces any outer scope entirely, so a nested replay never registers into a router
/// control registry it does not own. The previous scope is restored when `replay` returns or
/// unwinds.
pub fn with_replay_scope<T>(scope: ReplayScope, replay: impl FnOnce() -> T) -> anyhow::Result<T> {
    let lag_ms = scope.kv_event_lag_ms;
    anyhow::ensure!(
        lag_ms.is_finite() && lag_ms >= 0.0,
        "kv_event_lag_ms must be finite and non-negative, got {lag_ms}"
    );
    struct Restore(ReplayScope);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = std::mem::take(&mut self.0);
            REPLAY_SCOPE.with(|current| *current.borrow_mut() = previous);
        }
    }
    let _restore = Restore(REPLAY_SCOPE.with(|current| current.replace(scope)));
    Ok(replay())
}

/// Run `replay` with a simulated KV event lag for offline KV-router replay.
///
/// [`with_replay_scope`] with only [`ReplayScope::kv_event_lag_ms`] set; like it, this replaces
/// any outer scope.
pub fn with_kv_event_lag_ms<T>(lag_ms: f64, replay: impl FnOnce() -> T) -> anyhow::Result<T> {
    let scope = ReplayScope {
        kv_event_lag_ms: lag_ms,
        ..ReplayScope::default()
    };
    with_replay_scope(scope, replay)
}

/// The scope installed by the innermost [`with_replay_scope`] on this thread.
pub(crate) fn replay_scope() -> ReplayScope {
    REPLAY_SCOPE.with(|current| current.borrow().clone())
}

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

pub(crate) fn effective_agentic_lanes(
    requested_lanes: Option<usize>,
    play_count: usize,
) -> Option<usize> {
    requested_lanes.map(|lane_count| lane_count.min(play_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn replay_scope_is_validated_replaced_and_restored() {
        let lag_scope = |kv_event_lag_ms| ReplayScope {
            kv_event_lag_ms,
            ..Default::default()
        };
        assert!(with_replay_scope(lag_scope(f64::INFINITY), || ()).is_err());
        assert!(with_replay_scope(lag_scope(-1.0), || ()).is_err());
        let is_default = |scope: ReplayScope| {
            scope.kv_event_lag_ms == 0.0
                && scope.router_controls.is_none()
                && scope.determinism == ReplayDeterminism::Random
        };
        assert!(is_default(replay_scope()));

        let outer = ReplayScope {
            kv_event_lag_ms: 25.0,
            router_controls: Some(ReplayRouterControls::new()),
            determinism: ReplayDeterminism::CanonicalV1,
        };
        with_replay_scope(outer, || {
            let current = replay_scope();
            assert_eq!(current.kv_event_lag_ms, 25.0);
            assert!(current.router_controls.is_some());
            assert_eq!(current.determinism, ReplayDeterminism::CanonicalV1);
            // A nested scope replaces every setting, so a nested replay registers nowhere.
            with_replay_scope(ReplayScope::default(), || {
                assert!(is_default(replay_scope()))
            })
            .unwrap();
            let restored = replay_scope();
            assert_eq!(restored.kv_event_lag_ms, 25.0);
            assert!(restored.router_controls.is_some());
        })
        .unwrap();
        assert!(is_default(replay_scope()));

        // A replay that unwinds also restores the previous scope.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_replay_scope(lag_scope(5.0), || panic!("replay failed"))
        }));
        assert!(result.is_err());
        assert!(is_default(replay_scope()));
    }

    #[test]
    fn test_normalize_trace_requests_applies_arrival_speedup_ratio() {
        let requests = vec![
            DirectRequest {
                tokens: vec![1; 4],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(1)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(100.0),
                ..Default::default()
            },
            DirectRequest {
                tokens: vec![2; 4],
                max_output_tokens: 1,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(2)),
                dp_rank: 0,
                arrival_timestamp_ms: Some(200.0),
                ..Default::default()
            },
        ];

        let normalized = normalize_trace_requests(requests, 10.0).unwrap();
        let arrivals = normalized
            .into_iter()
            .map(|request| request.arrival_timestamp_ms.unwrap())
            .collect::<Vec<_>>();

        assert_eq!(arrivals, vec![0.0, 10.0]);
    }

    #[test]
    fn test_effective_agentic_lanes_are_bounded_by_play_count() {
        assert_eq!(effective_agentic_lanes(None, 3), None);
        assert_eq!(effective_agentic_lanes(Some(2), 3), Some(2));
        assert_eq!(effective_agentic_lanes(Some(usize::MAX), 3), Some(3));
    }
}
