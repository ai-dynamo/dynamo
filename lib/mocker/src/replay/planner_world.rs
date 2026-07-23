// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-deployment offline replay coordinator.
//!
//! Each deployment retains its existing aggregated or disaggregated DES. This
//! module adds a control-plane barrier around those runtimes: one authoritative
//! clock settles every data plane before freezing all same-time planner metrics,
//! invokes one world callback, applies its ordered cross-deployment actions, and
//! then re-settles same-time work.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;

use super::offline::planner_hook::PlannerTickMetrics;
use super::planner_handle::{PlannerReplayHandle, RuntimeKind};
use super::{TraceCollector, TraceSimulationReport};

/// One ordered replica-target mutation returned by the world planner.
///
/// Actions may target any deployment, including one whose local planner did not
/// tick at the current control timestamp. Vector order is application order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorldScalingAction {
    pub deployment_id: String,
    pub target_prefill: Option<usize>,
    pub target_decode: Option<usize>,
}

/// Result of one batched world-planner callback.
///
/// `next_ticks` must contain exactly one entry for every deployment present in
/// the callback batch. `None` stops that deployment's local planner; `Some(t)`
/// must be a finite absolute simulated timestamp strictly greater than the
/// current world time.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct WorldPlannerDecision {
    pub next_ticks: Vec<(String, Option<f64>)>,
    pub actions: Vec<WorldScalingAction>,
}

/// Planner callback driven once for all deployments due at the same timestamp.
///
/// The metrics vector is sorted by deployment ID and frozen only after every
/// deployment data plane has settled at the callback timestamp.
pub trait WorldPlannerHook {
    /// Initial absolute simulated tick time for one deployment. A non-finite
    /// value disables that deployment's planner. Finite times before the current
    /// world time are clamped to the current time, matching legacy replay.
    fn initial_tick_ms(&mut self, deployment_id: &str) -> Result<f64>;

    /// Evaluate all local planners due at the same exact simulated timestamp.
    fn on_ticks(
        &mut self,
        ticks: Vec<(String, PlannerTickMetrics)>,
    ) -> Result<WorldPlannerDecision>;
}

/// Final report for a multi-deployment replay.
#[derive(Debug, Clone, Serialize)]
pub struct ReplayWorldReport {
    /// Authoritative world control time when request work became quiescent.
    pub duration_ms: f64,
    /// Real execution time for the whole coordinated run.
    pub wall_time_ms: f64,
    /// Per-deployment reports in canonical deployment-ID order.
    pub deployments: Vec<(String, TraceSimulationReport)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::replay) struct DeploymentRuntimeState {
    pub request_work_pending: bool,
    pub lifecycle_work_pending: bool,
}

/// Internal facade over the four existing agg/disagg + routing compositions.
///
/// This deliberately contains no scheduling logic of its own: all work is
/// delegated to the runtime variants' existing fixed-point settlement and
/// scaling paths.
#[allow(clippy::large_enum_variant)]
pub(in crate::replay) struct DeploymentRuntime {
    inner: RuntimeKind,
}

impl DeploymentRuntime {
    fn new(inner: RuntimeKind) -> Result<Self> {
        let mut runtime = Self { inner };
        match &mut runtime.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.enable_world_planner()?,
            RuntimeKind::AggKv(runtime) => runtime.enable_world_planner()?,
            RuntimeKind::DisaggRoundRobin(runtime) => runtime.enable_world_planner()?,
            RuntimeKind::DisaggKv(runtime) => runtime.enable_world_planner()?,
        }
        Ok(runtime)
    }

    fn now_ms(&self) -> f64 {
        match &self.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.world_now_ms(),
            RuntimeKind::AggKv(runtime) => runtime.world_now_ms(),
            RuntimeKind::DisaggRoundRobin(runtime) => runtime.world_now_ms(),
            RuntimeKind::DisaggKv(runtime) => runtime.world_now_ms(),
        }
    }

    fn next_data_timestamp(&mut self) -> Option<f64> {
        match &mut self.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.world_next_data_timestamp(),
            RuntimeKind::AggKv(runtime) => runtime.world_next_data_timestamp(),
            RuntimeKind::DisaggRoundRobin(runtime) => runtime.world_next_data_timestamp(),
            RuntimeKind::DisaggKv(runtime) => runtime.world_next_data_timestamp(),
        }
    }

    fn settle_data_plane_to(&mut self, control_ms: f64) -> Result<()> {
        match &mut self.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.world_settle_data_plane_to(control_ms),
            RuntimeKind::AggKv(runtime) => runtime.world_settle_data_plane_to(control_ms),
            RuntimeKind::DisaggRoundRobin(runtime) => {
                runtime.world_settle_data_plane_to(control_ms)
            }
            RuntimeKind::DisaggKv(runtime) => runtime.world_settle_data_plane_to(control_ms),
        }
    }

    fn take_planner_metrics(&mut self) -> PlannerTickMetrics {
        match &mut self.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.world_take_planner_metrics(),
            RuntimeKind::AggKv(runtime) => runtime.world_take_planner_metrics(),
            RuntimeKind::DisaggRoundRobin(runtime) => runtime.world_take_planner_metrics(),
            RuntimeKind::DisaggKv(runtime) => runtime.world_take_planner_metrics(),
        }
    }

    fn apply_targets(
        &mut self,
        target_prefill: Option<usize>,
        target_decode: Option<usize>,
    ) -> Result<()> {
        match &mut self.inner {
            RuntimeKind::AggRoundRobin(runtime) => {
                if target_prefill.is_some() {
                    bail!("aggregated deployment does not have a prefill pool");
                }
                runtime.world_apply_targets(None, target_decode)
            }
            RuntimeKind::AggKv(runtime) => {
                if target_prefill.is_some() {
                    bail!("aggregated deployment does not have a prefill pool");
                }
                runtime.world_apply_targets(None, target_decode)
            }
            RuntimeKind::DisaggRoundRobin(runtime) => {
                runtime.world_apply_targets(target_prefill, target_decode)
            }
            RuntimeKind::DisaggKv(runtime) => {
                runtime.world_apply_targets(target_prefill, target_decode)
            }
        }
    }

    fn state(&self) -> DeploymentRuntimeState {
        let (request_work_pending, lifecycle_work_pending) = match &self.inner {
            RuntimeKind::AggRoundRobin(runtime) => (
                runtime.world_has_request_work(),
                runtime.world_has_lifecycle_work(),
            ),
            RuntimeKind::AggKv(runtime) => (
                runtime.world_has_request_work(),
                runtime.world_has_lifecycle_work(),
            ),
            RuntimeKind::DisaggRoundRobin(runtime) => (
                runtime.world_has_request_work(),
                runtime.world_has_lifecycle_work(),
            ),
            RuntimeKind::DisaggKv(runtime) => (
                runtime.world_has_request_work(),
                runtime.world_has_lifecycle_work(),
            ),
        };
        DeploymentRuntimeState {
            request_work_pending,
            lifecycle_work_pending,
        }
    }

    fn stop_planner(&mut self) {
        match &mut self.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.world_stop_planner(),
            RuntimeKind::AggKv(runtime) => runtime.world_stop_planner(),
            RuntimeKind::DisaggRoundRobin(runtime) => runtime.world_stop_planner(),
            RuntimeKind::DisaggKv(runtime) => runtime.world_stop_planner(),
        }
    }

    fn finish(self) -> TraceCollector {
        match self.inner {
            RuntimeKind::AggRoundRobin(runtime) => runtime.world_finish(),
            RuntimeKind::AggKv(runtime) => runtime.world_finish(),
            RuntimeKind::DisaggRoundRobin(runtime) => runtime.world_finish(),
            RuntimeKind::DisaggKv(runtime) => runtime.world_finish(),
        }
    }
}

struct WorldDeployment {
    id: String,
    runtime: DeploymentRuntime,
    next_planner_tick_ms: Option<f64>,
}

/// Owns multiple existing replay runtimes under one logical control clock.
pub struct ReplayWorldHandle {
    deployments: Vec<WorldDeployment>,
    now_ms: f64,
    started_at: Instant,
}

impl ReplayWorldHandle {
    /// Consume existing single-deployment handles into one coordinated world.
    ///
    /// Deployment IDs must be non-empty and unique. The world canonicalizes
    /// deployment order lexicographically, independent of caller insertion order.
    pub fn from_deployments(deployments: Vec<(String, PlannerReplayHandle)>) -> Result<Self> {
        if deployments.is_empty() {
            bail!("multi-deployment replay requires at least one deployment");
        }

        let mut seen = BTreeSet::new();
        let mut world_deployments = Vec::with_capacity(deployments.len());
        for (id, handle) in deployments {
            if id.trim().is_empty() {
                bail!("multi-deployment replay requires non-empty deployment IDs");
            }
            if !seen.insert(id.clone()) {
                bail!("duplicate replay deployment ID {id:?}");
            }
            world_deployments.push(WorldDeployment {
                id,
                runtime: DeploymentRuntime::new(handle.into_runtime_kind())?,
                next_planner_tick_ms: None,
            });
        }
        world_deployments.sort_by(|left, right| left.id.cmp(&right.id));

        Ok(Self {
            deployments: world_deployments,
            now_ms: 0.0,
            started_at: Instant::now(),
        })
    }

    /// Run until request work is quiescent across the whole world.
    ///
    /// A locally idle deployment continues receiving its scheduled planner ticks
    /// while any other deployment still has request work. Once world request work
    /// is quiescent, planner deadlines and startup-only lifecycle events no longer
    /// keep the run alive.
    pub fn run(mut self, mut hook: Box<dyn WorldPlannerHook>) -> Result<ReplayWorldReport> {
        self.settle_all_data_planes(0.0)?;
        self.seed_initial_ticks(hook.as_mut())?;

        while self.has_request_work() {
            let next_ms = self.next_control_timestamp()?.ok_or_else(|| {
                let blocked = self
                    .deployments
                    .iter()
                    .filter(|deployment| deployment.runtime.state().request_work_pending)
                    .map(|deployment| deployment.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow!(
                    "multi-deployment replay reached a dead end at {}ms with request work in: {blocked}",
                    self.now_ms
                )
            })?;
            if next_ms < self.now_ms {
                bail!(
                    "multi-deployment replay produced a past control timestamp {next_ms}ms while at {}ms",
                    self.now_ms
                );
            }

            self.settle_all_data_planes(next_ms)?;
            self.now_ms = next_ms;

            // Match legacy replay: once request work is globally quiescent, a
            // same-time planner heartbeat is dropped rather than keeping the run
            // alive. Startup-only lifecycle events are intentionally ignored here.
            if !self.has_request_work() {
                break;
            }

            let due_indices = self.due_planner_indices();
            if due_indices.is_empty() {
                continue;
            }

            let ticks = due_indices
                .iter()
                .map(|&index| {
                    let deployment = &mut self.deployments[index];
                    (
                        deployment.id.clone(),
                        deployment.runtime.take_planner_metrics(),
                    )
                })
                .collect();
            let decision = hook.on_ticks(ticks)?;

            let scheduled = self.validate_next_ticks(&due_indices, &decision.next_ticks)?;
            self.validate_actions(&decision.actions)?;
            self.install_next_ticks(&due_indices, scheduled);
            self.apply_actions(decision.actions)?;

            // Zero-startup scaling and router changes can make more data-plane
            // work possible without advancing logical time.
            self.settle_all_data_planes(self.now_ms)?;
        }

        self.finish()
    }

    fn seed_initial_ticks(&mut self, hook: &mut dyn WorldPlannerHook) -> Result<()> {
        for deployment in &mut self.deployments {
            let first_ms = hook.initial_tick_ms(&deployment.id).with_context(|| {
                format!(
                    "failed to obtain initial planner tick for deployment {:?}",
                    deployment.id
                )
            })?;
            if first_ms.is_finite() {
                deployment.next_planner_tick_ms = Some(first_ms.max(self.now_ms));
            } else {
                deployment.next_planner_tick_ms = None;
                deployment.runtime.stop_planner();
            }
        }
        Ok(())
    }

    fn has_request_work(&self) -> bool {
        self.deployments
            .iter()
            .any(|deployment| deployment.runtime.state().request_work_pending)
    }

    fn settle_all_data_planes(&mut self, control_ms: f64) -> Result<()> {
        for deployment in &mut self.deployments {
            deployment
                .runtime
                .settle_data_plane_to(control_ms)
                .with_context(|| {
                    format!(
                        "failed to settle deployment {:?} at {control_ms}ms",
                        deployment.id
                    )
                })?;
            if deployment.runtime.now_ms() != control_ms {
                bail!(
                    "deployment {:?} settled at {}ms instead of authoritative world time {control_ms}ms",
                    deployment.id,
                    deployment.runtime.now_ms()
                );
            }
        }
        Ok(())
    }

    fn next_control_timestamp(&mut self) -> Result<Option<f64>> {
        let mut next = None;
        for deployment in &mut self.deployments {
            let state = deployment.runtime.state();
            let next_data_timestamp = (state.request_work_pending || state.lifecycle_work_pending)
                .then(|| deployment.runtime.next_data_timestamp())
                .flatten();
            for candidate in [next_data_timestamp, deployment.next_planner_tick_ms]
                .into_iter()
                .flatten()
            {
                if !candidate.is_finite() {
                    bail!(
                        "deployment {:?} produced a non-finite control timestamp {candidate}",
                        deployment.id
                    );
                }
                next = Some(match next {
                    Some(current) if current <= candidate => current,
                    _ => candidate,
                });
            }
        }
        Ok(next)
    }

    fn due_planner_indices(&self) -> Vec<usize> {
        self.deployments
            .iter()
            .enumerate()
            .filter_map(|(index, deployment)| {
                (deployment.next_planner_tick_ms == Some(self.now_ms)).then_some(index)
            })
            .collect()
    }

    fn validate_next_ticks(
        &self,
        due_indices: &[usize],
        next_ticks: &[(String, Option<f64>)],
    ) -> Result<BTreeMap<String, Option<f64>>> {
        let due_ids = due_indices
            .iter()
            .map(|&index| self.deployments[index].id.as_str())
            .collect::<BTreeSet<_>>();
        let mut scheduled = BTreeMap::new();

        for (id, next_ms) in next_ticks {
            if !due_ids.contains(id.as_str()) {
                bail!(
                    "world planner returned a next tick for non-due deployment {id:?} at {}ms",
                    self.now_ms
                );
            }
            if scheduled.insert(id.clone(), *next_ms).is_some() {
                bail!("world planner returned duplicate next ticks for deployment {id:?}");
            }
            if let Some(next_ms) = next_ms
                && (!next_ms.is_finite() || *next_ms <= self.now_ms)
            {
                bail!(
                    "world planner next tick for deployment {id:?} must be finite and strictly greater than {}ms, got {next_ms}",
                    self.now_ms
                );
            }
        }

        if scheduled.len() != due_ids.len() {
            let missing = due_ids
                .into_iter()
                .filter(|id| !scheduled.contains_key(*id))
                .collect::<Vec<_>>()
                .join(", ");
            bail!("world planner omitted next ticks for due deployments: {missing}");
        }
        Ok(scheduled)
    }

    fn validate_actions(&self, actions: &[WorldScalingAction]) -> Result<()> {
        let deployment_ids = self
            .deployments
            .iter()
            .map(|deployment| deployment.id.as_str())
            .collect::<BTreeSet<_>>();
        for action in actions {
            if !deployment_ids.contains(action.deployment_id.as_str()) {
                bail!(
                    "world planner action targets unknown deployment {:?}",
                    action.deployment_id
                );
            }
            if action.target_prefill.is_none() && action.target_decode.is_none() {
                bail!(
                    "world planner action for deployment {:?} has no replica target",
                    action.deployment_id
                );
            }
            let deployment = self
                .deployments
                .iter()
                .find(|deployment| deployment.id == action.deployment_id)
                .expect("validated deployment must exist");
            if action.target_prefill.is_some()
                && matches!(
                    &deployment.runtime.inner,
                    RuntimeKind::AggRoundRobin(_) | RuntimeKind::AggKv(_)
                )
            {
                bail!(
                    "world planner action targets nonexistent prefill pool on aggregated deployment {:?}",
                    action.deployment_id
                );
            }
        }
        Ok(())
    }

    fn install_next_ticks(
        &mut self,
        due_indices: &[usize],
        mut scheduled: BTreeMap<String, Option<f64>>,
    ) {
        for &index in due_indices {
            let deployment = &mut self.deployments[index];
            deployment.next_planner_tick_ms = scheduled
                .remove(&deployment.id)
                .expect("validated next tick must exist");
            if deployment.next_planner_tick_ms.is_none() {
                deployment.runtime.stop_planner();
            }
        }
    }

    fn apply_actions(&mut self, actions: Vec<WorldScalingAction>) -> Result<()> {
        for action in actions {
            let deployment = self
                .deployments
                .iter_mut()
                .find(|deployment| deployment.id == action.deployment_id)
                .expect("validated deployment action target must exist");
            deployment
                .runtime
                .apply_targets(action.target_prefill, action.target_decode)
                .with_context(|| {
                    format!(
                        "failed to apply world planner action to deployment {:?} at {}ms",
                        action.deployment_id, self.now_ms
                    )
                })?;
        }
        Ok(())
    }

    fn finish(self) -> Result<ReplayWorldReport> {
        let wall_time_ms = self.started_at.elapsed().as_secs_f64() * 1000.0;
        let deployments = self
            .deployments
            .into_iter()
            .map(|deployment| {
                (
                    deployment.id,
                    deployment
                        .runtime
                        .finish()
                        .finish()
                        .with_wall_time_ms(wall_time_ms),
                )
            })
            .collect();
        Ok(ReplayWorldReport {
            duration_ms: self.now_ms,
            wall_time_ms,
            deployments,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::{ReplayWorldHandle, WorldPlannerDecision, WorldPlannerHook, WorldScalingAction};
    use crate::common::protocols::{MockEngineArgs, WorkerType};
    use crate::loadgen::{ArrivalSpec, DelaySpec, LengthSpec, SyntheticTraceSpec, Trace};
    use crate::replay::offline::extensions::kv_router::ReplayKvRouterConfig;
    use crate::replay::offline::planner_hook::PlannerTickMetrics;
    use crate::replay::{
        OfflineDisaggReplayConfig, PlannerReplayHandle, ReplayRouterMode, SlaThresholds,
    };

    fn small_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .enable_chunked_prefill(true)
            .speedup_ratio(1_000_000.0)
            .build()
            .unwrap()
    }

    fn one_request_trace(arrival_ms: f64) -> Trace {
        let mut trace = Trace::synthetic(SyntheticTraceSpec {
            block_size: 4,
            num_sessions: 1,
            turns_per_session: 1,
            input_tokens: LengthSpec {
                mean: 8,
                stddev: 0.0,
            },
            output_tokens: LengthSpec {
                mean: 1,
                stddev: 0.0,
            },
            shared_prefix_ratio: 0.0,
            num_prefix_groups: 0,
            first_turn_arrivals: ArrivalSpec::Burst,
            inter_turn_delays: DelaySpec::None,
            seed: 42,
        })
        .unwrap();
        trace.sessions[0].first_arrival_timestamp_ms = Some(arrival_ms);
        trace
    }

    fn handle_with_workers(arrival_ms: f64, num_workers: usize) -> PlannerReplayHandle {
        PlannerReplayHandle::from_trace(
            small_args(),
            None,
            None,
            one_request_trace(arrival_ms),
            num_workers,
            None,
            ReplayRouterMode::RoundRobin,
            SlaThresholds::default(),
        )
        .unwrap()
    }

    fn handle(arrival_ms: f64) -> PlannerReplayHandle {
        handle_with_workers(arrival_ms, 1)
    }

    fn queueing_handle_with_no_workers(arrival_ms: f64) -> PlannerReplayHandle {
        PlannerReplayHandle::from_trace(
            small_args(),
            Some(ReplayKvRouterConfig {
                router_queue_threshold: Some(0.5),
                ..ReplayKvRouterConfig::default()
            }),
            None,
            one_request_trace(arrival_ms),
            0,
            None,
            ReplayRouterMode::KvRouter,
            SlaThresholds::default(),
        )
        .unwrap()
    }

    fn disagg_handle(arrival_ms: f64) -> PlannerReplayHandle {
        PlannerReplayHandle::from_trace_disagg(
            OfflineDisaggReplayConfig {
                prefill_args: MockEngineArgs {
                    worker_type: WorkerType::Prefill,
                    ..small_args()
                },
                decode_args: MockEngineArgs {
                    worker_type: WorkerType::Decode,
                    ..small_args()
                },
                num_prefill_workers: 1,
                num_decode_workers: 1,
            },
            None,
            None,
            one_request_trace(arrival_ms),
            None,
            ReplayRouterMode::RoundRobin,
            SlaThresholds::default(),
        )
        .unwrap()
    }

    fn startup_handle(arrival_ms: f64) -> PlannerReplayHandle {
        let args = MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(128)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(false)
            .enable_chunked_prefill(true)
            .speedup_ratio(1_000_000.0)
            .startup_time(Some(1.0))
            .build()
            .unwrap();
        PlannerReplayHandle::from_trace(
            args,
            None,
            None,
            one_request_trace(arrival_ms),
            1,
            None,
            ReplayRouterMode::RoundRobin,
            SlaThresholds::default(),
        )
        .unwrap()
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TickObservation {
        deployment_id: String,
        now_ms: f64,
        num_req: usize,
        active_decode: usize,
    }

    struct OneBatchHook {
        initial_tick_ms: f64,
        observations: Rc<RefCell<Vec<Vec<TickObservation>>>>,
    }

    impl WorldPlannerHook for OneBatchHook {
        fn initial_tick_ms(&mut self, _deployment_id: &str) -> anyhow::Result<f64> {
            Ok(self.initial_tick_ms)
        }

        fn on_ticks(
            &mut self,
            ticks: Vec<(String, PlannerTickMetrics)>,
        ) -> anyhow::Result<WorldPlannerDecision> {
            let observations = ticks
                .iter()
                .map(|(deployment_id, metrics)| TickObservation {
                    deployment_id: deployment_id.clone(),
                    now_ms: metrics.now_ms,
                    num_req: metrics.traffic.num_req,
                    active_decode: metrics.active_decode_ids.len(),
                })
                .collect();
            let next_ticks = ticks
                .into_iter()
                .map(|(deployment_id, _)| (deployment_id, None))
                .collect();
            self.observations.borrow_mut().push(observations);
            Ok(WorldPlannerDecision {
                next_ticks,
                actions: Vec::new(),
            })
        }
    }

    #[test]
    fn batches_same_time_ticks_after_every_deployment_settles() {
        let observations = Rc::new(RefCell::new(Vec::new()));
        let report = ReplayWorldHandle::from_deployments(vec![
            ("zeta".to_string(), handle(2_000.0)),
            ("alpha".to_string(), handle(0.0)),
        ])
        .unwrap()
        .run(Box::new(OneBatchHook {
            initial_tick_ms: 500.0,
            observations: Rc::clone(&observations),
        }))
        .unwrap();

        let observations = observations.borrow();
        assert_eq!(observations.len(), 1);
        assert_eq!(
            observations[0]
                .iter()
                .map(|observation| observation.deployment_id.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
        assert!(
            observations[0]
                .iter()
                .all(|observation| observation.now_ms == 500.0)
        );
        assert_eq!(observations[0][0].num_req, 1);
        assert_eq!(observations[0][1].num_req, 0);
        assert_eq!(
            report
                .deployments
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
        assert!(
            report
                .deployments
                .iter()
                .all(|(_, report)| report.request_counts.completed_requests == 1)
        );
    }

    #[test]
    fn mixed_aggregated_and_disaggregated_deployments_share_a_barrier() {
        let observations = Rc::new(RefCell::new(Vec::new()));
        let report = ReplayWorldHandle::from_deployments(vec![
            ("disagg".to_string(), disagg_handle(100.0)),
            ("agg".to_string(), handle(100.0)),
        ])
        .unwrap()
        .run(Box::new(OneBatchHook {
            initial_tick_ms: 0.0,
            observations: Rc::clone(&observations),
        }))
        .unwrap();

        let observations = observations.borrow();
        assert_eq!(observations.len(), 1);
        assert_eq!(
            observations[0]
                .iter()
                .map(|observation| observation.deployment_id.as_str())
                .collect::<Vec<_>>(),
            vec!["agg", "disagg"]
        );
        assert!(
            report
                .deployments
                .iter()
                .all(|(_, report)| report.request_counts.completed_requests == 1)
        );
    }

    struct CrossDeploymentScaleHook {
        observations: Rc<RefCell<Vec<Vec<TickObservation>>>>,
        calls: usize,
    }

    impl WorldPlannerHook for CrossDeploymentScaleHook {
        fn initial_tick_ms(&mut self, _deployment_id: &str) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_ticks(
            &mut self,
            ticks: Vec<(String, PlannerTickMetrics)>,
        ) -> anyhow::Result<WorldPlannerDecision> {
            let observations = ticks
                .iter()
                .map(|(deployment_id, metrics)| TickObservation {
                    deployment_id: deployment_id.clone(),
                    now_ms: metrics.now_ms,
                    num_req: metrics.traffic.num_req,
                    active_decode: metrics.active_decode_ids.len(),
                })
                .collect();
            let next_ticks = ticks
                .iter()
                .map(|(deployment_id, _)| (deployment_id.clone(), (self.calls == 0).then_some(1.0)))
                .collect();
            let actions = if self.calls == 0 {
                vec![WorldScalingAction {
                    deployment_id: "beta".to_string(),
                    target_prefill: None,
                    target_decode: Some(2),
                }]
            } else {
                Vec::new()
            };

            self.calls += 1;
            self.observations.borrow_mut().push(observations);
            Ok(WorldPlannerDecision {
                next_ticks,
                actions,
            })
        }
    }

    #[test]
    fn applies_cross_deployment_action_before_the_next_barrier() {
        let observations = Rc::new(RefCell::new(Vec::new()));
        let report = ReplayWorldHandle::from_deployments(vec![
            ("alpha".to_string(), handle(1_000.0)),
            ("beta".to_string(), handle(1_000.0)),
        ])
        .unwrap()
        .run(Box::new(CrossDeploymentScaleHook {
            observations: Rc::clone(&observations),
            calls: 0,
        }))
        .unwrap();

        let observations = observations.borrow();
        assert_eq!(observations.len(), 2);
        assert!(observations.iter().all(|batch| {
            batch
                .iter()
                .map(|tick| tick.deployment_id.as_str())
                .eq(["alpha", "beta"])
        }));
        assert!(
            observations[0]
                .iter()
                .all(|observation| observation.active_decode == 1)
        );
        assert_eq!(
            observations[1]
                .iter()
                .find(|observation| observation.deployment_id == "beta")
                .unwrap()
                .active_decode,
            2
        );
        assert!(
            report
                .deployments
                .iter()
                .all(|(_, report)| report.request_counts.completed_requests == 1)
        );
    }

    struct WakeQueuedDeploymentHook {
        observation: Rc<RefCell<Option<TickObservation>>>,
    }

    impl WorldPlannerHook for WakeQueuedDeploymentHook {
        fn initial_tick_ms(&mut self, _deployment_id: &str) -> anyhow::Result<f64> {
            Ok(10.0)
        }

        fn on_ticks(
            &mut self,
            mut ticks: Vec<(String, PlannerTickMetrics)>,
        ) -> anyhow::Result<WorldPlannerDecision> {
            assert_eq!(ticks.len(), 1);
            let (deployment_id, metrics) = ticks.pop().unwrap();
            self.observation.replace(Some(TickObservation {
                deployment_id: deployment_id.clone(),
                now_ms: metrics.now_ms,
                num_req: metrics.traffic.num_req,
                active_decode: metrics.active_decode_ids.len(),
            }));
            Ok(WorldPlannerDecision {
                next_ticks: vec![(deployment_id.clone(), None)],
                actions: vec![WorldScalingAction {
                    deployment_id,
                    target_prefill: None,
                    target_decode: Some(1),
                }],
            })
        }
    }

    #[test]
    fn planner_barrier_wakes_queued_deployment_with_no_local_event() {
        let observation = Rc::new(RefCell::new(None));
        let report = ReplayWorldHandle::from_deployments(vec![(
            "sleeping".to_string(),
            queueing_handle_with_no_workers(0.0),
        )])
        .unwrap()
        .run(Box::new(WakeQueuedDeploymentHook {
            observation: Rc::clone(&observation),
        }))
        .unwrap();

        assert_eq!(
            observation.borrow().as_ref(),
            Some(&TickObservation {
                deployment_id: "sleeping".to_string(),
                now_ms: 10.0,
                num_req: 0,
                active_decode: 0,
            })
        );
        assert_eq!(report.deployments[0].1.request_counts.completed_requests, 1);
    }

    struct ScaleAndStopHook;

    impl WorldPlannerHook for ScaleAndStopHook {
        fn initial_tick_ms(&mut self, _deployment_id: &str) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_ticks(
            &mut self,
            ticks: Vec<(String, PlannerTickMetrics)>,
        ) -> anyhow::Result<WorldPlannerDecision> {
            assert_eq!(ticks.len(), 1);
            let deployment_id = ticks[0].0.clone();
            Ok(WorldPlannerDecision {
                next_ticks: vec![(deployment_id.clone(), None)],
                actions: vec![WorldScalingAction {
                    deployment_id,
                    target_prefill: None,
                    target_decode: Some(2),
                }],
            })
        }
    }

    #[test]
    fn pending_worker_startup_does_not_extend_world_past_request_quiescence() {
        let report =
            ReplayWorldHandle::from_deployments(vec![("startup".to_string(), startup_handle(0.0))])
                .unwrap()
                .run(Box::new(ScaleAndStopHook))
                .unwrap();

        assert_eq!(report.deployments[0].1.request_counts.completed_requests, 1);
        assert!(
            report.duration_ms < 1_000.0,
            "a startup-only lifecycle event must not keep the world alive: {}ms",
            report.duration_ms
        );
    }

    type RecordedTickBatches = Rc<RefCell<Vec<(f64, Vec<String>)>>>;

    struct RecurringHook {
        tick_ms: RecordedTickBatches,
        interval_ms: f64,
    }

    impl WorldPlannerHook for RecurringHook {
        fn initial_tick_ms(&mut self, _deployment_id: &str) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_ticks(
            &mut self,
            ticks: Vec<(String, PlannerTickMetrics)>,
        ) -> anyhow::Result<WorldPlannerDecision> {
            let now_ms = ticks[0].1.now_ms;
            let deployment_ids = ticks
                .iter()
                .map(|(deployment_id, _)| deployment_id.clone())
                .collect::<Vec<_>>();
            let next_ticks = deployment_ids
                .iter()
                .cloned()
                .map(|deployment_id| (deployment_id, Some(now_ms + self.interval_ms)))
                .collect();
            self.tick_ms.borrow_mut().push((now_ms, deployment_ids));
            Ok(WorldPlannerDecision {
                next_ticks,
                actions: Vec::new(),
            })
        }
    }

    #[test]
    fn idle_deployment_keeps_ticking_until_world_request_work_is_quiescent() {
        let tick_ms = Rc::new(RefCell::new(Vec::new()));
        let report = ReplayWorldHandle::from_deployments(vec![
            ("alpha".to_string(), handle(0.0)),
            ("beta".to_string(), handle(100.0)),
        ])
        .unwrap()
        .run(Box::new(RecurringHook {
            tick_ms: Rc::clone(&tick_ms),
            interval_ms: 10.0,
        }))
        .unwrap();

        let tick_ms = tick_ms.borrow();
        assert!(tick_ms.iter().any(|(now_ms, ids)| *now_ms > 10.0
            && *now_ms < 100.0
            && ids == &["alpha".to_string(), "beta".to_string()]));
        assert!(report.duration_ms >= 100.0);
        let (last_tick_ms, last_ids) = tick_ms.last().unwrap();
        assert_eq!(last_ids, &["alpha".to_string(), "beta".to_string()]);
        assert!(*last_tick_ms <= report.duration_ms);
        assert!(
            *last_tick_ms + 10.0 > report.duration_ms,
            "the world should terminate before the already-scheduled next heartbeat"
        );
        assert!(
            report
                .deployments
                .iter()
                .all(|(_, report)| report.request_counts.completed_requests == 1)
        );
    }

    struct InvalidNextTickHook;

    impl WorldPlannerHook for InvalidNextTickHook {
        fn initial_tick_ms(&mut self, _deployment_id: &str) -> anyhow::Result<f64> {
            Ok(0.0)
        }

        fn on_ticks(
            &mut self,
            ticks: Vec<(String, PlannerTickMetrics)>,
        ) -> anyhow::Result<WorldPlannerDecision> {
            Ok(WorldPlannerDecision {
                next_ticks: ticks
                    .into_iter()
                    .map(|(deployment_id, metrics)| (deployment_id, Some(metrics.now_ms)))
                    .collect(),
                actions: Vec::new(),
            })
        }
    }

    #[test]
    fn rejects_non_future_next_tick() {
        let error = ReplayWorldHandle::from_deployments(vec![("alpha".to_string(), handle(100.0))])
            .unwrap()
            .run(Box::new(InvalidNextTickHook))
            .unwrap_err();

        assert!(
            error.to_string().contains("strictly greater"),
            "unexpected error: {error:#}"
        );
    }
}
