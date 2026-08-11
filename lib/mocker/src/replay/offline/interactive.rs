// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public, polling-based control surface for causal offline replay.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use uuid::Uuid;

use crate::common::protocols::{EngineType, G1Backend, MockEngineArgs, WorkerType};
use crate::loadgen::{AgenticTrace, AgenticTurnTrace, WorkloadDriver};
use crate::replay::offline::agg::{ExternalAggRuntime, RoundRobinAggRuntime};
use crate::replay::offline::components::ReplayMode;
use crate::replay::offline::extensions::kv_router::AggRuntime as KvAggRuntime;
use crate::replay::{
    ReplayDeterminism, ReplayRouterMode, ReplayTerminalStatus, TraceSimulationReport,
    with_replay_determinism,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplaySessionRouter {
    External,
    RoundRobin,
    KvRouter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct WorkerTarget {
    pub worker_id: usize,
    pub dp_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayRequestSpec {
    pub logical_request_id: String,
    pub internal_uuid: Option<Uuid>,
    pub session_id: String,
    pub authored_turn_index: usize,
    pub ready_time_ms: f64,
    pub input_length: usize,
    pub hash_ids: Vec<u32>,
    pub trace_block_size: usize,
    pub output_length: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub target: Option<WorkerTarget>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayAgenticRequest {
    pub request: ReplayRequestSpec,
    pub wait_for: Vec<String>,
    pub dependency_delay_ms: f64,
    pub prefix_reset: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayAgenticWorkflow {
    pub trace_block_size: usize,
    pub requests: Vec<ReplayAgenticRequest>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayEventData {
    pub logical_request_id: String,
    pub internal_uuid: Uuid,
    pub session_id: String,
    pub authored_turn_index: usize,
    pub timestamp_ms: f64,
    pub worker_id: Option<usize>,
    pub dp_rank: Option<usize>,
    pub terminal_status: Option<ReplayTerminalStatus>,
    pub input_length: usize,
    pub requested_output_length: usize,
    pub emitted_output_count: usize,
    pub reused_input_tokens: Option<usize>,
    pub ttft_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event_type", content = "event", rename_all = "snake_case")]
pub enum ReplayEvent {
    PlacementNeeded(ReplayEventData),
    Routed(ReplayEventData),
    Queued(ReplayEventData),
    Admitted(ReplayEventData),
    FirstToken(ReplayEventData),
    Terminal(ReplayEventData),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReplayStepStatus {
    Advanced { now_ms: f64 },
    Quiescent { now_ms: f64 },
    Drained { now_ms: f64 },
}

impl ReplayStepStatus {
    pub fn now_ms(self) -> f64 {
        match self {
            Self::Advanced { now_ms } | Self::Quiescent { now_ms } | Self::Drained { now_ms } => {
                now_ms
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayPendingPlacement {
    pub logical_request_id: String,
    pub internal_uuid: Uuid,
    pub session_id: String,
    pub authored_turn_index: usize,
    pub ready_at_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayWorkerSnapshot {
    pub worker_id: usize,
    pub dp_rank: usize,
    pub active: bool,
    pub draining: bool,
    pub in_flight_requests: usize,
    pub queued_requests: Option<usize>,
    pub queued_tokens: Option<usize>,
    pub running_tokens: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplaySnapshot {
    pub now_ms: f64,
    pub admission_open: bool,
    pub pending_request_count: usize,
    pub pending_placement_count: usize,
    pub workers: Vec<ReplayWorkerSnapshot>,
}

#[derive(Debug, Clone)]
pub(crate) struct InteractiveRequestIdentity {
    pub logical_request_id: String,
    pub internal_uuid: Uuid,
    pub session_id: String,
    pub authored_turn_index: usize,
    pub input_length: usize,
    pub requested_output_length: usize,
    pub ready_at_ms: Option<f64>,
    pub worker: Option<WorkerTarget>,
    pub emitted_output_count: usize,
    pub reused_input_tokens: Option<usize>,
    pub first_token_ms: Option<f64>,
    pub terminal_ms: Option<f64>,
}

#[derive(Debug, Default)]
pub(crate) struct InteractiveCapture {
    external_placement: bool,
    identities: FxHashMap<Uuid, InteractiveRequestIdentity>,
    logical_to_uuid: FxHashMap<String, Uuid>,
    session_turn_to_uuid: FxHashMap<(String, usize), Uuid>,
    events: VecDeque<ReplayEvent>,
}

impl InteractiveCapture {
    pub(crate) fn new(external_placement: bool) -> Self {
        Self {
            external_placement,
            ..Self::default()
        }
    }

    pub(crate) fn uses_external_placement(&self) -> bool {
        self.external_placement
    }

    pub(crate) fn register(&mut self, identity: InteractiveRequestIdentity) -> anyhow::Result<()> {
        if identity.logical_request_id.trim().is_empty() {
            anyhow::bail!("interactive replay logical_request_id must not be empty");
        }
        if self
            .logical_to_uuid
            .contains_key(&identity.logical_request_id)
        {
            anyhow::bail!(
                "interactive replay duplicate logical_request_id {:?}",
                identity.logical_request_id
            );
        }
        let session_turn = (identity.session_id.clone(), identity.authored_turn_index);
        if let Some(existing) = self.session_turn_to_uuid.get(&session_turn) {
            anyhow::bail!(
                "interactive replay session {:?} authored turn {} conflicts with internal UUID {existing}",
                session_turn.0,
                session_turn.1
            );
        }
        if self.identities.contains_key(&identity.internal_uuid) {
            anyhow::bail!(
                "interactive replay duplicate internal UUID {}",
                identity.internal_uuid
            );
        }
        self.logical_to_uuid
            .insert(identity.logical_request_id.clone(), identity.internal_uuid);
        self.session_turn_to_uuid
            .insert(session_turn, identity.internal_uuid);
        self.identities.insert(identity.internal_uuid, identity);
        Ok(())
    }

    pub(crate) fn unregister(&mut self, uuid: Uuid) {
        if let Some(identity) = self.identities.remove(&uuid) {
            self.logical_to_uuid.remove(&identity.logical_request_id);
            self.session_turn_to_uuid
                .remove(&(identity.session_id, identity.authored_turn_index));
        }
    }

    pub(crate) fn uuid_for_logical_id(&self, logical_id: &str) -> Option<Uuid> {
        self.logical_to_uuid.get(logical_id).copied()
    }

    pub(crate) fn identity(&self, uuid: Uuid) -> Option<&InteractiveRequestIdentity> {
        self.identities.get(&uuid)
    }

    fn event_data(&self, uuid: Uuid, timestamp_ms: f64) -> anyhow::Result<ReplayEventData> {
        let identity = self.identities.get(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        let (worker_id, dp_rank) = identity
            .worker
            .map(|target| (Some(target.worker_id), Some(target.dp_rank)))
            .unwrap_or((None, None));
        Ok(ReplayEventData {
            logical_request_id: identity.logical_request_id.clone(),
            internal_uuid: identity.internal_uuid,
            session_id: identity.session_id.clone(),
            authored_turn_index: identity.authored_turn_index,
            timestamp_ms,
            worker_id,
            dp_rank,
            terminal_status: None,
            input_length: identity.input_length,
            requested_output_length: identity.requested_output_length,
            emitted_output_count: identity.emitted_output_count,
            reused_input_tokens: identity.reused_input_tokens,
            ttft_ms: identity
                .first_token_ms
                .map(|first| (first - identity.ready_at_ms.unwrap_or(first)).max(0.0)),
            e2e_latency_ms: identity
                .terminal_ms
                .map(|terminal| (terminal - identity.ready_at_ms.unwrap_or(terminal)).max(0.0)),
        })
    }

    pub(crate) fn mark_ready(&mut self, uuid: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let identity = self.identities.get_mut(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        identity.ready_at_ms.get_or_insert(now_ms);
        Ok(())
    }

    pub(crate) fn set_worker(&mut self, uuid: Uuid, target: WorkerTarget) -> anyhow::Result<()> {
        let identity = self.identities.get_mut(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        identity.worker = Some(target);
        Ok(())
    }

    pub(crate) fn emit_placement_needed(&mut self, uuid: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let data = self.event_data(uuid, now_ms)?;
        self.events.push_back(ReplayEvent::PlacementNeeded(data));
        Ok(())
    }

    pub(crate) fn emit_routed(&mut self, uuid: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let data = self.event_data(uuid, now_ms)?;
        self.events.push_back(ReplayEvent::Routed(data));
        Ok(())
    }

    pub(crate) fn emit_queued(&mut self, uuid: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let data = self.event_data(uuid, now_ms)?;
        self.events.push_back(ReplayEvent::Queued(data));
        Ok(())
    }

    pub(crate) fn emit_admitted(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        reused_input_tokens: usize,
    ) -> anyhow::Result<()> {
        let identity = self.identities.get_mut(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        identity.reused_input_tokens = Some(
            identity
                .reused_input_tokens
                .unwrap_or_default()
                .max(reused_input_tokens),
        );
        let data = self.event_data(uuid, now_ms)?;
        self.events.push_back(ReplayEvent::Admitted(data));
        Ok(())
    }

    pub(crate) fn on_output_token(&mut self, uuid: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let identity = self.identities.get_mut(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        identity.emitted_output_count += 1;
        if identity.first_token_ms.is_some() {
            return Ok(());
        }
        identity.first_token_ms = Some(now_ms);
        let data = self.event_data(uuid, now_ms)?;
        self.events.push_back(ReplayEvent::FirstToken(data));
        Ok(())
    }

    pub(crate) fn emit_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> anyhow::Result<()> {
        let identity = self.identities.get_mut(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        if identity.terminal_ms.replace(now_ms).is_some() {
            anyhow::bail!("interactive replay emitted duplicate terminal for request {uuid}");
        }
        let mut data = self.event_data(uuid, now_ms)?;
        data.terminal_status = Some(status);
        self.events.push_back(ReplayEvent::Terminal(data));
        Ok(())
    }

    pub(crate) fn drain_events(&mut self) -> Vec<ReplayEvent> {
        self.events.drain(..).collect()
    }

    pub(crate) fn pending(&self, uuids: impl Iterator<Item = Uuid>) -> Vec<ReplayPendingPlacement> {
        uuids
            .filter_map(|uuid| {
                let identity = self.identities.get(&uuid)?;
                Some(ReplayPendingPlacement {
                    logical_request_id: identity.logical_request_id.clone(),
                    internal_uuid: uuid,
                    session_id: identity.session_id.clone(),
                    authored_turn_index: identity.authored_turn_index,
                    ready_at_ms: identity.ready_at_ms.unwrap_or_default(),
                })
            })
            .collect()
    }
}

enum InteractiveRuntime {
    External(ExternalAggRuntime),
    RoundRobin(RoundRobinAggRuntime),
    KvRouter(Box<KvAggRuntime>),
}

impl InteractiveRuntime {
    fn register_identity(&mut self, identity: InteractiveRequestIdentity) -> Result<()> {
        match self {
            Self::External(runtime) => runtime.register_interactive_identity(identity),
            Self::RoundRobin(runtime) => runtime.register_interactive_identity(identity),
            Self::KvRouter(runtime) => runtime.register_interactive_identity(identity),
        }
    }

    fn unregister_identity(&mut self, uuid: Uuid) {
        match self {
            Self::External(runtime) => runtime.unregister_interactive_identity(uuid),
            Self::RoundRobin(runtime) => runtime.unregister_interactive_identity(uuid),
            Self::KvRouter(runtime) => runtime.unregister_interactive_identity(uuid),
        }
    }

    fn append(&mut self, trace: AgenticTrace, release_at_ms: f64) -> Result<()> {
        match self {
            Self::External(runtime) => {
                runtime.append_interactive_agentic_trace(trace, release_at_ms)
            }
            Self::RoundRobin(runtime) => {
                runtime.append_interactive_agentic_trace(trace, release_at_ms)
            }
            Self::KvRouter(runtime) => {
                runtime.append_interactive_agentic_trace(trace, release_at_ms)
            }
        }
    }

    fn close(&mut self) -> Result<()> {
        match self {
            Self::External(runtime) => runtime.interactive_close_admission(),
            Self::RoundRobin(runtime) => runtime.interactive_close_admission(),
            Self::KvRouter(runtime) => runtime.interactive_close_admission(),
        }
    }

    fn now_ms(&self) -> f64 {
        match self {
            Self::External(runtime) => runtime.interactive_now_ms(),
            Self::RoundRobin(runtime) => runtime.interactive_now_ms(),
            Self::KvRouter(runtime) => runtime.interactive_now_ms(),
        }
    }

    fn next_event_time_ms(&mut self) -> Option<f64> {
        match self {
            Self::External(runtime) => runtime.interactive_next_event_time_ms(),
            Self::RoundRobin(runtime) => runtime.interactive_next_event_time_ms(),
            Self::KvRouter(runtime) => runtime.interactive_next_event_time_ms(),
        }
    }

    fn advance_next(&mut self) -> Result<ReplayStepStatus> {
        match self {
            Self::External(runtime) => runtime.interactive_advance_next(),
            Self::RoundRobin(runtime) => runtime.interactive_advance_next(),
            Self::KvRouter(runtime) => runtime.interactive_advance_next(),
        }
    }

    fn advance_to(&mut self, target_ms: f64) -> Result<ReplayStepStatus> {
        match self {
            Self::External(runtime) => runtime.interactive_advance_to(target_ms),
            Self::RoundRobin(runtime) => runtime.interactive_advance_to(target_ms),
            Self::KvRouter(runtime) => runtime.interactive_advance_to(target_ms),
        }
    }

    fn settle_current_time(&mut self) -> Result<ReplayStepStatus> {
        match self {
            Self::External(runtime) => runtime.interactive_settle_current_time(),
            Self::RoundRobin(runtime) => runtime.interactive_settle_current_time(),
            Self::KvRouter(runtime) => runtime.interactive_settle_current_time(),
        }
    }

    fn drain_events(&mut self) -> Vec<ReplayEvent> {
        match self {
            Self::External(runtime) => runtime.drain_interactive_events(),
            Self::RoundRobin(runtime) => runtime.drain_interactive_events(),
            Self::KvRouter(runtime) => runtime.drain_interactive_events(),
        }
    }

    fn snapshot(&self) -> ReplaySnapshot {
        match self {
            Self::External(runtime) => runtime.interactive_snapshot(),
            Self::RoundRobin(runtime) => runtime.interactive_snapshot(),
            Self::KvRouter(runtime) => runtime.interactive_snapshot(),
        }
    }

    fn is_quiescent(&mut self) -> bool {
        match self {
            Self::External(runtime) => runtime.interactive_is_quiescent(),
            Self::RoundRobin(runtime) => runtime.interactive_is_quiescent(),
            Self::KvRouter(runtime) => runtime.interactive_is_quiescent(),
        }
    }

    fn is_drained(&self) -> bool {
        match self {
            Self::External(runtime) => runtime.interactive_is_drained(),
            Self::RoundRobin(runtime) => runtime.interactive_is_drained(),
            Self::KvRouter(runtime) => runtime.interactive_is_drained(),
        }
    }

    fn uuid_for_logical_id(&self, logical_id: &str) -> Option<Uuid> {
        match self {
            Self::External(runtime) => runtime.interactive_uuid_for_logical_id(logical_id),
            Self::RoundRobin(runtime) => runtime.interactive_uuid_for_logical_id(logical_id),
            Self::KvRouter(runtime) => runtime.interactive_uuid_for_logical_id(logical_id),
        }
    }

    fn pending_placements(&self) -> Vec<ReplayPendingPlacement> {
        match self {
            Self::External(runtime) => runtime.pending_interactive_placements(),
            Self::RoundRobin(_) | Self::KvRouter(_) => Vec::new(),
        }
    }

    fn preassign(&mut self, uuid: Uuid, target: WorkerTarget) -> Result<()> {
        match self {
            Self::External(runtime) => runtime.preassign_interactive(uuid, target),
            Self::RoundRobin(_) | Self::KvRouter(_) => {
                bail!("explicit WorkerTarget is only valid with external placement")
            }
        }
    }

    fn cancel_preassignment(&mut self, uuid: Uuid) {
        if let Self::External(runtime) = self {
            runtime.cancel_interactive_placement(uuid);
        }
    }

    fn assign(&mut self, uuid: Uuid, target: WorkerTarget) -> Result<()> {
        match self {
            Self::External(runtime) => runtime.assign_interactive(uuid, target),
            Self::RoundRobin(_) | Self::KvRouter(_) => {
                bail!("assign() is only valid with external placement")
            }
        }
    }

    fn finish(self) -> TraceSimulationReport {
        match self {
            Self::External(runtime) => runtime.finish_interactive(),
            Self::RoundRobin(runtime) => runtime.finish_interactive(),
            Self::KvRouter(runtime) => (*runtime).finish_interactive(),
        }
    }
}

/// One long-lived aggregated replay session driven by an external virtual-time
/// controller. Scheduler, router, and admission implementation types remain
/// private behind this concrete public type.
pub struct OfflineReplaySession {
    runtime: Option<InteractiveRuntime>,
    router: ReplaySessionRouter,
    determinism: ReplayDeterminism,
    trace_block_size: usize,
    admission_closed: bool,
}

impl OfflineReplaySession {
    pub fn new(
        args: &MockEngineArgs,
        num_workers: usize,
        trace_block_size: usize,
        router: ReplaySessionRouter,
    ) -> Result<Self> {
        Self::new_with_determinism(
            args,
            num_workers,
            trace_block_size,
            router,
            ReplayDeterminism::CanonicalV1,
        )
    }

    pub fn new_with_determinism(
        args: &MockEngineArgs,
        num_workers: usize,
        trace_block_size: usize,
        router: ReplaySessionRouter,
        determinism: ReplayDeterminism,
    ) -> Result<Self> {
        if num_workers == 0 {
            bail!("interactive replay requires at least one worker");
        }
        if trace_block_size == 0 {
            bail!("interactive replay trace_block_size must be greater than zero");
        }
        let args = args.clone().normalized()?;
        if args.engine_type != EngineType::Vllm {
            bail!("interactive replay currently supports only the vLLM mock engine");
        }
        if args.worker_type != WorkerType::Aggregated {
            bail!("interactive replay currently supports only aggregated workers");
        }
        if args.dp_size != 1 {
            bail!(
                "interactive replay milestone requires dp_size=1, got {}",
                args.dp_size
            );
        }
        if args.resolved_g1_backend() != G1Backend::Native {
            bail!("interactive replay milestone does not support KVBM/offload");
        }

        let engine_block_size = args.block_size;
        let mut runtime = with_replay_determinism(determinism, || -> Result<_> {
            Ok(match router {
                ReplaySessionRouter::External => {
                    let driver = WorkloadDriver::new_open_agentic_without_replay_hashes(
                        trace_block_size,
                        engine_block_size,
                    )?;
                    InteractiveRuntime::External(ExternalAggRuntime::new_external_workload(
                        &args,
                        driver,
                        num_workers,
                    )?)
                }
                ReplaySessionRouter::RoundRobin => {
                    let driver = WorkloadDriver::new_open_agentic_without_replay_hashes(
                        trace_block_size,
                        engine_block_size,
                    )?;
                    InteractiveRuntime::RoundRobin(RoundRobinAggRuntime::new_round_robin_workload(
                        &args,
                        driver,
                        num_workers,
                        ReplayMode::Trace,
                    )?)
                }
                ReplaySessionRouter::KvRouter => {
                    let driver =
                        WorkloadDriver::new_open_agentic(trace_block_size, engine_block_size)?;
                    InteractiveRuntime::KvRouter(Box::new(KvAggRuntime::new_workload(
                        &args,
                        None,
                        None,
                        driver,
                        num_workers,
                        ReplayMode::Trace,
                        ReplayRouterMode::KvRouter,
                    )?))
                }
            })
        })?;
        match &mut runtime {
            InteractiveRuntime::External(runtime) => runtime.enable_interactive_capture(true),
            InteractiveRuntime::RoundRobin(runtime) => runtime.enable_interactive_capture(false),
            InteractiveRuntime::KvRouter(runtime) => runtime.enable_interactive_capture(false),
        }
        Ok(Self {
            runtime: Some(runtime),
            router,
            determinism,
            trace_block_size,
            admission_closed: false,
        })
    }

    fn runtime(&self) -> Result<&InteractiveRuntime> {
        self.runtime
            .as_ref()
            .context("interactive replay session was already finalized")
    }

    fn runtime_mut(&mut self) -> Result<&mut InteractiveRuntime> {
        self.runtime
            .as_mut()
            .context("interactive replay session was already finalized")
    }

    fn deterministic_uuid(logical_request_id: &str) -> Uuid {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"dynamo-offline-interactive-request-v1\0");
        hasher.update(logical_request_id.as_bytes());
        let digest = hasher.finalize();
        let mut bytes = [0_u8; 16];
        bytes.copy_from_slice(&digest.as_bytes()[..16]);
        Uuid::from_bytes(bytes)
    }

    fn validate_request(&self, request: &ReplayRequestSpec) -> Result<()> {
        if request.logical_request_id.trim().is_empty() {
            bail!("interactive replay logical_request_id must not be empty");
        }
        if request.session_id.trim().is_empty() {
            bail!(
                "interactive replay request {} has an empty session_id",
                request.logical_request_id
            );
        }
        if !request.ready_time_ms.is_finite() || request.ready_time_ms < 0.0 {
            bail!(
                "interactive replay request {} has invalid ready_time_ms {}",
                request.logical_request_id,
                request.ready_time_ms
            );
        }
        if request.trace_block_size != self.trace_block_size {
            bail!(
                "interactive replay request {} trace_block_size {} does not match session block size {}",
                request.logical_request_id,
                request.trace_block_size,
                self.trace_block_size
            );
        }
        let required_hashes = request.input_length.div_ceil(self.trace_block_size);
        if request.hash_ids.len() < required_hashes {
            bail!(
                "interactive replay request {} input_length {} exceeds hash capacity {}",
                request.logical_request_id,
                request.input_length,
                request.hash_ids.len().saturating_mul(self.trace_block_size)
            );
        }
        if let Some(output_token_ids) = request.output_token_ids.as_ref()
            && output_token_ids.len() != request.output_length
        {
            bail!(
                "interactive replay request {} output_length {} does not match {} output_token_ids",
                request.logical_request_id,
                request.output_length,
                output_token_ids.len()
            );
        }
        if request.target.is_some() && self.router != ReplaySessionRouter::External {
            bail!(
                "interactive replay request {} supplies a WorkerTarget for native routing",
                request.logical_request_id
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use serde_json::Value;

    const TRACE_BLOCK_SIZE: usize = 4;
    const MAX_STEPS: usize = 10_000;

    fn replay_args(enable_prefix_caching: bool) -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(TRACE_BLOCK_SIZE)
            .num_gpu_blocks(128)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(4))
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn constrained_replay_args() -> MockEngineArgs {
        MockEngineArgs::builder()
            .block_size(TRACE_BLOCK_SIZE)
            .num_gpu_blocks(4)
            .max_num_batched_tokens(Some(16))
            .max_num_seqs(Some(2))
            .enable_prefix_caching(false)
            .enable_chunked_prefill(true)
            .speedup_ratio(0.0)
            .build()
            .unwrap()
    }

    fn session(
        router: ReplaySessionRouter,
        num_workers: usize,
        enable_prefix_caching: bool,
    ) -> OfflineReplaySession {
        OfflineReplaySession::new(
            &replay_args(enable_prefix_caching),
            num_workers,
            TRACE_BLOCK_SIZE,
            router,
        )
        .unwrap()
    }

    fn request(
        logical_request_id: &str,
        session_id: &str,
        authored_turn_index: usize,
        input_length: usize,
        hash_ids: &[u32],
        output_length: usize,
    ) -> ReplayRequestSpec {
        ReplayRequestSpec {
            logical_request_id: logical_request_id.to_string(),
            internal_uuid: None,
            session_id: session_id.to_string(),
            authored_turn_index,
            ready_time_ms: 0.0,
            input_length,
            hash_ids: hash_ids.to_vec(),
            trace_block_size: TRACE_BLOCK_SIZE,
            output_length,
            output_token_ids: Some(
                (0..output_length)
                    .map(|index| 50_000_u32 + index as u32)
                    .collect(),
            ),
            priority: 0,
            strict_priority: 0,
            policy_class: None,
            target: None,
        }
    }

    fn agentic_request(
        request: ReplayRequestSpec,
        wait_for: &[&str],
        dependency_delay_ms: f64,
    ) -> ReplayAgenticRequest {
        ReplayAgenticRequest {
            request,
            wait_for: wait_for
                .iter()
                .map(|dependency| dependency.to_string())
                .collect(),
            dependency_delay_ms,
            prefix_reset: false,
        }
    }

    fn terminal_count(events: &[ReplayEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, ReplayEvent::Terminal(_)))
            .count()
    }

    fn terminal_event<'a>(events: &'a [ReplayEvent], logical_id: &str) -> &'a ReplayEventData {
        events
            .iter()
            .find_map(|event| match event {
                ReplayEvent::Terminal(data) if data.logical_request_id == logical_id => Some(data),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing terminal event for {logical_id}"))
    }

    fn routed_event<'a>(events: &'a [ReplayEvent], logical_id: &str) -> &'a ReplayEventData {
        events
            .iter()
            .find_map(|event| match event {
                ReplayEvent::Routed(data) if data.logical_request_id == logical_id => Some(data),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing routed event for {logical_id}"))
    }

    fn admitted_reuse(events: &[ReplayEvent], logical_id: &str) -> usize {
        events
            .iter()
            .filter_map(|event| match event {
                ReplayEvent::Admitted(data) if data.logical_request_id == logical_id => {
                    data.reused_input_tokens
                }
                _ => None,
            })
            .max()
            .unwrap_or_else(|| panic!("missing admitted event for {logical_id}"))
    }

    fn settle_empty_open(session: &mut OfflineReplaySession) -> Result<ReplayStepStatus> {
        for _ in 0..32 {
            let status = session.settle_current_time()?;
            assert!(session.drain_events()?.is_empty());
            if !matches!(status, ReplayStepStatus::Advanced { .. }) {
                return Ok(status);
            }
        }
        panic!("empty interactive replay did not settle")
    }

    fn drive_to_terminal(
        session: &mut OfflineReplaySession,
        logical_id: &str,
    ) -> Result<Vec<ReplayEvent>> {
        let mut events = Vec::new();
        for _ in 0..MAX_STEPS {
            session.settle_current_time()?;
            events.extend(session.drain_events()?);
            if events.iter().any(|event| {
                matches!(event, ReplayEvent::Terminal(data) if data.logical_request_id == logical_id)
            }) {
                return Ok(events);
            }

            let status = session.advance_next()?;
            events.extend(session.drain_events()?);
            if events.iter().any(|event| {
                matches!(event, ReplayEvent::Terminal(data) if data.logical_request_id == logical_id)
            }) {
                return Ok(events);
            }
            if matches!(
                status,
                ReplayStepStatus::Quiescent { .. } | ReplayStepStatus::Drained { .. }
            ) {
                anyhow::bail!(
                    "interactive replay stopped before request {logical_id} reached a terminal state"
                );
            }
        }
        anyhow::bail!("interactive replay exceeded {MAX_STEPS} steps")
    }

    fn drive_to_first_token(
        session: &mut OfflineReplaySession,
        logical_id: &str,
    ) -> Result<Vec<ReplayEvent>> {
        let mut events = Vec::new();
        for _ in 0..MAX_STEPS {
            session.settle_current_time()?;
            events.extend(session.drain_events()?);
            if events.iter().any(|event| {
                matches!(event, ReplayEvent::FirstToken(data)
                    if data.logical_request_id == logical_id)
            }) {
                return Ok(events);
            }

            let status = session.advance_next()?;
            events.extend(session.drain_events()?);
            if events.iter().any(|event| {
                matches!(event, ReplayEvent::FirstToken(data)
                    if data.logical_request_id == logical_id)
            }) {
                return Ok(events);
            }
            if matches!(
                status,
                ReplayStepStatus::Quiescent { .. } | ReplayStepStatus::Drained { .. }
            ) {
                anyhow::bail!(
                    "interactive replay stopped before request {logical_id} emitted a token"
                );
            }
        }
        anyhow::bail!("interactive replay exceeded {MAX_STEPS} steps")
    }

    fn drive_to_pending_placement(session: &mut OfflineReplaySession) -> Result<Vec<ReplayEvent>> {
        let mut events = Vec::new();
        for _ in 0..32 {
            session.settle_current_time()?;
            events.extend(session.drain_events()?);
            if !session.pending_placements()?.is_empty() {
                return Ok(events);
            }
            session.advance_next()?;
            events.extend(session.drain_events()?);
            if !session.pending_placements()?.is_empty() {
                return Ok(events);
            }
        }
        anyhow::bail!("interactive replay never requested external placement")
    }

    fn drive_to_drained(session: &mut OfflineReplaySession) -> Result<Vec<ReplayEvent>> {
        let mut events = Vec::new();
        for _ in 0..MAX_STEPS {
            session.settle_current_time()?;
            events.extend(session.drain_events()?);
            if session.is_drained()? {
                return Ok(events);
            }

            let status = session.advance_next()?;
            events.extend(session.drain_events()?);
            if session.is_drained()? {
                return Ok(events);
            }
            if matches!(status, ReplayStepStatus::Quiescent { .. }) {
                anyhow::bail!("closed interactive replay became quiescent before draining");
            }
        }
        anyhow::bail!("interactive replay exceeded {MAX_STEPS} steps")
    }

    fn semantic_report_json(report: &TraceSimulationReport) -> (Value, Value) {
        let mut summary = serde_json::to_value(report).unwrap();
        let object = summary.as_object_mut().unwrap();
        for excluded in [
            "wall_time_ms",
            "processed_tokens_per_s",
            "processed_output_tokens_per_s",
        ] {
            let _ = object.remove(excluded);
        }
        let per_request = serde_json::to_value(&report.per_request).unwrap();
        (summary, per_request)
    }

    #[test]
    fn empty_open_is_quiescent_then_close_drains_and_finalize_consumes_session() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, false);
        assert!(matches!(
            settle_empty_open(&mut replay)?,
            ReplayStepStatus::Quiescent { now_ms: 0.0 }
        ));
        assert!(replay.is_quiescent()?);
        assert!(!replay.is_drained()?);
        assert_eq!(replay.next_event_time_ms()?, None);
        assert_eq!(replay.snapshot()?.pending_request_count, 0);
        assert!(
            replay
                .finalize()
                .unwrap_err()
                .to_string()
                .contains("admission remains open")
        );

        replay.close_admission()?;
        assert!(drive_to_drained(&mut replay)?.is_empty());
        let report = replay.finalize()?;
        assert_eq!(report.request_counts.num_requests, 0);
        assert_eq!(report.request_counts.completed_requests, 0);

        assert!(replay.now_ms().is_err());
        assert!(replay.snapshot().is_err());
        assert!(replay.drain_events().is_err());
        assert!(replay.finalize().is_err());
        assert!(
            replay.close_admission().is_err(),
            "every public operation must reject use after finalize"
        );
        Ok(())
    }

    #[test]
    fn future_submit_wakes_quiescent_session_and_advance_to_stops_at_arrival() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, false);
        settle_empty_open(&mut replay)?;

        let mut future = request("future", "future-session", 0, 4, &[11], 2);
        future.ready_time_ms = 50.0;
        replay.submit_request(future)?;
        assert_eq!(replay.next_event_time_ms()?, Some(50.0));

        let status = replay.advance_to(100.0)?;
        assert!(matches!(
            status,
            ReplayStepStatus::Advanced { now_ms: 50.0 }
        ));
        assert_eq!(replay.now_ms()?, 50.0);
        let mut events = replay.drain_events()?;
        assert_eq!(routed_event(&events, "future").timestamp_ms, 50.0);

        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 1);
        let report = replay.finalize()?;
        assert_eq!(report.request_counts.completed_requests, 1);
        Ok(())
    }

    #[test]
    fn same_time_dynamic_insertion_order_is_stable_and_event_drain_preserves_report() -> Result<()>
    {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 2, false);
        settle_empty_open(&mut replay)?;
        for (logical_id, hash) in [("first", 1), ("second", 2), ("third", 3)] {
            let mut submitted = request(logical_id, logical_id, 0, 4, &[hash], 1);
            submitted.ready_time_ms = 10.0;
            replay.submit_request(submitted)?;
        }
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;

        let routed = events
            .iter()
            .filter_map(|event| match event {
                ReplayEvent::Routed(data) if data.timestamp_ms == 10.0 => {
                    Some((data.logical_request_id.as_str(), data.worker_id))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            routed,
            vec![("first", Some(0)), ("second", Some(1)), ("third", Some(0))]
        );
        assert_eq!(terminal_count(&events), 3);
        assert!(replay.drain_events()?.is_empty());

        let report = replay.finalize()?;
        assert_eq!(report.request_counts.num_requests, 3);
        assert_eq!(report.request_counts.completed_requests, 3);
        assert_eq!(report.per_request.len(), terminal_count(&events));
        Ok(())
    }

    fn first_request_terminal_with_optional_late_arrival(
        include_late_arrival: bool,
    ) -> Result<(f64, Vec<ReplayEvent>)> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, false);
        replay.submit_request(request("long", "long", 0, 8, &[10, 11], 32))?;
        let mut events = drive_to_first_token(&mut replay, "long")?;
        let injection_time = replay.now_ms()?;

        if include_late_arrival {
            let before = replay.snapshot()?;
            assert_eq!(before.workers.len(), 1);
            assert_eq!(before.workers[0].in_flight_requests, 1);
            let hashes = (100_u32..116).collect::<Vec<_>>();
            let mut late = request("late", "late", 0, 64, &hashes, 2);
            late.ready_time_ms = injection_time;
            replay.submit_request(late)?;
            let snapshot = replay.snapshot()?;
            assert_eq!(snapshot.workers.len(), 1);
            assert_eq!(snapshot.workers[0].in_flight_requests, 1);
            assert_eq!(
                snapshot.pending_request_count,
                before.pending_request_count + 1
            );
        }

        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        let long_terminal = terminal_event(&events, "long").timestamp_ms;
        if include_late_arrival {
            let late_route = routed_event(&events, "late");
            assert_eq!(late_route.timestamp_ms, injection_time);
            assert!(late_route.timestamp_ms < long_terminal);
            assert_eq!(terminal_count(&events), 2);
        } else {
            assert_eq!(terminal_count(&events), 1);
        }
        let report = replay.finalize()?;
        assert_eq!(report.per_request.len(), terminal_count(&events));
        Ok((long_terminal, events))
    }

    #[test]
    fn late_arrival_changes_live_batch_completion_without_resetting_prior_state() -> Result<()> {
        let (isolated_terminal, _) = first_request_terminal_with_optional_late_arrival(false)?;
        let (contended_terminal, events) = first_request_terminal_with_optional_late_arrival(true)?;
        assert!(
            contended_terminal > isolated_terminal,
            "late overlapping work must change the already-running request's completion time"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ReplayEvent::Routed(data)
                    if data.logical_request_id == "long"))
                .count(),
            1,
            "appending work must not reset or redispatch existing engine state"
        );
        Ok(())
    }

    #[test]
    fn duplicate_identity_time_reversal_and_enqueue_after_close_are_rejected() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, false);
        assert!(matches!(
            replay.advance_to(10.0)?,
            ReplayStepStatus::Advanced { now_ms: 10.0 }
        ));
        let mut original = request("original", "session", 7, 4, &[1], 1);
        original.ready_time_ms = 10.0;
        original.internal_uuid = Some(Uuid::from_u128(1));
        replay.submit_request(original)?;

        let mut duplicate_logical = request("original", "other", 0, 4, &[2], 1);
        duplicate_logical.ready_time_ms = 10.0;
        duplicate_logical.internal_uuid = Some(Uuid::from_u128(2));
        assert!(
            replay
                .submit_request(duplicate_logical)
                .unwrap_err()
                .to_string()
                .contains("duplicate logical_request_id")
        );

        let mut duplicate_turn = request("other-logical", "session", 7, 4, &[3], 1);
        duplicate_turn.ready_time_ms = 10.0;
        duplicate_turn.internal_uuid = Some(Uuid::from_u128(3));
        assert!(
            replay
                .submit_request(duplicate_turn)
                .unwrap_err()
                .to_string()
                .contains("authored turn 7 conflicts")
        );

        let mut duplicate_uuid = request("third-logical", "third", 0, 4, &[4], 1);
        duplicate_uuid.ready_time_ms = 10.0;
        duplicate_uuid.internal_uuid = Some(Uuid::from_u128(1));
        assert!(
            replay
                .submit_request(duplicate_uuid)
                .unwrap_err()
                .to_string()
                .contains("duplicate internal UUID")
        );
        let mut empty_id = request("", "empty", 0, 4, &[5], 1);
        empty_id.ready_time_ms = 10.0;
        assert!(
            replay
                .submit_request(empty_id)
                .unwrap_err()
                .to_string()
                .contains("must not be empty")
        );

        assert!(replay.advance_to(9.0).is_err());
        let mut late = request("late", "late", 0, 4, &[6], 1);
        late.ready_time_ms = 9.0;
        assert!(
            replay
                .submit_request(late)
                .unwrap_err()
                .to_string()
                .contains("before current time")
        );

        replay.close_admission()?;
        assert!(
            replay
                .submit_request(request("after-close", "after-close", 0, 4, &[7], 1))
                .unwrap_err()
                .to_string()
                .contains("admission is closed")
        );
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 1);
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn external_placement_needed_assign_validates_worker_and_rank() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        settle_empty_open(&mut replay)?;
        replay.submit_request(request("external", "external-session", 0, 4, &[10], 2))?;

        let mut events = drive_to_pending_placement(&mut replay)?;
        let pending = replay.pending_placements()?;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].logical_request_id, "external");
        assert!(events.iter().any(|event| {
            matches!(event, ReplayEvent::PlacementNeeded(data)
                if data.logical_request_id == "external"
                    && data.worker_id.is_none()
                    && data.dp_rank.is_none())
        }));

        let decision_time = replay.now_ms()?;
        assert_eq!(replay.next_event_time_ms()?, None);
        assert!(matches!(
            replay.advance_to(decision_time + 100.0)?,
            ReplayStepStatus::Quiescent { now_ms } if now_ms == decision_time
        ));
        assert_eq!(replay.now_ms()?, decision_time);

        assert!(
            replay
                .assign(
                    "unknown",
                    WorkerTarget {
                        worker_id: 0,
                        dp_rank: 0
                    }
                )
                .is_err()
        );
        assert!(
            replay
                .assign(
                    "external",
                    WorkerTarget {
                        worker_id: 99,
                        dp_rank: 0,
                    },
                )
                .unwrap_err()
                .to_string()
                .contains("unavailable")
        );
        assert_eq!(replay.pending_placements()?.len(), 1);
        assert!(
            replay
                .assign(
                    "external",
                    WorkerTarget {
                        worker_id: 0,
                        dp_rank: 1,
                    },
                )
                .unwrap_err()
                .to_string()
                .contains("no active DP rank 1")
        );
        assert_eq!(replay.pending_placements()?.len(), 1);

        replay.assign(
            "external",
            WorkerTarget {
                worker_id: 1,
                dp_rank: 0,
            },
        )?;
        assert!(replay.pending_placements()?.is_empty());
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 1);
        assert_eq!(routed_event(&events, "external").worker_id, Some(1));
        assert_eq!(terminal_event(&events, "external").worker_id, Some(1));
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn pinned_target_emits_exactly_once_terminal_and_preserves_final_identity() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        settle_empty_open(&mut replay)?;

        let uuid = Uuid::from_u128(0xabc);
        let mut pinned = request("pinned", "pinned-session", 4, 8, &[20, 21], 3);
        pinned.internal_uuid = Some(uuid);
        pinned.target = Some(WorkerTarget {
            worker_id: 99,
            dp_rank: 0,
        });
        assert!(replay.submit_request(pinned.clone()).is_err());

        pinned.target = Some(WorkerTarget {
            worker_id: 1,
            dp_rank: 0,
        });
        replay.submit_request(pinned)?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert!(!events.iter().any(|event| {
            matches!(event, ReplayEvent::PlacementNeeded(data) if data.logical_request_id == "pinned")
        }));
        assert_eq!(terminal_count(&events), 1);
        assert!(replay.drain_events()?.is_empty());

        let terminal = terminal_event(&events, "pinned");
        assert_eq!(terminal.internal_uuid, uuid);
        assert_eq!(terminal.session_id, "pinned-session");
        assert_eq!(terminal.authored_turn_index, 4);
        assert_eq!(terminal.worker_id, Some(1));
        assert_eq!(terminal.dp_rank, Some(0));
        assert_eq!(
            terminal.terminal_status,
            Some(ReplayTerminalStatus::Completed)
        );
        assert_eq!(terminal.requested_output_length, 3);
        assert_eq!(terminal.emitted_output_count, 3);
        assert!(terminal.e2e_latency_ms.is_some());

        let report = replay.finalize()?;
        assert_eq!(report.request_counts.num_requests, 1);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.per_request.len(), 1);
        let record = &report.per_request[0];
        assert_eq!(record.logical_request_id.as_deref(), Some("pinned"));
        assert_eq!(record.session_id.as_deref(), Some("pinned-session"));
        assert_eq!(record.authored_turn_index, Some(4));
        assert_eq!(record.uuid, uuid.to_string());
        assert_eq!(record.requested_output_length, 3);
        assert_eq!(record.output_length, 3);
        assert_eq!(record.terminal_status, ReplayTerminalStatus::Completed);
        Ok(())
    }

    #[test]
    fn root_fanout_join_uses_actual_slowest_terminal_and_tool_delay() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 2, false);
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![
                agentic_request(request("root", "root-session", 0, 4, &[30], 1), &[], 0.0),
                agentic_request(
                    request("left", "left-session", 0, 4, &[31], 1),
                    &["root"],
                    0.0,
                ),
                agentic_request(
                    request("right", "right-session", 0, 4, &[32], 6),
                    &["root"],
                    0.0,
                ),
                agentic_request(
                    request("join", "join-session", 0, 4, &[33], 1),
                    &["left", "right"],
                    3.0,
                ),
            ],
        };
        replay.append_agentic_workflow(workflow, 5.0)?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 4);

        let root_terminal = terminal_event(&events, "root").timestamp_ms;
        assert_eq!(routed_event(&events, "left").timestamp_ms, root_terminal);
        assert_eq!(routed_event(&events, "right").timestamp_ms, root_terminal);

        let left_terminal = terminal_event(&events, "left").timestamp_ms;
        let right_terminal = terminal_event(&events, "right").timestamp_ms;
        assert!(right_terminal > left_terminal);
        let latest_parent_terminal = left_terminal.max(right_terminal);
        assert!(
            (routed_event(&events, "join").timestamp_ms - (latest_parent_terminal + 3.0)).abs()
                < 1e-9
        );

        let report = replay.finalize()?;
        assert_eq!(report.request_counts.completed_requests, 4);
        assert_eq!(report.per_request.len(), 4);
        Ok(())
    }

    #[test]
    fn rejected_parent_is_terminal_and_releases_zero_delay_child() -> Result<()> {
        let mut replay = OfflineReplaySession::new(
            &constrained_replay_args(),
            1,
            TRACE_BLOCK_SIZE,
            ReplaySessionRouter::RoundRobin,
        )?;
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![
                agentic_request(
                    request("oversized", "oversized-session", 0, 20, &[1, 2, 3, 4, 5], 1),
                    &[],
                    0.0,
                ),
                agentic_request(
                    request("after-rejection", "child-session", 0, 4, &[6], 1),
                    &["oversized"],
                    0.0,
                ),
            ],
        };
        replay.append_agentic_workflow(workflow, 0.0)?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 2);

        let rejection = terminal_event(&events, "oversized");
        assert_eq!(
            rejection.terminal_status,
            Some(ReplayTerminalStatus::Rejected)
        );
        assert_eq!(
            routed_event(&events, "after-rejection").timestamp_ms,
            rejection.timestamp_ms
        );
        assert_eq!(
            terminal_event(&events, "after-rejection").terminal_status,
            Some(ReplayTerminalStatus::Completed)
        );

        let report = replay.finalize()?;
        assert_eq!(report.request_counts.num_requests, 2);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.per_request.len(), 2);
        Ok(())
    }

    fn deterministic_fixture() -> Result<(Vec<ReplayEvent>, Value, Value)> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 2, true);
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![
                agentic_request(
                    request("det-root", "det-session", 0, 8, &[70, 71], 2),
                    &[],
                    0.0,
                ),
                agentic_request(
                    request("det-child", "det-session", 1, 8, &[70, 72], 3),
                    &["det-root"],
                    2.0,
                ),
            ],
        };
        replay.append_agentic_workflow(workflow, 7.0)?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        let report = replay.finalize()?;
        let (summary, per_request) = semantic_report_json(&report);
        Ok((events, summary, per_request))
    }

    #[test]
    fn canonical_session_repeats_identical_event_and_semantic_report_streams() -> Result<()> {
        let first = deterministic_fixture()?;
        let second = deterministic_fixture()?;
        assert_eq!(first, second);
        Ok(())
    }

    #[test]
    fn long_lived_session_preserves_shared_partial_and_divergent_prefix_state() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, true);
        settle_empty_open(&mut replay)?;

        let cases = [
            request("shared-seed", "shared-seed", 0, 8, &[100, 101], 1),
            request("shared-hit", "shared-hit", 0, 8, &[100, 101], 1),
            request("partial-seed", "partial-seed", 0, 4, &[200], 1),
            request("partial-hit", "partial-hit", 0, 6, &[200, 201], 1),
            request("diverge-seed", "diverge-seed", 0, 5, &[300, 301], 1),
            request(
                "one-token-divergence",
                "one-token-divergence",
                0,
                5,
                &[300, 302],
                1,
            ),
        ];
        let case_count = cases.len();
        let mut events = Vec::new();
        for mut case in cases {
            case.ready_time_ms = replay.now_ms()?;
            let logical_id = case.logical_request_id.clone();
            replay.submit_request(case)?;
            events.extend(drive_to_terminal(&mut replay, &logical_id)?);
            assert!(replay.is_quiescent()?);
        }

        assert!(admitted_reuse(&events, "shared-hit") > 0);
        assert_eq!(admitted_reuse(&events, "partial-hit"), TRACE_BLOCK_SIZE);
        assert_eq!(
            admitted_reuse(&events, "one-token-divergence"),
            TRACE_BLOCK_SIZE
        );

        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), case_count);
        let report = replay.finalize()?;
        assert!(report.prefix_cache_reused_ratio > 0.0);
        assert_eq!(report.per_request.len(), case_count);
        Ok(())
    }
}

impl OfflineReplaySession {
    pub fn now_ms(&self) -> Result<f64> {
        Ok(self.runtime()?.now_ms())
    }

    pub fn next_event_time_ms(&mut self) -> Result<Option<f64>> {
        let determinism = self.determinism;
        with_replay_determinism(determinism, || Ok(self.runtime_mut()?.next_event_time_ms()))
    }

    pub fn advance_next(&mut self) -> Result<ReplayStepStatus> {
        let determinism = self.determinism;
        with_replay_determinism(determinism, || self.runtime_mut()?.advance_next())
    }

    pub fn advance_to(&mut self, target_ms: f64) -> Result<ReplayStepStatus> {
        let determinism = self.determinism;
        with_replay_determinism(determinism, || self.runtime_mut()?.advance_to(target_ms))
    }

    pub fn settle_current_time(&mut self) -> Result<ReplayStepStatus> {
        let determinism = self.determinism;
        with_replay_determinism(determinism, || self.runtime_mut()?.settle_current_time())
    }

    pub fn drain_events(&mut self) -> Result<Vec<ReplayEvent>> {
        Ok(self.runtime_mut()?.drain_events())
    }

    pub fn pending_placements(&self) -> Result<Vec<ReplayPendingPlacement>> {
        Ok(self.runtime()?.pending_placements())
    }

    pub fn assign(&mut self, logical_request_id: &str, target: WorkerTarget) -> Result<()> {
        let uuid = self
            .runtime()?
            .uuid_for_logical_id(logical_request_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "interactive replay has no request with logical ID {logical_request_id:?}"
                )
            })?;
        let determinism = self.determinism;
        with_replay_determinism(determinism, || self.runtime_mut()?.assign(uuid, target))
    }

    pub fn snapshot(&self) -> Result<ReplaySnapshot> {
        Ok(self.runtime()?.snapshot())
    }

    pub fn close_admission(&mut self) -> Result<()> {
        self.runtime()?;
        if self.admission_closed {
            return Ok(());
        }
        self.runtime_mut()?.close()?;
        self.admission_closed = true;
        Ok(())
    }

    pub fn is_quiescent(&mut self) -> Result<bool> {
        Ok(self.runtime_mut()?.is_quiescent())
    }

    pub fn is_drained(&self) -> Result<bool> {
        Ok(self.runtime()?.is_drained())
    }

    pub fn finalize(&mut self) -> Result<TraceSimulationReport> {
        if !self.admission_closed {
            bail!("cannot finalize interactive replay while admission remains open");
        }
        if !self.runtime()?.is_drained() {
            bail!("cannot finalize interactive replay while work remains incomplete");
        }
        Ok(self
            .runtime
            .take()
            .context("interactive replay session was already finalized")?
            .finish())
    }
}

impl OfflineReplaySession {
    pub fn submit_request(&mut self, mut request: ReplayRequestSpec) -> Result<()> {
        let release_at_ms = request.ready_time_ms;
        request.ready_time_ms = 0.0;
        self.append_agentic_workflow(
            ReplayAgenticWorkflow {
                trace_block_size: self.trace_block_size,
                requests: vec![ReplayAgenticRequest {
                    request,
                    wait_for: Vec::new(),
                    dependency_delay_ms: 0.0,
                    prefix_reset: false,
                }],
            },
            release_at_ms,
        )
    }

    pub fn append_agentic_workflow(
        &mut self,
        workflow: ReplayAgenticWorkflow,
        release_at_ms: f64,
    ) -> Result<()> {
        if self.admission_closed {
            bail!("cannot append an interactive workflow after admission is closed");
        }
        if workflow.trace_block_size != self.trace_block_size {
            bail!(
                "workflow trace_block_size {} does not match session block size {}",
                workflow.trace_block_size,
                self.trace_block_size
            );
        }
        if workflow.requests.is_empty() {
            bail!("interactive replay workflow must contain at least one request");
        }
        let now_ms = self.now_ms()?;
        if !release_at_ms.is_finite() || release_at_ms < now_ms {
            bail!(
                "interactive replay release time {release_at_ms} ms is before current time {} ms",
                now_ms
            );
        }

        let mut logical_ids = FxHashSet::default();
        let mut internal_ids = FxHashSet::default();
        let mut session_turns = FxHashSet::default();
        let mut identities = Vec::with_capacity(workflow.requests.len());
        let mut turns = Vec::with_capacity(workflow.requests.len());
        let mut preassignments = Vec::new();
        for authored in workflow.requests {
            self.validate_request(&authored.request)?;
            if !authored.dependency_delay_ms.is_finite() || authored.dependency_delay_ms < 0.0 {
                bail!(
                    "interactive replay request {} has invalid dependency delay {}",
                    authored.request.logical_request_id,
                    authored.dependency_delay_ms
                );
            }
            if !logical_ids.insert(authored.request.logical_request_id.clone()) {
                bail!(
                    "interactive replay workflow duplicates logical_request_id {:?}",
                    authored.request.logical_request_id
                );
            }
            if !session_turns.insert((
                authored.request.session_id.clone(),
                authored.request.authored_turn_index,
            )) {
                bail!(
                    "interactive replay workflow duplicates session {:?} authored turn {}",
                    authored.request.session_id,
                    authored.request.authored_turn_index
                );
            }
            let uuid = authored.request.internal_uuid.unwrap_or_else(|| {
                if self.determinism == ReplayDeterminism::CanonicalV1 {
                    Self::deterministic_uuid(&authored.request.logical_request_id)
                } else {
                    Uuid::new_v4()
                }
            });
            if !internal_ids.insert(uuid) {
                bail!("interactive replay workflow duplicates internal UUID {uuid}");
            }
            if let Some(target) = authored.request.target {
                preassignments.push((uuid, target));
            }
            identities.push(InteractiveRequestIdentity {
                logical_request_id: authored.request.logical_request_id.clone(),
                internal_uuid: uuid,
                session_id: authored.request.session_id.clone(),
                authored_turn_index: authored.request.authored_turn_index,
                input_length: authored.request.input_length,
                requested_output_length: authored.request.output_length,
                ready_at_ms: None,
                worker: None,
                emitted_output_count: 0,
                reused_input_tokens: None,
                first_token_ms: None,
                terminal_ms: None,
            });

            let request = authored.request;
            let replay_key = request
                .output_token_ids
                .as_ref()
                .map(|_| request.logical_request_id.clone());
            turns.push(AgenticTurnTrace {
                request_id: request.logical_request_id,
                session_id: request.session_id,
                authored_turn_index: request.authored_turn_index,
                internal_uuid: Some(uuid),
                input_length: request.input_length,
                max_output_tokens: request.output_length,
                output_token_ids: request.output_token_ids,
                replay_key,
                hash_ids: request.hash_ids,
                first_ready_timestamp_ms: Some(request.ready_time_ms),
                delay_after_dependencies_ms: authored.dependency_delay_ms,
                priority: request.priority,
                strict_priority: request.strict_priority,
                policy_class: request.policy_class,
                wait_for: authored.wait_for,
                prefix_reset: authored.prefix_reset,
            });
        }

        let mut registered_uuids = Vec::with_capacity(identities.len());
        for identity in identities {
            let uuid = identity.internal_uuid;
            if let Err(error) = self.runtime_mut()?.register_identity(identity) {
                for uuid in &registered_uuids {
                    self.runtime_mut()?.unregister_identity(*uuid);
                }
                return Err(error);
            }
            registered_uuids.push(uuid);
        }
        for &(uuid, target) in &preassignments {
            if let Err(error) = self.runtime_mut()?.preassign(uuid, target) {
                for registered in &registered_uuids {
                    self.runtime_mut()?.cancel_preassignment(*registered);
                    self.runtime_mut()?.unregister_identity(*registered);
                }
                return Err(error);
            }
        }

        let determinism = self.determinism;
        let trace_block_size = self.trace_block_size;
        let append_result = with_replay_determinism(determinism, || {
            self.runtime_mut()?.append(
                AgenticTrace {
                    block_size: trace_block_size,
                    turns,
                },
                release_at_ms,
            )
        });
        if let Err(error) = append_result {
            for uuid in registered_uuids {
                self.runtime_mut()?.cancel_preassignment(uuid);
                self.runtime_mut()?.unregister_identity(uuid);
            }
            return Err(error);
        }
        Ok(())
    }
}
