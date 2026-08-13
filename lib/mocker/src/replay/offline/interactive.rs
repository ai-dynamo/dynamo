// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public, polling-based control surface for causal offline replay.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use uuid::Uuid;

use crate::common::protocols::{EngineType, G1Backend, MockEngineArgs, WorkerType};
pub use crate::loadgen::ReplayRoutingConstraints;
use crate::loadgen::{AgenticTrace, AgenticTurnTrace, WorkloadDriver};
use crate::replay::offline::agg::{ExternalAggRuntime, RoundRobinAggRuntime};
use crate::replay::offline::components::ReplayMode;
use crate::replay::offline::extensions::kv_router::AggRuntime as KvAggRuntime;
#[cfg(test)]
use crate::replay::offline::topology::DEFAULT_REPLAY_POOL_ID;
use crate::replay::offline::topology::{PoolSpec, ResolvedPoolTopology, WorkerTarget};
use crate::replay::{
    ReplayDeterminism, ReplayRouterMode, ReplayTerminalStatus, ReplayWorkerLifecycleStatus,
    TraceSimulationReport, with_replay_determinism,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplaySessionRouter {
    External,
    RoundRobin,
    KvRouter,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySessionOptions {
    /// Allow a native router to use authored session identity as an affinity
    /// hint. Disabled by default so session affinity is never implicit.
    #[serde(default)]
    pub session_affinity: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplayRequestSpec {
    pub logical_request_id: String,
    pub attempt_id: String,
    pub group_id: String,
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
    pub routing_constraints: ReplayRoutingConstraints,
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
    pub attempt_id: String,
    pub group_id: String,
    pub internal_uuid: Uuid,
    pub session_id: String,
    pub authored_turn_index: usize,
    pub timestamp_ms: f64,
    pub pool_id: Option<String>,
    pub worker_id: Option<usize>,
    pub dp_rank: Option<usize>,
    pub terminal_status: Option<ReplayTerminalStatus>,
    pub input_length: usize,
    /// Redacted until terminal so ordinary placement policies cannot inspect
    /// trace-recorded future output work.
    pub requested_output_length: Option<usize>,
    pub emitted_output_count: usize,
    pub reused_input_tokens: Option<usize>,
    pub ttft_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub routing_constraints: ReplayRoutingConstraints,
    pub eligible_pool_ids: Vec<String>,
    pub candidates: Vec<ReplayPlacementCandidate>,
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

impl ReplayEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::PlacementNeeded(_) => "placement_needed",
            Self::Routed(_) => "routed",
            Self::Queued(_) => "queued",
            Self::Admitted(_) => "admitted",
            Self::FirstToken(_) => "first_token",
            Self::Terminal(_) => "terminal",
        }
    }
}

/// Eager, immutable replay event snapshot used by high-volume adapters.
///
/// Unlike [`ReplayEventData`], repeated authored metadata and placement
/// candidates share their frozen backing allocations. Every lifecycle scalar
/// is copied at capture time, so this never observes later request mutation.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedReplayEventData {
    metadata: Arc<InteractiveRequestMetadata>,
    timestamp_ms: f64,
    pool_id: Option<Arc<str>>,
    worker_id: Option<usize>,
    dp_rank: Option<usize>,
    terminal_status: Option<ReplayTerminalStatus>,
    requested_output_length: Option<usize>,
    emitted_output_count: usize,
    reused_input_tokens: Option<usize>,
    ttft_ms: Option<f64>,
    e2e_latency_ms: Option<f64>,
    eligible_pool_ids: Arc<Vec<String>>,
    candidates: Arc<Vec<ReplayPlacementCandidate>>,
}

/// Borrowed declaration-order view of one immutable captured event.
#[doc(hidden)]
pub struct CapturedReplayEventDataView<'a> {
    pub logical_request_id: &'a str,
    pub attempt_id: &'a str,
    pub group_id: &'a str,
    pub internal_uuid: Uuid,
    pub session_id: &'a str,
    pub authored_turn_index: usize,
    pub timestamp_ms: f64,
    pub pool_id: Option<&'a str>,
    pub worker_id: Option<usize>,
    pub dp_rank: Option<usize>,
    pub terminal_status: Option<ReplayTerminalStatus>,
    pub input_length: usize,
    pub requested_output_length: Option<usize>,
    pub emitted_output_count: usize,
    pub reused_input_tokens: Option<usize>,
    pub ttft_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<&'a str>,
    pub routing_constraints: &'a ReplayRoutingConstraints,
    pub eligible_pool_ids: &'a [String],
    pub candidates: &'a [ReplayPlacementCandidate],
}

impl CapturedReplayEventData {
    pub fn view(&self) -> CapturedReplayEventDataView<'_> {
        CapturedReplayEventDataView {
            logical_request_id: &self.metadata.logical_request_id,
            attempt_id: &self.metadata.attempt_id,
            group_id: &self.metadata.group_id,
            internal_uuid: self.metadata.internal_uuid,
            session_id: &self.metadata.session_id,
            authored_turn_index: self.metadata.authored_turn_index,
            timestamp_ms: self.timestamp_ms,
            pool_id: self.pool_id.as_deref(),
            worker_id: self.worker_id,
            dp_rank: self.dp_rank,
            terminal_status: self.terminal_status,
            input_length: self.metadata.input_length,
            requested_output_length: self.requested_output_length,
            emitted_output_count: self.emitted_output_count,
            reused_input_tokens: self.reused_input_tokens,
            ttft_ms: self.ttft_ms,
            e2e_latency_ms: self.e2e_latency_ms,
            priority: self.metadata.priority,
            strict_priority: self.metadata.strict_priority,
            policy_class: self.metadata.policy_class.as_deref(),
            routing_constraints: &self.metadata.routing_constraints,
            eligible_pool_ids: self.eligible_pool_ids.as_slice(),
            candidates: self.candidates.as_slice(),
        }
    }

    pub fn into_owned(self) -> ReplayEventData {
        let metadata = self.metadata;
        ReplayEventData {
            logical_request_id: metadata.logical_request_id.clone(),
            attempt_id: metadata.attempt_id.clone(),
            group_id: metadata.group_id.clone(),
            internal_uuid: metadata.internal_uuid,
            session_id: metadata.session_id.clone(),
            authored_turn_index: metadata.authored_turn_index,
            timestamp_ms: self.timestamp_ms,
            pool_id: self.pool_id.as_deref().map(str::to_owned),
            worker_id: self.worker_id,
            dp_rank: self.dp_rank,
            terminal_status: self.terminal_status,
            input_length: metadata.input_length,
            requested_output_length: self.requested_output_length,
            emitted_output_count: self.emitted_output_count,
            reused_input_tokens: self.reused_input_tokens,
            ttft_ms: self.ttft_ms,
            e2e_latency_ms: self.e2e_latency_ms,
            priority: metadata.priority,
            strict_priority: metadata.strict_priority,
            policy_class: metadata.policy_class.clone(),
            routing_constraints: metadata.routing_constraints.clone(),
            eligible_pool_ids: Arc::unwrap_or_clone(self.eligible_pool_ids),
            candidates: Arc::unwrap_or_clone(self.candidates),
        }
    }
}

impl Serialize for CapturedReplayEventData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let data = self.view();
        let mut value = serializer.serialize_struct("ReplayEventData", 23)?;
        value.serialize_field("logical_request_id", data.logical_request_id)?;
        value.serialize_field("attempt_id", data.attempt_id)?;
        value.serialize_field("group_id", data.group_id)?;
        value.serialize_field("internal_uuid", &data.internal_uuid)?;
        value.serialize_field("session_id", data.session_id)?;
        value.serialize_field("authored_turn_index", &data.authored_turn_index)?;
        value.serialize_field("timestamp_ms", &data.timestamp_ms)?;
        value.serialize_field("pool_id", &data.pool_id)?;
        value.serialize_field("worker_id", &data.worker_id)?;
        value.serialize_field("dp_rank", &data.dp_rank)?;
        value.serialize_field("terminal_status", &data.terminal_status)?;
        value.serialize_field("input_length", &data.input_length)?;
        value.serialize_field("requested_output_length", &data.requested_output_length)?;
        value.serialize_field("emitted_output_count", &data.emitted_output_count)?;
        value.serialize_field("reused_input_tokens", &data.reused_input_tokens)?;
        value.serialize_field("ttft_ms", &data.ttft_ms)?;
        value.serialize_field("e2e_latency_ms", &data.e2e_latency_ms)?;
        value.serialize_field("priority", &data.priority)?;
        value.serialize_field("strict_priority", &data.strict_priority)?;
        value.serialize_field("policy_class", &data.policy_class)?;
        value.serialize_field("routing_constraints", data.routing_constraints)?;
        value.serialize_field("eligible_pool_ids", data.eligible_pool_ids)?;
        value.serialize_field("candidates", data.candidates)?;
        value.end()
    }
}

/// Eager immutable counterpart to [`ReplayEvent`] for direct adapters.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event_type", content = "event", rename_all = "snake_case")]
pub enum CapturedReplayEvent {
    PlacementNeeded(CapturedReplayEventData),
    Routed(CapturedReplayEventData),
    Queued(CapturedReplayEventData),
    Admitted(CapturedReplayEventData),
    FirstToken(CapturedReplayEventData),
    Terminal(CapturedReplayEventData),
}

impl CapturedReplayEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::PlacementNeeded(_) => "placement_needed",
            Self::Routed(_) => "routed",
            Self::Queued(_) => "queued",
            Self::Admitted(_) => "admitted",
            Self::FirstToken(_) => "first_token",
            Self::Terminal(_) => "terminal",
        }
    }

    pub fn data(&self) -> &CapturedReplayEventData {
        match self {
            Self::PlacementNeeded(data)
            | Self::Routed(data)
            | Self::Queued(data)
            | Self::Admitted(data)
            | Self::FirstToken(data)
            | Self::Terminal(data) => data,
        }
    }

    pub fn into_data(self) -> CapturedReplayEventData {
        match self {
            Self::PlacementNeeded(data)
            | Self::Routed(data)
            | Self::Queued(data)
            | Self::Admitted(data)
            | Self::FirstToken(data)
            | Self::Terminal(data) => data,
        }
    }

    pub fn into_owned(self) -> ReplayEvent {
        match self {
            Self::PlacementNeeded(data) => ReplayEvent::PlacementNeeded(data.into_owned()),
            Self::Routed(data) => ReplayEvent::Routed(data.into_owned()),
            Self::Queued(data) => ReplayEvent::Queued(data.into_owned()),
            Self::Admitted(data) => ReplayEvent::Admitted(data.into_owned()),
            Self::FirstToken(data) => ReplayEvent::FirstToken(data.into_owned()),
            Self::Terminal(data) => ReplayEvent::Terminal(data.into_owned()),
        }
    }
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
    pub attempt_id: String,
    pub group_id: String,
    pub internal_uuid: Uuid,
    pub session_id: String,
    pub authored_turn_index: usize,
    pub ready_at_ms: f64,
    /// Prompt work visible when the request becomes placeable. The recorded
    /// output length and every future timing value are deliberately absent.
    pub input_length: usize,
    pub priority: i32,
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub routing_constraints: ReplayRoutingConstraints,
    pub eligible_pool_ids: Vec<String>,
    pub candidates: Vec<ReplayPlacementCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayWorkerSnapshot {
    pub pool_id: String,
    pub worker_id: usize,
    pub dp_rank: usize,
    pub lifecycle_status: ReplayWorkerLifecycleStatus,
    /// Whether this worker is physically provisioned and therefore billed.
    /// Static-inactive, starting, and draining workers remain provisioned.
    pub provisioned: bool,
    pub active: bool,
    pub draining: bool,
    pub in_flight_requests: usize,
    /// Live scheduler queue depth, when supported by the engine backend.
    pub queued_requests: Option<usize>,
    /// Number of requests in the scheduler's running set.
    pub running_requests: Option<usize>,
    /// Materialized sequence tokens in the waiting queue. This excludes all
    /// future or recorded output-length information.
    pub queued_tokens: Option<usize>,
    /// Materialized sequence tokens in the running set (prompt plus generated
    /// tokens currently visible to the scheduler).
    pub running_tokens: Option<usize>,
    /// Configured finite runnable-sequence capacity; `None` is unbounded or
    /// unsupported depending on whether the other scheduler fields exist.
    pub max_num_seqs: Option<usize>,
    /// Exact cumulative worker/rank preemptions. Unsupported schedulers expose
    /// `None`; zero is reserved for a supported scheduler with no preemption.
    pub preemption_count: Option<u64>,
    /// Physical G1 KV capacity and occupancy, in engine blocks.
    pub kv_capacity_blocks: Option<usize>,
    pub kv_occupied_blocks: Option<usize>,
    pub kv_free_blocks: Option<usize>,
    pub tags: Vec<String>,
    pub taints: Vec<String>,
    pub capabilities: Vec<String>,
}

/// Complete request-specific, non-oracle observation available to an external
/// policy at one placement boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayPlacementCandidate {
    pub target: WorkerTarget,
    pub active: bool,
    pub draining: bool,
    pub eligible: bool,
    pub constraint_reason: Option<String>,
    pub in_flight_requests: usize,
    pub queued_requests: Option<usize>,
    pub running_requests: Option<usize>,
    pub queued_tokens: Option<usize>,
    pub running_tokens: Option<usize>,
    pub max_num_seqs: Option<usize>,
    pub preemption_count: Option<u64>,
    pub kv_prefix_overlap_tokens: Option<usize>,
    pub kv_capacity_blocks: Option<usize>,
    pub kv_occupied_blocks: Option<usize>,
    pub kv_free_blocks: Option<usize>,
    pub tags: Vec<String>,
    pub taints: Vec<String>,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReplaySnapshot {
    pub now_ms: f64,
    pub admission_open: bool,
    pub pending_request_count: usize,
    pub pending_placement_count: usize,
    pub workers: Vec<ReplayWorkerSnapshot>,
}

#[derive(Debug)]
pub(crate) struct InteractiveRequestIdentity {
    metadata: Arc<InteractiveRequestMetadata>,
    ready_at_ms: Option<f64>,
    worker: Option<InteractiveWorkerIdentity>,
    emitted_output_count: usize,
    reused_input_tokens: Option<usize>,
    first_token_ms: Option<f64>,
    terminal_ms: Option<f64>,
}

impl std::ops::Deref for InteractiveRequestIdentity {
    type Target = InteractiveRequestMetadata;

    fn deref(&self) -> &Self::Target {
        &self.metadata
    }
}

#[derive(Debug, PartialEq)]
pub(crate) struct InteractiveRequestMetadata {
    pub(crate) logical_request_id: String,
    pub(crate) attempt_id: String,
    pub(crate) group_id: String,
    pub(crate) internal_uuid: Uuid,
    pub(crate) session_id: String,
    pub(crate) authored_turn_index: usize,
    pub(crate) input_length: usize,
    pub(crate) requested_output_length: usize,
    pub(crate) priority: i32,
    pub(crate) strict_priority: u32,
    pub(crate) policy_class: Option<String>,
    pub(crate) routing_constraints: ReplayRoutingConstraints,
}

#[derive(Debug, Clone)]
struct InteractiveWorkerIdentity {
    pool_id: Arc<str>,
    worker_id: usize,
    dp_rank: usize,
}

#[derive(Debug, Default)]
pub(crate) struct InteractiveCapture {
    external_placement: bool,
    session_affinity: bool,
    eligible_pool_ids: Arc<Vec<String>>,
    identities: FxHashMap<Uuid, InteractiveRequestIdentity>,
    logical_to_uuid: FxHashMap<String, Uuid>,
    session_turn_to_uuid: FxHashMap<(String, usize), Uuid>,
    events: VecDeque<CapturedReplayEvent>,
    /// Exactly one externally controlled request may have a policy-visible
    /// observation at a time. Clearing this after assignment forces the next
    /// same-time request to observe the scheduler state changed by its
    /// predecessor. Retaining the candidates also makes PlacementNeeded,
    /// selected overlap, and Routed refer to one causally frozen observation
    /// instead of rebuilding equivalent scheduler views at each API boundary.
    announced_placement: Option<AnnouncedPlacementObservation>,
}

#[derive(Debug)]
struct AnnouncedPlacementObservation {
    request_id: Uuid,
    candidates: Arc<Vec<ReplayPlacementCandidate>>,
}

impl InteractiveCapture {
    pub(crate) fn new(
        external_placement: bool,
        session_affinity: bool,
        eligible_pool_ids: Vec<String>,
    ) -> Self {
        Self {
            external_placement,
            session_affinity,
            eligible_pool_ids: Arc::new(eligible_pool_ids),
            ..Self::default()
        }
    }

    pub(crate) fn uses_external_placement(&self) -> bool {
        self.external_placement
    }

    pub(crate) fn uses_session_affinity(&self) -> bool {
        self.session_affinity
    }

    pub(crate) fn register(&mut self, identity: InteractiveRequestIdentity) -> anyhow::Result<()> {
        let metadata = &identity.metadata;
        if metadata.logical_request_id.trim().is_empty() {
            anyhow::bail!("interactive replay logical_request_id must not be empty");
        }
        if self
            .logical_to_uuid
            .contains_key(&metadata.logical_request_id)
        {
            anyhow::bail!(
                "interactive replay duplicate logical_request_id {:?}",
                metadata.logical_request_id
            );
        }
        let session_turn = (metadata.session_id.clone(), metadata.authored_turn_index);
        if let Some(existing) = self.session_turn_to_uuid.get(&session_turn) {
            anyhow::bail!(
                "interactive replay session {:?} authored turn {} conflicts with internal UUID {existing}",
                session_turn.0,
                session_turn.1
            );
        }
        if self.identities.contains_key(&metadata.internal_uuid) {
            anyhow::bail!(
                "interactive replay duplicate internal UUID {}",
                metadata.internal_uuid
            );
        }
        self.logical_to_uuid
            .insert(metadata.logical_request_id.clone(), metadata.internal_uuid);
        self.session_turn_to_uuid
            .insert(session_turn, metadata.internal_uuid);
        self.identities.insert(metadata.internal_uuid, identity);
        Ok(())
    }

    pub(crate) fn unregister(&mut self, uuid: Uuid) {
        if self
            .announced_placement
            .as_ref()
            .is_some_and(|observation| observation.request_id == uuid)
        {
            self.announced_placement = None;
        }
        if let Some(identity) = self.identities.remove(&uuid) {
            self.logical_to_uuid
                .remove(&identity.metadata.logical_request_id);
            self.session_turn_to_uuid.remove(&(
                identity.metadata.session_id.clone(),
                identity.metadata.authored_turn_index,
            ));
        }
    }

    pub(crate) fn uuid_for_logical_id(&self, logical_id: &str) -> Option<Uuid> {
        self.logical_to_uuid.get(logical_id).copied()
    }

    pub(crate) fn identity(&self, uuid: Uuid) -> Option<&InteractiveRequestIdentity> {
        self.identities.get(&uuid)
    }

    fn event_data(
        &self,
        uuid: Uuid,
        timestamp_ms: f64,
        terminal: bool,
    ) -> anyhow::Result<CapturedReplayEventData> {
        let identity = self.identities.get(&uuid).ok_or_else(|| {
            anyhow::anyhow!("interactive replay has no authored identity for request {uuid}")
        })?;
        let (pool_id, worker_id, dp_rank) = identity
            .worker
            .as_ref()
            .map(|worker| {
                (
                    Some(Arc::clone(&worker.pool_id)),
                    Some(worker.worker_id),
                    Some(worker.dp_rank),
                )
            })
            .unwrap_or((None, None, None));
        Ok(CapturedReplayEventData {
            metadata: Arc::clone(&identity.metadata),
            timestamp_ms,
            pool_id,
            worker_id,
            dp_rank,
            terminal_status: None,
            requested_output_length: terminal.then_some(identity.metadata.requested_output_length),
            emitted_output_count: identity.emitted_output_count,
            reused_input_tokens: identity.reused_input_tokens,
            ttft_ms: identity
                .first_token_ms
                .map(|first| (first - identity.ready_at_ms.unwrap_or(first)).max(0.0)),
            e2e_latency_ms: identity
                .terminal_ms
                .map(|terminal| (terminal - identity.ready_at_ms.unwrap_or(terminal)).max(0.0)),
            eligible_pool_ids: Arc::clone(&self.eligible_pool_ids),
            candidates: Arc::default(),
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
        identity.worker = Some(InteractiveWorkerIdentity {
            pool_id: Arc::from(target.pool_id),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
        });
        Ok(())
    }

    pub(crate) fn placement_needed_event(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        candidates: Vec<ReplayPlacementCandidate>,
    ) -> anyhow::Result<ReplayEvent> {
        let (event, retained_candidates) =
            self.placement_needed_captured_event(uuid, now_ms, candidates)?;
        let event = event.into_owned();
        self.mark_shared_placement_observed(uuid, retained_candidates);
        Ok(event)
    }

    pub(crate) fn placement_needed_captured_event(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        candidates: Vec<ReplayPlacementCandidate>,
    ) -> anyhow::Result<(CapturedReplayEvent, Arc<Vec<ReplayPlacementCandidate>>)> {
        let candidates = Arc::new(candidates);
        let mut data = self.event_data(uuid, now_ms, false)?;
        data.eligible_pool_ids = Arc::new(eligible_pool_ids(&candidates));
        data.candidates = Arc::clone(&candidates);
        Ok((CapturedReplayEvent::PlacementNeeded(data), candidates))
    }

    pub(crate) fn mark_placement_observed(
        &mut self,
        uuid: Uuid,
        candidates: Vec<ReplayPlacementCandidate>,
    ) {
        self.mark_shared_placement_observed(uuid, Arc::new(candidates));
    }

    pub(crate) fn retain_captured_placement(
        &mut self,
        uuid: Uuid,
        candidates: Arc<Vec<ReplayPlacementCandidate>>,
    ) {
        self.mark_shared_placement_observed(uuid, candidates);
    }

    fn mark_shared_placement_observed(
        &mut self,
        uuid: Uuid,
        candidates: Arc<Vec<ReplayPlacementCandidate>>,
    ) {
        self.announced_placement = Some(AnnouncedPlacementObservation {
            request_id: uuid,
            candidates,
        });
    }

    pub(crate) fn placement_is_announced(&self, uuid: Uuid) -> bool {
        self.announced_placement
            .as_ref()
            .is_some_and(|observation| observation.request_id == uuid)
    }

    pub(crate) fn announced_placement_candidates(
        &self,
        uuid: Uuid,
    ) -> Option<&[ReplayPlacementCandidate]> {
        self.announced_placement
            .as_ref()
            .filter(|observation| observation.request_id == uuid)
            .map(|observation| observation.candidates.as_slice())
    }

    pub(crate) fn validate_placement_assignment(&self, uuid: Uuid) -> anyhow::Result<()> {
        if self.placement_is_announced(uuid) {
            return Ok(());
        }
        anyhow::bail!(
            "interactive replay request {uuid} has no current policy observation; drain events or refresh pending placements before assignment"
        )
    }

    pub(crate) fn complete_placement_assignment(
        &mut self,
        uuid: Uuid,
    ) -> anyhow::Result<Vec<ReplayPlacementCandidate>> {
        if !self.placement_is_announced(uuid) {
            anyhow::bail!(
                "interactive replay request {uuid} is not the current placement boundary"
            );
        }
        let candidates = self
            .announced_placement
            .take()
            .expect("placement observation was validated before consumption")
            .candidates;
        Ok(Arc::unwrap_or_clone(candidates))
    }

    pub(crate) fn cancel_placement_observation(&mut self, uuid: Uuid) {
        if self.placement_is_announced(uuid) {
            self.announced_placement = None;
        }
    }

    pub(crate) fn emit_routed(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        candidates: Vec<ReplayPlacementCandidate>,
    ) -> anyhow::Result<()> {
        let mut data = self.event_data(uuid, now_ms, false)?;
        data.eligible_pool_ids = Arc::new(eligible_pool_ids(&candidates));
        data.candidates = Arc::new(candidates);
        self.events.push_back(CapturedReplayEvent::Routed(data));
        Ok(())
    }

    pub(crate) fn emit_queued(&mut self, uuid: Uuid, now_ms: f64) -> anyhow::Result<()> {
        let data = self.event_data(uuid, now_ms, false)?;
        self.events.push_back(CapturedReplayEvent::Queued(data));
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
        let data = self.event_data(uuid, now_ms, false)?;
        self.events.push_back(CapturedReplayEvent::Admitted(data));
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
        let data = self.event_data(uuid, now_ms, false)?;
        self.events.push_back(CapturedReplayEvent::FirstToken(data));
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
        let mut data = self.event_data(uuid, now_ms, true)?;
        data.terminal_status = Some(status);
        self.events.push_back(CapturedReplayEvent::Terminal(data));
        Ok(())
    }

    pub(crate) fn drain_events(&mut self) -> Vec<ReplayEvent> {
        self.drain_captured_events()
            .into_iter()
            .map(CapturedReplayEvent::into_owned)
            .collect()
    }

    pub(crate) fn drain_captured_events(&mut self) -> Vec<CapturedReplayEvent> {
        self.events.drain(..).collect()
    }

    pub(crate) fn pending(&self, uuids: impl Iterator<Item = Uuid>) -> Vec<ReplayPendingPlacement> {
        uuids
            .filter_map(|uuid| {
                let identity = self.identities.get(&uuid)?;
                Some(ReplayPendingPlacement {
                    logical_request_id: identity.metadata.logical_request_id.clone(),
                    attempt_id: identity.metadata.attempt_id.clone(),
                    group_id: identity.metadata.group_id.clone(),
                    internal_uuid: uuid,
                    session_id: identity.metadata.session_id.clone(),
                    authored_turn_index: identity.metadata.authored_turn_index,
                    ready_at_ms: identity.ready_at_ms.unwrap_or_default(),
                    input_length: identity.metadata.input_length,
                    priority: identity.metadata.priority,
                    strict_priority: identity.metadata.strict_priority,
                    policy_class: identity.metadata.policy_class.clone(),
                    routing_constraints: identity.metadata.routing_constraints.clone(),
                    eligible_pool_ids: self.eligible_pool_ids.as_ref().clone(),
                    candidates: Vec::new(),
                })
            })
            .collect()
    }
}

fn eligible_pool_ids(candidates: &[ReplayPlacementCandidate]) -> Vec<String> {
    candidates
        .iter()
        .filter(|candidate| candidate.eligible)
        .map(|candidate| candidate.target.pool_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

enum InteractiveRuntime {
    External(ExternalAggRuntime),
    RoundRobin(RoundRobinAggRuntime),
    KvRouter(Box<KvAggRuntime>),
}

#[derive(Debug, Clone)]
struct StaticRoutingWorker {
    target: WorkerTarget,
    taints: BTreeSet<String>,
    is_eligible: bool,
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

    fn drain_captured_events(&mut self) -> Vec<CapturedReplayEvent> {
        match self {
            Self::External(runtime) => runtime.drain_interactive_captured_events(),
            Self::RoundRobin(runtime) => runtime.drain_interactive_captured_events(),
            Self::KvRouter(runtime) => runtime.drain_interactive_captured_events(),
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

    fn pending_placements(&mut self) -> Vec<ReplayPendingPlacement> {
        match self {
            Self::External(runtime) => runtime.pending_interactive_placements(),
            Self::RoundRobin(_) | Self::KvRouter(_) => Vec::new(),
        }
    }

    fn preassign(
        &mut self,
        uuid: Uuid,
        target: WorkerTarget,
        constraints: &ReplayRoutingConstraints,
    ) -> Result<()> {
        match self {
            Self::External(runtime) => runtime.preassign_interactive(
                uuid,
                target,
                constraints.required_taints.iter().cloned().collect(),
            ),
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

    fn assign_pool(&mut self, uuid: Uuid, pool_id: &str) -> Result<()> {
        match self {
            Self::External(runtime) => runtime.assign_pool_interactive(uuid, pool_id),
            Self::RoundRobin(_) | Self::KvRouter(_) => {
                bail!("assign_pool() is only valid with external placement")
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
    static_routing_workers: Vec<StaticRoutingWorker>,
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
        Self::new_with_determinism_and_options(
            args,
            num_workers,
            trace_block_size,
            router,
            determinism,
            ReplaySessionOptions::default(),
        )
    }

    pub fn new_with_options(
        args: &MockEngineArgs,
        num_workers: usize,
        trace_block_size: usize,
        router: ReplaySessionRouter,
        options: ReplaySessionOptions,
    ) -> Result<Self> {
        Self::new_with_determinism_and_options(
            args,
            num_workers,
            trace_block_size,
            router,
            ReplayDeterminism::CanonicalV1,
            options,
        )
    }

    pub fn new_with_determinism_and_options(
        args: &MockEngineArgs,
        num_workers: usize,
        trace_block_size: usize,
        router: ReplaySessionRouter,
        determinism: ReplayDeterminism,
        options: ReplaySessionOptions,
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
        if args.startup_time.is_some_and(|seconds| seconds != 0.0) {
            bail!(
                "interactive replay P0 topology is static; default pool must not configure startup_time"
            );
        }

        let engine_block_size = args.block_size;
        if trace_block_size != engine_block_size {
            bail!(
                "authoritative interactive replay trace_block_size {trace_block_size} does not match engine block_size {engine_block_size}"
            );
        }
        let static_routing_workers = (0..num_workers)
            .map(|worker_id| StaticRoutingWorker {
                target: WorkerTarget::default_pool(worker_id, 0),
                taints: args
                    .worker_taints
                    .get(worker_id)
                    .map(|taints| taints.iter().cloned().collect())
                    .unwrap_or_default(),
                is_eligible: true,
            })
            .collect();
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
            InteractiveRuntime::External(runtime) => {
                runtime.enable_interactive_capture(true, options.session_affinity)
            }
            InteractiveRuntime::RoundRobin(runtime) => {
                runtime.enable_interactive_capture(false, options.session_affinity)
            }
            InteractiveRuntime::KvRouter(runtime) => {
                runtime.enable_interactive_capture(false, options.session_affinity)
            }
        }
        Ok(Self {
            runtime: Some(runtime),
            router,
            determinism,
            trace_block_size,
            admission_closed: false,
            static_routing_workers,
        })
    }

    /// Create one external-placement session over two or more explicitly
    /// authored static pools. All workers execute inside one aggregated Mocker
    /// runtime and therefore one virtual clock.
    pub fn new_pooled(pools: Vec<PoolSpec>, trace_block_size: usize) -> Result<Self> {
        Self::new_pooled_with_determinism_and_options(
            pools,
            trace_block_size,
            ReplayDeterminism::CanonicalV1,
            ReplaySessionOptions::default(),
        )
    }

    pub fn new_pooled_with_determinism(
        pools: Vec<PoolSpec>,
        trace_block_size: usize,
        determinism: ReplayDeterminism,
    ) -> Result<Self> {
        Self::new_pooled_with_determinism_and_options(
            pools,
            trace_block_size,
            determinism,
            ReplaySessionOptions::default(),
        )
    }

    pub fn new_pooled_with_options(
        pools: Vec<PoolSpec>,
        trace_block_size: usize,
        options: ReplaySessionOptions,
    ) -> Result<Self> {
        Self::new_pooled_with_determinism_and_options(
            pools,
            trace_block_size,
            ReplayDeterminism::CanonicalV1,
            options,
        )
    }

    pub fn new_pooled_with_determinism_and_options(
        pools: Vec<PoolSpec>,
        trace_block_size: usize,
        determinism: ReplayDeterminism,
        options: ReplaySessionOptions,
    ) -> Result<Self> {
        let topology = ResolvedPoolTopology::resolve(pools, trace_block_size)?;
        let static_routing_workers = topology
            .workers
            .iter()
            .map(|worker| StaticRoutingWorker {
                target: worker.target.clone(),
                taints: worker.taints.clone(),
                is_eligible: worker.active && !worker.draining,
            })
            .collect();
        let engine_block_size = topology
            .workers
            .first()
            .expect("validated topology must contain a worker")
            .engine_args
            .block_size;
        let driver = WorkloadDriver::new_open_agentic_without_replay_hashes(
            trace_block_size,
            engine_block_size,
        )?;
        let mut runtime = with_replay_determinism(determinism, || -> Result<_> {
            Ok(InteractiveRuntime::External(
                ExternalAggRuntime::new_external_pooled_workload(driver, topology)?,
            ))
        })?;
        if let InteractiveRuntime::External(runtime) = &mut runtime {
            runtime.enable_interactive_capture(true, options.session_affinity);
        }
        Ok(Self {
            runtime: Some(runtime),
            router: ReplaySessionRouter::External,
            determinism,
            trace_block_size,
            admission_closed: false,
            static_routing_workers,
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
        if request.attempt_id.trim().is_empty() {
            bail!(
                "interactive replay request {} has an empty attempt_id",
                request.logical_request_id
            );
        }
        if request.group_id.trim().is_empty() {
            bail!(
                "interactive replay request {} has an empty group_id",
                request.logical_request_id
            );
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
        if request.hash_ids.len() != required_hashes {
            bail!(
                "interactive replay request {} input_length {} requires exactly {} hash IDs at trace_block_size {}, got {}",
                request.logical_request_id,
                request.input_length,
                required_hashes,
                self.trace_block_size,
                request.hash_ids.len(),
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
        let mut required_taints = FxHashSet::default();
        for taint in &request.routing_constraints.required_taints {
            if taint.is_empty() || taint.trim() != taint {
                bail!(
                    "interactive replay request {} has an empty or untrimmed required taint {:?}",
                    request.logical_request_id,
                    taint
                );
            }
            if !required_taints.insert(taint.as_str()) {
                bail!(
                    "interactive replay request {} duplicates required taint {:?}",
                    request.logical_request_id,
                    taint
                );
            }
        }
        for (taint, weight) in &request.routing_constraints.preferred_taints {
            if taint.is_empty() || taint.trim() != taint {
                bail!(
                    "interactive replay request {} has an empty or untrimmed preferred taint {:?}",
                    request.logical_request_id,
                    taint
                );
            }
            if !weight.is_finite() {
                bail!(
                    "interactive replay request {} has non-finite preferred-taint weight {} for {:?}",
                    request.logical_request_id,
                    weight,
                    taint
                );
            }
        }
        Ok(())
    }

    fn validate_static_routing(&self, request: &ReplayRequestSpec) -> Result<()> {
        let required_taints = request
            .routing_constraints
            .required_taints
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if !self.static_routing_workers.iter().any(|worker| {
            worker.is_eligible
                && required_taints
                    .iter()
                    .all(|required| worker.taints.contains(required))
        }) {
            bail!(
                "interactive replay request {} has no static active worker satisfying required taints {:?}",
                request.logical_request_id,
                required_taints
            );
        }

        let Some(target) = request.target.as_ref() else {
            return Ok(());
        };
        let worker = self
            .static_routing_workers
            .iter()
            .find(|worker| {
                worker.target.pool_id == target.pool_id
                    && worker.target.worker_id == target.worker_id
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "interactive replay worker {} is not a member of pool {:?}",
                    target.worker_id,
                    target.pool_id
                )
            })?;
        if target.dp_rank != 0 {
            bail!(
                "interactive replay worker {} has no active DP rank {} (active ranks: 1)",
                target.worker_id,
                target.dp_rank
            );
        }
        if !worker.is_eligible {
            bail!(
                "interactive replay worker {} in pool {:?} is unavailable",
                target.worker_id,
                target.pool_id
            );
        }
        if !required_taints
            .iter()
            .all(|required| worker.taints.contains(required))
        {
            bail!(
                "interactive replay worker {} in pool {:?} does not satisfy required taints {:?}",
                target.worker_id,
                target.pool_id,
                required_taints
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::replay::offline::topology::{PoolRouter, WorkerSpec};
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
            attempt_id: "0".to_string(),
            group_id: session_id.to_string(),
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
            routing_constraints: ReplayRoutingConstraints::default(),
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

    fn assert_invalid_workflow_rolls_back(
        constraints: ReplayRoutingConstraints,
        prefix_reset: bool,
        expected_error: &str,
    ) -> Result<()> {
        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, false);
        let mut first = request("rollback-first", "rollback-first-session", 0, 4, &[1], 1);
        first.internal_uuid = Some(Uuid::from_u128(0x501));
        let mut invalid = request(
            "rollback-invalid",
            "rollback-invalid-session",
            0,
            4,
            &[2],
            1,
        );
        invalid.routing_constraints = constraints;
        let mut invalid_authored = agentic_request(invalid, &[], 0.0);
        invalid_authored.prefix_reset = prefix_reset;
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![agentic_request(first.clone(), &[], 0.0), invalid_authored],
        };
        let error = replay
            .append_agentic_workflow(workflow, 0.0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(expected_error),
            "expected {expected_error:?} in {error:?}"
        );
        assert_eq!(replay.snapshot()?.pending_request_count, 0);

        // Reusing the exact authored logical/session/turn/UUID proves that the
        // failed public append did not leak a partially registered identity.
        replay.submit_request(first)?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 1);
        replay.finalize()?;
        Ok(())
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

    fn routed_event_count(events: &[ReplayEvent], logical_id: &str) -> usize {
        events
            .iter()
            .filter(|event| {
                matches!(event, ReplayEvent::Routed(data) if data.logical_request_id == logical_id)
            })
            .count()
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

    fn captured_variant(data: CapturedReplayEventData, ordinal: usize) -> CapturedReplayEvent {
        match ordinal {
            0 => CapturedReplayEvent::PlacementNeeded(data),
            1 => CapturedReplayEvent::Routed(data),
            2 => CapturedReplayEvent::Queued(data),
            3 => CapturedReplayEvent::Admitted(data),
            4 => CapturedReplayEvent::FirstToken(data),
            5 => CapturedReplayEvent::Terminal(data),
            _ => unreachable!("six replay event variants"),
        }
    }

    fn owned_variant(data: ReplayEventData, ordinal: usize) -> ReplayEvent {
        match ordinal {
            0 => ReplayEvent::PlacementNeeded(data),
            1 => ReplayEvent::Routed(data),
            2 => ReplayEvent::Queued(data),
            3 => ReplayEvent::Admitted(data),
            4 => ReplayEvent::FirstToken(data),
            5 => ReplayEvent::Terminal(data),
            _ => unreachable!("six replay event variants"),
        }
    }

    #[test]
    fn captured_all_event_schema_matches_owned_and_is_mutation_isolated() -> Result<()> {
        let uuid = Uuid::from_u128(0x5151);
        let metadata = Arc::new(InteractiveRequestMetadata {
            logical_request_id: "request-captured".to_string(),
            attempt_id: "attempt-captured".to_string(),
            group_id: "group-captured".to_string(),
            internal_uuid: uuid,
            session_id: "session-captured".to_string(),
            authored_turn_index: 7,
            input_length: 16,
            requested_output_length: 8,
            priority: -2,
            strict_priority: 3,
            policy_class: Some("latency".to_string()),
            routing_constraints: ReplayRoutingConstraints {
                required_taints: vec!["trusted".to_string()],
                preferred_taints: [("fast".to_string(), 1.25)].into(),
            },
        });
        let candidate = ReplayPlacementCandidate {
            target: WorkerTarget::new("pool-a", 7, 2),
            active: true,
            draining: false,
            eligible: true,
            constraint_reason: None,
            in_flight_requests: 4,
            queued_requests: Some(5),
            running_requests: Some(6),
            queued_tokens: Some(7),
            running_tokens: Some(8),
            max_num_seqs: Some(9),
            preemption_count: Some(10),
            kv_prefix_overlap_tokens: Some(11),
            kv_capacity_blocks: Some(12),
            kv_occupied_blocks: Some(13),
            kv_free_blocks: Some(14),
            tags: vec!["primary".to_string()],
            taints: vec!["trusted".to_string()],
            capabilities: vec!["chat".to_string()],
        };
        let captured_data = CapturedReplayEventData {
            metadata,
            timestamp_ms: 12.5,
            pool_id: Some(Arc::from("pool-a")),
            worker_id: Some(7),
            dp_rank: Some(2),
            terminal_status: Some(ReplayTerminalStatus::Completed),
            requested_output_length: Some(8),
            emitted_output_count: 8,
            reused_input_tokens: Some(4),
            ttft_ms: Some(1.5),
            e2e_latency_ms: Some(2.5),
            eligible_pool_ids: Arc::new(vec!["pool-a".to_string()]),
            candidates: Arc::new(vec![candidate]),
        };

        for ordinal in 0..6 {
            let captured = captured_variant(captured_data.clone(), ordinal);
            let owned = owned_variant(captured.clone().into_data().into_owned(), ordinal);
            assert_eq!(captured.event_type(), owned.event_type());
            assert_eq!(
                serde_json::to_value(&captured)?,
                serde_json::to_value(&owned)?,
                "captured variant {ordinal} changed the public schema"
            );
        }

        let captured = CapturedReplayEvent::PlacementNeeded(captured_data);
        let mut first = captured.clone().into_owned();
        let second = captured.into_owned();
        let ReplayEvent::PlacementNeeded(first_data) = &mut first else {
            unreachable!()
        };
        first_data.logical_request_id.push_str("-mutated");
        first_data.routing_constraints.required_taints.clear();
        first_data.eligible_pool_ids.clear();
        first_data.candidates[0].tags.clear();
        let ReplayEvent::PlacementNeeded(second_data) = second else {
            unreachable!()
        };
        assert_eq!(second_data.logical_request_id, "request-captured");
        assert_eq!(second_data.routing_constraints.required_taints, ["trusted"]);
        assert_eq!(second_data.eligible_pool_ids, ["pool-a"]);
        assert_eq!(second_data.candidates[0].tags, ["primary"]);
        Ok(())
    }

    #[test]
    fn placement_public_drain_does_not_retain_event_candidate_clone() -> Result<()> {
        let uuid = Uuid::from_u128(0x6161);
        let mut capture = InteractiveCapture::new(true, false, vec!["pool-a".to_string()]);
        capture.register(InteractiveRequestIdentity {
            metadata: Arc::new(InteractiveRequestMetadata {
                logical_request_id: "placement-unwrapped".to_string(),
                attempt_id: "attempt-unwrapped".to_string(),
                group_id: "group-unwrapped".to_string(),
                internal_uuid: uuid,
                session_id: "session-unwrapped".to_string(),
                authored_turn_index: 0,
                input_length: 4,
                requested_output_length: 1,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
                routing_constraints: ReplayRoutingConstraints::default(),
            }),
            ready_at_ms: Some(0.0),
            worker: None,
            emitted_output_count: 0,
            reused_input_tokens: None,
            first_token_ms: None,
            terminal_ms: None,
        })?;
        let candidate = ReplayPlacementCandidate {
            target: WorkerTarget::new("pool-a", 0, 0),
            active: true,
            draining: false,
            eligible: true,
            constraint_reason: None,
            in_flight_requests: 0,
            queued_requests: Some(0),
            running_requests: Some(0),
            queued_tokens: Some(0),
            running_tokens: Some(0),
            max_num_seqs: Some(1),
            preemption_count: Some(0),
            kv_prefix_overlap_tokens: Some(0),
            kv_capacity_blocks: Some(8),
            kv_occupied_blocks: Some(0),
            kv_free_blocks: Some(8),
            tags: vec!["stable".to_string()],
            taints: Vec::new(),
            capabilities: Vec::new(),
        };
        let event = capture.placement_needed_event(uuid, 0.0, vec![candidate])?;
        let ReplayEvent::PlacementNeeded(data) = event else {
            unreachable!()
        };
        assert_eq!(data.candidates[0].tags, ["stable"]);
        let candidates = capture.complete_placement_assignment(uuid)?;
        assert_eq!(candidates[0].tags, ["stable"]);
        Ok(())
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
    fn mid_pass_candidate_overlap_uses_only_committed_kv_state() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 1, true);
        replay.submit_request(request("seed", "seed-session", 0, 8, &[90, 91], 32))?;
        let mut events = drive_to_pending_placement(&mut replay)?;
        replay.assign("seed", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        replay.settle_current_time()?;
        events.extend(replay.drain_events()?);
        let pass_started_at = replay.now_ms()?;
        assert!(
            replay
                .next_event_time_ms()?
                .is_some_and(|completion| completion > pass_started_at),
            "fixture must observe a live in-progress pass"
        );

        let mut mid_pass = request("mid-pass", "mid-pass-session", 0, 8, &[90, 91], 1);
        mid_pass.ready_time_ms = pass_started_at;
        replay.submit_request(mid_pass)?;
        events.extend(drive_to_pending_placement(&mut replay)?);
        let mid_pass_overlap = events
            .iter()
            .rev()
            .find_map(|event| match event {
                ReplayEvent::PlacementNeeded(data) if data.logical_request_id == "mid-pass" => data
                    .candidates
                    .iter()
                    .find(|candidate| candidate.target.worker_id == 0)
                    .and_then(|candidate| candidate.kv_prefix_overlap_tokens),
                _ => None,
            })
            .expect("mid-pass candidate must expose native KV overlap");
        assert_eq!(
            mid_pass_overlap, 0,
            "eager future pass mutations must not leak into policy observations"
        );

        // External placement is a controller boundary: virtual time cannot
        // advance to the seed completion while this request is unassigned.
        // Queue it behind the live seed only after capturing the mid-pass
        // observation under test.
        replay.assign("mid-pass", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        events.extend(drive_to_terminal(&mut replay, "seed")?);
        let mut committed = request("committed", "committed-session", 0, 8, &[90, 91], 1);
        committed.ready_time_ms = replay.now_ms()?;
        replay.submit_request(committed)?;
        replay.settle_current_time()?;
        events.extend(replay.drain_events()?);
        let committed_overlap = events
            .iter()
            .rev()
            .find_map(|event| match event {
                ReplayEvent::PlacementNeeded(data) if data.logical_request_id == "committed" => {
                    data.candidates
                        .iter()
                        .find(|candidate| candidate.target.worker_id == 0)
                        .and_then(|candidate| candidate.kv_prefix_overlap_tokens)
                }
                _ => None,
            })
            .expect("post-completion candidate must expose native KV overlap");
        assert_eq!(committed_overlap, 8);

        replay.assign("committed", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 3);
        replay.finalize()?;
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
    fn routing_constraints_and_prefix_reset_fail_before_identity_registration() -> Result<()> {
        for required_taints in [
            vec!["".to_string()],
            vec![" untrimmed".to_string()],
            vec!["duplicate".to_string(), "duplicate".to_string()],
        ] {
            let expected = if required_taints.len() == 2 {
                "duplicates required taint"
            } else {
                "empty or untrimmed required taint"
            };
            assert_invalid_workflow_rolls_back(
                ReplayRoutingConstraints {
                    required_taints,
                    preferred_taints: Default::default(),
                },
                false,
                expected,
            )?;
        }

        let mut blank_preferred = std::collections::BTreeMap::new();
        blank_preferred.insert(" ".to_string(), 1.0);
        assert_invalid_workflow_rolls_back(
            ReplayRoutingConstraints {
                required_taints: Vec::new(),
                preferred_taints: blank_preferred,
            },
            false,
            "empty or untrimmed preferred taint",
        )?;

        let mut nan_preferred = std::collections::BTreeMap::new();
        nan_preferred.insert("finite-name".to_string(), f32::NAN);
        assert_invalid_workflow_rolls_back(
            ReplayRoutingConstraints {
                required_taints: Vec::new(),
                preferred_taints: nan_preferred,
            },
            false,
            "non-finite preferred-taint weight",
        )?;

        assert_invalid_workflow_rolls_back(
            ReplayRoutingConstraints::default(),
            true,
            "unsupported prefix_reset=true",
        )?;

        let mut replay = session(ReplaySessionRouter::RoundRobin, 1, false);
        let mut negative = request("negative-weight", "negative-weight", 0, 4, &[3], 1);
        negative
            .routing_constraints
            .preferred_taints
            .insert("avoid-if-possible".to_string(), -1.0);
        replay.submit_request(negative)?;
        replay.close_admission()?;
        assert_eq!(terminal_count(&drive_to_drained(&mut replay)?), 1);
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn non_pooled_static_config_rejects_startup_and_taint_cardinality() {
        let mut startup = replay_args(false);
        startup.startup_time = Some(1.0);
        let error =
            OfflineReplaySession::new(&startup, 2, TRACE_BLOCK_SIZE, ReplaySessionRouter::External)
                .err()
                .expect("nonzero startup must fail")
                .to_string();
        assert!(error.contains("must not configure startup_time"));

        for taint_count in [1, 3] {
            let mut args = replay_args(false);
            args.worker_taints = (0..taint_count)
                .map(|index| std::collections::HashSet::from([format!("taint-{index}")]))
                .collect();
            let error = OfflineReplaySession::new(
                &args,
                2,
                TRACE_BLOCK_SIZE,
                ReplaySessionRouter::External,
            )
            .err()
            .expect("wrong worker_taints cardinality must fail")
            .to_string();
            assert!(error.contains("worker_taints must be empty or contain exactly"));
        }

        let mut untrimmed = replay_args(false);
        untrimmed.worker_taints = vec![
            std::collections::HashSet::from([" secure ".to_string()]),
            std::collections::HashSet::new(),
        ];
        let error = OfflineReplaySession::new(
            &untrimmed,
            2,
            TRACE_BLOCK_SIZE,
            ReplaySessionRouter::External,
        )
        .err()
        .expect("untrimmed worker taint must fail")
        .to_string();
        assert!(error.contains("empty or untrimmed taint"));
    }

    #[test]
    fn external_placement_needed_assign_validates_worker_and_rank() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        settle_empty_open(&mut replay)?;
        let mut external = request("external", "external-session", 0, 4, &[10], 2);
        external.priority = -4;
        external.strict_priority = 9;
        external.policy_class = Some("latency-sensitive".to_string());
        external
            .routing_constraints
            .preferred_taints
            .insert("near".to_string(), 0.5);
        replay.submit_request(external)?;

        let mut events = drive_to_pending_placement(&mut replay)?;
        let pending = replay.pending_placements()?;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].logical_request_id, "external");
        assert_eq!(pending[0].input_length, 4);
        assert_eq!(pending[0].priority, -4);
        assert_eq!(pending[0].strict_priority, 9);
        assert_eq!(
            pending[0].policy_class.as_deref(),
            Some("latency-sensitive")
        );
        assert_eq!(
            pending[0].routing_constraints.preferred_taints.get("near"),
            Some(&0.5)
        );
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
                        pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
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
                        pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
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
                        pool_id: "missing-pool".to_string(),
                        worker_id: 0,
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
                        pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
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
                pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
                worker_id: 1,
                dp_rank: 0,
            },
        )?;
        assert!(replay.pending_placements()?.is_empty());
        assert!(
            replay
                .assign(
                    "external",
                    WorkerTarget {
                        pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
                        worker_id: 1,
                        dp_rank: 0,
                    },
                )
                .unwrap_err()
                .to_string()
                .contains("not awaiting placement")
        );
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 1);
        assert_eq!(routed_event(&events, "external").worker_id, Some(1));
        assert_eq!(terminal_event(&events, "external").worker_id, Some(1));
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn outstanding_external_placement_blocks_same_time_appended_preassignment() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        replay.submit_request(request("barrier", "barrier", 0, 4, &[18], 2))?;
        let mut events = drive_to_pending_placement(&mut replay)?;
        let barrier_time = replay.now_ms()?;
        assert_eq!(routed_event_count(&events, "barrier"), 0);

        let mut preassigned = request("preassigned", "preassigned", 0, 4, &[19], 2);
        preassigned.ready_time_ms = barrier_time;
        preassigned.target = Some(WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 1, 0));
        replay.submit_request(preassigned)?;
        let before = replay.snapshot()?;
        assert_eq!(before.pending_request_count, 2);
        assert_eq!(before.pending_placement_count, 1);
        assert!(
            before
                .workers
                .iter()
                .all(|worker| worker.in_flight_requests == 0)
        );

        assert!(matches!(
            replay.settle_current_time()?,
            ReplayStepStatus::Quiescent { now_ms } if now_ms == barrier_time
        ));
        assert!(matches!(
            replay.advance_to(barrier_time)?,
            ReplayStepStatus::Quiescent { now_ms } if now_ms == barrier_time
        ));
        assert!(replay.drain_events()?.is_empty());
        let blocked = replay.snapshot()?;
        assert_eq!(blocked.pending_request_count, 2);
        assert_eq!(blocked.pending_placement_count, 1);
        assert!(
            blocked
                .workers
                .iter()
                .all(|worker| worker.in_flight_requests == 0)
        );

        replay.assign("barrier", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        let released = replay.drain_events()?;
        assert_eq!(routed_event_count(&released, "barrier"), 1);
        assert_eq!(routed_event_count(&released, "preassigned"), 1);
        events.extend(released);

        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 2);
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn pinned_target_emits_exactly_once_terminal_and_preserves_final_identity() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        settle_empty_open(&mut replay)?;

        let uuid = Uuid::from_u128(0xabc);
        let mut pinned = request("pinned", "pinned-session", 4, 8, &[20, 21], 3);
        pinned.attempt_id = "attempt-7".to_string();
        pinned.group_id = "group-9".to_string();
        pinned.internal_uuid = Some(uuid);
        pinned.target = Some(WorkerTarget {
            pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
            worker_id: 99,
            dp_rank: 0,
        });
        assert!(replay.submit_request(pinned.clone()).is_err());

        pinned.target = Some(WorkerTarget {
            pool_id: DEFAULT_REPLAY_POOL_ID.to_string(),
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
        assert_eq!(terminal.attempt_id, "attempt-7");
        assert_eq!(terminal.group_id, "group-9");
        assert_eq!(terminal.session_id, "pinned-session");
        assert_eq!(terminal.authored_turn_index, 4);
        assert_eq!(terminal.worker_id, Some(1));
        assert_eq!(terminal.dp_rank, Some(0));
        assert_eq!(
            terminal.terminal_status,
            Some(ReplayTerminalStatus::Completed)
        );
        assert_eq!(terminal.requested_output_length, Some(3));
        assert_eq!(terminal.emitted_output_count, 3);
        assert!(terminal.e2e_latency_ms.is_some());

        let report = replay.finalize()?;
        assert_eq!(report.request_counts.num_requests, 1);
        assert_eq!(report.request_counts.completed_requests, 1);
        assert_eq!(report.per_request.len(), 1);
        let record = &report.per_request[0];
        assert_eq!(record.logical_request_id.as_deref(), Some("pinned"));
        assert_eq!(record.attempt_id.as_deref(), Some("attempt-7"));
        assert_eq!(record.group_id.as_deref(), Some("group-9"));
        assert_eq!(record.session_id.as_deref(), Some("pinned-session"));
        assert_eq!(record.authored_turn_index, Some(4));
        assert_eq!(record.uuid, uuid.to_string());
        assert_eq!(record.requested_output_length, 3);
        assert_eq!(record.output_length, 3);
        assert_eq!(record.terminal_status, ReplayTerminalStatus::Completed);
        let route = record
            .routing_history
            .last()
            .expect("authored exact placement must retain routing evidence");
        assert_eq!(route.pool_id.as_deref(), Some(DEFAULT_REPLAY_POOL_ID));
        assert_eq!(route.worker_id, Some(1));
        assert_eq!(route.dp_rank, Some(0));
        assert_eq!(route.logical_worker_id, Some(1));
        assert!(route.scheduler_id.is_some());
        assert!(route.routed_at_ms.is_some());
        assert_eq!(record.admission_history.len(), 1);
        assert!(record.first_admit_ms.is_some());
        assert!(record.terminal_time_ms >= record.first_admit_ms.unwrap());
        Ok(())
    }

    #[test]
    fn non_pooled_external_placement_uses_engine_worker_taints_consistently() -> Result<()> {
        let mut args = replay_args(false);
        args.worker_taints = vec![
            std::collections::HashSet::from(["plain".to_string()]),
            std::collections::HashSet::from(["secure".to_string()]),
        ];
        let mut replay =
            OfflineReplaySession::new(&args, 2, TRACE_BLOCK_SIZE, ReplaySessionRouter::External)?;

        let mut pinned = request("nonpool-pinned", "nonpool-pinned", 0, 4, &[24], 1);
        pinned.routing_constraints.required_taints = vec!["secure".to_string()];
        pinned.target = Some(WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0));
        assert!(
            replay
                .submit_request(pinned.clone())
                .unwrap_err()
                .to_string()
                .contains("does not satisfy required taints")
        );
        assert_eq!(replay.snapshot()?.pending_request_count, 0);

        pinned.target = Some(WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 1, 0));
        replay.submit_request(pinned)?;
        let mut live = request("nonpool-live", "nonpool-live", 0, 4, &[25], 1);
        live.routing_constraints.required_taints = vec!["secure".to_string()];
        replay.submit_request(live)?;
        let mut events = drive_to_pending_placement(&mut replay)?;
        let pending = &replay.pending_placements()?[0];
        assert_eq!(pending.logical_request_id, "nonpool-live");
        assert_eq!(
            pending
                .candidates
                .iter()
                .map(|candidate| (candidate.target.worker_id, candidate.eligible))
                .collect::<Vec<_>>(),
            vec![(0, false), (1, true)]
        );

        assert!(
            replay
                .assign(
                    "nonpool-live",
                    WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0)
                )
                .unwrap_err()
                .to_string()
                .contains("does not satisfy required taints")
        );
        assert_eq!(replay.pending_placements()?.len(), 1);
        replay.assign(
            "nonpool-live",
            WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 1, 0),
        )?;
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 2);
        let report = replay.finalize()?;
        assert!(
            report
                .per_request
                .iter()
                .all(|record| record.worker_id == Some(1))
        );
        Ok(())
    }

    #[test]
    fn authored_and_live_taint_rejection_are_atomic_and_recoverable() -> Result<()> {
        let mut plain = WorkerSpec::active(10);
        plain.taints = vec!["plain".to_string()];
        let mut secure = WorkerSpec::active(20);
        secure.taints = vec!["secure".to_string()];
        let mut replay = OfflineReplaySession::new_pooled(
            vec![PoolSpec {
                pool_id: "pool".to_string(),
                engine_args: replay_args(false),
                workers: vec![plain, secure],
                router: PoolRouter::RoundRobin,
            }],
            TRACE_BLOCK_SIZE,
        )?;

        let mut pinned = request("pinned-secure", "pinned-secure", 0, 4, &[25], 1);
        pinned.routing_constraints.required_taints = vec!["secure".to_string()];
        pinned.target = Some(WorkerTarget::new("pool", 10, 0));
        assert!(
            replay
                .submit_request(pinned.clone())
                .unwrap_err()
                .to_string()
                .contains("does not satisfy required taints")
        );
        assert_eq!(replay.snapshot()?.pending_request_count, 0);
        assert!(replay.pending_placements()?.is_empty());

        pinned.target = Some(WorkerTarget::new("pool", 20, 0));
        replay.submit_request(pinned)?;
        let mut live = request("live-secure", "live-secure", 0, 4, &[26], 1);
        live.routing_constraints.required_taints = vec!["secure".to_string()];
        replay.submit_request(live)?;
        let mut events = drive_to_pending_placement(&mut replay)?;
        let placement = events
            .iter()
            .find_map(|event| match event {
                ReplayEvent::PlacementNeeded(data) if data.logical_request_id == "live-secure" => {
                    Some(data)
                }
                _ => None,
            })
            .expect("live constrained request must reach placement");
        let plain = placement
            .candidates
            .iter()
            .find(|candidate| candidate.target.worker_id == 10)
            .unwrap();
        let secure = placement
            .candidates
            .iter()
            .find(|candidate| candidate.target.worker_id == 20)
            .unwrap();
        let announced_candidates = placement.candidates.clone();
        assert!(!plain.eligible);
        assert!(
            plain
                .constraint_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("missing required taints"))
        );
        assert!(secure.eligible);

        assert!(
            replay
                .assign("live-secure", WorkerTarget::new("pool", 10, 0))
                .unwrap_err()
                .to_string()
                .contains("does not satisfy required taints")
        );
        let pending_after_rejection = replay.pending_placements()?;
        assert_eq!(pending_after_rejection.len(), 1);
        assert_eq!(
            pending_after_rejection[0].candidates, announced_candidates,
            "a rejected assignment must retain its causally frozen observation"
        );
        replay.assign("live-secure", WorkerTarget::new("pool", 20, 0))?;
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(
            routed_event(&events, "live-secure").candidates,
            announced_candidates,
            "the successful retry must publish the original observation"
        );
        assert_eq!(terminal_count(&events), 2);
        let report = replay.finalize()?;
        assert_eq!(report.request_counts.completed_requests, 2);
        Ok(())
    }

    #[test]
    fn native_round_robin_enforces_required_worker_taints() -> Result<()> {
        let mut args = replay_args(false);
        args.worker_taints = vec![
            std::collections::HashSet::from(["general".to_string()]),
            std::collections::HashSet::from(["secure".to_string()]),
        ];
        let mut replay =
            OfflineReplaySession::new(&args, 2, TRACE_BLOCK_SIZE, ReplaySessionRouter::RoundRobin)?;
        let mut constrained = request("native-secure", "native-secure", 0, 4, &[27], 1);
        constrained.routing_constraints.required_taints = vec!["secure".to_string()];
        replay.submit_request(constrained)?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 1);
        let report = replay.finalize()?;
        let record = report
            .per_request
            .iter()
            .find(|record| record.logical_request_id.as_deref() == Some("native-secure"))
            .unwrap();
        assert_eq!(record.worker_id, Some(1));
        assert!(
            record
                .routing_history
                .iter()
                .all(|route| route.routed_at_ms.is_some()),
            "native round-robin final evidence must carry route time"
        );
        assert_eq!(
            record
                .routing_history
                .iter()
                .filter_map(|route| route.reported_overlap_tokens)
                .next(),
            None,
            "round-robin does not fabricate an unobserved KV overlap"
        );
        Ok(())
    }

    #[test]
    fn unsatisfiable_static_taints_reject_before_mutation_for_every_router() -> Result<()> {
        for router in [
            ReplaySessionRouter::External,
            ReplaySessionRouter::RoundRobin,
            ReplaySessionRouter::KvRouter,
        ] {
            let mut args = replay_args(false);
            args.worker_taints = vec![std::collections::HashSet::from(["plain".to_string()])];
            let mut replay = OfflineReplaySession::new(&args, 1, TRACE_BLOCK_SIZE, router)?;
            let mut authored = request("no-eligible", "no-eligible", 0, 4, &[28], 1);
            authored.internal_uuid = Some(Uuid::from_u128(0x601));
            authored.routing_constraints.required_taints = vec!["secure".to_string()];
            if router == ReplaySessionRouter::External {
                authored.target = Some(WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0));
            }

            let error = replay.submit_request(authored.clone()).unwrap_err();
            assert!(error.to_string().contains("no static active worker"));
            assert_eq!(replay.snapshot()?.pending_request_count, 0);
            assert!(replay.pending_placements()?.is_empty());
            assert!(replay.drain_events()?.is_empty());

            authored.routing_constraints = ReplayRoutingConstraints::default();
            replay.submit_request(authored)?;
            let mut events = Vec::new();
            replay.close_admission()?;
            events.extend(drive_to_drained(&mut replay)?);
            assert_eq!(terminal_count(&events), 1);
            let report = replay.finalize()?;
            assert_eq!(report.request_counts.num_requests, 1);
            assert_eq!(report.request_counts.completed_requests, 1);
        }
        Ok(())
    }

    #[test]
    fn unsatisfiable_child_rolls_back_the_complete_workflow() -> Result<()> {
        let mut args = replay_args(false);
        args.worker_taints = vec![std::collections::HashSet::from(["plain".to_string()])];
        let mut replay =
            OfflineReplaySession::new(&args, 1, TRACE_BLOCK_SIZE, ReplaySessionRouter::RoundRobin)?;
        let mut root = request("rollback-root", "rollback-root", 0, 4, &[29], 1);
        root.internal_uuid = Some(Uuid::from_u128(0x602));
        let mut child = request("rollback-child", "rollback-child", 0, 4, &[30], 1);
        child.internal_uuid = Some(Uuid::from_u128(0x603));
        child.routing_constraints.required_taints = vec!["secure".to_string()];
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![
                agentic_request(root.clone(), &[], 0.0),
                agentic_request(child.clone(), &["rollback-root"], 0.0),
            ],
        };

        let error = replay.append_agentic_workflow(workflow, 0.0).unwrap_err();
        assert!(error.to_string().contains("no static active worker"));
        assert_eq!(replay.snapshot()?.pending_request_count, 0);
        assert!(replay.drain_events()?.is_empty());

        child.routing_constraints = ReplayRoutingConstraints::default();
        replay.append_agentic_workflow(
            ReplayAgenticWorkflow {
                trace_block_size: TRACE_BLOCK_SIZE,
                requests: vec![
                    agentic_request(root, &[], 0.0),
                    agentic_request(child, &["rollback-root"], 0.0),
                ],
            },
            0.0,
        )?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 2);
        let report = replay.finalize()?;
        assert_eq!(report.request_counts.completed_requests, 2);
        Ok(())
    }

    #[test]
    fn pool_assignment_without_an_eligible_worker_is_recoverable() -> Result<()> {
        let mut plain = WorkerSpec::active(10);
        plain.taints = vec!["plain".to_string()];
        let mut secure = WorkerSpec::active(20);
        secure.taints = vec!["secure".to_string()];
        let mut replay = OfflineReplaySession::new_pooled(
            vec![
                PoolSpec {
                    pool_id: "plain".to_string(),
                    engine_args: replay_args(false),
                    workers: vec![plain],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "secure".to_string(),
                    engine_args: replay_args(false),
                    workers: vec![secure],
                    router: PoolRouter::RoundRobin,
                },
            ],
            TRACE_BLOCK_SIZE,
        )?;
        let mut constrained = request("pool-secure", "pool-secure", 0, 4, &[31], 1);
        constrained.routing_constraints.required_taints = vec!["secure".to_string()];
        replay.submit_request(constrained)?;
        let mut events = drive_to_pending_placement(&mut replay)?;

        let error = replay.assign_pool("pool-secure", "plain").unwrap_err();
        assert!(error.to_string().contains("no worker eligible"));
        assert_eq!(replay.pending_placements()?.len(), 1);
        replay.assign_pool("pool-secure", "secure")?;
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 1);
        let report = replay.finalize()?;
        assert_eq!(report.per_request[0].pool_id.as_deref(), Some("secure"));
        assert_eq!(report.per_request[0].worker_id, Some(20));
        Ok(())
    }

    #[test]
    fn heterogeneous_gpu_parallelism_fails_closed_in_report_accounting() {
        let one_gpu = replay_args(false);
        let mut two_gpu = replay_args(false);
        two_gpu.aic_tp_size = Some(2);
        let error = OfflineReplaySession::new_pooled(
            vec![
                PoolSpec {
                    pool_id: "one".to_string(),
                    engine_args: one_gpu,
                    workers: vec![WorkerSpec::active(1)],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "two".to_string(),
                    engine_args: two_gpu,
                    workers: vec![WorkerSpec::active(2)],
                    router: PoolRouter::RoundRobin,
                },
            ],
            TRACE_BLOCK_SIZE,
        )
        .err()
        .expect("mixed GPU counts must not silently use the first pool");
        assert!(
            error
                .to_string()
                .contains("heterogeneous GPUs-per-worker accounting is unsupported")
        );
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
    fn all_same_time_terminals_precede_dependent_placement_events() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![
                agentic_request(request("root-a", "session-a", 0, 4, &[41], 1), &[], 0.0),
                agentic_request(request("root-b", "session-b", 0, 4, &[42], 1), &[], 0.0),
                agentic_request(
                    request("child-a", "child-session-a", 0, 4, &[43], 1),
                    &["root-a"],
                    0.0,
                ),
                agentic_request(
                    request("child-b", "child-session-b", 0, 4, &[44], 1),
                    &["root-b"],
                    0.0,
                ),
            ],
        };
        replay.append_agentic_workflow(workflow, 0.0)?;
        let mut events = drive_to_pending_placement(&mut replay)?;
        assert_eq!(replay.pending_placements()?[0].logical_request_id, "root-a");
        replay.assign("root-a", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        events.extend(replay.drain_events()?);
        assert_eq!(replay.pending_placements()?[0].logical_request_id, "root-b");
        replay.assign("root-b", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 1, 0))?;

        for _ in 0..MAX_STEPS {
            replay.settle_current_time()?;
            events.extend(replay.drain_events()?);
            if !replay.pending_placements()?.is_empty() {
                break;
            }
            replay.advance_next()?;
            events.extend(replay.drain_events()?);
        }
        assert_eq!(
            replay.pending_placements()?[0].logical_request_id,
            "child-a"
        );
        replay.assign("child-a", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        events.extend(replay.drain_events()?);
        assert_eq!(
            replay.pending_placements()?[0].logical_request_id,
            "child-b"
        );
        let child_placements = events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                ReplayEvent::PlacementNeeded(data)
                    if data.logical_request_id == "child-a"
                        || data.logical_request_id == "child-b" =>
                {
                    Some((index, data.timestamp_ms))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(child_placements.len(), 2);
        let child_boundary = child_placements[0].1;
        let root_terminals = events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                ReplayEvent::Terminal(data)
                    if data.logical_request_id == "root-a"
                        || data.logical_request_id == "root-b" =>
                {
                    Some((index, data.timestamp_ms))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(root_terminals.len(), 2);
        assert!(
            root_terminals
                .iter()
                .all(|(_, timestamp)| *timestamp == child_boundary)
        );
        let first_child_index = child_placements
            .iter()
            .map(|(index, _)| *index)
            .min()
            .unwrap();
        assert!(
            root_terminals
                .iter()
                .all(|(index, _)| *index < first_child_index),
            "every terminal at t must be visible before any dependent placement at t"
        );

        replay.assign("child-b", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 1, 0))?;
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 4);
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn rejected_parent_cancels_child_with_parent_terminal_first() -> Result<()> {
        let mut replay = OfflineReplaySession::new(
            &constrained_replay_args(),
            1,
            TRACE_BLOCK_SIZE,
            ReplaySessionRouter::RoundRobin,
        )?;
        let mut child = request("after-rejection", "child-session", 0, 4, &[6], 1);
        child.attempt_id = "attempt-child".to_string();
        child.group_id = "failed-workflow".to_string();
        let workflow = ReplayAgenticWorkflow {
            trace_block_size: TRACE_BLOCK_SIZE,
            requests: vec![
                agentic_request(
                    request("oversized", "oversized-session", 0, 20, &[1, 2, 3, 4, 5], 1),
                    &[],
                    0.0,
                ),
                agentic_request(child, &["oversized"], 0.0),
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
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ReplayEvent::Routed(data)
                    if data.logical_request_id == "after-rejection")),
            "dependency-canceled child must never be routed"
        );
        let child_terminal = terminal_event(&events, "after-rejection");
        assert_eq!(
            child_terminal.terminal_status,
            Some(ReplayTerminalStatus::Canceled)
        );
        assert_eq!(child_terminal.attempt_id, "attempt-child");
        assert_eq!(child_terminal.group_id, "failed-workflow");
        let parent_terminal_index = events
            .iter()
            .position(|event| {
                matches!(event, ReplayEvent::Terminal(data)
                if data.logical_request_id == "oversized")
            })
            .expect("missing parent terminal");
        let child_terminal_index = events
            .iter()
            .position(|event| {
                matches!(event, ReplayEvent::Terminal(data)
                if data.logical_request_id == "after-rejection")
            })
            .expect("missing child terminal");
        assert!(parent_terminal_index < child_terminal_index);

        let report = replay.finalize()?;
        assert_eq!(report.request_counts.num_requests, 2);
        assert_eq!(report.request_counts.completed_requests, 0);
        assert_eq!(report.per_request.len(), 2);
        let child_record = report
            .per_request
            .iter()
            .find(|record| record.logical_request_id.as_deref() == Some("after-rejection"))
            .expect("missing dependency-canceled child record");
        assert_eq!(child_record.terminal_status, ReplayTerminalStatus::Canceled);
        assert_eq!(child_record.attempt_id.as_deref(), Some("attempt-child"));
        assert_eq!(child_record.group_id.as_deref(), Some("failed-workflow"));
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
        let first_bytes = serde_json::to_vec(&first)?;
        let second_bytes = serde_json::to_vec(&second)?;
        assert_eq!(first_bytes, second_bytes);
        assert_eq!(
            blake3::hash(&first_bytes),
            blake3::hash(&second_bytes),
            "canonical real-Mocker event/report digests must be byte stable"
        );
        Ok(())
    }

    fn contention_fixture(split: bool) -> Result<TraceSimulationReport> {
        let mut args = replay_args(false);
        args.max_num_seqs = Some(1);
        let mut replay = OfflineReplaySession::new_pooled(
            vec![
                PoolSpec {
                    pool_id: "left".to_string(),
                    engine_args: args.clone(),
                    workers: vec![WorkerSpec::active(0)],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "right".to_string(),
                    engine_args: args,
                    workers: vec![WorkerSpec::active(0)],
                    router: PoolRouter::RoundRobin,
                },
            ],
            TRACE_BLOCK_SIZE,
        )?;
        replay.submit_request(request("contend-a", "contend-a", 0, 8, &[1, 2], 24))?;
        replay.submit_request(request("contend-b", "contend-b", 0, 8, &[3, 4], 24))?;
        drive_to_pending_placement(&mut replay)?;
        replay.assign("contend-a", WorkerTarget::new("left", 0, 0))?;
        replay.drain_events()?;
        replay.assign(
            "contend-b",
            WorkerTarget::new(if split { "right" } else { "left" }, 0, 0),
        )?;
        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 2);
        replay.finalize()
    }

    #[test]
    fn colocated_requests_experience_real_mocker_contention_relative_to_split_pools() -> Result<()>
    {
        let colocated = contention_fixture(false)?;
        let split = contention_fixture(true)?;
        let last_terminal = |report: &TraceSimulationReport| {
            report
                .per_request
                .iter()
                .map(|record| record.terminal_time_ms)
                .max_by(f64::total_cmp)
                .unwrap()
        };
        assert!(
            last_terminal(&colocated) > last_terminal(&split),
            "max_num_seqs=1 must serialize co-located work while independent pools run concurrently"
        );
        assert_eq!(
            colocated
                .per_request
                .iter()
                .map(|record| record.pool_id.as_deref())
                .collect::<Vec<_>>(),
            [Some("left"), Some("left")]
        );
        assert!(
            split
                .per_request
                .iter()
                .any(|record| record.pool_id.as_deref() == Some("right"))
        );
        Ok(())
    }

    #[test]
    fn kv_reuse_and_candidate_overlap_are_pool_local() -> Result<()> {
        let args = replay_args(true);
        let mut replay = OfflineReplaySession::new_pooled(
            vec![
                PoolSpec {
                    pool_id: "cached".to_string(),
                    engine_args: args.clone(),
                    workers: vec![WorkerSpec::active(7)],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "cold".to_string(),
                    engine_args: args,
                    workers: vec![WorkerSpec::active(7)],
                    router: PoolRouter::RoundRobin,
                },
            ],
            TRACE_BLOCK_SIZE,
        )?;

        replay.submit_request(request("pool-seed", "pool-seed", 0, 8, &[80, 81], 1))?;
        drive_to_pending_placement(&mut replay)?;
        replay.assign("pool-seed", WorkerTarget::new("cached", 7, 0))?;
        drive_to_terminal(&mut replay, "pool-seed")?;

        let mut cold_probe = request("cold-probe", "cold-probe", 0, 8, &[80, 81], 1);
        cold_probe.ready_time_ms = replay.now_ms()?;
        replay.submit_request(cold_probe)?;
        let events = drive_to_pending_placement(&mut replay)?;
        let placement = events
            .iter()
            .find_map(|event| match event {
                ReplayEvent::PlacementNeeded(data) if data.logical_request_id == "cold-probe" => {
                    Some(data)
                }
                _ => None,
            })
            .expect("cold probe must request placement");
        let overlap = |pool_id: &str| {
            placement
                .candidates
                .iter()
                .find(|candidate| candidate.target.pool_id == pool_id)
                .and_then(|candidate| candidate.kv_prefix_overlap_tokens)
                .unwrap()
        };
        // Routing observes the full 8-token prefix. Scheduler admission below
        // deliberately recomputes the final block-aligned prompt block, so
        // the admitted reuse count is one 4-token block even though the
        // nonmutating native route fact reports both blocks.
        assert_eq!(overlap("cached"), 8);
        assert_eq!(overlap("cold"), 0);
        replay.assign("cold-probe", WorkerTarget::new("cold", 7, 0))?;
        let cold_events = drive_to_terminal(&mut replay, "cold-probe")?;
        assert_eq!(admitted_reuse(&cold_events, "cold-probe"), 0);

        let mut cached_probe = request("cached-probe", "cached-probe", 0, 8, &[80, 81], 1);
        cached_probe.ready_time_ms = replay.now_ms()?;
        replay.submit_request(cached_probe)?;
        drive_to_pending_placement(&mut replay)?;
        replay.assign_pool("cached-probe", "cached")?;
        let cached_events = drive_to_terminal(&mut replay, "cached-probe")?;
        assert_eq!(
            admitted_reuse(&cached_events, "cached-probe"),
            TRACE_BLOCK_SIZE
        );

        replay.close_admission()?;
        drive_to_drained(&mut replay)?;
        let report = replay.finalize()?;
        assert_eq!(report.request_counts.completed_requests, 3);
        let cached_record = report
            .per_request
            .iter()
            .find(|record| record.logical_request_id.as_deref() == Some("cached-probe"))
            .unwrap();
        assert_eq!(
            cached_record
                .routing_history
                .iter()
                .filter_map(|route| route.reported_overlap_tokens)
                .max(),
            Some(8),
            "pool-only assignment reports the selected worker's committed overlap"
        );
        Ok(())
    }

    #[test]
    fn external_pool_assignment_round_robins_eligible_workers_and_recovers() -> Result<()> {
        let mut trusted = WorkerSpec::active(10);
        trusted.taints.push("trusted".to_string());
        let mut replay = OfflineReplaySession::new_pooled(
            vec![
                PoolSpec {
                    pool_id: "rotate".to_string(),
                    engine_args: replay_args(false),
                    workers: vec![WorkerSpec::active(10), WorkerSpec::active(20)],
                    router: PoolRouter::RoundRobin,
                },
                PoolSpec {
                    pool_id: "trusted".to_string(),
                    engine_args: replay_args(false),
                    workers: vec![trusted],
                    router: PoolRouter::RoundRobin,
                },
            ],
            TRACE_BLOCK_SIZE,
        )?;

        for index in 0..3 {
            replay.submit_request(request(
                &format!("rotate-{index}"),
                &format!("rotate-session-{index}"),
                0,
                4,
                &[100 + index],
                1,
            ))?;
        }
        let mut events = drive_to_pending_placement(&mut replay)?;
        for index in 0..3 {
            let logical_id = format!("rotate-{index}");
            assert_eq!(
                replay.pending_placements()?[0].logical_request_id,
                logical_id
            );
            replay.assign_pool(&logical_id, "rotate")?;
            events.extend(replay.drain_events()?);
        }
        let routed_workers = (0..3)
            .map(|index| routed_event(&events, &format!("rotate-{index}")).worker_id)
            .collect::<Vec<_>>();
        assert_eq!(routed_workers, vec![Some(10), Some(20), Some(10)]);

        let mut constrained = request("pool-recovery", "pool-recovery", 0, 4, &[200], 1);
        constrained.routing_constraints.required_taints = vec!["trusted".to_string()];
        replay.submit_request(constrained)?;
        events.extend(drive_to_pending_placement(&mut replay)?);
        let error = replay
            .assign_pool("pool-recovery", "rotate")
            .unwrap_err()
            .to_string();
        assert!(error.contains("required taints") || error.contains("eligible"));
        assert_eq!(
            replay.pending_placements()?[0].logical_request_id,
            "pool-recovery"
        );
        replay.assign_pool("pool-recovery", "trusted")?;

        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        assert_eq!(terminal_count(&events), 4);
        let report = replay.finalize()?;
        let recovered = report
            .per_request
            .iter()
            .find(|record| record.logical_request_id.as_deref() == Some("pool-recovery"))
            .unwrap();
        assert_eq!(recovered.pool_id.as_deref(), Some("trusted"));
        assert_eq!(recovered.worker_id, Some(10));
        Ok(())
    }

    fn seed_then_probe_overlap(router: ReplaySessionRouter) -> Result<(usize, usize)> {
        let mut replay = session(router, 1, true);
        replay.submit_request(request("overlap-seed", "overlap-seed", 0, 6, &[60, 61], 1))?;
        let mut events = Vec::new();
        if router == ReplaySessionRouter::External {
            events.extend(drive_to_pending_placement(&mut replay)?);
            replay.assign(
                "overlap-seed",
                WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0),
            )?;
        }
        events.extend(drive_to_terminal(&mut replay, "overlap-seed")?);

        let mut probe = request("overlap-probe", "overlap-probe", 0, 6, &[60, 61], 1);
        probe.ready_time_ms = replay.now_ms()?;
        replay.submit_request(probe)?;
        let external_overlap = if router == ReplaySessionRouter::External {
            events.extend(drive_to_pending_placement(&mut replay)?);
            let overlap = events
                .iter()
                .rev()
                .find_map(|event| match event {
                    ReplayEvent::PlacementNeeded(data)
                        if data.logical_request_id == "overlap-probe" =>
                    {
                        data.candidates
                            .iter()
                            .find(|candidate| candidate.target.worker_id == 0)
                            .and_then(|candidate| candidate.kv_prefix_overlap_tokens)
                    }
                    _ => None,
                })
                .expect("external probe must expose a KV overlap fact");
            replay.assign(
                "overlap-probe",
                WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0),
            )?;
            overlap
        } else {
            0
        };
        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        let report = replay.finalize()?;
        let probe_record = report
            .per_request
            .iter()
            .find(|record| record.logical_request_id.as_deref() == Some("overlap-probe"))
            .unwrap();
        let routed_overlap = probe_record
            .routing_history
            .iter()
            .filter_map(|route| route.reported_overlap_tokens)
            .max()
            .unwrap();
        assert!(
            probe_record
                .routing_history
                .iter()
                .all(|route| route.routed_at_ms.is_some()),
            "every effective external/native KV route must carry route time"
        );
        if router == ReplaySessionRouter::KvRouter {
            let routed = routed_event(&events, "overlap-probe");
            assert_eq!(routed.requested_output_length, None);
            assert_eq!(routed.ttft_ms, None);
            assert_eq!(routed.e2e_latency_ms, None);
            assert_eq!(routed.candidates.len(), 1);
            assert_eq!(
                routed.candidates[0].kv_prefix_overlap_tokens,
                Some(routed_overlap),
                "native KV Routed must retain the actual shared pre-dispatch candidate view"
            );
            assert_eq!(
                routed.candidates[0].in_flight_requests, 0,
                "candidate load must be captured before physical dispatch"
            );
        }
        Ok((external_overlap, routed_overlap))
    }

    #[test]
    fn external_candidate_overlap_matches_native_kv_router_fact() -> Result<()> {
        let (external_overlap, external_route_fact) =
            seed_then_probe_overlap(ReplaySessionRouter::External)?;
        let (_, native_route_fact) = seed_then_probe_overlap(ReplaySessionRouter::KvRouter)?;
        assert_eq!(external_overlap, TRACE_BLOCK_SIZE);
        assert_eq!(external_route_fact, external_overlap);
        assert_eq!(external_overlap, native_route_fact);
        Ok(())
    }

    #[test]
    fn simultaneous_external_placements_observe_each_prior_assignment() -> Result<()> {
        let mut replay = session(ReplaySessionRouter::External, 2, false);
        for ordinal in 0..3 {
            replay.submit_request(request(
                &format!("fresh-{ordinal}"),
                &format!("fresh-session-{ordinal}"),
                0,
                8,
                &[700 + ordinal as u32, 800 + ordinal as u32],
                4,
            ))?;
        }

        let mut boundary_events = drive_to_pending_placement(&mut replay)?;
        let mut chosen_workers = Vec::new();
        let mut observed_loads = Vec::new();
        for ordinal in 0..3 {
            let placement_events = boundary_events
                .iter()
                .filter_map(|event| match event {
                    ReplayEvent::PlacementNeeded(data) => Some(data),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                placement_events.len(),
                1,
                "each drain exposes exactly one controller placement boundary"
            );
            let placement = placement_events[0];
            assert_eq!(placement.logical_request_id, format!("fresh-{ordinal}"));
            let loads = placement
                .candidates
                .iter()
                .map(|candidate| (candidate.target.worker_id, candidate.in_flight_requests))
                .collect::<Vec<_>>();
            observed_loads.push(loads);
            let selected = placement
                .candidates
                .iter()
                .filter(|candidate| candidate.eligible)
                .min_by_key(|candidate| (candidate.in_flight_requests, candidate.target.worker_id))
                .expect("two active candidates")
                .target
                .clone();
            let observed_candidates = placement.candidates.clone();
            chosen_workers.push(selected.worker_id);
            let logical_id = placement.logical_request_id.clone();
            replay.assign(&logical_id, selected)?;
            let next_events = replay.drain_events()?;
            let routed_candidates = next_events
                .iter()
                .find_map(|event| match event {
                    ReplayEvent::Routed(data) if data.logical_request_id == logical_id => {
                        Some(&data.candidates)
                    }
                    _ => None,
                })
                .expect("assignment must publish its routed observation");
            assert_eq!(
                routed_candidates, &observed_candidates,
                "Routed must reuse the exact causally frozen PlacementNeeded observation"
            );
            boundary_events = next_events;
        }

        assert_eq!(chosen_workers, vec![0, 1, 0]);
        assert_eq!(observed_loads[0], vec![(0, 0), (1, 0)]);
        assert_eq!(observed_loads[1], vec![(0, 1), (1, 0)]);
        assert_eq!(observed_loads[2], vec![(0, 1), (1, 1)]);

        replay.close_admission()?;
        let events = drive_to_drained(&mut replay)?;
        assert_eq!(terminal_count(&events), 3);
        replay.finalize()?;
        Ok(())
    }

    #[test]
    fn live_scheduler_and_final_accounting_reconcile() -> Result<()> {
        let mut args = replay_args(false);
        args.max_num_seqs = Some(1);
        let mut replay =
            OfflineReplaySession::new(&args, 1, TRACE_BLOCK_SIZE, ReplaySessionRouter::External)?;
        replay.submit_request(request("account-a", "account-a", 0, 8, &[10, 11], 8))?;
        replay.submit_request(request("account-b", "account-b", 0, 8, &[12, 13], 8))?;
        let mut events = drive_to_pending_placement(&mut replay)?;
        replay.assign("account-a", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        events.extend(replay.drain_events()?);
        replay.assign("account-b", WorkerTarget::new(DEFAULT_REPLAY_POOL_ID, 0, 0))?;
        replay.settle_current_time()?;
        events.extend(replay.drain_events()?);
        let snapshot = replay.snapshot()?;
        let worker = snapshot.workers.first().unwrap();
        assert_eq!(worker.running_requests, Some(1));
        assert_eq!(worker.queued_requests, Some(1));
        assert_eq!(worker.in_flight_requests, 2);
        assert!(worker.running_tokens.is_some_and(|tokens| tokens > 0));
        assert!(worker.queued_tokens.is_some_and(|tokens| tokens > 0));

        replay.close_admission()?;
        events.extend(drive_to_drained(&mut replay)?);
        let report = replay.finalize()?;
        assert_eq!(terminal_count(&events), 2);
        assert_eq!(report.request_counts.num_requests, 2);
        assert_eq!(report.request_counts.completed_requests, 2);
        assert_eq!(report.per_request.len(), 2);
        assert_eq!(
            report.request_counts.total_output_tokens,
            report
                .per_request
                .iter()
                .map(|record| record.output_length)
                .sum::<usize>()
        );
        assert!(
            report
                .per_request
                .iter()
                .all(|record| record.terminal_status == ReplayTerminalStatus::Completed)
        );
        assert!(
            (report.throughput.decode_worker_seconds - report.throughput.duration_ms / 1000.0)
                .abs()
                < 1e-9,
            "one static aggregated worker must accrue exactly the simulated duration"
        );
        assert_eq!(report.throughput.prefill_worker_seconds, 0.0);
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

    /// Drain eagerly captured events without cloning repeated owned metadata.
    ///
    /// This is a general adapter surface. Each item is a frozen snapshot and
    /// cannot observe later replay state; adapters must materialize independent
    /// public values before returning across a mutable foreign-language boundary.
    #[doc(hidden)]
    pub fn drain_captured_events(&mut self) -> Result<Vec<CapturedReplayEvent>> {
        Ok(self.runtime_mut()?.drain_captured_events())
    }

    pub fn pending_placements(&mut self) -> Result<Vec<ReplayPendingPlacement>> {
        Ok(self.runtime_mut()?.pending_placements())
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

    pub fn assign_pool(&mut self, logical_request_id: &str, pool_id: &str) -> Result<()> {
        let uuid = self
            .runtime()?
            .uuid_for_logical_id(logical_request_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "interactive replay has no request with logical ID {logical_request_id:?}"
                )
            })?;
        let determinism = self.determinism;
        with_replay_determinism(determinism, || {
            self.runtime_mut()?.assign_pool(uuid, pool_id)
        })
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
            if authored.prefix_reset {
                bail!(
                    "interactive replay request {} sets unsupported prefix_reset=true",
                    authored.request.logical_request_id
                );
            }
            self.validate_static_routing(&authored.request)?;
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
            if let Some(target) = authored.request.target.clone() {
                preassignments.push((uuid, target, authored.request.routing_constraints.clone()));
            }
            identities.push(InteractiveRequestIdentity {
                metadata: Arc::new(InteractiveRequestMetadata {
                    logical_request_id: authored.request.logical_request_id.clone(),
                    attempt_id: authored.request.attempt_id.clone(),
                    group_id: authored.request.group_id.clone(),
                    internal_uuid: uuid,
                    session_id: authored.request.session_id.clone(),
                    authored_turn_index: authored.request.authored_turn_index,
                    input_length: authored.request.input_length,
                    requested_output_length: authored.request.output_length,
                    priority: authored.request.priority,
                    strict_priority: authored.request.strict_priority,
                    policy_class: authored.request.policy_class.clone(),
                    routing_constraints: authored.request.routing_constraints.clone(),
                }),
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
                routing_constraints: request.routing_constraints.into_router_constraints(),
                wait_for: authored.wait_for,
                prefix_reset: authored.prefix_reset,
            });
        }

        let mut registered_uuids = Vec::with_capacity(identities.len());
        for identity in identities {
            let uuid = identity.metadata.internal_uuid;
            if let Err(error) = self.runtime_mut()?.register_identity(identity) {
                for uuid in &registered_uuids {
                    self.runtime_mut()?.unregister_identity(*uuid);
                }
                return Err(error);
            }
            registered_uuids.push(uuid);
        }
        for (uuid, target, constraints) in &preassignments {
            if let Err(error) = self
                .runtime_mut()?
                .preassign(*uuid, target.clone(), constraints)
            {
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
