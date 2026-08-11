// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use anyhow::{Context, Result, anyhow, bail};
use dynamo_kv_router::protocols::RoutingConstraints;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::trace::validate_synthesizable_prompt;
use super::types::{
    AgenticTrace, CascadedWorkloadTerminal, ReadyReplayTurn, ReadyTurn, ReplayRequestHashes,
    ReplayRequestPayload, Trace, WorkloadTerminalStatus,
};
use super::{SYNTHETIC_OUTPUT_SEED, planned_output_token_ids};
use crate::common::protocols::DirectRequest;

#[derive(Debug)]
enum SchedulingPolicy {
    Trace,
    Concurrency(ConcurrencyState),
    Agentic(Box<AgenticState>),
}

#[derive(Debug)]
struct ConcurrencyState {
    max_active_sessions: usize,
    next_pending_session: usize,
    active_sessions: usize,
}

#[derive(Debug)]
struct AgenticState {
    remaining_dependencies: Vec<usize>,
    ready_after_ms: Vec<f64>,
    dependents: FxHashMap<String, Vec<usize>>,
    request_identities: FxHashMap<String, AuthoredIdentity>,
    request_id_by_session_turn: FxHashMap<(String, usize), String>,
    request_id_by_internal_uuid: FxHashMap<Uuid, String>,
    output_rng: StdRng,
    open: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AuthoredIdentity {
    session_id: String,
    authored_turn_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptMode {
    Full,
    DeltaCumulative,
}

#[derive(Debug)]
struct TurnResolution {
    logical_request_id: Option<String>,
    session_ended: bool,
    status: WorkloadTerminalStatus,
}

#[derive(Debug)]
struct SessionRuntime {
    session_id: String,
    turns: Vec<TurnRuntime>,
    cumulative_tokens: Vec<u32>,
    next_turn_index: usize,
    next_ready_at_ms: Option<f64>,
    in_flight: Option<Uuid>,
}

#[derive(Debug)]
enum PromptTokens {
    // Full-prompt traces stay in their compact on-disk representation until
    // dispatch. Delta-cumulative traces remain eager because later turns append
    // generated output to already-materialized session history.
    Deferred {
        input_length: usize,
        hash_ids: Vec<u32>,
    },
    Materialized(Vec<u32>),
}

fn validate_exact_trace_hash_count(
    input_length: usize,
    hash_ids: &[u32],
    trace_block_size: usize,
) -> Result<()> {
    validate_synthesizable_prompt(input_length, hash_ids, trace_block_size)?;
    let required_hash_ids = input_length.div_ceil(trace_block_size);
    if hash_ids.len() != required_hash_ids {
        bail!(
            "input_length {input_length} requires exactly {required_hash_ids} hash IDs at trace_block_size {trace_block_size}, got {}",
            hash_ids.len()
        );
    }
    Ok(())
}

impl PromptTokens {
    fn deferred(input_length: usize, hash_ids: Vec<u32>, trace_block_size: usize) -> Result<Self> {
        validate_exact_trace_hash_count(input_length, &hash_ids, trace_block_size)?;
        Ok(Self::Deferred {
            input_length,
            hash_ids,
        })
    }

    fn input_length(&self) -> usize {
        match self {
            Self::Deferred { input_length, .. } => *input_length,
            Self::Materialized(tokens) => tokens.len(),
        }
    }

    fn take_deferred(&mut self) -> (usize, Vec<u32>) {
        match self {
            Self::Deferred {
                input_length,
                hash_ids,
            } => (*input_length, std::mem::take(hash_ids)),
            Self::Materialized(_) => {
                unreachable!("full-prompt turns must retain their deferred representation")
            }
        }
    }

    fn materialized(&self) -> &[u32] {
        match self {
            Self::Deferred { .. } => {
                unreachable!("delta-cumulative prompts are materialized during driver setup")
            }
            Self::Materialized(tokens) => tokens,
        }
    }
}

#[derive(Debug)]
struct TurnRuntime {
    logical_request_id: Option<String>,
    authored_turn_index: Option<usize>,
    replay_key: Option<String>,
    prompt_tokens: PromptTokens,
    max_output_tokens: usize,
    output_token_ids: Option<Vec<u32>>,
    delay_after_previous_ms: f64,
    priority: i32,
    strict_priority: u32,
    policy_class: Option<String>,
    routing_constraints: RoutingConstraints,
    internal_uuid: Option<Uuid>,
    /// Reporting-only source metadata. Prompt tokens/hash ids, not this flag,
    /// determine replay KV identity.
    prefix_reset: bool,
}

#[derive(Debug)]
struct StagedAgenticSession {
    session: SessionRuntime,
    logical_request_id: String,
    identity: AuthoredIdentity,
    dependencies: Vec<String>,
    internal_uuid: Option<Uuid>,
    root_ready_at_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct InFlightTurn {
    session_index: usize,
    turn_index: usize,
    emitted_output_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct ReadySession {
    ready_at_ms: f64,
    session_index: usize,
    turn_index: usize,
}

impl PartialEq for ReadySession {
    fn eq(&self, other: &Self) -> bool {
        self.ready_at_ms.to_bits() == other.ready_at_ms.to_bits()
            && self.session_index == other.session_index
            && self.turn_index == other.turn_index
    }
}

impl Eq for ReadySession {}

impl Ord for ReadySession {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .ready_at_ms
            .total_cmp(&self.ready_at_ms)
            .then_with(|| other.session_index.cmp(&self.session_index))
            .then_with(|| other.turn_index.cmp(&self.turn_index))
    }
}

impl PartialOrd for ReadySession {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SchedulingPolicy {
    fn schedules_sequential_turns(&self) -> bool {
        !matches!(self, Self::Agentic(_))
    }

    fn arrival_timestamp_ms(&self, scheduled_ready_at_ms: f64) -> Option<f64> {
        match self {
            Self::Concurrency(_) => None,
            Self::Trace | Self::Agentic(_) => Some(scheduled_ready_at_ms),
        }
    }

    fn dispatch_limit(&self, requested: usize, in_flight: usize) -> usize {
        match self {
            Self::Concurrency(state) => {
                requested.min(state.max_active_sessions.saturating_sub(in_flight))
            }
            Self::Trace | Self::Agentic(_) => requested,
        }
    }

    fn at_dispatch_capacity(&self, in_flight: usize) -> bool {
        matches!(
            self,
            Self::Concurrency(state) if in_flight >= state.max_active_sessions
        )
    }
}

impl ConcurrencyState {
    fn new(max_active_sessions: usize) -> Self {
        Self {
            max_active_sessions,
            next_pending_session: 0,
            active_sessions: 0,
        }
    }

    fn activate_pending(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        now_ms: f64,
    ) {
        while self.active_sessions < self.max_active_sessions
            && self.next_pending_session < sessions.len()
        {
            let session_index = self.next_pending_session;
            self.next_pending_session += 1;
            let session = &mut sessions[session_index];
            let turn_index = session.next_turn_index;
            session.next_ready_at_ms = Some(now_ms);
            ready_sessions.push(ReadySession {
                ready_at_ms: now_ms,
                session_index,
                turn_index,
            });
            self.active_sessions += 1;
        }
    }

    fn on_session_finished(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        now_ms: f64,
    ) {
        self.active_sessions = self.active_sessions.saturating_sub(1);
        self.activate_pending(sessions, ready_sessions, now_ms);
    }
}

impl AgenticState {
    fn new(open: bool) -> Self {
        Self {
            remaining_dependencies: Vec::new(),
            ready_after_ms: Vec::new(),
            dependents: FxHashMap::default(),
            request_identities: FxHashMap::default(),
            request_id_by_session_turn: FxHashMap::default(),
            request_id_by_internal_uuid: FxHashMap::default(),
            output_rng: StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED),
            open,
        }
    }

    fn release_dependents(
        &mut self,
        sessions: &mut [SessionRuntime],
        ready_sessions: &mut BinaryHeap<ReadySession>,
        request_id: &str,
        now_ms: f64,
    ) {
        let Some(dependent_sessions) = self.dependents.get(request_id).cloned() else {
            return;
        };
        for session_index in dependent_sessions {
            let Some(remaining) = self.remaining_dependencies.get_mut(session_index) else {
                continue;
            };
            if *remaining == 0 {
                continue;
            }
            *remaining -= 1;
            if let Some(ready_after_ms) = self.ready_after_ms.get_mut(session_index) {
                *ready_after_ms = ready_after_ms.max(now_ms);
            }
            if *remaining != 0 {
                continue;
            }

            let Some(session) = sessions.get_mut(session_index) else {
                continue;
            };
            if session.in_flight.is_some()
                || session.next_turn_index >= session.turns.len()
                || session.next_ready_at_ms.is_some()
            {
                continue;
            }
            let turn_index = session.next_turn_index;
            let ready_at_ms = self.ready_after_ms[session_index]
                + session.turns[turn_index].delay_after_previous_ms;
            session.next_ready_at_ms = Some(ready_at_ms);
            ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index,
                turn_index,
            });
        }
    }

    fn cancel_descendants(
        &mut self,
        sessions: &mut [SessionRuntime],
        request_id: &str,
    ) -> Result<Vec<CascadedWorkloadTerminal>> {
        let mut pending = VecDeque::new();
        if let Some(dependent_sessions) = self.dependents.get(request_id) {
            pending.extend(dependent_sessions.iter().copied());
        }

        let mut terminals = Vec::new();
        while let Some(session_index) = pending.pop_front() {
            let session = sessions
                .get_mut(session_index)
                .with_context(|| format!("unknown dependent workload session {session_index}"))?;
            if session.next_turn_index >= session.turns.len() {
                continue;
            }
            if let Some(in_flight) = session.in_flight {
                bail!(
                    "dependency-blocked session {} unexpectedly has in-flight request {in_flight}",
                    session.session_id
                );
            }

            let turn_index = session.next_turn_index;
            let turn = session
                .turns
                .get_mut(turn_index)
                .context("dependent workload turn disappeared during cancellation")?;
            let logical_request_id = turn.logical_request_id.clone().ok_or_else(|| {
                anyhow!(
                    "agentic dependent session {} turn {turn_index} has no logical request ID",
                    session.session_id
                )
            })?;
            let authored_turn_index = turn.authored_turn_index.ok_or_else(|| {
                anyhow!(
                    "agentic dependent request {logical_request_id} has no authored turn index"
                )
            })?;
            let request_uuid = if let Some(request_uuid) = turn.internal_uuid {
                request_uuid
            } else {
                let request_uuid = loop {
                    let candidate = Uuid::new_v4();
                    if !self.request_id_by_internal_uuid.contains_key(&candidate) {
                        break candidate;
                    }
                };
                turn.internal_uuid = Some(request_uuid);
                self.request_id_by_internal_uuid
                    .insert(request_uuid, logical_request_id.clone());
                request_uuid
            };
            let terminal = CascadedWorkloadTerminal {
                request_uuid,
                logical_request_id: logical_request_id.clone(),
                session_id: session.session_id.clone(),
                authored_turn_index,
                input_length: turn.prompt_tokens.input_length(),
                requested_output_length: turn.max_output_tokens,
                emitted_output_count: 0,
                status: WorkloadTerminalStatus::Canceled,
            };

            session.next_ready_at_ms = None;
            session.next_turn_index = session.turns.len();
            terminals.push(terminal);

            if let Some(dependent_sessions) = self.dependents.get(&logical_request_id) {
                pending.extend(dependent_sessions.iter().copied());
            }
        }
        Ok(terminals)
    }
}

#[derive(Debug)]
pub struct WorkloadDriver {
    policy: SchedulingPolicy,
    prompt_mode: PromptMode,
    emit_session_metadata: bool,
    trace_block_size: usize,
    engine_block_size: u32,
    include_replay_hashes: bool,
    sessions: Vec<SessionRuntime>,
    in_flight: FxHashMap<Uuid, InFlightTurn>,
    ready_sessions: BinaryHeap<ReadySession>,
}

impl WorkloadDriver {
    pub(crate) fn new_trace(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Trace,
            PromptMode::Full,
            true,
        )
    }

    pub(crate) fn new_trace_without_replay_hashes(
        trace: Trace,
        engine_block_size: usize,
        accumulate_session_deltas: bool,
    ) -> Result<Self> {
        trace.validate_for_trace_mode()?;
        let prompt_mode = if accumulate_session_deltas {
            PromptMode::DeltaCumulative
        } else {
            PromptMode::Full
        };
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Trace,
            prompt_mode,
            false,
        )
    }

    pub(crate) fn new_trace_accumulating_deltas(
        trace: Trace,
        engine_block_size: usize,
    ) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Trace,
            PromptMode::DeltaCumulative,
            true,
        )
    }

    /// Build a closed-loop concurrency driver. `max_in_flight` is the *session* cap
    /// (depth-first): a session holds its slot across all turns + think-time, and new
    /// sessions are admitted only while fewer than `max_in_flight` are active.
    pub(crate) fn new_concurrency(
        trace: Trace,
        engine_block_size: usize,
        max_in_flight: usize,
    ) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Concurrency(ConcurrencyState::new(max_in_flight)),
            PromptMode::Full,
            true,
        )
    }

    pub(crate) fn new_concurrency_without_replay_hashes(
        trace: Trace,
        engine_block_size: usize,
        max_in_flight: usize,
        accumulate_session_deltas: bool,
    ) -> Result<Self> {
        trace.validate_for_concurrency_mode()?;
        let prompt_mode = if accumulate_session_deltas {
            PromptMode::DeltaCumulative
        } else {
            PromptMode::Full
        };
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Concurrency(ConcurrencyState::new(max_in_flight)),
            prompt_mode,
            false,
        )
    }

    pub(crate) fn new_concurrency_accumulating_deltas(
        trace: Trace,
        engine_block_size: usize,
        max_in_flight: usize,
    ) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            SchedulingPolicy::Concurrency(ConcurrencyState::new(max_in_flight)),
            PromptMode::DeltaCumulative,
            true,
        )
    }

    pub(crate) fn new_agentic_trace(trace: AgenticTrace, engine_block_size: usize) -> Result<Self> {
        Self::new_agentic_trace_with_replay_hashes(trace, engine_block_size, true)
    }

    pub(crate) fn new_agentic_trace_without_replay_hashes(
        trace: AgenticTrace,
        engine_block_size: usize,
    ) -> Result<Self> {
        Self::new_agentic_trace_with_replay_hashes(trace, engine_block_size, false)
    }

    fn new_agentic_trace_with_replay_hashes(
        trace: AgenticTrace,
        engine_block_size: usize,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        let trace_block_size = trace.block_size;
        let mut driver = Self::new_open_agentic_with_replay_hashes(
            trace_block_size,
            engine_block_size,
            include_replay_hashes,
        )?;
        driver.append_agentic_trace(trace, 0.0)?;
        driver.close()?;
        Ok(driver)
    }

    /// Create an agentic driver whose admission remains open while idle.
    /// Appended traces must already share this driver's globally normalized
    /// hash-id namespace and trace block size.
    pub(crate) fn new_open_agentic(
        trace_block_size: usize,
        engine_block_size: usize,
    ) -> Result<Self> {
        Self::new_open_agentic_with_replay_hashes(trace_block_size, engine_block_size, true)
    }

    pub(crate) fn new_open_agentic_without_replay_hashes(
        trace_block_size: usize,
        engine_block_size: usize,
    ) -> Result<Self> {
        Self::new_open_agentic_with_replay_hashes(trace_block_size, engine_block_size, false)
    }

    fn new_open_agentic_with_replay_hashes(
        trace_block_size: usize,
        engine_block_size: usize,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        if trace_block_size == 0 {
            bail!("trace_block_size must be greater than 0");
        }
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        Ok(Self {
            policy: SchedulingPolicy::Agentic(Box::new(AgenticState::new(true))),
            prompt_mode: PromptMode::Full,
            emit_session_metadata: true,
            trace_block_size,
            engine_block_size: engine_block_size_u32,
            include_replay_hashes,
            sessions: Vec::new(),
            in_flight: FxHashMap::default(),
            ready_sessions: BinaryHeap::new(),
        })
    }

    /// Atomically append one independent authored DAG. Dependencies are local
    /// to the appended graph; authored request IDs, session/turn identities,
    /// and supplied internal UUIDs are unique across the whole driver.
    pub(crate) fn append_agentic_trace(
        &mut self,
        trace: AgenticTrace,
        release_at_ms: f64,
    ) -> Result<()> {
        if !release_at_ms.is_finite() || release_at_ms < 0.0 {
            bail!("release_at_ms must be finite and non-negative, got {release_at_ms}");
        }
        if trace.block_size != self.trace_block_size {
            bail!(
                "appended trace block_size {} does not match open driver block_size {}",
                trace.block_size,
                self.trace_block_size
            );
        }
        if let Some(turn) = trace.turns.iter().find(|turn| turn.prefix_reset) {
            bail!(
                "request {} sets unsupported prefix_reset=true; causal replay does not mutate KV state from trace metadata",
                turn.request_id
            );
        }
        trace.validate()?;

        let SchedulingPolicy::Agentic(state) = &self.policy else {
            bail!("append_agentic_trace requires an agentic workload driver");
        };
        if !state.open {
            bail!("cannot append an agentic workflow after admission is closed");
        }
        for turn in &trace.turns {
            if let Some(existing) = state.request_identities.get(&turn.request_id) {
                bail!(
                    "request_id {} is already registered as session {} authored turn {}",
                    turn.request_id,
                    existing.session_id,
                    existing.authored_turn_index
                );
            }
            let session_turn = (turn.session_id.clone(), turn.authored_turn_index);
            if let Some(existing) = state.request_id_by_session_turn.get(&session_turn) {
                bail!(
                    "request {} conflicts with existing request {existing} on session {} authored turn {}",
                    turn.request_id,
                    turn.session_id,
                    turn.authored_turn_index
                );
            }
            if let Some(internal_uuid) = turn.internal_uuid
                && let Some(existing) = state.request_id_by_internal_uuid.get(&internal_uuid)
            {
                bail!(
                    "request {} duplicates internal UUID {internal_uuid} already used by {existing}",
                    turn.request_id
                );
            }
        }

        let first_session_index = self.sessions.len();
        let mut output_rng = state.output_rng.clone();
        let mut staged_internal_uuids: FxHashMap<Uuid, String> = FxHashMap::default();
        let mut staged = Vec::with_capacity(trace.turns.len());
        for mut turn in trace.turns {
            let internal_uuid = turn.internal_uuid.or({
                #[cfg(feature = "replay-bench")]
                {
                    let session_index = first_session_index
                        .checked_add(staged.len())
                        .context("agentic session index overflow")?;
                    crate::replay::canonical_replay_active()
                        .then(|| Uuid::from_u128(session_index as u128 + 1))
                }
                #[cfg(not(feature = "replay-bench"))]
                {
                    None
                }
            });
            if let Some(internal_uuid) = internal_uuid {
                if let Some(existing) = state.request_id_by_internal_uuid.get(&internal_uuid) {
                    bail!(
                        "request {} maps to internal UUID {internal_uuid} already used by {existing}",
                        turn.request_id
                    );
                }
                if let Some(existing) =
                    staged_internal_uuids.insert(internal_uuid, turn.request_id.clone())
                {
                    bail!(
                        "requests {existing} and {} map to duplicate internal UUID {internal_uuid}",
                        turn.request_id
                    );
                }
            }

            let prompt_tokens = PromptTokens::deferred(
                turn.input_length,
                std::mem::take(&mut turn.hash_ids),
                self.trace_block_size,
            )?;
            let output_token_ids = Some(planned_output_token_ids(
                turn.output_token_ids,
                turn.max_output_tokens,
                &mut output_rng,
            ));
            let root_ready_at_ms = if turn.wait_for.is_empty() {
                let root_offset_ms = turn.first_ready_timestamp_ms.unwrap_or(0.0);
                let ready_at_ms = release_at_ms + root_offset_ms;
                if !ready_at_ms.is_finite() {
                    bail!(
                        "request {} root readiness overflows virtual time",
                        turn.request_id
                    );
                }
                Some(ready_at_ms)
            } else {
                None
            };
            let logical_request_id = turn.request_id;
            let identity = AuthoredIdentity {
                session_id: turn.session_id.clone(),
                authored_turn_index: turn.authored_turn_index,
            };
            staged.push(StagedAgenticSession {
                session: SessionRuntime {
                    session_id: turn.session_id,
                    turns: vec![TurnRuntime {
                        logical_request_id: Some(logical_request_id.clone()),
                        authored_turn_index: Some(turn.authored_turn_index),
                        replay_key: turn.replay_key,
                        prompt_tokens,
                        max_output_tokens: turn.max_output_tokens,
                        output_token_ids,
                        delay_after_previous_ms: turn.delay_after_dependencies_ms,
                        priority: turn.priority,
                        strict_priority: turn.strict_priority,
                        policy_class: turn.policy_class,
                        routing_constraints: turn.routing_constraints,
                        internal_uuid,
                        prefix_reset: turn.prefix_reset,
                    }],
                    cumulative_tokens: Vec::new(),
                    next_turn_index: 0,
                    next_ready_at_ms: root_ready_at_ms,
                    in_flight: None,
                },
                logical_request_id,
                identity,
                dependencies: turn.wait_for,
                internal_uuid,
                root_ready_at_ms,
            });
        }

        let SchedulingPolicy::Agentic(state) = &mut self.policy else {
            unreachable!("agentic policy was validated before staging")
        };
        state.output_rng = output_rng;
        for (offset, staged_session) in staged.into_iter().enumerate() {
            let session_index = first_session_index + offset;
            state
                .remaining_dependencies
                .push(staged_session.dependencies.len());
            state.ready_after_ms.push(0.0);
            for dependency in &staged_session.dependencies {
                state
                    .dependents
                    .entry(dependency.clone())
                    .or_default()
                    .push(session_index);
            }
            state.request_identities.insert(
                staged_session.logical_request_id.clone(),
                staged_session.identity.clone(),
            );
            state.request_id_by_session_turn.insert(
                (
                    staged_session.identity.session_id.clone(),
                    staged_session.identity.authored_turn_index,
                ),
                staged_session.logical_request_id.clone(),
            );
            if let Some(internal_uuid) = staged_session.internal_uuid {
                state
                    .request_id_by_internal_uuid
                    .insert(internal_uuid, staged_session.logical_request_id.clone());
            }
            if let Some(ready_at_ms) = staged_session.root_ready_at_ms {
                self.ready_sessions.push(ReadySession {
                    ready_at_ms,
                    session_index,
                    turn_index: 0,
                });
            }
            self.sessions.push(staged_session.session);
        }
        Ok(())
    }

    /// Close dynamic admission. Existing and dependency-blocked work remains
    /// live; an empty closed driver becomes drained.
    pub(crate) fn close(&mut self) -> Result<()> {
        let SchedulingPolicy::Agentic(state) = &mut self.policy else {
            bail!("close requires an agentic workload driver");
        };
        state.open = false;
        Ok(())
    }

    fn new(
        trace: Trace,
        engine_block_size: usize,
        policy: SchedulingPolicy,
        prompt_mode: PromptMode,
        include_replay_hashes: bool,
    ) -> Result<Self> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let trace_block_size = trace.block_size;
        let is_concurrency = matches!(&policy, SchedulingPolicy::Concurrency(_));
        let mut output_rng = StdRng::seed_from_u64(SYNTHETIC_OUTPUT_SEED);
        #[cfg(feature = "replay-bench")]
        let mut next_deterministic_request_id =
            crate::replay::canonical_replay_active().then_some(1_u128);
        let sessions: Vec<SessionRuntime> = trace
            .sessions
            .into_iter()
            .map(|session| -> Result<SessionRuntime> {
                let next_ready_at_ms = if is_concurrency {
                    None
                } else {
                    Some(session.first_arrival_timestamp_ms.unwrap_or(0.0))
                };
                let turns = session
                    .turns
                    .into_iter()
                    .map(|mut turn| -> Result<TurnRuntime> {
                        let prompt_tokens = match prompt_mode {
                            PromptMode::Full => PromptTokens::deferred(
                                turn.input_length,
                                std::mem::take(&mut turn.hash_ids),
                                trace_block_size,
                            )?,
                            PromptMode::DeltaCumulative => {
                                validate_exact_trace_hash_count(
                                    turn.input_length,
                                    &turn.hash_ids,
                                    trace_block_size,
                                )?;
                                PromptTokens::Materialized(
                                    turn.synthesize_tokens(trace_block_size)?,
                                )
                            }
                        };
                        let output_token_ids = Some(planned_output_token_ids(
                            turn.output_token_ids,
                            turn.max_output_tokens,
                            &mut output_rng,
                        ));
                        #[cfg(feature = "replay-bench")]
                        let internal_uuid = {
                            next_deterministic_request_id.map(|next_id| {
                                let request_id = Uuid::from_u128(next_id);
                                next_deterministic_request_id = Some(
                                    next_id
                                        .checked_add(1)
                                        .expect("deterministic replay request UUID overflow"),
                                );
                                request_id
                            })
                        };
                        #[cfg(not(feature = "replay-bench"))]
                        let internal_uuid = None;
                        Ok(TurnRuntime {
                            logical_request_id: None,
                            authored_turn_index: None,
                            prompt_tokens,
                            replay_key: turn.replay_key,
                            max_output_tokens: turn.max_output_tokens,
                            output_token_ids,
                            delay_after_previous_ms: turn.delay_after_previous_ms,
                            priority: turn.priority,
                            strict_priority: turn.strict_priority,
                            policy_class: turn.policy_class,
                            routing_constraints: turn.routing_constraints,
                            internal_uuid,
                            prefix_reset: false,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let cumulative_capacity = if prompt_mode == PromptMode::DeltaCumulative {
                    turns
                        .iter()
                        .map(|turn| {
                            turn.prompt_tokens.input_length()
                                + turn
                                    .output_token_ids
                                    .as_ref()
                                    .map_or(0, |output| output.len())
                        })
                        .sum()
                } else {
                    0
                };
                Ok(SessionRuntime {
                    session_id: session.session_id,
                    turns,
                    cumulative_tokens: Vec::with_capacity(cumulative_capacity),
                    next_turn_index: 0,
                    next_ready_at_ms,
                    in_flight: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let ready_sessions = sessions
            .iter()
            .enumerate()
            .filter_map(|(session_index, session)| {
                Some(ReadySession {
                    ready_at_ms: session.next_ready_at_ms?,
                    session_index,
                    turn_index: session.next_turn_index,
                })
            })
            .collect();

        let mut driver = Self {
            policy,
            prompt_mode,
            emit_session_metadata: true,
            trace_block_size,
            engine_block_size: engine_block_size_u32,
            include_replay_hashes,
            sessions,
            in_flight: FxHashMap::default(),
            ready_sessions,
        };
        if let SchedulingPolicy::Concurrency(state) = &mut driver.policy {
            state.activate_pending(&mut driver.sessions, &mut driver.ready_sessions, 0.0);
        }
        Ok(driver)
    }

    /// Use stable monotonically increasing UUIDs for replay parity fixtures.
    /// This is unavailable in production builds so normal request identity and
    /// randomness remain unchanged.
    #[cfg(any(test, feature = "replay-bench"))]
    pub fn with_deterministic_request_ids(mut self, first_id: u128) -> Self {
        let mut next_id = first_id;
        for session in &mut self.sessions {
            for turn in &mut session.turns {
                turn.internal_uuid = Some(Uuid::from_u128(next_id));
                next_id = next_id
                    .checked_add(1)
                    .expect("deterministic replay request UUID overflow");
            }
        }
        if let SchedulingPolicy::Agentic(state) = &mut self.policy {
            state.request_id_by_internal_uuid.clear();
            for session in &self.sessions {
                for turn in &session.turns {
                    if let (Some(internal_uuid), Some(logical_request_id)) =
                        (turn.internal_uuid, turn.logical_request_id.as_ref())
                    {
                        state
                            .request_id_by_internal_uuid
                            .insert(internal_uuid, logical_request_id.clone());
                    }
                }
            }
        }
        self
    }

    fn request_uuid(&mut self, session_index: usize, turn_index: usize) -> Uuid {
        if let Some(request_id) = self.sessions[session_index].turns[turn_index].internal_uuid {
            return request_id;
        }

        let request_id = loop {
            let candidate = Uuid::new_v4();
            let registered = matches!(
                &self.policy,
                SchedulingPolicy::Agentic(state)
                    if state.request_id_by_internal_uuid.contains_key(&candidate)
            );
            if !registered && !self.in_flight.contains_key(&candidate) {
                break candidate;
            }
        };
        let logical_request_id = self.sessions[session_index].turns[turn_index]
            .logical_request_id
            .clone();
        self.sessions[session_index].turns[turn_index].internal_uuid = Some(request_id);
        if let (SchedulingPolicy::Agentic(state), Some(logical_request_id)) =
            (&mut self.policy, logical_request_id)
        {
            state
                .request_id_by_internal_uuid
                .insert(request_id, logical_request_id);
        }
        request_id
    }

    pub(crate) fn without_session_metadata(mut self) -> Self {
        self.emit_session_metadata = false;
        self
    }

    /// Failure-path companion: release a cap slot and terminate the owning session.
    /// No-op if `on_complete` already ran. Used when a request task is cancelled
    /// or panics before reaching `on_complete`.
    ///
    /// Terminating the session (marking it exhausted) prevents `run_workload` from
    /// deadlocking: `pop_ready` skips sessions with `in_flight.is_some()`, so a
    /// leaked session would leave `is_drained` stuck at `false` forever.
    pub fn release_cap_slot(&mut self, request_uuid: Uuid, now_ms: f64) {
        // This legacy drop-guard helper cannot surface dependency-cascade
        // evidence. Agentic runtimes must call `on_terminal(Canceled)` and
        // consume the returned descriptors.
        let _ = self.on_terminal(request_uuid, now_ms, WorkloadTerminalStatus::Canceled);
    }

    pub fn pop_ready(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyTurn> {
        self.pop_ready_replay(now_ms, limit)
            .into_iter()
            .map(ReadyReplayTurn::into_ready_turn)
            .collect()
    }

    /// Pop ready turns without materializing compact Mooncake prompts.
    pub fn pop_ready_replay(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyReplayTurn> {
        let effective_limit = self.policy.dispatch_limit(limit, self.in_flight.len());
        if effective_limit == 0 {
            return Vec::new();
        }

        let mut emitted = Vec::new();
        while emitted.len() < effective_limit {
            let Some(ready_session) = self.ready_sessions.pop() else {
                break;
            };
            if ready_session.ready_at_ms > now_ms {
                self.ready_sessions.push(ready_session);
                break;
            }

            let session_index = ready_session.session_index;
            let Some((turn_index, scheduled_ready_at_ms)) = self
                .sessions
                .get(session_index)
                .filter(|session| {
                    session.in_flight.is_none()
                        && session.next_turn_index == ready_session.turn_index
                        && session.next_ready_at_ms == Some(ready_session.ready_at_ms)
                })
                .map(|session| {
                    (
                        session.next_turn_index,
                        session
                            .next_ready_at_ms
                            .expect("ready session must have a timestamp"),
                    )
                })
            else {
                continue;
            };
            let request_uuid = self.request_uuid(session_index, turn_index);
            let is_agentic = matches!(&self.policy, SchedulingPolicy::Agentic(_));
            let session = &mut self.sessions[session_index];
            let turn = &mut session.turns[turn_index];
            let authored_turn_index = turn.authored_turn_index.unwrap_or(turn_index);
            let arrival_timestamp_ms = self.policy.arrival_timestamp_ms(scheduled_ready_at_ms);
            let (request, replay_hashes) = match self.prompt_mode {
                PromptMode::Full => {
                    let (input_length, hash_ids) = turn.prompt_tokens.take_deferred();
                    let request_metadata = DirectRequest {
                        tokens: Vec::new(),
                        max_output_tokens: turn.max_output_tokens,
                        output_token_ids: if is_agentic {
                            turn.output_token_ids.clone()
                        } else {
                            turn.output_token_ids.take()
                        },
                        uuid: Some(request_uuid),
                        dp_rank: 0,
                        arrival_timestamp_ms,
                        priority: turn.priority,
                        strict_priority: turn.strict_priority,
                        policy_class: turn.policy_class.clone(),
                    };
                    let request = ReplayRequestPayload::deferred(
                        request_metadata,
                        input_length,
                        hash_ids,
                        self.trace_block_size,
                        turn.routing_constraints.clone(),
                    );
                    // The router needs engine-block hashes at arrival, but it
                    // does not need to retain the expanded prompt. Materialize
                    // once transiently for hashing, then keep only the compact
                    // payload until a worker admission.
                    // TODO: Derive engine-block hashes directly from the compact
                    // trace blocks so immediate dispatch does not materialize
                    // the prompt once for routing and again for admission.
                    // Preserve `ReplayRequestHashes::from_tokens` semantics when
                    // trace and engine block sizes differ.
                    let replay_hashes = self.include_replay_hashes.then(|| {
                        let request_tokens = request.prompt_tokens();
                        ReplayRequestHashes::from_tokens(&request_tokens, self.engine_block_size)
                    });
                    (request, replay_hashes)
                }
                PromptMode::DeltaCumulative => {
                    session
                        .cumulative_tokens
                        .extend_from_slice(turn.prompt_tokens.materialized());
                    let request_tokens = session.cumulative_tokens.clone();
                    let replay_hashes = self.include_replay_hashes.then(|| {
                        ReplayRequestHashes::from_tokens(&request_tokens, self.engine_block_size)
                    });
                    let request = ReplayRequestPayload::materialized_with_constraints(
                        DirectRequest {
                            tokens: request_tokens,
                            max_output_tokens: turn.max_output_tokens,
                            output_token_ids: turn.output_token_ids.clone(),
                            uuid: Some(request_uuid),
                            dp_rank: 0,
                            arrival_timestamp_ms,
                            priority: turn.priority,
                            strict_priority: turn.strict_priority,
                            policy_class: turn.policy_class.clone(),
                        },
                        turn.routing_constraints.clone(),
                    );
                    (request, replay_hashes)
                }
            };
            session.in_flight = Some(request_uuid);
            session.next_ready_at_ms = None;
            self.in_flight.insert(
                request_uuid,
                InFlightTurn {
                    session_index,
                    turn_index,
                    emitted_output_tokens: 0,
                },
            );
            emitted.push(ReadyReplayTurn {
                request_uuid,
                logical_request_id: turn.logical_request_id.clone(),
                session_id: session.session_id.clone(),
                turn_index: authored_turn_index,
                authored_turn_index,
                replay_key: turn.replay_key.clone(),
                scheduled_ready_at_ms,
                replay_hashes,
                prefix_reset: turn.prefix_reset,
                emit_session_metadata: self.emit_session_metadata,
                request,
            });
        }
        emitted
    }

    pub(crate) fn pop_ready_compact(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyReplayTurn> {
        self.pop_ready_replay(now_ms, limit)
    }

    pub fn on_output_token(&mut self, request_uuid: Uuid, token_id: u32) -> Result<()> {
        if self.prompt_mode == PromptMode::Full
            && !matches!(&self.policy, SchedulingPolicy::Agentic(_))
        {
            return Ok(());
        }
        let in_flight = self
            .in_flight
            .get(&request_uuid)
            .copied()
            .ok_or_else(|| anyhow!("unknown workload request output for {request_uuid}"))?;

        let turn = &self.sessions[in_flight.session_index].turns[in_flight.turn_index];
        let planned_output_tokens = turn
            .output_token_ids
            .as_ref()
            .expect("delta and agentic turns must have planned output tokens");
        let expected_token = planned_output_tokens
            .get(in_flight.emitted_output_tokens)
            .ok_or_else(|| {
                anyhow!(
                    "workload request {request_uuid} emitted more than {} planned output tokens",
                    planned_output_tokens.len()
                )
            })?;
        if token_id != *expected_token {
            bail!(
                "workload request {request_uuid} emitted token {token_id} at position {}, expected {}",
                in_flight.emitted_output_tokens,
                expected_token
            );
        }

        let in_flight = self
            .in_flight
            .get_mut(&request_uuid)
            .expect("validated in-flight request must still exist");
        in_flight.emitted_output_tokens = in_flight
            .emitted_output_tokens
            .checked_add(1)
            .context("workload emitted output token count overflow")?;
        Ok(())
    }

    pub fn on_complete(&mut self, request_uuid: Uuid, now_ms: f64) -> Result<()> {
        let cascaded = self.on_terminal(
            request_uuid,
            now_ms,
            WorkloadTerminalStatus::Completed,
        )?;
        debug_assert!(cascaded.is_empty());
        Ok(())
    }

    /// Resolve one engine terminal and apply workload-level dependency rules.
    ///
    /// Completed requests alone release agentic dependencies. Any other
    /// terminal status ends the owning workflow branch and returns every
    /// recursively canceled descendant for the replay runtime to record and
    /// emit exactly once.
    pub fn on_terminal(
        &mut self,
        request_uuid: Uuid,
        now_ms: f64,
        status: WorkloadTerminalStatus,
    ) -> Result<Vec<CascadedWorkloadTerminal>> {
        let Some(resolution) = self.resolve_turn(request_uuid, now_ms, status)? else {
            return Ok(Vec::new());
        };
        self.apply_resolution(resolution, now_ms)
    }

    fn resolve_turn(
        &mut self,
        request_uuid: Uuid,
        now_ms: f64,
        status: WorkloadTerminalStatus,
    ) -> Result<Option<TurnResolution>> {
        let Some(in_flight) = self.in_flight.get(&request_uuid).copied() else {
            return match status {
                WorkloadTerminalStatus::Canceled => Ok(None),
                WorkloadTerminalStatus::Completed
                | WorkloadTerminalStatus::Rejected
                | WorkloadTerminalStatus::Failed => Err(anyhow!(
                    "unknown workload request terminal for {request_uuid}"
                )),
            };
        };
        let session = self
            .sessions
            .get(in_flight.session_index)
            .ok_or_else(|| anyhow!("unknown workload session {}", in_flight.session_index))?;
        let turn = session.turns.get(in_flight.turn_index).ok_or_else(|| {
            anyhow!(
                "unknown workload turn {} for session {}",
                in_flight.turn_index,
                session.session_id
            )
        })?;
        if session.in_flight != Some(request_uuid) {
            bail!(
                "session {} resolution for {} does not match in-flight request {:?}",
                session.session_id,
                request_uuid,
                session.in_flight
            );
        }
        if session.next_turn_index != in_flight.turn_index {
            bail!(
                "session {} resolution for turn {} does not match next turn {}",
                session.session_id,
                in_flight.turn_index,
                session.next_turn_index
            );
        }

        let logical_request_id = turn.logical_request_id.clone();
        if status == WorkloadTerminalStatus::Rejected && in_flight.emitted_output_tokens != 0 {
            bail!(
                "rejected workload request {request_uuid} emitted {} output tokens",
                in_flight.emitted_output_tokens
            );
        }
        let completed_output_tokens = (status == WorkloadTerminalStatus::Completed
            && self.prompt_mode == PromptMode::DeltaCumulative)
            .then(|| {
                let planned_output_tokens = turn
                    .output_token_ids
                    .as_ref()
                    .expect("delta turns must have planned output tokens");
                planned_output_tokens[..in_flight.emitted_output_tokens].to_vec()
            });
        let (next_turn_index, next_ready_at_ms, session_ended) = match status {
            WorkloadTerminalStatus::Completed => {
                let next_turn_index = in_flight
                    .turn_index
                    .checked_add(1)
                    .context("workload turn index overflow")?;
                let has_more_turns = self.policy.schedules_sequential_turns()
                    && next_turn_index < session.turns.len();
                let next_ready_at_ms = has_more_turns
                    .then(|| now_ms + session.turns[next_turn_index].delay_after_previous_ms);
                (next_turn_index, next_ready_at_ms, !has_more_turns)
            }
            WorkloadTerminalStatus::Rejected
            | WorkloadTerminalStatus::Failed
            | WorkloadTerminalStatus::Canceled => (session.turns.len(), None, true),
        };

        self.in_flight
            .remove(&request_uuid)
            .expect("validated in-flight request must still exist");
        let session = &mut self.sessions[in_flight.session_index];
        session.in_flight = None;
        session.next_turn_index = next_turn_index;
        session.next_ready_at_ms = next_ready_at_ms;
        if next_ready_at_ms.is_some()
            && let Some(output_tokens) = completed_output_tokens
        {
            session.cumulative_tokens.extend(output_tokens);
        }
        if let Some(ready_at_ms) = next_ready_at_ms {
            self.ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index: in_flight.session_index,
                turn_index: next_turn_index,
            });
        }

        Ok(Some(TurnResolution {
            logical_request_id,
            session_ended,
            status,
        }))
    }

    fn apply_resolution(
        &mut self,
        resolution: TurnResolution,
        now_ms: f64,
    ) -> Result<Vec<CascadedWorkloadTerminal>> {
        match &mut self.policy {
            SchedulingPolicy::Trace => Ok(Vec::new()),
            SchedulingPolicy::Concurrency(state) => {
                if resolution.session_ended {
                    state.on_session_finished(&mut self.sessions, &mut self.ready_sessions, now_ms);
                }
                Ok(Vec::new())
            }
            SchedulingPolicy::Agentic(state) => {
                if let Some(request_id) = resolution.logical_request_id {
                    if resolution.status == WorkloadTerminalStatus::Completed {
                        state.release_dependents(
                            &mut self.sessions,
                            &mut self.ready_sessions,
                            &request_id,
                            now_ms,
                        );
                        Ok(Vec::new())
                    } else {
                        state.cancel_descendants(&mut self.sessions, &request_id)
                    }
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    pub fn next_ready_time_ms(&mut self) -> Option<f64> {
        if self.policy.at_dispatch_capacity(self.in_flight.len()) {
            return None;
        }
        loop {
            let ready_session = *self.ready_sessions.peek()?;
            let session = &self.sessions[ready_session.session_index];
            if session.in_flight.is_some()
                || session.next_turn_index != ready_session.turn_index
                || session.next_ready_at_ms != Some(ready_session.ready_at_ms)
            {
                self.ready_sessions.pop();
                continue;
            }
            return Some(ready_session.ready_at_ms);
        }
    }

    pub fn is_drained(&self) -> bool {
        let admission_closed = !matches!(
            &self.policy,
            SchedulingPolicy::Agentic(state) if state.open
        );
        admission_closed
            && self.in_flight.is_empty()
            && self
                .sessions
                .iter()
                .all(|session| session.next_turn_index >= session.turns.len())
    }

    pub(crate) fn is_open(&self) -> bool {
        matches!(
            &self.policy,
            SchedulingPolicy::Agentic(state) if state.open
        )
    }

    /// Number of submitted turns that have not reached a terminal outcome,
    /// including in-flight, dependency-blocked, and future-ready turns.
    pub(crate) fn pending_turns(&self) -> usize {
        self.sessions
            .iter()
            .map(|session| session.turns.len().saturating_sub(session.next_turn_index))
            .sum()
    }

    pub fn total_turns(&self) -> usize {
        self.sessions
            .iter()
            .map(|session| session.turns.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loadgen::{AgenticTrace, AgenticTurnTrace, SessionTrace, Trace, TurnTrace};

    fn assert_deterministic_output_plan(
        mut first_driver: WorkloadDriver,
        mut second_driver: WorkloadDriver,
        expected_len: usize,
    ) {
        let first = first_driver.pop_ready(0.0, usize::MAX);
        let second = second_driver.pop_ready(0.0, usize::MAX);

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(
            first[0].request.output_token_ids,
            second[0].request.output_token_ids
        );
        assert_eq!(
            first[0].request.output_token_ids.as_ref().map(Vec::len),
            Some(expected_len)
        );
    }

    #[test]
    fn hash_free_admission_preserves_request_without_router_metadata() {
        let trace = Trace {
            block_size: 2,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 4,
                    max_output_tokens: 1,
                    hash_ids: vec![10, 11],
                    ..Default::default()
                }],
            }],
        };
        let mut with_hashes = WorkloadDriver::new_trace(trace.clone(), 2).unwrap();
        let mut without_hashes =
            WorkloadDriver::new_trace_without_replay_hashes(trace, 2, false).unwrap();

        let with_hashes = with_hashes.pop_ready(0.0, 1).pop().unwrap();
        let without_hashes = without_hashes.pop_ready(0.0, 1).pop().unwrap();

        assert!(with_hashes.replay_hashes.is_some());
        assert!(without_hashes.replay_hashes.is_none());
        assert_eq!(without_hashes.request.tokens, with_hashes.request.tokens);
        assert_eq!(
            without_hashes.request.output_token_ids,
            with_hashes.request.output_token_ids
        );
    }

    fn two_session_trace() -> Trace {
        Trace {
            block_size: 1,
            sessions: vec![
                SessionTrace {
                    session_id: "a".into(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![
                        TurnTrace {
                            input_length: 2,
                            max_output_tokens: 1,
                            hash_ids: vec![1, 2],
                            delay_after_previous_ms: 0.0,
                            ..Default::default()
                        },
                        TurnTrace {
                            input_length: 2,
                            max_output_tokens: 1,
                            hash_ids: vec![3, 4],
                            delay_after_previous_ms: 5.0,
                            ..Default::default()
                        },
                    ],
                },
                SessionTrace {
                    session_id: "b".into(),
                    first_arrival_timestamp_ms: Some(0.0),
                    turns: vec![TurnTrace {
                        input_length: 2,
                        max_output_tokens: 1,
                        hash_ids: vec![5, 6],
                        delay_after_previous_ms: 0.0,
                        ..Default::default()
                    }],
                },
            ],
        }
    }

    /// A: 2 turns (turn-1 has a 5ms think-time). B, C: 1 turn each. Used for the cap>1
    /// transition / cancellation tests (w/ a third session pending behind a cap of 2).
    fn three_session_trace() -> Trace {
        let mut trace = two_session_trace();
        trace.sessions.push(SessionTrace {
            session_id: "c".into(),
            first_arrival_timestamp_ms: Some(0.0),
            turns: vec![TurnTrace {
                input_length: 2,
                max_output_tokens: 1,
                hash_ids: vec![7, 8],
                delay_after_previous_ms: 0.0,
                ..Default::default()
            }],
        });
        trace
    }

    #[test]
    fn full_prompts_remain_deferred_until_dispatch() {
        let mut driver = WorkloadDriver::new_trace(two_session_trace(), 1).unwrap();

        assert!(driver.sessions.iter().all(|session| {
            session
                .turns
                .iter()
                .all(|turn| matches!(turn.prompt_tokens, PromptTokens::Deferred { .. }))
        }));

        let ready = driver.pop_ready(0.0, 1);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].request.tokens, vec![1, 2]);
        assert!(ready[0].replay_hashes.is_some());
    }

    #[test]
    fn compact_dispatch_does_not_retain_materialized_prompt() {
        let mut driver = WorkloadDriver::new_trace(two_session_trace(), 1).unwrap();

        let mut ready = driver.pop_ready_compact(0.0, 1);

        assert_eq!(ready.len(), 1);
        let request = ready.pop().expect("one compact request").request;
        assert_eq!(request.input_length(), 2);
        assert!(request.metadata().tokens.is_empty());
        assert!(request.materialized_tokens().is_none());
        assert_eq!(request.into_direct_request().tokens, vec![1, 2]);
    }

    #[test]
    fn delta_cumulative_prompts_remain_materialized_during_setup() {
        let driver =
            WorkloadDriver::new_concurrency_accumulating_deltas(two_session_trace(), 1, 1).unwrap();

        assert!(driver.sessions.iter().all(|session| {
            session
                .turns
                .iter()
                .all(|turn| matches!(turn.prompt_tokens, PromptTokens::Materialized(_)))
        }));
    }

    #[test]
    fn deferred_prompt_validation_preserves_setup_errors() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "invalid".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 5,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    ..Default::default()
                }],
            }],
        };

        let error = WorkloadDriver::new_trace(trace, 4).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("input_length 5 exceeds synthesized capacity 4")
        );
    }

    #[test]
    fn unknown_completion_preserves_in_flight_state() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;

        let error = driver.on_complete(Uuid::new_v4(), 1.0).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unknown workload request terminal")
        );
        assert!(driver.in_flight.contains_key(&request_uuid));
        assert_eq!(driver.sessions[session_index].in_flight, Some(request_uuid));
    }

    #[test]
    fn unknown_cancellation_is_noop() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;

        driver.release_cap_slot(Uuid::new_v4(), 1.0);

        assert!(driver.in_flight.contains_key(&request_uuid));
        assert_eq!(driver.sessions[session_index].in_flight, Some(request_uuid));
    }

    #[test]
    fn inconsistent_session_mapping_preserves_in_flight_entry() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;
        driver.sessions[session_index].in_flight = Some(Uuid::new_v4());

        let error = driver.on_complete(request_uuid, 1.0).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("does not match in-flight request")
        );
        assert!(driver.in_flight.contains_key(&request_uuid));
        assert_eq!(driver.sessions[session_index].next_turn_index, 0);
    }

    #[test]
    fn cap_clamps_pop_ready_when_limit_is_unbounded() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        let second = driver.pop_ready(0.0, usize::MAX);
        assert!(
            second.is_empty(),
            "cap should block dispatch while slot is held"
        );
    }

    #[test]
    fn pop_ready_admits_next_turn_after_on_complete() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);
        let uuid = admitted[0].request_uuid;
        driver.on_complete(uuid, 10.0).unwrap();

        // next admitted turn is *this* session's turn-1
        // (ready at completion 10 + think-time 5 = 15)
        let next = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].turn_index, 1);
        assert_ne!(next[0].request_uuid, uuid);
    }

    #[test]
    fn concurrency_is_depth_first_holding_slot_across_think_time() {
        // Session A: 2 turns (turn-1 has a 5ms think-time). Session B: 1 turn. cap = 1.
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        // A.turn0 admitted; B is pending (not activated — cap is 1).
        let a0 = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(a0.len(), 1);
        assert_eq!(a0[0].turn_index, 0);
        let a0_uuid = a0[0].request_uuid;
        driver.on_complete(a0_uuid, 10.0).unwrap();

        // During A's think-time (turn-1 ready at 10+5=15), B must NOT slip in: A holds the slot.
        assert!(
            driver.pop_ready(10.0, usize::MAX).is_empty(),
            "B must not be admitted while A holds its slot in think-time"
        );

        // A.turn1 dispatches before B ever starts (depth-first).
        let a1 = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(a1.len(), 1);
        assert_eq!(a1[0].turn_index, 1);
        driver.on_complete(a1[0].request_uuid, 20.0).unwrap();

        // Only now that A is fully done is B activated.
        let b0 = driver.pop_ready(20.0, usize::MAX);
        assert_eq!(b0.len(), 1);
        assert_eq!(b0[0].turn_index, 0);
        assert_ne!(b0[0].request_uuid, a0_uuid);
        assert!(!driver.is_drained(), "B still in flight");
        driver.on_complete(b0[0].request_uuid, 30.0).unwrap();
        assert!(driver.is_drained());
    }

    #[test]
    fn concurrency_cap2_admits_pending_when_active_session_finishes() {
        // cap = 2: A (2 turns) and B (1 turn) start active; C (1 turn) is pending.
        let mut driver = WorkloadDriver::new_concurrency(three_session_trace(), 1, 2).unwrap();

        // Initial cohort: A.t0 and B.t0 (the cap-2 set); C stays pending.
        let first = driver.pop_ready(0.0, usize::MAX);
        let mut ids: Vec<&str> = first.iter().map(|r| r.session_id.as_str()).collect();
        ids.sort();
        assert_eq!(
            ids,
            vec!["a", "b"],
            "cap-2 admits exactly A and B; C pending"
        );
        let a0 = first
            .iter()
            .find(|r| r.session_id == "a")
            .unwrap()
            .request_uuid;
        let b0 = first
            .iter()
            .find(|r| r.session_id == "b")
            .unwrap()
            .request_uuid;

        // A finishes turn-0 → enters think-time (A.t1 ready at 10+5=15); A keeps its slot.
        driver.on_complete(a0, 10.0).unwrap();
        // B finishes its only turn → frees a slot → C is activated.
        driver.on_complete(b0, 10.0).unwrap();

        // At t=10 only C is admittable (its freed slot); A is mid-think-time and retains
        // its slot — neither dropped nor re-admitted early.
        let at_10 = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(at_10.len(), 1, "only C is admittable at t=10");
        assert_eq!(at_10[0].session_id, "c");
        assert_eq!(at_10[0].turn_index, 0);

        // A's retained slot resumes once its think-time elapses (t=15), proving it was
        // never evicted by C's admission.
        let at_15 = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(at_15.len(), 1);
        assert_eq!(
            (at_15[0].session_id.as_str(), at_15[0].turn_index),
            ("a", 1)
        );
    }

    #[test]
    fn release_cap_slot_terminates_inflight_session_and_admits_pending() {
        // Mirrors an online InFlightGuard drop (cancellation), which calls release_cap_slot.
        // cap = 2: A (2 turns) + B (1 turn) active, C (1 turn) pending. A is in think-time,
        // B is in flight and gets cancelled.
        let mut driver = WorkloadDriver::new_concurrency(three_session_trace(), 1, 2).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        let a0 = first
            .iter()
            .find(|r| r.session_id == "a")
            .unwrap()
            .request_uuid;
        let b0 = first
            .iter()
            .find(|r| r.session_id == "b")
            .unwrap()
            .request_uuid;

        // A → think-time (A.t1 ready at 15), retains its slot.
        driver.on_complete(a0, 10.0).unwrap();
        // B cancelled in flight: the online guard drop releases B's slot and terminates it.
        driver.release_cap_slot(b0, 10.0);

        // B's freed slot admits C; A's continuation is untouched.
        let at_10 = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(at_10.len(), 1);
        assert_eq!(
            at_10[0].session_id, "c",
            "C admitted into the slot freed by B's cancellation"
        );
        driver.on_complete(at_10[0].request_uuid, 12.0).unwrap();

        // A's continuation survived the cancellation and resumes after its think-time.
        let a1 = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(a1.len(), 1);
        assert_eq!((a1[0].session_id.as_str(), a1[0].turn_index), ("a", 1));
        driver.on_complete(a1[0].request_uuid, 20.0).unwrap();

        // A (2 turns), B (cancelled/terminated), C (1 turn) all resolved → drained.
        assert!(driver.is_drained());
    }

    #[test]
    fn next_ready_time_ms_returns_none_at_cap() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);

        assert!(
            driver.next_ready_time_ms().is_none(),
            "expected None while at cap even with ready sessions queued"
        );

        driver.on_complete(admitted[0].request_uuid, 10.0).unwrap();
        assert!(
            driver.next_ready_time_ms().is_some(),
            "expected readiness after a slot is freed"
        );
    }

    #[test]
    fn uncapped_concurrency_admits_all_sessions_up_to_caller_limit() {
        // usize::MAX cap == effectively uncapped: every session is activated, so the
        // caller's pop_ready limit is the only bound.
        let mut driver =
            WorkloadDriver::new_concurrency(two_session_trace(), 1, usize::MAX).unwrap();

        let admitted = driver.pop_ready(0.0, 5);
        assert_eq!(
            admitted.len(),
            2,
            "both sessions should admit when uncapped"
        );
        assert!(driver.next_ready_time_ms().is_none());
    }

    #[test]
    fn release_cap_slot_is_noop_after_on_complete() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        let uuid = admitted[0].request_uuid;
        driver.on_complete(uuid, 5.0).unwrap();

        // release_cap_slot after on_complete is a no-op (the in-flight entry is already
        // gone), so it must NOT double-decrement active_sessions. The session still holds its
        // slot for turn-1 (ready at 5 + think-time 5 = 10)
        driver.release_cap_slot(uuid, 5.0);

        let next = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].turn_index, 1);
        assert_ne!(next[0].request_uuid, uuid);
    }

    #[test]
    fn release_cap_slot_recovers_cap_when_on_complete_was_skipped() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);

        driver.release_cap_slot(admitted[0].request_uuid, 0.0);

        let next = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(
            next.len(),
            1,
            "cap slot should be available after release_cap_slot"
        );
    }

    #[test]
    fn release_cap_slot_terminates_session_so_is_drained_completes() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);
        let stuck_uuid = admitted[0].request_uuid;

        driver.release_cap_slot(stuck_uuid, 0.0);

        let neighbor = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(
            neighbor.len(),
            1,
            "other session must still be admissible after its neighbor was terminated"
        );
        driver.on_complete(neighbor[0].request_uuid, 1.0).unwrap();

        assert!(
            driver.is_drained(),
            "is_drained must become true so run_workload can exit"
        );
    }

    #[test]
    fn full_prompt_modes_plan_missing_output_token_ids_deterministically() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![TurnTrace {
                    input_length: 2,
                    max_output_tokens: 3,
                    hash_ids: vec![10, 11],
                    ..Default::default()
                }],
            }],
        };
        assert_deterministic_output_plan(
            WorkloadDriver::new_trace(trace.clone(), 1).unwrap(),
            WorkloadDriver::new_trace(trace, 1).unwrap(),
            3,
        );

        let trace = AgenticTrace {
            block_size: 1,
            turns: vec![AgenticTurnTrace {
                request_id: "r1".into(),
                session_id: "a".into(),
                input_length: 2,
                max_output_tokens: 3,
                hash_ids: vec![10, 11],
                first_ready_timestamp_ms: Some(0.0),
                prefix_reset: false,
                ..Default::default()
            }],
        };
        assert_deterministic_output_plan(
            WorkloadDriver::new_agentic_trace(trace.clone(), 1).unwrap(),
            WorkloadDriver::new_agentic_trace(trace, 1).unwrap(),
            3,
        );
    }

    #[test]
    fn accumulating_delta_mode_includes_previous_output_tokens() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 6,
                        max_output_tokens: 2,
                        output_token_ids: Some(vec![20, 21]),
                        replay_key: None,
                        hash_ids: vec![10, 11],
                        delay_after_previous_ms: 0.0,
                        priority: 3,
                        strict_priority: 4,
                        policy_class: None,
                        routing_constraints: Default::default(),
                    },
                    TurnTrace {
                        input_length: 3,
                        max_output_tokens: 1,
                        output_token_ids: None,
                        replay_key: None,
                        hash_ids: vec![12],
                        delay_after_previous_ms: 5.0,
                        priority: -2,
                        strict_priority: 7,
                        policy_class: None,
                        routing_constraints: Default::default(),
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 4, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].request.tokens, vec![10, 10, 10, 10, 11, 11]);
        assert_eq!(first[0].request.output_token_ids, Some(vec![20, 21]));
        assert_eq!(first[0].request.priority, 3);
        assert_eq!(first[0].request.strict_priority, 4);
        driver.on_output_token(first[0].request_uuid, 20).unwrap();
        driver.on_output_token(first[0].request_uuid, 21).unwrap();
        driver.on_complete(first[0].request_uuid, 10.0).unwrap();

        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(
            second[0].request.tokens,
            vec![10, 10, 10, 10, 11, 11, 20, 21, 12, 12, 12]
        );
        assert_eq!(second[0].request.priority, -2);
        assert_eq!(second[0].request.strict_priority, 7);
    }

    #[test]
    fn accumulating_delta_mode_plans_missing_output_token_ids() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 2,
                        max_output_tokens: 3,
                        hash_ids: vec![10, 11],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        ..Default::default()
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        let planned_output = first[0]
            .request
            .output_token_ids
            .clone()
            .expect("delta replay should plan synthetic outputs");
        assert_eq!(planned_output.len(), 3);
        for &token_id in &planned_output {
            driver
                .on_output_token(first[0].request_uuid, token_id)
                .unwrap();
        }
        driver.on_complete(first[0].request_uuid, 1.0).unwrap();

        let second = driver.pop_ready(1.0, usize::MAX);
        assert_eq!(second.len(), 1);
        let mut expected = vec![10, 11];
        expected.extend(planned_output);
        expected.push(12);
        assert_eq!(second[0].request.tokens, expected);
        assert_eq!(
            second[0].request.output_token_ids.as_ref().map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn accumulating_delta_mode_appends_only_emitted_output_tokens() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 3,
                        output_token_ids: Some(vec![20, 21, 22]),
                        hash_ids: vec![10],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        ..Default::default()
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        driver.on_output_token(first[0].request_uuid, 20).unwrap();
        driver.on_output_token(first[0].request_uuid, 21).unwrap();
        driver.on_complete(first[0].request_uuid, 1.0).unwrap();

        let second = driver.pop_ready(1.0, usize::MAX);
        assert_eq!(second[0].request.tokens, vec![10, 20, 21, 12]);
    }

    #[test]
    fn rejected_non_agentic_request_ends_sequential_session() {
        let trace = Trace {
            block_size: 1,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 2,
                        output_token_ids: Some(vec![20, 21]),
                        hash_ids: vec![10],
                        ..Default::default()
                    },
                    TurnTrace {
                        input_length: 1,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        ..Default::default()
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 1, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        let cascaded = driver
            .on_terminal(
                first[0].request_uuid,
                1.0,
                WorkloadTerminalStatus::Rejected,
            )
            .unwrap();

        assert!(cascaded.is_empty());
        assert!(driver.pop_ready(1.0, usize::MAX).is_empty());
        assert!(driver.is_drained());
    }

    #[test]
    fn agentic_mode_releases_turn_after_dependency_completion_plus_delay() {
        let trace = AgenticTrace {
            block_size: 1,
            turns: vec![
                AgenticTurnTrace {
                    request_id: "r1".into(),
                    session_id: "root".into(),
                    input_length: 2,
                    max_output_tokens: 1,
                    hash_ids: vec![1, 2],
                    first_ready_timestamp_ms: Some(0.0),
                    delay_after_dependencies_ms: 0.0,
                    wait_for: Vec::new(),
                    prefix_reset: false,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "r2".into(),
                    session_id: "root".into(),
                    authored_turn_index: 1,
                    input_length: 2,
                    max_output_tokens: 1,
                    hash_ids: vec![1, 3],
                    first_ready_timestamp_ms: Some(100.0),
                    delay_after_dependencies_ms: 5.0,
                    wait_for: vec!["r1".into()],
                    prefix_reset: false,
                    ..Default::default()
                },
            ],
        };
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].scheduled_ready_at_ms, 0.0);
        assert!(driver.pop_ready(14.0, usize::MAX).is_empty());

        driver.on_complete(first[0].request_uuid, 10.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(15.0));
        assert!(driver.pop_ready(14.0, usize::MAX).is_empty());
        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].scheduled_ready_at_ms, 15.0);
    }

    #[test]
    fn canceled_agentic_parent_cascades_and_never_releases_child() {
        let trace = AgenticTrace {
            block_size: 1,
            turns: vec![
                AgenticTurnTrace {
                    request_id: "r1".into(),
                    session_id: "root".into(),
                    input_length: 2,
                    max_output_tokens: 1,
                    hash_ids: vec![1, 2],
                    first_ready_timestamp_ms: Some(0.0),
                    delay_after_dependencies_ms: 0.0,
                    wait_for: Vec::new(),
                    prefix_reset: false,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "r2".into(),
                    session_id: "child".into(),
                    input_length: 2,
                    max_output_tokens: 1,
                    hash_ids: vec![1, 3],
                    first_ready_timestamp_ms: Some(100.0),
                    delay_after_dependencies_ms: 5.0,
                    wait_for: vec!["r1".into()],
                    prefix_reset: false,
                    ..Default::default()
                },
            ],
        };
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);

        let cascaded = driver
            .on_terminal(
                first[0].request_uuid,
                10.0,
                WorkloadTerminalStatus::Canceled,
            )
            .unwrap();

        assert_eq!(cascaded.len(), 1);
        assert_eq!(cascaded[0].logical_request_id, "r2");
        assert_eq!(cascaded[0].session_id, "child");
        assert_eq!(cascaded[0].authored_turn_index, 0);
        assert_eq!(cascaded[0].input_length, 2);
        assert_eq!(cascaded[0].requested_output_length, 1);
        assert_eq!(cascaded[0].emitted_output_count, 0);
        assert_eq!(cascaded[0].status, WorkloadTerminalStatus::Canceled);
        assert_eq!(driver.next_ready_time_ms(), None);
        assert!(driver.pop_ready(15.0, usize::MAX).is_empty());
        assert!(driver.is_drained());
    }

    #[test]
    fn agentic_mode_waits_for_slowest_dependency() {
        let trace = AgenticTrace {
            block_size: 1,
            turns: vec![
                AgenticTurnTrace {
                    request_id: "a".into(),
                    session_id: "a".into(),
                    input_length: 1,
                    max_output_tokens: 1,
                    hash_ids: vec![1],
                    first_ready_timestamp_ms: Some(0.0),
                    delay_after_dependencies_ms: 0.0,
                    wait_for: Vec::new(),
                    prefix_reset: false,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "b".into(),
                    session_id: "b".into(),
                    input_length: 1,
                    max_output_tokens: 1,
                    hash_ids: vec![2],
                    first_ready_timestamp_ms: Some(0.0),
                    delay_after_dependencies_ms: 0.0,
                    wait_for: Vec::new(),
                    prefix_reset: false,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "join".into(),
                    session_id: "root".into(),
                    input_length: 1,
                    max_output_tokens: 1,
                    hash_ids: vec![3],
                    first_ready_timestamp_ms: Some(1.0),
                    delay_after_dependencies_ms: 2.0,
                    wait_for: vec!["a".into(), "b".into()],
                    prefix_reset: false,
                    ..Default::default()
                },
            ],
        };
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let initial = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(initial.len(), 2);
        driver.on_complete(initial[0].request_uuid, 10.0).unwrap();
        assert!(driver.next_ready_time_ms().is_none());
        driver.on_complete(initial[1].request_uuid, 30.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(32.0));
    }

    fn open_agentic_turn(
        request_id: &str,
        session_id: &str,
        authored_turn_index: usize,
        wait_for: &[&str],
        first_ready_timestamp_ms: Option<f64>,
        delay_after_dependencies_ms: f64,
    ) -> AgenticTurnTrace {
        AgenticTurnTrace {
            request_id: request_id.into(),
            session_id: session_id.into(),
            authored_turn_index,
            internal_uuid: None,
            input_length: 1,
            max_output_tokens: 1,
            output_token_ids: Some(vec![100 + authored_turn_index as u32]),
            hash_ids: vec![authored_turn_index as u32 + 1],
            first_ready_timestamp_ms,
            delay_after_dependencies_ms,
            wait_for: wait_for
                .iter()
                .map(|dependency| (*dependency).into())
                .collect(),
            ..Default::default()
        }
    }

    fn open_agentic_trace(turns: Vec<AgenticTurnTrace>) -> AgenticTrace {
        AgenticTrace {
            block_size: 1,
            turns,
        }
    }

    #[test]
    fn open_agentic_quiescence_append_close_and_identity() {
        let mut driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        assert!(driver.is_open());
        assert!(!driver.is_drained());
        assert_eq!(driver.pending_turns(), 0);
        assert_eq!(driver.next_ready_time_ms(), None);

        let mut root = open_agentic_turn("logical-root", "session-a", 7, &[], Some(3.0), 0.0);
        root.internal_uuid = Some(Uuid::from_u128(77));
        driver
            .append_agentic_trace(open_agentic_trace(vec![root]), 10.0)
            .unwrap();
        assert_eq!(driver.total_turns(), 1);
        assert_eq!(driver.pending_turns(), 1);
        assert_eq!(driver.next_ready_time_ms(), Some(13.0));
        assert!(driver.pop_ready(12.0, usize::MAX).is_empty());

        let ready = driver.pop_ready(13.0, usize::MAX);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].logical_request_id.as_deref(), Some("logical-root"));
        assert_eq!(ready[0].session_id, "session-a");
        assert_eq!(ready[0].turn_index, 7);
        assert_eq!(ready[0].authored_turn_index, 7);
        assert_eq!(ready[0].request_uuid, Uuid::from_u128(77));
        assert_eq!(ready[0].request.uuid, Some(Uuid::from_u128(77)));
        assert!(!ready[0].prefix_reset);

        driver.on_output_token(ready[0].request_uuid, 107).unwrap();
        driver.on_complete(ready[0].request_uuid, 20.0).unwrap();
        assert_eq!(driver.pending_turns(), 0);
        assert_eq!(
            driver.total_turns(),
            1,
            "submitted total remains cumulative"
        );
        assert!(
            !driver.is_drained(),
            "open and idle is quiescent, not drained"
        );
        driver.close().unwrap();
        assert!(!driver.is_open());
        assert!(driver.is_drained());
        assert!(
            driver
                .append_agentic_trace(
                    open_agentic_trace(vec![open_agentic_turn("late", "late", 0, &[], None, 0.0)]),
                    20.0,
                )
                .unwrap_err()
                .to_string()
                .contains("admission is closed")
        );
    }

    #[test]
    fn open_agentic_rejects_global_identity_collisions_atomically() {
        let mut driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        let mut first = open_agentic_turn("one", "session", 0, &[], None, 0.0);
        first.internal_uuid = Some(Uuid::from_u128(1));
        driver
            .append_agentic_trace(open_agentic_trace(vec![first]), 0.0)
            .unwrap();

        let duplicate_id = open_agentic_turn("one", "other", 0, &[], None, 0.0);
        assert!(
            driver
                .append_agentic_trace(open_agentic_trace(vec![duplicate_id]), 0.0)
                .unwrap_err()
                .to_string()
                .contains("request_id one is already registered")
        );

        let duplicate_turn = open_agentic_turn("two", "session", 0, &[], None, 0.0);
        assert!(
            driver
                .append_agentic_trace(open_agentic_trace(vec![duplicate_turn]), 0.0)
                .unwrap_err()
                .to_string()
                .contains("conflicts with existing request one")
        );

        let mut duplicate_uuid = open_agentic_turn("three", "other", 0, &[], None, 0.0);
        duplicate_uuid.internal_uuid = Some(Uuid::from_u128(1));
        assert!(
            driver
                .append_agentic_trace(open_agentic_trace(vec![duplicate_uuid]), 0.0)
                .unwrap_err()
                .to_string()
                .contains("duplicates internal UUID")
        );
        assert_eq!(driver.total_turns(), 1);
        assert_eq!(driver.pending_turns(), 1);
    }

    #[test]
    fn open_agentic_validates_dependency_graph_before_mutation() {
        let mut driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        let missing = open_agentic_turn("child", "child", 0, &["missing"], None, 0.0);
        assert!(
            driver
                .append_agentic_trace(open_agentic_trace(vec![missing]), 0.0)
                .unwrap_err()
                .to_string()
                .contains("unknown request_id missing")
        );

        let self_edge = open_agentic_turn("self", "self", 0, &["self"], None, 0.0);
        assert!(
            driver
                .append_agentic_trace(open_agentic_trace(vec![self_edge]), 0.0)
                .unwrap_err()
                .to_string()
                .contains("cannot wait for itself")
        );

        let a = open_agentic_turn("a", "a", 0, &["b"], None, 0.0);
        let b = open_agentic_turn("b", "b", 0, &["a"], None, 0.0);
        assert!(
            driver
                .append_agentic_trace(open_agentic_trace(vec![a, b]), 0.0)
                .unwrap_err()
                .to_string()
                .contains("cycle detected")
        );
        assert_eq!(driver.total_turns(), 0);
    }

    #[test]
    fn static_and_open_agentic_reject_reporting_only_prefix_reset() {
        let mut turn = open_agentic_turn("root", "root", 0, &[], None, 0.0);
        turn.prefix_reset = true;
        let trace = open_agentic_trace(vec![turn]);

        let static_error = WorkloadDriver::new_agentic_trace(trace.clone(), 1).unwrap_err();
        assert!(
            static_error
                .to_string()
                .contains("unsupported prefix_reset=true")
        );

        let mut open_driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        let open_error = open_driver.append_agentic_trace(trace, 0.0).unwrap_err();
        assert!(
            open_error
                .to_string()
                .contains("unsupported prefix_reset=true")
        );
        assert_eq!(open_driver.total_turns(), 0);
        assert_eq!(open_driver.pending_turns(), 0);
    }

    #[test]
    fn open_agentic_fanout_and_join_use_actual_latest_terminal_time() {
        let trace = open_agentic_trace(vec![
            open_agentic_turn("root", "root", 0, &[], None, 0.0),
            open_agentic_turn("left", "left", 0, &["root"], None, 0.0),
            open_agentic_turn("right", "right", 0, &["root"], None, 0.0),
            open_agentic_turn("join", "join", 0, &["left", "right"], None, 2.0),
        ]);
        let mut driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        driver.append_agentic_trace(trace, 5.0).unwrap();

        let root = driver.pop_ready(5.0, usize::MAX);
        assert_eq!(root.len(), 1);
        driver.on_complete(root[0].request_uuid, 10.0).unwrap();

        let fanout = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(fanout.len(), 2);
        assert_eq!(fanout[0].logical_request_id.as_deref(), Some("left"));
        assert_eq!(fanout[1].logical_request_id.as_deref(), Some("right"));
        driver.on_complete(fanout[0].request_uuid, 20.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), None);
        driver.on_complete(fanout[1].request_uuid, 30.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), Some(32.0));
        let join = driver.pop_ready(32.0, usize::MAX);
        assert_eq!(join.len(), 1);
        assert_eq!(join[0].logical_request_id.as_deref(), Some("join"));
    }

    #[test]
    fn failed_join_parent_cancels_join_and_downstream_once() {
        let trace = open_agentic_trace(vec![
            open_agentic_turn("a", "a", 0, &[], None, 0.0),
            open_agentic_turn("b", "b", 0, &[], None, 0.0),
            open_agentic_turn("join", "join", 0, &["a", "b"], None, 0.0),
            open_agentic_turn("leaf", "leaf", 0, &["join"], None, 0.0),
        ]);
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1)
            .unwrap()
            .with_deterministic_request_ids(1);
        let roots = driver.pop_ready(0.0, usize::MAX);
        let a_uuid = roots
            .iter()
            .find(|ready| ready.logical_request_id.as_deref() == Some("a"))
            .unwrap()
            .request_uuid;
        let b_uuid = roots
            .iter()
            .find(|ready| ready.logical_request_id.as_deref() == Some("b"))
            .unwrap()
            .request_uuid;

        let cascaded = driver
            .on_terminal(a_uuid, 5.0, WorkloadTerminalStatus::Failed)
            .unwrap();

        assert_eq!(
            cascaded
                .iter()
                .map(|terminal| terminal.logical_request_id.as_str())
                .collect::<Vec<_>>(),
            vec!["join", "leaf"]
        );
        assert_eq!(cascaded[0].request_uuid, Uuid::from_u128(3));
        assert_eq!(cascaded[1].request_uuid, Uuid::from_u128(4));
        assert_eq!(driver.next_ready_time_ms(), None);
        assert!(!driver.is_drained(), "the independent root is still in flight");

        driver.on_complete(b_uuid, 7.0).unwrap();
        assert_eq!(driver.next_ready_time_ms(), None);
        assert!(driver.pop_ready(7.0, usize::MAX).is_empty());
        assert!(driver.is_drained());
    }

    #[test]
    fn non_completed_agentic_parent_cascades_for_all_statuses() {
        for status in [
            WorkloadTerminalStatus::Rejected,
            WorkloadTerminalStatus::Failed,
            WorkloadTerminalStatus::Canceled,
        ] {
            let trace = open_agentic_trace(vec![
                open_agentic_turn("root", "root", 0, &[], None, 0.0),
                open_agentic_turn("child", "child", 0, &["root"], None, 0.0),
                open_agentic_turn("leaf", "leaf", 0, &["child"], None, 0.0),
            ]);
            let mut driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
            driver.append_agentic_trace(trace, 0.0).unwrap();
            driver.close().unwrap();
            let mut driver = driver.with_deterministic_request_ids(1);
            let root = driver.pop_ready(0.0, usize::MAX);

            let cascaded = driver
                .on_terminal(root[0].request_uuid, 7.0, status)
                .unwrap();

            assert_eq!(
                cascaded
                    .iter()
                    .map(|terminal| terminal.logical_request_id.as_str())
                    .collect::<Vec<_>>(),
                vec!["child", "leaf"]
            );
            assert_eq!(cascaded[0].request_uuid, Uuid::from_u128(2));
            assert_eq!(cascaded[1].request_uuid, Uuid::from_u128(3));
            assert!(
                cascaded
                    .iter()
                    .all(|terminal| terminal.status == WorkloadTerminalStatus::Canceled)
            );
            assert_eq!(driver.next_ready_time_ms(), None);
            assert!(driver.pop_ready(7.0, usize::MAX).is_empty());
            assert_eq!(driver.pending_turns(), 0);
            assert!(driver.is_drained());
        }
    }

    #[test]
    fn open_agentic_forwards_and_validates_exact_output_tokens() {
        let mut turn = open_agentic_turn("root", "root", 0, &[], None, 0.0);
        turn.max_output_tokens = 2;
        turn.output_token_ids = Some(vec![9, 10]);
        let mut driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        driver
            .append_agentic_trace(open_agentic_trace(vec![turn]), 0.0)
            .unwrap();
        let ready = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(ready[0].request.output_token_ids, Some(vec![9, 10]));
        assert!(
            driver
                .on_output_token(ready[0].request_uuid, 8)
                .unwrap_err()
                .to_string()
                .contains("expected 9")
        );
        driver.on_output_token(ready[0].request_uuid, 9).unwrap();
        driver.on_output_token(ready[0].request_uuid, 10).unwrap();
        driver.on_complete(ready[0].request_uuid, 1.0).unwrap();
    }

    #[test]
    fn static_and_appendable_agentic_dispatch_are_equivalent() {
        let trace = open_agentic_trace(vec![
            open_agentic_turn("root", "session", 0, &[], Some(2.0), 0.0),
            open_agentic_turn("child", "session", 1, &["root"], None, 3.0),
        ]);
        let mut static_driver = WorkloadDriver::new_agentic_trace(trace.clone(), 1)
            .unwrap()
            .with_deterministic_request_ids(1);
        let mut open_driver = WorkloadDriver::new_open_agentic(1, 1).unwrap();
        open_driver.append_agentic_trace(trace, 0.0).unwrap();
        open_driver.close().unwrap();
        let mut open_driver = open_driver.with_deterministic_request_ids(1);

        let static_root = static_driver.pop_ready(2.0, usize::MAX);
        let open_root = open_driver.pop_ready(2.0, usize::MAX);
        assert_eq!(static_root[0].request_uuid, open_root[0].request_uuid);
        assert_eq!(
            static_root[0].logical_request_id,
            open_root[0].logical_request_id
        );
        assert_eq!(static_root[0].request.tokens, open_root[0].request.tokens);
        assert_eq!(
            static_root[0].request.output_token_ids,
            open_root[0].request.output_token_ids
        );
        static_driver
            .on_complete(static_root[0].request_uuid, 10.0)
            .unwrap();
        open_driver
            .on_complete(open_root[0].request_uuid, 10.0)
            .unwrap();
        let static_child = static_driver.pop_ready(13.0, usize::MAX);
        let open_child = open_driver.pop_ready(13.0, usize::MAX);
        assert_eq!(static_child[0].scheduled_ready_at_ms, 13.0);
        assert_eq!(
            static_child[0].logical_request_id,
            open_child[0].logical_request_id
        );
        assert_eq!(static_child[0].request.tokens, open_child[0].request.tokens);
    }
}
