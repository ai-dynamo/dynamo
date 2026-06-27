// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::{Context, Result, anyhow, bail};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::types::{AgenticTrace, ReadyTurn, ReplayRequestHashes, Trace};
use crate::common::protocols::DirectRequest;

#[derive(Debug)]
enum SchedulingPolicy {
    Trace,
    Concurrency(ConcurrencyState),
    Agentic(AgenticState),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptMode {
    Full,
    DeltaCumulative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    Completed,
    Cancelled,
}

#[derive(Debug)]
struct TurnResolution {
    request_id: Option<String>,
    session_ended: bool,
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
struct TurnRuntime {
    request_id: Option<String>,
    replay_key: Option<String>,
    tokens: Vec<u32>,
    max_output_tokens: usize,
    output_token_ids: Option<Vec<u32>>,
    delay_after_previous_ms: f64,
    priority: i32,
    strict_priority: u32,
    policy_class: Option<String>,
    replay_hashes: Option<ReplayRequestHashes>,
}

#[derive(Debug, Clone, Copy)]
struct InFlightTurn {
    session_index: usize,
    turn_index: usize,
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
}

#[derive(Debug)]
pub struct WorkloadDriver {
    policy: SchedulingPolicy,
    prompt_mode: PromptMode,
    engine_block_size: u32,
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
        )
    }

    pub(crate) fn new_agentic_trace(trace: AgenticTrace, engine_block_size: usize) -> Result<Self> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let trace_block_size = trace.block_size;

        let mut dependents: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        let mut remaining_dependencies = Vec::with_capacity(trace.turns.len());
        let mut ready_after_ms = Vec::with_capacity(trace.turns.len());
        let mut sessions = Vec::with_capacity(trace.turns.len());

        for (session_index, turn) in trace.turns.into_iter().enumerate() {
            for dependency in &turn.wait_for {
                dependents
                    .entry(dependency.clone())
                    .or_default()
                    .push(session_index);
            }
            remaining_dependencies.push(turn.wait_for.len());
            ready_after_ms.push(0.0);

            let replay_hashes = Some(turn.to_replay_hashes(trace_block_size, engine_block_size)?);
            let tokens = turn.synthesize_tokens(trace_block_size)?;
            let next_ready_at_ms = if turn.wait_for.is_empty() {
                Some(turn.first_ready_timestamp_ms.unwrap_or(0.0))
            } else {
                None
            };
            sessions.push(SessionRuntime {
                session_id: turn.session_id,
                turns: vec![TurnRuntime {
                    request_id: Some(turn.request_id),
                    replay_key: turn.replay_key,
                    tokens,
                    max_output_tokens: turn.max_output_tokens,
                    output_token_ids: turn.output_token_ids,
                    delay_after_previous_ms: turn.delay_after_dependencies_ms,
                    priority: turn.priority,
                    strict_priority: turn.strict_priority,
                    policy_class: turn.policy_class,
                    replay_hashes,
                }],
                cumulative_tokens: Vec::new(),
                next_turn_index: 0,
                next_ready_at_ms,
                in_flight: None,
            });
        }

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

        Ok(Self {
            policy: SchedulingPolicy::Agentic(AgenticState {
                remaining_dependencies,
                ready_after_ms,
                dependents,
            }),
            prompt_mode: PromptMode::Full,
            engine_block_size: engine_block_size_u32,
            sessions,
            in_flight: FxHashMap::default(),
            ready_sessions,
        })
    }

    fn new(
        trace: Trace,
        engine_block_size: usize,
        policy: SchedulingPolicy,
        prompt_mode: PromptMode,
    ) -> Result<Self> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let trace_block_size = trace.block_size;
        let is_concurrency = matches!(&policy, SchedulingPolicy::Concurrency(_));
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
                    .map(|turn| -> Result<TurnRuntime> {
                        let replay_hashes = if prompt_mode == PromptMode::Full {
                            Some(turn.to_replay_hashes(trace_block_size, engine_block_size)?)
                        } else {
                            None
                        };
                        Ok(TurnRuntime {
                            request_id: None,
                            tokens: turn.synthesize_tokens(trace_block_size)?,
                            replay_key: turn.replay_key,
                            max_output_tokens: turn.max_output_tokens,
                            output_token_ids: turn.output_token_ids,
                            delay_after_previous_ms: turn.delay_after_previous_ms,
                            priority: turn.priority,
                            strict_priority: turn.strict_priority,
                            policy_class: turn.policy_class,
                            replay_hashes,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let cumulative_capacity = if prompt_mode == PromptMode::DeltaCumulative {
                    turns.iter().map(|turn| turn.tokens.len()).sum()
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
            engine_block_size: engine_block_size_u32,
            sessions,
            in_flight: FxHashMap::default(),
            ready_sessions,
        };
        if let SchedulingPolicy::Concurrency(state) = &mut driver.policy {
            state.activate_pending(&mut driver.sessions, &mut driver.ready_sessions, 0.0);
        }
        Ok(driver)
    }

    /// Failure-path companion: release a cap slot and terminate the owning session.
    /// No-op if `on_complete` already ran. Used when a request task is cancelled
    /// or panics before reaching `on_complete`.
    ///
    /// Terminating the session (marking it exhausted) prevents `run_workload` from
    /// deadlocking: `pop_ready` skips sessions with `in_flight.is_some()`, so a
    /// leaked session would leave `is_drained` stuck at `false` forever.
    pub fn release_cap_slot(&mut self, request_uuid: Uuid, now_ms: f64) {
        let Ok(Some(resolution)) = self.resolve_turn(request_uuid, now_ms, TurnOutcome::Cancelled)
        else {
            return;
        };
        self.apply_resolution(resolution, now_ms);
    }

    pub fn pop_ready(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyTurn> {
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
            let session = &mut self.sessions[session_index];
            if session.in_flight.is_some()
                || session.next_turn_index != ready_session.turn_index
                || session.next_ready_at_ms != Some(ready_session.ready_at_ms)
            {
                continue;
            }
            let turn_index = session.next_turn_index;
            let scheduled_ready_at_ms = session
                .next_ready_at_ms
                .expect("ready session must have a timestamp");
            let request_uuid = Uuid::new_v4();
            let turn = &session.turns[turn_index];
            let arrival_timestamp_ms = self.policy.arrival_timestamp_ms(scheduled_ready_at_ms);
            let (request_tokens, replay_hashes) = match self.prompt_mode {
                PromptMode::Full => (
                    turn.tokens.clone(),
                    turn.replay_hashes
                        .as_ref()
                        .expect("full-prompt workload turns precompute replay hashes")
                        .clone(),
                ),
                PromptMode::DeltaCumulative => {
                    session.cumulative_tokens.extend_from_slice(&turn.tokens);
                    let request_tokens = session.cumulative_tokens.clone();
                    let replay_hashes =
                        ReplayRequestHashes::from_tokens(&request_tokens, self.engine_block_size);
                    (request_tokens, replay_hashes)
                }
            };
            let request = DirectRequest {
                tokens: request_tokens,
                max_output_tokens: turn.max_output_tokens,
                output_token_ids: turn.output_token_ids.clone(),
                uuid: Some(request_uuid),
                dp_rank: 0,
                arrival_timestamp_ms,
                priority: turn.priority,
                strict_priority: turn.strict_priority,
                policy_class: turn.policy_class.clone(),
            };
            session.in_flight = Some(request_uuid);
            session.next_ready_at_ms = None;
            self.in_flight.insert(
                request_uuid,
                InFlightTurn {
                    session_index,
                    turn_index,
                },
            );
            emitted.push(ReadyTurn {
                request_uuid,
                session_id: session.session_id.clone(),
                turn_index,
                replay_key: turn.replay_key.clone(),
                scheduled_ready_at_ms,
                replay_hashes: Some(replay_hashes),
                request,
            });
        }
        emitted
    }

    pub fn on_complete(&mut self, request_uuid: Uuid, now_ms: f64) -> Result<()> {
        let resolution = self
            .resolve_turn(request_uuid, now_ms, TurnOutcome::Completed)?
            .expect("completed turns require an in-flight request");
        self.apply_resolution(resolution, now_ms);
        Ok(())
    }

    fn resolve_turn(
        &mut self,
        request_uuid: Uuid,
        now_ms: f64,
        outcome: TurnOutcome,
    ) -> Result<Option<TurnResolution>> {
        let Some(in_flight) = self.in_flight.get(&request_uuid).copied() else {
            return match outcome {
                TurnOutcome::Completed => Err(anyhow!(
                    "unknown workload request completion for {request_uuid}"
                )),
                TurnOutcome::Cancelled => Ok(None),
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

        let request_id = turn.request_id.clone();
        let (next_turn_index, next_ready_at_ms, session_ended) = match outcome {
            TurnOutcome::Completed => {
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
            TurnOutcome::Cancelled => (session.turns.len(), None, true),
        };

        self.in_flight
            .remove(&request_uuid)
            .expect("validated in-flight request must still exist");
        let session = &mut self.sessions[in_flight.session_index];
        session.in_flight = None;
        session.next_turn_index = next_turn_index;
        session.next_ready_at_ms = next_ready_at_ms;
        if let Some(ready_at_ms) = next_ready_at_ms {
            self.ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index: in_flight.session_index,
                turn_index: next_turn_index,
            });
        }

        Ok(Some(TurnResolution {
            request_id,
            session_ended,
        }))
    }

    fn apply_resolution(&mut self, resolution: TurnResolution, now_ms: f64) {
        match &mut self.policy {
            SchedulingPolicy::Trace => {}
            SchedulingPolicy::Concurrency(state) => {
                if resolution.session_ended {
                    state.on_session_finished(&mut self.sessions, &mut self.ready_sessions, now_ms);
                }
            }
            SchedulingPolicy::Agentic(state) => {
                if let Some(request_id) = resolution.request_id {
                    state.release_dependents(
                        &mut self.sessions,
                        &mut self.ready_sessions,
                        &request_id,
                        now_ms,
                    );
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
        self.in_flight.is_empty()
            && self
                .sessions
                .iter()
                .all(|session| session.next_turn_index >= session.turns.len())
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
    fn unknown_completion_preserves_in_flight_state() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1, 1).unwrap();
        let admitted = driver.pop_ready(0.0, usize::MAX);
        let request_uuid = admitted[0].request_uuid;
        let session_index = driver.in_flight[&request_uuid].session_index;

        let error = driver.on_complete(Uuid::new_v4(), 1.0).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unknown workload request completion")
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
    fn accumulating_delta_mode_emits_cumulative_input_tokens() {
        let trace = Trace {
            block_size: 4,
            sessions: vec![SessionTrace {
                session_id: "a".into(),
                first_arrival_timestamp_ms: Some(0.0),
                turns: vec![
                    TurnTrace {
                        input_length: 6,
                        max_output_tokens: 1,
                        output_token_ids: None,
                        replay_key: None,
                        hash_ids: vec![10, 11],
                        delay_after_previous_ms: 0.0,
                        priority: 3,
                        strict_priority: 4,
                        policy_class: None,
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
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 4, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].request.tokens, vec![10, 10, 10, 10, 11, 11]);
        assert_eq!(first[0].request.priority, 3);
        assert_eq!(first[0].request.strict_priority, 4);
        driver.on_complete(first[0].request_uuid, 10.0).unwrap();

        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(
            second[0].request.tokens,
            vec![10, 10, 10, 10, 11, 11, 12, 12, 12]
        );
        assert_eq!(second[0].request.priority, -2);
        assert_eq!(second[0].request.strict_priority, 7);
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
                    prefix_reset: true,
                    ..Default::default()
                },
                AgenticTurnTrace {
                    request_id: "r2".into(),
                    session_id: "root".into(),
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
    fn agentic_mode_releases_dependents_when_cap_slot_is_released() {
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
                    prefix_reset: true,
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
                    prefix_reset: true,
                    ..Default::default()
                },
            ],
        };
        let mut driver = WorkloadDriver::new_agentic_trace(trace, 1).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);

        driver.release_cap_slot(first[0].request_uuid, 10.0);

        assert_eq!(driver.next_ready_time_ms(), Some(15.0));
        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].scheduled_ready_at_ms, 15.0);
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
                    prefix_reset: true,
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
                    prefix_reset: true,
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
}
