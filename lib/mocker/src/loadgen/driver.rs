// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::{Context, Result, anyhow, bail};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::types::{AgenticTrace, ReadyTurn, ReplayRequestHashes, Trace};
use crate::common::protocols::DirectRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverMode {
    Trace,
    Concurrency,
    AgenticTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptMode {
    Full,
    DeltaCumulative,
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
    tokens: Vec<u32>,
    max_output_tokens: usize,
    delay_after_previous_ms: f64,
    replay_hashes: Option<ReplayRequestHashes>,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct WorkloadDriver {
    mode: DriverMode,
    prompt_mode: PromptMode,
    engine_block_size: u32,
    sessions: Vec<SessionRuntime>,
    in_flight: FxHashMap<Uuid, InFlightTurn>,
    ready_sessions: BinaryHeap<ReadySession>,
    max_in_flight: Option<usize>,
    agentic_remaining_dependencies: Vec<usize>,
    agentic_ready_after_ms: Vec<f64>,
    agentic_dependents: FxHashMap<String, Vec<usize>>,
}

impl WorkloadDriver {
    pub(crate) fn new_trace(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            DriverMode::Trace,
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
            DriverMode::Trace,
            PromptMode::DeltaCumulative,
        )
    }

    pub(crate) fn new_concurrency(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            DriverMode::Concurrency,
            PromptMode::Full,
        )
    }

    pub(crate) fn new_concurrency_accumulating_deltas(
        trace: Trace,
        engine_block_size: usize,
    ) -> Result<Self> {
        Self::new(
            trace,
            engine_block_size,
            DriverMode::Concurrency,
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
                    tokens,
                    max_output_tokens: turn.max_output_tokens,
                    delay_after_previous_ms: turn.delay_after_dependencies_ms,
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
            mode: DriverMode::AgenticTrace,
            prompt_mode: PromptMode::Full,
            engine_block_size: engine_block_size_u32,
            sessions,
            in_flight: FxHashMap::default(),
            ready_sessions,
            max_in_flight: None,
            agentic_remaining_dependencies: remaining_dependencies,
            agentic_ready_after_ms: ready_after_ms,
            agentic_dependents: dependents,
        })
    }

    fn new(
        trace: Trace,
        engine_block_size: usize,
        mode: DriverMode,
        prompt_mode: PromptMode,
    ) -> Result<Self> {
        if engine_block_size == 0 {
            bail!("engine_block_size must be greater than 0");
        }
        let engine_block_size_u32 =
            u32::try_from(engine_block_size).context("engine_block_size does not fit in u32")?;
        let trace_block_size = trace.block_size;
        let sessions: Vec<SessionRuntime> = trace
            .sessions
            .into_iter()
            .map(|session| -> Result<SessionRuntime> {
                let next_ready_at_ms = Some(match mode {
                    DriverMode::Trace => session.first_arrival_timestamp_ms.unwrap_or(0.0),
                    DriverMode::Concurrency => 0.0,
                    DriverMode::AgenticTrace => {
                        unreachable!("agentic traces are constructed through new_agentic_trace")
                    }
                });
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
                            max_output_tokens: turn.max_output_tokens,
                            delay_after_previous_ms: turn.delay_after_previous_ms,
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

        Ok(Self {
            mode,
            prompt_mode,
            engine_block_size: engine_block_size_u32,
            sessions,
            in_flight: FxHashMap::default(),
            ready_sessions,
            max_in_flight: None,
            agentic_remaining_dependencies: Vec::new(),
            agentic_ready_after_ms: Vec::new(),
            agentic_dependents: FxHashMap::default(),
        })
    }

    /// Set a global in-flight cap. `pop_ready` will clamp by the remaining cap,
    /// and `next_ready_time_ms` returns `None` while at cap.
    pub fn set_max_in_flight(&mut self, cap: usize) {
        debug_assert!(
            self.in_flight.is_empty(),
            "set_max_in_flight called on a driver with pending work"
        );
        self.max_in_flight = Some(cap);
    }

    /// Failure-path companion: release a cap slot and terminate the owning session.
    /// No-op if `on_complete` already ran. Used when a request task is cancelled
    /// or panics before reaching `on_complete`.
    ///
    /// Terminating the session (marking it exhausted) prevents `run_workload` from
    /// deadlocking: `pop_ready` skips sessions with `in_flight.is_some()`, so a
    /// leaked session would leave `is_drained` stuck at `false` forever.
    pub fn release_cap_slot(&mut self, request_uuid: Uuid, now_ms: f64) {
        let Some(in_flight) = self.in_flight.remove(&request_uuid) else {
            return;
        };
        let Some(session) = self.sessions.get_mut(in_flight.session_index) else {
            return;
        };
        let request_id = session.turns[in_flight.turn_index].request_id.clone();
        if session.in_flight == Some(request_uuid) {
            session.in_flight = None;
            session.next_turn_index = session.turns.len();
            session.next_ready_at_ms = None;
        }
        if self.mode == DriverMode::AgenticTrace
            && let Some(request_id) = request_id
        {
            self.release_agentic_dependents(&request_id, now_ms);
        }
    }

    pub fn pop_ready(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyTurn> {
        let effective_limit = match self.max_in_flight {
            Some(cap) => limit.min(cap.saturating_sub(self.in_flight.len())),
            None => limit,
        };
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
            let arrival_timestamp_ms = match self.mode {
                DriverMode::Trace => Some(scheduled_ready_at_ms),
                DriverMode::Concurrency => None,
                DriverMode::AgenticTrace => Some(scheduled_ready_at_ms),
            };
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
                uuid: Some(request_uuid),
                dp_rank: 0,
                arrival_timestamp_ms,
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
                scheduled_ready_at_ms,
                replay_hashes: Some(replay_hashes),
                request,
            });
        }
        emitted
    }

    pub fn on_complete(&mut self, request_uuid: Uuid, now_ms: f64) -> Result<()> {
        let in_flight = self
            .in_flight
            .remove(&request_uuid)
            .ok_or_else(|| anyhow!("unknown workload request completion for {request_uuid}"))?;
        let session = self
            .sessions
            .get_mut(in_flight.session_index)
            .ok_or_else(|| anyhow!("unknown workload session {}", in_flight.session_index))?;
        if session.in_flight != Some(request_uuid) {
            bail!(
                "session {} completion for {} does not match in-flight request {:?}",
                session.session_id,
                request_uuid,
                session.in_flight
            );
        }

        session.in_flight = None;
        session.next_turn_index = in_flight.turn_index + 1;
        if self.mode != DriverMode::AgenticTrace && session.next_turn_index < session.turns.len() {
            let ready_at_ms =
                now_ms + session.turns[session.next_turn_index].delay_after_previous_ms;
            session.next_ready_at_ms = Some(ready_at_ms);
            self.ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index: in_flight.session_index,
                turn_index: session.next_turn_index,
            });
        } else {
            session.next_ready_at_ms = None;
        }
        if self.mode == DriverMode::AgenticTrace
            && let Some(request_id) = session.turns[in_flight.turn_index].request_id.clone()
        {
            self.release_agentic_dependents(&request_id, now_ms);
        }
        Ok(())
    }

    fn release_agentic_dependents(&mut self, request_id: &str, now_ms: f64) {
        let Some(dependent_sessions) = self.agentic_dependents.get(request_id).cloned() else {
            return;
        };
        for session_index in dependent_sessions {
            let Some(remaining) = self.agentic_remaining_dependencies.get_mut(session_index) else {
                continue;
            };
            if *remaining == 0 {
                continue;
            }
            *remaining -= 1;
            if let Some(ready_after_ms) = self.agentic_ready_after_ms.get_mut(session_index) {
                *ready_after_ms = ready_after_ms.max(now_ms);
            }
            if *remaining != 0 {
                continue;
            }

            let Some(session) = self.sessions.get_mut(session_index) else {
                continue;
            };
            if session.in_flight.is_some()
                || session.next_turn_index >= session.turns.len()
                || session.next_ready_at_ms.is_some()
            {
                continue;
            }
            let turn_index = session.next_turn_index;
            let ready_at_ms = self.agentic_ready_after_ms[session_index]
                + session.turns[turn_index].delay_after_previous_ms;
            session.next_ready_at_ms = Some(ready_at_ms);
            self.ready_sessions.push(ReadySession {
                ready_at_ms,
                session_index,
                turn_index,
            });
        }
    }

    pub fn next_ready_time_ms(&mut self) -> Option<f64> {
        if let Some(cap) = self.max_in_flight
            && self.in_flight.len() >= cap
        {
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
                        },
                        TurnTrace {
                            input_length: 2,
                            max_output_tokens: 1,
                            hash_ids: vec![3, 4],
                            delay_after_previous_ms: 5.0,
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
                    }],
                },
            ],
        }
    }

    #[test]
    fn cap_clamps_pop_ready_when_limit_is_unbounded() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();
        driver.set_max_in_flight(1);

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
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();
        driver.set_max_in_flight(1);

        let admitted = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(admitted.len(), 1);
        let uuid = admitted[0].request_uuid;
        driver.on_complete(uuid, 10.0).unwrap();

        let next = driver.pop_ready(10.0, usize::MAX);
        assert_eq!(next.len(), 1);
        assert_ne!(next[0].request_uuid, uuid);
    }

    #[test]
    fn next_ready_time_ms_returns_none_at_cap() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();
        driver.set_max_in_flight(1);

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
    fn no_cap_preserves_caller_limit_behavior() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();

        let admitted = driver.pop_ready(0.0, 5);
        assert_eq!(admitted.len(), 2, "both sessions should admit with no cap");
        assert!(driver.next_ready_time_ms().is_none());
    }

    #[test]
    fn release_cap_slot_is_noop_after_on_complete() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();
        driver.set_max_in_flight(1);

        let admitted = driver.pop_ready(0.0, usize::MAX);
        let uuid = admitted[0].request_uuid;
        driver.on_complete(uuid, 5.0).unwrap();

        driver.release_cap_slot(uuid, 5.0);

        let next = driver.pop_ready(5.0, usize::MAX);
        assert_eq!(next.len(), 1);
        assert_ne!(next[0].request_uuid, uuid);
    }

    #[test]
    fn release_cap_slot_recovers_cap_when_on_complete_was_skipped() {
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();
        driver.set_max_in_flight(1);

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
        let mut driver = WorkloadDriver::new_concurrency(two_session_trace(), 1).unwrap();
        driver.set_max_in_flight(1);

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
                        hash_ids: vec![10, 11],
                        delay_after_previous_ms: 0.0,
                    },
                    TurnTrace {
                        input_length: 3,
                        max_output_tokens: 1,
                        hash_ids: vec![12],
                        delay_after_previous_ms: 5.0,
                    },
                ],
            }],
        };
        let mut driver = WorkloadDriver::new_concurrency_accumulating_deltas(trace, 4).unwrap();

        let first = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].request.tokens, vec![10, 10, 10, 10, 11, 11]);
        driver.on_complete(first[0].request_uuid, 10.0).unwrap();

        let second = driver.pop_ready(15.0, usize::MAX);
        assert_eq!(second.len(), 1);
        assert_eq!(
            second[0].request.tokens,
            vec![10, 10, 10, 10, 11, 11, 12, 12, 12]
        );
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
