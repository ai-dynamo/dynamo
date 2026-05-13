// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::{Result, anyhow, bail};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::types::{ReadyTurn, ReplayRequestHashes, Trace};
use crate::common::protocols::DirectRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverMode {
    Trace,
    Concurrency,
}

#[derive(Debug)]
struct SessionRuntime {
    session_id: String,
    turns: Vec<TurnRuntime>,
    next_turn_index: usize,
    next_ready_at_ms: Option<f64>,
    in_flight: Option<Uuid>,
}

#[derive(Debug)]
struct TurnRuntime {
    tokens: Vec<u32>,
    max_output_tokens: usize,
    delay_after_previous_ms: f64,
    replay_hashes: ReplayRequestHashes,
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
    sessions: Vec<SessionRuntime>,
    in_flight: FxHashMap<Uuid, InFlightTurn>,
    ready_sessions: BinaryHeap<ReadySession>,
}

impl WorkloadDriver {
    pub(crate) fn new_trace(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(trace, engine_block_size, DriverMode::Trace)
    }

    pub(crate) fn new_concurrency(trace: Trace, engine_block_size: usize) -> Result<Self> {
        Self::new(trace, engine_block_size, DriverMode::Concurrency)
    }

    fn new(trace: Trace, engine_block_size: usize, mode: DriverMode) -> Result<Self> {
        let trace_block_size = trace.block_size;
        let sessions: Vec<SessionRuntime> = trace
            .sessions
            .into_iter()
            .map(|session| -> Result<SessionRuntime> {
                let next_ready_at_ms = Some(match mode {
                    DriverMode::Trace => session.first_arrival_timestamp_ms.unwrap_or(0.0),
                    DriverMode::Concurrency => 0.0,
                });
                let turns = session
                    .turns
                    .into_iter()
                    .map(|turn| -> Result<TurnRuntime> {
                        Ok(TurnRuntime {
                            tokens: turn.synthesize_tokens(trace_block_size)?,
                            max_output_tokens: turn.max_output_tokens,
                            delay_after_previous_ms: turn.delay_after_previous_ms,
                            replay_hashes: turn
                                .to_replay_hashes(trace_block_size, engine_block_size)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(SessionRuntime {
                    session_id: session.session_id,
                    turns,
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
            sessions,
            in_flight: FxHashMap::default(),
            ready_sessions,
        })
    }

    pub fn pop_ready(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyTurn> {
        if limit == 0 {
            return Vec::new();
        }

        let mut emitted = Vec::new();
        while emitted.len() < limit {
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
            };
            let request = DirectRequest {
                tokens: turn.tokens.clone(),
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
                replay_hashes: Some(turn.replay_hashes.clone()),
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
        if session.next_turn_index < session.turns.len() {
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
        Ok(())
    }

    pub fn next_ready_time_ms(&mut self) -> Option<f64> {
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
