// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use uuid::Uuid;

use super::types::{ReadyTurn, Trace, TurnTrace};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriverMode {
    Trace,
    Concurrency,
}

#[derive(Debug)]
struct SessionRuntime {
    session_id: String,
    session_index: usize,
    turns: Vec<TurnTrace>,
    next_turn_index: usize,
    next_ready_at_ms: Option<f64>,
    in_flight: Option<Uuid>,
}

#[derive(Debug)]
struct InFlightTurn {
    session_index: usize,
    turn_index: usize,
}

#[derive(Debug)]
pub struct WorkloadDriver {
    mode: DriverMode,
    block_size: usize,
    sessions: Vec<SessionRuntime>,
    in_flight: HashMap<Uuid, InFlightTurn>,
}

impl WorkloadDriver {
    pub(crate) fn new_trace(trace: Trace) -> Result<Self> {
        Self::new(trace, DriverMode::Trace)
    }

    pub(crate) fn new_concurrency(trace: Trace) -> Result<Self> {
        Self::new(trace, DriverMode::Concurrency)
    }

    fn new(trace: Trace, mode: DriverMode) -> Result<Self> {
        let sessions = trace
            .sessions
            .into_iter()
            .enumerate()
            .map(|(session_index, session)| SessionRuntime {
                session_id: session.session_id,
                session_index,
                turns: session.turns,
                next_turn_index: 0,
                next_ready_at_ms: Some(match mode {
                    DriverMode::Trace => session.first_arrival_timestamp_ms.unwrap_or(0.0),
                    DriverMode::Concurrency => 0.0,
                }),
                in_flight: None,
            })
            .collect();

        Ok(Self {
            mode,
            block_size: trace.block_size,
            sessions,
            in_flight: HashMap::new(),
        })
    }

    pub fn pop_ready(&mut self, now_ms: f64, limit: usize) -> Vec<ReadyTurn> {
        if limit == 0 {
            return Vec::new();
        }

        let mut ready_sessions = self
            .sessions
            .iter()
            .filter_map(|session| {
                let ready_at_ms = session.next_ready_at_ms?;
                if session.in_flight.is_some() || ready_at_ms > now_ms {
                    return None;
                }
                Some((ready_at_ms, session.session_index))
            })
            .collect::<Vec<_>>();
        ready_sessions.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });

        let mut emitted = Vec::new();
        for (_, session_index) in ready_sessions.into_iter().take(limit) {
            let session = &mut self.sessions[session_index];
            let turn_index = session.next_turn_index;
            let scheduled_ready_at_ms = session
                .next_ready_at_ms
                .expect("ready session must have a timestamp");
            let request_uuid = Uuid::new_v4();
            let arrival_timestamp_ms = match self.mode {
                DriverMode::Trace => Some(scheduled_ready_at_ms),
                DriverMode::Concurrency => None,
            };
            let request = session.turns[turn_index]
                .to_direct_request(self.block_size, request_uuid, arrival_timestamp_ms)
                .expect("validated trace should always synthesize into a direct request");
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
            session.next_ready_at_ms =
                Some(now_ms + session.turns[session.next_turn_index].delay_after_previous_ms);
        } else {
            session.next_ready_at_ms = None;
        }
        Ok(())
    }

    pub fn next_ready_time_ms(&self) -> Option<f64> {
        self.sessions
            .iter()
            .filter_map(|session| session.next_ready_at_ms)
            .min_by(|left, right| left.total_cmp(right))
    }

    pub fn is_drained(&self) -> bool {
        self.in_flight.is_empty()
            && self
                .sessions
                .iter()
                .all(|session| session.next_turn_index >= session.turns.len())
    }
}
