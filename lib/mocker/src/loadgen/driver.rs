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
    max_in_flight: Option<usize>,
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
            max_in_flight: None,
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
    pub fn release_cap_slot(&mut self, request_uuid: Uuid) {
        let Some(in_flight) = self.in_flight.remove(&request_uuid) else {
            return;
        };
        let Some(session) = self.sessions.get_mut(in_flight.session_index) else {
            return;
        };
        if session.in_flight == Some(request_uuid) {
            session.in_flight = None;
            session.next_turn_index = session.turns.len();
            session.next_ready_at_ms = None;
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
    use crate::loadgen::{SessionTrace, Trace, TurnTrace};

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

        driver.release_cap_slot(uuid);

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

        driver.release_cap_slot(admitted[0].request_uuid);

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

        driver.release_cap_slot(stuck_uuid);

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
}
