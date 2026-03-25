// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, VecDeque};

use dynamo_kv_router::protocols::RouterEvent;

use super::events::{SimulationEvent, SimulationEventKind, SimulationWorkerStage};
use crate::common::protocols::{DirectRequest, OutputSignal};

#[derive(Debug)]
pub(super) struct WorkerCompletionPayload {
    pub stage: SimulationWorkerStage,
    pub worker_idx: usize,
    pub completed_requests: usize,
    pub output_signals: Vec<OutputSignal>,
    pub kv_events: Vec<RouterEvent>,
}

pub(super) fn next_timestamp(
    next_arrival_ms: Option<f64>,
    next_event_ms: Option<f64>,
) -> Option<f64> {
    match (next_arrival_ms, next_event_ms) {
        (Some(arrival_ms), Some(event_ms)) => Some(arrival_ms.min(event_ms)),
        (Some(arrival_ms), None) => Some(arrival_ms),
        (None, Some(event_ms)) => Some(event_ms),
        (None, None) => None,
    }
}

pub(super) fn pop_next_trace_ready(
    pending: &mut VecDeque<DirectRequest>,
    now_ms: f64,
) -> Option<(DirectRequest, f64)> {
    let arrival_ms = pending
        .front()
        .and_then(|request| request.arrival_timestamp_ms)
        .filter(|arrival_ms| *arrival_ms <= now_ms)?;
    let request = pending
        .pop_front()
        .expect("front request must exist when arrival is ready");
    Some((request, arrival_ms))
}

pub(super) fn pop_next_concurrency_ready(
    pending: &mut VecDeque<DirectRequest>,
    now_ms: f64,
    cluster_in_flight: usize,
    max_in_flight: usize,
) -> Option<(DirectRequest, f64)> {
    if cluster_in_flight >= max_in_flight {
        return None;
    }
    let request = pending.pop_front()?;
    Some((request, now_ms))
}

pub(super) fn push_worker_completion(
    events: &mut BinaryHeap<SimulationEvent>,
    next_event_seq: &mut u64,
    at_ms: f64,
    payload: WorkerCompletionPayload,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::WorkerCompletion {
            stage: payload.stage,
            worker_idx: payload.worker_idx,
            completed_requests: payload.completed_requests,
            output_signals: payload.output_signals,
            kv_events: payload.kv_events,
        },
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_worker_completion(
    events: &mut BinaryHeap<SimulationEvent>,
    now_ms: f64,
) -> Option<WorkerCompletionPayload> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    let SimulationEventKind::WorkerCompletion { .. } = &event.kind else {
        return None;
    };
    let event = events.pop().expect("event must exist after peek");
    let (stage, worker_idx, completed_requests, output_signals, kv_events) = match event.kind {
        SimulationEventKind::WorkerCompletion {
            stage,
            worker_idx,
            completed_requests,
            output_signals,
            kv_events,
        } => (
            stage,
            worker_idx,
            completed_requests,
            output_signals,
            kv_events,
        ),
        SimulationEventKind::DecodeHandoff { .. } => {
            unreachable!("peeked worker completion event must match popped event")
        }
    };
    Some(WorkerCompletionPayload {
        stage,
        worker_idx,
        completed_requests,
        output_signals,
        kv_events,
    })
}

pub(super) fn push_decode_handoff(
    events: &mut BinaryHeap<SimulationEvent>,
    next_event_seq: &mut u64,
    at_ms: f64,
    uuid: uuid::Uuid,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::DecodeHandoff { uuid },
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_decode_handoff(
    events: &mut BinaryHeap<SimulationEvent>,
    now_ms: f64,
) -> Option<uuid::Uuid> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    let SimulationEventKind::DecodeHandoff { .. } = &event.kind else {
        return None;
    };
    let event = events.pop().expect("event must exist after peek");
    let SimulationEventKind::DecodeHandoff { uuid } = event.kind else {
        unreachable!("peeked decode handoff event must match popped event");
    };
    Some(uuid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::offline::events::SimulationWorkerStage;
    use uuid::Uuid;

    fn direct_request(uuid: u128, arrival_timestamp_ms: Option<f64>) -> DirectRequest {
        DirectRequest {
            tokens: vec![1; 8],
            max_output_tokens: 1,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms,
        }
    }

    #[test]
    fn test_next_timestamp_matches_current_choice_logic() {
        assert_eq!(next_timestamp(Some(1.0), Some(2.0)), Some(1.0));
        assert_eq!(next_timestamp(Some(2.0), Some(1.0)), Some(1.0));
        assert_eq!(next_timestamp(Some(3.0), None), Some(3.0));
        assert_eq!(next_timestamp(None, Some(4.0)), Some(4.0));
        assert_eq!(next_timestamp(None, None), None);
    }

    #[test]
    fn test_pop_next_trace_ready_releases_only_arrivals_at_or_before_now() {
        let mut pending = VecDeque::from(vec![
            direct_request(1, Some(1.0)),
            direct_request(2, Some(1.1)),
            direct_request(3, Some(2.0)),
        ]);

        let (request_1, arrival_1) = pop_next_trace_ready(&mut pending, 1.0).unwrap();
        assert_eq!(request_1.uuid, Some(Uuid::from_u128(1)));
        assert_eq!(arrival_1, 1.0);

        assert!(pop_next_trace_ready(&mut pending, 1.0).is_none());

        let (request_2, arrival_2) = pop_next_trace_ready(&mut pending, 1.1).unwrap();
        assert_eq!(request_2.uuid, Some(Uuid::from_u128(2)));
        assert_eq!(arrival_2, 1.1);
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_pop_next_concurrency_ready_stops_at_max_in_flight() {
        let mut pending = VecDeque::from(vec![direct_request(1, None), direct_request(2, None)]);

        assert!(pop_next_concurrency_ready(&mut pending, 5.0, 2, 2).is_none());

        let (request, arrival_ms) = pop_next_concurrency_ready(&mut pending, 5.0, 1, 2).unwrap();
        assert_eq!(request.uuid, Some(Uuid::from_u128(1)));
        assert_eq!(arrival_ms, 5.0);
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_worker_completion_helpers_preserve_same_time_sequence_ordering() {
        let mut events = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_completion(
            &mut events,
            &mut next_event_seq,
            10.0,
            WorkerCompletionPayload {
                stage: SimulationWorkerStage::Aggregated,
                worker_idx: 7,
                completed_requests: 1,
                output_signals: vec![OutputSignal {
                    uuid: Uuid::from_u128(7),
                    completed: true,
                    handoff_delay_ms: None,
                }],
                kv_events: Vec::new(),
            },
        );
        push_worker_completion(
            &mut events,
            &mut next_event_seq,
            10.0,
            WorkerCompletionPayload {
                stage: SimulationWorkerStage::Aggregated,
                worker_idx: 8,
                completed_requests: 2,
                output_signals: vec![OutputSignal {
                    uuid: Uuid::from_u128(8),
                    completed: false,
                    handoff_delay_ms: None,
                }],
                kv_events: Vec::new(),
            },
        );

        assert!(pop_ready_worker_completion(&mut events, 9.0).is_none());

        let first = pop_ready_worker_completion(&mut events, 10.0).unwrap();
        let second = pop_ready_worker_completion(&mut events, 10.0).unwrap();
        assert_eq!(first.stage, SimulationWorkerStage::Aggregated);
        assert_eq!(first.worker_idx, 7);
        assert_eq!(first.completed_requests, 1);
        assert_eq!(second.stage, SimulationWorkerStage::Aggregated);
        assert_eq!(second.worker_idx, 8);
        assert_eq!(second.completed_requests, 2);
        assert!(events.is_empty());
    }
}
