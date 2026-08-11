// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
#[cfg(test)]
use std::collections::VecDeque;

use super::components::ScheduledWorkerCompletions;
use super::core::{EngineEventBatch, EngineProgress};
use super::events::{
    SimulationEvent, SimulationEventKind, SimulationWorkerStage, WorkerCompletionPayload,
};
use crate::common::handoff::HandoffId;
#[cfg(test)]
use crate::common::protocols::DirectRequest;
#[cfg(test)]
use crate::common::protocols::OutputSignal;

// Keep the large singleton inline: boxing it would add an allocation to every
// DP1 and disaggregated completion solely to shrink this transient pop result.
#[allow(clippy::large_enum_variant)]
pub(super) enum ReadyWorkerCompletions<Events: EngineEventBatch = ()> {
    Single(WorkerCompletionPayload<Events>),
    Batch(Box<[WorkerCompletionPayload<Events>]>),
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

#[cfg(test)]
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

#[cfg(test)]
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

pub(super) fn push_worker_completions<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    scheduled: ScheduledWorkerCompletions<Events>,
) {
    let ScheduledWorkerCompletions {
        at_ms,
        mut payloads,
    } = scheduled;
    let payload_count =
        u64::try_from(payloads.len()).expect("completion payload count must fit in u64");
    assert!(
        payload_count > 0,
        "scheduled completion batch must not be empty"
    );
    let kind = if payloads.len() == 1 {
        let payload = payloads
            .pop()
            .expect("singleton scheduled completion must contain one payload");
        SimulationEventKind::WorkerCompletion {
            stage: payload.stage,
            worker_idx: payload.worker_idx,
            completed_requests: payload.completed_requests,
            output_signals: payload.output_signals,
            lifecycle_events: payload.lifecycle_events,
            engine_events: payload.engine_events,
            made_progress: payload.progress.made_progress,
            had_raw_observations: payload.progress.had_raw_observations,
            fpm: payload.fpm.map(Box::new),
            accept_length_output_tokens: payload.accept_length_output_tokens,
            accept_length_decode_forwards: payload.accept_length_decode_forwards,
        }
    } else {
        SimulationEventKind::WorkerCompletionBatch {
            payloads: payloads.into_boxed_slice(),
        }
    };
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind,
    });
    // Preserve the sequence numbers that later events would have received
    // before these payloads were represented by one heap entry.
    *next_event_seq = next_event_seq
        .checked_add(payload_count)
        .expect("offline replay event sequence overflow");
}

pub(super) fn pop_ready_worker_completions<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> Option<ReadyWorkerCompletions<Events>> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    if !matches!(
        event.kind,
        SimulationEventKind::WorkerCompletion { .. }
            | SimulationEventKind::WorkerCompletionBatch { .. }
    ) {
        return None;
    }
    let event = events.pop().expect("event must exist after peek");
    match event.kind {
        SimulationEventKind::WorkerCompletion {
            stage,
            worker_idx,
            completed_requests,
            output_signals,
            lifecycle_events,
            engine_events,
            made_progress,
            had_raw_observations,
            fpm,
            accept_length_output_tokens,
            accept_length_decode_forwards,
        } => Some(ReadyWorkerCompletions::Single(WorkerCompletionPayload {
            stage,
            worker_idx,
            completed_requests,
            output_signals,
            lifecycle_events,
            engine_events,
            progress: EngineProgress {
                made_progress,
                had_raw_observations,
            },
            fpm: fpm.map(|fpm| *fpm),
            accept_length_output_tokens,
            accept_length_decode_forwards,
        })),
        SimulationEventKind::WorkerCompletionBatch { payloads } => {
            Some(ReadyWorkerCompletions::Batch(payloads))
        }
        SimulationEventKind::TransferComplete { .. }
        | SimulationEventKind::WorkerReady { .. }
        | SimulationEventKind::ScalingTick => {
            unreachable!("peeked worker completion event must match popped event")
        }
    }
}

pub(super) fn push_transfer_complete<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
    handoff_id: HandoffId,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::TransferComplete { handoff_id },
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_transfer_complete<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> Option<HandoffId> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    let SimulationEventKind::TransferComplete { .. } = &event.kind else {
        return None;
    };
    let event = events.pop().expect("event must exist after peek");
    let SimulationEventKind::TransferComplete { handoff_id } = event.kind else {
        unreachable!("peeked decode handoff event must match popped event");
    };
    Some(handoff_id)
}

pub(super) fn push_worker_ready<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
    stage: SimulationWorkerStage,
    worker_id: usize,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::WorkerReady { stage, worker_id },
    });
    *next_event_seq += 1;
}

pub(super) fn pop_ready_worker_ready<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> Option<(SimulationWorkerStage, usize)> {
    let event = events.peek()?;
    if event.at_ms != now_ms {
        return None;
    }
    let SimulationEventKind::WorkerReady { .. } = &event.kind else {
        return None;
    };
    let event = events.pop().expect("event must exist after peek");
    let SimulationEventKind::WorkerReady { stage, worker_id } = event.kind else {
        unreachable!("peeked worker ready event must match popped event");
    };
    Some((stage, worker_id))
}

pub(super) fn push_scaling_tick<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    next_event_seq: &mut u64,
    at_ms: f64,
) {
    events.push(SimulationEvent {
        at_ms,
        seq_no: *next_event_seq,
        kind: SimulationEventKind::ScalingTick,
    });
    *next_event_seq += 1;
}

/// Pop a `ScalingTick` scheduled for exactly `now_ms` (peek-and-pop-at-now, like the
/// other `pop_ready_*` helpers). Payload-free, so it returns whether one fired.
pub(super) fn pop_ready_scaling_tick<Events: EngineEventBatch>(
    events: &mut BinaryHeap<SimulationEvent<Events>>,
    now_ms: f64,
) -> bool {
    let Some(event) = events.peek() else {
        return false;
    };
    if event.at_ms != now_ms {
        return false;
    }
    if !matches!(event.kind, SimulationEventKind::ScalingTick) {
        return false;
    }
    events.pop().expect("event must exist after peek");
    true
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
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(uuid)),
            dp_rank: 0,
            arrival_timestamp_ms,
            ..Default::default()
        }
    }

    fn completion_payload(worker_idx: usize, completed_requests: usize) -> WorkerCompletionPayload {
        WorkerCompletionPayload {
            stage: SimulationWorkerStage::Aggregated,
            worker_idx,
            completed_requests,
            output_signals: vec![OutputSignal {
                uuid: Uuid::from_u128(worker_idx as u128),
                token_id: None,
                completed: true,
                rejected: false,
                cached_tokens: None,
                handoff_delay_ms: None,
            }],
            lifecycle_events: Vec::new(),
            engine_events: (),
            progress: EngineProgress::default(),
            fpm: None,
            accept_length_output_tokens: 1,
            accept_length_decode_forwards: 1,
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
    fn test_worker_completion_batch_preserves_payload_and_event_ordering() {
        let mut events = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_completions(
            &mut events,
            &mut next_event_seq,
            ScheduledWorkerCompletions {
                at_ms: 10.0,
                payloads: vec![completion_payload(7, 1), completion_payload(8, 2)],
            },
        );
        assert_eq!(events.len(), 1);
        assert_eq!(next_event_seq, 2);

        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            10.0,
            SimulationWorkerStage::Aggregated,
            5,
        );
        assert_eq!(next_event_seq, 3);
        push_scaling_tick(&mut events, &mut next_event_seq, 10.0);
        assert_eq!(next_event_seq, 4);

        assert!(pop_ready_worker_completions(&mut events, 9.0).is_none());
        let ReadyWorkerCompletions::Batch(payloads) =
            pop_ready_worker_completions(&mut events, 10.0).unwrap()
        else {
            panic!("DP2 completions must use one batched heap event");
        };
        assert_eq!(
            payloads
                .iter()
                .map(|payload| (payload.worker_idx, payload.completed_requests))
                .collect::<Vec<_>>(),
            vec![(7, 1), (8, 2)]
        );
        assert_eq!(
            pop_ready_worker_ready(&mut events, 10.0),
            Some((SimulationWorkerStage::Aggregated, 5))
        );
        assert!(pop_ready_scaling_tick(&mut events, 10.0));
        assert!(events.is_empty());
    }

    #[test]
    fn test_worker_ready_push_pop_round_trip() {
        let mut events: BinaryHeap<SimulationEvent<()>> = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            100.0,
            SimulationWorkerStage::Aggregated,
            3,
        );

        // Not ready before the scheduled time.
        assert!(pop_ready_worker_ready(&mut events, 99.0).is_none());

        let (stage, worker_id) = pop_ready_worker_ready(&mut events, 100.0).unwrap();
        assert_eq!(stage, SimulationWorkerStage::Aggregated);
        assert_eq!(worker_id, 3);
        assert!(events.is_empty());
    }

    #[test]
    fn test_worker_ready_does_not_interfere_with_completion_pop() {
        let mut events: BinaryHeap<SimulationEvent<()>> = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            10.0,
            SimulationWorkerStage::Aggregated,
            1,
        );

        // pop_ready_worker_completions must return None (wrong event kind).
        assert!(pop_ready_worker_completions(&mut events, 10.0).is_none());
        // The event should still be in the heap.
        assert_eq!(events.len(), 1);
        // pop_ready_worker_ready should succeed.
        assert!(pop_ready_worker_ready(&mut events, 10.0).is_some());
    }

    #[test]
    fn test_worker_ready_interleaved_with_completion() {
        let mut events = BinaryHeap::new();
        let mut next_event_seq = 0;

        push_worker_completions(
            &mut events,
            &mut next_event_seq,
            ScheduledWorkerCompletions {
                at_ms: 10.0,
                payloads: vec![completion_payload(0, 1)],
            },
        );
        push_worker_ready(
            &mut events,
            &mut next_event_seq,
            10.0,
            SimulationWorkerStage::Aggregated,
            5,
        );

        // The completion was pushed first (lower seq_no) so it pops first.
        let ReadyWorkerCompletions::Single(completion) =
            pop_ready_worker_completions(&mut events, 10.0).unwrap()
        else {
            panic!("singleton completion must retain its existing representation");
        };
        assert_eq!(completion.worker_idx, 0);

        // Now the ready event is at the front.
        assert!(pop_ready_worker_completions(&mut events, 10.0).is_none());
        let (stage, worker_id) = pop_ready_worker_ready(&mut events, 10.0).unwrap();
        assert_eq!(stage, SimulationWorkerStage::Aggregated);
        assert_eq!(worker_id, 5);
        assert!(events.is_empty());
    }
}
