// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use super::core::{EngineEventBatch, EngineProgress};
use crate::common::handoff::HandoffId;
use crate::common::protocols::{ForwardPassSnapshot, OutputSignal};
use crate::scheduler::SchedulerLifecycleEvent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::replay::offline) enum SimulationWorkerStage {
    Aggregated,
    Prefill,
    Decode,
}

#[derive(Debug)]
pub(in crate::replay::offline) struct WorkerCompletionPayload<Events: EngineEventBatch = ()> {
    pub(in crate::replay::offline) stage: SimulationWorkerStage,
    pub(in crate::replay::offline) worker_idx: usize,
    pub(in crate::replay::offline) completed_requests: usize,
    pub(in crate::replay::offline) output_signals: Vec<OutputSignal>,
    pub(in crate::replay::offline) lifecycle_events: Vec<SchedulerLifecycleEvent>,
    pub(in crate::replay::offline) engine_events: Events,
    pub(in crate::replay::offline) progress: EngineProgress,
    pub(in crate::replay::offline) fpm: Option<ForwardPassSnapshot>,
    pub(in crate::replay::offline) accept_length_output_tokens: usize,
    pub(in crate::replay::offline) accept_length_decode_forwards: usize,
}

#[derive(Debug)]
pub(in crate::replay::offline) enum SimulationEventKind<Events: EngineEventBatch = ()> {
    WorkerCompletion {
        stage: SimulationWorkerStage,
        worker_idx: usize,
        completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        lifecycle_events: Vec<SchedulerLifecycleEvent>,
        engine_events: Events,
        made_progress: bool,
        had_raw_observations: bool,
        fpm: Option<Box<crate::common::protocols::ForwardPassSnapshot>>,
        accept_length_output_tokens: usize,
        accept_length_decode_forwards: usize,
    },
    WorkerCompletionBatch {
        payloads: Box<[WorkerCompletionPayload<Events>]>,
    },
    TransferComplete {
        handoff_id: HandoffId,
    },
    WorkerReady {
        stage: SimulationWorkerStage,
        worker_id: usize,
    },
    /// A recurring scaling heartbeat. Payload-free: the scaling snapshot is
    /// gathered from live runtime state when the tick fires. Re-enqueues itself
    /// at the time the scaling policy returns.
    ScalingTick,
}

impl<Events: EngineEventBatch> SimulationEventKind<Events> {
    /// Tie-breaker among events at the *same* `at_ms`. Every engine completion
    /// is exposed before controller/topology events, and scaling observes the
    /// fully settled timestamp. Sequence numbers remain the stable tie-breaker
    /// within each causal phase.
    fn ordering_rank(&self) -> u8 {
        match self {
            SimulationEventKind::WorkerCompletion { .. }
            | SimulationEventKind::WorkerCompletionBatch { .. } => 0,
            SimulationEventKind::TransferComplete { .. }
            | SimulationEventKind::WorkerReady { .. } => 1,
            SimulationEventKind::ScalingTick => 2,
        }
    }
}

#[derive(Debug)]
pub(in crate::replay::offline) struct SimulationEvent<Events: EngineEventBatch = ()> {
    pub(in crate::replay::offline) at_ms: f64,
    pub(in crate::replay::offline) seq_no: u64,
    pub(in crate::replay::offline) kind: SimulationEventKind<Events>,
}

impl<Events: EngineEventBatch> PartialEq for SimulationEvent<Events> {
    fn eq(&self, other: &Self) -> bool {
        self.at_ms.to_bits() == other.at_ms.to_bits() && self.seq_no == other.seq_no
    }
}

impl<Events: EngineEventBatch> Eq for SimulationEvent<Events> {}

impl<Events: EngineEventBatch> PartialOrd for SimulationEvent<Events> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Events: EngineEventBatch> Ord for SimulationEvent<Events> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at_ms
            .partial_cmp(&self.at_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.kind.ordering_rank().cmp(&self.kind.ordering_rank()))
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}
