// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use crate::common::protocols::OutputSignal;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimulationWorkerStage {
    Aggregated,
    Prefill,
    Decode,
}

#[derive(Debug)]
pub(crate) enum SimulationEventKind {
    WorkerCompletion {
        stage: SimulationWorkerStage,
        worker_idx: usize,
        completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        kv_events: Vec<dynamo_kv_router::protocols::RouterEvent>,
    },
    DecodeHandoff {
        uuid: Uuid,
    },
    WorkerReady {
        stage: SimulationWorkerStage,
        worker_id: usize,
    },
}

#[derive(Debug)]
pub(crate) struct SimulationEvent {
    pub(crate) at_ms: f64,
    pub(crate) seq_no: u64,
    pub(crate) kind: SimulationEventKind,
}

impl PartialEq for SimulationEvent {
    fn eq(&self, other: &Self) -> bool {
        self.at_ms.to_bits() == other.at_ms.to_bits() && self.seq_no == other.seq_no
    }
}

impl Eq for SimulationEvent {}

impl PartialOrd for SimulationEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SimulationEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at_ms
            .partial_cmp(&self.at_ms)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}
