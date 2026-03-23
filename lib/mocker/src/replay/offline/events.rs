// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;

use crate::common::protocols::OutputSignal;

#[derive(Debug)]
pub(crate) enum SimulationEventKind {
    WorkerCompletion {
        worker_idx: usize,
        completed_requests: usize,
        output_signals: Vec<OutputSignal>,
        kv_events: Vec<dynamo_kv_router::protocols::RouterEvent>,
    },
}

#[derive(Debug)]
pub(crate) struct SimulationEvent {
    pub(crate) at_ms: f64,
    pub(crate) seq_no: u64,
    pub(crate) kind: SimulationEventKind,
}

impl SimulationEvent {
    fn kind_priority(&self) -> u8 {
        0
    }
}

impl PartialEq for SimulationEvent {
    fn eq(&self, other: &Self) -> bool {
        self.at_ms.to_bits() == other.at_ms.to_bits()
            && self.seq_no == other.seq_no
            && self.kind_priority() == other.kind_priority()
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
            .then_with(|| self.kind_priority().cmp(&other.kind_priority()))
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}
