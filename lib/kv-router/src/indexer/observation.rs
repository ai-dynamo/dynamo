// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use tokio::sync::oneshot;

use crate::protocols::WorkerId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventCompletion {
    pub correlation_id: u32,
    pub finished_ns: u64,
    pub success: bool,
}

#[derive(Debug)]
pub struct EventCompletionWriter {
    epoch: Instant,
    records: Box<[EventCompletion]>,
    written: usize,
    overflow: bool,
}

impl EventCompletionWriter {
    pub fn new(epoch: Instant, capacity: usize) -> Self {
        Self {
            epoch,
            records: vec![EventCompletion::default(); capacity].into_boxed_slice(),
            written: 0,
            overflow: false,
        }
    }

    fn timestamp_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    fn record(&mut self, correlation_id: u32, success: bool) {
        let finished_ns = self.timestamp_ns();
        let Some(slot) = self.records.get_mut(self.written) else {
            self.overflow = true;
            return;
        };
        *slot = EventCompletion {
            correlation_id,
            finished_ns,
            success,
        };
        self.written += 1;
    }

    fn seal(&self) -> ObservationSeal {
        ObservationSeal {
            sealed_ns: self.timestamp_ns(),
            written: self.written,
            overflow: self.overflow,
        }
    }

    fn into_buffer(self) -> EventCompletionBuffer {
        EventCompletionBuffer {
            records: self.records,
            written: self.written,
            overflow: self.overflow,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObservationSeal {
    pub sealed_ns: u64,
    pub written: usize,
    pub overflow: bool,
}

#[derive(Debug)]
pub struct EventCompletionBuffer {
    records: Box<[EventCompletion]>,
    written: usize,
    overflow: bool,
}

impl EventCompletionBuffer {
    pub fn records(&self) -> &[EventCompletion] {
        &self.records[..self.written]
    }

    pub fn capacity(&self) -> usize {
        self.records.len()
    }

    pub fn overflowed(&self) -> bool {
        self.overflow
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ObservationError {
    #[error("an observation session is already active")]
    AlreadyActive,
    #[error("observation plan contains no workers")]
    EmptyPlan,
    #[error("observation plan contains duplicate worker {0}")]
    DuplicateWorker(u64),
    #[error("observation capacity overflow for event worker {0}")]
    CapacityOverflow(usize),
    #[error("worker {0} was not pre-registered")]
    UnknownWorker(u64),
    #[error("observed producer is closed")]
    ProducerClosed,
    #[error("event worker {0} is offline")]
    WorkerOffline(usize),
    #[error("event worker {0} rejected observation setup")]
    InstallRejected(usize),
    #[error("event worker {0} did not acknowledge observation setup")]
    InstallCanceled(usize),
    #[error("event worker {0} did not acknowledge the seal barrier")]
    SealCanceled(usize),
    #[error("event worker {0} rejected the seal barrier")]
    SealRejected(usize),
    #[error("event worker {0} seal task was already consumed")]
    SealTaskConsumed(usize),
    #[error("event worker {0} did not return its completion buffer")]
    HarvestCanceled(usize),
    #[error("event worker {0} harvest task was already consumed")]
    HarvestTaskConsumed(usize),
}

#[derive(Debug)]
pub struct ThreadPoolObservationPlan {
    pub epoch: Instant,
    pub expected_events_by_worker: Vec<(WorkerId, usize)>,
}

#[derive(Debug)]
pub struct ObservedEnqueueReceipt {
    pub accepted_ns: u64,
    pub event_worker: usize,
}

#[derive(Debug)]
pub struct ThreadPoolObservationSnapshot {
    pub seals: Vec<ObservationSeal>,
    pub buffers: Vec<EventCompletionBuffer>,
    pub queue_depth_at_stop: Vec<usize>,
}

pub struct WorkerObservationState {
    phase: ObservationPhase,
    writer: Option<EventCompletionWriter>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ObservationPhase {
    #[default]
    Idle,
    Recording,
    Sealed,
}

impl Default for WorkerObservationState {
    fn default() -> Self {
        Self {
            phase: ObservationPhase::Idle,
            writer: None,
        }
    }
}

impl WorkerObservationState {
    pub fn install(&mut self, writer: EventCompletionWriter, resp: oneshot::Sender<bool>) {
        let accepted = self.phase == ObservationPhase::Idle;
        if accepted {
            self.writer = Some(writer);
            self.phase = ObservationPhase::Recording;
        }
        let _ = resp.send(accepted);
    }

    pub fn record(&mut self, correlation_id: u32, success: bool) {
        if self.phase != ObservationPhase::Recording {
            return;
        }
        if let Some(writer) = self.writer.as_mut() {
            writer.record(correlation_id, success);
        }
    }

    pub fn seal(&mut self, resp: oneshot::Sender<Option<ObservationSeal>>) {
        let seal = if self.phase == ObservationPhase::Recording {
            self.phase = ObservationPhase::Sealed;
            self.writer.as_ref().map(EventCompletionWriter::seal)
        } else {
            None
        };
        let _ = resp.send(seal);
    }

    pub fn harvest(&mut self, resp: oneshot::Sender<EventCompletionBuffer>) {
        if self.phase != ObservationPhase::Sealed {
            return;
        }
        let Some(writer) = self.writer.take() else {
            return;
        };
        self.phase = ObservationPhase::Idle;
        let _ = resp.send(writer.into_buffer());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_writer_latches_overflow_without_growing() {
        let mut writer = EventCompletionWriter::new(Instant::now(), 1);
        writer.record(7, true);
        writer.record(8, false);

        let buffer = writer.into_buffer();
        assert_eq!(buffer.capacity(), 1);
        assert!(buffer.overflowed());
        assert_eq!(buffer.records().len(), 1);
        assert_eq!(buffer.records()[0].correlation_id, 7);
    }

    #[tokio::test]
    async fn observation_state_requires_seal_before_harvest() {
        let mut state = WorkerObservationState::default();
        let (install_tx, install_rx) = oneshot::channel();
        state.install(EventCompletionWriter::new(Instant::now(), 1), install_tx);
        assert!(install_rx.await.unwrap());

        state.record(9, true);
        let (seal_tx, seal_rx) = oneshot::channel();
        state.seal(seal_tx);
        let seal = seal_rx.await.unwrap().unwrap();
        assert_eq!(seal.written, 1);

        let (harvest_tx, harvest_rx) = oneshot::channel();
        state.harvest(harvest_tx);
        let buffer = harvest_rx.await.unwrap();
        assert_eq!(buffer.records()[0].correlation_id, 9);
    }
}
