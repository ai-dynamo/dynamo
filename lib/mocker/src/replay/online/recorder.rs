// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, anyhow, bail};
use tokio::sync::mpsc;
use tokio::time::Instant;
use uuid::Uuid;

use crate::live::ObservedAdmission;
use crate::replay::{ReplayTerminalStatus, SlaThresholds, TraceCollector, TraceSimulationReport};
use crate::scheduler::AdmissionEvent;

use super::state::ArrivalEvent;

pub(super) struct TerminalObservation {
    pub(super) uuid: Uuid,
    pub(super) token_times_ms: Vec<f64>,
    pub(super) terminal_time_ms: f64,
    pub(super) status: ReplayTerminalStatus,
}

#[derive(Clone, Copy, Default)]
pub(super) struct OnlineRecorderOptions {
    pub(super) capture_per_request: bool,
    pub(super) sla: SlaThresholds,
    pub(super) num_workers: usize,
    pub(super) gpus_per_worker: usize,
}

enum RecorderEvent {
    Arrival(ArrivalEvent),
    SessionMetadata {
        uuid: Uuid,
        session_id: String,
        turn_index: usize,
    },
    DecodeAssigned {
        uuid: Uuid,
        worker_idx: usize,
    },
    Admission {
        event: AdmissionEvent,
        at_ms: f64,
    },
    Terminal(TerminalObservation),
    Finish {
        wall_time_ms: f64,
    },
}

#[derive(Clone)]
pub(super) struct RecorderSender {
    tx: mpsc::UnboundedSender<RecorderEvent>,
    capture_per_request: bool,
}

impl RecorderSender {
    pub(super) fn record_arrival(&self, arrival: ArrivalEvent) -> Result<()> {
        self.tx
            .send(RecorderEvent::Arrival(arrival))
            .map_err(|_| anyhow!("online replay recorder closed while recording arrival"))
    }

    pub(super) fn record_admission(&self, event: AdmissionEvent, at_ms: f64) -> Result<()> {
        self.tx
            .send(RecorderEvent::Admission { event, at_ms })
            .map_err(|_| anyhow!("online replay recorder closed while recording admission"))
    }

    pub(super) fn record_session_metadata(
        &self,
        uuid: Uuid,
        session_id: String,
        turn_index: usize,
    ) -> Result<()> {
        if !self.capture_per_request {
            return Ok(());
        }
        self.tx
            .send(RecorderEvent::SessionMetadata {
                uuid,
                session_id,
                turn_index,
            })
            .map_err(|_| anyhow!("online replay recorder closed while recording session metadata"))
    }

    pub(super) fn record_decode_assignment(&self, uuid: Uuid, worker_idx: usize) -> Result<()> {
        if !self.capture_per_request {
            return Ok(());
        }
        self.tx
            .send(RecorderEvent::DecodeAssigned { uuid, worker_idx })
            .map_err(|_| anyhow!("online replay recorder closed while recording worker assignment"))
    }

    pub(super) fn record_terminal(&self, terminal: TerminalObservation) -> Result<()> {
        self.tx
            .send(RecorderEvent::Terminal(terminal))
            .map_err(|_| anyhow!("online replay recorder closed while recording terminal"))
    }
}

pub(super) struct OnlineTraceRecorder {
    tx: mpsc::UnboundedSender<RecorderEvent>,
    task: tokio::task::JoinHandle<Result<TraceSimulationReport>>,
    capture_per_request: bool,
}

impl OnlineTraceRecorder {
    pub(super) fn start(options: OnlineRecorderOptions) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            tx,
            task: tokio::spawn(run_recorder(rx, options)),
            capture_per_request: options.capture_per_request,
        }
    }

    pub(super) fn sender(&self) -> RecorderSender {
        RecorderSender {
            tx: self.tx.clone(),
            capture_per_request: self.capture_per_request,
        }
    }

    pub(super) async fn finish(self, wall_time_ms: f64) -> Result<TraceSimulationReport> {
        self.tx
            .send(RecorderEvent::Finish { wall_time_ms })
            .map_err(|_| anyhow!("online replay recorder closed before finalization"))?;
        drop(self.tx);
        self.task
            .await
            .map_err(|error| anyhow!("online replay recorder task failed: {error}"))?
    }
}

pub(super) async fn forward_admissions(
    start: Instant,
    mut admission_rx: mpsc::UnboundedReceiver<ObservedAdmission>,
    recorder: RecorderSender,
) -> Result<()> {
    while let Some(admission) = admission_rx.recv().await {
        let at_ms = admission
            .observed_at
            .saturating_duration_since(start)
            .as_secs_f64()
            * 1000.0;
        recorder.record_admission(admission.event, at_ms)?;
    }
    Ok(())
}

async fn run_recorder(
    mut rx: mpsc::UnboundedReceiver<RecorderEvent>,
    options: OnlineRecorderOptions,
) -> Result<TraceSimulationReport> {
    let mut collector = TraceCollector::default();
    collector.set_defer_token_timeline_finalization(true);
    collector.set_capture_per_request(options.capture_per_request);
    collector.set_sla_thresholds(options.sla);
    collector.set_static_worker_count(0, options.num_workers);
    collector.set_gpus_per_worker(0, options.gpus_per_worker);

    while let Some(event) = rx.recv().await {
        match event {
            RecorderEvent::Arrival(arrival) => collector.on_arrival(
                arrival.uuid,
                arrival.at_ms,
                arrival.input_tokens,
                arrival.output_tokens,
            ),
            RecorderEvent::SessionMetadata {
                uuid,
                session_id,
                turn_index,
            } => collector.on_session_metadata(uuid, session_id, turn_index),
            RecorderEvent::DecodeAssigned { uuid, worker_idx } => {
                collector.on_decode_assigned(uuid, worker_idx);
            }
            RecorderEvent::Admission { event, at_ms } => {
                collector.on_admit(event.uuid, at_ms, event.reused_input_tokens);
            }
            RecorderEvent::Terminal(terminal) => {
                for token_time_ms in terminal.token_times_ms {
                    collector.on_token(terminal.uuid, token_time_ms);
                }
                collector.on_terminal(terminal.uuid, terminal.terminal_time_ms, terminal.status);
            }
            RecorderEvent::Finish { wall_time_ms } => {
                return Ok(collector.finish().with_wall_time_ms(wall_time_ms));
            }
        }
    }

    bail!("online replay recorder channel closed before finalization")
}
