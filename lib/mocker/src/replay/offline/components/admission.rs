// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use anyhow::Result;
use uuid::Uuid;

use super::{ReadyArrival, ReplayMode};
use crate::common::protocols::DirectRequest;
use crate::loadgen::WorkloadDriver;
use crate::replay::ReplayTerminalStatus;

enum AdmissionSource {
    Requests(VecDeque<DirectRequest>),
    Workload(WorkloadDriver),
}

pub(in crate::replay::offline) struct AdmissionQueue {
    source: AdmissionSource,
    mode: ReplayMode,
}

impl AdmissionQueue {
    pub(in crate::replay::offline) fn new_requests(
        source: VecDeque<DirectRequest>,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Requests(source),
            mode,
        }
    }

    pub(in crate::replay::offline) fn new_workload(
        driver: WorkloadDriver,
        mode: ReplayMode,
    ) -> Self {
        Self {
            source: AdmissionSource::Workload(driver),
            mode,
        }
    }

    pub(in crate::replay::offline) fn mode(&self) -> ReplayMode {
        self.mode
    }

    pub(in crate::replay::offline) fn next_ready_time_ms(&mut self) -> Option<f64> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms),
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => driver.next_ready_time_ms(),
            // Concurrency: the driver owns the session cap and gates admission, so defer to
            // it directly (no in-flight clamp needed here).
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                driver.next_ready_time_ms()
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Requests(_)) => None,
        }
    }

    pub(in crate::replay::offline) fn drain_ready(
        &mut self,
        now_ms: f64,
        cluster_in_flight: usize,
    ) -> Result<Vec<ReadyArrival>> {
        match (&self.mode, &mut self.source) {
            (ReplayMode::Trace, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                loop {
                    let arrival_ms = pending
                        .front()
                        .and_then(|request| request.arrival_timestamp_ms)
                        .filter(|arrival_ms| *arrival_ms <= now_ms);
                    let Some(arrival_time_ms) = arrival_ms else {
                        break;
                    };
                    let request = pending
                        .pop_front()
                        .expect("front request must exist when arrival is ready");
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms,
                        replay_hashes: None,
                        session_id: None,
                        turn_index: None,
                    });
                }
                Ok(ready)
            }
            (ReplayMode::Trace, AdmissionSource::Workload(driver)) => Ok(driver
                .pop_ready(now_ms, usize::MAX)
                .into_iter()
                .map(|ready| ReadyArrival {
                    request: ready.request,
                    arrival_time_ms: ready.scheduled_ready_at_ms,
                    replay_hashes: ready.replay_hashes,
                    session_id: Some(ready.session_id),
                    turn_index: Some(ready.turn_index),
                })
                .collect()),
            (ReplayMode::Concurrency { max_in_flight }, AdmissionSource::Requests(pending)) => {
                let mut ready = Vec::new();
                let mut simulated_in_flight = cluster_in_flight;
                while simulated_in_flight < *max_in_flight {
                    let Some(request) = pending.pop_front() else {
                        break;
                    };
                    ready.push(ReadyArrival {
                        request,
                        arrival_time_ms: now_ms,
                        replay_hashes: None,
                        session_id: None,
                        turn_index: None,
                    });
                    simulated_in_flight += 1;
                }
                Ok(ready)
            }
            (ReplayMode::Concurrency { .. }, AdmissionSource::Workload(driver)) => {
                // The driver owns the session cap and only ever holds active sessions'
                // turns in its heap, so drain everything ready in heap (i.e. limit=usize MAX).
                Ok(driver
                    .pop_ready(now_ms, usize::MAX)
                    .into_iter()
                    .map(|ready| ReadyArrival {
                        request: ready.request,
                        arrival_time_ms: now_ms,
                        replay_hashes: ready.replay_hashes,
                        session_id: Some(ready.session_id),
                        turn_index: Some(ready.turn_index),
                    })
                    .collect())
            }
        }
    }

    pub(in crate::replay::offline) fn on_request_terminal(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        rejected: bool,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_terminal(uuid, now_ms, rejected)
    }

    pub(in crate::replay::offline) fn on_request_resolved(
        &mut self,
        uuid: Uuid,
        now_ms: f64,
        status: ReplayTerminalStatus,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        match status {
            ReplayTerminalStatus::Completed => driver.on_terminal(uuid, now_ms, false),
            ReplayTerminalStatus::Rejected => driver.on_terminal(uuid, now_ms, true),
            ReplayTerminalStatus::Canceled | ReplayTerminalStatus::Failed => {
                driver.release_cap_slot(uuid, now_ms);
                Ok(())
            }
        }
    }

    pub(in crate::replay::offline) fn on_output_token(
        &mut self,
        uuid: Uuid,
        token_id: u32,
    ) -> Result<()> {
        let AdmissionSource::Workload(driver) = &mut self.source else {
            return Ok(());
        };
        driver.on_output_token(uuid, token_id)
    }

    pub(in crate::replay::offline) fn is_drained(&self) -> bool {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.is_empty(),
            AdmissionSource::Workload(driver) => driver.is_drained(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_workload(&self) -> bool {
        matches!(self.source, AdmissionSource::Workload(_))
    }

    pub(in crate::replay::offline) fn total_requests(&self) -> usize {
        match &self.source {
            AdmissionSource::Requests(pending) => pending.len(),
            AdmissionSource::Workload(driver) => driver.total_turns(),
        }
    }
}
